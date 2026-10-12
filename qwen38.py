"""Pilot: GGUF models on llama.cpp, Modal GPUs. Defaults to Qwen3.8-27B (ISTA-DASLab GSQ-RCO GGUF) on an L4.

Separate app from `llm-inference` (index.py). Quality/speed pilot only —
Sudarshana stays on OpenRouter.

    modal run qwen38.py::download_model     # one-time, ~11.3GB onto hf-cache
    modal deploy qwen38.py
    modal app stop llm-inference-q38        # stop when done testing

Overrides at deploy time (read when the module loads):
    Q38_CTX=131072   context length
    Q38_KV=f16       KV cache type (q8_0 halves it if 128K doesn't fit)
    Q38_GPU=L4       Modal GPU type (e.g. A100-40GB)

Speed benchmark on any GPU (one-shot, no deploy):
    Q38_GPU=A100-40GB modal run qwen38.py::bench_main --bodies bench-bodies.json
"""

import os
import subprocess
import time
import urllib.request

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ID = os.environ.get("Q38_REPO", "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF")
GGUF_FILE = os.environ.get("Q38_FILE", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")  # 11.29 GiB incl. MTP head; no mmproj
MODEL_DIR = "/models/qwen38" if REPO_ID.startswith("ISTA-DASLab/Qwen3.8") else "/models/" + REPO_ID.split("/")[1]
MTP = os.environ.get("Q38_MTP", "on") == "on"   # only for GGUFs that carry an MTP head
PORT = 8000
MINUTES = 60

# Sudarshana's logged calls since the 70K summarization change peak at 82K
# input + up to 32K max_tokens -> ~114K, so 128K. KV is ~64 KiB/token f16
# (16 full-attention layers; the rest are fixed-size linear-attention state)
# -> ~8GB at 128K on top of ~11.3GB weights. Tight on the L4's ~22GB usable.
CTX = int(os.environ.get("Q38_CTX", "131072"))
KV_TYPE = os.environ.get("Q38_KV", "f16")
GPU = os.environ.get("Q38_GPU", "L4")

# The container re-imports this module without the local shell's env, so bake
# the overrides into the image to keep SERVER_CMD identical on both sides.
OVERRIDES = {k: v for k, v in os.environ.items() if k.startswith("Q38_")}

app = modal.App("llm-inference-q38")

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Official llama.cpp CUDA server image, unpinned (newest) — MTP support is recent.
image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.11")
    .entrypoint([])
    .pip_install("huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", **OVERRIDES})
)

SERVER_CMD = [
    "/app/llama-server",
    "-m", f"{MODEL_DIR}/{GGUF_FILE}",
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "-ngl", "99",                 # all layers on GPU
    "-fa", "on",                  # flash attention
    "-c", str(CTX),
    "--parallel", "1",            # one slot = full context + prompt-cache reuse across a run's calls
    "-ctk", KV_TYPE, "-ctv", KV_TYPE,
    "--jinja",                    # model's own chat template -> OpenAI tool calls
    "--reasoning-format", "deepseek",  # <think> -> reasoning_content, like the vLLM qwen3 parser
    "--alias", "qwen",
    "--metrics",
    "--slots",
] + (["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"] if MTP else [])  # speculative decoding via the MTP head


# ---------------------------------------------------------------------------
# One-time: pull the GGUF onto the volume.
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={"/models": hf_cache_vol}, timeout=60 * MINUTES)
def download_model():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, GGUF_FILE, local_dir=MODEL_DIR)
    hf_cache_vol.commit()
    print("model ready at", path)


# ---------------------------------------------------------------------------
# One-shot boot check before deploying: does it load at CTX, how much VRAM,
# does a tool call parse. Runs once and exits — no deploy, no retry loop.
#   modal run qwen38.py::smoke
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes={"/models": hf_cache_vol}, timeout=20 * MINUTES)
def smoke():
    import json

    proc = subprocess.Popen(SERVER_CMD)
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode})")
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(2)
    print(f"healthy after {time.time() - t0:.0f}s")
    print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout)
    body = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "What's the weather in Pune? Use the tool."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "max_tokens": 1024,
    }
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t1 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    print(f"request took {time.time() - t1:.1f}s")
    print(json.dumps(resp, indent=1)[:3000])
    proc.terminate()


# ---------------------------------------------------------------------------
# One-shot speed benchmark: boot, run each request body in order (prompt cache
# carries between them like a real run), report llama.cpp timings, exit.
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes={"/models": hf_cache_vol}, timeout=55 * MINUTES)
def bench(bodies: list[dict]) -> list[dict]:
    import json

    proc = subprocess.Popen(SERVER_CMD)
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode})")
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(2)
    print(f"healthy after {time.time() - t0:.0f}s")
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout)
    results = []
    for i, body in enumerate(bodies):
        body = {**body, "stream": False}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t1 = time.time()
        resp = json.loads(urllib.request.urlopen(req, timeout=20 * 60).read())
        t = resp.get("timings", {})
        r = {"i": i, "wall_s": round(time.time() - t1, 1),
             "prompt_n": t.get("prompt_n"), "prompt_tps": round(t.get("prompt_per_second", 0)),
             "gen_n": t.get("predicted_n"), "gen_tps": round(t.get("predicted_per_second", 0), 1),
             "draft_accept": round(t["draft_n_accepted"] / t["draft_n"], 2) if t.get("draft_n") else None,
             "finish": resp["choices"][0].get("finish_reason")}
        print(r, flush=True)
        msg = resp["choices"][0]["message"]
        r["content"] = (msg.get("content") or "")[:3000]
        r["reasoning"] = (msg.get("reasoning_content") or "")[:1500]
        r["tool_calls"] = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"][:1500]}
                           for c in msg.get("tool_calls") or []]
        results.append(r)
    proc.terminate()
    return results


@app.local_entrypoint()
def bench_main(bodies: str):
    import json

    results = bench.remote(json.load(open(bodies, encoding="utf-8")))
    json.dump(results, open(bodies.replace(".json", "-results.json"), "w", encoding="utf-8"), indent=1)


# ---------------------------------------------------------------------------
# Server. Same snapshot pattern as index.py: llama-server loads weights during
# the snapshot phase, Modal snapshots memory + VRAM, cold starts restore it.
# The snapshot holds the empty server — the prompt cache from earlier calls
# does not survive a scale-down.
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": hf_cache_vol},
    scaledown_window=20,  # 95% of in-run gaps between calls are <14s (llm-calls logs);
    #                       revisit once prefill speed is measured — each miss re-prefills 50-80K tokens.
    timeout=60 * MINUTES,
    max_containers=1,     # solo user: never a 2nd GPU
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=4)  # llama-server queues on its single slot
class LlamaServer:
    @modal.enter(snap=True)
    def start_engine(self):
        if not os.path.exists(f"{MODEL_DIR}/{GGUF_FILE}"):
            # Fail fast instead of crash-looping on a missing file.
            raise RuntimeError("model missing: run `modal run qwen38.py::download_model` first")
        self.proc = subprocess.Popen(SERVER_CMD)
        url = f"http://127.0.0.1:{PORT}/health"
        deadline = time.time() + 10 * MINUTES
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self.proc.returncode})")
            try:
                if urllib.request.urlopen(url, timeout=2).status == 200:
                    print("llama-server healthy")
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("llama-server did not become healthy in time")

    @modal.web_server(port=PORT, startup_timeout=12 * MINUTES)
    def serve(self):
        pass
