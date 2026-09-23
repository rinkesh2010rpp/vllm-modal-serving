import subprocess
import time
import urllib.request

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-14B-AWQ"  # official Qwen 4-bit AWQ - ~9.5GB weights, fits L4
VLLM_PORT = 8000
MINUTES = 60

app = modal.App("llm-inference")

# Generic HF cache volume (reuse it for any future model swap).
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Official vLLM image. v0.27.1 (Aug 2026) - last patch of the 2nd-newest minor,
# ~2.5 weeks stable, mature AWQ-Marlin + Qwen3 + hermes tool parser.
# Tags: https://hub.docker.com/r/vllm/vllm-openai/tags
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.27.1", add_python="3.11")
    .entrypoint([])
    .pip_install("huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_CACHE": "/models",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# KV math (Qwen3-14B: 40 layers, 8 KV heads, head_dim 128 -> 160 KB/token bf16):
#   L4 24GB @ 0.90 util ~= 21.6GB - 9.5GB weights - ~2GB activation/graphs
#   -> ~10GB KV pool / 160KB ~= ~60K tokens. Full 32K native context fits in bf16,
#   no fp8 KV needed. (chunked prefill + prefix caching are V1 defaults now.)
VLLM_CMD = [
    "vllm", "serve", MODEL_ID,
    "--host", "0.0.0.0",
    "--port", str(VLLM_PORT),
    "--download-dir", "/models",
    "--max-model-len", "32768",         # Qwen3-14B native max
    "--gpu-memory-utilization", "0.90",
    "--max-num-seqs", "2",
    "--enable-auto-tool-choice",        # required for tool-calling clients (deepagents)
    "--tool-call-parser", "hermes",     # Qwen3's tool-call format
    "--reasoning-parser", "qwen3",      # split <think>...</think> into reasoning_content,
    #                                     keeping `content` / tool_calls clean
    "--served-model-name", MODEL_ID, "qwen",
]


# ---------------------------------------------------------------------------
# One-time: pull the model onto the volume so the server never waits on a
# cold download.  No secret required - Qwen2.5 is ungated.
#   modal run index.py::download_model
# ---------------------------------------------------------------------------
@app.function(image=vllm_image, volumes={"/models": hf_cache_vol}, timeout=60 * MINUTES)
def download_model():
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL_ID,
        cache_dir="/models",
        ignore_patterns=["*.pt", "*.bin", "original/*"],  # safetensors only
    )
    hf_cache_vol.commit()
    print("model ready at", path)


# ---------------------------------------------------------------------------
# Server. `@modal.web_server` forwards traffic straight to vLLM's
# OpenAI-compatible API - no proxy code.
#
# enable_memory_snapshot + enable_gpu_snapshot: Modal boots vLLM once, snapshots
# the container memory AND GPU VRAM (weights loaded), then restores that image on
# future cold starts - skipping imports + weight load + warmup. ~72s cold start
# -> ~15-25s if the snapshot restores cleanly.
# ---------------------------------------------------------------------------
@app.cls(
    image=vllm_image,
    gpu="L4",
    volumes={"/models": hf_cache_vol},
    scaledown_window=25,  # ~half the ~50s cold start. Agent model calls are back-to-back;
    #                       this only needs to survive one slow shell tool call mid-loop.
    #                       Anything longer is pure waste - next invocation is an hour away.
    timeout=60 * MINUTES,
    max_containers=1,  # solo user: never spin a 2nd GPU, queue instead
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=2)
class VLLMServer:
    @modal.enter(snap=True)
    def start_engine(self):
        # Runs during the snapshotting phase (GPU present because of
        # enable_gpu_snapshot). vLLM loads weights here; Modal then snapshots.
        self.proc = subprocess.Popen(" ".join(VLLM_CMD), shell=True)
        url = f"http://127.0.0.1:{VLLM_PORT}/health"
        deadline = time.time() + 12 * MINUTES
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited early (code {self.proc.returncode})")
            try:
                if urllib.request.urlopen(url, timeout=2).status == 200:
                    print("vllm healthy")
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("vllm did not become healthy in time")

    @modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
    def serve(self):
        # Server is already running (started in start_engine, restored from the
        # snapshot on cold starts). Nothing to do here.
        pass
