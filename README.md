# vllm-modal-serving

A serverless, OpenAI-compatible LLM inference server: [vLLM](https://github.com/vllm-project/vllm) running on [Modal](https://modal.com)'s serverless GPUs. Deploy once, get an autoscaling `/v1/chat/completions` endpoint that scales to zero when idle and cold-starts in ~15-25s.

Not tied to any one model or project — swap `MODEL_ID` in [index.py](index.py) to serve whatever fits the target GPU's memory budget. Currently configured for `Qwen/Qwen3-14B-AWQ` (4-bit AWQ, ~9.5GB weights) on an L4 (24GB).

## Why this exists

Hosted inference APIs (OpenRouter, Groq, etc.) are convenient but come with per-token cost, rate limits, and no control over the serving stack. This runs the model on infra you control, billed only for GPU-seconds actually used — while still speaking the standard OpenAI API, so any OpenAI-compatible client (LangChain, `openai` SDK, etc.) can point at it with just a `base_url` change.

## How it works

- **`vllm/vllm-openai` Docker image** (not pip-installed vLLM) — avoids torch version conflicts between vLLM and its dependencies.
- **GPU memory snapshotting** (`enable_memory_snapshot` + `enable_gpu_snapshot`): Modal boots vLLM once, snapshots the container's memory *and* loaded GPU VRAM, then restores that snapshot on future cold starts — skipping the import + weight-load + warmup cycle. Cuts cold start from ~72s to ~15-25s.
- **Tool-calling support**: `--enable-auto-tool-choice --tool-call-parser hermes` so OpenAI-style function/tool calls work out of the box for agent frameworks (LangChain, deepagents, etc.).
- **`--reasoning-parser`** splits `<think>...</think>` reasoning traces into a separate `reasoning_content` field, keeping `content` and `tool_calls` clean for downstream parsing.
- **Scale-to-zero**: `max_containers=1` (single-tenant), `scaledown_window=25` — the container stays warm just long enough to survive a slow tool call mid-agent-loop, then shuts down since the next request is an hour away.

## Usage

```bash
# One-time: pull model weights onto the persistent volume
modal run index.py::download_model

# Deploy the server
modal deploy index.py
```

Then call it like any OpenAI-compatible endpoint (see [client.py](client.py) for a full example):

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-endpoint>.modal.run/v1", api_key="dummy")
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Explain what an API is in one simple sentence."}],
)
```

## Used by

Serves the LLM backend for [sudarshana](https://github.com/rinkesh2010rpp/sudarshana), an autonomous agent, but is otherwise a standalone, reusable serving layer.
