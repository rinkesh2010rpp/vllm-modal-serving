# llama.cpp pilot results (2026-10-10 / 11)

Results from [qwen38.py](qwen38.py): GGUF models on llama.cpp, Modal GPUs, measured against an autonomous agent's real logged calls (prompts not included in this repo). The goal was to understand the tradeoffs of self-hosting, not to beat a hosted API on cost.

## Setup

- `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, `IQ3_S-mtp` file (11.29 GiB, ~3.5 bits/weight, includes the multi-token-prediction head). No vision projector.
- Stock `ghcr.io/ggml-org/llama.cpp:server-cuda`, `-ngl 99 -fa on -c 131072 --parallel 1 --jinja`, speculative decoding with `--spec-type draft-mtp --spec-draft-n-max 3` (the flag was renamed from `mtp`).
- Comparison model: `unsloth/GLM-4.7-Flash-GGUF` `UD-Q4_K_XL` (16.3 GiB), no MTP head.

## Memory

| Model | GPU | Context | KV type | VRAM used |
|---|---|---|---|---|
| Qwen3.8-27B IQ3_S-mtp | L4 (23 GB) | 128K | f16 | 21.2 GB |
| Qwen3.8-27B IQ3_S-mtp | A100 40GB | 128K | f16 | 21.4 GB |
| GLM-4.7-Flash UD-Q4_K_XL | L4 | 128K | f16 | OOM (KV alone needs 6.8 GB) |
| GLM-4.7-Flash UD-Q4_K_XL | L4 | 128K | q8_0 | 20.7 GB |

Qwen3.8's KV is ~64 KiB/token at f16: only 16 of its layers are full attention, the rest are fixed-size linear attention.

## Speed

Same requests on each setup. Generation in tok/s; prompt processing (pp) in tok/s.

| Request | Qwen3.8 on L4 | Qwen3.8 on A100 40GB | GLM-4.7-Flash on L4 |
|---|---|---|---|
| Short tool call | 37 | — | 70 |
| ~3K-token free-form answer | ~31 | 75 | 67 |
| Real agent call, ~32-35K context | 32 | 84 | 35 |
| Real agent call, ~47-62K context | 22 | 56 | 23 |
| pp, ~32-35K prompt | 560 | 1,105 | 787 |
| pp, ~47-62K prompt | 464 | 1,053 | 469 |

- MTP draft acceptance: ~81% on tool-call JSON, ~47-62% on long free-form text.
- The A100 was ~2.5x the L4, not the ~5x its bandwidth suggests: at that bandwidth the IQ3_S dequant kernels, sequential MTP drafting and launch overhead become the limit.
- GLM-4.7-Flash (30B total, ~3B active) is ~2x faster than the dense 27B at short context, but the same speed at 30-60K context: all of its layers attend over the full KV, so attention dominates.
- Prompt-cache reuse works within a run: after the first call, later calls with the same prefix start in 2-10s.

## Quality

One-step replay of real agent calls, comparing each model's next action with the hosted model's (DeepSeek V4 Flash). Judged by hand.

| | Qwen3.8-27B | GLM-4.7-Flash |
|---|---|---|
| Answered | 12 of 16 (4 lost to Modal infra) | 16 of 16 |
| Matched or equally valid | 9 of 12 | 7 of 16 |
| Weaker but acceptable | 1 | 4 |
| Clear mistakes | 2 | 5 |

Qwen3.8's misses were judgment slips (a wrong date in a probe, a missed self-correction, a skipped house rule). GLM-4.7-Flash additionally wrote wrong conclusions and an invented result into the agent's records, which is the most damaging failure for an autonomous agent.

## Modal gotchas

- **Web endpoints return HTTP 303 if no response byte is sent within 150s.** A cold start or GPU-snapshot rebuild plus a long prompt crosses it. `qwen38.py::bench` runs requests inside the GPU function to avoid the web layer.
- **GPU containers can be preempted mid-request.** Non-preemptible execution costs 3x.
- Qwen3.8's chat template rejects system messages after the first one; clients that inject mid-conversation system messages need to send them as user turns.
- The GPU memory snapshot holds the freshly loaded server, so the prompt cache does not survive a scale-down.

## Cost (Modal, measured speeds, the agent's daily volume)

| | GPU $/hr | GPU hours/day | ~$/month |
|---|---|---|---|
| L4 | 0.80 | ~6.4 | ~155 |
| A100 40GB | 2.10 | ~2.9 | ~185 |

A single-user agent is the worst case for GPU economics: one request at a time leaves most of the GPU's compute idle. The hosted API ran ~$4-27/month for the same work.
