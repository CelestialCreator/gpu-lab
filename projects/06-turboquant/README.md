# Project 06: TurboQuant KV Cache Compression

**Status:** In Progress

Implementing Google's TurboQuant (ICLR 2026) to compress KV cache on consumer GPUs. Our Qwen3.5-27B kept crashing at 64K context — TurboQuant's 3-bit KV cache compression extends that to 195K tokens on the same hardware.

## Results

| Metric | Q8_0 KV (baseline) | TurboQuant turbo3 | Delta |
|--------|-------------------|-------------------|-------|
| **Max context** | ~60K (crashes at 64K) | **~195K** | **3.25x** |
| Decode (short) | 33.9 tok/s | 32.2 tok/s | -5% |
| Decode (32K ctx) | 33.8 tok/s | 24.7 tok/s | -27% |
| Decode (170K ctx) | impossible | 13 tok/s | N/A |
| Prefill (1K ctx) | 841 tok/s | 1,469 tok/s | +75% |
| Prefill (32K ctx) | 823 tok/s | 1,861 tok/s | +126% |
| 5090 VRAM | 30.9 GB | 26.7 GB | -4.2 GB |
| 3080 VRAM | 9.7 GB | 8.7 GB | -1 GB |

**Hardware:** RTX 5090 (32GB) + RTX 3080 (10GB) via `--tensor-split 32,8`
**Model:** Qwen3.5-27B-Q8_0 GGUF via llama.cpp

### Key Findings

- **3.25x more context** on the same GPUs — the main win
- **Prefill is 2x faster** — turbo3 compressed KV means less memory bandwidth during prefill
- **Decode slows at long context** — 27% slower at 32K from turbo3 dequantization overhead, but this is acceptable
- **5.2 GB VRAM saved** — enough headroom that 64K no longer crashes

## Background

### What is TurboQuant?

[TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026) compresses the **KV cache** — the memory that stores past token representations during inference. It's complementary to weight quantization (GGUF, GPTQ, AWQ).

The algorithm has two stages:
1. **PolarQuant:** Random rotation (Hadamard transform) makes all coordinates follow a known Beta distribution, then Lloyd-Max optimal scalar quantization bins them efficiently
2. **QJL Correction:** 1-bit residual correction for unbiased inner products (disabled in practice — softmax amplifies the noise)

Key properties:
- **Data-oblivious** — no calibration data needed, applied at inference time
- **Near-optimal** — MSE within 2.7x of information-theoretic lower bound
- **3-bit = 5.3x compression** vs FP16 with near-zero quality loss

### Why This Project?

Our Qwen3.5-27B-Q8_0 deployment uses 28.6 GB for model weights, leaving only ~11.4 GB across both GPUs for KV cache. At Q8_0, the KV cache grows ~0.15 MB per token — so 64K tokens needs ~9.6 GB, which overflows.

TurboQuant at 3-bit reduces KV cache to ~0.06 MB per token — 64K tokens needs only ~3.8 GB.

## Implementation

### Phase 1: Algorithm Deep-Dive

Implemented TurboQuant from scratch in a Jupyter notebook to understand the math:
- Random orthogonal rotation via QR decomposition
- Lloyd-Max optimal quantizer for Beta distribution
- MSE validation: **our 3-bit MSE = 0.034, matching the paper exactly**
- Simulated attention with quantized KV cache
- VRAM savings calculator for Qwen3.5-27B architecture

### Phase 2: llama.cpp Integration

Used [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) — a community CUDA fork that adds `turbo3` and `turbo4` KV cache types to llama.cpp.

The deployment change is minimal:
```yaml
# Before (crashes at 64K)
--cache-type-k q8_0
--cache-type-v q8_0

# After (works at 195K)
--cache-type-k turbo3
--cache-type-v turbo3
```

### Challenges

1. **containerd NVIDIA runtime config lost** — After a kubelet restart, the `runtimes.nvidia` block was dropped from `/etc/containerd/config.toml`. Pods got GPU devices via CDI but no driver libraries (`libcuda.so.1`). Fix: manually insert the nvidia runtime block back into the main config.

2. **envsubst not exporting GPU_UUID** — `source .env` sets the variable but doesn't export it. `envsubst` only reads exported vars. Fix: use `export GPU_UUID=... &&` before envsubst.

3. **TurboQuant fork chat template parsing** — The fork's response parser fails on Qwen3.5's `<think>` tags for long reasoning chains (500+ token think blocks). Short responses and `reasoning_content` field work fine.

## Quick Start

1. **Build TurboQuant llama-server image:**
```bash
cd workloads/llama-server
docker build -t localhost:5000/llama-server:turboquant -f Dockerfile.turboquant .
docker push localhost:5000/llama-server:turboquant
```

2. **Deploy with turbo3 KV cache:**
```bash
# Set GPU UUIDs for 5090 + 3080
export GPU_UUID="GPU-xxx,GPU-yyy"
envsubst '${GPU_UUID}' < deployment-turboquant.yaml | kubectl apply -f -
```

3. **Run benchmarks:**
```bash
python3 projects/06-turboquant/scripts/baseline_benchmark.py
```

## Files

| File | Purpose |
|------|---------|
| `notebooks/01-algorithm-deep-dive.ipynb` | TurboQuant algorithm implementation from scratch |
| `scripts/baseline_benchmark.py` | Benchmark script (tok/s, needle-in-haystack, math) |
| `scripts/kv_cache_analysis.py` | Apply TurboQuant to real KV cache tensors |
| `k8s/job-kv-analysis.yaml` | K8s job for KV cache tensor analysis |
| `benchmarks/baseline_27b.json` | Raw benchmark data |
| `../../workloads/llama-server/Dockerfile.turboquant` | llama.cpp TurboQuant fork Docker image |
| `../../workloads/llama-server/deployment-turboquant.yaml` | K8s deployment with turbo3 KV cache |

## References

- [TurboQuant paper (arxiv 2504.19874)](https://arxiv.org/abs/2504.19874) — Zandieh et al., ICLR 2026
- [Google Research blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) — CUDA fork used
- [llama.cpp TurboQuant discussion](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [llama.cpp feature request #20977](https://github.com/ggml-org/llama.cpp/issues/20977)
