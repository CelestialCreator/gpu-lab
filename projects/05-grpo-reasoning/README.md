# Project 05: GRPO Reasoning Training

Teaching Qwen3.5-0.8B to reason using **GRPO (Group Relative Policy Optimization)** — the RL technique behind DeepSeek-R1 — with verifiable math rewards on a single RTX 5090.

**Model:** [celestialcreator/Qwen3.5-0.8B-GRPO-Math](https://huggingface.co/celestialcreator/Qwen3.5-0.8B-GRPO-Math)
**Dataset:** [celestialcreator/Qwen3.5-0.8B-GRPO-Math-Dataset](https://huggingface.co/datasets/celestialcreator/Qwen3.5-0.8B-GRPO-Math-Dataset)

## Results

| Model | GSM8K 8-shot CoT | GSM8K Zero-shot |
|-------|:-----------------:|:---------------:|
| Qwen3.5-0.8B (baseline) | 53.5% | 52.1% |
| **+ SFT + GRPO (ours)** | 50.4% | **58.0% (+5.9pp)** |

The model was trained to reason using `<think>` tags. Zero-shot performance improved significantly because the model internalized step-by-step reasoning — it no longer needs few-shot examples. The 8-shot drop is expected: few-shot examples conflict with the model's learned reasoning format.

## Overview

GRPO eliminates the need for a separate reward model and critic network (unlike PPO). For each prompt, it:
1. Samples G completions from the policy
2. Scores each with a verifiable reward (exact math answer checking)
3. Normalizes rewards within the group (relative advantage)
4. Updates the policy using a clipped surrogate objective

Only 2 models in memory (policy + reference) instead of 4, making it feasible on consumer GPUs.

## Training Pipeline

### Phase 1: SFT Warmup
- **Data:** 3,558 reasoning examples (1K Claude-generated math chains + 250 TeichAI Opus reasoning + 2.3K Opus 4.6 Reasoning)
- **Purpose:** Teach the model `<think>` tag format before RL
- **Stats:** 1 epoch, loss 0.932, 78% token accuracy

### Phase 2: GRPO Training
- **Data:** GSM8K train split (7,473 math word problems)
- **Rewards:** Math correctness (1.0/0.0) + format reward (0.3 for `<think>` tags, 0.2 for `####` answer)
- **Config:** 8 generations/prompt, batch size 1 x 8 grad accum, lr 1e-6, beta 0.04
- **Hardware:** Single NVIDIA RTX 5090 (32GB VRAM)
- **Duration:** ~77 hours, stopped at epoch 2.13 (checkpoint-15900)

## Quick Start

### 1. Set up secrets
```bash
cp .env.example .env
# Edit .env with your HF_TOKEN, WANDB_API_KEY, GPU_UUID
```

### 2. Build Docker image
```bash
docker build -t localhost:5000/grpo-reasoning:latest -f Dockerfile.grpo .
docker push localhost:5000/grpo-reasoning:latest
```

### 3. Generate SFT data (optional — uses Claude API)
```bash
source .env
python3 scripts/generate_sft_data.py --num_examples 1000
python3 scripts/merge_sft_data.py
```

### 4. Run training (SFT + GRPO)
```bash
./apply.sh k8s/job-grpo-train.yaml
kubectl logs -f job/grpo-train
```

### 5. Evaluate
```bash
./apply.sh k8s/job-grpo-eval.yaml
kubectl logs -f job/grpo-eval
```

### 6. Monitor training
```bash
./scripts/monitor.sh 300  # check every 5 minutes with desktop notifications
```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.grpo` | Container with TRL + transformers + FLA for Qwen3.5 |
| `apply.sh` | Helper to inject `.env` secrets into k8s YAML via envsubst |
| `scripts/generate_sft_data.py` | Generate reasoning chains using Claude API |
| `scripts/merge_sft_data.py` | Merge SFT data from multiple sources |
| `scripts/train_sft.py` | SFT warmup on reasoning chains |
| `scripts/train_grpo.py` | GRPO training using TRL's GRPOTrainer |
| `scripts/reward.py` | Math answer verification reward functions |
| `scripts/eval_gsm8k.py` | GSM8K evaluation (8-shot CoT + zero-shot) |
| `scripts/upload_to_hf.py` | Upload trained model to HuggingFace |
| `scripts/monitor.sh` | Training monitor with desktop notifications |
| `k8s/job-grpo-train.yaml` | K8s training job (SFT → GRPO pipeline) |
| `k8s/job-grpo-eval.yaml` | K8s evaluation job |

## Key Findings

- **Qwen3.5-0.8B uses DeltaNet** (hybrid Gated DeltaNet + Gated Attention). Install `flash-linear-attention` + `causal-conv1d` for fast generation — without FLA, torch fallback is ~10x slower.
- **SDPA is faster than FLA for inference** (3.6x on first call). Use `attn_implementation="sdpa"` for eval.
- **0.8B may be near capacity ceiling for GRPO** — the model can internalize reasoning format but has limited room to improve math accuracy. Consider 1.5B+ for stronger gains.
- **Zero-shot is the right eval for RL-trained models** — few-shot examples conflict with learned reasoning patterns.

## VRAM Budget (~10-12 GB on RTX 5090)

- Model (bf16): ~1.6 GB
- Reference model: ~1.6 GB
- 8 completions x 512 tokens: ~2-4 GB
- Gradients + optimizer: ~4 GB
- Headroom: ~20 GB free
