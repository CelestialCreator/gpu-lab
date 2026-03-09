"""Upload GRPO-trained model and dataset to HuggingFace Hub."""

import argparse
import os

from huggingface_hub import HfApi, create_repo


MODEL_CARD = """---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
- reasoning
- math
- grpo
- reinforcement-learning
- rlvr
- qwen3.5
datasets:
- gsm8k
- celestialcreator/Qwen3.5-0.8B-GRPO-Math-Dataset
pipeline_tag: text-generation
---

# Qwen3.5-0.8B-GRPO-Math

A reasoning-enhanced version of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), trained using **GRPO (Group Relative Policy Optimization)** — the RL technique behind DeepSeek-R1 — on a single RTX 5090.

## Results

| Model | GSM8K 8-shot CoT | GSM8K Zero-shot |
|-------|:-----------------:|:---------------:|
| Qwen3.5-0.8B (baseline) | 53.5% | 52.1% |
| **Qwen3.5-0.8B-GRPO-Math** | 50.4% | **58.0% (+5.9pp)** |

The model was trained to reason using `<think>` tags. **Zero-shot performance improved by +5.9 percentage points** because the model internalized step-by-step reasoning — it no longer needs few-shot examples. The 8-shot drop is expected: few-shot examples conflict with the model's learned reasoning format.

## Training Pipeline

### Phase 1: SFT Warmup
- **Data:** [3,558 reasoning examples](https://huggingface.co/datasets/celestialcreator/Qwen3.5-0.8B-GRPO-Math-Dataset) (Claude-generated math chains + community reasoning data)
- **Purpose:** Teach the model `<think>` tag format before RL
- **Stats:** 1 epoch, loss 0.932, 78% token accuracy

### Phase 2: GRPO Training
- **Data:** GSM8K train split (7,473 math word problems)
- **Rewards:** Math correctness (1.0/0.0) + format reward (0.3 for `<think>` tags, 0.2 for `####` answer)
- **Config:** 8 generations/prompt, batch size 1 × 8 grad accum, lr 1e-6, β=0.04
- **Hardware:** Single NVIDIA RTX 5090 (32GB VRAM)
- **Duration:** ~77 hours, 15,900 steps (epoch 2.13)

### What is GRPO?

GRPO eliminates the need for a separate reward model and critic network (unlike PPO). For each prompt, it:
1. Samples G completions from the policy
2. Scores each with a verifiable reward (exact math answer checking)
3. Normalizes rewards within the group (relative advantage)
4. Updates the policy using a clipped surrogate objective

This means only 2 models in memory (policy + reference) instead of 4, making it feasible on consumer GPUs.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "celestialcreator/Qwen3.5-0.8B-GRPO-Math"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant that thinks step by step. Show your reasoning inside <think> tags before giving your final answer. End math answers with: #### <number>"},
    {"role": "user", "content": "If a train travels at 60 mph for 2.5 hours, how far does it go?"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training Code

Full training pipeline (Dockerfile, k8s configs, scripts): [github.com/CelestialCreator/gpu-lab/tree/main/projects/05-grpo-reasoning](https://github.com/CelestialCreator/gpu-lab/tree/main/projects/05-grpo-reasoning)

## Key Findings

- **Qwen3.5-0.8B uses DeltaNet** (hybrid Gated DeltaNet + Gated Attention layers). Install `flash-linear-attention` + `causal-conv1d` for fast generation.
- **SDPA > FLA for inference** — 3.6x faster on first call. Use `attn_implementation="sdpa"`.
- **Zero-shot is the right eval for RL-trained reasoning models** — few-shot examples conflict with learned reasoning patterns.
- **0.8B is near the capacity ceiling for GRPO** — the model internalizes reasoning format but has limited room for math accuracy gains. Consider 1.5B+ for stronger results.

## Acknowledgments

- [TRL](https://github.com/huggingface/trl) for the GRPOTrainer implementation
- [Qwen Team](https://github.com/QwenLM) for the base model
- [DeepSeek](https://arxiv.org/abs/2402.03300) for the GRPO algorithm

## Citation

```bibtex
@misc{qwen35-grpo-math-2026,
  author = {Akshay Mhaskar},
  title = {Qwen3.5-0.8B-GRPO-Math: Teaching a Small Model to Reason with RL},
  year = {2026},
  url = {https://huggingface.co/celestialcreator/Qwen3.5-0.8B-GRPO-Math},
}
```
"""

DATASET_CARD = """---
language:
- en
license: apache-2.0
tags:
- reasoning
- math
- sft
- grpo
- chain-of-thought
size_categories:
- 1K<n<10K
---

# Qwen3.5-0.8B-GRPO-Math Training Dataset

SFT warmup dataset used to train [celestialcreator/Qwen3.5-0.8B-GRPO-Math](https://huggingface.co/celestialcreator/Qwen3.5-0.8B-GRPO-Math).

## Dataset Description

**3,558 reasoning examples** from 3 sources, standardized to use `<think>` tags:

| Source | Examples | Description |
|--------|:--------:|-------------|
| Claude Sonnet math chains | 1,000 | GSM8K questions solved by Claude with step-by-step reasoning |
| TeichAI Opus reasoning | ~250 | General reasoning examples with `<think>` tags |
| Opus 4.6 Reasoning 3000x | ~2,300 | Mixed reasoning with thinking + solution fields |

## Format

Each example is a chat-format JSONL line:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant that thinks step by step. Show your reasoning inside <think> tags before giving your final answer."},
    {"role": "user", "content": "If a train travels at 60 mph for 2.5 hours, how far does it go?"},
    {"role": "assistant", "content": "<think>\\nDistance = speed × time\\nDistance = 60 × 2.5 = 150 miles\\n</think>\\n\\nThe train travels 150 miles.\\n#### 150"}
  ]
}
```

## Files

- `sft_combined.jsonl` — Full merged dataset (3,558 examples) used for SFT training
- `sft_train.jsonl` — Claude-generated math reasoning chains (1,000 examples)
- `raw_chains.jsonl` — Raw Claude API outputs before formatting

## Usage

This dataset was used as **Phase 1 (SFT warmup)** before GRPO training. The SFT phase teaches the model the `<think>` tag reasoning format, giving it a template before RL exploration.

```python
from datasets import load_dataset
ds = load_dataset("celestialcreator/Qwen3.5-0.8B-GRPO-Math-Dataset", data_files="sft_combined.jsonl", split="train")
```

## Training Code

[github.com/CelestialCreator/gpu-lab/tree/main/projects/05-grpo-reasoning](https://github.com/CelestialCreator/gpu-lab/tree/main/projects/05-grpo-reasoning)
"""


def upload_model(api: HfApi, model_dir: str, repo_name: str):
    """Upload model checkpoint to HuggingFace."""
    print(f"Creating repo {repo_name}...")
    create_repo(repo_name, exist_ok=True)

    # Upload model files (skip optimizer, scheduler, training state)
    skip_patterns = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "trainer_state.json"]

    print(f"Uploading model from {model_dir}...")
    for fname in os.listdir(model_dir):
        if fname in skip_patterns:
            print(f"  Skipping {fname}")
            continue
        fpath = os.path.join(model_dir, fname)
        if os.path.isfile(fpath):
            print(f"  Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=repo_name,
            )

    # Upload model card
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
    )
    print(f"Model uploaded: https://huggingface.co/{repo_name}")


def upload_dataset(api: HfApi, data_dir: str, repo_name: str):
    """Upload SFT dataset to HuggingFace."""
    print(f"Creating dataset repo {repo_name}...")
    create_repo(repo_name, repo_type="dataset", exist_ok=True)

    data_files = ["sft_combined.jsonl", "sft_train.jsonl", "raw_chains.jsonl"]
    for fname in data_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            print(f"  Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=repo_name,
                repo_type="dataset",
            )

    # Upload dataset card
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
    )
    print(f"Dataset uploaded: https://huggingface.co/datasets/{repo_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/workspace/grpo-output/checkpoint-15900")
    parser.add_argument("--data_dir", type=str, default="/home/akshay/grpo-sft-data")
    parser.add_argument("--model_repo", type=str, default="celestialcreator/Qwen3.5-0.8B-GRPO-Math")
    parser.add_argument("--dataset_repo", type=str, default="celestialcreator/Qwen3.5-0.8B-GRPO-Math-Dataset")
    parser.add_argument("--skip_model", action="store_true")
    parser.add_argument("--skip_dataset", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    if not args.skip_dataset:
        upload_dataset(api, args.data_dir, args.dataset_repo)

    if not args.skip_model:
        upload_model(api, args.model_dir, args.model_repo)

    print("\nDone!")


if __name__ == "__main__":
    main()
