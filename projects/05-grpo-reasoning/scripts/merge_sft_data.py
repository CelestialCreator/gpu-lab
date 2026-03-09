"""
Merge SFT datasets from 3 sources:
1. Our Claude Sonnet math reasoning chains (1000) — <thought> tags → converted to <think>
2. TeichAI/claude-4.5-opus-high-reasoning-250x (250) — <think> tags (general reasoning)
3. nohurry/Opus-4.6-Reasoning-3000x-filtered (2326) — thinking + solution fields (Opus 4.6)

Standardizes on <think></think> tags.
Output: combined JSONL ready for SFT training.
"""

import json
import argparse

from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a helpful assistant that thinks step by step. "
    "Show your reasoning inside <think> tags before giving your final answer."
)


def convert_thought_to_think(text: str) -> str:
    """Convert <thought>...</thought> tags to <think>...</think>."""
    return text.replace("<thought>", "<think>").replace("</thought>", "</think>")


def load_our_math_data(path: str) -> list[dict]:
    """Load our Claude-generated math reasoning chains."""
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msgs[1]["content"]},
                    {"role": "assistant", "content": convert_thought_to_think(msgs[2]["content"])},
                ],
                "source": "sonnet-math",
            })
    return examples


def load_teichai_opus() -> list[dict]:
    """Load TeichAI/claude-4.5-opus-high-reasoning-250x."""
    ds = load_dataset("TeichAI/claude-4.5-opus-high-reasoning-250x", split="train")
    examples = []
    for ex in ds:
        msgs = ex["messages"]
        assistant_content = msgs[2]["content"] if len(msgs) > 2 else ""
        if not assistant_content or "<think>" not in assistant_content:
            continue
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msgs[1]["content"]},
                {"role": "assistant", "content": assistant_content},
            ],
            "source": "teichai-opus",
        })
    return examples


def load_opus46_reasoning() -> list[dict]:
    """Load nohurry/Opus-4.6-Reasoning-3000x-filtered.

    Has separate 'thinking' and 'solution' fields — combine into <think> format.
    """
    ds = load_dataset("nohurry/Opus-4.6-Reasoning-3000x-filtered", split="train")
    examples = []
    for ex in ds:
        thinking = ex.get("thinking", "").strip()
        solution = ex.get("solution", "").strip()
        problem = ex.get("problem", "").strip()

        if not thinking or not solution or not problem:
            continue

        # Combine thinking + solution into <think> format
        assistant_content = f"<think>\n{thinking}\n</think>\n\n{solution}"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": assistant_content},
            ],
            "source": "opus46-reasoning",
            "difficulty": ex.get("difficulty", "unknown"),
            "category": ex.get("category", "unknown"),
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Merge SFT datasets")
    parser.add_argument("--our_data", type=str, default="/home/akshay/grpo-sft-data/sft_train.jsonl")
    parser.add_argument("--output", type=str, default="/home/akshay/grpo-sft-data/sft_combined.jsonl")
    parser.add_argument("--max_opus46", type=int, default=2326,
                        help="Max examples from Opus 4.6 dataset (use all by default)")
    args = parser.parse_args()

    print("=== Loading datasets ===")

    print("1. Our Sonnet math reasoning chains...")
    math_data = load_our_math_data(args.our_data)
    print(f"   {len(math_data)} examples")

    print("2. TeichAI Opus general reasoning...")
    teichai_data = load_teichai_opus()
    print(f"   {len(teichai_data)} examples")

    print("3. Opus 4.6 Reasoning 3000x filtered...")
    opus46_data = load_opus46_reasoning()
    if len(opus46_data) > args.max_opus46:
        opus46_data = opus46_data[:args.max_opus46]
    print(f"   {len(opus46_data)} examples")

    # Combine all
    combined = math_data + teichai_data + opus46_data
    print(f"\n=== Combined: {len(combined)} total ===")

    # Stats by source
    from collections import Counter
    sources = Counter(ex.get("source", "unknown") for ex in combined)
    for src, count in sources.items():
        print(f"  {src}: {count}")

    has_think = sum(1 for ex in combined if "<think>" in ex["messages"][2]["content"])
    avg_len = sum(len(ex["messages"][2]["content"]) for ex in combined) / len(combined)
    print(f"  With <think> tags: {has_think}/{len(combined)}")
    print(f"  Avg response length: {avg_len:.0f} chars")

    # Write (strip source/metadata, keep only messages)
    with open(args.output, "w") as f:
        for ex in combined:
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
