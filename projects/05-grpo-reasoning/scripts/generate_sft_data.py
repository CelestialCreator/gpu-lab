"""
Generate reasoning chain SFT data using Claude as the teacher model.

Takes GSM8K questions and asks Claude to solve them step-by-step with
<thought> tags, producing training data for SFT warmup before GRPO.

This solves the "cold start" problem: a 0.8B model may struggle to
discover good reasoning patterns via pure RL exploration. A small
amount of SFT on high-quality reasoning chains gives it a template.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

try:
    import anthropic
except ImportError:
    print("Installing anthropic SDK...")
    import subprocess
    subprocess.check_call(["pip", "install", "anthropic"])
    import anthropic


TEACHER_SYSTEM_PROMPT = """\
You are a math tutor helping a student learn to solve word problems step by step.

Format your response EXACTLY as follows:
1. Start with <thought> tag containing your step-by-step reasoning
2. End with </thought> tag
3. Then state the final answer as: #### <number>

Example format:
<thought>
Let me break this down step by step.
Step 1: [identify what we know]
Step 2: [set up the calculation]
Step 3: [solve]
</thought>
#### 42

Be thorough but concise in your reasoning. Show each arithmetic step clearly.\
"""

TEACHER_USER_TEMPLATE = "Solve this math problem:\n\n{question}"


def generate_chain(client: anthropic.Anthropic, question: str, answer: str, model: str = "claude-sonnet-4-20250514") -> dict | None:
    """Call Claude to generate a reasoning chain for one question."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=TEACHER_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": TEACHER_USER_TEMPLATE.format(question=question)}
            ],
        )
        text = response.content[0].text

        # Verify the response has the expected format
        if "<thought>" not in text or "####" not in text:
            return None

        return {
            "question": question,
            "ground_truth": answer,
            "reasoning_chain": text,
            "model": model,
        }
    except anthropic.RateLimitError:
        time.sleep(5)
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def format_for_sft(example: dict) -> dict:
    """Format a generated example into chat format for SFT."""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful math assistant. Solve problems step by step, "
                    "showing your reasoning inside <thought> tags. End with the final "
                    "answer as: #### <number>"
                ),
            },
            {
                "role": "user",
                "content": example["question"],
            },
            {
                "role": "assistant",
                "content": example["reasoning_chain"],
            },
        ]
    }


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract ground truth number from GSM8K answer field."""
    import re
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return answer_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate SFT data using Claude")
    parser.add_argument("--output_dir", type=str, default="/home/akshay/grpo-workspace/sft-data")
    parser.add_argument("--num_examples", type=int, default=1000,
                        help="Number of examples to generate (from GSM8K train)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model to use as teacher")
    parser.add_argument("--max_workers", type=int, default=5,
                        help="Parallel API calls")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Source .env first.")

    client = anthropic.Anthropic(api_key=api_key)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_path = os.path.join(args.output_dir, "raw_chains.jsonl")
    sft_path = os.path.join(args.output_dir, "sft_train.jsonl")

    # Load existing data if resuming
    existing = set()
    if args.resume and os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                d = json.loads(line)
                existing.add(d["question"])
        print(f"Resuming: {len(existing)} examples already generated")

    # Load GSM8K train
    print("Loading GSM8K train set...")
    ds = load_dataset("gsm8k", "main", split="train")
    print(f"Total GSM8K train: {len(ds)} examples")

    # Select examples to generate
    to_generate = []
    for ex in ds:
        if len(to_generate) >= args.num_examples:
            break
        if ex["question"] not in existing:
            to_generate.append(ex)

    print(f"Generating reasoning chains for {len(to_generate)} examples using {args.model}...")

    generated = 0
    failed = 0

    with open(raw_path, "a") as raw_f, open(sft_path, "a") as sft_f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for ex in to_generate:
                gt = extract_gsm8k_answer(ex["answer"])
                future = executor.submit(generate_chain, client, ex["question"], gt, args.model)
                futures[future] = ex

            for future in as_completed(futures):
                result = future.result()
                if result:
                    raw_f.write(json.dumps(result) + "\n")
                    raw_f.flush()

                    sft_example = format_for_sft(result)
                    sft_f.write(json.dumps(sft_example) + "\n")
                    sft_f.flush()

                    generated += 1
                else:
                    failed += 1

                total = generated + failed
                if total % 50 == 0:
                    print(f"  Progress: {generated} generated, {failed} failed, {total}/{len(to_generate)}")

    print(f"\nDone! Generated {generated} reasoning chains ({failed} failed)")
    print(f"Raw chains: {raw_path}")
    print(f"SFT data:   {sft_path}")


if __name__ == "__main__":
    main()
