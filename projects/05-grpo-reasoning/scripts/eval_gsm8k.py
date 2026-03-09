"""
GSM8K evaluation script for measuring reasoning accuracy.
Supports both baseline and GRPO-trained models.
Uses 8-shot chain-of-thought prompting (standard GSM8K eval protocol).
"""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a helpful assistant that thinks step by step. "
    "Show your reasoning inside <think> tags before giving your final answer. "
    "End math answers with: #### <number>"
)

# 8 few-shot exemplars (subset of GSM8K train, standard practice)
FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.\n#### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops.\n#### 8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 + 2 = 4 more toys. 5 + 4 = 9.\n#### 9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29.\n#### 29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each is 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left.\n#### 8",
    },
]


def extract_answer(text: str) -> str | None:
    """Extract numerical answer from model output."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return _normalize(match.group(1))
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        num = re.search(r"-?[\d,]+\.?\d*", match.group(1))
        if num:
            return _normalize(num.group(0))
    match = re.search(r"(?:the answer is|answer:)\s*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return _normalize(match.group(1))
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return _normalize(numbers[-1])
    return None


def _normalize(s: str) -> str:
    s = s.replace(",", "")
    try:
        val = float(s)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return s


def extract_gsm8k_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return _normalize(match.group(1))
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return _normalize(numbers[-1])
    return answer_text.strip()


def build_few_shot_messages(question: str) -> list[dict]:
    """Build 8-shot CoT prompt in chat format."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def build_zero_shot_messages(question: str) -> list[dict]:
    """Build zero-shot prompt in chat format."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HF model name or local path")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of test samples to evaluate (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--zero_shot", action="store_true",
                        help="Use zero-shot instead of 8-shot CoT")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print(f"=== GSM8K Evaluation ===")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'zero-shot' if args.zero_shot else '8-shot CoT'}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda()
    model.eval()

    # Load test set
    print("Loading GSM8K test set...")
    test_ds = load_dataset("gsm8k", "main", split="test")
    if args.num_samples:
        test_ds = test_ds.select(range(min(args.num_samples, len(test_ds))))
    print(f"Evaluating on {len(test_ds)} examples")

    build_messages = build_zero_shot_messages if args.zero_shot else build_few_shot_messages

    correct = 0
    total = 0
    results = []
    start_time = time.time()

    # Process in batches
    for i in tqdm(range(0, len(test_ds), args.batch_size), desc="Evaluating"):
        batch = test_ds[i : i + args.batch_size]
        questions = batch["question"]
        answers = batch["answer"]

        # Build prompts
        all_messages = [build_messages(q) for q in questions]
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]

        # Tokenize
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the new tokens
        for j in range(len(questions)):
            input_len = inputs["input_ids"][j].shape[0]
            generated = outputs[j][input_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True)

            predicted = extract_answer(response)
            expected = extract_gsm8k_answer(answers[j])
            is_correct = predicted is not None and predicted == expected

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": questions[j],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "response": response[:500],  # truncate for storage
            })

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0

    print(f"\n=== Results ===")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.1f}s per example)")

    # Save results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        summary = {
            "model": args.model_name,
            "mode": "zero-shot" if args.zero_shot else "8-shot-cot",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "time_seconds": elapsed,
            "examples": results,
        }
        with open(args.output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
