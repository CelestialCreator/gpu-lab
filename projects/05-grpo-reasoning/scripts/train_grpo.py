"""
GRPO training script for teaching Qwen3.5-0.8B to reason on math problems.

Uses HuggingFace TRL's GRPOTrainer with verifiable math rewards (RLVR).
Designed to run on a single RTX 5090 (32GB VRAM).
"""

import argparse
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward import extract_answer, _normalize_number


SYSTEM_PROMPT = (
    "You are a helpful assistant that thinks step by step. "
    "Show your reasoning inside <think> tags before giving your final answer. "
    "End math answers with: #### <number>"
)


def build_prompt(question: str) -> list[dict]:
    """Build a chat-format prompt for the model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract ground truth number from GSM8K answer field."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return _normalize_number(match.group(1))
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return _normalize_number(numbers[-1])
    return answer_text.strip()


def math_reward(completions: list[list[dict]], ground_truths: list[str], **kwargs) -> list[float]:
    """Reward function: 1.0 if answer matches ground truth, else 0.0."""
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        predicted = extract_answer(text)
        expected = _normalize_number(truth)

        if predicted is not None and predicted == expected:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Bonus reward for step-by-step reasoning and clear answer format."""
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        reward = 0.0
        # Reward using <think> tags for reasoning
        if "<think>" in text and "</think>" in text:
            reward += 0.3
        # Reward clear answer formatting
        if re.search(r"(####|\\boxed|[Tt]he answer is)", text):
            reward += 0.2
        rewards.append(reward)
    return rewards


def prepare_dataset(tokenizer, dataset_name="gsm8k", split="train"):
    """Load and format GSM8K for GRPO training."""
    ds = load_dataset(dataset_name, "main", split=split)

    def format_example(example):
        return {
            "prompt": build_prompt(example["question"]),
            "ground_truths": extract_gsm8k_answer(example["answer"]),
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds


def main():
    parser = argparse.ArgumentParser(description="GRPO training on GSM8K")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output_dir", type=str, default="/workspace/grpo-output")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--use_format_reward", action="store_true",
                        help="Add format reward in addition to correctness reward")
    parser.add_argument("--wandb_project", type=str, default="grpo-qwen35-math")
    args = parser.parse_args()

    print(f"=== GRPO Training Configuration ===")
    print(f"Model: {args.model_name}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Max completion length: {args.max_completion_length}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load tokenizer
    print("=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("=== Loading GSM8K dataset ===")
    train_dataset = prepare_dataset(tokenizer)
    print(f"Training examples: {len(train_dataset)}")

    # Build reward functions list
    reward_fns = [math_reward]
    if args.use_format_reward:
        reward_fns.append(format_reward)
        print("Using math_reward + format_reward")
    else:
        print("Using math_reward only")

    # Configure GRPO
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"grpo-{args.model_name.split('/')[-1]}",
        log_on_each_node=False,
        # GRPO specific
        beta=0.04,  # KL penalty coefficient
        # Memory optimization
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
    )

    # Load model
    print("=== Loading model ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")

    # Create trainer
    print("=== Initializing GRPOTrainer ===")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("=== Starting GRPO training ===")
    trainer.train()

    # Save final model
    print("=== Saving final model ===")
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")

    print("=== Training complete ===")


if __name__ == "__main__":
    main()
