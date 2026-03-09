"""
SFT warmup: fine-tune Qwen3.5-0.8B on Claude-generated reasoning chains.

This is a short SFT phase to give the model a "reasoning template" before
GRPO training. Only 1 epoch on ~1000 high-quality examples — enough to
teach the format without overfitting.
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_sft_dataset(path: str) -> Dataset:
    """Load the JSONL SFT dataset produced by generate_sft_data.py."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(description="SFT warmup on reasoning chains")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--sft_data", type=str, default="/workspace/sft-data/sft_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="/workspace/sft-output")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    args = parser.parse_args()

    print(f"=== SFT Warmup ===")
    print(f"Model: {args.model_name}")
    print(f"Data: {args.sft_data}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading SFT dataset...")
    dataset = load_sft_dataset(args.sft_data)
    print(f"Training examples: {len(dataset)}")

    # Config — TRL 0.29+ moved max_seq_length out of SFTConfig
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="sft-warmup-qwen35-0.8b",
        warmup_ratio=0.05,
    )

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")

    # Train
    print("Starting SFT...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"SFT model saved to {final_dir}")


if __name__ == "__main__":
    main()
