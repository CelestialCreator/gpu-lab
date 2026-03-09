"""
Math answer verification reward functions for GRPO training.
Extracts numerical answers from model completions and compares to ground truth.
"""

import re


def extract_answer(text: str) -> str | None:
    """Extract the final numerical answer from a model completion.

    Handles formats like:
    - "#### 42"
    - "The answer is 42"
    - "\\boxed{42}"
    - Last number in the text as fallback
    """
    # Try #### format (GSM8K standard)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return _normalize_number(match.group(1))

    # Try \boxed{} format
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        inner = match.group(1).strip()
        # Try to extract number from boxed content
        num_match = re.search(r"-?[\d,]+\.?\d*", inner)
        if num_match:
            return _normalize_number(num_match.group(0))

    # Try "the answer is X" pattern
    match = re.search(
        r"(?:the answer is|answer:|final answer:?)\s*(-?[\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if match:
        return _normalize_number(match.group(1))

    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def _normalize_number(s: str) -> str:
    """Normalize a number string: remove commas, trailing zeros."""
    s = s.replace(",", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the ground truth answer from GSM8K answer field.
    GSM8K answers end with '#### <number>'."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return _normalize_number(match.group(1))
    # Fallback
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return _normalize_number(numbers[-1])
    return answer_text.strip()


def math_reward_fn(completions: list[list[dict]], ground_truths: list[str], **kwargs) -> list[float]:
    """Reward function for GRPOTrainer.

    Args:
        completions: List of completion message lists from the model.
                     Each element is [{"role": "assistant", "content": "..."}]
        ground_truths: List of ground truth answers (pre-extracted numbers).

    Returns:
        List of float rewards (1.0 for correct, 0.0 for incorrect).
    """
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        # Extract text from the completion
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        predicted = extract_answer(text)
        expected = _normalize_number(truth) if truth else None

        if predicted is not None and expected is not None and predicted == expected:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
    """Bonus reward for completions that use <think> tags and clear formatting.

    Gives partial credit for:
    - Using <think>...</think> tags (0.3)
    - Including a clear final answer marker (0.2)
    """
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
