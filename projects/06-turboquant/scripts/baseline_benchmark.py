#!/usr/bin/env python3
"""
Baseline benchmark for Qwen3.5-27B via llama-server.
Measures tok/s and response quality at various context lengths.
This gives us "before" numbers for comparison with TurboQuant KV cache.
"""

import json
import time
import sys
import subprocess
import requests

API_URL = "http://localhost:30080"

def get_gpu_memory():
    """Get current GPU memory usage via nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        gpus.append({
            "name": parts[0],
            "used_mb": int(parts[1]),
            "total_mb": int(parts[2]),
            "free_mb": int(parts[3]),
        })
    return gpus

def generate(prompt, max_tokens=256, temperature=0.0):
    """Send completion request and measure timing."""
    payload = {
        "model": "qwen3.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    t0 = time.time()
    try:
        resp = requests.post(f"{API_URL}/v1/chat/completions", json=payload, timeout=300)
        elapsed = time.time() - t0

        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "elapsed": elapsed}

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        timings = data.get("timings", {})

        # Qwen3.5 uses reasoning_content for thinking, content for final answer
        msg = choice["message"]
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")

        return {
            "content": content,
            "reasoning": reasoning[:200] if reasoning else "",
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "elapsed": elapsed,
            "tok_per_sec": timings.get("predicted_per_second",
                           usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0),
            "prefill_tok_per_sec": timings.get("prompt_per_second", 0),
        }
    except requests.exceptions.Timeout:
        return {"error": "TIMEOUT", "elapsed": time.time() - t0}
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - t0}

def build_context_prompt(target_tokens):
    """Build a prompt with approximately target_tokens of context."""
    # Use a repeating pattern to fill context, then ask a question
    filler = "The quick brown fox jumps over the lazy dog. " * 20  # ~200 tokens per repeat
    repeats = max(1, target_tokens // 200)

    # Insert a "needle" fact in the middle
    needle_pos = repeats // 2
    parts = []
    for i in range(repeats):
        if i == needle_pos:
            parts.append("IMPORTANT FACT: The secret project codename is 'TurboQuant-7X'. Remember this. ")
        parts.append(filler)

    parts.append("\nBased on the text above, what is the secret project codename? Answer in one short sentence.")
    return "".join(parts)

def main():
    print("=" * 70)
    print("Qwen3.5-27B Baseline Benchmark (Q8_0 KV Cache)")
    print("=" * 70)

    # GPU state before
    print("\nGPU Memory (before benchmarks):")
    for gpu in get_gpu_memory():
        print(f"  {gpu['name']}: {gpu['used_mb']}MB / {gpu['total_mb']}MB ({gpu['free_mb']}MB free)")

    results = []

    # Test 1: Short context — generation speed
    print("\n--- Test 1: Short context generation speed ---")
    prompt = "Write a Python function to compute fibonacci numbers efficiently using memoization. Include a docstring."
    res = generate(prompt, max_tokens=1024, temperature=0.0)
    if "error" not in res:
        print(f"  Prompt tokens:      {res['prompt_tokens']}")
        print(f"  Completion tokens:  {res['completion_tokens']}")
        print(f"  Time:               {res['elapsed']:.2f}s")
        print(f"  Decode speed:       {res['tok_per_sec']:.1f} tok/s")
        print(f"  Prefill speed:      {res['prefill_tok_per_sec']:.1f} tok/s")
        results.append({"test": "short_gen", "ctx": res['prompt_tokens'], **res})
    else:
        print(f"  ERROR: {res['error']}")

    # Test 2: Increasing context lengths — needle in haystack
    print("\n--- Test 2: Context length scaling (needle-in-haystack) ---")
    for target_ctx in [1000, 4000, 8000, 16000, 32000, 64000]:
        print(f"\n  Target context: ~{target_ctx} tokens")
        prompt = build_context_prompt(target_ctx)

        # Capture GPU memory during generation
        res = generate(prompt, max_tokens=512, temperature=0.0)
        gpus = get_gpu_memory()

        if "error" not in res:
            # Check both content and reasoning for needle
            full_text = (res['content'] + " " + res.get('reasoning', '')).lower()
            found_needle = "turboquant" in full_text or "7x" in full_text
            print(f"    Actual prompt tokens: {res['prompt_tokens']}")
            print(f"    Decode speed:        {res['tok_per_sec']:.1f} tok/s")
            print(f"    Prefill speed:       {res['prefill_tok_per_sec']:.1f} tok/s")
            print(f"    Needle found:        {'YES' if found_needle else 'NO'}")
            print(f"    GPU 5090:            {gpus[2]['used_mb']}MB / {gpus[2]['total_mb']}MB")
            print(f"    GPU 3080:            {gpus[0]['used_mb']}MB / {gpus[0]['total_mb']}MB")
            answer = res['content'][:100] if res['content'] else "(in reasoning only)"
            print(f"    Response:            {answer}")
            results.append({
                "test": "needle", "target_ctx": target_ctx,
                "needle_found": found_needle,
                "gpu_5090_used": gpus[2]['used_mb'],
                "gpu_3080_used": gpus[0]['used_mb'],
                **res
            })
        else:
            print(f"    ERROR: {res['error']}")
            results.append({"test": "needle", "target_ctx": target_ctx, "error": res['error']})
            if "TIMEOUT" in str(res.get('error', '')):
                print("    Stopping — server may be OOM")
                break

    # Test 3: Math reasoning (baseline for quality comparison)
    print("\n--- Test 3: Math reasoning quality ---")
    math_problems = [
        ("What is 247 * 83?", "20501"),
        ("If a train travels 120 km in 1.5 hours, what is its speed in km/h?", "80"),
        ("A rectangle has area 48 and width 6. What is its perimeter?", "28"),
    ]

    correct = 0
    for problem, expected in math_problems:
        res = generate(f"Solve this: {problem} Give only the final number.", max_tokens=1024, temperature=0.0)
        if "error" not in res:
            is_correct = expected in res['content'] or expected in res.get('reasoning', '')
            correct += int(is_correct)
            print(f"  Q: {problem}")
            print(f"  A: {res['content'][:80]}  {'CORRECT' if is_correct else 'WRONG (expected: ' + expected + ')'}")

    print(f"\n  Math accuracy: {correct}/{len(math_problems)}")

    # Save results
    output_path = "/home/akshay/gpu-lab/projects/06-turboquant/benchmarks/baseline_27b.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Final GPU state
    print("\nGPU Memory (after benchmarks):")
    for gpu in get_gpu_memory():
        print(f"  {gpu['name']}: {gpu['used_mb']}MB / {gpu['total_mb']}MB ({gpu['free_mb']}MB free)")

if __name__ == "__main__":
    main()
