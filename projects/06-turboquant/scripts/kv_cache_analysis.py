#!/usr/bin/env python3
"""
Day 2: Apply TurboQuant to real Qwen3.5 KV cache tensors.
Runs inside the GRPO container with torch + transformers.
Uses Qwen3.5-0.8B (same architecture as 27B, same head_dim=128).

Measures:
- MSE of quantized vs original KV cache
- Cosine similarity of attention outputs
- Memory savings
- Per-layer analysis
"""

import torch
import numpy as np
import json
import time
import os
from scipy.special import gamma
from scipy.integrate import quad

# ============================================================
# TurboQuant implementation (same as notebook 01, PyTorch version)
# ============================================================

def beta_pdf(x, d):
    coeff = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    return coeff * (1 - x**2) ** ((d - 3) / 2)

def lloyd_max_quantizer(d, bits, n_iters=100):
    n_levels = 2 ** bits
    pdf = lambda x: beta_pdf(x, d)
    sigma = 1.0 / np.sqrt(d)
    lo, hi = -4 * sigma, 4 * sigma
    boundaries = np.linspace(lo, hi, n_levels + 1)
    levels = np.zeros(n_levels)
    for _ in range(n_iters):
        for i in range(n_levels):
            a, b_bound = boundaries[i], boundaries[i + 1]
            num, _ = quad(lambda x: x * pdf(x), a, b_bound)
            den, _ = quad(pdf, a, b_bound)
            levels[i] = num / den if den > 1e-15 else (a + b_bound) / 2
        for i in range(1, n_levels):
            boundaries[i] = (levels[i - 1] + levels[i]) / 2
    return boundaries, levels


class TurboQuantTorch:
    """TurboQuant MSE-only (Stage 1) in PyTorch for GPU execution."""

    def __init__(self, d, bits, device='cuda'):
        self.d = d
        self.bits = bits
        self.device = device

        # Compute Lloyd-Max codebook (numpy, then convert)
        boundaries_np, levels_np = lloyd_max_quantizer(d, bits)
        self.boundaries = torch.tensor(boundaries_np, dtype=torch.float32, device=device)
        self.levels = torch.tensor(levels_np, dtype=torch.float32, device=device)

        # Random orthogonal rotation matrix
        G = torch.randn(d, d, device=device)
        Q, R = torch.linalg.qr(G)
        self.Pi = Q @ torch.diag(torch.sign(torch.diag(R)))

    def quantize(self, x):
        """x: (..., d) tensor. Returns indices, norms."""
        shape = x.shape
        x_flat = x.reshape(-1, self.d)

        norms = torch.norm(x_flat, dim=1, keepdim=True)
        x_unit = x_flat / (norms + 1e-10)

        # Rotate
        y = x_unit @ self.Pi.T

        # Scalar quantize (bucketize)
        indices = torch.bucketize(y, self.boundaries[1:-1])
        indices = indices.clamp(0, 2**self.bits - 1)

        return indices, norms.squeeze(-1), shape

    def dequantize(self, indices, norms, shape):
        """Reconstruct from indices."""
        y_hat = self.levels[indices]
        x_hat = y_hat @ self.Pi
        x_hat = x_hat * norms.unsqueeze(-1)
        return x_hat.reshape(shape)

    def round_trip(self, x):
        """Quantize then dequantize."""
        indices, norms, shape = self.quantize(x)
        return self.dequantize(indices, norms, shape)


# ============================================================
# Extract real KV cache from Qwen3.5-0.8B
# ============================================================

def extract_kv_cache(model, tokenizer, text, device='cuda'):
    """Run model forward pass and extract KV cache tensors."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False, use_cache=True)

    past_kv = outputs.past_key_values
    return past_kv, inputs['input_ids'].shape[1]


def analyze_kv_cache(past_kv, bits_list=[3, 4], device='cuda'):
    """Apply TurboQuant to each layer's K and V, measure quality."""
    results = []

    n_layers = len(past_kv)
    # Get dimensions from first layer
    k_sample = past_kv[0][0]  # (batch, n_heads, seq_len, head_dim)
    head_dim = k_sample.shape[-1]
    n_heads = k_sample.shape[1]
    seq_len = k_sample.shape[2]

    print(f"KV Cache shape: {n_layers} layers, {n_heads} heads, seq_len={seq_len}, head_dim={head_dim}")

    for bits in bits_list:
        print(f"\n{'='*60}")
        print(f"TurboQuant {bits}-bit analysis")
        print(f"{'='*60}")

        tq = TurboQuantTorch(head_dim, bits, device=device)

        layer_mse_k, layer_mse_v = [], []
        layer_cosine_k, layer_cosine_v = [], []
        total_orig_bytes = 0
        total_quant_bytes = 0

        for layer_idx in range(n_layers):
            K = past_kv[layer_idx][0].squeeze(0).float()  # (n_heads, seq_len, head_dim)
            V = past_kv[layer_idx][1].squeeze(0).float()

            # Reshape for TurboQuant: (n_heads * seq_len, head_dim)
            K_flat = K.reshape(-1, head_dim)
            V_flat = V.reshape(-1, head_dim)

            # Quantize + dequantize
            K_recon = tq.round_trip(K_flat).reshape(K.shape)
            V_recon = tq.round_trip(V_flat).reshape(V.shape)

            # MSE (normalized)
            k_mse = torch.mean((K - K_recon)**2).item() / (torch.mean(K**2).item() + 1e-10)
            v_mse = torch.mean((V - V_recon)**2).item() / (torch.mean(V**2).item() + 1e-10)
            layer_mse_k.append(k_mse)
            layer_mse_v.append(v_mse)

            # Cosine similarity
            k_cos = torch.nn.functional.cosine_similarity(K_flat, K_recon.reshape(-1, head_dim), dim=1).mean().item()
            v_cos = torch.nn.functional.cosine_similarity(V_flat, V_recon.reshape(-1, head_dim), dim=1).mean().item()
            layer_cosine_k.append(k_cos)
            layer_cosine_v.append(v_cos)

            # Memory
            orig_bytes = K.numel() * 2 + V.numel() * 2  # FP16 = 2 bytes
            quant_bytes = (K.numel() * bits / 8 + V.numel() * bits / 8 +
                          K_flat.shape[0] * 4 * 2)  # norms as FP32
            total_orig_bytes += orig_bytes
            total_quant_bytes += quant_bytes

            if layer_idx % 6 == 0 or layer_idx == n_layers - 1:
                print(f"  Layer {layer_idx:2d}: K_mse={k_mse:.6f} V_mse={v_mse:.6f} "
                      f"K_cos={k_cos:.6f} V_cos={v_cos:.6f}")

        avg_k_mse = np.mean(layer_mse_k)
        avg_v_mse = np.mean(layer_mse_v)
        avg_k_cos = np.mean(layer_cosine_k)
        avg_v_cos = np.mean(layer_cosine_v)
        compression = total_orig_bytes / total_quant_bytes

        print(f"\n  Summary ({bits}-bit):")
        print(f"    Avg K MSE (normalized): {avg_k_mse:.6f}")
        print(f"    Avg V MSE (normalized): {avg_v_mse:.6f}")
        print(f"    Avg K cosine sim:       {avg_k_cos:.6f}")
        print(f"    Avg V cosine sim:       {avg_v_cos:.6f}")
        print(f"    Memory: {total_orig_bytes/1e6:.1f} MB → {total_quant_bytes/1e6:.1f} MB ({compression:.1f}x)")

        results.append({
            'bits': bits,
            'avg_k_mse': avg_k_mse,
            'avg_v_mse': avg_v_mse,
            'avg_k_cosine': avg_k_cos,
            'avg_v_cosine': avg_v_cos,
            'orig_mb': total_orig_bytes / 1e6,
            'quant_mb': total_quant_bytes / 1e6,
            'compression_ratio': compression,
            'per_layer_k_mse': layer_mse_k,
            'per_layer_v_mse': layer_mse_v,
        })

    return results


def attention_fidelity_test(past_kv, bits, device='cuda'):
    """
    Test: how much do attention outputs change when KV cache is quantized?
    Simulates full attention computation with original vs quantized KV.
    """
    print(f"\n{'='*60}")
    print(f"Attention Fidelity Test ({bits}-bit TurboQuant)")
    print(f"{'='*60}")

    n_layers = len(past_kv)
    head_dim = past_kv[0][0].shape[-1]
    n_heads = past_kv[0][0].shape[1]
    seq_len = past_kv[0][0].shape[2]

    tq = TurboQuantTorch(head_dim, bits, device=device)

    output_cosines = []

    for layer_idx in range(n_layers):
        K = past_kv[layer_idx][0].squeeze(0).float()  # (n_heads, seq_len, head_dim)
        V = past_kv[layer_idx][1].squeeze(0).float()

        # Simulate a new query attending to cached KV
        Q = torch.randn(n_heads, 1, head_dim, device=device)

        # Original attention
        scores_orig = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_orig = torch.softmax(scores_orig, dim=-1)
        out_orig = torch.matmul(attn_orig, V)

        # Quantized KV attention
        K_flat = K.reshape(-1, head_dim)
        V_flat = V.reshape(-1, head_dim)
        K_q = tq.round_trip(K_flat).reshape(K.shape)
        V_q = tq.round_trip(V_flat).reshape(V.shape)

        scores_q = torch.matmul(Q, K_q.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_q = torch.softmax(scores_q, dim=-1)
        out_q = torch.matmul(attn_q, V_q)

        # Cosine similarity of attention outputs
        cos = torch.nn.functional.cosine_similarity(
            out_orig.reshape(-1), out_q.reshape(-1), dim=0
        ).item()
        output_cosines.append(cos)

    avg_cos = np.mean(output_cosines)
    min_cos = np.min(output_cosines)
    print(f"  Attention output cosine similarity:")
    print(f"    Mean: {avg_cos:.6f}")
    print(f"    Min:  {min_cos:.6f}")
    print(f"    Per-layer range: [{min_cos:.6f}, {np.max(output_cosines):.6f}]")

    return {'bits': bits, 'mean_cosine': avg_cos, 'min_cosine': min_cos,
            'per_layer': output_cosines}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3.5-0.8B"
    cache_dir = os.environ.get("HF_HOME", "/workspace/hf_cache")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device).eval()

    print(f"Model loaded on {device}")
    if device == 'cuda':
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # Use a realistic prompt for KV cache extraction
    test_text = """You are a helpful AI assistant. The user has asked you to solve
a complex mathematical problem involving calculus and linear algebra.

Problem: Find the eigenvalues of the matrix A = [[3, 1, 0], [1, 3, 1], [0, 1, 3]]
and determine if A is positive definite.

Solution: Let me work through this step by step.

First, we need to find the eigenvalues by solving det(A - λI) = 0.

The characteristic polynomial is:
det(A - λI) = (3-λ)[(3-λ)(3-λ) - 1] - 1[(3-λ) - 0] + 0
= (3-λ)[(3-λ)² - 1] - (3-λ)
= (3-λ)[(3-λ)² - 1 - 1]
= (3-λ)[(3-λ)² - 2]

Setting this to zero:
(3-λ) = 0, so λ₁ = 3
(3-λ)² = 2, so λ₂ = 3-√2, λ₃ = 3+√2

All eigenvalues are positive (3-√2 ≈ 1.586, 3, 3+√2 ≈ 4.414), so A is positive definite."""

    # Extract KV cache
    print(f"\nExtracting KV cache...")
    past_kv, seq_len = extract_kv_cache(model, tokenizer, test_text, device)
    print(f"Sequence length: {seq_len} tokens")

    # Analyze with different bit-widths
    results = analyze_kv_cache(past_kv, bits_list=[2, 3, 4], device=device)

    # Attention fidelity tests
    attn_results = []
    for bits in [3, 4]:
        attn_res = attention_fidelity_test(past_kv, bits, device=device)
        attn_results.append(attn_res)

    # Save results
    output = {
        'model': model_name,
        'seq_len': seq_len,
        'quantization_results': results,
        'attention_fidelity': attn_results,
    }

    output_path = "/workspace/turboquant_kv_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
