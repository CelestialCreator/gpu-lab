# Multi-Token Prediction via Self-Distillation

**Status: Planned**

An investigation into predicting multiple tokens simultaneously during language model inference, breaking the one-token-at-a-time bottleneck of standard autoregressive generation.

---

## Background

Standard large language models generate text one token at a time. Each forward pass through the model produces a probability distribution over the vocabulary, a single token is sampled, and the process repeats. For a 500-token response, that means 500 sequential forward passes -- each one waiting for the previous token before it can begin.

**Multi-Token Prediction (MTP)** challenges this paradigm. Instead of predicting one token per forward pass, the model predicts multiple future tokens at once. If a model can accurately predict 4 tokens per step, inference throughput could theoretically approach a 4x improvement.

The core idea is conceptually simple, but the engineering and quality tradeoffs are where the real challenge lies.

## How Multi-Token Prediction Works

The approach described in the Meta research paper ("Better & Faster Large Language Models via Multi-token Prediction," Gloeckle et al., 2024) modifies the training objective:

1. **Standard training**: Given tokens `[t1, t2, t3, t4]`, the model learns to predict `t2` from `t1`, `t3` from `[t1, t2]`, and so on -- always the single next token.

2. **Multi-token training**: The model learns to predict `[t2, t3, t4, t5]` simultaneously from `t1`. Multiple prediction heads share the same trunk (transformer backbone), and the training loss is the sum of the individual next-token losses across all heads.

3. **Self-distillation**: The model's own predictions are used as soft targets to train the additional prediction heads, reducing the need for extra data or a separate teacher model.

At inference time, the additional prediction heads can be used speculatively -- predict multiple tokens, verify them with a single forward pass, and accept the ones that match. This is related to speculative decoding, but with the speculation built into the model itself rather than requiring a separate draft model.

## Goals for This Project

1. **Understand the technique**: Read and annotate the MTP paper, trace through the architecture changes required to add multiple prediction heads to a transformer

2. **Implement a minimal version**: Starting with a small model (likely a 125M-1B parameter range), add multi-token prediction heads and train with the modified objective

3. **Measure the speedup-quality tradeoff**: The central question is how much inference speed can be gained and at what cost to output quality. Specifically:
   - Tokens per second with 2, 4, and 8 prediction heads
   - Perplexity impact compared to single-token baseline
   - Acceptance rate of speculative tokens (what fraction of predicted tokens are actually correct?)
   - Memory overhead of the additional prediction heads

4. **Compare with speculative decoding**: Multi-token prediction builds the draft model into the main model. How does this compare to external speculative decoding with a separate small draft model?

## References

- Gloeckle, F., Idrissi, B. Y., Roziere, B., Lopez-Paz, D., & Synnaeve, G. (2024). *Better & Faster Large Language Models via Multi-token Prediction*. arXiv:2404.19737.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
- DeepSeek-AI. (2024). *DeepSeek-V3 Technical Report*. (Uses multi-token prediction in production at scale.)

## Current Status

This project is in the planning stage. The directory will be populated with configuration files, training scripts, evaluation benchmarks, and results as work progresses.

```
04-multi-token-prediction/
  (empty -- work has not yet begun)
```

The plan is to start with a thorough reading of the MTP paper and the DeepSeek-V3 technical report (which deployed MTP at production scale), then implement the training objective modification, and finally benchmark inference throughput against baselines.
