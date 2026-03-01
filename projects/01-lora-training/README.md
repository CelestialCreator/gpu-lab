# Training a Character LoRA on a Home GPU Lab

How I trained a custom character LoRA adapter for Stable Diffusion, learned the hard way that VRAM is not the only memory that matters, and built a repeatable pipeline from dataset creation to inference.

---

## The Goal

Train a LoRA (Low-Rank Adaptation) model that teaches Stable Diffusion to generate images of a specific character. The character does not exist in the base model's training data, so the model needs to learn a new visual concept from a small set of reference images.

The hardware: an NVIDIA RTX 5090 with 32GB of VRAM, running inside a Kubernetes cluster on a machine with 16GB of system RAM.

## The FLUX.1-dev Attempt (and Why It Failed)

FLUX.1-dev from Black Forest Labs was the obvious first choice. It is one of the most capable open image generation models available, with approximately 12 billion parameters and outstanding prompt adherence. On paper, the RTX 5090's 32GB of VRAM should handle it.

The training config (`configs/flux-dev-failed.yaml`) was set up with every memory optimization available:

- **Quantization**: `qfloat8` to reduce weight precision
- **Low VRAM mode**: `low_vram: true` to offload layers to system RAM during training
- **8-bit optimizer**: `adamw8bit` to halve optimizer state memory
- **Gradient checkpointing**: enabled to trade compute for memory

None of it mattered. The failure happened before training even started.

### The Problem: System RAM, Not VRAM

When a model loads, the weights first pass through system RAM before being transferred to the GPU. FLUX.1-dev's 24GB of weights need to be deserialized on the CPU side, which requires 20-30GB of system RAM even temporarily. On a machine with only 16GB of RAM, this immediately pushed the system into swap.

The consequences were severe. The Linux kernel's swap thrashing did not just slow down training -- it froze the entire machine. The Kubernetes API server became unresponsive. Websocket connections to the training pod timed out. For several minutes, the cluster was completely unreachable.

### The Debugging Journey

The initial symptoms were misleading:

1. **Websocket timeouts** when connecting to the training pod -- looked like a networking issue
2. **Kubernetes API server unresponsive** -- looked like a cluster infrastructure problem
3. **Pod showing "Running" but producing no output** -- looked like a code bug

The actual root cause only became clear after SSH-ing directly into the node and watching `htop`. System RAM was pegged at 100%, swap was being thrashed at maximum I/O bandwidth, and the OOM killer had not yet triggered because the process was technically within limits (swap counts as available memory).

Two key learnings emerged:

- Setting `vm.swappiness=10` (down from the default 60) reduces the kernel's eagerness to swap, making OOM kills happen faster and more cleanly instead of the slow death spiral of swap thrashing.
- The `low_vram=true` flag is critical for large models -- but it only helps with VRAM. The initial CPU-side model loading still needs enough system RAM to hold the full weights.

## Pivoting to SDXL

With FLUX.1-dev ruled out for the current hardware, I pivoted to Stable Diffusion XL (SDXL). Here is how the three main model families compare:

| Model | Parameters | Model Size | Minimum VRAM | System RAM Needed |
|---|---|---|---|---|
| SD 1.5 | ~860M | ~2GB | ~4GB | ~8GB |
| **SDXL** | **~3.5B** | **~7GB** | **~8GB** | **~12GB** |
| FLUX.1-dev | ~12B | ~24GB | ~16GB+ | ~20-30GB |

SDXL sits in the sweet spot: large enough to produce high-quality 1024x1024 images, small enough to train comfortably on a single GPU with 16GB of system RAM. The base model (`stabilityai/stable-diffusion-xl-base-1.0`) loads without any memory pressure.

## Training Strategy: Three Iterations

Rather than guessing at hyperparameters and running a single long training job, I structured the work as three progressive iterations. Each config file is in the `configs/` directory with full inline documentation.

### Test Run: 500 Steps (`configs/sdxl-test-500steps.yaml`)

The first priority was validating that the entire pipeline works end-to-end.

- **Steps**: 500 (enough to see if the model is learning, not enough for quality)
- **LoRA rank**: 16 (lower rank = fewer trainable parameters = faster iteration)
- **Batch size**: 1
- **Learning rate**: 1e-4
- **Resolution**: 512, 768 (multi-resolution training)
- **Sampler**: ddpm with guidance scale 7.5

This run completed in minutes and confirmed that data loading, latent caching, training, checkpointing, and sample generation all worked correctly. The sample images showed early signs of learning the character's features.

### v2: 3000 Steps (`configs/sdxl-v2-3000steps.yaml`)

With the pipeline validated, the second run pushed for quality.

- **Steps**: 3000
- **LoRA rank**: 32 (doubled from test run for better detail capture)
- **Gradient accumulation**: 2 (effective batch size of 2 without doubling memory)
- **Learning rate**: 5e-5 (halved -- higher rank benefits from lower LR)
- **Resolution**: 512, 768, 1024

The increased rank and steps produced noticeably better likeness. Gradient accumulation smoothed out the training signal without requiring more VRAM.

### v3: 10000 Steps (`configs/sdxl-v3-10000steps.yaml`)

The production run with all optimizations from previous iterations.

- **Steps**: 10000
- **LoRA rank**: 32
- **Gradient accumulation**: 2
- **Learning rate**: 5e-5
- **Resolution**: 512, 768, 1024 (three resolution buckets)
- **Save every**: 1000 steps (with max 4 checkpoint saves retained)

This run produced the final model with strong likeness capture and good prompt adherence.

## The Dataset

The training dataset consists of 17 images covering a variety of poses, angles, and expressions. Rather than using photographs, the images were generated using a ComfyUI pipeline built around the Qwen Image Edit model (see [project 02: Dataset Creation](../02-dataset-creation/)).

Each image is paired with a `.txt` caption file. The captions follow the format:

```
TRIGGER_WORD, description of the image content
```

The trigger word teaches the model to associate a unique token with the character's appearance. At inference time, including the trigger word in the prompt activates the LoRA's learned features.

Auto-captioning was done with Florence2 (`microsoft/Florence-2-base`), which generates detailed descriptions of image content. The trigger word was prepended programmatically. See the `dataset/` directory for all caption files.

## Configs

All four training configuration files are in the `configs/` directory:

| Config | Model | Steps | Rank | Status |
|---|---|---|---|---|
| `flux-dev-failed.yaml` | FLUX.1-dev | 1500 | 32 | Failed (OOM) |
| `sdxl-test-500steps.yaml` | SDXL | 500 | 16 | Success (validation) |
| `sdxl-v2-3000steps.yaml` | SDXL | 3000 | 32 | Success (improved) |
| `sdxl-v3-10000steps.yaml` | SDXL | 10000 | 32 | Success (production) |

Each YAML file includes detailed header comments explaining the rationale for that run's settings. The configs use the [ai-toolkit](https://github.com/ostris/ai-toolkit) format.

## Using the LoRA in ComfyUI

The trained LoRA adapter is a `.safetensors` file that can be loaded in ComfyUI using the **LoraLoader** node. The inference pipeline is:

```
CheckpointLoaderSimple (SDXL base)
    --> LoraLoader (character LoRA)
        --> CLIPTextEncode (prompt with trigger word)
            --> KSampler
                --> VAEDecode
                    --> SaveImage
```

Recommended inference settings:

- **LoRA strength**: 0.7 - 1.0 (start at 0.8, adjust to taste)
- **Steps**: 25
- **CFG scale**: 7.5
- **Sampler**: euler
- **Prompt**: Include the trigger word followed by the scene description

Higher LoRA strength produces stronger likeness at the cost of reduced prompt flexibility. Lower strength blends the character more naturally into diverse scenes.

## Key Lessons

**Start small, validate pipeline before long runs.** The 500-step test run caught configuration issues in minutes instead of hours. Running 10000 steps with a broken data pipeline would have been a significant waste of GPU time.

**Understand the full memory hierarchy.** GPU training involves three distinct memory pools: system RAM (model loading, data preprocessing), VRAM (model weights, activations, optimizer states), and swap (overflow). A bottleneck in any one of them can be catastrophic, and the symptoms often point to the wrong layer.

**LoRA rank is a quality-vs-efficiency tradeoff.** Rank 16 was sufficient for validation but rank 32 captured finer details. For a 17-image dataset of a single character, rank 32 was the sweet spot -- higher ranks showed diminishing returns.

**Synthetic training data works.** Generating the training dataset with AI (via Qwen Image Edit) instead of using real photographs produced a LoRA that generalizes well. The consistency of the AI-generated images may actually help, since there is less noise from varying lighting, camera settings, and backgrounds.

---

## Directory Structure

```
01-lora-training/
  configs/
    flux-dev-failed.yaml        # FLUX.1-dev config (OOM failure)
    sdxl-test-500steps.yaml     # SDXL validation run
    sdxl-v2-3000steps.yaml      # SDXL improved run
    sdxl-v3-10000steps.yaml     # SDXL production run
  dataset/
    captions/                   # 17 .txt caption files (committed)
    *.png                       # 17 training images (gitignored)
  results/
    *.safetensors               # Trained LoRA adapter (gitignored)
```
