---
license: creativeml-openrail-m
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags:
  - text-to-image
  - stable-diffusion-xl
  - lora
  - character
pipeline_tag: text-to-image
---

# My Character — SDXL LoRA

A LoRA adapter for [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) trained on a custom character dataset.

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | stabilityai/stable-diffusion-xl-base-1.0 |
| **LoRA Rank** | 32 |
| **LoRA Alpha** | 32 |
| **Training Steps** | 10,000 |
| **Learning Rate** | 5e-5 |
| **Optimizer** | AdamW 8-bit |
| **Precision** | bf16 |
| **Dataset** | 17 images with text captions |
| **Trigger Word** | `TRIGGER_WORD` |

## Usage

### ComfyUI

1. Place the `.safetensors` file in `ComfyUI/models/loras/`
2. Use the **LoraLoader** node:
   - **lora_name**: `my-character.safetensors`
   - **strength_model**: 0.7 - 1.0
   - **strength_clip**: 0.7 - 1.0
3. Include the trigger word in your prompt

### Diffusers

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("your-username/my-character-sdxl-lora")

image = pipe(
    "a photo of TRIGGER_WORD in a garden, natural lighting",
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]
```

## Training Details

- **Training Tool**: [AI-Toolkit](https://github.com/ostris/ai-toolkit) by ostris
- **Hardware**: NVIDIA RTX 5090 (32 GB VRAM)
- **Dataset**: 17 AI-generated character images with Florence2 auto-captions
- **Caption Format**: `TRIGGER_WORD, description of the image content`
- **Caption Dropout**: 5%
- **Resolutions**: 512, 768, 1024

The full training config and dataset details are available in the [gpu-lab repository](../../projects/01-lora-training/).

## Dataset

The training dataset was generated using ComfyUI with a Qwen Image Edit pipeline, covering:
- Full body (turnaround, t-pose, walking, sitting, standing)
- Portraits (front-facing, close-up)
- Expressions (happy, surprised, angry, sad, laughing, contemplative)

## Limitations

- Trained on AI-generated images, not photographs — style may lean synthetic
- Small dataset (17 images) — may not generalize to all poses/contexts
- Best results at LoRA strength 0.7-0.9; higher values may cause overfitting artifacts

## License

This model is released under the CreativeML Open RAIL-M license, consistent with the SDXL base model license.
