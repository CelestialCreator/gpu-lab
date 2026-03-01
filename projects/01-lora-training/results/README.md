# Training Results

The SDXL v3 run (10,000 steps) produced the final character LoRA adapter with strong likeness capture and good prompt adherence.

---

## Model Summary

| Property | Value |
|---|---|
| Base model | Stable Diffusion XL 1.0 |
| Adapter type | LoRA (.safetensors) |
| Training steps | 10,000 |
| LoRA rank | 32 |
| Training images | 17 |
| Training resolution | 512, 768, 1024 (multi-bucket) |
| Optimizer | AdamW 8-bit |
| Learning rate | 5e-5 |
| Gradient accumulation | 2 |

## Quality Assessment

The v3 model represents a significant improvement over earlier iterations:

- **v1 (500 steps, rank 16)**: Basic feature recognition. The model learned to produce images that vaguely resembled the character, but details were soft and inconsistent across generations. Useful only for pipeline validation.
- **v2 (3000 steps, rank 32)**: Recognizable likeness. Facial features became consistent, and the model responded well to the trigger word. Some poses and expressions still showed artifacts.
- **v3 (10000 steps, rank 32)**: Strong likeness with natural variation. The character is immediately recognizable across different prompts, poses, and lighting conditions. Prompt adherence remains high -- the model follows scene descriptions without overfitting to the training data's specific backgrounds or compositions.

## Using the LoRA in ComfyUI

### Node Pipeline

The inference workflow uses six standard ComfyUI nodes connected in sequence:

```
CheckpointLoaderSimple
        |
        v
    LoraLoader
        |
        v
  CLIPTextEncode
        |
        v
     KSampler
        |
        v
    VAEDecode
        |
        v
    SaveImage
```

### Step-by-Step Setup

1. **CheckpointLoaderSimple**: Load the SDXL base checkpoint (`sd_xl_base_1.0.safetensors`). The LoRA was trained on this base, so inference should use the same checkpoint for best results.

2. **LoraLoader**: Load the trained `.safetensors` LoRA file. Connect the MODEL and CLIP outputs from the checkpoint loader to the LoRA loader's inputs.

3. **CLIPTextEncode**: Write your prompt. Always include the trigger word at the beginning, followed by the scene description. For example: `TRIGGER_WORD, portrait photo at a coffee shop, warm lighting, shallow depth of field`.

4. **KSampler**: Connect the model from LoraLoader and the conditioning from CLIPTextEncode. Use the recommended settings below.

5. **VAEDecode**: Connect the latent output from KSampler. Uses the VAE from the checkpoint loader.

6. **SaveImage**: Connect the decoded image for output.

### Recommended Settings

| Setting | Value | Notes |
|---|---|---|
| LoRA strength (model) | 0.7 - 1.0 | Start at 0.8. Higher = stronger likeness, lower = more prompt flexibility |
| LoRA strength (CLIP) | 0.7 - 1.0 | Generally keep equal to model strength |
| Steps | 25 | Standard for SDXL. 20 is acceptable for drafts |
| CFG scale | 7.5 | Standard for SDXL. Lower (5-6) for more creative results |
| Sampler | euler | Clean and reliable. `dpmpp_2m` with `karras` scheduler also works well |
| Resolution | 1024 x 1024 | SDXL's native resolution. Can use 768x1024 or 1024x768 for non-square |

### Prompt Tips

- Always start the prompt with the trigger word
- Be specific about the scene, lighting, and composition
- The negative prompt `blurry, low quality, deformed` helps with overall quality
- For portraits, adding `detailed face, sharp focus` improves facial clarity
- The model generalizes well to scenes not in the training data -- try environments, outfits, and art styles not present in the original 17 images

## File Inventory

```
results/
  *.safetensors       # Trained LoRA adapter (gitignored — too large for git)
  model-cards/        # HuggingFace model card template (if present)
```

The `.safetensors` files are not committed to git due to their size. The trained model is stored locally and will be published to HuggingFace with a proper model card documenting the training configuration, intended use, and limitations.

## Next Steps

- Publish the LoRA adapter to HuggingFace with a model card
- Test with community SDXL fine-tunes (not just the base checkpoint)
- Experiment with LoRA merging to combine the character with style LoRAs
- Evaluate whether a rank-64 model provides meaningful improvement over rank-32
