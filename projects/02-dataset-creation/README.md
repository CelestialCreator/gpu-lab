# Automated Character Dataset Creation with ComfyUI

A programmatically generated ComfyUI workflow that transforms a single reference photo into a complete LoRA training dataset -- 18 character variations with auto-captioning, upscaling, and paired output.

---

## Why Generate Training Data with AI?

Traditional LoRA training datasets require collecting 15-30 photographs of a subject in varied poses, expressions, and lighting conditions. This is time-consuming, requires a cooperative subject, and the results are inconsistent (different cameras, environments, times of day).

This pipeline takes a different approach: start with a single reference image and use **Qwen Image Edit** to generate controlled variations. The result is a consistent, comprehensive dataset that covers angles, poses, and expressions systematically.

## Pipeline Architecture

The pipeline has two parts, designed to run sequentially.

### Part 1: Character Variation Generation

Uses the **Qwen Image Edit** model (`qwen_image_edit_1.1`) to generate 18 variations from a single input photo. Each variation is a separate generation group in the ComfyUI workflow with its own prompt, seed, and output path.

The 18 generation groups cover:

| Group | Category | Description |
|---|---|---|
| 01 | Turnaround | Front, side, back, and three-quarter views |
| 02 | Portrait | Clean front-facing headshot, neutral expression |
| 03 | Close-up | Extreme facial close-up with skin detail |
| 04 | T-Pose | Full body, arms extended horizontally |
| 05 | Sitting | Casual seated pose, indoor setting |
| 06 | Standing Side | Side view, natural stance |
| 07 | Back View | Rear view showing outfit |
| 08 | Walking | Mid-stride, outdoor golden hour |
| 09-14 | Expressions | Happy, surprised, angry, sad, laughing, contemplative |
| 15-16 | Virtual Try-On | Character wearing a reference outfit (requires clothing reference image) |
| 17-18 | Pose Transfer | Character in a reference pose (requires pose reference image) |

The expression groups (09-14) chain off the portrait group's output, using the clean headshot as their input for more consistent facial editing. The virtual try-on and pose transfer groups use a secondary reference image for the target clothing or pose.

### Part 2: Dataset Preparation

Runs after Part 1 completes. This stage processes the raw generated images into a training-ready dataset:

1. **Load**: `LoadImagesFromFolderKJ` reads all generated images from the output directory
2. **Caption**: `Florence2Run` (`microsoft/Florence-2-base`) generates detailed natural-language descriptions of each image
3. **Trigger word**: `JoinStrings` prepends the trigger word to each caption (format: `TRIGGER_WORD, caption text`)
4. **Upscale**: `4x-UltraSharp` upscaling model increases resolution for maximum training detail
5. **Resize**: `ImageResizeKJv2` normalizes all images to 1024x1024
6. **Save**: Paired `.png` image and `.txt` caption files are written to the dataset output directory

Part 2 nodes are **muted by default** (mode=2) in the workflow. Unmute them in ComfyUI before running the dataset preparation stage.

## The Workflow Generator Script

The ComfyUI workflow is not hand-built in the UI. It is generated programmatically by `generate_workflow.py` (799 lines of Python).

### Why Script It?

A workflow with 18 generation groups, shared model infrastructure, and a dataset preparation pipeline would contain hundreds of nodes. Building this by hand in ComfyUI's visual editor would be error-prone and nearly impossible to modify systematically. The script approach provides:

- **Reproducibility**: The workflow can be regenerated with different parameters at any time
- **Consistency**: Every generation group uses the exact same node structure and connections
- **Maintainability**: Changing a shared parameter (model, resolution, sampler settings) updates all 18 groups automatically
- **Documentation**: The Python code serves as documentation for the workflow's logic

### Script Architecture

The script uses a `WorkflowBuilder` class that manages nodes, links, and groups:

```python
class WorkflowBuilder:
    def add_node(...)     # Create a ComfyUI node with position, size, widgets
    def add_input(...)    # Add an input slot to a node
    def add_output(...)   # Add an output slot to a node
    def connect(...)      # Wire two nodes together (creates a link)
    def add_group(...)    # Add a visual group box
    def build()           # Serialize to ComfyUI JSON format
```

The workflow is assembled in three phases:

1. **`build_shared_infrastructure()`** -- Creates shared model loaders (GGUF model, CLIP, VAE), parameter primitives (steps, CFG, width, height), and input image loaders. Uses `SetNode`/`GetNode` pairs to share values across groups without tangled wires.

2. **`build_generation_group()`** -- Called 18 times, once per variation. Each group gets: GetNodes for shared resources, TextEncodeQwenImageEditPlus (with the variation prompt), EmptySD3LatentImage, KSampler, VAEDecode, and SaveImage.

3. **`build_part2_dataset_prep()`** -- Creates the Florence2 captioning, 4x-UltraSharp upscaling, resize, and paired save nodes.

### Output

Running the script produces:

```
$ python generate_workflow.py
Workflow saved to character_dataset_creator.json
Total nodes: 295
Total links: 283
Total groups: 20
```

The output JSON is approximately 235KB -- a workflow of this complexity would be impractical to manage by hand.

## Workflow Files

```
02-dataset-creation/
  generate_workflow.py                          # Workflow generator script (799 lines)
  workflows/
    character_dataset_creator.json              # Generated workflow (295 nodes, 283 links)
    qwen_image_edit_1.1.json                    # Original manual workflow (inspiration)
```

- **`character_dataset_creator.json`** is the output of `generate_workflow.py`. Load this in ComfyUI to run the pipeline.
- **`qwen_image_edit_1.1.json`** is the manually-created workflow that was used to prototype the Qwen Image Edit approach before scripting it. It served as the reference implementation for the generation groups.

## Required Custom Nodes

The workflow depends on several ComfyUI custom nodes that must be installed before loading:

| Custom Node | Purpose |
|---|---|
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | Load GGUF-quantized models (Qwen Image Edit) |
| [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) | Florence2 auto-captioning |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Utility nodes: LoadImagesFromFolder, JoinStrings, ImageResize, StringConstant |
| [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | Auxiliary preprocessing for pose/depth |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | Quality-of-life workflow utilities |

Install via ComfyUI Manager or clone directly into the `custom_nodes/` directory.

## Usage

### Prerequisites

1. ComfyUI with the custom nodes listed above
2. Model files:
   - `Qwen-Image-Edit-2509-Q5_0.gguf` (GGUF-quantized Qwen Image Edit)
   - `qwen_2.5_vl_7b_fp8_scaled.safetensors` (CLIP encoder)
   - `qwen_image_vae.safetensors` (VAE)
   - `Qwen-Image-Lightning-4steps-V2.0.safetensors` (Lightning LoRA for 4-step generation)
   - `4x-UltraSharp.pth` (upscale model)
   - `Florence-2-base` (auto-captioning model, downloaded automatically)

### Running the Pipeline

1. Load `character_dataset_creator.json` in ComfyUI
2. Set your reference photo in the **INPUT PHOTO** LoadImage node
3. (Optional) Set clothing and pose reference images for groups 15-18
4. Set your trigger word in the **TRIGGER WORD** StringConstant node
5. Queue the workflow -- Part 1 generates all 18 variations
6. After Part 1 completes, unmute the Part 2 nodes (select all Part 2 nodes, press `M`)
7. Queue again -- Part 2 captions, upscales, and saves the paired dataset

### Regenerating the Workflow

To modify the workflow parameters and regenerate:

```bash
python generate_workflow.py
```

The script writes the output to the configured path. Copy the JSON to ComfyUI's workflow directory or load it directly.
