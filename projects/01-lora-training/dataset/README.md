# Training Dataset

17 images of a single character, generated via ComfyUI using the Qwen Image Edit pipeline. Each image is paired with a text caption file for training.

---

## How the Dataset Was Created

Rather than collecting real photographs, the training images were generated programmatically using a ComfyUI workflow built around the **Qwen Image Edit** model. A single reference photo was transformed into 17 distinct variations covering different poses, angles, and expressions. See [project 02: Dataset Creation](../../02-dataset-creation/) for the full pipeline.

This approach has several advantages for LoRA training:

- **Consistency**: Every image has the same character identity with controlled variation
- **Coverage**: Systematic pose and expression variety that would be difficult to collect manually
- **Reproducibility**: The workflow can regenerate the dataset with different settings at any time

## Image Inventory

| # | Filename Prefix | Category | Count | Description |
|---|---|---|---|---|
| 01 | `01_turnaround` | Turnaround | 1 | Front, side, back, and three-quarter views on white background |
| 02 | `02_portrait` | Portrait | 2 | Clean front-facing headshots, neutral expression |
| 03 | `03_closeup` | Close-up | 2 | Extreme close-up showing facial detail and skin texture |
| 04 | `04_tpose` | T-Pose | 1 | Full body with arms extended horizontally |
| 05 | `05_sitting` | Sitting | 1 | Casual seated pose, indoor setting |
| 06 | `06_standing_side` | Standing Side | 1 | Side view, natural stance |
| 07 | `07_back_view` | Back View | 1 | Rear view showing outfit from behind |
| 08 | `08_walking` | Walking | 1 | Mid-stride in a park, golden hour lighting |
| 09 | `09_happy` | Expression | 1 | Bright warm smile, joyful eyes |
| 10 | `10_surprised` | Expression | 1 | Wide eyes, slightly open mouth |
| 11 | `11_angry` | Expression | 1 | Furrowed brows, intense gaze |
| 12 | `12_sad` | Expression | 1 | Melancholic expression, downcast eyes |
| 13 | `13_laughing` | Expression | 2 | Genuine open-mouth laughter |
| 14 | `14_contemplative` | Expression | 1 | Thoughtful gaze, looking into the distance |

**Total: 17 images**

The variety is intentional. LoRA training benefits from seeing the subject in diverse conditions -- different angles teach the model the character's 3D structure, different expressions teach it which features are constant (identity) versus variable (emotion), and different lighting conditions improve generalization.

## Caption Format

Each image has a matching `.txt` file in the `captions/` directory. The caption format is:

```
TRIGGER_WORD, description of the image content
```

For example:

```
TRIGGER_WORD, a character turnaround sheet showing a woman from front view, side view,
back view, and three-quarter view, full body, white background
```

```
TRIGGER_WORD, a woman looking genuinely happy with a bright warm smile, joyful eyes, portrait
```

The trigger word is a unique token that the LoRA learns to associate with the character's appearance. During inference, including this token in the prompt activates the learned features.

Captions were generated automatically using **Florence2** (`microsoft/Florence-2-base`), which produces detailed natural-language descriptions of image content. The trigger word was prepended programmatically by the dataset creation pipeline.

## What Is Committed vs. Gitignored

- **Committed**: All `.txt` caption files in the `captions/` directory (small text files, essential for reproducibility)
- **Gitignored**: The `.png` image files (too large for git, can be regenerated from the ComfyUI workflow)

To regenerate the images, run the `character_dataset_creator.json` workflow from [project 02](../../02-dataset-creation/).

## Directory Structure

```
dataset/
  captions/
    01_turnaround_00001_.txt
    02_portrait_00001_.txt
    02_portrait_00002_.txt
    03_closeup_00001_.txt
    03_closeup_00002_.txt
    04_tpose_00001_.txt
    05_sitting_00001_.txt
    06_standing_side_00001_.txt
    07_back_view_00001_.txt
    08_walking_00001_.txt
    09_happy_00001_.txt
    10_surprised_00001_.txt
    11_angry_00001_.txt
    12_sad_00001_.txt
    13_laughing_00001_.txt
    13_laughing_00002_.txt
    14_contemplative_00001_.txt
  *.png                          # Training images (gitignored)
```
