# AI Music Generation with ACE-Step

Using ComfyUI and the ACE-Step model to generate music from text descriptions and lyrics -- demonstrating that ComfyUI's node-based workflow engine extends well beyond image generation.

---

## What Is ACE-Step?

ACE-Step is an AI music generation model that produces full audio tracks from text descriptions. Given a set of tags (genre, mood, instruments, vocal style) and optional lyrics, it generates complete musical compositions with vocals, instrumentation, and structure.

This project uses **ACE-Step 1.5 Turbo**, an optimized variant that generates high-quality audio in fewer sampling steps. The all-in-one (AIO) checkpoint is approximately 10GB and includes both the audio generation model and the text encoder.

## Why ComfyUI for Music?

ComfyUI was originally built for image generation workflows, but its architecture -- a directed acyclic graph of processing nodes -- is fundamentally model-agnostic. The same infrastructure that chains together CLIP encoding, latent sampling, and VAE decoding for images can chain together text encoding, audio sampling, and audio decoding for music.

This makes ComfyUI a surprisingly effective tool for audio generation:

- Visual workflow editing makes it easy to experiment with different settings
- The queue system handles long generation jobs gracefully
- Workflows can be saved, shared, and version-controlled as JSON files
- The same GPU that trains image LoRAs can generate music between training runs

## ComfyUI Workflow

The workflow is stored in `workflows/audio_ace_step_1_5_checkpoint.json` and follows a straightforward six-node pipeline:

```
CheckpointLoaderSimple
        |
        v
ModelSamplingAuraFlow (shift=3)
        |
        v
TextEncodeAceStepAudio1.5 (tags + lyrics)
        |
        v
     KSampler
        |
        v
  VAEDecodeAudio
        |
        v
   SaveAudioMP3
```

### Node Details

1. **CheckpointLoaderSimple**: Loads the ACE-Step 1.5 Turbo AIO checkpoint. This single file contains both the diffusion model and the text encoder.

2. **ModelSamplingAuraFlow**: Applies a frequency shift of 3 to the sampling schedule. This is specific to the AuraFlow-based architecture that ACE-Step uses and is required for correct generation.

3. **TextEncodeAceStepAudio1.5**: The text conditioning node. Takes two inputs:
   - **Tags**: Genre, mood, instruments, vocal style, tempo, and other musical attributes (comma-separated)
   - **Lyrics**: The actual lyrics for the song, with structural markers for verse, chorus, etc.

4. **KSampler**: The core sampling node. Generates the audio latent through iterative denoising.

5. **VAEDecodeAudio**: Decodes the latent representation into a raw audio waveform.

6. **SaveAudioMP3**: Encodes and saves the output as an MP3 file.

## Recommended Settings

| Setting | Value | Notes |
|---|---|---|
| Steps | 16 | ACE-Step Turbo is optimized for fewer steps |
| Sampler | euler | Clean and fast |
| CFG scale | 1 | ACE-Step uses classifier-free guidance sparingly |
| Shift | 3 | Required for AuraFlow sampling schedule |
| Scheduler | simple | Standard scheduler works well with euler |

### Tag Format

Tags are comma-separated descriptors that guide the overall musical style:

```
indie rock, melancholic, electric guitar, male vocals, mid-tempo, 120 bpm,
reverb, atmospheric, minor key
```

Effective tags typically include:

- **Genre**: rock, pop, jazz, electronic, hip-hop, classical, ambient
- **Mood**: melancholic, upbeat, aggressive, dreamy, nostalgic
- **Instruments**: electric guitar, piano, synthesizer, drums, bass, strings
- **Vocals**: male vocals, female vocals, choir, no vocals, whispered
- **Tempo**: slow, mid-tempo, fast, specific BPM

### Lyrics Format

Lyrics can include structural markers:

```
[Verse 1]
Walking down the empty street
Shadows dancing at my feet

[Chorus]
We were golden in the light
Everything was burning bright
```

The model responds to these markers and will attempt to create musical structure that matches (verse sections, chorus hooks, bridges, etc.).

## Required Custom Node

| Custom Node | Purpose |
|---|---|
| [ComfyUI_ACE-Step](https://github.com/lks-ai/ComfyUI_ACE-Step) | ACE-Step audio generation nodes (TextEncode, VAEDecodeAudio, SaveAudioMP3) |

Install via ComfyUI Manager or clone into the `custom_nodes/` directory.

## Directory Structure

```
03-music-generation/
  workflows/
    audio_ace_step_1_5_checkpoint.json    # ComfyUI workflow
```

## Observations

ACE-Step 1.5 Turbo produces surprisingly coherent music. The model handles genre blending well (for example, "jazz fusion with electronic elements") and generates vocals that follow the provided lyrics with reasonable accuracy. Instrumental sections are particularly strong.

Limitations worth noting:

- Vocal clarity varies -- some generations have clear singing, others are more abstract
- Long-form structure (songs over 2-3 minutes) can lose coherence
- The model works best when tags and lyrics are stylistically consistent
- Generation is not real-time; a single track takes roughly 30-60 seconds on the RTX 5090

This project demonstrates that a single GPU lab setup can serve multiple creative AI workflows. The same hardware and ComfyUI instance that trains character LoRAs can generate music, making the GPU investment more versatile than a single-purpose training rig.
