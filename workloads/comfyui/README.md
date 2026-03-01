# ComfyUI Workload

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) — a node-based UI for diffusion models and other generative AI workflows.

## Deployment

The manifests create:
- [`namespace.yaml`](namespace.yaml) — `comfyui` namespace
- [`comfyui-deployment.yaml`](comfyui-deployment.yaml) — Single replica using `mmartial/comfyui-nvidia-docker`
- [`comfyui-service.yaml`](comfyui-service.yaml) — NodePort exposing the web UI on port **30188**

## Use Cases

ComfyUI is used for three different projects in this lab:

| Project | What It Does | Custom Nodes |
|---------|-------------|--------------|
| [Dataset Creation](../../projects/02-dataset-creation/) | Qwen Image Edit + Florence2 captioning pipeline | ComfyUI-GGUF, ComfyUI-Florence2, ComfyUI-KJNodes, comfyui_controlnet_aux, rgthree-comfy |
| [Music Generation](../../projects/03-music-generation/) | ACE-Step 1.5 audio generation | ComfyUI_ACE-Step |
| LoRA Inference | Generate images using trained LoRA adapters | (built-in LoraLoader node) |

## Custom Nodes Installed

| Node | Purpose |
|------|---------|
| ComfyUI_ACE-Step | ACE-Step music generation |
| comfyui_controlnet_aux | ControlNet preprocessors (DWPose, etc.) |
| ComfyUI-Florence2 | Florence2 image captioning |
| ComfyUI-GGUF | GGUF model format support |
| ComfyUI-KJNodes | Utility nodes |
| ComfyUI-Manager | Package manager for custom nodes |
| comfyui-videohelpersuite | Video processing utilities |
| rgthree-comfy | UI utilities (Set/Get nodes, etc.) |

## GPU Assignment

Unlike AI-Toolkit, ComfyUI does **not** use UUID pinning. It requests `nvidia.com/gpu: "1"` and gets whichever GPU the Kubernetes scheduler assigns. This is fine for inference workloads that fit in 8-10 GB VRAM.

## Storage

Uses hostPath volumes:
- `/data/comfyui` → `/workspace` (models, outputs, custom nodes)
- `/data/comfyui/run` → `/comfy/mnt` (runtime data)
- `dshm` emptyDir → `/dev/shm` (8 GiB shared memory)

## Deploying

```bash
kubectl apply -f namespace.yaml
kubectl apply -f comfyui-deployment.yaml
kubectl apply -f comfyui-service.yaml
```

Access at `http://<node-ip>:30188`
