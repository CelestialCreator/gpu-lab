# AI-Toolkit Workload

[AI-Toolkit](https://github.com/ostris/ai-toolkit) by ostris — a web-based interface for training LoRA adapters on diffusion models.

## Deployment

The manifest at [`ai-toolkit-k8s.yaml`](ai-toolkit-k8s.yaml) creates:
- **Namespace**: `ai-toolkit`
- **Deployment**: Single replica running `ostris/aitoolkit:latest`
- **Service**: NodePort exposing the web UI on port **30675**

## GPU Assignment

The pod is pinned to the **RTX 5090** (32 GB VRAM) via `NVIDIA_VISIBLE_DEVICES` UUID targeting. This is the only GPU with enough VRAM for training larger models. See [GPU Assignment](../../docs/03-gpu-assignment.md) for the strategy.

## Environment Variables

| Variable | Value | Notes |
|----------|-------|-------|
| `AI_TOOLKIT_AUTH` | `YOUR_WEBUI_PASSWORD` | Replace with your password |
| `HF_TOKEN` | `YOUR_HF_TOKEN_HERE` | Replace with your HuggingFace token |
| `HF_HOME` | `/workspace/hf_cache` | Cache directory for HF models |
| `NVIDIA_VISIBLE_DEVICES` | `GPU-01b9a...` | RTX 5090 UUID |

## Storage

Uses a hostPath volume at `/data/ai-toolkit` mounted to `/workspace` inside the container. This persists training configs, datasets, outputs, and cached models across pod restarts.

## Resource Limits

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
    memory: "30Gi"
  requests:
    nvidia.com/gpu: "1"
    memory: "8Gi"
```

The `dshm` (shared memory) volume is mounted at `/dev/shm` with a 1 GiB limit — required for PyTorch DataLoader workers.

## Before Deploying

1. Replace `YOUR_HF_TOKEN_HERE` with your HuggingFace token
2. Replace `YOUR_WEBUI_PASSWORD` with a strong password
3. Ensure `/data/ai-toolkit` exists on the host
4. Verify the GPU UUID matches your target GPU (`nvidia-smi -L`)

```bash
kubectl apply -f ai-toolkit-k8s.yaml
```
