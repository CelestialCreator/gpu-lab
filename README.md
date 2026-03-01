# GPU Lab

Hands-on AI/ML infrastructure on a single Debian server with Kubernetes and consumer GPUs. This repository documents everything I built, broke, and learned — from bare-metal GPU setup to training LoRA models and generating music with AI.

## What This Is

A learning lab for AI/ML infrastructure. Instead of using cloud GPU rentals, I built everything on a single physical server: Kubernetes, multi-GPU management, model training pipelines, and creative AI workflows. Each project is a self-contained learning module that documents the full journey — including the failures.

## Hardware

| Component | Spec |
|-----------|------|
| **CPU** | AMD Ryzen 9 5900X (12-core) |
| **RAM** | 16 GB DDR4 |
| **Swap** | 32 GB (btrfs swapfile) |
| **Storage** | 1.9 TB NVMe |
| **GPU 1** | NVIDIA RTX 5090 — 32 GB VRAM |
| **GPU 2** | NVIDIA RTX 3080 — 10 GB VRAM |
| **GPU 3** | NVIDIA RTX 2070 SUPER — 8 GB VRAM |
| **OS** | Debian 13 (trixie) |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Debian 13 — Kernel 6.12 — NVIDIA Driver 590.48     │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Kubernetes v1.35.0 (kubeadm, single-node)      │ │
│  │  CNI: Cilium 1.18.5                              │ │
│  │  GPU: NVIDIA Device Plugin 0.17.1                │ │
│  │                                                   │ │
│  │  ┌──────────────┐  ┌──────────────────────────┐  │ │
│  │  │  AI-Toolkit  │  │  ComfyUI                 │  │ │
│  │  │  (RTX 5090)  │  │  (scheduler-assigned)    │  │ │
│  │  │  LoRA train  │  │  Image gen, music gen,   │  │ │
│  │  │  :30675      │  │  dataset creation :30188 │  │ │
│  │  └──────────────┘  └──────────────────────────┘  │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  GPU 0: RTX 3080 ─── GPU 1: RTX 2070S ─── GPU 2: RTX 5090  │
└─────────────────────────────────────────────────────┘
```

## Projects

| # | Project | Status | Description |
|---|---------|--------|-------------|
| 01 | [LoRA Training](projects/01-lora-training/) | Done | FLUX.1-dev OOM → SDXL pivot → 10k-step LoRA on custom character |
| 02 | [Dataset Creation](projects/02-dataset-creation/) | Done | ComfyUI pipeline: Qwen Image Edit + Florence2 auto-captioning |
| 03 | [Music Generation](projects/03-music-generation/) | Done | ACE-Step 1.5 music generation via ComfyUI |
| 04 | [Multi-Token Prediction](projects/04-multi-token-prediction/) | Planned | MTP via Self-Distillation — next project |

## Documentation

| Guide | What It Covers |
|-------|---------------|
| [Kubernetes Setup](docs/01-kubernetes-setup.md) | Single-node kubeadm on Debian 13 with Cilium CNI |
| [NVIDIA GPU Setup](docs/02-nvidia-gpu-setup.md) | Driver, containerd runtime, CDI, device plugin |
| [GPU Assignment](docs/03-gpu-assignment.md) | UUID pinning for training vs. scheduler for inference |
| [Known Issues](docs/04-known-issues.md) | OOM debugging, swap thrashing, device plugin cache |

## Infrastructure

| Component | Details |
|-----------|---------|
| [Cilium CNI](kubernetes/cilium/) | Helm install for Cilium 1.18.5 |
| [NVIDIA Device Plugin](kubernetes/nvidia-device-plugin/) | Helm install for nvdp 0.17.1 |
| [AI-Toolkit](workloads/ai-toolkit/) | Kubernetes deployment for LoRA training |
| [ComfyUI](workloads/comfyui/) | Kubernetes deployment for image/audio generation |
| [System Configs](system/) | containerd, sysctl, modprobe configs |

## Tech Stack

- **Orchestration**: Kubernetes v1.35.0 (kubeadm)
- **CNI**: Cilium 1.18.5
- **Container Runtime**: containerd 1.7.28 with NVIDIA runtime
- **GPU Management**: NVIDIA Device Plugin 0.17.1
- **NVIDIA Driver**: 590.48.01 | CUDA 12.8
- **Training**: [AI-Toolkit](https://github.com/ostris/ai-toolkit) (ostris)
- **Workflows**: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 1.38.13

## Roadmap

- [ ] **Multi-Token Prediction** — Implement MTP via Self-Distillation
- [ ] **HuggingFace Publishing** — Publish LoRA adapter with proper model card
- [ ] **Multi-node Kubernetes** — Scale beyond single server
- [ ] **Automated training pipelines** — CronJob-based retraining workflows

## Repository Structure

```
gpu-lab/
├── docs/                    # Infrastructure setup guides
├── system/                  # OS-level configs (sysctl, containerd, modprobe)
├── kubernetes/              # Helm install docs (Cilium, NVIDIA plugin)
├── workloads/               # Kubernetes manifests (AI-Toolkit, ComfyUI)
├── projects/                # Self-contained learning modules
│   ├── 01-lora-training/    # LoRA fine-tuning on SDXL
│   ├── 02-dataset-creation/ # Training dataset pipeline
│   ├── 03-music-generation/ # ACE-Step music generation
│   └── 04-multi-token-prediction/  # (Planned)
├── model-cards/             # HuggingFace model card templates
└── assets/                  # Screenshots and diagrams
```
