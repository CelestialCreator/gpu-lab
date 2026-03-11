# GPU Lab

Hands-on AI/ML infrastructure on a single Debian server with Kubernetes and consumer GPUs. This repository documents everything I built, broke, and learned — from bare-metal GPU setup to training LoRA models and generating music with AI.

## What This Is

A learning lab for AI/ML on consumer hardware. Instead of cloud GPU rentals, everything runs on a single physical server: Kubernetes, multi-GPU management, model training pipelines, RL reasoning, and creative AI workflows. Each project is a self-contained module that documents the full journey — including the failures.

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
┌───────────────────────────────────────────────────────────────┐
│  Debian 13 — Kernel 6.12 — NVIDIA Driver 590.48              │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Kubernetes v1.35.0 (kubeadm, single-node)              │  │
│  │  CNI: Cilium 1.18.5 | GPU: NVIDIA Device Plugin         │  │
│  │                                                          │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐  │  │
│  │  │ Training     │ │ Inference    │ │ Services        │  │  │
│  │  │              │ │              │ │                  │  │  │
│  │  │ - SFT/GRPO   │ │ - llama-srv  │ │ - ChatterBox    │  │  │
│  │  │ - LoRA       │ │   (Qwen3.5)  │ │ - ComfyUI       │  │  │
│  │  │ - MTP        │ │ - Ollama     │ │ - AI-Toolkit    │  │  │
│  │  └──────────────┘ └──────────────┘ └─────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────── GPU Pool (50 GB VRAM) ──────────────┐  │
│  │                                                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │  │
│  │  │  RTX 5090   │  │  RTX 3080   │  │  RTX 2070S  │    │  │
│  │  │  32 GB      │  │  10 GB      │  │  8 GB       │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │  │
│  │                                                         │  │
│  │  GPUs are dynamically assigned to workloads via         │  │
│  │  UUID pinning + NVIDIA Device Plugin. Typical configs:  │  │
│  │                                                         │  │
│  │  Training:   5090 (32GB)          — SFT, GRPO, LoRA    │  │
│  │  Inference:  5090 + 2070S (40GB)  — llama-server        │  │
│  │  Services:   3080 (10GB)          — ChatterBox, Ollama  │  │
│  │  Full pool:  all 3 GPUs (50GB)    — large model serving │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

## Projects

| # | Project | Status | Description |
|---|---------|--------|-------------|
| 01 | [LoRA Training](projects/01-lora-training/) | Done | FLUX.1-dev OOM → SDXL pivot → 10k-step LoRA on custom character |
| 02 | [Dataset Creation](projects/02-dataset-creation/) | Done | ComfyUI pipeline: Qwen Image Edit + Florence2 auto-captioning |
| 03 | [Music Generation](projects/03-music-generation/) | Done | ACE-Step 1.5 music generation via ComfyUI |
| 04 | [Multi-Token Prediction](projects/04-multi-token-prediction/) | Done | Reproduced Meta's MTP paper on single RTX 5090 (1.8x inference speedup) |
| 05 | [GRPO Reasoning](projects/05-grpo-reasoning/) | Done | Taught Qwen3.5-0.8B to reason like DeepSeek-R1 (+5.9pp zero-shot GSM8K) |

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
| [llama-server](workloads/llama-server/) | Qwen3.5-35B-A3B inference via llama.cpp (multi-GPU) |
| [System Configs](system/) | containerd, sysctl, modprobe configs |

## Tech Stack

- **Orchestration**: Kubernetes v1.35.0 (kubeadm)
- **CNI**: Cilium 1.18.5
- **Container Runtime**: containerd 1.7.28 with NVIDIA runtime
- **GPU Management**: NVIDIA Device Plugin 0.17.1
- **NVIDIA Driver**: 590.48.01 | CUDA 12.8
- **Training**: [AI-Toolkit](https://github.com/ostris/ai-toolkit) (ostris)
- **Workflows**: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 1.38.13

## Published Models

| Model | Description | Link |
|-------|-------------|------|
| Llama-3.2-1B-MTP-k8 | Multi-Token Prediction reproduction (1.8x speedup) | [HuggingFace](https://huggingface.co/celestialcreator/Llama-3.2-1B-MTP-k8) |
| Qwen3.5-0.8B-GRPO-Math | GRPO reasoning training (+5.9pp zero-shot GSM8K) | [HuggingFace](https://huggingface.co/celestialcreator/Qwen3.5-0.8B-GRPO-Math) |

## Roadmap

- [x] **Multi-Token Prediction** — Reproduced MTP via Self-Distillation (1.8x inference speedup)
- [x] **GRPO Reasoning** — SFT + GRPO on Qwen3.5-0.8B, zero-shot GSM8K 52.1% → 58.0%
- [ ] **Model Abliteration** — Remove refusals from multimodal models for domain-specific use
- [ ] **Multi-node Kubernetes** — Scale beyond single server
- [ ] **Automated training pipelines** — CronJob-based retraining workflows

## Repository Structure

```
gpu-lab/
├── docs/                    # Infrastructure setup guides
├── system/                  # OS-level configs (sysctl, containerd, modprobe)
├── kubernetes/              # Helm install docs (Cilium, NVIDIA plugin)
├── workloads/               # Kubernetes manifests (AI-Toolkit, ComfyUI, llama-server)
├── projects/                # Self-contained learning modules
│   ├── 01-lora-training/    # LoRA fine-tuning on SDXL
│   ├── 02-dataset-creation/ # Training dataset pipeline
│   ├── 03-music-generation/ # ACE-Step music generation
│   ├── 04-multi-token-prediction/  # MTP paper reproduction
│   └── 05-grpo-reasoning/   # GRPO reasoning training (DeepSeek-R1 technique)
├── model-cards/             # HuggingFace model card templates
└── assets/                  # Screenshots and diagrams
```
