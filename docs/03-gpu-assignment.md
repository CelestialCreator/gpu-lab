# GPU Assignment Strategies

With three GPUs of different capabilities, we need a strategy for assigning the right GPU to the right workload. This guide covers the two approaches we use: UUID pinning for training workloads and scheduler-based assignment for inference workloads.

## GPU Inventory

| GPU | VRAM | UUID | Primary Use |
|-----|------|------|-------------|
| RTX 3080 | 10 GB | `GPU-33ec8b21-2088-c655-d31f-62ab54fb29e5` | Inference |
| RTX 2070 SUPER | 8 GB | `GPU-458be0b4-b6c4-0d85-964d-c85563411524` | Inference |
| RTX 5090 | 32 GB | `GPU-01b9a2f0-30fe-acf6-0080-279a43ad5d4f` | Training |

You can find your GPU UUIDs with:

```bash
nvidia-smi -L
```

Or in more detail:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
```

## Two Assignment Strategies

There are two fundamentally different ways to assign GPUs to pods in Kubernetes:

### Strategy 1: UUID Pinning (Deterministic)

Pin a specific GPU to a specific pod using the `NVIDIA_VISIBLE_DEVICES` environment variable. The pod always gets the exact same GPU, regardless of what the Kubernetes scheduler thinks.

**When to use:** Training workloads that need a specific GPU -- typically the one with the most VRAM.

### Strategy 2: Scheduler Assignment (Dynamic)

Request a GPU count through `nvidia.com/gpu` resource limits and let Kubernetes + the NVIDIA Device Plugin decide which physical GPU to assign.

**When to use:** Inference workloads that can run on any GPU with sufficient VRAM.

## UUID Pinning: AI-Toolkit Example

LoRA training on FLUX.1-dev or SDXL needs as much VRAM as possible. The RTX 5090 with 32 GB is the only GPU in the cluster with enough headroom. We pin it by UUID.

From [`workloads/ai-toolkit/ai-toolkit-k8s.yaml`](../workloads/ai-toolkit/ai-toolkit-k8s.yaml):

```yaml
spec:
  containers:
  - name: ai-toolkit
    image: ostris/aitoolkit:latest
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "GPU-01b9a2f0-30fe-acf6-0080-279a43ad5d4f"
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: "30Gi"
      requests:
        nvidia.com/gpu: "1"
        memory: "8Gi"
```

### How NVIDIA_VISIBLE_DEVICES works

The `NVIDIA_VISIBLE_DEVICES` environment variable is intercepted by the NVIDIA container runtime (not by Kubernetes). Here is the flow:

1. The Kubernetes scheduler sees `nvidia.com/gpu: "1"` in the resource limits and allocates one GPU from the node's pool of 3.
2. The NVIDIA Device Plugin sets `NVIDIA_VISIBLE_DEVICES` to the UUID of whichever GPU it allocated.
3. **But** -- if the pod spec already defines `NVIDIA_VISIBLE_DEVICES`, the pod's value takes precedence.
4. The NVIDIA container runtime reads this variable at container creation time and configures the container to see only that specific GPU.

So we are effectively overriding the device plugin's allocation. The pod still requests `nvidia.com/gpu: "1"` so the scheduler knows to reserve a GPU slot, but the actual GPU is determined by our hardcoded UUID.

### Important: You still need the resource limit

Even with UUID pinning, you must include `nvidia.com/gpu: "1"` in the resource limits. Without it:

- The Kubernetes scheduler does not know the pod needs a GPU
- The device plugin does not get involved in the pod's lifecycle
- The container runtime may not set up GPU access correctly

The resource limit is what triggers the NVIDIA device plugin to inject the container runtime hooks. The UUID override then selects which physical GPU is exposed.

## Scheduler Assignment: ComfyUI Example

ComfyUI runs inference workloads (image generation, music generation) that can run on any GPU. We let Kubernetes assign whichever GPU is available.

From [`workloads/comfyui/comfyui-deployment.yaml`](../workloads/comfyui/comfyui-deployment.yaml):

```yaml
spec:
  containers:
  - name: comfyui
    image: mmartial/comfyui-nvidia-docker:ubuntu24_cuda12.8-latest
    resources:
      limits:
        nvidia.com/gpu: "1"
      requests:
        nvidia.com/gpu: "1"
```

No `NVIDIA_VISIBLE_DEVICES` is set. The flow:

1. The scheduler sees the GPU request and schedules the pod on the node (which has GPUs available).
2. The NVIDIA Device Plugin allocates one GPU from the available pool and sets `NVIDIA_VISIBLE_DEVICES` to that GPU's UUID.
3. The container sees exactly one GPU via `nvidia-smi`.

Which physical GPU gets assigned depends on what is available at scheduling time. If the RTX 5090 is already consumed by AI-Toolkit (via UUID pinning), ComfyUI gets one of the other two.

## Resource Limits: nvidia.com/gpu

The `nvidia.com/gpu` resource works like any other Kubernetes resource (CPU, memory):

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"    # Pod gets exactly 1 GPU
  requests:
    nvidia.com/gpu: "1"    # Pod needs at least 1 GPU
```

Key properties:

- **GPUs are whole units.** You cannot request `0.5` GPUs. Each GPU is exclusively allocated to one pod.
- **Limits and requests should match.** Unlike CPU and memory, there is no concept of "burstable" GPU access. A pod either has a GPU or it does not.
- **No overcommit.** If all 3 GPUs are allocated, the next pod requesting a GPU stays in `Pending` until a GPU is freed.

### Checking GPU allocation

To see current GPU allocation across the cluster:

```bash
kubectl describe node zosmaai | grep -A 10 "Allocated resources"
```

Or list all pods with GPU requests:

```bash
kubectl get pods --all-namespaces -o json | \
  jq -r '.items[] | select(.spec.containers[].resources.limits["nvidia.com/gpu"] != null) |
  "\(.metadata.namespace)/\(.metadata.name)"'
```

## Memory and Shared Memory Configuration

GPU workloads often need more than just GPU VRAM. They also need:

1. **System memory (RAM)** -- for model loading, data preprocessing, and CPU-side operations
2. **Shared memory (/dev/shm)** -- for PyTorch DataLoader workers and inter-process communication

### Setting memory limits

From the AI-Toolkit deployment:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
    memory: "30Gi"      # System RAM limit
  requests:
    nvidia.com/gpu: "1"
    memory: "8Gi"        # Minimum guaranteed RAM
```

The `memory` limit here is system RAM, not VRAM. VRAM is managed entirely by the NVIDIA driver and is not visible to the Kubernetes resource system.

### Shared memory (dshm)

PyTorch DataLoaders with `num_workers > 0` use `/dev/shm` for shared memory between worker processes. Kubernetes defaults `/dev/shm` to 64 MB, which is far too small for training workloads.

Mount a larger shared memory volume:

```yaml
volumeMounts:
- name: dshm
  mountPath: /dev/shm
volumes:
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: 1Gi
```

The `medium: Memory` tells Kubernetes to back this `emptyDir` with tmpfs (RAM-backed filesystem). The `sizeLimit: 1Gi` caps it at 1 GB.

## Multi-GPU Considerations

Currently, each workload gets a single GPU. For future multi-GPU training:

### Requesting multiple GPUs

```yaml
resources:
  limits:
    nvidia.com/gpu: "2"    # Pod gets 2 GPUs
```

The pod would see two GPUs via `nvidia-smi`. PyTorch's `DataParallel` or `DistributedDataParallel` can then split training across them.

### Multi-GPU with UUID pinning

To pin multiple specific GPUs, set `NVIDIA_VISIBLE_DEVICES` to a comma-separated list of UUIDs:

```yaml
env:
- name: NVIDIA_VISIBLE_DEVICES
  value: "GPU-01b9a2f0-30fe-acf6-0080-279a43ad5d4f,GPU-33ec8b21-2088-c655-d31f-62ab54fb29e5"
```

This would give the pod both the RTX 5090 and RTX 3080. However, multi-GPU training across cards with different VRAM sizes and architectures (Blackwell + Ampere) introduces complexity -- different memory capacities mean the smaller card becomes the bottleneck.

### PCIe topology matters

For multi-GPU training, the physical PCIe topology affects communication speed between GPUs. Use `nvidia-smi topo -m` to see the interconnect matrix. GPUs connected through the CPU (via PCIe bridge) are slower to communicate than GPUs on the same PCIe switch.

## Summary

| Workload | Strategy | GPU | Mechanism |
|----------|----------|-----|-----------|
| AI-Toolkit (training) | UUID pinning | RTX 5090 (32 GB) | `NVIDIA_VISIBLE_DEVICES` env var |
| ComfyUI (inference) | Scheduler | Any available | `nvidia.com/gpu` resource limit |

UUID pinning guarantees your most memory-hungry workload gets the most capable GPU. Scheduler assignment keeps things simple for workloads that are GPU-flexible. Using both together gives you deterministic control where it matters and easy scaling where it does not.

## Next Steps

For issues encountered during GPU setup and workload operation, see [Known Issues](04-known-issues.md).
