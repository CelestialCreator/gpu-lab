# NVIDIA GPU Setup in Kubernetes

This guide covers making NVIDIA GPUs available to Kubernetes pods -- from driver installation to the device plugin that exposes GPUs as schedulable resources. This was the most trial-and-error-heavy part of the entire setup.

## Hardware

| GPU | VRAM | Role |
|-----|------|------|
| **RTX 5090** | 32 GB | LoRA training (AI-Toolkit) |
| **RTX 3080** | 10 GB | Inference / generation (ComfyUI) |
| **RTX 2070 SUPER** | 8 GB | Inference / generation (ComfyUI) |

## Software Versions

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 590.48.01 |
| CUDA | 12.8 |
| containerd | 1.7.28 |
| NVIDIA Device Plugin (nvdp) | 0.17.1 |

## Step 1: Blacklist the Nouveau Driver

The open-source `nouveau` driver conflicts with NVIDIA's proprietary driver. It must be blacklisted before installing the NVIDIA driver.

Create `/etc/modprobe.d/blacklist-nouveau.conf`:

```
blacklist nouveau
options nouveau modeset=0
```

Regenerate initramfs and reboot:

```bash
sudo update-initramfs -u
sudo reboot
```

The config is stored at [`system/modprobe.d/blacklist-nouveau.conf`](../system/modprobe.d/blacklist-nouveau.conf).

After reboot, verify nouveau is not loaded:

```bash
lsmod | grep nouveau
# Should return nothing
```

## Step 2: Install the NVIDIA Driver

Install the NVIDIA driver 590.48.01 with CUDA 12.8 support. Follow NVIDIA's official instructions for Debian, or install from the `.run` file:

```bash
# Verify installation
nvidia-smi
```

Expected output should show all three GPUs:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080        Off | 00000000:0A:00.0   Off |                  N/A |
|   1  NVIDIA GeForce RTX 2070 SUPER  Off | 00000000:0B:00.0   Off |                  N/A |
|   2  NVIDIA GeForce RTX 5090        Off | 00000000:01:00.0   Off |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## Step 3: Install the NVIDIA Container Toolkit

The NVIDIA Container Toolkit lets container runtimes access GPUs. It includes `nvidia-ctk`, the tool that configures containerd and generates CDI specs.

```bash
# Add the NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

## Step 4: Configure containerd with NVIDIA as Default Runtime

This is a critical detail that many guides get wrong. There are two ways to configure the NVIDIA runtime in containerd:

1. **Available runtime** -- NVIDIA is registered but not used unless explicitly requested
2. **Default runtime** -- Every container automatically uses the NVIDIA runtime

We set NVIDIA as the **default runtime**. This means every pod gets GPU access by default, and the NVIDIA Device Plugin can detect and allocate GPUs without requiring a RuntimeClass on every pod spec.

The key line in `config.toml`:

```toml
[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "nvidia"
```

You can configure this with `nvidia-ctk`:

```bash
sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
sudo systemctl restart containerd
```

Or edit `/etc/containerd/config.toml` directly. The full configuration is at [`system/containerd/config.toml`](../system/containerd/config.toml).

### Why default and not just available?

If NVIDIA is only an available (non-default) runtime, you need to create a `RuntimeClass` and reference it in every pod spec. With it as the default runtime, containerd uses it for all containers automatically. The NVIDIA Device Plugin then manages which containers actually get GPU access through the `nvidia.com/gpu` resource and `NVIDIA_VISIBLE_DEVICES` environment variable.

## Step 5: Generate the CDI Specification

CDI (Container Device Interface) is the modern way to expose devices to containers. It replaces the older `--gpus` flag approach with a standardized specification.

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

Verify the CDI spec was created:

```bash
nvidia-ctk cdi list
```

This should list all three GPUs with their CDI identifiers.

## Step 6: Create a RuntimeClass (Optional)

Even though we set NVIDIA as the default runtime, creating a RuntimeClass is good practice for explicit documentation in pod specs:

```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
```

```bash
kubectl apply -f nvidia-runtimeclass.yaml
```

Pods can then reference this RuntimeClass in their spec:

```yaml
spec:
  runtimeClassName: nvidia
```

In our setup this is optional because the default runtime is already NVIDIA, but it makes the intent explicit.

## Step 7: Install the NVIDIA Device Plugin

The device plugin is what actually makes GPUs visible to the Kubernetes scheduler as allocatable resources (`nvidia.com/gpu`). Without it, Kubernetes has no idea GPUs exist.

### The GPU Operator Saga

Before arriving at the standalone device plugin, we tried the **NVIDIA GPU Operator** first. The GPU Operator is NVIDIA's all-in-one solution: it bundles the driver, container toolkit, device plugin, and monitoring into a single Helm chart with an operator managing everything.

**Why it failed for us:**

The GPU Operator assumes it manages the entire NVIDIA stack. It tries to install its own driver containers, its own container toolkit, and its own device plugin. On a system where we had already installed the driver and toolkit at the host level, the Operator conflicted with the existing installation. Driver version mismatches between host and operator containers caused init container crashes. The Operator's validator pods failed health checks because they detected the host-installed driver but could not reconcile it with their expected state.

After debugging for several hours, the pragmatic choice was clear: uninstall the GPU Operator and use the standalone device plugin instead.

```bash
# Uninstall the GPU Operator (if you went down this path)
helm uninstall gpu-operator -n gpu-operator
kubectl delete namespace gpu-operator
```

### GPU Operator vs. Standalone Device Plugin

| Aspect | GPU Operator | Standalone Device Plugin |
|--------|-------------|------------------------|
| **Scope** | Full stack: driver, toolkit, plugin, monitoring | Device plugin only |
| **Best for** | Fresh clusters where NVIDIA manages everything | Systems with existing host-level driver/toolkit |
| **Complexity** | Higher -- operator manages many components | Lower -- single DaemonSet |
| **Flexibility** | Less -- fights host-level installations | More -- works alongside existing setup |
| **Our choice** | Tried first, failed | Used successfully |

### Install the Standalone Device Plugin with Helm

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --version 0.17.1 \
  --namespace nvidia-device-plugin \
  --create-namespace
```

### Verify GPUs are visible

After the device plugin DaemonSet is running, check that the node reports GPU resources:

```bash
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
```

Expected output:

```
"3"
```

You can also describe the node to see GPU details:

```bash
kubectl describe node zosmaai | grep -A 5 "Allocatable:" | grep nvidia
```

```
nvidia.com/gpu: 3
```

## Step 8: Validate with a Test Pod

Run a quick test pod to confirm GPU access from inside a container:

```bash
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:12.8.0-base-ubuntu22.04 \
  --limits=nvidia.com/gpu=1 \
  -- nvidia-smi
```

This should print the `nvidia-smi` output from inside the pod, showing one GPU allocated to it.

## Configuration Files Reference

| File | Location | Purpose |
|------|----------|---------|
| [`system/containerd/config.toml`](../system/containerd/config.toml) | `/etc/containerd/config.toml` | containerd config with NVIDIA as default runtime |
| [`system/modprobe.d/blacklist-nouveau.conf`](../system/modprobe.d/blacklist-nouveau.conf) | `/etc/modprobe.d/blacklist-nouveau.conf` | Blacklist nouveau driver |
| CDI spec (generated) | `/etc/cdi/nvidia.yaml` | CDI device specification for all GPUs |

## Troubleshooting

**"Failed to initialize NVML" in pods**: The NVIDIA runtime is not configured as default, or containerd was not restarted after configuration changes. Verify `default_runtime_name = "nvidia"` in `config.toml` and restart containerd.

**Device plugin pod in CrashLoopBackOff**: Check that the NVIDIA driver is loaded on the host (`nvidia-smi` works) and that the containerd socket is accessible. The device plugin needs both.

**"0" GPUs allocatable**: The device plugin is not running or cannot detect GPUs. Check device plugin logs: `kubectl logs -n nvidia-device-plugin -l app.kubernetes.io/name=nvidia-device-plugin`.

## Next Steps

With GPUs available to Kubernetes, the next step is [GPU Assignment](03-gpu-assignment.md) to control which workload gets which GPU.
