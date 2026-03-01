# NVIDIA Device Plugin

The [NVIDIA Device Plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin) exposes GPUs to the Kubernetes scheduler, allowing pods to request `nvidia.com/gpu` resources.

## Version

- **NVIDIA Device Plugin**: 0.17.1

## Installation

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

helm install nvdp nvdp/nvidia-device-plugin \
  --version 0.17.1 \
  --namespace nvidia-device-plugin \
  --create-namespace
```

## Why Not the GPU Operator?

Initially tried the full [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html), which bundles the driver, container runtime, device plugin, and monitoring into a single install. It didn't work reliably on this setup (Debian 13 with manually installed drivers), so it was uninstalled:

```bash
helm uninstall gpu-operator -n gpu-operator
kubectl delete namespace gpu-operator --force --grace-period=0
```

The standalone device plugin via Helm is simpler and more predictable when you already have:
- NVIDIA drivers installed at the OS level
- containerd configured with the NVIDIA runtime
- CDI (Container Device Interface) generated

## Verification

```bash
# Check the device plugin daemonset is running
kubectl -n nvidia-device-plugin get pods

# Check GPUs are visible to the scheduler
kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'
# Expected output: 3

# List GPU devices
kubectl -n nvidia-device-plugin logs ds/nvdp-nvidia-device-plugin | grep "Device found"
```

## Troubleshooting

If pods show `CreateContainerConfigError` with "endpoint not found in cache":

```bash
kubectl -n nvidia-device-plugin rollout restart daemonset nvdp-nvidia-device-plugin
```

See [Known Issues](../../docs/04-known-issues.md) for more details.
