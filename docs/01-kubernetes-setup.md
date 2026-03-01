# Kubernetes Setup: Single-Node kubeadm on Debian 13

This guide walks through setting up a single-node Kubernetes cluster on Debian 13 (trixie) using kubeadm, containerd, and Cilium CNI. The setup is purpose-built for running GPU workloads on a single physical server -- no cloud, no multi-node complexity.

## Overview

| Component | Version / Detail |
|-----------|-----------------|
| **OS** | Debian 13 (trixie) |
| **Kubernetes** | v1.35.0 |
| **kubectl** | v1.33.5 |
| **containerd** | 1.7.28 |
| **CNI** | Cilium 1.18.5 |
| **Node name** | zosmaai |

This setup is based on [max-pfeiffer's blog guide](https://max-pfeiffer.ch) for kubeadm on Debian, with several custom tweaks for single-node GPU workloads -- most notably keeping swap enabled and replacing kube-proxy with Cilium.

## Prerequisites

A Debian 13 (trixie) server with:
- Root or sudo access
- A static IP or stable DHCP lease
- Internet access for pulling packages and container images

## Step 1: Install containerd

Kubernetes needs a container runtime. We use containerd 1.7.28, installed from the Docker apt repository.

```bash
# Add Docker's official GPG key and repository
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y containerd.io
```

Generate the default config and enable SystemdCgroup:

```bash
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml
```

Edit `/etc/containerd/config.toml` to set `SystemdCgroup = true` under the runc options. This is required for kubeadm to work correctly with systemd as the init system. The full config is stored in this repository at [`system/containerd/config.toml`](../system/containerd/config.toml).

```bash
sudo systemctl restart containerd
sudo systemctl enable containerd
```

## Step 2: Kernel Module and Sysctl Configuration

Kubernetes networking requires specific kernel modules and sysctl parameters.

### Load required kernel modules

```bash
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter
```

### Configure sysctl parameters

Create `/etc/sysctl.d/k8s.conf` to enable IP forwarding and bridge netfilter:

```ini
net.ipv4.ip_forward = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
```

Apply immediately:

```bash
sudo sysctl --system
```

The actual config file is stored at [`system/sysctl.d/k8s.conf`](../system/sysctl.d/k8s.conf).

## Step 3: Install kubeadm, kubelet, and kubectl

Add the Kubernetes apt repository and install the components:

```bash
sudo apt-get install -y apt-transport-https
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.35/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.35/deb/ /' | \
  sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
```

## Step 4: Initialize the Cluster

### About swap -- why we keep it

Most Kubernetes guides tell you to disable swap. We do not. This server has only 16 GB of physical RAM but runs large model training workloads that can spike memory usage unpredictably. A 32 GB btrfs swapfile acts as a safety net.

Kubernetes has supported swap since v1.28 (beta). We pass `--ignore-preflight-errors=Swap` to kubeadm and let the kubelet handle it. Later, in [Known Issues](04-known-issues.md), we discuss tuning `vm.swappiness` to prevent swap thrashing from freezing the system.

### Initialize with kubeadm

We skip the default kube-proxy installation because Cilium will replace it entirely:

```bash
sudo kubeadm init \
  --node-name zosmaai \
  --skip-phases=addon/kube-proxy \
  --ignore-preflight-errors=Swap
```

The `--skip-phases=addon/kube-proxy` flag is critical. Cilium operates as a full kube-proxy replacement using eBPF, and having both running causes routing conflicts.

### Configure kubectl for your user

```bash
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

## Step 5: Remove the Control-Plane Taint

By default, kubeadm taints the control-plane node so that no workload pods can be scheduled on it. On a multi-node cluster this makes sense -- you want the control plane dedicated to cluster management. On a single-node cluster, this taint means nothing can run at all.

Remove it:

```bash
kubectl taint nodes zosmaai node-role.kubernetes.io/control-plane:NoSchedule-
```

Verify the taint is gone:

```bash
kubectl describe node zosmaai | grep -i taint
# Should show: Taints: <none>
```

## Step 6: Install Cilium CNI

Cilium is a high-performance CNI that uses eBPF for networking, observability, and security. We use it as a complete replacement for kube-proxy.

### Install the Cilium CLI

```bash
CILIUM_CLI_VERSION=$(curl -s https://raw.githubusercontent.com/cilium/cilium-cli/main/stable.txt)
CLI_ARCH=amd64
curl -L --fail --remote-name-all \
  https://github.com/cilium/cilium-cli/releases/download/${CILIUM_CLI_VERSION}/cilium-linux-${CLI_ARCH}.tar.gz
sudo tar xzvfC cilium-linux-${CLI_ARCH}.tar.gz /usr/local/bin
rm cilium-linux-${CLI_ARCH}.tar.gz
```

### Install Cilium with Helm

```bash
helm repo add cilium https://helm.cilium.io/
helm repo update

helm install cilium cilium/cilium --version 1.18.5 \
  --namespace kube-system \
  --set kubeProxyReplacement=true \
  --set k8sServiceHost=<API_SERVER_IP> \
  --set k8sServicePort=6443
```

The `kubeProxyReplacement=true` flag tells Cilium to handle all service routing via eBPF, replacing kube-proxy entirely.

### Cilium sysctl override

Cilium requires `rp_filter` (reverse path filtering) to be disabled on its interfaces. Without this, the kernel drops packets that Cilium legitimately routes through its virtual interfaces.

Create `/etc/sysctl.d/99-zzz-override_cilium.conf`:

```ini
# Disable rp_filter on Cilium interfaces since it may cause mangled packets to be dropped
-net.ipv4.conf.lxc*.rp_filter = 0
-net.ipv4.conf.cilium_*.rp_filter = 0
# The kernel uses max(conf.all, conf.{dev}) as its value, so we need to set .all. to 0 as well.
# Otherwise it will overrule the device specific settings.
net.ipv4.conf.all.rp_filter = 0
```

The `99-zzz-` prefix ensures this file is loaded last, overriding any earlier sysctl configs. The actual config is at [`system/sysctl.d/99-zzz-override_cilium.conf`](../system/sysctl.d/99-zzz-override_cilium.conf).

```bash
sudo sysctl --system
```

### Verify Cilium is healthy

```bash
cilium status
```

You should see all components reporting `OK`. You can also run the connectivity test:

```bash
cilium connectivity test
```

## Step 7: Verify the Cluster

At this point, the single-node cluster should be fully operational:

```bash
kubectl get nodes
```

Expected output:

```
NAME       STATUS   ROLES           AGE   VERSION
zosmaai    Ready    control-plane   ...   v1.35.0
```

Check that all system pods are running:

```bash
kubectl get pods -n kube-system
```

You should see the Cilium agent, Cilium operator, CoreDNS, etcd, kube-apiserver, kube-controller-manager, and kube-scheduler all in `Running` state. There should be no kube-proxy pod since we skipped that phase.

## Configuration Files Reference

All system configuration files are stored in the [`system/`](../system/) directory:

| File | Purpose |
|------|---------|
| [`system/sysctl.d/k8s.conf`](../system/sysctl.d/k8s.conf) | IP forwarding and bridge netfilter for Kubernetes |
| [`system/sysctl.d/99-zzz-override_cilium.conf`](../system/sysctl.d/99-zzz-override_cilium.conf) | Disable rp_filter for Cilium |
| [`system/containerd/config.toml`](../system/containerd/config.toml) | containerd configuration with SystemdCgroup and NVIDIA runtime |

## Key Decisions and Tradeoffs

**Why single-node?** This is a learning lab. Single-node eliminates networking complexity, storage replication, and node scheduling concerns. It lets us focus on the GPU workload side. Multi-node is on the [roadmap](../README.md).

**Why Cilium over kube-proxy?** eBPF-based networking is more efficient and provides better observability. For GPU workloads where we want low overhead, Cilium is a good fit. It also means one fewer component (kube-proxy) to manage.

**Why keep swap?** With 16 GB RAM and models that can spike to 12+ GB during loading, swap is a safety net. The tradeoff is that swap thrashing can freeze the system (see [Known Issues](04-known-issues.md)), but with `vm.swappiness=1`, the kernel prefers OOM-killing over thrashing.

**Why kubeadm over k3s or microk8s?** kubeadm produces a standard upstream Kubernetes cluster. What you learn here transfers directly to production environments. k3s and microk8s are excellent tools, but they abstract away details that are valuable to understand.

## Next Steps

With the cluster running, the next step is [NVIDIA GPU Setup](02-nvidia-gpu-setup.md) to make the GPUs available to Kubernetes workloads.
