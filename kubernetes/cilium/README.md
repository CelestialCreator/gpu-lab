# Cilium CNI

[Cilium](https://cilium.io/) is used as the Container Network Interface (CNI) for this cluster, replacing the default kube-proxy.

## Version

- **Cilium**: 1.18.5

## Installation

```bash
helm repo add cilium https://helm.cilium.io/
helm repo update

helm install cilium cilium/cilium \
  --version 1.18.5 \
  --namespace kube-system
```

## Sysctl Override

Cilium requires `rp_filter` to be disabled on its interfaces to prevent mangled packets from being dropped. See [`system/sysctl.d/99-zzz-override_cilium.conf`](../../system/sysctl.d/99-zzz-override_cilium.conf):

```
-net.ipv4.conf.lxc*.rp_filter = 0
-net.ipv4.conf.cilium_*.rp_filter = 0
net.ipv4.conf.all.rp_filter = 0
```

## Verification

```bash
# Check Cilium pods are running
kubectl -n kube-system get pods -l app.kubernetes.io/name=cilium-agent

# Check Cilium status
kubectl -n kube-system exec ds/cilium -- cilium status
```
