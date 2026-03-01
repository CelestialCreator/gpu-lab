# Known Issues and Lessons Learned

Every infrastructure project has war stories. This document captures the issues we hit, how we debugged them, and the fixes we applied. If you are building a similar setup, this is the page that will save you the most time.

## Issue 1: FLUX OOM Freeze on 16 GB RAM

**Symptom:** Attempting to load FLUX.1-dev (~12 billion parameters) for LoRA training caused the entire system to freeze. Not just the training pod -- the entire server became unresponsive. SSH sessions hung. The Kubernetes API server on port 6443 stopped responding. Websocket connections timed out.

**Root cause:** FLUX.1-dev requires roughly 12+ GB of system RAM just to load the model weights into CPU memory before transferring them to GPU VRAM. On a server with 16 GB of physical RAM, this left almost nothing for the kernel, Kubernetes components, and other system processes.

The kernel did not OOM-kill the training process. Instead, with 32 GB of swap available and the default `vm.swappiness=60`, the kernel tried to accommodate the memory pressure by swapping aggressively. This triggered swap thrashing: the system spent all its time moving pages between RAM and swap, with no CPU cycles left for actual work.

The critical chain of failure:

1. FLUX model loading consumed ~12 GB RAM
2. Kernel swapped out Kubernetes components (kubelet, API server, etcd) to make room
3. Kubernetes components needed to run (health checks, API calls) and got swapped back in
4. This evicted the model loading pages, which then got swapped back in
5. Repeat -- the system entered a thrashing death spiral
6. The Kubernetes API server on port 6443 could not respond to health checks
7. Websocket connections to the API server timed out
8. The entire system became effectively frozen despite not technically being "out of memory"

**Fix:** Set `vm.swappiness=1`:

```bash
sudo sysctl vm.swappiness=1
```

To make it persistent:

```bash
echo "vm.swappiness=1" | sudo tee -a /etc/sysctl.d/99-swap.conf
sudo sysctl --system
```

With `vm.swappiness=1`, the kernel strongly prefers killing the offending process (via OOM killer) over swapping. This is the correct behavior for our use case: it is far better to have one pod get OOM-killed and restarted than to have the entire server freeze.

**Lesson:** Swap is a safety net, not a substitute for RAM. With `swappiness=1`, the kernel uses swap only as a last resort before invoking the OOM killer, rather than aggressively trying to accommodate every allocation. For ML workloads with unpredictable memory spikes, this is the right tradeoff.

**Related:** This OOM issue ultimately led to the pivot from FLUX.1-dev to SDXL for LoRA training. SDXL fits comfortably in the available RAM and VRAM. See [Project 01: LoRA Training](../projects/01-lora-training/) for the full story.

---

## Issue 2: Device Plugin Cache Loss After System Freeze

**Symptom:** After recovering from the FLUX OOM freeze (Issue 1), GPU workloads refused to start. Pods were stuck in `CreateContainerConfigError`. The error in events:

```
Warning  Failed  ...  kubelet  Error: endpoint not found in cache for a registered resource: nvidia.com/gpu
```

**Root cause:** The NVIDIA Device Plugin DaemonSet maintains an in-memory cache of GPU endpoints (the mapping between `nvidia.com/gpu` resources and actual physical GPUs). When the system froze and was hard-rebooted, this cache was lost. The kubelet still believed GPUs were allocated (from its checkpoint file), but the device plugin had no record of those allocations.

The disconnect:

- **kubelet** thought: "I have 3 GPUs, 1 is allocated to pod X"
- **device plugin** thought: "I just started fresh, I have no allocations"
- When a pod tried to start, kubelet asked the device plugin for the GPU endpoint, and the plugin could not find it in its empty cache

**Fix:** Restart the NVIDIA Device Plugin DaemonSet to force re-registration of all GPUs:

```bash
kubectl rollout restart daemonset -n nvidia-device-plugin \
  $(kubectl get daemonset -n nvidia-device-plugin -o jsonpath='{.items[0].metadata.name}')
```

After the restart, the device plugin re-enumerates all GPUs and registers them with the kubelet. Existing pods may need to be deleted and recreated (the deployment controller handles this automatically if the pod is managed by a Deployment).

Verify GPUs are re-registered:

```bash
kubectl describe node zosmaai | grep nvidia.com/gpu
```

Should show:

```
nvidia.com/gpu:  3
nvidia.com/gpu:  3
```

(One line for Capacity, one for Allocatable.)

**Lesson:** The device plugin's cache is ephemeral. Any hard reboot or system crash can invalidate it. The fix is always the same: rolling restart the DaemonSet. In a production environment, you would want monitoring to detect this state and trigger an automatic restart.

---

## Issue 3: AI-Toolkit Dashboard DATASETS_FOLDER Misconfiguration

**Symptom:** The AI-Toolkit web dashboard loaded correctly, but the dataset list was empty. The API endpoint for listing datasets returned an empty array, even though dataset directories existed on disk.

**Root cause:** AI-Toolkit's Next.js dashboard stores its configuration in a SQLite database. One of the settings in the `Settings` table is `DATASETS_FOLDER`, which tells the API where to look for dataset directories.

The value was pointing to a path inside the dataset directory rather than its parent directory. For example:

- **Incorrect:** `/workspace/datasets/my-dataset` (points to a specific dataset)
- **Correct:** `/workspace/datasets` (points to the parent containing all datasets)

The API's `readdirSync` on the incorrect path found files (images, captions) instead of dataset directories, and since none of them were directories, it returned an empty list.

**Fix:** Update the SQLite database directly:

```bash
# Find the SQLite database
find /data/ai-toolkit -name "*.db" -o -name "*.sqlite" 2>/dev/null

# Connect and update the setting
sqlite3 /data/ai-toolkit/aitoolkit.db
```

```sql
-- Check current value
SELECT * FROM Settings WHERE key = 'DATASETS_FOLDER';

-- Fix it
UPDATE Settings SET value = '/workspace/datasets' WHERE key = 'DATASETS_FOLDER';
```

After updating, restart the AI-Toolkit pod:

```bash
kubectl rollout restart deployment ai-toolkit -n ai-toolkit
```

**Lesson:** When a web UI works but data is missing, check the configuration layer between the API and the filesystem. SQLite databases embedded in applications are easy to overlook during debugging because they are not part of the Kubernetes manifest or the container environment variables.

---

## Issue 4: Node.js Symlink Resolution with fs.readdirSync

**Symptom:** A dataset directory was created as a symlink (to avoid duplicating large training images), but AI-Toolkit's dashboard did not recognize it as a directory. It appeared as a regular file in the API's directory listing, and the dataset was invisible in the UI.

**Root cause:** AI-Toolkit's Next.js API uses `fs.readdirSync()` with the `withFileTypes` option to list dataset directories. The `Dirent` objects returned by this method have an `isDirectory()` method that checks the type of the directory entry itself -- not what it points to.

For symlinks, `isDirectory()` returns `false` because the directory entry is a symlink, not a directory. The correct method would be `isSymbolicLink()` followed by `fs.statSync()` to resolve the target, or using `fs.readdirSync` without `withFileTypes` and calling `fs.statSync` on each entry (which follows symlinks).

```javascript
// This is what the code does:
const entries = fs.readdirSync(path, { withFileTypes: true });
const dirs = entries.filter(e => e.isDirectory());  // Symlinks are excluded!

// What it should do to follow symlinks:
const entries = fs.readdirSync(path, { withFileTypes: true });
const dirs = entries.filter(e => e.isDirectory() || e.isSymbolicLink());
```

**Fix:** Replace the symlink with an actual copy of the dataset directory:

```bash
# Remove the symlink
rm /data/ai-toolkit/datasets/my-dataset

# Copy the actual data
cp -r /data/source/my-dataset /data/ai-toolkit/datasets/my-dataset
```

This is a workaround, not a fix for the upstream code. The tradeoff is disk space duplication, but for training datasets (typically a few GB of images), this is acceptable.

**Lesson:** Symlinks and Node.js's `fs.readdirSync` with `withFileTypes` do not mix well. When containers mount hostPath volumes, symlinks that work on the host may not resolve correctly inside the container anyway (if the symlink target is not also mounted). Prefer actual copies for data that needs to be visible inside containers.

---

## Issue 5: Pod Memory Limits and Shared Memory

**Symptom:** Training pods crashed with obscure errors during data loading, or PyTorch DataLoader workers died silently. Sometimes the pod was OOM-killed by Kubernetes despite GPU VRAM being fine.

**Root cause:** Two separate memory-related issues that look similar but have different fixes:

### 5a: Missing memory limits

Without explicit memory limits, the kubelet does not reserve memory for the pod. Under memory pressure, the kernel's OOM killer can target the pod even if other processes are the real culprits. Conversely, without limits, a runaway training process can consume all system RAM and trigger the FLUX OOM issue (Issue 1).

### 5b: Insufficient /dev/shm

PyTorch's DataLoader with `num_workers > 0` uses shared memory (`/dev/shm`) to pass data between worker processes and the main training process. Kubernetes defaults `/dev/shm` to 64 MB. Training workloads with multiple workers and large image batches easily exceed this.

When `/dev/shm` fills up, worker processes crash. PyTorch may report this as a DataLoader error, a broken pipe, or sometimes just a silent hang.

**Fix:** Set both resource limits and shared memory in the pod spec:

```yaml
spec:
  containers:
  - name: ai-toolkit
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: "30Gi"        # System RAM hard limit
      requests:
        nvidia.com/gpu: "1"
        memory: "8Gi"          # Guaranteed minimum
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory           # Backed by tmpfs (RAM)
      sizeLimit: 1Gi           # 1 GB shared memory
```

Key details:

- **`memory: "30Gi"` limit** -- Sets a hard ceiling on system RAM usage. If the process exceeds this, Kubernetes OOM-kills the pod (which is recoverable) instead of letting it consume all system RAM (which causes Issue 1).
- **`memory: "8Gi"` request** -- Guarantees this much RAM is reserved for the pod. The scheduler only places the pod on nodes with at least 8 GB available.
- **`emptyDir` with `medium: Memory`** -- Creates a tmpfs mount at `/dev/shm`. The `sizeLimit: 1Gi` prevents a misbehaving DataLoader from consuming unbounded RAM.

**Lesson:** GPU workloads in Kubernetes need three types of memory management: VRAM (handled by the driver), system RAM (handled by Kubernetes resource limits), and shared memory (handled by the dshm volume mount). Missing any one of these causes different failure modes that can be hard to diagnose.

---

## Quick Reference: Common Recovery Commands

```bash
# Check GPU status
nvidia-smi

# Restart the device plugin after system crash
kubectl rollout restart daemonset -n nvidia-device-plugin \
  $(kubectl get daemonset -n nvidia-device-plugin -o jsonpath='{.items[0].metadata.name}')

# Check for GPU allocation issues
kubectl describe node zosmaai | grep -A 5 nvidia

# Check pod events for GPU errors
kubectl get events --all-namespaces --sort-by='.lastTimestamp' | grep -i gpu

# Check device plugin logs
kubectl logs -n nvidia-device-plugin -l app.kubernetes.io/name=nvidia-device-plugin --tail=50

# Verify swap configuration
cat /proc/sys/vm/swappiness  # Should be 1
swapon --show                # Should show the 32 GB swapfile
```
