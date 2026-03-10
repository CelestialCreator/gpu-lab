# llama-server — Local LLM Inference on Kubernetes

Serves **Qwen3.5-35B-A3B** (Q8_0, 36.9GB) via llama.cpp's `llama-server` on Kubernetes with multi-GPU support. Provides an OpenAI-compatible API for use with Claude Code, Cursor, opencode, or any chat client.

## Architecture

```
Claude Code / Cursor / Browser UI
            ↓
   http://localhost:30080/v1  (NodePort)
            ↓
┌──────────────────────────────────────────────┐
│  K8s Pod: llama-server                       │
│                                              │
│  Model: Qwen3.5-35B-A3B (Q8_0, 36.9GB)     │
│  GPU 0: RTX 5090  — 28.2 GB  (80% layers)  │
│  GPU 1: RTX 2070S —  6.5 GB  (20% layers)  │
│  API: OpenAI-compatible /v1/chat/completions │
│  Speed: ~75 tok/s generation                 │
└──────────────────────────────────────────────┘
```

## Why This Stack

- **Qwen3.5-35B-A3B** — MoE with 35B total / 3B active params per token. Best quality-per-FLOP at this size.
- **Q8_0 quantization** — Near-lossless 8-bit, fits across two GPUs with room for KV cache.
- **llama-server** — Lightweight single binary, native multi-GPU tensor splitting, OpenAI-compatible API, built-in chat UI.
- **Multi-GPU** — `--tensor-split 32,8` distributes layers proportionally across RTX 5090 (32GB) + RTX 2070 SUPER (8GB).

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds llama.cpp with CUDA from source |
| `deployment.yaml` | K8s Deployment + NodePort Service |
| `.env` | GPU UUIDs (gitignored) |
| `apply.sh` | Deploys with envsubst for GPU pinning |

## Setup

### 1. Build & Push Docker Image

```bash
cd workloads/llama-server
docker build -t localhost:5000/llama-server:latest .
docker push localhost:5000/llama-server:latest
```

### 2. Download Model

```bash
mkdir -p /home/akshay/llama-workspace/models
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'unsloth/Qwen3.5-35B-A3B-GGUF',
    local_dir='/home/akshay/llama-workspace/models/Qwen3.5-35B-A3B-GGUF',
    allow_patterns=['*Q8_0*']
)
"
```

### 3. Configure `.env`

```bash
# .env
GPU_UUID=<5090-uuid>,<2070s-uuid>
```

### 4. Deploy

```bash
./apply.sh deployment.yaml
```

### 5. Verify

```bash
# Check pod
kubectl logs -f deployment/llama-server

# Test API
curl http://localhost:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-35B-A3B-Q8_0.gguf","messages":[{"role":"user","content":"Hello!"}]}'
```

## Usage

### Browser UI

Open `http://localhost:30080` — llama-server has a built-in chat interface.

### SSH Port Forward (Remote Access)

```bash
ssh -p 4224 -L 30080:localhost:30080 your_username@devserver.zosma.ai
# Then open http://localhost:30080 in your browser
```

### Claude Code

Add to `~/.claude/settings.json` (prevents KV cache invalidation):
```json
{
  "env": {
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

Then run:
```bash
ANTHROPIC_BASE_URL="http://localhost:30080" \
ANTHROPIC_API_KEY="sk-no-key-required" \
claude --model Qwen3.5-35B-A3B-Q8_0.gguf
```

### Cursor / opencode / Any OpenAI Client

Set base URL to `http://localhost:30080/v1` and API key to any non-empty string.

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--ctx-size` | 16384 | Reduced from 32K to fit VRAM budget |
| `--cache-type-k` | q8_0 | Quantized KV cache to save VRAM |
| `--cache-type-v` | q8_0 | Same |
| `--tensor-split` | 32,8 | VRAM ratio: 5090 gets 80%, 2070S gets 20% |
| `-ngl` | 99 | Offload all layers to GPU |
| `--temp` | 0.6 | Default sampling temperature |
