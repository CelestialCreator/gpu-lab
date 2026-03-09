#!/bin/bash
# Apply a k8s job YAML with secrets from .env
# Usage: ./apply.sh k8s/job-grpo-train.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Error: .env file not found at $ENV_FILE"
  exit 1
fi

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <yaml-file>"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

envsubst '${HF_TOKEN} ${WANDB_API_KEY} ${GPU_UUID}' < "$1" | kubectl apply -f -
