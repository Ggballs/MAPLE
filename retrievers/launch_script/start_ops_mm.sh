#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

GPU=0
PORT=18088
HOST=0.0.0.0
ENV_NAME="${MAPLE_VLM_ENV_NAME:-maple-vlm}"
LOG_DIR="$(services_default_log_dir)"
MODEL_PATH="${OPS_MM_MODEL_PATH:-}"
EXTRA_PYTHONPATH="${MAPLE_EXTRA_PYTHONPATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --pythonpath) EXTRA_PYTHONPATH="$2"; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 1 ;;
  esac
done

services_ensure_log_dir "${LOG_DIR}"
cmd=(
  retrievers/models/servers/serve_multimodal_embedding.py
  --model ops-mm-embed-7b
  --device cuda:0
  --host "${HOST}"
  --port "${PORT}"
)
if [[ -n "${MODEL_PATH}" ]]; then
  cmd+=(--model-path "${MODEL_PATH}")
fi
services_start_python_service \
  "${ENV_NAME}" \
  "${GPU}" \
  "${LOG_DIR}/ops_mm_${PORT}.log" \
  "http://127.0.0.1:${PORT}/health" \
  "${EXTRA_PYTHONPATH}" \
  "${cmd[@]}"
