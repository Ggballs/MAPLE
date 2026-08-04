#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash retrievers/launch_script/start_service.sh <model> [args...]

Models:
  qwen3-embed-8b
  bge-m3
  multi-model
  gritlm-7b
  qwen3-vl-embed-8b
  ops-mm-embed-7b
EOF
  exit 1
fi

model="$1"
shift

case "${model}" in
  qwen3-embed-8b) exec bash "${SCRIPT_DIR}/start_qwen3_embed.sh" "$@" ;;
  bge-m3) exec bash "${SCRIPT_DIR}/start_bge_m3.sh" "$@" ;;
  multi-model) exec bash "${SCRIPT_DIR}/start_multi_model.sh" "$@" ;;
  gritlm-7b) exec bash "${SCRIPT_DIR}/start_gritlm.sh" "$@" ;;
  qwen3-vl-embed-8b) exec bash "${SCRIPT_DIR}/start_qwen3_vl.sh" "$@" ;;
  ops-mm-embed-7b) exec bash "${SCRIPT_DIR}/start_ops_mm.sh" "$@" ;;
  *)
    printf 'Unknown model: %s\n' "${model}" >&2
    exit 1
    ;;
esac
