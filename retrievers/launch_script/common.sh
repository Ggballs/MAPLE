#!/usr/bin/env bash

set -euo pipefail

services_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/../.." && pwd
}

services_default_log_dir() {
  if [[ -n "${MAPLE_SERVICE_LOG_DIR:-}" ]]; then
    printf '%s\n' "${MAPLE_SERVICE_LOG_DIR}"
  else
    printf '%s\n' "outputs/experiments/service_logs"
  fi
}

services_ensure_log_dir() {
  local log_dir="$1"
  mkdir -p "${log_dir}"
}

services_conda_sh() {
  if [[ -n "${CONDA_SH_PATH:-}" ]]; then
    printf '%s\n' "${CONDA_SH_PATH}"
    return
  fi
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    printf '%s\n' "${HOME}/anaconda3/etc/profile.d/conda.sh"
    return
  fi
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    printf '%s\n' "${HOME}/miniconda3/etc/profile.d/conda.sh"
    return
  fi
  printf '%s\n' "Unable to locate conda.sh. Set CONDA_SH_PATH explicitly." >&2
  return 1
}

services_wait_for_health() {
  local url="$1"
  local timeout_seconds="${2:-120}"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

services_start_python_service() {
  local env_name="$1"
  local gpu="$2"
  local log_path="$3"
  local health_url="$4"
  local extra_pythonpath="$5"
  shift 5

  local conda_sh
  conda_sh="$(services_conda_sh)"
  local repo_root
  repo_root="$(services_repo_root)"

  local shell_cmd
  local full_pythonpath="${repo_root}"
  if [[ -n "${extra_pythonpath}" ]]; then
    full_pythonpath="${full_pythonpath}:${extra_pythonpath}"
  fi

  local no_proxy_value="${NO_PROXY:-${no_proxy:-}}"
  if [[ -z "${no_proxy_value}" ]]; then
    no_proxy_value="127.0.0.1,localhost"
  elif [[ ",${no_proxy_value}," != *",127.0.0.1,"* ]]; then
    no_proxy_value="${no_proxy_value},127.0.0.1,localhost"
  fi

  shell_cmd=$(
    printf 'source %q && conda activate %q && cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q NO_PROXY=%q no_proxy=%q python %s' \
      "${conda_sh}" \
      "${env_name}" \
      "${repo_root}" \
      "${gpu}" \
      "${full_pythonpath}" \
      "${no_proxy_value}" \
      "${no_proxy_value}" \
      "$(printf '%q ' "$@")"
  )

  nohup bash -lc "${shell_cmd}" >"${log_path}" 2>&1 &
  local pid=$!

  if services_wait_for_health "${health_url}" 180; then
    printf 'Started pid=%s health=%s log=%s\n' "${pid}" "${health_url}" "${log_path}"
    return 0
  fi

  printf 'Service failed to become healthy. pid=%s log=%s\n' "${pid}" "${log_path}" >&2
  tail -n 120 "${log_path}" >&2 || true
  return 1
}
