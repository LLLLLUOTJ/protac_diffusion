#!/usr/bin/env bash
# shellcheck shell=bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/server_config.sh"

require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[error] conda not found in PATH" >&2
    exit 1
  fi
}

run_in_env() {
  require_conda
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" "$@"
}

append_if_set() {
  local value="$1"
  shift
  if [[ -n "${value}" ]]; then
    printf '%s\0' "$@" "${value}"
  fi
}

print_config_summary() {
  echo "[config] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[config] CONDA_ENV_NAME=${CONDA_ENV_NAME}"
  echo "[config] DEVICE=${DEVICE}"
}
