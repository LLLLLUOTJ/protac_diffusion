#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "${TRAIN_ANALYSIS_OUT_DIR}"

echo "[run] analyzing train log ${TRAIN_LOG_PATH}"
run_in_env python "${PROJECT_ROOT}/analyze_train_log.py" \
  --log "${TRAIN_LOG_PATH}" \
  --out-dir "${TRAIN_ANALYSIS_OUT_DIR}"

echo "[done] analysis_dir=${TRAIN_ANALYSIS_OUT_DIR}"
