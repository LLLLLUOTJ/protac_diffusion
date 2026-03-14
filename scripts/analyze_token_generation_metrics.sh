#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "${TOKEN_METRICS_OUT_DIR}"

echo "[run] analyzing token generation metrics"
run_in_env python "${PROJECT_ROOT}/analyze_token_generation_metrics.py" \
  --all_generations_json "${TOKEN_METRICS_ALL_GENERATIONS_JSON}" \
  --train_weak_anchor_csv "${TOKEN_METRICS_TRAIN_CSV}" \
  --out_dir "${TOKEN_METRICS_OUT_DIR}"

echo "[done] token_metrics_dir=${TOKEN_METRICS_OUT_DIR}"
