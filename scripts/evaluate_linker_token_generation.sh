#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "${TOKEN_EVAL_OUT_DIR}"

TOKEN_EVAL_ARGS=(
  python "${PROJECT_ROOT}/evaluate_linker_token_generation.py"
  --tensor-pt "${WEAK_ANCHOR_TOKEN_PT}"
  --token-ckpt "${TOKEN_CKPT}"
  --max-source-samples "${TOKEN_EVAL_MAX_SOURCE_SAMPLES}"
  --start-index "${TOKEN_EVAL_START_INDEX}"
  --num-samples-per-source "${TOKEN_EVAL_NUM_SAMPLES_PER_SOURCE}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --log-every "${TOKEN_EVAL_LOG_EVERY}"
  --out-dir "${TOKEN_EVAL_OUT_DIR}"
)

if [[ "${TOKEN_EVAL_SHUFFLE}" == "true" ]]; then
  TOKEN_EVAL_ARGS+=(--shuffle)
fi
if [[ "${TOKEN_EVAL_SHOW_PROGRESS}" == "true" ]]; then
  TOKEN_EVAL_ARGS+=(--show-progress)
fi
if [[ "${TOKEN_EVAL_SAVE_IMAGES}" == "true" ]]; then
  TOKEN_EVAL_ARGS+=(--save-images)
fi

echo "[run] evaluating token linker generation"
run_in_env "${TOKEN_EVAL_ARGS[@]}"

echo "[done] token_eval_dir=${TOKEN_EVAL_OUT_DIR}"
