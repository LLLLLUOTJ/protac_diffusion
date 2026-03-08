#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "${EVAL_OUT_DIR}"

EVAL_ARGS=(
  python "${PROJECT_ROOT}/evaluate_linker_generation.py"
  --tensor-pt "${WEAK_ANCHOR_TENSOR_PT}"
  --node-ckpt "${NODE_CKPT}"
  --mode "${EVAL_MODE}"
  --max-source-samples "${EVAL_MAX_SOURCE_SAMPLES}"
  --start-index "${EVAL_START_INDEX}"
  --num-samples-per-source "${EVAL_NUM_SAMPLES_PER_SOURCE}"
  --edge-threshold "${EVAL_EDGE_THRESHOLD}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --log-every "${EVAL_LOG_EVERY}"
  --out-dir "${EVAL_OUT_DIR}"
)

if [[ "${EVAL_MODE}" == "joint" ]]; then
  EVAL_ARGS+=(--edge-ckpt "${EDGE_CKPT}")
fi
if [[ "${EVAL_SHUFFLE}" == "true" ]]; then
  EVAL_ARGS+=(--shuffle)
fi
if [[ "${EVAL_SHOW_PROGRESS}" == "true" ]]; then
  EVAL_ARGS+=(--show-progress)
fi
if [[ "${EVAL_SAVE_IMAGES}" == "true" ]]; then
  EVAL_ARGS+=(--save-images)
fi

echo "[run] evaluating linker generation"
run_in_env "${EVAL_ARGS[@]}"

echo "[done] eval_dir=${EVAL_OUT_DIR}"
