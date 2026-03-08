#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "${SAMPLE_OUT_DIR}"

SAMPLE_ARGS=(
  python "${PROJECT_ROOT}/sample_linker.py"
  --tensor-pt "${WEAK_ANCHOR_TENSOR_PT}"
  --node-ckpt "${NODE_CKPT}"
  --mode "${SAMPLE_MODE}"
  --sample-index "${SAMPLE_INDEX}"
  --num-samples "${NUM_GENERATIONS}"
  --edge-threshold "${EDGE_THRESHOLD}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --log-every "${SAMPLE_LOG_EVERY}"
  --out-dir "${SAMPLE_OUT_DIR}"
)

if [[ "${SAMPLE_MODE}" == "joint" ]]; then
  SAMPLE_ARGS+=(--edge-ckpt "${EDGE_CKPT}")
fi

if [[ -n "${SAMPLE_ID}" ]]; then
  SAMPLE_ARGS+=(--sample-id "${SAMPLE_ID}")
fi
if [[ "${SAMPLE_SHOW_PROGRESS}" == "true" ]]; then
  SAMPLE_ARGS+=(--show-progress)
fi
if [[ "${SAMPLE_SAVE_IMAGES}" == "true" ]]; then
  SAMPLE_ARGS+=(--save-images)
fi

echo "[run] sampling linker generations"
run_in_env "${SAMPLE_ARGS[@]}"

echo "[done] samples_dir=${SAMPLE_OUT_DIR}"
