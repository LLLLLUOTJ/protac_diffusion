#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "${TOKEN_SAMPLE_OUT_DIR}"

TOKEN_SAMPLE_ARGS=(
  python "${PROJECT_ROOT}/sample_linker_token.py"
  --tensor-pt "${WEAK_ANCHOR_TOKEN_PT}"
  --ckpt "${TOKEN_CKPT}"
  --sample-index "${TOKEN_SAMPLE_INDEX}"
  --num-samples "${TOKEN_NUM_GENERATIONS}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --log-every "${TOKEN_SAMPLE_LOG_EVERY}"
  --out-dir "${TOKEN_SAMPLE_OUT_DIR}"
)

if [[ -n "${TOKEN_SAMPLE_ID}" ]]; then
  TOKEN_SAMPLE_ARGS+=(--sample-id "${TOKEN_SAMPLE_ID}")
fi
if [[ "${TOKEN_SAMPLE_SHOW_PROGRESS}" == "true" ]]; then
  TOKEN_SAMPLE_ARGS+=(--show-progress)
fi
if [[ "${TOKEN_SAMPLE_SAVE_IMAGES}" == "true" ]]; then
  TOKEN_SAMPLE_ARGS+=(--save-images)
fi

echo "[run] sampling token-conditioned linkers"
run_in_env "${TOKEN_SAMPLE_ARGS[@]}"

echo "[done] token_samples_dir=${TOKEN_SAMPLE_OUT_DIR}"
