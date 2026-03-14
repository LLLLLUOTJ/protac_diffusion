#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "${CHECKPOINT_DIR}"

TOKEN_ARGS=(
  python "${PROJECT_ROOT}/train_linker_token_diffusion.py"
  --tensor-pt "${WEAK_ANCHOR_TOKEN_PT}"
  --epochs "${TOKEN_EPOCHS}"
  --batch-size "${TOKEN_BATCH_SIZE}"
  --lr "${TOKEN_LR}"
  --weight-decay "${TOKEN_WEIGHT_DECAY}"
  --hidden-dim "${TOKEN_HIDDEN_DIM}"
  --layers "${TOKEN_LAYERS}"
  --heads "${TOKEN_HEADS}"
  --dropout "${TOKEN_DROPOUT}"
  --condition-dropout "${TOKEN_CONDITION_DROPOUT}"
  --timesteps "${TOKEN_TIMESTEPS}"
  --beta-start "${TOKEN_BETA_START}"
  --beta-end "${TOKEN_BETA_END}"
  --val-ratio "${TOKEN_VAL_RATIO}"
  --patience "${TOKEN_PATIENCE}"
  --min-delta "${TOKEN_MIN_DELTA}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --out "${TOKEN_CKPT}"
)
if [[ -n "${TOKEN_MAX_SAMPLES}" ]]; then
  TOKEN_ARGS+=(--max-samples "${TOKEN_MAX_SAMPLES}")
fi

echo "[run] training token diffusion"
run_in_env "${TOKEN_ARGS[@]}"

echo "[done] token_ckpt=${TOKEN_CKPT}"
