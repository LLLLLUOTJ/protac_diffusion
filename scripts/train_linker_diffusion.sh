#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "${CHECKPOINT_DIR}"

NODE_ARGS=(
  python "${PROJECT_ROOT}/train_linker_node_diffusion.py"
  --tensor-pt "${WEAK_ANCHOR_TENSOR_PT}"
  --epochs "${NODE_EPOCHS}"
  --batch-size "${NODE_BATCH_SIZE}"
  --lr "${NODE_LR}"
  --weight-decay "${NODE_WEIGHT_DECAY}"
  --hidden-dim "${NODE_HIDDEN_DIM}"
  --layers "${NODE_LAYERS}"
  --dropout "${NODE_DROPOUT}"
  --condition-dropout "${NODE_CONDITION_DROPOUT}"
  --timesteps "${NODE_TIMESTEPS}"
  --beta-start "${NODE_BETA_START}"
  --beta-end "${NODE_BETA_END}"
  --val-ratio "${NODE_VAL_RATIO}"
  --patience "${NODE_PATIENCE}"
  --min-delta "${NODE_MIN_DELTA}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --out "${NODE_CKPT}"
)
if [[ -n "${NODE_MAX_SAMPLES}" ]]; then
  NODE_ARGS+=(--max-samples "${NODE_MAX_SAMPLES}")
fi

EDGE_ARGS=(
  python "${PROJECT_ROOT}/train_linker_edge_diffusion.py"
  --tensor-pt "${WEAK_ANCHOR_TENSOR_PT}"
  --epochs "${EDGE_EPOCHS}"
  --batch-size "${EDGE_BATCH_SIZE}"
  --lr "${EDGE_LR}"
  --weight-decay "${EDGE_WEIGHT_DECAY}"
  --hidden-dim "${EDGE_HIDDEN_DIM}"
  --layers "${EDGE_LAYERS}"
  --dropout "${EDGE_DROPOUT}"
  --condition-dropout "${EDGE_CONDITION_DROPOUT}"
  --timesteps "${EDGE_TIMESTEPS}"
  --beta-start "${EDGE_BETA_START}"
  --beta-end "${EDGE_BETA_END}"
  --val-ratio "${EDGE_VAL_RATIO}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --out "${EDGE_CKPT}"
)
if [[ -n "${EDGE_MAX_SAMPLES}" ]]; then
  EDGE_ARGS+=(--max-samples "${EDGE_MAX_SAMPLES}")
fi

if [[ "${TRAIN_NODE}" == "true" ]]; then
  echo "[run] training node diffusion"
  run_in_env "${NODE_ARGS[@]}"
fi

if [[ "${TRAIN_EDGE}" == "true" ]]; then
  echo "[run] training edge diffusion"
  run_in_env "${EDGE_ARGS[@]}"
fi

echo "[done] node_ckpt=${NODE_CKPT}"
echo "[done] edge_ckpt=${EDGE_CKPT}"
