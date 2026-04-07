#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "$(dirname "${SMILES_EVAL_OUTPUT_JSON}")"

cmd=(
  python "${PROJECT_ROOT}/eval_molecules.py"
  --generated "${SMILES_EVAL_GENERATED_PATH}"
  --smiles-col "${SMILES_EVAL_SMILES_COL}"
  --output "${SMILES_EVAL_OUTPUT_JSON}"
)

if [[ -n "${SMILES_EVAL_TRAIN_PATH}" ]]; then
  cmd+=(--train "${SMILES_EVAL_TRAIN_PATH}")
fi
if [[ -n "${SMILES_EVAL_TRAIN_SMILES_COL}" ]]; then
  cmd+=(--train-smiles-col "${SMILES_EVAL_TRAIN_SMILES_COL}")
fi

echo "[run] evaluating generated smiles"
run_in_env "${cmd[@]}"
echo "[done] smiles_eval_json=${SMILES_EVAL_OUTPUT_JSON}"
