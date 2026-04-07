#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "$(dirname "${SMILES_COMPARE_OUTPUT_JSON}")"

cmd=(
  python "${PROJECT_ROOT}/analyze_molecule_distributions.py"
  --generated "${SMILES_COMPARE_GENERATED_PATH}"
  --train "${SMILES_COMPARE_TRAIN_PATH}"
  --smiles-col "${SMILES_COMPARE_SMILES_COL}"
  --train-smiles-col "${SMILES_COMPARE_TRAIN_SMILES_COL}"
  --output-json "${SMILES_COMPARE_OUTPUT_JSON}"
  --output-plot "${SMILES_COMPARE_OUTPUT_PLOT}"
  --output-csv "${SMILES_COMPARE_OUTPUT_CSV}"
)

echo "[run] analyzing molecule distributions"
run_in_env "${cmd[@]}"
echo "[done] compare_json=${SMILES_COMPARE_OUTPUT_JSON}"
echo "[done] compare_plot=${SMILES_COMPARE_OUTPUT_PLOT}"
echo "[done] compare_csv=${SMILES_COMPARE_OUTPUT_CSV}"
