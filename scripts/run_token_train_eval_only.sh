#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary

if [[ ! -f "${WEAK_ANCHOR_TOKEN_PT}" ]]; then
  echo "[error] token tensor dataset not found: ${WEAK_ANCHOR_TOKEN_PT}" >&2
  echo "[hint] run bash scripts/build_weak_anchor_token_data.sh first" >&2
  exit 1
fi

if [[ -n "${WEAK_ANCHOR_CSV:-}" && -f "${WEAK_ANCHOR_CSV}" ]]; then
  if [[ -z "${TOKEN_METRICS_TRAIN_CSV:-}" || ! -f "${TOKEN_METRICS_TRAIN_CSV}" ]]; then
    export TOKEN_METRICS_TRAIN_CSV="${WEAK_ANCHOR_CSV}"
  fi
  if [[ -z "${SMILES_EVAL_TRAIN_PATH:-}" || ! -f "${SMILES_EVAL_TRAIN_PATH}" ]]; then
    export SMILES_EVAL_TRAIN_PATH="${WEAK_ANCHOR_CSV}"
  fi
  if [[ -z "${SMILES_COMPARE_TRAIN_PATH:-}" || ! -f "${SMILES_COMPARE_TRAIN_PATH}" ]]; then
    export SMILES_COMPARE_TRAIN_PATH="${WEAK_ANCHOR_CSV}"
  fi
fi

echo "[run] reusing token tensor dataset"
echo "[data] token_tensor=${WEAK_ANCHOR_TOKEN_PT}"
if [[ -n "${WEAK_ANCHOR_CSV:-}" ]]; then
  echo "[data] weak_anchor_csv=${WEAK_ANCHOR_CSV}"
fi

bash "${SCRIPT_DIR}/train_linker_token_diffusion.sh"
bash "${SCRIPT_DIR}/sample_linker_token.sh"
bash "${SCRIPT_DIR}/evaluate_linker_token_generation.sh"
bash "${SCRIPT_DIR}/analyze_token_generation_metrics.sh"
bash "${SCRIPT_DIR}/evaluate_generated_molecules.sh"
bash "${SCRIPT_DIR}/analyze_molecule_distributions.sh"
