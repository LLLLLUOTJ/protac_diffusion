#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary
mkdir -p "${WEAK_ANCHOR_OUT_DIR}" "$(dirname "${WEAK_ANCHOR_TENSOR_PT}")"

BUILD_ARGS=(
  python "${PROJECT_ROOT}/build_weak_anchor_dataset.py"
  --protac_csv "${PROTAC_CSV}"
  --linker_csv "${LINKER_CSV}"
  --out_csv "${WEAK_ANCHOR_CSV}"
  --rej_csv "${WEAK_ANCHOR_REJ_CSV}"
  --summary_json "${WEAK_ANCHOR_SUMMARY_JSON}"
  --dedupe_protacs "${DEDUPE_PROTACS}"
  --dedupe_linkers "${DEDUPE_LINKERS}"
  --min_fragment_heavy_atoms "${MIN_FRAGMENT_HEAVY_ATOMS}"
  --min_linker_heavy_atoms "${MIN_LINKER_HEAVY_ATOMS}"
  --min_anchor_graph_distance "${MIN_ANCHOR_GRAPH_DISTANCE}"
  --min_linker_ratio_pct "${MIN_LINKER_RATIO_PCT}"
  --max_linker_ratio_pct "${MAX_LINKER_RATIO_PCT}"
  --log_no_match_rejections "${LOG_NO_MATCH_REJECTIONS}"
)

if [[ -n "${MAX_PAIRS}" ]]; then
  BUILD_ARGS+=(--max_pairs "${MAX_PAIRS}")
fi
if [[ -n "${WEAK_ANCHOR_DEBUG_JSON}" ]]; then
  BUILD_ARGS+=(--debug_json "${WEAK_ANCHOR_DEBUG_JSON}")
fi

echo "[run] building weak-anchor csv"
run_in_env "${BUILD_ARGS[@]}"

TENSOR_ARGS=(
  python "${PROJECT_ROOT}/build_weak_anchor_tensor_dataset.py"
  --csv "${WEAK_ANCHOR_CSV}"
  --out "${WEAK_ANCHOR_TENSOR_PT}"
)
if [[ "${INCLUDE_PAIR_MASK}" == "true" ]]; then
  TENSOR_ARGS+=(--include-pair-mask)
fi

echo "[run] building weak-anchor tensor dataset"
run_in_env "${TENSOR_ARGS[@]}"

echo "[done] csv=${WEAK_ANCHOR_CSV}"
echo "[done] tensor=${WEAK_ANCHOR_TENSOR_PT}"
