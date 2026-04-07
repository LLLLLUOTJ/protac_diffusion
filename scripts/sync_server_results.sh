#!/usr/bin/env bash
# shellcheck shell=bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/server_config.sh"

REMOTE_HOST="${REMOTE_HOST:-gpu1-server}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/yanglh/L/protac_diffusion}"
SERVER_SYNC_ROOT="${SERVER_SYNC_ROOT:-${PROJECT_ROOT}/outputs/server_sync}"
SERVER_SYNC_STAMP="${SERVER_SYNC_STAMP:-$(date +%F)}"
SERVER_SYNC_KEEP="${SERVER_SYNC_KEEP:-1}"
SERVER_SYNC_INCLUDE_IMAGES="${SERVER_SYNC_INCLUDE_IMAGES:-true}"

SYNC_DIR="${SERVER_SYNC_ROOT}/${SERVER_SYNC_STAMP}"

copy_from_server() {
  local remote_rel="$1"
  local local_path="$2"
  mkdir -p "$(dirname "${local_path}")"
  scp "${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/${remote_rel}" "${local_path}"
}

copy_from_server_if_exists() {
  local remote_rel="$1"
  local local_path="$2"
  if ssh -o BatchMode=yes "${REMOTE_HOST}" "test -f '${REMOTE_PROJECT_ROOT}/${remote_rel}'"; then
    copy_from_server "${remote_rel}" "${local_path}"
  fi
}

prune_old_snapshots() {
  local keep="${1}"
  local index=0
  while IFS= read -r snapshot_path; do
    local snapshot="${snapshot_path##*/}"
    index=$((index + 1))
    if (( index > keep )); then
      rm -rf "${SERVER_SYNC_ROOT}/${snapshot}"
      echo "[prune] removed ${SERVER_SYNC_ROOT}/${snapshot}"
    fi
  done < <(
    find "${SERVER_SYNC_ROOT}" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name '20??-??-??' | sort -r
  )
}

echo "[sync] remote=${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}"
echo "[sync] local=${SYNC_DIR}"

rm -rf "${SYNC_DIR}"
mkdir -p "${SYNC_DIR}/images"

copy_from_server "outputs/linker_token_eval/summary.json" "${SYNC_DIR}/linker_token_eval.summary.json"
copy_from_server "outputs/linker_token_eval/per_source_summary.csv" "${SYNC_DIR}/linker_token_eval.per_source_summary.csv"
copy_from_server "outputs/linker_token_eval/all_generations.json" "${SYNC_DIR}/linker_token_eval.all_generations.json"
copy_from_server "outputs/linker_token_metrics/summary.json" "${SYNC_DIR}/linker_token_metrics.summary.json"
copy_from_server "checkpoints/linker_token_diffusion.summary.json" "${SYNC_DIR}/linker_token_diffusion.summary.json"
copy_from_server_if_exists "outputs/linker_smiles_compare/summary.json" "${SYNC_DIR}/linker_smiles_compare.summary.json"
copy_from_server_if_exists "outputs/linker_smiles_compare/distributions.png" "${SYNC_DIR}/images/linker_smiles_compare.distributions.png"
copy_from_server_if_exists "outputs/linker_smiles_compare/descriptors.csv" "${SYNC_DIR}/linker_smiles_compare.descriptors.csv"

if [[ "${SERVER_SYNC_INCLUDE_IMAGES}" == "true" ]]; then
  copy_from_server "outputs/linker_token_eval/evaluation_overview.png" "${SYNC_DIR}/images/linker_token_eval.evaluation_overview.png"
  copy_from_server "outputs/linker_token_metrics/overview.png" "${SYNC_DIR}/images/linker_token_metrics.overview.png"
  copy_from_server "outputs/linker_token_sampling/sample_000_anchored_linker.png" "${SYNC_DIR}/images/sample_000_anchored_linker.png"
  copy_from_server "outputs/linker_token_sampling/sample_000_full.png" "${SYNC_DIR}/images/sample_000_full.png"
  copy_from_server "outputs/linker_token_sampling/sample_001_anchored_linker.png" "${SYNC_DIR}/images/sample_001_anchored_linker.png"
  copy_from_server "outputs/linker_token_sampling/sample_001_full.png" "${SYNC_DIR}/images/sample_001_full.png"
fi

ln -sfn "${SERVER_SYNC_STAMP}" "${SERVER_SYNC_ROOT}/latest"
prune_old_snapshots "${SERVER_SYNC_KEEP}"

echo "[done] synced into ${SYNC_DIR}"
