#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

print_config_summary

TOKENIZE_ARGS=(
  python "${PROJECT_ROOT}/build_weak_anchor_tokenized_dataset.py"
  --weak_anchor_csv "${WEAK_ANCHOR_CSV}"
  --out_csv "${WEAK_ANCHOR_TOKENIZED_CSV}"
  --summary_json "${WEAK_ANCHOR_TOKENIZED_SUMMARY_JSON}"
)

TOKEN_EMBED_ARGS=(
  python "${PROJECT_ROOT}/train_core_token_embedding.py"
  --tokenized_csv "${WEAK_ANCHOR_TOKENIZED_CSV}"
  --out_dir "${TOKEN_EMBED_DIR}"
  --embedding_dim "${TOKEN_EMBED_DIM}"
  --window_size "${TOKEN_EMBED_WINDOW_SIZE}"
  --negative_samples "${TOKEN_EMBED_NEGATIVE_SAMPLES}"
  --epochs "${TOKEN_EMBED_EPOCHS}"
  --batch_size "${TOKEN_EMBED_BATCH_SIZE}"
  --learning_rate "${TOKEN_EMBED_LR}"
  --pad_to_length "${TOKEN_PAD_TO_LENGTH}"
  --pad_token "${TOKEN_PAD_TOKEN}"
  --learn_pad_token "${TOKEN_LEARN_PAD_TOKEN}"
  --device cpu
)

BUILD_PT_ARGS=(
  python "${PROJECT_ROOT}/build_weak_anchor_token_dataset.py"
  --weak_anchor_csv "${WEAK_ANCHOR_CSV}"
  --token_vocab_json "${TOKEN_VOCAB_JSON}"
  --token_embeddings_pt "${TOKEN_EMBED_PT}"
  --pad_to_length "${TOKEN_PAD_TO_LENGTH}"
  --pad_token "${TOKEN_PAD_TOKEN}"
  --reject_overlength "${TOKEN_REJECT_OVERLENGTH}"
  --learn_pad_positions "${TOKEN_LEARN_PAD_POSITIONS}"
  --out_pt "${WEAK_ANCHOR_TOKEN_PT}"
)

echo "[run] building weak-anchor oriented tokenized csv"
run_in_env "${TOKENIZE_ARGS[@]}"

echo "[run] training oriented token embeddings"
run_in_env "${TOKEN_EMBED_ARGS[@]}"

echo "[run] building weak-anchor token tensor dataset"
run_in_env "${BUILD_PT_ARGS[@]}"

echo "[done] tokenized_csv=${WEAK_ANCHOR_TOKENIZED_CSV}"
echo "[done] token_vocab=${TOKEN_VOCAB_JSON}"
echo "[done] token_embeddings=${TOKEN_EMBED_PT}"
echo "[done] token_tensor=${WEAK_ANCHOR_TOKEN_PT}"
