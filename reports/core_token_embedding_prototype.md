# Core Linker Token Embedding Prototype

Date: 2026-03-11

## Goal

Train a small, stable, token-level embedding space for core linker tokens (about 30 high-frequency tokens), while preserving attachment semantics such as `*C*`, `*O*`, `*N*`, `*C(*)=O`, `*c1ccc(*)cc1`, `*N1CCN(*)CC1`.

This prototype is **token embedding only** (not linker/sample embedding).

## Training scheme (minimal)

- Model: Skip-gram with negative sampling (SGNS)
- Framework: PyTorch
- Input unit: token sequence per sample (`token_smiles_list_json`)
- Context: fixed window (`window_size`, default 2)
- Small embedding dimension (default 16)
- Weighted by `sample_weight` from input rows
- Deterministic seed for stable runs

Why this setup:
- simple and fast on small vocab/data
- easy to extend to larger vocab later
- keeps exact token strings (including `*`) as independent symbols

## Input format

Primary input:
- `data/processed/linker_anchor_tokenized_core10.csv`

Required columns:
- `sample_id`
- `linker_id`
- `sample_weight`
- `token_smiles_list_json` (JSON list of token strings)

Fallback input:
- `instances_csv` with `sample_id`, `token_smiles`, `token_index`, `sample_weight`

Optional frequency reference:
- `data/processed/linker_anchor_fragment_library_core10.csv`

## Run

```bash
conda run -n diffusion python /Users/lintianjian/diffusion/train_core_token_embedding.py \
  --tokenized_csv /Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_core10.csv \
  --library_csv /Users/lintianjian/diffusion/data/processed/linker_anchor_fragment_library_core10.csv \
  --out_dir /Users/lintianjian/diffusion/data/processed/core_token_embedding \
  --embedding_dim 16 \
  --window_size 2 \
  --negative_samples 6 \
  --epochs 120 \
  --batch_size 1024 \
  --learning_rate 0.02 \
  --device cpu
```

## Output files

In `out_dir`:

- `token_vocab.json`
  - token list
  - `token_to_id`
  - sequence/library frequencies
- `token_embeddings.pt`
  - embedding tensor + metadata
- `token_embeddings.npy`
  - embedding matrix only
- `training_summary.json`

## Neighbor analysis

```bash
conda run -n diffusion python /Users/lintianjian/diffusion/analyze_token_neighbors.py \
  --vocab_json /Users/lintianjian/diffusion/data/processed/core_token_embedding/token_vocab.json \
  --embeddings_pt /Users/lintianjian/diffusion/data/processed/core_token_embedding/token_embeddings.pt \
  --top_k 5 \
  --query_tokens "*C*,*O*,*N*,*C(*)=O,*c1ccc(*)cc1,*N1CCN(*)CC1" \
  --out_json /Users/lintianjian/diffusion/data/processed/core_token_embedding/token_neighbor_report.json
```

Analysis outputs:
- nearest neighbors by cosine similarity
- motif-level quick separation stats (intra/inter-group mean cosine)

## Notes for extension

- To include medium-frequency tokens: adjust upstream core filtering threshold and retrain.
- To initialize larger models: load `token_embeddings.pt` and map by `token_to_id`.
- Token strings are kept exact, so `*O*`, `*O`, `O` remain different entries.
