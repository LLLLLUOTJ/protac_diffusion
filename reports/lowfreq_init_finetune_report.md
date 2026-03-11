# Low-Frequency Initialization + Full-Vocab Fine-Tune

Date: 2026-03-11

## Workflow

1. Split tokens by frequency threshold (`freq >= 10` as core).
2. For each low-frequency token, assign nearest core token by chemical-rule + structure similarity.
3. Initialize low-frequency token embeddings from assigned core embedding when confidence is `high` or `medium`.
4. Keep uncertain tokens as random init.
5. Continue SGNS training on full tokenized samples.

## Mapping stats

- total tokens: `103`
- core tokens: `30`
- low-frequency tokens: `73`
- assigned high: `5`
- assigned medium: `10`
- uncertain: `58`

Files:
- `/Users/lintianjian/diffusion/data/processed/core_token_table.csv`
- `/Users/lintianjian/diffusion/data/processed/lowfreq_to_core_assignment.csv`
- `/Users/lintianjian/diffusion/data/processed/lowfreq_uncertain_tokens.csv`
- `/Users/lintianjian/diffusion/reports/lowfreq_grouping_table.md`

## Fine-tune run

Command:

```bash
conda run -n diffusion python /Users/lintianjian/diffusion/train_full_token_embedding_with_init.py \
  --tokenized_csv /Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_multi.csv \
  --coverage_csv /Users/lintianjian/diffusion/data/processed/linker_anchor_token_coverage_stats.csv \
  --assignment_csv /Users/lintianjian/diffusion/data/processed/lowfreq_to_core_assignment.csv \
  --core_embedding_pt /Users/lintianjian/diffusion/data/processed/core_token_embedding/token_embeddings.pt \
  --out_dir /Users/lintianjian/diffusion/data/processed/full_token_embedding_initialized \
  --use_uncertain_suggestions false \
  --epochs 80 --batch_size 1024 --window_size 2 --negative_samples 6 --learning_rate 0.015 --device cpu
```

Run summary:
- sequences: `2721`
- vocab: `103`
- skip-gram pairs: `101912`
- embedding dim: `16`
- init sources:
  - `core_exact`: `30`
  - `mapped_high`: `5`
  - `mapped_medium`: `10`
  - `random_unmapped`: `58`

## Outputs

- `/Users/lintianjian/diffusion/data/processed/full_token_embedding_initialized/token_vocab.json`
- `/Users/lintianjian/diffusion/data/processed/full_token_embedding_initialized/token_embeddings.pt`
- `/Users/lintianjian/diffusion/data/processed/full_token_embedding_initialized/token_embeddings.npy`
- `/Users/lintianjian/diffusion/data/processed/full_token_embedding_initialized/token_init_sources.csv`
- `/Users/lintianjian/diffusion/data/processed/full_token_embedding_initialized/training_summary.json`
- `/Users/lintianjian/diffusion/data/processed/full_token_embedding_initialized/token_neighbor_report.json`
