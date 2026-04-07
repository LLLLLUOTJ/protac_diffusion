# Fragment-Token Pipeline Report (Based on Available Logs)

Compiled on: 2026-04-02  
Scope: summarize current fragment/token method status using only the files provided in `Downloads`.

## Inputs Used

- [/Users/lintianjian/Downloads/run_token_pipeline.log](/Users/lintianjian/Downloads/run_token_pipeline.log)
- [/Users/lintianjian/Downloads/run_pipeline.log](/Users/lintianjian/Downloads/run_pipeline.log)
- [/Users/lintianjian/Downloads/summary.json](/Users/lintianjian/Downloads/summary.json)

## Executive Summary

- The fragment/token pipeline has completed end-to-end and is stable.
- Token diffusion training converged on CUDA with strong validation loss (`best_val=0.0144`).
- Generation quality is operationally strong (`decode_rate=1.0`, `assembly_rate=1.0`).
- The model is no longer length-locked (`same_token_length_rate=0.0703`), which is consistent with explicit PAD-position learning.
- Compared to the older node/edge graph pipeline run, the token pipeline is currently the stronger route for downstream iteration.

## 1) Fragment/Token Training Results

### 1.1 Oriented Token Embedding Stage

Evidence: `run_token_pipeline.log`

- Dataset: `5020` sequences
- Vocab size: `126`
- Skip-gram pairs: `432148`
- Device: `cpu`
- Final logged embedding loss: `weighted_loss=2.358146` (epoch 40)
- Output:
  - `token_vocab.json`
  - `token_embeddings.pt`
  - `token_embeddings.npy`

Interpretation:

- Embedding pretraining ran successfully and produced all expected artifacts.
- This stage is currently CPU-bound by script config; this is expected behavior.

### 1.2 Token Diffusion Stage

Evidence: `run_token_pipeline.log`

- Dataset (post-filter):
  - total `4976`
  - train `4479`
  - val `497`
  - filtered (`token_sequence_too_long`): `44`
- PAD setup in effect:
  - `learn_pad_positions=True`
  - `pad_token=<PAD>`
- Device: `cuda`
- Convergence:
  - epoch 1: `train=0.3863`, `val=0.1364`
  - best: `best_epoch=72`, `best_val=0.0144`
  - last logged epoch: `87` (`train=0.0341`, `val=0.0157`)
  - early stop at epoch `87` with patience `15`
- Output checkpoint:
  - `checkpoints/linker_token_diffusion.pt`

Interpretation:

- Training is stable, convergent, and not showing the failure pattern seen in earlier graph sampling runs.

## 2) Generation/Evaluation Results (Token Method)

Evidence: `summary.json` + `run_token_pipeline.log`

### 2.1 Core Success Metrics

| Metric | Value |
|---|---:|
| num_source_samples | 32 |
| num_requested | 128 |
| decode_rate | 1.0000 |
| assembly_rate | 1.0000 |
| unique_anchored | 128 |
| unique_full | 128 |

### 2.2 Memorization/Novelty/Diversity

| Metric | Value |
|---|---:|
| exact_source_match_rate | 0.0000 |
| exact_train_match_rate | 0.0469 |
| exact_token_match_rate | 0.0000 |
| train_nn_similarity_mean | 0.5763 |
| source_similarity_mean | 0.1816 |
| internal_diversity_mean | 0.8105 |

Interpretation:

- No direct source-copy behavior observed (`exact_source_match_rate=0`).
- Diversity is high for this sample size (`internal_diversity_mean=0.8105`).
- Some training-space proximity remains (`train_nn_similarity_mean=0.5763`), but not in a collapse regime.

### 2.3 Length Behavior (PAD/stop-related)

| Metric | Value |
|---|---:|
| same_token_length_rate | 0.0703 |

Interpretation:

- Generated length is now variable and no longer tied to source length, consistent with PAD-position learning being active.

### 2.4 Descriptor Drift (Generated - Source, mean)

| Descriptor | Delta Mean |
|---|---:|
| heavy_atoms | +1.0547 |
| hetero_atoms | -0.5156 |
| ring_count | +0.4219 |
| rotatable_bonds | -0.5703 |
| tpsa | -8.8531 |
| anchor_distance | +0.1172 |

Interpretation:

- Current generation still trends slightly toward lower polarity (`TPSA` down) and somewhat more ring-rich structures.
- Drift is milder than earlier token snapshots where TPSA/hetero drift was larger.

## 3) Comparison With Older Graph Pipeline Run

Evidence: `run_pipeline.log`

- Old graph run (`node + edge`) completed training:
  - node best val around `0.1213` (epoch 30)
  - edge best val around `0.5065` (epoch 99)
- But same run’s sample step ended with:
  - `decoded=0/8`, `assembled=0/8`
- And evaluation script was not executed due:
  - `Permission denied` on `evaluate_linker_generation.sh`

Interpretation:

- For the current objective, the fragment/token path is clearly ahead in end-to-end usability and output validity.

## 4) What Is Missing From This Report

Not available in the provided bundle:

- `checkpoints/linker_token_diffusion.history.csv`
- `checkpoints/linker_token_diffusion.summary.json`
- full per-row generation file for this exact run (`all_generations.json/csv`)

Without these, this report cannot include:

- epoch-by-epoch checkpoint gap diagnostics
- stop-index histogram and PAD suffix violation rate for this exact run

## 5) Recommended Next Step

Proceed with token pipeline as primary branch and keep graph pipeline as baseline reference only.

Immediate follow-up once files are available:

1. Add stop-index/PAD suffix consistency metrics for this run.
2. Run a controlled comparison across 2-3 random seeds for token diffusion.
3. If needed, tighten PAD-suffix behavior with an explicit suffix regularization term.
