# Server Run

## Files

- env config: [environment.server.yml](/Users/lintianjian/diffusion/environment.server.yml)
- central runtime config: [scripts/server_config.sh](/Users/lintianjian/diffusion/scripts/server_config.sh)
- env setup: [scripts/setup_env.sh](/Users/lintianjian/diffusion/scripts/setup_env.sh)
- build weak-anchor data: [scripts/build_weak_anchor_data.sh](/Users/lintianjian/diffusion/scripts/build_weak_anchor_data.sh)
- train node + edge diffusion: [scripts/train_linker_diffusion.sh](/Users/lintianjian/diffusion/scripts/train_linker_diffusion.sh)
- sample from trained checkpoints: [scripts/sample_linker.sh](/Users/lintianjian/diffusion/scripts/sample_linker.sh)
- build token diffusion data: [scripts/build_weak_anchor_token_data.sh](/Users/lintianjian/diffusion/scripts/build_weak_anchor_token_data.sh)
- train token diffusion: [scripts/train_linker_token_diffusion.sh](/Users/lintianjian/diffusion/scripts/train_linker_token_diffusion.sh)
- sample token diffusion: [scripts/sample_linker_token.sh](/Users/lintianjian/diffusion/scripts/sample_linker_token.sh)
- evaluate token diffusion: [scripts/evaluate_linker_token_generation.sh](/Users/lintianjian/diffusion/scripts/evaluate_linker_token_generation.sh)
- analyze train log: [scripts/analyze_train_log.sh](/Users/lintianjian/diffusion/scripts/analyze_train_log.sh)
- end-to-end wrapper: [scripts/run_server_pipeline.sh](/Users/lintianjian/diffusion/scripts/run_server_pipeline.sh)
- token end-to-end wrapper: [scripts/run_token_server_pipeline.sh](/Users/lintianjian/diffusion/scripts/run_token_server_pipeline.sh)

## 1) Create Environment

Default is CUDA 12.1:

```bash
bash scripts/setup_env.sh
```

The setup script now fails fast if `TORCH_VARIANT=cuda121` but `torch.cuda.is_available()` is still `False`.

CPU-only:

```bash
TORCH_VARIANT=cpu bash scripts/setup_env.sh
```

## 2) Edit Runtime Config

Change only [scripts/server_config.sh](/Users/lintianjian/diffusion/scripts/server_config.sh) if you need different:

- CSV input paths
- output directories
- linker filtering thresholds
- training epochs / batch size
- sample count / sample index

Most common values:

```bash
CONDA_ENV_NAME=diffusion-server
DEVICE=auto
MIN_LINKER_RATIO_PCT=15
MAX_LINKER_RATIO_PCT=35
NODE_EPOCHS=100
NODE_PATIENCE=15
NODE_CONDITION_DROPOUT=0.1
EDGE_CONDITION_DROPOUT=0.1
TRAIN_EDGE=true
SAMPLE_MODE=joint
EVAL_MODE=joint
```

You can also override per run without editing the file:

```bash
CONDA_ENV_NAME=diffusion \
DEVICE=cuda \
NODE_EPOCHS=20 \
EDGE_EPOCHS=20 \
bash scripts/train_linker_diffusion.sh
```

On a multi-GPU server, pin one card explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda bash scripts/train_linker_diffusion.sh
```

## 3) Build Weak-Anchor Dataset

```bash
bash scripts/build_weak_anchor_data.sh
```

Outputs:

- weak-anchor csv: `outputs/weak_anchor_best/weak_anchor_dataset.csv`
- rejection log: `outputs/weak_anchor_best/weak_anchor_rejections.csv`
- summary: `outputs/weak_anchor_best/summary.json`
- tensor dataset: `data/processed/weak_anchor_tensors.pt`

## 4) Train Diffusion Models

```bash
bash scripts/train_linker_diffusion.sh
```

Defaults now follow the current project focus:

- `TRAIN_NODE=true`
- `TRAIN_EDGE=true`
- `NODE_PATIENCE=15`

Outputs:

- node checkpoint: `checkpoints/linker_node_diffusion.pt`
- edge checkpoint: `checkpoints/linker_edge_diffusion.pt`

## 5) Sample Linkers

```bash
bash scripts/sample_linker.sh
```

Default mode is `joint`.
If you want a more conservative node-only sampler that reuses the source linker topology:

```bash
SAMPLE_MODE=node_only bash scripts/sample_linker.sh
```

Outputs:

- samples csv: `outputs/linker_sampling/generated_samples.csv`
- samples json: `outputs/linker_sampling/generated_samples.json`
- summary json: `outputs/linker_sampling/summary.json`
- optional PNGs when `SAMPLE_SAVE_IMAGES=true`

## Token Version

This is the new downstream path where:

- left/right fragments are still graph-conditioned
- linker is no longer trained as atom `node + edge`
- linker is represented as an oriented token embedding sequence

Build token data and embeddings:

```bash
bash scripts/build_weak_anchor_token_data.sh
```

Train token diffusion:

```bash
bash scripts/train_linker_token_diffusion.sh
```

Sample token diffusion:

```bash
bash scripts/sample_linker_token.sh
```

Evaluate token diffusion:

```bash
bash scripts/evaluate_linker_token_generation.sh
```

Most relevant outputs:

- tokenized weak-anchor csv: `data/processed/weak_anchor_tokenized_oriented.csv`
- token embeddings: `data/processed/task/oriented_token_embedding/`
  - trained with `PAD=<PAD>` and `pad_to_length=23`
- token tensor dataset: `data/processed/weak_anchor_token_tensors.pt`
  - linker token sequences are padded to length `23`; overlength samples are rejected instead of truncated
- token checkpoint: `checkpoints/linker_token_diffusion.pt`
- token samples: `outputs/linker_token_sampling/`
- token eval: `outputs/linker_token_eval/`

## 6) Evaluate Generation

```bash
bash scripts/evaluate_linker_generation.sh
```

Outputs:

- `outputs/linker_eval/all_generations.csv`
- `outputs/linker_eval/per_source_summary.csv`
- `outputs/linker_eval/summary.json`
- `outputs/linker_eval/evaluation_overview.png`

Token version:

```bash
bash scripts/evaluate_linker_token_generation.sh
```

Outputs:

- `outputs/linker_token_eval/all_generations.csv`
- `outputs/linker_token_eval/per_source_summary.csv`
- `outputs/linker_token_eval/summary.json`
- `outputs/linker_token_eval/evaluation_overview.png`

## 7) Analyze Training Log

```bash
bash scripts/analyze_train_log.sh
```

Outputs:

- `outputs/train_log_analysis/loss_curves.png`
- `outputs/train_log_analysis/diagnostics.png`
- `outputs/train_log_analysis/summary.json`
- `outputs/train_log_analysis/REPORT.md`

## 8) End-to-End

```bash
bash scripts/run_server_pipeline.sh
```

Token pipeline:

```bash
bash scripts/run_token_server_pipeline.sh
```

For background execution:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
nohup bash scripts/run_server_pipeline.sh > run_pipeline.log 2>&1 &
```

Token pipeline in background:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
nohup bash scripts/run_token_server_pipeline.sh > run_token_pipeline.log 2>&1 &
```

Check progress:

```bash
tail -f run_pipeline.log
```

## Smoke Example

Use the current local env and smoke checkpoints:

```bash
CONDA_ENV_NAME=diffusion \
NODE_CKPT=/Users/lintianjian/diffusion/checkpoints/linker_node_diffusion_smoke.pt \
EDGE_CKPT=/Users/lintianjian/diffusion/checkpoints/linker_edge_diffusion_smoke.pt \
NUM_GENERATIONS=4 \
SAMPLE_OUT_DIR=/Users/lintianjian/diffusion/outputs/linker_sampling_smoke_script \
bash scripts/sample_linker.sh
```
