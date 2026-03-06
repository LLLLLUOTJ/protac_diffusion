# Anchor GNN Split Report

Checkpoint: `checkpoints/anchor_gnn.pt`
Source: `data/csv/linker.csv`
Seed: `42`
Validation ratio: `0.1`

## Split Sizes

- Train samples: `2452`
- Validation samples: `272`

## Validation Success

- Raw ordered exact (same as training log metric): `98/272` = `0.3603`
- Ordered exact success: `181/272` = `0.6654`
- Unordered exact success: `264/272` = `0.9706`

## Files

- `train_split.csv`: dataset index, compound ID, SMILES, target anchor labels
- `val_split.csv`: dataset index, compound ID, SMILES, target anchors, raw predictions, generation-time predictions, success flags
