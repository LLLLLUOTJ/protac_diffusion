# Anchor Pipeline (Dummy Atom + Mask Tensors)

## 1) Build Anchored Tensor Dataset

```bash
python build_anchored_tensor_dataset.py \
  --csv data/csv/linker.csv \
  --out data/processed/anchored_linker_tensors.pt \
  --include-pair-mask
```

Output records include:

- `x`, `edge_index`, `edge_attr`, `node_type`
- `can_add_bond` (node-level valence mask)
- `degree`, `dummy_mask`, `anchor_mask`
- optional `pair_mask` (`--include-pair-mask`)

## 2) Train Anchor Model Directly From Tensor Dataset

```bash
python train_anchor.py \
  --tensor-pt data/processed/anchored_linker_tensors.pt \
  --epochs 20 \
  --out checkpoints/anchor_gnn.pt
```

## 3) Generate Anchored Linker From Plain Linker

```bash
python generate_anchored_linker.py \
  --ckpt checkpoints/anchor_gnn.pt \
  --smiles 'CCOCCOCCNC(=O)CN'
```

The output will be an anchored linker with `[*:1]` and `[*:2]`.
