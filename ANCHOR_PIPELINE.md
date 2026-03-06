# Anchor Pipeline (Dummy Atom + Mask Tensors)

## 1) Build Anchored Tensor Dataset

```bash
python build_anchored_tensor_dataset.py \
  --csv data/csv/linker.csv \
  --out data/processed/anchored_linker_tensors.pt \
  --include-pair-mask
```

Example output:

```text
[done] samples=2746 reasons={'missing_r1_r2': 3} include_pair_mask=True out=data/processed/anchored_linker_tensors.pt
```

Output records include:

- `x`, `edge_index`, `edge_attr`, `node_type`
- `can_add_bond` (node-level valence mask)
- `degree`, `dummy_mask`, `anchor_mask`
- optional `pair_mask` (`--include-pair-mask`)

Example first record:

```text
meta_num_samples 2746
meta_reason_counts {'missing_r1_r2': 3}
first_x (9, 4) torch.float32
first_edge_index (2, 16) torch.int64
first_edge_attr (16, 4) torch.float32
first_node_type (9,) torch.int64
first_can_add_bond (9,) torch.bool
first_pair_mask (9, 9) torch.bool
first_anchored_smiles C(COC[*:2])OCC[*:1]
```

## 2) Train Anchor Model Directly From Tensor Dataset

```bash
python train_anchor.py \
  --tensor-pt data/processed/anchored_linker_tensors.pt \
  --epochs 20 \
  --out checkpoints/anchor_gnn.pt
```

Example smoke-run output:

```text
[data] source=tensor_pt=data/processed/anchored_linker_tensors.pt total=2746 train=2472 val=274 reasons={'missing_r1_r2': 3}
[train] device=cpu class_weights=[0.0960945338010788, 1.4519526958465576, 1.4519526958465576]
[epoch 001] train_loss=0.5471 train_node_acc=0.9199 train_anchor_exact=0.1744 val_loss=0.4604 val_node_acc=0.9461 val_anchor_exact=0.4197
[epoch 002] train_loss=0.4579 train_node_acc=0.9475 train_anchor_exact=0.2213 val_loss=0.4532 val_node_acc=0.9500 val_anchor_exact=0.4453
```

## 3) Generate Anchored Linker From Plain Linker

```bash
python generate_anchored_linker.py \
  --ckpt checkpoints/anchor_gnn.pt \
  --smiles 'CCOCCOCCNC(=O)CN'
```

The output will be an anchored linker with `[*:1]` and `[*:2]`.

Example output:

```text
[input] CCOCCOCCNC(=O)CN
[pred_anchor_idx] left=12 right=0
[output] O=C(CN[*:1])NCCOCCOCC[*:2]
```

## 4) Use Masks During Diffusion Sampling

`diffusion/DDPM.sample()` now supports generic tensor shapes and mask-based constraints.

Typical graph usage:

```python
record = torch.load("data/processed/anchored_linker_tensors.pt")["records"][0]

# Example: edge tensor shape [B, N, N, bond_dim]
pair_mask = record["pair_mask"].unsqueeze(0).unsqueeze(-1)   # valid bond slots
fixed_mask = record["dummy_mask"].unsqueeze(0).unsqueeze(-1) # optionally freeze dummy nodes/features

x = ddpm.sample(
    shape=(1, record["x"].shape[0], record["x"].shape[0], 4),
    sample_mask=pair_mask,
    fixed_mask=fixed_mask,
    fixed_values=0.0,
    show_progress=False,
)
```

Mask semantics:

- `sample_mask`: multiplicative validity mask, invalid positions are forced to zero every step.
- `fixed_mask`: immutable positions, values are overwritten with `fixed_values` every step.
- `post_step_fn`: optional callback for step-wise projection after each reverse step.

Example output from a masked DDPM smoke test:

```text
sample_shape (1, 3, 3, 2)
masked_01 [0.0, 0.0]
masked_10 [0.0, -0.0]
valid_00_nonzero True
```

This confirms invalid pair slots stay zero while valid slots continue to evolve during sampling.
