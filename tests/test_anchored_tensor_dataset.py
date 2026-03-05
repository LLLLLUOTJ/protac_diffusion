from __future__ import annotations

import csv

from data.anchored_tensor_dataset import (
    AnchoredTensorDataset,
    AnchoredTensorPTDataset,
    attach_anchor_dummies,
    collate_anchor_train_samples,
    degree_from_bidir_edge_index,
    pair_add_mask,
    serialize_anchored_tensor_dataset,
)
from rdkit import Chem
import torch


def test_attach_anchor_dummies() -> None:
    mol = Chem.MolFromSmiles("CCOCC")
    assert mol is not None
    out = attach_anchor_dummies(mol, 0, 4)
    assert out is not None
    dummies = [a for a in out.GetAtoms() if a.GetAtomicNum() == 0]
    assert len(dummies) == 2
    maps = sorted(a.GetAtomMapNum() for a in dummies)
    assert maps == [1, 2]


def test_degree_and_pair_mask() -> None:
    mol = Chem.MolFromSmiles("[*:1]CCOCC[*:2]")
    assert mol is not None
    from molgraph import encode_mol

    g = encode_mol(mol)
    deg = degree_from_bidir_edge_index(g["edge_index"], g["x"].shape[0])
    # internal atoms are degree 2, terminal dummies degree 1 in this linker.
    assert int(deg.max().item()) >= 2

    can_add = deg < 4
    pm = pair_add_mask(g["edge_index"], can_add)
    n = g["x"].shape[0]
    assert pm.shape == (n, n)
    # diagonal should be disabled by default
    assert not bool(pm.diag().any().item())


def test_anchored_tensor_dataset_build(tmp_path) -> None:
    csv_path = tmp_path / "tiny_linker.csv"
    rows = [
        {"Compound ID": "1", "Smiles": "CCOCCOC", "Smiles_R": "[R1]CCOCCOC[R2]"},
        {"Compound ID": "2", "Smiles": "CCOCC", "Smiles_R": "[R1]CCOCC[R2]"},
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles", "Smiles_R"])
        writer.writeheader()
        writer.writerows(rows)

    ds = AnchoredTensorDataset(str(csv_path), include_pair_mask=True)
    assert len(ds) == 2

    sample = ds[0]
    assert sample.x.ndim == 2
    assert sample.edge_index.shape[0] == 2
    assert sample.edge_attr.ndim == 2
    assert sample.node_type.ndim == 1
    assert int((sample.dummy_mask).sum().item()) == 2
    assert int((sample.anchor_mask).sum().item()) == 2
    assert sample.can_add_bond.dtype == torch.bool
    assert sample.pair_mask is not None


def test_pt_dataset_and_train_collate(tmp_path) -> None:
    csv_path = tmp_path / "tiny_linker.csv"
    rows = [
        {"Compound ID": "1", "Smiles": "CCOCCOC", "Smiles_R": "[R1]CCOCCOC[R2]"},
        {"Compound ID": "2", "Smiles": "CCOCC", "Smiles_R": "[R1]CCOCC[R2]"},
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles", "Smiles_R"])
        writer.writeheader()
        writer.writerows(rows)

    ds = AnchoredTensorDataset(str(csv_path), include_pair_mask=False)
    out_pt = tmp_path / "anchored.pt"
    serialize_anchored_tensor_dataset(ds, str(out_pt), include_pair_mask=False)

    pt_ds = AnchoredTensorPTDataset(str(out_pt))
    assert len(pt_ds) == 2
    batch = collate_anchor_train_samples([pt_ds[0], pt_ds[1]])
    assert batch["x"].ndim == 2
    assert batch["edge_index"].shape[0] == 2
    assert batch["y"].ndim == 1
