from __future__ import annotations

import csv

from data.anchored_tensor_dataset import (
    AnchoredTensorDataset,
    AnchoredTensorPTDataset,
    WeakAnchorTensorDataset,
    WeakAnchorTensorPTDataset,
    attach_anchor_dummies,
    collate_graph_tensor_blocks,
    collate_anchor_train_samples,
    collate_weak_anchor_tensor_samples,
    degree_from_bidir_edge_index,
    pair_add_mask,
    serialize_anchored_tensor_dataset,
    serialize_weak_anchor_tensor_dataset,
)
from build_weak_anchor_dataset import MolRecord, process_pair
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


def _make_record(smiles: str, row_id: str) -> MolRecord:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return MolRecord(
        row_id=row_id,
        smiles=smiles,
        mol=mol,
        canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
        num_atoms=mol.GetNumAtoms(),
        source_row={},
    )


def _make_weak_anchor_rows() -> list[dict[str, str]]:
    full_1 = "c1ccccc1CCOCCNc2ccccc2"
    linker_1 = "CCOCCN"
    full_2 = "c1ccccc1CCOCCOCCNc2ccccc2"
    linker_2 = "CCOCCOCCN"

    accepted_1, rejection_1 = process_pair(
        _make_record(full_1, "p1"),
        _make_record(linker_1, "l1"),
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=1,
        min_anchor_graph_distance=1,
        min_linker_ratio_pct=0.0,
        max_linker_ratio_pct=100.0,
    )
    accepted_2, rejection_2 = process_pair(
        _make_record(full_2, "p2"),
        _make_record(linker_2, "l2"),
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=1,
        min_anchor_graph_distance=1,
        min_linker_ratio_pct=0.0,
        max_linker_ratio_pct=100.0,
    )
    assert accepted_1 is not None, rejection_1
    assert accepted_2 is not None, rejection_2

    accepted_1["sample_id"] = "1"
    accepted_2["sample_id"] = "2"
    return [
        {k: str(v) for k, v in accepted_1.items() if not k.startswith("_")},
        {k: str(v) for k, v in accepted_2.items() if not k.startswith("_")},
    ]


def test_weak_anchor_tensor_dataset_build(tmp_path) -> None:
    csv_path = tmp_path / "weak_anchor.csv"
    rows = _make_weak_anchor_rows()
    fieldnames = [
        "sample_id",
        "protac_id",
        "linker_id",
        "full_protac_smiles",
        "linker_smiles",
        "anchored_linker_smiles",
        "left_fragment_smiles",
        "right_fragment_smiles",
        "anchor_left_atom_idx_in_full",
        "anchor_right_atom_idx_in_full",
        "num_atoms_full",
        "num_atoms_linker",
        "num_atoms_left",
        "num_atoms_right",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ds = WeakAnchorTensorDataset(str(csv_path), include_pair_mask=True)
    assert len(ds) == 2

    sample = ds[0]
    assert sample.linker_graph.x.ndim == 2
    assert sample.linker_graph.edge_index.shape[0] == 2
    assert sample.linker_graph.pair_mask is not None
    assert int(sample.linker_graph.dummy_mask.sum().item()) == 2
    assert int(sample.left_graph.dummy_mask.sum().item()) == 1
    assert int(sample.right_graph.dummy_mask.sum().item()) == 1
    assert 0.0 < sample.linker_ratio_pct < 100.0


def test_weak_anchor_pt_dataset_and_collate(tmp_path) -> None:
    csv_path = tmp_path / "weak_anchor.csv"
    rows = _make_weak_anchor_rows()
    fieldnames = [
        "sample_id",
        "protac_id",
        "linker_id",
        "full_protac_smiles",
        "linker_smiles",
        "anchored_linker_smiles",
        "left_fragment_smiles",
        "right_fragment_smiles",
        "anchor_left_atom_idx_in_full",
        "anchor_right_atom_idx_in_full",
        "num_atoms_full",
        "num_atoms_linker",
        "num_atoms_left",
        "num_atoms_right",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ds = WeakAnchorTensorDataset(str(csv_path), include_pair_mask=True)
    out_pt = tmp_path / "weak_anchor.pt"
    serialize_weak_anchor_tensor_dataset(ds, str(out_pt), include_pair_mask=True)

    pt_ds = WeakAnchorTensorPTDataset(str(out_pt))
    assert len(pt_ds) == 2
    batch = collate_weak_anchor_tensor_samples([pt_ds[0], pt_ds[1]])
    assert batch["linker_graph"]["x"].ndim == 2
    assert batch["linker_graph"]["edge_index"].shape[0] == 2
    assert batch["left_graph"]["x"].ndim == 2
    assert batch["right_graph"]["x"].ndim == 2
    assert batch["linker_graph"]["pair_mask"].ndim == 2
    assert len(batch["sample_id"]) == 2
    assert batch["linker_ratio_pct"].shape[0] == 2

    linker_only = collate_graph_tensor_blocks([pt_ds[0]["linker_graph"], pt_ds[1]["linker_graph"]])
    assert linker_only["x"].ndim == 2
    assert linker_only["graph_ptr"].shape[0] == 3
