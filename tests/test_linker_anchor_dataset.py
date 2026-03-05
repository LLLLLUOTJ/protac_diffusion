from __future__ import annotations

import csv

from rdkit import Chem

from data.linker_anchor_dataset import (
    LinkerAnchorDataset,
    collate_linker_anchor_samples,
    extract_anchor_indices_from_smiles_pair,
    normalize_smiles_r,
)


def test_normalize_smiles_r() -> None:
    raw = "[R1]CCOCC[R2]"
    assert normalize_smiles_r(raw) == "[*:1]CCOCC[*:2]"


def test_extract_anchor_indices_from_smiles_pair() -> None:
    smiles = "CCOCCOC"
    smiles_r = "[R1]CCOCCOC[R2]"
    anchors, reason = extract_anchor_indices_from_smiles_pair(smiles, smiles_r)
    assert anchors is not None, reason
    left, right = anchors
    assert left != right


def test_dataset_and_collate(tmp_path) -> None:
    csv_path = tmp_path / "linker_tiny.csv"
    rows = [
        {"Compound ID": "1", "Smiles": "CCOCCOC", "Smiles_R": "[R1]CCOCCOC[R2]"},
        {"Compound ID": "2", "Smiles": "CCOCC", "Smiles_R": "[R1]CCOCC[R2]"},
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles", "Smiles_R"])
        writer.writeheader()
        writer.writerows(rows)

    dataset = LinkerAnchorDataset(str(csv_path))
    assert len(dataset) == 2

    batch = collate_linker_anchor_samples([dataset[0], dataset[1]])
    assert batch["x"].ndim == 2
    assert batch["edge_index"].shape[0] == 2
    assert batch["y"].ndim == 1
    assert batch["graph_ptr"].shape[0] == 3
    assert int((batch["y"] == 1).sum().item()) == 2
    assert int((batch["y"] == 2).sum().item()) == 2

    # Ensure y is a valid 3-class target.
    assert set(batch["y"].tolist()).issubset({0, 1, 2})

    # Sanity: first sample still decodable by RDKit.
    mol = Chem.MolFromSmiles(dataset[0].smiles)
    assert mol is not None
