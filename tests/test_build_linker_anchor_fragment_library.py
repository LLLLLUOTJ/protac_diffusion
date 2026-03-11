from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

from build_linker_anchor_fragment_library import (
    add_anchor_endpoint_dummies,
    anchor_path_single_bonds,
    infer_anchor_pairs_from_smiles_r,
)


def test_infer_anchor_pairs_from_smiles_r_linear() -> None:
    mol = Chem.MolFromSmiles("CCOCCOC")
    assert mol is not None
    anchors, reason = infer_anchor_pairs_from_smiles_r(mol, "[R1]CCOCCOC[R2]")
    assert reason is None
    assert anchors is not None
    assert len(anchors) == 1
    a1, a2 = anchors[0]
    path, cuts = anchor_path_single_bonds(mol, a1, a2, include_ring_single_bonds=False)
    assert len(path) == 7
    assert len(cuts) == 6


def test_infer_anchor_pairs_requires_r_labels() -> None:
    mol = Chem.MolFromSmiles("CCOCC")
    assert mol is not None
    anchors, reason = infer_anchor_pairs_from_smiles_r(mol, "CCOCC")
    assert anchors is None
    assert reason == "MISSING_R_LABELS"


def test_add_anchor_endpoint_dummies_includes_both_labels() -> None:
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None
    anchored = add_anchor_endpoint_dummies(mol, 0, 1)
    smi = Chem.MolToSmiles(anchored, canonical=True)
    assert "[*:1]" in smi
    assert "[*:2]" in smi


def test_anchor_script_outputs_accept_and_reject(tmp_path: Path) -> None:
    in_csv = tmp_path / "linker.csv"
    tokenized_csv = tmp_path / "tokenized.csv"
    instances_csv = tmp_path / "instances.csv"
    library_csv = tmp_path / "library.csv"
    rej_csv = tmp_path / "rejections.csv"
    summary_json = tmp_path / "summary.json"

    rows = [
        {"Compound ID": "1", "Smiles": "CCOCCOC", "Smiles_R": "[R1]CCOCCOC[R2]"},
        {"Compound ID": "2", "Smiles": "CCOCC", "Smiles_R": "CCOCC"},
        {"Compound ID": "3", "Smiles": "CCCC", "Smiles_R": "[R1]CC[R2]"},
    ]
    with in_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles", "Smiles_R"])
        writer.writeheader()
        writer.writerows(rows)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "build_linker_anchor_fragment_library.py"),
        "--in_csv",
        str(in_csv),
        "--tokenized_csv",
        str(tokenized_csv),
        "--instances_csv",
        str(instances_csv),
        "--library_csv",
        str(library_csv),
        "--rej_csv",
        str(rej_csv),
        "--summary_json",
        str(summary_json),
    ]
    subprocess.run(cmd, check=True)

    with tokenized_csv.open("r", encoding="utf-8", newline="") as f:
        tokenized_rows = list(csv.DictReader(f))
    with instances_csv.open("r", encoding="utf-8", newline="") as f:
        instance_rows = list(csv.DictReader(f))
    with library_csv.open("r", encoding="utf-8", newline="") as f:
        library_rows = list(csv.DictReader(f))
    with rej_csv.open("r", encoding="utf-8", newline="") as f:
        rejected_rows = list(csv.DictReader(f))
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    row1 = [r for r in tokenized_rows if r["linker_id"] == "1"]
    assert len(row1) == 1
    assert int(row1[0]["num_cuts"]) == 6
    assert int(row1[0]["num_fragments"]) == 7
    assert "[*:1]" in row1[0]["anchored_linker_smiles"]
    assert "[*:2]" in row1[0]["anchored_linker_smiles"]
    vocab_tokens = json.loads(row1[0]["token_smiles_list_json"])
    mapped_tokens = json.loads(row1[0]["token_smiles_with_maps_list_json"])
    assert len(vocab_tokens) == len(mapped_tokens) == 7
    assert any("[*:" in tok for tok in mapped_tokens)

    row3 = [r for r in tokenized_rows if r["linker_id"] == "3"]
    assert len(row3) >= 2
    assert all(float(r["sample_weight"]) <= 1.0 for r in row3)
    assert abs(sum(float(r["sample_weight"]) for r in row3) - 1.0) < 1e-6

    assert len(instance_rows) >= 7
    assert len(library_rows) >= 1
    assert len(rejected_rows) == 1
    assert rejected_rows[0]["rejection_reason"] == "MISSING_R_LABELS"

    assert summary["accepted"] >= 3
    assert summary["rejected"] == 1
    assert summary["rows_with_multi_anchor_pairs"] >= 1
