from __future__ import annotations

import csv
import json
from pathlib import Path

from eval_molecules import (
    canonicalize_smiles,
    compute_novelty,
    compute_property_scores,
    compute_uniqueness,
    compute_validity,
    load_smiles_file,
)


def test_compute_validity_handles_invalid_smiles() -> None:
    result = compute_validity(["CCO", "not_a_smiles", "c1ccccc1"])
    assert result["total_count"] == 3
    assert result["valid_count"] == 2
    assert result["validity"] == 2.0 / 3.0
    assert len(result["valid_mols"]) == 2


def test_compute_uniqueness_uses_unique_canonical_smiles() -> None:
    ethanol_1 = canonicalize_smiles("CCO")
    ethanol_2 = canonicalize_smiles("OCC")
    benzene = canonicalize_smiles("c1ccccc1")
    assert ethanol_1 is not None
    assert ethanol_2 is not None
    assert benzene is not None
    result = compute_uniqueness([ethanol_1, ethanol_2, benzene])
    assert result["unique_valid_count"] == 2
    assert result["uniqueness"] == 2.0 / 3.0


def test_compute_novelty_excludes_training_overlap() -> None:
    unique_valid = [canonicalize_smiles("CCO"), canonicalize_smiles("CCN"), canonicalize_smiles("c1ccccc1")]
    unique_valid = [smiles for smiles in unique_valid if smiles is not None]
    result = compute_novelty(unique_valid, ["OCC", "CCCl", "invalid"])
    assert result["train_valid_unique_count"] == 2
    assert result["novel_unique_count"] == 2
    assert result["novelty"] == 2.0 / 3.0


def test_load_smiles_file_supports_json_objects_and_strings(tmp_path: Path) -> None:
    objects_path = tmp_path / "objects.json"
    objects_path.write_text(json.dumps([{"smiles": "CCO"}, {"smiles": "CCN"}]), encoding="utf-8")
    strings_path = tmp_path / "strings.json"
    strings_path.write_text(json.dumps(["CCO", "CCN"]), encoding="utf-8")

    assert load_smiles_file(objects_path, smiles_col="smiles") == ["CCO", "CCN"]
    assert load_smiles_file(strings_path, smiles_col="smiles") == ["CCO", "CCN"]


def test_load_smiles_file_supports_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "generated.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles"])
        writer.writeheader()
        writer.writerow({"smiles": "CCO"})
        writer.writerow({"smiles": "CCN"})
    assert load_smiles_file(csv_path, smiles_col="smiles") == ["CCO", "CCN"]


def test_compute_property_scores_returns_none_for_empty_input() -> None:
    result = compute_property_scores([])
    assert result["mean_qed"] is None
    assert result["mean_sa"] is None
