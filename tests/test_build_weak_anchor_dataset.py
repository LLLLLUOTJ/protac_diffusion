from __future__ import annotations

import csv
import json
import sys

from rdkit import Chem

from build_weak_anchor_dataset import (
    MolRecord,
    extract_linker_left_right,
    fragment_with_dummies,
    get_crossing_bonds,
    get_unique_match,
    main,
    process_pair,
)


def make_record(smiles: str, row_id: str = "x") -> MolRecord:
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


def test_process_pair_accepts_unique_match_with_two_boundaries() -> None:
    protac = make_record("c1ccccc1CCOCCNc2ccccc2", "p1")
    linker = make_record("CCOCCN", "l1")

    accepted, rejection = process_pair(
        protac,
        linker,
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=1,
        min_anchor_graph_distance=1,
        min_linker_ratio_pct=0.0,
        max_linker_ratio_pct=100.0,
    )

    assert rejection is None
    assert accepted is not None
    assert "[*:1]" in accepted["anchored_linker_smiles"]
    assert "[*:2]" in accepted["anchored_linker_smiles"]
    assert accepted["anchor_left_atom_idx_in_full"] != accepted["anchor_right_atom_idx_in_full"]


def test_get_unique_match_no_match() -> None:
    protac = Chem.MolFromSmiles("CCOCC")
    linker = Chem.MolFromSmiles("NCC")
    assert protac is not None and linker is not None

    match, reason = get_unique_match(protac, linker)

    assert match is None
    assert reason == "NO_MATCH"


def test_get_unique_match_multi_match() -> None:
    protac = Chem.MolFromSmiles("CCOCCOCC")
    linker = Chem.MolFromSmiles("CO")
    assert protac is not None and linker is not None

    match, reason = get_unique_match(protac, linker)

    assert match is None
    assert reason == "MULTI_MATCH"


def test_process_pair_boundary_not_two() -> None:
    protac = make_record("CCOCCN", "p1")
    linker = make_record("CCN", "l1")

    accepted, rejection = process_pair(
        protac,
        linker,
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=1,
        min_anchor_graph_distance=1,
        min_linker_ratio_pct=0.0,
        max_linker_ratio_pct=100.0,
    )

    assert accepted is None
    assert rejection is not None
    assert rejection["rejection_reason"] == "BOUNDARY_NE_2"


def test_fragmentation_produces_mapped_dummy_linker() -> None:
    protac = Chem.MolFromSmiles("c1ccccc1CCOCCNc2ccccc2")
    linker = Chem.MolFromSmiles("CCOCCN")
    assert protac is not None and linker is not None

    match, reason = get_unique_match(protac, linker)
    assert match is not None, reason
    bond_ids, anchor_ids = get_crossing_bonds(protac, match)
    assert len(bond_ids) == 2
    assert len(set(anchor_ids)) == 2

    fragmented = fragment_with_dummies(protac, bond_ids)
    anchored_linker, left_fragment, right_fragment, frag_reason = extract_linker_left_right(fragmented)

    assert frag_reason is None
    assert anchored_linker is not None
    assert left_fragment is not None
    assert right_fragment is not None
    anchored_smiles = Chem.MolToSmiles(anchored_linker, canonical=True)
    assert "[*:1]" in anchored_smiles
    assert "[*:2]" in anchored_smiles


def test_process_pair_linker_too_small() -> None:
    protac = make_record("c1ccccc1CCOCCNc2ccccc2", "p1")
    linker = make_record("CCN", "l1")

    accepted, rejection = process_pair(
        protac,
        linker,
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=4,
        min_anchor_graph_distance=1,
        min_linker_ratio_pct=0.0,
        max_linker_ratio_pct=100.0,
    )

    assert accepted is None
    assert rejection is not None
    assert rejection["rejection_reason"] == "LINKER_TOO_SMALL"


def test_process_pair_anchor_distance_too_short() -> None:
    protac = make_record("c1ccccc1CCOCCNc2ccccc2", "p1")
    linker = make_record("CCOCCN", "l1")

    accepted, rejection = process_pair(
        protac,
        linker,
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=1,
        min_anchor_graph_distance=10,
        min_linker_ratio_pct=0.0,
        max_linker_ratio_pct=100.0,
    )

    assert accepted is None
    assert rejection is not None
    assert rejection["rejection_reason"] == "ANCHOR_DISTANCE_TOO_SHORT"


def test_process_pair_linker_ratio_out_of_range() -> None:
    protac = make_record("c1ccccc1CCOCCNc2ccccc2", "p1")
    linker = make_record("CCOCCN", "l1")

    accepted, rejection = process_pair(
        protac,
        linker,
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=1,
        min_anchor_graph_distance=1,
        min_linker_ratio_pct=40.0,
        max_linker_ratio_pct=100.0,
    )

    assert accepted is None
    assert rejection is not None
    assert rejection["rejection_reason"] == "LINKER_RATIO_OUT_OF_RANGE"


def test_main_writes_outputs_for_small_synthetic_dataset(tmp_path, monkeypatch) -> None:
    protac_csv = tmp_path / "protac.csv"
    linker_csv = tmp_path / "linker.csv"
    out_csv = tmp_path / "weak_anchor_dataset.csv"
    rej_csv = tmp_path / "weak_anchor_rejections.csv"
    summary_json = tmp_path / "summary.json"

    with protac_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles"])
        writer.writeheader()
        writer.writerows(
            [
                {"Compound ID": "p1", "Smiles": "c1ccccc1CCOCCNc2ccccc2"},
                {"Compound ID": "p2", "Smiles": "CCOCCN"},
            ]
        )

    with linker_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles"])
        writer.writeheader()
        writer.writerows(
            [
                {"Compound ID": "l1", "Smiles": "CCOCCN"},
                {"Compound ID": "l2", "Smiles": "N#N"},
            ]
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_weak_anchor_dataset.py",
            "--protac_csv",
            str(protac_csv),
            "--linker_csv",
            str(linker_csv),
            "--out_csv",
            str(out_csv),
            "--rej_csv",
            str(rej_csv),
            "--summary_json",
            str(summary_json),
            "--log_no_match_rejections",
            "true",
            "--min_fragment_heavy_atoms",
            "1",
            "--min_linker_heavy_atoms",
            "1",
            "--min_anchor_graph_distance",
            "1",
            "--min_linker_ratio_pct",
            "0",
            "--max_linker_ratio_pct",
            "100",
        ],
    )

    main()

    assert out_csv.exists()
    assert rej_csv.exists()
    assert summary_json.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert "[*:1]" in rows[0]["anchored_linker_smiles"]
    assert "[*:2]" in rows[0]["anchored_linker_smiles"]

    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["accepted"] == 1
    assert summary["total_pairs_tested"] >= 1


def test_main_keeps_only_best_linker_per_protac(tmp_path, monkeypatch) -> None:
    protac_csv = tmp_path / "protac.csv"
    linker_csv = tmp_path / "linker.csv"
    out_csv = tmp_path / "weak_anchor_dataset.csv"
    rej_csv = tmp_path / "weak_anchor_rejections.csv"
    summary_json = tmp_path / "summary.json"

    with protac_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles"])
        writer.writeheader()
        writer.writerow({"Compound ID": "p1", "Smiles": "c1ccccc1CCOCCNc2ccccc2"})

    with linker_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles"])
        writer.writeheader()
        writer.writerows(
            [
                {"Compound ID": "small", "Smiles": "CCN"},
                {"Compound ID": "big", "Smiles": "CCOCCN"},
            ]
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_weak_anchor_dataset.py",
            "--protac_csv",
            str(protac_csv),
            "--linker_csv",
            str(linker_csv),
            "--out_csv",
            str(out_csv),
            "--rej_csv",
            str(rej_csv),
            "--summary_json",
            str(summary_json),
            "--min_fragment_heavy_atoms",
            "1",
            "--min_linker_heavy_atoms",
            "1",
            "--min_anchor_graph_distance",
            "1",
            "--min_linker_ratio_pct",
            "0",
            "--max_linker_ratio_pct",
            "100",
        ],
    )

    main()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["linker_id"] == "big"


def test_main_dedupes_duplicate_protac_rows(tmp_path, monkeypatch) -> None:
    protac_csv = tmp_path / "protac.csv"
    linker_csv = tmp_path / "linker.csv"
    out_csv = tmp_path / "weak_anchor_dataset.csv"
    rej_csv = tmp_path / "weak_anchor_rejections.csv"
    summary_json = tmp_path / "summary.json"

    with protac_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles"])
        writer.writeheader()
        writer.writerows(
            [
                {"Compound ID": "p1", "Smiles": "c1ccccc1CCOCCNc2ccccc2"},
                {"Compound ID": "p1_dup", "Smiles": "c1ccccc1CCOCCNc2ccccc2"},
            ]
        )

    with linker_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Compound ID", "Smiles"])
        writer.writeheader()
        writer.writerow({"Compound ID": "l1", "Smiles": "CCOCCN"})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_weak_anchor_dataset.py",
            "--protac_csv",
            str(protac_csv),
            "--linker_csv",
            str(linker_csv),
            "--out_csv",
            str(out_csv),
            "--rej_csv",
            str(rej_csv),
            "--summary_json",
            str(summary_json),
            "--min_fragment_heavy_atoms",
            "1",
            "--min_linker_heavy_atoms",
            "1",
            "--min_anchor_graph_distance",
            "1",
            "--min_linker_ratio_pct",
            "0",
            "--max_linker_ratio_pct",
            "100",
            "--dedupe_protacs",
            "true",
        ],
    )

    main()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1

    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["input_protac_rows"] == 2
    assert summary["protacs_after_dedupe"] == 1
