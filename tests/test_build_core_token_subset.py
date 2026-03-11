from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_build_core_token_subset_filters_low_freq_and_rebalances(tmp_path: Path) -> None:
    tokenized_csv = tmp_path / "tokenized.csv"
    instances_csv = tmp_path / "instances.csv"

    tokenized_rows = [
        {"sample_id": "A#1", "linker_id": "A", "num_anchor_pair_accepted": "2", "sample_weight": "0.5"},
        {"sample_id": "A#2", "linker_id": "A", "num_anchor_pair_accepted": "2", "sample_weight": "0.5"},
        {"sample_id": "B#1", "linker_id": "B", "num_anchor_pair_accepted": "1", "sample_weight": "1.0"},
    ]
    with tokenized_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tokenized_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tokenized_rows)

    instance_rows = [
        {"sample_id": "A#1", "linker_id": "A", "num_anchor_pair_accepted": "2", "sample_weight": "0.5", "token_smiles": "TOK_X"},
        {"sample_id": "A#1", "linker_id": "A", "num_anchor_pair_accepted": "2", "sample_weight": "0.5", "token_smiles": "TOK_X"},
        {"sample_id": "A#2", "linker_id": "A", "num_anchor_pair_accepted": "2", "sample_weight": "0.5", "token_smiles": "TOK_X"},
        {"sample_id": "A#2", "linker_id": "A", "num_anchor_pair_accepted": "2", "sample_weight": "0.5", "token_smiles": "TOK_TAIL"},
        {"sample_id": "B#1", "linker_id": "B", "num_anchor_pair_accepted": "1", "sample_weight": "1.0", "token_smiles": "TOK_X"},
        {"sample_id": "B#1", "linker_id": "B", "num_anchor_pair_accepted": "1", "sample_weight": "1.0", "token_smiles": "TOK_Y"},
        {"sample_id": "B#1", "linker_id": "B", "num_anchor_pair_accepted": "1", "sample_weight": "1.0", "token_smiles": "TOK_Y"},
    ]
    with instances_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(instance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(instance_rows)

    out_tokenized = tmp_path / "tokenized_core.csv"
    out_instances = tmp_path / "instances_core.csv"
    out_library = tmp_path / "library_core.csv"
    out_dropped = tmp_path / "dropped.csv"
    out_summary = tmp_path / "summary.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "build_core_token_subset.py"),
        "--tokenized_csv",
        str(tokenized_csv),
        "--instances_csv",
        str(instances_csv),
        "--out_tokenized_csv",
        str(out_tokenized),
        "--out_instances_csv",
        str(out_instances),
        "--out_library_csv",
        str(out_library),
        "--out_dropped_csv",
        str(out_dropped),
        "--out_summary_json",
        str(out_summary),
        "--min_token_freq",
        "2",
    ]
    subprocess.run(cmd, check=True)

    kept_tokenized = list(csv.DictReader(out_tokenized.open("r", encoding="utf-8", newline="")))
    kept_instances = list(csv.DictReader(out_instances.open("r", encoding="utf-8", newline="")))
    dropped = list(csv.DictReader(out_dropped.open("r", encoding="utf-8", newline="")))
    library = list(csv.DictReader(out_library.open("r", encoding="utf-8", newline="")))
    summary = json.loads(out_summary.read_text(encoding="utf-8"))

    kept_ids = {row["sample_id"] for row in kept_tokenized}
    assert kept_ids == {"A#1", "B#1"}
    assert len(dropped) == 1
    assert dropped[0]["sample_id"] == "A#2"
    assert dropped[0]["dropped_reason"] == "HAS_LOW_FREQ_TOKEN"

    # Rebalanced weights should be 1.0 for both kept linkers.
    for row in kept_tokenized:
        assert abs(float(row["sample_weight"]) - 1.0) < 1e-8
        assert row["sample_weight_before_core_filter"] in {"0.5", "1.0"}

    # No tail token in kept instances.
    assert all(row["token_smiles"] != "TOK_TAIL" for row in kept_instances)

    lib_tokens = {row["token_smiles"] for row in library}
    assert "TOK_TAIL" not in lib_tokens
    assert {"TOK_X", "TOK_Y"}.issubset(lib_tokens)

    assert summary["output"]["num_samples_kept"] == 2
    assert summary["output"]["num_samples_dropped"] == 1
