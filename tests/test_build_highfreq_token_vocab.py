from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_build_highfreq_token_vocab(tmp_path: Path) -> None:
    instances_csv = tmp_path / "instances.csv"
    stats_csv = tmp_path / "stats.csv"
    vocab_txt = tmp_path / "vocab.txt"
    vocab_json = tmp_path / "vocab.json"
    summary_json = tmp_path / "summary.json"

    rows = [
        {"sample_id": "a#1", "linker_id": "a", "sample_weight": "1.0", "token_smiles": "*C*"},
        {"sample_id": "a#1", "linker_id": "a", "sample_weight": "1.0", "token_smiles": "*C*"},
        {"sample_id": "a#1", "linker_id": "a", "sample_weight": "1.0", "token_smiles": "*O*"},
        {"sample_id": "b#1", "linker_id": "b", "sample_weight": "1.0", "token_smiles": "*C*"},
        {"sample_id": "b#1", "linker_id": "b", "sample_weight": "1.0", "token_smiles": "*N*"},
    ]
    with instances_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "linker_id", "sample_weight", "token_smiles"])
        writer.writeheader()
        writer.writerows(rows)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "build_highfreq_token_vocab.py"),
        "--instances_csv",
        str(instances_csv),
        "--stats_csv",
        str(stats_csv),
        "--vocab_txt",
        str(vocab_txt),
        "--vocab_json",
        str(vocab_json),
        "--summary_json",
        str(summary_json),
        "--min_freq",
        "1",
        "--target_cumulative_coverage_pct",
        "80",
    ]
    subprocess.run(cmd, check=True)

    stats_rows = list(csv.DictReader(stats_csv.open("r", encoding="utf-8", newline="")))
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    vocab_data = json.loads(vocab_json.read_text(encoding="utf-8"))
    vocab_tokens = vocab_txt.read_text(encoding="utf-8").strip().splitlines()

    assert len(stats_rows) == 3
    assert stats_rows[0]["token_smiles"] == "*C*"
    assert int(stats_rows[0]["frequency"]) == 3
    assert summary["total_instances"] == 5
    assert summary["total_unique_tokens"] == 3
    assert summary["selected_occurrence_coverage_pct"] >= 80.0
    assert "*C*" in vocab_data["selected_tokens"]
    assert vocab_tokens[0] == "<PAD>"
    assert vocab_tokens[1] == "<UNK>"
    assert vocab_tokens[2] == "<MASK>"
