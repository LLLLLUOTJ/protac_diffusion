from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_group_lowfreq_tokens_by_core(tmp_path: Path) -> None:
    coverage_csv = tmp_path / "coverage.csv"
    core_table_csv = tmp_path / "core_table.csv"
    assignment_csv = tmp_path / "assign.csv"
    uncertain_csv = tmp_path / "uncertain.csv"
    grouped_json = tmp_path / "grouped.json"
    summary_json = tmp_path / "summary.json"

    rows = [
        {"rank": "1", "token_smiles": "*C*", "frequency": "100"},
        {"rank": "2", "token_smiles": "*O*", "frequency": "80"},
        {"rank": "3", "token_smiles": "*N*", "frequency": "70"},
        {"rank": "4", "token_smiles": "*C(*)=O", "frequency": "60"},
        {"rank": "5", "token_smiles": "*c1ccc(*)cc1", "frequency": "20"},
        {"rank": "6", "token_smiles": "*C(*)(C)C", "frequency": "2"},
        {"rank": "7", "token_smiles": "*N(*)CC", "frequency": "3"},
        {"rank": "8", "token_smiles": "*C(*)=S", "frequency": "2"},
        {"rank": "9", "token_smiles": "BADTOKEN", "frequency": "1"},
    ]
    with coverage_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "token_smiles", "frequency"])
        writer.writeheader()
        writer.writerows(rows)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "group_lowfreq_tokens_by_core.py"),
        "--coverage_csv",
        str(coverage_csv),
        "--core_freq_threshold",
        "10",
        "--out_core_table_csv",
        str(core_table_csv),
        "--out_assignment_csv",
        str(assignment_csv),
        "--out_uncertain_csv",
        str(uncertain_csv),
        "--out_grouped_json",
        str(grouped_json),
        "--out_summary_json",
        str(summary_json),
        "--verbose",
        "false",
    ]
    subprocess.run(cmd, check=True)

    core_rows = list(csv.DictReader(core_table_csv.open("r", encoding="utf-8", newline="")))
    assign_rows = list(csv.DictReader(assignment_csv.open("r", encoding="utf-8", newline="")))
    uncertain_rows = list(csv.DictReader(uncertain_csv.open("r", encoding="utf-8", newline="")))
    grouped = json.loads(grouped_json.read_text(encoding="utf-8"))
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    assert len(core_rows) == 5
    assert len(assign_rows) == 4
    assert "BADTOKEN" in {row["low_token"] for row in uncertain_rows}
    assert summary["num_core_tokens"] == 5
    assert summary["num_tail_tokens"] == 4
    assert set(summary["assignment_counts"].keys()) == {"high", "medium", "uncertain"}

    bad_row = next(row for row in assign_rows if row["low_token"] == "BADTOKEN")
    assert bad_row["confidence"] == "uncertain"
    assert bad_row["uncertain_reason"] == "LOW_TOKEN_PARSE_FAIL"

    sul_row = next(row for row in assign_rows if row["low_token"] == "*C(*)=S")
    assert sul_row["assigned_core_token"] in {"*C(*)=O", "*C*"}
    assert sul_row["confidence"] in {"high", "medium", "uncertain"}

    assert "*C*" in grouped
