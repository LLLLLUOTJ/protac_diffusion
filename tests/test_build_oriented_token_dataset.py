from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_build_oriented_token_dataset(tmp_path: Path) -> None:
    in_csv = tmp_path / "tokenized.csv"
    out_csv = tmp_path / "oriented.csv"
    summary_json = tmp_path / "summary.json"

    rows = [
        {
            "sample_id": "56#1",
            "linker_id": "56",
            "token_smiles_list_json": json.dumps(["*N*", "*C*", "*C*", "*c1cn(*)nn1", "*C*", "*O*"]),
            "token_smiles_with_maps_list_json": json.dumps(
                [
                    "N([*:1])[*:3]",
                    "C([*:3])[*:4]",
                    "C([*:4])[*:5]",
                    "c1c([*:6])nnn1[*:5]",
                    "C([*:6])[*:7]",
                    "O([*:2])[*:7]",
                ]
            ),
        }
    ]
    with in_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "build_oriented_token_dataset.py"),
        "--in_csv",
        str(in_csv),
        "--out_csv",
        str(out_csv),
        "--summary_json",
        str(summary_json),
    ]
    subprocess.run(cmd, check=True)

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        out_rows = list(csv.DictReader(f))
    assert len(out_rows) == 1

    oriented = json.loads(out_rows[0]["token_smiles_list_json"])
    assert oriented == json.loads(out_rows[0]["oriented_token_smiles_list_json"])
    assert json.loads(out_rows[0]["base_token_smiles_list_json"]) == ["*N*", "*C*", "*C*", "*c1cn(*)nn1", "*C*", "*O*"]
    assert oriented[0] == "N([*:1])[*:2]"
    assert oriented[3] == "c1c([*:2])nnn1[*:1]"

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["total_rows"] == 1
    assert summary["converted_rows"] == 1
    assert summary["failed_rows"] == 0
