from __future__ import annotations

import json
from pathlib import Path

from evaluate_generated_linker_feasibility import _normalize_row, _read_rows


def test_normalize_row_uses_generated_and_source_fragment_fields() -> None:
    row = {
        "sample_id": "s1",
        "source_dataset_index": 12,
        "repeat_index": 3,
        "generated_anchored_linker_smiles": "[*:1]CC[*:2]",
        "source_left_fragment_smiles": "C[*:1]",
        "source_right_fragment_smiles": "N[*:2]",
    }
    out = _normalize_row(row, mode="free_full")
    assert out is not None
    assert out["mode"] == "free_full"
    assert out["sample_id"] == "s1__12_3"
    assert out["anchored_linker_smiles"] == "[*:1]CC[*:2]"
    assert out["left_fragment_smiles"] == "C[*:1]"
    assert out["right_fragment_smiles"] == "N[*:2]"


def test_read_rows_supports_json_array(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    payload = [{"sample_id": "a"}, {"sample_id": "b"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = _read_rows(path)
    assert [row["sample_id"] for row in rows] == ["a", "b"]
