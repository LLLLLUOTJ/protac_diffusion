from __future__ import annotations

import json
from pathlib import Path

from archive.delinker_baseline_2026_04_08.convert_delinker_sampling_to_generated_json import (
    _extract_anchored_linker,
    main,
)


def test_extract_anchored_linker_from_generated_full() -> None:
    left = "c1ccccc1[*:1]"
    right = "N#C[*:2]"
    full = "N#CCOc1ccccc1"
    anchored, reason = _extract_anchored_linker(full, left, right)
    assert reason is None
    assert anchored is not None
    assert "[*:1]" in anchored
    assert "[*:2]" in anchored


def test_convert_delinker_output_to_generated_json(tmp_path: Path, monkeypatch) -> None:
    source_rows = [
        {
            "sample_id": "token_1",
            "source_dataset_index": "7",
            "source_left_fragment_smiles": "c1ccccc1[*:1]",
            "source_right_fragment_smiles": "N#C[*:2]",
            "source_anchored_linker_smiles": "[*:1]CC[*:2]",
        }
    ]
    source_json = tmp_path / "sources.json"
    source_json.write_text(json.dumps(source_rows), encoding="utf-8")

    pairs_smi = tmp_path / "pairs.smi"
    pairs_smi.write_text("c1ccccc1[*:1].N#C[*:2] N#CCCCc1ccccc1\n", encoding="utf-8")

    sampling_smi = tmp_path / "delinker.smi"
    sampling_smi.write_text(
        "c1ccccc1[*:1].N#C[*:2] N#CCCCc1ccccc1 N#CCOc1ccccc1\n",
        encoding="utf-8",
    )

    output_json = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert_delinker_sampling_to_generated_json.py",
            "--input-smi",
            str(sampling_smi),
            "--pairs-smi",
            str(pairs_smi),
            "--source-json",
            str(source_json),
            "--output-json",
            str(output_json),
        ],
    )
    main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(payload) == 1
    row = payload[0]
    assert row["source_dataset_index"] == "7"
    assert row["generated_full_smiles"] == "N#CCOc1ccccc1"
    assert row["generated_anchored_linker_smiles"] is not None
    assert row["decode_reason"] is None
