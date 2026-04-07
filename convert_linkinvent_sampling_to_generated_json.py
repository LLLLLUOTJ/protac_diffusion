from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Link-INVENT sampling CSV into a JSON format compatible with existing evaluation scripts."
    )
    parser.add_argument("--input-csv", type=str, required=True, help="Link-INVENT sampling CSV")
    parser.add_argument("--output-json", type=str, required=True, help="output JSON path")
    return parser.parse_args()


def _label_fragment(smiles: str, label: int) -> str:
    text = str(smiles or "").strip()
    if not text:
        raise ValueError("fragment smiles is empty")
    if "[*:" in text:
        return text
    replaced = text.replace("*", f"[*:{int(label)}]", 1)
    if replaced == text:
        raise ValueError(f"fragment is missing '*' attachment point: {text}")
    return replaced


def _label_linker(smiles: str) -> str:
    text = str(smiles or "").strip()
    if not text:
        raise ValueError("linker smiles is empty")
    if "[*:" in text:
        return text
    tokenized = re.sub(r"\[\*\]", "*", text)
    star_count = tokenized.count("*")
    if star_count != 2:
        raise ValueError(f"expected linker with exactly 2 '*' attachment points, found {star_count}: {text}")
    left, middle, right = tokenized.split("*")
    return f"{left}[*:1]{middle}[*:2]{right}"


def _normalize_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    warheads = str(row.get("Warheads", "")).strip()
    linker = str(row.get("Linker", "")).strip()
    full = str(row.get("SMILES", "")).strip()
    parts = warheads.split("|")
    if len(parts) != 2:
        raise ValueError(f"expected two warheads separated by '|', got: {warheads}")
    left = _label_fragment(parts[0], label=1)
    right = _label_fragment(parts[1], label=2)
    anchored_linker = _label_linker(linker)
    return {
        "sample_id": f"linkinvent_{row_index:04d}",
        "repeat_index": row_index - 1,
        "source_left_fragment_smiles": left,
        "source_right_fragment_smiles": right,
        "source_anchored_linker_smiles": None,
        "generated_anchored_linker_smiles": anchored_linker,
        "generated_full_smiles": full or None,
        "decode_reason": None,
        "assemble_reason": None,
        "nll": row.get("NLL"),
        "mode": "linkinvent",
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_json)

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            rows.append(_normalize_row(dict(row), row_index=idx))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] converted={len(rows)} out={output_path}")


if __name__ == "__main__":
    main()
