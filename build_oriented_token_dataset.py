from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sampling.token_linker_codec import normalize_mapped_token_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert tokenized linker CSV into oriented-token training data")
    parser.add_argument(
        "--in_csv",
        type=str,
        default="data/processed/linker_anchor_tokenized_multi.csv",
        help="Input tokenized CSV containing token_smiles_with_maps_list_json",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/processed/linker_anchor_tokenized_oriented.csv",
        help="Output CSV with token_smiles_list_json replaced by oriented token templates",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="data/processed/linker_anchor_tokenized_oriented.summary.json",
        help="Conversion summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    summary_json = Path(args.summary_json)

    if not in_csv.exists():
        raise FileNotFoundError(f"input csv not found: {in_csv}")

    converted_rows: list[dict[str, str]] = []
    total = 0
    failed = 0
    for path in [out_csv, summary_json]:
        path.parent.mkdir(parents=True, exist_ok=True)

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"csv has no header: {in_csv}")
        fieldnames = list(reader.fieldnames)
        extra_fields = ["base_token_smiles_list_json", "oriented_token_smiles_list_json"]
        out_fields = fieldnames[:]
        for name in extra_fields:
            if name not in out_fields:
                out_fields.append(name)

        for row in reader:
            total += 1
            mapped_text = str(row.get("token_smiles_with_maps_list_json", "")).strip()
            if not mapped_text:
                failed += 1
                continue
            mapped_tokens = json.loads(mapped_text)
            oriented_tokens = normalize_mapped_token_sequence(mapped_tokens)

            new_row = {str(k): str(v) for k, v in row.items()}
            base_tokens = str(new_row.get("token_smiles_list_json", "")).strip()
            oriented_json = json.dumps(oriented_tokens, ensure_ascii=False)
            new_row["base_token_smiles_list_json"] = base_tokens
            new_row["oriented_token_smiles_list_json"] = oriented_json
            new_row["token_smiles_list_json"] = oriented_json
            converted_rows.append(new_row)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(converted_rows)

    summary = {
        "input_csv": str(in_csv),
        "output_csv": str(out_csv),
        "total_rows": int(total),
        "converted_rows": int(len(converted_rows)),
        "failed_rows": int(failed),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] total={total} converted={len(converted_rows)} failed={failed}", flush=True)
    print(f"[files] out_csv={out_csv} summary_json={summary_json}", flush=True)


if __name__ == "__main__":
    main()
