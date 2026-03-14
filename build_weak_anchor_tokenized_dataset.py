from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sampling.token_linker_codec import tokenize_anchored_linker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize weak-anchor linker samples into oriented token sequences")
    parser.add_argument("--weak_anchor_csv", type=str, default="outputs/weak_anchor_best/weak_anchor_dataset.csv")
    parser.add_argument("--out_csv", type=str, default="data/processed/weak_anchor_tokenized_oriented.csv")
    parser.add_argument("--summary_json", type=str, default="data/processed/weak_anchor_tokenized_oriented.summary.json")
    parser.add_argument("--include_ring_single_bonds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_csv = Path(args.weak_anchor_csv)
    out_csv = Path(args.out_csv)
    summary_json = Path(args.summary_json)
    if not in_csv.exists():
        raise FileNotFoundError(f"weak-anchor csv not found: {in_csv}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, str]] = []
    total = 0
    failed = 0
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            anchored_linker_smiles = str(row.get("anchored_linker_smiles", "")).strip()
            if not anchored_linker_smiles:
                failed += 1
                continue
            try:
                tokenized = tokenize_anchored_linker(
                    anchored_linker_smiles,
                    include_ring_single_bonds=bool(args.include_ring_single_bonds),
                )
            except Exception:
                failed += 1
                continue

            rows_out.append(
                {
                    "sample_id": str(row.get("sample_id", "")).strip(),
                    "protac_id": str(row.get("protac_id", "")).strip(),
                    "linker_id": str(row.get("linker_id", "")).strip(),
                    "sample_weight": "1.0",
                    "anchored_linker_smiles": anchored_linker_smiles,
                    "left_fragment_smiles": str(row.get("left_fragment_smiles", "")).strip(),
                    "right_fragment_smiles": str(row.get("right_fragment_smiles", "")).strip(),
                    "token_smiles_list_json": json.dumps(tokenized["oriented_token_smiles"], ensure_ascii=False),
                    "base_token_smiles_list_json": json.dumps(tokenized["token_smiles"], ensure_ascii=False),
                    "token_smiles_with_maps_list_json": json.dumps(tokenized["token_smiles_with_maps"], ensure_ascii=False),
                    "anchor_path_atom_indices": "-".join(str(int(x)) for x in tokenized["anchor_path_atom_indices"]),
                    "anchor_path_single_bond_cut_indices": "-".join(
                        str(int(x)) for x in tokenized["anchor_path_single_bond_cut_indices"]
                    ),
                    "anchor_graph_distance": str(int(tokenized["anchor_graph_distance"])),
                    "num_cuts": str(int(tokenized["num_cuts"])),
                    "num_fragments": str(int(tokenized["num_fragments"])),
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "protac_id",
                "linker_id",
                "sample_weight",
                "anchored_linker_smiles",
                "left_fragment_smiles",
                "right_fragment_smiles",
                "token_smiles_list_json",
                "base_token_smiles_list_json",
                "token_smiles_with_maps_list_json",
                "anchor_path_atom_indices",
                "anchor_path_single_bond_cut_indices",
                "anchor_graph_distance",
                "num_cuts",
                "num_fragments",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "input_csv": str(in_csv),
        "output_csv": str(out_csv),
        "total_rows": int(total),
        "converted_rows": int(len(rows_out)),
        "failed_rows": int(failed),
        "include_ring_single_bonds": bool(args.include_ring_single_bonds),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] total={total} converted={len(rows_out)} failed={failed}", flush=True)
    print(f"[files] out_csv={out_csv} summary_json={summary_json}", flush=True)


if __name__ == "__main__":
    main()
