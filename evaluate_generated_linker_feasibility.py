from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sampling.linker_feasibility import FeasibilityThresholds, evaluate_record, write_feasibility_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate free_full or fixed feasibility directly from generated linker result files"
    )
    parser.add_argument("--input", type=str, required=True, help="generated_samples/all_generations csv or json/jsonl")
    parser.add_argument("--mode", type=str, default="free_full", choices=["free_full", "fixed"])
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    parser.add_argument("--skip-failed-decode", action="store_true")
    parser.add_argument("--num-conformers", type=int, default=100)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-top-k", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="outputs/generated_linker_feasibility")
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON payload must be a list")
        return [dict(item) for item in payload]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(dict(json.loads(line)))
    return rows


def _normalize_row(row: dict[str, Any], mode: str) -> dict[str, Any] | None:
    generated = str(row.get("generated_anchored_linker_smiles") or row.get("anchored_linker_smiles") or "").strip()
    left = str(row.get("source_left_fragment_smiles") or row.get("left_fragment_smiles") or "").strip()
    right = str(row.get("source_right_fragment_smiles") or row.get("right_fragment_smiles") or "").strip()
    if not generated or not left or not right:
        return None
    out = dict(row)
    out["mode"] = str(mode)
    base_sample_id = str(row.get("sample_id", "sample")).strip() or "sample"
    source_dataset_index = str(row.get("source_dataset_index", "")).strip()
    repeat_index = str(row.get("repeat_index", "")).strip()
    if source_dataset_index or repeat_index:
        suffix_parts = [part for part in [source_dataset_index, repeat_index] if part]
        out["sample_id"] = f"{base_sample_id}__{'_'.join(suffix_parts)}"
    else:
        out["sample_id"] = base_sample_id
    out["anchored_linker_smiles"] = generated
    out["left_fragment_smiles"] = left
    out["right_fragment_smiles"] = right
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    rows = _read_rows(input_path)
    if args.skip_failed_decode:
        rows = [row for row in rows if str(row.get("generated_anchored_linker_smiles") or "").strip()]

    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = _normalize_row(row, mode=args.mode)
        if item is not None:
            normalized.append(item)
    if args.max_rows > 0:
        normalized = normalized[: int(args.max_rows)]
    if not normalized:
        raise RuntimeError(f"no evaluable rows found in {input_path}")

    thresholds = FeasibilityThresholds()
    results = []
    for idx, row in enumerate(normalized, start=1):
        result = evaluate_record(
            record=row,
            num_conformers=args.num_conformers,
            max_iters=args.max_iters,
            random_seed=args.seed + idx - 1,
            thresholds=thresholds,
            save_sdf_dir=Path(args.out_dir) / "best_sdf",
            save_top_k=args.save_top_k,
        )
        results.append(result)
        if idx == 1 or idx % 10 == 0 or idx == len(normalized):
            print(
                f"[progress] {idx}/{len(normalized)} pass={sum(r.status == 'pass' for r in results)} "
                f"borderline={sum(r.status == 'borderline' for r in results)} "
                f"fail={sum(r.status == 'fail' for r in results)}",
                flush=True,
            )

    summary = write_feasibility_outputs(results, out_dir=args.out_dir)
    summary_with_meta = dict(summary)
    summary_with_meta.update(
        {
            "input": str(input_path),
            "mode": str(args.mode),
            "num_conformers": int(args.num_conformers),
            "max_iters": int(args.max_iters),
            "max_rows": int(args.max_rows),
            "skip_failed_decode": bool(args.skip_failed_decode),
        }
    )
    with (Path(args.out_dir) / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_with_meta, f, indent=2, ensure_ascii=False)
    print(
        f"[done] mode={args.mode} samples={summary_with_meta['num_samples']} "
        f"pass={summary_with_meta['num_pass']} borderline={summary_with_meta['num_borderline']} "
        f"fail={summary_with_meta['num_fail']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
