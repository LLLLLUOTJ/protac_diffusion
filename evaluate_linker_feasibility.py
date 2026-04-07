from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from sampling.linker_feasibility import (
    FeasibilityThresholds,
    evaluate_linker_feasibility,
    evaluate_full_protac_generation,
    evaluate_record,
    load_mol_from_path,
    load_optional_protein,
    write_feasibility_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether generated anchored linkers can satisfy fixed left/right 3D fragment geometry"
    )
    parser.add_argument("--input-json", type=str, default=None, help="JSON list or JSONL file of batch records")
    parser.add_argument("--mode", type=str, default="fixed", choices=["fixed", "free_full"])
    parser.add_argument("--left-fragment-path", type=str, default=None)
    parser.add_argument("--right-fragment-path", type=str, default=None)
    parser.add_argument("--left-fragment-smiles", type=str, default=None)
    parser.add_argument("--right-fragment-smiles", type=str, default=None)
    parser.add_argument("--anchored-linker-smiles", type=str, default=None)
    parser.add_argument("--sample-id", type=str, default="sample")
    parser.add_argument("--left-anchor-atom-idx", type=int, default=None)
    parser.add_argument("--right-anchor-atom-idx", type=int, default=None)
    parser.add_argument("--left-core-atom-indices", type=str, default=None, help="comma-separated indices")
    parser.add_argument("--right-core-atom-indices", type=str, default=None, help="comma-separated indices")
    parser.add_argument("--protein-path", type=str, default=None)
    parser.add_argument("--num-conformers", type=int, default=200)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-top-k", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="outputs/linker_feasibility")
    return parser.parse_args()


def _parse_int_list(text: str | None) -> list[int] | None:
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    return [int(part.strip()) for part in stripped.split(",") if part.strip()]


def _load_batch_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("input-json list payload must be a JSON array")
        return [dict(item) for item in payload]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(dict(json.loads(line)))
    return rows


def main() -> None:
    args = parse_args()
    thresholds = FeasibilityThresholds()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = out_dir / "best_sdf"

    if args.input_json:
        records = _load_batch_records(args.input_json)
        results = [
            evaluate_record(
                record=record,
                num_conformers=args.num_conformers,
                max_iters=args.max_iters,
                random_seed=args.seed,
                thresholds=thresholds,
                save_sdf_dir=sdf_dir,
                save_top_k=args.save_top_k,
            )
            for record in records
        ]
        summary = write_feasibility_outputs(results, out_dir=out_dir)
        print(
            f"[done] samples={summary['num_samples']} pass={summary['num_pass']} "
            f"borderline={summary['num_borderline']} fail={summary['num_fail']}",
            flush=True,
        )
        return

    if not args.anchored_linker_smiles:
        raise ValueError("single-sample mode requires --anchored-linker-smiles")

    if args.left_fragment_path:
        left = load_mol_from_path(args.left_fragment_path)
    else:
        left = Chem.MolFromSmiles(args.left_fragment_smiles) if args.left_fragment_smiles else None
    if args.right_fragment_path:
        right = load_mol_from_path(args.right_fragment_path)
    else:
        right = Chem.MolFromSmiles(args.right_fragment_smiles) if args.right_fragment_smiles else None
    if left is None or right is None:
        raise ValueError(
            "single-sample mode requires left/right fragments via --left-fragment-path/--right-fragment-path "
            "or --left-fragment-smiles/--right-fragment-smiles"
        )
    linker = Chem.MolFromSmiles(args.anchored_linker_smiles)
    if linker is None:
        raise ValueError(f"failed to parse anchored linker smiles: {args.anchored_linker_smiles}")
    protein = load_optional_protein(args.protein_path)

    evaluator = evaluate_linker_feasibility if args.mode == "fixed" else evaluate_full_protac_generation
    result = evaluator(
        left_fragment=left,
        anchored_linker=linker,
        right_fragment=right,
        sample_id=args.sample_id,
        num_conformers=args.num_conformers,
        max_iters=args.max_iters,
        random_seed=args.seed,
        thresholds=thresholds,
        left_anchor_atom_idx=args.left_anchor_atom_idx,
        right_anchor_atom_idx=args.right_anchor_atom_idx,
        left_core_atom_indices=_parse_int_list(args.left_core_atom_indices),
        right_core_atom_indices=_parse_int_list(args.right_core_atom_indices),
        protein=protein,
        save_sdf_dir=sdf_dir,
        save_top_k=args.save_top_k,
    )
    write_feasibility_outputs([result], out_dir=out_dir)
    print(
        f"[done] sample={result.sample_id} status={result.status} "
        f"scored={result.num_conformers_scored}/{result.num_conformers_requested}",
        flush=True,
    )


if __name__ == "__main__":
    main()
