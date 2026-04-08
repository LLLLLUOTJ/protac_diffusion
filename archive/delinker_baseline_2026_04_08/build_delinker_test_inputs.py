from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem

from sampling.linker_generation import assemble_full_molecule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DeLinker test-mode inputs from token generation source records."
    )
    parser.add_argument("--input-json", type=str, required=True, help="all_generations.json with source_* fields")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max-sources", type=int, default=0, help="Optional source limit for smoke tests")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--minimize", action="store_true", help="Run MMFF/UFF after embedding")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("input JSON must be a list of objects")
    return [dict(row) for row in payload]


def _unique_sources(rows: list[dict[str, Any]], max_sources: int = 0) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        source_idx = str(row.get("source_dataset_index", ""))
        left = str(row.get("source_left_fragment_smiles", "")).strip()
        right = str(row.get("source_right_fragment_smiles", "")).strip()
        linker = str(row.get("source_anchored_linker_smiles", "")).strip()
        if not left or not right or not linker:
            continue
        key = (source_idx, left, right, linker)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "source_dataset_index": source_idx,
                "sample_id": str(row.get("sample_id", "")),
                "left_fragment_smiles": left,
                "right_fragment_smiles": right,
                "anchored_linker_smiles": linker,
            }
        )
        if max_sources > 0 and len(unique) >= max_sources:
            break
    return unique


def _embed_molecule(mol: Chem.Mol, seed: int, minimize: bool = False) -> Chem.Mol | None:
    work = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(work, params)
    if status != 0:
        return None
    if minimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(work):
                AllChem.MMFFOptimizeMolecule(work, maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(work, maxIters=200)
        except Exception:
            pass
    return Chem.RemoveHs(work)


def _canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_path)
    sources = _unique_sources(rows, max_sources=int(args.max_sources))

    pairs_path = out_dir / "pairs.smi"
    sdf_path = out_dir / "reference_full.sdf"
    summary_path = out_dir / "build_summary.json"

    writer = Chem.SDWriter(str(sdf_path))
    written = 0
    failures: list[dict[str, str]] = []

    with pairs_path.open("w", encoding="utf-8") as pairs_file:
        for idx, row in enumerate(sources):
            left = Chem.MolFromSmiles(row["left_fragment_smiles"])
            right = Chem.MolFromSmiles(row["right_fragment_smiles"])
            linker = Chem.MolFromSmiles(row["anchored_linker_smiles"])
            if left is None or right is None or linker is None:
                failures.append({"sample_id": row["sample_id"], "reason": "parse_failed"})
                continue

            assembled, reason = assemble_full_molecule(left, linker, right)
            if assembled is None:
                failures.append({"sample_id": row["sample_id"], "reason": f"assemble_failed: {reason}"})
                continue

            embedded = _embed_molecule(assembled, seed=int(args.seed) + idx, minimize=bool(args.minimize))
            if embedded is None:
                failures.append({"sample_id": row["sample_id"], "reason": "embed_failed"})
                continue

            full_smiles = _canonical_smiles(embedded)
            fragments_smiles = f"{row['left_fragment_smiles']}.{row['right_fragment_smiles']}"
            embedded.SetProp("_Name", row["sample_id"])
            embedded.SetProp("sample_id", row["sample_id"])
            embedded.SetProp("source_dataset_index", row["source_dataset_index"])
            embedded.SetProp("fragments_smiles", fragments_smiles)
            embedded.SetProp("full_smiles", full_smiles)
            writer.write(embedded)
            pairs_file.write(f"{fragments_smiles} {full_smiles}\n")
            written += 1

    writer.close()

    summary = {
        "input_json": str(input_path),
        "num_rows": len(rows),
        "num_unique_sources": len(sources),
        "num_written": written,
        "num_failures": len(failures),
        "pairs_path": str(pairs_path),
        "sdf_path": str(sdf_path),
        "failures": failures[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] unique_sources={len(sources)} written={written} failures={len(failures)}")
    print(f"[done] pairs={pairs_path}")
    print(f"[done] sdf={sdf_path}")


if __name__ == "__main__":
    main()
