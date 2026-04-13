from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from build_weak_anchor_dataset import extract_linker_left_right, fragment_with_dummies
from prepare_difflinker_inputs import _choose_non_overlapping_matches, _strip_single_dummy_and_get_anchor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DiffLinker per-sample SDF outputs into evaluation JSON/CSV")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--generated-subdir", type=str, default="generated")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    return parser.parse_args()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list")
    return [dict(row) for row in payload]


def _find_crossing_bonds_by_fragment_side(
    mol: Chem.Mol,
    left_match: tuple[int, ...],
    right_match: tuple[int, ...],
) -> tuple[int, int] | tuple[None, None]:
    left_set = set(int(x) for x in left_match)
    right_set = set(int(x) for x in right_match)
    fragment_set = left_set | right_set
    left_crossings: list[int] = []
    right_crossings: list[int] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_in = begin in fragment_set
        end_in = end in fragment_set
        if begin_in == end_in:
            continue
        inside = begin if begin_in else end
        if inside in left_set:
            left_crossings.append(bond.GetIdx())
        elif inside in right_set:
            right_crossings.append(bond.GetIdx())
    if len(left_crossings) != 1 or len(right_crossings) != 1:
        return None, None
    return int(left_crossings[0]), int(right_crossings[0])


def extract_anchored_linker_from_generated(
    full_mol: Chem.Mol,
    left_fragment_smiles: str,
    right_fragment_smiles: str,
) -> tuple[str | None, str | None]:
    try:
        left_core, _ = _strip_single_dummy_and_get_anchor(left_fragment_smiles)
        right_core, _ = _strip_single_dummy_and_get_anchor(right_fragment_smiles)
        left_match, right_match = _choose_non_overlapping_matches(full_mol, left_core, right_core)
    except Exception as exc:
        return None, f"fragment_match_failed: {exc}"

    left_bond, right_bond = _find_crossing_bonds_by_fragment_side(full_mol, left_match, right_match)
    if left_bond is None or right_bond is None:
        return None, "boundary_not_two"

    try:
        fragmented = fragment_with_dummies(full_mol, [left_bond, right_bond])
        anchored_linker, _, _, reason = extract_linker_left_right(fragmented, min_fragment_heavy_atoms=0)
        if anchored_linker is None:
            return None, reason
        return Chem.MolToSmiles(anchored_linker, canonical=True), None
    except Exception as exc:
        return None, f"fragment_extract_failed: {exc}"


def convert_rows(manifest_rows: list[dict[str, Any]], generated_subdir: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in manifest_rows:
        sample_id = str(row["sample_id"])
        sample_dir = Path(row["fragments_path"]).parent
        generated_dir = sample_dir / generated_subdir
        sdf_paths = sorted(generated_dir.glob("*.sdf"))
        for repeat_index, sdf_path in enumerate(sdf_paths):
            mol = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)[0]
            decode_reason = None
            generated_full_smiles = None
            generated_anchored_linker_smiles = None
            if mol is None:
                decode_reason = "generated_sdf_parse_failed"
            else:
                try:
                    generated_full_smiles = Chem.MolToSmiles(mol, canonical=True)
                except Exception as exc:
                    decode_reason = f"generated_smiles_failed: {exc}"
                if decode_reason is None:
                    generated_anchored_linker_smiles, decode_reason = extract_anchored_linker_from_generated(
                        mol,
                        left_fragment_smiles=str(row["left_fragment_smiles"]),
                        right_fragment_smiles=str(row["right_fragment_smiles"]),
                    )
            out.append(
                {
                    "repeat_index": int(repeat_index),
                    "sample_id": sample_id,
                    "protac_id": str(row["protac_id"]),
                    "source_dataset_index": int(row.get("source_dataset_index", 0)),
                    "source_anchored_linker_smiles": str(row["anchored_linker_smiles"]),
                    "source_left_fragment_smiles": str(row["left_fragment_smiles"]),
                    "source_right_fragment_smiles": str(row["right_fragment_smiles"]),
                    "generated_anchored_linker_smiles": generated_anchored_linker_smiles,
                    "generated_full_smiles": generated_full_smiles,
                    "decode_reason": decode_reason,
                    "assemble_reason": None,
                    "generated_sdf_path": str(sdf_path),
                    "mode": "difflinker",
                }
            )
    return out


def _write_outputs(rows: list[dict[str, Any]], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "repeat_index",
        "sample_id",
        "protac_id",
        "source_dataset_index",
        "source_anchored_linker_smiles",
        "source_left_fragment_smiles",
        "source_right_fragment_smiles",
        "generated_anchored_linker_smiles",
        "generated_full_smiles",
        "decode_reason",
        "assemble_reason",
        "generated_sdf_path",
        "mode",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_rows = _load_manifest(Path(args.manifest))
    rows = convert_rows(manifest_rows, generated_subdir=args.generated_subdir)
    _write_outputs(rows, Path(args.output_json), Path(args.output_csv))
    print(f"[done] converted={len(rows)} json={args.output_json} csv={args.output_csv}")


if __name__ == "__main__":
    main()
