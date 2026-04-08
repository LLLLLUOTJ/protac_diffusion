from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem

from build_weak_anchor_dataset import (
    canonical_smiles,
    extract_linker_left_right,
    get_crossing_bonds,
    is_induced_subgraph_connected,
    sanitize_copy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DeLinker sampling output into the JSON format used by existing evaluation scripts."
    )
    parser.add_argument("--input-smi", type=str, required=True, help="DeLinker .smi output with 3 columns")
    parser.add_argument("--pairs-smi", type=str, required=True, help="pairs.smi used to build the DeLinker inputs")
    parser.add_argument("--source-json", type=str, required=True, help="token all_generations.json with source_* fields")
    parser.add_argument("--output-json", type=str, required=True, help="output JSON path")
    parser.add_argument("--mode-name", type=str, default="delinker_same_length")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("source JSON must be a list of objects")
    return [dict(row) for row in payload]


def _unique_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return unique


def _canonical_smiles_or_none(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        raise ValueError(f"failed to parse smiles: {smiles}")
    sane = sanitize_copy(mol) or mol
    return canonical_smiles(sane)


def _normalize_fragments_smiles(fragments_smiles: str) -> str:
    parts = [part.strip() for part in str(fragments_smiles or "").split(".") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected exactly 2 fragment parts, got {len(parts)}: {fragments_smiles}")
    return ".".join(sorted(parts))


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 2:
            raise ValueError(f"expected 2 fields in pairs.smi, got {len(parts)}: {text}")
        pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def _read_delinker_output(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 3:
            raise ValueError(f"expected 3 fields in DeLinker output, got {len(parts)}: {text}")
        rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return rows


def _prepare_fragment_core(fragment_smiles: str) -> tuple[Chem.Mol, Chem.Mol, str]:
    fragment = Chem.MolFromSmiles(fragment_smiles)
    if fragment is None:
        raise ValueError(f"failed to parse fragment smiles: {fragment_smiles}")
    dummy_atoms = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        raise ValueError(f"expected fragment with exactly one dummy atom: {fragment_smiles}")
    dummy_idx = dummy_atoms[0].GetIdx()
    rw = Chem.RWMol(fragment)
    rw.RemoveAtom(dummy_idx)
    core = rw.GetMol()
    sane_fragment = sanitize_copy(fragment) or fragment
    sane_core = sanitize_copy(core) or core
    return sane_fragment, sane_core, canonical_smiles(sane_fragment)


def _label_dummies_with_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() in {1, 2}:
            atom.SetAtomMapNum(int(atom.GetIsotope()))
            atom.SetIsotope(0)
    return rw.GetMol()


def _fragment_with_custom_labels(mol: Chem.Mol, crossing_bonds: list[int], bond_labels: list[int]) -> Chem.Mol:
    fragmented = Chem.FragmentOnBonds(
        mol,
        list(crossing_bonds),
        addDummies=True,
        dummyLabels=[(int(label), int(label)) for label in bond_labels],
    )
    return _label_dummies_with_atom_maps(fragmented)


def _extract_anchored_linker(
    full_smiles: str,
    left_fragment_smiles: str,
    right_fragment_smiles: str,
) -> tuple[str | None, str | None]:
    full = Chem.MolFromSmiles(full_smiles)
    if full is None:
        return None, "generated_full_parse_failed"

    try:
        left_fragment, left_core, left_fragment_can = _prepare_fragment_core(left_fragment_smiles)
        right_fragment, right_core, right_fragment_can = _prepare_fragment_core(right_fragment_smiles)
    except Exception as exc:
        return None, f"fragment_prepare_failed: {exc}"

    left_matches = full.GetSubstructMatches(left_core, uniquify=True)
    right_matches = full.GetSubstructMatches(right_core, uniquify=True)
    if not left_matches:
        return None, "left_fragment_no_match"
    if not right_matches:
        return None, "right_fragment_no_match"

    for left_match in left_matches:
        left_atoms = set(int(idx) for idx in left_match)
        for right_match in right_matches:
            right_atoms = set(int(idx) for idx in right_match)
            if left_atoms & right_atoms:
                continue

            fragment_atoms = left_atoms | right_atoms
            linker_atoms = [idx for idx in range(full.GetNumAtoms()) if idx not in fragment_atoms]
            if not linker_atoms:
                continue
            if not is_induced_subgraph_connected(full, linker_atoms):
                continue

            crossing_bonds, _ = get_crossing_bonds(full, linker_atoms)
            if len(crossing_bonds) != 2:
                continue
            linker_atom_set = set(int(idx) for idx in linker_atoms)
            bond_labels: list[int] = []
            valid_labels = True
            for bond_idx in crossing_bonds:
                bond = full.GetBondWithIdx(int(bond_idx))
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                outside_atom = begin if begin not in linker_atom_set else end
                if outside_atom in left_atoms:
                    bond_labels.append(1)
                elif outside_atom in right_atoms:
                    bond_labels.append(2)
                else:
                    valid_labels = False
                    break
            if not valid_labels or sorted(bond_labels) != [1, 2]:
                continue

            fragmented = _fragment_with_custom_labels(full, crossing_bonds, bond_labels)
            anchored_linker, left_frag, right_frag, reason = extract_linker_left_right(
                fragmented,
                min_fragment_heavy_atoms=0,
            )
            if reason is not None or anchored_linker is None or left_frag is None or right_frag is None:
                continue

            linker_sane = sanitize_copy(anchored_linker)
            left_sane = sanitize_copy(left_frag)
            right_sane = sanitize_copy(right_frag)
            if linker_sane is None or left_sane is None or right_sane is None:
                continue

            if canonical_smiles(left_sane) != left_fragment_can:
                continue
            if canonical_smiles(right_sane) != right_fragment_can:
                continue
            return canonical_smiles(linker_sane), None

    return None, "linker_extraction_failed"


def _normalize_pair_key(fragments_smiles: str, full_smiles: str) -> tuple[str, str]:
    return _normalize_fragments_smiles(fragments_smiles), _canonical_smiles_or_none(full_smiles)


def main() -> None:
    args = parse_args()
    input_smi = Path(args.input_smi)
    pairs_smi = Path(args.pairs_smi)
    source_json = Path(args.source_json)
    output_json = Path(args.output_json)

    source_rows = _load_rows(source_json)
    unique_sources = _unique_sources(source_rows)
    pairs = _read_pairs(pairs_smi)
    if len(pairs) != len(unique_sources):
        raise ValueError(
            f"pairs/source length mismatch: {len(pairs)} pairs vs {len(unique_sources)} unique source rows"
        )

    pair_to_source: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, source in zip(pairs, unique_sources):
        pair_to_source[_normalize_pair_key(pair[0], pair[1])] = source

    repeat_counters: defaultdict[tuple[str, str], int] = defaultdict(int)
    output_rows: list[dict[str, Any]] = []

    for row_index, (fragments_smiles, reference_full_smiles, generated_full_smiles) in enumerate(
        _read_delinker_output(input_smi),
        start=1,
    ):
        source = pair_to_source.get(_normalize_pair_key(fragments_smiles, reference_full_smiles))
        if source is None:
            raise KeyError(
                "could not map DeLinker row back to source metadata for "
                f"{fragments_smiles} / {reference_full_smiles}"
            )

        repeat_key = _normalize_pair_key(fragments_smiles, reference_full_smiles)
        repeat_index = repeat_counters[repeat_key]
        repeat_counters[repeat_key] += 1

        anchored_linker, reason = _extract_anchored_linker(
            full_smiles=generated_full_smiles,
            left_fragment_smiles=str(source["left_fragment_smiles"]),
            right_fragment_smiles=str(source["right_fragment_smiles"]),
        )

        output_rows.append(
            {
                "sample_id": f"delinker_{row_index:04d}",
                "source_dataset_index": str(source["source_dataset_index"]),
                "repeat_index": int(repeat_index),
                "source_left_fragment_smiles": str(source["left_fragment_smiles"]),
                "source_right_fragment_smiles": str(source["right_fragment_smiles"]),
                "source_anchored_linker_smiles": str(source["anchored_linker_smiles"]),
                "reference_full_smiles": str(reference_full_smiles),
                "generated_full_smiles": str(generated_full_smiles),
                "generated_anchored_linker_smiles": anchored_linker,
                "decode_reason": reason,
                "assemble_reason": None,
                "mode": str(args.mode_name),
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    decoded = sum(row["generated_anchored_linker_smiles"] is not None for row in output_rows)
    print(f"[done] converted={len(output_rows)} decoded={decoded} out={output_json}")


if __name__ == "__main__":
    main()
