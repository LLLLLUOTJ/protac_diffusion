from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class PreparedInput:
    sample_id: str
    protac_id: str
    source_dataset_index: int
    fragments_path: str
    anchors: str
    linker_size: int
    full_protac_smiles: str
    left_fragment_smiles: str
    right_fragment_smiles: str
    anchored_linker_smiles: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DiffLinker 3D fragment inputs from weak-anchor samples")
    parser.add_argument("--weak-anchor-csv", type=str, default="outputs/weak_anchor_best/weak_anchor_dataset.csv")
    parser.add_argument(
        "--selection-json",
        type=str,
        default="outputs/linker_token_eval/all_generations.json",
        help="JSON rows used to choose source samples; unique (sample_id, source_dataset_index) will be kept",
    )
    parser.add_argument(
        "--protac-sdf",
        type=str,
        default="",
        help="Optional PROTAC SDF table. If provided, rows will be looked up by protac_id/_Name and used as the topology source.",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _load_selection(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("selection JSON must be a list of rows")
    return [dict(row) for row in payload]


def _unique_source_keys(rows: Sequence[dict]) -> list[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    ordered: list[tuple[str, int]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        source_idx = int(row.get("source_dataset_index", 0))
        key = (sample_id, source_idx)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _find_dummy_neighbor(mol: Chem.Mol) -> tuple[int, int]:
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        raise ValueError(f"expected exactly one dummy atom, found {len(dummy_atoms)}")
    dummy = dummy_atoms[0]
    neighbors = list(dummy.GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError("dummy atom must have exactly one neighbor")
    return dummy.GetIdx(), neighbors[0].GetIdx()


def _strip_single_dummy_and_get_anchor(smiles: str) -> tuple[Chem.Mol, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"failed to parse fragment smiles: {smiles}")
    dummy_idx, anchor_old_idx = _find_dummy_neighbor(mol)
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(anchor_old_idx).SetAtomMapNum(900)
    rw.RemoveAtom(dummy_idx)
    core = rw.GetMol()
    Chem.SanitizeMol(core)
    anchor_idx = None
    for atom in core.GetAtoms():
        if atom.GetAtomMapNum() == 900:
            anchor_idx = atom.GetIdx()
            # Relax attachment-site query constraints so the fragment can match
            # the corresponding atom in the full molecule after the linker-side bond is restored.
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            atom.SetNumRadicalElectrons(0)
            atom.SetAtomMapNum(0)
            break
    if anchor_idx is None:
        raise ValueError("failed to recover anchor atom after dummy removal")
    core.UpdatePropertyCache(strict=False)
    return core, int(anchor_idx)


def _remove_dummies(mol: Chem.Mol) -> Chem.Mol:
    dummy_indices = sorted([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0], reverse=True)
    rw = Chem.RWMol(mol)
    for idx in dummy_indices:
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"failed to parse smiles: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def _embed_molecule_topology(mol: Chem.Mol, seed: int) -> Chem.Mol:
    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(mol_h, params)
    if status != 0:
        raise ValueError(f"ETKDG embedding failed with code {status}")
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
    mol_no_h = Chem.RemoveHs(mol_h)
    Chem.SanitizeMol(mol_no_h)
    return mol_no_h


def _embed_full_molecule(smiles: str, seed: int) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"failed to parse full PROTAC smiles: {smiles}")
    return _embed_molecule_topology(mol, seed=seed)


def _has_nonzero_z(mol: Chem.Mol) -> bool:
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    return any(abs(conf.GetAtomPosition(i).z) > 1e-6 for i in range(mol.GetNumAtoms()))


def _load_protac_sdf_index(path: Path) -> dict[str, list[Chem.Mol]]:
    by_name: dict[str, list[Chem.Mol]] = {}
    suppl = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    for mol in suppl:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if not name:
            continue
        by_name.setdefault(str(name), []).append(Chem.Mol(mol))
    return by_name


def _select_full_molecule_from_sdf(
    protac_id: str,
    expected_smiles: str,
    sdf_index: dict[str, list[Chem.Mol]],
    seed: int,
) -> Chem.Mol:
    candidates = sdf_index.get(str(protac_id), [])
    if not candidates:
        raise KeyError(f"protac_id {protac_id} not found in protac SDF")
    expected_canonical = _canonicalize_smiles(expected_smiles)
    chosen = None
    for candidate in candidates:
        candidate_smiles = candidate.GetProp("Smiles") if candidate.HasProp("Smiles") else ""
        if not candidate_smiles:
            continue
        try:
            if _canonicalize_smiles(candidate_smiles) == expected_canonical:
                chosen = candidate
                break
        except Exception:
            continue
    if chosen is None:
        chosen = candidates[0]

    chosen = Chem.Mol(chosen)
    if chosen.GetNumConformers() == 0 or not _has_nonzero_z(chosen):
        return _embed_molecule_topology(chosen, seed=seed)
    Chem.SanitizeMol(chosen)
    return chosen


def _choose_non_overlapping_matches(full_mol: Chem.Mol, left_core: Chem.Mol, right_core: Chem.Mol) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left_matches = sorted(full_mol.GetSubstructMatches(left_core, uniquify=True))
    right_matches = sorted(full_mol.GetSubstructMatches(right_core, uniquify=True))
    if not left_matches:
        raise ValueError("no left fragment substructure match in full molecule")
    if not right_matches:
        raise ValueError("no right fragment substructure match in full molecule")
    for left_match in left_matches:
        left_set = set(left_match)
        for right_match in right_matches:
            if left_set.isdisjoint(right_match):
                return tuple(int(x) for x in left_match), tuple(int(x) for x in right_match)
    raise ValueError("failed to find non-overlapping left/right fragment matches")


def _copy_fragment_with_coordinates(core: Chem.Mol, full_mol: Chem.Mol, match: Sequence[int]) -> Chem.Mol:
    frag = Chem.Mol(core)
    conf = Chem.Conformer(frag.GetNumAtoms())
    full_conf = full_mol.GetConformer()
    for frag_idx, full_idx in enumerate(match):
        pos = full_conf.GetAtomPosition(int(full_idx))
        conf.SetAtomPosition(frag_idx, pos)
    frag.RemoveAllConformers()
    frag.AddConformer(conf, assignId=True)
    return frag


def _combine_fragment_conformers(left: Chem.Mol, right: Chem.Mol) -> Chem.Mol:
    combo = Chem.CombineMols(left, right)
    conf = Chem.Conformer(combo.GetNumAtoms())
    left_conf = left.GetConformer()
    right_conf = right.GetConformer()
    for idx in range(left.GetNumAtoms()):
        conf.SetAtomPosition(idx, left_conf.GetAtomPosition(idx))
    offset = left.GetNumAtoms()
    for idx in range(right.GetNumAtoms()):
        conf.SetAtomPosition(offset + idx, right_conf.GetAtomPosition(idx))
    combo.RemoveAllConformers()
    combo.AddConformer(conf, assignId=True)
    return combo


def _compute_linker_size(anchored_linker_smiles: str) -> int:
    mol = Chem.MolFromSmiles(anchored_linker_smiles)
    if mol is None:
        raise ValueError(f"failed to parse anchored linker smiles: {anchored_linker_smiles}")
    core = _remove_dummies(mol)
    return int(core.GetNumAtoms())


def _write_sdf(mol: Chem.Mol, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def prepare_inputs(
    rows: pd.DataFrame,
    selection_rows: Sequence[dict],
    out_dir: Path,
    seed: int,
    max_sources: int = 0,
    protac_sdf: str = "",
) -> list[PreparedInput]:
    selected_keys = _unique_source_keys(selection_rows)
    if max_sources > 0:
        selected_keys = selected_keys[: int(max_sources)]

    rows_by_sample_id = {str(row["sample_id"]): row for _, row in rows.iterrows()}
    sdf_index = _load_protac_sdf_index(Path(protac_sdf)) if protac_sdf else {}
    prepared: list[PreparedInput] = []
    for offset, (sample_id, source_idx) in enumerate(selected_keys):
        row = rows_by_sample_id.get(str(sample_id))
        if row is None:
            raise KeyError(f"sample_id {sample_id} not found in weak-anchor csv")

        if sdf_index:
            full = _select_full_molecule_from_sdf(
                protac_id=str(row["protac_id"]),
                expected_smiles=str(row["full_protac_smiles"]),
                sdf_index=sdf_index,
                seed=seed + offset,
            )
        else:
            full = _embed_full_molecule(str(row["full_protac_smiles"]), seed=seed + offset)
        left_core, left_anchor = _strip_single_dummy_and_get_anchor(str(row["left_fragment_smiles"]))
        right_core, right_anchor = _strip_single_dummy_and_get_anchor(str(row["right_fragment_smiles"]))
        left_match, right_match = _choose_non_overlapping_matches(full, left_core, right_core)
        left_3d = _copy_fragment_with_coordinates(left_core, full, left_match)
        right_3d = _copy_fragment_with_coordinates(right_core, full, right_match)
        combined = _combine_fragment_conformers(left_3d, right_3d)

        sample_dir = out_dir / f"sample_{int(sample_id):04d}"
        fragments_path = sample_dir / "fragments.sdf"
        _write_sdf(combined, fragments_path)

        anchors = f"{left_anchor + 1},{left_3d.GetNumAtoms() + right_anchor + 1}"
        linker_size = _compute_linker_size(str(row["anchored_linker_smiles"]))
        prepared.append(
            PreparedInput(
                sample_id=str(sample_id),
                protac_id=str(row["protac_id"]),
                source_dataset_index=int(source_idx),
                fragments_path=str(fragments_path),
                anchors=anchors,
                linker_size=linker_size,
                full_protac_smiles=str(row["full_protac_smiles"]),
                left_fragment_smiles=str(row["left_fragment_smiles"]),
                right_fragment_smiles=str(row["right_fragment_smiles"]),
                anchored_linker_smiles=str(row["anchored_linker_smiles"]),
            )
        )
    return prepared


def _write_manifest(records: Iterable[PreparedInput], out_dir: Path) -> None:
    rows = [record.__dict__ for record in records]
    manifest_csv = out_dir / "manifest.csv"
    manifest_json = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with manifest_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        manifest_csv.write_text("", encoding="utf-8")
    manifest_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    weak_anchor = pd.read_csv(args.weak_anchor_csv, dtype={"sample_id": str, "protac_id": str, "linker_id": str})
    selection_rows = _load_selection(Path(args.selection_json))
    out_dir = Path(args.out_dir)
    prepared = prepare_inputs(
        weak_anchor,
        selection_rows,
        out_dir=out_dir,
        seed=args.seed,
        max_sources=args.max_sources,
        protac_sdf=args.protac_sdf,
    )
    _write_manifest(prepared, out_dir)
    print(f"[done] prepared={len(prepared)} out={out_dir}")


if __name__ == "__main__":
    main()
