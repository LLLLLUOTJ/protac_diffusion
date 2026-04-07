from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms
from rdkit.Geometry import Point3D


def sanitize_copy(mol: Chem.Mol) -> tuple[Chem.Mol | None, str | None]:
    clone = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(clone)
        return clone, None
    except Exception as first_err:
        try:
            flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(clone, sanitizeOps=flags)
            Chem.MolToSmiles(clone, canonical=True)
            return clone, None
        except Exception as second_err:
            return None, f"sanitize_failed: {first_err}; fallback_failed: {second_err}"


def _first_conformer(mol: Chem.Mol) -> Chem.Conformer:
    if mol.GetNumConformers() == 0:
        raise ValueError("molecule is missing a conformer")
    return mol.GetConformer(0)


def _copy_point(point: Point3D) -> Point3D:
    return Point3D(float(point.x), float(point.y), float(point.z))


def _distance(a: Point3D, b: Point3D) -> float:
    dx = float(a.x) - float(b.x)
    dy = float(a.y) - float(b.y)
    dz = float(a.z) - float(b.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def _dummy_label(atom: Chem.Atom) -> int:
    if atom.GetAtomicNum() != 0:
        return 0
    atom_map = int(atom.GetAtomMapNum())
    if atom_map > 0:
        return atom_map
    isotope = int(atom.GetIsotope())
    if isotope > 0:
        return isotope
    return 0


def _find_labeled_dummy(fragment: Chem.Mol, label: int) -> int | None:
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() == 0 and _dummy_label(atom) == int(label):
            return int(atom.GetIdx())
    return None


def _attach_anchor_dummy(fragment: Chem.Mol, label: int, anchor_atom_idx: int) -> Chem.Mol:
    if anchor_atom_idx < 0 or anchor_atom_idx >= fragment.GetNumAtoms():
        raise IndexError(f"anchor_atom_idx out of range: {anchor_atom_idx}")
    rw = Chem.RWMol(fragment)
    dummy = Chem.Atom(0)
    dummy.SetAtomMapNum(int(label))
    dummy_idx = rw.AddAtom(dummy)
    rw.AddBond(int(anchor_atom_idx), int(dummy_idx), Chem.BondType.SINGLE)
    out = rw.GetMol()
    if fragment.GetNumConformers() > 0:
        old_conf = fragment.GetConformer(0)
        new_conf = Chem.Conformer(out.GetNumAtoms())
        for idx in range(fragment.GetNumAtoms()):
            new_conf.SetAtomPosition(idx, _copy_point(old_conf.GetAtomPosition(idx)))
        anchor_point = old_conf.GetAtomPosition(int(anchor_atom_idx))
        new_conf.SetAtomPosition(int(dummy_idx), _copy_point(anchor_point))
        out.RemoveAllConformers()
        out.AddConformer(new_conf, assignId=True)
    sane, reason = sanitize_copy(out)
    if sane is None:
        raise ValueError(f"failed to attach dummy label {label}: {reason}")
    return sane


def ensure_anchor_dummy(fragment: Chem.Mol, label: int, anchor_atom_idx: int | None = None) -> Chem.Mol:
    """Ensure one fragment carries a single dummy atom with the requested label."""

    existing = _find_labeled_dummy(fragment, label=label)
    if existing is not None:
        return Chem.Mol(fragment)
    if anchor_atom_idx is None:
        raise ValueError(f"fragment is missing [*:{label}] and no anchor_atom_idx was provided")
    return _attach_anchor_dummy(fragment=fragment, label=label, anchor_atom_idx=int(anchor_atom_idx))


def _dummy_neighbor(fragment: Chem.Mol, dummy_idx: int) -> int:
    atom = fragment.GetAtomWithIdx(int(dummy_idx))
    neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
    if len(neighbors) != 1:
        raise ValueError(f"dummy atom {dummy_idx} must have exactly one neighbor")
    return int(neighbors[0])


def assemble_protac_2d(
    left_fragment: Chem.Mol,
    anchored_linker: Chem.Mol,
    right_fragment: Chem.Mol,
    *,
    left_anchor_atom_idx: int | None = None,
    right_anchor_atom_idx: int | None = None,
) -> tuple[Chem.Mol | None, str | None]:
    """Assemble a full PROTAC in 2D without preserving input fragment coordinates."""

    try:
        left = ensure_anchor_dummy(left_fragment, label=1, anchor_atom_idx=left_anchor_atom_idx)
        right = ensure_anchor_dummy(right_fragment, label=2, anchor_atom_idx=right_anchor_atom_idx)
    except Exception as exc:
        return None, f"fragment_anchor_prepare_failed: {exc}"

    sane_linker, reason = sanitize_copy(anchored_linker)
    if sane_linker is None:
        return None, f"anchored_linker_sanitize_failed: {reason}"
    linker = sane_linker

    combo = Chem.CombineMols(Chem.CombineMols(left, linker), right)
    rw = Chem.RWMol(combo)

    anchors: dict[int, list[int]] = {1: [], 2: []}
    for atom in rw.GetAtoms():
        label = _dummy_label(atom)
        if atom.GetAtomicNum() == 0 and label in anchors:
            anchors[int(label)].append(int(atom.GetIdx()))

    add_bonds: list[tuple[int, int]] = []
    remove_atoms: list[int] = []
    for label, dummy_indices in anchors.items():
        if len(dummy_indices) != 2:
            return None, f"expected exactly two dummy atoms for label {label}, found {len(dummy_indices)}"
        neighbors = []
        for idx in dummy_indices:
            neighbors.append(_dummy_neighbor(rw, idx))
        add_bonds.append((neighbors[0], neighbors[1]))
        remove_atoms.extend(dummy_indices)

    for a, b in add_bonds:
        if rw.GetBondBetweenAtoms(int(a), int(b)) is None:
            rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    for idx in sorted(set(remove_atoms), reverse=True):
        rw.RemoveAtom(int(idx))

    mol = rw.GetMol()
    sane, reason = sanitize_copy(mol)
    if sane is None:
        return None, f"assembled_sanitize_failed: {reason}"
    return sane, None


def anchor_neighbors_from_anchored_linker(linker: Chem.Mol) -> tuple[int, int]:
    neighbors: dict[int, int] = {}
    for atom in linker.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        label = _dummy_label(atom)
        if label not in (1, 2):
            continue
        attached = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        if len(attached) != 1:
            raise ValueError(f"linker dummy [*:{label}] must have exactly one neighbor")
        neighbors[label] = int(attached[0])
    if 1 not in neighbors or 2 not in neighbors:
        raise ValueError("anchored linker must contain [*:1] and [*:2]")
    return neighbors[1], neighbors[2]


@dataclass(frozen=True)
class FeasibilityThresholds:
    pass_anchor_closure_error: float = 0.35
    borderline_anchor_closure_error: float = 0.75
    pass_internal_clash_score: float = 0.50
    borderline_internal_clash_score: float = 1.50
    pass_strain_penalty: float = 1.00
    borderline_strain_penalty: float = 1.80
    pass_protein_clash_score: float = 0.50
    borderline_protein_clash_score: float = 1.50
    hard_span_ratio_fail: float = 1.60


@dataclass
class AssembledProtac:
    mol: Chem.Mol
    left_atom_indices: list[int]
    linker_atom_indices: list[int]
    right_atom_indices: list[int]
    fixed_atom_indices: list[int]
    left_anchor_atom_idx: int
    right_anchor_atom_idx: int
    left_linker_atom_idx: int
    right_linker_atom_idx: int
    left_fragment_smiles: str
    anchored_linker_smiles: str
    right_fragment_smiles: str


@dataclass
class ConformerMetrics:
    conformer_index: int
    force_field: str
    minimize_status: int
    anchor_closure_error: float
    core_rmsd: float
    internal_clash_score: float
    internal_clash_count: int
    internal_clash_max_overlap: float
    torsion_penalty: float
    strain_penalty: float
    force_field_energy: float
    force_field_energy_per_linker_heavy: float
    protein_clash_score: float | None
    protein_clash_count: int | None
    ranking_score: float
    status: str


@dataclass
class FeasibilityResult:
    sample_id: str
    status: str
    reason: str | None
    full_protac_smiles: str | None
    span_estimate: dict[str, Any]
    num_conformers_requested: int
    num_conformers_embedded: int
    num_conformers_scored: int
    best_conformer: dict[str, Any] | None
    conformers: list[dict[str, Any]]


def assemble_protac_3d(
    left_fragment: Chem.Mol,
    anchored_linker: Chem.Mol,
    right_fragment: Chem.Mol,
    *,
    left_anchor_atom_idx: int | None = None,
    right_anchor_atom_idx: int | None = None,
    left_core_atom_indices: Sequence[int] | None = None,
    right_core_atom_indices: Sequence[int] | None = None,
) -> tuple[AssembledProtac | None, str | None]:
    """Assemble one full PROTAC while preserving input fragment coordinates."""

    if left_fragment.GetNumConformers() == 0:
        return None, "left_fragment_missing_3d"
    if right_fragment.GetNumConformers() == 0:
        return None, "right_fragment_missing_3d"

    try:
        left = ensure_anchor_dummy(left_fragment, label=1, anchor_atom_idx=left_anchor_atom_idx)
        right = ensure_anchor_dummy(right_fragment, label=2, anchor_atom_idx=right_anchor_atom_idx)
    except Exception as exc:
        return None, f"fragment_anchor_prepare_failed: {exc}"

    sane_linker, reason = sanitize_copy(anchored_linker)
    if sane_linker is None:
        return None, f"anchored_linker_sanitize_failed: {reason}"
    linker = sane_linker

    left_dummy = _find_labeled_dummy(left, label=1)
    right_dummy = _find_labeled_dummy(right, label=2)
    linker_dummy_1 = _find_labeled_dummy(linker, label=1)
    linker_dummy_2 = _find_labeled_dummy(linker, label=2)
    if None in (left_dummy, right_dummy, linker_dummy_1, linker_dummy_2):
        return None, "missing_required_anchor_dummies"

    try:
        left_core_old_local = _dummy_neighbor(left, int(left_dummy))
        right_core_old_local = _dummy_neighbor(right, int(right_dummy))
        left_linker_old_local = _dummy_neighbor(linker, int(linker_dummy_1))
        right_linker_old_local = _dummy_neighbor(linker, int(linker_dummy_2))
    except Exception as exc:
        return None, f"dummy_neighbor_failed: {exc}"

    left_n = left.GetNumAtoms()
    linker_n = linker.GetNumAtoms()
    right_n = right.GetNumAtoms()
    linker_offset = left_n
    right_offset = left_n + linker_n

    combo = Chem.CombineMols(Chem.CombineMols(left, linker), right)
    conf = Chem.Conformer(combo.GetNumAtoms())
    left_conf = left.GetConformer(0)
    right_conf = right.GetConformer(0)
    linker_conf = linker.GetConformer(0) if linker.GetNumConformers() > 0 else None
    for idx in range(left_n):
        conf.SetAtomPosition(idx, _copy_point(left_conf.GetAtomPosition(idx)))
    for idx in range(linker_n):
        point = linker_conf.GetAtomPosition(idx) if linker_conf is not None else Point3D(0.0, 0.0, 0.0)
        conf.SetAtomPosition(linker_offset + idx, _copy_point(point))
    for idx in range(right_n):
        conf.SetAtomPosition(right_offset + idx, _copy_point(right_conf.GetAtomPosition(idx)))
    combo.RemoveAllConformers()
    combo.AddConformer(conf, assignId=True)

    left_dummy_old = int(left_dummy)
    right_dummy_old = right_offset + int(right_dummy)
    linker_dummy_1_old = linker_offset + int(linker_dummy_1)
    linker_dummy_2_old = linker_offset + int(linker_dummy_2)
    left_core_old = int(left_core_old_local)
    right_core_old = right_offset + int(right_core_old_local)
    left_linker_old = linker_offset + int(left_linker_old_local)
    right_linker_old = linker_offset + int(right_linker_old_local)

    rw = Chem.RWMol(combo)
    if rw.GetBondBetweenAtoms(left_core_old, left_linker_old) is None:
        rw.AddBond(left_core_old, left_linker_old, Chem.BondType.SINGLE)
    if rw.GetBondBetweenAtoms(right_core_old, right_linker_old) is None:
        rw.AddBond(right_core_old, right_linker_old, Chem.BondType.SINGLE)

    removed = sorted({left_dummy_old, right_dummy_old, linker_dummy_1_old, linker_dummy_2_old})
    old_to_new: dict[int, int] = {}
    removed_seen = 0
    removed_set = set(removed)
    for old_idx in range(combo.GetNumAtoms()):
        if old_idx in removed_set:
            removed_seen += 1
            continue
        old_to_new[old_idx] = old_idx - removed_seen

    for atom_idx in sorted(removed, reverse=True):
        rw.RemoveAtom(int(atom_idx))
    assembled = rw.GetMol()
    sane_assembled, reason = sanitize_copy(assembled)
    if sane_assembled is None:
        return None, f"assembled_sanitize_failed: {reason}"
    assembled = sane_assembled

    left_indices = [old_to_new[idx] for idx in range(left_n) if idx != left_dummy_old]
    linker_indices = [
        old_to_new[idx]
        for idx in range(linker_offset, linker_offset + linker_n)
        if idx not in {linker_dummy_1_old, linker_dummy_2_old}
    ]
    right_indices = [
        old_to_new[idx]
        for idx in range(right_offset, right_offset + right_n)
        if idx != right_dummy_old
    ]

    if left_core_atom_indices is None:
        left_core_atom_indices = [idx for idx in range(left_n) if idx != int(left_dummy)]
    if right_core_atom_indices is None:
        right_core_atom_indices = [idx for idx in range(right_n) if idx != int(right_dummy)]

    fixed_atom_indices: list[int] = []
    for idx in left_core_atom_indices:
        if int(idx) == int(left_dummy):
            continue
        fixed_atom_indices.append(old_to_new[int(idx)])
    for idx in right_core_atom_indices:
        if int(idx) == int(right_dummy):
            continue
        fixed_atom_indices.append(old_to_new[right_offset + int(idx)])
    fixed_atom_indices = sorted(set(fixed_atom_indices))

    return (
        AssembledProtac(
            mol=assembled,
            left_atom_indices=left_indices,
            linker_atom_indices=linker_indices,
            right_atom_indices=right_indices,
            fixed_atom_indices=fixed_atom_indices,
            left_anchor_atom_idx=old_to_new[left_core_old],
            right_anchor_atom_idx=old_to_new[right_core_old],
            left_linker_atom_idx=old_to_new[left_linker_old],
            right_linker_atom_idx=old_to_new[right_linker_old],
            left_fragment_smiles=_canonical_smiles(left),
            anchored_linker_smiles=_canonical_smiles(linker),
            right_fragment_smiles=_canonical_smiles(right),
        ),
        None,
    )


def _bond_span_length(bond: Chem.Bond) -> float:
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return 1.50
    if bond_type == Chem.BondType.DOUBLE:
        return 1.34
    if bond_type == Chem.BondType.TRIPLE:
        return 1.20
    if bond_type == Chem.BondType.AROMATIC:
        return 1.40
    return 1.50


def estimate_linker_span(
    anchored_linker: Chem.Mol,
    reference_mol: Chem.Mol,
    left_anchor_atom_idx: int,
    right_anchor_atom_idx: int,
    *,
    hard_fail_ratio: float = 1.60,
) -> dict[str, Any]:
    ref_conf = _first_conformer(reference_mol)
    left_point = ref_conf.GetAtomPosition(int(left_anchor_atom_idx))
    right_point = ref_conf.GetAtomPosition(int(right_anchor_atom_idx))
    anchor_distance = _distance(left_point, right_point)

    linker_anchor_left, linker_anchor_right = anchor_neighbors_from_anchored_linker(anchored_linker)
    if int(linker_anchor_left) == int(linker_anchor_right):
        path = [int(linker_anchor_left)]
    else:
        path = list(Chem.GetShortestPath(anchored_linker, int(linker_anchor_left), int(linker_anchor_right)))
    ideal_path_length = 0.0
    if len(path) >= 2:
        for src, dst in zip(path[:-1], path[1:]):
            bond = anchored_linker.GetBondBetweenAtoms(int(src), int(dst))
            if bond is None:
                raise ValueError("shortest path contains a non-bonded atom pair")
            ideal_path_length += _bond_span_length(bond)
    span_ratio = anchor_distance / max(ideal_path_length, 1e-8)

    if span_ratio > hard_fail_ratio:
        status = "fail"
    elif span_ratio > 1.10:
        status = "borderline"
    elif span_ratio < 0.35:
        status = "borderline"
    else:
        status = "pass"
    return {
        "status": status,
        "anchor_distance": float(anchor_distance),
        "ideal_path_length": float(ideal_path_length),
        "span_ratio": float(span_ratio),
        "anchor_path_num_atoms": int(len(path)),
        "anchor_path_atom_indices": [int(idx) for idx in path],
    }


def _build_coord_map(mol: Chem.Mol, atom_indices: Sequence[int]) -> dict[int, Point3D]:
    conf = _first_conformer(mol)
    return {int(idx): _copy_point(conf.GetAtomPosition(int(idx))) for idx in atom_indices}


def _embed_constrained_trial(
    base_mol: Chem.Mol,
    coord_map: Mapping[int, Point3D],
    seed: int,
) -> Chem.Mol | None:
    trial = Chem.Mol(base_mol)
    trial.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    params.clearConfs = True
    params.enforceChirality = True
    params.numThreads = 0
    if hasattr(params, "coordMap"):
        params.coordMap = dict(coord_map)
    status = AllChem.EmbedMolecule(trial, params)
    if status != 0:
        return None
    return trial


def _embed_unconstrained_trial(base_mol: Chem.Mol, seed: int) -> Chem.Mol | None:
    trial = Chem.Mol(base_mol)
    trial.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    params.clearConfs = True
    params.enforceChirality = True
    params.numThreads = 0
    status = AllChem.EmbedMolecule(trial, params)
    if status != 0:
        return None
    return trial


def _align_trial_to_reference(
    trial: Chem.Mol,
    reference_mol: Chem.Mol,
    fixed_atom_indices: Sequence[int],
) -> None:
    atom_map = [(int(idx), int(idx)) for idx in fixed_atom_indices]
    if not atom_map:
        return
    rdMolAlign.AlignMol(trial, reference_mol, atomMap=atom_map)


def _minimize_trial(
    mol: Chem.Mol,
    *,
    fixed_atom_indices: Sequence[int],
    max_iters: int,
) -> tuple[str, int, float] | None:
    force_field_name = ""
    ff = None
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
        force_field_name = "MMFF94"
    if ff is None:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
        force_field_name = "UFF"
    if ff is None:
        return None

    if hasattr(ff, "AddFixedPoint"):
        for atom_idx in fixed_atom_indices:
            ff.AddFixedPoint(int(atom_idx))
    elif force_field_name == "MMFF94" and hasattr(ff, "MMFFAddPositionConstraint"):
        for atom_idx in fixed_atom_indices:
            ff.MMFFAddPositionConstraint(int(atom_idx), 0.0, 200.0)
    elif force_field_name == "UFF" and hasattr(ff, "UFFAddPositionConstraint"):
        for atom_idx in fixed_atom_indices:
            ff.UFFAddPositionConstraint(int(atom_idx), 0.0, 200.0)

    if hasattr(ff, "Initialize"):
        ff.Initialize()
    status = int(ff.Minimize(maxIts=int(max_iters)))
    energy = float(ff.CalcEnergy())
    return force_field_name, status, energy


def _fixed_core_errors(mol: Chem.Mol, fixed_atom_indices: Sequence[int], reference_positions: Mapping[int, Point3D]) -> float:
    conf = _first_conformer(mol)
    sq_sum = 0.0
    count = 0
    for atom_idx in fixed_atom_indices:
        if int(atom_idx) not in reference_positions:
            continue
        current = conf.GetAtomPosition(int(atom_idx))
        ref = reference_positions[int(atom_idx)]
        dist = _distance(current, ref)
        sq_sum += dist * dist
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(sq_sum / float(count))


def _anchor_closure_error(
    mol: Chem.Mol,
    left_anchor_atom_idx: int,
    right_anchor_atom_idx: int,
    reference_positions: Mapping[int, Point3D],
) -> float:
    conf = _first_conformer(mol)
    errors: list[float] = []
    for atom_idx in [int(left_anchor_atom_idx), int(right_anchor_atom_idx)]:
        ref = reference_positions[int(atom_idx)]
        errors.append(_distance(conf.GetAtomPosition(int(atom_idx)), ref))
    return float(sum(errors) / max(len(errors), 1))


def _rotatable_linker_bonds(mol: Chem.Mol, linker_atom_indices: Sequence[int]) -> list[Chem.Bond]:
    linker_set = set(int(idx) for idx in linker_atom_indices)
    bonds: list[Chem.Bond] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin not in linker_set or end not in linker_set:
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        begin_atom = mol.GetAtomWithIdx(begin)
        end_atom = mol.GetAtomWithIdx(end)
        if begin_atom.GetAtomicNum() == 1 or end_atom.GetAtomicNum() == 1:
            continue
        heavy_begin = sum(1 for nbr in begin_atom.GetNeighbors() if nbr.GetAtomicNum() > 1 and nbr.GetIdx() != end)
        heavy_end = sum(1 for nbr in end_atom.GetNeighbors() if nbr.GetAtomicNum() > 1 and nbr.GetIdx() != begin)
        if heavy_begin == 0 or heavy_end == 0:
            continue
        bonds.append(bond)
    return bonds


def _pick_torsion_neighbor(atom: Chem.Atom, exclude_idx: int) -> int | None:
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1 and nbr.GetIdx() != int(exclude_idx)]
    if heavy:
        return int(heavy[0])
    any_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != int(exclude_idx)]
    return int(any_neighbors[0]) if any_neighbors else None


def linker_torsion_penalty(mol: Chem.Mol, linker_atom_indices: Sequence[int]) -> float:
    conf = _first_conformer(mol)
    penalties: list[float] = []
    for bond in _rotatable_linker_bonds(mol, linker_atom_indices=linker_atom_indices):
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        a = _pick_torsion_neighbor(mol.GetAtomWithIdx(begin), exclude_idx=end)
        d = _pick_torsion_neighbor(mol.GetAtomWithIdx(end), exclude_idx=begin)
        if a is None or d is None:
            continue
        theta_deg = float(rdMolTransforms.GetDihedralDeg(conf, int(a), int(begin), int(end), int(d)))
        theta_rad = math.radians(theta_deg)
        penalties.append(0.5 * (1.0 + math.cos(3.0 * theta_rad)))
    if not penalties:
        return 0.0
    return float(sum(penalties) / float(len(penalties)))


def _vdw_radius(atomic_num: int) -> float:
    periodic_table = Chem.GetPeriodicTable()
    try:
        return float(periodic_table.GetRvdw(int(atomic_num)))
    except Exception:
        return 1.7


def internal_vdw_clash_score(
    mol: Chem.Mol,
    linker_atom_indices: Sequence[int],
    *,
    clash_scale: float = 0.75,
    min_topological_distance: int = 3,
) -> tuple[float, int, float]:
    conf = _first_conformer(mol)
    linker_set = set(int(idx) for idx in linker_atom_indices)
    heavy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    topo = Chem.GetDistanceMatrix(mol)
    total = 0.0
    count = 0
    max_overlap = 0.0

    for pos, atom_i in enumerate(heavy_atoms):
        if atom_i not in linker_set and all(other not in linker_set for other in heavy_atoms[pos + 1 :]):
            continue
        for atom_j in heavy_atoms[pos + 1 :]:
            if atom_i not in linker_set and atom_j not in linker_set:
                continue
            if int(topo[int(atom_i), int(atom_j)]) < int(min_topological_distance):
                continue
            point_i = conf.GetAtomPosition(int(atom_i))
            point_j = conf.GetAtomPosition(int(atom_j))
            dist = _distance(point_i, point_j)
            radius_i = _vdw_radius(mol.GetAtomWithIdx(int(atom_i)).GetAtomicNum())
            radius_j = _vdw_radius(mol.GetAtomWithIdx(int(atom_j)).GetAtomicNum())
            cutoff = clash_scale * (radius_i + radius_j)
            overlap = cutoff - dist
            if overlap <= 0.0:
                continue
            total += overlap * overlap
            count += 1
            if overlap > max_overlap:
                max_overlap = overlap
    return float(total), int(count), float(max_overlap)


def heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    return [int(atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def protein_clash_score(
    mol: Chem.Mol,
    protein: Chem.Mol,
    linker_atom_indices: Sequence[int],
    *,
    clash_scale: float = 0.75,
) -> tuple[float, int]:
    if protein.GetNumConformers() == 0:
        raise ValueError("protein is missing a conformer")
    conf = _first_conformer(mol)
    protein_conf = protein.GetConformer(0)
    total = 0.0
    count = 0
    linker_heavy = [idx for idx in linker_atom_indices if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() > 1]
    protein_heavy = [atom.GetIdx() for atom in protein.GetAtoms() if atom.GetAtomicNum() > 1]
    for atom_i in linker_heavy:
        point_i = conf.GetAtomPosition(int(atom_i))
        radius_i = _vdw_radius(mol.GetAtomWithIdx(int(atom_i)).GetAtomicNum())
        for atom_j in protein_heavy:
            point_j = protein_conf.GetAtomPosition(int(atom_j))
            radius_j = _vdw_radius(protein.GetAtomWithIdx(int(atom_j)).GetAtomicNum())
            dist = _distance(point_i, point_j)
            cutoff = clash_scale * (radius_i + radius_j)
            overlap = cutoff - dist
            if overlap <= 0.0:
                continue
            total += overlap * overlap
            count += 1
    return float(total), int(count)


def _classify_conformer(
    *,
    anchor_closure_error: float,
    internal_clash_score: float,
    strain_penalty: float,
    protein_clash_score_value: float | None,
    thresholds: FeasibilityThresholds,
) -> str:
    protein_pass = True if protein_clash_score_value is None else protein_clash_score_value <= thresholds.pass_protein_clash_score
    protein_borderline = (
        True if protein_clash_score_value is None else protein_clash_score_value <= thresholds.borderline_protein_clash_score
    )
    if (
        anchor_closure_error <= thresholds.pass_anchor_closure_error
        and internal_clash_score <= thresholds.pass_internal_clash_score
        and strain_penalty <= thresholds.pass_strain_penalty
        and protein_pass
    ):
        return "pass"
    if (
        anchor_closure_error <= thresholds.borderline_anchor_closure_error
        and internal_clash_score <= thresholds.borderline_internal_clash_score
        and strain_penalty <= thresholds.borderline_strain_penalty
        and protein_borderline
    ):
        return "borderline"
    return "fail"


def _ranking_score(
    *,
    anchor_closure_error: float,
    internal_clash_score: float,
    strain_penalty: float,
    protein_clash_score_value: float | None,
) -> float:
    score = 4.0 * float(anchor_closure_error) + 2.0 * float(internal_clash_score) + 1.5 * float(strain_penalty)
    if protein_clash_score_value is not None:
        score += 2.0 * float(protein_clash_score_value)
    return float(score)


def score_conformer(
    mol: Chem.Mol,
    assembled: AssembledProtac,
    *,
    conformer_index: int,
    force_field: str,
    minimize_status: int,
    force_field_energy: float,
    reference_positions: Mapping[int, Point3D],
    protein: Chem.Mol | None,
    thresholds: FeasibilityThresholds,
) -> ConformerMetrics:
    anchor_error = _anchor_closure_error(
        mol,
        left_anchor_atom_idx=assembled.left_anchor_atom_idx,
        right_anchor_atom_idx=assembled.right_anchor_atom_idx,
        reference_positions=reference_positions,
    )
    core_rmsd = _fixed_core_errors(mol, assembled.fixed_atom_indices, reference_positions=reference_positions)
    clash_score, clash_count, max_overlap = internal_vdw_clash_score(mol, linker_atom_indices=assembled.linker_atom_indices)
    torsion_pen = linker_torsion_penalty(mol, linker_atom_indices=assembled.linker_atom_indices)
    linker_heavy_atoms = max(
        1,
        sum(1 for idx in assembled.linker_atom_indices if mol.GetAtomWithIdx(int(idx)).GetAtomicNum() > 1),
    )
    energy_per_heavy = float(force_field_energy) / float(linker_heavy_atoms)
    strain_pen = float(torsion_pen + 0.10 * max(0.0, energy_per_heavy))
    protein_score = None
    protein_count = None
    if protein is not None:
        protein_score, protein_count = protein_clash_score(mol, protein=protein, linker_atom_indices=assembled.linker_atom_indices)
    status = _classify_conformer(
        anchor_closure_error=anchor_error,
        internal_clash_score=clash_score,
        strain_penalty=strain_pen,
        protein_clash_score_value=protein_score,
        thresholds=thresholds,
    )
    return ConformerMetrics(
        conformer_index=int(conformer_index),
        force_field=str(force_field),
        minimize_status=int(minimize_status),
        anchor_closure_error=float(anchor_error),
        core_rmsd=float(core_rmsd),
        internal_clash_score=float(clash_score),
        internal_clash_count=int(clash_count),
        internal_clash_max_overlap=float(max_overlap),
        torsion_penalty=float(torsion_pen),
        strain_penalty=float(strain_pen),
        force_field_energy=float(force_field_energy),
        force_field_energy_per_linker_heavy=float(energy_per_heavy),
        protein_clash_score=None if protein_score is None else float(protein_score),
        protein_clash_count=None if protein_count is None else int(protein_count),
        ranking_score=_ranking_score(
            anchor_closure_error=anchor_error,
            internal_clash_score=clash_score,
            strain_penalty=strain_pen,
            protein_clash_score_value=protein_score,
        ),
        status=status,
    )


def _make_prop_dict(metrics: ConformerMetrics, sample_id: str, overall_status: str) -> dict[str, str]:
    props = {"sample_id": str(sample_id), "overall_status": str(overall_status)}
    for key, value in asdict(metrics).items():
        props[str(key)] = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
    return props


def save_best_conformers(
    sample_id: str,
    conformer_records: Sequence[tuple[ConformerMetrics, Chem.Mol]],
    out_dir: str | Path,
    *,
    overall_status: str,
    top_k: int = 1,
) -> list[str]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    writer = Chem.SDWriter(str(output_dir / f"{sample_id}.best.sdf"))
    try:
        for rank, (metrics, mol) in enumerate(sorted(conformer_records, key=lambda item: item[0].ranking_score)[: max(1, int(top_k))], start=1):
            clone = Chem.Mol(mol)
            for key, value in _make_prop_dict(metrics, sample_id=sample_id, overall_status=overall_status).items():
                clone.SetProp(str(key), str(value))
            clone.SetProp("rank", str(int(rank)))
            writer.write(clone)
        saved_paths.append(str(output_dir / f"{sample_id}.best.sdf"))
    finally:
        writer.close()
    return saved_paths


def evaluate_linker_feasibility(
    *,
    left_fragment: Chem.Mol,
    anchored_linker: Chem.Mol,
    right_fragment: Chem.Mol,
    sample_id: str = "sample",
    num_conformers: int = 200,
    max_iters: int = 200,
    random_seed: int = 13,
    thresholds: FeasibilityThresholds | None = None,
    left_anchor_atom_idx: int | None = None,
    right_anchor_atom_idx: int | None = None,
    left_core_atom_indices: Sequence[int] | None = None,
    right_core_atom_indices: Sequence[int] | None = None,
    protein: Chem.Mol | None = None,
    save_sdf_dir: str | Path | None = None,
    save_top_k: int = 1,
) -> FeasibilityResult:
    """Evaluate whether a generated linker can satisfy fixed left/right 3D anchor geometry."""

    thresholds = thresholds or FeasibilityThresholds()
    assembled, reason = assemble_protac_3d(
        left_fragment=left_fragment,
        anchored_linker=anchored_linker,
        right_fragment=right_fragment,
        left_anchor_atom_idx=left_anchor_atom_idx,
        right_anchor_atom_idx=right_anchor_atom_idx,
        left_core_atom_indices=left_core_atom_indices,
        right_core_atom_indices=right_core_atom_indices,
    )
    if assembled is None:
        return FeasibilityResult(
            sample_id=str(sample_id),
            status="fail",
            reason=str(reason),
            full_protac_smiles=None,
            span_estimate={},
            num_conformers_requested=int(num_conformers),
            num_conformers_embedded=0,
            num_conformers_scored=0,
            best_conformer=None,
            conformers=[],
        )

    span = estimate_linker_span(
        anchored_linker=anchored_linker,
        reference_mol=assembled.mol,
        left_anchor_atom_idx=assembled.left_anchor_atom_idx,
        right_anchor_atom_idx=assembled.right_anchor_atom_idx,
        hard_fail_ratio=thresholds.hard_span_ratio_fail,
    )
    if span["span_ratio"] > thresholds.hard_span_ratio_fail:
        return FeasibilityResult(
            sample_id=str(sample_id),
            status="fail",
            reason="hard_span_ratio_fail",
            full_protac_smiles=_canonical_smiles(assembled.mol),
            span_estimate=span,
            num_conformers_requested=int(num_conformers),
            num_conformers_embedded=0,
            num_conformers_scored=0,
            best_conformer=None,
            conformers=[],
        )

    base_mol = Chem.AddHs(Chem.Mol(assembled.mol), addCoords=True)
    reference_positions = _build_coord_map(base_mol, assembled.fixed_atom_indices)
    conformer_records: list[tuple[ConformerMetrics, Chem.Mol]] = []
    num_embedded = 0

    for conf_idx in range(int(num_conformers)):
        trial = _embed_constrained_trial(base_mol=base_mol, coord_map=reference_positions, seed=int(random_seed) + conf_idx)
        if trial is None:
            continue
        _align_trial_to_reference(trial, reference_mol=base_mol, fixed_atom_indices=assembled.fixed_atom_indices)
        num_embedded += 1
        minimized = _minimize_trial(trial, fixed_atom_indices=assembled.fixed_atom_indices, max_iters=int(max_iters))
        if minimized is None:
            continue
        force_field_name, minimize_status, energy = minimized
        metrics = score_conformer(
            trial,
            assembled=assembled,
            conformer_index=conf_idx,
            force_field=force_field_name,
            minimize_status=minimize_status,
            force_field_energy=energy,
            reference_positions=reference_positions,
            protein=protein,
            thresholds=thresholds,
        )
        conformer_records.append((metrics, trial))

    if not conformer_records:
        return FeasibilityResult(
            sample_id=str(sample_id),
            status="fail",
            reason="no_valid_conformers",
            full_protac_smiles=_canonical_smiles(assembled.mol),
            span_estimate=span,
            num_conformers_requested=int(num_conformers),
            num_conformers_embedded=int(num_embedded),
            num_conformers_scored=0,
            best_conformer=None,
            conformers=[],
        )

    ordered = sorted(conformer_records, key=lambda item: item[0].ranking_score)
    best_metrics, _best_mol = ordered[0]
    statuses = [metrics.status for metrics, _ in conformer_records]
    overall_status = "pass" if "pass" in statuses else ("borderline" if "borderline" in statuses else "fail")
    if save_sdf_dir is not None:
        save_best_conformers(
            sample_id=str(sample_id),
            conformer_records=ordered,
            out_dir=save_sdf_dir,
            overall_status=overall_status,
            top_k=int(save_top_k),
        )

    return FeasibilityResult(
        sample_id=str(sample_id),
        status=overall_status,
        reason=None,
        full_protac_smiles=_canonical_smiles(assembled.mol),
        span_estimate=span,
        num_conformers_requested=int(num_conformers),
        num_conformers_embedded=int(num_embedded),
        num_conformers_scored=int(len(conformer_records)),
        best_conformer=asdict(best_metrics),
        conformers=[asdict(metrics) for metrics, _ in ordered],
    )


def evaluate_full_protac_generation(
    *,
    left_fragment: Chem.Mol,
    anchored_linker: Chem.Mol,
    right_fragment: Chem.Mol,
    sample_id: str = "sample",
    num_conformers: int = 200,
    max_iters: int = 200,
    random_seed: int = 13,
    thresholds: FeasibilityThresholds | None = None,
    left_anchor_atom_idx: int | None = None,
    right_anchor_atom_idx: int | None = None,
    left_core_atom_indices: Sequence[int] | None = None,
    right_core_atom_indices: Sequence[int] | None = None,
    protein: Chem.Mol | None = None,
    save_sdf_dir: str | Path | None = None,
    save_top_k: int = 1,
) -> FeasibilityResult:
    """Free whole-PROTAC conformer generation without fixed anchor geometry.

    This is a looser screen than fixed-geometry feasibility:
    - fragments are assembled in 2D
    - the full PROTAC is embedded and minimized as one molecule
    - no anchor closure term is enforced
    """

    thresholds = thresholds or FeasibilityThresholds()
    assembled, reason = assemble_protac_2d(
        left_fragment=left_fragment,
        anchored_linker=anchored_linker,
        right_fragment=right_fragment,
        left_anchor_atom_idx=left_anchor_atom_idx,
        right_anchor_atom_idx=right_anchor_atom_idx,
    )
    if assembled is None:
        return FeasibilityResult(
            sample_id=str(sample_id),
            status="fail",
            reason=str(reason),
            full_protac_smiles=None,
            span_estimate={"mode": "free_full_protac"},
            num_conformers_requested=int(num_conformers),
            num_conformers_embedded=0,
            num_conformers_scored=0,
            best_conformer=None,
            conformers=[],
        )

    linker_set = set()
    left_labeled = ensure_anchor_dummy(left_fragment, label=1, anchor_atom_idx=left_anchor_atom_idx)
    right_labeled = ensure_anchor_dummy(right_fragment, label=2, anchor_atom_idx=right_anchor_atom_idx)
    linker_sane, _ = sanitize_copy(anchored_linker)
    assert linker_sane is not None
    left_plain = Chem.MolFromSmiles(_canonical_smiles(left_labeled).replace("[*:1]", "[H]"))
    right_plain = Chem.MolFromSmiles(_canonical_smiles(right_labeled).replace("[*:2]", "[H]"))
    if left_plain is not None and right_plain is not None:
        approx_linker_heavy = max(
            1,
            assembled.GetNumHeavyAtoms() - left_plain.GetNumHeavyAtoms() - right_plain.GetNumHeavyAtoms(),
        )
    else:
        approx_linker_heavy = max(1, linker_sane.GetNumHeavyAtoms() - 2)

    base_mol = Chem.AddHs(Chem.Mol(assembled), addCoords=False)
    conformer_records: list[tuple[ConformerMetrics, Chem.Mol]] = []
    num_embedded = 0
    for conf_idx in range(int(num_conformers)):
        trial = _embed_unconstrained_trial(base_mol=base_mol, seed=int(random_seed) + conf_idx)
        if trial is None:
            continue
        num_embedded += 1
        minimized = _minimize_trial(trial, fixed_atom_indices=[], max_iters=int(max_iters))
        if minimized is None:
            continue
        force_field_name, minimize_status, energy = minimized
        clash_score, clash_count, max_overlap = internal_vdw_clash_score(
            trial,
            linker_atom_indices=heavy_atom_indices(trial),
        )
        torsion_pen = linker_torsion_penalty(trial, linker_atom_indices=heavy_atom_indices(trial))
        energy_per_heavy = float(energy) / float(max(1, approx_linker_heavy))
        strain_pen = float(torsion_pen + 0.10 * max(0.0, energy_per_heavy))
        protein_score = None
        protein_count = None
        if protein is not None:
            protein_score, protein_count = protein_clash_score(trial, protein=protein, linker_atom_indices=heavy_atom_indices(trial))

        # Free generation has no anchor closure target; keep this neutral and classify by clash/strain.
        status = _classify_conformer(
            anchor_closure_error=0.0,
            internal_clash_score=clash_score,
            strain_penalty=strain_pen,
            protein_clash_score_value=protein_score,
            thresholds=thresholds,
        )
        metrics = ConformerMetrics(
            conformer_index=int(conf_idx),
            force_field=str(force_field_name),
            minimize_status=int(minimize_status),
            anchor_closure_error=0.0,
            core_rmsd=0.0,
            internal_clash_score=float(clash_score),
            internal_clash_count=int(clash_count),
            internal_clash_max_overlap=float(max_overlap),
            torsion_penalty=float(torsion_pen),
            strain_penalty=float(strain_pen),
            force_field_energy=float(energy),
            force_field_energy_per_linker_heavy=float(energy_per_heavy),
            protein_clash_score=None if protein_score is None else float(protein_score),
            protein_clash_count=None if protein_count is None else int(protein_count),
            ranking_score=_ranking_score(
                anchor_closure_error=0.0,
                internal_clash_score=clash_score,
                strain_penalty=strain_pen,
                protein_clash_score_value=protein_score,
            ),
            status=status,
        )
        conformer_records.append((metrics, trial))

    if not conformer_records:
        return FeasibilityResult(
            sample_id=str(sample_id),
            status="fail",
            reason="no_valid_conformers",
            full_protac_smiles=_canonical_smiles(assembled),
            span_estimate={"mode": "free_full_protac"},
            num_conformers_requested=int(num_conformers),
            num_conformers_embedded=int(num_embedded),
            num_conformers_scored=0,
            best_conformer=None,
            conformers=[],
        )

    ordered = sorted(conformer_records, key=lambda item: item[0].ranking_score)
    best_metrics, _best_mol = ordered[0]
    statuses = [metrics.status for metrics, _ in conformer_records]
    overall_status = "pass" if "pass" in statuses else ("borderline" if "borderline" in statuses else "fail")
    if save_sdf_dir is not None:
        save_best_conformers(
            sample_id=str(sample_id),
            conformer_records=ordered,
            out_dir=save_sdf_dir,
            overall_status=overall_status,
            top_k=int(save_top_k),
        )

    return FeasibilityResult(
        sample_id=str(sample_id),
        status=overall_status,
        reason=None,
        full_protac_smiles=_canonical_smiles(assembled),
        span_estimate={"mode": "free_full_protac"},
        num_conformers_requested=int(num_conformers),
        num_conformers_embedded=int(num_embedded),
        num_conformers_scored=int(len(conformer_records)),
        best_conformer=asdict(best_metrics),
        conformers=[asdict(metrics) for metrics, _ in ordered],
    )


def load_mol_from_path(path: str | Path) -> Chem.Mol:
    mol_path = Path(path)
    suffix = mol_path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(mol_path), removeHs=False)
        mol = supplier[0] if supplier is not None and len(supplier) > 0 else None
        if mol is None and suffix == ".mol":
            mol = Chem.MolFromMolFile(str(mol_path), removeHs=False)
    elif suffix in {".pdb", ".ent"}:
        mol = Chem.MolFromPDBFile(str(mol_path), sanitize=False, removeHs=False)
    else:
        raise ValueError(f"unsupported molecule file type: {mol_path}")
    if mol is None:
        raise ValueError(f"failed to load molecule from {mol_path}")
    return mol


def load_optional_protein(path: str | Path | None) -> Chem.Mol | None:
    if path is None:
        return None
    protein = load_mol_from_path(path)
    if protein.GetNumConformers() == 0:
        raise ValueError(f"protein structure is missing a conformer: {path}")
    return protein


def _int_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return None
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(values, Iterable):
        return [int(v) for v in values]
    raise ValueError(f"unsupported integer-list value: {values!r}")


def evaluate_record(
    record: Mapping[str, Any],
    *,
    num_conformers: int,
    max_iters: int,
    random_seed: int,
    thresholds: FeasibilityThresholds,
    save_sdf_dir: str | Path | None,
    save_top_k: int,
) -> FeasibilityResult:
    sample_id = str(record.get("sample_id", "sample")).strip() or "sample"

    linker_smiles = str(
        record.get("anchored_linker_smiles")
        or record.get("generated_anchored_linker_smiles")
        or ""
    ).strip()
    if not linker_smiles:
        return FeasibilityResult(
            sample_id=sample_id,
            status="fail",
            reason="missing_anchored_linker_smiles",
            full_protac_smiles=None,
            span_estimate={},
            num_conformers_requested=int(num_conformers),
            num_conformers_embedded=0,
            num_conformers_scored=0,
            best_conformer=None,
            conformers=[],
        )

    left_path = record.get("left_fragment_path") or record.get("left_fragment_sdf") or record.get("left_path")
    right_path = record.get("right_fragment_path") or record.get("right_fragment_sdf") or record.get("right_path")
    linker = Chem.MolFromSmiles(linker_smiles)
    if linker is None:
        raise ValueError(f"failed to parse anchored linker smiles: {linker_smiles}")
    mode = str(record.get("mode", "fixed")).strip().lower() or "fixed"
    if left_path is not None:
        left_fragment = load_mol_from_path(left_path)
    else:
        left_smiles = str(record.get("left_fragment_smiles", "")).strip()
        left_fragment = Chem.MolFromSmiles(left_smiles) if left_smiles else None
    if right_path is not None:
        right_fragment = load_mol_from_path(right_path)
    else:
        right_smiles = str(record.get("right_fragment_smiles", "")).strip()
        right_fragment = Chem.MolFromSmiles(right_smiles) if right_smiles else None
    if left_fragment is None or right_fragment is None:
        if mode == "fixed":
            raise ValueError("fixed mode requires left/right fragment 3D paths or pre-embedded molecules")
        raise ValueError("free_full mode requires left/right fragment paths or fragment smiles")
    protein = load_optional_protein(record.get("protein_path"))

    evaluator = evaluate_linker_feasibility if mode == "fixed" else evaluate_full_protac_generation
    return evaluator(
        left_fragment=left_fragment,
        anchored_linker=linker,
        right_fragment=right_fragment,
        sample_id=sample_id,
        num_conformers=int(num_conformers),
        max_iters=int(max_iters),
        random_seed=int(random_seed),
        thresholds=thresholds,
        left_anchor_atom_idx=record.get("left_anchor_atom_idx"),
        right_anchor_atom_idx=record.get("right_anchor_atom_idx"),
        left_core_atom_indices=_int_list(record.get("left_core_atom_indices")),
        right_core_atom_indices=_int_list(record.get("right_core_atom_indices")),
        protein=protein,
        save_sdf_dir=save_sdf_dir,
        save_top_k=int(save_top_k),
    )


def write_feasibility_outputs(results: Sequence[FeasibilityResult], out_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample_metrics.json"
    summary_path = output_dir / "summary.json"

    rows = [asdict(result) for result in results]
    summary = {
        "num_samples": int(len(rows)),
        "num_pass": int(sum(1 for row in rows if row["status"] == "pass")),
        "num_borderline": int(sum(1 for row in rows if row["status"] == "borderline")),
        "num_fail": int(sum(1 for row in rows if row["status"] == "fail")),
        "num_with_best_conformer": int(sum(1 for row in rows if row["best_conformer"] is not None)),
        "pass_rate": float(sum(1 for row in rows if row["status"] == "pass") / max(len(rows), 1)),
    }
    with per_sample_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
