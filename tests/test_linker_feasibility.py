from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from sampling.linker_feasibility import (
    assemble_protac_3d,
    estimate_linker_span,
    evaluate_full_protac_generation,
    evaluate_linker_feasibility,
    write_feasibility_outputs,
)


def _fragment_triplet_from_reference() -> tuple[Chem.Mol, Chem.Mol, Chem.Mol]:
    full = Chem.AddHs(Chem.MolFromSmiles("CCOCCN"))
    assert full is not None
    status = AllChem.EmbedMolecule(full, AllChem.ETKDGv3())
    assert status == 0
    AllChem.UFFOptimizeMolecule(full, maxIters=200)
    full = Chem.RemoveHs(full)

    cut_bonds = []
    for a, b in [(1, 2), (4, 5)]:
        bond = full.GetBondBetweenAtoms(a, b)
        assert bond is not None
        cut_bonds.append(int(bond.GetIdx()))
    fragmented = Chem.FragmentOnBonds(full, cut_bonds, addDummies=True, dummyLabels=[(1, 1), (2, 2)])
    frags = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)

    left = None
    linker = None
    right = None
    for frag in frags:
        labels = sorted(
            int(atom.GetAtomMapNum()) if int(atom.GetAtomMapNum()) > 0 else int(atom.GetIsotope())
            for atom in frag.GetAtoms()
            if atom.GetAtomicNum() == 0
        )
        if labels == [1]:
            left = frag
        elif labels == [1, 2]:
            linker = frag
        elif labels == [2]:
            right = frag
    assert left is not None
    assert linker is not None
    assert right is not None
    Chem.SanitizeMol(left)
    Chem.SanitizeMol(linker)
    Chem.SanitizeMol(right)
    return left, linker, right


def _shift_mol(mol: Chem.Mol, dx: float, dy: float, dz: float) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer(0)
    for idx in range(out.GetNumAtoms()):
        point = conf.GetAtomPosition(idx)
        conf.SetAtomPosition(idx, (float(point.x) + dx, float(point.y) + dy, float(point.z) + dz))
    return out


def test_assemble_protac_3d_returns_component_mappings() -> None:
    left, linker, right = _fragment_triplet_from_reference()
    assembled, reason = assemble_protac_3d(left, linker, right)
    assert reason is None
    assert assembled is not None
    assert assembled.mol.GetNumConformers() == 1
    assert len(assembled.left_atom_indices) > 0
    assert len(assembled.linker_atom_indices) > 0
    assert len(assembled.right_atom_indices) > 0
    assert assembled.left_anchor_atom_idx in assembled.left_atom_indices
    assert assembled.right_anchor_atom_idx in assembled.right_atom_indices


def test_estimate_linker_span_for_reference_is_reasonable() -> None:
    left, linker, right = _fragment_triplet_from_reference()
    assembled, reason = assemble_protac_3d(left, linker, right)
    assert reason is None
    assert assembled is not None
    span = estimate_linker_span(
        linker,
        reference_mol=assembled.mol,
        left_anchor_atom_idx=assembled.left_anchor_atom_idx,
        right_anchor_atom_idx=assembled.right_anchor_atom_idx,
    )
    assert span["anchor_distance"] > 0.0
    assert span["ideal_path_length"] > 0.0
    assert span["status"] in {"pass", "borderline"}


def test_evaluate_linker_feasibility_passes_on_reference_like_geometry(tmp_path: Path) -> None:
    left, linker, right = _fragment_triplet_from_reference()
    result = evaluate_linker_feasibility(
        left_fragment=left,
        anchored_linker=linker,
        right_fragment=right,
        sample_id="reference",
        num_conformers=8,
        max_iters=100,
        random_seed=7,
        save_sdf_dir=tmp_path,
    )
    assert result.status in {"pass", "borderline"}
    assert result.num_conformers_scored > 0
    assert result.best_conformer is not None
    assert (tmp_path / "reference.best.sdf").exists()


def test_evaluate_linker_feasibility_fails_for_hard_span_mismatch() -> None:
    left, _linker, right = _fragment_triplet_from_reference()
    stretched_right = _shift_mol(right, dx=10.0, dy=0.0, dz=0.0)
    short_linker = Chem.MolFromSmiles("[*:1]C[*:2]")
    assert short_linker is not None
    result = evaluate_linker_feasibility(
        left_fragment=left,
        anchored_linker=short_linker,
        right_fragment=stretched_right,
        sample_id="too_far",
        num_conformers=8,
        max_iters=100,
        random_seed=7,
    )
    assert result.status == "fail"
    assert result.reason in {"hard_span_ratio_fail", "no_valid_conformers"}


def test_write_feasibility_outputs_writes_summary_and_metrics(tmp_path: Path) -> None:
    left, linker, right = _fragment_triplet_from_reference()
    result = evaluate_linker_feasibility(
        left_fragment=left,
        anchored_linker=linker,
        right_fragment=right,
        sample_id="writeout",
        num_conformers=4,
        max_iters=50,
        random_seed=11,
    )
    summary = write_feasibility_outputs([result], out_dir=tmp_path)
    assert summary["num_samples"] == 1
    payload = json.loads((tmp_path / "per_sample_metrics.json").read_text(encoding="utf-8"))
    assert payload[0]["sample_id"] == "writeout"
    assert (tmp_path / "summary.json").exists()


def test_evaluate_full_protac_generation_runs_without_fixed_geometry(tmp_path: Path) -> None:
    left, linker, right = _fragment_triplet_from_reference()
    left_2d = Chem.MolFromSmiles(Chem.MolToSmiles(left))
    right_2d = Chem.MolFromSmiles(Chem.MolToSmiles(right))
    assert left_2d is not None
    assert right_2d is not None
    result = evaluate_full_protac_generation(
        left_fragment=left_2d,
        anchored_linker=linker,
        right_fragment=right_2d,
        sample_id="free_full",
        num_conformers=8,
        max_iters=100,
        random_seed=5,
        save_sdf_dir=tmp_path,
    )
    assert result.num_conformers_scored > 0
    assert result.best_conformer is not None
    assert result.span_estimate["mode"] == "free_full_protac"
    assert (tmp_path / "free_full.best.sdf").exists()
