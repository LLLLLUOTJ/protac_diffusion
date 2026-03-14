from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from rdkit import Chem

from build_linker_anchor_fragment_library import (
    anchor_path_single_bonds,
    fragment_on_path_single_bonds,
    ordered_fragment_tokens,
)


def sanitize_copy(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[str]]:
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


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def infer_token_attachment_labels(token_index: int, num_tokens: int) -> Tuple[int, int]:
    """Return the left/right dummy labels a token should use in a chain."""

    if num_tokens < 1:
        raise ValueError("num_tokens must be >= 1")
    if token_index < 0 or token_index >= num_tokens:
        raise ValueError(f"token_index out of range: {token_index} for num_tokens={num_tokens}")

    left_label = 1 if token_index == 0 else token_index + 2
    right_label = 2 if token_index == (num_tokens - 1) else token_index + 3
    return left_label, right_label


def _dummy_atoms(mol: Chem.Mol) -> list[Chem.Atom]:
    return [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"failed to parse smiles: {smiles}")
    return mol


def anchor_neighbors_from_anchored_mol(mol: Chem.Mol) -> Tuple[int, int]:
    """Return the two linker anchor atom indices adjacent to [*:1] and [*:2]."""

    neighbors: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        label = int(atom.GetAtomMapNum())
        if label not in (1, 2):
            continue
        attached = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        if len(attached) != 1:
            raise ValueError(f"dummy atom [*:{label}] must have exactly one neighbor")
        neighbors[label] = int(attached[0])
    if 1 not in neighbors or 2 not in neighbors:
        raise ValueError("anchored linker must contain dummy labels 1 and 2")
    if neighbors[1] == neighbors[2]:
        raise ValueError("anchor neighbors for [*:1] and [*:2] must be distinct")
    return neighbors[1], neighbors[2]


def tokenize_anchored_linker(
    anchored_linker: str | Chem.Mol,
    *,
    include_ring_single_bonds: bool = False,
) -> Dict[str, object]:
    """Tokenize one anchored linker directly into base/oriented/mapped token sequences."""

    mol = anchored_linker if isinstance(anchored_linker, Chem.Mol) else _mol_from_smiles(str(anchored_linker))
    mol = Chem.Mol(mol)
    sane, reason = sanitize_copy(mol)
    if sane is None:
        raise ValueError(f"failed to sanitize anchored linker: {reason}")

    anchor_1, anchor_2 = anchor_neighbors_from_anchored_mol(sane)
    path, cut_bond_indices = anchor_path_single_bonds(
        sane,
        anchor_1=int(anchor_1),
        anchor_2=int(anchor_2),
        include_ring_single_bonds=bool(include_ring_single_bonds),
    )
    if len(path) <= 1:
        raise ValueError("anchor path too short")

    fragmented = fragment_on_path_single_bonds(
        sane,
        cut_bond_indices=cut_bond_indices,
        dummy_label_start=3,
    )
    tokens = ordered_fragment_tokens(fragmented, path=path)
    if not tokens:
        raise ValueError("failed to derive fragment tokens from anchored linker")

    token_smiles = [str(item["token_smiles"]) for item in tokens]
    token_smiles_with_maps = [str(item["token_smiles_with_maps"]) for item in tokens]
    oriented_token_smiles = normalize_mapped_token_sequence(token_smiles_with_maps)
    return {
        "token_smiles": token_smiles,
        "token_smiles_with_maps": token_smiles_with_maps,
        "oriented_token_smiles": oriented_token_smiles,
        "anchor_path_atom_indices": [int(x) for x in path],
        "anchor_path_single_bond_cut_indices": [int(x) for x in cut_bond_indices],
        "num_cuts": int(len(cut_bond_indices)),
        "num_fragments": int(len(tokens)),
        "anchor_graph_distance": int(len(path) - 1),
    }


def build_default_token_template(token_smiles: str) -> str:
    """Build a fallback template by ordering dummy atoms by atom index."""

    mol = _mol_from_smiles(token_smiles)
    dummies = sorted(_dummy_atoms(mol), key=lambda atom: atom.GetIdx())
    if len(dummies) != 2:
        raise ValueError(f"expected exactly 2 attachment dummies in token: {token_smiles}")

    for atom in dummies:
        atom.SetAtomMapNum(0)
    dummies[0].SetAtomMapNum(1)
    dummies[1].SetAtomMapNum(2)

    sane, reason = sanitize_copy(mol)
    if sane is None:
        raise ValueError(f"failed to sanitize default token template: {reason}")
    return canonical_smiles(sane)


def normalize_token_template(token_smiles_with_maps: str, left_label: int, right_label: int) -> str:
    """Normalize a mapped token into a reusable [*:1]/[*:2] orientation template."""

    if left_label == right_label:
        raise ValueError("left_label and right_label must be distinct")

    mol = _mol_from_smiles(token_smiles_with_maps)
    dummies = _dummy_atoms(mol)
    if len(dummies) != 2:
        raise ValueError(f"expected exactly 2 attachment dummies in mapped token: {token_smiles_with_maps}")

    by_label = {int(atom.GetAtomMapNum()): atom for atom in dummies}
    if left_label not in by_label or right_label not in by_label:
        raise ValueError(
            f"mapped token does not contain expected left/right labels {left_label}/{right_label}: {token_smiles_with_maps}"
        )

    for atom in dummies:
        atom.SetAtomMapNum(0)
    by_label[left_label].SetAtomMapNum(1)
    by_label[right_label].SetAtomMapNum(2)

    sane, reason = sanitize_copy(mol)
    if sane is None:
        raise ValueError(f"failed to sanitize normalized token template: {reason}")
    return canonical_smiles(sane)


def instantiate_token_template(template_smiles: str, left_label: int, right_label: int) -> str:
    """Instantiate a [*:1]/[*:2] template with concrete chain labels."""

    if left_label == right_label:
        raise ValueError("left_label and right_label must be distinct")

    mol = _mol_from_smiles(template_smiles)
    dummies = _dummy_atoms(mol)
    if len(dummies) != 2:
        raise ValueError(f"expected exactly 2 attachment dummies in token template: {template_smiles}")

    seen = set()
    for atom in dummies:
        label = int(atom.GetAtomMapNum())
        seen.add(label)
        if label == 1:
            atom.SetAtomMapNum(int(left_label))
        elif label == 2:
            atom.SetAtomMapNum(int(right_label))
        else:
            raise ValueError(f"token template must only contain dummy labels 1/2: {template_smiles}")
    if seen != {1, 2}:
        raise ValueError(f"token template must contain both dummy labels 1 and 2: {template_smiles}")

    sane, reason = sanitize_copy(mol)
    if sane is None:
        raise ValueError(f"failed to sanitize instantiated token template: {reason}")
    return canonical_smiles(sane)


def build_token_templates_from_rows(rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
    """Build one orientation template per token_smiles from instance-style rows."""

    rows_by_sample: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        rows_by_sample[sample_id].append(row)

    template_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample_rows in rows_by_sample.values():
        valid_rows: list[tuple[int, Mapping[str, str]]] = []
        for row in sample_rows:
            try:
                token_index = int(str(row.get("token_index", "")).strip())
            except Exception:
                continue
            valid_rows.append((token_index, row))
        if not valid_rows:
            continue

        valid_rows.sort(key=lambda item: item[0])
        num_tokens = valid_rows[-1][0] + 1
        for token_index, row in valid_rows:
            token_smiles = str(row.get("token_smiles", "")).strip()
            token_smiles_with_maps = str(row.get("token_smiles_with_maps", "")).strip()
            if not token_smiles or not token_smiles_with_maps:
                continue
            left_label, right_label = infer_token_attachment_labels(token_index=token_index, num_tokens=num_tokens)
            normalized = normalize_token_template(
                token_smiles_with_maps=token_smiles_with_maps,
                left_label=left_label,
                right_label=right_label,
            )
            template_counts[token_smiles][normalized] += 1

    templates: Dict[str, str] = {}
    for token_smiles, counts in template_counts.items():
        template, _ = counts.most_common(1)[0]
        templates[token_smiles] = template
    return templates


def build_token_templates_from_instances_csv(instances_csv: str | Path) -> Dict[str, str]:
    path = Path(instances_csv)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({str(k): str(v) for k, v in row.items()})
    return build_token_templates_from_rows(rows)


def mapped_tokens_from_token_sequence(
    token_smiles_sequence: Sequence[str],
    token_templates: Mapping[str, str],
    *,
    strict: bool = False,
) -> list[str]:
    """Map a token sequence into concrete token smiles with chain-specific dummy labels."""

    tokens = [str(token).strip() for token in token_smiles_sequence if str(token).strip()]
    if not tokens:
        raise ValueError("token_smiles_sequence must contain at least one token")

    mapped: list[str] = []
    num_tokens = len(tokens)
    for token_index, token_smiles in enumerate(tokens):
        template = token_templates.get(token_smiles)
        if template is None:
            if strict:
                raise KeyError(f"missing token template for token: {token_smiles}")
            template = build_default_token_template(token_smiles)
        left_label, right_label = infer_token_attachment_labels(token_index=token_index, num_tokens=num_tokens)
        mapped.append(
            instantiate_token_template(
                template_smiles=template,
                left_label=left_label,
                right_label=right_label,
            )
        )
    return mapped


def normalize_mapped_token_sequence(mapped_tokens: Sequence[str]) -> list[str]:
    """Convert one concrete mapped chain into oriented [*:1]/[*:2] token templates."""

    tokens = [str(token).strip() for token in mapped_tokens if str(token).strip()]
    if not tokens:
        raise ValueError("mapped_tokens must contain at least one token")

    normalized: list[str] = []
    num_tokens = len(tokens)
    for token_index, token_smiles_with_maps in enumerate(tokens):
        left_label, right_label = infer_token_attachment_labels(token_index=token_index, num_tokens=num_tokens)
        normalized.append(
            normalize_token_template(
                token_smiles_with_maps=token_smiles_with_maps,
                left_label=left_label,
                right_label=right_label,
            )
        )
    return normalized


def mapped_tokens_from_oriented_sequence(oriented_tokens: Sequence[str]) -> list[str]:
    """Instantiate an oriented token sequence back into concrete mapped tokens."""

    tokens = [str(token).strip() for token in oriented_tokens if str(token).strip()]
    if not tokens:
        raise ValueError("oriented_tokens must contain at least one token")

    mapped: list[str] = []
    num_tokens = len(tokens)
    for token_index, template_smiles in enumerate(tokens):
        left_label, right_label = infer_token_attachment_labels(token_index=token_index, num_tokens=num_tokens)
        mapped.append(
            instantiate_token_template(
                template_smiles=template_smiles,
                left_label=left_label,
                right_label=right_label,
            )
        )
    return mapped


def stitch_mapped_tokens(mapped_tokens: Sequence[str]) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Stitch a mapped token sequence back into one anchored linker mol."""

    token_mols: list[Chem.Mol] = []
    for token_smiles in mapped_tokens:
        token_mols.append(_mol_from_smiles(token_smiles))
    if not token_mols:
        return None, "empty_token_sequence"

    combo = token_mols[0]
    for mol in token_mols[1:]:
        combo = Chem.CombineMols(combo, mol)
    rw = Chem.RWMol(combo)

    label_to_dummy_indices: dict[int, list[int]] = defaultdict(list)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        label = int(atom.GetAtomMapNum())
        if label <= 0:
            return None, "dummy_without_positive_map"
        label_to_dummy_indices[label].append(atom.GetIdx())

    for anchor_label in (1, 2):
        if len(label_to_dummy_indices.get(anchor_label, [])) != 1:
            return None, f"anchor_label_{anchor_label}_count_ne_1"

    remove_atoms: list[int] = []
    for label, dummy_indices in sorted(label_to_dummy_indices.items()):
        if label in (1, 2):
            continue
        if len(dummy_indices) != 2:
            return None, f"internal_label_{label}_count_ne_2"

        neighbors: list[int] = []
        for dummy_idx in dummy_indices:
            atom = rw.GetAtomWithIdx(dummy_idx)
            attached = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
            if len(attached) != 1:
                return None, f"dummy_label_{label}_degree_ne_1"
            neighbors.append(attached[0])
        if neighbors[0] == neighbors[1]:
            return None, f"internal_label_{label}_same_neighbor"
        if rw.GetBondBetweenAtoms(neighbors[0], neighbors[1]) is None:
            rw.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
        remove_atoms.extend(dummy_indices)

    for atom_idx in sorted(set(remove_atoms), reverse=True):
        rw.RemoveAtom(atom_idx)

    mol = rw.GetMol()
    sane, reason = sanitize_copy(mol)
    if sane is None:
        return None, reason

    remaining_anchor_maps = sorted(
        int(atom.GetAtomMapNum())
        for atom in sane.GetAtoms()
        if atom.GetAtomicNum() == 0 and int(atom.GetAtomMapNum()) > 0
    )
    if remaining_anchor_maps != [1, 2]:
        return None, f"remaining_anchor_maps_invalid:{remaining_anchor_maps}"
    return sane, None


def nearest_vocab_token_ids(
    token_embeddings: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    *,
    top_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return nearest token ids by cosine similarity."""

    if token_embeddings.ndim == 1:
        token_embeddings = token_embeddings.unsqueeze(0)
    if vocab_embeddings.ndim != 2:
        raise ValueError(f"vocab_embeddings must be rank-2, got shape {tuple(vocab_embeddings.shape)}")
    if token_embeddings.shape[-1] != vocab_embeddings.shape[-1]:
        raise ValueError(
            "token_embeddings and vocab_embeddings must share the same embedding dimension: "
            f"{token_embeddings.shape[-1]} vs {vocab_embeddings.shape[-1]}"
        )
    if top_k < 1 or top_k > vocab_embeddings.shape[0]:
        raise ValueError(f"top_k must be in [1, {vocab_embeddings.shape[0]}], got {top_k}")

    q = F.normalize(token_embeddings.float(), p=2, dim=-1)
    v = F.normalize(vocab_embeddings.float(), p=2, dim=-1)
    scores = q @ v.transpose(0, 1)
    top_scores, top_ids = torch.topk(scores, k=top_k, dim=-1)
    return top_ids, top_scores


def decode_token_embeddings(
    token_embeddings: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    vocab_tokens: Sequence[str],
    *,
    top_k: int = 1,
) -> Tuple[list[str], torch.Tensor]:
    """Project predicted token embeddings back to nearest vocab tokens."""

    token_ids, scores = nearest_vocab_token_ids(
        token_embeddings=token_embeddings,
        vocab_embeddings=vocab_embeddings,
        top_k=top_k,
    )
    if int(token_ids.shape[-1]) != 1:
        raise ValueError("decode_token_embeddings currently expects top_k=1")
    decoded = [str(vocab_tokens[int(idx)]) for idx in token_ids.squeeze(-1).tolist()]
    return decoded, scores.squeeze(-1)


def decode_embedding_sequence_to_linker(
    token_embeddings: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    vocab_tokens: Sequence[str],
    token_templates: Mapping[str, str],
) -> Dict[str, object]:
    """Decode token embeddings into tokens, mapped token smiles, and one anchored linker."""

    token_smiles, scores = decode_token_embeddings(
        token_embeddings=token_embeddings,
        vocab_embeddings=vocab_embeddings,
        vocab_tokens=vocab_tokens,
        top_k=1,
    )
    mapped_tokens = mapped_tokens_from_token_sequence(token_smiles, token_templates)
    mol, reason = stitch_mapped_tokens(mapped_tokens)
    return {
        "token_smiles": token_smiles,
        "mapped_tokens": mapped_tokens,
        "scores": scores.detach().cpu(),
        "mol": mol,
        "anchored_linker_smiles": canonical_smiles(mol) if mol is not None else None,
        "reason": reason,
    }


def decode_oriented_embedding_sequence_to_linker(
    token_embeddings: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    vocab_tokens: Sequence[str],
    *,
    stop_token: str | None = None,
) -> Dict[str, object]:
    """Decode oriented-token embeddings back into one anchored linker."""

    oriented_tokens, scores = decode_token_embeddings(
        token_embeddings=token_embeddings,
        vocab_embeddings=vocab_embeddings,
        vocab_tokens=vocab_tokens,
        top_k=1,
    )
    raw_oriented_tokens = list(oriented_tokens)
    stop_index: int | None = None
    if stop_token is not None:
        try:
            stop_index = raw_oriented_tokens.index(str(stop_token))
        except ValueError:
            stop_index = None
        if stop_index is not None:
            oriented_tokens = raw_oriented_tokens[:stop_index]

    if not oriented_tokens:
        return {
            "oriented_token_smiles_raw": raw_oriented_tokens,
            "oriented_token_smiles": [],
            "mapped_tokens": [],
            "scores": scores.detach().cpu(),
            "mol": None,
            "anchored_linker_smiles": None,
            "reason": "empty_sequence_after_stop_token" if stop_index is not None else "empty_token_sequence",
            "stop_index": stop_index,
        }

    mapped_tokens = mapped_tokens_from_oriented_sequence(oriented_tokens)
    mol, reason = stitch_mapped_tokens(mapped_tokens)
    return {
        "oriented_token_smiles_raw": raw_oriented_tokens,
        "oriented_token_smiles": oriented_tokens,
        "mapped_tokens": mapped_tokens,
        "scores": scores.detach().cpu(),
        "mol": mol,
        "anchored_linker_smiles": canonical_smiles(mol) if mol is not None else None,
        "reason": reason,
        "stop_index": stop_index,
    }
