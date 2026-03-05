from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from rdkit import Chem

# edge_attr uses one-hot encoding in this order.
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
BOND_TYPE_TO_INDEX = {bond_type: idx for idx, bond_type in enumerate(BOND_TYPES)}

# node_type semantic labels.
NODE_NORMAL = 0
NODE_ANCHOR_L = 1
NODE_ANCHOR_R = 2

# x[:, channel]
CH_ATOMIC_NUM = 0
CH_FORMAL_CHARGE = 1
CH_IS_AROMATIC = 2
CH_NUM_HS = 3
NUM_NODE_FEATURES = 4


def _safe_atomic_num(value: int) -> int:
    if value < 0:
        return 6
    if value == 0:
        return 0
    if value > 118:
        return 6
    return value


def _bond_one_hot(bond: Chem.Bond) -> list[float]:
    one_hot = [0.0] * len(BOND_TYPES)
    idx = BOND_TYPE_TO_INDEX.get(bond.GetBondType(), 0)
    one_hot[idx] = 1.0
    return one_hot


def find_anchor_indices(
    mol: Chem.Mol,
    explicit_anchor_indices: Optional[Sequence[int]] = None,
) -> tuple[Optional[int], Optional[int]]:
    """Find left/right anchors with deterministic fallback.

    Priority:
      1) explicit indices, when provided;
      2) dummy atoms with atom-map numbers 1 and 2;
      3) first two dummy atoms by atom index.
    """

    num_atoms = mol.GetNumAtoms()
    if explicit_anchor_indices is not None:
        if len(explicit_anchor_indices) != 2:
            raise ValueError("explicit_anchor_indices must contain exactly two indices")
        left, right = int(explicit_anchor_indices[0]), int(explicit_anchor_indices[1])
        if left < 0 or right < 0 or left >= num_atoms or right >= num_atoms:
            raise ValueError("explicit anchor index out of range")
        return left, right

    anchor_l: Optional[int] = None
    anchor_r: Optional[int] = None
    dummy_indices: list[int] = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() != 0:
            continue
        dummy_indices.append(idx)
        map_num = atom.GetAtomMapNum()
        if map_num == 1:
            anchor_l = idx
        elif map_num == 2:
            anchor_r = idx

    if anchor_l is None or anchor_r is None:
        remaining = [idx for idx in sorted(dummy_indices) if idx not in {anchor_l, anchor_r}]
        if anchor_l is None and remaining:
            anchor_l = remaining.pop(0)
        if anchor_r is None and remaining:
            anchor_r = remaining.pop(0)

    return anchor_l, anchor_r


def encode_mol(
    mol: Chem.Mol,
    include_original_smiles: bool = False,
    explicit_anchor_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Encode an RDKit Mol to graph tensors compatible with PyG-style inputs.

    Returns dict keys:
      - x: [N, F] float32
      - edge_index: [2, E] int64
      - edge_attr: [E, B] float32
      - node_type: [N] int64 (0 normal, 1 ANCHOR_L, 2 ANCHOR_R)
      - meta: decode metadata (atom_map_numbers, optional original_smiles)
    """

    if mol is None:
        raise ValueError("encode_mol received None")

    num_atoms = mol.GetNumAtoms()
    x = torch.zeros((num_atoms, NUM_NODE_FEATURES), dtype=torch.float32)
    node_type = torch.zeros((num_atoms,), dtype=torch.long)
    atom_map_numbers: list[int] = []
    anchor_l, anchor_r = find_anchor_indices(mol, explicit_anchor_indices=explicit_anchor_indices)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        atom_map_num = atom.GetAtomMapNum()

        x[idx, CH_ATOMIC_NUM] = float(atomic_num)
        x[idx, CH_FORMAL_CHARGE] = float(atom.GetFormalCharge())
        x[idx, CH_IS_AROMATIC] = float(atom.GetIsAromatic())
        x[idx, CH_NUM_HS] = float(atom.GetTotalNumHs(includeNeighbors=True))

        atom_map_numbers.append(int(atom_map_num))
        if idx == anchor_l:
            node_type[idx] = NODE_ANCHOR_L
        elif idx == anchor_r:
            node_type[idx] = NODE_ANCHOR_R

    src: list[int] = []
    dst: list[int] = []
    edge_attr_rows: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = _bond_one_hot(bond)
        src.extend([i, j])
        dst.extend([j, i])
        edge_attr_rows.extend([feat, feat])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_TYPES)), dtype=torch.float32)

    meta: Dict[str, Any] = {
        "atom_map_numbers": atom_map_numbers,
        "anchor_indices": [anchor_l, anchor_r],
    }
    if include_original_smiles:
        meta["original_smiles"] = Chem.MolToSmiles(mol, canonical=True)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_type": node_type,
        "meta": meta,
    }


def _sanitize_with_fallback(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    try:
        Chem.SanitizeMol(mol)
        return mol, None
    except Exception as first_err:
        try:
            flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(mol, sanitizeOps=flags)
            # Ensure resulting molecule can still be serialized.
            Chem.MolToSmiles(mol, canonical=True)
            return mol, None
        except Exception as second_err:
            return None, f"sanitize_failed: {first_err}; fallback_failed: {second_err}"


def decode_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_type: Optional[torch.Tensor] = None,
    meta: Optional[Dict[str, Any]] = None,
    return_reason: bool = False,
) -> Union[Optional[Chem.Mol], Tuple[Optional[Chem.Mol], Optional[str]]]:
    """Decode graph tensors back into an RDKit Mol.

    Args:
      x: [N, F] float tensor. Channels expected:
        0 atomic_num, 1 formal_charge, 2 is_aromatic, 3 num_hs(optional).
      edge_index: [2, E] COO tensor.
      edge_attr: [E, 4] one-hot bond type in SINGLE/DOUBLE/TRIPLE/AROMATIC.
      node_type: [N] tensor with anchor labels (0 normal, 1 left, 2 right).
      meta: optional dict from encode_mol containing atom_map_numbers.
      return_reason: when True, returns (mol_or_none, reason_or_none).
    """

    x = torch.as_tensor(x)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)

    if x.ndim != 2:
        result = (None, "x must be rank-2 [N, F]")
        return result if return_reason else result[0]
    if x.shape[1] <= CH_ATOMIC_NUM:
        result = (None, "x must contain atomic_num channel at x[:, 0]")
        return result if return_reason else result[0]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        result = (None, "edge_index must have shape [2, E]")
        return result if return_reason else result[0]
    if edge_attr.ndim != 2:
        result = (None, "edge_attr must be rank-2 [E, B]")
        return result if return_reason else result[0]
    if edge_index.shape[1] != edge_attr.shape[0]:
        result = (None, "edge_index and edge_attr edge count mismatch")
        return result if return_reason else result[0]

    num_nodes = int(x.shape[0])
    if node_type is None:
        node_type_t = torch.zeros((num_nodes,), dtype=torch.long)
    else:
        node_type_t = torch.as_tensor(node_type, dtype=torch.long)
        if node_type_t.shape[0] != num_nodes:
            result = (None, "node_type length mismatch with x")
            return result if return_reason else result[0]

    atom_map_numbers = []
    if meta is not None:
        atom_map_numbers = list(meta.get("atom_map_numbers", []))

    rw_mol = Chem.RWMol()
    for idx in range(num_nodes):
        atomic_num = _safe_atomic_num(int(round(float(x[idx, CH_ATOMIC_NUM].item()))))
        formal_charge = int(round(float(x[idx, CH_FORMAL_CHARGE].item()))) if x.shape[1] > CH_FORMAL_CHARGE else 0
        is_aromatic = bool(round(float(x[idx, CH_IS_AROMATIC].item()))) if x.shape[1] > CH_IS_AROMATIC else False

        ntype = int(node_type_t[idx].item())
        if ntype in (NODE_ANCHOR_L, NODE_ANCHOR_R):
            atomic_num = 0

        atom = Chem.Atom(int(atomic_num))
        atom.SetFormalCharge(formal_charge)
        if atomic_num != 0:
            atom.SetIsAromatic(is_aromatic)
        else:
            atom.SetIsAromatic(False)

        map_num = 0
        if idx < len(atom_map_numbers):
            map_num = int(atom_map_numbers[idx])
        if map_num <= 0 and ntype in (NODE_ANCHOR_L, NODE_ANCHOR_R):
            map_num = ntype
        if map_num > 0:
            atom.SetAtomMapNum(map_num)

        rw_mol.AddAtom(atom)

    seen_pairs: set[tuple[int, int]] = set()
    for edge_id in range(edge_index.shape[1]):
        i = int(edge_index[0, edge_id].item())
        j = int(edge_index[1, edge_id].item())
        if i == j:
            continue
        if i < 0 or j < 0 or i >= num_nodes or j >= num_nodes:
            continue

        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))

        bond_type_idx = int(torch.argmax(edge_attr[edge_id]).item()) if edge_attr.shape[1] > 0 else 0
        if bond_type_idx < 0 or bond_type_idx >= len(BOND_TYPES):
            bond_type_idx = 0
        bond_type = BOND_TYPES[bond_type_idx]

        try:
            rw_mol.AddBond(a, b, bond_type)
            bond = rw_mol.GetBondBetweenAtoms(a, b)
            if bond is not None and bond_type == Chem.BondType.AROMATIC:
                bond.SetIsAromatic(True)
                begin_atom = rw_mol.GetAtomWithIdx(a)
                end_atom = rw_mol.GetAtomWithIdx(b)
                if begin_atom.GetAtomicNum() != 0:
                    begin_atom.SetIsAromatic(True)
                if end_atom.GetAtomicNum() != 0:
                    end_atom.SetIsAromatic(True)
        except Exception:
            continue

    mol = rw_mol.GetMol()
    mol, reason = _sanitize_with_fallback(mol)
    if return_reason:
        return mol, reason
    return mol
