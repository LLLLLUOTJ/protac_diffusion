from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import Draw

from data.anchored_tensor_dataset import GraphTensorBlock
from data.weak_anchor_diffusion import collate_weak_anchor_diffusion_batch
from molgraph.featurize import (
    CH_ATOMIC_NUM,
    CH_FORMAL_CHARGE,
    CH_IS_AROMATIC,
    CH_NUM_HS,
    NODE_ANCHOR_L,
    NODE_ANCHOR_R,
    decode_graph,
)
from molgraph.masks import DEFAULT_MAX_VALENCE, MAX_VALENCE_BY_ATOMIC_NUM, allowed_bond_mask


ALLOWED_ATOMIC_NUMS = torch.tensor([0, 6, 7, 8, 9, 15, 16, 17, 35, 53], dtype=torch.float32)
ALLOWED_FORMAL_CHARGES = torch.tensor([-1, 0, 1], dtype=torch.float32)
BOND_DIM = 4
ALLOWED_MUTABLE_ATOMIC_NUMS = torch.tensor([6, 7, 8, 9, 15, 16, 17, 35, 53], dtype=torch.float32)
AROMATIC_CAPABLE_ATOMIC_NUMS = {6, 7, 8, 16}
BOND_ORDER_WEIGHTS = torch.tensor([1.0, 2.0, 3.0, 1.5], dtype=torch.float32)


def project_node_features(x: torch.Tensor, fixed_mask: torch.Tensor | None = None, fixed_values: torch.Tensor | None = None) -> torch.Tensor:
    """Project continuous node features back to a decode-friendly discrete space."""

    out = x.clone()
    values = out[..., CH_ATOMIC_NUM].unsqueeze(-1)
    atom_choices = ALLOWED_MUTABLE_ATOMIC_NUMS.to(device=out.device, dtype=out.dtype)
    atom_idx = torch.argmin(torch.abs(values - atom_choices.view(*((1,) * (values.ndim - 1)), -1)), dim=-1)
    out[..., CH_ATOMIC_NUM] = atom_choices[atom_idx]

    # Keep mutable linker atoms neutral by default; dummy anchors are restored from fixed_values below.
    out[..., CH_FORMAL_CHARGE] = 0.0
    out[..., CH_IS_AROMATIC] = (out[..., CH_IS_AROMATIC] >= 0.7).to(dtype=out.dtype)
    out[..., CH_NUM_HS] = torch.clamp(torch.round(out[..., CH_NUM_HS]), min=0.0, max=4.0)

    aromatic_atomic = torch.zeros_like(out[..., CH_IS_AROMATIC], dtype=torch.bool)
    for atomic_num in AROMATIC_CAPABLE_ATOMIC_NUMS:
        aromatic_atomic = aromatic_atomic | (out[..., CH_ATOMIC_NUM].round().long() == atomic_num)
    out[..., CH_IS_AROMATIC] = out[..., CH_IS_AROMATIC] * aromatic_atomic.to(dtype=out.dtype)

    if fixed_mask is not None:
        mask = fixed_mask
        if mask.ndim == out.ndim - 1:
            mask = mask.unsqueeze(-1)
        if fixed_values is None:
            raise ValueError("fixed_values must be provided when fixed_mask is used")
        out = torch.where(mask.bool(), fixed_values, out)
    return out


def project_edge_features(x: torch.Tensor, fixed_mask: torch.Tensor | None = None, fixed_values: torch.Tensor | None = None) -> torch.Tensor:
    """Project dense edge features to symmetric [0,1] bond logits with optional frozen slots."""

    out = x.clone()
    out = 0.5 * (out + out.transpose(-3, -2))
    out = torch.clamp(out, min=0.0, max=1.0)
    if fixed_mask is not None:
        if fixed_values is None:
            raise ValueError("fixed_values must be provided when fixed_mask is used")
        out = torch.where(fixed_mask.bool(), fixed_values, out)
    return out


def dense_edge_to_coo(
    edge_tensor: torch.Tensor,
    node_x: torch.Tensor,
    node_type: torch.Tensor,
    score_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dense edge tensor [N,N,4] to doubled COO edges with simple valence-aware pruning."""

    if edge_tensor.ndim != 3:
        raise ValueError(f"Expected edge_tensor rank-3 [N,N,B], got {tuple(edge_tensor.shape)}")
    num_nodes = int(edge_tensor.shape[0])
    scores = edge_tensor.max(dim=-1).values
    atomic_num = node_x[:, CH_ATOMIC_NUM].round().long().cpu()
    aromatic_node = node_x[:, CH_IS_AROMATIC] >= 0.5

    degree = torch.zeros((num_nodes,), dtype=torch.long)
    valence = torch.zeros((num_nodes,), dtype=torch.float32)
    src: list[int] = []
    dst: list[int] = []
    attrs: list[torch.Tensor] = []

    pairs: list[tuple[float, int, int]] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if scores[i, j].item() < score_threshold:
                continue
            pairs.append((float(scores[i, j].item()), i, j))
    pairs.sort(reverse=True)

    for _, i, j in pairs:
        if i == j:
            continue
        # Keep anchor dummies to a single incident bond.
        if int(node_type[i].item()) in (NODE_ANCHOR_L, NODE_ANCHOR_R) and int(degree[i].item()) >= 1:
            continue
        if int(node_type[j].item()) in (NODE_ANCHOR_L, NODE_ANCHOR_R) and int(degree[j].item()) >= 1:
            continue

        bond_scores = edge_tensor[i, j].detach().cpu().float()
        bond_type_idx = int(torch.argmax(bond_scores).item())
        if bond_type_idx == 3 and (not bool(aromatic_node[i].item()) or not bool(aromatic_node[j].item())):
            non_aromatic = bond_scores[:3]
            bond_type_idx = int(torch.argmax(non_aromatic).item())
        bond_order = float(BOND_ORDER_WEIGHTS[bond_type_idx].item())

        max_i = MAX_VALENCE_BY_ATOMIC_NUM.get(int(atomic_num[i].item()), DEFAULT_MAX_VALENCE)
        max_j = MAX_VALENCE_BY_ATOMIC_NUM.get(int(atomic_num[j].item()), DEFAULT_MAX_VALENCE)
        if float(valence[i].item()) + bond_order > float(max_i):
            continue
        if float(valence[j].item()) + bond_order > float(max_j):
            continue

        can_add = allowed_bond_mask(current_degree=degree, atomic_num=atomic_num)
        if not bool(can_add[i].item()) or not bool(can_add[j].item()):
            continue

        trial_degree = degree.clone()
        trial_degree[i] += 1
        trial_degree[j] += 1
        degree = trial_degree
        valence[i] += bond_order
        valence[j] += bond_order
        one_hot = torch.zeros((edge_tensor.shape[-1],), dtype=edge_tensor.dtype)
        one_hot[bond_type_idx] = 1.0
        src.extend([i, j])
        dst.extend([j, i])
        attrs.extend([one_hot, one_hot.clone()])

    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, edge_tensor.shape[-1]), dtype=edge_tensor.dtype)
    return torch.tensor([src, dst], dtype=torch.long), torch.stack(attrs, dim=0)


def assemble_full_molecule(
    left_fragment: Chem.Mol,
    anchored_linker: Chem.Mol,
    right_fragment: Chem.Mol,
) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Attach left/right fragments onto an anchored linker by dummy map numbers."""

    combo = Chem.CombineMols(Chem.CombineMols(left_fragment, anchored_linker), right_fragment)
    rw = Chem.RWMol(combo)

    anchors: Dict[int, list[int]] = {1: [], 2: []}
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() in anchors:
            anchors[int(atom.GetAtomMapNum())].append(atom.GetIdx())

    add_bonds: list[tuple[int, int]] = []
    remove_atoms: list[int] = []
    for label, dummy_indices in anchors.items():
        if len(dummy_indices) != 2:
            return None, f"expected exactly two dummy atoms for label {label}, found {len(dummy_indices)}"
        neighbors = []
        for idx in dummy_indices:
            atom = rw.GetAtomWithIdx(idx)
            neigh = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neigh) != 1:
                return None, f"dummy atom {idx} for label {label} does not have exactly one neighbor"
            neighbors.append(neigh[0])
        add_bonds.append((neighbors[0], neighbors[1]))
        remove_atoms.extend(dummy_indices)

    for a, b in add_bonds:
        if rw.GetBondBetweenAtoms(a, b) is None:
            rw.AddBond(a, b, Chem.BondType.SINGLE)
    for idx in sorted(set(remove_atoms), reverse=True):
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        return mol, None
    except Exception as exc:
        return None, str(exc)


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


def extract_anchored_component(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Keep only the fragment containing exactly dummy labels 1 and 2."""

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    candidates: List[Chem.Mol] = []
    for frag in frags:
        maps = sorted(int(atom.GetAtomMapNum()) for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0)
        if maps != [1, 2]:
            continue
        dummy_count = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0)
        if dummy_count != 2:
            continue
        sanitized, reason = sanitize_copy(frag)
        if sanitized is None:
            return None, reason
        candidates.append(sanitized)

    if len(candidates) == 0:
        return None, "anchored_component_missing"
    if len(candidates) > 1:
        return None, "multiple_anchored_components"
    return candidates[0], None


def batch_to_model_kwargs(batch: Dict[str, Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    out = {}
    for name in ["linker_graph", "left_graph", "right_graph"]:
        out[name] = {}
        for key, value in batch[name].items():
            out[name][key] = value.to(device) if torch.is_tensor(value) else value
    return out


def sample_index_to_batch(dataset, index: int):
    return collate_weak_anchor_diffusion_batch([dataset[index]])


def clone_graph_batch(graph: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in graph.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = value
    return out


def meta_from_node_type(node_type: torch.Tensor) -> Dict[str, List[int]]:
    maps: List[int] = []
    for value in node_type.long().tolist():
        if int(value) == NODE_ANCHOR_L:
            maps.append(1)
        elif int(value) == NODE_ANCHOR_R:
            maps.append(2)
        else:
            maps.append(0)
    return {"atom_map_numbers": maps}


def fixed_edge_template_from_graph(linker_graph: Dict[str, torch.Tensor], node_x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Bootstrap an edge-conditioning graph using only fixed dummy-adjacent edges."""

    graph = clone_graph_batch(linker_graph)
    graph["x"] = node_x.clone()

    dummy_mask = graph["dummy_mask"].bool()
    edge_index = graph["edge_index"].long()
    edge_attr = graph["edge_attr"].float()
    if edge_index.numel() == 0:
        graph["edge_index"] = torch.empty((2, 0), dtype=torch.long, device=node_x.device)
        graph["edge_attr"] = torch.empty((0, BOND_DIM), dtype=node_x.dtype, device=node_x.device)
        return graph

    src = edge_index[0]
    dst = edge_index[1]
    keep = dummy_mask[src] | dummy_mask[dst]
    graph["edge_index"] = edge_index[:, keep]
    graph["edge_attr"] = edge_attr[keep]
    return graph


def _split_dense_edge_tensor(edge_tensor: torch.Tensor, graph_ptr: torch.Tensor) -> List[torch.Tensor]:
    dense = edge_tensor.squeeze(0) if edge_tensor.ndim == 4 else edge_tensor
    ptr = graph_ptr.long().tolist()
    blocks: List[torch.Tensor] = []
    for start, end in zip(ptr[:-1], ptr[1:]):
        blocks.append(dense[start:end, start:end])
    return blocks


def _split_node_tensor(node_x: torch.Tensor, graph_ptr: torch.Tensor) -> List[torch.Tensor]:
    dense = node_x.squeeze(0) if node_x.ndim == 3 else node_x
    ptr = graph_ptr.long().tolist()
    blocks: List[torch.Tensor] = []
    for start, end in zip(ptr[:-1], ptr[1:]):
        blocks.append(dense[start:end])
    return blocks


def update_linker_graph_from_dense_edges(
    linker_graph: Dict[str, torch.Tensor],
    edge_tensor: torch.Tensor,
    node_x: torch.Tensor,
    score_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Refresh batched linker topology from dense sampled edge scores."""

    graph = clone_graph_batch(linker_graph)
    flat_node_x = node_x.squeeze(0) if node_x.ndim == 3 else node_x
    graph["x"] = flat_node_x.clone()

    edge_blocks = _split_dense_edge_tensor(edge_tensor=edge_tensor, graph_ptr=graph["graph_ptr"])
    node_blocks = _split_node_tensor(node_x=flat_node_x, graph_ptr=graph["graph_ptr"])

    edge_index_parts: List[torch.Tensor] = []
    edge_attr_parts: List[torch.Tensor] = []
    ptr = graph["graph_ptr"].long().tolist()
    node_type = graph["node_type"].long()

    for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
        local_node_type = node_type[start:end]
        local_edge_index, local_edge_attr = dense_edge_to_coo(
            edge_tensor=edge_blocks[graph_idx],
            node_x=node_blocks[graph_idx],
            node_type=local_node_type,
            score_threshold=score_threshold,
        )
        if local_edge_index.numel() == 0:
            continue
        edge_index_parts.append(local_edge_index + start)
        edge_attr_parts.append(local_edge_attr.to(dtype=flat_node_x.dtype, device=flat_node_x.device))

    if edge_index_parts:
        graph["edge_index"] = torch.cat(edge_index_parts, dim=1).to(device=flat_node_x.device)
        graph["edge_attr"] = torch.cat(edge_attr_parts, dim=0).to(device=flat_node_x.device)
    else:
        graph["edge_index"] = torch.empty((2, 0), dtype=torch.long, device=flat_node_x.device)
        graph["edge_attr"] = torch.empty((0, BOND_DIM), dtype=flat_node_x.dtype, device=flat_node_x.device)
    return graph


def decode_generated_linker_batch(
    node_x: torch.Tensor,
    edge_tensor: torch.Tensor,
    linker_graph: Dict[str, torch.Tensor],
    score_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Decode one sampled batch into per-graph linker molecules."""

    node_blocks = _split_node_tensor(node_x=node_x, graph_ptr=linker_graph["graph_ptr"])
    edge_blocks = _split_dense_edge_tensor(edge_tensor=edge_tensor, graph_ptr=linker_graph["graph_ptr"])
    ptr = linker_graph["graph_ptr"].long().tolist()
    node_type = linker_graph["node_type"].long()

    results: List[Dict[str, Any]] = []
    for graph_idx, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
        local_node_type = node_type[start:end]
        local_edge_index, local_edge_attr = dense_edge_to_coo(
            edge_tensor=edge_blocks[graph_idx],
            node_x=node_blocks[graph_idx],
            node_type=local_node_type,
            score_threshold=score_threshold,
        )
        mol, reason = decode_graph(
            x=node_blocks[graph_idx].cpu(),
            edge_index=local_edge_index.cpu(),
            edge_attr=local_edge_attr.cpu(),
            node_type=local_node_type.cpu(),
            meta=meta_from_node_type(local_node_type.cpu()),
            return_reason=True,
        )
        if mol is not None:
            mol, fragment_reason = extract_anchored_component(mol)
            if fragment_reason is not None:
                reason = fragment_reason
        results.append(
            {
                "graph_idx": graph_idx,
                "node_x": node_blocks[graph_idx].cpu(),
                "edge_index": local_edge_index.cpu(),
                "edge_attr": local_edge_attr.cpu(),
                "node_type": local_node_type.cpu(),
                "mol": mol,
                "reason": reason,
                "anchored_linker_smiles": Chem.MolToSmiles(mol, canonical=True) if mol is not None else None,
            }
        )
    return results


def draw_molecule(mol: Chem.Mol, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Draw.MolToFile(mol, str(path), size=(800, 500))
