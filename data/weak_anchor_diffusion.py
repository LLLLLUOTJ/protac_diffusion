from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch

from data.anchored_tensor_dataset import GraphTensorBlock, collate_graph_tensor_blocks


@dataclass
class DiffusionTensorPack:
    """DDPM-ready tensor pack with masks and fixed values."""

    x_start: torch.Tensor
    sample_mask: torch.Tensor
    fixed_mask: torch.Tensor
    fixed_values: torch.Tensor
    loss_mask: torch.Tensor


def mutable_linker_node_mask(graph: GraphTensorBlock) -> torch.Tensor:
    """Only diffuse non-dummy linker atoms; anchor dummies stay fixed."""

    return (~graph.dummy_mask.bool()) & (graph.node_type.long() == 0)


def fixed_linker_node_mask(graph: GraphTensorBlock) -> torch.Tensor:
    """Freeze anchor dummies during forward and reverse diffusion."""

    return graph.dummy_mask.bool() | (graph.node_type.long() > 0)


def dense_edge_tensor_from_graph(graph: GraphTensorBlock) -> torch.Tensor:
    """Convert COO edges into a dense [N, N, B] edge tensor."""

    num_nodes = int(graph.x.shape[0])
    bond_dim = int(graph.edge_attr.shape[1]) if graph.edge_attr.ndim == 2 else 0
    dense = torch.zeros((num_nodes, num_nodes, bond_dim), dtype=graph.edge_attr.dtype)
    if graph.edge_index.numel() == 0:
        return dense
    src = graph.edge_index[0].long()
    dst = graph.edge_index[1].long()
    dense[src, dst] = graph.edge_attr
    return dense


def mutable_linker_edge_mask(graph: GraphTensorBlock) -> torch.Tensor:
    """Diffuse only edges fully inside the mutable linker interior."""

    node_mask = mutable_linker_node_mask(graph)
    pair_mask = node_mask[:, None] & node_mask[None, :]
    pair_mask.fill_diagonal_(False)
    return pair_mask


def fixed_linker_edge_mask(graph: GraphTensorBlock) -> torch.Tensor:
    """Freeze all edge slots touching dummy anchors, plus the diagonal."""

    dummy = graph.dummy_mask.bool()
    fixed_pair = dummy[:, None] | dummy[None, :]
    fixed_pair.fill_diagonal_(True)
    return fixed_pair


def build_linker_node_diffusion_pack(graph: GraphTensorBlock) -> DiffusionTensorPack:
    """Prepare node-feature diffusion tensors for a linker graph."""

    sample_mask = mutable_linker_node_mask(graph)
    fixed_mask = fixed_linker_node_mask(graph)
    return DiffusionTensorPack(
        x_start=graph.x,
        sample_mask=sample_mask,
        fixed_mask=fixed_mask,
        fixed_values=graph.x,
        loss_mask=sample_mask,
    )


def build_linker_edge_diffusion_pack(graph: GraphTensorBlock) -> DiffusionTensorPack:
    """Prepare dense edge-feature diffusion tensors for a linker graph."""

    x_start = dense_edge_tensor_from_graph(graph)
    sample_mask = mutable_linker_edge_mask(graph).unsqueeze(-1)
    fixed_mask = fixed_linker_edge_mask(graph).unsqueeze(-1)
    return DiffusionTensorPack(
        x_start=x_start,
        sample_mask=sample_mask,
        fixed_mask=fixed_mask,
        fixed_values=x_start,
        loss_mask=sample_mask,
    )


def _concat_node_packs(packs: Sequence[DiffusionTensorPack]) -> Dict[str, torch.Tensor]:
    if len(packs) == 0:
        raise ValueError("Cannot collate empty node pack list")
    x_start = torch.cat([pack.x_start for pack in packs], dim=0).unsqueeze(0)
    sample_mask = torch.cat([pack.sample_mask for pack in packs], dim=0).unsqueeze(0)
    fixed_mask = torch.cat([pack.fixed_mask for pack in packs], dim=0).unsqueeze(0)
    fixed_values = torch.cat([pack.fixed_values for pack in packs], dim=0).unsqueeze(0)
    loss_mask = torch.cat([pack.loss_mask for pack in packs], dim=0).unsqueeze(0)
    return {
        "x_start": x_start,
        "sample_mask": sample_mask,
        "fixed_mask": fixed_mask,
        "fixed_values": fixed_values,
        "loss_mask": loss_mask,
    }


def _block_diag_last3(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 0:
        raise ValueError("Cannot build block diagonal tensor from empty list")
    feat_dim = int(tensors[0].shape[-1])
    total_nodes = sum(int(t.shape[0]) for t in tensors)
    out = torch.zeros((total_nodes, total_nodes, feat_dim), dtype=tensors[0].dtype)
    offset = 0
    for tensor in tensors:
        n = int(tensor.shape[0])
        out[offset : offset + n, offset : offset + n] = tensor
        offset += n
    return out


def _block_diag_masks(masks: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(masks) == 0:
        raise ValueError("Cannot build block diagonal mask from empty list")
    total_nodes = sum(int(mask.shape[0]) for mask in masks)
    out = torch.zeros((total_nodes, total_nodes, 1), dtype=masks[0].dtype)
    offset = 0
    for mask in masks:
        n = int(mask.shape[0])
        out[offset : offset + n, offset : offset + n] = mask
        offset += n
    return out


def _concat_edge_packs(packs: Sequence[DiffusionTensorPack]) -> Dict[str, torch.Tensor]:
    if len(packs) == 0:
        raise ValueError("Cannot collate empty edge pack list")
    x_start = _block_diag_last3([pack.x_start for pack in packs]).unsqueeze(0)
    sample_mask = _block_diag_masks([pack.sample_mask for pack in packs]).unsqueeze(0)
    fixed_mask = _block_diag_masks([pack.fixed_mask for pack in packs]).unsqueeze(0)
    fixed_values = _block_diag_last3([pack.fixed_values for pack in packs]).unsqueeze(0)
    loss_mask = _block_diag_masks([pack.loss_mask for pack in packs]).unsqueeze(0)
    return {
        "x_start": x_start,
        "sample_mask": sample_mask,
        "fixed_mask": fixed_mask,
        "fixed_values": fixed_values,
        "loss_mask": loss_mask,
    }


def collate_weak_anchor_diffusion_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a DDPM-ready batch with linker masks and fragment context graphs.

    The masking policy is:
    - linker interior nodes/edges: diffuse
    - anchor dummy nodes and their incident edge slots: fixed
    - left/right fragments: conditioning context only, never noised here
    """

    if len(samples) == 0:
        raise ValueError("Cannot collate empty sample list")

    linker_graphs = [sample["linker_graph"] for sample in samples]
    left_graphs = [sample["left_graph"] for sample in samples]
    right_graphs = [sample["right_graph"] for sample in samples]

    node_packs = [build_linker_node_diffusion_pack(graph) for graph in linker_graphs]
    edge_packs = [build_linker_edge_diffusion_pack(graph) for graph in linker_graphs]

    return {
        "linker_node": _concat_node_packs(node_packs),
        "linker_edge": _concat_edge_packs(edge_packs),
        "linker_graph": collate_graph_tensor_blocks(linker_graphs),
        "left_graph": collate_graph_tensor_blocks(left_graphs),
        "right_graph": collate_graph_tensor_blocks(right_graphs),
        "sample_id": [str(sample["sample_id"]) for sample in samples],
        "protac_id": [str(sample["protac_id"]) for sample in samples],
        "linker_id": [str(sample["linker_id"]) for sample in samples],
        "full_protac_smiles": [str(sample["full_protac_smiles"]) for sample in samples],
        "linker_smiles": [str(sample["linker_smiles"]) for sample in samples],
        "anchored_linker_smiles": [str(sample["anchored_linker_smiles"]) for sample in samples],
        "left_fragment_smiles": [str(sample["left_fragment_smiles"]) for sample in samples],
        "right_fragment_smiles": [str(sample["right_fragment_smiles"]) for sample in samples],
        "anchor_left_atom_idx_in_full": torch.tensor(
            [int(sample["anchor_left_atom_idx_in_full"]) for sample in samples],
            dtype=torch.long,
        ),
        "anchor_right_atom_idx_in_full": torch.tensor(
            [int(sample["anchor_right_atom_idx_in_full"]) for sample in samples],
            dtype=torch.long,
        ),
        "linker_ratio_pct": torch.tensor([float(sample["linker_ratio_pct"]) for sample in samples], dtype=torch.float32),
        "left_ratio_pct": torch.tensor([float(sample["left_ratio_pct"]) for sample in samples], dtype=torch.float32),
        "right_ratio_pct": torch.tensor([float(sample["right_ratio_pct"]) for sample in samples], dtype=torch.float32),
    }
