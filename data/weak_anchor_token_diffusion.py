from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch

from data.anchored_tensor_dataset import collate_graph_tensor_blocks


@dataclass
class TokenDiffusionPack:
    x_start: torch.Tensor
    sample_mask: torch.Tensor
    fixed_mask: torch.Tensor
    fixed_values: torch.Tensor
    loss_mask: torch.Tensor
    token_ids: torch.Tensor
    lengths: torch.Tensor


def build_linker_token_diffusion_pack(samples: Sequence[Dict[str, Any]]) -> TokenDiffusionPack:
    if len(samples) == 0:
        raise ValueError("Cannot build token diffusion pack from empty sample list")

    batch_size = len(samples)
    embedding_dim = int(samples[0]["linker_token_embeddings"].shape[1])
    lengths = torch.tensor(
        [int(sample.get("linker_length", sample["linker_token_embeddings"].shape[0])) for sample in samples],
        dtype=torch.long,
    )
    padded_lengths = torch.tensor([int(sample["linker_token_embeddings"].shape[0]) for sample in samples], dtype=torch.long)
    max_len = int(padded_lengths.max().item())

    x_start = torch.zeros((batch_size, max_len, embedding_dim), dtype=torch.float32)
    sample_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    fixed_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    fixed_values = torch.zeros((batch_size, max_len, embedding_dim), dtype=torch.float32)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    token_ids = torch.full((batch_size, max_len), fill_value=-1, dtype=torch.long)

    for idx, sample in enumerate(samples):
        emb = sample["linker_token_embeddings"].float()
        ids = sample["linker_token_ids"].long()
        full_length = int(emb.shape[0])
        length = int(sample.get("linker_length", full_length))
        x_start[idx, :full_length] = emb
        token_ids[idx, :full_length] = ids
        sample_mask[idx, :length] = True
        loss_mask[idx, :length] = True
        if full_length > length:
            fixed_mask[idx, length:full_length] = True
            fixed_values[idx, length:full_length] = emb[length:full_length]

    return TokenDiffusionPack(
        x_start=x_start,
        sample_mask=sample_mask,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        loss_mask=loss_mask,
        token_ids=token_ids,
        lengths=lengths,
    )


def collate_weak_anchor_token_diffusion_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("Cannot collate empty token diffusion batch")

    token_pack = build_linker_token_diffusion_pack(samples)
    return {
        "linker_token": {
            "x_start": token_pack.x_start,
            "sample_mask": token_pack.sample_mask,
            "fixed_mask": token_pack.fixed_mask,
            "fixed_values": token_pack.fixed_values,
            "loss_mask": token_pack.loss_mask,
            "token_ids": token_pack.token_ids,
            "lengths": token_pack.lengths,
        },
        "left_graph": collate_graph_tensor_blocks([sample["left_graph"] for sample in samples]),
        "right_graph": collate_graph_tensor_blocks([sample["right_graph"] for sample in samples]),
        "token_smiles": [list(sample["token_smiles"]) for sample in samples],
        "token_smiles_with_maps": [list(sample["token_smiles_with_maps"]) for sample in samples],
        "oriented_token_smiles": [list(sample["oriented_token_smiles"]) for sample in samples],
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
