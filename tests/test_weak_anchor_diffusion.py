from __future__ import annotations

import csv

import torch
from rdkit import Chem
from torch import nn

from build_weak_anchor_dataset import MolRecord, process_pair
from data.anchored_tensor_dataset import WeakAnchorTensorDataset, WeakAnchorTensorPTDataset, serialize_weak_anchor_tensor_dataset
from data.weak_anchor_diffusion import (
    build_linker_edge_diffusion_pack,
    build_linker_node_diffusion_pack,
    collate_weak_anchor_diffusion_batch,
)
from diffusion.ddpm import DDPM


class ZeroNoiseModel(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _make_record(smiles: str, row_id: str) -> MolRecord:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return MolRecord(
        row_id=row_id,
        smiles=smiles,
        mol=mol,
        canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
        num_atoms=mol.GetNumAtoms(),
        source_row={},
    )


def _make_rows() -> list[dict[str, str]]:
    accepted_1, rej_1 = process_pair(
        _make_record("c1ccccc1CCOCCNc2ccccc2", "p1"),
        _make_record("CCOCCN", "l1"),
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=1,
        min_anchor_graph_distance=1,
        min_linker_ratio_pct=0.0,
        max_linker_ratio_pct=100.0,
    )
    accepted_2, rej_2 = process_pair(
        _make_record("c1ccccc1CCOCCOCCNc2ccccc2", "p2"),
        _make_record("CCOCCOCCN", "l2"),
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=1,
        min_anchor_graph_distance=1,
        min_linker_ratio_pct=0.0,
        max_linker_ratio_pct=100.0,
    )
    assert accepted_1 is not None, rej_1
    assert accepted_2 is not None, rej_2
    accepted_1["sample_id"] = "1"
    accepted_2["sample_id"] = "2"
    return [
        {k: str(v) for k, v in accepted_1.items() if not k.startswith("_")},
        {k: str(v) for k, v in accepted_2.items() if not k.startswith("_")},
    ]


def _build_pt_dataset(tmp_path) -> WeakAnchorTensorPTDataset:
    csv_path = tmp_path / "weak_anchor.csv"
    rows = _make_rows()
    fieldnames = [
        "sample_id",
        "protac_id",
        "linker_id",
        "full_protac_smiles",
        "linker_smiles",
        "anchored_linker_smiles",
        "left_fragment_smiles",
        "right_fragment_smiles",
        "anchor_left_atom_idx_in_full",
        "anchor_right_atom_idx_in_full",
        "num_atoms_full",
        "num_atoms_linker",
        "num_atoms_left",
        "num_atoms_right",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ds = WeakAnchorTensorDataset(str(csv_path), include_pair_mask=True)
    pt_path = tmp_path / "weak_anchor.pt"
    serialize_weak_anchor_tensor_dataset(ds, str(pt_path), include_pair_mask=True)
    return WeakAnchorTensorPTDataset(str(pt_path))


def test_linker_node_pack_freezes_dummy_nodes(tmp_path) -> None:
    ds = _build_pt_dataset(tmp_path)
    sample = ds[0]
    pack = build_linker_node_diffusion_pack(sample["linker_graph"])

    assert pack.x_start.ndim == 2
    assert int(pack.fixed_mask.sum().item()) == 2
    assert int(pack.sample_mask.sum().item()) == sample["linker_graph"].x.shape[0] - 2
    assert torch.all(pack.fixed_values[pack.fixed_mask] == sample["linker_graph"].x[pack.fixed_mask])


def test_linker_edge_pack_freezes_dummy_incident_edges(tmp_path) -> None:
    ds = _build_pt_dataset(tmp_path)
    sample = ds[0]
    graph = sample["linker_graph"]
    pack = build_linker_edge_diffusion_pack(graph)

    assert pack.x_start.ndim == 3
    assert pack.sample_mask.shape[-1] == 1
    assert pack.fixed_mask.shape[-1] == 1

    dummy_idx = torch.nonzero(graph.dummy_mask, as_tuple=False).view(-1)
    assert dummy_idx.numel() == 2
    first_dummy = int(dummy_idx[0].item())
    assert bool(pack.fixed_mask[first_dummy].any().item())
    assert not bool(pack.sample_mask[first_dummy].any().item())


def test_collate_weak_anchor_diffusion_batch_shapes(tmp_path) -> None:
    ds = _build_pt_dataset(tmp_path)
    batch = collate_weak_anchor_diffusion_batch([ds[0], ds[1]])

    assert batch["linker_node"]["x_start"].ndim == 3
    assert batch["linker_edge"]["x_start"].ndim == 4
    assert batch["linker_node"]["sample_mask"].ndim == 2
    assert batch["linker_edge"]["sample_mask"].ndim == 4
    assert batch["left_graph"]["x"].ndim == 2
    assert batch["right_graph"]["x"].ndim == 2
    assert len(batch["sample_id"]) == 2


def test_ddpm_respects_linker_masks_for_nodes_and_edges(tmp_path) -> None:
    ds = _build_pt_dataset(tmp_path)
    sample = ds[0]
    node_pack = build_linker_node_diffusion_pack(sample["linker_graph"])
    edge_pack = build_linker_edge_diffusion_pack(sample["linker_graph"])
    ddpm = DDPM(model=ZeroNoiseModel(), timesteps=4)

    node_out = ddpm.sample(
        shape=(1,) + tuple(node_pack.x_start.shape),
        initial_noise=torch.ones((1,) + tuple(node_pack.x_start.shape)),
        sample_mask=node_pack.sample_mask.unsqueeze(0),
        fixed_mask=node_pack.fixed_mask.unsqueeze(0),
        fixed_values=node_pack.fixed_values.unsqueeze(0),
        show_progress=False,
    )
    fixed_node_mask = node_pack.fixed_mask.unsqueeze(0).unsqueeze(-1).expand_as(node_out)
    assert torch.all(node_out[fixed_node_mask] == node_pack.fixed_values.unsqueeze(0)[fixed_node_mask])

    edge_out = ddpm.sample(
        shape=(1,) + tuple(edge_pack.x_start.shape),
        initial_noise=torch.ones((1,) + tuple(edge_pack.x_start.shape)),
        sample_mask=edge_pack.sample_mask.unsqueeze(0),
        fixed_mask=edge_pack.fixed_mask.unsqueeze(0),
        fixed_values=edge_pack.fixed_values.unsqueeze(0),
        show_progress=False,
    )
    fixed_edge_mask = edge_pack.fixed_mask.unsqueeze(0).expand_as(edge_out)
    assert torch.all(edge_out[fixed_edge_mask] == edge_pack.fixed_values.unsqueeze(0)[fixed_edge_mask])
