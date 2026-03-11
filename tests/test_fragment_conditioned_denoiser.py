from __future__ import annotations

import csv
from typing import Any

import torch
from rdkit import Chem

from build_weak_anchor_dataset import MolRecord, process_pair
from data.anchored_tensor_dataset import WeakAnchorTensorDataset, WeakAnchorTensorPTDataset, serialize_weak_anchor_tensor_dataset
from data.weak_anchor_diffusion import collate_weak_anchor_diffusion_batch
from diffusion.ddpm import DDPM
from models.fragment_conditioned_denoiser import FragmentConditionedEdgeDenoiser, FragmentConditionedNodeDenoiser


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


def _build_batch(tmp_path):
    rows = []
    for sample_id, (protac, linker, pid, lid) in enumerate(
        [
            ("c1ccccc1CCOCCNc2ccccc2", "CCOCCN", "p1", "l1"),
            ("c1ccccc1CCOCCOCCNc2ccccc2", "CCOCCOCCN", "p2", "l2"),
        ],
        start=1,
    ):
        accepted, rejection = process_pair(
            _make_record(protac, pid),
            _make_record(linker, lid),
            min_fragment_heavy_atoms=1,
            min_linker_heavy_atoms=1,
            min_anchor_graph_distance=1,
            min_linker_ratio_pct=0.0,
            max_linker_ratio_pct=100.0,
        )
        assert accepted is not None, rejection
        accepted["sample_id"] = str(sample_id)
        rows.append({k: str(v) for k, v in accepted.items() if not k.startswith("_")})

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
    csv_path = tmp_path / "weak_anchor.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ds = WeakAnchorTensorDataset(str(csv_path), include_pair_mask=True)
    pt_path = tmp_path / "weak_anchor.pt"
    serialize_weak_anchor_tensor_dataset(ds, str(pt_path), include_pair_mask=True)
    pt_ds = WeakAnchorTensorPTDataset(str(pt_path))
    return collate_weak_anchor_diffusion_batch([pt_ds[0], pt_ds[1]])


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, dict):
            out[key] = _clone_batch(value)
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _batch_with_modified_fragment_features(batch: dict[str, Any]) -> dict[str, Any]:
    out = _clone_batch(batch)
    out["left_graph"]["x"] = torch.zeros_like(out["left_graph"]["x"])
    out["right_graph"]["x"] = torch.full_like(out["right_graph"]["x"], fill_value=3.0)
    return out

def test_fragment_conditioned_denoiser_forward_shape(tmp_path) -> None:
    batch = _build_batch(tmp_path)
    model = FragmentConditionedNodeDenoiser(in_dim=4, hidden_dim=64, num_layers=3)

    out = model(
        batch["linker_node"]["x_start"],
        torch.tensor([3], dtype=torch.long),
        linker_graph=batch["linker_graph"],
        left_graph=batch["left_graph"],
        right_graph=batch["right_graph"],
    )

    assert out.shape == batch["linker_node"]["x_start"].shape


def test_ddpm_p_losses_accepts_model_kwargs(tmp_path) -> None:
    batch = _build_batch(tmp_path)
    model = FragmentConditionedNodeDenoiser(in_dim=4, hidden_dim=64, num_layers=3)
    ddpm = DDPM(model=model, timesteps=8)

    loss = ddpm.p_losses(
        x_start=batch["linker_node"]["x_start"],
        t=torch.tensor([2], dtype=torch.long),
        sample_mask=batch["linker_node"]["sample_mask"],
        fixed_mask=batch["linker_node"]["fixed_mask"],
        fixed_values=batch["linker_node"]["fixed_values"],
        loss_mask=batch["linker_node"]["loss_mask"],
        model_kwargs={
            "linker_graph": batch["linker_graph"],
            "left_graph": batch["left_graph"],
            "right_graph": batch["right_graph"],
        },
    )

    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_fragment_conditioned_edge_denoiser_forward_shape(tmp_path) -> None:
    batch = _build_batch(tmp_path)
    model = FragmentConditionedEdgeDenoiser(node_in_dim=4, edge_in_dim=4, hidden_dim=64, num_layers=3)

    out = model(
        batch["linker_edge"]["x_start"],
        torch.tensor([3], dtype=torch.long),
        linker_graph=batch["linker_graph"],
        left_graph=batch["left_graph"],
        right_graph=batch["right_graph"],
    )

    assert out.shape == batch["linker_edge"]["x_start"].shape


def test_ddpm_p_losses_accepts_edge_model_kwargs(tmp_path) -> None:
    batch = _build_batch(tmp_path)
    model = FragmentConditionedEdgeDenoiser(node_in_dim=4, edge_in_dim=4, hidden_dim=64, num_layers=3)
    ddpm = DDPM(model=model, timesteps=8)

    loss = ddpm.p_losses(
        x_start=batch["linker_edge"]["x_start"],
        t=torch.tensor([2], dtype=torch.long),
        sample_mask=batch["linker_edge"]["sample_mask"],
        fixed_mask=batch["linker_edge"]["fixed_mask"],
        fixed_values=batch["linker_edge"]["fixed_values"],
        loss_mask=batch["linker_edge"]["loss_mask"],
        model_kwargs={
            "linker_graph": batch["linker_graph"],
            "left_graph": batch["left_graph"],
            "right_graph": batch["right_graph"],
        },
    )

    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_node_condition_dropout_train_mode_drops_fragment_context(tmp_path) -> None:
    batch = _build_batch(tmp_path)
    changed = _batch_with_modified_fragment_features(batch)

    model = FragmentConditionedNodeDenoiser(
        in_dim=4,
        hidden_dim=64,
        num_layers=3,
        dropout=0.0,
        condition_dropout=1.0,
    )
    model.train()

    t = torch.tensor([3], dtype=torch.long)
    out_a = model(
        batch["linker_node"]["x_start"],
        t,
        linker_graph=batch["linker_graph"],
        left_graph=batch["left_graph"],
        right_graph=batch["right_graph"],
    )
    out_b = model(
        changed["linker_node"]["x_start"],
        t,
        linker_graph=changed["linker_graph"],
        left_graph=changed["left_graph"],
        right_graph=changed["right_graph"],
    )

    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_edge_condition_dropout_train_mode_drops_fragment_context(tmp_path) -> None:
    batch = _build_batch(tmp_path)
    changed = _batch_with_modified_fragment_features(batch)

    model = FragmentConditionedEdgeDenoiser(
        node_in_dim=4,
        edge_in_dim=4,
        hidden_dim=64,
        num_layers=3,
        dropout=0.0,
        condition_dropout=1.0,
    )
    model.train()

    t = torch.tensor([3], dtype=torch.long)
    out_a = model(
        batch["linker_edge"]["x_start"],
        t,
        linker_graph=batch["linker_graph"],
        left_graph=batch["left_graph"],
        right_graph=batch["right_graph"],
    )
    out_b = model(
        changed["linker_edge"]["x_start"],
        t,
        linker_graph=changed["linker_graph"],
        left_graph=changed["left_graph"],
        right_graph=changed["right_graph"],
    )

    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)
