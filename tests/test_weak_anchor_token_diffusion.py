from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from rdkit import Chem

from build_weak_anchor_dataset import MolRecord, process_pair
from data.weak_anchor_token_dataset import WeakAnchorTokenDataset, WeakAnchorTokenPTDataset, serialize_weak_anchor_token_dataset
from data.weak_anchor_token_diffusion import collate_weak_anchor_token_diffusion_batch
from diffusion.ddpm import DDPM
from models.fragment_conditioned_denoiser import FragmentConditionedTokenDenoiser
from sampling.token_linker_codec import tokenize_anchored_linker


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


def _write_token_resources(tmp_path: Path, tokens: list[str], dim: int = 8) -> tuple[Path, Path]:
    vocab_json = tmp_path / "token_vocab.json"
    emb_pt = tmp_path / "token_embeddings.pt"
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    vocab_json.write_text(
        json.dumps(
            {
                "tokens": tokens,
                "token_to_id": token_to_id,
                "id_to_token": tokens,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    embeddings = torch.randn((len(tokens), dim), dtype=torch.float32)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    torch.save(
        {
            "embeddings": embeddings,
            "token_to_id": token_to_id,
            "id_to_token": tokens,
            "embedding_dim": dim,
        },
        emb_pt,
    )
    return vocab_json, emb_pt


def _build_weak_anchor_csv(tmp_path: Path) -> Path:
    accepted, rejection = process_pair(
        _make_record("c1ccccc1CCOCCNc2ccccc2", "p1"),
        _make_record("CCOCCN", "l1"),
        min_fragment_heavy_atoms=1,
        min_linker_heavy_atoms=1,
        min_anchor_graph_distance=1,
        min_linker_ratio_pct=0.0,
        max_linker_ratio_pct=100.0,
    )
    assert accepted is not None, rejection
    accepted["sample_id"] = "1"

    csv_path = tmp_path / "weak_anchor.csv"
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
        writer.writerow({k: str(v) for k, v in accepted.items() if k in fieldnames})
    return csv_path


def test_weak_anchor_token_dataset_and_collate(tmp_path: Path) -> None:
    csv_path = _build_weak_anchor_csv(tmp_path)
    row = next(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    tokenized = tokenize_anchored_linker(row["anchored_linker_smiles"])
    vocab_json, emb_pt = _write_token_resources(tmp_path, ["<PAD>"] + list(tokenized["oriented_token_smiles"]), dim=8)

    ds = WeakAnchorTokenDataset(
        csv_path=str(csv_path),
        token_vocab_json=str(vocab_json),
        token_embeddings_pt=str(emb_pt),
        pad_to_length=23,
        pad_token="<PAD>",
        reject_overlength=True,
        learn_pad_positions=True,
    )
    assert len(ds) == 1
    assert ds[0]["linker_token_ids"].shape[0] == 23
    assert ds[0]["linker_length"] == len(tokenized["oriented_token_smiles"])

    pt_path = tmp_path / "weak_anchor_token.pt"
    serialize_weak_anchor_token_dataset(ds, str(pt_path))
    pt_ds = WeakAnchorTokenPTDataset(str(pt_path))
    batch = collate_weak_anchor_token_diffusion_batch([pt_ds[0], pt_ds[0]])

    assert batch["linker_token"]["x_start"].shape[0] == 2
    assert batch["linker_token"]["x_start"].shape[1] == 23
    assert batch["linker_token"]["x_start"].shape[-1] == 8
    assert batch["left_graph"]["x"].ndim == 2
    assert batch["right_graph"]["x"].ndim == 2
    assert batch["linker_token"]["sample_mask"].dtype == torch.bool
    assert bool(batch["linker_token"]["learn_pad_positions"]) is True
    assert int(batch["linker_token"]["sample_mask"][0].sum().item()) == 23
    assert not bool(batch["linker_token"]["fixed_mask"][0].any().item())


def test_fragment_conditioned_token_denoiser_forward_and_loss(tmp_path: Path) -> None:
    csv_path = _build_weak_anchor_csv(tmp_path)
    row = next(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    tokenized = tokenize_anchored_linker(row["anchored_linker_smiles"])
    vocab_json, emb_pt = _write_token_resources(tmp_path, ["<PAD>"] + list(tokenized["oriented_token_smiles"]), dim=8)

    ds = WeakAnchorTokenDataset(
        csv_path=str(csv_path),
        token_vocab_json=str(vocab_json),
        token_embeddings_pt=str(emb_pt),
        pad_to_length=23,
        pad_token="<PAD>",
        reject_overlength=True,
        learn_pad_positions=True,
    )
    batch = collate_weak_anchor_token_diffusion_batch([ds[0], ds[0]])

    model = FragmentConditionedTokenDenoiser(
        embed_dim=8,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        condition_dropout=0.0,
    )
    out = model(
        batch["linker_token"]["x_start"],
        torch.tensor([2, 2], dtype=torch.long),
        left_graph=batch["left_graph"],
        right_graph=batch["right_graph"],
        token_mask=batch["linker_token"]["sample_mask"],
    )
    assert out.shape == batch["linker_token"]["x_start"].shape

    ddpm = DDPM(model=model, timesteps=8)
    loss = ddpm.p_losses(
        x_start=batch["linker_token"]["x_start"],
        t=torch.tensor([2, 2], dtype=torch.long),
        sample_mask=batch["linker_token"]["sample_mask"],
        fixed_mask=batch["linker_token"]["fixed_mask"],
        fixed_values=batch["linker_token"]["fixed_values"],
        loss_mask=batch["linker_token"]["loss_mask"],
        model_kwargs={
            "left_graph": batch["left_graph"],
            "right_graph": batch["right_graph"],
            "token_mask": batch["linker_token"]["sample_mask"],
        },
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0
