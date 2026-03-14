from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from data.anchored_tensor_dataset import (
    GraphTensorBlock,
    collate_graph_tensor_blocks,
    graph_block_from_mol,
    graph_block_to_record,
)
from sampling.token_linker_codec import tokenize_anchored_linker


@dataclass
class WeakAnchorTokenSample:
    left_graph: GraphTensorBlock
    right_graph: GraphTensorBlock
    linker_token_ids: torch.Tensor
    linker_token_embeddings: torch.Tensor
    linker_length: int
    token_smiles: list[str]
    token_smiles_with_maps: list[str]
    oriented_token_smiles: list[str]
    sample_id: str
    protac_id: str
    linker_id: str
    full_protac_smiles: str
    linker_smiles: str
    anchored_linker_smiles: str
    left_fragment_smiles: str
    right_fragment_smiles: str
    anchor_left_atom_idx_in_full: int
    anchor_right_atom_idx_in_full: int
    linker_ratio_pct: float
    left_ratio_pct: float
    right_ratio_pct: float


def _record_to_graph_block(record: Dict[str, Any]) -> GraphTensorBlock:
    return GraphTensorBlock(
        x=record["x"].float(),
        edge_index=record["edge_index"].long(),
        edge_attr=record["edge_attr"].float(),
        node_type=record["node_type"].long(),
        can_add_bond=record["can_add_bond"].bool(),
        degree=record["degree"].long(),
        dummy_mask=record["dummy_mask"].bool(),
        anchor_mask=record["anchor_mask"].bool(),
        pair_mask=record.get("pair_mask"),
    )


def load_token_embedding_resources(
    vocab_json: str | Path,
    embeddings_pt: str | Path,
) -> tuple[list[str], Dict[str, int], torch.Tensor]:
    vocab_path = Path(vocab_json)
    emb_path = Path(embeddings_pt)
    if not vocab_path.exists():
        raise FileNotFoundError(f"token vocab json not found: {vocab_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"token embeddings pt not found: {emb_path}")

    vocab_payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    emb_payload: Dict[str, Any] = torch.load(emb_path, map_location="cpu")

    vocab_tokens = [str(x) for x in vocab_payload.get("tokens", [])]
    token_to_id = {str(k): int(v) for k, v in vocab_payload.get("token_to_id", {}).items()}
    embeddings = emb_payload.get("embeddings")
    if not torch.is_tensor(embeddings):
        raise RuntimeError(f"embeddings payload missing tensor: {emb_path}")
    embeddings = embeddings.detach().cpu().float()

    if len(vocab_tokens) != embeddings.shape[0]:
        raise RuntimeError(
            f"vocab size and embedding rows differ: {len(vocab_tokens)} vs {int(embeddings.shape[0])}"
        )
    if set(vocab_tokens) != set(token_to_id.keys()):
        raise RuntimeError("token list and token_to_id keys differ")
    return vocab_tokens, token_to_id, embeddings


def _ratio_pct(part_atoms: int, full_atoms: int) -> float:
    if full_atoms <= 0:
        return 0.0
    return 100.0 * float(part_atoms) / float(full_atoms)


class WeakAnchorTokenDataset(Dataset):
    """Weak-anchor dataset with linker represented as an oriented token embedding sequence."""

    def __init__(
        self,
        csv_path: str,
        token_vocab_json: str,
        token_embeddings_pt: str,
        max_samples: Optional[int] = None,
        include_ring_single_bonds: bool = False,
        pad_to_length: int = 0,
        pad_token: str = "<PAD>",
        reject_overlength: bool = False,
        learn_pad_positions: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"weak-anchor csv not found: {self.csv_path}")

        vocab_tokens, token_to_id, token_embeddings = load_token_embedding_resources(
            vocab_json=token_vocab_json,
            embeddings_pt=token_embeddings_pt,
        )
        self.vocab_tokens = vocab_tokens
        self.token_to_id = token_to_id
        self.token_embeddings = token_embeddings
        self.embedding_dim = int(token_embeddings.shape[1])
        self.include_ring_single_bonds = bool(include_ring_single_bonds)
        self.pad_to_length = int(pad_to_length)
        self.pad_token = str(pad_token)
        self.reject_overlength = bool(reject_overlength)
        self.learn_pad_positions = bool(learn_pad_positions)
        self.pad_token_id = self.token_to_id.get(self.pad_token, None)
        if self.pad_to_length < 0:
            raise ValueError(f"pad_to_length must be >= 0, got {self.pad_to_length}")
        if self.pad_to_length > 0 and self.pad_token_id is None:
            raise ValueError(
                f"pad_to_length={self.pad_to_length} requested but pad token {self.pad_token!r} is missing from vocab"
            )
        self.samples: list[WeakAnchorTokenSample] = []
        self.reason_counts: Dict[str, int] = {}
        self._build(max_samples=max_samples)

    def _inc_reason(self, reason: str) -> None:
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def _build(self, max_samples: Optional[int]) -> None:
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = str(row.get("sample_id", "")).strip()
                protac_id = str(row.get("protac_id", "")).strip()
                linker_id = str(row.get("linker_id", "")).strip()
                full_protac_smiles = str(row.get("full_protac_smiles", "")).strip()
                linker_smiles = str(row.get("linker_smiles", "")).strip()
                anchored_linker_smiles = str(row.get("anchored_linker_smiles", "")).strip()
                left_fragment_smiles = str(row.get("left_fragment_smiles", "")).strip()
                right_fragment_smiles = str(row.get("right_fragment_smiles", "")).strip()

                if not anchored_linker_smiles or not left_fragment_smiles or not right_fragment_smiles:
                    self._inc_reason("missing_smiles")
                    continue

                mol_left = Chem.MolFromSmiles(left_fragment_smiles)
                if mol_left is None:
                    self._inc_reason("left_fragment_parse")
                    continue
                mol_right = Chem.MolFromSmiles(right_fragment_smiles)
                if mol_right is None:
                    self._inc_reason("right_fragment_parse")
                    continue

                try:
                    tokenized = tokenize_anchored_linker(
                        anchored_linker_smiles,
                        include_ring_single_bonds=self.include_ring_single_bonds,
                    )
                except Exception:
                    self._inc_reason("linker_tokenize")
                    continue

                oriented_tokens = [str(x) for x in tokenized["oriented_token_smiles"]]
                if not oriented_tokens:
                    self._inc_reason("empty_token_sequence")
                    continue

                try:
                    token_ids = torch.tensor([self.token_to_id[token] for token in oriented_tokens], dtype=torch.long)
                except KeyError:
                    self._inc_reason("token_oov")
                    continue

                linker_length = int(token_ids.shape[0])
                if self.pad_to_length > 0:
                    if linker_length > self.pad_to_length and self.reject_overlength:
                        self._inc_reason("token_sequence_too_long")
                        continue
                    if linker_length < self.pad_to_length:
                        assert self.pad_token_id is not None
                        pad_count = self.pad_to_length - linker_length
                        pad_ids = torch.full((pad_count,), fill_value=int(self.pad_token_id), dtype=torch.long)
                        token_ids = torch.cat([token_ids, pad_ids], dim=0)
                token_emb = self.token_embeddings[token_ids].clone()
                left_graph = graph_block_from_mol(mol_left, include_pair_mask=False)
                right_graph = graph_block_from_mol(mol_right, include_pair_mask=False)

                num_atoms_full = int(str(row.get("num_atoms_full", "0") or "0"))
                num_atoms_linker = int(str(row.get("num_atoms_linker", "0") or "0"))
                num_atoms_left = int(str(row.get("num_atoms_left", "0") or "0"))
                num_atoms_right = int(str(row.get("num_atoms_right", "0") or "0"))

                self.samples.append(
                    WeakAnchorTokenSample(
                        left_graph=left_graph,
                        right_graph=right_graph,
                        linker_token_ids=token_ids,
                        linker_token_embeddings=token_emb,
                        linker_length=linker_length,
                        token_smiles=[str(x) for x in tokenized["token_smiles"]],
                        token_smiles_with_maps=[str(x) for x in tokenized["token_smiles_with_maps"]],
                        oriented_token_smiles=oriented_tokens,
                        sample_id=sample_id,
                        protac_id=protac_id,
                        linker_id=linker_id,
                        full_protac_smiles=full_protac_smiles,
                        linker_smiles=linker_smiles,
                        anchored_linker_smiles=anchored_linker_smiles,
                        left_fragment_smiles=left_fragment_smiles,
                        right_fragment_smiles=right_fragment_smiles,
                        anchor_left_atom_idx_in_full=int(row.get("anchor_left_atom_idx_in_full", 0)),
                        anchor_right_atom_idx_in_full=int(row.get("anchor_right_atom_idx_in_full", 0)),
                        linker_ratio_pct=_ratio_pct(num_atoms_linker, num_atoms_full),
                        left_ratio_pct=_ratio_pct(num_atoms_left, num_atoms_full),
                        right_ratio_pct=_ratio_pct(num_atoms_right, num_atoms_full),
                    )
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "left_graph": sample.left_graph,
            "right_graph": sample.right_graph,
            "linker_token_ids": sample.linker_token_ids.clone(),
            "linker_token_embeddings": sample.linker_token_embeddings.clone(),
            "linker_length": int(sample.linker_length),
            "learn_pad_positions": bool(self.learn_pad_positions),
            "token_smiles": list(sample.token_smiles),
            "token_smiles_with_maps": list(sample.token_smiles_with_maps),
            "oriented_token_smiles": list(sample.oriented_token_smiles),
            "sample_id": sample.sample_id,
            "protac_id": sample.protac_id,
            "linker_id": sample.linker_id,
            "full_protac_smiles": sample.full_protac_smiles,
            "linker_smiles": sample.linker_smiles,
            "anchored_linker_smiles": sample.anchored_linker_smiles,
            "left_fragment_smiles": sample.left_fragment_smiles,
            "right_fragment_smiles": sample.right_fragment_smiles,
            "anchor_left_atom_idx_in_full": sample.anchor_left_atom_idx_in_full,
            "anchor_right_atom_idx_in_full": sample.anchor_right_atom_idx_in_full,
            "linker_ratio_pct": sample.linker_ratio_pct,
            "left_ratio_pct": sample.left_ratio_pct,
            "right_ratio_pct": sample.right_ratio_pct,
        }


def serialize_weak_anchor_token_dataset(
    dataset: WeakAnchorTokenDataset,
    out_path: str,
) -> None:
    records = []
    max_token_length = 0
    for sample in dataset.samples:
        max_token_length = max(max_token_length, int(sample.linker_token_ids.shape[0]))
        records.append(
            {
                "left_graph": graph_block_to_record(sample.left_graph, include_pair_mask=False),
                "right_graph": graph_block_to_record(sample.right_graph, include_pair_mask=False),
                "linker_token_ids": sample.linker_token_ids,
                "linker_token_embeddings": sample.linker_token_embeddings,
                "linker_length": int(sample.linker_length),
                "learn_pad_positions": bool(dataset.learn_pad_positions),
                "token_smiles": sample.token_smiles,
                "token_smiles_with_maps": sample.token_smiles_with_maps,
                "oriented_token_smiles": sample.oriented_token_smiles,
                "sample_id": sample.sample_id,
                "protac_id": sample.protac_id,
                "linker_id": sample.linker_id,
                "full_protac_smiles": sample.full_protac_smiles,
                "linker_smiles": sample.linker_smiles,
                "anchored_linker_smiles": sample.anchored_linker_smiles,
                "left_fragment_smiles": sample.left_fragment_smiles,
                "right_fragment_smiles": sample.right_fragment_smiles,
                "anchor_left_atom_idx_in_full": sample.anchor_left_atom_idx_in_full,
                "anchor_right_atom_idx_in_full": sample.anchor_right_atom_idx_in_full,
                "linker_ratio_pct": sample.linker_ratio_pct,
                "left_ratio_pct": sample.left_ratio_pct,
                "right_ratio_pct": sample.right_ratio_pct,
            }
        )

    payload = {
        "records": records,
        "meta": {
            "num_samples": len(records),
            "reason_counts": dict(dataset.reason_counts),
            "token_vocab": list(dataset.vocab_tokens),
            "token_to_id": dict(dataset.token_to_id),
            "token_embeddings": dataset.token_embeddings.clone(),
            "embedding_dim": int(dataset.embedding_dim),
            "max_token_length": int(max_token_length),
            "pad_to_length": int(dataset.pad_to_length),
            "pad_token": dataset.pad_token,
            "pad_token_id": int(dataset.pad_token_id) if dataset.pad_token_id is not None else None,
            "reject_overlength": bool(dataset.reject_overlength),
            "learn_pad_positions": bool(dataset.learn_pad_positions),
            "include_ring_single_bonds": bool(dataset.include_ring_single_bonds),
            "source_format": "weak_anchor_dataset_csv",
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)


class WeakAnchorTokenPTDataset(Dataset):
    def __init__(self, pt_path: str, max_samples: Optional[int] = None) -> None:
        self.pt_path = Path(pt_path)
        if not self.pt_path.exists():
            raise FileNotFoundError(f"token tensor dataset file not found: {self.pt_path}")

        payload: Dict[str, Any] = torch.load(self.pt_path, map_location="cpu")
        records = list(payload.get("records", []))
        if max_samples is not None:
            records = records[:max_samples]
        self.records = records
        self.meta = dict(payload.get("meta", {}))
        self.reason_counts = dict(self.meta.get("reason_counts", {}))
        self.vocab_tokens = [str(x) for x in self.meta.get("token_vocab", [])]
        self.token_to_id = {str(k): int(v) for k, v in self.meta.get("token_to_id", {}).items()}
        token_embeddings = self.meta.get("token_embeddings")
        self.token_embeddings = token_embeddings.float() if torch.is_tensor(token_embeddings) else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        return {
            "left_graph": _record_to_graph_block(record["left_graph"]),
            "right_graph": _record_to_graph_block(record["right_graph"]),
            "linker_token_ids": record["linker_token_ids"].long(),
            "linker_token_embeddings": record["linker_token_embeddings"].float(),
            "linker_length": int(record.get("linker_length", record["linker_token_ids"].shape[0])),
            "learn_pad_positions": bool(record.get("learn_pad_positions", self.meta.get("learn_pad_positions", False))),
            "token_smiles": [str(x) for x in record.get("token_smiles", [])],
            "token_smiles_with_maps": [str(x) for x in record.get("token_smiles_with_maps", [])],
            "oriented_token_smiles": [str(x) for x in record.get("oriented_token_smiles", [])],
            "sample_id": str(record.get("sample_id", "")),
            "protac_id": str(record.get("protac_id", "")),
            "linker_id": str(record.get("linker_id", "")),
            "full_protac_smiles": str(record.get("full_protac_smiles", "")),
            "linker_smiles": str(record.get("linker_smiles", "")),
            "anchored_linker_smiles": str(record.get("anchored_linker_smiles", "")),
            "left_fragment_smiles": str(record.get("left_fragment_smiles", "")),
            "right_fragment_smiles": str(record.get("right_fragment_smiles", "")),
            "anchor_left_atom_idx_in_full": int(record.get("anchor_left_atom_idx_in_full", 0)),
            "anchor_right_atom_idx_in_full": int(record.get("anchor_right_atom_idx_in_full", 0)),
            "linker_ratio_pct": float(record.get("linker_ratio_pct", 0.0)),
            "left_ratio_pct": float(record.get("left_ratio_pct", 0.0)),
            "right_ratio_pct": float(record.get("right_ratio_pct", 0.0)),
        }


def collate_weak_anchor_token_tensor_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("Cannot collate empty token sample list")

    return {
        "left_graph": collate_graph_tensor_blocks([sample["left_graph"] for sample in samples]),
        "right_graph": collate_graph_tensor_blocks([sample["right_graph"] for sample in samples]),
        "linker_token_ids": [sample["linker_token_ids"] for sample in samples],
        "linker_token_embeddings": [sample["linker_token_embeddings"] for sample in samples],
        "linker_length": torch.tensor([int(sample["linker_length"]) for sample in samples], dtype=torch.long),
        "learn_pad_positions": bool(samples[0].get("learn_pad_positions", False)),
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
