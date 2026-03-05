from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from data.linker_anchor_dataset import (
    extract_anchor_indices_from_smiles_pair,
    normalize_smiles_r,
)
from molgraph import allowed_bond_mask, encode_mol


def _sanitize_with_fallback(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(mol, sanitizeOps=flags)
            return mol
        except Exception:
            return None


def attach_anchor_dummies(mol: Chem.Mol, anchor_l: int, anchor_r: int) -> Optional[Chem.Mol]:
    """Attach [*:1] and [*:2] to two anchor atom indices on a plain linker mol."""
    if anchor_l == anchor_r:
        return None
    if anchor_l < 0 or anchor_r < 0 or anchor_l >= mol.GetNumAtoms() or anchor_r >= mol.GetNumAtoms():
        return None

    rw = Chem.RWMol(mol)
    d1 = Chem.Atom(0)
    d1.SetAtomMapNum(1)
    d2 = Chem.Atom(0)
    d2.SetAtomMapNum(2)
    d1_idx = rw.AddAtom(d1)
    d2_idx = rw.AddAtom(d2)
    rw.AddBond(anchor_l, d1_idx, Chem.BondType.SINGLE)
    rw.AddBond(anchor_r, d2_idx, Chem.BondType.SINGLE)
    return _sanitize_with_fallback(rw.GetMol())


def degree_from_bidir_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute undirected degree from doubled COO edge list.

    edge_index is expected to contain both directions for each undirected bond.
    Counting outgoing edges per node yields the undirected degree directly.
    """

    deg = torch.zeros((num_nodes,), dtype=torch.long)
    if edge_index.numel() == 0:
        return deg
    src = edge_index[0].long()
    ones = torch.ones_like(src, dtype=torch.long)
    deg.index_add_(0, src, ones)
    return deg


def pair_add_mask(
    edge_index: torch.Tensor,
    can_add_bond: torch.Tensor,
    include_diagonal: bool = False,
) -> torch.Tensor:
    """Build [N, N] candidate mask for adding new bonds.

    A pair is allowed when:
      - both endpoint nodes still pass valence mask;
      - pair is not an existing bond;
      - i != j unless include_diagonal=True.
    """

    num_nodes = int(can_add_bond.shape[0])
    node_ok = can_add_bond.bool()
    mask = node_ok[:, None] & node_ok[None, :]
    if not include_diagonal:
        mask.fill_diagonal_(False)

    if edge_index.numel() > 0:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        mask[src, dst] = False
    return mask


@dataclass
class AnchoredTensorSample:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_type: torch.Tensor
    can_add_bond: torch.Tensor
    degree: torch.Tensor
    dummy_mask: torch.Tensor
    anchor_mask: torch.Tensor
    pair_mask: Optional[torch.Tensor]
    smiles: str
    smiles_r: str
    anchored_smiles: str
    compound_id: str


@dataclass
class AnchorTrainSample:
    """Minimal sample schema consumed by train_anchor.py."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    smiles: str
    smiles_r: str
    compound_id: str


class AnchoredTensorDataset(Dataset):
    """Dataset of linker graphs with explicit dummy anchors and valence masks."""

    def __init__(
        self,
        csv_path: str,
        max_samples: Optional[int] = None,
        include_pair_mask: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.include_pair_mask = include_pair_mask
        self.samples: list[AnchoredTensorSample] = []
        self.reason_counts: Dict[str, int] = {}
        self._build(max_samples=max_samples)

    def _inc_reason(self, reason: str) -> None:
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def _build(self, max_samples: Optional[int]) -> None:
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles = (row.get("Smiles") or "").strip()
                smiles_r_raw = (row.get("Smiles_R") or "").strip()
                smiles_r = normalize_smiles_r(smiles_r_raw)
                compound_id = str(row.get("Compound ID") or "").strip()

                mol_plain = Chem.MolFromSmiles(smiles)
                if mol_plain is None:
                    self._inc_reason("plain_parse")
                    continue

                mol_anchored = Chem.MolFromSmiles(smiles_r)
                if mol_anchored is None:
                    # Fallback: derive anchor nodes from pair and attach dummy atoms.
                    anchors, reason = extract_anchor_indices_from_smiles_pair(smiles, smiles_r_raw)
                    if anchors is None:
                        self._inc_reason(reason)
                        continue
                    mol_anchored = attach_anchor_dummies(mol_plain, anchors[0], anchors[1])
                    if mol_anchored is None:
                        self._inc_reason("attach_dummy_failed")
                        continue
                else:
                    # Ensure both anchor dummies exist.
                    has_map1 = any(a.GetAtomicNum() == 0 and a.GetAtomMapNum() == 1 for a in mol_anchored.GetAtoms())
                    has_map2 = any(a.GetAtomicNum() == 0 and a.GetAtomMapNum() == 2 for a in mol_anchored.GetAtoms())
                    if not (has_map1 and has_map2):
                        anchors, reason = extract_anchor_indices_from_smiles_pair(smiles, smiles_r_raw)
                        if anchors is None:
                            self._inc_reason(reason)
                            continue
                        mol_anchored = attach_anchor_dummies(mol_plain, anchors[0], anchors[1])
                        if mol_anchored is None:
                            self._inc_reason("attach_dummy_failed")
                            continue

                graph = encode_mol(mol_anchored, include_original_smiles=True)
                x = graph["x"]
                edge_index = graph["edge_index"]
                edge_attr = graph["edge_attr"]
                node_type = graph["node_type"].long()

                atomic_num = x[:, 0].long()
                degree = degree_from_bidir_edge_index(edge_index=edge_index, num_nodes=x.shape[0])
                can_add = allowed_bond_mask(current_degree=degree, atomic_num=atomic_num)
                pair_mask_tensor = pair_add_mask(edge_index=edge_index, can_add_bond=can_add) if self.include_pair_mask else None

                sample = AnchoredTensorSample(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    node_type=node_type,
                    can_add_bond=can_add.bool(),
                    degree=degree,
                    dummy_mask=(atomic_num == 0),
                    anchor_mask=(node_type > 0),
                    pair_mask=pair_mask_tensor,
                    smiles=smiles,
                    smiles_r=smiles_r_raw,
                    anchored_smiles=Chem.MolToSmiles(mol_anchored, canonical=True),
                    compound_id=compound_id,
                )
                self.samples.append(sample)

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> AnchoredTensorSample:
        return self.samples[idx]


def serialize_anchored_tensor_dataset(
    dataset: AnchoredTensorDataset,
    out_path: str,
    include_pair_mask: bool = False,
) -> None:
    """Save dataset samples as a .pt file of plain tensors + metadata."""

    records = []
    for sample in dataset.samples:
        record = {
            "x": sample.x,
            "edge_index": sample.edge_index,
            "edge_attr": sample.edge_attr,
            "node_type": sample.node_type,
            "can_add_bond": sample.can_add_bond,
            "degree": sample.degree,
            "dummy_mask": sample.dummy_mask,
            "anchor_mask": sample.anchor_mask,
            "smiles": sample.smiles,
            "smiles_r": sample.smiles_r,
            "anchored_smiles": sample.anchored_smiles,
            "compound_id": sample.compound_id,
        }
        if include_pair_mask:
            record["pair_mask"] = sample.pair_mask
        records.append(record)

    payload = {
        "records": records,
        "meta": {
            "num_samples": len(records),
            "reason_counts": dataset.reason_counts,
            "feature_spec": {
                "x_channels": ["atomic_num", "formal_charge", "is_aromatic", "num_hs"],
                "edge_attr_channels": ["single", "double", "triple", "aromatic"],
                "node_type": {"0": "normal", "1": "anchor_l", "2": "anchor_r"},
                "can_add_bond": "allowed_bond_mask(current_degree, atomic_num)",
            },
            "include_pair_mask": include_pair_mask,
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)


class AnchoredTensorPTDataset(Dataset):
    """Load serialized anchored tensor dataset (.pt) for training/inference."""

    def __init__(self, pt_path: str, max_samples: Optional[int] = None) -> None:
        self.pt_path = Path(pt_path)
        if not self.pt_path.exists():
            raise FileNotFoundError(f"Tensor dataset file not found: {self.pt_path}")

        payload: Dict[str, Any] = torch.load(self.pt_path, map_location="cpu")
        records = list(payload.get("records", []))
        if max_samples is not None:
            records = records[:max_samples]
        self.records = records
        self.meta = payload.get("meta", {})
        self.reason_counts = dict(self.meta.get("reason_counts", {}))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> AnchorTrainSample:
        record = self.records[idx]
        y = record.get("node_type")
        if y is None:
            raise KeyError("record missing required key: node_type")
        return AnchorTrainSample(
            x=record["x"].float(),
            edge_index=record["edge_index"].long(),
            edge_attr=record["edge_attr"].float(),
            y=y.long(),
            smiles=str(record.get("smiles", "")),
            smiles_r=str(record.get("smiles_r", "")),
            compound_id=str(record.get("compound_id", "")),
        )


def collate_anchor_train_samples(samples: Sequence[AnchorTrainSample]) -> Dict[str, torch.Tensor]:
    """Collate variable-size graph samples into a single training batch."""

    if len(samples) == 0:
        raise ValueError("Cannot collate empty sample list")

    x_list = []
    y_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []
    graph_ptr = [0]
    offset = 0

    for graph_id, sample in enumerate(samples):
        n = sample.x.shape[0]
        x_list.append(sample.x)
        y_list.append(sample.y)
        edge_index_list.append(sample.edge_index + offset)
        edge_attr_list.append(sample.edge_attr)
        batch_list.append(torch.full((n,), graph_id, dtype=torch.long))
        offset += n
        graph_ptr.append(offset)

    return {
        "x": torch.cat(x_list, dim=0),
        "y": torch.cat(y_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "batch": torch.cat(batch_list, dim=0),
        "graph_ptr": torch.tensor(graph_ptr, dtype=torch.long),
    }
