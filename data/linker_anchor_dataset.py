from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from molgraph import encode_mol


def normalize_smiles_r(smiles_r: str) -> str:
    """Convert linker placeholders from [R1]/[R2] to RDKit-compatible [*:1]/[*:2]."""
    return smiles_r.replace("[R1]", "[*:1]").replace("[R2]", "[*:2]")


def _sanitize_with_fallback(mol: Chem.Mol) -> bool:
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        try:
            flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(mol, sanitizeOps=flags)
            return True
        except Exception:
            return False


def extract_anchor_indices_from_smiles_pair(
    smiles: str,
    smiles_r: str,
) -> Tuple[Optional[Tuple[int, int]], str]:
    """Extract anchor atom indices (left, right) on plain linker from (Smiles, Smiles_R).

    Returns:
      ((anchor_l, anchor_r), "ok") on success, else (None, reason)
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "plain_parse"

    mol_r = Chem.MolFromSmiles(normalize_smiles_r(smiles_r))
    if mol_r is None:
        return None, "smiles_r_parse"

    dummy_l = next((a for a in mol_r.GetAtoms() if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == 1), None)
    dummy_r = next((a for a in mol_r.GetAtoms() if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == 2), None)
    if dummy_l is None or dummy_r is None:
        return None, "missing_r1_r2"

    neigh_l = dummy_l.GetNeighbors()
    neigh_r = dummy_r.GetNeighbors()
    if len(neigh_l) != 1 or len(neigh_r) != 1:
        return None, "dummy_degree_not_one"

    anchor_l_r = neigh_l[0].GetIdx()
    anchor_r_r = neigh_r[0].GetIdx()
    if anchor_l_r == anchor_r_r:
        return None, "same_anchor_atom"

    # Mark anchor neighbors and remove dummies to obtain the plain linker core.
    rw = Chem.RWMol(mol_r)
    rw.GetAtomWithIdx(anchor_l_r).SetAtomMapNum(101)
    rw.GetAtomWithIdx(anchor_r_r).SetAtomMapNum(102)
    for idx in sorted([dummy_l.GetIdx(), dummy_r.GetIdx()], reverse=True):
        rw.RemoveAtom(idx)
    core = rw.GetMol()
    if not _sanitize_with_fallback(core):
        return None, "core_sanitize_failed"

    core_l = None
    core_r = None
    for atom in core.GetAtoms():
        if atom.GetAtomMapNum() == 101:
            core_l = atom.GetIdx()
        elif atom.GetAtomMapNum() == 102:
            core_r = atom.GetIdx()
    if core_l is None or core_r is None:
        return None, "anchor_maps_lost"

    matches = mol.GetSubstructMatches(core, uniquify=False)
    if not matches:
        return None, "no_substruct_match"

    # Deterministic choice for symmetry cases.
    match = sorted(matches)[0]
    return (int(match[core_l]), int(match[core_r])), "ok"


@dataclass
class LinkerAnchorSample:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    smiles: str
    smiles_r: str
    compound_id: str


class LinkerAnchorDataset(Dataset):
    """Supervised dataset for anchor prediction from linker.csv."""

    def __init__(
        self,
        csv_path: str,
        max_samples: Optional[int] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.samples: list[LinkerAnchorSample] = []
        self.reason_counts: Dict[str, int] = {}
        self._build(max_samples=max_samples)

    def _inc_reason(self, reason: str) -> None:
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def _build(self, max_samples: Optional[int]) -> None:
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles = (row.get("Smiles") or "").strip()
                smiles_r = (row.get("Smiles_R") or "").strip()
                compound_id = str(row.get("Compound ID") or "").strip()

                anchors, reason = extract_anchor_indices_from_smiles_pair(smiles, smiles_r)
                if anchors is None:
                    self._inc_reason(reason)
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    self._inc_reason("plain_parse")
                    continue

                graph = encode_mol(mol, explicit_anchor_indices=anchors)
                sample = LinkerAnchorSample(
                    x=graph["x"],
                    edge_index=graph["edge_index"],
                    edge_attr=graph["edge_attr"],
                    y=graph["node_type"].long(),
                    smiles=smiles,
                    smiles_r=smiles_r,
                    compound_id=compound_id,
                )
                self.samples.append(sample)

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> LinkerAnchorSample:
        return self.samples[idx]


def collate_linker_anchor_samples(samples: Sequence[LinkerAnchorSample]) -> Dict[str, torch.Tensor]:
    """Collate variable-size molecular graphs into a single batch tensor dict."""

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
