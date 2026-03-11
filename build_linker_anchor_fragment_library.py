from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem


def parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean: {text}")


def find_smiles_column(fieldnames: Sequence[str]) -> str:
    for name in ["Smiles", "SMILES", "smiles"]:
        if name in fieldnames:
            return name
    for name in fieldnames:
        if "smiles" in name.lower() and "smiles_r" not in name.lower():
            return name
    raise KeyError("cannot find smiles column")


def find_smiles_r_column(fieldnames: Sequence[str]) -> Optional[str]:
    for name in ["Smiles_R", "smiles_r", "SMILES_R"]:
        if name in fieldnames:
            return name
    for name in fieldnames:
        low = name.lower()
        if "smiles" in low and "_r" in low:
            return name
    return None


def find_id_column(fieldnames: Sequence[str]) -> str:
    for name in ["Compound ID", "compound_id", "id", "ID"]:
        if name in fieldnames:
            return name
    return fieldnames[0]


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def canonical_token_smiles(mol: Chem.Mol) -> str:
    clone = Chem.Mol(mol)
    for atom in clone.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
    return canonical_smiles(clone)


def sanitize_copy(mol: Chem.Mol) -> Optional[Chem.Mol]:
    clone = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(clone)
        return clone
    except Exception:
        try:
            flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(clone, sanitizeOps=flags)
            return clone
        except Exception:
            return None


def heavy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _remove_dummy_atoms(mol: Chem.Mol) -> Tuple[Chem.Mol, Dict[int, int]]:
    rw = Chem.RWMol(mol)
    old_to_new: Dict[int, int] = {idx: idx for idx in range(mol.GetNumAtoms())}
    to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    for old_idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(old_idx)
        for key, value in list(old_to_new.items()):
            if value == old_idx:
                del old_to_new[key]
            elif value > old_idx:
                old_to_new[key] = value - 1
    return rw.GetMol(), old_to_new


def infer_anchor_pairs_from_smiles_r(linker_mol: Chem.Mol, smiles_r: str) -> Tuple[Optional[List[Tuple[int, int]]], Optional[str]]:
    text = (smiles_r or "").strip()
    if "[R1]" not in text or "[R2]" not in text:
        return None, "MISSING_R_LABELS"

    anchored_smiles = text.replace("[R1]", "[*:1]").replace("[R2]", "[*:2]")
    anchored_mol = Chem.MolFromSmiles(anchored_smiles)
    if anchored_mol is None:
        return None, "SMILES_R_PARSE_FAIL"

    dummy_1 = None
    dummy_2 = None
    for atom in anchored_mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        if atom.GetAtomMapNum() == 1:
            dummy_1 = atom
        elif atom.GetAtomMapNum() == 2:
            dummy_2 = atom
    if dummy_1 is None or dummy_2 is None:
        return None, "MISSING_DUMMY_MAP"
    if dummy_1.GetDegree() != 1 or dummy_2.GetDegree() != 1:
        return None, "BAD_DUMMY_DEGREE"

    neigh_1_old = int(dummy_1.GetNeighbors()[0].GetIdx())
    neigh_2_old = int(dummy_2.GetNeighbors()[0].GetIdx())
    core_mol, old_to_new = _remove_dummy_atoms(anchored_mol)

    core_sane = sanitize_copy(core_mol)
    if core_sane is None:
        return None, "CORE_SANITIZE_FAIL"
    if neigh_1_old not in old_to_new or neigh_2_old not in old_to_new:
        return None, "CORE_ANCHOR_MAP_FAIL"
    neigh_1_core = int(old_to_new[neigh_1_old])
    neigh_2_core = int(old_to_new[neigh_2_old])

    matches = linker_mol.GetSubstructMatches(core_sane, uniquify=True)
    if len(matches) == 0:
        return None, "NO_MATCH_TO_LINKER"

    pair_to_match: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    for match in matches:
        a1 = int(match[neigh_1_core])
        a2 = int(match[neigh_2_core])
        if a1 == a2:
            continue
        key = (a1, a2)
        if key not in pair_to_match:
            pair_to_match[key] = match
    if not pair_to_match:
        return None, "SAME_ANCHOR"
    pairs = sorted(pair_to_match.keys())
    return pairs, None


def add_anchor_endpoint_dummies(mol: Chem.Mol, anchor_1: int, anchor_2: int) -> Chem.Mol:
    """Return mol augmented to [*:1]-mol-[*:2] using given anchor atoms."""
    rw = Chem.RWMol(mol)

    d1 = Chem.Atom(0)
    d1.SetAtomMapNum(1)
    d1_idx = rw.AddAtom(d1)
    rw.AddBond(int(anchor_1), int(d1_idx), Chem.BondType.SINGLE)

    d2 = Chem.Atom(0)
    d2.SetAtomMapNum(2)
    d2_idx = rw.AddAtom(d2)
    rw.AddBond(int(anchor_2), int(d2_idx), Chem.BondType.SINGLE)
    return rw.GetMol()


def anchor_path_single_bonds(
    mol: Chem.Mol,
    anchor_1: int,
    anchor_2: int,
    include_ring_single_bonds: bool = False,
) -> Tuple[List[int], List[int]]:
    path = [int(x) for x in Chem.GetShortestPath(mol, int(anchor_1), int(anchor_2))]
    if len(path) <= 1:
        return path, []

    cut_bond_indices: List[int] = []
    for i in range(len(path) - 1):
        a = int(path[i])
        b = int(path[i + 1])
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if (not include_ring_single_bonds) and bond.IsInRing():
            continue
        cut_bond_indices.append(int(bond.GetIdx()))
    return path, cut_bond_indices


def _convert_dummy_isotope_to_map(mol: Chem.Mol) -> Chem.Mol:
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        isotope = int(atom.GetIsotope())
        if isotope > 0:
            atom.SetAtomMapNum(isotope)
            atom.SetIsotope(0)
    return out


def fragment_on_path_single_bonds(mol: Chem.Mol, cut_bond_indices: Sequence[int], dummy_label_start: int = 3) -> Chem.Mol:
    if not cut_bond_indices:
        return Chem.Mol(mol)
    labels = [(dummy_label_start + i, dummy_label_start + i) for i in range(len(cut_bond_indices))]
    frag = Chem.FragmentOnBonds(
        mol,
        bondIndices=[int(idx) for idx in cut_bond_indices],
        addDummies=True,
        dummyLabels=labels,
    )
    return _convert_dummy_isotope_to_map(frag)


def ordered_fragment_tokens(fragmented: Chem.Mol, path: Sequence[int]) -> Optional[List[Dict[str, object]]]:
    atom_frags = list(Chem.GetMolFrags(fragmented, asMols=False, sanitizeFrags=False))
    mol_frags = list(Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False))
    if len(atom_frags) != len(mol_frags):
        return None

    items: List[Dict[str, object]] = []
    for idx, (atom_ids, frag) in enumerate(zip(atom_frags, mol_frags)):
        sane = sanitize_copy(frag)
        if sane is None:
            return None
        atom_set = set(int(x) for x in atom_ids)
        path_positions = [pos for pos, atom_idx in enumerate(path) if int(atom_idx) in atom_set]
        if path_positions:
            span_start = min(path_positions)
            span_end = max(path_positions)
        else:
            span_start = 10**9
            span_end = 10**9
        token_mapped = canonical_smiles(sane)
        token_vocab = canonical_token_smiles(sane)
        num_dummies = sum(1 for atom in sane.GetAtoms() if atom.GetAtomicNum() == 0)
        items.append(
            {
                "frag_index_raw": idx,
                "token_smiles_with_maps": token_mapped,
                "token_smiles": token_vocab,
                "token_heavy_atoms": heavy_atom_count(sane),
                "num_attachment_dummies": num_dummies,
                "path_span_start": span_start,
                "path_span_end": span_end,
            }
        )

    items.sort(
        key=lambda x: (
            int(x["path_span_start"]),
            int(x["path_span_end"]),
            str(x["token_smiles"]),
            str(x["token_smiles_with_maps"]),
            int(x["frag_index_raw"]),
        )
    )
    for token_index, item in enumerate(items):
        item["token_index"] = token_index
    return items


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build linker fragment library by anchor-guided path cuts")
    parser.add_argument("--in_csv", type=str, default="data/csv/linker.csv")
    parser.add_argument("--tokenized_csv", type=str, default="data/processed/linker_anchor_tokenized.csv")
    parser.add_argument("--instances_csv", type=str, default="data/processed/linker_anchor_fragment_instances.csv")
    parser.add_argument("--library_csv", type=str, default="data/processed/linker_anchor_fragment_library.csv")
    parser.add_argument("--rej_csv", type=str, default="data/processed/linker_anchor_fragment_rejections.csv")
    parser.add_argument("--summary_json", type=str, default="data/processed/linker_anchor_fragment_summary.json")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--dedupe_smiles", type=parse_bool, default=False)
    parser.add_argument("--include_ring_single_bonds", type=parse_bool, default=False)
    parser.add_argument("--min_internal_chain_nodes", type=int, default=0)
    parser.add_argument("--min_num_cuts", type=int, default=0)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"input csv not found: {in_csv}")

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise RuntimeError("input csv has no header")

    id_col = find_id_column(fieldnames)
    smiles_col = find_smiles_column(fieldnames)
    smiles_r_col = find_smiles_r_column(fieldnames)
    if smiles_r_col is None:
        raise RuntimeError("cannot find Smiles_R column for anchor inference")

    tokenized_rows: List[Dict[str, str]] = []
    instance_rows: List[Dict[str, str]] = []
    rejected_rows: List[Dict[str, str]] = []

    token_counter: Counter[str] = Counter()
    token_linker_ids: Dict[str, set[str]] = defaultdict(set)
    path_len_counter: Counter[int] = Counter()
    num_cut_counter: Counter[int] = Counter()
    num_frag_counter: Counter[int] = Counter()
    rejection_counter: Counter[str] = Counter()
    anchor_candidate_counter: Counter[int] = Counter()
    rows_with_multi_anchor_pairs = 0
    total_anchor_pair_candidates = 0

    seen_smiles: set[str] = set()
    total = 0

    for row in iter_rows(in_csv):
        total += 1
        if args.max_rows is not None and total > args.max_rows:
            break

        linker_id = str(row.get(id_col, "")).strip()
        smiles = str(row.get(smiles_col, "")).strip()
        smiles_r = str(row.get(smiles_r_col, "")).strip()

        if not smiles:
            reason = "EMPTY_SMILES"
            rejected_rows.append(
                {
                    "linker_id": linker_id,
                    "linker_smiles": smiles,
                    "smiles_r": smiles_r,
                    "rejection_reason": reason,
                }
            )
            rejection_counter[reason] += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            reason = "SMILES_PARSE_FAIL"
            rejected_rows.append(
                {
                    "linker_id": linker_id,
                    "linker_smiles": smiles,
                    "smiles_r": smiles_r,
                    "rejection_reason": reason,
                }
            )
            rejection_counter[reason] += 1
            continue

        linker_smiles = canonical_smiles(mol)
        if args.dedupe_smiles:
            if linker_smiles in seen_smiles:
                continue
            seen_smiles.add(linker_smiles)

        if len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)) != 1:
            reason = "MULTI_COMPONENT"
            rejected_rows.append(
                {
                    "linker_id": linker_id,
                    "linker_smiles": linker_smiles,
                    "smiles_r": smiles_r,
                    "rejection_reason": reason,
                }
            )
            rejection_counter[reason] += 1
            continue

        anchor_pairs, reason = infer_anchor_pairs_from_smiles_r(mol, smiles_r=smiles_r)
        if anchor_pairs is None or len(anchor_pairs) == 0:
            rejected_rows.append(
                {
                    "linker_id": linker_id,
                    "linker_smiles": linker_smiles,
                    "smiles_r": smiles_r,
                    "rejection_reason": str(reason),
                }
            )
            rejection_counter[str(reason)] += 1
            continue
        anchor_candidate_counter[len(anchor_pairs)] += 1
        total_anchor_pair_candidates += len(anchor_pairs)
        if len(anchor_pairs) > 1:
            rows_with_multi_anchor_pairs += 1
        variant_payloads: List[Dict[str, object]] = []
        for pair_rank, (anchor_1, anchor_2) in enumerate(anchor_pairs):
            path, cut_bond_indices = anchor_path_single_bonds(
                mol,
                anchor_1=int(anchor_1),
                anchor_2=int(anchor_2),
                include_ring_single_bonds=bool(args.include_ring_single_bonds),
            )
            if len(path) <= 1:
                reason = "PATH_TOO_SHORT"
                rejected_rows.append(
                    {
                        "linker_id": linker_id,
                        "linker_smiles": linker_smiles,
                        "smiles_r": smiles_r,
                        "rejection_reason": reason,
                    }
                )
                rejection_counter[reason] += 1
                continue

            chain_nodes_including_anchors = len(path)
            chain_nodes_internal = max(0, chain_nodes_including_anchors - 2)
            if chain_nodes_internal < int(args.min_internal_chain_nodes):
                reason = "CHAIN_TOO_SHORT"
                rejected_rows.append(
                    {
                        "linker_id": linker_id,
                        "linker_smiles": linker_smiles,
                        "smiles_r": smiles_r,
                        "rejection_reason": reason,
                    }
                )
                rejection_counter[reason] += 1
                continue
            if len(cut_bond_indices) < int(args.min_num_cuts):
                reason = "TOO_FEW_CUTS"
                rejected_rows.append(
                    {
                        "linker_id": linker_id,
                        "linker_smiles": linker_smiles,
                        "smiles_r": smiles_r,
                        "rejection_reason": reason,
                    }
                )
                rejection_counter[reason] += 1
                continue

            anchored_mol = add_anchor_endpoint_dummies(mol, anchor_1=int(anchor_1), anchor_2=int(anchor_2))
            anchored_mol = sanitize_copy(anchored_mol)
            if anchored_mol is None:
                reason = "ANCHORED_SANITIZE_FAIL"
                rejected_rows.append(
                    {
                        "linker_id": linker_id,
                        "linker_smiles": linker_smiles,
                        "smiles_r": smiles_r,
                        "rejection_reason": reason,
                    }
                )
                rejection_counter[reason] += 1
                continue

            fragmented = fragment_on_path_single_bonds(
                anchored_mol,
                cut_bond_indices=cut_bond_indices,
                dummy_label_start=3,
            )
            tokens = ordered_fragment_tokens(fragmented, path=path)
            if tokens is None or len(tokens) == 0:
                reason = "TOKENIZE_FAIL"
                rejected_rows.append(
                    {
                        "linker_id": linker_id,
                        "linker_smiles": linker_smiles,
                        "smiles_r": smiles_r,
                        "rejection_reason": reason,
                    }
                )
                rejection_counter[reason] += 1
                continue

            token_list = [str(item["token_smiles"]) for item in tokens]
            token_with_maps_list = [str(item["token_smiles_with_maps"]) for item in tokens]
            anchored_linker_smiles = canonical_smiles(anchored_mol)
            sample_id = f"{linker_id}#{pair_rank + 1}"

            variant_payloads.append(
                {
                    "sample_id": sample_id,
                    "anchor_pair_rank": pair_rank + 1,
                    "anchor_1": int(anchor_1),
                    "anchor_2": int(anchor_2),
                    "chain_nodes_including_anchors": chain_nodes_including_anchors,
                    "chain_nodes_internal": chain_nodes_internal,
                    "path": path,
                    "cut_bond_indices": cut_bond_indices,
                    "token_list": token_list,
                    "token_with_maps_list": token_with_maps_list,
                    "tokens": tokens,
                    "anchored_linker_smiles": anchored_linker_smiles,
                }
            )

        accepted_for_this_linker = len(variant_payloads)
        if accepted_for_this_linker == 0:
            # no additional row-level reject here; pair-level reasons already logged.
            pass
        else:
            sample_weight = 1.0 / float(accepted_for_this_linker)
            for payload in variant_payloads:
                path = [int(x) for x in payload["path"]]  # type: ignore[index]
                cut_bond_indices = [int(x) for x in payload["cut_bond_indices"]]  # type: ignore[index]
                token_list = [str(x) for x in payload["token_list"]]  # type: ignore[index]
                token_with_maps_list = [str(x) for x in payload["token_with_maps_list"]]  # type: ignore[index]
                tokens = payload["tokens"]  # type: ignore[index]
                sample_id = str(payload["sample_id"])  # type: ignore[index]
                pair_rank = int(payload["anchor_pair_rank"])  # type: ignore[index]
                anchor_1 = int(payload["anchor_1"])  # type: ignore[index]
                anchor_2 = int(payload["anchor_2"])  # type: ignore[index]
                chain_nodes_including_anchors = int(payload["chain_nodes_including_anchors"])  # type: ignore[index]
                chain_nodes_internal = int(payload["chain_nodes_internal"])  # type: ignore[index]
                anchored_linker_smiles = str(payload["anchored_linker_smiles"])  # type: ignore[index]

                tokenized_rows.append(
                    {
                        "sample_id": sample_id,
                        "linker_id": linker_id,
                        "anchor_pair_rank": str(pair_rank),
                        "num_anchor_pair_candidates": str(len(anchor_pairs)),
                        "num_anchor_pair_accepted": str(accepted_for_this_linker),
                        "sample_weight": f"{sample_weight:.8f}",
                        "linker_smiles": linker_smiles,
                        "anchored_linker_smiles": anchored_linker_smiles,
                        "smiles_r": smiles_r,
                        "anchor1_atom_idx": str(anchor_1),
                        "anchor2_atom_idx": str(anchor_2),
                        "anchor_graph_distance": str(len(path) - 1),
                        "chain_nodes_including_anchors": str(chain_nodes_including_anchors),
                        "chain_nodes_internal": str(chain_nodes_internal),
                        "anchor_path_atom_indices": "-".join(str(int(x)) for x in path),
                        "anchor_path_single_bond_cut_indices": "-".join(str(int(x)) for x in cut_bond_indices),
                        "num_cuts": str(len(cut_bond_indices)),
                        "num_fragments": str(len(tokens)),
                        "token_smiles_list_json": json.dumps(token_list, ensure_ascii=False),
                        "token_smiles_with_maps_list_json": json.dumps(token_with_maps_list, ensure_ascii=False),
                    }
                )

                path_len_counter[len(path)] += 1
                num_cut_counter[len(cut_bond_indices)] += 1
                num_frag_counter[len(tokens)] += 1
                for item in tokens:
                    token = str(item["token_smiles"])
                    token_with_maps = str(item["token_smiles_with_maps"])
                    instance_rows.append(
                        {
                            "sample_id": sample_id,
                            "linker_id": linker_id,
                            "anchor_pair_rank": str(pair_rank),
                            "num_anchor_pair_candidates": str(len(anchor_pairs)),
                            "num_anchor_pair_accepted": str(accepted_for_this_linker),
                            "sample_weight": f"{sample_weight:.8f}",
                            "linker_smiles": linker_smiles,
                            "anchored_linker_smiles": anchored_linker_smiles,
                            "anchor1_atom_idx": str(anchor_1),
                            "anchor2_atom_idx": str(anchor_2),
                            "chain_nodes_including_anchors": str(chain_nodes_including_anchors),
                            "chain_nodes_internal": str(chain_nodes_internal),
                            "token_index": str(int(item["token_index"])),
                            "token_smiles": token,
                            "token_smiles_with_maps": token_with_maps,
                            "token_heavy_atoms": str(int(item["token_heavy_atoms"])),
                            "num_attachment_dummies": str(int(item["num_attachment_dummies"])),
                            "path_span_start": str(int(item["path_span_start"])),
                            "path_span_end": str(int(item["path_span_end"])),
                        }
                    )
                    token_counter[token] += 1
                    token_linker_ids[token].add(linker_id)

        if total == 1 or total % 500 == 0:
            print(
                f"[progress] rows={total} accepted={len(tokenized_rows)} rejected={len(rejected_rows)} "
                f"unique_tokens={len(token_counter)}",
                flush=True,
            )

    tokenized_csv = Path(args.tokenized_csv)
    instances_csv = Path(args.instances_csv)
    library_csv = Path(args.library_csv)
    rej_csv = Path(args.rej_csv)
    summary_json = Path(args.summary_json)
    for path in [tokenized_csv, instances_csv, library_csv, rej_csv, summary_json]:
        path.parent.mkdir(parents=True, exist_ok=True)

    with tokenized_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "linker_id",
                "anchor_pair_rank",
                "num_anchor_pair_candidates",
                "num_anchor_pair_accepted",
                "sample_weight",
                "linker_smiles",
                "anchored_linker_smiles",
                "smiles_r",
                "anchor1_atom_idx",
                "anchor2_atom_idx",
                "anchor_graph_distance",
                "chain_nodes_including_anchors",
                "chain_nodes_internal",
                "anchor_path_atom_indices",
                "anchor_path_single_bond_cut_indices",
                "num_cuts",
                "num_fragments",
                "token_smiles_list_json",
                "token_smiles_with_maps_list_json",
            ],
        )
        writer.writeheader()
        writer.writerows(tokenized_rows)

    with instances_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "linker_id",
                "anchor_pair_rank",
                "num_anchor_pair_candidates",
                "num_anchor_pair_accepted",
                "sample_weight",
                "linker_smiles",
                "anchored_linker_smiles",
                "anchor1_atom_idx",
                "anchor2_atom_idx",
                "chain_nodes_including_anchors",
                "chain_nodes_internal",
                "token_index",
                "token_smiles",
                "token_smiles_with_maps",
                "token_heavy_atoms",
                "num_attachment_dummies",
                "path_span_start",
                "path_span_end",
            ],
        )
        writer.writeheader()
        writer.writerows(instance_rows)

    library_rows = []
    for token, freq in token_counter.most_common():
        library_rows.append(
            {
                "token_smiles": token,
                "frequency": str(int(freq)),
                "num_unique_linkers": str(len(token_linker_ids[token])),
            }
        )
    with library_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["token_smiles", "frequency", "num_unique_linkers"],
        )
        writer.writeheader()
        writer.writerows(library_rows)

    with rej_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["linker_id", "linker_smiles", "smiles_r", "rejection_reason"],
        )
        writer.writeheader()
        writer.writerows(rejected_rows)

    summary = {
        "total_rows": total,
        "accepted": len(tokenized_rows),
        "rejected": len(rejected_rows),
        "accepted_unique_linkers": len({row["linker_id"] for row in tokenized_rows}),
        "rows_with_multi_anchor_pairs": int(rows_with_multi_anchor_pairs),
        "total_anchor_pair_candidates": int(total_anchor_pair_candidates),
        "anchor_candidate_count_distribution": {
            str(k): int(v) for k, v in sorted(anchor_candidate_counter.items())
        },
        "sum_sample_weights": round(sum(float(row["sample_weight"]) for row in tokenized_rows), 6),
        "instances": len(instance_rows),
        "unique_tokens": len(token_counter),
        "dedupe_smiles": bool(args.dedupe_smiles),
        "include_ring_single_bonds": bool(args.include_ring_single_bonds),
        "min_internal_chain_nodes": int(args.min_internal_chain_nodes),
        "min_num_cuts": int(args.min_num_cuts),
        "rejection_counts": dict(rejection_counter),
        "path_len_counts": {str(k): int(v) for k, v in sorted(path_len_counter.items())},
        "num_cut_counts": {str(k): int(v) for k, v in sorted(num_cut_counter.items())},
        "num_fragment_counts": {str(k): int(v) for k, v in sorted(num_frag_counter.items())},
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[done] total={total} accepted={len(tokenized_rows)} rejected={len(rejected_rows)} "
        f"instances={len(instance_rows)} unique_tokens={len(token_counter)}",
        flush=True,
    )
    print(
        f"[files] tokenized_csv={tokenized_csv} instances_csv={instances_csv} "
        f"library_csv={library_csv} rej_csv={rej_csv} summary_json={summary_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
