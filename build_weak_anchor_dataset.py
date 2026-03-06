from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from rdkit import Chem

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised implicitly in the current env
    pd = None


REJECTION_REASONS = [
    "NO_MATCH",
    "MULTI_MATCH",
    "DISCONNECTED_LINKER",
    "BOUNDARY_NE_2",
    "SAME_ANCHOR",
    "LINKER_TOO_SMALL",
    "ANCHOR_DISTANCE_TOO_SHORT",
    "LINKER_RATIO_OUT_OF_RANGE",
    "LINKER_FRAGMENT_FAIL",
    "FRAGMENT_TOO_SMALL",
    "SANITIZE_FAIL",
    "EXCEPTION",
]


@dataclass(frozen=True)
class MolRecord:
    row_id: str
    smiles: str
    mol: Chem.Mol
    canonical_smiles: str
    num_atoms: int
    source_row: Dict[str, Any]


def count_heavy_atoms(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def parse_bool(value: str) -> bool:
    value_l = value.strip().lower()
    if value_l in {"1", "true", "t", "yes", "y"}:
        return True
    if value_l in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_table(path: str) -> Any:
    """Load a CSV as string-only columns to preserve IDs exactly."""

    if pd is not None:
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def infer_columns(df: Any) -> Dict[str, str]:
    """Infer ID and SMILES columns from a table."""

    if pd is not None and isinstance(df, pd.DataFrame):
        columns = list(df.columns)
    else:
        columns = list(df[0].keys()) if df else []
    if not columns:
        raise ValueError("CSV has no columns")

    if pd is not None and isinstance(df, pd.DataFrame):
        id_col = "Compound ID" if "Compound ID" in df.columns else columns[0]
    else:
        id_col = "Compound ID" if "Compound ID" in columns else columns[0]

    smiles_col = None
    if "Smiles" in columns:
        smiles_col = "Smiles"
    else:
        for col in columns:
            if "smiles" in col.lower():
                smiles_col = col
                break

    if smiles_col is None:
        raise ValueError("Could not infer SMILES column")

    return {"id_col": id_col, "smiles_col": smiles_col}


def iter_rows(table: Any) -> Iterator[Dict[str, str]]:
    if pd is not None and isinstance(table, pd.DataFrame):
        for _, row in table.iterrows():
            yield {str(k): str(v) for k, v in row.to_dict().items()}
    else:
        for row in table:
            yield {str(k): str(v) for k, v in row.items()}


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    smiles = (smiles or "").strip()
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def sanitize_copy(mol: Chem.Mol) -> Optional[Chem.Mol]:
    copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copy)
        return copy
    except Exception:
        try:
            flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(copy, sanitizeOps=flags)
            Chem.MolToSmiles(copy, canonical=True)
            return copy
        except Exception:
            return None


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def anchor_graph_distance(mol: Chem.Mol, anchor_left: int, anchor_right: int) -> int:
    return len(Chem.GetShortestPath(mol, int(anchor_left), int(anchor_right))) - 1


def linker_ratio_pct(linker_heavy_atoms: int, left_heavy_atoms: int, right_heavy_atoms: int) -> float:
    total = linker_heavy_atoms + left_heavy_atoms + right_heavy_atoms
    if total <= 0:
        raise ValueError("Total heavy atom count must be positive")
    return linker_heavy_atoms / total * 100.0


def _label_dummies_with_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() in {1, 2}:
            atom.SetAtomMapNum(int(atom.GetIsotope()))
            atom.SetIsotope(0)
    return rw.GetMol()


def get_unique_match(protac_mol: Chem.Mol, linker_mol: Chem.Mol) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    """Return the unique match tuple or a rejection reason."""

    matches = protac_mol.GetSubstructMatches(linker_mol, uniquify=True, maxMatches=2)
    if len(matches) == 0:
        return None, "NO_MATCH"
    if len(matches) > 1:
        return None, "MULTI_MATCH"
    return tuple(int(idx) for idx in matches[0]), None


def is_induced_subgraph_connected(mol: Chem.Mol, atom_indices: Sequence[int]) -> bool:
    atom_set = set(int(idx) for idx in atom_indices)
    if not atom_set:
        return False

    start = next(iter(atom_set))
    visited = {start}
    stack = [start]

    while stack:
        current = stack.pop()
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            neigh_idx = neighbor.GetIdx()
            if neigh_idx not in atom_set or neigh_idx in visited:
                continue
            visited.add(neigh_idx)
            stack.append(neigh_idx)

    return visited == atom_set


def get_crossing_bonds(mol: Chem.Mol, atom_indices: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Return crossing bond indices and linker-side anchor atom indices in stable order."""

    atom_set = set(int(idx) for idx in atom_indices)
    crossings: List[Tuple[int, int, int]] = []

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_in = begin in atom_set
        end_in = end in atom_set
        if begin_in == end_in:
            continue
        inside = begin if begin_in else end
        outside = end if begin_in else begin
        crossings.append((inside, bond.GetIdx(), outside))

    crossings.sort(key=lambda item: (item[0], item[1], item[2]))
    bond_ids = [item[1] for item in crossings]
    anchor_ids = [item[0] for item in crossings]
    return bond_ids, anchor_ids


def fragment_with_dummies(mol: Chem.Mol, crossing_bond_indices: Sequence[int]) -> Chem.Mol:
    if len(crossing_bond_indices) != 2:
        raise ValueError("Expected exactly two crossing bonds")
    fragmented = Chem.FragmentOnBonds(
        mol,
        list(crossing_bond_indices),
        addDummies=True,
        dummyLabels=[(1, 1), (2, 2)],
    )
    return _label_dummies_with_atom_maps(fragmented)


def _dummy_map_numbers(mol: Chem.Mol) -> List[int]:
    return sorted(
        int(atom.GetAtomMapNum())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0
    )


def _fragment_has_only_dummy_label(mol: Chem.Mol, label: int) -> bool:
    return _dummy_map_numbers(mol) == [label]


def _fragment_sort_key(mol: Chem.Mol, index: int) -> Tuple[int, int]:
    return (-mol.GetNumAtoms(), index)


def extract_linker_left_right(
    fragmented_mol: Chem.Mol,
    min_fragment_heavy_atoms: int = 3,
) -> Tuple[Optional[Chem.Mol], Optional[Chem.Mol], Optional[Chem.Mol], Optional[str]]:
    """Identify anchored linker and left/right fragments from a fragmented molecule."""

    fragments = list(Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=False))
    linker_idx: Optional[int] = None

    for idx, frag in enumerate(fragments):
        dummy_maps = _dummy_map_numbers(frag)
        num_dummy_atoms = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0)
        if dummy_maps == [1, 2] and num_dummy_atoms == 2:
            linker_idx = idx
            break

    if linker_idx is None:
        return None, None, None, "LINKER_FRAGMENT_FAIL"

    anchored_linker = fragments[linker_idx]
    other_fragments = [(idx, frag) for idx, frag in enumerate(fragments) if idx != linker_idx]
    if len(other_fragments) < 2:
        return None, None, None, "FRAGMENT_TOO_SMALL"

    left_candidates = [(idx, frag) for idx, frag in other_fragments if _fragment_has_only_dummy_label(frag, 1)]
    right_candidates = [(idx, frag) for idx, frag in other_fragments if _fragment_has_only_dummy_label(frag, 2)]

    left: Optional[Chem.Mol] = None
    right: Optional[Chem.Mol] = None

    if left_candidates and right_candidates:
        left = sorted(left_candidates, key=lambda item: _fragment_sort_key(item[1], item[0]))[0][1]
        right = sorted(right_candidates, key=lambda item: _fragment_sort_key(item[1], item[0]))[0][1]
    else:
        largest = sorted(other_fragments, key=lambda item: _fragment_sort_key(item[1], item[0]))[:2]
        if len(largest) < 2:
            return None, None, None, "FRAGMENT_TOO_SMALL"
        frag_a = largest[0][1]
        frag_b = largest[1][1]
        smiles_a = canonical_smiles(sanitize_copy(frag_a) or frag_a)
        smiles_b = canonical_smiles(sanitize_copy(frag_b) or frag_b)
        if smiles_a <= smiles_b:
            left, right = frag_a, frag_b
        else:
            left, right = frag_b, frag_a

    if left is None or right is None:
        return None, None, None, "LINKER_FRAGMENT_FAIL"

    if count_heavy_atoms(left) < min_fragment_heavy_atoms or count_heavy_atoms(right) < min_fragment_heavy_atoms:
        return None, None, None, "FRAGMENT_TOO_SMALL"

    return anchored_linker, left, right, None


def build_record_from_row(row: Mapping[str, Any], id_col: str, smiles_col: str) -> Optional[MolRecord]:
    row_id = str(row.get(id_col, "")).strip()
    smiles = str(row.get(smiles_col, "")).strip()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return MolRecord(
        row_id=row_id,
        smiles=smiles,
        mol=mol,
        canonical_smiles=canonical_smiles(mol),
        num_atoms=mol.GetNumAtoms(),
        source_row=dict(row),
    )


def dedupe_records(records: Iterable[MolRecord]) -> List[MolRecord]:
    deduped: List[MolRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.canonical_smiles in seen:
            continue
        seen.add(record.canonical_smiles)
        deduped.append(record)
    return deduped


def process_pair(
    protac_record: MolRecord,
    linker_record: MolRecord,
    min_fragment_heavy_atoms: int = 9,
    min_linker_heavy_atoms: int = 8,
    min_anchor_graph_distance: int = 5,
    min_linker_ratio_pct: float = 15.0,
    max_linker_ratio_pct: float = 35.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    """Process one (full PROTAC, candidate linker) pair."""

    try:
        linker_heavy_atoms = count_heavy_atoms(linker_record.mol)
        if linker_heavy_atoms < min_linker_heavy_atoms:
            return None, {
                "protac_id": protac_record.row_id,
                "linker_id": linker_record.row_id,
                "full_protac_smiles": protac_record.smiles,
                "linker_smiles": linker_record.smiles,
                "rejection_reason": "LINKER_TOO_SMALL",
            }

        match, reason = get_unique_match(protac_record.mol, linker_record.mol)
        if match is None:
            return None, {
                "protac_id": protac_record.row_id,
                "linker_id": linker_record.row_id,
                "full_protac_smiles": protac_record.smiles,
                "linker_smiles": linker_record.smiles,
                "rejection_reason": str(reason),
            }

        if not is_induced_subgraph_connected(protac_record.mol, match):
            return None, {
                "protac_id": protac_record.row_id,
                "linker_id": linker_record.row_id,
                "full_protac_smiles": protac_record.smiles,
                "linker_smiles": linker_record.smiles,
                "rejection_reason": "DISCONNECTED_LINKER",
            }

        crossing_bond_indices, anchor_atom_indices = get_crossing_bonds(protac_record.mol, match)
        if len(crossing_bond_indices) != 2:
            return None, {
                "protac_id": protac_record.row_id,
                "linker_id": linker_record.row_id,
                "full_protac_smiles": protac_record.smiles,
                "linker_smiles": linker_record.smiles,
                "rejection_reason": "BOUNDARY_NE_2",
            }

        if len(set(anchor_atom_indices)) != 2:
            return None, {
                "protac_id": protac_record.row_id,
                "linker_id": linker_record.row_id,
                "full_protac_smiles": protac_record.smiles,
                "linker_smiles": linker_record.smiles,
                "rejection_reason": "SAME_ANCHOR",
            }

        graph_distance = anchor_graph_distance(protac_record.mol, anchor_atom_indices[0], anchor_atom_indices[1])
        if graph_distance < min_anchor_graph_distance:
            return None, {
                "protac_id": protac_record.row_id,
                "linker_id": linker_record.row_id,
                "full_protac_smiles": protac_record.smiles,
                "linker_smiles": linker_record.smiles,
                "rejection_reason": "ANCHOR_DISTANCE_TOO_SHORT",
            }

        fragmented = fragment_with_dummies(protac_record.mol, crossing_bond_indices)
        anchored_linker, left_fragment, right_fragment, frag_reason = extract_linker_left_right(
            fragmented,
            min_fragment_heavy_atoms=min_fragment_heavy_atoms,
        )
        if frag_reason is not None or anchored_linker is None or left_fragment is None or right_fragment is None:
            return None, {
                "protac_id": protac_record.row_id,
                "linker_id": linker_record.row_id,
                "full_protac_smiles": protac_record.smiles,
                "linker_smiles": linker_record.smiles,
                "rejection_reason": str(frag_reason),
            }

        anchored_linker_sane = sanitize_copy(anchored_linker)
        left_sane = sanitize_copy(left_fragment)
        right_sane = sanitize_copy(right_fragment)
        if anchored_linker_sane is None or left_sane is None or right_sane is None:
            return None, {
                "protac_id": protac_record.row_id,
                "linker_id": linker_record.row_id,
                "full_protac_smiles": protac_record.smiles,
                "linker_smiles": linker_record.smiles,
                "rejection_reason": "SANITIZE_FAIL",
            }

        left_heavy_atoms = count_heavy_atoms(left_sane)
        right_heavy_atoms = count_heavy_atoms(right_sane)
        ratio_pct = linker_ratio_pct(linker_heavy_atoms, left_heavy_atoms, right_heavy_atoms)
        if ratio_pct < min_linker_ratio_pct or ratio_pct > max_linker_ratio_pct:
            return None, {
                "protac_id": protac_record.row_id,
                "linker_id": linker_record.row_id,
                "full_protac_smiles": protac_record.smiles,
                "linker_smiles": linker_record.smiles,
                "rejection_reason": "LINKER_RATIO_OUT_OF_RANGE",
            }

        return {
            "protac_id": protac_record.row_id,
            "linker_id": linker_record.row_id,
            "full_protac_smiles": protac_record.canonical_smiles,
            "linker_smiles": linker_record.canonical_smiles,
            "anchored_linker_smiles": canonical_smiles(anchored_linker_sane),
            "left_fragment_smiles": canonical_smiles(left_sane),
            "right_fragment_smiles": canonical_smiles(right_sane),
            "anchor_left_atom_idx_in_full": int(anchor_atom_indices[0]),
            "anchor_right_atom_idx_in_full": int(anchor_atom_indices[1]),
            "num_atoms_full": int(protac_record.num_atoms),
            "num_atoms_linker": int(linker_record.num_atoms),
            "num_atoms_left": int(left_sane.GetNumAtoms()),
            "num_atoms_right": int(right_sane.GetNumAtoms()),
            "_num_heavy_atoms_linker": int(linker_heavy_atoms),
            "_num_heavy_atoms_left": int(left_heavy_atoms),
            "_num_heavy_atoms_right": int(right_heavy_atoms),
            "_anchor_graph_distance": int(graph_distance),
            "_linker_ratio_pct": float(ratio_pct),
        }, None
    except Exception:
        return None, {
            "protac_id": protac_record.row_id,
            "linker_id": linker_record.row_id,
            "full_protac_smiles": protac_record.smiles,
            "linker_smiles": linker_record.smiles,
            "rejection_reason": "EXCEPTION",
        }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a weakly supervised PROTAC linker-anchor dataset")
    parser.add_argument("--protac_csv", type=str, required=True)
    parser.add_argument("--linker_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--rej_csv", type=str, required=True)
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--max_pairs", type=int, default=None, help="Stop after processing this many pair comparisons")
    parser.add_argument("--dedupe_protacs", type=parse_bool, default=True)
    parser.add_argument("--dedupe_linkers", type=parse_bool, default=True)
    parser.add_argument(
        "--min_fragment_heavy_atoms",
        type=int,
        default=9,
        help="Require both side fragments to have at least this many heavy atoms. Default 9 enforces '> 8'.",
    )
    parser.add_argument(
        "--min_linker_heavy_atoms",
        type=int,
        default=8,
        help="Require matched linker to have at least this many heavy atoms.",
    )
    parser.add_argument(
        "--min_anchor_graph_distance",
        type=int,
        default=5,
        help="Require shortest path distance between linker-side anchor atoms in the full PROTAC to be at least this value.",
    )
    parser.add_argument(
        "--min_linker_ratio_pct",
        type=float,
        default=15.0,
        help="Require linker heavy-atom ratio to be at least this percent of linker+left+right.",
    )
    parser.add_argument(
        "--max_linker_ratio_pct",
        type=float,
        default=35.0,
        help="Require linker heavy-atom ratio to be at most this percent of linker+left+right.",
    )
    parser.add_argument(
        "--log_no_match_rejections",
        type=parse_bool,
        default=False,
        help="Whether to write NO_MATCH rows to rejection CSV. Disabled by default because the file can become huge.",
    )
    parser.add_argument(
        "--debug_json",
        type=str,
        default="",
        help="Optional path to write up to 20 accepted examples with anchor indices and smiles.",
    )
    return parser


def candidate_rank_key(row: Dict[str, Any]) -> Tuple[int, int, float, int, int, str, str]:
    """Higher is better; final string terms keep the choice deterministic."""

    return (
        int(row["_num_heavy_atoms_linker"]),
        int(row["_anchor_graph_distance"]),
        float(row["_linker_ratio_pct"]),
        int(row["num_atoms_linker"]),
        min(int(row["_num_heavy_atoms_left"]), int(row["_num_heavy_atoms_right"])),
        int(row["_num_heavy_atoms_left"]) + int(row["_num_heavy_atoms_right"]),
        str(row["anchored_linker_smiles"]),
        str(row["linker_id"]),
    )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    protac_df = load_table(args.protac_csv)
    linker_df = load_table(args.linker_csv)
    protac_cols = infer_columns(protac_df)
    linker_cols = infer_columns(linker_df)

    protac_records_raw = [
        build_record_from_row(row, protac_cols["id_col"], protac_cols["smiles_col"])
        for row in iter_rows(protac_df)
    ]
    linker_records_raw = [
        build_record_from_row(row, linker_cols["id_col"], linker_cols["smiles_col"])
        for row in iter_rows(linker_df)
    ]

    protac_invalid = sum(record is None for record in protac_records_raw)
    linker_invalid = sum(record is None for record in linker_records_raw)
    protac_records = [record for record in protac_records_raw if record is not None]
    linker_records = [record for record in linker_records_raw if record is not None]
    protac_input_count = len(protac_records)
    linker_input_count = len(linker_records)
    if args.dedupe_protacs:
        protac_records = dedupe_records(protac_records)
    if args.dedupe_linkers:
        linker_records = dedupe_records(linker_records)
    linker_pool = [
        (count_heavy_atoms(record.mol), record)
        for record in linker_records
    ]
    linker_pool.sort(
        key=lambda item: (
            item[0],
            item[1].num_atoms,
            item[1].canonical_smiles,
            item[1].row_id,
        ),
        reverse=True,
    )

    print(
        f"[load] protacs={len(protac_records)} invalid_protacs={protac_invalid} "
        f"dedupe_protacs={args.dedupe_protacs} input_protacs={protac_input_count}"
    )
    print(
        f"[load] linkers={len(linker_records)} invalid_linkers={linker_invalid} "
        f"dedupe_linkers={args.dedupe_linkers} input_linkers={linker_input_count}"
    )

    out_path = Path(args.out_csv)
    rej_path = Path(args.rej_csv)
    summary_path = Path(args.summary_json)
    if args.debug_json:
        debug_path = Path(args.debug_json)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        debug_path = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rej_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    accepted_fields = [
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
    rejection_fields = [
        "protac_id",
        "linker_id",
        "full_protac_smiles",
        "linker_smiles",
        "rejection_reason",
    ]

    summary: Dict[str, Any] = {
        "total_pairs_tested": 0,
        "accepted": 0,
        "rejected": 0,
        "invalid_protac_rows": protac_invalid,
        "invalid_linker_rows": linker_invalid,
        "input_protac_rows": protac_input_count,
        "input_linker_rows": linker_input_count,
        "protacs_after_dedupe": len(protac_records),
        "linkers_after_dedupe": len(linker_records),
        "dedupe_protacs": bool(args.dedupe_protacs),
        "dedupe_linkers": bool(args.dedupe_linkers),
        "log_no_match_rejections": bool(args.log_no_match_rejections),
        "min_fragment_heavy_atoms": int(args.min_fragment_heavy_atoms),
        "min_linker_heavy_atoms": int(args.min_linker_heavy_atoms),
        "min_anchor_graph_distance": int(args.min_anchor_graph_distance),
        "min_linker_ratio_pct": float(args.min_linker_ratio_pct),
        "max_linker_ratio_pct": float(args.max_linker_ratio_pct),
    }
    for reason in REJECTION_REASONS:
        summary[reason] = 0
    summary["valid_candidates"] = 0
    summary["protacs_with_valid_candidate"] = 0
    summary["protacs_without_valid_candidate"] = 0

    sample_id = 0
    debug_examples: List[Dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8", newline="") as out_f, rej_path.open(
        "w", encoding="utf-8", newline=""
    ) as rej_f:
        accepted_writer = csv.DictWriter(out_f, fieldnames=accepted_fields)
        rejection_writer = csv.DictWriter(rej_f, fieldnames=rejection_fields)
        accepted_writer.writeheader()
        rejection_writer.writeheader()

        stop_early = False
        for protac_idx, protac_record in enumerate(protac_records, start=1):
            if protac_idx == 1 or protac_idx % 100 == 0:
                print(
                    f"[progress] protac={protac_idx}/{len(protac_records)} "
                    f"tested={summary['total_pairs_tested']} selected={summary['accepted']} rejected={summary['rejected']}"
                )

            best_row: Optional[Dict[str, Any]] = None
            best_rank: Optional[Tuple[int, int, float, int, int, str, str]] = None
            protac_heavy = count_heavy_atoms(protac_record.mol)

            for linker_heavy, linker_record in linker_pool:
                if protac_record.num_atoms <= linker_record.num_atoms:
                    continue
                if protac_heavy <= linker_heavy:
                    continue
                if best_rank is not None and linker_heavy < best_rank[0]:
                    break

                summary["total_pairs_tested"] += 1
                accepted_row, rejection_row = process_pair(
                    protac_record,
                    linker_record,
                    min_fragment_heavy_atoms=args.min_fragment_heavy_atoms,
                    min_linker_heavy_atoms=args.min_linker_heavy_atoms,
                    min_anchor_graph_distance=args.min_anchor_graph_distance,
                    min_linker_ratio_pct=args.min_linker_ratio_pct,
                    max_linker_ratio_pct=args.max_linker_ratio_pct,
                )

                if accepted_row is not None:
                    summary["valid_candidates"] += 1
                    rank = candidate_rank_key(accepted_row)
                    if best_rank is None or rank > best_rank:
                        best_row = accepted_row
                        best_rank = rank
                else:
                    assert rejection_row is not None
                    reason = rejection_row["rejection_reason"]
                    summary[reason] = summary.get(reason, 0) + 1
                    summary["rejected"] += 1
                    if args.log_no_match_rejections or reason != "NO_MATCH":
                        rejection_writer.writerow(rejection_row)

                if args.max_pairs is not None and summary["total_pairs_tested"] >= args.max_pairs:
                    stop_early = True
                    break

            if best_row is not None:
                summary["protacs_with_valid_candidate"] += 1
                sample_id += 1
                best_row["sample_id"] = sample_id
                accepted_writer.writerow({field: best_row[field] for field in accepted_fields})
                summary["accepted"] += 1
                if debug_path is not None and len(debug_examples) < 20:
                    debug_examples.append({k: v for k, v in best_row.items() if not k.startswith("_")})
            else:
                summary["protacs_without_valid_candidate"] += 1

            if stop_early:
                print(f"[stop] reached max_pairs={args.max_pairs}")
                break

    if debug_path is not None:
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(debug_examples, f, indent=2)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
