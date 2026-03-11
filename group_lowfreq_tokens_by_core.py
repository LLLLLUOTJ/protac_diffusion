from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors


MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


@dataclass
class TokenFeatures:
    token: str
    freq: int
    parse_ok: bool
    mol: Optional[Chem.Mol]
    fp: Optional[DataStructs.ExplicitBitVect]
    attachment_count: int
    heavy_atoms: int
    aromatic_atoms: int
    ring_count: int
    count_c: int
    count_n: int
    count_o: int
    count_s: int
    count_p: int
    count_halogen: int
    has_carbonyl: bool
    has_imine: bool
    has_azo: bool
    has_triple: bool
    has_aromatic: bool
    motif_class: str


def parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean: {text}")


def infer_motif_class(token: str, c_n: int, c_o: int, c_s: int, has_aromatic: bool, has_carbonyl: bool, has_triple: bool) -> str:
    if has_aromatic:
        return "aromatic"
    if has_carbonyl:
        return "carbonyl"
    if has_triple:
        return "unsaturated"
    if c_n > 0 and c_o == 0 and c_s == 0:
        return "amine_like"
    if c_o > 0 and c_n == 0:
        return "oxygen_like"
    if c_s > 0:
        return "sulfur_like"
    if "C" in token:
        return "aliphatic"
    return "other"


def featurize_token(token: str, freq: int) -> TokenFeatures:
    mol = Chem.MolFromSmiles(token)
    if mol is None:
        return TokenFeatures(
            token=token,
            freq=freq,
            parse_ok=False,
            mol=None,
            fp=None,
            attachment_count=token.count("*"),
            heavy_atoms=0,
            aromatic_atoms=0,
            ring_count=0,
            count_c=0,
            count_n=0,
            count_o=0,
            count_s=0,
            count_p=0,
            count_halogen=0,
            has_carbonyl=("=O" in token),
            has_imine=("C=N" in token or "N=C" in token),
            has_azo=("N=N" in token),
            has_triple=("#" in token),
            has_aromatic=any(ch in token for ch in ["c", "n", "o", "s"]),
            motif_class="unparsed",
        )

    atom_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    attachment_count = sum(1 for z in atom_nums if z == 0)
    heavy_atoms = sum(1 for z in atom_nums if z > 1)
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ring_count = int(mol.GetRingInfo().NumRings())
    count_c = sum(1 for z in atom_nums if z == 6)
    count_n = sum(1 for z in atom_nums if z == 7)
    count_o = sum(1 for z in atom_nums if z == 8)
    count_s = sum(1 for z in atom_nums if z == 16)
    count_p = sum(1 for z in atom_nums if z == 15)
    count_halogen = sum(1 for z in atom_nums if z in {9, 17, 35, 53})
    has_carbonyl = bool(mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]=[#8]")))
    has_imine = bool(mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]=[#7]")) or mol.HasSubstructMatch(Chem.MolFromSmarts("[#7]=[#6]")))
    has_azo = bool(mol.HasSubstructMatch(Chem.MolFromSmarts("[#7]=[#7]")))
    has_triple = any(b.GetBondType() == Chem.BondType.TRIPLE for b in mol.GetBonds())
    has_aromatic = aromatic_atoms > 0
    motif_class = infer_motif_class(token, count_n, count_o, count_s, has_aromatic, has_carbonyl, has_triple)
    try:
        fp = MORGAN_GEN.GetFingerprint(mol)
    except Exception:
        # fallback for older RDKit builds
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
    return TokenFeatures(
        token=token,
        freq=freq,
        parse_ok=True,
        mol=mol,
        fp=fp,
        attachment_count=attachment_count,
        heavy_atoms=heavy_atoms,
        aromatic_atoms=aromatic_atoms,
        ring_count=ring_count,
        count_c=count_c,
        count_n=count_n,
        count_o=count_o,
        count_s=count_s,
        count_p=count_p,
        count_halogen=count_halogen,
        has_carbonyl=has_carbonyl,
        has_imine=has_imine,
        has_azo=has_azo,
        has_triple=has_triple,
        has_aromatic=has_aromatic,
        motif_class=motif_class,
    )


def descriptor_similarity(a: TokenFeatures, b: TokenFeatures) -> float:
    dist = 0.0
    dist += 2.5 * abs(a.attachment_count - b.attachment_count)
    dist += 1.0 * abs(a.heavy_atoms - b.heavy_atoms)
    dist += 1.2 * abs(a.aromatic_atoms - b.aromatic_atoms)
    dist += 1.0 * abs(a.ring_count - b.ring_count)
    dist += 1.0 * abs(a.count_n - b.count_n)
    dist += 1.0 * abs(a.count_o - b.count_o)
    dist += 1.0 * abs(a.count_s - b.count_s)
    dist += 0.8 * abs(a.count_halogen - b.count_halogen)
    dist += 0.7 * abs(a.count_p - b.count_p)
    dist += 0.9 * (1.0 if a.has_carbonyl != b.has_carbonyl else 0.0)
    dist += 0.8 * (1.0 if a.has_imine != b.has_imine else 0.0)
    dist += 0.8 * (1.0 if a.has_azo != b.has_azo else 0.0)
    dist += 0.8 * (1.0 if a.has_triple != b.has_triple else 0.0)
    dist += 0.6 * (1.0 if a.has_aromatic != b.has_aromatic else 0.0)
    return float(math.exp(-dist / 6.0))


def score_pair(low: TokenFeatures, core: TokenFeatures) -> Tuple[float, float, float, List[str]]:
    reasons: List[str] = []
    if not low.parse_ok or not core.parse_ok or low.fp is None or core.fp is None:
        return 0.0, 0.0, 0.0, ["PARSE_FAIL"]

    tanimoto = float(DataStructs.TanimotoSimilarity(low.fp, core.fp))
    dsim = descriptor_similarity(low, core)
    score = 0.72 * tanimoto + 0.28 * dsim

    if low.attachment_count != core.attachment_count:
        score -= 0.25
        reasons.append("ATTACHMENT_MISMATCH")
    else:
        reasons.append("ATTACHMENT_MATCH")

    if low.has_aromatic != core.has_aromatic:
        score -= 0.08
        reasons.append("AROMATIC_MISMATCH")
    else:
        reasons.append("AROMATIC_MATCH")

    if low.has_carbonyl and core.has_carbonyl:
        score += 0.06
        reasons.append("CARBONYL_MATCH")
    elif low.has_carbonyl != core.has_carbonyl:
        score -= 0.06
        reasons.append("CARBONYL_MISMATCH")

    if low.has_triple and core.has_triple:
        score += 0.04
        reasons.append("TRIPLE_MATCH")
    elif low.has_triple != core.has_triple:
        score -= 0.04
        reasons.append("TRIPLE_MISMATCH")

    if low.has_imine and core.has_imine:
        score += 0.03
        reasons.append("IMINE_MATCH")
    elif low.has_imine != core.has_imine:
        score -= 0.03
        reasons.append("IMINE_MISMATCH")

    if low.has_azo and core.has_azo:
        score += 0.03
        reasons.append("AZO_MATCH")
    elif low.has_azo != core.has_azo:
        score -= 0.03
        reasons.append("AZO_MISMATCH")

    if low.motif_class == core.motif_class:
        score += 0.04
        reasons.append("MOTIF_CLASS_MATCH")

    score = float(max(0.0, min(1.0, score)))
    return score, tanimoto, dsim, reasons


def choose_assignment(low: TokenFeatures, cores: Sequence[TokenFeatures]) -> Dict[str, object]:
    scored: List[Dict[str, object]] = []
    for core in cores:
        score, tanimoto, dsim, reasons = score_pair(low, core)
        scored.append(
            {
                "core_token": core.token,
                "core_freq": core.freq,
                "score": score,
                "tanimoto": tanimoto,
                "descriptor_similarity": dsim,
                "reasons": reasons,
                "attachment_match": int(low.attachment_count == core.attachment_count),
            }
        )
    scored.sort(key=lambda x: (-float(x["score"]), -int(x["core_freq"]), str(x["core_token"])))
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    margin = float(best["score"]) - (float(second["score"]) if second is not None else 0.0)

    confidence = "uncertain"
    uncertain_reason = ""
    attach_ok = int(best.get("attachment_match", 0)) == 1
    if not low.parse_ok:
        confidence = "uncertain"
        uncertain_reason = "LOW_TOKEN_PARSE_FAIL"
    elif (float(best["score"]) >= 0.78) and (margin >= 0.08) and attach_ok:
        confidence = "high"
    elif (float(best["score"]) >= 0.62) and (margin >= 0.05) and attach_ok:
        confidence = "medium"
    else:
        confidence = "uncertain"
        if not attach_ok:
            uncertain_reason = "ATTACHMENT_MISMATCH_WITH_BEST_CORE"
        elif margin < 0.05:
            uncertain_reason = "AMBIGUOUS_NEAREST_CORES"
        else:
            uncertain_reason = "LOW_SIMILARITY_SCORE"

    top3 = scored[:3]
    return {
        "best": best,
        "second": second,
        "margin": margin,
        "confidence": confidence,
        "uncertain_reason": uncertain_reason,
        "top3": top3,
    }


def read_frequency_rows(coverage_csv: Path) -> List[Tuple[str, int, int]]:
    rows: List[Tuple[str, int, int]] = []
    with coverage_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = str(row.get("token_smiles", "")).strip()
            if not token:
                continue
            try:
                rank = int(str(row.get("rank", "0")).strip())
            except Exception:
                rank = 0
            try:
                freq = int(str(row.get("frequency", "0")).strip())
            except Exception:
                freq = 0
            rows.append((token, freq, rank))
    rows.sort(key=lambda x: (-x[1], x[2], x[0]))
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Group low-frequency linker tokens by core token using chemical rules")
    parser.add_argument(
        "--coverage_csv",
        type=str,
        default="data/processed/linker_anchor_token_coverage_stats.csv",
    )
    parser.add_argument("--core_freq_threshold", type=int, default=10)
    parser.add_argument("--out_core_table_csv", type=str, default="data/processed/core_token_table.csv")
    parser.add_argument("--out_assignment_csv", type=str, default="data/processed/lowfreq_to_core_assignment.csv")
    parser.add_argument("--out_uncertain_csv", type=str, default="data/processed/lowfreq_uncertain_tokens.csv")
    parser.add_argument("--out_grouped_json", type=str, default="data/processed/lowfreq_grouped_by_core.json")
    parser.add_argument("--out_summary_json", type=str, default="data/processed/lowfreq_grouping_summary.json")
    parser.add_argument("--verbose", type=parse_bool, default=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    coverage_csv = Path(args.coverage_csv)
    if not coverage_csv.exists():
        raise FileNotFoundError(f"coverage csv not found: {coverage_csv}")

    freq_rows = read_frequency_rows(coverage_csv)
    core_thr = int(args.core_freq_threshold)
    core_rows = [(tok, freq, rank) for tok, freq, rank in freq_rows if freq >= core_thr]
    tail_rows = [(tok, freq, rank) for tok, freq, rank in freq_rows if freq < core_thr]
    if not core_rows:
        raise RuntimeError(f"no core tokens found with threshold={core_thr}")

    core_features = [featurize_token(tok, freq) for tok, freq, _ in core_rows]
    tail_features = [featurize_token(tok, freq) for tok, freq, _ in tail_rows]
    core_feature_by_token = {f.token: f for f in core_features}

    core_table_rows: List[Dict[str, str]] = []
    for tok, freq, rank in core_rows:
        core_table_rows.append(
            {
                "rank": str(rank),
                "token_smiles": tok,
                "frequency": str(freq),
            }
        )

    assignment_rows: List[Dict[str, str]] = []
    uncertain_rows: List[Dict[str, str]] = []
    grouped: Dict[str, List[Dict[str, object]]] = {f.token: [] for f in core_features}

    high_count = 0
    medium_count = 0
    uncertain_count = 0
    for low in tail_features:
        out = choose_assignment(low, core_features)
        best = out["best"]  # type: ignore[assignment]
        second = out["second"]  # type: ignore[assignment]
        confidence = str(out["confidence"])
        uncertain_reason = str(out["uncertain_reason"])
        margin = float(out["margin"])
        top3 = out["top3"]  # type: ignore[assignment]

        assigned_core = str(best["core_token"])
        row = {
            "low_token": low.token,
            "low_freq": str(low.freq),
            "assigned_core_token": assigned_core,
            "assigned_core_freq": str(best["core_freq"]),
            "confidence": confidence,
            "score": f"{float(best['score']):.6f}",
            "tanimoto": f"{float(best['tanimoto']):.6f}",
            "descriptor_similarity": f"{float(best['descriptor_similarity']):.6f}",
            "margin_vs_second": f"{margin:.6f}",
            "low_attachment_count": str(low.attachment_count),
            "core_attachment_count": str(core_feature_by_token[assigned_core].attachment_count),
            "low_motif_class": low.motif_class,
            "core_motif_class": core_feature_by_token[assigned_core].motif_class,
            "rule_flags_json": json.dumps(best["reasons"], ensure_ascii=False),  # type: ignore[arg-type]
            "top3_candidates_json": json.dumps(top3, ensure_ascii=False),
            "uncertain_reason": uncertain_reason,
        }
        assignment_rows.append(row)

        grouped.setdefault(assigned_core, []).append(
            {
                "low_token": low.token,
                "low_freq": low.freq,
                "confidence": confidence,
                "score": float(best["score"]),
            }
        )

        if confidence == "high":
            high_count += 1
        elif confidence == "medium":
            medium_count += 1
        else:
            uncertain_count += 1
            uncertain_rows.append(row)

    for token in grouped:
        grouped[token].sort(key=lambda x: (-float(x["score"]), -int(x["low_freq"]), str(x["low_token"])))

    out_core_table_csv = Path(args.out_core_table_csv)
    out_assignment_csv = Path(args.out_assignment_csv)
    out_uncertain_csv = Path(args.out_uncertain_csv)
    out_grouped_json = Path(args.out_grouped_json)
    out_summary_json = Path(args.out_summary_json)

    write_csv(
        out_core_table_csv,
        core_table_rows,
        fieldnames=["rank", "token_smiles", "frequency"],
    )
    write_csv(
        out_assignment_csv,
        assignment_rows,
        fieldnames=[
            "low_token",
            "low_freq",
            "assigned_core_token",
            "assigned_core_freq",
            "confidence",
            "score",
            "tanimoto",
            "descriptor_similarity",
            "margin_vs_second",
            "low_attachment_count",
            "core_attachment_count",
            "low_motif_class",
            "core_motif_class",
            "rule_flags_json",
            "top3_candidates_json",
            "uncertain_reason",
        ],
    )
    write_csv(
        out_uncertain_csv,
        uncertain_rows,
        fieldnames=[
            "low_token",
            "low_freq",
            "assigned_core_token",
            "assigned_core_freq",
            "confidence",
            "score",
            "tanimoto",
            "descriptor_similarity",
            "margin_vs_second",
            "low_attachment_count",
            "core_attachment_count",
            "low_motif_class",
            "core_motif_class",
            "rule_flags_json",
            "top3_candidates_json",
            "uncertain_reason",
        ],
    )
    out_grouped_json.parent.mkdir(parents=True, exist_ok=True)
    out_grouped_json.write_text(json.dumps(grouped, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "coverage_csv": str(coverage_csv),
        "core_freq_threshold": core_thr,
        "num_total_tokens": len(freq_rows),
        "num_core_tokens": len(core_rows),
        "num_tail_tokens": len(tail_rows),
        "assignment_counts": {
            "high": high_count,
            "medium": medium_count,
            "uncertain": uncertain_count,
        },
        "files": {
            "core_table_csv": str(out_core_table_csv),
            "assignment_csv": str(out_assignment_csv),
            "uncertain_csv": str(out_uncertain_csv),
            "grouped_json": str(out_grouped_json),
        },
    }
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if bool(args.verbose):
        print(
            f"[done] core={len(core_rows)} tail={len(tail_rows)} "
            f"high={high_count} medium={medium_count} uncertain={uncertain_count}",
            flush=True,
        )
        print(
            f"[files] core_table={out_core_table_csv} assignments={out_assignment_csv} "
            f"uncertain={out_uncertain_csv} grouped={out_grouped_json} summary={out_summary_json}",
            flush=True,
        )


if __name__ == "__main__":
    main()
