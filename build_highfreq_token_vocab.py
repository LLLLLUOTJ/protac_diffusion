from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean: {text}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build token coverage stats and high-frequency vocabulary")
    parser.add_argument(
        "--instances_csv",
        type=str,
        default="data/processed/linker_anchor_fragment_instances_multi.csv",
    )
    parser.add_argument(
        "--stats_csv",
        type=str,
        default="data/processed/linker_anchor_token_coverage_stats.csv",
    )
    parser.add_argument(
        "--vocab_txt",
        type=str,
        default="data/processed/linker_anchor_token_vocab_highfreq.txt",
    )
    parser.add_argument(
        "--vocab_json",
        type=str,
        default="data/processed/linker_anchor_token_vocab_highfreq.json",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="data/processed/linker_anchor_token_vocab_summary.json",
    )
    parser.add_argument("--min_freq", type=int, default=20)
    parser.add_argument("--min_token_occurrence_pct", type=float, default=0.0)
    parser.add_argument("--target_cumulative_coverage_pct", type=float, default=95.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--add_special_tokens", type=parse_bool, default=True)
    parser.add_argument("--special_tokens", type=str, default="<PAD>,<UNK>,<MASK>")
    return parser


def parse_special_tokens(text: str) -> List[str]:
    tokens = [x.strip() for x in str(text).split(",")]
    return [x for x in tokens if x]


def safe_float(text: str, default: float = 1.0) -> float:
    try:
        return float(text)
    except Exception:
        return default


def ensure_parent(paths: Sequence[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = make_parser().parse_args()
    instances_csv = Path(args.instances_csv)
    if not instances_csv.exists():
        raise FileNotFoundError(f"instances csv not found: {instances_csv}")

    freq: Counter[str] = Counter()
    weighted_freq: Dict[str, float] = defaultdict(float)
    token_to_sample_ids: Dict[str, set[str]] = defaultdict(set)
    token_to_linker_ids: Dict[str, set[str]] = defaultdict(set)
    all_sample_ids: set[str] = set()
    all_linker_ids: set[str] = set()
    total_instances = 0
    total_weighted_instances = 0.0

    with instances_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = str(row.get("token_smiles", "")).strip()
            if not token:
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            linker_id = str(row.get("linker_id", "")).strip()
            weight = safe_float(str(row.get("sample_weight", "1.0")))

            total_instances += 1
            total_weighted_instances += weight
            freq[token] += 1
            weighted_freq[token] += weight
            if sample_id:
                token_to_sample_ids[token].add(sample_id)
                all_sample_ids.add(sample_id)
            if linker_id:
                token_to_linker_ids[token].add(linker_id)
                all_linker_ids.add(linker_id)

    if total_instances == 0:
        raise RuntimeError("no token instances found")

    total_samples = max(1, len(all_sample_ids))
    total_linkers = max(1, len(all_linker_ids))
    total_weighted_instances = max(total_weighted_instances, 1e-12)

    ordered_tokens = sorted(freq.keys(), key=lambda t: (-freq[t], t))
    stats_rows: List[Dict[str, str]] = []
    cumulative = 0
    weighted_cumulative = 0.0
    for rank, token in enumerate(ordered_tokens, start=1):
        f = int(freq[token])
        wf = float(weighted_freq[token])
        cumulative += f
        weighted_cumulative += wf
        token_occ_pct = 100.0 * f / float(total_instances)
        token_occ_pct_weighted = 100.0 * wf / float(total_weighted_instances)
        sample_cov_pct = 100.0 * len(token_to_sample_ids[token]) / float(total_samples)
        linker_cov_pct = 100.0 * len(token_to_linker_ids[token]) / float(total_linkers)
        cum_cov_pct = 100.0 * cumulative / float(total_instances)
        cum_cov_pct_weighted = 100.0 * weighted_cumulative / float(total_weighted_instances)
        stats_rows.append(
            {
                "rank": str(rank),
                "token_smiles": token,
                "frequency": str(f),
                "weighted_frequency": f"{wf:.8f}",
                "occurrence_coverage_pct": f"{token_occ_pct:.6f}",
                "weighted_occurrence_coverage_pct": f"{token_occ_pct_weighted:.6f}",
                "sample_coverage_pct": f"{sample_cov_pct:.6f}",
                "linker_coverage_pct": f"{linker_cov_pct:.6f}",
                "cumulative_occurrence_coverage_pct": f"{cum_cov_pct:.6f}",
                "cumulative_weighted_occurrence_coverage_pct": f"{cum_cov_pct_weighted:.6f}",
                "num_samples_with_token": str(len(token_to_sample_ids[token])),
                "num_linkers_with_token": str(len(token_to_linker_ids[token])),
            }
        )

    min_freq = int(args.min_freq)
    min_occ_pct = float(args.min_token_occurrence_pct)
    target_cov_pct = float(args.target_cumulative_coverage_pct)
    top_k = args.top_k if args.top_k is None else int(args.top_k)

    selected_tokens: List[str] = []
    selected_cumulative = 0
    for row in stats_rows:
        token = row["token_smiles"]
        f = int(row["frequency"])
        occ_pct = float(row["occurrence_coverage_pct"])
        if f < min_freq:
            continue
        if occ_pct < min_occ_pct:
            continue
        selected_tokens.append(token)
        selected_cumulative += f
        if top_k is not None and len(selected_tokens) >= top_k:
            break
        if target_cov_pct > 0:
            current_cov = 100.0 * selected_cumulative / float(total_instances)
            if current_cov >= target_cov_pct:
                break

    if len(selected_tokens) == 0:
        # fallback: keep the top-1 token to avoid empty vocab
        selected_tokens = [stats_rows[0]["token_smiles"]]
        selected_cumulative = int(stats_rows[0]["frequency"])

    special_tokens: List[str] = []
    if bool(args.add_special_tokens):
        special_tokens = parse_special_tokens(args.special_tokens)
    vocab_tokens = special_tokens + selected_tokens
    token_to_id = {tok: idx for idx, tok in enumerate(vocab_tokens)}

    selected_set = set(selected_tokens)
    selected_weighted_sum = sum(float(weighted_freq[t]) for t in selected_tokens)

    summary = {
        "instances_csv": str(instances_csv),
        "total_instances": int(total_instances),
        "total_weighted_instances": float(total_weighted_instances),
        "total_unique_tokens": len(freq),
        "total_unique_samples": len(all_sample_ids),
        "total_unique_linkers": len(all_linker_ids),
        "selection_config": {
            "min_freq": min_freq,
            "min_token_occurrence_pct": min_occ_pct,
            "target_cumulative_coverage_pct": target_cov_pct,
            "top_k": top_k,
            "add_special_tokens": bool(args.add_special_tokens),
            "special_tokens": special_tokens,
        },
        "selected_vocab_size_without_special": len(selected_tokens),
        "selected_vocab_size_with_special": len(vocab_tokens),
        "selected_occurrence_coverage_pct": 100.0 * float(selected_cumulative) / float(total_instances),
        "selected_weighted_occurrence_coverage_pct": 100.0 * selected_weighted_sum / float(total_weighted_instances),
        "selected_top_20_tokens": selected_tokens[:20],
        "coverage_at_topk": {
            "5": 100.0 * sum(int(stats_rows[i]["frequency"]) for i in range(min(5, len(stats_rows)))) / float(total_instances),
            "10": 100.0 * sum(int(stats_rows[i]["frequency"]) for i in range(min(10, len(stats_rows)))) / float(total_instances),
            "20": 100.0 * sum(int(stats_rows[i]["frequency"]) for i in range(min(20, len(stats_rows)))) / float(total_instances),
            "50": 100.0 * sum(int(stats_rows[i]["frequency"]) for i in range(min(50, len(stats_rows)))) / float(total_instances),
            "100": 100.0 * sum(int(stats_rows[i]["frequency"]) for i in range(min(100, len(stats_rows)))) / float(total_instances),
        },
    }

    stats_csv = Path(args.stats_csv)
    vocab_txt = Path(args.vocab_txt)
    vocab_json = Path(args.vocab_json)
    summary_json = Path(args.summary_json)
    ensure_parent([stats_csv, vocab_txt, vocab_json, summary_json])

    with stats_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "token_smiles",
                "frequency",
                "weighted_frequency",
                "occurrence_coverage_pct",
                "weighted_occurrence_coverage_pct",
                "sample_coverage_pct",
                "linker_coverage_pct",
                "cumulative_occurrence_coverage_pct",
                "cumulative_weighted_occurrence_coverage_pct",
                "num_samples_with_token",
                "num_linkers_with_token",
            ],
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    with vocab_txt.open("w", encoding="utf-8") as f:
        for token in vocab_tokens:
            f.write(f"{token}\n")

    with vocab_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tokens": vocab_tokens,
                "token_to_id": token_to_id,
                "selected_tokens": selected_tokens,
                "special_tokens": special_tokens,
                "selection_config": summary["selection_config"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[done] total_instances={total_instances} unique_tokens={len(freq)} "
        f"selected={len(selected_tokens)} selected_with_special={len(vocab_tokens)}",
        flush=True,
    )
    print(
        f"[coverage] selected_occurrence_coverage_pct={summary['selected_occurrence_coverage_pct']:.4f} "
        f"selected_weighted_occurrence_coverage_pct={summary['selected_weighted_occurrence_coverage_pct']:.4f}",
        flush=True,
    )
    print(
        f"[files] stats_csv={stats_csv} vocab_txt={vocab_txt} vocab_json={vocab_json} summary_json={summary_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
