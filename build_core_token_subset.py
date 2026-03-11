from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build core-token-only subset by dropping samples containing low-frequency tokens")
    parser.add_argument(
        "--tokenized_csv",
        type=str,
        default="data/processed/linker_anchor_tokenized_multi.csv",
    )
    parser.add_argument(
        "--instances_csv",
        type=str,
        default="data/processed/linker_anchor_fragment_instances_multi.csv",
    )
    parser.add_argument(
        "--out_tokenized_csv",
        type=str,
        default="data/processed/linker_anchor_tokenized_core10.csv",
    )
    parser.add_argument(
        "--out_instances_csv",
        type=str,
        default="data/processed/linker_anchor_fragment_instances_core10.csv",
    )
    parser.add_argument(
        "--out_library_csv",
        type=str,
        default="data/processed/linker_anchor_fragment_library_core10.csv",
    )
    parser.add_argument(
        "--out_dropped_csv",
        type=str,
        default="data/processed/linker_anchor_core10_dropped_samples.csv",
    )
    parser.add_argument(
        "--out_summary_json",
        type=str,
        default="data/processed/linker_anchor_core10_summary.json",
    )
    parser.add_argument("--min_token_freq", type=int, default=10)
    return parser


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = make_parser().parse_args()
    tokenized_csv = Path(args.tokenized_csv)
    instances_csv = Path(args.instances_csv)
    if not tokenized_csv.exists():
        raise FileNotFoundError(f"tokenized csv not found: {tokenized_csv}")
    if not instances_csv.exists():
        raise FileNotFoundError(f"instances csv not found: {instances_csv}")

    tokenized_rows = read_csv_rows(tokenized_csv)
    instance_rows = read_csv_rows(instances_csv)
    if not tokenized_rows or not instance_rows:
        raise RuntimeError("input csv is empty")

    token_freq: Counter[str] = Counter()
    sample_to_tokens: Dict[str, List[str]] = defaultdict(list)
    sample_to_linker: Dict[str, str] = {}
    sample_to_instance_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in instance_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        linker_id = str(row.get("linker_id", "")).strip()
        token = str(row.get("token_smiles", "")).strip()
        if not sample_id:
            continue
        sample_to_linker[sample_id] = linker_id
        sample_to_instance_rows[sample_id].append(row)
        if token:
            token_freq[token] += 1
            sample_to_tokens[sample_id].append(token)

    min_token_freq = int(args.min_token_freq)
    core_tokens = {tok for tok, freq in token_freq.items() if freq >= min_token_freq}

    dropped_rows: List[Dict[str, str]] = []
    kept_sample_ids: set[str] = set()
    for sample_id, tokens in sample_to_tokens.items():
        bad_tokens = sorted({tok for tok in tokens if tok not in core_tokens})
        if bad_tokens:
            dropped_rows.append(
                {
                    "sample_id": sample_id,
                    "linker_id": sample_to_linker.get(sample_id, ""),
                    "dropped_reason": "HAS_LOW_FREQ_TOKEN",
                    "low_freq_tokens_json": json.dumps(bad_tokens, ensure_ascii=False),
                }
            )
            continue
        kept_sample_ids.add(sample_id)

    kept_tokenized = [row for row in tokenized_rows if str(row.get("sample_id", "")).strip() in kept_sample_ids]
    kept_instances = [row for row in instance_rows if str(row.get("sample_id", "")).strip() in kept_sample_ids]

    # Re-balance sample weights after filtering: each linker's kept samples sum to 1.
    linker_to_samples: Dict[str, List[str]] = defaultdict(list)
    for row in kept_tokenized:
        linker_to_samples[str(row.get("linker_id", "")).strip()].append(str(row.get("sample_id", "")).strip())
    linker_to_weight: Dict[str, float] = {}
    for linker_id, samples in linker_to_samples.items():
        if not samples:
            continue
        linker_to_weight[linker_id] = 1.0 / float(len(samples))

    sample_to_weight: Dict[str, float] = {}
    sample_to_num_accepted: Dict[str, int] = {}
    for linker_id, samples in linker_to_samples.items():
        w = linker_to_weight.get(linker_id, 1.0)
        n = len(samples)
        for sample_id in samples:
            sample_to_weight[sample_id] = w
            sample_to_num_accepted[sample_id] = n

    for row in kept_tokenized:
        sample_id = str(row.get("sample_id", "")).strip()
        old_weight = str(row.get("sample_weight", ""))
        if "sample_weight_before_core_filter" not in row:
            row["sample_weight_before_core_filter"] = old_weight
        row["sample_weight"] = f"{sample_to_weight.get(sample_id, 1.0):.8f}"
        if "num_anchor_pair_accepted_before_core_filter" not in row:
            row["num_anchor_pair_accepted_before_core_filter"] = str(row.get("num_anchor_pair_accepted", ""))
        row["num_anchor_pair_accepted"] = str(sample_to_num_accepted.get(sample_id, 1))

    for row in kept_instances:
        sample_id = str(row.get("sample_id", "")).strip()
        old_weight = str(row.get("sample_weight", ""))
        if "sample_weight_before_core_filter" not in row:
            row["sample_weight_before_core_filter"] = old_weight
        row["sample_weight"] = f"{sample_to_weight.get(sample_id, 1.0):.8f}"
        if "num_anchor_pair_accepted_before_core_filter" not in row:
            row["num_anchor_pair_accepted_before_core_filter"] = str(row.get("num_anchor_pair_accepted", ""))
        row["num_anchor_pair_accepted"] = str(sample_to_num_accepted.get(sample_id, 1))

    out_tokenized_csv = Path(args.out_tokenized_csv)
    out_instances_csv = Path(args.out_instances_csv)
    out_library_csv = Path(args.out_library_csv)
    out_dropped_csv = Path(args.out_dropped_csv)
    out_summary_json = Path(args.out_summary_json)

    tokenized_fields = list(kept_tokenized[0].keys()) if kept_tokenized else list(tokenized_rows[0].keys())
    instance_fields = list(kept_instances[0].keys()) if kept_instances else list(instance_rows[0].keys())
    dropped_fields = ["sample_id", "linker_id", "dropped_reason", "low_freq_tokens_json"]
    write_csv_rows(out_tokenized_csv, kept_tokenized, tokenized_fields)
    write_csv_rows(out_instances_csv, kept_instances, instance_fields)
    write_csv_rows(out_dropped_csv, dropped_rows, dropped_fields)

    # Recompute library on kept instances.
    kept_freq: Counter[str] = Counter()
    kept_linker_ids: Dict[str, set[str]] = defaultdict(set)
    for row in kept_instances:
        token = str(row.get("token_smiles", "")).strip()
        linker_id = str(row.get("linker_id", "")).strip()
        if not token:
            continue
        kept_freq[token] += 1
        if linker_id:
            kept_linker_ids[token].add(linker_id)

    library_rows: List[Dict[str, str]] = []
    for token, freq in kept_freq.most_common():
        library_rows.append(
            {
                "token_smiles": token,
                "frequency": str(int(freq)),
                "num_unique_linkers": str(len(kept_linker_ids[token])),
            }
        )
    write_csv_rows(
        out_library_csv,
        library_rows,
        ["token_smiles", "frequency", "num_unique_linkers"],
    )

    # Coverage summary.
    total_samples = len(sample_to_tokens)
    kept_samples = len(kept_sample_ids)
    dropped_samples = total_samples - kept_samples
    total_instances = len(instance_rows)
    kept_instances_count = len(kept_instances)
    dropped_instances_count = total_instances - kept_instances_count
    tail_tokens = sorted(tok for tok, freq in token_freq.items() if freq < min_token_freq)
    linker_weight_sum: Dict[str, float] = defaultdict(float)
    for row in kept_tokenized:
        linker_weight_sum[str(row.get("linker_id", "")).strip()] += float(row.get("sample_weight", "1.0"))
    bad_weight_linkers = [lid for lid, w in linker_weight_sum.items() if abs(w - 1.0) > 1e-6]

    summary = {
        "min_token_freq": min_token_freq,
        "input": {
            "tokenized_csv": str(tokenized_csv),
            "instances_csv": str(instances_csv),
            "num_samples": total_samples,
            "num_instances": total_instances,
            "num_unique_tokens": len(token_freq),
        },
        "core_tokens": {
            "num_core_tokens": len(core_tokens),
            "num_tail_tokens": len(tail_tokens),
            "tail_tokens_top_50": tail_tokens[:50],
        },
        "output": {
            "out_tokenized_csv": str(out_tokenized_csv),
            "out_instances_csv": str(out_instances_csv),
            "out_library_csv": str(out_library_csv),
            "out_dropped_csv": str(out_dropped_csv),
            "out_summary_json": str(out_summary_json),
            "num_samples_kept": kept_samples,
            "num_samples_dropped": dropped_samples,
            "num_instances_kept": kept_instances_count,
            "num_instances_dropped": dropped_instances_count,
            "sample_keep_rate_pct": 100.0 * kept_samples / max(1, total_samples),
            "instance_keep_rate_pct": 100.0 * kept_instances_count / max(1, total_instances),
            "num_unique_tokens_kept": len(kept_freq),
            "linkers_with_nonunit_weight_sum_after_rebalance": len(bad_weight_linkers),
        },
    }
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[done] min_token_freq={min_token_freq} "
        f"samples_kept={kept_samples}/{total_samples} "
        f"instances_kept={kept_instances_count}/{total_instances} "
        f"core_tokens={len(core_tokens)}/{len(token_freq)}",
        flush=True,
    )
    print(
        f"[files] tokenized={out_tokenized_csv} instances={out_instances_csv} "
        f"library={out_library_csv} dropped={out_dropped_csv} summary={out_summary_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
