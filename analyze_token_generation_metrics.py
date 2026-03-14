from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.warning")

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canonicalize_smiles(smiles: str) -> str | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def morgan_fp(smiles: str) -> Any | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return MORGAN_GEN.GetFingerprint(mol)


def safe_json_list(text: str) -> list[str]:
    if not text:
        return []
    try:
        values = json.loads(text)
    except Exception:
        return []
    if not isinstance(values, list):
        return []
    return [str(x) for x in values]


def multiset_overlap_ratio(a: Sequence[str], b: Sequence[str]) -> float | None:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    return float(common) / float(max(len(b), 1))


def linker_descriptors(smiles: str) -> dict[str, float | int | None] | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None

    dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    anchor_distance: int | None = None
    if len(dummy_atoms) >= 2:
        path = Chem.rdmolops.GetShortestPath(mol, dummy_atoms[0], dummy_atoms[1])
        if path:
            anchor_distance = len(path) - 1

    return {
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "hetero_atoms": int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (0, 1, 6))),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "anchor_distance": anchor_distance,
    }


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(statistics.mean(values))


def median_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(statistics.median(values))


def min_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(min(values))


def max_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(max(values))


def load_rows(all_generations_json: str | Path) -> list[dict[str, Any]]:
    path = Path(all_generations_json)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"expected list in {path}")
    return [{str(k): v for k, v in row.items()} for row in rows]


def load_train_linkers(train_weak_anchor_csv: str | Path) -> tuple[list[str], set[str]]:
    path = Path(train_weak_anchor_csv)
    linkers: list[str] = []
    canonical: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = str(row.get("anchored_linker_smiles", "")).strip()
            if not smiles:
                continue
            linkers.append(smiles)
            canon = canonicalize_smiles(smiles)
            if canon is not None:
                canonical.add(canon)
    if not linkers:
        raise RuntimeError(f"no anchored_linker_smiles found in {path}")
    return linkers, canonical


def compute_row_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    train_linkers: Sequence[str],
    train_canonical: set[str],
) -> list[dict[str, Any]]:
    train_fps = [fp for fp in (morgan_fp(smiles) for smiles in train_linkers) if fp is not None]
    out: list[dict[str, Any]] = []

    for row in rows:
        generated = str(row.get("generated_anchored_linker_smiles", "") or "")
        source = str(row.get("source_anchored_linker_smiles", "") or "")
        generated_fp = morgan_fp(generated)
        source_fp = morgan_fp(source)
        generated_canon = canonicalize_smiles(generated)

        source_tokens = safe_json_list(str(row.get("source_oriented_token_smiles", "") or ""))
        generated_tokens = safe_json_list(str(row.get("generated_oriented_token_smiles", "") or ""))

        source_similarity = None
        if generated_fp is not None and source_fp is not None:
            source_similarity = float(DataStructs.TanimotoSimilarity(generated_fp, source_fp))

        train_nn_similarity = None
        if generated_fp is not None and train_fps:
            train_nn_similarity = float(max(DataStructs.BulkTanimotoSimilarity(generated_fp, train_fps)))

        gen_desc = linker_descriptors(generated)
        src_desc = linker_descriptors(source)

        item: dict[str, Any] = dict(row)
        item["generated_canonical_smiles"] = generated_canon
        item["source_canonical_smiles"] = canonicalize_smiles(source)
        item["exact_source_match"] = bool(generated == source)
        item["exact_train_match"] = bool(generated_canon in train_canonical) if generated_canon is not None else False
        item["source_similarity"] = source_similarity
        item["train_nn_similarity"] = train_nn_similarity
        item["source_token_length"] = len(source_tokens)
        item["generated_token_length"] = len(generated_tokens)
        item["same_token_length"] = bool(len(source_tokens) == len(generated_tokens))
        item["exact_token_match"] = bool(source_tokens == generated_tokens)
        item["token_overlap_ratio"] = multiset_overlap_ratio(source_tokens, generated_tokens)

        for prefix, desc in [("generated", gen_desc), ("source", src_desc)]:
            if desc is None:
                for key in ["heavy_atoms", "hetero_atoms", "ring_count", "rotatable_bonds", "tpsa", "anchor_distance"]:
                    item[f"{prefix}_{key}"] = None
            else:
                for key, value in desc.items():
                    item[f"{prefix}_{key}"] = value

        for key in ["heavy_atoms", "hetero_atoms", "ring_count", "rotatable_bonds", "tpsa", "anchor_distance"]:
            gen_value = item.get(f"generated_{key}")
            src_value = item.get(f"source_{key}")
            if gen_value is None or src_value is None:
                item[f"delta_{key}"] = None
            else:
                item[f"delta_{key}"] = float(gen_value) - float(src_value)

        out.append(item)
    return out


def summarize_descriptor_pairs(row_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    descriptor_keys = ["heavy_atoms", "hetero_atoms", "ring_count", "rotatable_bonds", "tpsa", "anchor_distance"]
    for key in descriptor_keys:
        gen_values = [float(row[f"generated_{key}"]) for row in row_metrics if row.get(f"generated_{key}") is not None]
        src_values = [float(row[f"source_{key}"]) for row in row_metrics if row.get(f"source_{key}") is not None]
        delta_values = [float(row[f"delta_{key}"]) for row in row_metrics if row.get(f"delta_{key}") is not None]
        summary[key] = {
            "generated_mean": mean_or_none(gen_values),
            "generated_median": median_or_none(gen_values),
            "source_mean": mean_or_none(src_values),
            "source_median": median_or_none(src_values),
            "delta_mean": mean_or_none(delta_values),
            "delta_median": median_or_none(delta_values),
        }
    return summary


def build_per_source_quality(row_metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in row_metrics:
        grouped[str(row["source_dataset_index"])].append(row)

    out: list[dict[str, Any]] = []
    for source_idx, items in sorted(grouped.items(), key=lambda kv: int(kv[0])):
        generated_anchored = [str(item["generated_anchored_linker_smiles"]) for item in items if item.get("generated_anchored_linker_smiles")]
        generated_full = [str(item["generated_full_smiles"]) for item in items if item.get("generated_full_smiles")]

        source_sim_values = [float(item["source_similarity"]) for item in items if item.get("source_similarity") is not None]
        train_nn_values = [float(item["train_nn_similarity"]) for item in items if item.get("train_nn_similarity") is not None]
        token_overlap_values = [float(item["token_overlap_ratio"]) for item in items if item.get("token_overlap_ratio") is not None]

        fps = [morgan_fp(smiles) for smiles in generated_anchored]
        fps = [fp for fp in fps if fp is not None]
        pairwise_distances: list[float] = []
        for idx in range(len(fps)):
            for jdx in range(idx + 1, len(fps)):
                pairwise_distances.append(1.0 - float(DataStructs.TanimotoSimilarity(fps[idx], fps[jdx])))

        out.append(
            {
                "source_dataset_index": int(source_idx),
                "sample_id": items[0]["sample_id"],
                "protac_id": items[0]["protac_id"],
                "linker_id": items[0]["linker_id"],
                "num_requested": len(items),
                "decode_rate": sum(1 for item in items if item.get("generated_anchored_linker_smiles")) / max(len(items), 1),
                "assembly_rate": sum(1 for item in items if item.get("generated_full_smiles")) / max(len(items), 1),
                "unique_anchored": len(set(generated_anchored)),
                "unique_full": len(set(generated_full)),
                "source_similarity_mean": mean_or_none(source_sim_values),
                "train_nn_similarity_mean": mean_or_none(train_nn_values),
                "token_overlap_mean": mean_or_none(token_overlap_values),
                "same_token_length_rate": mean_or_none(float(bool(item["same_token_length"])) for item in items),
                "exact_source_match_rate": mean_or_none(float(bool(item["exact_source_match"])) for item in items),
                "exact_train_match_rate": mean_or_none(float(bool(item["exact_train_match"])) for item in items),
                "internal_diversity": mean_or_none(pairwise_distances),
            }
        )
    return out


def build_summary(
    row_metrics: Sequence[dict[str, Any]],
    per_source_rows: Sequence[dict[str, Any]],
    *,
    train_unique_count: int,
) -> dict[str, Any]:
    decoded = [row for row in row_metrics if row.get("generated_anchored_linker_smiles")]
    assembled = [row for row in row_metrics if row.get("generated_full_smiles")]

    source_sims = [float(row["source_similarity"]) for row in row_metrics if row.get("source_similarity") is not None]
    train_nn_sims = [float(row["train_nn_similarity"]) for row in row_metrics if row.get("train_nn_similarity") is not None]
    token_overlaps = [float(row["token_overlap_ratio"]) for row in row_metrics if row.get("token_overlap_ratio") is not None]
    internal_diversities = [float(row["internal_diversity"]) for row in per_source_rows if row.get("internal_diversity") is not None]

    return {
        "num_source_samples": len({str(row["source_dataset_index"]) for row in row_metrics}),
        "num_requested": len(row_metrics),
        "num_decoded_anchored": len(decoded),
        "num_assembled_full": len(assembled),
        "decode_rate": float(len(decoded) / max(len(row_metrics), 1)),
        "assembly_rate": float(len(assembled) / max(len(row_metrics), 1)),
        "unique_anchored": len({row["generated_anchored_linker_smiles"] for row in decoded}),
        "unique_full": len({row["generated_full_smiles"] for row in assembled}),
        "train_unique_linkers": int(train_unique_count),
        "exact_source_match_rate": mean_or_none(float(bool(row["exact_source_match"])) for row in row_metrics),
        "exact_train_match_rate": mean_or_none(float(bool(row["exact_train_match"])) for row in row_metrics),
        "exact_token_match_rate": mean_or_none(float(bool(row["exact_token_match"])) for row in row_metrics),
        "same_token_length_rate": mean_or_none(float(bool(row["same_token_length"])) for row in row_metrics),
        "source_similarity_mean": mean_or_none(source_sims),
        "source_similarity_median": median_or_none(source_sims),
        "source_similarity_min": min_or_none(source_sims),
        "source_similarity_max": max_or_none(source_sims),
        "train_nn_similarity_mean": mean_or_none(train_nn_sims),
        "train_nn_similarity_median": median_or_none(train_nn_sims),
        "train_nn_similarity_min": min_or_none(train_nn_sims),
        "train_nn_similarity_max": max_or_none(train_nn_sims),
        "token_overlap_mean": mean_or_none(token_overlaps),
        "token_overlap_median": median_or_none(token_overlaps),
        "token_overlap_min": min_or_none(token_overlaps),
        "token_overlap_max": max_or_none(token_overlaps),
        "internal_diversity_mean": mean_or_none(internal_diversities),
        "internal_diversity_median": median_or_none(internal_diversities),
        "internal_diversity_min": min_or_none(internal_diversities),
        "internal_diversity_max": max_or_none(internal_diversities),
        "descriptor_summary": summarize_descriptor_pairs(row_metrics),
    }


def write_csv(rows: Sequence[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(row_metrics: Sequence[dict[str, Any]], per_source_rows: Sequence[dict[str, Any]], out_path: Path) -> None:
    source_sims = [float(row["source_similarity"]) for row in row_metrics if row.get("source_similarity") is not None]
    train_nn_sims = [float(row["train_nn_similarity"]) for row in row_metrics if row.get("train_nn_similarity") is not None]
    token_overlaps = [float(row["token_overlap_ratio"]) for row in row_metrics if row.get("token_overlap_ratio") is not None]
    internal_diversities = [float(row["internal_diversity"]) for row in per_source_rows if row.get("internal_diversity") is not None]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = np.asarray(axes)

    if source_sims:
        axes[0, 0].hist(source_sims, bins=20, color="#1f77b4", alpha=0.85)
    axes[0, 0].set_title("Generated vs Source Morgan Similarity")
    axes[0, 0].set_xlabel("Tanimoto")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(alpha=0.25, linestyle=":")

    if train_nn_sims:
        axes[0, 1].hist(train_nn_sims, bins=20, color="#ff7f0e", alpha=0.85)
    axes[0, 1].set_title("Nearest-Train Morgan Similarity")
    axes[0, 1].set_xlabel("Tanimoto")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(alpha=0.25, linestyle=":")

    if token_overlaps:
        axes[1, 0].hist(token_overlaps, bins=20, color="#2ca02c", alpha=0.85)
    axes[1, 0].set_title("Source vs Generated Token Overlap")
    axes[1, 0].set_xlabel("Overlap ratio")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(alpha=0.25, linestyle=":")

    if internal_diversities:
        axes[1, 1].hist(internal_diversities, bins=20, color="#d62728", alpha=0.85)
    axes[1, 1].set_title("Per-Source Internal Diversity")
    axes[1, 1].set_xlabel("1 - mean pairwise Tanimoto")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(alpha=0.25, linestyle=":")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    descriptor = summary["descriptor_summary"]
    lines = [
        "# Token Generation Metrics",
        "",
        "## Core Metrics",
        "",
        f"- `num_source_samples`: {summary['num_source_samples']}",
        f"- `num_requested`: {summary['num_requested']}",
        f"- `decode_rate`: {summary['decode_rate']:.4f}",
        f"- `assembly_rate`: {summary['assembly_rate']:.4f}",
        f"- `unique_anchored`: {summary['unique_anchored']}",
        f"- `unique_full`: {summary['unique_full']}",
        f"- `exact_source_match_rate`: {summary['exact_source_match_rate']:.4f}",
        f"- `exact_train_match_rate`: {summary['exact_train_match_rate']:.4f}",
        f"- `exact_token_match_rate`: {summary['exact_token_match_rate']:.4f}",
        f"- `same_token_length_rate`: {summary['same_token_length_rate']:.4f}",
        "",
        "## Similarity and Diversity",
        "",
        f"- `source_similarity_mean`: {summary['source_similarity_mean']:.4f}",
        f"- `train_nn_similarity_mean`: {summary['train_nn_similarity_mean']:.4f}",
        f"- `token_overlap_mean`: {summary['token_overlap_mean']:.4f}",
        f"- `internal_diversity_mean`: {summary['internal_diversity_mean']:.4f}",
        "",
        "## Descriptor Drift",
        "",
        "| descriptor | generated_mean | source_mean | delta_mean |",
        "|---|---:|---:|---:|",
    ]
    for key in ["heavy_atoms", "hetero_atoms", "ring_count", "rotatable_bonds", "tpsa", "anchor_distance"]:
        item = descriptor[key]
        lines.append(
            f"| {key} | {item['generated_mean']:.3f} | {item['source_mean']:.3f} | {item['delta_mean']:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze token-generation outputs with similarity, diversity, and descriptor metrics")
    parser.add_argument("--all_generations_json", type=str, required=True)
    parser.add_argument("--train_weak_anchor_csv", type=str, default="outputs/weak_anchor_best/weak_anchor_dataset.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/token_generation_metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.all_generations_json)
    train_linkers, train_canonical = load_train_linkers(args.train_weak_anchor_csv)
    row_metrics = compute_row_metrics(rows, train_linkers=train_linkers, train_canonical=train_canonical)
    per_source_rows = build_per_source_quality(row_metrics)
    summary = build_summary(row_metrics, per_source_rows, train_unique_count=len(train_canonical))

    write_csv(row_metrics, out_dir / "row_metrics.csv")
    write_csv(per_source_rows, out_dir / "per_source_quality.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_overview(row_metrics, per_source_rows, out_dir / "overview.png")
    write_report(summary, out_dir / "REPORT.md")

    print(f"[done] rows={len(row_metrics)} out_dir={out_dir}", flush=True)
    print(
        f"[summary] decode_rate={summary['decode_rate']:.4f} assembly_rate={summary['assembly_rate']:.4f} "
        f"source_sim={summary['source_similarity_mean']:.4f} train_nn={summary['train_nn_similarity_mean']:.4f} "
        f"diversity={summary['internal_diversity_mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
