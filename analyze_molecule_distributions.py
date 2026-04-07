"""Compare generated and training SMILES distributions with lightweight RDKit metrics.

This script evaluates generated molecules and a reference/training set side by side.
It computes validity/uniqueness-style summaries for both sets, extracts a few simple
descriptors, and saves overlay distribution plots.

Metrics/plots:
- length (heavy atom count)
- molecular weight
- rotatable bonds
- QED
- SA score

Example:
    python analyze_molecule_distributions.py \
      --generated outputs/linker_token_eval/all_generations.json \
      --train outputs/weak_anchor_best/weak_anchor_dataset.csv \
      --smiles-col generated_anchored_linker_smiles \
      --train-smiles-col anchored_linker_smiles \
      --output-json outputs/linker_smiles_compare/summary.json \
      --output-plot outputs/linker_smiles_compare/distributions.png \
      --output-csv outputs/linker_smiles_compare/descriptors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED

from eval_molecules import (
    SASCORER,
    compute_property_scores,
    compute_uniqueness,
    compute_validity,
    load_smiles_file,
)


def compute_descriptor_record(mol: Chem.Mol, canonical_smiles: str) -> dict[str, Any]:
    return {
        "canonical_smiles": canonical_smiles,
        "length_heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "qed": float(QED.qed(mol)),
        "sa": float(SASCORER.calculateScore(mol)),
    }


def build_dataset_summary(name: str, smiles_list: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validity_result = compute_validity(smiles_list)
    uniqueness_result = compute_uniqueness(validity_result["valid_canonical_smiles"])
    property_result = compute_property_scores(validity_result["valid_mols"])

    records = [
        compute_descriptor_record(mol, canonical_smiles)
        for mol, canonical_smiles in zip(
            validity_result["valid_mols"],
            validity_result["valid_canonical_smiles"],
            strict=True,
        )
    ]

    descriptor_summary: dict[str, dict[str, float] | None] = {}
    for key in ["length_heavy_atoms", "molecular_weight", "rotatable_bonds", "qed", "sa"]:
        values = [float(record[key]) for record in records]
        if not values:
            descriptor_summary[key] = None
            continue
        values_sorted = sorted(values)
        mid = len(values_sorted) // 2
        if len(values_sorted) % 2 == 0:
            median = (values_sorted[mid - 1] + values_sorted[mid]) / 2.0
        else:
            median = values_sorted[mid]
        descriptor_summary[key] = {
            "mean": float(sum(values) / len(values)),
            "median": float(median),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    summary = {
        "name": name,
        "total_count": int(validity_result["total_count"]),
        "valid_count": int(validity_result["valid_count"]),
        "validity": float(validity_result["validity"]),
        "unique_valid_count": int(uniqueness_result["unique_valid_count"]),
        "uniqueness": float(uniqueness_result["uniqueness"]),
        "mean_qed": property_result["mean_qed"],
        "mean_sa": property_result["mean_sa"],
        "descriptor_summary": descriptor_summary,
    }
    return summary, records


def write_descriptor_csv(
    output_csv: Path,
    generated_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "dataset",
        "canonical_smiles",
        "length_heavy_atoms",
        "molecular_weight",
        "rotatable_bonds",
        "qed",
        "sa",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_name, records in (("generated", generated_records), ("train", train_records)):
            for record in records:
                writer.writerow({"dataset": dataset_name, **record})


def plot_distributions(
    generated_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    output_plot: Path,
) -> None:
    metrics = [
        ("length_heavy_atoms", "Length (heavy atoms)"),
        ("molecular_weight", "MW"),
        ("rotatable_bonds", "Rotatable bonds"),
        ("qed", "QED"),
        ("sa", "SA"),
    ]

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = list(axes.flatten())

    for axis, (key, title) in zip(axes_flat, metrics):
        generated_values = [float(record[key]) for record in generated_records]
        train_values = [float(record[key]) for record in train_records]
        axis.hist(train_values, bins=24, alpha=0.55, label="train", color="#4C78A8", density=True)
        axis.hist(generated_values, bins=24, alpha=0.55, label="generated", color="#F58518", density=True)
        axis.set_title(title)
        axis.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)

    axes_flat[-1].axis("off")
    fig.suptitle("Generated vs Train Molecule Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_results(
    *,
    generated_path: str,
    train_path: str,
    smiles_col: str,
    train_smiles_col: str,
    generated_summary: dict[str, Any],
    train_summary: dict[str, Any],
) -> dict[str, Any]:
    descriptor_delta: dict[str, dict[str, float] | None] = {}
    for key in ["length_heavy_atoms", "molecular_weight", "rotatable_bonds", "qed", "sa"]:
        generated_descriptor = generated_summary["descriptor_summary"].get(key)
        train_descriptor = train_summary["descriptor_summary"].get(key)
        if generated_descriptor is None or train_descriptor is None:
            descriptor_delta[key] = None
            continue
        descriptor_delta[key] = {
            "mean_delta": float(generated_descriptor["mean"] - train_descriptor["mean"]),
            "median_delta": float(generated_descriptor["median"] - train_descriptor["median"]),
        }

    return {
        "generated_path": generated_path,
        "train_path": train_path,
        "smiles_col": smiles_col,
        "train_smiles_col": train_smiles_col,
        "length_definition": "heavy atom count",
        "generated": generated_summary,
        "train": train_summary,
        "descriptor_delta": descriptor_delta,
    }


def print_summary(results: dict[str, Any]) -> None:
    print("[analyze] generated vs train molecule distributions", flush=True)
    print(f"generated: {results['generated_path']}", flush=True)
    print(f"train: {results['train_path']}", flush=True)
    for dataset_name in ["generated", "train"]:
        summary = results[dataset_name]
        print(
            f"{dataset_name}: total={summary['total_count']} valid={summary['valid_count']} "
            f"validity={summary['validity']:.4f} unique={summary['unique_valid_count']} "
            f"uniqueness={summary['uniqueness']:.4f} qed={summary['mean_qed']:.4f} sa={summary['mean_sa']:.4f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated and training SMILES distributions")
    parser.add_argument("--generated", required=True, help="generated CSV or JSON file")
    parser.add_argument("--train", required=True, help="training/reference CSV or JSON file")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES field/column in generated file")
    parser.add_argument(
        "--train-smiles-col",
        default=None,
        help="SMILES field/column in train file (defaults to --smiles-col)",
    )
    parser.add_argument("--output-json", required=True, help="summary JSON output path")
    parser.add_argument("--output-plot", required=True, help="distribution plot output path")
    parser.add_argument("--output-csv", default=None, help="optional per-molecule descriptor CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_smiles_col = args.train_smiles_col or args.smiles_col

    generated_smiles = load_smiles_file(args.generated, smiles_col=args.smiles_col)
    train_smiles = load_smiles_file(args.train, smiles_col=train_smiles_col)

    generated_summary, generated_records = build_dataset_summary("generated", generated_smiles)
    train_summary, train_records = build_dataset_summary("train", train_smiles)

    results = build_results(
        generated_path=str(args.generated),
        train_path=str(args.train),
        smiles_col=args.smiles_col,
        train_smiles_col=train_smiles_col,
        generated_summary=generated_summary,
        train_summary=train_summary,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_distributions(generated_records, train_records, Path(args.output_plot))

    if args.output_csv:
        write_descriptor_csv(Path(args.output_csv), generated_records, train_records)

    print_summary(results)
    print(f"plot: {args.output_plot}", flush=True)
    if args.output_csv:
        print(f"csv: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
