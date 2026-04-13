"""Compare descriptor distributions between two generated SMILES datasets.

Example:
    python compare_generated_molecule_distributions.py \
      --a-path outputs/server_sync/2026-04-07/linker_token_eval.all_generations.json \
      --a-smiles-col generated_anchored_linker_smiles \
      --a-label token \
      --b-path outputs/server_sync/2026-04-07/linkinvent_compare/sampling_32x4.csv \
      --b-smiles-col Linker \
      --b-label linkinvent \
      --output-json outputs/server_sync/2026-04-09/token_vs_linkinvent/summary.json \
      --output-plot outputs/server_sync/2026-04-09/token_vs_linkinvent/distributions.png
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

from analyze_molecule_distributions import build_dataset_summary
from eval_molecules import load_smiles_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two generated SMILES distributions")
    parser.add_argument("--a-path", required=True)
    parser.add_argument("--a-smiles-col", required=True)
    parser.add_argument("--a-label", default="model_a")
    parser.add_argument("--b-path", required=True)
    parser.add_argument("--b-smiles-col", required=True)
    parser.add_argument("--b-label", default="model_b")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-plot", required=True)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


METRICS: list[tuple[str, str]] = [
    ("length_heavy_atoms", "Length (heavy atoms)"),
    ("molecular_weight", "MW"),
    ("rotatable_bonds", "Rotatable bonds"),
    ("qed", "QED"),
    ("sa", "SA"),
]


def _descriptor_delta(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
) -> dict[str, dict[str, float] | None]:
    out: dict[str, dict[str, float] | None] = {}
    for key, _ in METRICS:
        left = summary_a["descriptor_summary"].get(key)
        right = summary_b["descriptor_summary"].get(key)
        if left is None or right is None:
            out[key] = None
            continue
        out[key] = {
            "mean_delta_b_minus_a": float(right["mean"] - left["mean"]),
            "median_delta_b_minus_a": float(right["median"] - left["median"]),
        }
    return out


def _write_descriptor_csv(
    path: Path,
    label_a: str,
    label_b: str,
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, records in ((label_a, records_a), (label_b, records_b)):
            for record in records:
                writer.writerow({"dataset": label, **record})


def _plot(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = list(axes.flatten())
    colors = ["#4C78A8", "#F58518"]

    for axis, (key, title) in zip(axes_flat, METRICS):
        values_a = [float(record[key]) for record in records_a]
        values_b = [float(record[key]) for record in records_b]
        axis.hist(values_a, bins=24, alpha=0.55, label=label_a, color=colors[0], density=True)
        axis.hist(values_b, bins=24, alpha=0.55, label=label_b, color=colors[1], density=True)
        axis.set_title(title)
        axis.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)

    axes_flat[-1].axis("off")
    fig.suptitle(f"{label_a} vs {label_b} Generated Molecule Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    smiles_a = load_smiles_file(args.a_path, smiles_col=args.a_smiles_col)
    smiles_b = load_smiles_file(args.b_path, smiles_col=args.b_smiles_col)

    summary_a, records_a = build_dataset_summary(args.a_label, smiles_a)
    summary_b, records_b = build_dataset_summary(args.b_label, smiles_b)

    results = {
        "a_path": str(args.a_path),
        "a_smiles_col": args.a_smiles_col,
        "a_label": args.a_label,
        "b_path": str(args.b_path),
        "b_smiles_col": args.b_smiles_col,
        "b_label": args.b_label,
        "length_definition": "heavy atom count",
        args.a_label: summary_a,
        args.b_label: summary_b,
        "descriptor_delta": _descriptor_delta(summary_a, summary_b),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot(records_a, records_b, args.a_label, args.b_label, Path(args.output_plot))

    if args.output_csv:
        _write_descriptor_csv(Path(args.output_csv), args.a_label, args.b_label, records_a, records_b)

    print(f"[done] summary={args.output_json}")
    print(f"[done] plot={args.output_plot}")
    if args.output_csv:
        print(f"[done] csv={args.output_csv}")


if __name__ == "__main__":
    main()
