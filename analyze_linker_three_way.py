"""Three-way linker distribution comparison.

Compares:
- raw linker library
- weak-anchor training linkers
- generated linkers

This is a lightweight helper to understand how weak-supervision filtering and
generation shift the linker distribution relative to the original linker pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_molecule_distributions import build_dataset_summary
from eval_molecules import load_smiles_file


def plot_three_way_distributions(
    records_by_label: dict[str, list[dict[str, Any]]],
    output_plot: Path,
) -> None:
    metrics = [
        ("length_heavy_atoms", "Length (heavy atoms)"),
        ("molecular_weight", "MW"),
        ("rotatable_bonds", "Rotatable bonds"),
        ("qed", "QED"),
        ("sa", "SA"),
    ]
    colors = {
        "raw_linker": "#4C78A8",
        "weak_anchor": "#54A24B",
        "generated": "#F58518",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = list(axes.flatten())

    for axis, (key, title) in zip(axes_flat, metrics):
        for label, records in records_by_label.items():
            values = [float(record[key]) for record in records]
            axis.hist(
                values,
                bins=24,
                alpha=0.45,
                density=True,
                label=label,
                color=colors.get(label),
            )
        axis.set_title(title)
        axis.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)

    axes_flat[-1].axis("off")
    fig.suptitle("Raw vs Weak-Anchor vs Generated Linker Distributions", fontsize=14)
    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)


def delta_summary(summary_a: dict[str, Any], summary_b: dict[str, Any]) -> dict[str, dict[str, float] | None]:
    out: dict[str, dict[str, float] | None] = {}
    for key in ["length_heavy_atoms", "molecular_weight", "rotatable_bonds", "qed", "sa"]:
        a = summary_a["descriptor_summary"].get(key)
        b = summary_b["descriptor_summary"].get(key)
        if a is None or b is None:
            out[key] = None
            continue
        out[key] = {
            "mean_delta": float(a["mean"] - b["mean"]),
            "median_delta": float(a["median"] - b["median"]),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw, weak-anchor, and generated linker distributions")
    parser.add_argument("--raw", required=True, help="raw linker CSV/JSON path")
    parser.add_argument("--weak", required=True, help="weak-anchor linker CSV/JSON path")
    parser.add_argument("--generated", required=True, help="generated linker CSV/JSON path")
    parser.add_argument("--raw-smiles-col", default="smiles", help="raw smiles column")
    parser.add_argument("--weak-smiles-col", default="smiles", help="weak smiles column")
    parser.add_argument("--generated-smiles-col", default="smiles", help="generated smiles column")
    parser.add_argument("--output-json", required=True, help="summary JSON output path")
    parser.add_argument("--output-plot", required=True, help="distribution plot output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_smiles = load_smiles_file(args.raw, smiles_col=args.raw_smiles_col)
    weak_smiles = load_smiles_file(args.weak, smiles_col=args.weak_smiles_col)
    generated_smiles = load_smiles_file(args.generated, smiles_col=args.generated_smiles_col)

    raw_summary, raw_records = build_dataset_summary("raw_linker", raw_smiles)
    weak_summary, weak_records = build_dataset_summary("weak_anchor", weak_smiles)
    gen_summary, gen_records = build_dataset_summary("generated", generated_smiles)

    output = {
        "raw_path": str(args.raw),
        "weak_path": str(args.weak),
        "generated_path": str(args.generated),
        "length_definition": "heavy atom count",
        "raw_linker": raw_summary,
        "weak_anchor": weak_summary,
        "generated": gen_summary,
        "delta_weak_minus_raw": delta_summary(weak_summary, raw_summary),
        "delta_generated_minus_raw": delta_summary(gen_summary, raw_summary),
        "delta_generated_minus_weak": delta_summary(gen_summary, weak_summary),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_three_way_distributions(
        {
            "raw_linker": raw_records,
            "weak_anchor": weak_records,
            "generated": gen_records,
        },
        Path(args.output_plot),
    )

    print("[analyze] raw vs weak-anchor vs generated", flush=True)
    for label, summary in [
        ("raw_linker", raw_summary),
        ("weak_anchor", weak_summary),
        ("generated", gen_summary),
    ]:
        print(
            f"{label}: total={summary['total_count']} valid={summary['valid_count']} "
            f"unique={summary['unique_valid_count']} qed={summary['mean_qed']:.4f} sa={summary['mean_sa']:.4f}",
            flush=True,
        )
    print(f"plot: {args.output_plot}", flush=True)


if __name__ == "__main__":
    main()
