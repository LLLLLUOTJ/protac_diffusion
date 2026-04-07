from __future__ import annotations

import csv
from pathlib import Path

from analyze_molecule_distributions import (
    build_dataset_summary,
    build_results,
    plot_distributions,
    write_descriptor_csv,
)


def write_csv(path: Path, column: str, values: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[column])
        writer.writeheader()
        for value in values:
            writer.writerow({column: value})


def test_build_dataset_summary_handles_invalid_smiles() -> None:
    summary, records = build_dataset_summary("generated", ["CCO", "bad_smiles", "c1ccccc1"])
    assert summary["total_count"] == 3
    assert summary["valid_count"] == 2
    assert summary["validity"] == 2.0 / 3.0
    assert summary["unique_valid_count"] == 2
    assert len(records) == 2
    assert summary["descriptor_summary"]["length_heavy_atoms"] is not None


def test_cli_outputs_summary_plot_and_csv(tmp_path: Path) -> None:
    generated_csv = tmp_path / "generated.csv"
    train_csv = tmp_path / "train.csv"
    write_csv(generated_csv, "smiles", ["CCO", "CCN", "CCCl"])
    write_csv(train_csv, "smiles", ["CCO", "c1ccccc1", "CCBr"])

    generated_summary, generated_records = build_dataset_summary("generated", ["CCO", "CCN", "CCCl"])
    train_summary, train_records = build_dataset_summary("train", ["CCO", "c1ccccc1", "CCBr"])

    results = build_results(
        generated_path=str(generated_csv),
        train_path=str(train_csv),
        smiles_col="smiles",
        train_smiles_col="smiles",
        generated_summary=generated_summary,
        train_summary=train_summary,
    )

    assert results["length_definition"] == "heavy atom count"
    assert results["generated"]["valid_count"] == 3
    assert results["train"]["valid_count"] == 3
    assert "molecular_weight" in results["descriptor_delta"]
    assert len(generated_records) == 3
    assert len(train_records) == 3

    plot_path = tmp_path / "distributions.png"
    csv_path = tmp_path / "descriptors.csv"
    plot_distributions(generated_records, train_records, plot_path)
    write_descriptor_csv(csv_path, generated_records, train_records)

    assert plot_path.exists()
    assert csv_path.exists()
