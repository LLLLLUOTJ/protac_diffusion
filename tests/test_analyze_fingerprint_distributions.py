from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyze_fingerprint_distributions import (
    analyze_dataset,
    apply_reference_metrics,
    build_special_comparison_sets,
    load_smiles_records,
    normalize_input_smiles,
    summary_dataframe,
)


def test_load_smiles_records_from_csv_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"linker_smiles": ["CCO", "invalid"]}).to_csv(csv_path, index=False)

    json_path = tmp_path / "generated.json"
    json_path.write_text(json.dumps([{"generated_smiles": "CCN"}, {"generated_smiles": "CCC"}]), encoding="utf-8")

    csv_smiles, csv_field = load_smiles_records(csv_path)
    json_smiles, json_field = load_smiles_records(json_path)

    assert csv_smiles == ["CCO", "invalid"]
    assert csv_field == "linker_smiles"
    assert json_smiles == ["CCN", "CCC"]
    assert json_field == "generated_smiles"


def test_normalize_input_smiles_converts_raw_anchor_notation() -> None:
    assert normalize_input_smiles("[R1]CCO[R2]", smiles_field="Smiles_R") == "[*:1]CCO[*:2]"
    assert normalize_input_smiles("CCO", smiles_field="smiles") == "CCO"


def test_analyze_dataset_and_reference_metrics(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    gen_path = tmp_path / "generated.json"

    pd.DataFrame({"anchored_linker_smiles": ["CCO", "CCN", "invalid"]}).to_csv(train_path, index=False)
    gen_path.write_text(
        json.dumps(
            [
                {"generated_anchored_linker_smiles": "CCO"},
                {"generated_anchored_linker_smiles": "CCC"},
                {"generated_anchored_linker_smiles": "broken"},
            ]
        ),
        encoding="utf-8",
    )

    train = analyze_dataset({"name": "train", "path": str(train_path), "role": "train"})
    generated = analyze_dataset({"name": "gen_16d", "path": str(gen_path), "role": "generated"})

    apply_reference_metrics([train, generated], train)
    summary_df = summary_dataframe([train, generated])

    train_row = summary_df[summary_df["dataset_name"] == "train"].iloc[0]
    gen_row = summary_df[summary_df["dataset_name"] == "gen_16d"].iloc[0]

    assert train.parsed_samples == 2
    assert generated.parsed_samples == 2
    assert float(gen_row["uniqueness"]) == 1.0
    assert float(gen_row["novelty"]) == 0.5
    assert gen_row["mean_nn_tanimoto"] is not None
    assert train_row["internal_diversity"] is not None


def test_build_special_comparison_sets_detects_expected_groups(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(train_path, index=False)

    datasets = [
        analyze_dataset({"name": "train", "path": str(train_path), "role": "train"}),
        analyze_dataset({"name": "raw_linker", "path": str(train_path), "role": "raw"}),
        analyze_dataset({"name": "gen_16d", "path": str(train_path), "role": "generated"}),
        analyze_dataset({"name": "gen_32d", "path": str(train_path), "role": "generated"}),
        analyze_dataset({"name": "gen_pad_suffix_ce015", "path": str(train_path), "role": "generated"}),
        analyze_dataset({"name": "gen_pad_suffix_ce005", "path": str(train_path), "role": "generated"}),
        analyze_dataset({"name": "link_invent", "path": str(train_path), "role": "baseline"}),
    ]

    apply_reference_metrics(datasets, datasets[0])
    summary_df = summary_dataframe(datasets)
    special_sets = build_special_comparison_sets(datasets, summary_df)

    assert "train_vs_16d_vs_32d" in special_sets
    assert "train_vs_pad_suffix_versions" in special_sets
    assert "train_vs_baseline_vs_best_model" in special_sets
    assert "train_vs_raw_vs_best_model" in special_sets
    assert "train_vs_raw_vs_gen_16d" in special_sets
