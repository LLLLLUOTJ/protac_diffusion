from __future__ import annotations

from analyze_token_generation_metrics import (
    build_per_source_quality,
    build_summary,
    canonicalize_smiles,
    compute_row_metrics,
)


def test_compute_row_metrics_and_summary() -> None:
    rows = [
        {
            "source_dataset_index": 0,
            "sample_id": "s1",
            "protac_id": "p1",
            "linker_id": "l1",
            "source_anchored_linker_smiles": "[*:1]CC[*:2]",
            "generated_anchored_linker_smiles": "[*:1]CC[*:2]",
            "generated_full_smiles": "CCC",
            "source_oriented_token_smiles": "[\"C([*:1])[*:2]\", \"C([*:1])[*:2]\"]",
            "generated_oriented_token_smiles": "[\"C([*:1])[*:2]\", \"C([*:1])[*:2]\"]",
        },
        {
            "source_dataset_index": 0,
            "sample_id": "s1",
            "protac_id": "p1",
            "linker_id": "l1",
            "source_anchored_linker_smiles": "[*:1]CC[*:2]",
            "generated_anchored_linker_smiles": "[*:1]CO[*:2]",
            "generated_full_smiles": "CCO",
            "source_oriented_token_smiles": "[\"C([*:1])[*:2]\", \"C([*:1])[*:2]\"]",
            "generated_oriented_token_smiles": "[\"C([*:1])[*:2]\", \"O([*:1])[*:2]\"]",
        },
    ]
    train_linkers = ["[*:1]CC[*:2]", "[*:1]CO[*:2]"]
    train_canonical = {canonicalize_smiles(smiles) for smiles in train_linkers}

    row_metrics = compute_row_metrics(rows, train_linkers=train_linkers, train_canonical=train_canonical)
    assert len(row_metrics) == 2
    assert row_metrics[0]["exact_source_match"] is True
    assert row_metrics[1]["exact_source_match"] is False
    assert row_metrics[0]["exact_train_match"] is True
    assert row_metrics[1]["exact_train_match"] is True
    assert row_metrics[0]["exact_token_match"] is True
    assert row_metrics[1]["exact_token_match"] is False
    assert row_metrics[0]["same_token_length"] is True

    per_source = build_per_source_quality(row_metrics)
    assert len(per_source) == 1
    assert per_source[0]["unique_anchored"] == 2
    assert per_source[0]["decode_rate"] == 1.0

    summary = build_summary(row_metrics, per_source, train_unique_count=2)
    assert summary["num_requested"] == 2
    assert summary["decode_rate"] == 1.0
    assert summary["assembly_rate"] == 1.0
    assert abs(summary["exact_source_match_rate"] - 0.5) < 1e-8
    assert abs(summary["exact_token_match_rate"] - 0.5) < 1e-8
