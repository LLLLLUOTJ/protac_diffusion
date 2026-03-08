from __future__ import annotations

from evaluate_linker_generation import build_per_source_rows, summarize_rows


def test_summarize_rows_counts_decode_and_assembly() -> None:
    rows = [
        {
            "source_dataset_index": 0,
            "generated_anchored_linker_smiles": "C[*:1].[*:2]",
            "generated_full_smiles": None,
        },
        {
            "source_dataset_index": 0,
            "generated_anchored_linker_smiles": "CC[*:1].[*:2]",
            "generated_full_smiles": "CCC",
        },
        {
            "source_dataset_index": 1,
            "generated_anchored_linker_smiles": None,
            "generated_full_smiles": None,
        },
    ]
    summary = summarize_rows(rows)
    assert summary["num_source_samples"] == 2
    assert summary["num_requested"] == 3
    assert summary["num_decoded_anchored"] == 2
    assert summary["num_assembled_full"] == 1
    assert abs(summary["decode_rate"] - (2 / 3)) < 1e-8
    assert abs(summary["assembly_rate"] - (1 / 3)) < 1e-8


def test_build_per_source_rows_groups_by_source() -> None:
    rows = [
        {
            "source_dataset_index": 2,
            "sample_id": "a",
            "protac_id": "p1",
            "linker_id": "l1",
            "source_anchored_linker_smiles": "src1",
            "source_left_fragment_smiles": "L1",
            "source_right_fragment_smiles": "R1",
            "generated_anchored_linker_smiles": "gen1",
            "generated_full_smiles": None,
        },
        {
            "source_dataset_index": 2,
            "sample_id": "a",
            "protac_id": "p1",
            "linker_id": "l1",
            "source_anchored_linker_smiles": "src1",
            "source_left_fragment_smiles": "L1",
            "source_right_fragment_smiles": "R1",
            "generated_anchored_linker_smiles": "gen1",
            "generated_full_smiles": "full1",
        },
        {
            "source_dataset_index": 3,
            "sample_id": "b",
            "protac_id": "p2",
            "linker_id": "l2",
            "source_anchored_linker_smiles": "src2",
            "source_left_fragment_smiles": "L2",
            "source_right_fragment_smiles": "R2",
            "generated_anchored_linker_smiles": None,
            "generated_full_smiles": None,
        },
    ]
    grouped = build_per_source_rows(rows)
    assert len(grouped) == 2
    first = grouped[0]
    assert first["source_dataset_index"] == 2
    assert first["num_requested"] == 2
    assert first["num_decoded_anchored"] == 2
    assert first["num_assembled_full"] == 1
    assert first["unique_anchored"] == 1
