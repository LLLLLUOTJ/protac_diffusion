from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch


def test_train_full_token_embedding_with_init(tmp_path: Path) -> None:
    core_pt = tmp_path / "core.pt"
    coverage_csv = tmp_path / "coverage.csv"
    assignment_csv = tmp_path / "assign.csv"
    tokenized_csv = tmp_path / "tokenized.csv"
    out_dir = tmp_path / "out"

    core_tokens = ["*C*", "*O*"]
    core_tensor = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]], dtype=torch.float32)
    torch.save(
        {
            "embeddings": core_tensor,
            "token_to_id": {"*C*": 0, "*O*": 1},
            "id_to_token": core_tokens,
        },
        core_pt,
    )

    with coverage_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "token_smiles", "frequency"])
        writer.writeheader()
        writer.writerows(
            [
                {"rank": "1", "token_smiles": "*C*", "frequency": "100"},
                {"rank": "2", "token_smiles": "*O*", "frequency": "80"},
                {"rank": "3", "token_smiles": "*C(*)C", "frequency": "5"},
            ]
        )

    with assignment_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "low_token",
                "low_freq",
                "assigned_core_token",
                "assigned_core_freq",
                "confidence",
                "score",
                "tanimoto",
                "descriptor_similarity",
                "margin_vs_second",
                "low_attachment_count",
                "core_attachment_count",
                "low_motif_class",
                "core_motif_class",
                "rule_flags_json",
                "top3_candidates_json",
                "uncertain_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "low_token": "*C(*)C",
                "low_freq": "5",
                "assigned_core_token": "*C*",
                "assigned_core_freq": "100",
                "confidence": "high",
                "score": "0.9",
                "tanimoto": "0.9",
                "descriptor_similarity": "0.8",
                "margin_vs_second": "0.2",
                "low_attachment_count": "2",
                "core_attachment_count": "2",
                "low_motif_class": "aliphatic",
                "core_motif_class": "aliphatic",
                "rule_flags_json": "[]",
                "top3_candidates_json": "[]",
                "uncertain_reason": "",
            }
        )

    with tokenized_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "linker_id", "sample_weight", "token_smiles_list_json"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "linker_id": "l1",
                "sample_weight": "1.0",
                "token_smiles_list_json": json.dumps(["*C*", "*C(*)C", "*O*"]),
            }
        )
        writer.writerow(
            {
                "sample_id": "s2",
                "linker_id": "l2",
                "sample_weight": "1.0",
                "token_smiles_list_json": json.dumps(["*O*", "*C*", "*O*"]),
            }
        )

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "train_full_token_embedding_with_init.py"),
        "--tokenized_csv",
        str(tokenized_csv),
        "--coverage_csv",
        str(coverage_csv),
        "--assignment_csv",
        str(assignment_csv),
        "--core_embedding_pt",
        str(core_pt),
        "--out_dir",
        str(out_dir),
        "--epochs",
        "8",
        "--batch_size",
        "16",
        "--window_size",
        "1",
        "--negative_samples",
        "2",
        "--learning_rate",
        "0.03",
        "--device",
        "cpu",
        "--log_every",
        "4",
    ]
    subprocess.run(cmd, check=True)

    vocab_json = out_dir / "token_vocab.json"
    emb_pt = out_dir / "token_embeddings.pt"
    init_csv = out_dir / "token_init_sources.csv"
    summary_json = out_dir / "training_summary.json"
    assert vocab_json.exists()
    assert emb_pt.exists()
    assert init_csv.exists()
    assert summary_json.exists()

    vocab = json.loads(vocab_json.read_text(encoding="utf-8"))
    assert "*C(*)C" in vocab["token_to_id"]

    init_rows = list(csv.DictReader(init_csv.open("r", encoding="utf-8", newline="")))
    low_row = next(r for r in init_rows if r["token"] == "*C(*)C")
    assert low_row["init_source"] == "mapped_high"

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["vocab_size"] == 3
