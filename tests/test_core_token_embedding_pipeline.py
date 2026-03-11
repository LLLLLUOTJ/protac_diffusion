from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def test_core_token_embedding_train_and_analyze(tmp_path: Path) -> None:
    tokenized_csv = tmp_path / "tokenized.csv"
    out_dir = tmp_path / "emb_out"
    neighbor_json = tmp_path / "neighbors.json"

    rows = [
        {
            "sample_id": "s1",
            "linker_id": "l1",
            "sample_weight": "1.0",
            "token_smiles_list_json": json.dumps(["*C*", "*O*", "*C*"]),
        },
        {
            "sample_id": "s2",
            "linker_id": "l2",
            "sample_weight": "1.0",
            "token_smiles_list_json": json.dumps(["*N*", "*C(*)=O", "*N*"]),
        },
        {
            "sample_id": "s3",
            "linker_id": "l3",
            "sample_weight": "1.0",
            "token_smiles_list_json": json.dumps(["*c1ccc(*)cc1", "*C*", "*c1ccc(*)cc1"]),
        },
    ]
    with tokenized_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    train_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "train_core_token_embedding.py"),
        "--tokenized_csv",
        str(tokenized_csv),
        "--out_dir",
        str(out_dir),
        "--embedding_dim",
        "8",
        "--epochs",
        "20",
        "--batch_size",
        "32",
        "--window_size",
        "1",
        "--negative_samples",
        "2",
        "--learning_rate",
        "0.05",
        "--device",
        "cpu",
    ]
    subprocess.run(train_cmd, check=True)

    vocab_json = out_dir / "token_vocab.json"
    emb_pt = out_dir / "token_embeddings.pt"
    emb_npy = out_dir / "token_embeddings.npy"
    summary_json = out_dir / "training_summary.json"
    assert vocab_json.exists()
    assert emb_pt.exists()
    assert emb_npy.exists()
    assert summary_json.exists()

    vocab_data = json.loads(vocab_json.read_text(encoding="utf-8"))
    assert "*C*" in vocab_data["token_to_id"]
    assert "*O*" in vocab_data["token_to_id"]
    assert "*N*" in vocab_data["token_to_id"]
    assert "O" not in vocab_data["token_to_id"]

    obj = torch.load(emb_pt, map_location="cpu")
    arr = obj["embeddings"].detach().cpu().numpy()
    arr_npy = np.load(emb_npy)
    assert arr.shape[1] == 8
    assert arr_npy.shape == arr.shape

    analyze_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "analyze_token_neighbors.py"),
        "--vocab_json",
        str(vocab_json),
        "--embeddings_pt",
        str(emb_pt),
        "--out_json",
        str(neighbor_json),
        "--query_tokens",
        "*C*,*O*",
        "--top_k",
        "2",
    ]
    subprocess.run(analyze_cmd, check=True)

    report = json.loads(neighbor_json.read_text(encoding="utf-8"))
    assert report["vocab_size"] >= 4
    assert "*C*" in report["neighbors"]
    assert len(report["neighbors"]["*C*"]) == 2
