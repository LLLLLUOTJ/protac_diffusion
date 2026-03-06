from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from data.anchored_tensor_dataset import AnchoredTensorPTDataset
from data.linker_anchor_dataset import LinkerAnchorDataset
from generate_anchored_linker import select_distinct_anchors
from models.anchor_gnn import AnchorGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report train/val split and validation accuracy for anchor_gnn checkpoint")
    parser.add_argument("--ckpt", type=str, default="checkpoints/anchor_gnn.pt", help="trained checkpoint path")
    parser.add_argument("--out-dir", type=str, default="reports/anchor_gnn_split", help="report output directory")
    return parser.parse_args()


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_n = max(1, int(n * val_ratio))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:] or val_idx
    return train_idx, val_idx


def build_dataset_from_train_config(train_cfg: Dict[str, Any]) -> Dataset:
    tensor_pt = train_cfg.get("tensor_pt")
    max_samples = train_cfg.get("max_samples")
    if tensor_pt:
        return AnchoredTensorPTDataset(pt_path=tensor_pt, max_samples=max_samples)
    return LinkerAnchorDataset(csv_path=train_cfg["csv"], max_samples=max_samples)


def find_anchor_positions(node_type: torch.Tensor) -> Tuple[int | None, int | None]:
    left = (node_type == 1).nonzero(as_tuple=False).view(-1)
    right = (node_type == 2).nonzero(as_tuple=False).view(-1)
    left_idx = int(left[0].item()) if left.numel() == 1 else None
    right_idx = int(right[0].item()) if right.numel() == 1 else None
    return left_idx, right_idx


def evaluate_dataset(
    model: AnchorGNN,
    dataset: Dataset,
    indices: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    raw_ordered_success = 0
    ordered_success = 0
    unordered_success = 0
    total = 0

    for dataset_index in indices:
        sample = dataset[dataset_index]
        x = sample.x.float()
        edge_index = sample.edge_index.long()
        y = sample.y.long()

        with torch.no_grad():
            logits = model(x, edge_index)
        raw_pred_l = int(torch.argmax(logits[:, 1]).item())
        raw_pred_r = int(torch.argmax(logits[:, 2]).item())
        pred_l, pred_r = select_distinct_anchors(logits)
        true_l, true_r = find_anchor_positions(y)

        raw_ordered_ok = int((raw_pred_l, raw_pred_r) == (true_l, true_r))
        ordered_ok = int((pred_l, pred_r) == (true_l, true_r))
        unordered_ok = int({pred_l, pred_r} == {true_l, true_r})
        raw_ordered_success += raw_ordered_ok
        ordered_success += ordered_ok
        unordered_success += unordered_ok
        total += 1

        rows.append(
            {
                "dataset_index": dataset_index,
                "compound_id": getattr(sample, "compound_id", ""),
                "smiles": getattr(sample, "smiles", ""),
                "smiles_r": getattr(sample, "smiles_r", ""),
                "true_anchor_l": true_l,
                "true_anchor_r": true_r,
                "raw_pred_anchor_l": raw_pred_l,
                "raw_pred_anchor_r": raw_pred_r,
                "pred_anchor_l": pred_l,
                "pred_anchor_r": pred_r,
                "raw_ordered_success": raw_ordered_ok,
                "ordered_success": ordered_ok,
                "unordered_success": unordered_ok,
            }
        )

    metrics = {
        "count": float(total),
        "raw_ordered_success_count": float(raw_ordered_success),
        "ordered_success_count": float(ordered_success),
        "unordered_success_count": float(unordered_success),
        "raw_ordered_success_rate": raw_ordered_success / max(total, 1),
        "ordered_success_rate": ordered_success / max(total, 1),
        "unordered_success_rate": unordered_success / max(total, 1),
    }
    return rows, metrics


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    ckpt_path: str,
    train_cfg: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    val_metrics: Dict[str, float],
) -> None:
    lines = [
        "# Anchor GNN Split Report",
        "",
        f"Checkpoint: `{ckpt_path}`",
        f"Source: `{train_cfg.get('tensor_pt') or train_cfg.get('csv')}`",
        f"Seed: `{train_cfg.get('seed')}`",
        f"Validation ratio: `{train_cfg.get('val_ratio')}`",
        "",
        "## Split Sizes",
        "",
        f"- Train samples: `{len(train_rows)}`",
        f"- Validation samples: `{len(val_rows)}`",
        "",
        "## Validation Success",
        "",
        f"- Raw ordered exact (same as training log metric): `{int(val_metrics['raw_ordered_success_count'])}/{int(val_metrics['count'])}` = `{val_metrics['raw_ordered_success_rate']:.4f}`",
        f"- Ordered exact success: `{int(val_metrics['ordered_success_count'])}/{int(val_metrics['count'])}` = `{val_metrics['ordered_success_rate']:.4f}`",
        f"- Unordered exact success: `{int(val_metrics['unordered_success_count'])}/{int(val_metrics['count'])}` = `{val_metrics['unordered_success_rate']:.4f}`",
        "",
        "## Files",
        "",
        "- `train_split.csv`: dataset index, compound ID, SMILES, target anchor labels",
        "- `val_split.csv`: dataset index, compound ID, SMILES, target anchors, raw predictions, generation-time predictions, success flags",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_cfg = ckpt["train_config"]

    dataset = build_dataset_from_train_config(train_cfg)
    train_idx, val_idx = split_indices(len(dataset), val_ratio=float(train_cfg["val_ratio"]), seed=int(train_cfg["seed"]))

    model_cfg = ckpt["model_config"]
    model = AnchorGNN(
        in_dim=model_cfg["in_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_classes=model_cfg.get("num_classes", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_rows = []
    for dataset_index in train_idx:
        sample = dataset[dataset_index]
        true_l, true_r = find_anchor_positions(sample.y.long())
        train_rows.append(
            {
                "dataset_index": dataset_index,
                "compound_id": getattr(sample, "compound_id", ""),
                "smiles": getattr(sample, "smiles", ""),
                "smiles_r": getattr(sample, "smiles_r", ""),
                "true_anchor_l": true_l,
                "true_anchor_r": true_r,
            }
        )

    val_rows, val_metrics = evaluate_dataset(model, dataset, val_idx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train_split.csv"
    val_csv = out_dir / "val_split.csv"
    summary_md = out_dir / "summary.md"

    write_csv(train_csv, train_rows)
    write_csv(val_csv, val_rows)
    write_summary(summary_md, args.ckpt, train_cfg, train_rows, val_rows, val_metrics)

    print(f"[report] train_csv={train_csv}")
    print(f"[report] val_csv={val_csv}")
    print(f"[report] summary={summary_md}")
    print(
        f"[validation] raw_ordered_exact={int(val_metrics['raw_ordered_success_count'])}/{int(val_metrics['count'])}={val_metrics['raw_ordered_success_rate']:.4f}"
    )
    print(
        f"[validation] ordered_exact={int(val_metrics['ordered_success_count'])}/{int(val_metrics['count'])}={val_metrics['ordered_success_rate']:.4f}"
    )
    print(
        f"[validation] unordered_exact={int(val_metrics['unordered_success_count'])}/{int(val_metrics['count'])}={val_metrics['unordered_success_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
