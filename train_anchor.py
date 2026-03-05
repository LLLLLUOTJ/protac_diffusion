from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from data.anchored_tensor_dataset import AnchoredTensorPTDataset, collate_anchor_train_samples
from data.linker_anchor_dataset import LinkerAnchorDataset, collate_linker_anchor_samples
from models.anchor_gnn import AnchorGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anchor predictor from linker.csv or processed .pt tensors")
    parser.add_argument("--csv", type=str, default="data/csv/linker.csv", help="linker csv path")
    parser.add_argument("--tensor-pt", type=str, default=None, help="optional processed tensor dataset (.pt)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--out", type=str, default="checkpoints/anchor_gnn.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_n = max(1, int(n * val_ratio))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def compute_class_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
    counts = torch.zeros((3,), dtype=torch.float32)
    for batch in loader:
        y = batch["y"]
        counts += torch.bincount(y, minlength=3).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights.to(device)


def anchor_exact_match(logits: torch.Tensor, y: torch.Tensor, graph_ptr: torch.Tensor) -> Tuple[int, int]:
    exact = 0
    total = 0
    for i in range(graph_ptr.shape[0] - 1):
        start = int(graph_ptr[i].item())
        end = int(graph_ptr[i + 1].item())
        if end <= start:
            continue

        logits_g = logits[start:end]
        y_g = y[start:end]

        true_l = (y_g == 1).nonzero(as_tuple=False).view(-1)
        true_r = (y_g == 2).nonzero(as_tuple=False).view(-1)
        if true_l.numel() != 1 or true_r.numel() != 1:
            continue

        pred_l = int(torch.argmax(logits_g[:, 1]).item())
        pred_r = int(torch.argmax(logits_g[:, 2]).item())
        exact += int(pred_l == int(true_l[0].item()) and pred_r == int(true_r[0].item()))
        total += 1
    return exact, total


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_nodes = 0
    total_correct = 0
    total_exact = 0
    total_graphs = 0

    for batch in loader:
        x = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        y = batch["y"].to(device)
        graph_ptr = batch["graph_ptr"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x, edge_index)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * int(y.shape[0])
        total_nodes += int(y.shape[0])
        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred == y).sum().item())

        exact, graphs = anchor_exact_match(logits, y, graph_ptr)
        total_exact += exact
        total_graphs += graphs

    return {
        "loss": total_loss / max(total_nodes, 1),
        "node_acc": total_correct / max(total_nodes, 1),
        "anchor_exact_acc": total_exact / max(total_graphs, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    if args.tensor_pt:
        dataset: Dataset = AnchoredTensorPTDataset(pt_path=args.tensor_pt, max_samples=args.max_samples)
        collate_fn = collate_anchor_train_samples
        source = f"tensor_pt={args.tensor_pt}"
    else:
        dataset = LinkerAnchorDataset(csv_path=args.csv, max_samples=args.max_samples)
        collate_fn = collate_linker_anchor_samples
        source = f"csv={args.csv}"

    if len(dataset) == 0:
        raise RuntimeError("No valid training samples found for the selected data source")

    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = AnchorGNN(
        in_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    class_weights = compute_class_weights(train_loader, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reason_counts = getattr(dataset, "reason_counts", {})
    print(
        f"[data] source={source} total={len(dataset)} train={len(train_set)} val={len(val_set)} "
        f"reasons={reason_counts}",
        flush=True,
    )
    print(f"[train] device={device} class_weights={class_weights.tolist()}", flush=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = run_epoch(model, val_loader, None, criterion, device)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_node_acc={train_metrics['node_acc']:.4f} "
            f"train_anchor_exact={train_metrics['anchor_exact_acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_node_acc={val_metrics['node_acc']:.4f} "
            f"val_anchor_exact={val_metrics['anchor_exact_acc']:.4f}",
            flush=True,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "in_dim": 4,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.layers,
                        "dropout": args.dropout,
                        "num_classes": 3,
                    },
                    "train_config": vars(args),
                    "class_weights": class_weights.detach().cpu(),
                    "best_val_loss": best_val,
                },
                out_path,
            )
            print(f"[checkpoint] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
