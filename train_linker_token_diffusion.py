from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from data.weak_anchor_token_dataset import WeakAnchorTokenPTDataset
from data.weak_anchor_token_diffusion import collate_weak_anchor_token_diffusion_batch
from diffusion.ddpm import DDPM
from models.fragment_conditioned_denoiser import FragmentConditionedTokenDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fragment-conditioned linker-token diffusion on weak-anchor token tensors")
    parser.add_argument("--tensor-pt", type=str, default="data/processed/weak_anchor_token_tensors.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="checkpoints/linker_token_diffusion.pt")
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


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def run_epoch(
    diffusion: DDPM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    timesteps: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    diffusion.train() if is_train else diffusion.eval()

    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        linker_token = batch["linker_token"]
        model_kwargs = {
            "left_graph": batch["left_graph"],
            "right_graph": batch["right_graph"],
            "token_mask": linker_token["sample_mask"],
        }
        x_start = linker_token["x_start"]
        t = torch.randint(0, timesteps, (x_start.shape[0],), device=device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            loss = diffusion.p_losses(
                x_start=x_start,
                t=t,
                sample_mask=linker_token["sample_mask"],
                fixed_mask=linker_token["fixed_mask"],
                fixed_values=linker_token["fixed_values"],
                loss_mask=linker_token["loss_mask"],
                model_kwargs=model_kwargs,
            )
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return {"loss": total_loss / max(total_batches, 1)}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    dataset = WeakAnchorTokenPTDataset(pt_path=args.tensor_pt, max_samples=args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError("No valid weak-anchor token samples found")
    embed_dim = int(dataset.meta.get("embedding_dim", 0))
    if embed_dim <= 0:
        raise RuntimeError("Token dataset metadata missing embedding_dim")

    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_weak_anchor_token_diffusion_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_weak_anchor_token_diffusion_batch,
    )

    model = FragmentConditionedTokenDenoiser(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        condition_dropout=args.condition_dropout,
    )
    diffusion = DDPM(
        model=model,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device_name,
    ).to(device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[data] source={args.tensor_pt} total={len(dataset)} train={len(train_set)} val={len(val_set)} "
        f"reasons={dataset.reason_counts}",
        flush=True,
    )
    print(
        f"[train] device={device} batch_size={args.batch_size} embed_dim={embed_dim} "
        f"hidden_dim={args.hidden_dim} layers={args.layers} heads={args.heads} "
        f"timesteps={args.timesteps} condition_dropout={args.condition_dropout}",
        flush=True,
    )

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[Dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(diffusion, train_loader, optimizer, device, timesteps=args.timesteps)
        val_metrics = run_epoch(diffusion, val_loader, None, device, timesteps=args.timesteps)
        epoch_time = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(val_metrics["loss"]),
                "epoch_time_s": float(epoch_time),
            }
        )
        print(
            f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}",
            flush=True,
        )
        if val_metrics["loss"] < (best_val - args.min_delta):
            best_val = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": diffusion.model.state_dict(),
                    "model_config": {
                        "embed_dim": embed_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.layers,
                        "num_heads": args.heads,
                        "dropout": args.dropout,
                        "condition_dropout": args.condition_dropout,
                    },
                    "diffusion_config": {
                        "timesteps": args.timesteps,
                        "beta_start": args.beta_start,
                        "beta_end": args.beta_end,
                    },
                    "train_config": vars(args),
                    "best_val_loss": best_val,
                    "token_meta": {
                        "token_vocab": dataset.vocab_tokens,
                        "token_to_id": dataset.token_to_id,
                        "embedding_dim": embed_dim,
                    },
                },
                out_path,
            )
            print(f"[checkpoint] saved {out_path}", flush=True)
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"[early_stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_val={best_val:.4f} patience={args.patience}",
                flush=True,
            )
            break

    history_path = out_path.with_suffix(".history.csv")
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "epoch_time_s"])
        writer.writeheader()
        writer.writerows(history)

    summary_path = out_path.with_suffix(".summary.json")
    total_time = sum(float(item["epoch_time_s"]) for item in history)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_path": str(out_path),
                "history_path": str(history_path),
                "num_epochs": len(history),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "stopped_early": bool(args.patience > 0 and len(history) < args.epochs),
                "final_train_loss": history[-1]["train_loss"] if history else None,
                "final_val_loss": history[-1]["val_loss"] if history else None,
                "total_epoch_time_s": total_time,
                "avg_epoch_time_s": (total_time / len(history)) if history else None,
                "train_config": vars(args),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[history] csv={history_path}", flush=True)
    print(f"[summary] json={summary_path}", flush=True)


if __name__ == "__main__":
    main()
