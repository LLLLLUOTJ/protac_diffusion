from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

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
    parser.add_argument(
        "--sample-weight-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "linker_id_inv",
            "linker_id_inv_sqrt",
            "anchored_inv",
            "anchored_inv_sqrt",
        ],
        help="Optional weighted sampling mode for the training split.",
    )
    parser.add_argument(
        "--token-ce-weight",
        type=float,
        default=0.15,
        help="Weight for token-level auxiliary CE loss computed from predicted x0 embeddings.",
    )
    parser.add_argument(
        "--token-ce-temperature",
        type=float,
        default=0.10,
        help="Cosine-logit temperature for token auxiliary CE loss.",
    )
    parser.add_argument(
        "--pad-suffix-weight",
        type=float,
        default=2.0,
        help="Extra weight multiplier for PAD suffix positions in token auxiliary loss.",
    )
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


def build_weighted_sampler(train_set: Subset, sample_weight_mode: str) -> tuple[WeightedRandomSampler | None, dict[str, Any]]:
    if sample_weight_mode == "none":
        return None, {"mode": "none"}

    if not isinstance(train_set.dataset, WeakAnchorTokenPTDataset):
        raise TypeError("weighted sampling currently expects WeakAnchorTokenPTDataset as the base dataset")

    if sample_weight_mode.startswith("linker_id_"):
        field = "linker_id"
    elif sample_weight_mode.startswith("anchored_"):
        field = "anchored_linker_smiles"
    else:
        raise ValueError(f"Unsupported sample_weight_mode: {sample_weight_mode}")

    if sample_weight_mode.endswith("_inv"):
        power = 1.0
    elif sample_weight_mode.endswith("_inv_sqrt"):
        power = 0.5
    else:
        raise ValueError(f"Unsupported sample_weight_mode: {sample_weight_mode}")

    group_values = [str(train_set.dataset.records[int(i)].get(field, "")) for i in train_set.indices]
    counts = Counter(group_values)
    raw_weights = torch.tensor([1.0 / (counts[value] ** power) for value in group_values], dtype=torch.double)
    raw_weights = raw_weights * (len(raw_weights) / raw_weights.sum().item())
    sampler = WeightedRandomSampler(weights=raw_weights, num_samples=len(raw_weights), replacement=True)

    count_values = list(counts.values())
    summary = {
        "mode": sample_weight_mode,
        "field": field,
        "num_groups": len(counts),
        "min_group_count": int(min(count_values)),
        "median_group_count": float(sorted(count_values)[len(count_values) // 2]),
        "max_group_count": int(max(count_values)),
        "top_groups": counts.most_common(10),
        "weight_min": float(raw_weights.min().item()),
        "weight_max": float(raw_weights.max().item()),
        "weight_mean": float(raw_weights.mean().item()),
    }
    return sampler, summary


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def compute_token_auxiliary_terms(
    *,
    x0_pred: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    pad_token_id: int | None,
    temperature: float,
    pad_suffix_weight: float,
) -> Dict[str, torch.Tensor]:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    norm_x = F.normalize(x0_pred.float(), p=2, dim=-1)
    norm_vocab = F.normalize(vocab_embeddings.float(), p=2, dim=-1)
    logits = (norm_x @ norm_vocab.transpose(0, 1)) / float(temperature)

    per_token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        token_ids.reshape(-1),
        reduction="none",
        ignore_index=-1,
    ).reshape_as(token_ids)

    base_weights = loss_mask.float()
    pad_suffix_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    if pad_token_id is not None:
        pad_suffix_mask = (token_ids == int(pad_token_id)) & loss_mask.bool()

    weights = base_weights
    if pad_suffix_weight != 1.0:
        weights = weights * torch.where(
            pad_suffix_mask,
            torch.full_like(base_weights, float(pad_suffix_weight)),
            torch.ones_like(base_weights),
        )

    denom = weights.sum().clamp(min=1.0)
    base_denom = base_weights.sum().clamp(min=1.0)
    pad_denom = pad_suffix_mask.float().sum().clamp(min=1.0)

    return {
        "loss": (per_token_ce * weights).sum() / denom,
        "token_ce": (per_token_ce * base_weights).sum() / base_denom,
        "pad_suffix_ce": (per_token_ce * pad_suffix_mask.float()).sum() / pad_denom,
        "pad_suffix_fraction": pad_suffix_mask.float().mean(),
    }


def run_epoch(
    diffusion: DDPM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    timesteps: int,
    vocab_embeddings: torch.Tensor,
    pad_token_id: int | None,
    token_ce_weight: float,
    token_ce_temperature: float,
    pad_suffix_weight: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    diffusion.train() if is_train else diffusion.eval()

    total_loss = 0.0
    total_diffusion_loss = 0.0
    total_token_aux_loss = 0.0
    total_token_ce = 0.0
    total_pad_suffix_ce = 0.0
    total_pad_suffix_fraction = 0.0
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
            terms = diffusion.training_terms(
                x_start=x_start,
                t=t,
                sample_mask=linker_token["sample_mask"],
                fixed_mask=linker_token["fixed_mask"],
                fixed_values=linker_token["fixed_values"],
                loss_mask=linker_token["loss_mask"],
                model_kwargs=model_kwargs,
            )
            diffusion_loss = terms["loss"]
            token_aux_loss = torch.zeros((), device=device)
            token_ce = torch.zeros((), device=device)
            pad_suffix_ce = torch.zeros((), device=device)
            pad_suffix_fraction = torch.zeros((), device=device)
            if token_ce_weight > 0.0:
                x0_pred = diffusion.predict_x0_from_noise(
                    x_t=terms["x_t"],
                    t=t,
                    predicted_noise=terms["predicted_noise"],
                )
                aux_terms = compute_token_auxiliary_terms(
                    x0_pred=x0_pred,
                    vocab_embeddings=vocab_embeddings,
                    token_ids=linker_token["token_ids"],
                    loss_mask=linker_token["loss_mask"],
                    pad_token_id=pad_token_id,
                    temperature=token_ce_temperature,
                    pad_suffix_weight=pad_suffix_weight,
                )
                token_aux_loss = aux_terms["loss"]
                token_ce = aux_terms["token_ce"]
                pad_suffix_ce = aux_terms["pad_suffix_ce"]
                pad_suffix_fraction = aux_terms["pad_suffix_fraction"]
            loss = diffusion_loss + float(token_ce_weight) * token_aux_loss
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_diffusion_loss += float(diffusion_loss.item())
        total_token_aux_loss += float(token_aux_loss.item())
        total_token_ce += float(token_ce.item())
        total_pad_suffix_ce += float(pad_suffix_ce.item())
        total_pad_suffix_fraction += float(pad_suffix_fraction.item())
        total_batches += 1

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "diffusion_loss": total_diffusion_loss / denom,
        "token_aux_loss": total_token_aux_loss / denom,
        "token_ce": total_token_ce / denom,
        "pad_suffix_ce": total_pad_suffix_ce / denom,
        "pad_suffix_fraction": total_pad_suffix_fraction / denom,
    }


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
    learn_pad_positions = bool(dataset.meta.get("learn_pad_positions", False))
    pad_token = str(dataset.meta.get("pad_token", ""))
    pad_token_id_raw = dataset.meta.get("pad_token_id", None)
    pad_token_id = int(pad_token_id_raw) if pad_token_id_raw is not None else None
    vocab_embeddings = dataset.meta.get("token_embeddings")
    if not torch.is_tensor(vocab_embeddings):
        raise RuntimeError("Token dataset metadata missing token_embeddings")
    vocab_embeddings = vocab_embeddings.detach().cpu().float().to(device)

    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    train_sampler, sampler_summary = build_weighted_sampler(train_set, sample_weight_mode=args.sample_weight_mode)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
        f"reasons={dataset.reason_counts} learn_pad_positions={learn_pad_positions} "
        f"pad_token={pad_token or 'None'}",
        flush=True,
    )
    print(
        f"[train] device={device} batch_size={args.batch_size} embed_dim={embed_dim} "
        f"hidden_dim={args.hidden_dim} layers={args.layers} heads={args.heads} "
        f"timesteps={args.timesteps} condition_dropout={args.condition_dropout} "
        f"sample_weight_mode={args.sample_weight_mode} token_ce_weight={args.token_ce_weight} "
        f"token_ce_temperature={args.token_ce_temperature} pad_suffix_weight={args.pad_suffix_weight}",
        flush=True,
    )
    if args.sample_weight_mode != "none":
        print(f"[sampler] {json.dumps(sampler_summary, ensure_ascii=False)}", flush=True)

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[Dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            diffusion,
            train_loader,
            optimizer,
            device,
            timesteps=args.timesteps,
            vocab_embeddings=vocab_embeddings,
            pad_token_id=pad_token_id,
            token_ce_weight=args.token_ce_weight,
            token_ce_temperature=args.token_ce_temperature,
            pad_suffix_weight=args.pad_suffix_weight,
        )
        val_metrics = run_epoch(
            diffusion,
            val_loader,
            None,
            device,
            timesteps=args.timesteps,
            vocab_embeddings=vocab_embeddings,
            pad_token_id=pad_token_id,
            token_ce_weight=args.token_ce_weight,
            token_ce_temperature=args.token_ce_temperature,
            pad_suffix_weight=args.pad_suffix_weight,
        )
        epoch_time = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(val_metrics["loss"]),
                "train_diffusion_loss": float(train_metrics["diffusion_loss"]),
                "val_diffusion_loss": float(val_metrics["diffusion_loss"]),
                "train_token_aux_loss": float(train_metrics["token_aux_loss"]),
                "val_token_aux_loss": float(val_metrics["token_aux_loss"]),
                "train_pad_suffix_ce": float(train_metrics["pad_suffix_ce"]),
                "val_pad_suffix_ce": float(val_metrics["pad_suffix_ce"]),
                "epoch_time_s": float(epoch_time),
            }
        )
        print(
            f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"train_diff={train_metrics['diffusion_loss']:.4f} "
            f"val_diff={val_metrics['diffusion_loss']:.4f} "
            f"train_aux={train_metrics['token_aux_loss']:.4f} "
            f"val_aux={val_metrics['token_aux_loss']:.4f} "
            f"val_pad_ce={val_metrics['pad_suffix_ce']:.4f}",
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
                    "sampler_summary": sampler_summary,
                    "best_val_loss": best_val,
                    "token_meta": {
                        "token_vocab": dataset.vocab_tokens,
                        "token_to_id": dataset.token_to_id,
                        "pad_token": pad_token,
                        "pad_token_id": pad_token_id,
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
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "train_diffusion_loss",
                "val_diffusion_loss",
                "train_token_aux_loss",
                "val_token_aux_loss",
                "train_pad_suffix_ce",
                "val_pad_suffix_ce",
                "epoch_time_s",
            ],
        )
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
                "final_train_diffusion_loss": history[-1]["train_diffusion_loss"] if history else None,
                "final_val_diffusion_loss": history[-1]["val_diffusion_loss"] if history else None,
                "final_train_token_aux_loss": history[-1]["train_token_aux_loss"] if history else None,
                "final_val_token_aux_loss": history[-1]["val_token_aux_loss"] if history else None,
                "final_train_pad_suffix_ce": history[-1]["train_pad_suffix_ce"] if history else None,
                "final_val_pad_suffix_ce": history[-1]["val_pad_suffix_ce"] if history else None,
                "total_epoch_time_s": total_time,
                "avg_epoch_time_s": (total_time / len(history)) if history else None,
                "train_config": vars(args),
                "sampler_summary": sampler_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[history] csv={history_path}", flush=True)
    print(f"[summary] json={summary_path}", flush=True)


if __name__ == "__main__":
    main()
