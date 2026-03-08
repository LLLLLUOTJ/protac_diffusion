from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_RE = re.compile(r"^\[run\] training (?P<stage>node|edge) diffusion$")
DATA_RE = re.compile(
    r"^\[data\] source=(?P<source>\S+) total=(?P<total>\d+) train=(?P<train>\d+) val=(?P<val>\d+) reasons=(?P<reasons>.+)$"
)
TRAIN_RE = re.compile(
    r"^\[train\] device=(?P<device>\S+) batch_size=(?P<batch_size>\d+) hidden_dim=(?P<hidden_dim>\d+) "
    r"layers=(?P<layers>\d+) timesteps=(?P<timesteps>\d+)$"
)
EPOCH_RE = re.compile(
    r"^\[epoch (?P<epoch>\d+)\] train_loss=(?P<train_loss>[-+]?\d*\.?\d+) val_loss=(?P<val_loss>[-+]?\d*\.?\d+)$"
)
CHECKPOINT_RE = re.compile(r"^\[checkpoint\] saved (?P<path>\S+)$")


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    checkpoint_saved: bool = False


@dataclass
class StageHistory:
    stage: str
    source: str = ""
    total: int = 0
    train_size: int = 0
    val_size: int = 0
    reasons: str = ""
    device: str = ""
    batch_size: int = 0
    hidden_dim: int = 0
    layers: int = 0
    timesteps: int = 0
    epochs: List[EpochRecord] = field(default_factory=list)

    def add_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.epochs.append(EpochRecord(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

    def mark_last_checkpoint(self) -> None:
        if self.epochs:
            self.epochs[-1].checkpoint_saved = True


def parse_log(path: str) -> Dict[str, StageHistory]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    stages: Dict[str, StageHistory] = {}
    current: Optional[StageHistory] = None

    for line in lines:
        match = RUN_RE.match(line)
        if match:
            stage = match.group("stage")
            current = StageHistory(stage=stage)
            stages[stage] = current
            continue

        if current is None:
            continue

        match = DATA_RE.match(line)
        if match:
            current.source = match.group("source")
            current.total = int(match.group("total"))
            current.train_size = int(match.group("train"))
            current.val_size = int(match.group("val"))
            current.reasons = match.group("reasons")
            continue

        match = TRAIN_RE.match(line)
        if match:
            current.device = match.group("device")
            current.batch_size = int(match.group("batch_size"))
            current.hidden_dim = int(match.group("hidden_dim"))
            current.layers = int(match.group("layers"))
            current.timesteps = int(match.group("timesteps"))
            continue

        match = EPOCH_RE.match(line)
        if match:
            current.add_epoch(
                epoch=int(match.group("epoch")),
                train_loss=float(match.group("train_loss")),
                val_loss=float(match.group("val_loss")),
            )
            continue

        match = CHECKPOINT_RE.match(line)
        if match:
            current.mark_last_checkpoint()
            continue

    return stages


def _rolling_mean(values: List[float], window: int = 5) -> List[float]:
    out: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(sum(chunk) / max(len(chunk), 1))
    return out


def stage_summary(stage: StageHistory) -> Dict[str, Any]:
    if not stage.epochs:
        return {
            "stage": stage.stage,
            "num_epochs": 0,
        }

    train_losses = [epoch.train_loss for epoch in stage.epochs]
    val_losses = [epoch.val_loss for epoch in stage.epochs]
    best_idx = min(range(len(val_losses)), key=lambda idx: val_losses[idx])
    best_epoch = stage.epochs[best_idx]
    first_epoch = stage.epochs[0]
    last_epoch = stage.epochs[-1]

    improvement_abs = first_epoch.val_loss - best_epoch.val_loss
    improvement_pct = (improvement_abs / first_epoch.val_loss * 100.0) if first_epoch.val_loss != 0 else math.nan
    last_10 = val_losses[-10:] if len(val_losses) >= 10 else val_losses[:]
    recent_mean = sum(last_10) / max(len(last_10), 1)
    last_5 = val_losses[-5:] if len(val_losses) >= 5 else val_losses[:]
    last_5_mean = sum(last_5) / max(len(last_5), 1)
    checkpoint_epochs = [epoch.epoch for epoch in stage.epochs if epoch.checkpoint_saved]

    summary = {
        "stage": stage.stage,
        "source": stage.source,
        "device": stage.device,
        "batch_size": stage.batch_size,
        "hidden_dim": stage.hidden_dim,
        "layers": stage.layers,
        "timesteps": stage.timesteps,
        "dataset_total": stage.total,
        "dataset_train": stage.train_size,
        "dataset_val": stage.val_size,
        "num_epochs": len(stage.epochs),
        "first_epoch": first_epoch.epoch,
        "first_train_loss": first_epoch.train_loss,
        "first_val_loss": first_epoch.val_loss,
        "last_epoch": last_epoch.epoch,
        "last_train_loss": last_epoch.train_loss,
        "last_val_loss": last_epoch.val_loss,
        "best_epoch": best_epoch.epoch,
        "best_train_loss": best_epoch.train_loss,
        "best_val_loss": best_epoch.val_loss,
        "val_improvement_abs": improvement_abs,
        "val_improvement_pct": improvement_pct,
        "final_gap_train_minus_val": last_epoch.train_loss - last_epoch.val_loss,
        "best_gap_train_minus_val": best_epoch.train_loss - best_epoch.val_loss,
        "recent_val_mean_last_10": recent_mean,
        "recent_val_mean_last_5": last_5_mean,
        "checkpoint_epochs": checkpoint_epochs,
    }
    return summary


def write_stage_csv(stage: StageHistory, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "checkpoint_saved", "generalization_gap"],
        )
        writer.writeheader()
        for epoch in stage.epochs:
            writer.writerow(
                {
                    "epoch": epoch.epoch,
                    "train_loss": epoch.train_loss,
                    "val_loss": epoch.val_loss,
                    "checkpoint_saved": int(epoch.checkpoint_saved),
                    "generalization_gap": epoch.train_loss - epoch.val_loss,
                }
            )


def plot_loss_curves(stages: Dict[str, StageHistory], out_path: Path) -> None:
    available = [stage for stage in ["node", "edge"] if stage in stages and stages[stage].epochs]
    if not available:
        raise RuntimeError("No stage data available for plotting")

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 4.8 * len(available)), sharex=False)
    axes = np.atleast_1d(axes).ravel().tolist()

    for ax, stage_name in zip(axes, available):
        stage = stages[stage_name]
        epochs = [epoch.epoch for epoch in stage.epochs]
        train_losses = [epoch.train_loss for epoch in stage.epochs]
        val_losses = [epoch.val_loss for epoch in stage.epochs]
        best_idx = min(range(len(val_losses)), key=lambda idx: val_losses[idx])
        rolling = _rolling_mean(val_losses, window=5)

        ax.plot(epochs, train_losses, label="train_loss", color="#1f77b4", linewidth=1.8)
        ax.plot(epochs, val_losses, label="val_loss", color="#d62728", linewidth=1.8)
        ax.plot(epochs, rolling, label="val_loss_ma5", color="#ff9896", linewidth=1.2, linestyle="--")

        ckpt_epochs = [epoch.epoch for epoch in stage.epochs if epoch.checkpoint_saved]
        ckpt_vals = [epoch.val_loss for epoch in stage.epochs if epoch.checkpoint_saved]
        if ckpt_epochs:
            ax.scatter(ckpt_epochs, ckpt_vals, label="checkpoint", color="#2ca02c", s=28, zorder=3)

        ax.scatter(
            [stage.epochs[best_idx].epoch],
            [stage.epochs[best_idx].val_loss],
            color="#000000",
            s=40,
            zorder=4,
            label=f"best_val={stage.epochs[best_idx].val_loss:.4f}",
        )
        ax.set_title(
            f"{stage_name.capitalize()} Diffusion Loss Trend "
            f"(device={stage.device}, batch={stage.batch_size}, epochs={len(stage.epochs)})"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gap_curves(stages: Dict[str, StageHistory], out_path: Path) -> None:
    available = [stage for stage in ["node", "edge"] if stage in stages and stages[stage].epochs]
    if not available:
        raise RuntimeError("No stage data available for plotting")

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 4.2 * len(available)), sharex=False)
    axes = np.atleast_1d(axes).ravel().tolist()

    for ax, stage_name in zip(axes, available):
        stage = stages[stage_name]
        epochs = [epoch.epoch for epoch in stage.epochs]
        val_losses = [epoch.val_loss for epoch in stage.epochs]
        best_so_far = []
        current_best = float("inf")
        for val in val_losses:
            current_best = min(current_best, val)
            best_so_far.append(current_best)
        gap = [epoch.train_loss - epoch.val_loss for epoch in stage.epochs]

        ax.plot(epochs, best_so_far, label="best_val_so_far", color="#2ca02c", linewidth=1.8)
        ax.plot(epochs, gap, label="train_minus_val", color="#9467bd", linewidth=1.5)
        ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
        ax.set_title(f"{stage_name.capitalize()} Diffusion Convergence Diagnostics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown_report(stages: Dict[str, StageHistory], summary: Dict[str, Any], out_path: Path, log_path: str) -> None:
    lines: List[str] = []
    lines.append("# Training Analysis")
    lines.append("")
    lines.append(f"- log: `{log_path}`")
    lines.append(f"- stages: {', '.join(sorted(stages.keys()))}")
    lines.append("")

    for stage_name in ["node", "edge"]:
        if stage_name not in summary:
            continue
        item = summary[stage_name]
        lines.append(f"## {stage_name.capitalize()} Diffusion")
        lines.append("")
        lines.append(f"- device: `{item.get('device', '')}`")
        lines.append(
            f"- dataset: total `{item.get('dataset_total', 0)}`, train `{item.get('dataset_train', 0)}`, val `{item.get('dataset_val', 0)}`"
        )
        lines.append(
            f"- config: batch `{item.get('batch_size', 0)}`, hidden `{item.get('hidden_dim', 0)}`, "
            f"layers `{item.get('layers', 0)}`, timesteps `{item.get('timesteps', 0)}`"
        )
        lines.append(
            f"- best epoch: `{item.get('best_epoch', 0)}` with val loss `{item.get('best_val_loss', 0.0):.4f}`"
        )
        lines.append(
            f"- first val loss: `{item.get('first_val_loss', 0.0):.4f}` -> "
            f"best val loss: `{item.get('best_val_loss', 0.0):.4f}` "
            f"(`{item.get('val_improvement_pct', 0.0):.2f}%` improvement)"
        )
        lines.append(
            f"- final epoch: `{item.get('last_epoch', 0)}` with train `{item.get('last_train_loss', 0.0):.4f}`, "
            f"val `{item.get('last_val_loss', 0.0):.4f}`"
        )
        lines.append(
            f"- last-10 val mean: `{item.get('recent_val_mean_last_10', 0.0):.4f}`; "
            f"last-5 val mean: `{item.get('recent_val_mean_last_5', 0.0):.4f}`"
        )
        ckpts = item.get("checkpoint_epochs", [])
        if ckpts:
            lines.append(f"- checkpoint epochs: `{ckpts}`")
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("- `loss_curves.png`: train/val loss trend and checkpoint epochs")
    lines.append("- `diagnostics.png`: best-so-far validation curve and train-minus-val gap")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze diffusion training log and generate plots/statistics")
    parser.add_argument("--log", type=str, required=True, help="path to train_diffusion.log")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/train_log_analysis",
        help="directory to write plots and summary files",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    stages = parse_log(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}
    for stage_name, stage in stages.items():
        summary[stage_name] = stage_summary(stage)
        write_stage_csv(stage, out_dir / f"{stage_name}_history.csv")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_loss_curves(stages, out_dir / "loss_curves.png")
    plot_gap_curves(stages, out_dir / "diagnostics.png")
    write_markdown_report(stages, summary, out_dir / "REPORT.md", log_path=args.log)

    print(f"[done] stages={sorted(stages.keys())} out_dir={out_dir}", flush=True)
    for stage_name in ["node", "edge"]:
        if stage_name not in summary:
            continue
        item = summary[stage_name]
        print(
            f"[{stage_name}] epochs={item['num_epochs']} best_epoch={item['best_epoch']} "
            f"best_val={item['best_val_loss']:.4f} last_val={item['last_val_loss']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
