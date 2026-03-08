from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.anchored_tensor_dataset import WeakAnchorTensorPTDataset
from sample_linker import (
    build_sampling_batch,
    choose_device,
    decode_results,
    load_edge_diffusion,
    load_node_diffusion,
    sample_edges,
    sample_nodes,
    select_source_sample,
    set_seed,
    source_edge_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate linker generation over multiple weak-anchor samples")
    parser.add_argument("--tensor-pt", type=str, default="data/processed/weak_anchor_tensors.pt")
    parser.add_argument("--node-ckpt", type=str, required=True)
    parser.add_argument("--edge-ckpt", type=str, default="")
    parser.add_argument("--mode", type=str, default="joint", choices=["node_only", "joint"])
    parser.add_argument("--max-source-samples", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num-samples-per-source", type=int, default=4)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/linker_eval")
    return parser.parse_args()


def select_indices(dataset_len: int, start_index: int, max_source_samples: int, seed: int, shuffle: bool) -> List[int]:
    if start_index < 0 or start_index >= dataset_len:
        raise IndexError(f"start_index out of range: {start_index}")
    indices = list(range(start_index, dataset_len))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices[: max_source_samples]


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    anchored = [row["generated_anchored_linker_smiles"] for row in rows if row["generated_anchored_linker_smiles"]]
    full = [row["generated_full_smiles"] for row in rows if row["generated_full_smiles"]]
    source_ids = sorted({str(row["source_dataset_index"]) for row in rows})
    return {
        "num_source_samples": len(source_ids),
        "num_requested": len(rows),
        "num_decoded_anchored": len(anchored),
        "num_assembled_full": len(full),
        "decode_rate": (len(anchored) / len(rows)) if rows else 0.0,
        "assembly_rate": (len(full) / len(rows)) if rows else 0.0,
        "unique_anchored": len(set(anchored)),
        "unique_full": len(set(full)),
    }


def build_per_source_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_source: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        key = int(row["source_dataset_index"])
        by_source.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for source_idx, items in sorted(by_source.items()):
        anchored = [row["generated_anchored_linker_smiles"] for row in items if row["generated_anchored_linker_smiles"]]
        full = [row["generated_full_smiles"] for row in items if row["generated_full_smiles"]]
        out.append(
            {
                "source_dataset_index": source_idx,
                "sample_id": items[0]["sample_id"],
                "protac_id": items[0]["protac_id"],
                "linker_id": items[0]["linker_id"],
                "source_anchored_linker_smiles": items[0]["source_anchored_linker_smiles"],
                "source_left_fragment_smiles": items[0]["source_left_fragment_smiles"],
                "source_right_fragment_smiles": items[0]["source_right_fragment_smiles"],
                "num_requested": len(items),
                "num_decoded_anchored": len(anchored),
                "num_assembled_full": len(full),
                "decode_rate": (len(anchored) / len(items)) if items else 0.0,
                "assembly_rate": (len(full) / len(items)) if items else 0.0,
                "unique_anchored": len(set(anchored)),
                "unique_full": len(set(full)),
            }
        )
    return out


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_eval(per_source_rows: List[Dict[str, Any]], summary: Dict[str, Any], out_path: Path) -> None:
    if not per_source_rows:
        raise RuntimeError("No per-source rows available for plotting")

    indices = [row["source_dataset_index"] for row in per_source_rows]
    decode_rates = [row["decode_rate"] for row in per_source_rows]
    assembly_rates = [row["assembly_rate"] for row in per_source_rows]
    unique_anchored = [row["unique_anchored"] for row in per_source_rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].plot(indices, decode_rates, label="decode_rate", color="#1f77b4", linewidth=1.6)
    axes[0].plot(indices, assembly_rates, label="assembly_rate", color="#d62728", linewidth=1.6)
    axes[0].set_title(
        "Per-Source Generation Success "
        f"(overall decode={summary['decode_rate']:.3f}, assembly={summary['assembly_rate']:.3f})"
    )
    axes[0].set_xlabel("Source dataset index")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25, linestyle=":")
    axes[0].legend(loc="best")

    axes[1].hist(unique_anchored, bins=range(0, max(unique_anchored) + 2), color="#2ca02c", alpha=0.85, align="left")
    axes[1].set_title("Unique Anchored Linkers per Source Sample")
    axes[1].set_xlabel("Unique anchored linker count")
    axes[1].set_ylabel("Number of source samples")
    axes[1].grid(alpha=0.25, linestyle=":")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    dataset = WeakAnchorTensorPTDataset(pt_path=args.tensor_pt)
    indices = select_indices(
        dataset_len=len(dataset),
        start_index=args.start_index,
        max_source_samples=args.max_source_samples,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    node_diffusion = load_node_diffusion(args.node_ckpt, device=device)
    edge_diffusion = None
    if args.mode == "joint":
        if not args.edge_ckpt:
            raise ValueError("--edge-ckpt is required when --mode joint")
        edge_diffusion = load_edge_diffusion(args.edge_ckpt, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for pos, dataset_index in enumerate(indices, start=1):
        source_sample = select_source_sample(dataset, sample_index=dataset_index, sample_id=None)
        batch = build_sampling_batch(source_sample, num_samples=args.num_samples_per_source)

        sampled_node_x = sample_nodes(
            node_diffusion,
            batch=batch,
            device=device,
            show_progress=args.show_progress,
            log_every=args.log_every,
        )
        if args.mode == "joint":
            assert edge_diffusion is not None
            sampled_edge_x = sample_edges(
                edge_diffusion,
                batch=batch,
                sampled_node_x=sampled_node_x,
                device=device,
                edge_threshold=args.edge_threshold,
                show_progress=args.show_progress,
                log_every=args.log_every,
            )
        else:
            sampled_edge_x = source_edge_tensor(batch=batch, device=device)

        rows = decode_results(
            batch=batch,
            sampled_node_x=sampled_node_x,
            sampled_edge_x=sampled_edge_x,
            edge_threshold=args.edge_threshold,
            out_dir=out_dir / f"source_{dataset_index:05d}",
            save_images=args.save_images,
        )
        for row in rows:
            row["source_dataset_index"] = dataset_index
            row["mode"] = args.mode
        all_rows.extend(rows)

        if pos == 1 or pos % 10 == 0 or pos == len(indices):
            decoded = sum(row["generated_anchored_linker_smiles"] is not None for row in all_rows)
            assembled = sum(row["generated_full_smiles"] is not None for row in all_rows)
            print(
                f"[progress] source={pos}/{len(indices)} total_generated={len(all_rows)} "
                f"decoded={decoded} assembled={assembled}",
                flush=True,
            )

    summary = summarize_rows(all_rows)
    summary.update(
        {
            "mode": args.mode,
            "seed": args.seed,
            "num_samples_per_source": args.num_samples_per_source,
            "max_source_samples": args.max_source_samples,
            "start_index": args.start_index,
            "shuffle": args.shuffle,
            "tensor_pt": args.tensor_pt,
            "node_ckpt": args.node_ckpt,
            "edge_ckpt": args.edge_ckpt,
        }
    )
    per_source_rows = build_per_source_rows(all_rows)

    write_csv(all_rows, out_dir / "all_generations.csv")
    write_csv(per_source_rows, out_dir / "per_source_summary.csv")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / "all_generations.json").open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    if per_source_rows:
        plot_eval(per_source_rows, summary, out_dir / "evaluation_overview.png")

    print(f"[done] mode={args.mode} out_dir={out_dir}", flush=True)
    print(
        f"[summary] requested={summary['num_requested']} decoded={summary['num_decoded_anchored']} "
        f"assembled={summary['num_assembled_full']} decode_rate={summary['decode_rate']:.4f} "
        f"assembly_rate={summary['assembly_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
