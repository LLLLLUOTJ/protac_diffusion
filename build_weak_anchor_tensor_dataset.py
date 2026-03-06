from __future__ import annotations

import argparse

from data.anchored_tensor_dataset import (
    WeakAnchorTensorDataset,
    serialize_weak_anchor_tensor_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tensor dataset with linker/left/right graph tensors from weak_anchor_dataset.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/weak_anchor_best/weak_anchor_dataset.csv",
        help="input weak-anchor csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/weak_anchor_tensors.pt",
        help="output .pt file",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="optional cap for quick debugging")
    parser.add_argument(
        "--include-pair-mask",
        action="store_true",
        help="also store [N,N] pair mask for linker graph bond-add candidates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = WeakAnchorTensorDataset(
        csv_path=args.csv,
        max_samples=args.max_samples,
        include_pair_mask=args.include_pair_mask,
    )
    serialize_weak_anchor_tensor_dataset(
        dataset=dataset,
        out_path=args.out,
        include_pair_mask=args.include_pair_mask,
    )

    print(
        f"[done] samples={len(dataset)} reasons={dataset.reason_counts} "
        f"include_pair_mask={args.include_pair_mask} out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
