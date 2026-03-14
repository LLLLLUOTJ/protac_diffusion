from __future__ import annotations

import argparse
from pathlib import Path

from data.weak_anchor_token_dataset import WeakAnchorTokenDataset, serialize_weak_anchor_token_dataset


def parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean: {text}")


def _resolve_optional_path(value: str, candidates: list[str], label: str) -> str:
    text = str(value).strip()
    if text:
        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return str(path)
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"could not resolve {label}; tried candidates: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak-anchor token diffusion dataset from anchored linkers")
    parser.add_argument("--weak_anchor_csv", type=str, default="outputs/weak_anchor_best/weak_anchor_dataset.csv")
    parser.add_argument("--token_vocab_json", type=str, default="")
    parser.add_argument("--token_embeddings_pt", type=str, default="")
    parser.add_argument("--out_pt", type=str, default="data/processed/weak_anchor_token_tensors.pt")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--include_ring_single_bonds", action="store_true")
    parser.add_argument("--pad_to_length", type=int, default=0)
    parser.add_argument("--pad_token", type=str, default="<PAD>")
    parser.add_argument("--reject_overlength", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token_vocab_json = _resolve_optional_path(
        args.token_vocab_json,
        candidates=[
            "data/processed/task/oriented_token_embedding/token_vocab.json",
            "data/processed/task/oriented_token_embedding_smoke/token_vocab.json",
            "data/processed/task/oriented_token_embedding_full/token_vocab.json",
        ],
        label="token_vocab_json",
    )
    token_embeddings_pt = _resolve_optional_path(
        args.token_embeddings_pt,
        candidates=[
            "data/processed/task/oriented_token_embedding/token_embeddings.pt",
            "data/processed/task/oriented_token_embedding_smoke/token_embeddings.pt",
            "data/processed/task/oriented_token_embedding_full/token_embeddings.pt",
        ],
        label="token_embeddings_pt",
    )

    dataset = WeakAnchorTokenDataset(
        csv_path=args.weak_anchor_csv,
        token_vocab_json=token_vocab_json,
        token_embeddings_pt=token_embeddings_pt,
        max_samples=args.max_samples,
        include_ring_single_bonds=bool(args.include_ring_single_bonds),
        pad_to_length=int(args.pad_to_length),
        pad_token=str(args.pad_token),
        reject_overlength=bool(args.reject_overlength),
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid weak-anchor token samples found")

    serialize_weak_anchor_token_dataset(dataset, out_path=args.out_pt)
    print(
        f"[done] samples={len(dataset)} reasons={dataset.reason_counts} out={args.out_pt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
