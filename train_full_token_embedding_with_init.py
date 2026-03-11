from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from train_core_token_embedding import (
    SkipGramNS,
    build_skipgram_pairs,
    choose_device,
    map_sequences_to_ids,
    read_tokenized_sequences,
    set_seed,
)


def parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean: {text}")


def load_core_embedding(core_embedding_pt: Path) -> Tuple[torch.Tensor, Dict[str, int], List[str]]:
    obj = torch.load(core_embedding_pt, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError("core embedding file must be a dict with embeddings/token_to_id")
    if "embeddings" not in obj:
        raise RuntimeError("missing embeddings in core embedding file")
    emb = obj["embeddings"]
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb, dtype=torch.float32)
    token_to_id = {str(k): int(v) for k, v in obj["token_to_id"].items()}
    if "id_to_token" in obj:
        id_to_token = [str(t) for t in obj["id_to_token"]]
    else:
        id_to_token = [None] * len(token_to_id)  # type: ignore[list-item]
        for tok, idx in token_to_id.items():
            id_to_token[idx] = tok
    return emb.detach().cpu().float(), token_to_id, id_to_token


def load_full_vocab_from_coverage(coverage_csv: Path) -> Tuple[List[str], Dict[str, int]]:
    rows = list(csv.DictReader(coverage_csv.open("r", encoding="utf-8", newline="")))
    parsed: List[Tuple[str, int, int]] = []
    for row in rows:
        tok = str(row.get("token_smiles", "")).strip()
        if not tok:
            continue
        try:
            freq = int(str(row.get("frequency", "0")).strip())
        except Exception:
            freq = 0
        try:
            rank = int(str(row.get("rank", "0")).strip())
        except Exception:
            rank = 0
        parsed.append((tok, freq, rank))
    parsed.sort(key=lambda x: (-x[1], x[2], x[0]))
    tokens = [tok for tok, _, _ in parsed]
    freq_map = {tok: freq for tok, freq, _ in parsed}
    return tokens, freq_map


def load_assignment_map(assignment_csv: Path) -> Dict[str, Dict[str, str]]:
    rows = list(csv.DictReader(assignment_csv.open("r", encoding="utf-8", newline="")))
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        low = str(row.get("low_token", "")).strip()
        if not low:
            continue
        out[low] = {k: str(v) for k, v in row.items()}
    return out


def build_initialized_matrix(
    full_tokens: List[str],
    full_freq: Dict[str, int],
    core_emb: torch.Tensor,
    core_token_to_id: Dict[str, int],
    assignment_map: Dict[str, Dict[str, str]],
    use_uncertain_suggestions: bool,
    noise_std: float,
    random_std: float,
    seed: int,
) -> Tuple[torch.Tensor, List[Dict[str, str]]]:
    rng = np.random.default_rng(seed)
    dim = int(core_emb.shape[1])
    mat = np.zeros((len(full_tokens), dim), dtype=np.float32)
    rows: List[Dict[str, str]] = []

    for idx, tok in enumerate(full_tokens):
        source = ""
        mapped_core = ""
        conf = ""
        score = ""
        if tok in core_token_to_id:
            vec = core_emb[core_token_to_id[tok]].numpy().astype(np.float32)
            source = "core_exact"
        else:
            assign = assignment_map.get(tok)
            if assign is not None:
                mapped_core = str(assign.get("assigned_core_token", "")).strip()
                conf = str(assign.get("confidence", "")).strip()
                score = str(assign.get("score", "")).strip()
                can_map = mapped_core in core_token_to_id and (
                    conf in {"high", "medium"} or (use_uncertain_suggestions and conf == "uncertain")
                )
                if can_map:
                    base = core_emb[core_token_to_id[mapped_core]].numpy().astype(np.float32)
                    vec = base + rng.normal(0.0, noise_std, size=base.shape).astype(np.float32)
                    source = f"mapped_{conf}"
                else:
                    vec = rng.normal(0.0, random_std, size=(dim,)).astype(np.float32)
                    source = "random_unmapped"
            else:
                vec = rng.normal(0.0, random_std, size=(dim,)).astype(np.float32)
                source = "random_missing_assignment"
        mat[idx] = vec
        rows.append(
            {
                "token": tok,
                "frequency": str(int(full_freq.get(tok, 0))),
                "init_source": source,
                "mapped_core_token": mapped_core,
                "mapped_confidence": conf,
                "mapped_score": score,
            }
        )
    return torch.tensor(mat, dtype=torch.float32), rows


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm-start full token embedding from core embedding + lowfreq mapping")
    parser.add_argument(
        "--tokenized_csv",
        type=str,
        default="data/processed/linker_anchor_tokenized_multi.csv",
    )
    parser.add_argument(
        "--coverage_csv",
        type=str,
        default="data/processed/linker_anchor_token_coverage_stats.csv",
    )
    parser.add_argument(
        "--assignment_csv",
        type=str,
        default="data/processed/lowfreq_to_core_assignment.csv",
    )
    parser.add_argument(
        "--core_embedding_pt",
        type=str,
        default="data/processed/core_token_embedding/token_embeddings.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/full_token_embedding_initialized",
    )
    parser.add_argument("--use_uncertain_suggestions", type=parse_bool, default=False)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--negative_samples", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.015)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--random_std", type=float, default=0.05)
    parser.add_argument("--l2_normalize", type=parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log_every", type=int, default=10)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    set_seed(int(args.seed))

    tokenized_csv = Path(args.tokenized_csv)
    coverage_csv = Path(args.coverage_csv)
    assignment_csv = Path(args.assignment_csv)
    core_embedding_pt = Path(args.core_embedding_pt)
    for p in [tokenized_csv, coverage_csv, assignment_csv, core_embedding_pt]:
        if not p.exists():
            raise FileNotFoundError(f"missing required file: {p}")

    core_emb, core_token_to_id, core_tokens = load_core_embedding(core_embedding_pt)
    core_dim = int(core_emb.shape[1])
    if args.embedding_dim is not None and int(args.embedding_dim) != core_dim:
        raise ValueError(f"embedding_dim mismatch: requested={args.embedding_dim}, core_dim={core_dim}")

    full_tokens, full_freq = load_full_vocab_from_coverage(coverage_csv)
    full_token_to_id = {tok: i for i, tok in enumerate(full_tokens)}
    assignment_map = load_assignment_map(assignment_csv)
    init_mat, init_rows = build_initialized_matrix(
        full_tokens=full_tokens,
        full_freq=full_freq,
        core_emb=core_emb,
        core_token_to_id=core_token_to_id,
        assignment_map=assignment_map,
        use_uncertain_suggestions=bool(args.use_uncertain_suggestions),
        noise_std=float(args.noise_std),
        random_std=float(args.random_std),
        seed=int(args.seed),
    )

    sequences = read_tokenized_sequences(tokenized_csv)
    mapped = map_sequences_to_ids(sequences, token_to_id=full_token_to_id, unk_token=None)
    centers, contexts, pair_weights = build_skipgram_pairs(mapped, window_size=int(args.window_size))

    device = choose_device(str(args.device))
    model = SkipGramNS(vocab_size=len(full_tokens), embed_dim=core_dim).to(device)
    with torch.no_grad():
        model.in_embed.weight.copy_(init_mat.to(device))
        model.out_embed.weight.copy_(init_mat.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    token_counts = np.array([max(1, int(full_freq.get(tok, 1))) for tok in full_tokens], dtype=np.float64)
    token_counts = np.power(token_counts, 0.75)
    token_counts /= np.clip(token_counts.sum(), a_min=1e-12, a_max=None)
    neg_dist = torch.tensor(token_counts, dtype=torch.float32, device=device)

    centers = centers.to(device)
    contexts = contexts.to(device)
    pair_weights = pair_weights.to(device)
    num_pairs = centers.shape[0]
    batch_size = int(args.batch_size)
    neg_k = int(args.negative_samples)
    log_every = max(1, int(args.log_every))

    print(
        f"[data] sequences={len(sequences)} vocab={len(full_tokens)} pairs={num_pairs} "
        f"core_vocab={len(core_tokens)} device={device}",
        flush=True,
    )
    for epoch in range(1, int(args.epochs) + 1):
        perm = torch.randperm(num_pairs, device=device)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, num_pairs, batch_size):
            idx = perm[start : start + batch_size]
            c = centers[idx]
            p = contexts[idx]
            w = pair_weights[idx]
            bsz = c.shape[0]
            neg = torch.multinomial(neg_dist, num_samples=bsz * neg_k, replacement=True).view(bsz, neg_k)
            optimizer.zero_grad(set_to_none=True)
            loss = model(c, p, neg, pair_weights=w)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * float(w.sum().item())
            total_weight += float(w.sum().item())
        mean_loss = total_loss / max(total_weight, 1e-8)
        if epoch == 1 or epoch % log_every == 0 or epoch == int(args.epochs):
            print(f"[epoch {epoch:03d}] weighted_loss={mean_loss:.6f}", flush=True)

    emb = model.export_embeddings(l2_normalize=bool(args.l2_normalize)).detach().cpu()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_json = out_dir / "token_vocab.json"
    emb_pt = out_dir / "token_embeddings.pt"
    emb_npy = out_dir / "token_embeddings.npy"
    init_csv = out_dir / "token_init_sources.csv"
    summary_json = out_dir / "training_summary.json"

    vocab_payload = {
        "tokens": full_tokens,
        "token_to_id": full_token_to_id,
        "id_to_token": full_tokens,
        "token_frequency": {tok: int(full_freq.get(tok, 0)) for tok in full_tokens},
        "core_vocab_size": len(core_tokens),
    }
    vocab_json.write_text(json.dumps(vocab_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(
        {
            "embeddings": emb,
            "token_to_id": full_token_to_id,
            "id_to_token": full_tokens,
            "embedding_dim": core_dim,
            "config": vars(args),
        },
        emb_pt,
    )
    np.save(emb_npy, emb.numpy())

    with init_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["token", "frequency", "init_source", "mapped_core_token", "mapped_confidence", "mapped_score"],
        )
        writer.writeheader()
        writer.writerows(init_rows)

    init_source_counts: Dict[str, int] = {}
    for row in init_rows:
        init_source_counts[row["init_source"]] = init_source_counts.get(row["init_source"], 0) + 1

    summary = {
        "tokenized_csv": str(tokenized_csv),
        "coverage_csv": str(coverage_csv),
        "assignment_csv": str(assignment_csv),
        "core_embedding_pt": str(core_embedding_pt),
        "num_sequences": len(sequences),
        "num_pairs": int(num_pairs),
        "vocab_size": len(full_tokens),
        "core_vocab_size": len(core_tokens),
        "embedding_dim": core_dim,
        "device": str(device),
        "init_source_counts": init_source_counts,
        "use_uncertain_suggestions": bool(args.use_uncertain_suggestions),
        "files": {
            "token_vocab_json": str(vocab_json),
            "token_embeddings_pt": str(emb_pt),
            "token_embeddings_npy": str(emb_npy),
            "token_init_sources_csv": str(init_csv),
        },
        "config": vars(args),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[done] vocab={len(full_tokens)} embedding_dim={core_dim} out_dir={out_dir}",
        flush=True,
    )
    print(f"[init] counts={init_source_counts}", flush=True)
    print(f"[files] vocab={vocab_json} emb_pt={emb_pt} emb_npy={emb_npy} init={init_csv}", flush=True)


if __name__ == "__main__":
    main()
