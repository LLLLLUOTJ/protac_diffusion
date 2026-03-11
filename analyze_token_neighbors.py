from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def load_vocab(vocab_json: Path) -> Tuple[List[str], Dict[str, int]]:
    data = json.loads(vocab_json.read_text(encoding="utf-8"))
    if "tokens" in data:
        tokens = [str(t) for t in data["tokens"]]
    elif "id_to_token" in data:
        tokens = [str(t) for t in data["id_to_token"]]
    else:
        raise RuntimeError("token list not found in vocab json")
    if "token_to_id" in data:
        token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
    else:
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
    return tokens, token_to_id


def load_embeddings(
    embeddings_pt: Optional[Path],
    embeddings_npy: Optional[Path],
    expected_vocab_size: int,
) -> np.ndarray:
    if embeddings_pt is not None and embeddings_pt.exists():
        obj = torch.load(embeddings_pt, map_location="cpu")
        if isinstance(obj, dict) and "embeddings" in obj:
            emb = obj["embeddings"]
            if isinstance(emb, torch.Tensor):
                arr = emb.detach().cpu().numpy()
            else:
                arr = np.asarray(emb, dtype=np.float32)
        elif isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        else:
            raise RuntimeError("unsupported .pt embedding format")
    elif embeddings_npy is not None and embeddings_npy.exists():
        arr = np.load(embeddings_npy)
    else:
        raise FileNotFoundError("no embeddings file found")

    if arr.ndim != 2:
        raise RuntimeError(f"embedding array must be 2D, got shape={arr.shape}")
    if arr.shape[0] != expected_vocab_size:
        raise RuntimeError(
            f"embedding vocab mismatch: embedding_rows={arr.shape[0]} expected_vocab_size={expected_vocab_size}"
        )
    return arr.astype(np.float32)


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def nearest_neighbors(
    emb_norm: np.ndarray,
    tokens: Sequence[str],
    token_to_id: Dict[str, int],
    query_token: str,
    top_k: int,
) -> List[Tuple[str, float]]:
    if query_token not in token_to_id:
        raise KeyError(f"token not in vocab: {query_token}")
    idx = token_to_id[query_token]
    sims = emb_norm @ emb_norm[idx]
    order = np.argsort(-sims)
    out: List[Tuple[str, float]] = []
    for j in order:
        if int(j) == int(idx):
            continue
        out.append((tokens[int(j)], float(sims[int(j)])))
        if len(out) >= top_k:
            break
    return out


def token_group(token: str) -> str:
    # Simple motif tags for quick, interpretable diagnostics.
    if re.search(r"[cnosp]", token):
        return "aromatic_like"
    if "=O" in token:
        return "carbonyl_like"
    if "N" in token:
        return "amine_like"
    if "O" in token:
        return "oxygen_like"
    return "aliphatic_like"


def mean_pairwise_cosine(emb_norm: np.ndarray, indices: Sequence[int]) -> float:
    idx = list(indices)
    if len(idx) < 2:
        return float("nan")
    mat = emb_norm[idx]
    sims = mat @ mat.T
    n = len(idx)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = sims[mask]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def mean_cross_group_cosine(emb_norm: np.ndarray, a_idx: Sequence[int], b_idx: Sequence[int]) -> float:
    if not a_idx or not b_idx:
        return float("nan")
    sims = emb_norm[list(a_idx)] @ emb_norm[list(b_idx)].T
    if sims.size == 0:
        return float("nan")
    return float(sims.mean())


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze nearest neighbors of core token embeddings")
    parser.add_argument(
        "--vocab_json",
        type=str,
        default="data/processed/core_token_embedding/token_vocab.json",
    )
    parser.add_argument(
        "--embeddings_pt",
        type=str,
        default="data/processed/core_token_embedding/token_embeddings.pt",
    )
    parser.add_argument(
        "--embeddings_npy",
        type=str,
        default="data/processed/core_token_embedding/token_embeddings.npy",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--query_tokens",
        type=str,
        default="",
        help="comma-separated tokens; default analyzes all tokens",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="data/processed/core_token_embedding/token_neighbor_report.json",
    )
    parser.add_argument("--print_all", type=str, default="false")
    return parser


def parse_query_tokens(text: str) -> List[str]:
    if not text.strip():
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean: {text}")


def main() -> None:
    args = make_parser().parse_args()
    vocab_json = Path(args.vocab_json)
    embeddings_pt = Path(args.embeddings_pt) if args.embeddings_pt else None
    embeddings_npy = Path(args.embeddings_npy) if args.embeddings_npy else None

    tokens, token_to_id = load_vocab(vocab_json)
    emb = load_embeddings(embeddings_pt, embeddings_npy, expected_vocab_size=len(tokens))
    emb_norm = l2_normalize(emb)
    top_k = int(args.top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_tokens = parse_query_tokens(str(args.query_tokens))
    if not query_tokens:
        query_tokens = list(tokens)
    query_tokens = [tok for tok in query_tokens if tok in token_to_id]
    if not query_tokens:
        raise RuntimeError("no valid query tokens after filtering")

    report_neighbors: Dict[str, List[Dict[str, float | str]]] = {}
    for tok in query_tokens:
        nn = nearest_neighbors(emb_norm, tokens, token_to_id, tok, top_k=top_k)
        report_neighbors[tok] = [{"token": t, "cosine": float(s)} for t, s in nn]

    # Motif-level quick diagnostics.
    groups: Dict[str, List[int]] = {}
    for tok in tokens:
        grp = token_group(tok)
        groups.setdefault(grp, []).append(token_to_id[tok])

    intra: Dict[str, float] = {}
    for grp, idx in groups.items():
        if len(idx) >= 2:
            intra[grp] = mean_pairwise_cosine(emb_norm, idx)

    inter: Dict[str, float] = {}
    group_names = sorted(groups.keys())
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i + 1 :]:
            v = mean_cross_group_cosine(emb_norm, groups[g1], groups[g2])
            inter[f"{g1}__vs__{g2}"] = v

    report = {
        "vocab_size": len(tokens),
        "embedding_dim": int(emb.shape[1]),
        "top_k": top_k,
        "query_tokens": query_tokens,
        "neighbors": report_neighbors,
        "motif_groups": {k: [tokens[i] for i in v] for k, v in groups.items()},
        "motif_intra_group_mean_cosine": intra,
        "motif_inter_group_mean_cosine": inter,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[info] vocab_size={len(tokens)} embedding_dim={int(emb.shape[1])}", flush=True)
    for tok in query_tokens:
        neighbors = report_neighbors[tok]
        if parse_bool(str(args.print_all)) or tok in query_tokens[: min(12, len(query_tokens))]:
            readable = ", ".join(f"{x['token']}({x['cosine']:.3f})" for x in neighbors)
            print(f"[neighbors] {tok}: {readable}", flush=True)
    print(f"[done] report={out_json}", flush=True)


if __name__ == "__main__":
    main()
