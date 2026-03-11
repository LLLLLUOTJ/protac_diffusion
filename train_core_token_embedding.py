from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean: {text}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SampleSequence:
    sample_id: str
    linker_id: str
    tokens: List[str]
    sample_weight: float


def read_tokenized_sequences(tokenized_csv: Path) -> List[SampleSequence]:
    rows: List[SampleSequence] = []
    with tokenized_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            linker_id = str(row.get("linker_id", "")).strip()
            seq_text = str(row.get("token_smiles_list_json", "")).strip()
            if not seq_text:
                continue
            try:
                tokens = list(json.loads(seq_text))
            except Exception as exc:
                raise RuntimeError(f"failed to parse token_smiles_list_json for sample_id={sample_id}") from exc
            tokens = [str(t) for t in tokens if str(t).strip()]
            if not tokens:
                continue
            weight_text = str(row.get("sample_weight", "1.0")).strip()
            try:
                sample_weight = float(weight_text)
            except Exception:
                sample_weight = 1.0
            rows.append(
                SampleSequence(
                    sample_id=sample_id,
                    linker_id=linker_id,
                    tokens=tokens,
                    sample_weight=sample_weight,
                )
            )
    if not rows:
        raise RuntimeError(f"no valid sequences loaded from: {tokenized_csv}")
    return rows


def read_sequences_from_instances(instances_csv: Path) -> List[SampleSequence]:
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    linker_of: Dict[str, str] = {}
    weight_of: Dict[str, float] = {}
    with instances_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            token = str(row.get("token_smiles", "")).strip()
            if not sample_id or not token:
                continue
            try:
                token_index = int(str(row.get("token_index", "0")).strip())
            except Exception:
                token_index = 0
            grouped.setdefault(sample_id, []).append((token_index, token))
            linker_of[sample_id] = str(row.get("linker_id", "")).strip()
            try:
                weight_of[sample_id] = float(str(row.get("sample_weight", "1.0")).strip())
            except Exception:
                weight_of[sample_id] = 1.0

    rows: List[SampleSequence] = []
    for sample_id, indexed_tokens in grouped.items():
        indexed_tokens.sort(key=lambda x: x[0])
        tokens = [tok for _, tok in indexed_tokens]
        if not tokens:
            continue
        rows.append(
            SampleSequence(
                sample_id=sample_id,
                linker_id=linker_of.get(sample_id, ""),
                tokens=tokens,
                sample_weight=weight_of.get(sample_id, 1.0),
            )
        )
    if not rows:
        raise RuntimeError(f"no valid sequences loaded from: {instances_csv}")
    return rows


def build_vocab(
    sequences: Sequence[SampleSequence],
    min_count: int = 1,
    add_unk: bool = True,
    unk_token: str = "<UNK>",
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    freq: Dict[str, int] = {}
    for sample in sequences:
        for token in sample.tokens:
            freq[token] = freq.get(token, 0) + 1
    kept = [tok for tok, count in freq.items() if count >= min_count]
    kept.sort(key=lambda t: (-freq[t], t))
    vocab_tokens = kept[:]
    if add_unk and unk_token not in vocab_tokens:
        vocab_tokens.append(unk_token)
    token_to_id = {tok: i for i, tok in enumerate(vocab_tokens)}
    return vocab_tokens, token_to_id, freq


def map_sequences_to_ids(
    sequences: Sequence[SampleSequence],
    token_to_id: Dict[str, int],
    unk_token: Optional[str] = "<UNK>",
) -> List[Tuple[List[int], float, str, str]]:
    mapped: List[Tuple[List[int], float, str, str]] = []
    unk_id = token_to_id.get(unk_token, -1) if unk_token else -1
    for sample in sequences:
        ids: List[int] = []
        for token in sample.tokens:
            if token in token_to_id:
                ids.append(token_to_id[token])
            elif unk_id >= 0:
                ids.append(unk_id)
        if len(ids) < 2:
            continue
        mapped.append((ids, float(sample.sample_weight), sample.sample_id, sample.linker_id))
    if not mapped:
        raise RuntimeError("no mapped sequences with length >= 2")
    return mapped


def build_skipgram_pairs(
    mapped_sequences: Sequence[Tuple[List[int], float, str, str]],
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers: List[int] = []
    contexts: List[int] = []
    weights: List[float] = []
    for ids, sample_weight, _, _ in mapped_sequences:
        n = len(ids)
        for i in range(n):
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            center = ids[i]
            for j in range(left, right):
                if j == i:
                    continue
                centers.append(center)
                contexts.append(ids[j])
                weights.append(sample_weight)
    if not centers:
        raise RuntimeError("no skip-gram pairs generated")
    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.xavier_uniform_(self.out_embed.weight)

    def forward(
        self,
        center_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        pair_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        center_vec = self.in_embed(center_ids)  # [B, D]
        pos_vec = self.out_embed(pos_ids)  # [B, D]
        pos_score = torch.sum(center_vec * pos_vec, dim=1)  # [B]
        pos_logprob = F.logsigmoid(pos_score)  # [B]

        neg_vec = self.out_embed(neg_ids)  # [B, K, D]
        neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(2)).squeeze(2)  # [B, K]
        neg_logprob = F.logsigmoid(-neg_score).sum(dim=1)  # [B]

        per_pair_loss = -(pos_logprob + neg_logprob)  # [B]
        if pair_weights is not None:
            denom = torch.clamp(pair_weights.sum(), min=1e-8)
            return torch.sum(per_pair_loss * pair_weights) / denom
        return per_pair_loss.mean()

    def export_embeddings(self, l2_normalize: bool = True) -> torch.Tensor:
        emb = 0.5 * (self.in_embed.weight.data + self.out_embed.weight.data)
        if l2_normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train minimal core linker token embeddings")
    parser.add_argument(
        "--tokenized_csv",
        type=str,
        default="data/processed/linker_anchor_tokenized_core10.csv",
        help="CSV with token_smiles_list_json per sample",
    )
    parser.add_argument(
        "--instances_csv",
        type=str,
        default=None,
        help="Fallback CSV with token_smiles/token_index per sample if tokenized_csv is unavailable",
    )
    parser.add_argument(
        "--library_csv",
        type=str,
        default="data/processed/linker_anchor_fragment_library_core10.csv",
        help="Optional frequency reference table",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/core_token_embedding",
    )
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--negative_samples", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--add_unk", type=parse_bool, default=False)
    parser.add_argument("--unk_token", type=str, default="<UNK>")
    parser.add_argument("--l2_normalize", type=parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log_every", type=int, default=10)
    return parser


def load_reference_frequency(library_csv: Optional[Path]) -> Dict[str, int]:
    if library_csv is None or (not library_csv.exists()):
        return {}
    freq: Dict[str, int] = {}
    with library_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = str(row.get("token_smiles", "")).strip()
            if not token:
                continue
            try:
                val = int(str(row.get("frequency", "0")).strip())
            except Exception:
                val = 0
            freq[token] = val
    return freq


def choose_device(user_device: str) -> torch.device:
    if user_device == "cpu":
        return torch.device("cpu")
    if user_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = make_parser().parse_args()
    set_seed(int(args.seed))

    tokenized_csv = Path(args.tokenized_csv)
    instances_csv = Path(args.instances_csv) if args.instances_csv else None
    if tokenized_csv.exists():
        sequences = read_tokenized_sequences(tokenized_csv)
        sequence_source = str(tokenized_csv)
    elif instances_csv is not None and instances_csv.exists():
        sequences = read_sequences_from_instances(instances_csv)
        sequence_source = str(instances_csv)
    else:
        raise FileNotFoundError("neither tokenized_csv nor instances_csv could be loaded")

    vocab_tokens, token_to_id, seq_freq = build_vocab(
        sequences=sequences,
        min_count=int(args.min_count),
        add_unk=bool(args.add_unk),
        unk_token=str(args.unk_token),
    )
    mapped = map_sequences_to_ids(
        sequences=sequences,
        token_to_id=token_to_id,
        unk_token=str(args.unk_token) if bool(args.add_unk) else None,
    )
    centers, contexts, pair_weights = build_skipgram_pairs(
        mapped_sequences=mapped,
        window_size=int(args.window_size),
    )

    vocab_size = len(vocab_tokens)
    device = choose_device(str(args.device))
    model = SkipGramNS(vocab_size=vocab_size, embed_dim=int(args.embedding_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    token_counts = np.array([seq_freq.get(tok, 1) for tok in vocab_tokens], dtype=np.float64)
    token_counts = np.power(token_counts, 0.75)
    token_counts /= np.clip(token_counts.sum(), a_min=1e-12, a_max=None)
    neg_dist = torch.tensor(token_counts, dtype=torch.float32, device=device)

    num_pairs = centers.shape[0]
    batch_size = int(args.batch_size)
    neg_k = int(args.negative_samples)
    epochs = int(args.epochs)
    log_every = max(1, int(args.log_every))

    centers = centers.to(device)
    contexts = contexts.to(device)
    pair_weights = pair_weights.to(device)

    print(
        f"[data] source={sequence_source} sequences={len(sequences)} vocab={vocab_size} "
        f"pairs={num_pairs} device={device}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
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
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"[epoch {epoch:03d}] weighted_loss={mean_loss:.6f}", flush=True)

    emb = model.export_embeddings(l2_normalize=bool(args.l2_normalize)).detach().cpu()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "token_vocab.json"
    emb_pt_path = out_dir / "token_embeddings.pt"
    emb_npy_path = out_dir / "token_embeddings.npy"
    meta_path = out_dir / "training_summary.json"

    ref_freq = load_reference_frequency(Path(args.library_csv) if args.library_csv else None)
    token_frequency = {tok: int(seq_freq.get(tok, 0)) for tok in vocab_tokens}
    token_frequency_ref = {tok: int(ref_freq.get(tok, 0)) for tok in vocab_tokens if tok in ref_freq}

    vocab_payload = {
        "tokens": vocab_tokens,
        "token_to_id": token_to_id,
        "id_to_token": vocab_tokens,
        "token_frequency_from_sequences": token_frequency,
        "token_frequency_from_library_csv": token_frequency_ref,
        "notes": "Token strings preserve attachment semantics, including '*' and mapped motif forms.",
    }
    vocab_path.write_text(json.dumps(vocab_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    torch.save(
        {
            "embeddings": emb,
            "token_to_id": token_to_id,
            "id_to_token": vocab_tokens,
            "embedding_dim": int(args.embedding_dim),
            "l2_normalize": bool(args.l2_normalize),
            "token_frequency": token_frequency,
            "config": vars(args),
        },
        emb_pt_path,
    )
    np.save(emb_npy_path, emb.numpy())

    summary = {
        "sequence_source": sequence_source,
        "num_sequences": len(sequences),
        "num_pairs": int(num_pairs),
        "vocab_size": vocab_size,
        "embedding_dim": int(args.embedding_dim),
        "device": str(device),
        "out_dir": str(out_dir),
        "files": {
            "token_vocab_json": str(vocab_path),
            "token_embeddings_pt": str(emb_pt_path),
            "token_embeddings_npy": str(emb_npy_path),
        },
        "config": vars(args),
    }
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[done] vocab={vocab_size} embedding_dim={int(args.embedding_dim)} "
        f"out_dir={out_dir}",
        flush=True,
    )
    print(f"[files] vocab={vocab_path} emb_pt={emb_pt_path} emb_npy={emb_npy_path}", flush=True)


if __name__ == "__main__":
    main()
