from __future__ import annotations

import argparse

import torch
from rdkit import Chem

from models.anchor_gnn import AnchorGNN
from molgraph import encode_mol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict anchor nodes and attach [*:1]/[*:2] to a linker.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/anchor_gnn.pt", help="trained model checkpoint")
    parser.add_argument("--smiles", type=str, required=True, help="input linker smiles (without anchors)")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    return parser.parse_args()


def _sanitize_with_fallback(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(mol, sanitizeOps=flags)
            return mol
        except Exception:
            return None


def attach_anchor_dummies(mol: Chem.Mol, idx_l: int, idx_r: int) -> Chem.Mol | None:
    if idx_l == idx_r:
        return None
    rw = Chem.RWMol(mol)
    d1 = Chem.Atom(0)
    d1.SetAtomMapNum(1)
    d2 = Chem.Atom(0)
    d2.SetAtomMapNum(2)
    d1_idx = rw.AddAtom(d1)
    d2_idx = rw.AddAtom(d2)
    rw.AddBond(idx_l, d1_idx, Chem.BondType.SINGLE)
    rw.AddBond(idx_r, d2_idx, Chem.BondType.SINGLE)
    return _sanitize_with_fallback(rw.GetMol())


def select_distinct_anchors(logits: torch.Tensor) -> tuple[int, int]:
    left_scores = logits[:, 1]
    right_scores = logits[:, 2]

    pred_l = int(torch.argmax(left_scores).item())
    pred_r = int(torch.argmax(right_scores).item())
    if pred_l != pred_r:
        return pred_l, pred_r

    # If both heads choose the same node, pick the best alternative by pair score.
    top_l = torch.topk(left_scores, k=min(5, left_scores.shape[0])).indices.tolist()
    top_r = torch.topk(right_scores, k=min(5, right_scores.shape[0])).indices.tolist()
    best_pair = None
    best_score = -1e30
    for l_idx in top_l:
        for r_idx in top_r:
            if l_idx == r_idx:
                continue
            score = float(left_scores[l_idx].item() + right_scores[r_idx].item())
            if score > best_score:
                best_score = score
                best_pair = (int(l_idx), int(r_idx))
    if best_pair is not None:
        return best_pair

    # Last fallback for very small graphs.
    if logits.shape[0] >= 2:
        return 0, 1
    return 0, 0


def main() -> None:
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint = torch.load(args.ckpt, map_location=device)
    cfg = checkpoint["model_config"]
    model = AnchorGNN(
        in_dim=cfg["in_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_classes=cfg.get("num_classes", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {args.smiles}")
    graph = encode_mol(mol)

    x = graph["x"].to(device)
    edge_index = graph["edge_index"].to(device)
    with torch.no_grad():
        logits = model(x, edge_index)
    pred_l, pred_r = select_distinct_anchors(logits)

    anchored = attach_anchor_dummies(mol, pred_l, pred_r)
    if anchored is None:
        raise RuntimeError("Failed to attach anchor dummies")

    print(f"[input] {Chem.MolToSmiles(mol, canonical=True)}")
    print(f"[pred_anchor_idx] left={pred_l} right={pred_r}")
    print(f"[output] {Chem.MolToSmiles(anchored, canonical=True)}")


if __name__ == "__main__":
    main()
