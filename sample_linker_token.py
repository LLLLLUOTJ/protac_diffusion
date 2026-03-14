from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from rdkit import Chem

from data.weak_anchor_token_dataset import WeakAnchorTokenPTDataset
from data.weak_anchor_token_diffusion import collate_weak_anchor_token_diffusion_batch
from diffusion.ddpm import DDPM
from models.fragment_conditioned_denoiser import FragmentConditionedTokenDenoiser
from sampling.linker_generation import assemble_full_molecule, draw_molecule
from sampling.token_linker_codec import decode_oriented_embedding_sequence_to_linker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample anchored PROTAC linkers from token diffusion")
    parser.add_argument("--tensor-pt", type=str, default="data/processed/weak_anchor_token_tensors.pt")
    parser.add_argument("--ckpt", type=str, default="checkpoints/linker_token_diffusion.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/linker_token_sampling")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    return obj


def load_token_diffusion(ckpt_path: str, device: torch.device) -> DDPM:
    payload = torch.load(ckpt_path, map_location="cpu")
    model = FragmentConditionedTokenDenoiser(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    diffusion = DDPM(model=model, device=str(device), **payload["diffusion_config"]).to(device)
    diffusion.eval()
    return diffusion


def select_source_sample(dataset: WeakAnchorTokenPTDataset, sample_index: int, sample_id: str | None) -> Dict[str, Any]:
    if sample_id is not None:
        for record in dataset:
            if str(record["sample_id"]) == str(sample_id):
                return record
        raise KeyError(f"sample_id not found: {sample_id}")
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index out of range: {sample_index}")
    return dataset[sample_index]


def build_sampling_batch(sample: Dict[str, Any], num_samples: int) -> Dict[str, Any]:
    return collate_weak_anchor_token_diffusion_batch([sample for _ in range(num_samples)])


def sample_tokens(
    diffusion: DDPM,
    batch: Dict[str, Any],
    device: torch.device,
    *,
    show_progress: bool,
    log_every: int,
) -> torch.Tensor:
    linker_token = move_to_device(batch["linker_token"], device)
    model_kwargs = {
        "left_graph": move_to_device(batch["left_graph"], device),
        "right_graph": move_to_device(batch["right_graph"], device),
        "token_mask": linker_token["sample_mask"],
    }

    def post_step(x: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        return x / norms

    sampled = diffusion.sample(
        shape=tuple(linker_token["x_start"].shape),
        device=str(device),
        show_progress=show_progress,
        log_every=log_every,
        sample_mask=linker_token["sample_mask"],
        fixed_mask=linker_token["fixed_mask"],
        fixed_values=linker_token["fixed_values"],
        post_step_fn=post_step,
        model_kwargs=model_kwargs,
    )
    return F.normalize(sampled, p=2, dim=-1)


def decode_results(
    dataset: WeakAnchorTokenPTDataset,
    batch: Dict[str, Any],
    sampled_token_x: torch.Tensor,
    *,
    out_dir: Path,
    save_images: bool,
) -> List[Dict[str, Any]]:
    vocab_tokens = list(dataset.vocab_tokens)
    vocab_embeddings = dataset.token_embeddings
    if vocab_embeddings is None:
        raise RuntimeError("token dataset metadata missing token_embeddings")

    rows: List[Dict[str, Any]] = []
    mask = batch["linker_token"]["sample_mask"].bool()
    for idx in range(sampled_token_x.shape[0]):
        length = int(mask[idx].sum().item())
        decoded = decode_oriented_embedding_sequence_to_linker(
            token_embeddings=sampled_token_x[idx, :length].cpu(),
            vocab_embeddings=vocab_embeddings,
            vocab_tokens=vocab_tokens,
        )

        left_smiles = batch["left_fragment_smiles"][idx]
        right_smiles = batch["right_fragment_smiles"][idx]
        left_mol = Chem.MolFromSmiles(left_smiles)
        right_mol = Chem.MolFromSmiles(right_smiles)
        full_mol = None
        full_reason = None
        if decoded["mol"] is not None and left_mol is not None and right_mol is not None:
            full_mol, full_reason = assemble_full_molecule(
                left_fragment=left_mol,
                anchored_linker=decoded["mol"],
                right_fragment=right_mol,
            )
        else:
            full_reason = "linker_decode_failed_or_fragment_parse_failed"

        row = {
            "repeat_index": idx,
            "sample_id": batch["sample_id"][idx],
            "protac_id": batch["protac_id"][idx],
            "linker_id": batch["linker_id"][idx],
            "source_anchored_linker_smiles": batch["anchored_linker_smiles"][idx],
            "source_left_fragment_smiles": left_smiles,
            "source_right_fragment_smiles": right_smiles,
            "source_oriented_token_smiles": json.dumps(batch["oriented_token_smiles"][idx], ensure_ascii=False),
            "generated_oriented_token_smiles": json.dumps(decoded["oriented_token_smiles"], ensure_ascii=False),
            "generated_anchored_linker_smiles": decoded["anchored_linker_smiles"],
            "generated_full_smiles": Chem.MolToSmiles(full_mol, canonical=True) if full_mol is not None else None,
            "decode_reason": decoded["reason"],
            "assemble_reason": full_reason,
        }
        rows.append(row)

        if save_images and decoded["mol"] is not None:
            draw_molecule(decoded["mol"], str(out_dir / f"sample_{idx:03d}_anchored_linker.png"))
        if save_images and full_mol is not None:
            draw_molecule(full_mol, str(out_dir / f"sample_{idx:03d}_full.png"))

    return rows


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "generated_samples.csv"
    json_path = out_dir / "generated_samples.json"
    summary_path = out_dir / "summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    anchored = [row["generated_anchored_linker_smiles"] for row in rows if row["generated_anchored_linker_smiles"]]
    full = [row["generated_full_smiles"] for row in rows if row["generated_full_smiles"]]
    summary = {
        "num_requested": len(rows),
        "num_decoded_anchored": len(anchored),
        "num_assembled_full": len(full),
        "decode_rate": (len(anchored) / len(rows)) if rows else 0.0,
        "assembly_rate": (len(full) / len(rows)) if rows else 0.0,
        "unique_anchored": len(set(anchored)),
        "unique_full": len(set(full)),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    dataset = WeakAnchorTokenPTDataset(pt_path=args.tensor_pt)
    sample = select_source_sample(dataset, sample_index=args.sample_index, sample_id=args.sample_id)
    batch = build_sampling_batch(sample, num_samples=args.num_samples)

    print(
        f"[sample] device={device} source_sample_id={sample['sample_id']} "
        f"num_samples={args.num_samples}",
        flush=True,
    )
    print(
        f"[source] anchored={sample['anchored_linker_smiles']} "
        f"tokens={sample['oriented_token_smiles']}",
        flush=True,
    )

    diffusion = load_token_diffusion(args.ckpt, device=device)
    sampled = sample_tokens(
        diffusion,
        batch=batch,
        device=device,
        show_progress=args.show_progress,
        log_every=args.log_every,
    )

    out_dir = Path(args.out_dir)
    rows = decode_results(
        dataset=dataset,
        batch=batch,
        sampled_token_x=sampled,
        out_dir=out_dir,
        save_images=args.save_images,
    )
    if not rows:
        raise RuntimeError("No decoded token samples were produced")
    write_outputs(rows, out_dir=out_dir)

    decoded = sum(row["generated_anchored_linker_smiles"] is not None for row in rows)
    assembled = sum(row["generated_full_smiles"] is not None for row in rows)
    print(
        f"[done] decoded={decoded}/{len(rows)} assembled={assembled}/{len(rows)} "
        f"csv={out_dir / 'generated_samples.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
