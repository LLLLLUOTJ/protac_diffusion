from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from rdkit import Chem

from data.anchored_tensor_dataset import WeakAnchorTensorPTDataset
from data.weak_anchor_diffusion import collate_weak_anchor_diffusion_batch
from diffusion.ddpm import DDPM
from models.fragment_conditioned_denoiser import (
    FragmentConditionedEdgeDenoiser,
    FragmentConditionedNodeDenoiser,
)
from sampling.linker_generation import (
    assemble_full_molecule,
    batch_to_model_kwargs,
    decode_generated_linker_batch,
    draw_molecule,
    fixed_edge_template_from_graph,
    project_edge_features,
    project_node_features,
    update_linker_graph_from_dense_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample anchored PROTAC linkers conditioned on left/right fragments")
    parser.add_argument("--tensor-pt", type=str, default="data/processed/weak_anchor_tensors.pt")
    parser.add_argument("--node-ckpt", type=str, default="checkpoints/linker_node_diffusion_smoke.pt")
    parser.add_argument("--edge-ckpt", type=str, default="checkpoints/linker_edge_diffusion_smoke.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/linker_sampling")
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


def load_node_diffusion(ckpt_path: str, device: torch.device) -> DDPM:
    payload = torch.load(ckpt_path, map_location="cpu")
    model = FragmentConditionedNodeDenoiser(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    diffusion = DDPM(model=model, device=str(device), **payload["diffusion_config"]).to(device)
    diffusion.eval()
    return diffusion


def load_edge_diffusion(ckpt_path: str, device: torch.device) -> DDPM:
    payload = torch.load(ckpt_path, map_location="cpu")
    model = FragmentConditionedEdgeDenoiser(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    diffusion = DDPM(model=model, device=str(device), **payload["diffusion_config"]).to(device)
    diffusion.eval()
    return diffusion


def select_source_sample(dataset: WeakAnchorTensorPTDataset, sample_index: int, sample_id: str | None) -> Dict[str, Any]:
    if sample_id is not None:
        for record in dataset:
            if str(record["sample_id"]) == str(sample_id):
                return record
        raise KeyError(f"sample_id not found: {sample_id}")
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index out of range: {sample_index}")
    return dataset[sample_index]


def build_sampling_batch(sample: Dict[str, Any], num_samples: int) -> Dict[str, Any]:
    return collate_weak_anchor_diffusion_batch([sample for _ in range(num_samples)])


def sample_nodes(
    diffusion: DDPM,
    batch: Dict[str, Any],
    device: torch.device,
    *,
    show_progress: bool,
    log_every: int,
) -> torch.Tensor:
    linker_node = move_to_device(batch["linker_node"], device)
    model_kwargs = batch_to_model_kwargs(batch, device)

    def node_post_step(x: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        return project_node_features(
            x,
            fixed_mask=linker_node["fixed_mask"],
            fixed_values=linker_node["fixed_values"],
        )

    node_x = diffusion.sample(
        shape=tuple(linker_node["x_start"].shape),
        device=str(device),
        show_progress=show_progress,
        log_every=log_every,
        sample_mask=linker_node["sample_mask"],
        fixed_mask=linker_node["fixed_mask"],
        fixed_values=linker_node["fixed_values"],
        post_step_fn=node_post_step,
        model_kwargs=model_kwargs,
    )
    return project_node_features(
        node_x,
        fixed_mask=linker_node["fixed_mask"],
        fixed_values=linker_node["fixed_values"],
    )


def sample_edges(
    diffusion: DDPM,
    batch: Dict[str, Any],
    sampled_node_x: torch.Tensor,
    device: torch.device,
    *,
    edge_threshold: float,
    show_progress: bool,
    log_every: int,
) -> torch.Tensor:
    linker_edge = move_to_device(batch["linker_edge"], device)
    model_kwargs = batch_to_model_kwargs(batch, device)
    model_kwargs["linker_graph"] = fixed_edge_template_from_graph(
        model_kwargs["linker_graph"],
        node_x=sampled_node_x.squeeze(0),
    )

    def edge_post_step(x: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        projected = project_edge_features(
            x,
            fixed_mask=linker_edge["fixed_mask"],
            fixed_values=linker_edge["fixed_values"],
        )
        model_kwargs["linker_graph"] = update_linker_graph_from_dense_edges(
            linker_graph=model_kwargs["linker_graph"],
            edge_tensor=projected,
            node_x=sampled_node_x.squeeze(0),
            score_threshold=edge_threshold,
        )
        return projected

    bootstrap = project_edge_features(
        linker_edge["fixed_values"],
        fixed_mask=linker_edge["fixed_mask"],
        fixed_values=linker_edge["fixed_values"],
    )
    model_kwargs["linker_graph"] = update_linker_graph_from_dense_edges(
        linker_graph=model_kwargs["linker_graph"],
        edge_tensor=bootstrap,
        node_x=sampled_node_x.squeeze(0),
        score_threshold=edge_threshold,
    )

    edge_x = diffusion.sample(
        shape=tuple(linker_edge["x_start"].shape),
        device=str(device),
        show_progress=show_progress,
        log_every=log_every,
        sample_mask=linker_edge["sample_mask"],
        fixed_mask=linker_edge["fixed_mask"],
        fixed_values=linker_edge["fixed_values"],
        post_step_fn=edge_post_step,
        model_kwargs=model_kwargs,
    )
    return project_edge_features(
        edge_x,
        fixed_mask=linker_edge["fixed_mask"],
        fixed_values=linker_edge["fixed_values"],
    )


def decode_results(
    batch: Dict[str, Any],
    sampled_node_x: torch.Tensor,
    sampled_edge_x: torch.Tensor,
    *,
    edge_threshold: float,
    out_dir: Path,
    save_images: bool,
) -> List[Dict[str, Any]]:
    linker_graph = batch["linker_graph"]
    results = decode_generated_linker_batch(
        node_x=sampled_node_x.cpu(),
        edge_tensor=sampled_edge_x.cpu(),
        linker_graph=move_to_device(linker_graph, torch.device("cpu")),
        score_threshold=edge_threshold,
    )

    rows: List[Dict[str, Any]] = []
    for idx, decoded in enumerate(results):
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

        anchored_smiles = decoded["anchored_linker_smiles"]
        full_smiles = Chem.MolToSmiles(full_mol, canonical=True) if full_mol is not None else None
        row = {
            "repeat_index": idx,
            "sample_id": batch["sample_id"][idx],
            "protac_id": batch["protac_id"][idx],
            "linker_id": batch["linker_id"][idx],
            "source_anchored_linker_smiles": batch["anchored_linker_smiles"][idx],
            "source_left_fragment_smiles": left_smiles,
            "source_right_fragment_smiles": right_smiles,
            "generated_anchored_linker_smiles": anchored_smiles,
            "generated_full_smiles": full_smiles,
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

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    dataset = WeakAnchorTensorPTDataset(pt_path=args.tensor_pt)
    source_sample = select_source_sample(dataset, sample_index=args.sample_index, sample_id=args.sample_id)
    batch = build_sampling_batch(source_sample, num_samples=args.num_samples)

    print(
        f"[sample] device={device} source_sample_id={source_sample['sample_id']} "
        f"num_samples={args.num_samples}",
        flush=True,
    )
    print(
        f"[source] anchored={source_sample['anchored_linker_smiles']} "
        f"left={source_sample['left_fragment_smiles']} "
        f"right={source_sample['right_fragment_smiles']}",
        flush=True,
    )

    node_diffusion = load_node_diffusion(args.node_ckpt, device=device)
    edge_diffusion = load_edge_diffusion(args.edge_ckpt, device=device)

    sampled_node_x = sample_nodes(
        node_diffusion,
        batch=batch,
        device=device,
        show_progress=args.show_progress,
        log_every=args.log_every,
    )
    sampled_edge_x = sample_edges(
        edge_diffusion,
        batch=batch,
        sampled_node_x=sampled_node_x,
        device=device,
        edge_threshold=args.edge_threshold,
        show_progress=args.show_progress,
        log_every=args.log_every,
    )

    out_dir = Path(args.out_dir)
    rows = decode_results(
        batch=batch,
        sampled_node_x=sampled_node_x,
        sampled_edge_x=sampled_edge_x,
        edge_threshold=args.edge_threshold,
        out_dir=out_dir,
        save_images=args.save_images,
    )
    if not rows:
        raise RuntimeError("No decoded samples were produced")
    write_outputs(rows, out_dir=out_dir)

    decoded = sum(row["generated_anchored_linker_smiles"] is not None for row in rows)
    assembled = sum(row["generated_full_smiles"] is not None for row in rows)
    print(
        f"[done] decoded={decoded}/{len(rows)} assembled={assembled}/{len(rows)} "
        f"csv={out_dir / 'generated_samples.csv'}",
        flush=True,
    )
    for row in rows[: min(3, len(rows))]:
        print(
            f"[result] idx={row['repeat_index']} "
            f"anchored={row['generated_anchored_linker_smiles']} "
            f"full={row['generated_full_smiles']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
