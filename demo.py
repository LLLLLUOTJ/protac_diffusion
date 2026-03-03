from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.train_config import TrainConfig
from data.base import DiffusionSample
from data.image_dataset import build_image_dataset
from diffusion.ddpm import DDPM
from models.simple_unet import SimpleUNet
from sampling.sampler import sample_and_save_images, sample_batch
from trainers.trainer import Trainer
from utils.common import set_seed


def collate_diffusion_samples(batch: list[DiffusionSample]) -> dict:
    x = torch.stack([item.x for item in batch], dim=0)
    cond_items = [item.cond for item in batch]
    cond = None
    if cond_items and all(c is not None for c in cond_items):
        cond = torch.stack([c for c in cond_items if c is not None], dim=0)
    meta = [item.meta for item in batch]
    return {"x": x, "cond": cond, "meta": meta}


def build_dataloader(config: TrainConfig) -> DataLoader:
    if config.modality != "image":
        raise ValueError(f"当前demo仅实现image最小闭环, 收到: {config.modality}")

    dataset = build_image_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_diffusion_samples,
        drop_last=True,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainConfig(device=device)
    set_seed(config.seed)

    dataloader = build_dataloader(config)

    model = SimpleUNet(in_channels=config.channels, base_channels=64, time_dim=128)
    diffusion = DDPM(
        model=model,
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=config.device,
    )

    trainer = Trainer(diffusion=diffusion, dataloader=dataloader, config=config)
    trainer.train()

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": diffusion.model.state_dict(),
            "config": config.__dict__,
        },
        checkpoint_path,
    )
    print(f"[checkpoint] saved to {checkpoint_path}")

    sample_batch(diffusion=diffusion, config=config, out_path="samples.pt")
    sample_and_save_images(diffusion=diffusion, config=config)


if __name__ == "__main__":
    main()
