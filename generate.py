from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs.train_config import TrainConfig
from diffusion.ddpm import DDPM
from models.simple_unet import SimpleUNet
from sampling.sampler import sample_and_save_images
from utils.common import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load trained checkpoint and generate images.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/ddpm.pt", help="checkpoint path")
    parser.add_argument("--num", type=int, default=8, help="number of images to generate")
    parser.add_argument("--out", type=str, default="outputs/samples", help="output image directory")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device in {"cpu", "cuda"}:
        device = args.device

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    config_dict = checkpoint.get("config", {})
    config = TrainConfig(**{k: v for k, v in config_dict.items() if k in TrainConfig.__dataclass_fields__})
    config.device = device
    config.sample_batch_size = args.num
    config.sample_output_dir = args.out

    set_seed(config.seed)

    model = SimpleUNet(in_channels=config.channels, base_channels=64, time_dim=128)
    model.load_state_dict(checkpoint["model_state_dict"])

    diffusion = DDPM(
        model=model,
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=config.device,
    ).to(device)

    saved = sample_and_save_images(diffusion=diffusion, config=config, out_dir=config.sample_output_dir)
    print("[done] generated files:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
