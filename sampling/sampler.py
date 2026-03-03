from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from configs.train_config import TrainConfig


@torch.no_grad()
def sample_batch(diffusion, config: TrainConfig, out_path: str = "samples.pt") -> None:
    diffusion.eval()
    x = diffusion.sample(
        shape=(config.sample_batch_size, config.channels, config.image_size, config.image_size),
        device=config.device,
    )
    x = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
    torch.save(x.cpu(), out_path)
    print(f"[sample] saved to {out_path}, shape={tuple(x.shape)}")


def save_tensor_images(x: torch.Tensor, out_dir: str, prefix: str = "sample") -> list[str]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    x = torch.clamp(x.detach().cpu(), 0.0, 1.0)
    for index in range(x.shape[0]):
        img = x[index]
        if img.shape[0] == 1:
            array = (img[0].numpy() * 255.0).round().astype("uint8")
            pil = Image.fromarray(array, mode="L")
        else:
            array = (img.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            pil = Image.fromarray(array, mode="RGB")
        path = output_dir / f"{prefix}_{index:03d}.png"
        pil.save(path)
        saved_paths.append(str(path))
    return saved_paths


@torch.no_grad()
def sample_and_save_images(diffusion, config: TrainConfig, out_dir: str | None = None) -> list[str]:
    diffusion.eval()
    x = diffusion.sample(
        shape=(config.sample_batch_size, config.channels, config.image_size, config.image_size),
        device=config.device,
    )
    x = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
    target_dir = out_dir or config.sample_output_dir
    paths = save_tensor_images(x, out_dir=target_dir)
    print(f"[sample] saved {len(paths)} images to {target_dir}")
    return paths
