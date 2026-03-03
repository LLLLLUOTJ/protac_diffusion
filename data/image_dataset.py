from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import Tensor

from configs.train_config import TrainConfig
from data.base import BaseDiffusionDataset, DiffusionSample


class SyntheticImageDataset(BaseDiffusionDataset):
    def __init__(
        self,
        size: int = 1024,
        image_size: int = 32,
        channels: int = 3,
        seed: int = 42,
    ) -> None:
        self.size = size
        self.image_size = image_size
        self.channels = channels
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> DiffusionSample:
        generator = torch.Generator().manual_seed(self.seed + index)
        x: Tensor = torch.rand(
            self.channels,
            self.image_size,
            self.image_size,
            generator=generator,
            dtype=torch.float32,
        )
        x = x * 2.0 - 1.0
        label = torch.tensor(index % 10, dtype=torch.long)
        return DiffusionSample(x=x, cond=label, meta={"index": index})


def _pil_to_normalized_tensor(image: Image.Image) -> Tensor:
    image = image.convert("RGB")
    byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    x = byte_tensor.float().view(image.height, image.width, 3).permute(2, 0, 1) / 255.0
    return x * 2.0 - 1.0


class FolderImageDataset(BaseDiffusionDataset):
    def __init__(self, folder: str, image_size: int = 32, channels: int = 3) -> None:
        self.folder = Path(folder)
        self.image_size = image_size
        self.channels = channels
        self.paths = self._collect_paths(self.folder)
        if len(self.paths) == 0:
            raise ValueError(f"图片目录为空或无有效图片: {self.folder}")

    @staticmethod
    def _collect_paths(folder: Path) -> list[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if not folder.exists():
            raise ValueError(f"图片目录不存在: {folder}")
        return [path for path in folder.rglob("*") if path.suffix.lower() in extensions]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> DiffusionSample:
        path = self.paths[index]
        with Image.open(path) as img:
            img = img.resize((self.image_size, self.image_size), resample=Image.Resampling.BILINEAR)
            x = _pil_to_normalized_tensor(img)
        if self.channels == 1:
            x = x[:1]
        elif self.channels == 3:
            pass
        else:
            raise ValueError(f"当前仅支持channels=1或3, 收到: {self.channels}")
        return DiffusionSample(x=x, cond=None, meta={"path": str(path)})


class CIFAR10DatasetAdapter(BaseDiffusionDataset):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        image_size: int,
        channels: int,
    ) -> None:
        if channels != 3:
            raise ValueError("CIFAR10为RGB数据, 当前仅支持channels=3")
        try:
            from torchvision import datasets, transforms
        except Exception as exc:
            raise ImportError("使用cifar10需要安装torchvision: pip install torchvision") from exc

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )
        self.dataset = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DiffusionSample:
        x, label = self.dataset[index]
        return DiffusionSample(x=x, cond=torch.tensor(label, dtype=torch.long), meta={"index": index})


def build_image_dataset(config: TrainConfig) -> BaseDiffusionDataset:
    builders: dict[str, Callable[[TrainConfig], BaseDiffusionDataset]] = {
        "synthetic": lambda cfg: SyntheticImageDataset(
            size=cfg.synthetic_dataset_size,
            image_size=cfg.image_size,
            channels=cfg.channels,
            seed=cfg.seed,
        ),
        "folder": lambda cfg: FolderImageDataset(
            folder=cfg.image_folder,
            image_size=cfg.image_size,
            channels=cfg.channels,
        ),
        "cifar10": lambda cfg: CIFAR10DatasetAdapter(
            root=cfg.cifar10_root,
            train=cfg.cifar10_train,
            download=cfg.cifar10_download,
            image_size=cfg.image_size,
            channels=cfg.channels,
        ),
    }
    if config.image_source not in builders:
        raise ValueError(f"未知image_source: {config.image_source}, 可选: {list(builders.keys())}")
    return builders[config.image_source](config)
