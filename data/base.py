from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class DiffusionSample:
    x: Tensor
    cond: Optional[Tensor] = None
    meta: Optional[Dict[str, Any]] = None


class BaseDiffusionDataset(Dataset):
    def __getitem__(self, index: int) -> DiffusionSample:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
