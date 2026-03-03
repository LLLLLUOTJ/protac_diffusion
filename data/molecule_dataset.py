from __future__ import annotations

from data.base import BaseDiffusionDataset


class MoleculeDatasetAdapter(BaseDiffusionDataset):
    def __len__(self) -> int:
        raise NotImplementedError("实现分子图或token解析后再返回长度")

    def __getitem__(self, index: int):
        raise NotImplementedError("建议输出统一字段: DiffusionSample(x, cond, meta)")
