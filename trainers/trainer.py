from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs.train_config import TrainConfig
from data.base import DiffusionSample


class Trainer:
    def __init__(self, diffusion: nn.Module, dataloader: DataLoader, config: TrainConfig) -> None:
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.config = config
        self.device = torch.device(config.device)
        self.optimizer = Adam(self.diffusion.parameters(), lr=config.lr)

    def train(self) -> None:
        self.diffusion.to(self.device)
        self.diffusion.train()

        step = 0
        while step < self.config.train_steps:
            for batch in self.dataloader:
                if step >= self.config.train_steps:
                    break

                if isinstance(batch, DiffusionSample):
                    x = batch.x
                elif isinstance(batch, dict):
                    x = batch["x"]
                else:
                    x = batch[0]

                x = x.to(self.device)
                t = torch.randint(0, self.config.timesteps, (x.shape[0],), device=self.device)

                loss = self.diffusion.p_losses(x, t)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                step += 1
                if step % self.config.log_every == 0 or step == 1:
                    print(f"[train] step={step} loss={loss.item():.6f}")
