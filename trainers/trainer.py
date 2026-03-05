from __future__ import annotations

import sys
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from configs.train_config import TrainConfig
from data.base import DiffusionSample


class Trainer:
    def __init__(self, diffusion: nn.Module, dataloader: DataLoader, config: TrainConfig) -> None:
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.config = config
        self.device = torch.device(config.device)
        self.optimizer = Adam(self.diffusion.parameters(), lr=config.lr)
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train(self) -> None:
        self.diffusion.to(self.device)
        self.diffusion.train()

        step = 0
        total_steps = self.config.train_steps
        train_start = time.perf_counter()
        pbar = None
        if self.config.progress_bar and tqdm is not None:
            pbar = tqdm(
                total=total_steps,
                desc="train",
                dynamic_ncols=True,
                leave=True,
                disable=not sys.stdout.isatty(),
            )
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

                step_start = time.perf_counter()
                x = x.to(self.device, non_blocking=self.device.type == "cuda")
                t = torch.randint(0, self.config.timesteps, (x.shape[0],), device=self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self.diffusion.p_losses(x, t)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                step += 1
                if pbar is not None:
                    pbar.update(1)
                if step % self.config.log_every == 0 or step == 1 or step == total_steps:
                    elapsed = time.perf_counter() - train_start
                    avg_step = elapsed / step
                    eta = avg_step * max(total_steps - step, 0)
                    step_time = time.perf_counter() - step_start
                    progress = (step / total_steps) * 100.0
                    if pbar is not None:
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            step_s=f"{step_time:.3f}",
                            eta_s=f"{eta:.1f}",
                        )
                    print(
                        f"[train] step={step}/{total_steps} ({progress:.1f}%) "
                        f"loss={loss.item():.6f} step_time={step_time:.3f}s "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
        if pbar is not None:
            pbar.close()
