from __future__ import annotations

import sys
import time

import torch
from torch import Tensor, nn
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


class DDPM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device_name = device

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        # alphas[t]
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # alphas[t - 1]
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8),
        )

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    # sum of ||epsilon - epsilon_predicted||^2 for all samples in the batch
    def p_losses(self, x_start: Tensor, t: Tensor) -> Tensor:
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_t, t)
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: Tensor, t: Tensor) -> Tensor:
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = torch.rsqrt(self.alphas[t]).view(-1, 1, 1, 1)

        model_mean = sqrt_recip_alpha_t * (x - betas_t * self.model(x, t) / sqrt_one_minus_alpha_bar_t)

        if (t == 0).all():
            return model_mean

        posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(torch.clamp(posterior_var_t, min=1e-20)) * noise

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int, int],
        device: str | None = None,
        show_progress: bool = True,
        log_every: int = 20,
    ) -> Tensor:
        actual_device = device or self.device_name
        x = torch.randn(shape, device=actual_device)
        start = time.perf_counter()
        total = self.timesteps
        every = max(1, log_every)
        iterator = reversed(range(self.timesteps))
        use_tqdm = bool(show_progress and tqdm is not None)
        if use_tqdm:
            iterator = tqdm(
                iterator,
                total=total,
                desc="sample",
                dynamic_ncols=True,
                leave=True,
                disable=not sys.stdout.isatty(),
            )
        for index, step in enumerate(iterator, start=1):
            t = torch.full((shape[0],), step, device=actual_device, dtype=torch.long)
            x = self.p_sample(x, t)
            if show_progress and (index == 1 or index % every == 0 or index == total):
                elapsed = time.perf_counter() - start
                avg = elapsed / index
                eta = avg * max(total - index, 0)
                progress = (index / total) * 100.0
                if use_tqdm:
                    iterator.set_postfix(t=step, eta_s=f"{eta:.1f}")
                else:
                    print(
                        f"[sample] step={index}/{total} ({progress:.1f}%) "
                        f"t={step} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
        if use_tqdm:
            iterator.close()
        return x
