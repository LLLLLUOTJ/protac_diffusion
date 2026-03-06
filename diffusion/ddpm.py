from __future__ import annotations

import sys
import time
from typing import Callable, Optional

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

    def _extract(self, values: Tensor, t: Tensor, reference: Tensor) -> Tensor:
        coeff = values[t]
        view_shape = (coeff.shape[0],) + (1,) * (reference.ndim - 1)
        return coeff.view(view_shape)

    def _broadcast_like(self, value: Tensor | float | bool, reference: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        tensor = torch.as_tensor(value, device=reference.device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)

        if tensor.shape == reference.shape:
            return tensor
        if tensor.shape == reference.shape[:-1]:
            return tensor.unsqueeze(-1)
        if tensor.shape == reference.shape[1:]:
            return tensor.unsqueeze(0)
        if reference.ndim >= 2 and tensor.shape == reference.shape[1:-1]:
            return tensor.unsqueeze(0).unsqueeze(-1)

        while tensor.ndim < reference.ndim:
            tensor = tensor.unsqueeze(-1)
        try:
            return torch.broadcast_to(tensor, reference.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Constraint shape {tuple(torch.as_tensor(value).shape)} is not broadcastable to {tuple(reference.shape)}"
            ) from exc

    def _apply_sample_mask(self, x: Tensor, sample_mask: Tensor | None = None) -> Tensor:
        if sample_mask is None:
            return x
        mask = self._broadcast_like(sample_mask, x, dtype=x.dtype)
        return x * mask

    def _apply_fixed_mask(
        self,
        x: Tensor,
        fixed_mask: Tensor | None = None,
        fixed_values: Tensor | None = None,
    ) -> Tensor:
        if fixed_mask is None:
            return x
        mask = self._broadcast_like(fixed_mask, x, dtype=torch.bool)
        if fixed_values is None:
            values = torch.zeros_like(x)
        else:
            values = self._broadcast_like(fixed_values, x, dtype=x.dtype)
        return torch.where(mask, values, x)

    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Tensor | None = None,
        sample_mask: Tensor | None = None,
        fixed_mask: Tensor | None = None,
        fixed_values: Tensor | None = None,
    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        noise = self._apply_sample_mask(noise, sample_mask=sample_mask)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_start)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start)
        x_t = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
        x_t = self._apply_sample_mask(x_t, sample_mask=sample_mask)
        fixed_base = x_start if fixed_values is None else fixed_values
        return self._apply_fixed_mask(x_t, fixed_mask=fixed_mask, fixed_values=fixed_base)

    # sum of ||epsilon - epsilon_predicted||^2 for all samples in the batch
    def p_losses(
        self,
        x_start: Tensor,
        t: Tensor,
        sample_mask: Tensor | None = None,
        fixed_mask: Tensor | None = None,
        fixed_values: Tensor | None = None,
        loss_mask: Tensor | None = None,
        model_kwargs: Optional[dict] = None,
    ) -> Tensor:
        noise = torch.randn_like(x_start)
        noise = self._apply_sample_mask(noise, sample_mask=sample_mask)
        x_t = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise,
            sample_mask=sample_mask,
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
        )
        if model_kwargs is None:
            predicted_noise = self.model(x_t, t)
        else:
            predicted_noise = self.model(x_t, t, **model_kwargs)
        predicted_noise = self._apply_sample_mask(predicted_noise, sample_mask=sample_mask)

        mse = torch.nn.functional.mse_loss(predicted_noise, noise, reduction="none")
        effective_mask = torch.ones_like(mse, dtype=mse.dtype)
        if sample_mask is not None:
            effective_mask = effective_mask * self._broadcast_like(sample_mask, mse, dtype=mse.dtype)
        if fixed_mask is not None:
            fixed_bool = self._broadcast_like(fixed_mask, mse, dtype=torch.bool)
            effective_mask = effective_mask * (~fixed_bool).to(dtype=mse.dtype)
        if loss_mask is not None:
            effective_mask = effective_mask * self._broadcast_like(loss_mask, mse, dtype=mse.dtype)

        denom = effective_mask.sum().clamp(min=1.0)
        return (mse * effective_mask).sum() / denom

    @torch.no_grad()
    def p_sample(
        self,
        x: Tensor,
        t: Tensor,
        sample_mask: Tensor | None = None,
        fixed_mask: Tensor | None = None,
        fixed_values: Tensor | None = None,
        post_step_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        model_kwargs: Optional[dict] = None,
    ) -> Tensor:
        betas_t = self._extract(self.betas, t, x)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x)
        sqrt_recip_alpha_t = torch.rsqrt(self._extract(self.alphas, t, x))

        if model_kwargs is None:
            predicted_noise = self.model(x, t)
        else:
            predicted_noise = self.model(x, t, **model_kwargs)
        predicted_noise = self._apply_sample_mask(predicted_noise, sample_mask=sample_mask)
        model_mean = sqrt_recip_alpha_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_bar_t)

        posterior_var_t = self._extract(self.posterior_variance, t, x)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).to(dtype=x.dtype).view((t.shape[0],) + (1,) * (x.ndim - 1))
        x_prev = model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_var_t, min=1e-20)) * noise
        x_prev = self._apply_sample_mask(x_prev, sample_mask=sample_mask)
        x_prev = self._apply_fixed_mask(x_prev, fixed_mask=fixed_mask, fixed_values=fixed_values)
        if post_step_fn is not None:
            x_prev = post_step_fn(x_prev, t)
            x_prev = self._apply_sample_mask(x_prev, sample_mask=sample_mask)
            x_prev = self._apply_fixed_mask(x_prev, fixed_mask=fixed_mask, fixed_values=fixed_values)
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        device: str | None = None,
        show_progress: bool = True,
        log_every: int = 20,
        initial_noise: Tensor | None = None,
        sample_mask: Tensor | None = None,
        fixed_mask: Tensor | None = None,
        fixed_values: Tensor | None = None,
        post_step_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        model_kwargs: Optional[dict] = None,
    ) -> Tensor:
        actual_device = device or self.device_name
        if initial_noise is None:
            x = torch.randn(shape, device=actual_device)
        else:
            x = initial_noise.to(actual_device)
            if tuple(x.shape) != tuple(shape):
                raise ValueError(f"initial_noise shape {tuple(x.shape)} does not match requested shape {tuple(shape)}")
        x = self._apply_sample_mask(x, sample_mask=sample_mask)
        x = self._apply_fixed_mask(x, fixed_mask=fixed_mask, fixed_values=fixed_values)
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
            x = self.p_sample(
                x,
                t,
                sample_mask=sample_mask,
                fixed_mask=fixed_mask,
                fixed_values=fixed_values,
                post_step_fn=post_step_fn,
                model_kwargs=model_kwargs,
            )
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
