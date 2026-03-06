from __future__ import annotations

import torch
from torch import nn

from diffusion.ddpm import DDPM


class ZeroNoiseModel(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_q_sample_supports_rank3_and_fixed_mask() -> None:
    ddpm = DDPM(model=ZeroNoiseModel(), timesteps=8)
    x_start = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    t = torch.tensor([1, 3], dtype=torch.long)
    fixed_mask = torch.tensor(
        [
            [True, False, False, False],
            [False, False, True, False],
        ],
        dtype=torch.bool,
    )
    fixed_values = torch.full_like(x_start, fill_value=-5.0)

    x_t = ddpm.q_sample(
        x_start=x_start,
        t=t,
        noise=torch.zeros_like(x_start),
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )

    assert x_t.shape == x_start.shape
    assert torch.all(x_t[0, 0] == -5.0)
    assert torch.all(x_t[1, 2] == -5.0)
    assert not torch.all(x_t[0, 1] == -5.0)


def test_sample_mask_zeroes_invalid_rank4_slots() -> None:
    ddpm = DDPM(model=ZeroNoiseModel(), timesteps=4)
    initial_noise = torch.ones((1, 3, 3, 2), dtype=torch.float32)
    pair_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ],
        dtype=torch.bool,
    )

    out = ddpm.sample(
        shape=(1, 3, 3, 2),
        initial_noise=initial_noise,
        sample_mask=pair_mask,
        show_progress=False,
    )

    assert out.shape == initial_noise.shape
    assert torch.all(out[:, 0, 1] == 0)
    assert torch.all(out[:, 1, 0] == 0)
    assert torch.all(out[:, 1, 2] == 0)


def test_fixed_mask_freezes_values_during_sampling() -> None:
    ddpm = DDPM(model=ZeroNoiseModel(), timesteps=4)
    fixed_mask = torch.tensor([[True, False, False, True]], dtype=torch.bool)
    fixed_values = torch.tensor(
        [
            [[9.0, 9.0], [0.0, 0.0], [0.0, 0.0], [7.0, 7.0]],
        ]
    )

    out = ddpm.sample(
        shape=(1, 4, 2),
        initial_noise=torch.zeros((1, 4, 2)),
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        show_progress=False,
    )

    assert torch.all(out[:, 0] == fixed_values[:, 0])
    assert torch.all(out[:, 3] == fixed_values[:, 3])


def test_p_sample_skips_noise_only_for_zero_t_samples() -> None:
    ddpm = DDPM(model=ZeroNoiseModel(), timesteps=4)
    x = torch.zeros((2, 3, 2), dtype=torch.float32)
    t = torch.tensor([0, 1], dtype=torch.long)

    torch.manual_seed(0)
    out = ddpm.p_sample(x, t)

    assert torch.allclose(out[0], torch.zeros_like(out[0]))
    assert not torch.allclose(out[1], torch.zeros_like(out[1]))
