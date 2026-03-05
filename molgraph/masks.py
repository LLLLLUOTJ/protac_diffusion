from __future__ import annotations

from typing import Union

import torch

# Conservative valence limits for common molecular generation atoms.
MAX_VALENCE_BY_ATOMIC_NUM = {
    0: 4,   # dummy/anchor
    1: 1,   # H
    6: 4,   # C
    7: 3,   # N
    8: 2,   # O
    9: 1,   # F
    15: 5,  # P
    16: 6,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}

DEFAULT_MAX_VALENCE = 4


def _max_valence_tensor(atomic_nums: torch.Tensor) -> torch.Tensor:
    max_vals = torch.full_like(atomic_nums, fill_value=DEFAULT_MAX_VALENCE, dtype=torch.long)
    for atomic_num, max_valence in MAX_VALENCE_BY_ATOMIC_NUM.items():
        max_vals = torch.where(atomic_nums == atomic_num, torch.full_like(max_vals, max_valence), max_vals)
    return max_vals


def allowed_bond_mask(
    current_degree: Union[int, torch.Tensor],
    atomic_num: Union[int, torch.Tensor],
) -> torch.Tensor:
    """Return whether adding one more bond is allowed under simple valence rules.

    This helper is intentionally lightweight for downstream diffusion masking logic.
    It supports scalar integers or tensors and returns a boolean tensor.
    """

    deg = torch.as_tensor(current_degree, dtype=torch.long)
    atomic = torch.as_tensor(atomic_num, dtype=torch.long, device=deg.device)
    deg, atomic = torch.broadcast_tensors(deg, atomic)
    max_vals = _max_valence_tensor(atomic)
    return deg < max_vals
