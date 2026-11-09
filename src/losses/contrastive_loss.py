"""Vectorized user-view InfoNCE loss (FairSkew Module B)."""
from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F


def info_nce(view_one: Tensor, view_two: Tensor, temperature: float) -> Tensor:
    """Same row is positive; every other sampled user is an in-batch negative."""
    if len(view_one) < 2:
        return view_one.new_zeros(())
    if temperature <= 0:
        raise ValueError("contrastive_temperature must be positive")
    a, b = F.normalize(view_one, dim=-1, eps=1e-8), F.normalize(view_two, dim=-1, eps=1e-8)
    return F.cross_entropy(a @ b.T / temperature, torch.arange(len(a), device=a.device))
