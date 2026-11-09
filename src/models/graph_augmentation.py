"""Training-only gender-conditioned graph augmentation (FairSkew Module A)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class ItemSkew:
    normalized: Tensor
    dominant_group: Tensor


def item_gender_skew(edges: np.ndarray, group_ids: np.ndarray, num_items: int,
                     smoothing: float = 1.0) -> ItemSkew:
    """Compute the proposed smoothed logit distance and dominant M/F group.

    Group ids are 0=M, 1=F, -1=unknown. Unknown interactions do not influence
    gender skew, but their graph edges are retained (drop probability zero).
    """
    male = np.zeros(num_items, dtype=np.float64)
    female = np.zeros(num_items, dtype=np.float64)
    for user, item in edges:
        group = group_ids[int(user)]
        if group == 0:
            male[int(item)] += 1
        elif group == 1:
            female[int(item)] += 1
    total = male.sum() + female.sum()
    overall = male.sum() / total if total else 0.5
    eps = 1e-7
    p = (male + smoothing) / (male + female + 2 * smoothing)
    logit = lambda x: np.log(np.clip(x, eps, 1 - eps) / (1 - np.clip(x, eps, 1 - eps)))
    skew = np.abs(logit(p) - logit(overall))
    maximum = skew.max(initial=0.0)
    normalized = skew / maximum if maximum > eps else np.zeros_like(skew)
    # Ties and zero-observation items default to male, with zero skew/drop impact.
    dominant = np.where(male >= female, 0, 1).astype(np.int64)
    return ItemSkew(torch.tensor(normalized, dtype=torch.float32), torch.tensor(dominant))


def sample_augmented_edges(edges: Tensor, group_ids: Tensor, skew: ItemSkew, rho: float,
                           generator: torch.Generator | None = None) -> Tensor:
    """Independently sample one graph view; recommendation positives remain intact."""
    if not 0 <= rho <= 1:
        raise ValueError("edge_drop_rho must be in [0, 1]")
    users, items = edges[:, 0], edges[:, 1]
    groups = group_ids[users]
    probs = rho * skew.normalized.to(edges.device)[items]
    probs = probs * (groups == skew.dominant_group.to(edges.device)[items]).float()
    keep = torch.rand(len(edges), device=edges.device, generator=generator) >= probs
    # Degenerate views make sparse propagation invalid; retain one edge deterministically.
    if not bool(keep.any()):
        keep[0] = True
    return edges[keep]
