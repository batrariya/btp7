"""Differentiable exposure-parity objectives used only during training."""
from __future__ import annotations

import torch
from torch import Tensor


def _js_divergence(left: Tensor, right: Tensor, epsilon: float = 1e-8) -> Tensor:
    left = left.clamp_min(epsilon)
    right = right.clamp_min(epsilon)
    left, right = left / left.sum(), right / right.sum()
    midpoint = (left + right) / 2
    return 0.5 * ((left * (left / midpoint).log()).sum() + (right * (right / midpoint).log()).sum())


def soft_exposure_losses(user_embeddings: Tensor, item_embeddings: Tensor, sampled_users: Tensor,
                         group_ids: Tensor, train_edges: Tensor, num_items: int,
                         temperature: float = 0.1) -> tuple[Tensor, Tensor]:
    """Return soft JS-DP and relevance-conditioned soft JS-EO surrogates.

    DP uses a low-temperature (Top-K-like) score distribution over the complete
    catalogue. EO aggregates calibrated relevance weights on observed training
    positives. Both are direct, differentiable distribution comparisons: unlike
    the original FairSkew modules, they explicitly penalize the quantities that
    produce disparate item exposure. Test/validation interactions are never read.
    """
    if temperature <= 0:
        raise ValueError("exposure_temperature must be positive")
    local_groups = group_ids[sampled_users]
    masks = [(local_groups == group) for group in (0, 1)]
    if not all(bool(mask.any()) for mask in masks):
        zero = user_embeddings.new_zeros(())
        return zero, zero
    scores = user_embeddings[sampled_users] @ item_embeddings.T
    probabilities = torch.softmax(scores / temperature, dim=1)
    # Average user-level score distributions within each group to obtain one
    # comparable per-item exposure distribution per group.
    dp = [_js_distribution(probabilities[mask].mean(dim=0)) for mask in masks]
    dp_loss = _js_divergence(dp[0], dp[1])

    # Map global user IDs in the sample to local score rows without a dense
    # user-by-item relevance matrix.
    locations = torch.full((group_ids.numel(),), -1, device=sampled_users.device, dtype=torch.long)
    locations[sampled_users] = torch.arange(len(sampled_users), device=sampled_users.device)
    edge_locations = locations[train_edges[:, 0]]
    edge_mask = edge_locations >= 0
    eo_counts = []
    for group, user_mask in enumerate(masks):
        local_for_group = user_mask[edge_locations[edge_mask]]
        edge_users = edge_locations[edge_mask][local_for_group]
        edge_items = train_edges[edge_mask][:, 1][local_for_group]
        exposure = probabilities.new_zeros(num_items)
        if len(edge_items):
            # Sigmoid score is a calibrated soft relevance estimate for an
            # observed-positive item. It avoids using held-out ground truth and
            # supplies denser gradients than a full-catalog softmax tail.
            relevance = torch.sigmoid(scores[edge_users, edge_items] / temperature)
            exposure.scatter_add_(0, edge_items, relevance)
        eo_counts.append(_js_distribution(exposure))
    eo_loss = _js_divergence(eo_counts[0], eo_counts[1])
    return dp_loss, eo_loss


def _js_distribution(values: Tensor) -> Tensor:
    return values / values.sum().clamp_min(1e-8)
