"""FairGo-inspired sensitive-attribute discriminator."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class EgoSensitiveDiscriminator(nn.Module):
    """Predicts a sensitive group from node and ego-graph representations.

    The discriminator follows FairGo's observation that a user's sensitive
    attribute can leak through both the user embedding and the first-order
    neighborhood. The recommender is optimized to make this task difficult,
    while the discriminator is separately optimized to remain strong.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or embedding_dim
        self.network = nn.Sequential(nn.Linear(embedding_dim * 2, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 2))

    def forward(self, user_embeddings: Tensor, ego_embeddings: Tensor) -> Tensor:
        return self.network(torch.cat((user_embeddings, ego_embeddings), dim=-1))


def ego_item_summary(user_embeddings: Tensor, item_embeddings: Tensor, train_edges: Tensor) -> Tensor:
    """Mean first-order item-neighborhood embedding for every user."""
    summary = torch.zeros_like(user_embeddings)
    summary.index_add_(0, train_edges[:, 0], item_embeddings[train_edges[:, 1]])
    degree = torch.bincount(train_edges[:, 0], minlength=len(user_embeddings)).to(user_embeddings.dtype).clamp_min(1)
    return summary / degree.unsqueeze(-1)
