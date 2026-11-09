"""LightGCN plus FairSkew-GCN training graph-view support."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .graph_augmentation import ItemSkew, item_gender_skew, sample_augmented_edges
from .lightgcn import LightGCN, normalized_bipartite_adjacency


class FairSkewGCN(LightGCN):
    """FairSkew model. Inference defaults to the original, gender-free graph."""
    def __init__(self, *args, train_edges: np.ndarray, group_ids: np.ndarray,
                 smoothing: float = 1.0, rho: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho
        self.register_buffer("train_edges", torch.as_tensor(train_edges, dtype=torch.long))
        self.register_buffer("group_ids", torch.as_tensor(group_ids, dtype=torch.long))
        skew: ItemSkew = item_gender_skew(train_edges, group_ids, self.num_items, smoothing)
        self.register_buffer("item_skew", skew.normalized)
        self.register_buffer("dominant_groups", skew.dominant_group)

    def full_adjacency(self) -> Tensor:
        return normalized_bipartite_adjacency(self.num_users, self.num_items, self.train_edges, self.train_edges.device)

    def augmented_adjacency(self) -> Tensor:
        skew = ItemSkew(self.item_skew, self.dominant_groups)
        view = sample_augmented_edges(self.train_edges, self.group_ids, skew, self.rho)
        return normalized_bipartite_adjacency(self.num_users, self.num_items, view, self.train_edges.device)

    def training_embeddings(self, augmentation: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Original embeddings plus two independently augmented views per epoch."""
        users, items = self.propagate(self.full_adjacency())
        if not augmentation:
            return users, items, users, users
        view1, view2 = self.augmented_adjacency(), self.augmented_adjacency()
        u1, _ = self.propagate(view1)
        u2, _ = self.propagate(view2)
        return users, items, u1, u2
