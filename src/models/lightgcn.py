"""Sparse LightGCN backbone."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def normalized_bipartite_adjacency(num_users: int, num_items: int, edges: Tensor,
                                   device: torch.device) -> Tensor:
    """Return D^-1/2 A D^-1/2 as a coalesced torch COO sparse tensor."""
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape [n_edges, 2]")
    users, items = edges[:, 0].long(), edges[:, 1].long() + num_users
    indices = torch.stack((torch.cat((users, items)), torch.cat((items, users))))
    total = num_users + num_items
    degree = torch.bincount(indices[0], minlength=total).float().clamp_min(1)
    values = (degree[indices[0]] * degree[indices[1]]).rsqrt()
    return torch.sparse_coo_tensor(indices, values, (total, total), device=device).coalesce()


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int):
        super().__init__()
        self.num_users, self.num_items, self.num_layers = num_users, num_items, num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def propagate(self, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        layers = [embeddings]
        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layers.append(embeddings)
        output = torch.stack(layers, dim=0).mean(dim=0)
        return output[:self.num_users], output[self.num_users:]

    @staticmethod
    def score(users: Tensor, items: Tensor, user_embeddings: Tensor, item_embeddings: Tensor) -> Tensor:
        return (user_embeddings[users] * item_embeddings[items]).sum(dim=-1)
