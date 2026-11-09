"""Bayesian Personalized Ranking loss."""
from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F


def bpr_per_example(positive_scores: Tensor, negative_scores: Tensor) -> Tensor:
    """-log sigmoid(y_ui-y_uj), retaining one loss per interaction."""
    return F.softplus(negative_scores - positive_scores)
