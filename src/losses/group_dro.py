"""Stable, detached Group DRO weighting (FairSkew Module C)."""
from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor


class GroupDRO:
    def __init__(self, group_names: Sequence[str], eta: float, device: torch.device):
        if not group_names:
            raise ValueError("Fairness training needs at least one known sensitive group.")
        self.group_names, self.eta = list(group_names), eta
        self.log_weights = torch.full((len(group_names),), -torch.log(torch.tensor(float(len(group_names)))), device=device)

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.log_weights, dim=0)

    @torch.no_grad()
    def update(self, losses: Tensor, active: Tensor) -> None:
        """Apply w_g <- normalize(w_g exp(eta L_g)) for observed groups only."""
        if active.any():
            self.log_weights[active] += self.eta * losses.detach()[active]
            self.log_weights -= torch.logsumexp(self.log_weights, dim=0)

    def state_dict(self) -> dict:
        return {"group_names": self.group_names, "eta": self.eta, "log_weights": self.log_weights.detach().cpu()}

    def load_state_dict(self, state: dict) -> None:
        if state["group_names"] != self.group_names:
            raise ValueError("Checkpoint group names do not match dataset.")
        self.log_weights = state["log_weights"].to(self.log_weights.device)
