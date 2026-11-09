import torch

from losses.bpr_loss import bpr_per_example
from losses.contrastive_loss import info_nce
from losses.group_dro import GroupDRO
from losses.fairness_loss import soft_exposure_losses


def test_losses_are_finite_and_prefer_positive_scores():
    assert bpr_per_example(torch.tensor([2.0]), torch.tensor([0.0])) < bpr_per_example(torch.tensor([0.0]), torch.tensor([2.0]))
    assert torch.isfinite(info_nce(torch.randn(3, 4), torch.randn(3, 4), 0.2))


def test_group_dro_increases_higher_loss_weight():
    dro = GroupDRO(["M", "F"], 1.0, torch.device("cpu"))
    dro.update(torch.tensor([0.1, 1.0]), torch.tensor([True, True]))
    assert dro.weights[1] > dro.weights[0]


def test_soft_exposure_losses_are_finite():
    users, items = torch.randn(4, 3, requires_grad=True), torch.randn(5, 3, requires_grad=True)
    edges = torch.tensor([[0, 0], [0, 1], [1, 1], [2, 2], [3, 3]])
    dp, eo = soft_exposure_losses(users, items, torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 0, 1]), edges, 5)
    (dp + eo).backward()
    assert torch.isfinite(dp) and torch.isfinite(eo) and users.grad is not None
