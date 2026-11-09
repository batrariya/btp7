import torch

from models.fairskew_gcn import FairSkewGCN


def test_propagation_and_augmentation(tiny_data):
    model = FairSkewGCN(4, 6, 8, 2, train_edges=tiny_data.train, group_ids=[0, 1, 0, 1])
    users, items, first, second = model.training_embeddings(True)
    assert users.shape == (4, 8) and items.shape == (6, 8)
    assert first.shape == second.shape == (4, 8)
    assert torch.isfinite(users).all()
