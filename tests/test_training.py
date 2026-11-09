import torch

from models.fairskew_gcn import FairSkewGCN
from training.trainer import Trainer


def test_short_training_creates_checkpoint(tiny_data, tmp_path):
    config = {
        "learning_rate": 0.01, "experiment_name": "test", "seed": 1, "output_dir": str(tmp_path),
        "evaluation_ks": [1, 2, 3], "group_dro_eta": 0.1, "enable_group_dro": True,
        "enable_augmentation": True, "enable_contrastive": True, "contrastive_batch_size": 4,
        "contrastive_temperature": 0.2, "contrastive_coefficient": 0.1, "regularization": 1e-4,
        "epochs": 2, "validation_interval": 1, "checkpoint_interval": 2, "resume": None,
    }
    model = FairSkewGCN(4, 6, 4, 1, train_edges=tiny_data.train, group_ids=[0, 1, 0, 1])
    _, directory = Trainer(model, tiny_data, config, torch.device("cpu")).train()
    assert (directory / "checkpoints" / "final.pt").exists()
    assert (directory / "results" / "results.json").exists()
