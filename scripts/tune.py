"""Validation-only FairSkew-GCN hyperparameter screen.

Every candidate uses the unchanged FairSkew-GCN architecture.  It never calls
the test evaluator; candidate ranking is NDCG@30 on validation data only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.evaluator import Evaluator
from main import build_data
from models.fairskew_gcn import FairSkewGCN
from training.trainer import Trainer
from utils.config import load_config
from utils.seed import set_seed


# This compact screen varies every approved tuning dimension while remaining
# practical on CPU. The selected config is always retrained from scratch.
TRIALS = [
    {"learning_rate": 0.005, "regularization": 1e-8, "embedding_dim": 64,
     "edge_drop_rho": 0.10, "contrastive_coefficient": 0.005,
     "contrastive_temperature": 0.20, "group_dro_eta": 0.05},
    {"learning_rate": 0.010, "regularization": 1e-8, "embedding_dim": 64,
     "edge_drop_rho": 0.10, "contrastive_coefficient": 0.005,
     "contrastive_temperature": 0.20, "group_dro_eta": 0.05},
    {"learning_rate": 0.005, "regularization": 1e-7, "embedding_dim": 64,
     "edge_drop_rho": 0.25, "contrastive_coefficient": 0.010,
     "contrastive_temperature": 0.15, "group_dro_eta": 0.10},
    {"learning_rate": 0.010, "regularization": 1e-7, "embedding_dim": 32,
     "edge_drop_rho": 0.40, "contrastive_coefficient": 0.020,
     "contrastive_temperature": 0.20, "group_dro_eta": 0.10},
    {"learning_rate": 0.001, "regularization": 1e-6, "embedding_dim": 64,
     "edge_drop_rho": 0.20, "contrastive_coefficient": 0.001,
     "contrastive_temperature": 0.20, "group_dro_eta": 0.05},
    {"learning_rate": 0.002, "regularization": 1e-6, "embedding_dim": 64,
     "edge_drop_rho": 0.40, "contrastive_coefficient": 0.001,
     "contrastive_temperature": 0.20, "group_dro_eta": 0.10},
]


def make_model(data, config, device):
    labels = {"M": 0, "F": 1}
    groups = [labels.get(str(group), -1) for group in data.user_groups]
    return FairSkewGCN(data.num_users, data.num_items, config["embedding_dim"], config["num_layers"],
                        train_edges=data.train, group_ids=groups, smoothing=config["skew_smoothing"],
                        rho=config["edge_drop_rho"]).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/movielens1m.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="screening epochs per candidate")
    parser.add_argument("--trials", default="all", help="comma-separated 1-based trial IDs, or 'all'")
    args = parser.parse_args()
    base = load_config(args.config)
    device = torch.device("cuda" if base["device"] == "auto" and torch.cuda.is_available() else "cpu")
    data = build_data(base)
    summary: list[dict] = []
    selected = range(1, len(TRIALS) + 1) if args.trials == "all" else [int(value) for value in args.trials.split(",")]
    for number in selected:
        if number < 1 or number > len(TRIALS):
            raise ValueError(f"Trial {number} is outside 1..{len(TRIALS)}")
        overrides = TRIALS[number - 1]
        config = {**base, **overrides, "epochs": args.epochs, "final_test_evaluation": False,
                  "experiment_name": f"fairskew-tune-{number:02d}"}
        set_seed(config["seed"])
        model = make_model(data, config, device)
        _, run_dir = Trainer(model, data, config, device).train()
        evaluator = Evaluator(data, config["evaluation_ks"], device, config["fairness_metric_variant"])
        validation = evaluator.evaluate(model, model.full_adjacency(), "validation")
        record = {"trial": number, "parameters": overrides, "run_dir": str(run_dir), "validation": validation,
                  "selection_ndcg_at_30": validation["NDCG"][30], "selection_recall_at_30": validation["Recall"][30]}
        summary.append(record)
        print(f"trial={number} validation NDCG@30={record['selection_ndcg_at_30']:.6f} "
              f"Recall@30={record['selection_recall_at_30']:.6f}")
    summary.sort(key=lambda value: value["selection_ndcg_at_30"], reverse=True)
    destination = Path(base["output_dir"]) / f"tuning-{datetime.now():%Y%m%d-%H%M%S}.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"selected_trial={summary[0]['trial']} summary={destination}")


if __name__ == "__main__":
    main()
