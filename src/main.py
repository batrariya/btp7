"""Command-line entry point."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from datasets.movielens1m import MovieLens1MDataset
from models.fairskew_gcn import FairSkewGCN
from training.trainer import Trainer
from utils.config import load_config
from utils.seed import set_seed


def build_data(config: dict):
    if config.get("dataset") != "movielens1m":
        raise ValueError("Only movielens1m adapter is bundled; implement another adapter using RecommendationData.")
    return MovieLens1MDataset(config["data_dir"], config["rating_threshold"], tuple(config["split"]), config["seed"], config.get("dataset_url")).load()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/movielens1m.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume: config["resume"] = args.resume
    if args.epochs: config["epochs"] = args.epochs
    set_seed(config["seed"])
    device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else config["device"] if config["device"] != "auto" else "cpu")
    data = build_data(config)
    ids = {"M": 0, "F": 1}
    groups = [ids.get(str(group), -1) for group in data.user_groups]
    model = FairSkewGCN(data.num_users, data.num_items, config["embedding_dim"], config["num_layers"], train_edges=data.train,
                         group_ids=groups, smoothing=config["skew_smoothing"], rho=config["edge_drop_rho"]).to(device)
    Trainer(model, data, config, device).train()


if __name__ == "__main__":
    main()
