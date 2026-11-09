"""Evaluate a saved checkpoint independently on the ML-1M test split."""
from pathlib import Path
import argparse
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from main import build_data
from models.fairskew_gcn import FairSkewGCN
from training.checkpoint import load_checkpoint
from evaluation.evaluator import Evaluator
from utils.config import load_config
from utils.result_formatter import format_table

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config", default="configs/movielens1m.yaml")
args = parser.parse_args()
config = load_config(args.config)
device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else "cpu")
data = build_data(config)
groups = [{"M": 0, "F": 1}.get(str(x), -1) for x in data.user_groups]
model = FairSkewGCN(data.num_users, data.num_items, config["embedding_dim"], config["num_layers"], train_edges=data.train,
                     group_ids=groups, smoothing=config["skew_smoothing"], rho=config["edge_drop_rho"]).to(device)
state = load_checkpoint(args.checkpoint, model, device=device)
metrics = Evaluator(
    data, config["evaluation_ks"], device,
    config.get("fairness_metric_variant", "divergence"),
    config.get("rerank_strength", 0.0),
    config.get("rerank_pool_size", 100),
    config.get("rerank_common_relevance", False),
    config.get("rerank_eo_proxy_strength", 0.0),
).evaluate(model, model.full_adjacency())
print(format_table("FairSkew-GCN", state["epoch"], metrics, config["evaluation_ks"]))
