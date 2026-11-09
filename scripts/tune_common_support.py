"""Select a common-support reranking strength on validation data only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.evaluator import Evaluator
from main import build_data
from models.fairskew_gcn import FairSkewGCN
from training.checkpoint import load_checkpoint
from utils.config import load_config
from utils.seed import set_seed


def at(metrics: dict, name: str, k: int) -> float:
    return float(metrics[name][k])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/fairskew_common_support.yaml")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75])
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--min-ndcg-retention", type=float, default=0.85)
    parser.add_argument("--common-relevance", action=argparse.BooleanOptionalAction, default=None,
                        help="use the training-only common-relevance prior (default: config value)")
    parser.add_argument("--eo-proxy-strength", nargs="+", type=float, default=None,
                        help="training-only EO correction values (default: config value)")
    parser.add_argument("--output", default="outputs/common_support_validation_selection.json")
    args = parser.parse_args()
    if not 0 < args.min_ndcg_retention <= 1:
        raise ValueError("--min-ndcg-retention must be in (0, 1]")
    config = load_config(args.config)
    common_relevance = config.get("rerank_common_relevance", False) if args.common_relevance is None else args.common_relevance
    eo_strengths = args.eo_proxy_strength or [config.get("rerank_eo_proxy_strength", 0.0)]
    if any(not 0 <= value <= 1 for value in [*args.strengths, *eo_strengths]):
        raise ValueError("all reranking strengths must be in [0, 1]")
    if args.k not in config["evaluation_ks"]:
        raise ValueError("--k must appear in evaluation_ks")
    set_seed(config["seed"])
    device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else "cpu")
    data = build_data(config)
    group_ids = [{"M": 0, "F": 1}.get(str(value), -1) for value in data.user_groups]
    model = FairSkewGCN(
        data.num_users, data.num_items, config["embedding_dim"], config["num_layers"],
        train_edges=data.train, group_ids=group_ids, smoothing=config["skew_smoothing"],
        rho=config["edge_drop_rho"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    records = []
    for strength in sorted(set(args.strengths)):
        for eo_strength in sorted(set(eo_strengths)):
            metrics = Evaluator(
                data, config["evaluation_ks"], device, config.get("fairness_metric_variant", "divergence"),
                strength, config.get("rerank_pool_size", 100), common_relevance, eo_strength,
            ).evaluate(model, model.full_adjacency(), split="validation")
            records.append({
                "strength": strength,
                "eo_proxy_strength": eo_strength,
                "ndcg": at(metrics, "NDCG", args.k),
                "recall": at(metrics, "Recall", args.k),
                "js_dp": at(metrics, "JS-DP", args.k),
                "js_eo": at(metrics, "JS-EO", args.k),
                "fairness_score": (at(metrics, "JS-DP", args.k) + at(metrics, "JS-EO", args.k)) / 2,
                "metrics": metrics,
            })
    baseline = next(record for record in records if record["strength"] == 0.0 and record["eo_proxy_strength"] == 0.0)
    threshold = baseline["ndcg"] * args.min_ndcg_retention
    feasible = [record for record in records if record["ndcg"] >= threshold]
    # Choose fairness first only among policies that meet the declared utility floor.
    selected = min(feasible or records, key=lambda row: (row["fairness_score"], -row["ndcg"], -row["recall"]))
    payload = {
        "split": "validation",
        "selection_rule": "minimize mean(JS-DP, JS-EO) subject to NDCG retention",
        "k": args.k,
        "baseline_ndcg": baseline["ndcg"],
        "min_ndcg_retention": args.min_ndcg_retention,
        "common_relevance": common_relevance,
        "eo_proxy_strengths": eo_strengths,
        "ndcg_threshold": threshold,
        "candidates": records,
        "selected": selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("split", "selection_rule", "ndcg_threshold", "selected")}, indent=2))


if __name__ == "__main__":
    main()
