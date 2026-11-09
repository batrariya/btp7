"""Leakage-safe full-catalog ranking evaluation."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import torch

from datasets.base_dataset import RecommendationData
from models.lightgcn import LightGCN
from .fairness_metrics import js_dp, js_eo
from .ranking_metrics import ndcg_at_k, recall_at_k
from .reranker import exposure_parity_rerank


class Evaluator:
    def __init__(self, data: RecommendationData, ks: list[int], device: torch.device,
                 fairness_metric_variant: str = "divergence", rerank_strength: float = 0.0,
                 rerank_pool_size: int = 100, rerank_common_relevance: bool = False,
                 rerank_eo_proxy_strength: float = 0.0):
        self.data, self.ks, self.device = data, sorted(ks), device
        self.fairness_metric_variant = fairness_metric_variant
        self.rerank_strength, self.rerank_pool_size = rerank_strength, rerank_pool_size
        self.rerank_common_relevance = rerank_common_relevance
        self.rerank_eo_proxy_strength = rerank_eo_proxy_strength

    def _common_training_relevance(self) -> np.ndarray:
        """Geometric-mean M/F training-positive item prior for slate targeting."""
        counts = np.zeros((2, self.data.num_items), dtype=np.float64)
        for user, item in self.data.train:
            label = str(self.data.user_groups[int(user)])
            if label == "M":
                counts[0, int(item)] += 1
            elif label == "F":
                counts[1, int(item)] += 1
        distributions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        return np.sqrt(distributions[0] * distributions[1])

    def _group_training_relevance(self) -> np.ndarray:
        """Smoothed p(item is relevant | group), estimated from train only."""
        counts = np.ones((2, self.data.num_items), dtype=np.float64)
        for user, item in self.data.train:
            label = str(self.data.user_groups[int(user)])
            if label == "M":
                counts[0, int(item)] += 1
            elif label == "F":
                counts[1, int(item)] += 1
        return counts / counts.sum(axis=1, keepdims=True)

    @torch.no_grad()
    def evaluate(self, model: LightGCN, adjacency: torch.Tensor, split: str = "test") -> dict:
        """Score all catalogue items after removing known training (and validation for test) items."""
        relevant = self.data.validation if split == "validation" else self.data.test
        seen = self.data.train_items_by_user()
        if split == "test":
            for user, items in self.data.validation.items():
                seen[user].update(items)
        user_embeddings, item_embeddings = model.propagate(adjacency)
        max_k = min(max(self.ks), self.data.num_items)
        results: dict[int, list[int]] = {}
        candidate_scores: dict[int, list[float]] = {}
        candidate_count = min(max(max_k, self.rerank_pool_size), self.data.num_items)
        for user in relevant:
            scores = (user_embeddings[user] @ item_embeddings.T).clone()
            if seen[user]:
                scores[torch.tensor(list(seen[user]), device=self.device)] = -torch.inf
            ranked_scores, ranked_ids = torch.topk(scores, candidate_count)
            results[user] = ranked_ids.cpu().tolist()
            candidate_scores[user] = ranked_scores.cpu().tolist()
        if self.rerank_strength > 0:
            prior = self._common_training_relevance() if self.rerank_common_relevance else None
            conditional = self._group_training_relevance() if self.rerank_eo_proxy_strength else None
            results = exposure_parity_rerank(
                results, candidate_scores, self.data.user_groups, max_k, self.rerank_strength,
                prior, self.data.num_items, conditional, self.rerank_eo_proxy_strength,
            )
        values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for user, ranked in results.items():
            group = str(self.data.user_groups[user])
            rel = set(relevant[user])
            for k in self.ks:
                values["NDCG"][k].append(ndcg_at_k(ranked, rel, k))
                values["Recall"][k].append(recall_at_k(ranked, rel, k))
                if group == "M":
                    values["Male NDCG"][k].append(ndcg_at_k(ranked, rel, k))
                    values["Male Recall"][k].append(recall_at_k(ranked, rel, k))
                elif group == "F":
                    values["Female NDCG"][k].append(ndcg_at_k(ranked, rel, k))
                    values["Female Recall"][k].append(recall_at_k(ranked, rel, k))
        metrics = {name: {k: float(np.mean(vals[k])) if vals[k] else float("nan") for k in self.ks}
                   for name, vals in values.items()}
        # Ensure fixed public output rows even if an unusual split has no known group.
        for name in ("NDCG", "Recall", "Male NDCG", "Female NDCG", "Male Recall", "Female Recall"):
            metrics.setdefault(name, {k: float("nan") for k in self.ks})
        for k in self.ks:
            at_k = {u: ranked[:k] for u, ranked in results.items()}
            metrics.setdefault("JS-DP", {})[k] = js_dp(at_k, self.data.user_groups, self.data.num_items,
                                                        variant=self.fairness_metric_variant)
            metrics.setdefault("JS-EO", {})[k] = js_eo(at_k, relevant, self.data.user_groups, self.data.num_items,
                                                        variant=self.fairness_metric_variant)
        return metrics
