"""Per-user top-K relevance metrics."""
from __future__ import annotations
import math
from typing import Iterable


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0
