"""Dataset-independent recommendation data structures.

Adapters provide contiguous user/item indices and split interactions; model code never
depends on source column names or MovieLens paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


Interaction = Tuple[int, int]


@dataclass
class RecommendationData:
    name: str
    num_users: int
    num_items: int
    train: np.ndarray  # shape [n, 2], user/item contiguous indices
    validation: Dict[int, List[int]]
    test: Dict[int, List[int]]
    user_groups: np.ndarray  # string labels, "unknown" is permitted
    group_names: List[str]
    user_id_map: Mapping[str, int]
    item_id_map: Mapping[str, int]

    def train_items_by_user(self) -> Dict[int, set[int]]:
        seen: Dict[int, set[int]] = {u: set() for u in range(self.num_users)}
        for u, i in self.train.tolist():
            seen[int(u)].add(int(i))
        return seen

    def eligible_groups(self) -> List[str]:
        """Groups with train observations, excluding unknown sensitive values."""
        present = set(self.user_groups[self.train[:, 0]]) if len(self.train) else set()
        return [g for g in self.group_names if g != "unknown" and g in present]

