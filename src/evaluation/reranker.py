"""Global, score-only target-exposure slate optimization."""
from __future__ import annotations

from typing import Dict

import numpy as np


def exposure_parity_rerank(candidates: Dict[int, list[int]], candidate_scores: Dict[int, list[float]],
                           groups: np.ndarray, k: int, strength: float,
                           common_relevance_prior: np.ndarray | None = None,
                           num_items: int | None = None,
                           conditional_relevance_priors: np.ndarray | None = None,
                           eo_proxy_strength: float = 0.0) -> Dict[int, list[int]]:
    """Select slates that track one shared common-support exposure target.

    The target favors items that are available with high predicted utility to
    *both* groups, rather than the union of each group's distinct candidates.
    An optional prior from training-positive item distributions focuses this
    common support on items that are historically relevant to both groups. This
    directly addresses the disjoint-support cause of high JS-DP and JS-EO.
    Held-out relevance is never accessed.
    """
    if strength <= 0:
        return {user: items[:k] for user, items in candidates.items()}
    if strength > 1:
        raise ValueError("global rerank strength must lie in [0, 1]")
    if not 0 <= eo_proxy_strength <= 1:
        raise ValueError("EO proxy strength must lie in [0, 1]")
    known = {label: sorted(user for user in candidates if groups[user] == label) for label in ("M", "F")}
    inferred_items = max((max(items, default=-1) for items in candidates.values()), default=-1) + 1
    item_count = int(num_items) if num_items is not None else inferred_items
    if item_count < inferred_items:
        raise ValueError("num_items cannot exclude a candidate item")
    output: Dict[int, list[int]] = {user: items[:k] for user, items in candidates.items() if groups[user] not in {"M", "F"}}
    if item_count <= 0 or not known["M"] or not known["F"]:
        return {user: items[:k] for user, items in candidates.items()}

    target = np.full(item_count, 1e-12, dtype=np.float64)
    normalized_scores: Dict[int, np.ndarray] = {}
    availability = {label: np.zeros(item_count, dtype=np.float64) for label in ("M", "F")}
    utility = {label: np.zeros(item_count, dtype=np.float64) for label in ("M", "F")}
    for user, items in candidates.items():
        scores = np.asarray(candidate_scores[user], dtype=np.float64)
        normalized = (scores - scores.min()) / max(float(scores.max() - scores.min()), 1e-12)
        normalized_scores[user] = normalized
        if groups[user] in {"M", "F"}:
            weights = np.exp((normalized - normalized.max()) / 0.25)
            label = str(groups[user])
            indices = np.asarray(items)
            availability[label][indices] += 1
            utility[label][indices] += weights / weights.sum()
    # Geometric means are zero when an item is available to only one group,
    # preventing a group-exclusive candidate from becoming the parity target.
    availability_m = availability["M"] / len(known["M"])
    availability_f = availability["F"] / len(known["F"])
    utility_m = utility["M"] / len(known["M"])
    utility_f = utility["F"] / len(known["F"])
    target += np.sqrt(availability_m * availability_f) * np.sqrt(utility_m * utility_f)
    if common_relevance_prior is not None:
        if len(common_relevance_prior) != item_count:
            raise ValueError("common relevance prior and candidate item space differ")
        target *= np.sqrt(np.clip(common_relevance_prior, 1e-12, None))
    target /= target.sum()
    if conditional_relevance_priors is not None:
        if conditional_relevance_priors.shape != (2, item_count):
            raise ValueError("conditional relevance priors must have shape [2, num_items]")

    for group in ("M", "F"):
        users = known[group]
        desired_distribution = target
        if conditional_relevance_priors is not None and eo_proxy_strength:
            # q_g(i) p_g(i) estimates group-g true-positive exposure. Scaling
            # a common target by 1 / p_g therefore targets equal conditional
            # exposure without ever reading validation or test relevance.
            group_index = 0 if group == "M" else 1
            eo_desired = target / conditional_relevance_priors[group_index]
            eo_desired /= eo_desired.sum()
            desired_distribution = (1.0 - eo_proxy_strength) * target + eo_proxy_strength * eo_desired
        desired = desired_distribution * (len(users) * k)
        exposure = np.zeros(item_count, dtype=np.float64)
        for user in users:
            items = candidates[user]
            item_indices = np.asarray(items)
            utility = normalized_scores[user].copy()
            available = np.ones(len(items), dtype=bool)
            chosen: list[int] = []
            for _ in range(min(k, len(items))):
                deficit = (desired[item_indices] - exposure[item_indices]) / (desired[item_indices] + 1.0)
                objective = utility + strength * np.clip(deficit, -1.0, 1.0)
                objective[~available] = -np.inf
                position = int(np.argmax(objective))
                item = items[position]
                chosen.append(item)
                exposure[item] += 1
                available[position] = False
            output[user] = chosen
    return output
