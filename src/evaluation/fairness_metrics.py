"""Jensen-Shannon exposure metrics used when a source-paper definition is absent.

JS-DP is the JS divergence of normalized top-K item exposure distributions for
the two sensitive groups. JS-EO uses the same distribution restricted to
recommendations that are relevant held-out items. This operationalizes DP as
unconditional exposure parity and EO as exposure parity conditional on a true
positive. Both are bounded in [0, log(2)] (natural log) and use epsilon
smoothing for zero exposure. It is deliberately configurable/documented rather
than claimed to be a particular paper's unprovided definition.
"""
from __future__ import annotations
import numpy as np


def _js(first: np.ndarray, second: np.ndarray, epsilon: float = 1e-12,
        variant: str = "divergence") -> float:
    """Return JS divergence or its square-root distance compatibility form.

    SciPy's ``jensenshannon`` used by the supplied FairIB and UGRec repositories
    returns the square root of JS (the JS *distance*). ``distance`` reproduces
    that convention; ``divergence`` preserves this repository's original,
    natural-log JS divergence. Both compare exactly the same exposure vectors.
    """
    first, second = first.astype(float) + epsilon, second.astype(float) + epsilon
    first, second = first / first.sum(), second / second.sum()
    mean = (first + second) / 2
    divergence = float(0.5 * ((first * np.log(first / mean)).sum() + (second * np.log(second / mean)).sum()))
    if variant == "divergence":
        return divergence
    if variant == "distance":
        return float(np.sqrt(max(divergence, 0.0)))
    raise ValueError("fairness metric variant must be 'divergence' or 'distance'")


def js_dp(recommendations: dict[int, list[int]], groups: np.ndarray, num_items: int,
          first_group: str = "M", second_group: str = "F", variant: str = "divergence") -> float:
    counts = [np.zeros(num_items), np.zeros(num_items)]
    for user, items in recommendations.items():
        target = 0 if groups[user] == first_group else 1 if groups[user] == second_group else -1
        if target >= 0:
            counts[target][items] += 1
    return _js(counts[0], counts[1], variant=variant)


def js_eo(recommendations: dict[int, list[int]], relevant: dict[int, list[int]], groups: np.ndarray,
          num_items: int, first_group: str = "M", second_group: str = "F", variant: str = "divergence") -> float:
    counts = [np.zeros(num_items), np.zeros(num_items)]
    for user, items in recommendations.items():
        target = 0 if groups[user] == first_group else 1 if groups[user] == second_group else -1
        if target >= 0:
            positives = set(relevant.get(user, []))
            hits = [item for item in items if item in positives]
            counts[target][hits] += 1
    return _js(counts[0], counts[1], variant=variant)
