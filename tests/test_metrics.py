import numpy as np

from evaluation.ranking_metrics import ndcg_at_k, recall_at_k
from evaluation.fairness_metrics import js_dp, js_eo


def test_ranking_and_jensen_shannon_metrics():
    assert recall_at_k([1, 2], {1, 3}, 2) == 0.5
    assert 0 < ndcg_at_k([1, 2], {1, 3}, 2) <= 1
    recs = {0: [0, 1], 1: [2, 3]}
    groups = np.array(["M", "F"], dtype=object)
    assert js_dp(recs, groups, 4) > 0
    assert js_eo(recs, {0: [0], 1: [2]}, groups, 4) > 0


def test_js_distance_is_square_root_of_divergence():
    recs = {0: [0, 1], 1: [2, 3]}
    groups = np.array(["M", "F"], dtype=object)
    divergence = js_dp(recs, groups, 4, variant="divergence")
    assert np.isclose(js_dp(recs, groups, 4, variant="distance"), np.sqrt(divergence))
