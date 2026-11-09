import numpy as np

from evaluation.reranker import exposure_parity_rerank


def test_reranker_returns_unique_top_k_lists():
    candidates = {0: [0, 1, 2], 1: [2, 1, 0], 2: [0, 2, 1], 3: [2, 0, 1]}
    scores = {user: [3.0, 2.0, 1.0] for user in candidates}
    output = exposure_parity_rerank(candidates, scores, np.array(["M", "F", "M", "F"], dtype=object), 2, 0.2)
    assert all(len(items) == 2 and len(set(items)) == 2 for items in output.values())
