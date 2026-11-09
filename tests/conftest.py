from __future__ import annotations

import numpy as np
import pytest

from datasets.base_dataset import RecommendationData


@pytest.fixture
def tiny_data():
    train = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4]], dtype=np.int64)
    return RecommendationData(
        "tiny", 4, 6, train, {0: [4], 1: [4], 2: [4], 3: [5]}, {0: [5], 1: [5], 2: [5], 3: [0]},
        np.array(["M", "F", "M", "F"], dtype=object), ["M", "F", "unknown"],
        {str(u): u for u in range(4)}, {str(i): i for i in range(6)},
    )
