import numpy as np
import pytest


def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    similarity = dot_product / (magnitude_v1 * magnitude_v2)
    return np.round(similarity, 3)


@pytest.mark.parametrize(
    "v1, v2, expected",
    [
        (np.array([1, 2, 3]), np.array([2, 4, 6]), 1.0),
        (np.array([1, 2, 3]), np.array([-1, -2, -3]), -1.0),
        (np.array([1, 0, 7]), np.array([0, 1, 3]), 0.939),
    ],
)
def test_cosine_similarity(v1, v2, expected):
    got = cosine_similarity(v1, v2)
    assert got == expected
