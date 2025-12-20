import numpy as np
import pytest


def calculate_covariance_matrix_np(vectors: list[list[float]]) -> list[list[float]]:
    arr = np.asarray(vectors, dtype=float)
    n_feats, n_obs = arr.shape
    if n_obs == 0:
        return [[0.0] * n_feats for _ in range(n_feats)]
    if n_obs == 1:
        cov = np.zeros((n_feats, n_feats), dtype=float)
    else:
        cov = np.cov(arr, bias=False)
    return cov.tolist()


# TODO: write calculate_covariance_matrix without numpy


@pytest.mark.parametrize(
    "vectors, expected",
    [
        ([[1, 2, 3], [4, 5, 6]], [[1.0, 1.0], [1.0, 1.0]]),
        ([[1, 5, 6], [2, 3, 4], [7, 8, 9]], [[7.0, 2.5, 2.5], [2.5, 1.0, 1.0], [2.5, 1.0, 1.0]]),
    ],
)
def test_calculate_covariance_matrix(vectors, expected):
    got = calculate_covariance_matrix_np(vectors)
    assert got == expected
