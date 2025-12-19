import numpy as np
import pytest


def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    a, b = matrix[0]
    c, d = matrix[1]

    det = a * d - b * c

    eps = 1e-12
    if abs(det) <= eps:
        return None

    inv_det = 1.0 / det
    inverse = [[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]]
    return [[round(float(x), 1) for x in row] for row in inverse]


def inverse_2x2_np(matrix: list[list[float]]) -> list[list[float]] | None:
    try:
        m = np.array(matrix)
        inverse = np.linalg.inv(m)
        inverse = np.round(inverse, decimals=1)
        return inverse.tolist()
    except Exception:
        return None


@pytest.mark.parametrize(
    "matrix, expected",
    [
        ([[4, 7], [2, 6]], [[0.6, -0.7], [-0.2, 0.4]]),
        ([[2, 1], [6, 2]], [[-1.0, 0.5], [3.0, -1.0]]),
        ([[1, 2], [2, 4]], None),
    ],
)
def test_inverse_2x2(matrix, expected):
    got = inverse_2x2(matrix)
    assert got == expected
