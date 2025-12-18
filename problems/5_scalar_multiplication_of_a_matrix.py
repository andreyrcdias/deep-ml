import numpy as np
import pytest


def scalar_multiply_np(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
    m = np.array(matrix)
    result = m * scalar
    return result.tolist()


def scalar_multiply(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    result = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        row = matrix[i]
        for j in range(len(row)):
            result[i][j] = row[j] * scalar
    return result


@pytest.mark.parametrize(
    "matrix, scalar, expected",
    [
        ([[1, 2], [3, 4]], 2, [[2, 4], [6, 8]]),
        ([[0, -1], [1, 0]], -1, [[0, 1], [-1, 0]]),
    ],
)
def test_scalar_multiply(matrix, scalar, expected):
    got = scalar_multiply(matrix, scalar)
    assert got == expected
