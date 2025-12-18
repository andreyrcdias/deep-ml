import pytest


def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    rows, columns = len(a), len(a[0])
    b = [[0] * rows for _ in range(columns)]
    for i in range(rows):
        for j in range(columns):
            b[j][i] = a[i][j]
    return b


def transpose_matrix_v2(a: list[list[int | float]]) -> list[list[int | float]]:
    return [list(col) for col in zip(*a)]


def transpose_matrix_np(a: list[list[int | float]]) -> list[list[int | float]]:
    import numpy as np

    a_np = np.array(a)
    return np.transpose(a_np).tolist()


@pytest.mark.parametrize(
    "a, expected",
    [
        ([[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]]),
        ([[1, 2], [3, 4], [5, 6]], [[1, 3, 5], [2, 4, 6]]),
        ([[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]]),
    ],
)
def test_transpose_matrix(a, expected):
    got = transpose_matrix(a)
    assert got == expected
