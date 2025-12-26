import numpy as np
import pytest


def matrixmul(a: list[list[int | float]], b: list[list[int | float]]) -> list[list[int | float]]:
    m, n = len(a), len(a[0])
    if any(len(row) != n for row in a):
        return -1

    p, q = len(b), len(b[0])
    if any(len(row) != q for row in b):
        return -1

    if n != p:
        return -1

    result = [[0 for _ in range(q)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            aij = a[i][j]
            if aij == 0:
                continue
            bj_row = b[j]
            res_row = result[i]
            for k in range(q):
                res_row[k] += aij * bj_row[k]
    return result


def matrixmul_np(a: list[list[int | float]], b: list[list[int | float]]) -> list[list[int | float]]:
    A = np.array(a)
    B = np.array(b)
    if A.shape[1] != B.shape[0]:
        return -1
    c = A @ B
    return c.tolist()


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([[1, 2], [2, 4]], [[2, 1], [3, 4]], [[8, 9], [16, 18]]),
        (
            [[0, 0], [2, 4], [1, 2]],
            [[0, 0], [2, 4]],
            [[0, 0], [8, 16], [4, 8]],
        ),
        ([[0, 0], [2, 4], [1, 2]], [[0, 0, 1], [2, 4, 1], [1, 2, 3]], -1),
    ],
)
def test_matrixmul(a, b, expected):
    got = matrixmul(a, b)
    assert got == expected
