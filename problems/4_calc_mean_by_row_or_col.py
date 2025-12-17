import numpy as np
import pytest


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    arr = np.array(matrix, dtype=float)

    if mode == "column":
        return arr.mean(axis=0).tolist()
    if mode == "row":
        return arr.mean(axis=1).tolist()

    raise ValueError("mode must be `row` or `column`")


@pytest.mark.parametrize(
    "matrix, mode, expected",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "column", [4.0, 5.0, 6.0]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "row", [2.0, 5.0, 8.0]),
        ([[1, 2, 3, 4], [5, 6, 7, 8]], "row", [2.5, 6.5]),
        ([[1, 2, 3, 4], [5, 6, 7, 8]], "column", [3.0, 4.0, 5.0, 6.0]),
    ],
)
def test_calculate_matrix_mean(matrix, mode, expected):
    got = calculate_matrix_mean(matrix, mode)
    assert got == expected
