import numpy as np
import pytest


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode not in ("column", "row"):
        raise ValueError("mode must be `row` or `column`")
    arr = np.array(matrix, dtype=float)
    axis = {
        "column": 0,
        "row": 1,
    }
    return arr.mean(axis=axis[mode]).tolist()


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
