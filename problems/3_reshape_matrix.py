import numpy as np
import pytest


def reshape_matrix(
    a: list[list[int | float]], new_shape: tuple[int, int]
) -> list[list[int | float]]:
    try:
        reshaped_matrix = np.array(a).reshape(new_shape)
        return reshaped_matrix.tolist()
    except ValueError:
        return []


@pytest.mark.parametrize(
    "a, new_shape, expected",
    [
        ([[1, 2, 3, 4], [5, 6, 7, 8]], (4, 2), [[1, 2], [3, 4], [5, 6], [7, 8]]),
        ([[1, 2, 3, 4], [5, 6, 7, 8]], (4, 2), [[1, 2], [3, 4], [5, 6], [7, 8]]),
        ([[1, 2, 3, 4], [5, 6, 7, 8]], (1, 4), []),
        ([[1, 2, 3], [4, 5, 6]], (3, 2), [[1, 2], [3, 4], [5, 6]]),
        ([[1, 2, 3, 4], [5, 6, 7, 8]], (2, 4), [[1, 2, 3, 4], [5, 6, 7, 8]]),
    ],
)
def test_reshape_matrix(a, new_shape, expected):
    got = reshape_matrix(a, new_shape)
    assert got == expected
