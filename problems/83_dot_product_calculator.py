import numpy as np
import pytest


def calculate_dot_product(vec1, vec2) -> float:
    return vec1.dot(vec2)


@pytest.mark.parametrize(
    "vec1, vec2, expected",
    [
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 32),
        (np.array([-1, 2, 3]), np.array([4, -5, 6]), 4),
        (np.array([1, 0]), np.array([0, 1]), 0),
    ],
)
def test_calculate_dot_product(vec1, vec2, expected):
    got = calculate_dot_product(vec1, vec2)
    assert got == expected
