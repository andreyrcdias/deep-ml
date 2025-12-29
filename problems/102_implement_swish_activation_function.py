import math

import pytest


def swish(x: float) -> float:
    x = x * (1.0 / (1.0 + math.exp(-x)))
    return round(x, 4)


@pytest.mark.parametrize(
    "x, expected",
    [
        (1, 0.7311),
        (0, 0.0),
        (-1, -0.2689),
        (10, 9.9995),
        (-10, -0.0005),
    ],
)
def test_swish(x, expected):
    got = swish(x)
    assert got == expected
