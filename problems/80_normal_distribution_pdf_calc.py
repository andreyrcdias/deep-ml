import math

import numpy as np
import pytest


def normal_pdf(x, mean, std_dev):
    """
    f(x) = (1 / (σ √(2π))) · exp(−(x − μ)² / (2 σ²))

    Where:
    μ       is the mean,
    σ > 0   is the standard deviation,
    exp     is the exponential function.
    """

    coeff = 1.0 / (std_dev * math.sqrt(2 * math.pi))
    exp = -0.5 * ((x - mean) / std_dev) ** 2
    val = coeff * math.exp(exp)
    return round(val, 5)


@pytest.mark.parametrize(
    "x, mean, std_dev, expected",
    [(16, 15, 2.04, 0.17342), (0, 0, 1, 0.39894), (1, 0, 0.5, 0.10798)],
)
def test_normal_pdf(x, mean, std_dev, expected):
    got = normal_pdf(x, mean, std_dev)
    assert got == expected
