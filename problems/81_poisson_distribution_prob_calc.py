import math

import numpy as np
import pytest


def poisson_probability(k, lam):
    # (P(k, λ) = ((e^-λ)(λ^k))/k!

    # P(k, λ): The probability of exactly x occurrences
    # e:        Euler's number, approximately 2.71828
    # λ:        The average number of events in the given interval (the mean rate)
    # x:        The actual number of occurrences you're interested in (e.g., 0, 1, 2, ...)
    # x!:       The factorial of x(x X (x-1) X ... X1))

    # e^(-λ)
    val = math.exp(-lam)

    # (λ^k)/k!
    for i in range(k):
        val *= lam
        val /= i + 1

    return round(val, 5)


def poisson_probability_np(k, lam):
    k_arr = np.asarray(k)
    lam_arr = np.asarray(lam)
    lgamma_vec = np.vectorize(math.lgamma)
    val = np.exp(-lam_arr + k_arr * np.log(lam_arr) - lgamma_vec(k_arr + 1))
    return float(np.round(val, 5))


@pytest.mark.parametrize(
    "k, lam, expected",
    [(3, 5, 0.14037), (0, 5, 0.00674), (2, 10, 0.00227)],
)
def test_poisson_probability(k, lam, expected):
    got = poisson_probability_np(k, lam)
    assert got == expected
