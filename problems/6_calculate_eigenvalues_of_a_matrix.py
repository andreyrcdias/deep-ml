import cmath

import pytest


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    # λ² - tr(A)λ + det(A) = 0
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]

    # sum of the main diagonal
    trace = a + d

    det = a * d - b * c

    # Δ = tr(A)² - 4*det(A)
    discriminant = trace * trace - 4 * det
    sqrt_disc = cmath.sqrt(discriminant)
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2
    return [lambda1, lambda2]


@pytest.mark.parametrize(
    "matrix, expected",
    [
        ([[2, 1], [1, 2]], [3.0, 1.0]),
        ([[4, -2], [1, 1]], [3.0, 2.0]),
    ],
)
def test_calculate_eigenvalues(matrix, expected):
    got = calculate_eigenvalues(matrix)
    assert got == expected
