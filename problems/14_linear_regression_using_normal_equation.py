import numpy as np
import pytest


def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    X_transpose = X.T
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    theta = np.round(theta, 4).flatten().tolist()
    return theta


@pytest.mark.parametrize(
    "X, y, expected",
    [
        ([[1, 1], [1, 2], [1, 3]], [1, 2, 3], [0.0, 1.0]),
        ([[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1], [4.0, -1.0, -0.0]),
    ],
)
def test_linear_regression_normal_equation(X, y, expected):
    got = linear_regression_normal_equation(X, y)
    assert got == expected
