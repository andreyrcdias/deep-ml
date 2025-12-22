import numpy as np
import pytest


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))

    for _ in range(iterations):
        # predictions: (m,1)
        preds = X @ theta
        # error: (m,1)
        error = preds - y
        # gradient: (n,1)
        grad = (X.T @ error) / m
        # update
        theta = theta - alpha * grad

    return np.round(theta.flatten(), 4)


@pytest.mark.parametrize(
    "X, y, alpha, iterations, expected",
    [
        (
            np.array([[1, 1], [1, 2], [1, 3]]),
            np.array([1, 2, 3]),
            0.01,
            1000,
            np.array([0.1107, 0.9513]),
        ),
        (
            np.array([[1, 1], [1, 2], [1, 3]]),
            np.array([1, 2, 3]),
            0.01,
            1000,
            np.array([0.1107, 0.9513]),
        ),
    ],
)
def test_linear_regression_gradient_descent(X, y, alpha, iterations, expected):
    got = linear_regression_gradient_descent(X, y, alpha, iterations)
    assert np.array_equal(got, expected, equal_nan=True)
