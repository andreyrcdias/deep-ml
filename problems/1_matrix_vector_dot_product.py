import pytest


def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:
    # Return a list where each element is the dot product of a row of 'a' with 'b'.
    # If the number of columns in 'a' does not match the length of 'b', return -1.
    if len(a[0]) != len(b):
        return -1
    result: list[int] = []
    for row_a in a:
        dot_product = sum(row_a[i] * b[i] for i in range(len(b)))
        result.append(dot_product)
    return result


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([[1, 2], [2, 4]], [1, 2], [5, 10]),
        ([[1, 2, 3], [2, 4, 5], [6, 8, 9]], [1, 2, 3], [14, 25, 49]),
        ([[1, 2], [2, 4], [6, 8], [12, 4]], [1, 2, 3], -1),
        ([[1.5, 2.5], [3.0, 4.0]], [2, 1], [5.5, 10.0]),
    ],
)
def test_matrix_dot_vector(a, b, expected):
    got = matrix_dot_vector(a, b)
    assert got == expected
