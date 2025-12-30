from collections import Counter

import numpy as np
import pytest


def mean(data: list | np.ndarray) -> int | float:
    s = data.sum() if hasattr(data, "sum") else sum(data)
    return s / len(data)


def median(data: list | np.ndarray) -> int | float:
    if hasattr(data, "median"):
        return data.median()

    seq = sorted(data)
    n = len(seq)
    mid = n // 2
    if n % 2 == 1:
        return seq[mid]
    return (seq[mid - 1] + seq[mid]) / 2


def mode(data: list | np.ndarray) -> int | float:
    if hasattr(data, "mode"):
        return data.mode()

    cnt = Counter(data)
    max_count = max(cnt.values())
    candidates = [k for k, v in cnt.items() if v == max_count]
    return min(candidates)


def variance(data: list | np.ndarray) -> int | float:
    ddof = 0
    if hasattr(data, "dtype") or isinstance(data, (np.ndarray,)):
        arr = np.asarray(data, dtype=float)
        return float(np.var(arr, ddof=ddof))

    n = len(data)
    mu = sum(data) / n
    ssd = sum((x - mu) ** 2 for x in data)
    var = ssd / (n - ddof)
    return round(var, 4)


def standard_deviation(data: list | np.ndarray) -> int | float:
    ddof = 0
    if hasattr(data, "dtype") or isinstance(data, (np.ndarray,)):
        arr = np.asarray(data, dtype=float)
        return float(np.var(arr, ddof=ddof))

    n = len(data)
    mu = sum(data) / n
    ssd = sum((x - mu) ** 2 for x in data)
    var = ssd / (n - ddof)
    std_var = var**0.5
    return round(std_var, 4)


def percentile(data: list | np.ndarray, percent: float) -> int | float:
    seq = sorted(data)
    n = len(seq)
    rank = percent * (n - 1)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return seq[lower]
    weight = rank - lower
    return seq[lower] * (1 - weight) + seq[upper] * weight


def interquartile_range(data: list | np.ndarray) -> int | float:
    q1 = percentile(data, 0.25)
    q3 = percentile(data, 0.75)
    return q3 - q1


def descriptive_statistics(data: list | np.ndarray) -> dict:
    return {
        "mean": mean(data),
        "median": median(data),
        "mode": mode(data),
        "variance": variance(data),
        "standard_deviation": standard_deviation(data),
        "25th_percentile": percentile(data, 0.25),
        "50th_percentile": percentile(data, 0.50),
        "75th_percentile": percentile(data, 0.75),
        "interquartile_range": interquartile_range(data),
    }


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            [1, 2, 2, 3, 4, 4, 4, 5],
            {
                "mean": 3.125,
                "median": 3.5,
                "mode": 4,
                "variance": 1.6094,
                "standard_deviation": 1.2686,
                "25th_percentile": 2.0,
                "50th_percentile": 3.5,
                "75th_percentile": 4.0,
                "interquartile_range": 2.0,
            },
        ),
        (
            [10, 20, 20, 30, 40],
            {
                "mean": 24.0,
                "median": 20.0,
                "mode": 20,
                "variance": 104.0,
                "standard_deviation": 10.198,
                "25th_percentile": 20.0,
                "50th_percentile": 20.0,
                "75th_percentile": 30.0,
                "interquartile_range": 10.0,
            },
        ),
        (
            # testing np.ndarray
            np.array([100]),
            {
                "mean": 100.0,
                "median": 100.0,
                "mode": 100,
                "variance": 0.0,
                "standard_deviation": 0.0,
                "25th_percentile": 100.0,
                "50th_percentile": 100.0,
                "75th_percentile": 100.0,
                "interquartile_range": 0.0,
            },
        ),
    ],
)
def test_descriptive_statistics(data, expected):
    got = descriptive_statistics(data)
    assert got == expected
