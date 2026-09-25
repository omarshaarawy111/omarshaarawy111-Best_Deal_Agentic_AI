"""
Regression metrics for Best Deal models.

Description:
    This module contains pure metrics extracted from model-evaluation work.
    Model training, plotting, and model-selection experiments remain notebooks.

Responsibilities:
    - Compute MAE, RMSE, MAPE, and R-squared.
    - Compute tolerance accuracy for price predictions.
"""

from __future__ import annotations

import math


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """
    Calculate mean absolute error.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        Mean absolute error.
    """
    _validate_pairs(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def root_mean_squared_error(actual: list[float], predicted: list[float]) -> float:
    """
    Calculate root mean squared error.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        Root mean squared error.
    """
    _validate_pairs(actual, predicted)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def mean_absolute_percentage_error(actual: list[float], predicted: list[float]) -> float:
    """
    Calculate mean absolute percentage error while excluding zero actual values.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        MAPE as a percentage.
    """
    _validate_pairs(actual, predicted)
    values = [abs(a - p) / abs(a) * 100 for a, p in zip(actual, predicted) if a != 0]
    return sum(values) / len(values) if values else 0.0


def r_squared(actual: list[float], predicted: list[float]) -> float:
    """
    Calculate coefficient of determination.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        R-squared value.
    """
    _validate_pairs(actual, predicted)
    mean_actual = sum(actual) / len(actual)
    total = sum((value - mean_actual) ** 2 for value in actual)
    residual = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 0.0 if total == 0 else 1 - residual / total


def within_percentage(actual: list[float], predicted: list[float], tolerance: float) -> float:
    """
    Calculate the fraction of predictions within a percentage tolerance.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.
        tolerance: Relative tolerance as a fraction, such as 0.2 for 20%.

    Returns:
        Fraction of predictions inside the tolerance.
    """
    _validate_pairs(actual, predicted)
    valid = [abs(a - p) / abs(a) <= tolerance for a, p in zip(actual, predicted) if a != 0]
    return sum(valid) / len(valid) if valid else 0.0


def _validate_pairs(actual: list[float], predicted: list[float]) -> None:
    """
    Validate that metric input arrays are non-empty and aligned.

    Args:
        actual: Ground-truth values.
        predicted: Model predictions.

    Returns:
        None.
    """
    if not actual or len(actual) != len(predicted):
        raise ValueError("actual and predicted must be non-empty and have equal length")
