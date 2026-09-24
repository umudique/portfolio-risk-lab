"""Shared risk-measurement helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def finite_array(values: Any, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, array)


def terminal_losses(terminal_returns: Any) -> FloatArray:
    returns = finite_array(terminal_returns, name="terminal_returns")
    return -returns


def validate_confidence_level(confidence_level: float) -> None:
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")


def empirical_var(losses: FloatArray, confidence_level: float) -> float:
    validate_confidence_level(confidence_level)
    return float(np.quantile(losses, confidence_level, method="lower"))
