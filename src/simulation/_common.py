"""Shared simulation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from src.domain import AnalysisConfig, Portfolio, ReturnEstimation

FloatArray = npt.NDArray[np.float64]


def validate_config(config: AnalysisConfig) -> None:
    if config.horizon <= 0:
        raise ValueError("horizon must be positive")
    if config.n_simulations <= 0:
        raise ValueError("n_simulations must be positive")


def weights_array(weights: Sequence[float], n_assets: int) -> FloatArray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (n_assets,):
        raise ValueError("portfolio weights must match the number of assets")
    if not np.all(np.isfinite(values)):
        raise ValueError("portfolio weights must be finite")
    return cast(FloatArray, values)


def covariance_array(estimation: ReturnEstimation, n_assets: int) -> FloatArray:
    values = np.asarray(estimation.covariance_matrix, dtype=np.float64)
    if values.shape != (n_assets, n_assets):
        raise ValueError("covariance matrix shape must match assets")
    if not np.all(np.isfinite(values)):
        raise ValueError("covariance matrix must be finite")
    return cast(FloatArray, values)


def historical_returns_array(estimation: ReturnEstimation, n_assets: int) -> FloatArray:
    values = np.asarray(estimation.returns, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != n_assets or values.shape[0] == 0:
        raise ValueError("historical returns must be a non-empty matrix aligned to assets")
    if not np.all(np.isfinite(values)):
        raise ValueError("historical returns must be finite")
    return cast(FloatArray, values)


def validate_portfolio_estimation(portfolio: Portfolio, estimation: ReturnEstimation) -> int:
    if portfolio.tickers != estimation.tickers:
        raise ValueError("portfolio tickers must match estimation tickers")
    return len(portfolio.tickers)


def terminal_compounded_returns(portfolio_paths: Any) -> FloatArray:
    paths = np.asarray(portfolio_paths, dtype=np.float64)
    if paths.ndim != 2:
        raise ValueError("portfolio paths must have shape (n_simulations, horizon)")
    terminal = np.prod(1.0 + paths, axis=1) - 1.0
    return terminal
