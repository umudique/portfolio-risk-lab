"""Volatility estimation contracts."""

from __future__ import annotations

import math
from typing import cast

import pandas as pd


class VolatilityEstimator:
    """Estimate per-asset annualized volatility from empirical returns.

    Args:
        returns: A pandas ``DataFrame`` of asset returns where columns are
            tickers and rows are observations under a documented return
            convention.
        annualization_factor: Number of return periods per year. The MVP
            default is ``252`` for daily trading observations.

    Returns:
        A pandas ``Series`` indexed by ticker containing sample standard
        deviations with ``ddof=1``, annualized by multiplying by
        ``sqrt(annualization_factor)``.

    Raises:
        ValueError: If returns are empty, non-numeric, or contain fewer than two
            observations.

    Invariants:
        Volatility estimation is independently testable and does not compute
        covariance, correlation, simulation paths, or risk metrics.
    """

    default_annualization_factor: int = 252

    def estimate(
        self,
        returns: pd.DataFrame,
        *,
        annualization_factor: int = 252,
    ) -> pd.Series:
        """Estimate annualized per-asset volatility.

        Args:
            returns: Asset return matrix.
            annualization_factor: Periods per year used for annualization.

        Returns:
            Annualized volatility per asset.

        Raises:
            ValueError: If inputs cannot produce sample volatility estimates.
        """
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("returns must be a pandas DataFrame")
        if returns.empty or len(returns) < 2:
            raise ValueError("returns must contain at least two observations")
        if annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")

        try:
            numeric_returns = returns.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("returns must be numeric") from exc

        if bool(numeric_returns.isna().any().any()):
            raise ValueError("returns must not contain missing values")

        return cast(pd.Series, numeric_returns.std(ddof=1) * math.sqrt(float(annualization_factor)))
