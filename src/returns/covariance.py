"""Covariance estimation contracts."""

from __future__ import annotations

from typing import cast

import pandas as pd


class CovarianceEstimator:
    """Estimate the asset covariance matrix from empirical returns.

    Args:
        returns: A pandas ``DataFrame`` of asset returns with one column per
            ticker and one row per observation.

    Returns:
        A pandas ``DataFrame`` indexed and columned by ticker containing the
        sample covariance matrix using ``ddof=1``.

    Raises:
        ValueError: If returns are empty, non-numeric, or contain fewer than two
            observations.

    Invariants:
        The output covariance matrix must be symmetric and positive
        semi-definite for valid finite input. This component is independent of
        volatility, correlation, simulation, and risk metric calculation.
    """

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate a sample covariance matrix.

        Args:
            returns: Asset return matrix.

        Returns:
            Sample covariance matrix.

        Raises:
            ValueError: If the covariance matrix cannot be estimated.
        """
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("returns must be a pandas DataFrame")
        if returns.empty or len(returns) < 2:
            raise ValueError("returns must contain at least two observations")

        try:
            numeric_returns = returns.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("returns must be numeric") from exc

        if bool(numeric_returns.isna().any().any()):
            raise ValueError("returns must not contain missing values")

        return cast(pd.DataFrame, numeric_returns.cov())
