"""Correlation estimation contracts."""

from __future__ import annotations

from typing import cast

import pandas as pd


class CorrelationEstimator:
    """Estimate the asset correlation matrix from empirical returns.

    Args:
        returns: A pandas ``DataFrame`` of finite asset returns with one column
            per ticker and one row per observation.

    Returns:
        A pandas ``DataFrame`` indexed and columned by ticker containing the
        sample Pearson correlation matrix.

    Raises:
        ValueError: If returns are empty, non-numeric, contain fewer than two
            observations, or include a zero-variance asset for which
            correlations are undefined.

    Invariants:
        The diagonal of a valid correlation matrix is exactly ``1.0`` and all
        entries lie in ``[-1.0, 1.0]``. This component does not estimate means,
        simulate returns, or calculate risk metrics.
    """

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate a sample correlation matrix.

        Args:
            returns: Asset return matrix.

        Returns:
            Sample Pearson correlation matrix.

        Raises:
            ValueError: If the correlation matrix cannot be estimated.
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
        if bool((numeric_returns.std(ddof=1) == 0.0).any()):
            raise ValueError("correlation is undefined for zero-variance assets")

        correlation = numeric_returns.corr().copy()
        for ticker in correlation.columns:
            correlation.loc[ticker, ticker] = 1.0
        return cast(pd.DataFrame, correlation)
