"""Return estimation result contracts."""

from __future__ import annotations

from typing import Literal, cast

import pandas as pd

from src.domain import ReturnEstimation
from src.returns.calculation import ReturnConvention

MeanMethod = Literal["historical"]


class MeanReturnEstimator:
    """Estimate expected asset returns from empirical returns.

    Args:
        returns: A pandas ``DataFrame`` of asset returns.
        method: Mean convention. The MVP uses ``"historical"``, meaning the
            arithmetic sample mean of each asset's observed return series under
            the documented return convention.

    Returns:
        A pandas ``Series`` indexed by ticker containing one expected return per
        asset.

    Raises:
        ValueError: If returns are empty, non-numeric, or if an unsupported mean
            method is requested.

    Invariants:
        The mean vector shape must match the number and order of asset columns.
        This estimator does not simulate, bootstrap, or calculate risk metrics.
    """

    default_method: MeanMethod = "historical"

    def estimate(
        self,
        returns: pd.DataFrame,
        *,
        method: MeanMethod = "historical",
    ) -> pd.Series:
        """Estimate one expected return per asset.

        Args:
            returns: Asset return matrix.
            method: ``"historical"`` sample mean method.

        Returns:
            Mean return vector indexed by ticker.

        Raises:
            ValueError: If mean returns cannot be estimated.
        """
        if method != "historical":
            raise ValueError(f"unsupported mean return method: {method}")
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("returns must be a pandas DataFrame")
        if returns.empty:
            raise ValueError("returns must not be empty")

        try:
            numeric_returns = returns.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("returns must be numeric") from exc

        if bool(numeric_returns.isna().any().any()):
            raise ValueError("returns must not contain missing values")

        return cast(pd.Series, numeric_returns.mean())


class EstimationResultBuilder:
    """Build the canonical ``ReturnEstimation`` domain object.

    Args:
        returns: Empirical return matrix.
        mean_returns: Mean return vector aligned to return columns.
        volatilities: Volatility vector aligned to return columns.
        covariance_matrix: Covariance matrix aligned to return columns.
        correlation_matrix: Correlation matrix aligned to return columns.
        return_convention: Return convention used to produce ``returns``.
        covariance_method: Method label used to estimate covariance.

    Returns:
        A canonical :class:`src.domain.ReturnEstimation` with all fields
        populated. ``observation_count`` is the number of rows in ``returns``.
        ``estimation_window`` is non-empty metadata that records at least the
        return convention and covariance method.

    Raises:
        ValueError: If any input is missing, empty, or dimensionally
            inconsistent.

    Invariants:
        This builder only assembles already-computed return-estimation outputs.
        It does not fetch market data, run simulations, or calculate risk
        metrics.
    """

    default_covariance_method: str = "sample_ddof_1"

    def build(
        self,
        *,
        returns: pd.DataFrame,
        mean_returns: pd.Series,
        volatilities: pd.Series,
        covariance_matrix: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        return_convention: ReturnConvention = "log",
        covariance_method: str = "sample_ddof_1",
    ) -> ReturnEstimation:
        """Build the canonical return-estimation result.

        Args:
            returns: Asset return matrix.
            mean_returns: Mean return vector.
            volatilities: Volatility vector.
            covariance_matrix: Asset covariance matrix.
            correlation_matrix: Asset correlation matrix.
            return_convention: Convention used for return calculation.
            covariance_method: Covariance method metadata.

        Returns:
            Canonical immutable ``ReturnEstimation``.

        Raises:
            ValueError: If input shapes or metadata are invalid.
        """
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("returns must be a pandas DataFrame")
        if returns.empty:
            raise ValueError("returns must not be empty")
        if not return_convention:
            raise ValueError("return_convention must not be empty")
        if not covariance_method:
            raise ValueError("covariance_method must not be empty")

        tickers = tuple(str(ticker) for ticker in returns.columns)
        estimation_window = f"{return_convention}, {covariance_method}, observations={len(returns)}"

        return ReturnEstimation(
            tickers=tickers,
            returns=returns,
            mean_returns=mean_returns,
            volatilities=volatilities,
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            observation_count=len(returns),
            estimation_window=estimation_window,
        )
