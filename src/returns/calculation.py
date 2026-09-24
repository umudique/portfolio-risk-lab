"""Return calculation contracts."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

ReturnConvention = Literal["log", "simple"]


class ReturnCalculator:
    """Convert an aligned price matrix into asset return series.

    Args:
        prices: A pandas ``DataFrame`` of strictly positive aligned asset prices.
            Columns are ticker symbols and rows are chronological observations.
        method: Return convention. The default and canonical MVP convention is
            ``"log"``, calculated as ``ln(P_t / P_{t-1})``. The explicit
            alternative is ``"simple"``, calculated as ``P_t / P_{t-1} - 1``.

    Returns:
        A pandas ``DataFrame`` with one fewer row than ``prices`` and the same
        asset columns, containing returns under the requested convention.

    Raises:
        ValueError: If prices are empty, non-positive, non-numeric, or if fewer
            than two price observations are provided.

    Invariants:
        This component only transforms aligned prices into empirical returns.
        It does not estimate risk metrics, simulate paths, fetch market data,
        or choose portfolio weights. Downstream components must receive the
        documented return convention in ``ReturnEstimation.estimation_window``.
    """

    default_method: ReturnConvention = "log"

    def calculate(
        self,
        prices: pd.DataFrame,
        *,
        method: ReturnConvention = "log",
    ) -> pd.DataFrame:
        """Calculate asset returns from aligned prices.

        Args:
            prices: Aligned positive price matrix.
            method: ``"log"`` for canonical log returns or ``"simple"`` for
                explicit simple returns.

        Returns:
            Asset return matrix with the same columns as ``prices``.

        Raises:
            ValueError: If the price matrix or return convention is invalid.
        """
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("prices must be a pandas DataFrame")
        if prices.empty or len(prices) < 2:
            raise ValueError("prices must contain at least two observations")

        try:
            numeric_prices = prices.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("prices must be numeric") from exc

        if bool(numeric_prices.isna().any().any()):
            raise ValueError("prices must not contain missing values")
        if bool((numeric_prices <= 0.0).any().any()):
            raise ValueError("prices must be strictly positive")

        price_ratio = numeric_prices / numeric_prices.shift(1)
        if method == "log":
            returns = np.log(price_ratio)
        elif method == "simple":
            returns = price_ratio - 1.0
        else:
            raise ValueError(f"unsupported return convention: {method}")

        return cast(pd.DataFrame, returns.iloc[1:])
