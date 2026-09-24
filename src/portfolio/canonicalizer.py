"""Portfolio canonicalization contract."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.domain import Portfolio


class PortfolioCanonicalizer:
    """Produce the canonical portfolio ordering used across module boundaries.

    Args:
        rows: Schema-valid and weight-valid portfolio data.

    Returns:
        A :class:`src.domain.Portfolio` with tickers ordered alphabetically by
        their normalized ticker text. Weights are moved with their associated
        tickers, so the portfolio composition is preserved exactly.

    Raises:
        ValueError: If the input cannot produce a non-empty canonical
            :class:`src.domain.Portfolio`.

    Invariants:
        Canonicalization is deterministic: two CSV files containing the same
        ticker/weight pairs in different row orders produce equal portfolios.
        This component does not load market data, estimate returns, simulate
        paths, or calculate risk metrics.
    """

    def canonicalize(self, rows: Any) -> Portfolio:
        """Build a deterministic immutable Portfolio from validated rows.

        Args:
            rows: Schema-valid and weight-valid portfolio data.

        Returns:
            Canonical immutable portfolio.

        Raises:
            ValueError: If the canonical portfolio cannot be produced.
        """
        if not isinstance(rows, pd.DataFrame):
            raise ValueError("portfolio rows must be a pandas DataFrame")

        ordered = rows.sort_values(by="ticker", kind="mergesort")
        tickers = tuple(str(ticker) for ticker in ordered["ticker"])
        weights = tuple(float(weight) for weight in ordered["weight"])

        return Portfolio(tickers=tickers, weights=weights)
