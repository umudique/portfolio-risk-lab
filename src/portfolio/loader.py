"""Portfolio CSV loading contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domain import Portfolio
from src.portfolio.canonicalizer import PortfolioCanonicalizer
from src.portfolio.validation import PortfolioSchemaValidator, WeightValidator


class PortfolioLoader:
    """Load a portfolio CSV and return the canonical domain contract.

    Args:
        csv_path: File system path to a CSV containing only portfolio composition
            data. The required public columns are ``ticker`` and ``weight``.

    Returns:
        A canonical :class:`src.domain.Portfolio` whose tickers and weights have
        been schema-validated, weight-validated, and deterministically ordered by
        :class:`PortfolioCanonicalizer`.

    Raises:
        ValueError: If the CSV is empty, missing required columns, contains
            duplicate tickers, contains non-parseable values, or violates the
            documented weight rule.

    Invariants:
        This loader reads the portfolio CSV only. It performs no market-data
        calls, return estimation, simulation, or risk metric calculation.
    """

    def load(self, csv_path: str | Path) -> Portfolio:
        """Load, validate, and canonicalize a portfolio CSV.

        Args:
            csv_path: Path to the portfolio CSV.

        Returns:
            The canonical immutable portfolio.

        Raises:
            ValueError: If any schema, duplicate ticker, empty CSV, or weight
                validation rule is violated.
        """
        try:
            rows = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError("portfolio CSV is empty or has no data rows") from exc

        schema_valid_rows = PortfolioSchemaValidator().validate(rows)
        weight_valid_rows = WeightValidator().validate(schema_valid_rows)
        return PortfolioCanonicalizer().canonicalize(weight_valid_rows)
