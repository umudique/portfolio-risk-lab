"""Portfolio validation contracts."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


class PortfolioSchemaValidator:
    """Validate portfolio CSV schema without silently correcting input.

    Args:
        rows: A tabular portfolio representation produced by the CSV loader.
            The concrete implementation may use a pandas ``DataFrame``.

    Returns:
        A schema-valid tabular representation with required columns present and
        values parseable for downstream weight validation.

    Raises:
        ValueError: If required columns are absent, the CSV has no data rows,
            tickers are missing or duplicated, or any weight value cannot be
            parsed as numeric.

    Invariants:
        The validator never normalizes weights, reorders rows, fetches market
        data, estimates returns, or calculates risk metrics.
    """

    def validate(self, rows: Any) -> pd.DataFrame:
        """Validate required columns and parseable ticker/weight values.

        Args:
            rows: Tabular portfolio data.

        Returns:
            Schema-valid tabular portfolio data.

        Raises:
            ValueError: If the schema contract is violated.
        """
        if not isinstance(rows, pd.DataFrame):
            raise ValueError("portfolio rows must be a pandas DataFrame")

        required_columns = {"ticker", "weight"}
        missing_columns = required_columns.difference(rows.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"missing required portfolio column: {missing}")

        if rows.empty:
            raise ValueError("portfolio CSV is empty or has no data rows")

        validated = rows.loc[:, ["ticker", "weight"]].copy()

        ticker_text = validated["ticker"].astype("string")
        missing_ticker = ticker_text.isna() | (ticker_text.str.strip() == "")
        if bool(missing_ticker.any()):
            raise ValueError("ticker values must be present")

        if bool(ticker_text.duplicated().any()):
            raise ValueError("duplicate ticker values are not allowed")

        try:
            validated["weight"] = pd.to_numeric(validated["weight"], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("weight values must be numeric") from exc

        return validated


class WeightValidator:
    """Validate portfolio weights using a strict sum-to-one policy.

    The MVP policy is deterministic and non-normalizing: all weights must be
    numeric and their sum must equal ``1.0`` within an absolute tolerance of
    ``1e-9``. Inputs whose weights sum to any other value are rejected with
    ``ValueError`` rather than being corrected or rescaled.

    Args:
        rows: Schema-valid portfolio data containing numeric weights.

    Returns:
        Weight-valid portfolio data, unchanged except for implementation-level
        numeric typing needed by downstream canonicalization.

    Raises:
        ValueError: If any weight is non-finite or if weights do not sum to
            ``1.0`` within the documented tolerance.

    Invariants:
        This validator does not normalize weights and does not perform return,
        simulation, scenario, or risk calculations.
    """

    tolerance: float = 1e-9

    def validate(self, rows: Any) -> pd.DataFrame:
        """Validate the documented strict sum-to-one weight rule.

        Args:
            rows: Schema-valid portfolio data.

        Returns:
            Weight-valid portfolio data.

        Raises:
            ValueError: If weights are non-finite or fail the sum-to-one rule.
        """
        if not isinstance(rows, pd.DataFrame):
            raise ValueError("portfolio rows must be a pandas DataFrame")

        weights = [float(weight) for weight in rows["weight"]]
        if any(not math.isfinite(weight) for weight in weights):
            raise ValueError("weight values must be finite")

        total_weight = math.fsum(weights)
        if not math.isclose(total_weight, 1.0, abs_tol=self.tolerance, rel_tol=0.0):
            raise ValueError("portfolio weight sum must equal 1.0")

        return rows
