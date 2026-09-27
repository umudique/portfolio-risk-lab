"""Application adapter for Portfolio Risk Lab analysis execution."""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from src.domain import AnalysisConfig, DiagnosticReport, Portfolio
from src.pipeline.analysis import AnalysisPipeline
from src.portfolio.canonicalizer import PortfolioCanonicalizer
from src.portfolio.validation import PortfolioSchemaValidator, WeightValidator

_DEFAULT_LOOKBACK_DAYS = 756


class ConcreteAnalysisOrchestrator:
    """Thin app adapter for the AnalysisOrchestrator protocol.

    Runs app/infrastructure work only: portfolio CSV parsing, market-price
    fetching, and delegation to ``src.pipeline.AnalysisPipeline``.
    """

    def run(self, portfolio_csv: Any, config: AnalysisConfig) -> DiagnosticReport:
        """Execute the analysis request through the application pipeline.

        Args:
            portfolio_csv: Streamlit UploadedFile or any object with
                ``.getvalue()`` returning portfolio CSV bytes.
            config: Canonical analysis configuration from the dashboard form.

        Returns:
            Fully assembled diagnostic report from ``AnalysisPipeline``.

        Raises:
            ValueError: If portfolio parsing, price fetching, or pipeline
                execution rejects the request.
        """
        portfolio = _load_portfolio(portfolio_csv)
        prices = _fetch_prices(portfolio.tickers, config.estimation_window)
        return AnalysisPipeline().run(portfolio, prices, config)


def _load_portfolio(portfolio_csv: Any) -> Portfolio:
    content = portfolio_csv.getvalue().decode("utf-8")
    rows = pd.read_csv(io.StringIO(content))
    schema_valid = PortfolioSchemaValidator().validate(rows)
    weight_valid = WeightValidator().validate(schema_valid)
    return PortfolioCanonicalizer().canonicalize(weight_valid)


def _fetch_prices(tickers: tuple[str, ...], estimation_window: int | None) -> pd.DataFrame:
    lookback = estimation_window if estimation_window is not None else _DEFAULT_LOOKBACK_DAYS
    end = datetime.today()
    start = end - timedelta(days=lookback)
    data = yf.download(
        list(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = list(tickers)
    prices.columns = list(tickers)
    prices = prices.dropna()
    if prices.empty or len(prices) < 2:
        raise ValueError("insufficient price history fetched for the requested tickers")
    return prices  # type: ignore[no-any-return]
