from __future__ import annotations

from pathlib import Path

import pytest

from src.domain import Portfolio
from src.portfolio import PortfolioLoader


def write_portfolio_csv(tmp_path: Path, content: str, filename: str = "portfolio.csv") -> Path:
    csv_path = tmp_path / filename
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_valid_csv_loads_portfolio_with_expected_tickers_and_weights(tmp_path: Path) -> None:
    csv_path = write_portfolio_csv(
        tmp_path,
        "ticker,weight\nAAPL,0.30\nMSFT,0.25\nSPY,0.25\nTLT,0.20\n",
    )

    portfolio = PortfolioLoader().load(csv_path)

    assert portfolio == Portfolio(
        tickers=("AAPL", "MSFT", "SPY", "TLT"),
        weights=(0.30, 0.25, 0.25, 0.20),
    )


def test_missing_ticker_column_is_rejected(tmp_path: Path) -> None:
    csv_path = write_portfolio_csv(tmp_path, "symbol,weight\nAAPL,1.0\n")

    with pytest.raises(ValueError, match="ticker"):
        PortfolioLoader().load(csv_path)


def test_missing_weight_column_is_rejected(tmp_path: Path) -> None:
    csv_path = write_portfolio_csv(tmp_path, "ticker,allocation\nAAPL,1.0\n")

    with pytest.raises(ValueError, match="weight"):
        PortfolioLoader().load(csv_path)


def test_non_numeric_weight_is_rejected(tmp_path: Path) -> None:
    csv_path = write_portfolio_csv(tmp_path, "ticker,weight\nAAPL,not-a-number\n")

    with pytest.raises(ValueError, match="weight"):
        PortfolioLoader().load(csv_path)


def test_duplicate_ticker_is_rejected(tmp_path: Path) -> None:
    csv_path = write_portfolio_csv(tmp_path, "ticker,weight\nAAPL,0.5\nAAPL,0.5\n")

    with pytest.raises(ValueError, match="duplicate|ticker"):
        PortfolioLoader().load(csv_path)


def test_weights_that_sum_to_one_are_accepted(tmp_path: Path) -> None:
    csv_path = write_portfolio_csv(tmp_path, "ticker,weight\nAAPL,0.4\nMSFT,0.6\n")

    portfolio = PortfolioLoader().load(csv_path)

    assert portfolio.weights == (0.4, 0.6)


def test_weights_that_do_not_sum_to_one_are_rejected(tmp_path: Path) -> None:
    csv_path = write_portfolio_csv(tmp_path, "ticker,weight\nAAPL,0.4\nMSFT,0.4\n")

    with pytest.raises(ValueError, match="sum|1.0|weight"):
        PortfolioLoader().load(csv_path)


def test_empty_csv_with_header_only_is_rejected(tmp_path: Path) -> None:
    csv_path = write_portfolio_csv(tmp_path, "ticker,weight\n")

    with pytest.raises(ValueError, match="empty|row"):
        PortfolioLoader().load(csv_path)


def test_single_asset_portfolio_is_accepted(tmp_path: Path) -> None:
    csv_path = write_portfolio_csv(tmp_path, "ticker,weight\nSPY,1.0\n")

    portfolio = PortfolioLoader().load(csv_path)

    assert portfolio == Portfolio(tickers=("SPY",), weights=(1.0,))


def test_canonical_ticker_ordering_is_deterministic_for_shuffled_rows(tmp_path: Path) -> None:
    first_path = write_portfolio_csv(
        tmp_path,
        "ticker,weight\nMSFT,0.25\nTLT,0.20\nAAPL,0.30\nSPY,0.25\n",
        filename="first.csv",
    )
    second_path = write_portfolio_csv(
        tmp_path,
        "ticker,weight\nSPY,0.25\nAAPL,0.30\nTLT,0.20\nMSFT,0.25\n",
        filename="second.csv",
    )

    first = PortfolioLoader().load(first_path)
    second = PortfolioLoader().load(second_path)

    assert first == second
    assert first.tickers == ("AAPL", "MSFT", "SPY", "TLT")
