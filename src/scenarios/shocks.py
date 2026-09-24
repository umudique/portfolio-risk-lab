"""Deterministic shock application contracts."""

from __future__ import annotations

from src.domain import Portfolio, ScenarioDefinition
from src.scenarios.validation import ScenarioValidator


class ShockApplier:
    """Apply deterministic asset shocks using portfolio weights.

    Args:
        portfolio: Canonical portfolio with tickers and weights.
        scenario: Validated scenario definition whose shocks reference portfolio
            tickers.

    Returns:
        Weighted scenario portfolio return calculated as
        ``sum(weight_i * shock_i)``. Missing tickers are treated as zero shock
        only after validation has confirmed every supplied shock ticker is
        allowed.

    Raises:
        ValueError: If portfolio and scenario inputs are inconsistent.

    Invariants:
        This component is deterministic, uses no random seed, makes no market
        data calls, imports no simulation module, and never calculates VaR,
        Expected Shortfall, stochastic loss probability, or diagnostics.
    """

    def apply(self, portfolio: Portfolio, scenario: ScenarioDefinition) -> float:
        """Calculate weighted deterministic scenario return.

        Args:
            portfolio: Canonical portfolio.
            scenario: Validated scenario definition.

        Returns:
            Weighted portfolio scenario return.

        Raises:
            ValueError: If inputs are inconsistent.
        """
        ScenarioValidator().validate(portfolio, scenario)
        shocks_by_ticker = {
            scenario_shock.ticker: scenario_shock.shock for scenario_shock in scenario.shocks
        }
        return float(
            sum(
                weight * shocks_by_ticker.get(ticker, 0.0)
                for ticker, weight in zip(portfolio.tickers, portfolio.weights, strict=True)
            )
        )
