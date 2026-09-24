"""Scenario validation contracts."""

from __future__ import annotations

import math

from src.domain import Portfolio, ScenarioDefinition


class ScenarioValidator:
    """Validate deterministic scenario definitions against a portfolio.

    Args:
        portfolio: Canonical portfolio whose tickers define the allowed shock
            universe.
        scenario: Scenario definition containing explicit asset-level shocks.

    Returns:
        The validated scenario definition, unchanged.

    Raises:
        ValueError: If a shock ticker is not in the portfolio, if duplicate
            shock tickers are supplied, or if any shock magnitude is outside the
            documented MVP bound ``[-1.0, +5.0]``.

    Invariants:
        Validation never silently corrects inputs. Scenarios are deterministic
        and require no random seed. This component never imports simulation
        code, calls market data providers, assigns probabilities, or merges
        scenario results with stochastic outputs.
    """

    min_shock: float = -1.0
    max_shock: float = 5.0

    def validate(
        self,
        portfolio: Portfolio,
        scenario: ScenarioDefinition,
    ) -> ScenarioDefinition:
        """Validate a scenario against portfolio tickers and shock bounds.

        Args:
            portfolio: Canonical portfolio.
            scenario: Scenario definition to validate.

        Returns:
            The validated scenario definition.

        Raises:
            ValueError: If any validation rule is violated.
        """
        if not scenario.name:
            raise ValueError("scenario name must not be empty")
        if len(scenario.shocks) == 0:
            raise ValueError("scenario must contain at least one shock")

        portfolio_tickers = set(portfolio.tickers)
        seen_tickers: set[str] = set()
        for scenario_shock in scenario.shocks:
            if scenario_shock.ticker not in portfolio_tickers:
                raise ValueError(f"shock ticker {scenario_shock.ticker} is not in portfolio")
            if scenario_shock.ticker in seen_tickers:
                raise ValueError(f"duplicate shock ticker: {scenario_shock.ticker}")
            if not math.isfinite(scenario_shock.shock):
                raise ValueError("shock values must be finite")
            if not self.min_shock <= scenario_shock.shock <= self.max_shock:
                raise ValueError(
                    f"shock must be within documented bound [{self.min_shock}, {self.max_shock}]"
                )
            seen_tickers.add(scenario_shock.ticker)

        return scenario
