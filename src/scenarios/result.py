"""Scenario result building contract."""

from __future__ import annotations

import math

from src.domain import ScenarioDefinition, ScenarioResult


class ScenarioResultBuilder:
    """Build the canonical ``ScenarioResult`` domain object.

    Args:
        scenario: Validated scenario definition.
        portfolio_loss: Positive deterministic scenario loss magnitude.

    Returns:
        A canonical :class:`src.domain.ScenarioResult` containing the scenario
        name, all explicit asset shocks as ``dict[str, float]``, and the
        positive portfolio loss magnitude.

    Raises:
        ValueError: If the scenario name is empty, shocks are invalid, or loss
            is negative/non-finite.

    Invariants:
        This builder assembles deterministic scenario output only. Scenario
        results are never averaged with or merged into probabilistic simulation
        results and never carry probabilities in the MVP.
    """

    def build(
        self,
        *,
        scenario: ScenarioDefinition,
        portfolio_loss: float,
    ) -> ScenarioResult:
        """Build a canonical scenario result.

        Args:
            scenario: Validated scenario definition.
            portfolio_loss: Positive scenario loss magnitude.

        Returns:
            Canonical immutable ``ScenarioResult``.

        Raises:
            ValueError: If fields are invalid.
        """
        if not scenario.name:
            raise ValueError("scenario name must not be empty")
        if len(scenario.shocks) == 0:
            raise ValueError("scenario must contain at least one shock")
        if not math.isfinite(portfolio_loss) or portfolio_loss < 0.0:
            raise ValueError("portfolio_loss must be a finite non-negative loss magnitude")

        asset_shocks: dict[str, float] = {}
        for scenario_shock in scenario.shocks:
            if scenario_shock.ticker in asset_shocks:
                raise ValueError(f"duplicate shock ticker: {scenario_shock.ticker}")
            if not math.isfinite(scenario_shock.shock):
                raise ValueError("shock values must be finite")
            asset_shocks[scenario_shock.ticker] = scenario_shock.shock

        return ScenarioResult(
            scenario_name=scenario.name,
            asset_shocks=asset_shocks,
            portfolio_loss=portfolio_loss,
        )
