"""Scenario loss calculation contracts."""

from __future__ import annotations

import math


class ScenarioLossCalculator:
    """Convert deterministic scenario return into positive loss magnitude.

    Args:
        scenario_return: Weighted scenario portfolio return from
            ``ShockApplier``.

    Returns:
        Positive scenario loss magnitude using the project risk sign
        convention: ``max(-scenario_return, 0.0)``. A net scenario gain is
        reported as zero loss rather than a negative loss.

    Raises:
        ValueError: If the scenario return is not finite.

    Invariants:
        This calculator handles deterministic scenario impact only. It never
        assigns probabilities, reruns simulation, averages scenario outputs
        with stochastic results, or calculates VaR/ES.
    """

    def calculate(self, scenario_return: float) -> float:
        """Calculate positive deterministic scenario loss.

        Args:
            scenario_return: Weighted portfolio scenario return.

        Returns:
            Positive loss magnitude.

        Raises:
            ValueError: If input is invalid.
        """
        if not math.isfinite(scenario_return):
            raise ValueError("scenario_return must be finite")
        return max(-scenario_return, 0.0)
