"""Expected Shortfall calculation contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.risk._common import terminal_losses, validate_confidence_level


class ExpectedShortfallCalculator:
    """Calculate Expected Shortfall using the project loss sign convention.

    Args:
        terminal_returns: Portfolio terminal returns from ``SimulationResult``.
        confidence_level: External analysis confidence level used to determine
            the VaR threshold.

    Returns:
        Expected Shortfall ``ES_alpha = E[Loss | Loss >= VaR_alpha]`` as a
        positive loss magnitude, where ``Loss = -terminal_return``. The VaR
        threshold must use the same empirical quantile and positive-loss
        convention as ``VaRCalculator``.

    Raises:
        ValueError: If returns are empty, non-finite, or confidence level is
            outside ``(0, 1)``.

    Invariants:
        This component consumes terminal returns only. It never reruns
        simulation, chooses its own confidence level, calculates VaR for
        reporting separately from the threshold, or generates diagnostics.
    """

    def calculate(self, terminal_returns: Any, *, confidence_level: float) -> float:
        """Calculate Expected Shortfall from terminal returns.

        Args:
            terminal_returns: Simulated portfolio terminal returns.
            confidence_level: ES confidence level supplied by configuration.

        Returns:
            Positive loss magnitude conditional on losses meeting or exceeding
            the VaR threshold.

        Raises:
            ValueError: If inputs are invalid.
        """
        validate_confidence_level(confidence_level)
        losses = terminal_losses(terminal_returns)
        threshold = float(np.quantile(losses, confidence_level, method="lower"))
        tail_losses = losses[losses >= threshold]
        if tail_losses.size == 0:
            raise ValueError("no losses meet or exceed the VaR threshold")
        return float(np.mean(tail_losses))
