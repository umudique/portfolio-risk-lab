"""Loss probability calculation contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.risk._common import terminal_losses


class LossProbabilityCalculator:
    """Calculate the probability that simulated loss exceeds a threshold.

    Args:
        terminal_returns: Portfolio terminal returns from ``SimulationResult``.
        loss_threshold: External threshold from analysis configuration.

    Returns:
        The empirical fraction ``P(Loss > loss_threshold)`` where
        ``Loss = -terminal_return`` and losses are positive magnitudes.

    Raises:
        ValueError: If returns are empty, non-finite, or the threshold is
            negative.

    Invariants:
        This calculator consumes simulation output only. It does not choose its
        own threshold, rerun simulation, assign scenario probabilities, or
        calculate VaR/ES/drawdown metrics.
    """

    def calculate(self, terminal_returns: Any, *, loss_threshold: float) -> float:
        """Calculate empirical probability of loss above threshold.

        Args:
            terminal_returns: Simulated portfolio terminal returns.
            loss_threshold: Positive loss threshold supplied by configuration.

        Returns:
            Fraction of losses strictly greater than the threshold.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not np.isfinite(loss_threshold) or loss_threshold < 0.0:
            raise ValueError("loss_threshold must be non-negative")

        losses = terminal_losses(terminal_returns)
        return float(np.mean(losses > loss_threshold))
