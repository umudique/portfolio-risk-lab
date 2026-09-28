"""Mean and worst simulated loss calculation."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.risk._common import terminal_losses


class LossStatisticsCalculator:
    """Calculate mean and worst simulated loss from terminal returns.

    Args:
        terminal_returns: Portfolio terminal returns from ``SimulationResult``.

    Returns:
        A tuple ``(mean_simulated_loss, worst_simulated_loss)`` where both are
        positive loss magnitudes. ``mean_simulated_loss`` is the average of
        losses that are strictly positive; ``worst_simulated_loss`` is the
        largest observed loss floored at zero.

    Raises:
        ValueError: If terminal returns are empty or non-finite.

    Invariants:
        Consumes simulation output only. Never reruns simulation, calculates
        VaR/ES, or imports from app or scenarios modules.
    """

    def calculate(self, terminal_returns: Any) -> tuple[float, float]:
        """Calculate mean and worst simulated loss.

        Args:
            terminal_returns: Simulated portfolio terminal returns.

        Returns:
            Tuple of ``(mean_simulated_loss, worst_simulated_loss)``.

        Raises:
            ValueError: If inputs are empty or non-finite.
        """
        losses = terminal_losses(terminal_returns)
        positive_losses = losses[losses > 0]
        mean_loss = float(np.mean(positive_losses)) if len(positive_losses) > 0 else 0.0
        worst_loss = max(float(np.max(losses)), 0.0)
        return mean_loss, worst_loss
