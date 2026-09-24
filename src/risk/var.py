"""Value at Risk calculation contract."""

from __future__ import annotations

from typing import Any

from src.risk._common import empirical_var, terminal_losses


class VaRCalculator:
    """Calculate Value at Risk as a positive loss magnitude.

    Args:
        terminal_returns: Portfolio terminal returns from ``SimulationResult``.
            Negative returns are losses and positive returns are gains.
        confidence_level: External analysis confidence level. The calculator
            never chooses or mutates this value.

    Returns:
        VaR at ``confidence_level`` as a positive loss magnitude. The sign
        convention is explicit: losses are computed as ``-terminal_returns``;
        a 4.2% portfolio loss is returned as ``0.042``. The MVP empirical
        quantile convention is the smallest observed loss whose empirical CDF
        is at least ``confidence_level``.

    Raises:
        ValueError: If returns are empty, non-finite, or confidence level is
            outside ``(0, 1)``.

    Invariants:
        This component consumes simulation output only. It never runs
        simulation, generates stochastic draws, calculates Expected Shortfall,
        loss probability, drawdown metrics, scenarios, or diagnostics.
    """

    def calculate(self, terminal_returns: Any, *, confidence_level: float) -> float:
        """Calculate VaR from terminal returns.

        Args:
            terminal_returns: Simulated portfolio terminal returns.
            confidence_level: VaR confidence level supplied by configuration.

        Returns:
            Positive loss magnitude at the configured confidence level.

        Raises:
            ValueError: If inputs are invalid.
        """
        losses = terminal_losses(terminal_returns)
        return empirical_var(losses, confidence_level)
