"""Drawdown calculation contracts."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.risk._common import finite_array


class DrawdownCalculator:
    """Calculate drawdown paths from simulated portfolio return paths.

    Args:
        portfolio_paths: Portfolio-level simulated returns from
            ``SimulationResult.portfolio_paths`` with shape
            ``(n_simulations, horizon)`` or a single one-dimensional path.

    Returns:
        Drawdown magnitudes with the same path shape. Wealth starts at ``1.0``;
        cumulative wealth is ``cumprod(1 + r_t)``; drawdown at each step is
        ``(running_peak - wealth_t) / running_peak`` and is therefore a
        non-negative magnitude.

    Raises:
        ValueError: If paths are missing, empty, non-finite, or would produce
            non-positive wealth.

    Invariants:
        Drawdown calculation operates only on provided simulated portfolio
        paths. It never reruns simulation and never calculates VaR, Expected
        Shortfall, loss probability, scenarios, or diagnostics.
    """

    def calculate(self, portfolio_paths: Any) -> Any:
        """Calculate drawdown path magnitudes.

        Args:
            portfolio_paths: Simulated portfolio return paths.

        Returns:
            Drawdown path magnitudes.

        Raises:
            ValueError: If inputs are invalid.
        """
        paths = finite_array(portfolio_paths, name="portfolio_paths")
        was_one_dimensional = paths.ndim == 1
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        if paths.ndim != 2:
            raise ValueError("portfolio_paths must be one- or two-dimensional")

        wealth = np.cumprod(1.0 + paths, axis=1)
        if np.any(wealth <= 0.0):
            raise ValueError("portfolio paths must produce positive wealth")

        running_peak = np.maximum.accumulate(wealth, axis=1)
        drawdowns = (running_peak - wealth) / running_peak
        if was_one_dimensional:
            return drawdowns[0]
        return drawdowns


class DrawdownDistributionAnalyzer:
    """Summarize maximum drawdown distribution from drawdown paths.

    Args:
        drawdown_paths: Drawdown magnitudes produced by ``DrawdownCalculator``.

    Returns:
        Summary values needed by ``RiskResult``: mean maximum drawdown and the
        95th percentile of maximum drawdowns.

    Raises:
        ValueError: If drawdown paths are empty or non-finite.

    Invariants:
        This analyzer summarizes already-computed drawdown paths. It does not
        rerun simulation or compute VaR, ES, loss probabilities, scenarios, or
        diagnostics.
    """

    def summarize(self, drawdown_paths: Any) -> tuple[float, float]:
        """Summarize mean and 95th percentile maximum drawdown.

        Args:
            drawdown_paths: Drawdown magnitudes by simulation path.

        Returns:
            ``(max_drawdown_mean, max_drawdown_q95)``.

        Raises:
            ValueError: If inputs are invalid.
        """
        drawdowns = finite_array(drawdown_paths, name="drawdown_paths")
        if drawdowns.ndim == 1:
            drawdowns = drawdowns.reshape(1, -1)
        if drawdowns.ndim != 2:
            raise ValueError("drawdown_paths must be one- or two-dimensional")
        if np.any(drawdowns < 0.0):
            raise ValueError("drawdown_paths must contain non-negative magnitudes")

        max_drawdowns = np.max(drawdowns, axis=1)
        return float(np.mean(max_drawdowns)), float(np.quantile(max_drawdowns, 0.95))
