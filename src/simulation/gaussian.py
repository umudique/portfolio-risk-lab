"""Gaussian simulation contract."""

from __future__ import annotations

import numpy as np

from src.domain import AnalysisConfig, Portfolio, ReturnEstimation, SimulationResult
from src.simulation._common import (
    covariance_array,
    terminal_compounded_returns,
    validate_config,
    validate_portfolio_estimation,
)
from src.simulation.aggregation import PortfolioPathAggregator
from src.simulation.base import SimulationModel
from src.simulation.random_state import RandomStateManager
from src.simulation.result import SimulationResultBuilder


class GaussianSimulator(SimulationModel):
    """Generate correlated portfolio return paths under Gaussian assumptions.

    Args:
        portfolio: Canonical portfolio whose weights align to
            ``estimation.tickers``.
        estimation: Canonical return estimation containing mean returns and the
            covariance matrix used for multivariate Gaussian draws.
        config: Analysis configuration providing horizon, simulation count, and
            explicit random seed.

    Returns:
        A ``SimulationResult`` with portfolio-level paths shaped
        ``(config.n_simulations, config.horizon)`` and terminal returns shaped
        ``(config.n_simulations,)``. Terminal returns use compounded path
        returns: ``prod(1 + r_t) - 1``.

    Raises:
        ValueError: If portfolio weights, estimation dimensions, horizon,
            simulation count, or covariance inputs are invalid.

    Invariants:
        This simulator implements the common ``SimulationModel`` contract. It
        generates simulated returns only; it never calculates VaR, Expected
        Shortfall, loss probabilities, drawdowns, scenarios, or diagnostics.
        All randomness must come from ``RandomStateManager``.
    """

    def simulate(
        self,
        portfolio: Portfolio,
        estimation: ReturnEstimation,
        config: AnalysisConfig,
    ) -> SimulationResult:
        """Simulate Gaussian portfolio returns.

        Args:
            portfolio: Canonical portfolio.
            estimation: Return and dependence estimates.
            config: Simulation configuration and seed.

        Returns:
            Canonical simulation result.

        Raises:
            ValueError: If simulation inputs are invalid.
        """
        validate_config(config)
        n_assets = validate_portfolio_estimation(portfolio, estimation)
        covariance = covariance_array(estimation, n_assets)
        rng = RandomStateManager().generator(config.random_seed)

        draws_by_step = rng.multivariate_normal(
            mean=np.zeros(n_assets, dtype=np.float64),
            cov=covariance,
            size=(config.horizon, config.n_simulations),
        )
        asset_paths = np.transpose(draws_by_step, axes=(1, 0, 2))
        portfolio_paths = PortfolioPathAggregator().aggregate(asset_paths, portfolio.weights)
        terminal_returns = terminal_compounded_returns(portfolio_paths)

        return SimulationResultBuilder().build(
            model_name="Gaussian",
            terminal_returns=terminal_returns,
            portfolio_paths=portfolio_paths,
            random_seed=config.random_seed,
            simulation_count=config.n_simulations,
            horizon=config.horizon,
            model_params={"assumption": "gaussian"},
        )
