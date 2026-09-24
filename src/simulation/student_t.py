"""Student-t simulation contract."""

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


class StudentTSimulator(SimulationModel):
    """Generate correlated portfolio return paths under Student-t assumptions.

    Args:
        portfolio: Canonical portfolio whose weights align to
            ``estimation.tickers``.
        estimation: Canonical return estimation containing location and
            dependence inputs.
        config: Analysis configuration providing horizon, simulation count,
            explicit random seed, and ``student_t_degrees_of_freedom``.

    Returns:
        A ``SimulationResult`` with portfolio-level paths shaped
        ``(config.n_simulations, config.horizon)`` and terminal returns shaped
        ``(config.n_simulations,)``. Terminal returns use the same compounded
        convention as Gaussian: ``prod(1 + r_t) - 1``. ``model_params`` must
        record ``degrees_of_freedom`` so downstream diagnostics can report the
        fat-tail assumption.

    Raises:
        ValueError: If dimensions are invalid or degrees of freedom are not
            valid for the documented Student-t model.

    Invariants:
        This simulator implements the common ``SimulationModel`` contract and
        uses the same portfolio, horizon, simulation count, and seed conventions
        as the Gaussian and bootstrap models. It never calculates risk metrics
        or model-risk findings. All randomness must come from
        ``RandomStateManager``.
    """

    def simulate(
        self,
        portfolio: Portfolio,
        estimation: ReturnEstimation,
        config: AnalysisConfig,
    ) -> SimulationResult:
        """Simulate Student-t portfolio returns.

        Args:
            portfolio: Canonical portfolio.
            estimation: Return and dependence estimates.
            config: Simulation configuration and seed.

        Returns:
            Canonical simulation result with Student-t model parameters.

        Raises:
            ValueError: If simulation inputs are invalid.
        """
        validate_config(config)
        degrees_of_freedom = float(config.student_t_degrees_of_freedom)
        if degrees_of_freedom <= 2.0:
            raise ValueError("student_t_degrees_of_freedom must be greater than 2")

        n_assets = validate_portfolio_estimation(portfolio, estimation)
        covariance = covariance_array(estimation, n_assets)
        rng = RandomStateManager().generator(config.random_seed)

        cholesky = np.linalg.cholesky(covariance)
        normal_draws = (
            rng.normal(size=(config.n_simulations, config.horizon, n_assets)) @ cholesky.T
        )
        chi_square_draws = rng.chisquare(
            df=degrees_of_freedom,
            size=(config.n_simulations, config.horizon, 1),
        )
        asset_paths = normal_draws / np.sqrt(chi_square_draws / degrees_of_freedom)
        portfolio_paths = PortfolioPathAggregator().aggregate(asset_paths, portfolio.weights)
        terminal_returns = terminal_compounded_returns(portfolio_paths)

        return SimulationResultBuilder().build(
            model_name="Student-t",
            terminal_returns=terminal_returns,
            portfolio_paths=portfolio_paths,
            random_seed=config.random_seed,
            simulation_count=config.n_simulations,
            horizon=config.horizon,
            model_params={"degrees_of_freedom": degrees_of_freedom},
        )
