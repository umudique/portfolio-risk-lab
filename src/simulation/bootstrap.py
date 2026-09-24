"""Historical bootstrap simulation contract."""

from __future__ import annotations

from src.domain import AnalysisConfig, Portfolio, ReturnEstimation, SimulationResult
from src.simulation._common import (
    historical_returns_array,
    terminal_compounded_returns,
    validate_config,
    validate_portfolio_estimation,
)
from src.simulation.aggregation import PortfolioPathAggregator
from src.simulation.base import SimulationModel
from src.simulation.random_state import RandomStateManager
from src.simulation.result import SimulationResultBuilder


class HistoricalBootstrapSimulator(SimulationModel):
    """Generate portfolio return paths by resampling historical joint vectors.

    Args:
        portfolio: Canonical portfolio whose weights align to
            ``estimation.tickers``.
        estimation: Canonical return estimation containing the empirical
            historical return matrix used as the bootstrap population.
        config: Analysis configuration providing horizon, simulation count, and
            explicit random seed.

    Returns:
        A ``SimulationResult`` with reproducible portfolio-level paths shaped
        ``(config.n_simulations, config.horizon)`` and terminal returns using
        ``prod(1 + r_t) - 1``. For bootstrap integrity checks, every sampled
        joint asset-return vector exposed in ``model_params["asset_paths"]``
        must be exactly one complete row from ``estimation.returns``.

    Raises:
        ValueError: If historical returns are missing, dimensions are invalid,
            or the bootstrap configuration cannot preserve complete joint
            return vectors.

    Invariants:
        The bootstrap resampling unit is the full same-date multi-asset return
        vector. Per-asset independent resampling is prohibited because it
        destroys observed dependence. This simulator never calculates VaR,
        Expected Shortfall, loss probabilities, drawdowns, or diagnostics. All
        randomness must come from ``RandomStateManager``.
    """

    def simulate(
        self,
        portfolio: Portfolio,
        estimation: ReturnEstimation,
        config: AnalysisConfig,
    ) -> SimulationResult:
        """Simulate historical-bootstrap portfolio returns.

        Args:
            portfolio: Canonical portfolio.
            estimation: Historical return observations and dependence estimates.
            config: Simulation configuration and seed.

        Returns:
            Canonical simulation result.

        Raises:
            ValueError: If simulation inputs are invalid.
        """
        validate_config(config)
        n_assets = validate_portfolio_estimation(portfolio, estimation)
        historical_returns = historical_returns_array(estimation, n_assets)
        rng = RandomStateManager().generator(config.random_seed)

        sample_indices = rng.integers(
            low=0,
            high=historical_returns.shape[0],
            size=(config.n_simulations, config.horizon),
        )
        asset_paths = historical_returns[sample_indices]
        portfolio_paths = PortfolioPathAggregator().aggregate(asset_paths, portfolio.weights)
        terminal_returns = terminal_compounded_returns(portfolio_paths)

        return SimulationResultBuilder().build(
            model_name="Historical Bootstrap",
            terminal_returns=terminal_returns,
            portfolio_paths=portfolio_paths,
            random_seed=config.random_seed,
            simulation_count=config.n_simulations,
            horizon=config.horizon,
            model_params={
                "bootstrap_method": "joint_vector_resampling",
                "asset_paths": asset_paths,
            },
        )
