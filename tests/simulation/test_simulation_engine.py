from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.domain import AnalysisConfig, Portfolio, ReturnEstimation, SimulationResult
from src.simulation import (
    GaussianSimulator,
    HistoricalBootstrapSimulator,
    PortfolioPathAggregator,
    RandomStateManager,
    SimulationResultBuilder,
    StudentTSimulator,
)


def make_portfolio() -> Portfolio:
    return Portfolio(tickers=("AAPL", "MSFT"), weights=(0.6, 0.4))


def make_estimation() -> ReturnEstimation:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.03, -0.01],
            "MSFT": [0.02, -0.01, 0.01, -0.03],
        }
    )
    covariance = pd.DataFrame(
        [[0.0004, 0.0001], [0.0001, 0.0009]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    correlation = pd.DataFrame(
        [[1.0, 0.1666666667], [0.1666666667, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    return ReturnEstimation(
        tickers=("AAPL", "MSFT"),
        returns=returns,
        mean_returns=pd.Series({"AAPL": 0.0, "MSFT": 0.0}),
        volatilities=pd.Series({"AAPL": 0.02, "MSFT": 0.03}),
        covariance_matrix=covariance,
        correlation_matrix=correlation,
        observation_count=len(returns),
        estimation_window="log, sample_ddof_1",
    )


def make_config(
    *,
    n_simulations: int = 8,
    horizon: int = 3,
    random_seed: int = 1234,
    student_t_degrees_of_freedom: float = 5.0,
) -> AnalysisConfig:
    return AnalysisConfig(
        confidence_level=0.95,
        horizon=horizon,
        n_simulations=n_simulations,
        loss_threshold=0.10,
        random_seed=random_seed,
        student_t_degrees_of_freedom=student_t_degrees_of_freedom,
    )


def empirical_kurtosis(values: np.ndarray) -> float:
    centered = values - values.mean()
    variance = np.mean(centered**2)
    return float(np.mean(centered**4) / variance**2)


def test_gaussian_simulator_output_terminal_return_shape() -> None:
    result = GaussianSimulator().simulate(make_portfolio(), make_estimation(), make_config())

    assert isinstance(result, SimulationResult)
    assert np.asarray(result.terminal_returns).shape == (8,)


def test_gaussian_simulator_reproducibility_with_same_seed() -> None:
    first = GaussianSimulator().simulate(make_portfolio(), make_estimation(), make_config())
    second = GaussianSimulator().simulate(make_portfolio(), make_estimation(), make_config())

    np.testing.assert_array_equal(first.terminal_returns, second.terminal_returns)


@pytest.mark.slow
def test_gaussian_simulator_zero_mean_recovery_on_large_sample() -> None:
    config = make_config(n_simulations=100_000, horizon=1, random_seed=77)

    result = GaussianSimulator().simulate(make_portfolio(), make_estimation(), config)

    assert float(np.mean(result.terminal_returns)) == pytest.approx(0.0, abs=0.01)


def test_student_t_simulator_output_shape_and_reproducibility() -> None:
    config = make_config(random_seed=222)

    first = StudentTSimulator().simulate(make_portfolio(), make_estimation(), config)
    second = StudentTSimulator().simulate(make_portfolio(), make_estimation(), config)

    assert np.asarray(first.terminal_returns).shape == (8,)
    np.testing.assert_array_equal(first.terminal_returns, second.terminal_returns)


@pytest.mark.slow
def test_student_t_simulator_has_heavier_tails_than_gaussian() -> None:
    config = make_config(n_simulations=50_000, horizon=1, random_seed=909)

    gaussian = GaussianSimulator().simulate(make_portfolio(), make_estimation(), config)
    student_t = StudentTSimulator().simulate(make_portfolio(), make_estimation(), config)

    assert empirical_kurtosis(np.asarray(student_t.terminal_returns)) > empirical_kurtosis(
        np.asarray(gaussian.terminal_returns)
    )


def test_historical_bootstrap_output_shape_and_reproducibility() -> None:
    config = make_config(random_seed=333)

    first = HistoricalBootstrapSimulator().simulate(make_portfolio(), make_estimation(), config)
    second = HistoricalBootstrapSimulator().simulate(make_portfolio(), make_estimation(), config)

    assert np.asarray(first.terminal_returns).shape == (8,)
    np.testing.assert_array_equal(first.terminal_returns, second.terminal_returns)


def test_historical_bootstrap_resamples_only_complete_historical_vectors() -> None:
    estimation = make_estimation()
    result = HistoricalBootstrapSimulator().simulate(make_portfolio(), estimation, make_config())
    historical_vectors = {tuple(row) for row in estimation.returns.to_numpy()}

    sampled_paths = np.asarray(result.model_params["asset_paths"])
    sampled_vectors = sampled_paths.reshape(-1, sampled_paths.shape[-1])

    assert all(tuple(vector) in historical_vectors for vector in sampled_vectors)


def test_all_simulation_models_accept_the_common_signature() -> None:
    expected = ("self", "portfolio", "estimation", "config")

    assert tuple(inspect.signature(GaussianSimulator.simulate).parameters) == expected
    assert tuple(inspect.signature(StudentTSimulator.simulate).parameters) == expected
    assert tuple(inspect.signature(HistoricalBootstrapSimulator.simulate).parameters) == expected


def test_portfolio_path_aggregator_uses_weighted_sum_per_step() -> None:
    asset_return_paths = np.array(
        [
            [[0.10, 0.00], [0.00, 0.20]],
            [[-0.10, 0.10], [0.05, -0.05]],
        ]
    )
    weights = (0.6, 0.4)

    portfolio_paths = PortfolioPathAggregator().aggregate(asset_return_paths, weights)

    expected = np.array([[0.06, 0.08], [-0.02, 0.01]])
    np.testing.assert_allclose(portfolio_paths, expected)


def test_student_t_result_records_degrees_of_freedom() -> None:
    config = make_config(student_t_degrees_of_freedom=7.0)

    result = StudentTSimulator().simulate(make_portfolio(), make_estimation(), config)

    assert result.model_params["degrees_of_freedom"] == 7.0


def test_simulation_result_random_seed_matches_analysis_config() -> None:
    config = make_config(random_seed=9876)

    result = GaussianSimulator().simulate(make_portfolio(), make_estimation(), config)

    assert result.random_seed == config.random_seed


def test_random_state_manager_returns_reproducible_numpy_generator() -> None:
    first = RandomStateManager().generator(42)
    second = RandomStateManager().generator(42)

    assert isinstance(first, np.random.Generator)
    np.testing.assert_array_equal(first.normal(size=5), second.normal(size=5))


def test_simulation_result_builder_populates_canonical_result() -> None:
    terminal_returns = np.array([0.01, -0.02, 0.03])
    portfolio_paths = np.array([[0.01], [-0.02], [0.03]])

    result = SimulationResultBuilder().build(
        model_name="Gaussian",
        terminal_returns=terminal_returns,
        portfolio_paths=portfolio_paths,
        random_seed=123,
        simulation_count=3,
        horizon=1,
        model_params={"assumption": "gaussian"},
    )

    assert result == SimulationResult(
        model_name="Gaussian",
        terminal_returns=terminal_returns,
        portfolio_paths=portfolio_paths,
        random_seed=123,
        simulation_count=3,
        horizon=1,
        model_params={"assumption": "gaussian"},
    )
