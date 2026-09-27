from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from src.domain import ReturnEstimation
from src.returns import (
    CorrelationEstimator,
    CovarianceEstimator,
    EstimationResultBuilder,
    MeanReturnEstimator,
    ReturnCalculator,
    VolatilityEstimator,
)


def test_log_return_calculation_matches_known_price_fixture() -> None:
    prices = pd.DataFrame({"AAPL": [100.0, 110.0, 99.0]})
    expected = np.array([math.log(110.0 / 100.0), math.log(99.0 / 110.0)])

    returns = ReturnCalculator().calculate(prices, method="log")

    np.testing.assert_array_almost_equal(returns["AAPL"].to_numpy(), expected, decimal=6)


def test_simple_return_calculation_is_explicit_alternative() -> None:
    prices = pd.DataFrame({"AAPL": [100.0, 110.0, 99.0]})
    expected = np.array([0.10, -0.10])

    returns = ReturnCalculator().calculate(prices, method="simple")

    np.testing.assert_array_almost_equal(returns["AAPL"].to_numpy(), expected, decimal=6)


def test_volatility_estimate_matches_known_annualized_fixture() -> None:
    returns = pd.DataFrame({"AAPL": [0.01, 0.02, 0.03]})
    expected = 0.01 * math.sqrt(252.0)

    volatilities = VolatilityEstimator().estimate(returns, annualization_factor=252)

    assert volatilities["AAPL"] == pytest.approx(expected, abs=1e-12)


@st.composite
def finite_return_frames(draw: st.DrawFn) -> pd.DataFrame:
    rows = draw(st.integers(min_value=3, max_value=12))
    values = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False),
            ),
            min_size=rows,
            max_size=rows,
        )
    )
    return pd.DataFrame(values, columns=["AAPL", "MSFT", "SPY"])


@given(finite_return_frames())
@seed(20260924)
@settings(max_examples=25, deadline=None)
def test_covariance_matrix_is_symmetric(returns: pd.DataFrame) -> None:
    covariance = CovarianceEstimator().estimate(returns)
    values = covariance.to_numpy()

    np.testing.assert_allclose(values, values.T, atol=1e-12)


@given(finite_return_frames())
@seed(20260924)
@settings(max_examples=25, deadline=None)
def test_covariance_matrix_is_positive_semidefinite(returns: pd.DataFrame) -> None:
    covariance = CovarianceEstimator().estimate(returns)
    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())

    assert np.all(eigenvalues >= -1e-12)


def test_correlation_matrix_diagonal_is_exactly_one() -> None:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, -0.01, 0.03],
            "MSFT": [0.02, -0.01, 0.04, 0.01],
            "SPY": [-0.01, 0.01, 0.02, 0.03],
        }
    )

    correlation = CorrelationEstimator().estimate(returns)

    assert tuple(np.diag(correlation.to_numpy())) == (1.0, 1.0, 1.0)


def test_correlation_matrix_values_are_within_unit_bounds() -> None:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, -0.01, 0.03],
            "MSFT": [0.02, -0.01, 0.04, 0.01],
            "SPY": [-0.01, 0.01, 0.02, 0.03],
        }
    )

    correlation = CorrelationEstimator().estimate(returns)
    values = correlation.to_numpy()

    assert np.all(values >= -1.0)
    assert np.all(values <= 1.0)


def test_mean_return_shape_matches_number_of_assets() -> None:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.03],
            "MSFT": [0.02, 0.01, 0.00],
            "SPY": [0.00, 0.01, 0.02],
        }
    )

    mean_returns = MeanReturnEstimator().estimate(returns, method="historical")

    assert mean_returns.shape == (3,)
    assert tuple(mean_returns.index) == ("AAPL", "MSFT", "SPY")


def test_estimation_result_builder_populates_required_fields_and_observation_count() -> None:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.03],
            "MSFT": [0.02, 0.01, 0.00],
        }
    )
    mean_returns = pd.Series({"AAPL": 0.02, "MSFT": 0.01})
    volatilities = pd.Series({"AAPL": 0.01, "MSFT": 0.01})
    covariance_matrix = pd.DataFrame(
        [[0.0001, 0.0], [0.0, 0.0001]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    estimation = EstimationResultBuilder().build(
        returns=returns,
        mean_returns=mean_returns,
        volatilities=volatilities,
        covariance_matrix=covariance_matrix,
        correlation_matrix=correlation_matrix,
        return_convention="log",
        covariance_method="sample_ddof_1",
    )

    assert isinstance(estimation, ReturnEstimation)
    assert estimation.tickers == ("AAPL", "MSFT")
    assert estimation.returns is returns
    assert estimation.mean_returns is mean_returns
    assert estimation.volatilities is volatilities
    assert estimation.covariance_matrix is covariance_matrix
    assert estimation.correlation_matrix is correlation_matrix
    assert estimation.observation_count == 3


def test_estimation_window_records_return_convention_and_is_not_empty() -> None:
    returns = pd.DataFrame({"AAPL": [0.01, 0.02], "MSFT": [0.02, 0.01]})

    estimation = EstimationResultBuilder().build(
        returns=returns,
        mean_returns=pd.Series({"AAPL": 0.015, "MSFT": 0.015}),
        volatilities=pd.Series({"AAPL": 0.01, "MSFT": 0.01}),
        covariance_matrix=pd.DataFrame(
            [[0.0001, 0.0], [0.0, 0.0001]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"],
        ),
        correlation_matrix=pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"],
        ),
        return_convention="log",
        covariance_method="sample_ddof_1",
    )

    assert estimation.estimation_window
    assert "log" in estimation.estimation_window
