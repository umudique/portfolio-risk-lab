from __future__ import annotations

import numpy as np
import pytest

from src.domain import RiskResult
from src.risk import (
    DrawdownCalculator,
    DrawdownDistributionAnalyzer,
    ExpectedShortfallCalculator,
    LossProbabilityCalculator,
    RiskResultBuilder,
    VaRCalculator,
)


def test_var_calculator_uses_known_empirical_loss_quantile() -> None:
    losses = np.arange(1000, dtype=float) / 1000.0
    terminal_returns = -losses

    var = VaRCalculator().calculate(terminal_returns, confidence_level=0.95)

    assert var == pytest.approx(0.949)


def test_var_sign_convention_is_positive_loss_magnitude() -> None:
    terminal_returns = np.array([0.10, -0.20, -0.05, 0.03])

    var = VaRCalculator().calculate(terminal_returns, confidence_level=0.75)

    assert var > 0.0


def test_expected_shortfall_uses_losses_at_or_above_var_threshold() -> None:
    terminal_returns = -np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    expected_shortfall = ExpectedShortfallCalculator().calculate(
        terminal_returns,
        confidence_level=0.80,
    )

    assert expected_shortfall == pytest.approx(4.5)


def test_loss_probability_uses_known_fraction_above_threshold() -> None:
    terminal_returns = -np.array([0.01, 0.05, 0.12, 0.20])

    probability = LossProbabilityCalculator().calculate(
        terminal_returns,
        loss_threshold=0.10,
    )

    assert probability == pytest.approx(0.5)


def test_drawdown_calculator_known_single_path_sequence() -> None:
    portfolio_path = np.array([0.10, -0.20, 0.05])

    drawdowns = DrawdownCalculator().calculate(portfolio_path)

    expected = np.array([0.0, 0.20, 0.16])
    np.testing.assert_allclose(drawdowns, expected)


def test_drawdown_distribution_analyzer_mean_and_q95() -> None:
    # Four drawdown paths; max per path is [0.0, 0.10, 0.20, 0.30]
    drawdown_paths = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.10, 0.10],
            [0.0, 0.20, 0.15],
            [0.0, 0.30, 0.25],
        ]
    )

    mean_dd, q95_dd = DrawdownDistributionAnalyzer().summarize(drawdown_paths)

    assert mean_dd == pytest.approx(0.15)
    assert q95_dd == pytest.approx(0.285)


def test_risk_result_builder_populates_all_domain_fields() -> None:
    result = RiskResultBuilder().build(
        model_name="Gaussian",
        confidence_level=0.95,
        var=0.04,
        expected_shortfall=0.06,
        loss_threshold=0.10,
        probability_loss_exceeds_threshold=0.02,
        max_drawdown_mean=0.08,
        max_drawdown_q95=0.15,
        mean_simulated_loss=0.01,
        worst_simulated_loss=0.25,
    )

    assert result == RiskResult(
        model_name="Gaussian",
        confidence_level=0.95,
        var=0.04,
        expected_shortfall=0.06,
        loss_threshold=0.10,
        probability_loss_exceeds_threshold=0.02,
        max_drawdown_mean=0.08,
        max_drawdown_q95=0.15,
        mean_simulated_loss=0.01,
        worst_simulated_loss=0.25,
    )
