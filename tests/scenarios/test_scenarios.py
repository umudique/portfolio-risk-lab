from __future__ import annotations

import pytest

from src.domain import Portfolio, ScenarioDefinition, ScenarioResult, ScenarioShock
from src.scenarios import (
    ScenarioLossCalculator,
    ScenarioResultBuilder,
    ScenarioValidator,
    ShockApplier,
)


def make_portfolio() -> Portfolio:
    return Portfolio(tickers=("AAPL", "MSFT"), weights=(0.5, 0.5))


def test_known_two_asset_shock_produces_expected_positive_portfolio_loss() -> None:
    scenario = ScenarioDefinition(
        name="tech selloff",
        shocks=(
            ScenarioShock(ticker="AAPL", shock=-0.10),
            ScenarioShock(ticker="MSFT", shock=-0.05),
        ),
    )

    ScenarioValidator().validate(make_portfolio(), scenario)
    scenario_return = ShockApplier().apply(make_portfolio(), scenario)
    portfolio_loss = ScenarioLossCalculator().calculate(scenario_return)

    assert portfolio_loss == pytest.approx(0.075)


def test_scenario_validator_rejects_shock_for_ticker_not_in_portfolio() -> None:
    scenario = ScenarioDefinition(
        name="unknown ticker",
        shocks=(ScenarioShock(ticker="TSLA", shock=-0.10),),
    )

    with pytest.raises(ValueError, match="ticker|portfolio"):
        ScenarioValidator().validate(make_portfolio(), scenario)


def test_scenario_validator_rejects_shock_outside_documented_bounds() -> None:
    too_negative = ScenarioDefinition(
        name="impossible loss",
        shocks=(ScenarioShock(ticker="AAPL", shock=-1.01),),
    )
    too_positive = ScenarioDefinition(
        name="impossible gain",
        shocks=(ScenarioShock(ticker="MSFT", shock=5.01),),
    )

    validator = ScenarioValidator()
    with pytest.raises(ValueError, match="-1.0|5.0|bound|shock"):
        validator.validate(make_portfolio(), too_negative)
    with pytest.raises(ValueError, match="-1.0|5.0|bound|shock"):
        validator.validate(make_portfolio(), too_positive)


def test_offsetting_shocks_produce_zero_portfolio_loss() -> None:
    scenario = ScenarioDefinition(
        name="offsetting",
        shocks=(
            ScenarioShock(ticker="AAPL", shock=0.10),
            ScenarioShock(ticker="MSFT", shock=-0.10),
        ),
    )

    scenario_return = ShockApplier().apply(make_portfolio(), scenario)
    portfolio_loss = ScenarioLossCalculator().calculate(scenario_return)

    assert scenario_return == pytest.approx(0.0)
    assert portfolio_loss == pytest.approx(0.0)


def test_shock_applier_uses_weighted_sum_formula_independently() -> None:
    portfolio = Portfolio(tickers=("AAPL", "MSFT"), weights=(0.25, 0.75))
    scenario = ScenarioDefinition(
        name="weighted move",
        shocks=(
            ScenarioShock(ticker="AAPL", shock=-0.20),
            ScenarioShock(ticker="MSFT", shock=0.04),
        ),
    )

    scenario_return = ShockApplier().apply(portfolio, scenario)

    assert scenario_return == pytest.approx(-0.02)


def test_scenario_result_carries_all_shock_and_loss_fields() -> None:
    scenario = ScenarioDefinition(
        name="tech selloff",
        shocks=(
            ScenarioShock(ticker="AAPL", shock=-0.10),
            ScenarioShock(ticker="MSFT", shock=-0.05),
        ),
    )

    result = ScenarioResultBuilder().build(scenario=scenario, portfolio_loss=0.075)

    assert result == ScenarioResult(
        scenario_name="tech selloff",
        asset_shocks={"AAPL": -0.10, "MSFT": -0.05},
        portfolio_loss=0.075,
    )
