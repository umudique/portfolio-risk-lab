from __future__ import annotations

from src.diagnostics.verdict import ModelRiskVerdictCalculator
from src.domain import RiskResult


def _risk_result(model_name: str, *, var: float, expected_shortfall: float) -> RiskResult:
    return RiskResult(
        model_name=model_name,
        confidence_level=0.95,
        var=var,
        expected_shortfall=expected_shortfall,
        loss_threshold=0.10,
        probability_loss_exceeds_threshold=0.05,
        max_drawdown_mean=0.03,
        max_drawdown_q95=0.05,
        mean_simulated_loss=0.01,
        worst_simulated_loss=0.08,
    )


def test_model_risk_verdict_high_when_var_range_exceeds_three_percentage_points() -> None:
    risk_results = [
        _risk_result("Gaussian", var=0.02, expected_shortfall=0.03),
        _risk_result("Student-t", var=0.051, expected_shortfall=0.04),
    ]

    assert ModelRiskVerdictCalculator().calculate(risk_results) == "HIGH"


def test_model_risk_verdict_high_when_es_range_exceeds_five_percentage_points() -> None:
    risk_results = [
        _risk_result("Gaussian", var=0.02, expected_shortfall=0.03),
        _risk_result("Student-t", var=0.03, expected_shortfall=0.081),
    ]

    assert ModelRiskVerdictCalculator().calculate(risk_results) == "HIGH"


def test_model_risk_verdict_moderate_when_var_range_is_one_to_three_points() -> None:
    risk_results = [
        _risk_result("Gaussian", var=0.02, expected_shortfall=0.03),
        _risk_result("Student-t", var=0.04, expected_shortfall=0.04),
    ]

    assert ModelRiskVerdictCalculator().calculate(risk_results) == "MODERATE"


def test_model_risk_verdict_moderate_when_es_range_is_two_to_five_points() -> None:
    risk_results = [
        _risk_result("Gaussian", var=0.02, expected_shortfall=0.03),
        _risk_result("Student-t", var=0.025, expected_shortfall=0.06),
    ]

    assert ModelRiskVerdictCalculator().calculate(risk_results) == "MODERATE"


def test_model_risk_verdict_low_when_var_and_es_ranges_are_below_low_thresholds() -> None:
    risk_results = [
        _risk_result("Gaussian", var=0.02, expected_shortfall=0.03),
        _risk_result("Student-t", var=0.029, expected_shortfall=0.049),
    ]

    assert ModelRiskVerdictCalculator().calculate(risk_results) == "LOW"


def test_model_risk_verdict_unknown_when_fewer_than_two_results() -> None:
    risk_results = [_risk_result("Gaussian", var=0.02, expected_shortfall=0.03)]

    assert ModelRiskVerdictCalculator().calculate(risk_results) == "UNKNOWN"
