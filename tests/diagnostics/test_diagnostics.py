from __future__ import annotations

from datetime import date

import pytest

from src.diagnostics import (
    DiagnosticReportBuilder,
    ModelComparisonTableBuilder,
    ModelRiskFindingGenerator,
    RelativeDifferenceCalculator,
    TailRiskComparator,
)
from src.domain import DiagnosticReport, ModelRiskFinding, RiskResult, ScenarioResult


def make_risk_result(
    model_name: str,
    *,
    var: float,
    expected_shortfall: float,
    loss_probability: float,
) -> RiskResult:
    return RiskResult(
        model_name=model_name,
        confidence_level=0.95,
        var=var,
        expected_shortfall=expected_shortfall,
        loss_threshold=0.10,
        probability_loss_exceeds_threshold=loss_probability,
        max_drawdown_mean=0.08,
        max_drawdown_q95=0.15,
        mean_simulated_loss=0.01,
        worst_simulated_loss=0.25,
    )


def make_three_model_results() -> list[RiskResult]:
    return [
        make_risk_result("Gaussian", var=4.2, expected_shortfall=6.0, loss_probability=1.3),
        make_risk_result("Student-t", var=5.8, expected_shortfall=9.3, loss_probability=4.4),
        make_risk_result("Bootstrap", var=5.1, expected_shortfall=7.8, loss_probability=3.1),
    ]


def test_model_comparison_table_aligns_three_models_with_required_rows() -> None:
    rows = ModelComparisonTableBuilder().build(make_three_model_results())

    assert rows[:3] == [
        {"metric": "VaR", "Gaussian": 4.2, "Student-t": 5.8, "Bootstrap": 5.1},
        {
            "metric": "Expected Shortfall",
            "Gaussian": 6.0,
            "Student-t": 9.3,
            "Bootstrap": 7.8,
        },
        {
            "metric": "P(terminal loss > threshold)",
            "Gaussian": 1.3,
            "Student-t": 4.4,
            "Bootstrap": 3.1,
        },
    ]
    assert any(row["metric"] == "Mean Max Drawdown" for row in rows)
    assert any(row["metric"] == "95th Percentile Max Drawdown" for row in rows)


def test_model_comparison_table_appends_var_and_es_range_rows() -> None:
    rows = ModelComparisonTableBuilder().build(make_three_model_results())

    var_range_row = next((r for r in rows if r.get("metric") == "VaR range"), None)
    es_range_row = next((r for r in rows if r.get("metric") == "ES range"), None)

    assert var_range_row is not None, "VaR range row must be present"
    assert es_range_row is not None, "ES range row must be present"
    assert var_range_row["value"] == pytest.approx(5.8 - 4.2, abs=1e-4)
    assert es_range_row["value"] == pytest.approx(9.3 - 6.0, abs=1e-4)


def test_relative_difference_calculator_known_expected_shortfall_values() -> None:
    relative_difference = RelativeDifferenceCalculator().calculate(
        baseline=6.0,
        comparison=9.3,
    )

    assert relative_difference == pytest.approx(0.55)


def test_tail_risk_comparator_identifies_highest_expected_shortfall_model() -> None:
    highest_model = TailRiskComparator().highest_expected_shortfall_model(
        make_three_model_results()
    )

    assert highest_model == "Student-t"


def test_model_risk_finding_generator_emits_finding_above_threshold() -> None:
    finding = ModelRiskFindingGenerator().generate(
        metric="Expected Shortfall",
        model_a="Gaussian",
        model_b="Student-t",
        value_a=6.0,
        value_b=9.3,
        materiality_threshold=0.10,
    )

    assert finding == ModelRiskFinding(
        finding_text="Student-t Expected Shortfall differs from Gaussian by 55.0%",
        metric="Expected Shortfall",
        model_a="Gaussian",
        model_b="Student-t",
        absolute_difference=3.3,
        relative_difference=0.55,
    )


def test_model_risk_finding_generator_suppresses_immaterial_difference() -> None:
    finding = ModelRiskFindingGenerator().generate(
        metric="Expected Shortfall",
        model_a="Gaussian",
        model_b="Bootstrap",
        value_a=6.0,
        value_b=6.3,
        materiality_threshold=0.10,
    )

    assert finding is None


def test_diagnostic_report_builder_assembles_report_from_precomputed_outputs() -> None:
    comparison_rows = [
        {"metric": "VaR", "Gaussian": 4.2, "Student-t": 5.8, "Bootstrap": 5.1},
        {"metric": "Expected Shortfall", "Gaussian": 6.0, "Student-t": 9.3, "Bootstrap": 7.8},
        {
            "metric": "P(terminal loss > threshold)",
            "Gaussian": 1.3,
            "Student-t": 4.4,
            "Bootstrap": 3.1,
        },
    ]
    findings = [
        ModelRiskFinding(
            finding_text="Student-t Expected Shortfall differs from Gaussian by 55.0%",
            metric="Expected Shortfall",
            model_a="Gaussian",
            model_b="Student-t",
            absolute_difference=3.3,
            relative_difference=0.55,
        )
    ]
    scenario_results = [
        ScenarioResult(
            scenario_name="tech selloff",
            asset_shocks={"AAPL": -0.10, "MSFT": -0.05},
            portfolio_loss=0.075,
        )
    ]

    report = DiagnosticReportBuilder().build(
        comparison_rows=comparison_rows,
        findings=findings,
        scenario_results=scenario_results,
        estimation_window="2020-01-01/2024-12-31",
        observation_count=1000,
        run_date=date(2026, 9, 24),
    )

    assert report == DiagnosticReport(
        comparison_rows=comparison_rows,
        findings=findings,
        scenario_results=scenario_results,
        estimation_window="2020-01-01/2024-12-31",
        observation_count=1000,
        run_date=date(2026, 9, 24),
    )


def test_diagnostic_report_builder_passes_portfolio_weights_through() -> None:
    weights = {"SPY": 0.60, "TLT": 0.40}
    comparison_rows = [{"metric": "VaR", "Gaussian": 4.2, "Student-t": 5.8, "Bootstrap": 5.1}]

    report = DiagnosticReportBuilder().build(
        comparison_rows=comparison_rows,
        findings=[],
        scenario_results=[],
        estimation_window="756 trading days",
        observation_count=756,
        run_date=date(2026, 9, 27),
        portfolio_weights=weights,
    )

    assert report.portfolio_weights == weights
