from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.domain import (
    AnalysisConfig,
    DiagnosticReport,
    ModelRiskFinding,
    Portfolio,
    ReturnEstimation,
    RiskResult,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioShock,
    SimulationResult,
)
from src.pipeline.analysis import AnalysisPipeline


class SpySimulator:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[tuple[Portfolio, Any, AnalysisConfig]] = []

    def simulate(
        self,
        portfolio: Portfolio,
        estimation: Any,
        config: AnalysisConfig,
    ) -> SimulationResult:
        self.calls.append((portfolio, estimation, config))
        return SimulationResult(
            model_name=self.model_name,
            terminal_returns=[-0.01, -0.02, 0.01],
            portfolio_paths=[[1.0, 0.99], [1.0, 0.98]],
            random_seed=config.random_seed,
            simulation_count=3,
            horizon=config.horizon,
        )


class SpyRiskEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[SimulationResult, AnalysisConfig]] = []

    def measure(self, simulation_result: SimulationResult, config: AnalysisConfig) -> RiskResult:
        self.calls.append((simulation_result, config))
        offset = float(len(self.calls)) / 1000.0
        return RiskResult(
            model_name=simulation_result.model_name,
            confidence_level=config.confidence_level,
            var=0.02 + offset,
            expected_shortfall=0.03 + offset,
            loss_threshold=config.loss_threshold,
            probability_loss_exceeds_threshold=0.10,
            max_drawdown_mean=0.04,
            max_drawdown_q95=0.05,
            mean_simulated_loss=0.01,
            worst_simulated_loss=0.08,
        )


class SpyScenarioEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[Portfolio, tuple[ScenarioDefinition, ...]]] = []

    def run(
        self,
        portfolio: Portfolio,
        scenario_definitions: tuple[ScenarioDefinition, ...],
    ) -> list[ScenarioResult]:
        self.calls.append((portfolio, scenario_definitions))
        return [
            ScenarioResult(
                scenario_name="Market Crash",
                asset_shocks={"SPY": -0.30},
                portfolio_loss=0.30,
            )
        ]


class SpyVerdictCalculator:
    def __init__(self) -> None:
        self.calls: list[list[RiskResult]] = []

    def calculate(self, risk_results: list[RiskResult]) -> str:
        self.calls.append(risk_results)
        return "HIGH"


def _portfolio() -> Portfolio:
    return Portfolio(tickers=("SPY", "TLT"), weights=(0.60, 0.40))


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 99.0, 102.0],
            "TLT": [100.0, 99.0, 100.0, 101.0],
        }
    )


def _config() -> AnalysisConfig:
    return AnalysisConfig(
        confidence_level=0.95,
        horizon=2,
        n_simulations=3,
        loss_threshold=0.10,
        random_seed=42,
        scenario_definitions=(
            ScenarioDefinition(
                name="Market Crash",
                shocks=(ScenarioShock(ticker="SPY", shock=-0.30),),
            ),
        ),
    )


def _pipeline_fixture() -> tuple[
    AnalysisPipeline,
    tuple[SpySimulator, SpySimulator, SpySimulator],
    SpyRiskEngine,
    SpyScenarioEngine,
    SpyVerdictCalculator,
]:
    simulators = (
        SpySimulator("Gaussian"),
        SpySimulator("Student-t"),
        SpySimulator("Historical Bootstrap"),
    )
    risk_engine = SpyRiskEngine()
    scenario_engine = SpyScenarioEngine()
    verdict_calculator = SpyVerdictCalculator()
    pipeline = AnalysisPipeline(
        simulators=simulators,
        risk_engine=risk_engine,
        scenario_engine=scenario_engine,
        verdict_calculator=verdict_calculator,
    )
    return pipeline, simulators, risk_engine, scenario_engine, verdict_calculator


def test_analysis_pipeline_run_returns_diagnostic_report() -> None:
    pipeline, _, _, _, _ = _pipeline_fixture()

    report = pipeline.run(_portfolio(), _prices(), _config())

    assert isinstance(report, DiagnosticReport)


def test_analysis_pipeline_runs_all_three_model_families() -> None:
    pipeline, simulators, _, _, _ = _pipeline_fixture()

    pipeline.run(_portfolio(), _prices(), _config())

    assert [simulator.model_name for simulator in simulators] == [
        "Gaussian",
        "Student-t",
        "Historical Bootstrap",
    ]
    assert all(len(simulator.calls) == 1 for simulator in simulators)


def test_analysis_pipeline_calls_risk_engine_once_per_simulation() -> None:
    pipeline, simulators, risk_engine, _, _ = _pipeline_fixture()

    pipeline.run(_portfolio(), _prices(), _config())

    assert len(risk_engine.calls) == len(simulators)


def test_analysis_pipeline_calls_scenario_engine_independently_of_simulation() -> None:
    pipeline, simulators, _, scenario_engine, _ = _pipeline_fixture()

    pipeline.run(_portfolio(), _prices(), _config())

    assert len(scenario_engine.calls) == 1
    assert all(len(simulator.calls) == 1 for simulator in simulators)
    assert scenario_engine.calls[0][1] == _config().scenario_definitions


def test_analysis_pipeline_uses_verdict_calculator_for_report_model_risk_level() -> None:
    pipeline, _, _, _, verdict_calculator = _pipeline_fixture()

    report = pipeline.run(_portfolio(), _prices(), _config())

    assert len(verdict_calculator.calls) == 1
    assert report.model_risk_level == "HIGH"


def test_analysis_pipeline_report_contains_one_simulation_result_per_model() -> None:
    pipeline, simulators, _, _, _ = _pipeline_fixture()

    report = pipeline.run(_portfolio(), _prices(), _config())

    assert len(report.simulation_results) == len(simulators)
    assert [result.model_name for result in report.simulation_results] == [
        "Gaussian",
        "Student-t",
        "Historical Bootstrap",
    ]


def test_analysis_pipeline_report_return_estimation_is_populated() -> None:
    pipeline, _, _, _, _ = _pipeline_fixture()

    report = pipeline.run(_portfolio(), _prices(), _config())

    assert isinstance(report.return_estimation, ReturnEstimation)


def test_analysis_pipeline_report_contains_diagnostics_outputs() -> None:
    pipeline, _, _, _, _ = _pipeline_fixture()

    report = pipeline.run(_portfolio(), _prices(), _config())

    assert report.comparison_rows
    assert all(isinstance(finding, ModelRiskFinding) for finding in report.findings)
    assert report.scenario_results
    assert report.run_date == date.today()


def test_analysis_pipeline_report_contains_portfolio_weights() -> None:
    pipeline, _, _, _, _ = _pipeline_fixture()
    portfolio = _portfolio()

    report = pipeline.run(portfolio, _prices(), _config())

    assert report.portfolio_weights is not None
    assert report.portfolio_weights == dict(zip(portfolio.tickers, portfolio.weights))


def test_pipeline_analysis_does_not_import_from_app() -> None:
    tree = ast.parse(Path("src/pipeline/analysis.py").read_text())
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert not any(module == "app" or module.startswith("app.") for module in imported_modules)


def test_app_orchestrator_phase2_target_has_no_forbidden_src_imports() -> None:
    prohibited_modules = ("src.simulation", "src.risk", "src.returns", "src.scenarios")
    tree = ast.parse(Path("app/orchestrator.py").read_text())
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert not any(
        module == prohibited or module.startswith(f"{prohibited}.")
        for module in imported_modules
        for prohibited in prohibited_modules
    )
