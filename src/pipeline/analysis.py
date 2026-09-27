"""Application-service coordination contract for the analytical pipeline."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.diagnostics.comparison import ModelComparisonTableBuilder
from src.diagnostics.findings import ModelRiskFindingGenerator
from src.diagnostics.report import DiagnosticReportBuilder
from src.diagnostics.verdict import ModelRiskVerdictCalculator
from src.domain import (
    AnalysisConfig,
    DiagnosticReport,
    Portfolio,
    RiskResult,
    ScenarioDefinition,
    ScenarioShock,
    SimulationResult,
)
from src.returns.calculation import ReturnCalculator
from src.returns.correlation import CorrelationEstimator
from src.returns.covariance import CovarianceEstimator
from src.returns.estimation import EstimationResultBuilder, MeanReturnEstimator
from src.returns.volatility import VolatilityEstimator
from src.risk.drawdown import DrawdownCalculator, DrawdownDistributionAnalyzer
from src.risk.expected_shortfall import ExpectedShortfallCalculator
from src.risk.loss_probability import LossProbabilityCalculator
from src.risk.loss_statistics import LossStatisticsCalculator
from src.risk.result import RiskResultBuilder
from src.risk.var import VaRCalculator
from src.scenarios.loss import ScenarioLossCalculator
from src.scenarios.result import ScenarioResultBuilder
from src.scenarios.shocks import ShockApplier
from src.simulation.bootstrap import HistoricalBootstrapSimulator
from src.simulation.gaussian import GaussianSimulator
from src.simulation.student_t import StudentTSimulator

_COMPARED_METRICS: tuple[tuple[str, str], ...] = (
    ("VaR", "var"),
    ("Expected Shortfall", "expected_shortfall"),
    ("Loss Probability", "probability_loss_exceeds_threshold"),
)
_DEFAULT_SCENARIO_SHOCKS: tuple[tuple[str, float], ...] = (
    ("Market Crash", -0.30),
    ("Sector Correction", -0.15),
    ("Mild Downturn", -0.08),
)


class RiskMeasurementEngine:
    """Coordinate risk calculators for one simulation result."""

    def measure(self, simulation_result: SimulationResult, config: AnalysisConfig) -> RiskResult:
        """Measure risk for one completed simulation result."""
        var = VaRCalculator().calculate(
            simulation_result.terminal_returns,
            confidence_level=config.confidence_level,
        )
        expected_shortfall = ExpectedShortfallCalculator().calculate(
            simulation_result.terminal_returns,
            confidence_level=config.confidence_level,
        )
        loss_probability = LossProbabilityCalculator().calculate(
            simulation_result.terminal_returns,
            loss_threshold=config.loss_threshold,
        )
        drawdown_paths = DrawdownCalculator().calculate(simulation_result.portfolio_paths)
        drawdown_mean, drawdown_q95 = DrawdownDistributionAnalyzer().summarize(drawdown_paths)
        mean_loss, worst_loss = LossStatisticsCalculator().calculate(
            simulation_result.terminal_returns
        )

        return RiskResultBuilder().build(
            model_name=simulation_result.model_name,
            confidence_level=config.confidence_level,
            var=var,
            expected_shortfall=expected_shortfall,
            loss_threshold=config.loss_threshold,
            probability_loss_exceeds_threshold=loss_probability,
            max_drawdown_mean=drawdown_mean,
            max_drawdown_q95=drawdown_q95,
            mean_simulated_loss=mean_loss,
            worst_simulated_loss=worst_loss,
        )


class ScenarioAnalysisEngine:
    """Coordinate deterministic scenario application."""

    def run(
        self,
        portfolio: Portfolio,
        scenario_definitions: tuple[ScenarioDefinition, ...],
    ) -> list[Any]:
        """Run deterministic scenario definitions independently of simulation."""
        results = []
        for scenario in scenario_definitions:
            scenario_return = ShockApplier().apply(portfolio, scenario)
            portfolio_loss = ScenarioLossCalculator().calculate(scenario_return)
            results.append(
                ScenarioResultBuilder().build(
                    scenario=scenario,
                    portfolio_loss=portfolio_loss,
                )
            )
        return results


class AnalysisPipeline:
    """Coordinate return estimation, simulation, risk, scenarios, and diagnostics.

    Args:
        simulators: Optional three-model simulation family sequence. The Phase 2
            default is Gaussian, Student-t, and Historical Bootstrap.
        risk_engine: Optional risk-measurement boundary used once per
            simulation result.
        scenario_engine: Optional deterministic scenario boundary, independent
            of stochastic simulation.
        verdict_calculator: Optional diagnostics verdict calculator used to set
            ``DiagnosticReport.model_risk_level``.

    Returns:
        An application-service object whose ``run`` method produces a
        :class:`src.domain.DiagnosticReport`.

    Raises:
        ValueError: If injected collaborators are incomplete or inconsistent.

    Invariants:
        The pipeline coordinates existing ``src`` modules only. It may invoke
        return estimation, simulation, risk, scenario, and diagnostics
        components in order, but it contains no mathematical logic of its own,
        imports nothing from ``app/``, performs no market-data fetching, and
        never stores UI session state.
    """

    def __init__(
        self,
        *,
        simulators: Any = None,
        risk_engine: Any = None,
        scenario_engine: Any = None,
        verdict_calculator: Any = None,
    ) -> None:
        """Create a pipeline with optional test or production collaborators.

        Args:
            simulators: Optional simulation model collaborators.
            risk_engine: Optional risk-measurement collaborator.
            scenario_engine: Optional deterministic scenario collaborator.
            verdict_calculator: Optional model-risk verdict collaborator.

        Returns:
            ``None`` after storing collaborator configuration for ``run``.

        Raises:
            ValueError: If collaborators are invalid.

        Invariants:
            Constructor injection is for orchestration boundaries only. It must
            not perform calculations, fetch prices, or call analytical engines.
        """
        self._simulators = simulators or (
            GaussianSimulator(),
            StudentTSimulator(),
            HistoricalBootstrapSimulator(),
        )
        self._risk_engine = risk_engine or RiskMeasurementEngine()
        self._scenario_engine = scenario_engine or ScenarioAnalysisEngine()
        self._verdict_calculator = verdict_calculator or ModelRiskVerdictCalculator()

    def run(
        self,
        portfolio: Portfolio,
        prices: Any,
        config: AnalysisConfig,
    ) -> DiagnosticReport:
        """Run the complete analytical workflow from prices to report.

        Args:
            portfolio: Canonical portfolio loaded and validated before pipeline
                entry.
            prices: Already-fetched, aligned price matrix. Price fetching stays
                outside the pipeline in the app/infrastructure adapter.
            config: Canonical analysis configuration used consistently across
                estimation, simulation, risk, scenarios, and diagnostics.

        Returns:
            Fully assembled ``DiagnosticReport`` containing comparison rows,
            findings, scenario results, model-risk verdict, simulation results,
            and return estimation.

        Raises:
            ValueError: If any coordinated module rejects its inputs or if the
                pipeline cannot produce the required three-model comparison.

        Invariants:
            The pipeline is coordination-only. It calls return estimation before
            simulation, risk once per simulation result, scenarios
            independently of simulation, then diagnostics/report assembly. It
            contains no ``numpy`` math, no quantiles, no means, no ``max`` over
            domain values, no scenario loss arithmetic, no model-risk threshold
            logic, no yfinance calls, and no imports from ``app``.
        """
        returns = ReturnCalculator().calculate(prices)
        mean_returns = MeanReturnEstimator().estimate(returns)
        volatilities = VolatilityEstimator().estimate(returns)
        covariance_matrix = CovarianceEstimator().estimate(returns)
        correlation_matrix = CorrelationEstimator().estimate(returns)
        estimation = EstimationResultBuilder().build(
            returns=returns,
            mean_returns=mean_returns,
            volatilities=volatilities,
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
        )

        simulation_results = [
            simulator.simulate(portfolio, estimation, config) for simulator in self._simulators
        ]
        risk_results = [
            self._risk_engine.measure(simulation_result, config)
            for simulation_result in simulation_results
        ]

        scenario_definitions = config.scenario_definitions or _default_scenarios(portfolio)
        scenario_results = self._scenario_engine.run(portfolio, scenario_definitions)
        comparison_rows = ModelComparisonTableBuilder().build(risk_results)
        findings = _generate_findings(risk_results)
        model_risk_level = self._verdict_calculator.calculate(risk_results)

        return DiagnosticReportBuilder().build(
            comparison_rows=comparison_rows,
            findings=findings,
            scenario_results=scenario_results,
            estimation_window=estimation.estimation_window,
            observation_count=estimation.observation_count,
            run_date=date.today(),
            model_risk_level=model_risk_level,
            simulation_results=tuple(simulation_results),
            return_estimation=estimation,
            portfolio_weights=dict(zip(portfolio.tickers, portfolio.weights)),
        )


def _default_scenarios(portfolio: Portfolio) -> tuple[ScenarioDefinition, ...]:
    return tuple(
        ScenarioDefinition(
            name=name,
            shocks=tuple(ScenarioShock(ticker=ticker, shock=shock) for ticker in portfolio.tickers),
        )
        for name, shock in _DEFAULT_SCENARIO_SHOCKS
    )


def _generate_findings(risk_results: list[RiskResult]) -> list[Any]:
    findings = []
    generator = ModelRiskFindingGenerator()
    baseline = risk_results[0]
    for other in risk_results[1:]:
        for metric, attr in _COMPARED_METRICS:
            value_a = getattr(baseline, attr)
            value_b = getattr(other, attr)
            if value_a == 0.0:
                continue
            finding = generator.generate(
                metric=metric,
                model_a=baseline.model_name,
                model_b=other.model_name,
                value_a=value_a,
                value_b=value_b,
            )
            if finding is not None:
                findings.append(finding)
    return findings
