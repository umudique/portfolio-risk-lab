"""Concrete analysis orchestrator — wires all domain modules into the full pipeline."""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from src.diagnostics.comparison import ModelComparisonTableBuilder
from src.diagnostics.findings import ModelRiskFindingGenerator
from src.diagnostics.report import DiagnosticReportBuilder
from src.domain import (
    AnalysisConfig,
    DiagnosticReport,
    ModelRiskFinding,
    Portfolio,
    RiskResult,
    ScenarioDefinition,
    ScenarioShock,
)
from src.portfolio.canonicalizer import PortfolioCanonicalizer
from src.portfolio.validation import PortfolioSchemaValidator, WeightValidator
from src.returns.calculation import ReturnCalculator
from src.returns.correlation import CorrelationEstimator
from src.returns.covariance import CovarianceEstimator
from src.returns.estimation import EstimationResultBuilder, MeanReturnEstimator
from src.returns.volatility import VolatilityEstimator
from src.risk.drawdown import DrawdownCalculator, DrawdownDistributionAnalyzer
from src.risk.expected_shortfall import ExpectedShortfallCalculator
from src.risk.loss_probability import LossProbabilityCalculator
from src.risk.result import RiskResultBuilder
from src.risk.var import VaRCalculator
from src.scenarios.result import ScenarioResultBuilder
from src.scenarios.shocks import ShockApplier
from src.simulation.bootstrap import HistoricalBootstrapSimulator
from src.simulation.gaussian import GaussianSimulator
from src.simulation.student_t import StudentTSimulator

_DEFAULT_LOOKBACK_DAYS = 756
_COMPARED_METRICS: list[tuple[str, str]] = [
    ("VaR", "var"),
    ("Expected Shortfall", "expected_shortfall"),
    ("Loss Probability", "probability_loss_exceeds_threshold"),
]
_DEFAULT_SCENARIO_SHOCKS = [
    ("Market Crash", -0.30),
    ("Sector Correction", -0.15),
    ("Mild Downturn", -0.08),
]


class ConcreteAnalysisOrchestrator:
    """Concrete implementation of the AnalysisOrchestrator protocol.

    Runs the full pipeline: portfolio loading → price fetch → return estimation
    → simulation (Gaussian, Student-t, Bootstrap) → risk metrics → scenarios
    → diagnostic comparison → report assembly.
    """

    def run(self, portfolio_csv: Any, config: AnalysisConfig) -> DiagnosticReport:
        """Execute the full analysis pipeline.

        Args:
            portfolio_csv: Streamlit UploadedFile or any object with .getvalue()
                returning portfolio CSV bytes.
            config: Canonical analysis configuration from the dashboard form.

        Returns:
            Fully assembled diagnostic report.

        Raises:
            ValueError: If any pipeline stage rejects its inputs.
        """
        portfolio = _load_portfolio(portfolio_csv)
        prices = _fetch_prices(portfolio.tickers, config.estimation_window)
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

        risk_results, sim_results = _run_simulations(portfolio, estimation, config)
        scenario_definitions = config.scenario_definitions or _default_scenarios(portfolio)
        scenario_results = _run_scenarios(portfolio, scenario_definitions)
        comparison_rows = ModelComparisonTableBuilder().build(risk_results)
        findings = _generate_findings(risk_results)
        model_risk_level = _compute_model_risk_level(risk_results)

        return DiagnosticReportBuilder().build(
            comparison_rows=comparison_rows,
            findings=findings,
            scenario_results=scenario_results,
            estimation_window=estimation.estimation_window,
            observation_count=estimation.observation_count,
            run_date=date.today(),
            model_risk_level=model_risk_level,
            simulation_results=tuple(sim_results),
            return_estimation=estimation,
        )


def _load_portfolio(portfolio_csv: Any) -> Portfolio:
    content = portfolio_csv.getvalue().decode("utf-8")
    rows = pd.read_csv(io.StringIO(content))
    schema_valid = PortfolioSchemaValidator().validate(rows)
    weight_valid = WeightValidator().validate(schema_valid)
    return PortfolioCanonicalizer().canonicalize(weight_valid)


def _fetch_prices(tickers: tuple[str, ...], estimation_window: int | None) -> pd.DataFrame:
    lookback = estimation_window if estimation_window is not None else _DEFAULT_LOOKBACK_DAYS
    end = datetime.today()
    start = end - timedelta(days=lookback)
    data = yf.download(
        list(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = list(tickers)
    prices.columns = list(tickers)
    prices = prices.dropna()
    if prices.empty or len(prices) < 2:
        raise ValueError("insufficient price history fetched for the requested tickers")
    return prices  # type: ignore[no-any-return]


def _run_simulations(
    portfolio: Portfolio,
    estimation: Any,
    config: AnalysisConfig,
) -> tuple[list[RiskResult], list[Any]]:
    from src.domain import SimulationResult

    simulators = [GaussianSimulator(), StudentTSimulator(), HistoricalBootstrapSimulator()]
    risk_results: list[RiskResult] = []
    sim_results: list[SimulationResult] = []

    for simulator in simulators:
        sim_result = simulator.simulate(portfolio, estimation, config)
        sim_results.append(sim_result)
        terminal = np.asarray(sim_result.terminal_returns, dtype=np.float64)
        paths = np.asarray(sim_result.portfolio_paths, dtype=np.float64)

        var = VaRCalculator().calculate(terminal, confidence_level=config.confidence_level)
        es = ExpectedShortfallCalculator().calculate(
            terminal, confidence_level=config.confidence_level
        )
        loss_prob = LossProbabilityCalculator().calculate(
            terminal, loss_threshold=config.loss_threshold
        )
        drawdown_paths = DrawdownCalculator().calculate(paths)
        dd_mean, dd_q95 = DrawdownDistributionAnalyzer().summarize(drawdown_paths)

        losses = -terminal
        positive_losses = losses[losses > 0]
        mean_loss = float(np.mean(positive_losses)) if len(positive_losses) > 0 else 0.0
        worst_loss = max(float(np.max(losses)), 0.0)

        risk_results.append(
            RiskResultBuilder().build(
                model_name=sim_result.model_name,
                confidence_level=config.confidence_level,
                var=var,
                expected_shortfall=es,
                loss_threshold=config.loss_threshold,
                probability_loss_exceeds_threshold=loss_prob,
                max_drawdown_mean=dd_mean,
                max_drawdown_q95=dd_q95,
                mean_simulated_loss=mean_loss,
                worst_simulated_loss=worst_loss,
            )
        )

    return risk_results, sim_results


def _compute_model_risk_level(risk_results: list[RiskResult]) -> str:
    if len(risk_results) < 2:
        return "UNKNOWN"
    vars_ = [r.var for r in risk_results]
    es_ = [r.expected_shortfall for r in risk_results]
    var_range = max(vars_) - min(vars_)
    es_range = max(es_) - min(es_)
    if var_range > 0.03 or es_range > 0.05:
        return "HIGH"
    if var_range > 0.01 or es_range > 0.02:
        return "MODERATE"
    return "LOW"


def _run_scenarios(
    portfolio: Portfolio,
    scenario_definitions: tuple[ScenarioDefinition, ...],
) -> list[Any]:
    results = []
    for scenario in scenario_definitions:
        scenario_return = ShockApplier().apply(portfolio, scenario)
        portfolio_loss = max(-scenario_return, 0.0)
        results.append(
            ScenarioResultBuilder().build(scenario=scenario, portfolio_loss=portfolio_loss)
        )
    return results


def _default_scenarios(portfolio: Portfolio) -> tuple[ScenarioDefinition, ...]:
    return tuple(
        ScenarioDefinition(
            name=name,
            shocks=tuple(ScenarioShock(ticker=ticker, shock=shock) for ticker in portfolio.tickers),
        )
        for name, shock in _DEFAULT_SCENARIO_SHOCKS
    )


def _generate_findings(risk_results: list[RiskResult]) -> list[ModelRiskFinding]:
    findings: list[ModelRiskFinding] = []
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
