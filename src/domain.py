"""Canonical data contracts — the only definitions used across all modules.

Every public type here is immutable once produced by its source module.
No module may redefine these types or introduce parallel dataclasses that
serve the same conceptual role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

# ---------------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapConfig:
    block_size: int = 1


@dataclass(frozen=True)
class ScenarioShock:
    ticker: str
    shock: float


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    shocks: tuple[ScenarioShock, ...]


@dataclass(frozen=True)
class AnalysisConfig:
    confidence_level: float
    horizon: int
    n_simulations: int
    loss_threshold: float
    random_seed: int
    estimation_window: int | None = None
    student_t_degrees_of_freedom: float = 5.0
    bootstrap_config: BootstrapConfig = field(default_factory=BootstrapConfig)
    scenario_definitions: tuple[ScenarioDefinition, ...] = field(default_factory=tuple)
    use_mean_returns: bool = False


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Portfolio:
    tickers: tuple[str, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.tickers) != len(self.weights):
            raise ValueError("tickers and weights must have the same length")
        if len(self.tickers) == 0:
            raise ValueError("portfolio must contain at least one asset")


# ---------------------------------------------------------------------------
# Return estimation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReturnEstimation:
    tickers: tuple[str, ...]
    returns: Any
    mean_returns: Any
    volatilities: Any
    covariance_matrix: Any
    correlation_matrix: Any
    observation_count: int
    estimation_window: str


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    model_name: str
    terminal_returns: Any
    portfolio_paths: Any | None
    random_seed: int
    simulation_count: int
    horizon: int
    model_params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskResult:
    model_name: str
    confidence_level: float
    var: float
    expected_shortfall: float
    loss_threshold: float
    probability_loss_exceeds_threshold: float
    max_drawdown_mean: float
    max_drawdown_q95: float
    mean_simulated_loss: float
    worst_simulated_loss: float


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioResult:
    scenario_name: str
    asset_shocks: dict[str, float]
    portfolio_loss: float


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRiskFinding:
    finding_text: str
    metric: str
    model_a: str
    model_b: str
    absolute_difference: float
    relative_difference: float


@dataclass(frozen=True)
class DiagnosticReport:
    comparison_rows: list[dict[str, Any]]
    findings: list[ModelRiskFinding]
    scenario_results: list[ScenarioResult]
    estimation_window: str
    observation_count: int
    run_date: date
    # UI presentation fields — populated by the orchestrator for dashboard rendering
    model_risk_level: str = "UNKNOWN"
    simulation_results: tuple[Any, ...] = field(default_factory=tuple)
    return_estimation: Any = None
    portfolio_weights: dict[str, float] | None = None
