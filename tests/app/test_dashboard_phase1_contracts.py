from __future__ import annotations

import ast
import dataclasses
import inspect
import sys
from dataclasses import fields
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

from app import dashboard
from src.domain import (
    AnalysisConfig,
    BootstrapConfig,
    DiagnosticReport,
    ReturnEstimation,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioShock,
    SimulationResult,
)


class StreamlitRecorder(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.session_state: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        recorder = self

        class _CallableContextManager:
            def __init__(self) -> None:
                self._n: int = 1

            def __call__(self, *args: Any, **kwargs: Any) -> _CallableContextManager:
                recorder.calls.append((name, args, kwargs))
                first = args[0] if args else 1
                if isinstance(first, int):
                    self._n = first
                else:
                    try:
                        self._n = len(first)
                    except TypeError:
                        self._n = 1
                return self

            def __getattr__(self, attr_name: str) -> Any:
                def _col_method(*args: Any, **kwargs: Any) -> _CallableContextManager:
                    recorder.calls.append((attr_name, args, kwargs))
                    return self

                return _col_method

            def __iter__(self) -> Any:
                for _ in range(self._n):
                    yield self

            def __enter__(self) -> _CallableContextManager:
                return self

            def __exit__(self, *_: Any) -> None:
                pass

        return _CallableContextManager()


class ScenarioRunnerSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def run(self, portfolio: Any, scenario_definition: Any) -> ScenarioResult:
        self.calls.append((portfolio, scenario_definition))
        return ScenarioResult(
            scenario_name=scenario_definition.name,
            asset_shocks={shock.ticker: shock.shock for shock in scenario_definition.shocks},
            portfolio_loss=0.10,
        )


def _config() -> AnalysisConfig:
    return AnalysisConfig(
        confidence_level=0.95,
        horizon=10,
        n_simulations=5_000,
        loss_threshold=0.10,
        random_seed=42,
        estimation_window=756,
        student_t_degrees_of_freedom=5.0,
        bootstrap_config=BootstrapConfig(block_size=1),
        scenario_definitions=(
            ScenarioDefinition(
                name="Custom",
                shocks=(ScenarioShock(ticker="SPY", shock=-0.10),),
            ),
        ),
        use_mean_returns=False,
    )


def _report(
    *,
    model_risk_level: str = "HIGH",
    return_estimation: ReturnEstimation | None = None,
) -> DiagnosticReport:
    return DiagnosticReport(
        comparison_rows=[
            {"metric": "VaR", "Gaussian": 0.041, "Student-t": 0.063, "Bootstrap": 0.058},
            {
                "metric": "Expected Shortfall",
                "Gaussian": 0.052,
                "Student-t": 0.091,
                "Bootstrap": 0.074,
            },
            {"metric": "VaR range", "value": 0.022},
            {"metric": "ES range", "value": 0.039},
        ],
        findings=[],
        scenario_results=[
            ScenarioResult(
                scenario_name="Market Crash",
                asset_shocks={"SPY": -0.30},
                portfolio_loss=0.30,
            )
        ],
        estimation_window="756 trading days",
        observation_count=756,
        run_date=date(2026, 9, 27),
        model_risk_level=model_risk_level,
        simulation_results=(
            SimulationResult(
                model_name="Gaussian",
                terminal_returns=[-0.01, -0.02, 0.01],
                portfolio_paths=[[1.0, 0.99, 1.01], [1.0, 0.98, 0.97]],
                random_seed=42,
                simulation_count=3,
                horizon=2,
            ),
            SimulationResult(
                model_name="Student-t",
                terminal_returns=[-0.03, -0.04, 0.02],
                portfolio_paths=[[1.0, 0.97, 0.96], [1.0, 0.99, 0.95]],
                random_seed=42,
                simulation_count=3,
                horizon=2,
            ),
        ),
        return_estimation=return_estimation,
    )


def _estimation() -> ReturnEstimation:
    return ReturnEstimation(
        tickers=("SPY", "TLT"),
        returns=[[0.01, -0.01]],
        mean_returns={"SPY": 0.001, "TLT": -0.001},
        volatilities={"SPY": 0.010, "TLT": 0.007},
        covariance_matrix=[[0.01, -0.02], [-0.02, 0.01]],
        correlation_matrix=[[1.0, -0.4], [-0.4, 1.0]],
        observation_count=756,
        estimation_window="756 trading days",
    )


class _AttrSpy:
    """Wraps any object and records every attribute access by name."""

    def __init__(self, wrapped: Any) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "accessed_attrs", set())

    def __getattr__(self, name: str) -> Any:
        object.__getattribute__(self, "accessed_attrs").add(name)
        return getattr(object.__getattribute__(self, "_wrapped"), name)


def _install_streamlit(monkeypatch: Any) -> StreamlitRecorder:
    recorder = StreamlitRecorder()
    monkeypatch.setitem(sys.modules, "streamlit", recorder)
    return recorder


def test_public_dashboard_functions_have_contract_docstrings() -> None:
    public_functions = [
        member
        for name, member in inspect.getmembers(dashboard, inspect.isfunction)
        if not name.startswith("_")
    ]

    assert public_functions
    for function in public_functions:
        docstring = inspect.getdoc(function)
        assert docstring is not None
        for section in ("Args:", "Returns:", "Raises:", "Invariants:"):
            assert section in docstring


def test_model_risk_banner_is_first_element_after_title(monkeypatch: Any) -> None:
    st = _install_streamlit(monkeypatch)

    dashboard.display_diagnostic_report(_report(model_risk_level="HIGH"), _config())

    # display_diagnostic_report does not call st.title() — that belongs to render_dashboard.
    # The first element it renders must be the verdict banner.
    assert st.calls, "display_diagnostic_report rendered nothing"
    # columns() call precedes the chip; verdict must appear within the first two calls.
    assert "HIGH" in str(st.calls[:3]), f"Expected verdict banner early; got {st.calls[:3]}"


def test_model_risk_banner_displays_each_supported_level(monkeypatch: Any) -> None:
    for level in ("HIGH", "MODERATE", "LOW"):
        st = _install_streamlit(monkeypatch)

        dashboard.display_diagnostic_report(_report(model_risk_level=level), _config())

        assert level in str(st.calls)


def test_var_and_es_ranges_are_read_from_comparison_rows(monkeypatch: Any) -> None:
    st = _install_streamlit(monkeypatch)
    report = _report()

    dashboard.render_model_risk_summary(report, _config())

    rendered = str(st.calls)
    assert "2.2 pp" in rendered
    assert "3.9 pp" in rendered


def test_loss_distribution_charts_use_terminal_returns(monkeypatch: Any) -> None:
    _install_streamlit(monkeypatch)
    base = _report()
    spies = [_AttrSpy(s) for s in base.simulation_results]
    report = dataclasses.replace(base, simulation_results=tuple(spies))

    dashboard.render_loss_distribution_comparison(report)

    for spy in spies:
        assert "terminal_returns" in spy.accessed_attrs, (
            f"render_loss_distribution_comparison did not access terminal_returns "
            f"on {spy._wrapped.model_name}"  # type: ignore[attr-defined]
        )


def test_drawdown_charts_use_portfolio_paths(monkeypatch: Any) -> None:
    _install_streamlit(monkeypatch)
    base = _report()
    spies = [_AttrSpy(s) for s in base.simulation_results]
    report = dataclasses.replace(base, simulation_results=tuple(spies))

    dashboard.render_drawdown_comparison(report)

    for spy in spies:
        assert "portfolio_paths" in spy.accessed_attrs, (
            f"render_drawdown_comparison did not access portfolio_paths "
            f"on {spy._wrapped.model_name}"  # type: ignore[attr-defined]
        )


def test_portfolio_structure_shows_weight_when_portfolio_weights_in_report(
    monkeypatch: Any,
) -> None:
    st = _install_streamlit(monkeypatch)
    report = dataclasses.replace(
        _report(return_estimation=_estimation()),
        portfolio_weights={"SPY": 0.60, "TLT": 0.40},
    )

    dashboard.render_portfolio_structure(report)

    rendered = str(st.calls)
    assert "60.0%" in rendered or "0.60" in rendered, (
        "weight must appear in portfolio structure table"
    )


def test_correlation_heatmap_uses_return_estimation(monkeypatch: Any) -> None:
    _install_streamlit(monkeypatch)
    spy = _AttrSpy(_estimation())
    report = dataclasses.replace(_report(), return_estimation=spy)

    dashboard.render_portfolio_structure(report)

    assert "correlation_matrix" in spy.accessed_attrs, (
        "render_portfolio_structure did not access return_estimation.correlation_matrix"
    )


def test_correlation_heatmap_placeholder_when_return_estimation_is_absent(
    monkeypatch: Any,
) -> None:
    st = _install_streamlit(monkeypatch)

    dashboard.render_portfolio_structure(_report(return_estimation=None))

    assert "placeholder" in str(st.calls).lower()


def test_scenario_table_reads_scenario_results_directly(monkeypatch: Any) -> None:
    st = _install_streamlit(monkeypatch)
    report = _report()

    dashboard.render_scenarios(report)

    rendered = str(st.calls)
    assert "Market Crash" in rendered
    assert "30.0%" in rendered


def test_custom_scenario_calls_only_scenario_boundary(monkeypatch: Any) -> None:
    st = _install_streamlit(monkeypatch)
    runner = ScenarioRunnerSpy()
    scenario_definition = ScenarioDefinition(
        name="Inline Shock",
        shocks=(ScenarioShock(ticker="SPY", shock=-0.10),),
    )

    dashboard.render_custom_scenario(runner, object(), scenario_definition)

    assert len(runner.calls) == 1
    assert "simulation" not in str(st.calls).lower()


def test_assumptions_expander_renders_all_analysis_config_parameters(
    monkeypatch: Any,
) -> None:
    st = _install_streamlit(monkeypatch)
    config = _config()

    dashboard.render_assumptions(config)

    rendered = str(st.calls)
    assert "Assumptions and configuration." in rendered
    assert "'expanded': False" in rendered
    for field in fields(AnalysisConfig):
        assert field.name in rendered


def test_render_order_is_exactly_the_ui_architecture_order(monkeypatch: Any) -> None:
    st = _install_streamlit(monkeypatch)

    dashboard.display_diagnostic_report(_report(return_estimation=_estimation()), _config())

    rendered = str(st.calls).lower()
    expected = [
        "model risk",
        "loss distribution",
        "drawdown",
        "portfolio structure",
        "scenario",
        "assumptions and configuration",
    ]
    positions = [rendered.index(label) for label in expected]
    assert positions == sorted(positions)


def test_dashboard_does_not_import_forbidden_analytical_modules() -> None:
    prohibited_modules = ("src.simulation", "src.risk", "src.returns", "src.scenarios")
    tree = ast.parse(Path("app/dashboard.py").read_text(), filename="app/dashboard.py")
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


def test_session_state_contains_only_config_and_report_keys() -> None:
    tree = ast.parse(Path("app/dashboard.py").read_text(), filename="app/dashboard.py")
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "session_state"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)

    assert keys <= {"analysis_config", "diagnostic_report"}
