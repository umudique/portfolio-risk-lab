"""Streamlit dashboard contracts for Portfolio Risk Lab.

The dashboard is a thin presentation and parameter-collection layer. It calls
the analysis orchestrator and displays a precomputed ``DiagnosticReport``; it
does not import analytical engines or recompute report content.
"""

from __future__ import annotations

import dataclasses
from io import StringIO
from typing import Any, Protocol

from src.domain import AnalysisConfig, DiagnosticReport, Portfolio, ScenarioDefinition

_CSS = """
<style>
@font-face {
    font-family: 'IBM Plex Sans';
    font-weight: 400;
    src: url('/app/static/fonts/IBMPlexSans-Regular.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Sans';
    font-weight: 500;
    src: url('/app/static/fonts/IBMPlexSans-Medium.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Sans';
    font-weight: 600;
    src: url('/app/static/fonts/IBMPlexSans-SemiBold.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Sans';
    font-weight: 700;
    src: url('/app/static/fonts/IBMPlexSans-Bold.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Mono';
    font-weight: 400;
    src: url('/app/static/fonts/IBMPlexMono-Regular.woff2') format('woff2');
}
@font-face {
    font-family: 'IBM Plex Mono';
    font-weight: 600;
    src: url('/app/static/fonts/IBMPlexMono-SemiBold.woff2') format('woff2');
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 400;
}
h1, .stTitle {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 700 !important;
}
h2, h3, h4,
[data-testid="stHeading"],
.stSubheader {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-testid="stCaption"],
code, pre, .stCodeBlock {
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"],
.metric-value {
    font-size: 1.55rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"],
.metric-label {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
.metric-value.bold { font-weight: 700 !important; }
table td, table th {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
table th {
    font-weight: 600 !important;
}
button[kind="secondary"] {
    padding: 0.1rem 0.5rem !important;
    font-size: 0.72rem !important;
    min-height: unset !important;
    line-height: 1.4 !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] strong {
    margin-bottom: 0 !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaption"] {
    margin-top: 0.1rem !important;
    margin-bottom: 0.35rem !important;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}
[data-testid="column"]:nth-child(2) {
    overflow-y: auto;
    max-height: 62vh;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stVerticalBlock"] > [data-testid="element-container"] {
    margin-bottom: -0.5rem !important;
}
[data-testid="stPlotlyChart"] {
    margin-top: -0.25rem !important;
}
</style>
"""

_VERDICT_COLORS = {
    "HIGH": "#b03535",
    "MODERATE": "#c4890a",
    "LOW": "#1e7a48",
}


class AnalysisOrchestrator(Protocol):
    """Protocol for the application-level analysis orchestrator.

    Args:
        portfolio_csv: User-uploaded portfolio CSV passed through by the UI.
        config: Canonical analysis configuration built from form inputs.

    Returns:
        A precomputed :class:`src.domain.DiagnosticReport` ready for display.

    Raises:
        ValueError: If upstream modules reject the portfolio, parameters, or
            scenario definitions.

    Invariants:
        The orchestrator is the only object the dashboard uses to start an
        analysis. The dashboard never calls simulation, returns, risk, or
        scenario modules directly.
    """

    def run(self, portfolio_csv: object, config: AnalysisConfig) -> DiagnosticReport:
        """Run the full analysis pipeline outside the dashboard.

        Args:
            portfolio_csv: Uploaded CSV object forwarded from the UI.
            config: Analysis configuration collected by the dashboard.

        Returns:
            Fully assembled diagnostic report.

        Raises:
            ValueError: If the analysis pipeline rejects the request.
        """
        ...


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "—"
    return f"{val:.1%}"


def _fmt_pp(val: Any) -> str:
    if val is None:
        return "—"
    return f"{val:+.1%} pp"


def _fmt_num(val: Any) -> str:
    if val is None:
        return "—"
    return f"{float(val):.2f}"


def _fmt_num3(val: Any) -> str:
    if val is None:
        return "—"
    return f"{float(val):.3f}"


def _metric_html(label: str, value: str) -> str:
    return (
        f"<div style=\"font-family:'IBM Plex Mono',monospace; min-width:7rem\">"
        f'<div style="font-size:0.68rem; color:#888; letter-spacing:0.07em;'
        f' margin-bottom:0.12rem">{label}</div>'
        f'<div style="font-size:1.3rem; font-weight:400; color:#222">{value}</div>'
        f"</div>"
    )


def render_dashboard(orchestrator: AnalysisOrchestrator) -> None:
    """Render the Streamlit dashboard and delegate analysis to the orchestrator.

    Args:
        orchestrator: Application orchestrator responsible for all analytical
            workflow coordination.

    Returns:
        ``None`` after rendering the interactive dashboard.

    Raises:
        ValueError: If UI-level validation rejects user input before handoff.

    Invariants:
        The dashboard contains no simulation mathematics, no risk metrics, no
        scenario calculations, and no return estimation. Session state stores
        only ``AnalysisConfig`` and ``DiagnosticReport`` values.
    """
    import streamlit as st

    from app.validation import AnalysisParameterValidator, PortfolioUploadValidator

    st.markdown(_CSS, unsafe_allow_html=True)

    if st.session_state.get("diagnostic_report") is not None:
        report = st.session_state["diagnostic_report"]
        config: Any = st.session_state.get("analysis_config")
        display_diagnostic_report(report, config)
        return

    _, form_col, _ = st.columns([1, 3, 1])
    with form_col:
        st.title("Portfolio Risk Lab")
        st.caption(
            "Upload a portfolio CSV and configure the analysis to see how sensitive "
            "your risk estimate is to modelling assumptions."
        )

        uploaded_csv = st.file_uploader("Portfolio CSV", type=["csv"])

        with st.form("analysis_parameters"):
            confidence_level = st.number_input(
                "Confidence level",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
            )
            horizon = st.number_input("Horizon", min_value=1, value=10, step=1)
            n_simulations = st.number_input("Simulations", min_value=1, value=10_000, step=1_000)
            loss_threshold = st.number_input("Loss threshold", min_value=0.0, value=0.10, step=0.01)
            random_seed = st.number_input("Random seed", value=42, step=1)
            submitted = st.form_submit_button(
                "Run analysis", type="primary", use_container_width=True
            )

    if not submitted:
        return

    if uploaded_csv is None:
        st.error("Upload a portfolio CSV before running analysis.")
        return

    text_stream = StringIO(uploaded_csv.getvalue().decode("utf-8"))
    PortfolioUploadValidator().validate(text_stream)
    config = AnalysisParameterValidator().validate(
        confidence_level=float(confidence_level),
        horizon=int(horizon),
        n_simulations=int(n_simulations),
        loss_threshold=float(loss_threshold),
        random_seed=int(random_seed),
    )
    st.session_state["analysis_config"] = config
    report = orchestrator.run(uploaded_csv, config)
    st.session_state["diagnostic_report"] = report
    st.rerun()


def display_diagnostic_report(report: DiagnosticReport, config: AnalysisConfig) -> None:
    """Display the report layers in the required model-risk-first order.

    Args:
        report: Diagnostic report assembled by the diagnostics layer.
        config: Analysis configuration that produced the report.

    Returns:
        ``None`` after rendering the six report layers.

    Raises:
        ValueError: If the supplied report is missing required display fields.

    Invariants:
        Rendering order is exactly: verdict banner, loss distributions,
        drawdown comparison, portfolio structure, scenarios, assumptions.
        Every displayed analytical value is read from ``DiagnosticReport`` or
        ``AnalysisConfig``. The dashboard performs no simulation, risk, return,
        or scenario mathematics.
    """
    import streamlit as st

    render_model_risk_summary(report, config)
    left, right = st.columns(2)
    with left:
        render_loss_distribution_comparison(report)
        render_drawdown_comparison(report)
    with right:
        render_portfolio_structure(report)
        render_scenarios(report)
    render_assumptions(config)


def render_model_risk_summary(report: DiagnosticReport, config: AnalysisConfig) -> None:
    """Render Layer 1: the model-risk verdict and comparison evidence.

    Args:
        report: Diagnostic report containing ``model_risk_level`` and
            ``comparison_rows`` supplied by the diagnostics layer.
        config: Analysis configuration used to render the report context line.

    Returns:
        ``None`` after rendering the full-width verdict banner, VaR/ES
        comparison strip, and precomputed VaR/ES range values.

    Raises:
        ValueError: If the report lacks a supported ``model_risk_level`` or
            comparison rows required for Layer 1.

    Invariants:
        ``DiagnosticReport.model_risk_level`` is the first report element after
        ``st.title()`` and must display HIGH, MODERATE, or LOW. VaR and ES
        ranges are read from ``DiagnosticReport.comparison_rows`` and are never
        recomputed in the dashboard.
    """
    import streamlit as st

    if report.model_risk_level not in {"HIGH", "MODERATE", "LOW"}:
        raise ValueError("diagnostic report must contain HIGH, MODERATE, or LOW model risk")

    comparison_rows = report.comparison_rows
    if not comparison_rows:
        raise ValueError("diagnostic report must contain comparison rows")

    _tickers = (
        " / ".join(report.return_estimation.tickers)
        if report.return_estimation is not None
        else "—"
    )
    color = _VERDICT_COLORS[report.model_risk_level]
    var_range = next(
        (row.get("value") for row in comparison_rows if row.get("metric") == "VaR range"),
        None,
    )
    es_range = next(
        (row.get("value") for row in comparison_rows if row.get("metric") == "ES range"),
        None,
    )
    loss_prob = next(
        (
            row.get(next((k for k in row if k != "metric"), ""))
            for row in comparison_rows
            if row.get("metric") == "P(loss > threshold)"
        ),
        None,
    )

    # MODEL RISK subtitle: find most conservative model by ES
    es_row = next((r for r in comparison_rows if r.get("metric") == "Expected Shortfall"), {})
    model_keys = [k for k in es_row if k != "metric" and k != "value"]
    most_conservative = (
        max(model_keys, key=lambda m: float(es_row.get(m) or 0)) if model_keys else None
    )
    subtitle = "Tail-risk estimates vary across model assumptions" + (
        f", with {most_conservative} producing the most conservative VaR and ES."
        if most_conservative
        else "."
    )

    c0, c1, c2, c3 = st.columns(4)
    c0.markdown(
        f'<div style="margin-top:0.25rem">'
        f'<div style="font-size:0.72rem; color:#888; margin-bottom:0.15rem;'
        f' letter-spacing:0.06em">MODEL RISK</div>'
        f'<div style="font-size:1.9rem; font-weight:700; color:{color};'
        f' line-height:1.1">{report.model_risk_level}</div>'
        f'<div style="font-size:0.72rem; color:#666; margin-top:0.25rem; line-height:1.3">'
        f"{subtitle}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    c1.metric(
        "VaR disagreement", f"{float(var_range) * 100:.1f} pp" if var_range is not None else "—"
    )
    c2.metric("ES disagreement", f"{float(es_range) * 100:.1f} pp" if es_range is not None else "—")
    c3.metric(f"P(loss > {_fmt_pct(config.loss_threshold)})", _fmt_pct(loss_prob))

    _ctx_parts = [
        _tickers,
        f"{int(config.confidence_level * 100)}% confidence",
        f"{config.horizon}-day horizon",
        f"{report.observation_count:,} observations",
        f"{config.n_simulations:,} simulations",
    ]
    ctx_col, btn_col = st.columns([10, 1])
    ctx_col.markdown(
        f'<div style="margin-top:0.6rem; font-size:0.8rem; color:#888">'
        f"{' · '.join(_ctx_parts)}</div>",
        unsafe_allow_html=True,
    )
    with btn_col:
        if st.button("← Reset", key="reset_analysis", use_container_width=True):
            st.session_state.pop("diagnostic_report", None)
            st.session_state.pop("analysis_config", None)
            st.rerun()

    tbl_col, chart_col = st.columns(2)

    tbl_col.markdown("**Model Comparison**")
    tbl_col.caption("VaR and ES at the configured confidence level, per model.")
    strip_rows = [
        {
            k: (_fmt_pct(v) if k != "metric" and isinstance(v, (int, float)) else v)
            for k, v in row.items()
        }
        for row in comparison_rows
        if row.get("metric") in {"VaR", "Expected Shortfall"}
    ]
    tbl_col.table(strip_rows)

    chart_col.markdown("**VaR / ES by Model**")
    chart_col.caption("Grouped comparison across all three models.")
    _render_model_comparison_chart(comparison_rows, chart_col)


def _render_model_comparison_chart(comparison_rows: list[Any], container: Any) -> None:
    import plotly.graph_objects as go

    var_row = next((r for r in comparison_rows if r.get("metric") == "VaR"), {})
    es_row = next((r for r in comparison_rows if r.get("metric") == "Expected Shortfall"), {})
    models = [k for k in var_row if k != "metric" and k != "value"]
    if not models:
        return

    _PALETTE = ["#6b8cae", "#c4890a", "#1e7a48"]
    fig = go.Figure()
    for idx, model in enumerate(models):
        color = _PALETTE[idx % len(_PALETTE)]
        var_val = var_row.get(model)
        es_val = es_row.get(model)
        fig.add_trace(
            go.Bar(
                name=model,
                x=["VaR", "ES"],
                y=[
                    float(var_val) if var_val is not None else 0,
                    float(es_val) if es_val is not None else 0,
                ],
                marker_color=color,
                text=[
                    _fmt_pct(float(var_val)) if var_val is not None else "",
                    _fmt_pct(float(es_val)) if es_val is not None else "",
                ],
                textposition="outside",
                textfont={"size": 9},
            )
        )

    fig.update_layout(
        barmode="group",
        height=220,
        margin={"t": 8, "b": 32, "l": 8, "r": 8},
        yaxis_tickformat=".0%",
        yaxis_showticklabels=False,
        xaxis_tickfont={"size": 11},
        legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"size": 9}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Mono, monospace", "size": 10},
    )
    container.plotly_chart(fig, use_container_width=True)


def render_loss_distribution_comparison(report: DiagnosticReport) -> None:
    """Render Layer 2: terminal loss distribution comparison charts.

    Args:
        report: Diagnostic report whose ``simulation_results`` entries expose
            model names and ``terminal_returns``.

    Returns:
        ``None`` after rendering all loss distribution charts.

    Raises:
        ValueError: If required simulation results or terminal returns are
            absent from the diagnostic report.

    Invariants:
        Charts use ``SimulationResult.terminal_returns`` from each entry in
        ``DiagnosticReport.simulation_results``. The dashboard never calls a
        simulation or risk module and performs no analytical recalculation.
    """
    import streamlit as st

    st.markdown(
        '<div style="margin-top:-6rem; margin-bottom:0.1rem">'
        "<strong>Loss Distributions</strong><br>"
        '<span style="font-size:0.8rem; color:#888">'
        "Do the three models agree on the shape of the loss distribution? "
        "Fatter tails indicate elevated tail risk. "
        "<em>Dashed line = VaR · Dotted line = ES</em>"
        "</span></div>",
        unsafe_allow_html=True,
    )
    import numpy as np
    import plotly.graph_objects as go

    _PALETTE = ["#6b8cae", "#c4890a", "#1e7a48"]
    comparison_rows = report.comparison_rows
    fig = go.Figure()

    for idx, sim_result in enumerate(report.simulation_results):
        color = _PALETTE[idx % len(_PALETTE)]
        arr = np.asarray(sim_result.terminal_returns)
        fig.add_trace(
            go.Histogram(
                x=arr,
                nbinsx=60,
                name=sim_result.model_name,
                marker_color=color,
                opacity=0.55,
            )
        )

    var_row = next((r for r in comparison_rows if r.get("metric") == "VaR"), {})
    es_row_dist = next((r for r in comparison_rows if r.get("metric") == "Expected Shortfall"), {})
    for idx, sim_result in enumerate(report.simulation_results):
        var_val = var_row.get(sim_result.model_name)
        es_val = es_row_dist.get(sim_result.model_name)
        color = _PALETTE[idx % len(_PALETTE)]
        if var_val is not None:
            fig.add_vline(x=-float(var_val), line_width=1, line_dash="dash", line_color=color)
        if es_val is not None:
            fig.add_vline(x=-float(es_val), line_width=1, line_dash="dot", line_color=color)

    fig.update_layout(
        barmode="overlay",
        height=280,
        margin={"t": 16, "b": 40, "l": 48, "r": 16},
        xaxis_title="Terminal Return",
        xaxis_tickformat=".0%",
        yaxis_title="Paths",
        legend={"orientation": "h", "y": 1.18, "x": 0, "font": {"size": 10}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Mono, monospace", "size": 10},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_comparison(report: DiagnosticReport) -> None:
    """Render Layer 3: drawdown distribution comparison charts.

    Args:
        report: Diagnostic report whose ``simulation_results`` entries expose
            model names and ``portfolio_paths``.

    Returns:
        ``None`` after rendering drawdown comparisons for each model.

    Raises:
        ValueError: If simulation results or portfolio paths needed for
            drawdown display are absent.

    Invariants:
        Drawdown charts read ``SimulationResult.portfolio_paths`` from
        ``DiagnosticReport.simulation_results``. The dashboard never derives
        paths or imports the simulation, risk, or returns engines.
    """
    import numpy as np
    import plotly.graph_objects as go
    import streamlit as st

    st.markdown(
        '<div style="margin-top:-3rem; margin-bottom:0.1rem">'
        "<strong>Maximum Drawdown by Model</strong><br>"
        '<span style="font-size:0.8rem; color:#888">'
        "Distribution of per-simulation maximum drawdown across all paths. "
        "Wider spread or heavier left tail signals higher path-level tail risk."
        "</span></div>",
        unsafe_allow_html=True,
    )

    _PALETTE = ["#6b8cae", "#c4890a", "#1e7a48"]
    fig = go.Figure()
    for idx, sim_result in enumerate(report.simulation_results):
        arr = np.asarray(sim_result.portfolio_paths)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        # portfolio_paths are period returns → convert to cumulative value first
        ones = np.ones((arr.shape[0], 1))
        cum = np.hstack([ones, np.cumprod(1.0 + arr, axis=1)])
        running_max = np.maximum.accumulate(cum, axis=1)
        per_path_dd = np.min(cum / running_max - 1.0, axis=1)
        color = _PALETTE[idx % len(_PALETTE)]
        fig.add_trace(
            go.Violin(
                y=per_path_dd,
                name=sim_result.model_name,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.6,
                line_color=color,
                points=False,
            )
        )

    fig.update_layout(
        height=280,
        margin={"t": 8, "b": 40, "l": 56, "r": 16},
        yaxis_title="Max Drawdown",
        yaxis_tickformat=".0%",
        showlegend=False,
        violingap=0.3,
        violinmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Mono, monospace", "size": 10},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_portfolio_structure(
    report: DiagnosticReport, portfolio: Portfolio | None = None
) -> None:
    """Render Layer 4: portfolio composition and correlation context.

    Args:
        report: Diagnostic report carrying ``return_estimation`` for
            mean-return, volatility, and correlation display.
        portfolio: Optional portfolio object supplied by the orchestrator for
            composition display when available.

    Returns:
        ``None`` after rendering portfolio structure content, or a placeholder
        when return estimation is unavailable.

    Raises:
        ValueError: If a supplied portfolio is malformed for display.

    Invariants:
        The correlation heatmap reads
        ``DiagnosticReport.return_estimation.correlation_matrix``. When
        ``return_estimation`` is ``None``, the dashboard renders a placeholder
        instead of computing or fetching estimation data.
    """
    import streamlit as st

    st.markdown("**Portfolio Structure**")
    st.caption(
        "Return estimation inputs used by all three models — composition, "
        "mean returns, volatility, and cross-asset correlation."
    )

    estimation = report.return_estimation
    if estimation is None:
        st.info("Portfolio structure placeholder: return estimation is unavailable.")
        return

    import numpy as np
    import plotly.graph_objects as go

    tickers = estimation.tickers
    vols = estimation.volatilities

    def _vol(key: str, idx: int) -> str:
        if isinstance(vols, dict):
            return _fmt_pct(vols[key])
        return _fmt_pct(float(np.asarray(vols).flat[idx]))

    weights = report.portfolio_weights
    if portfolio is not None:
        weights = dict(zip(portfolio.tickers, portfolio.weights))
    if weights is not None:
        rows = [
            {"ticker": t, "weight": _fmt_pct(weights.get(t, 0.0)), "volatility": _vol(t, i)}
            for i, t in enumerate(tickers)
        ]
    else:
        rows = [{"ticker": t, "volatility": _vol(t, i)} for i, t in enumerate(tickers)]
    st.table(rows)

    corr = np.asarray(estimation.correlation_matrix)
    n = len(tickers)
    fig = go.Figure(
        go.Heatmap(
            z=corr,
            x=list(tickers),
            y=list(tickers),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=[[f"{corr[i, j]:.2f}" for j in range(n)] for i in range(n)],
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=40 * n + 60,
        font={"family": "IBM Plex Mono, monospace", "size": 11},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"side": "bottom", "tickfont": {"size": 11}},
        yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_scenarios(report: DiagnosticReport) -> None:
    """Render Layer 5: deterministic scenario results and custom scenario entry.

    Args:
        report: Diagnostic report whose ``scenario_results`` contain the
            precomputed deterministic scenario table.

    Returns:
        ``None`` after rendering scenario results and any custom scenario UI.

    Raises:
        ValueError: If scenario result rows are malformed for presentation.

    Invariants:
        The scenario table reads ``DiagnosticReport.scenario_results``
        directly. The custom scenario entry may trigger only the scenario
        boundary, never the simulation engine or risk calculations.
    """
    import streamlit as st

    st.markdown("**Scenario Analysis**")
    st.caption(
        "Deterministic losses under named market shocks. "
        "These are mechanical calculations, independent of the simulation."
    )

    def _fmt_shocks(shocks: dict[str, float]) -> str:
        return " · ".join(f"{t} {v * 100:+.0f}%" for t, v in shocks.items())

    rows = [
        {
            "scenario": r.scenario_name,
            "shocks": _fmt_shocks(r.asset_shocks),
            "portfolio loss": _fmt_pct(r.portfolio_loss),
        }
        for r in report.scenario_results
    ]
    st.table(rows)


def render_custom_scenario(
    scenario_runner: Any,
    portfolio: Portfolio,
    scenario_definition: ScenarioDefinition,
) -> None:
    """Render and submit a custom deterministic scenario request.

    Args:
        scenario_runner: Scenario-only boundary used for the permitted partial
            recalculation.
        portfolio: Portfolio to shock deterministically.
        scenario_definition: User-entered scenario definition.

    Returns:
        ``None`` after displaying the custom scenario result.

    Raises:
        ValueError: If the scenario boundary rejects the custom scenario.

    Invariants:
        This function calls only a scenario boundary. It never imports or calls
        simulation, risk, return-estimation, or diagnostics engines.
    """
    import streamlit as st

    scenario_result = scenario_runner.run(portfolio, scenario_definition)
    st.write(
        {
            "scenario_name": scenario_result.scenario_name,
            "asset_shocks": scenario_result.asset_shocks,
            "portfolio_loss": scenario_result.portfolio_loss,
        }
    )


def render_assumptions(config: AnalysisConfig) -> None:
    """Render Layer 6: collapsed assumptions and configuration expander.

    Args:
        config: Canonical analysis configuration whose public fields must all
            be displayed.

    Returns:
        ``None`` after rendering the collapsed assumptions expander.

    Raises:
        ValueError: If the configuration object cannot be displayed.

    Invariants:
        The expander is labelled ``Assumptions and configuration.`` and is
        collapsed by default. Every parameter from ``AnalysisConfig`` is
        displayed directly from the config object with no business logic stored
        in Streamlit session state.
    """
    import streamlit as st

    st.markdown("**Assumptions**")
    st.caption("Analysis configuration used to produce this report.")
    with st.expander("Assumptions and configuration.", expanded=False):
        st.write(
            {
                field.name: getattr(config, field.name)
                for field in dataclasses.fields(AnalysisConfig)
            }
        )
