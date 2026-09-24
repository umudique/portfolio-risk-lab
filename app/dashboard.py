"""Streamlit dashboard contract for Portfolio Risk Lab.

The dashboard is a thin presentation and parameter-collection layer. It calls
the analysis orchestrator and displays a precomputed ``DiagnosticReport``; it
does not import analytical engines or recompute report content.
"""

from __future__ import annotations

from typing import Protocol

from src.domain import AnalysisConfig, DiagnosticReport


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

    st.title("Portfolio Risk Lab")
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
        submitted = st.form_submit_button("Run analysis")

    if not submitted:
        return

    if uploaded_csv is None:
        st.error("Upload a portfolio CSV before running analysis.")
        return

    text_stream = uploaded_csv.getvalue().decode("utf-8")
    from io import StringIO

    PortfolioUploadValidator().validate(StringIO(text_stream))
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
    display_diagnostic_report(report)


def display_diagnostic_report(report: DiagnosticReport) -> None:
    """Display a precomputed diagnostic report without recomputation.

    Args:
        report: Diagnostic report assembled by the diagnostics layer.

    Returns:
        ``None`` after rendering report sections.

    Raises:
        ValueError: If the supplied report is missing required display fields.

    Invariants:
        Comparison rows, model-risk findings, and scenario results are read
        directly from ``DiagnosticReport``. The dashboard never recomputes,
        reformats, or merges analytical outputs.
    """
    import streamlit as st

    if len(report.comparison_rows) == 0:
        raise ValueError("diagnostic report must contain comparison rows")

    st.header("Risk Comparison")
    st.dataframe(report.comparison_rows, hide_index=True)

    st.header("Model Risk Findings")
    for finding in report.findings:
        st.write(finding.finding_text)

    st.header("Scenario Results")
    st.dataframe(report.scenario_results, hide_index=True)

    st.caption(f"Estimation window: {report.estimation_window}")
    st.caption(f"Observation count: {report.observation_count}")
    st.caption(f"Run date: {report.run_date.isoformat()}")
