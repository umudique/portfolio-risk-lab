"""Diagnostic report building contracts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from src.domain import DiagnosticReport, ModelRiskFinding, ScenarioResult


class DiagnosticReportBuilder:
    """Assemble the canonical diagnostic report from pre-computed outputs.

    Args:
        comparison_rows: Rows produced by ``ModelComparisonTableBuilder``.
        findings: Evidence-based ``ModelRiskFinding`` entries.
        scenario_results: Deterministic scenario results from the scenarios
            module.
        estimation_window: Label or date range describing the estimation
            window used upstream.
        observation_count: Number of observations used in return estimation.
        run_date: Report run date.

    Returns:
        A canonical :class:`src.domain.DiagnosticReport`.

    Raises:
        ValueError: If required report fields are missing or invalid.

    Invariants:
        The builder assembles pre-computed risk and scenario outputs only. It
        never reruns simulation, recalculates returns, calls market data
        providers, or fabricates findings without numerical evidence.
    """

    def build(
        self,
        *,
        comparison_rows: list[dict[str, float | str]],
        findings: Sequence[ModelRiskFinding],
        scenario_results: Sequence[ScenarioResult],
        estimation_window: str,
        observation_count: int,
        run_date: date,
    ) -> DiagnosticReport:
        """Build a canonical diagnostic report.

        Args:
            comparison_rows: Aligned model comparison rows.
            findings: Evidence-backed model-risk findings.
            scenario_results: Deterministic scenario outputs.
            estimation_window: Upstream estimation-window metadata.
            observation_count: Upstream return-estimation observation count.
            run_date: Report date.

        Returns:
            Canonical immutable diagnostic report.

        Raises:
            ValueError: If report fields are invalid.
        """
        if len(comparison_rows) == 0:
            raise ValueError("comparison_rows must be non-empty")
        if estimation_window == "":
            raise ValueError("estimation_window must be non-empty")
        if observation_count < 0:
            raise ValueError("observation_count must be non-negative")

        return DiagnosticReport(
            comparison_rows=comparison_rows,
            findings=list(findings),
            scenario_results=list(scenario_results),
            estimation_window=estimation_window,
            observation_count=observation_count,
            run_date=run_date,
        )
