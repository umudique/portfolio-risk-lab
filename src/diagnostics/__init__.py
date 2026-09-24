"""Diagnostics and model-risk comparison contracts."""

from src.diagnostics.comparison import ModelComparisonTableBuilder, TailRiskComparator
from src.diagnostics.differences import RelativeDifferenceCalculator
from src.diagnostics.findings import ModelRiskFindingGenerator
from src.diagnostics.report import DiagnosticReportBuilder

__all__ = [
    "DiagnosticReportBuilder",
    "ModelComparisonTableBuilder",
    "ModelRiskFindingGenerator",
    "RelativeDifferenceCalculator",
    "TailRiskComparator",
]
