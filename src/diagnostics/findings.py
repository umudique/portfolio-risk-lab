"""Model-risk finding generation contracts."""

from __future__ import annotations

import math

from src.diagnostics.differences import RelativeDifferenceCalculator
from src.domain import ModelRiskFinding


class ModelRiskFindingGenerator:
    """Generate evidence-based model-risk findings from metric differences.

    Args:
        metric: Name of the compared metric, such as ``"Expected Shortfall"``.
        model_a: Baseline model name.
        model_b: Comparison model name.
        value_a: Baseline metric value.
        value_b: Comparison metric value.
        materiality_threshold: Absolute relative-difference threshold required
            to generate a finding. The MVP default is ``0.10`` (10%).

    Returns:
        A ``ModelRiskFinding`` when the absolute relative difference is greater
        than or equal to ``materiality_threshold``; otherwise ``None``.

    Raises:
        ValueError: If metric/model names are empty, values are non-finite, or
            the threshold is negative.

    Invariants:
        Every generated finding must cite metric, both model names, absolute
        difference, and relative difference. The generator must not emit
        narrative without numerical evidence, rerun simulation, or recalculate
        risk metrics.
    """

    def generate(
        self,
        *,
        metric: str,
        model_a: str,
        model_b: str,
        value_a: float,
        value_b: float,
        materiality_threshold: float = 0.10,
    ) -> ModelRiskFinding | None:
        """Generate a material model-risk finding when warranted.

        Args:
            metric: Compared metric name.
            model_a: Baseline model name.
            model_b: Comparison model name.
            value_a: Baseline metric value.
            value_b: Comparison metric value.
            materiality_threshold: Minimum absolute relative difference.

        Returns:
            Evidence-backed finding or ``None`` if immaterial.

        Raises:
            ValueError: If inputs are invalid.
        """
        if metric == "":
            raise ValueError("metric must be non-empty")
        if model_a == "" or model_b == "":
            raise ValueError("model names must be non-empty")
        if not math.isfinite(value_a) or not math.isfinite(value_b):
            raise ValueError("metric values must be finite")
        if not math.isfinite(materiality_threshold) or materiality_threshold < 0:
            raise ValueError("materiality_threshold must be a non-negative finite value")

        relative_difference = RelativeDifferenceCalculator().calculate(
            baseline=value_a,
            comparison=value_b,
        )
        if abs(relative_difference) < materiality_threshold:
            return None

        rounded_absolute_difference = round(abs(value_b - value_a), 4)
        rounded_relative_difference = round(relative_difference, 4)
        percent_difference = abs(rounded_relative_difference) * 100.0

        return ModelRiskFinding(
            finding_text=(
                f"{model_b} {metric} differs from {model_a} by {percent_difference:.1f}%"
            ),
            metric=metric,
            model_a=model_a,
            model_b=model_b,
            absolute_difference=rounded_absolute_difference,
            relative_difference=rounded_relative_difference,
        )
