"""Model-risk verdict calculation contract."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.domain import RiskResult


class ModelRiskVerdictCalculator:
    """Classify model-risk severity from completed risk results.

    Args:
        risk_results: Precomputed model risk results, usually one each for
            Gaussian, Student-t, and Historical Bootstrap.

    Returns:
        ``"HIGH"``, ``"MODERATE"``, ``"LOW"``, or ``"UNKNOWN"``.

    Raises:
        ValueError: If supplied risk results contain invalid metric values.

    Invariants:
        This diagnostics component reads completed ``RiskResult`` values only.
        It never reruns simulation, calculates VaR or Expected Shortfall from
        raw paths, imports risk/simulation/app modules, or mutates results. The
        thresholds must match ``docs/ui_architecture.md``: VaR range greater
        than 3 percentage points or ES range greater than 5 percentage points is
        HIGH; VaR range from 1 to 3 percentage points or ES range from 2 to 5
        percentage points is MODERATE; VaR range below 1 percentage point and
        ES range below 2 percentage points is LOW; fewer than two results is
        UNKNOWN.
    """

    def calculate(self, risk_results: Sequence[RiskResult]) -> str:
        """Calculate the model-risk verdict from precomputed risk results.

        Args:
            risk_results: Sequence of completed ``RiskResult`` objects.

        Returns:
            One of ``"HIGH"``, ``"MODERATE"``, ``"LOW"``, or ``"UNKNOWN"``.

        Raises:
            ValueError: If risk results contain invalid model names or
                non-finite risk metrics.

        Invariants:
            The calculation belongs to diagnostics because it is a model-risk
            domain decision. The dashboard and pipeline must consume this
            verdict rather than duplicating threshold logic.
        """
        if len(risk_results) < 2:
            return "UNKNOWN"

        for result in risk_results:
            if result.model_name == "":
                raise ValueError("model names must be non-empty")
            if not math.isfinite(result.var) or not math.isfinite(result.expected_shortfall):
                raise ValueError("risk metrics must be finite")

        var_values = [result.var for result in risk_results]
        es_values = [result.expected_shortfall for result in risk_results]
        var_range = max(var_values) - min(var_values)
        es_range = max(es_values) - min(es_values)

        if var_range > 0.03 or es_range > 0.05:
            return "HIGH"
        if var_range >= 0.01 or es_range >= 0.02:
            return "MODERATE"
        return "LOW"
