"""Model comparison table contracts."""

from __future__ import annotations

from collections.abc import Sequence

from src.domain import RiskResult


def _validate_risk_results(risk_results: Sequence[RiskResult]) -> None:
    if len(risk_results) < 3:
        raise ValueError("at least three risk results are required")

    model_names = [result.model_name for result in risk_results]
    if any(name == "" for name in model_names):
        raise ValueError("model names must be non-empty")
    if len(set(model_names)) != len(model_names):
        raise ValueError("model names must be unique")


class ModelComparisonTableBuilder:
    """Align pre-computed model risk results into comparison rows.

    Args:
        risk_results: Pre-computed ``RiskResult`` objects, typically one each
            for Gaussian, Student-t, and historical bootstrap models.

    Returns:
        A list of row dictionaries suitable for the final report. Required
        primary rows are ``"VaR"``, ``"Expected Shortfall"``, and
        ``"P(loss > threshold)"``. Secondary drawdown rows must be clearly
        labeled, for example ``"Mean Max Drawdown"`` and
        ``"95th Percentile Max Drawdown"``.

    Raises:
        ValueError: If model names are missing, duplicated, or if fewer than
            the required model results are supplied.

    Invariants:
        The diagnostics layer consumes completed risk outputs only. It never
        reruns simulation, recalculates raw returns, calls market data sources,
        or modifies risk results.
    """

    def build(self, risk_results: Sequence[RiskResult]) -> list[dict[str, float | str]]:
        """Build aligned comparison rows.

        Args:
            risk_results: Pre-computed risk results to align by model name.

        Returns:
            Comparison table rows containing metric labels and per-model values.

        Raises:
            ValueError: If inputs cannot form a comparable table.
        """
        _validate_risk_results(risk_results)

        rows: list[dict[str, float | str]] = [
            {"metric": "VaR"},
            {"metric": "Expected Shortfall"},
            {"metric": "P(terminal loss > threshold)"},
            {"metric": "Mean Max Drawdown"},
            {"metric": "95th Percentile Max Drawdown"},
            {"metric": "Mean Simulated Loss"},
            {"metric": "Worst Simulated Loss"},
        ]
        for result in risk_results:
            rows[0][result.model_name] = result.var
            rows[1][result.model_name] = result.expected_shortfall
            rows[2][result.model_name] = result.probability_loss_exceeds_threshold
            rows[3][result.model_name] = result.max_drawdown_mean
            rows[4][result.model_name] = result.max_drawdown_q95
            rows[5][result.model_name] = result.mean_simulated_loss
            rows[6][result.model_name] = result.worst_simulated_loss

        var_vals = [r.var for r in risk_results]
        es_vals = [r.expected_shortfall for r in risk_results]
        rows.append({"metric": "VaR range", "value": round(max(var_vals) - min(var_vals), 4)})
        rows.append({"metric": "ES range", "value": round(max(es_vals) - min(es_vals), 4)})

        return rows


class TailRiskComparator:
    """Compare tail-risk metrics across pre-computed risk results.

    Args:
        risk_results: Pre-computed ``RiskResult`` objects.

    Returns:
        The model name with the highest Expected Shortfall.

    Raises:
        ValueError: If no risk results are supplied or if model names are
            missing.

    Invariants:
        This comparator reads already-computed metrics. It never calculates VaR
        or Expected Shortfall from simulation outputs and never reruns
        stochastic models.
    """

    def highest_expected_shortfall_model(self, risk_results: Sequence[RiskResult]) -> str:
        """Return the model with the largest Expected Shortfall.

        Args:
            risk_results: Pre-computed risk results.

        Returns:
            Model name with highest Expected Shortfall.

        Raises:
            ValueError: If inputs are invalid.
        """
        if len(risk_results) == 0:
            raise ValueError("at least one risk result is required")
        if any(result.model_name == "" for result in risk_results):
            raise ValueError("model names must be non-empty")

        return max(risk_results, key=lambda result: result.expected_shortfall).model_name
