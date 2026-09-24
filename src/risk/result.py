"""Risk result building contract."""

from __future__ import annotations

import math

from src.domain import RiskResult


class RiskResultBuilder:
    """Build the canonical ``RiskResult`` domain object.

    Args:
        model_name: Name from the source ``SimulationResult``.
        confidence_level: External confidence level used for VaR/ES.
        var: Positive VaR loss magnitude.
        expected_shortfall: Positive ES loss magnitude.
        loss_threshold: External loss threshold used for loss probability.
        probability_loss_exceeds_threshold: Empirical probability that loss is
            strictly above ``loss_threshold``.
        max_drawdown_mean: Mean of simulated maximum drawdowns.
        max_drawdown_q95: 95th percentile of simulated maximum drawdowns.
        mean_simulated_loss: Mean terminal loss magnitude.
        worst_simulated_loss: Worst terminal loss magnitude.

    Returns:
        A canonical :class:`src.domain.RiskResult` with all fields populated.

    Raises:
        ValueError: If required values are missing, non-finite, or inconsistent
            with the positive-loss sign convention.

    Invariants:
        The builder assembles measured risk outputs only. It never runs
        simulation, calculates stochastic paths, or generates model-risk
        findings.
    """

    def build(
        self,
        *,
        model_name: str,
        confidence_level: float,
        var: float,
        expected_shortfall: float,
        loss_threshold: float,
        probability_loss_exceeds_threshold: float,
        max_drawdown_mean: float,
        max_drawdown_q95: float,
        mean_simulated_loss: float,
        worst_simulated_loss: float,
    ) -> RiskResult:
        """Build a canonical risk result.

        Args:
            model_name: Source simulation model name.
            confidence_level: VaR/ES confidence level.
            var: Positive VaR loss magnitude.
            expected_shortfall: Positive ES loss magnitude.
            loss_threshold: Loss probability threshold.
            probability_loss_exceeds_threshold: Empirical exceedance
                probability.
            max_drawdown_mean: Mean maximum drawdown.
            max_drawdown_q95: 95th percentile maximum drawdown.
            mean_simulated_loss: Mean terminal loss.
            worst_simulated_loss: Worst terminal loss.

        Returns:
            Canonical immutable ``RiskResult``.

        Raises:
            ValueError: If fields are invalid.
        """
        if not model_name:
            raise ValueError("model_name must not be empty")

        numeric_fields = {
            "confidence_level": confidence_level,
            "var": var,
            "expected_shortfall": expected_shortfall,
            "loss_threshold": loss_threshold,
            "probability_loss_exceeds_threshold": probability_loss_exceeds_threshold,
            "max_drawdown_mean": max_drawdown_mean,
            "max_drawdown_q95": max_drawdown_q95,
            "mean_simulated_loss": mean_simulated_loss,
            "worst_simulated_loss": worst_simulated_loss,
        }
        for field_name, value in numeric_fields.items():
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")

        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if not 0.0 <= probability_loss_exceeds_threshold <= 1.0:
            raise ValueError("probability_loss_exceeds_threshold must be in [0, 1]")

        non_negative_fields = {
            "var": var,
            "expected_shortfall": expected_shortfall,
            "loss_threshold": loss_threshold,
            "max_drawdown_mean": max_drawdown_mean,
            "max_drawdown_q95": max_drawdown_q95,
            "mean_simulated_loss": mean_simulated_loss,
            "worst_simulated_loss": worst_simulated_loss,
        }
        for field_name, value in non_negative_fields.items():
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")

        return RiskResult(
            model_name=model_name,
            confidence_level=confidence_level,
            var=var,
            expected_shortfall=expected_shortfall,
            loss_threshold=loss_threshold,
            probability_loss_exceeds_threshold=probability_loss_exceeds_threshold,
            max_drawdown_mean=max_drawdown_mean,
            max_drawdown_q95=max_drawdown_q95,
            mean_simulated_loss=mean_simulated_loss,
            worst_simulated_loss=worst_simulated_loss,
        )
