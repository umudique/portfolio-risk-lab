"""Relative-difference calculation contracts."""

from __future__ import annotations

import math


class RelativeDifferenceCalculator:
    """Calculate relative differences between pre-computed metric values.

    Args:
        baseline: Baseline model metric value.
        comparison: Comparison model metric value.

    Returns:
        Relative difference ``(comparison - baseline) / baseline``. For
        example, Gaussian ES ``6.0`` and Student-t ES ``9.3`` produce ``0.55``.

    Raises:
        ValueError: If inputs are non-finite or if the baseline is zero.

    Invariants:
        This calculator operates only on completed scalar metrics. It never
        reruns simulation, computes risk metrics from raw returns, or generates
        narrative findings.
    """

    def calculate(self, *, baseline: float, comparison: float) -> float:
        """Calculate relative metric difference.

        Args:
            baseline: Baseline model value.
            comparison: Comparison model value.

        Returns:
            Relative difference as a decimal fraction.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not math.isfinite(baseline) or not math.isfinite(comparison):
            raise ValueError("baseline and comparison must be finite")
        if baseline == 0:
            raise ValueError("baseline must be non-zero")

        return (comparison - baseline) / baseline
