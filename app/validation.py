"""Surface-level dashboard input validation contracts.

The Streamlit layer performs only user-experience validation needed to catch
obvious form errors before invoking the analysis orchestrator. Analytical
validation remains owned by the corresponding ``src`` modules.
"""

from __future__ import annotations

import csv
import math
from typing import TextIO

from src.domain import AnalysisConfig


class PortfolioUploadValidator:
    """Validate portfolio CSV uploads at the dashboard boundary.

    Args:
        uploaded_csv: Text stream supplied by the Streamlit file uploader.

    Returns:
        ``None`` when the upload contains the minimal columns required to hand
        the file to the portfolio loader.

    Raises:
        ValueError: If the upload is empty, unreadable, or missing required
            surface-level columns such as ``ticker`` or ``weight``.

    Invariants:
        This validator does not canonicalize portfolios, normalize weights,
        estimate returns, call market-data providers, or perform simulation
        mathematics. Full analytical validation is delegated to
        ``src.portfolio`` via the analysis orchestrator.
    """

    def validate(self, uploaded_csv: TextIO) -> None:
        """Validate an uploaded portfolio CSV before orchestrator handoff.

        Args:
            uploaded_csv: Text stream containing user-uploaded CSV content.

        Returns:
            ``None`` when the upload passes surface-level checks.

        Raises:
            ValueError: If required dashboard-level CSV preconditions fail.
        """
        try:
            position = uploaded_csv.tell()
        except (AttributeError, OSError):
            position = None

        try:
            if position is not None:
                uploaded_csv.seek(0)
            reader = csv.DictReader(uploaded_csv)
            columns = set(reader.fieldnames or ())
        except csv.Error as exc:
            raise ValueError("portfolio CSV could not be read") from exc
        finally:
            if position is not None:
                uploaded_csv.seek(position)

        required_columns = {"ticker", "weight"}
        missing_columns = required_columns - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"portfolio CSV missing required columns: {missing}")


class AnalysisParameterValidator:
    """Validate analysis parameter form values at the dashboard boundary.

    Args:
        confidence_level: Requested confidence level; the UI requires a value
            strictly between 0 and 1.
        horizon: Requested simulation horizon; the UI requires a positive
            integer.
        n_simulations: Requested simulation count.
        loss_threshold: User-selected loss threshold for reporting.
        random_seed: Reproducibility seed passed through to ``AnalysisConfig``.

    Returns:
        A canonical :class:`src.domain.AnalysisConfig` suitable for passing to
        the analysis orchestrator.

    Raises:
        ValueError: If form values are outside basic UI bounds.

    Invariants:
        This validator performs no simulation, risk, return, scenario, or
        diagnostic calculations. It only packages user inputs for the
        orchestrator after surface-level checks.
    """

    def validate(
        self,
        *,
        confidence_level: float,
        horizon: int,
        n_simulations: int,
        loss_threshold: float,
        random_seed: int,
    ) -> AnalysisConfig:
        """Validate dashboard analysis settings.

        Args:
            confidence_level: Confidence level from the form.
            horizon: Positive analysis horizon from the form.
            n_simulations: Positive requested simulation count.
            loss_threshold: Loss threshold for risk reporting.
            random_seed: Random seed to preserve reproducibility.

        Returns:
            Canonical ``AnalysisConfig`` assembled from form inputs.

        Raises:
            ValueError: If any surface-level form validation fails.
        """
        if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be greater than 0 and less than 1")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if not math.isfinite(loss_threshold) or loss_threshold < 0:
            raise ValueError("loss_threshold must be non-negative")

        return AnalysisConfig(
            confidence_level=confidence_level,
            horizon=horizon,
            n_simulations=n_simulations,
            loss_threshold=loss_threshold,
            random_seed=random_seed,
        )
