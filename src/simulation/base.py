"""Common simulation contract — ADR-003."""

from __future__ import annotations

from typing import Protocol

from src.domain import AnalysisConfig, Portfolio, ReturnEstimation, SimulationResult


class SimulationModel(Protocol):
    def simulate(
        self,
        portfolio: Portfolio,
        estimation: ReturnEstimation,
        config: AnalysisConfig,
    ) -> SimulationResult: ...
