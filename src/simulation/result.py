"""Simulation result building contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from src.domain import SimulationResult


class ComparableArray:
    """NumPy-compatible array wrapper with scalar equality semantics."""

    def __init__(self, values: npt.ArrayLike) -> None:
        self._values = np.asarray(values)

    def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray[Any, Any]:
        if dtype is None:
            return self._values
        return np.asarray(self._values, dtype=dtype)

    def __eq__(self, other: object) -> bool:
        return bool(np.array_equal(self._values, np.asarray(other)))

    def __iter__(self) -> Any:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(self._values)


def _comparable_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ComparableArray(value)
    return value


class SimulationResultBuilder:
    """Build the canonical ``SimulationResult`` domain object.

    Args:
        model_name: Human-readable model identifier.
        terminal_returns: Portfolio-level terminal returns, one per simulation.
        portfolio_paths: Portfolio-level simulated return paths shaped
            ``(simulation_count, horizon)``. Asset-level bootstrap samples are
            stored in ``model_params["asset_paths"]`` rather than here.
        random_seed: Seed used to create all random draws.
        simulation_count: Number of generated simulation paths.
        horizon: Number of simulated return steps per path.
        model_params: Model-specific assumptions, such as Student-t degrees of
            freedom or bootstrap settings.

    Returns:
        A canonical :class:`src.domain.SimulationResult`.

    Raises:
        ValueError: If required metadata is missing or dimensionally
            inconsistent.

    Invariants:
        The builder standardizes simulation outputs only. It does not generate
        random draws, calculate risk metrics, or generate model-risk findings.
    """

    def build(
        self,
        *,
        model_name: str,
        terminal_returns: Any,
        portfolio_paths: Any | None,
        random_seed: int,
        simulation_count: int,
        horizon: int,
        model_params: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """Build a canonical simulation result.

        Args:
            model_name: Simulation model identifier.
            terminal_returns: Portfolio terminal return vector.
            portfolio_paths: Simulated path payload.
            random_seed: Reproducibility seed.
            simulation_count: Number of simulations.
            horizon: Simulation horizon.
            model_params: Model-specific assumptions and settings.

        Returns:
            Canonical immutable ``SimulationResult``.

        Raises:
            ValueError: If required fields are invalid.
        """
        if not model_name:
            raise ValueError("model_name must not be empty")
        if simulation_count <= 0:
            raise ValueError("simulation_count must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        return SimulationResult(
            model_name=model_name,
            terminal_returns=_comparable_payload(terminal_returns),
            portfolio_paths=_comparable_payload(portfolio_paths),
            random_seed=random_seed,
            simulation_count=simulation_count,
            horizon=horizon,
            model_params={} if model_params is None else model_params,
        )
