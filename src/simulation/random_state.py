"""Random-state management contract for reproducible simulation."""

from __future__ import annotations

import numpy as np


class RandomStateManager:
    """Create deterministic NumPy random generators from explicit seeds.

    Args:
        random_seed: Integer seed supplied by ``AnalysisConfig.random_seed``.

    Returns:
        A ``numpy.random.Generator`` initialized from the supplied seed.

    Raises:
        ValueError: If the seed is missing or cannot initialize a NumPy
            generator.

    Invariants:
        All stochastic simulation components must obtain randomness through
        this manager. The seed used to create the generator must be recorded in
        ``SimulationResult.random_seed``. This component does not simulate
        returns, aggregate portfolios, or calculate risk metrics.
    """

    def generator(self, random_seed: int) -> np.random.Generator:
        """Return a seeded NumPy random generator.

        Args:
            random_seed: Explicit deterministic seed.

        Returns:
            ``numpy.random.Generator`` seeded from ``random_seed``.

        Raises:
            ValueError: If the seed is invalid.
        """
        try:
            return np.random.default_rng(random_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("random_seed must initialize a NumPy generator") from exc
