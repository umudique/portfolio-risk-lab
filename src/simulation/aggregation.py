"""Portfolio path aggregation contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class PortfolioPathAggregator:
    """Convert asset-level simulated returns into portfolio return paths.

    Args:
        asset_return_paths: A NumPy array whose final dimension corresponds to
            assets ordered exactly as ``Portfolio.tickers``.
        weights: Portfolio weights ordered exactly as the asset dimension.

    Returns:
        A NumPy array with the asset dimension removed. For every simulation
        and time step, the portfolio return is
        ``sum(weight_i * asset_return_i)``.

    Raises:
        ValueError: If the asset dimension does not match the number of weights
            or if inputs are not finite numeric arrays.

    Invariants:
        Aggregation performs only weighted sums of simulated asset returns. It
        never calculates VaR, Expected Shortfall, loss probabilities,
        drawdowns, or model-risk findings.
    """

    def aggregate(
        self,
        asset_return_paths: FloatArray,
        weights: Sequence[float],
    ) -> FloatArray:
        """Aggregate asset-level returns using portfolio weights.

        Args:
            asset_return_paths: Simulated asset returns with assets on the last
                dimension.
            weights: Portfolio weights aligned to that asset dimension.

        Returns:
            Weighted portfolio return paths.

        Raises:
            ValueError: If shapes or values are invalid.
        """
        asset_paths = np.asarray(asset_return_paths, dtype=np.float64)
        weight_values = np.asarray(weights, dtype=np.float64)

        if asset_paths.ndim < 2:
            raise ValueError("asset_return_paths must include path and asset dimensions")
        if asset_paths.shape[-1] != weight_values.shape[0]:
            raise ValueError("weights must match the asset dimension")
        if not np.all(np.isfinite(asset_paths)):
            raise ValueError("asset_return_paths must be finite")
        if not np.all(np.isfinite(weight_values)):
            raise ValueError("weights must be finite")

        return cast(FloatArray, np.einsum("...a,a->...", asset_paths, weight_values))
