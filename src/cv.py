"""Cross-validation strategies for spatial geochemistry data.

Implements Leave-One-Scene-Out (LOSO) and spatial block CV to address
spatial autocorrelation. Ordinary k-fold provided as comparison baseline.

Typical usage::

    from src.cv import LeaveOneSceneOut, spatial_block_cv

    loso = LeaveOneSceneOut(scene_labels=df["scene_id"])
    for train_idx, test_idx in loso.split(X, y):
        model.fit(X[train_idx], y[train_idx])
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


class LeaveOneSceneOut(BaseCrossValidator):
    """Leave-One-Scene-Out cross-validator.

    Each split holds out all samples from one Tanager scene as the test
    set. This is the primary CV strategy to address spatial autocorrelation.

    Parameters
    ----------
    scene_labels : array-like
        Scene identifier for each sample (e.g. scene_id or acquisition date).
    """

    def __init__(self, scene_labels: np.ndarray | pd.Series) -> None:
        self.scene_labels = np.asarray(scene_labels)

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return number of splits (= number of unique scenes)."""
        return len(np.unique(self.scene_labels))

    def _iter_test_indices(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ):
        """Yield test indices for each scene."""
        # TODO: implement
        # 1. For each unique scene label, yield indices where label matches
        raise NotImplementedError


def spatial_block_cv(
    lats: np.ndarray,
    lons: np.ndarray,
    n_splits: int = 5,
    block_size_deg: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate spatial block cross-validation splits.

    Divides the study area into spatial blocks and assigns blocks to
    folds, ensuring nearby samples are in the same fold.

    Parameters
    ----------
    lats : np.ndarray
        Sample latitudes.
    lons : np.ndarray
        Sample longitudes.
    n_splits : int
        Number of CV folds.
    block_size_deg : float
        Block size in degrees.

    Returns
    -------
    list of (train_indices, test_indices)
    """
    # TODO: implement
    # 1. Create spatial grid with block_size_deg
    # 2. Assign each sample to a block
    # 3. Assign blocks to folds (round-robin or clustering)
    # 4. Yield train/test index arrays
    raise NotImplementedError


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Compute R² and RMSE with bootstrap confidence intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    n_bootstrap : int
        Number of bootstrap iterations.
    ci : float
        Confidence interval width (0–1).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"r2": (median, lower, upper), "rmse": (median, lower, upper)}``
    """
    # TODO: implement
    # 1. Resample with replacement n_bootstrap times
    # 2. Compute R² and RMSE for each resample
    # 3. Return median and CI bounds
    raise NotImplementedError
