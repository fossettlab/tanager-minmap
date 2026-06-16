"""Unit tests for SAM and classification on synthetic data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tanager_rocks.speclib import Endmember
from tanager_rocks.unmix import sam_classify, spectral_angle

_WL = np.array([1000.0, 1100.0, 1200.0])
# pixel 0 = (2,0,0) parallel to endmember (1,0,0); pixel 1 = (0,1,0) orthogonal.
_CUBE = xr.DataArray(
    np.array([[[2.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]]]),
    dims=("band", "y", "x"),
    coords={"band": _WL},
)
_EM = {"a": Endmember("a", "s", "ASD", _WL, np.array([1.0, 0.0, 0.0]))}


def test_spectral_angle_parallel_and_orthogonal():
    ang = spectral_angle(_CUBE, _EM)["a"].values
    assert np.isclose(ang[0, 0], 0.0, atol=1e-6)  # parallel -> 0 rad
    assert np.isclose(ang[0, 1], np.pi / 2, atol=1e-6)  # orthogonal -> pi/2


def test_sam_classify_threshold():
    angles = spectral_angle(_CUBE, _EM)
    classes, minerals = sam_classify(angles, max_angle_rad=0.1)
    assert minerals == ["a"]
    assert classes.values[0, 0] == 0  # within threshold -> class 0
    assert classes.values[0, 1] == -1  # angle pi/2 > 0.1 -> unclassified
