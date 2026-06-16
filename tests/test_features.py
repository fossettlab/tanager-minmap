"""Unit tests for continuum-removed band depth on synthetic spectra."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tanager_rocks.features import (
    FeatureDef,
    band_depth,
    diagnostic_feature_maps,
    shoulders_from_endmember,
)

# Five bands; a triangular absorption at 2200 nm between flat 0.5 shoulders.
WL = np.array([2100.0, 2150.0, 2200.0, 2250.0, 2300.0])
# pixel 0 has the absorption (0.4 at center); pixel 1 is flat.
_DATA = np.array(
    [
        [[0.5, 0.5]],
        [[0.5, 0.5]],
        [[0.4, 0.5]],
        [[0.5, 0.5]],
        [[0.5, 0.5]],
    ]
)
CUBE = xr.DataArray(_DATA, dims=("band", "y", "x"), coords={"band": WL})
FEATURE = FeatureDef(
    "aloh_test", center_nm=2200, lo_shoulder_nm=2100, hi_shoulder_nm=2300, source="synthetic"
)


def test_band_depth_matches_hand_calc():
    bd = band_depth(CUBE, WL, FEATURE)
    assert bd.dims == ("y", "x")
    # continuum at center = 0.5 (flat shoulders); depth = 1 - 0.4/0.5 = 0.2.
    assert np.isclose(bd.values[0, 0], 0.2)
    # flat pixel: no absorption.
    assert np.isclose(bd.values[0, 1], 0.0)


def test_shoulders_must_bracket_center():
    with pytest.raises(ValueError, match="bracket the center"):
        FeatureDef("bad", center_nm=2200, lo_shoulder_nm=2300, hi_shoulder_nm=2100, source="x")


def test_diagnostic_feature_maps_assembles_dataset():
    ds = diagnostic_feature_maps(CUBE, WL, [FEATURE])
    assert "aloh_test" in ds
    assert np.isclose(ds["aloh_test"].values[0, 0], 0.2)


def test_shoulders_from_endmember_picks_bracketing_maxima():
    wl = np.array([2100.0, 2130.0, 2160.0, 2200.0, 2240.0, 2270.0, 2300.0])
    refl = np.array([0.50, 0.60, 0.55, 0.40, 0.55, 0.62, 0.50])  # min at 2200
    lo, hi = shoulders_from_endmember(wl, refl, center_nm=2200.0, half_window_nm=100.0)
    assert (lo, hi) == (2130.0, 2270.0)
