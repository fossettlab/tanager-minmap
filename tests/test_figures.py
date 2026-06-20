"""Tests for the submission presentation figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from tanager_rocks.figures import (  # noqa: E402
    _nearest,
    _normalize,
    representative_spectra,
    rgb_context,
    spectra_story,
    validation_pair,
)
from tanager_rocks.speclib import Endmember  # noqa: E402


def _cube(values: np.ndarray, wl: np.ndarray) -> xr.DataArray:
    nb, ny, nx = values.shape
    return xr.DataArray(
        values.astype(float),
        dims=("band", "y", "x"),
        coords={"band": np.arange(nb), "y": np.arange(ny), "x": np.arange(nx) * 30.0},
    )


def test_nearest_band_index():
    wl = np.array([400.0, 500.0, 600.0, 700.0])
    assert _nearest(wl, 550.0) in (1, 2)
    assert _nearest(wl, 610.0) == 2


def test_normalize_robust_to_outlier():
    # a clean ramp plus two extreme outliers; the 2-98 percentile range should
    # trim the outliers so the ramp keeps its spread rather than collapsing.
    spec = np.concatenate([np.linspace(0.0, 1.0, 100), [100.0, 100.0]])
    out = _normalize(spec)
    assert np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0
    assert out[-1] == 1.0  # outliers clip to the top
    assert out[50] > 0.3  # mid-ramp value is not crushed toward zero


def test_representative_spectra_picks_top_pixels():
    # alunite abundance high only at pixel (0,0); its reflectance should be returned.
    wl = np.array([450.0, 550.0, 650.0])
    refl = np.zeros((3, 2, 2))
    refl[:, 0, 0] = [0.1, 0.2, 0.3]
    refl[:, 1, 1] = [0.9, 0.8, 0.7]
    cube = _cube(refl, wl)
    ds = xr.Dataset(
        {
            "alunite_mf": xr.DataArray([[1.0, 0.0], [0.0, 0.0]], dims=("y", "x")),
            "alunite_infeas": xr.DataArray(np.zeros((2, 2)), dims=("y", "x")),
        }
    )
    out = representative_spectra(cube, ds, ["alunite"], top_n=1, quantile=0.5)
    assert np.allclose(out["alunite"], [0.1, 0.2, 0.3])


def test_representative_spectra_skips_undetected():
    wl = np.array([450.0, 550.0, 650.0])
    cube = _cube(np.ones((3, 2, 2)), wl)
    ds = xr.Dataset(
        {
            "alunite_mf": xr.DataArray(np.zeros((2, 2)), dims=("y", "x")),
            "alunite_infeas": xr.DataArray(np.zeros((2, 2)), dims=("y", "x")),
        }
    )
    assert representative_spectra(cube, ds, ["alunite"]) == {}


def test_rgb_context_renders():
    wl = np.linspace(450.0, 700.0, 8)
    cube = _cube(np.random.default_rng(0).uniform(0.05, 0.4, (8, 6, 6)), wl)
    fig = rgb_context(cube, wl, title="test", scale_bar_m=None)
    assert fig.axes


def test_spectra_story_renders():
    wl = np.linspace(450.0, 2450.0, 50)
    em = Endmember("alunite", "s", "ASD", wl, np.linspace(0.2, 0.5, 50))
    fig = spectra_story(
        {"alunite": em},
        {"alunite": np.linspace(0.1, 0.4, 50)},
        wl,
        ["alunite"],
        absorptions={"Al-OH 2200": 2200.0},
    )
    assert fig.axes


def test_validation_pair_renders():
    score = xr.DataArray(np.random.default_rng(1).uniform(0, 1, (5, 5)), dims=("y", "x"))
    reference = xr.DataArray(np.array([[3, 3, 0, 1, 1]] * 5), dims=("y", "x"))
    fig = validation_pair(
        score, reference, frozenset({3}), mineral="alunite", title="t", excluded=frozenset({0})
    )
    assert len(fig.axes) >= 2
