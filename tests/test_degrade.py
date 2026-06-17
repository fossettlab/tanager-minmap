"""Tests for spectral band ablation on synthetic SRFs and spectra."""

from __future__ import annotations

import numpy as np
import xarray as xr
from tanager_spec.srf import gaussian_srf

from tanager_rocks.degrade import (
    degrade_cube,
    degrade_spectra,
    pair_spectral_angle,
    separability,
    srf_band_stats,
)
from tanager_rocks.speclib import Endmember

# Source spectra grid and a two-band synthetic SRF well inside it.
_SRC = np.arange(400.0, 2401.0, 10.0)
_GRID = np.arange(400.0, 2401.0, 1.0)
_SRF = gaussian_srf(["b1", "b2"], np.array([1000.0, 2200.0]), np.array([50.0, 180.0]), _GRID)


def test_srf_band_stats_recovers_centers_and_fwhm():
    centers, fwhm = srf_band_stats(_SRF)
    assert np.allclose(centers, [1000.0, 2200.0], atol=2.0)
    assert np.allclose(fwhm, [50.0, 180.0], atol=4.0)


def test_degrade_spectra_preserves_flat_reflectance():
    # A flat 0.5 spectrum degrades to ~0.5 in every band.
    flat = np.full((1, _SRC.size), 0.5)
    out = degrade_spectra(flat, _SRC, _SRF)
    assert out.shape == (1, 2)
    assert np.allclose(out, 0.5, atol=1e-3)


def test_pair_spectral_angle_parallel_and_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert np.isclose(pair_spectral_angle(a, 2 * a), 0.0, atol=1e-6)
    assert np.isclose(pair_spectral_angle(a, b), np.pi / 2, atol=1e-6)


def test_degrade_cube_shape_coords_and_crs():
    cube = xr.DataArray(
        np.full((_SRC.size, 2, 3), 0.4),
        dims=("band", "y", "x"),
        coords={"band": _SRC, "y": [0, 1], "x": [0, 1, 2]},
    ).rio.write_crs("EPSG:32612")
    out = degrade_cube(cube, _SRC, _SRF)
    assert out.dims == ("band", "y", "x")
    assert out.shape == (2, 2, 3)
    assert list(out.band.values) == ["b1", "b2"]
    assert out.rio.crs.to_epsg() == 32612
    assert np.allclose(out.values, 0.4, atol=1e-3)


def test_separability_drops_when_band_distinguishing_feature_is_lost():
    # Two spectra differing only by an absorption notch near 2200 nm. Each SRF
    # has a common anchor band (1000 nm) plus a 2200 nm band that is either
    # narrow (resolves the notch) or broad (washes it out). With two bands the
    # spectral angle is well defined, and the narrow SRF must separate the pair
    # better than the broad one.
    base = np.full(_SRC.size, 0.5)
    notched = base.copy()
    notch = (_SRC >= 2185.0) & (_SRC <= 2215.0)
    notched[notch] = 0.2
    ema = Endmember("a", "s", "ASD", _SRC, base)
    emb = Endmember("b", "s", "ASD", _SRC, notched)
    broad = gaussian_srf(
        ["anchor", "swir"], np.array([1000.0, 2200.0]), np.array([50.0, 300.0]), _GRID
    )
    narrow = gaussian_srf(
        ["anchor", "swir"], np.array([1000.0, 2200.0]), np.array([50.0, 15.0]), _GRID
    )
    sep_broad = separability({"a": ema, "b": emb}, _SRC, broad, [("a", "b")])
    sep_narrow = separability({"a": ema, "b": emb}, _SRC, narrow, [("a", "b")])
    # Full-resolution angle identical; degraded angle larger for the narrow SRF.
    assert sep_narrow[("a", "b")][1] > sep_broad[("a", "b")][1]
