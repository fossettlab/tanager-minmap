"""Unit tests for SAM and classification on synthetic data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tanager_rocks.speclib import Endmember
from tanager_rocks.unmix import matched_filter_maps, mtmf, sam_classify, spectral_angle

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


def test_matched_filter_scores_target_high():
    rng = np.random.default_rng(0)
    wl = np.linspace(2000.0, 2400.0, 6)
    mu_b = np.array([0.40, 0.42, 0.41, 0.43, 0.40, 0.42])
    delta = np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0])  # target signature
    n = 300
    px = mu_b + rng.normal(0.0, 0.01, size=(n, 6))
    px[0] = mu_b + delta  # one planted pure-target pixel
    cube = xr.DataArray(px.T[:, None, :], dims=("band", "y", "x"), coords={"band": wl})
    em = {"t": Endmember("t", "s", "ASD", wl, mu_b + delta)}
    mf = matched_filter_maps(cube, em)["t"].values  # (1, n)
    assert np.isclose(mf[0, 0], 1.0, atol=0.3)  # target pixel scores ~1
    assert mf[0, 0] > np.nanpercentile(mf[0, 1:], 99)  # well above background


def test_mtmf_infeasibility_separates_mixture_from_anomaly():
    rng = np.random.default_rng(1)
    wl = np.linspace(2000.0, 2400.0, 6)
    mu = np.array([0.40, 0.42, 0.41, 0.43, 0.40, 0.42])
    delta = np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0])  # target direction
    samples = mu + rng.normal(0.0, 0.01, size=(400, 6))
    samples[0] = mu + 0.5 * delta  # clean 50% target mixture (on the line)
    samples[1] = mu.copy()
    samples[1][5] += 0.10  # anomaly orthogonal to the target direction
    cube = xr.DataArray(samples.T[:, None, :], dims=("band", "y", "x"), coords={"band": wl})
    em = {"t": Endmember("t", "s", "ASD", wl, mu + delta)}
    ds = mtmf(cube, em, ridge=1e-3)
    mf = ds["t_mf"].values[0]
    inf = ds["t_infeas"].values[0]
    assert np.isclose(mf[0], 0.5, atol=0.2)  # mixture pixel abundance ~0.5
    assert inf[0] < inf[1]  # mixture is more feasible than the anomaly
