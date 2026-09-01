"""Tests for cross-sensor comparison metrics on synthetic cubes/maps."""

from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from tanager_minmap.compare import (
    detection_agreement,
    mean_spectrum,
    resample_spectrum,
    spectral_agreement,
)


def test_resample_spectrum_linear_ramp():
    src = np.array([0.0, 10.0])
    spec = np.array([0.0, 1.0])  # ramp 0->1 over 0..10
    dst = np.array([5.0, 10.0, 20.0])
    out = resample_spectrum(spec, src, dst)
    assert np.isclose(out[0], 0.5)
    assert np.isclose(out[1], 1.0)
    assert np.isnan(out[2])  # outside source range -> NaN, not extrapolated


def _cube(band_nm, ny, nx, spectrum):
    data = np.broadcast_to(np.asarray(spectrum)[:, None, None], (len(band_nm), ny, nx)).astype(
        float
    )
    return xr.DataArray(
        data.copy(),
        dims=("band", "y", "x"),
        coords={"band": np.asarray(band_nm, float), "y": np.arange(ny), "x": np.arange(nx)},
    )


def test_mean_spectrum_ignores_nan():
    c = _cube([500, 600], 2, 2, [0.2, 0.4])
    c.values[0, 0, 0] = np.nan
    ms = mean_spectrum(c)
    assert np.allclose(ms, [0.2, 0.4])


def test_spectral_agreement_high_for_shared_shape():
    nm = np.linspace(500, 2400, 50)
    shape = 0.3 + 0.1 * np.sin(nm / 200.0)
    tan = _cube(nm, 4, 4, shape)
    emit = _cube(nm, 3, 3, shape * 1.05 + 0.01)  # scaled+offset but same shape
    agree, common_nm, tan_on, emit_mean = spectral_agreement(tan, nm, emit, nm)
    assert agree.pearson_r > 0.99
    assert agree.spectral_angle_deg < 5.0
    assert common_nm.size == emit_mean.size


def _map(values, crs="EPSG:4326"):
    ny, nx = values.shape
    da = xr.DataArray(
        values.astype(float),
        dims=("y", "x"),
        coords={"y": np.linspace(1.0, 0.0, ny), "x": np.linspace(0.0, 1.0, nx)},
    )
    return da.rio.write_crs(crs).rio.write_transform()


def test_detection_agreement_positive_for_aligned_maps():
    rng = np.random.default_rng(0)
    base = rng.random((8, 8))
    tan = xr.Dataset({"alunite": _map(base)})
    emit = xr.Dataset({"alunite": _map(base + 0.05 * rng.random((8, 8)))})
    out = detection_agreement(tan, emit, ["alunite", "absent"])
    assert "absent" not in out  # layers missing from a sensor are skipped
    assert out["alunite"].pearson_r > 0.8
    assert out["alunite"].n_pixels > 0
