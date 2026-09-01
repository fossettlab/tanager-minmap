"""Tests for the authoritative Tanager scene-quality policy."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from tanager_rocks.quality import load_tanager_quality_metadata, mask_tanager_scene


def _cube() -> tuple[xr.DataArray, np.ndarray]:
    wavelengths = np.array([500.0, 760.0, 1400.0, 2000.0])
    cube = xr.DataArray(
        np.ones((4, 2, 3), dtype=np.float32),
        dims=("band", "y", "x"),
        coords={"band": wavelengths, "y": [15.0, 5.0], "x": [5.0, 15.0, 25.0]},
    )
    return cube.rio.write_crs("EPSG:32611"), wavelengths


def _write_quality_file(path: Path, wavelengths: np.ndarray, *, unknown: bool = False) -> None:
    with h5py.File(path, "w") as handle:
        fields = handle.create_group("HDFEOS/GRIDS/HYP/Data Fields")
        reflectance = fields.create_dataset(
            "surface_reflectance", data=np.ones((4, 2, 3), dtype=np.float32)
        )
        reflectance.attrs["wavelengths"] = wavelengths
        reflectance.attrs["good_wavelengths"] = np.array([1, 1, 0, 1], dtype=np.uint8)
        cloud = np.zeros((2, 3), dtype=np.uint8)
        cloud[0, 1] = 7 if unknown else 1
        cirrus = np.zeros((2, 3), dtype=np.uint8)
        cirrus[1, 0] = 1
        nodata = np.zeros((2, 3), dtype=np.uint8)
        nodata[1, 2] = 1
        fields.create_dataset("beta_cloud_mask", data=cloud)
        fields.create_dataset("beta_cirrus_mask", data=cirrus)
        fields.create_dataset("nodata_pixels", data=nodata)


def test_mask_tanager_scene_applies_qa_and_band_union(tmp_path: Path):
    cube, wavelengths = _cube()
    path = tmp_path / "scene.h5"
    _write_quality_file(path, wavelengths)

    masked, report = mask_tanager_scene(cube, wavelengths, path)

    # QA-invalid pixels are removed through the full spectral axis.
    assert np.isnan(masked[:, 0, 1]).all()
    assert np.isnan(masked[:, 1, 0]).all()
    assert np.isnan(masked[:, 1, 2]).all()
    # The configured 760/1400-nm windows and the product-bad 1400-nm flag
    # combine, leaving only the 500- and 2000-nm channels in this fixture.
    assert np.isnan(masked.sel(band=760.0)).all()
    assert np.isnan(masked.sel(band=1400.0)).all()
    np.testing.assert_allclose(masked.sel(band=500.0).values[[0, 1], [0, 1]], [1.0, 1.0])
    np.testing.assert_allclose(masked.sel(band=2000.0).values[[0, 1], [0, 1]], [1.0, 1.0])
    assert report.invalid_pixels == 3
    assert report.retained_bands == 2


def test_quality_metadata_rejects_unknown_qa_value(tmp_path: Path):
    cube, wavelengths = _cube()
    path = tmp_path / "scene.h5"
    _write_quality_file(path, wavelengths, unknown=True)

    with pytest.raises(ValueError, match="undocumented QA values"):
        load_tanager_quality_metadata(path, cube, wavelengths)


def test_quality_metadata_rejects_wavelength_mismatch(tmp_path: Path):
    cube, wavelengths = _cube()
    path = tmp_path / "scene.h5"
    _write_quality_file(path, wavelengths + 0.5)

    with pytest.raises(ValueError, match="wavelength metadata"):
        load_tanager_quality_metadata(path, cube, wavelengths)
