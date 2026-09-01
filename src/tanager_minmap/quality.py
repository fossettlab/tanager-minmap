"""Authoritative quality masking for Planet Tanager surface reflectance.

The Tanager product embeds three beta usable-data masks and a per-band
``good_wavelengths`` flag.  This module owns their interpretation so every
analysis path applies the same scene-level policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from tanager_spec.bands import indices_in_windows
from tanager_spec.config import ABSORPTION_MASKS_NM, TANAGER_HDF5_GRID, TANAGER_SR_FIELD
from tanager_spec.mask import invalid_pixel_mask, require_band_y_x

logger = logging.getLogger(__name__)

QA_FIELDS = ("beta_cloud_mask", "beta_cirrus_mask", "nodata_pixels")
QA_ALLOWED_VALUES = frozenset({0, 1, 255})


@dataclass(frozen=True)
class TanagerQualityReport:
    """Counts recorded whenever the authoritative scene mask is applied."""

    total_pixels: int
    nodata_pixels: int
    cloud_pixels: int
    cirrus_pixels: int
    invalid_pixels: int
    product_bad_bands: int
    configured_window_bands: int
    retained_bands: int

    @property
    def invalid_fraction(self) -> float:
        """Fraction of spatial pixels excluded from analysis."""
        return self.invalid_pixels / self.total_pixels if self.total_pixels else float("nan")


def _data_fields_path(grid: str) -> str:
    return f"HDFEOS/GRIDS/{grid}/Data Fields"


def load_tanager_quality_metadata(
    path: str | Path,
    cube: xr.DataArray,
    wavelengths: np.ndarray,
    *,
    grid: str = TANAGER_HDF5_GRID,
    field: str = TANAGER_SR_FIELD,
) -> tuple[xr.Dataset, np.ndarray]:
    """Load embedded QA layers and product wavelength-validity flags.

    Unknown QA values, spatial mismatches, and wavelength mismatches raise
    rather than being interpreted silently.
    """
    cube = require_band_y_x(cube)
    expected_wavelengths = np.asarray(wavelengths, dtype=float)
    with h5py.File(path, "r") as handle:
        fields = handle[_data_fields_path(grid)]
        reflectance = fields[field]
        file_wavelengths = np.asarray(reflectance.attrs["wavelengths"], dtype=float)
        product_good = np.asarray(reflectance.attrs["good_wavelengths"], dtype=bool)
        qa_arrays = {name: np.asarray(fields[name][...]) for name in QA_FIELDS}

    if file_wavelengths.shape != expected_wavelengths.shape or not np.allclose(
        file_wavelengths,
        expected_wavelengths,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("HDF5 wavelength metadata does not match the loaded cube")
    if product_good.shape != expected_wavelengths.shape:
        raise ValueError("good_wavelengths length does not match the loaded cube")

    expected_shape = (cube.sizes["y"], cube.sizes["x"])
    for name, values in qa_arrays.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} shape {values.shape} != cube shape {expected_shape}")
        unknown = set(int(value) for value in np.unique(values)) - QA_ALLOWED_VALUES
        if unknown:
            raise ValueError(f"{name} contains undocumented QA values: {sorted(unknown)}")

    qa = xr.Dataset(
        {
            name: xr.DataArray(
                values,
                dims=("y", "x"),
                coords={"y": cube.coords["y"], "x": cube.coords["x"]},
            )
            for name, values in qa_arrays.items()
        }
    )
    return qa, product_good


def mask_tanager_scene(
    cube: xr.DataArray,
    wavelengths: np.ndarray,
    path: str | Path,
    *,
    windows: list[tuple[float, float]] | None = None,
) -> tuple[xr.DataArray, TanagerQualityReport]:
    """Apply Tanager pixel QA and spectral-channel exclusions.

    Pixels are excluded when any embedded beta mask is nonzero (cloud, cirrus,
    no-data, or the QA layer's own fill value) or the reflectance cube is
    non-finite. Spectral channels are excluded when Planet marks
    ``good_wavelengths == 0`` or when they fall in the project's existing
    atmospheric-absorption windows. No numeric reflectance clamp is applied:
    Planet describes surface reflectance as *typically* 0--1, not as a strict
    validity interval.
    """
    cube = require_band_y_x(cube)
    wl = np.asarray(wavelengths, dtype=float)
    if windows is None:
        windows = ABSORPTION_MASKS_NM

    qa, product_good = load_tanager_quality_metadata(path, cube, wl)
    qa_invalid = np.logical_or.reduce([np.asarray(qa[name].values) != 0 for name in QA_FIELDS])
    combined_qa = xr.DataArray(
        qa_invalid.astype(np.uint8),
        dims=("y", "x"),
        coords={"y": cube.coords["y"], "x": cube.coords["x"]},
    )
    invalid = invalid_pixel_mask(cube, qa=combined_qa, qa_valid_values=[0])

    configured_bad = indices_in_windows(wl, windows)
    retained = product_good & ~configured_bad
    out = cube.where(~invalid).astype(float)
    out.values[~retained, :, :] = np.nan

    report = TanagerQualityReport(
        total_pixels=cube.sizes["y"] * cube.sizes["x"],
        nodata_pixels=int((qa["nodata_pixels"].values != 0).sum()),
        cloud_pixels=int((qa["beta_cloud_mask"].values != 0).sum()),
        cirrus_pixels=int((qa["beta_cirrus_mask"].values != 0).sum()),
        invalid_pixels=int(invalid.sum()),
        product_bad_bands=int((~product_good).sum()),
        configured_window_bands=int(configured_bad.sum()),
        retained_bands=int(retained.sum()),
    )
    logger.info(
        "Tanager QA: excluded %d/%d pixels (%.2f%%); retained %d/%d bands",
        report.invalid_pixels,
        report.total_pixels,
        100 * report.invalid_fraction,
        report.retained_bands,
        wl.size,
    )
    return out, report


__all__ = [
    "QA_ALLOWED_VALUES",
    "QA_FIELDS",
    "TanagerQualityReport",
    "load_tanager_quality_metadata",
    "mask_tanager_scene",
]
