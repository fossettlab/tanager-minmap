"""Spectral data extraction and preprocessing for Tanager-1 imagery.

Handles reading HDF5/ENVI hyperspectral data, extracting spectra at
sample locations, and masking atmospheric absorption bands.

Typical usage::

    from src.spectra import extract_spectra_at_points, mask_atmospheric_bands

    spectra = extract_spectra_at_points(scene_ds, sample_gdf)
    spectra_clean = mask_atmospheric_bands(spectra)
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Tanager-1 band parameters
N_BANDS = 426
WAVELENGTH_RANGE_NM = (380, 2500)

# Atmospheric absorption bands to mask (nm ranges)
ATMOSPHERIC_MASK_RANGES_NM = [
    (755, 770),    # O2 absorption
    (1350, 1450),  # H2O absorption
    (1800, 1950),  # H2O absorption
]


def get_wavelengths() -> np.ndarray:
    """Return Tanager-1 center wavelengths for all 426 bands.

    Returns
    -------
    np.ndarray
        Wavelengths in nanometers, shape (426,).
    """
    # TODO: implement
    # 1. Load from metadata or compute as linearly spaced
    #    (approximate; replace with actual wavelength table when available)
    raise NotImplementedError


def extract_spectra_at_points(
    scene: xr.DataArray,
    points: gpd.GeoDataFrame,
    buffer_m: float = 0.0,
) -> pd.DataFrame:
    """Extract spectral signatures at geochemistry sample locations.

    Parameters
    ----------
    scene : xr.DataArray
        Hyperspectral data cube with dims (band, y, x).
    points : gpd.GeoDataFrame
        Sample locations with geometry (Point) and sample_id column.
    buffer_m : float
        Buffer radius in meters around each point. If > 0, averages
        spectra within the buffer.

    Returns
    -------
    pd.DataFrame
        Rows = samples, columns = band wavelengths (nm).
        Index is sample_id.
    """
    # TODO: implement
    # 1. Reproject points to scene CRS if needed
    # 2. For each point, extract pixel value (or mean within buffer)
    # 3. Return DataFrame with wavelength columns
    raise NotImplementedError


def mask_atmospheric_bands(
    spectra: pd.DataFrame,
    wavelengths: np.ndarray | None = None,
    mask_ranges: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Remove atmospheric absorption bands from spectral data.

    Parameters
    ----------
    spectra : pd.DataFrame
        Spectral data with wavelength columns (nm).
    wavelengths : np.ndarray, optional
        Band center wavelengths. If None, inferred from column names.
    mask_ranges : list of (float, float), optional
        Wavelength ranges to mask. Defaults to ATMOSPHERIC_MASK_RANGES_NM.

    Returns
    -------
    pd.DataFrame
        Spectra with atmospheric bands removed.
    """
    # TODO: implement
    # 1. Identify columns within mask_ranges
    # 2. Drop those columns
    # 3. Log how many bands were removed
    raise NotImplementedError


def load_hdf5_scene(filepath: str) -> xr.DataArray:
    """Load a Tanager scene from HDF5 format.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.

    Returns
    -------
    xr.DataArray
        Hyperspectral data cube.
    """
    # TODO: implement
    # 1. Open with h5py
    # 2. Extract reflectance/radiance dataset
    # 3. Attach wavelength and spatial coordinates
    raise NotImplementedError
