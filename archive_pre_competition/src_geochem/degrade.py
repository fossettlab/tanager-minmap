"""Spectral degradation: convolve Tanager-1 data to multispectral bandpasses.

Simulates what ASTER, Sentinel-2, and Landsat-8/9 would observe by convolving
the 426-band Tanager spectra with published spectral response functions (SRFs).

Typical usage::

    from src.degrade import load_srf, convolve_to_sensor

    srf = load_srf("aster")
    aster_bands = convolve_to_sensor(tanager_spectra, srf)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Supported sensors and their band counts
SUPPORTED_SENSORS = {
    "aster": {"n_bands": 9, "short_name": "ASTER"},
    "sentinel2": {"n_bands": 13, "short_name": "S2"},
    "landsat8": {"n_bands": 7, "short_name": "L8"},
    "landsat9": {"n_bands": 7, "short_name": "L9"},
}

SRF_DIR = Path("data/srf")


def load_srf(
    sensor: str,
    srf_dir: str | Path = SRF_DIR,
) -> dict[str, np.ndarray]:
    """Load spectral response functions for a multispectral sensor.

    Parameters
    ----------
    sensor : str
        Sensor name: ``"aster"``, ``"sentinel2"``, ``"landsat8"``, ``"landsat9"``.
    srf_dir : str or Path
        Directory containing SRF CSV files.

    Returns
    -------
    dict
        Keys are band names (e.g. ``"B1"``, ``"SWIR1"``), values are
        2D arrays of shape (n_wavelengths, 2) with columns
        [wavelength_nm, response].
    """
    # TODO: implement
    # 1. Validate sensor name
    # 2. Read SRF CSV from srf_dir / f"{sensor}_srf.csv"
    # 3. Parse into per-band response arrays
    raise NotImplementedError


def convolve_to_sensor(
    spectra: pd.DataFrame,
    srf: dict[str, np.ndarray],
    wavelengths: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convolve hyperspectral data to multispectral bandpasses.

    Applies the spectral response function to simulate what a
    multispectral sensor would observe from the same surface.

    Parameters
    ----------
    spectra : pd.DataFrame
        Hyperspectral data (rows = samples, columns = wavelengths in nm).
    srf : dict
        Spectral response functions from :func:`load_srf`.
    wavelengths : np.ndarray, optional
        Tanager band center wavelengths. If None, inferred from columns.

    Returns
    -------
    pd.DataFrame
        Simulated multispectral bands (rows = samples, columns = band names).
    """
    # TODO: implement
    # 1. Interpolate SRF to Tanager wavelength grid
    # 2. For each band: weighted average = sum(spectra * srf) / sum(srf)
    # 3. Return DataFrame with sensor band columns
    raise NotImplementedError


def degrade_all_sensors(
    spectra: pd.DataFrame,
    wavelengths: np.ndarray | None = None,
    srf_dir: str | Path = SRF_DIR,
) -> dict[str, pd.DataFrame]:
    """Convolve spectra to all supported multispectral sensors.

    Parameters
    ----------
    spectra : pd.DataFrame
        Hyperspectral data.
    wavelengths : np.ndarray, optional
        Tanager band center wavelengths.
    srf_dir : str or Path
        Directory containing SRF files.

    Returns
    -------
    dict
        Sensor name → degraded DataFrame.
    """
    # TODO: implement
    # 1. For each sensor in SUPPORTED_SENSORS, load SRF and convolve
    raise NotImplementedError
