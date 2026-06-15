"""Continuum removal and diagnostic-absorption feature mapping.

Implements spec.md pipeline step 3: continuum-remove the masked SR cube and
map the diagnostic absorptions that distinguish the alteration assemblage —
the 2200 nm Al-OH doublet (alunite vs kaolinite/dickite), 2265 nm jarosite,
2340 nm gypsum/carbonate, and the VNIR Fe-oxide features (hematite/goethite).

Input cubes come from :func:`tanager_spec.io.load_tanager_sr_hdf5` after
:mod:`tanager_spec.mask`; the wavelength axis comes from the same cube.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .config import DIAGNOSTIC_NM


def continuum_removed(cube: xr.DataArray, wavelengths_nm: np.ndarray) -> xr.DataArray:
    """Continuum-remove a reflectance cube along the spectral axis.

    Parameters
    ----------
    cube : xr.DataArray
        Masked surface-reflectance cube with dims (band, y, x).
    wavelengths_nm : np.ndarray
        Band centres (nm) aligned to the band dimension.

    Returns
    -------
    xr.DataArray
        Continuum-removed cube, same shape as ``cube``.
    """
    # TODO (spec step 3): upper-convex-hull continuum per pixel spectrum,
    # divide reflectance by the hull. Operate over valid (unmasked) bands only.
    raise NotImplementedError


def absorption_depth(
    cr_cube: xr.DataArray,
    wavelengths_nm: np.ndarray,
    center_nm: float,
    window_nm: float = 30.0,
) -> xr.DataArray:
    """Band-depth of a single diagnostic absorption.

    Parameters
    ----------
    cr_cube : xr.DataArray
        Continuum-removed cube from :func:`continuum_removed`.
    wavelengths_nm : np.ndarray
        Band centres (nm).
    center_nm : float
        Diagnostic absorption centre (see :data:`tanager_rocks.config.DIAGNOSTIC_NM`).
    window_nm : float
        Half-width of the search window around ``center_nm`` for the minimum.

    Returns
    -------
    xr.DataArray
        Per-pixel band depth (1 - min continuum-removed reflectance), dims (y, x).
    """
    # TODO (spec step 3): locate the continuum-removed minimum within the
    # window, return 1 - that value as band depth.
    raise NotImplementedError


def diagnostic_feature_maps(
    cube: xr.DataArray,
    wavelengths_nm: np.ndarray,
    diagnostics: dict[str, float] = DIAGNOSTIC_NM,
) -> xr.Dataset:
    """Compute band-depth maps for every diagnostic absorption.

    Parameters
    ----------
    cube : xr.DataArray
        Masked surface-reflectance cube (band, y, x).
    wavelengths_nm : np.ndarray
        Band centres (nm).
    diagnostics : dict
        Feature name -> centre wavelength (nm).

    Returns
    -------
    xr.Dataset
        One band-depth variable per diagnostic feature.
    """
    # TODO (spec step 3): continuum_removed once, then absorption_depth per
    # diagnostic; assemble into a Dataset for the hero/Goldfield figures.
    raise NotImplementedError
