"""Spectral band ablation: degrade Tanager to a coarser sensor (spec.md step 5).

The novelty lever. Tanager's full VSWIR reflectance is convolved to Sentinel-2's
bands with published spectral response functions (`tanager_spec.srf`), and the
loss is quantified — chiefly that S2's single broad SWIR band at ~2200 nm (B12)
integrates the entire 2100-2280 nm Al-OH region, collapsing the doublet that
separates alunite from kaolinite. SRF convolution and the S2 response tables
live in the shared data layer; this module reshapes between the ``(band, y, x)``
cube and ``simulate``'s ``(pixel, band)`` contract, degrades the reference
endmembers the same way (so maps and library stay comparable), and measures
pairwise spectral-angle separability in each sensor's band space.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from tanager_spec.srf import SpectralResponse, simulate

from .speclib import Endmember


def srf_band_stats(srf: SpectralResponse) -> tuple[np.ndarray, np.ndarray]:
    """Response-weighted band centers and FWHM (nm), one per band.

    Center is the response-weighted mean wavelength; FWHM is the span of the
    contiguous wavelengths whose response is at least half the band's peak.
    """
    wl = srf.wavelength_nm
    resp = srf.response  # (n_band, n_grid)
    centers = (wl[None, :] * resp).sum(axis=1) / resp.sum(axis=1)
    fwhm = np.empty(resp.shape[0])
    for i, r in enumerate(resp):
        above = wl[r >= 0.5 * r.max()]
        fwhm[i] = float(above.max() - above.min()) if above.size else np.nan
    return centers, fwhm


def degrade_spectra(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    srf: SpectralResponse,
    min_coverage: float = 0.5,
) -> np.ndarray:
    """Convolve ``(n, n_source_band)`` reflectance to the SRF's target bands."""
    return simulate(np.atleast_2d(spectra), np.asarray(wavelengths, float), srf, min_coverage)


def degrade_cube(
    cube: xr.DataArray,
    wavelengths: np.ndarray,
    srf: SpectralResponse,
    min_coverage: float = 0.5,
) -> xr.DataArray:
    """Degrade a ``(band, y, x)`` SR cube to the SRF's target bands.

    Returns a cube on the same spatial grid with one layer per target band; the
    ``band`` coordinate carries the target band names. Spatial metadata (CRS,
    transform) is preserved when present.
    """
    data = cube.transpose("band", "y", "x").values
    nb, ny, nx = data.shape
    out = degrade_spectra(data.reshape(nb, ny * nx).T, wavelengths, srf, min_coverage)
    degraded = out.T.reshape(out.shape[1], ny, nx)
    da = xr.DataArray(
        degraded,
        dims=("band", "y", "x"),
        coords={"band": list(srf.band_names), "y": cube.y, "x": cube.x},
    )
    if cube.rio.crs is not None:
        da = da.rio.write_crs(cube.rio.crs).rio.write_transform(cube.rio.transform())
    return da


def degrade_endmembers(
    endmembers: dict[str, Endmember],
    wavelengths: np.ndarray,
    srf: SpectralResponse,
    min_coverage: float = 0.5,
) -> dict[str, np.ndarray]:
    """Degrade each endmember's reflectance to the SRF's target bands."""
    names = list(endmembers)
    mat = np.vstack([endmembers[n].reflectance for n in names])
    deg = degrade_spectra(mat, wavelengths, srf, min_coverage)
    return {n: deg[i] for i, n in enumerate(names)}


def pair_spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral angle (radians) between two spectra over their shared finite bands.

    Larger angle = more separable. Returns NaN if fewer than two shared bands.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    av, bv = a[m], b[m]
    cos = float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def separability(
    endmembers: dict[str, Endmember],
    wavelengths: np.ndarray,
    srf: SpectralResponse,
    pairs: list[tuple[str, str]],
    min_coverage: float = 0.5,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Per-pair spectral angle (radians) in full vs. degraded band space.

    Returns ``{(a, b): (full_angle, degraded_angle)}``. A large drop from full to
    degraded means the coarser sensor cannot separate that mineral pair.
    """
    deg = degrade_endmembers(endmembers, wavelengths, srf, min_coverage)
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for a, b in pairs:
        full = pair_spectral_angle(endmembers[a].reflectance, endmembers[b].reflectance)
        coarse = pair_spectral_angle(deg[a], deg[b])
        out[(a, b)] = (full, coarse)
    return out
