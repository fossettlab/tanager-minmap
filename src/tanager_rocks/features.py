"""Continuum-removed diagnostic-absorption band-depth mapping.

Implements spec.md step 3 with the Clark & Roush (1984) continuum-removed
band-depth method: for each diagnostic absorption a straight continuum is drawn
between two shoulder wavelengths, and the band depth is ``1 - R_center / R_cont``.
This is the USGS Tetracorder-style per-feature approach, and it targets the
named absorptions directly (2200 nm Al-OH, 2265 nm jarosite, 2340 nm gypsum).

Feature definitions are NOT hard-coded here. A :class:`FeatureDef` carries its
shoulder wavelengths plus a ``source`` string; the shoulders are meant to be
derived from the reference spectral-library endmembers
(:mod:`tanager_rocks.speclib`) so they are data-driven, not invented. Input
cubes come from :func:`tanager_spec.io.load_tanager_sr_hdf5` after masking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class FeatureDef:
    """A diagnostic absorption: its center, two continuum shoulders, provenance.

    ``source`` records where the wavelengths came from (a library endmember or a
    citation) so no value in a map traces back to a guess.
    """

    name: str
    center_nm: float
    lo_shoulder_nm: float
    hi_shoulder_nm: float
    source: str

    def __post_init__(self) -> None:
        if not self.lo_shoulder_nm < self.center_nm < self.hi_shoulder_nm:
            raise ValueError(
                f"{self.name}: shoulders must bracket the center "
                f"({self.lo_shoulder_nm} < {self.center_nm} < {self.hi_shoulder_nm})"
            )


def _nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    """Index of the band whose center is nearest ``target_nm``."""
    return int(np.argmin(np.abs(np.asarray(wavelengths, dtype=float) - target_nm)))


def band_depth(
    cube: xr.DataArray,
    wavelengths: np.ndarray,
    feature: FeatureDef,
) -> xr.DataArray:
    """Continuum-removed band depth of one diagnostic absorption.

    The continuum at the center is the linear interpolation, in wavelength,
    between the reflectance at the two shoulder bands. Band depth is
    ``1 - R_center / R_continuum``: 0 where the feature is absent, larger where
    the absorption is deeper. Pixels with a non-positive continuum are ``NaN``.

    Parameters
    ----------
    cube : xr.DataArray
        Masked surface-reflectance cube, dims ``("band", "y", "x")``, with
        ``band`` aligned to ``wavelengths``.
    wavelengths : np.ndarray
        Band-center wavelengths (nm).
    feature : FeatureDef
        The absorption to map.

    Returns
    -------
    xr.DataArray
        Band depth, dims ``("y", "x")``, named ``feature.name``.
    """
    wl = np.asarray(wavelengths, dtype=float)
    lo_i = _nearest_band(wl, feature.lo_shoulder_nm)
    hi_i = _nearest_band(wl, feature.hi_shoulder_nm)
    c_i = _nearest_band(wl, feature.center_nm)

    r_lo = cube.isel(band=lo_i)
    r_hi = cube.isel(band=hi_i)
    r_c = cube.isel(band=c_i)

    # Linear continuum evaluated at the center band's actual wavelength.
    frac = (wl[c_i] - wl[lo_i]) / (wl[hi_i] - wl[lo_i])
    continuum = r_lo + (r_hi - r_lo) * frac

    depth = 1.0 - r_c / continuum.where(continuum > 0)
    return depth.rename(feature.name).drop_vars("band", errors="ignore")


def diagnostic_feature_maps(
    cube: xr.DataArray,
    wavelengths: np.ndarray,
    features: list[FeatureDef],
) -> xr.Dataset:
    """Band-depth map for every diagnostic feature, assembled into a Dataset.

    Parameters
    ----------
    cube : xr.DataArray
        Masked surface-reflectance cube, dims ``("band", "y", "x")``.
    wavelengths : np.ndarray
        Band-center wavelengths (nm).
    features : list of FeatureDef
        Diagnostic absorptions to map (shoulders sourced from the library).

    Returns
    -------
    xr.Dataset
        One band-depth variable per feature.
    """
    return xr.Dataset({f.name: band_depth(cube, wavelengths, f) for f in features})


def shoulders_from_endmember(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    center_nm: float,
    half_window_nm: float = 100.0,
) -> tuple[float, float]:
    """Derive continuum shoulders for an absorption from a reference spectrum.

    Within ``center_nm +/- half_window_nm`` the absorption is taken as the band
    of minimum reflectance, and each shoulder is the band of maximum reflectance
    on its side of that minimum. This makes the band-depth continuum data-driven
    (read off a library endmember) rather than a hand-picked pair of numbers.

    Parameters
    ----------
    wavelengths : np.ndarray
        Band centres (nm).
    reflectance : np.ndarray
        Reference reflectance aligned to ``wavelengths`` (may contain NaN).
    center_nm : float
        Nominal absorption center.
    half_window_nm : float
        Half-width of the search window.

    Returns
    -------
    tuple of float
        ``(lo_shoulder_nm, hi_shoulder_nm)``.
    """
    wl = np.asarray(wavelengths, dtype=float)
    refl = np.asarray(reflectance, dtype=float)
    in_window = (wl >= center_nm - half_window_nm) & (wl <= center_nm + half_window_nm)
    win = in_window & np.isfinite(refl)
    wl_w, refl_w = wl[win], refl[win]
    if wl_w.size < 3:
        raise ValueError(f"too few finite bands within +/-{half_window_nm} nm of {center_nm}")

    c_wl = wl_w[int(np.argmin(refl_w))]
    lo_side = wl_w <= c_wl
    hi_side = wl_w >= c_wl
    lo = float(wl_w[lo_side][int(np.argmax(refl_w[lo_side]))])
    hi = float(wl_w[hi_side][int(np.argmax(refl_w[hi_side]))])
    if not lo < center_nm < hi:
        raise ValueError(
            f"derived shoulders do not bracket {center_nm} (lo={lo}, hi={hi}); "
            "widen the window or check the endmember"
        )
    return lo, hi


def locate_feature(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    search_lo_nm: float,
    search_hi_nm: float,
) -> tuple[float, float, float]:
    """Locate an absorption (center + both shoulders) within a search window.

    Like :func:`shoulders_from_endmember` but for a feature whose center is not
    fixed in advance (e.g. the VNIR Fe-oxide band, whose position the spec does
    not pin): the center is the minimum-reflectance band in ``[search_lo_nm,
    search_hi_nm]`` and each shoulder is the maximum on its side.

    Returns
    -------
    tuple of float
        ``(center_nm, lo_shoulder_nm, hi_shoulder_nm)``.
    """
    wl = np.asarray(wavelengths, dtype=float)
    refl = np.asarray(reflectance, dtype=float)
    win = (wl >= search_lo_nm) & (wl <= search_hi_nm) & np.isfinite(refl)
    wl_w, refl_w = wl[win], refl[win]
    if wl_w.size < 3:
        raise ValueError(f"too few finite bands in [{search_lo_nm}, {search_hi_nm}] nm")

    center = float(wl_w[int(np.argmin(refl_w))])
    lo_side = wl_w <= center
    hi_side = wl_w >= center
    lo = float(wl_w[lo_side][int(np.argmax(refl_w[lo_side]))])
    hi = float(wl_w[hi_side][int(np.argmax(refl_w[hi_side]))])
    if not lo < center < hi:
        raise ValueError(
            f"located feature does not bracket its center (lo={lo}, center={center}, hi={hi})"
        )
    return center, lo, hi
