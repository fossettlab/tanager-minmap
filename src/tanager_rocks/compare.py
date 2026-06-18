"""Cross-sensor comparison metrics: Tanager vs EMIT (spec.md step 6).

Given the *same* alteration-mapping products (diagnostic band depths, MTMF
abundance) computed independently on a Tanager scene and an overlapping EMIT
scene, this module quantifies the three things spec step 6 asks for:

- **Spectral correlation** — do the two spectrometers see the same reflectance
  shape over the shared ground? Compared on the scene-mean spectrum, resampled
  to a common wavelength axis.
- **Mineral-detection agreement** — do the per-mineral MTMF maps agree
  spatially? The finer Tanager map is reprojected onto the coarser EMIT grid
  (no upsampling) and correlated pixel-for-pixel.
- **Spatial detail** — the resolution ratio, reported honestly (Tanager 30 m
  vs EMIT 60 m → 4× the pixel density, a smaller minimum mappable feature).

The maps themselves come from the existing `features`/`unmix` pipeline run on
each sensor; nothing here recomputes mineralogy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from rasterio.enums import Resampling


def resample_spectrum(spectrum: np.ndarray, src_nm: np.ndarray, dst_nm: np.ndarray) -> np.ndarray:
    """Linearly resample a spectrum from ``src_nm`` to ``dst_nm`` (NaN-aware)."""
    finite = np.isfinite(spectrum)
    return np.interp(dst_nm, src_nm[finite], spectrum[finite], left=np.nan, right=np.nan)


def mean_spectrum(cube: xr.DataArray) -> np.ndarray:
    """Scene-mean reflectance spectrum, ``(band,)``, ignoring NaN."""
    return np.nanmean(cube.values.reshape(cube.sizes["band"], -1), axis=1)


def _pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Pearson r over the finite overlap of two vectors; also return the count."""
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan"), n
    return float(np.corrcoef(a[m], b[m])[0, 1]), n


def _spectral_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral angle (degrees) over the finite overlap of two spectra."""
    m = np.isfinite(a) & np.isfinite(b)
    av, bv = a[m], b[m]
    if av.size < 2:
        return float("nan")
    cos = float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@dataclass(frozen=True)
class SpectralAgreement:
    """Scene-mean spectral agreement between two sensors on a common axis."""

    pearson_r: float
    spectral_angle_deg: float
    n_bands: int


def spectral_agreement(
    tan_cube: xr.DataArray,
    tan_nm: np.ndarray,
    emit_cube: xr.DataArray,
    emit_nm: np.ndarray,
) -> tuple[SpectralAgreement, np.ndarray, np.ndarray, np.ndarray]:
    """Compare the two sensors' scene-mean spectra on the coarser (EMIT) axis.

    Returns the agreement metrics plus ``(common_nm, tan_mean_on_common,
    emit_mean)`` for plotting. Tanager's finer spectrum is resampled to EMIT's
    285-band axis; bands outside either sensor's finite range are dropped.
    """
    tan_mean = mean_spectrum(tan_cube)
    emit_mean = mean_spectrum(emit_cube)
    tan_on_emit = resample_spectrum(tan_mean, np.asarray(tan_nm, float), np.asarray(emit_nm, float))
    r, _ = _pearson(tan_on_emit, emit_mean)
    angle = _spectral_angle_deg(tan_on_emit, emit_mean)
    both = np.isfinite(tan_on_emit) & np.isfinite(emit_mean)
    agree = SpectralAgreement(pearson_r=r, spectral_angle_deg=angle, n_bands=int(both.sum()))
    return agree, np.asarray(emit_nm, float), tan_on_emit, emit_mean


@dataclass(frozen=True)
class DetectionAgreement:
    """Spatial agreement of one mineral's MTMF map between two sensors."""

    mineral: str
    pearson_r: float
    n_pixels: int


def reproject_to(score: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Reproject a score map onto ``like``'s grid (bilinear; continuous data)."""
    return score.rio.reproject_match(like, resampling=Resampling.bilinear)


def reproject_crs(score: xr.DataArray, dst_crs, resolution: float | None = None) -> xr.DataArray:
    """Reproject a score map to ``dst_crs`` (optionally at a target resolution).

    Used to put the two sensors' maps in the same CRS for side-by-side display
    while keeping each at its own ground sampling — so Tanager's finer grain
    stays visible rather than being resampled onto EMIT's coarser grid.
    """
    return score.rio.reproject(dst_crs, resolution=resolution, resampling=Resampling.bilinear)


def detection_agreement(
    tan_scores: xr.Dataset,
    emit_scores: xr.Dataset,
    minerals: list[str],
) -> dict[str, DetectionAgreement]:
    """Per-mineral spatial correlation of MTMF maps on the common (EMIT) grid.

    Each Tanager mineral map is reprojected onto the EMIT grid and correlated
    with EMIT's own map over the pixels finite in both. A positive correlation
    means the two independent sensors light up the same ground for that mineral.
    """
    out: dict[str, DetectionAgreement] = {}
    for m in minerals:
        if m not in tan_scores or m not in emit_scores:
            continue
        tan_on_emit = reproject_to(tan_scores[m], emit_scores[m])
        r, n = _pearson(tan_on_emit.values.ravel(), emit_scores[m].values.ravel())
        out[m] = DetectionAgreement(mineral=m, pearson_r=r, n_pixels=n)
    return out


__all__ = [
    "SpectralAgreement",
    "DetectionAgreement",
    "resample_spectrum",
    "mean_spectrum",
    "spectral_agreement",
    "detection_agreement",
    "reproject_to",
]
