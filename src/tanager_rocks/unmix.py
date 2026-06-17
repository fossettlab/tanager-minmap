"""Spectral unmixing: SAM and MTMF against the reference library.

Implements spec.md pipeline step 4. MTMF (a covariance-aware matched filter)
is the primary method because the methodology suite found Tanager's
information lives in covariance-aware statistics; SAM is the band-independent
comparison. Both consume endmembers from :mod:`tanager_rocks.speclib` and a
masked SR cube from the :mod:`tanager_spec` data layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .speclib import Endmember


def spectral_angle(cube: xr.DataArray, endmembers: dict[str, Endmember]) -> xr.Dataset:
    """Spectral Angle Mapper score per endmember.

    For each endmember the per-pixel spectral angle is
    ``arccos(<R, e> / (|R| |e|))`` over the bands valid in both the cube and the
    endmember. Smaller = better match. Nodata pixels (NaN across the cube) come
    back as ``NaN``.

    Parameters
    ----------
    cube : xr.DataArray
        Masked surface-reflectance cube (band, y, x).
    endmembers : dict
        Mineral -> :class:`tanager_rocks.speclib.Endmember`, resampled to the
        cube's wavelength axis.

    Returns
    -------
    xr.Dataset
        One spectral-angle variable per mineral (radians).
    """
    data = cube.transpose("band", "y", "x").values
    # Bands carrying data somewhere (drops the all-NaN absorption-masked bands).
    band_has_data = np.isfinite(data).any(axis=(1, 2))

    out: dict[str, xr.DataArray] = {}
    for mineral, em in endmembers.items():
        valid = band_has_data & np.isfinite(em.reflectance)
        cube_v = data[valid]  # (nb, y, x)
        e = em.reflectance[valid]  # (nb,)
        dot = np.tensordot(e, cube_v, axes=(0, 0))  # (y, x)
        cube_norm = np.sqrt((cube_v**2).sum(axis=0))
        cos = dot / (cube_norm * np.linalg.norm(e))
        angle = np.arccos(np.clip(cos, -1.0, 1.0))
        out[mineral] = xr.DataArray(angle, dims=("y", "x"), coords={"y": cube.y, "x": cube.x})
    return xr.Dataset(out)


def sam_classify(angles: xr.Dataset, max_angle_rad: float) -> tuple[xr.DataArray, list[str]]:
    """Assign each pixel to its best-matching mineral, gated by a max angle.

    Parameters
    ----------
    angles : xr.Dataset
        Per-mineral spectral-angle maps from :func:`spectral_angle`.
    max_angle_rad : float
        Pixels whose smallest angle exceeds this are left unclassified (-1).
        This is a tunable acceptance threshold, not a physical constant.

    Returns
    -------
    classes : xr.DataArray
        Integer class codes (index into ``minerals``; -1 = unclassified / NaN).
    minerals : list of str
        Class labels in code order.
    """
    minerals = list(angles.data_vars)
    stack = np.stack([angles[m].values for m in minerals], axis=0)  # (mineral, y, x)
    best = np.nanargmin(np.where(np.isnan(stack), np.inf, stack), axis=0)
    min_angle = np.nanmin(stack, axis=0)
    codes = np.where(np.isfinite(min_angle) & (min_angle <= max_angle_rad), best, -1)
    classes = xr.DataArray(codes, dims=("y", "x"), coords={"y": angles.y, "x": angles.x})
    return classes, minerals


@dataclass
class _Background:
    """Shared background statistics for matched filtering on a cube's valid data.

    The full-band VSWIR covariance is ill-conditioned (collinear adjacent bands),
    so ``cov_inv`` is the inverse of the diagonally-loaded covariance.
    """

    valid_b: np.ndarray  # (n_band,) bool — bands used
    px_valid: np.ndarray  # (ny*nx,) bool — finite pixels
    centered: np.ndarray  # (n_px, n_valid_band) — pixels minus background mean
    mu: np.ndarray  # (n_valid_band,) background mean
    cov_inv: np.ndarray  # (n_valid_band, n_valid_band) inverse loaded covariance
    ny: int
    nx: int


def _background(cube: xr.DataArray, endmembers: dict[str, Endmember], ridge: float) -> _Background:
    data = cube.transpose("band", "y", "x").values
    n_band, ny, nx = data.shape
    flat = data.reshape(n_band, ny * nx)

    band_has_data = np.isfinite(flat).any(axis=1)
    em_finite = np.all([np.isfinite(e.reflectance) for e in endmembers.values()], axis=0)
    valid_b = band_has_data & em_finite
    px_valid = np.isfinite(flat[valid_b]).all(axis=0)

    samples = flat[valid_b][:, px_valid].T  # (n_px, n_valid_band)
    mu = samples.mean(axis=0)
    centered = samples - mu
    cov = centered.T @ centered / (samples.shape[0] - 1)
    cov += ridge * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])  # diagonal loading
    return _Background(valid_b, px_valid, centered, mu, np.linalg.inv(cov), ny, nx)


def _to_map(values: np.ndarray, bg: _Background, coords: dict) -> xr.DataArray:
    full = np.full(bg.ny * bg.nx, np.nan)
    full[bg.px_valid] = values
    return xr.DataArray(full.reshape(bg.ny, bg.nx), dims=("y", "x"), coords=coords)


def matched_filter_maps(
    cube: xr.DataArray,
    endmembers: dict[str, Endmember],
    ridge: float = 1e-2,
) -> xr.Dataset:
    """Covariance-aware matched-filter abundance per endmember.

    For each endmember ``t`` the matched-filter score per pixel ``x`` is
    ``(t - mu)^T C^-1 (x - mu) / (t - mu)^T C^-1 (t - mu)`` against the scene's
    own background mean ``mu`` and band covariance ``C``: ``1`` at the target
    spectrum, ``0`` at the background mean, so larger = more target-like. Unlike
    SAM this uses the full band covariance, which is where the spec expects
    Tanager's signal to live; it is the abundance half of MTMF.

    ``C`` is stabilised by diagonal loading (``C + ridge * mean(diag(C)) * I``),
    a numerical parameter, not physical. All-NaN (absorption-masked) bands and
    bands where an endmember is non-finite are dropped; nodata pixels are ``NaN``.

    Parameters
    ----------
    cube : xr.DataArray
        Masked surface-reflectance cube (band, y, x).
    endmembers : dict
        Mineral -> :class:`tanager_rocks.speclib.Endmember`.
    ridge : float
        Diagonal-loading fraction applied to the covariance before inversion.

    Returns
    -------
    xr.Dataset
        One matched-filter abundance variable per mineral.
    """
    bg = _background(cube, endmembers, ridge)
    coords = {"y": cube.y, "x": cube.x}
    out: dict[str, xr.DataArray] = {}
    for mineral, em in endmembers.items():
        d = em.reflectance[bg.valid_b] - bg.mu
        weight = bg.cov_inv @ d
        alpha = bg.centered @ weight / float(d @ weight)  # =1 at target, 0 at background
        out[mineral] = _to_map(alpha, bg, coords)
    return xr.Dataset(out)


def mtmf(
    cube: xr.DataArray,
    endmembers: dict[str, Endmember],
    ridge: float = 1e-2,
) -> xr.Dataset:
    """Mixture-tuned matched filter: abundance + infeasibility per endmember.

    Builds on :func:`matched_filter_maps`. In the background-whitened metric
    (``C^-1``) the matched filter explains the pixel's component along the target
    direction with abundance ``alpha``; the *infeasibility* is the magnitude of
    everything left over — the residual orthogonal to the target mixing
    direction — in background-sigma units:

        ``infeas(x)^2 = (x-mu)^T C^-1 (x-mu) - alpha^2 (t-mu)^T C^-1 (t-mu)``

    (the RX anomaly score minus the matched-filter-explained part), divided by
    ``sqrt(n_band - 1)``. A true sub-pixel occurrence has high ``alpha`` *and* low
    infeasibility; a false positive has a large residual the mixture model cannot
    explain (high infeasibility). This is the operational MTMF feasibility check
    (Boardman 1998) implemented from the whitened residual — not ENVI's exact
    (unpublished) normalisation. The absolute scale is not unit-variance (the
    diagonal loading shrinks the whitening), so gate by the infeasibility
    distribution, not a fixed sigma.

    Parameters
    ----------
    cube : xr.DataArray
        Masked surface-reflectance cube (band, y, x).
    endmembers : dict
        Mineral -> :class:`tanager_rocks.speclib.Endmember`.
    ridge : float
        Diagonal-loading fraction applied to the covariance before inversion.

    Returns
    -------
    xr.Dataset
        ``<mineral>_mf`` (abundance) and ``<mineral>_infeas`` per mineral.
    """
    bg = _background(cube, endmembers, ridge)
    coords = {"y": cube.y, "x": cube.x}
    n_dim = int(bg.valid_b.sum())
    # RX anomaly score (x-mu)^T C^-1 (x-mu), computed once for all endmembers.
    whitened = bg.centered @ bg.cov_inv  # (n_px, n_band)
    rx = np.einsum("ij,ij->i", whitened, bg.centered)
    norm = np.sqrt(max(n_dim - 1, 1))

    out: dict[str, xr.DataArray] = {}
    for mineral, em in endmembers.items():
        d = em.reflectance[bg.valid_b] - bg.mu
        weight = bg.cov_inv @ d
        eta = float(d @ weight)
        alpha = bg.centered @ weight / eta
        infeas = np.sqrt(np.clip(rx - alpha**2 * eta, 0.0, None)) / norm
        out[f"{mineral}_mf"] = _to_map(alpha, bg, coords)
        out[f"{mineral}_infeas"] = _to_map(infeas, bg, coords)
    return xr.Dataset(out)
