"""Spectral unmixing: SAM and MTMF against the reference library.

Implements spec.md pipeline step 4. MTMF (a covariance-aware matched filter)
is the primary method because the methodology suite found Tanager's
information lives in covariance-aware statistics; SAM is the band-independent
comparison. Both consume endmembers from :mod:`tanager_rocks.speclib` and a
masked SR cube from the :mod:`tanager_spec` data layer.
"""

from __future__ import annotations

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
    Tanager's signal to live; it is the abundance half of MTMF (the mixture-
    tuned infeasibility gate is a separate, still-to-build step).

    Adjacent VSWIR bands are nearly collinear, so the full-band covariance is
    ill-conditioned/singular. ``C`` is therefore stabilised by diagonal loading
    (``C + ridge * mean(diag(C)) * I``) — standard regularisation for matched
    filtering, controlled by ``ridge`` (a numerical parameter, not physical).

    Bands that are all-NaN (absorption-masked) and any band where an endmember
    is non-finite are dropped; nodata pixels come back as ``NaN``.

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
    cov_inv = np.linalg.inv(cov)

    out: dict[str, xr.DataArray] = {}
    for mineral, em in endmembers.items():
        d = em.reflectance[valid_b] - mu
        weight = cov_inv @ d
        score = centered @ weight / float(d @ weight)  # =1 at target, 0 at background
        full = np.full(ny * nx, np.nan)
        full[px_valid] = score
        out[mineral] = xr.DataArray(
            full.reshape(ny, nx), dims=("y", "x"), coords={"y": cube.y, "x": cube.x}
        )
    return xr.Dataset(out)
