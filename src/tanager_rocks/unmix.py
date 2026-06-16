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


def mtmf(
    cube: xr.DataArray,
    endmembers: dict[str, Endmember],
    n_components: int | None = None,
) -> xr.Dataset:
    """Mixture-Tuned Matched Filter abundance + infeasibility per endmember.

    Parameters
    ----------
    cube : xr.DataArray
        Masked surface-reflectance cube (band, y, x).
    endmembers : dict
        Mineral -> :class:`tanager_rocks.speclib.Endmember`.
    n_components : int, optional
        MNF components to retain before matched filtering. If None, chosen
        from the noise-whitened eigenvalue spectrum.

    Returns
    -------
    xr.Dataset
        Per mineral: matched-filter abundance score and MTMF infeasibility,
        the pair used to threshold confident detections (spec step 4).
    """
    # TODO (spec step 4): MNF transform, per-endmember matched filter on the
    # whitened cube, infeasibility from the orthogonal residual; return both
    # so detections can be gated on (high abundance, low infeasibility).
    raise NotImplementedError
