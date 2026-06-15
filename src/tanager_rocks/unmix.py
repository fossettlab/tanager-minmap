"""Spectral unmixing: SAM and MTMF against the reference library.

Implements spec.md pipeline step 4. MTMF (a covariance-aware matched filter)
is the primary method because the methodology suite found Tanager's
information lives in covariance-aware statistics; SAM is the band-independent
comparison. Both consume endmembers from :mod:`tanager_rocks.speclib` and a
masked SR cube from the :mod:`tanager_spec` data layer.
"""

from __future__ import annotations

import xarray as xr

from .speclib import Endmember


def spectral_angle(cube: xr.DataArray, endmembers: dict[str, Endmember]) -> xr.Dataset:
    """Spectral Angle Mapper score per endmember.

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
        One spectral-angle variable per mineral (radians; smaller = better match).
    """
    # TODO (spec step 4): per-pixel arccos of normalised dot product against
    # each endmember; mask invalid pixels.
    raise NotImplementedError


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
