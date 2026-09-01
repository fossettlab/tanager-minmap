"""Spectral unmixing: SAM and MTMF against the reference library.

Implements spec.md pipeline step 4. MTMF (a covariance-aware matched filter)
is the primary method because the methodology suite found Tanager's
information lives in covariance-aware statistics; SAM is the band-independent
comparison. Both consume endmembers from :mod:`tanager_minmap.speclib` and a
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
        Mineral -> :class:`tanager_minmap.speclib.Endmember`, resampled to the
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


@dataclass(frozen=True)
class MtmfBackground:
    """One fitted MTMF background shared by every frozen endmember."""

    valid_bands: np.ndarray
    mean: np.ndarray
    covariance_inverse: np.ndarray
    sample_count: int
    ridge: float


def _spatial_mask(
    mask: np.ndarray | xr.DataArray | None,
    shape: tuple[int, int],
    *,
    name: str,
) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    values = np.asarray(mask, dtype=bool)
    if values.shape != shape:
        raise ValueError(f"{name} shape {values.shape} does not match cube shape {shape}")
    return values


def _validate_mtmf_inputs(
    cube: xr.DataArray,
    endmembers: dict[str, Endmember],
) -> tuple[np.ndarray, tuple[int, int]]:
    if not endmembers:
        raise ValueError("at least one endmember is required")
    data = cube.transpose("band", "y", "x").values
    shape = (data.shape[1], data.shape[2])
    for mineral, endmember in endmembers.items():
        if np.asarray(endmember.reflectance).shape != (data.shape[0],):
            raise ValueError(f"{mineral} endmember length does not match the cube band count")
    return data, shape


def fit_mtmf_background(
    cube: xr.DataArray,
    endmembers: dict[str, Endmember],
    ridge: float = 1e-2,
    *,
    fit_mask: np.ndarray | xr.DataArray | None = None,
    batch_size: int = 65_536,
) -> MtmfBackground:
    """Fit the MTMF mean and loaded covariance on explicitly selected pixels.

    ``fit_mask`` is applied before band support, the mean, or covariance is
    calculated. Consequently, values outside the mask cannot influence any
    fitted background quantity. Moments are accumulated in deterministic
    row-major batches to avoid materialising a full pixel-by-band copy.
    """
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    data, shape = _validate_mtmf_inputs(cube, endmembers)
    selected = _spatial_mask(fit_mask, shape, name="fit_mask").reshape(-1)
    if not np.any(selected):
        raise ValueError("fit_mask selects no pixels")
    flat = data.reshape(data.shape[0], -1)
    endmember_finite = np.logical_and.reduce(
        [np.isfinite(endmember.reflectance) for endmember in endmembers.values()]
    )
    band_has_training_data = np.asarray(
        [np.any(np.isfinite(flat[index, selected])) for index in range(flat.shape[0])]
    )
    valid_bands = endmember_finite & band_has_training_data
    if not np.any(valid_bands):
        raise ValueError("no bands are finite in both the training pixels and all endmembers")

    finite_pixels = selected.copy()
    for index in np.flatnonzero(valid_bands):
        finite_pixels &= np.isfinite(flat[index])
    sample_indices = np.flatnonzero(finite_pixels)
    if sample_indices.size < 2:
        raise ValueError("at least two pairwise-complete training pixels are required")

    n_dimensions = int(np.count_nonzero(valid_bands))
    band_indices = np.flatnonzero(valid_bands)
    count = 0
    mean = np.zeros(n_dimensions, dtype=float)
    sum_squares = np.zeros((n_dimensions, n_dimensions), dtype=float)
    for start in range(0, sample_indices.size, batch_size):
        indices = sample_indices[start : start + batch_size]
        batch = flat[np.ix_(band_indices, indices)].T
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_centered = batch - batch_mean
        batch_sum_squares = batch_centered.T @ batch_centered
        if count == 0:
            mean = batch_mean
            sum_squares = batch_sum_squares
            count = batch_count
            continue
        delta = batch_mean - mean
        combined = count + batch_count
        sum_squares += batch_sum_squares + np.outer(delta, delta) * (count * batch_count / combined)
        mean += delta * (batch_count / combined)
        count = combined

    covariance = sum_squares / (count - 1)
    loading_scale = float(np.trace(covariance) / covariance.shape[0])
    if not np.isfinite(loading_scale) or loading_scale <= 0:
        raise ValueError("training covariance has non-positive or non-finite mean variance")
    loaded = covariance + ridge * loading_scale * np.eye(covariance.shape[0])
    try:
        covariance_inverse = np.linalg.inv(loaded)
    except np.linalg.LinAlgError as error:
        raise ValueError("loaded training covariance is singular") from error
    return MtmfBackground(
        valid_bands=valid_bands,
        mean=mean,
        covariance_inverse=covariance_inverse,
        sample_count=count,
        ridge=float(ridge),
    )


def _score_with_background(
    cube: xr.DataArray,
    endmembers: dict[str, Endmember],
    background: MtmfBackground,
    *,
    score_mask: np.ndarray | xr.DataArray | None,
    include_infeasibility: bool,
    batch_size: int,
) -> xr.Dataset:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    data, shape = _validate_mtmf_inputs(cube, endmembers)
    if background.valid_bands.shape != (data.shape[0],):
        raise ValueError("background band mask does not match the cube")
    if int(np.count_nonzero(background.valid_bands)) != background.mean.size:
        raise ValueError("background dimensions are internally inconsistent")
    selected = _spatial_mask(score_mask, shape, name="score_mask").reshape(-1)
    flat = data.reshape(data.shape[0], -1)
    band_indices = np.flatnonzero(background.valid_bands)
    finite_pixels = selected.copy()
    for index in band_indices:
        finite_pixels &= np.isfinite(flat[index])
    score_indices = np.flatnonzero(finite_pixels)

    minerals = tuple(endmembers)
    weights: dict[str, np.ndarray] = {}
    eta: dict[str, float] = {}
    for mineral, endmember in endmembers.items():
        direction = endmember.reflectance[background.valid_bands] - background.mean
        weight = background.covariance_inverse @ direction
        denominator = float(direction @ weight)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"{mineral} has a degenerate MTMF target direction")
        weights[mineral] = weight
        eta[mineral] = denominator

    values = {
        f"{mineral}_{suffix}": np.full(flat.shape[1], np.nan, dtype=float)
        for mineral in minerals
        for suffix in (("mf", "infeas") if include_infeasibility else ("mf",))
    }
    normalizer = np.sqrt(max(background.mean.size - 1, 1))
    for start in range(0, score_indices.size, batch_size):
        indices = score_indices[start : start + batch_size]
        centered = flat[np.ix_(band_indices, indices)].T - background.mean
        if include_infeasibility:
            whitened = centered @ background.covariance_inverse
            rx = np.einsum("ij,ij->i", whitened, centered)
        for mineral in minerals:
            alpha = centered @ weights[mineral] / eta[mineral]
            values[f"{mineral}_mf"][indices] = alpha
            if include_infeasibility:
                residual = np.clip(rx - alpha**2 * eta[mineral], 0.0, None)
                values[f"{mineral}_infeas"][indices] = np.sqrt(residual) / normalizer

    coords = {"y": cube.y, "x": cube.x}
    return xr.Dataset(
        {
            name: xr.DataArray(array.reshape(shape), dims=("y", "x"), coords=coords)
            for name, array in values.items()
        }
    )


def score_mtmf_background(
    cube: xr.DataArray,
    endmembers: dict[str, Endmember],
    background: MtmfBackground,
    *,
    score_mask: np.ndarray | xr.DataArray | None = None,
    batch_size: int = 65_536,
) -> xr.Dataset:
    """Score MTMF abundance and infeasibility from one pre-fitted background."""
    return _score_with_background(
        cube,
        endmembers,
        background,
        score_mask=score_mask,
        include_infeasibility=True,
        batch_size=batch_size,
    )


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
        Mineral -> :class:`tanager_minmap.speclib.Endmember`.
    ridge : float
        Diagonal-loading fraction applied to the covariance before inversion.

    Returns
    -------
    xr.Dataset
        One matched-filter abundance variable per mineral.
    """
    background = fit_mtmf_background(cube, endmembers, ridge)
    scored = _score_with_background(
        cube,
        endmembers,
        background,
        score_mask=None,
        include_infeasibility=False,
        batch_size=65_536,
    )
    return xr.Dataset({mineral: scored[f"{mineral}_mf"] for mineral in endmembers})


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
    diagonal loading shrinks the whitening), so the fixed ``max_infeas = 1.0``
    the pipeline applies is a distribution-informed feasibility filter, not a
    calibrated sigma: the background sits near 0.2, so the gate passes ~99.9% of
    pixels and removes only the extreme-misfit tail. Detection is defined by the
    downstream per-mineral upper-decile abundance floor, not by this gate.

    Parameters
    ----------
    cube : xr.DataArray
        Masked surface-reflectance cube (band, y, x).
    endmembers : dict
        Mineral -> :class:`tanager_minmap.speclib.Endmember`.
    ridge : float
        Diagonal-loading fraction applied to the covariance before inversion.

    Returns
    -------
    xr.Dataset
        ``<mineral>_mf`` (abundance) and ``<mineral>_infeas`` per mineral.
    """
    background = fit_mtmf_background(cube, endmembers, ridge)
    return score_mtmf_background(cube, endmembers, background)
