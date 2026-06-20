"""Submission presentation figures (Visualization & Storytelling).

Composite, presentation-grade figures for the competition story page, distinct
from the per-stage analytical panels in :mod:`tanager_rocks.viz`. They assemble
true-color context, the diagnostic-spectra story, and the validation
side-by-side from the same products the pipeline computes — nothing here
re-derives analysis, it only presents it.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch
from tanager_spec.mask import invalid_pixel_mask

from .speclib import Endmember
from .viz import MINERAL_COLORS, _scale_bar

# Reflectance bands nearest these centres make the true-color composite.
RGB_NM = (640.0, 550.0, 470.0)
# Display valid-reflectance bounds for the RGB stretch — drops nodata fill and
# atmospheric-correction overshoot so the percentile stretch is not blown out.
RGB_VALID_RANGE = (0.0, 1.5)


def _nearest(wl: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(np.asarray(wl, dtype=float) - target_nm)))


def rgb_context(
    cube: xr.DataArray,
    wl: np.ndarray,
    *,
    title: str,
    scale_bar_m: float | None = 5000.0,
    rgb_nm: tuple[float, float, float] = RGB_NM,
    pct: tuple[float, float] = (2.0, 98.0),
) -> matplotlib.figure.Figure:
    """True-color composite of a Tanager scene (the "these are real scenes" view).

    The three visible bands nearest ``rgb_nm`` are stacked and percentile-
    stretched per channel over the valid pixels; invalid pixels (nodata fill /
    out-of-range overshoot, from :func:`tanager_spec.mask.invalid_pixel_mask`)
    are rendered white. Expects a raw (un-absorption-masked) cube so the visible
    bands are present.

    Parameters
    ----------
    cube : xr.DataArray
        Surface-reflectance cube, dims ``("band", "y", "x")``.
    wl : np.ndarray
        Band-centre wavelengths (nm).
    title : str
        Figure title.
    scale_bar_m : float, optional
        Scale-bar length in projected metres; ``None`` to omit.
    rgb_nm : tuple of float
        Target wavelengths (nm) for the R, G, B channels.
    pct : tuple of float
        Lower/upper percentiles for the per-channel display stretch.

    Returns
    -------
    matplotlib.figure.Figure
    """
    idx = [_nearest(wl, t) for t in rgb_nm]
    sub = cube.isel(band=idx)
    invalid = invalid_pixel_mask(sub, valid_range=RGB_VALID_RANGE).values

    rgb = np.stack([sub.isel(band=i).values for i in range(3)], axis=-1).astype(float)
    lo_p, hi_p = pct
    for c in range(3):
        chan = rgb[..., c]
        lo, hi = np.nanpercentile(chan[~invalid], [lo_p, hi_p]) if (~invalid).any() else (0.0, 1.0)
        rgb[..., c] = np.clip((chan - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    rgb[invalid] = 1.0  # white background for off-scene / invalid pixels

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if scale_bar_m is not None and "x" in cube.coords:
        _scale_bar(ax, np.asarray(cube["x"].values), scale_bar_m)
    fig.tight_layout()
    return fig


def representative_spectra(
    cube: xr.DataArray,
    mtmf_ds: xr.Dataset,
    minerals: list[str],
    *,
    top_n: int = 100,
    max_infeas: float = 1.0,
    quantile: float = 0.98,
) -> dict[str, np.ndarray]:
    """Mean reflectance spectrum of the most strongly-detected pixels per mineral.

    For each mineral the infeasibility-gated MTMF abundance selects its top
    ``quantile`` pixels; the mean reflectance of the strongest ``top_n`` of those
    is returned. This is the bridge from a label back to the data: the spectrum
    of the pixels the model calls "alunite" should carry alunite's absorptions.

    Returns
    -------
    dict
        Mineral -> mean reflectance spectrum (length = n bands), for minerals
        with at least one gated detection.
    """
    refl = cube.transpose("band", "y", "x").values
    nb = refl.shape[0]
    flat_refl = refl.reshape(nb, -1)
    out: dict[str, np.ndarray] = {}
    for mineral in minerals:
        if f"{mineral}_mf" not in mtmf_ds:
            continue
        gated = mtmf_ds[f"{mineral}_mf"].where(mtmf_ds[f"{mineral}_infeas"] < max_infeas).values
        flat = gated.ravel()
        finite = np.isfinite(flat) & (flat > 0)
        if not finite.any():
            continue
        thr = float(np.quantile(flat[finite], quantile))
        sel = finite & (flat >= thr)
        if not sel.any():
            continue
        order = np.argsort(np.where(sel, flat, -np.inf))[::-1][:top_n]
        out[mineral] = np.nanmean(flat_refl[:, order], axis=1)
    return out


def _normalize(spectrum: np.ndarray) -> np.ndarray:
    """Scale a spectrum to [0, 1] for shape comparison.

    Uses a robust 2nd–98th percentile range and clips, so a single noisy
    band at a mask edge does not compress the curve or spike off-scale.
    """
    finite = np.isfinite(spectrum)
    if not finite.any():
        return spectrum
    lo, hi = np.nanpercentile(spectrum, [2.0, 98.0])
    return np.clip((spectrum - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def spectra_story(
    endmembers: dict[str, Endmember],
    pixel_spectra: dict[str, np.ndarray],
    wl: np.ndarray,
    minerals: list[str],
    *,
    absorptions: dict[str, float],
    title: str = "Diagnostic spectra: library vs. Tanager scene",
) -> matplotlib.figure.Figure:
    """Offset-stacked library vs scene spectra with diagnostic absorptions marked.

    For each mineral the USGS library endmember (solid) and the mean spectrum of
    the scene's most-detected pixels (dashed) are normalised to [0, 1] and offset
    vertically, so a reader can see the same diagnostic absorptions in both. The
    named absorptions are drawn as vertical guides.

    Parameters
    ----------
    endmembers : dict
        Mineral -> :class:`Endmember` (reflectance on ``wl``).
    pixel_spectra : dict
        Mineral -> mean scene spectrum (from :func:`representative_spectra`).
    wl : np.ndarray
        Band-centre wavelengths (nm).
    minerals : list of str
        Minerals to stack, bottom to top.
    absorptions : dict
        Label -> wavelength (nm) for the diagnostic guides.
    title : str
        Figure title.
    """
    wl = np.asarray(wl, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 7))
    step = 1.25
    for i, mineral in enumerate(minerals):
        offset = i * step
        color = MINERAL_COLORS.get(mineral, f"C{i}")
        if mineral in endmembers:
            ax.plot(wl, _normalize(endmembers[mineral].reflectance) + offset, color=color, lw=1.4)
        if mineral in pixel_spectra:
            scene = _normalize(pixel_spectra[mineral]) + offset
            ax.plot(wl, scene, color=color, lw=1.2, ls="--", alpha=0.9)
        ax.text(wl[-1], offset + 0.55, mineral, color=color, ha="right", va="center", fontsize=10)

    top = len(minerals) * step
    for label, nm in absorptions.items():
        ax.axvline(nm, color="0.5", lw=0.8, ls=":")
        ax.text(nm, top + 0.05, label, rotation=90, va="bottom", ha="center", fontsize=8, c="0.4")

    ax.set_xlim(float(wl.min()), float(wl.max()))
    ax.set_ylim(-0.15, top + 0.55)  # headroom for the absorption labels
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("normalised reflectance (offset per mineral)")
    ax.set_yticks([])
    ax.set_title(title)
    handles = [
        plt.Line2D([], [], color="0.2", lw=1.4, label="USGS splib07a library"),
        plt.Line2D([], [], color="0.2", lw=1.2, ls="--", label="Tanager scene (top pixels)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def validation_pair(
    score: xr.DataArray,
    reference: xr.DataArray,
    positive_classes: frozenset[int] | set[int],
    *,
    mineral: str,
    title: str,
    excluded: frozenset[int] | set[int] = frozenset(),
    vmax_quantile: float = 0.98,
) -> matplotlib.figure.Figure:
    """Tanager score map beside the reference alteration zone on a shared grid.

    Left: the continuous Tanager MTMF abundance for ``mineral``. Right: the
    aligned Rockwell reference reduced to three tones — the mineral's published
    positive zone, other classified ground, and excluded/nodata — so the rank-AUC
    agreement is visually legible.

    Parameters
    ----------
    score : xr.DataArray
        Tanager score map, dims ``("y", "x")``.
    reference : xr.DataArray
        Aligned categorical Rockwell reference, dims ``("y", "x")``.
    positive_classes : set of int
        Reference class values that are positive for ``mineral``.
    mineral : str
        Mineral name (titling + positive-zone colour).
    title : str
        Figure suptitle.
    excluded : set of int
        Reference classes treated as nodata (rendered white).
    vmax_quantile : float
        Upper quantile for the score colour stretch.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6.5))

    vmax = max(float(np.nanquantile(score.values, vmax_quantile)), 1e-3)
    im = score.plot.imshow(ax=ax0, cmap="cividis", vmin=0.0, vmax=vmax, add_colorbar=False)  # type: ignore[attr-defined]
    ax0.set_title(f"Tanager {mineral} MTMF abundance")
    ax0.set_aspect("equal")
    ax0.set_xlabel("")
    ax0.set_ylabel("")
    ax0.set_xticks([])
    ax0.set_yticks([])
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label="MF abundance")

    ref = reference.values
    pos_color = MINERAL_COLORS.get(mineral, "#d95f02")
    rgba = np.zeros((*ref.shape, 4), dtype=float)
    is_excluded = np.isin(ref, list(excluded))
    is_pos = np.isin(ref, list(positive_classes))
    is_other = ~is_pos & ~is_excluded
    rgba[is_excluded] = (0.97, 0.97, 0.97, 1.0)  # near-white nodata
    rgba[is_other] = (0.62, 0.62, 0.62, 1.0)  # medium grey = classified footprint
    rgba[is_pos] = (*matplotlib.colors.to_rgb(pos_color), 1.0)
    ax1.imshow(rgba, interpolation="nearest")
    ax1.set_title(f"Rockwell ASTER reference — {mineral} zone")
    ax1.set_aspect("equal")
    ax1.set_xticks([])
    ax1.set_yticks([])
    handles = [
        Patch(facecolor=pos_color, label=f"{mineral} zone (positive)"),
        Patch(facecolor=(0.62, 0.62, 0.62), label="other classified"),
        Patch(facecolor=(0.97, 0.97, 0.97), edgecolor="0.6", label="excluded / nodata"),
    ]
    ax1.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
