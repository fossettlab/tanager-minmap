"""Visualization for mineral maps, band-ablation panels, and EMIT comparison.

Produces the submission figures (spec.md "Visualization & Storytelling"):
the Bingham hero mineral map, the Sentinel-2 band-ablation panel showing the
Al-OH doublet that S2 cannot split, and the EMIT cross-sensor comparison.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# Publication figure defaults (PNG @ 300 DPI per project convention).
FIGURE_DPI = 300
FIGURE_FORMAT = "png"


def setup_style() -> None:
    """Apply consistent matplotlib style for submission figures."""
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
            "savefig.dpi": FIGURE_DPI,
        }
    )


def score_panel(
    scores: xr.Dataset,
    title: str,
    cbar_label: str = "value",
    vmax_quantile: float = 0.98,
    ncols: int = 4,
) -> matplotlib.figure.Figure:
    """Plot each variable in a Dataset as a panel in a shared grid.

    Parameters
    ----------
    scores : xr.Dataset
        One 2-D ``(y, x)`` variable per panel (band depths, matched-filter
        scores, etc.).
    title : str
        Figure suptitle.
    cbar_label : str
        Colorbar label.
    vmax_quantile : float
        Upper quantile for each panel's color stretch, so a few outliers do
        not flatten the map. Lower bound is fixed at 0.
    ncols : int
        Panels per row.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(scores.data_vars)
    ncols = min(ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    for ax, name in zip(flat_axes, names, strict=False):
        da = scores[name]
        vmax = float(da.quantile(vmax_quantile).item())
        im = da.plot.imshow(  # type: ignore[attr-defined]
            ax=ax, cmap="cividis", vmin=0.0, vmax=max(vmax, 1e-3), add_colorbar=False
        )
        ax.set_title(name)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    for ax in flat_axes[len(names) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def band_depth_panel(
    depths: xr.Dataset,
    title: str = "Continuum-removed band depth",
    vmax_quantile: float = 0.98,
) -> matplotlib.figure.Figure:
    """Plot diagnostic band-depth maps (thin wrapper over :func:`score_panel`)."""
    return score_panel(depths, title, cbar_label="band depth", vmax_quantile=vmax_quantile)


def classification_map(
    classes: xr.DataArray,
    labels: list[str],
    title: str = "SAM mineral classification",
) -> matplotlib.figure.Figure:
    """Render an integer class map (-1 = unclassified) with a categorical legend.

    Parameters
    ----------
    classes : xr.DataArray
        Integer class codes, dims ``("y", "x")``; -1 is unclassified.
    labels : list of str
        Class labels in code order (index 0..n-1).
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(labels)
    # tab20 is categorical and reasonably colorblind-tolerant for ~8 classes.
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(n, 1)))
    cmap = ListedColormap([(0.85, 0.85, 0.85, 1.0), *colors])  # grey = unclassified
    norm = BoundaryNorm(np.arange(-1.5, n + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(classes.values, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [Patch(facecolor=colors[i], label=labels[i]) for i in range(n)]
    handles.append(Patch(facecolor=(0.85, 0.85, 0.85), label="unclassified"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def zone_discrimination_panel(
    scores: xr.Dataset,
    reference: xr.DataArray,
    mapping: dict[str, frozenset[int]],
    discriminations: dict,
    title: str = "Score by reference alteration zone",
    ncols: int = 4,
) -> matplotlib.figure.Figure:
    """Box plots of each score inside vs. outside its reference alteration zone.

    For every validated layer, the score distribution in the published positive
    zone is drawn beside the distribution over the other classified ground; the
    panel title carries the rank AUC. This is the visual companion to
    :func:`tanager_rocks.validate.validate_scores`.

    Parameters
    ----------
    scores : xr.Dataset
        Score maps (one ``(y, x)`` variable per layer).
    reference : xr.DataArray
        Aligned categorical Rockwell reference.
    mapping : dict
        Layer -> positive Rockwell class set.
    discriminations : dict
        Layer -> ``Discrimination`` from :func:`validate_scores` (for the AUC).
    title : str
        Figure suptitle.
    ncols : int
        Panels per row.
    """
    from .validate import analysis_domain

    layers = [m for m in mapping if m in discriminations]
    ncols = min(ncols, max(len(layers), 1))
    nrows = int(np.ceil(max(len(layers), 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.0 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    domain = analysis_domain(reference)
    for ax, layer in zip(flat_axes, layers, strict=False):
        sc = scores[layer].values
        ref = reference.values
        use = domain & np.isfinite(sc)
        is_pos = np.isin(ref, list(mapping[layer])) & use
        is_neg = use & ~np.isin(ref, list(mapping[layer]))
        ax.boxplot([sc[is_pos], sc[is_neg]], showfliers=False, widths=0.6)
        ax.set_xticks([1, 2], ["in zone", "out"])  # version-independent labels
        d = discriminations[layer]
        ax.set_title(f"{layer}\nAUC={d.auc:.2f}")
        ax.set_ylabel("score")
    for ax in flat_axes[len(layers) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# Accessible categorical palette (Brewer "Dark2" / "Set2" hues) for the named
# alteration minerals — chosen to stay distinguishable for common colorblindness
# and ordered so the alteration story (clays, micas, Fe-oxides, sulfates) reads.
MINERAL_COLORS: dict[str, str] = {
    "alunite": "#d95f02",  # advanced argillic
    "kaolinite": "#e7298a",  # argillic
    "dickite": "#a6761d",  # argillic (high-T)
    "muscovite": "#1b9e77",  # sericite / phyllic
    "jarosite": "#e6ab02",  # acid-sulfate / AMD
    "hematite": "#7570b3",  # ferric oxide
    "goethite": "#66a61e",  # ferric oxyhydroxide
    "gypsum": "#666666",  # sulfate
}


def _scale_bar(ax, x_coords: np.ndarray, length_m: float = 5000.0) -> None:
    """Draw a simple scale bar (projected metres) in the lower-left of an axis.

    ``x_coords`` are the map's projected x coordinates (UTM metres); the bar is
    drawn ``length_m`` wide in pixel space using the coordinate spacing.
    """
    px = float(abs(x_coords[1] - x_coords[0]))  # metres per pixel
    n_px = length_m / px
    span = abs(ax.get_xlim()[1] - ax.get_xlim()[0])  # axis width in pixels
    frac = n_px / span
    x_lo, y_lo = 0.06, 0.06  # axes-fraction anchor
    ax.plot(
        [x_lo, x_lo + frac],
        [y_lo, y_lo],
        transform=ax.transAxes,
        color="black",
        lw=3,
        solid_capstyle="butt",
    )
    ax.text(
        x_lo + frac / 2,
        y_lo + 0.02,
        f"{length_m / 1000:.0f} km",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def mineral_map(
    abundance: xr.Dataset,
    title: str = "Mineral map",
    per_mineral_quantile: float = 0.90,
    vmax_quantile: float = 0.98,
    scale_bar_m: float | None = 5000.0,
) -> matplotlib.figure.Figure:
    """Composite per-mineral MTMF abundance into a dominant-mineral hero map.

    Each mineral is gated to its own upper tail: only pixels whose abundance
    exceeds that mineral's ``per_mineral_quantile`` are kept, and those are
    normalised by the mineral's own threshold so the layers are comparable
    despite differing absolute matched-filter scales. At each pixel the dominant
    mineral is then the one most strongly expressed *relative to its own
    detection floor*, colored from :data:`MINERAL_COLORS` with opacity scaled by
    that strength. Pixels where no mineral clears its threshold (or off-scene /
    infeasibility-gated NaN) are left light grey. The per-mineral gate keeps the
    pervasive low-level soil signal (e.g. background Fe-oxide) from washing the
    map and lets the alteration centres read.

    Parameters
    ----------
    abundance : xr.Dataset
        Per-mineral abundance layers (one ``(y, x)`` var each) from
        :mod:`tanager_rocks.unmix`, typically infeasibility-gated.
    title : str
        Figure title.
    per_mineral_quantile : float
        Upper quantile of each mineral's positive abundance used as its
        detection threshold (0.90 keeps the top 10 % per mineral).
    vmax_quantile : float
        Upper quantile of the normalised strength used for the opacity stretch.
    scale_bar_m : float, optional
        Scale-bar length in metres (projected CRS); ``None`` to omit.

    Returns
    -------
    matplotlib.figure.Figure
    """
    minerals = list(abundance.data_vars)
    stack = np.stack([abundance[m].values for m in minerals], axis=0)  # (M, y, x)
    # Per-mineral threshold + normalise by that threshold so layers are comparable.
    strength = np.full_like(stack, np.nan, dtype=float)
    for i in range(len(minerals)):
        v = stack[i]
        pos = v[np.isfinite(v) & (v > 0)]
        if pos.size == 0:
            continue
        thr = float(np.quantile(pos, per_mineral_quantile))
        if thr <= 0:
            continue
        keep = np.isfinite(v) & (v >= thr)
        strength[i][keep] = v[keep] / thr  # >= 1 where kept

    finite = np.isfinite(strength)
    filled = np.where(finite, strength, -np.inf)
    dominant = np.argmax(filled, axis=0)
    peak = np.max(filled, axis=0)  # -inf where no mineral clears its threshold
    classified = np.isfinite(peak)
    max_val = np.where(classified, peak, np.nan)

    vmax = float(np.nanquantile(max_val[classified], vmax_quantile)) if classified.any() else 2.0
    # Opacity floor 0.4 so every kept detection is visible; saturates at vmax.
    alpha = 0.4 + 0.6 * np.clip((max_val - 1.0) / max(vmax - 1.0, 1e-9), 0.0, 1.0)

    ny, nx = dominant.shape
    rgba = np.zeros((ny, nx, 4), dtype=float)
    fallback = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(minerals), 1)))
    for i, m in enumerate(minerals):
        color = MINERAL_COLORS.get(m)
        rgb = matplotlib.colors.to_rgb(color) if color else tuple(fallback[i][:3])
        sel = classified & (dominant == i)
        rgba[sel, :3] = rgb
        rgba[sel, 3] = alpha[sel]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.full((ny, nx), 0.92), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    present = [m for i, m in enumerate(minerals) if (classified & (dominant == i)).any()]
    handles = [Patch(facecolor=MINERAL_COLORS.get(m, "0.5"), label=m) for m in present]
    handles.append(Patch(facecolor=(0.92, 0.92, 0.92), label="no detection"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    if scale_bar_m is not None and "x" in abundance.coords:
        _scale_bar(ax, np.asarray(abundance["x"].values), scale_bar_m)
    fig.tight_layout()
    return fig


# Sequential acid-generating-potential palette (ColorBrewer YlOrRd hues),
# keyed by the ordinal tier codes in :mod:`tanager_rocks.hazard`. Background is
# the same light grey as the hero map; low/moderate/high escalate in luminance
# so the ramp reads for common colorblindness. Off-domain pixels are white.
AGP_TIER_COLORS: dict[int, tuple[float, float, float] | str] = {
    0: (0.92, 0.92, 0.92),  # background (in-scene, no indicator)
    1: "#fecc5c",  # low / neutralised
    2: "#fd8d3c",  # moderate
    3: "#e31a1c",  # high
}


def amd_map(
    tiers: xr.DataArray,
    title: str = "Acid-generating-potential proxy",
    labels: dict[int, str] | None = None,
    scale_bar_m: float | None = 5000.0,
) -> matplotlib.figure.Figure:
    """Render the ordinal AMD acid-generating-potential map (spec step 7).

    ``tiers`` is the ordinal AGP code per pixel from
    :func:`tanager_rocks.hazard.acid_generating_potential` (``NaN`` off the
    in-scene domain). Each tier is drawn in its :data:`AGP_TIER_COLORS` hue over
    a white base; off-domain pixels stay transparent (white). Only tiers present
    in the map appear in the legend.

    Parameters
    ----------
    tiers : xr.DataArray
        Ordinal AGP map, dims ``("y", "x")``.
    title : str
        Figure title.
    labels : dict, optional
        Tier-code -> legend label. Defaults to a generic ramp; callers pass
        :data:`tanager_rocks.hazard.AGP_LABELS` for the science-grounded text.
    scale_bar_m : float, optional
        Scale-bar length in metres (projected CRS); ``None`` to omit.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = labels or {0: "background", 1: "low", 2: "moderate", 3: "high"}
    vals = tiers.values
    ny, nx = vals.shape

    rgba = np.zeros((ny, nx, 4), dtype=float)
    rgba[..., :3] = 1.0  # white base; off-domain stays transparent over it
    present: list[int] = []
    for code, color in AGP_TIER_COLORS.items():
        sel = np.isfinite(vals) & (vals == code)
        if not sel.any():
            continue
        rgba[sel, :3] = matplotlib.colors.to_rgb(color)
        rgba[sel, 3] = 1.0
        present.append(code)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.ones((ny, nx)), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [Patch(facecolor=AGP_TIER_COLORS[c], label=labels[c]) for c in sorted(present)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    if scale_bar_m is not None and "x" in tiers.coords:
        _scale_bar(ax, np.asarray(tiers["x"].values), scale_bar_m)
    fig.tight_layout()
    return fig


def emit_comparison_panel(
    common_nm: np.ndarray,
    tan_mean: np.ndarray,
    emit_mean: np.ndarray,
    pearson_r: float,
    spectral_angle_deg: float,
    tan_score: xr.DataArray,
    emit_score: xr.DataArray,
    mineral: str,
    detection_r: float,
    title: str = "Tanager vs EMIT cross-sensor comparison",
    vmax_quantile: float = 0.98,
) -> matplotlib.figure.Figure:
    """Three-panel cross-sensor comparison (spec step 6).

    Left: the two scene-mean reflectance spectra on EMIT's wavelength axis with
    the spectral-agreement metrics. Middle/right: one mineral's MTMF map from
    each sensor at its native resolution (Tanager 30 m vs EMIT 60 m), sharing a
    color stretch, with the spatial-detection correlation annotated.

    Parameters
    ----------
    common_nm, tan_mean, emit_mean : np.ndarray
        Common wavelength axis and the two scene-mean spectra (from
        :func:`tanager_rocks.compare.spectral_agreement`).
    pearson_r, spectral_angle_deg : float
        Scene-mean spectral agreement.
    tan_score, emit_score : xr.DataArray
        The chosen mineral's MTMF map from each sensor (native grids).
    mineral : str
        Mineral name for titling.
    detection_r : float
        Spatial correlation of the two maps on the common grid.
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5))

    ax0.plot(common_nm, tan_mean, color="#1b9e77", lw=1.2, label="Tanager (resampled)")
    ax0.plot(common_nm, emit_mean, color="#7570b3", lw=1.2, label="EMIT")
    ax0.set_xlabel("wavelength (nm)")
    ax0.set_ylabel("scene-mean reflectance")
    ax0.set_title("Scene-mean spectra")
    ax0.legend(fontsize=8, frameon=False)
    ax0.annotate(
        f"Pearson r = {pearson_r:.3f}\nspectral angle = {spectral_angle_deg:.2f}°",
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.5"},
    )

    # Per-map stretch: MTMF abundance is not absolutely comparable across sensors
    # (different band sets and covariance), so each map is scaled to its own
    # distribution; the detection correlation carries the quantitative agreement.
    for ax, da, label in ((ax1, tan_score, "Tanager 30 m"), (ax2, emit_score, "EMIT 60 m")):
        vmax = max(float(np.nanquantile(da.values, vmax_quantile)), 1e-3)
        im = da.plot.imshow(  # type: ignore[attr-defined]
            ax=ax, cmap="cividis", vmin=0.0, vmax=vmax, add_colorbar=False
        )
        ax.set_title(f"{mineral} MTMF — {label}")
        ax.set_aspect("equal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="abundance")
    ax2.annotate(
        f"detection r = {detection_r:.3f}",
        xy=(0.5, 0.97),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.5"},
    )
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def band_ablation_panel(
    wavelengths: np.ndarray,
    endmembers: dict[str, np.ndarray],
    degraded: dict[str, np.ndarray],
    s2_centers: np.ndarray,
    s2_fwhm: np.ndarray,
    full_angle_deg: float,
    s2_angle_deg: float,
    minerals: tuple[str, str] = ("alunite", "kaolinite"),
    title: str = "Tanager vs. Sentinel-2: the Al-OH doublet",
) -> matplotlib.figure.Figure:
    """Show what Sentinel-2 loses by collapsing the 2200 nm Al-OH doublet.

    Left: the two minerals' full Tanager VSWIR spectra with the SRF-degraded S2
    band values overplotted. Right: a zoom on the 2000-2350 nm Al-OH region with
    the S2 SWIR bands' FWHM shaded, making explicit that one broad S2 band (B12)
    spans the whole doublet. The annotation carries the spectral-angle
    separability in each sensor's band space (novelty lever, spec step 5).

    Parameters
    ----------
    wavelengths : np.ndarray
        Tanager wavelength axis (nm).
    endmembers, degraded : dict
        ``mineral -> reflectance`` at full Tanager resolution and degraded to S2.
    s2_centers, s2_fwhm : np.ndarray
        Sentinel-2 band centers and FWHM (nm), from :func:`degrade.srf_band_stats`.
    full_angle_deg, s2_angle_deg : float
        ``minerals``-pair spectral angle (degrees) in Tanager vs. S2 band space.
    minerals : tuple of str
        The two minerals to contrast (default alunite vs kaolinite).
    title : str
        Figure suptitle.
    """
    colors = {minerals[0]: "#1b9e77", minerals[1]: "#d95f02"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    for m in minerals:
        ax0.plot(wavelengths, endmembers[m], color=colors[m], lw=1.2, label=f"{m} (Tanager)")
        ax0.plot(
            s2_centers,
            degraded[m],
            "o--",
            color=colors[m],
            ms=5,
            lw=1.0,
            alpha=0.8,
            label=f"{m} (Sentinel-2)",
        )
    ax0.set_xlabel("wavelength (nm)")
    ax0.set_ylabel("reflectance")
    ax0.set_title("Full VSWIR vs. 13 S2 bands")
    ax0.legend(fontsize=8, frameon=False)

    lo, hi = 2000.0, 2350.0
    win = (wavelengths >= lo) & (wavelengths <= hi)
    for m in minerals:
        ax1.plot(wavelengths[win], np.asarray(endmembers[m])[win], color=colors[m], lw=1.6, label=m)
    in_win = (s2_centers >= lo) & (s2_centers <= hi)
    for c, w in zip(s2_centers[in_win], s2_fwhm[in_win], strict=False):
        ax1.axvspan(c - w / 2, c + w / 2, color="0.6", alpha=0.18)
        ax1.axvline(c, color="0.4", lw=0.8, ls=":")
    for m in minerals:
        ax1.plot(s2_centers[in_win], np.asarray(degraded[m])[in_win], "o", color=colors[m], ms=7)
    ax1.set_xlim(lo, hi)
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("reflectance")
    ax1.set_title("Al-OH region — S2 band FWHM shaded")
    ax1.legend(fontsize=8, frameon=False, loc="lower left")
    ax1.annotate(
        f"{minerals[0]}–{minerals[1]} spectral angle\n"
        f"Tanager {full_angle_deg:.1f}°  →  S2 {s2_angle_deg:.1f}°  "
        f"({100 * (1 - s2_angle_deg / full_angle_deg):.0f}% loss)",
        xy=(0.5, 0.97),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.5"},
    )
    fig.suptitle(title)
    fig.tight_layout()
    return fig
