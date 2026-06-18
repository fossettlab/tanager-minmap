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


def mineral_map(
    abundance: xr.Dataset,
    title: str = "Mineral map",
) -> matplotlib.figure.Figure:
    """Render a multi-mineral abundance map (hero figure).

    Parameters
    ----------
    abundance : xr.Dataset
        Per-mineral abundance/detection layers from :mod:`tanager_rocks.unmix`.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO (spec step 9): composite per-mineral layers with an accessible
    # categorical palette; overlay the site footprint and a scale bar.
    raise NotImplementedError


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
