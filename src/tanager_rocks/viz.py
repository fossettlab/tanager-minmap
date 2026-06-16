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


def band_depth_panel(
    depths: xr.Dataset,
    title: str = "Continuum-removed band depth",
    vmax_quantile: float = 0.98,
) -> matplotlib.figure.Figure:
    """Plot each diagnostic band-depth map as a panel with a shared style.

    Parameters
    ----------
    depths : xr.Dataset
        One band-depth variable per diagnostic feature (from
        :func:`tanager_rocks.features.diagnostic_feature_maps`).
    title : str
        Figure suptitle.
    vmax_quantile : float
        Upper quantile used to set each panel's color stretch, so a few
        high-depth outliers do not flatten the map.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(depths.data_vars)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        da = depths[name]
        vmax = float(da.quantile(vmax_quantile).item())
        im = da.plot.imshow(  # type: ignore[attr-defined]
            ax=ax, cmap="cividis", vmin=0.0, vmax=max(vmax, 1e-3), add_colorbar=False
        )
        ax.set_title(name)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="band depth")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


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


def band_ablation_panel(
    tanager_feature: xr.DataArray,
    s2_feature: xr.DataArray,
    title: str = "Tanager vs. Sentinel-2: Al-OH doublet",
) -> matplotlib.figure.Figure:
    """Side-by-side panel quantifying what Sentinel-2 loses (novelty lever).

    Parameters
    ----------
    tanager_feature, s2_feature : xr.DataArray
        The same diagnostic feature from full Tanager vs. SRF-degraded S2 bands
        (see spec step 5 / :func:`tanager_spec.srf.simulate`).
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO (spec step 6): paired maps + a difference panel; annotate that S2
    # cannot resolve the Al-OH doublet, so it cannot split alunite/kaolinite.
    raise NotImplementedError
