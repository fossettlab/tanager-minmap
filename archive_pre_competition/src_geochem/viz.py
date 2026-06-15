"""Shared visualization functions for the Tanager Rocks project.

Provides consistent plotting for spectral data, geochemistry maps,
SHAP overlays, and model comparison figures.

Typical usage::

    from src.viz import plot_spectra, plot_shap_spectrum

    fig = plot_spectra(spectra_df, highlight_bands=[2200, 900])
    fig = plot_shap_spectrum(shap_values, wavelengths, oxide="SiO2")
"""

from __future__ import annotations

import logging

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Publication figure defaults
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
COLORMAP_SPECTRA = "viridis"
COLORMAP_MAP = "RdYlBu_r"


def setup_style() -> None:
    """Apply consistent matplotlib style for publication figures."""
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "savefig.bbox": "tight",
        "savefig.dpi": FIGURE_DPI,
    })


def plot_spectra(
    spectra: pd.DataFrame,
    wavelengths: np.ndarray | None = None,
    highlight_bands: list[float] | None = None,
    title: str = "Tanager-1 Reflectance Spectra",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot spectral signatures with optional band highlights.

    Parameters
    ----------
    spectra : pd.DataFrame
        Spectral data (rows = samples, columns = wavelengths).
    wavelengths : np.ndarray, optional
        X-axis wavelengths. If None, inferred from column names.
    highlight_bands : list of float, optional
        Wavelengths to mark with vertical lines (e.g. mineral absorptions).
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO: implement
    # 1. Create figure/axes if needed
    # 2. Plot each spectrum as a line
    # 3. Add vertical lines for highlight_bands
    # 4. Label axes (Wavelength nm, Reflectance)
    raise NotImplementedError


def plot_sample_map(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray | None = None,
    scene_footprints: object | None = None,
    title: str = "Sample Locations",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot geochemistry sample locations on a map.

    Parameters
    ----------
    lats, lons : np.ndarray
        Sample coordinates.
    values : np.ndarray, optional
        Values to color-code points (e.g. oxide concentration).
    scene_footprints : GeoDataFrame, optional
        Tanager scene footprints to overlay.
    title : str
        Plot title.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO: implement
    raise NotImplementedError


def plot_shap_spectrum(
    shap_values: np.ndarray,
    wavelengths: np.ndarray,
    oxide: str,
    mineral_features: dict[str, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot SHAP values overlaid on wavelength axis with mineral annotations.

    Parameters
    ----------
    shap_values : np.ndarray
        Mean absolute SHAP values per band.
    wavelengths : np.ndarray
        Band center wavelengths (nm).
    oxide : str
        Target oxide name for title.
    mineral_features : dict, optional
        Mineral name → diagnostic wavelength (nm) for annotations.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO: implement
    # 1. Bar or line plot of SHAP values vs wavelength
    # 2. Annotate mineral absorption features
    # 3. Shade atmospheric mask regions
    raise NotImplementedError


def plot_sensor_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "r2",
    title: str = "Hyperspectral vs. Multispectral Performance",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Bar chart comparing model performance across sensors.

    Parameters
    ----------
    results : dict
        Sensor name → {oxide: metric_value} nested dict.
    metric : str
        Metric to plot (``"r2"`` or ``"rmse"``).
    title : str
        Plot title.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO: implement
    # 1. Grouped bar chart: sensors on x-axis, bars grouped by oxide
    # 2. Add error bars if CI available
    raise NotImplementedError
