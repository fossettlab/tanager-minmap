"""Smoke tests for the hero mineral-map compositing logic."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display needed for the compositing logic

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from tanager_rocks.hazard import AGP_LABELS  # noqa: E402
from tanager_rocks.viz import amd_map, mineral_map  # noqa: E402


def _layer(values):
    ny, nx = values.shape
    return xr.DataArray(
        values.astype(float),
        dims=("y", "x"),
        coords={"y": np.arange(ny), "x": np.arange(nx) * 30.0},
    )


def test_mineral_map_dominant_selection_and_legend():
    # alunite strong in the top half, muscovite strong in the bottom half
    a = np.zeros((4, 4))
    a[:2, :] = 1.0
    m = np.zeros((4, 4))
    m[2:, :] = 1.0
    ds = xr.Dataset({"alunite": _layer(a), "muscovite": _layer(m)})
    fig = mineral_map(ds, per_mineral_quantile=0.5)
    labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    # both detected minerals plus the unclassified entry appear
    assert {"alunite", "muscovite", "no detection"} <= labels


def test_mineral_map_all_zero_is_blank():
    # No positive abundance anywhere -> nothing classified, only "no detection".
    z = _layer(np.zeros((3, 3)))
    fig = mineral_map(xr.Dataset({"alunite": z, "muscovite": z}))
    labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert labels == {"no detection"}


def test_amd_map_legend_reflects_present_tiers():
    # tiers row: high | moderate | low | background | off-domain(NaN)
    tiers = _layer(np.array([[3.0, 2.0, 1.0, 0.0, np.nan]]))
    fig = amd_map(tiers, labels=AGP_LABELS)
    labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    # every realised tier is labelled; the NaN pixel contributes no legend entry.
    assert labels == {AGP_LABELS[c] for c in (0, 1, 2, 3)}
