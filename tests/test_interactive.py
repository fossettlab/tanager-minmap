"""Tests for the interactive-map overlay helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr
from rasterio.transform import from_origin

from tanager_minmap.interactive import class_rgba, reproject_classes_4326


def test_class_rgba_colors_and_transparency():
    arr = np.array([[0, 1], [2, -1]])
    colors = {0: "#ff0000", 1: "#00ff00", 2: (0.0, 0.0, 1.0)}
    rgba = class_rgba(arr, colors, transparent=frozenset({0}))
    assert rgba.shape == (2, 2, 4)
    assert tuple(rgba[0, 1]) == (0, 255, 0, 255)  # class 1 opaque green
    assert tuple(rgba[1, 0]) == (0, 0, 255, 255)  # class 2 opaque blue
    assert rgba[0, 0, 3] == 0  # class 0 is in `transparent`
    assert rgba[1, 1, 3] == 0  # -1 is unmapped -> transparent


def test_reproject_classes_4326_bounds_and_shape():
    arr = np.array([[0, 1, 2], [2, 1, 0], [0, 1, 2]])
    # UTM 11N coords near Goldfield; the coords must match the transform.
    x = 400000.0 + np.arange(3) * 30.0
    y = 4150000.0 - np.arange(3) * 30.0
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": y, "x": x})
    transform = from_origin(400000.0, 4150000.0, 30.0, 30.0)
    out, bounds = reproject_classes_4326(da, "EPSG:32611", transform)
    (south, west), (north, east) = bounds
    assert south < north and west < east
    assert -125.0 < west < -110.0 and 35.0 < south < 42.0  # western-US lat/lon
    assert out.ndim == 2
