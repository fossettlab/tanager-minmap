"""Tests for the Rockwell ASTER validation-reference class table and alignment."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tanager_minmap.config import TARGET_MINERALS
from tanager_minmap.reference import (
    FEATURE_TO_ROCKWELL,
    MINERAL_TO_ROCKWELL,
    ROCKWELL_CLASSES,
    ROCKWELL_EXCLUDED,
    align_reference,
)


def test_mapped_classes_exist_and_are_not_excluded():
    # Every positive class in either mapping must be a real Rockwell class and
    # must not be one of the excluded (nodata/vegetation/corrupted) values.
    for mapping in (MINERAL_TO_ROCKWELL, FEATURE_TO_ROCKWELL):
        for layer, classes in mapping.items():
            assert classes, f"{layer} has no positive classes"
            for c in classes:
                assert c in ROCKWELL_CLASSES, f"{layer}: class {c} not in table"
                assert c not in ROCKWELL_EXCLUDED, f"{layer}: class {c} is excluded"


def test_target_minerals_mapped_except_gypsum():
    # Gypsum has no Rockwell class (no sulfate-evaporite class) and is the only
    # documented omission; every other target mineral must be mapped.
    mapped = set(MINERAL_TO_ROCKWELL)
    assert "gypsum" not in mapped
    assert set(TARGET_MINERALS) - mapped == {"gypsum"}


def _categorical_da(values: np.ndarray, x0: float, y0: float, res: float) -> xr.DataArray:
    ny, nx = values.shape
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": y0 - res * (np.arange(ny) + 0.5),  # north-up
            "x": x0 + res * (np.arange(nx) + 0.5),
        },
    )
    return da.rio.write_crs("EPSG:4326")


def test_align_reference_nearest_preserves_class_codes():
    # A 2x2 categorical block upsampled by nearest must introduce no new codes.
    ref = _categorical_da(np.array([[3, 5], [8, 1]], dtype="uint8"), 0.0, 1.0, 0.5)
    like = _categorical_da(np.zeros((4, 4), dtype="uint8"), 0.0, 1.0, 0.25)
    out = align_reference(ref, like)
    assert set(np.unique(out.values)) <= {1, 3, 5, 8}
    assert out.shape == (4, 4)
