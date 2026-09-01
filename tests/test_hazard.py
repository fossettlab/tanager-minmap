"""Tests for the ordinal AMD acid-generating-potential proxy."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tanager_minmap.hazard import (
    AGP_BACKGROUND,
    AGP_HIGH,
    AGP_LOW,
    AGP_MODERATE,
    acid_generating_potential,
)

# All four indicator minerals the proxy reads; goethite is included but left
# empty in most fixtures so hematite alone drives the Fe-oxide (moderate) tier.
_MINERALS = ("jarosite", "hematite", "goethite", "gypsum")


def _ds(mf: dict[str, list[float]], infeas: dict[str, list[float]] | None = None) -> xr.Dataset:
    """Build a synthetic MTMF dataset (one row) with _mf and _infeas per mineral.

    Minerals absent from ``mf`` are filled with zeros; infeasibility defaults to
    0 (feasible) for every mineral unless overridden.
    """
    nx = len(next(iter(mf.values())))
    coords = {"y": np.arange(1), "x": np.arange(nx) * 30.0}
    data: dict[str, xr.DataArray] = {}
    for mineral in _MINERALS:
        vals = np.asarray(mf.get(mineral, [0.0] * nx), dtype=float).reshape(1, nx)
        inf = np.asarray((infeas or {}).get(mineral, [0.0] * nx), dtype=float).reshape(1, nx)
        data[f"{mineral}_mf"] = xr.DataArray(vals, dims=("y", "x"), coords=coords)
        data[f"{mineral}_infeas"] = xr.DataArray(inf, dims=("y", "x"), coords=coords)
    return xr.Dataset(data)


def test_tier_priority_most_acidic_indicator_wins():
    # Columns: jarosite | hematite | gypsum-only | none | gypsum+hematite
    ds = _ds(
        {
            "jarosite": [1.0, 0.0, 0.0, 0.0, 0.0],
            "hematite": [0.0, 1.0, 0.0, 0.0, 1.0],
            "gypsum": [0.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    tiers = acid_generating_potential(ds, quantile=0.5).tiers.values.ravel()
    # last column carries both gypsum and Fe-oxide -> Fe-oxide (moderate) wins.
    assert list(tiers) == [AGP_HIGH, AGP_MODERATE, AGP_LOW, AGP_BACKGROUND, AGP_MODERATE]


def test_infeasibility_gate_demotes_detection():
    # Jarosite is abundant but spectrally infeasible -> not a detection -> background.
    ds = _ds(
        {"jarosite": [1.0, 0.0], "hematite": [0.0, 1.0]},
        infeas={"jarosite": [2.0, 0.0]},  # col0 above the gate
    )
    tiers = acid_generating_potential(ds, max_infeas=1.0, quantile=0.5).tiers.values.ravel()
    assert list(tiers) == [AGP_BACKGROUND, AGP_MODERATE]


def test_off_domain_pixels_are_nan_not_background():
    # A non-finite raw matched-filter score = off-scene/nodata -> NaN, not 0.
    ds = _ds({"jarosite": [1.0, np.nan], "gypsum": [0.0, 0.0]})
    tiers = acid_generating_potential(ds, quantile=0.5).tiers.values.ravel()
    assert tiers[0] == AGP_HIGH
    assert np.isnan(tiers[1])


def test_counts_and_domain_consistency():
    ds = _ds(
        {
            "jarosite": [1.0, 0.0, np.nan],
            "hematite": [0.0, 1.0, 0.0],
            "gypsum": [0.0, 0.0, 0.0],
        }
    )
    result = acid_generating_potential(ds, quantile=0.5)
    assert result.domain.tolist() == [[True, True, False]]
    # in-domain tier counts sum to the in-domain pixel count.
    assert sum(result.counts.values()) == int(result.domain.sum())
    assert result.counts[AGP_HIGH] == 1
    assert result.counts[AGP_MODERATE] == 1


def test_requires_jarosite_layer():
    ds = _ds({"hematite": [1.0, 0.0], "gypsum": [0.0, 1.0]})
    ds = ds.drop_vars(["jarosite_mf", "jarosite_infeas"])
    with pytest.raises(ValueError, match="jarosite"):
        acid_generating_potential(ds)
