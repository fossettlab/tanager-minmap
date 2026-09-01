"""Tests for EMIT granule selection and GLT orthorectification on synthetic data."""

from __future__ import annotations

import h5py
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
from shapely.geometry import Polygon, box

from tanager_minmap.emit import (
    EmitGranule,
    _granule_polygon,
    _ortho_window,
    load_emit_reflectance,
    rank_granules,
    select_granule,
)


def _umm(lons, lats, date, cloud, ur):
    pts = [{"Longitude": x, "Latitude": y} for x, y in zip(lons, lats, strict=True)]
    return {
        "GranuleUR": ur,
        "CloudCover": cloud,
        "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": date}},
        "SpatialExtent": {
            "HorizontalSpatialDomain": {"Geometry": {"GPolygons": [{"Boundary": {"Points": pts}}]}}
        },
    }


class _FakeResult(dict):
    def data_links(self):
        return ["https://example/RFL.nc"]


def test_granule_polygon_parses_boundary():
    umm = _umm([0, 2, 2, 0], [0, 0, 2, 2], "2023-08-04T00:00:00Z", 4.0, "g1")
    poly = _granule_polygon(umm)
    assert isinstance(poly, Polygon)
    assert poly.area == 4.0


def test_rank_and_select_picks_clearest_full_overlap():
    foot = box(0, 0, 1, 1)
    # full-overlap cloudy, full-overlap clear, partial-overlap clear
    cloudy = _FakeResult(
        umm=_umm([-1, 2, 2, -1], [-1, -1, 2, 2], "2023-01-01T00:00:00Z", 80.0, "c")
    )
    clear = _FakeResult(umm=_umm([-1, 2, 2, -1], [-1, -1, 2, 2], "2023-06-01T00:00:00Z", 5.0, "k"))
    partial = _FakeResult(
        umm=_umm([0.5, 2, 2, 0.5], [0, 0, 2, 2], "2023-07-01T00:00:00Z", 1.0, "p")
    )
    ranked = rank_granules([cloudy, clear, partial], foot)
    assert ranked[0].overlap == 1.0  # full-overlap granules rank first
    chosen = select_granule(ranked, max_cloud=10.0, min_overlap=0.99)
    assert chosen.granule_ur == "k"  # clearest among full-overlap


def test_select_granule_raises_without_coverage():
    g = EmitGranule("x", "2023", 1.0, 0.4, ())
    try:
        select_granule([g], min_overlap=0.99)
    except ValueError as e:
        assert "covers" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for insufficient coverage")


def test_ortho_window_clamps_to_grid():
    # geotransform: x0=0, dx=1, y0=10, dy=-1; grid 10x10
    gt = np.array([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
    r0, r1, c0, c1 = _ortho_window(gt, [2.0, 3.0, 5.0, 6.0], (10, 10))
    assert 0 <= r0 < r1 <= 10
    assert 0 <= c0 < c1 <= 10
    # bbox cols [2,5] -> cols ~2..5; rows for y in [3,6] with y0=10 -> rows ~4..7
    assert c0 <= 2 and c1 >= 6
    assert r0 <= 4 and r1 >= 7


def _write_synthetic_emit(path):
    # 3 downtrack x 2 crosstrack x 4 bands; distinct value per (dt, ct, band)
    refl = np.zeros((3, 2, 4), dtype="float32")
    for dt in range(3):
        for ct in range(2):
            refl[dt, ct, :] = 10 * dt + ct + np.arange(4) * 0.1
    wl = np.array([500.0, 600.0, 700.0, 800.0], dtype="float32")
    good = np.array([1, 1, 0, 1], dtype="uint8")  # band 2 is flagged bad
    # 2x2 ortho grid; (1,1) is GLT fill
    glt_x = np.array([[1, 2], [1, 0]], dtype="int32")
    glt_y = np.array([[1, 1], [2, 0]], dtype="int32")
    with h5py.File(path, "w") as f:
        f.create_dataset("reflectance", data=refl)
        sbp = f.create_group("sensor_band_parameters")
        sbp.create_dataset("wavelengths", data=wl)
        sbp.create_dataset("good_wavelengths", data=good)
        loc = f.create_group("location")
        loc.create_dataset("glt_x", data=glt_x)
        loc.create_dataset("glt_y", data=glt_y)
        f.attrs["geotransform"] = np.array([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
    return refl


def test_load_emit_orthorectifies_via_glt(tmp_path):
    path = tmp_path / "EMIT_L2A_RFL_synthetic.nc"
    refl = _write_synthetic_emit(path)
    cube, wl = load_emit_reflectance(path)
    assert cube.dims == ("band", "y", "x")
    assert cube.shape == (4, 2, 2)
    assert cube.rio.crs.to_epsg() == 4326
    # GLT maps: out[0,0]=refl[0,0], out[0,1]=refl[0,1], out[1,0]=refl[1,0]; out[1,1]=fill
    assert np.allclose(cube.values[0, 0, 0], refl[0, 0, 0])
    assert np.allclose(cube.values[0, 0, 1], refl[0, 1, 0])
    assert np.allclose(cube.values[0, 1, 0], refl[1, 0, 0])
    assert np.isnan(cube.values[:, 1, 1]).all()  # GLT-fill pixel
    assert np.isnan(cube.values[2]).all()  # band 2 flagged not-good
    assert np.allclose(wl, [500.0, 600.0, 700.0, 800.0])


def test_load_emit_empty_window_returns_all_nan(tmp_path):
    # A GLT with no valid cells (bbox entirely off-swath) must yield an all-NaN
    # cube, not raise on the empty rows_needed reduction.
    path = tmp_path / "EMIT_L2A_RFL_empty.nc"
    with h5py.File(path, "w") as f:
        f.create_dataset("reflectance", data=np.zeros((3, 2, 4), dtype="float32"))
        sbp = f.create_group("sensor_band_parameters")
        sbp.create_dataset("wavelengths", data=np.array([500.0, 600.0, 700.0, 800.0]))
        sbp.create_dataset("good_wavelengths", data=np.array([1, 1, 1, 1], dtype="uint8"))
        loc = f.create_group("location")
        loc.create_dataset("glt_x", data=np.zeros((2, 2), dtype="int32"))  # all GLT fill
        loc.create_dataset("glt_y", data=np.zeros((2, 2), dtype="int32"))
        f.attrs["geotransform"] = np.array([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
    cube, _ = load_emit_reflectance(path)
    assert cube.shape == (4, 2, 2)
    assert np.isnan(cube.values).all()
