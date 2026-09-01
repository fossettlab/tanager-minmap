"""Focused tests for pipeline input and quality-policy wiring."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import xarray as xr

from tanager_minmap.config import SiteSpec
from tanager_minmap.pipeline import (
    PipelinePaths,
    _ensure_emit_granule,
    _load_masked_cube,
    _write_amd_counts_csv,
)
from tanager_minmap.quality import TanagerQualityReport


def _paths(tmp_path: Path) -> PipelinePaths:
    return PipelinePaths(
        raw_dir=tmp_path / "raw",
        speclib_dir=tmp_path / "speclib",
        reference_dir=tmp_path / "reference",
        emit_dir=tmp_path / "emit",
        maps_dir=tmp_path / "maps",
        figures_dir=tmp_path / "figures",
        intermediate_dir=tmp_path / "intermediate",
    )


def _write_emit_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["geotransform"] = np.array([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
        handle.create_dataset("reflectance", data=np.ones((1, 1, 2), dtype=np.float32))
        bands = handle.create_group("sensor_band_parameters")
        bands.create_dataset("wavelengths", data=np.array([500.0, 600.0]))
        bands.create_dataset("good_wavelengths", data=np.array([1, 1], dtype=np.uint8))
        location = handle.create_group("location")
        location.create_dataset("glt_x", data=np.array([[1]], dtype=np.int16))
        location.create_dataset("glt_y", data=np.array([[1]], dtype=np.int16))


def test_pipeline_load_routes_cube_through_shared_quality_policy(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    scene_id = "scene"
    site = SiteSpec("test", "Test", "hero", 1, ((0.0, 0.0),), (scene_id,), True)
    wavelengths = np.array([500.0, 2000.0])
    raw = xr.DataArray(
        np.ones((2, 1, 2)),
        dims=("band", "y", "x"),
        coords={"band": wavelengths, "y": [0.0], "x": [0.0, 1.0]},
    )
    masked = raw.copy()
    masked.values[:, 0, 1] = np.nan
    calls: list[tuple[Path, Path]] = []

    def fake_load(path: Path):
        calls.append((path, Path("load")))
        return raw, wavelengths

    def fake_mask(cube, wl, path):
        assert cube is raw
        np.testing.assert_array_equal(wl, wavelengths)
        calls.append((path, Path("mask")))
        return masked, TanagerQualityReport(2, 1, 0, 0, 1, 0, 0, 2)

    monkeypatch.setattr("tanager_minmap.pipeline.load_tanager_sr_hdf5", fake_load)
    monkeypatch.setattr("tanager_minmap.pipeline.mask_tanager_scene", fake_mask)
    expected_path = paths.raw_dir / f"{scene_id}_ortho_sr_hdf5.h5"
    expected_path.parent.mkdir()
    expected_path.touch()

    result, result_wavelengths = _load_masked_cube(site, paths)

    assert calls == [(expected_path, Path("load")), (expected_path, Path("mask"))]
    assert result is masked
    np.testing.assert_array_equal(result_wavelengths, wavelengths)


def test_pinned_emit_cache_does_not_require_catalog_login(tmp_path: Path):
    granule = "EMIT_L2A_RFL_001_test"
    emit_dir = tmp_path / "emit"
    emit_dir.mkdir()
    cached = emit_dir / f"{granule}.nc"
    _write_emit_file(cached)

    result = _ensure_emit_granule(
        [-1.0, -1.0, 1.0, 1.0],
        emit_dir,
        expected_granule_ur=granule,
    )

    assert result == cached


def test_missing_scene_error_names_download_command(tmp_path: Path):
    paths = _paths(tmp_path)
    site = SiteSpec("test", "Test", "hero", 1, ((0.0, 0.0),), ("scene",), True)

    with pytest.raises(FileNotFoundError, match=r"download_scenes\.py --site test"):
        _load_masked_cube(site, paths)


def test_invalid_pinned_emit_cache_fails_before_catalog_query(tmp_path: Path):
    granule = "EMIT_L2A_RFL_001_test"
    emit_dir = tmp_path / "emit"
    emit_dir.mkdir()
    (emit_dir / f"{granule}.nc").write_bytes(b"truncated")

    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        _ensure_emit_granule(
            [-1.0, -1.0, 1.0, 1.0],
            emit_dir,
            expected_granule_ur=granule,
        )


def test_missing_emit_cache_requires_environment_credentials(tmp_path: Path, monkeypatch):
    for name in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="Earthdata credentials are not configured"):
        _ensure_emit_granule([-1.0, -1.0, 1.0, 1.0], tmp_path / "emit")


def test_emit_download_returns_actual_written_filename(tmp_path: Path, monkeypatch):
    emit_dir = tmp_path / "emit"
    chosen = SimpleNamespace(
        granule_ur="EMIT_L2A_RFL_001_catalog_name",
        data_links=("https://example.test/actual-rfl-name_RFL_.nc",),
    )

    def download(_links, destination):
        actual = Path(destination) / "actual-rfl-name_RFL_.nc"
        _write_emit_file(actual)
        return [str(actual)]

    fake_earthaccess = SimpleNamespace(
        login=lambda **_kwargs: None,
        search_data=lambda **_kwargs: [],
        download=download,
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)
    monkeypatch.setenv("EARTHDATA_USERNAME", "configured")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "configured")
    monkeypatch.setattr("tanager_minmap.pipeline.rank_granules", lambda *_args: [chosen])
    monkeypatch.setattr("tanager_minmap.pipeline.select_granule", lambda _ranked: chosen)

    result = _ensure_emit_granule([-1.0, -1.0, 1.0, 1.0], emit_dir)

    assert result == emit_dir / "actual-rfl-name_RFL_.nc"


def test_write_amd_counts_csv_records_counts_and_gates(tmp_path: Path):
    path = tmp_path / "counts.csv"

    _write_amd_counts_csv(
        path,
        {3: 7, 0: 11, 2: 5, 1: 3},
        in_scene_pixels=26,
        max_infeas=1.0,
        quantile=0.9,
    )

    assert path.read_text().splitlines() == [
        "tier_code,tier_label,pixel_count,in_scene_pixels,max_infeas,detection_quantile",
        "0,background (no indicator),11,26,1.0,0.9",
        "1,low / neutralised (gypsum),3,26,1.0,0.9",
        "2,moderate (Fe-oxide),5,26,1.0,0.9",
        "3,high (jarosite),7,26,1.0,0.9",
    ]
