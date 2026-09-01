"""Synthetic tests for the pre-result native/basic-to-ortho framework."""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_coordinates

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_LAUNCHER_PATH = _SCRIPTS_DIR / "run_basic_ortho_sensitivity_launcher.py"
_RUNNER_PATH = _SCRIPTS_DIR / "run_basic_ortho_sensitivity.py"
_LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "run_basic_ortho_sensitivity_launcher",
    _LAUNCHER_PATH,
)
assert _LAUNCHER_SPEC is not None and _LAUNCHER_SPEC.loader is not None
launcher = importlib.util.module_from_spec(_LAUNCHER_SPEC)
sys.modules[_LAUNCHER_SPEC.name] = launcher
_LAUNCHER_SPEC.loader.exec_module(launcher)
runner = launcher.load_runner_module(
    _RUNNER_PATH,
    module_name="run_basic_ortho_sensitivity",
)
_RUNTIME_BINDING = runner._bootstrap_runtime()
basic_ortho = _RUNTIME_BINDING.basic_ortho

FROZEN_ACQUISITION_MANIFEST_SHA256 = basic_ortho.FROZEN_ACQUISITION_MANIFEST_SHA256
FROZEN_ORTHO_MANIFEST_SHA256 = basic_ortho.FROZEN_ORTHO_MANIFEST_SHA256
FROZEN_PREREGISTRATION_SHA256 = basic_ortho.FROZEN_PREREGISTRATION_SHA256
FROZEN_SCENES = basic_ortho.FROZEN_SCENES
MAPPING_ARTIFACT_NAMES = basic_ortho.MAPPING_ARTIFACT_NAMES
RESOURCE_PILOT_DEFAULT_BRANCH = basic_ortho.RESOURCE_PILOT_DEFAULT_BRANCH
RESOURCE_PILOT_DEFAULT_SITE = basic_ortho.RESOURCE_PILOT_DEFAULT_SITE
TARGET_BASIC_QA_INVALID = basic_ortho.TARGET_BASIC_QA_INVALID
TARGET_MAPPED = basic_ortho.TARGET_MAPPED
TARGET_ORTHO_QA_INVALID = basic_ortho.TARGET_ORTHO_QA_INVALID
CleanupResidueError = basic_ortho.CleanupResidueError
FrozenSceneInput = basic_ortho.FrozenSceneInput
NativeToOrthoMapping = basic_ortho.NativeToOrthoMapping
OrthoGrid = basic_ortho.OrthoGrid
ProtocolError = basic_ortho.ProtocolError
ResourcePilotTelemetry = basic_ortho.ResourcePilotTelemetry
SpectrumCopyAudit = basic_ortho.SpectrumCopyAudit
ValidatedInputs = basic_ortho.ValidatedInputs
design_document = basic_ortho.design_document
exact_spectrum_copy_audit = basic_ortho.exact_spectrum_copy_audit
execution_identity = basic_ortho.execution_identity
inspect_basic_schema = basic_ortho.inspect_basic_schema
inspect_ortho_schema = basic_ortho.inspect_ortho_schema
load_basic_geolocation_and_qa = basic_ortho.load_basic_geolocation_and_qa
load_ortho_grid = basic_ortho.load_ortho_grid
load_ortho_qa = basic_ortho.load_ortho_qa
map_native_to_ortho = basic_ortho.map_native_to_ortho
project_scalar_nearest = basic_ortho.project_scalar_nearest
resource_pilot_document = basic_ortho.resource_pilot_document
schema_document = basic_ortho.schema_document
select_resource_pilot_scene = basic_ortho.select_resource_pilot_scene
sha256_file = basic_ortho.sha256_file
strict_json_dump = basic_ortho.strict_json_dump
validate_frozen_inputs = basic_ortho.validate_frozen_inputs
validate_protocol_file = basic_ortho.validate_protocol_file
validate_schema_pair = basic_ortho.validate_schema_pair
write_mapping_bundle = basic_ortho.write_mapping_bundle

GOVERNING_HASHES = {
    key: hashlib.sha256(key.encode("utf-8")).hexdigest() for key in basic_ortho.GOVERNING_FILE_KEYS
}

_RUNTIME_EXPORTS = (
    "FROZEN_ACQUISITION_MANIFEST_SHA256",
    "FROZEN_ORTHO_MANIFEST_SHA256",
    "FROZEN_PREREGISTRATION_SHA256",
    "FROZEN_SCENES",
    "MAPPING_ARTIFACT_NAMES",
    "RESOURCE_PILOT_DEFAULT_BRANCH",
    "RESOURCE_PILOT_DEFAULT_SITE",
    "TARGET_BASIC_QA_INVALID",
    "TARGET_MAPPED",
    "TARGET_ORTHO_QA_INVALID",
    "CleanupResidueError",
    "FrozenSceneInput",
    "NativeToOrthoMapping",
    "OrthoGrid",
    "ProtocolError",
    "ResourcePilotTelemetry",
    "SpectrumCopyAudit",
    "ValidatedInputs",
    "design_document",
    "exact_spectrum_copy_audit",
    "execution_identity",
    "inspect_basic_schema",
    "inspect_ortho_schema",
    "load_basic_geolocation_and_qa",
    "load_ortho_grid",
    "load_ortho_qa",
    "map_native_to_ortho",
    "project_scalar_nearest",
    "resource_pilot_document",
    "schema_document",
    "select_resource_pilot_scene",
    "sha256_file",
    "strict_json_dump",
    "validate_frozen_inputs",
    "validate_protocol_file",
    "validate_schema_pair",
    "write_mapping_bundle",
)


def _install_test_runtime(binding: object) -> None:
    module = binding.basic_ortho
    globals()["_RUNTIME_BINDING"] = binding
    globals()["basic_ortho"] = module
    for name in _RUNTIME_EXPORTS:
        globals()[name] = getattr(module, name)
    globals()["GOVERNING_HASHES"] = {
        key: hashlib.sha256(key.encode("utf-8")).hexdigest() for key in module.GOVERNING_FILE_KEYS
    }


runner._teardown_runtime(_RUNTIME_BINDING)


@pytest.fixture(scope="module", autouse=True)
def _active_runtime_capsule():
    binding = runner._bootstrap_runtime()
    _install_test_runtime(binding)
    yield
    runner._teardown_runtime(binding)


def _sr_attributes(dataset: h5py.Dataset, bands: int) -> None:
    dataset.attrs["wavelengths"] = np.linspace(500.0, 800.0, bands, dtype=np.float32)
    dataset.attrs["good_wavelengths"] = np.ones(bands, dtype=np.uint8)
    dataset.attrs["_FillValue"] = np.float32(-9999.0)


def _write_basic(
    path: Path,
    *,
    shape: tuple[int, int] = (3, 4),
    bands: int = 2,
    geolocation_shape: tuple[int, int] | None = None,
    qa_shape: tuple[int, int] | None = None,
    spectral_axis: int = 0,
    label_square_axes: bool = False,
) -> None:
    geolocation_shape = geolocation_shape or shape
    qa_shape = qa_shape or geolocation_shape
    cube_shape = list(shape)
    cube_shape.insert(spectral_axis, bands)
    with h5py.File(path, "w") as handle:
        reflectance = handle.create_dataset(
            "HDFEOS/SWATHS/HYP/Data Fields/surface_reflectance",
            shape=tuple(cube_shape),
            dtype="float32",
            fillvalue=-9999.0,
        )
        _sr_attributes(reflectance, bands)
        if label_square_axes:
            spatial_axes = [axis for axis in range(3) if axis != spectral_axis]
            reflectance.dims[spatial_axes[0]].label = "row"
            reflectance.dims[spatial_axes[1]].label = "column"
        geo = handle.require_group("HDFEOS/SWATHS/HYP/Geolocation Fields")
        latitude = geo.create_dataset("Latitude", shape=geolocation_shape, dtype="float64")
        longitude = geo.create_dataset("Longitude", shape=geolocation_shape, dtype="float64")
        latitude.attrs["_FillValue"] = -9999.0
        longitude.attrs["_FillValue"] = -9999.0
        fields = handle["HDFEOS/SWATHS/HYP/Data Fields"]
        for name in ("beta_cloud_mask", "beta_cirrus_mask", "nodata_pixels"):
            dataset = fields.create_dataset(name, shape=qa_shape, dtype="uint8")
            dataset.attrs["_FillValue"] = np.uint8(255)


def _ortho_struct(shape: tuple[int, int], zone: int = 11) -> str:
    ny, nx = shape
    ulx, uly, pixel = 500000.0, 4100000.0, 30.0
    return (
        "GROUP=GridStructure\nGROUP=GRID_1\n"
        'GridName="HYP"\n'
        f"XDim={nx}\nYDim={ny}\n"
        f"UpperLeftPointMtrs=({ulx},{uly})\n"
        f"LowerRightMtrs=({ulx + nx * pixel},{uly - ny * pixel})\n"
        "Projection=HE5_GCTP_UTM\n"
        f"ZoneCode={zone}\nEND_GROUP=GRID_1\nEND_GROUP=GridStructure\n"
    )


def _write_ortho(path: Path, *, shape: tuple[int, int] = (4, 5), bands: int = 2) -> None:
    with h5py.File(path, "w") as handle:
        reflectance = handle.create_dataset(
            "HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance",
            shape=(bands, *shape),
            dtype="float32",
            fillvalue=-9999.0,
        )
        _sr_attributes(reflectance, bands)
        fields = handle["HDFEOS/GRIDS/HYP/Data Fields"]
        for name in ("beta_cloud_mask", "beta_cirrus_mask", "nodata_pixels"):
            dataset = fields.create_dataset(name, shape=shape, dtype="uint8")
            dataset.attrs["_FillValue"] = np.uint8(255)
        handle.create_dataset(
            "HDFEOS INFORMATION/StructMetadata.0",
            data=np.bytes_(_ortho_struct(shape)),
        )


def test_schema_inspection_proves_non_square_axes_without_reading_cubes(tmp_path: Path):
    basic_path = tmp_path / "basic.h5"
    ortho_path = tmp_path / "ortho.h5"
    _write_basic(basic_path, shape=(3, 4), bands=2)
    _write_ortho(ortho_path, shape=(4, 5), bands=2)

    basic = inspect_basic_schema(basic_path, expected_shape=(3, 4))
    ortho = inspect_ortho_schema(ortho_path, expected_shape=(4, 5))
    validate_schema_pair(basic, ortho)

    assert basic.reflectance.shape == (2, 3, 4)
    assert basic.spectral_axis == 0
    assert basic.spatial_axes == (1, 2)
    assert basic.geolocation["latitude"].shape == (3, 4)
    assert set(basic.qa) == {"beta_cloud_mask", "beta_cirrus_mask", "nodata_pixels"}
    assert ortho.spatial_shape == (4, 5)
    assert basic.wavelengths_sha256 == ortho.wavelengths_sha256


def test_schema_accepts_band_last_when_spatial_order_is_proved(tmp_path: Path):
    path = tmp_path / "basic.h5"
    _write_basic(path, shape=(3, 4), bands=2, spectral_axis=2)
    schema = inspect_basic_schema(path, expected_shape=(3, 4))
    assert schema.spectral_axis == 2
    assert schema.spatial_axes == (0, 1)


def test_schema_rejects_transposed_geolocation_or_qa(tmp_path: Path):
    transposed = tmp_path / "transposed.h5"
    _write_basic(transposed, shape=(3, 4), geolocation_shape=(4, 3))
    with pytest.raises(ProtocolError, match="spatial-axis order"):
        inspect_basic_schema(transposed, expected_shape=(4, 3))

    misaligned_qa = tmp_path / "misaligned_qa.h5"
    _write_basic(misaligned_qa, shape=(3, 4), qa_shape=(4, 3))
    with pytest.raises(ProtocolError, match="not integer or aligned"):
        inspect_basic_schema(misaligned_qa, expected_shape=(3, 4))


def test_square_schema_requires_explicit_spatial_labels(tmp_path: Path):
    ambiguous = tmp_path / "ambiguous.h5"
    _write_basic(ambiguous, shape=(3, 3))
    with pytest.raises(ProtocolError, match="orientation cannot be proved"):
        inspect_basic_schema(ambiguous, expected_shape=(3, 3))

    labelled = tmp_path / "labelled.h5"
    _write_basic(labelled, shape=(3, 3), label_square_axes=True)
    assert inspect_basic_schema(labelled, expected_shape=(3, 3)).spatial_shape == (3, 3)


def test_basic_geolocation_and_qa_loader_preserves_values_fill_and_policy(tmp_path: Path):
    path = tmp_path / "basic.h5"
    _write_basic(path, shape=(3, 4))
    schema = inspect_basic_schema(path, expected_shape=(3, 4))
    with h5py.File(path, "r+") as handle:
        geo = handle["HDFEOS/SWATHS/HYP/Geolocation Fields"]
        geo["Longitude"][...] = np.arange(12, dtype=np.float64).reshape(3, 4)
        geo["Latitude"][...] = np.arange(12, dtype=np.float64).reshape(3, 4) + 30.0
        geo["Longitude"][0, 1] = -9999.0
        fields = handle["HDFEOS/SWATHS/HYP/Data Fields"]
        fields["beta_cloud_mask"][1, 2] = 1
        fields["nodata_pixels"][2, 3] = 255

    longitude, latitude, valid, counts = load_basic_geolocation_and_qa(path, schema)
    assert longitude.shape == latitude.shape == valid.shape == (3, 4)
    assert np.isnan(longitude[0, 1])
    assert latitude[0, 1] == 31.0
    assert not valid[1, 2]
    assert not valid[2, 3]
    assert counts == {
        "beta_cloud_mask": 1,
        "beta_cirrus_mask": 0,
        "nodata_pixels": 1,
    }


def test_basic_geolocation_and_qa_loader_rejects_unknown_qa(tmp_path: Path):
    path = tmp_path / "basic.h5"
    _write_basic(path, shape=(3, 4))
    schema = inspect_basic_schema(path, expected_shape=(3, 4))
    with h5py.File(path, "r+") as handle:
        handle["HDFEOS/SWATHS/HYP/Data Fields/beta_cirrus_mask"][0, 0] = 7
    with pytest.raises(ProtocolError, match="undocumented QA values"):
        load_basic_geolocation_and_qa(path, schema)


def test_ortho_qa_loader_preserves_alignment_and_rejects_unknown_values(tmp_path: Path):
    path = tmp_path / "ortho.h5"
    _write_ortho(path, shape=(3, 4))
    schema = inspect_ortho_schema(path, expected_shape=(3, 4))
    with h5py.File(path, "r+") as handle:
        fields = handle["HDFEOS/GRIDS/HYP/Data Fields"]
        fields["beta_cloud_mask"][1, 2] = 1
        fields["nodata_pixels"][2, 3] = 255

    valid, counts = load_ortho_qa(path, schema)
    assert valid.shape == (3, 4)
    assert not valid[1, 2]
    assert not valid[2, 3]
    assert counts == {
        "beta_cloud_mask": 1,
        "beta_cirrus_mask": 0,
        "nodata_pixels": 1,
    }

    with h5py.File(path, "r+") as handle:
        handle["HDFEOS/GRIDS/HYP/Data Fields/beta_cirrus_mask"][0, 0] = 7
    with pytest.raises(ProtocolError, match="undocumented QA values"):
        load_ortho_qa(path, schema)


def test_ortho_grid_metadata_is_bound_to_shape_crs_and_resolution(tmp_path: Path):
    path = tmp_path / "ortho.h5"
    _write_ortho(path, shape=(3, 4))
    grid = load_ortho_grid(
        path,
        expected_shape=(3, 4),
        expected_crs="EPSG:32611",
        expected_resolution_m=30.0,
    )
    assert grid.shape == (3, 4)
    assert grid.transform.a == 30.0
    assert grid.transform.e == -30.0

    with pytest.raises(ProtocolError, match="resolution"):
        load_ortho_grid(
            path,
            expected_shape=(3, 4),
            expected_crs="EPSG:32611",
            expected_resolution_m=31.0,
        )


def test_schema_document_records_frozen_stac_and_hdf_grid_geometry(tmp_path: Path):
    basic_path = tmp_path / "basic.h5"
    ortho_path = tmp_path / "ortho.h5"
    _write_basic(basic_path, shape=(3, 4))
    _write_ortho(ortho_path, shape=(4, 5))
    basic = inspect_basic_schema(basic_path, expected_shape=(3, 4))
    ortho = inspect_ortho_schema(ortho_path, expected_shape=(4, 5))
    grid = load_ortho_grid(
        ortho_path,
        expected_shape=(4, 5),
        expected_crs="EPSG:32611",
        expected_resolution_m=30.0,
    )
    scene = FrozenSceneInput(
        site="goldfield",
        scene_id="scene",
        basic_path=basic_path,
        basic_size_bytes=basic_path.stat().st_size,
        basic_sha256="basic",
        ortho_path=ortho_path,
        ortho_size_bytes=ortho_path.stat().st_size,
        ortho_sha256="ortho",
        basic_stac_shape=(3, 4),
        basic_stac_crs="EPSG:4326",
        basic_stac_resolution_m=40.86,
        ortho_stac_shape=(4, 5),
        ortho_stac_crs="EPSG:32611",
        ortho_stac_resolution_m=30.0,
    )
    inputs = ValidatedInputs(
        acquisition_manifest_path=tmp_path / "acquisition.json",
        acquisition_manifest_sha256="acquisition",
        ortho_manifest_path=tmp_path / "ortho.json",
        ortho_manifest_sha256="ortho-manifest",
        scenes=(scene,),
    )
    document = schema_document(
        inputs,
        {"goldfield": (basic, ortho)},
        {"goldfield": grid},
        protocol_sha256="protocol",
        governing_files=GOVERNING_HASHES,
    )
    assert document["mode"] == "schema-only"
    assert document["frozen_input_geometry"]["goldfield"]["basic_stac_resolution_m"] == 40.86
    assert document["scenes"]["goldfield"]["ortho_grid"]["resolution_m"] == (30.0, 30.0)


def _mapping_fixture() -> tuple[NativeToOrthoMapping, OrthoGrid]:
    grid = OrthoGrid(
        shape=(2, 3),
        transform=from_origin(500000.0, 4100000.0, 30.0, 30.0),
        crs=CRS.from_epsg(32611),
    )
    projected_x = np.array(
        [500015.0, 500015.0, 500045.0, 500015.0, 499000.0, 500075.0, 500085.0, 500075.0]
    ).reshape(2, 4)
    projected_y = np.array(
        [4099985.0, 4099985.0, 4099990.0, 4099985.0, 4101000.0, 4099955.0, 4099955.0, 4099985.0]
    ).reshape(2, 4)
    longitude_raw, latitude_raw = transform_coordinates(
        grid.crs, "EPSG:4326", projected_x.ravel(), projected_y.ravel()
    )
    longitude = np.asarray(longitude_raw).reshape(2, 4)
    latitude = np.asarray(latitude_raw).reshape(2, 4)
    longitude[0, 3] = np.nan
    source_qa_valid = np.ones((2, 4), dtype=bool)
    source_qa_valid[0, 2] = False
    target_qa_valid = np.ones(grid.shape, dtype=bool)
    target_qa_valid[1, 0] = False
    return (
        map_native_to_ortho(
            longitude,
            latitude,
            source_qa_valid,
            target_qa_valid,
            grid,
        ),
        grid,
    )


def test_mapping_accounts_reuse_no_calls_invalid_qa_and_exact_ties():
    mapping, _ = _mapping_fixture()
    counts = mapping.counts

    assert counts.total_source_samples == 8
    assert counts.invalid_qa_source_samples == 1
    assert counts.invalid_geolocation_source_samples == 1
    assert counts.used_source_samples == 4
    assert counts.unused_source_samples == 4
    assert counts.sources_with_multiple_target_cells == 1
    assert counts.duplicate_target_assignments == 1
    assert counts.total_target_cells == 6
    assert counts.invalid_qa_target_cells == 1
    assert counts.basic_qa_no_call_target_cells == 1
    assert counts.no_geolocated_source_target_cells == 0
    assert counts.mapped_target_cells == 4
    assert counts.unmapped_target_cells == 2
    assert mapping.target_status[0, 1] == TARGET_BASIC_QA_INVALID
    assert mapping.target_status[1, 0] == TARGET_ORTHO_QA_INVALID
    assert mapping.source_multiplicity[1, 1] == 2
    assert mapping.source_multiplicity[1, 2] == 2
    assert mapping.target_count_per_source[1, 1] == 2
    assert (mapping.source_row[0, 0], mapping.source_col[0, 0]) == (0, 0)
    assert mapping.source_flat_index[0, 0] == 0
    assert mapping.mapping_distance_m[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_mapping_reports_no_geolocated_source_without_inventing_a_fallback():
    grid = OrthoGrid(
        shape=(1, 2),
        transform=from_origin(500000.0, 4100000.0, 30.0, 30.0),
        crs=CRS.from_epsg(32611),
    )
    mapping = map_native_to_ortho(
        np.full((2, 3), np.nan),
        np.full((2, 3), np.nan),
        np.ones((2, 3), dtype=bool),
        np.ones(grid.shape, dtype=bool),
        grid,
    )
    assert mapping.counts.no_geolocated_source_target_cells == 2
    assert mapping.counts.mapped_target_cells == 0
    assert mapping.counts.unmapped_target_cells == 2
    assert (mapping.source_flat_index == -1).all()


def test_scalar_projection_copies_selected_source_exactly_and_rejects_cubes():
    mapping, _ = _mapping_fixture()
    scalar = np.arange(8, dtype=np.float64).reshape(2, 4)
    scalar_before = scalar.copy()
    ledger_before = mapping.source_flat_index.copy()
    projected = project_scalar_nearest(scalar, mapping)
    mapped = mapping.target_status == TARGET_MAPPED

    np.testing.assert_array_equal(
        projected[mapped], scalar.ravel()[mapping.source_flat_index[mapped]]
    )
    assert np.isnan(projected[~mapped]).all()
    assert mapping.source_flat_index[0, 1] >= 0
    assert np.isnan(projected[0, 1])
    np.testing.assert_array_equal(scalar, scalar_before)
    np.testing.assert_array_equal(mapping.source_flat_index, ledger_before)
    with pytest.raises(ProtocolError, match="2-D scalar"):
        project_scalar_nearest(np.zeros((2, 2, 4)), mapping)


def test_exact_spectrum_copy_audit_separates_any_and_selected_matches():
    mapping, _ = _mapping_fixture()
    basic = np.empty((2, 2, 4), dtype=np.float32)
    basic[0] = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
    basic[1] = basic[0] + 100.0
    ortho = np.full((2, 2, 3), -500.0, dtype=np.float32)
    mapped_targets = np.flatnonzero(mapping.target_status.ravel() == TARGET_MAPPED)
    for target in mapped_targets:
        source = mapping.source_flat_index.ravel()[target]
        ortho.reshape(2, -1)[:, target] = basic.reshape(2, -1)[:, source]
    ortho.reshape(2, -1)[0, mapped_targets[0]] += 0.25
    ortho.reshape(2, -1)[:, 1] = basic.reshape(2, -1)[:, 0]

    basic_valid = np.ones((2, 4), dtype=bool)
    basic_valid[0, 2] = False
    basic_valid[0, 3] = False
    ortho_valid = np.ones((2, 3), dtype=bool)
    ortho_valid[1, 0] = False

    audit = exact_spectrum_copy_audit(
        basic,
        ortho,
        mapping,
        retained_bands=np.ones(2, dtype=bool),
        basic_valid=basic_valid,
        ortho_valid=ortho_valid,
    )
    assert audit.valid_basic_spectra == 6
    assert audit.valid_ortho_spectra == 5
    assert audit.mapped_valid_ortho_spectra == 4
    assert audit.mapped_exact_match_to_selected_basic == 3
    assert audit.mapped_exact_match_to_selected_basic_fraction == pytest.approx(3 / 4)
    assert audit.ortho_exact_match_to_any_basic == 4
    assert audit.ortho_exact_match_to_any_basic_fraction == pytest.approx(4 / 5)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_input_manifests(root: Path, *, partial_basic: bool = False) -> tuple[Path, Path]:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    acquisition_assets = []
    ortho_records = []
    for site, scene in (
        ("goldfield", "20240925_185504_87_4001"),
        ("bingham", "20250911_191523_58_4001"),
    ):
        basic_name = f"{scene}_basic_sr_hdf5.h5" + (
            ".part" if partial_basic and site == "goldfield" else ""
        )
        basic = raw / basic_name
        ortho = raw / f"{scene}_ortho_sr_hdf5.h5"
        basic.write_bytes(f"basic-{site}".encode())
        ortho.write_bytes(f"ortho-{site}".encode())
        acquisition_assets.append(
            {
                "site": site,
                "scene_id": scene,
                "asset_key": "basic_sr_hdf5",
                "local_path": str(basic.relative_to(root)),
                "content_length": basic.stat().st_size,
                "sha256": _sha(basic),
                "stac_proj_shape": [3, 4],
                "stac_proj_code": "EPSG:4326",
                "stac_raster_spatial_resolution_m": 40.0,
                "paired_ortho_proj_shape": [4, 5],
                "paired_ortho_proj_code": "EPSG:32611",
                "paired_ortho_raster_spatial_resolution_m": 30.0,
            }
        )
        ortho_records.append(
            {
                "id": f"tanager-{site}-1",
                "logical_path": str(ortho.relative_to(root)),
                "size_bytes": ortho.stat().st_size,
                "sha256": _sha(ortho),
            }
        )
    acquisition = root / "acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "preregistration": {"sha256": FROZEN_PREREGISTRATION_SHA256},
                "assets": acquisition_assets,
                "scientific_endpoint_values_inspected": False,
            }
        ),
        encoding="utf-8",
    )
    ortho_manifest = root / "ortho.json"
    ortho_manifest.write_text(
        json.dumps({"hash_algorithm": "sha256", "inputs": ortho_records}),
        encoding="utf-8",
    )
    return acquisition, ortho_manifest


def test_input_validation_is_hash_bound_and_rejects_partial_files(tmp_path: Path):
    acquisition, ortho = _write_input_manifests(tmp_path)
    validated = validate_frozen_inputs(
        acquisition,
        ortho,
        root=tmp_path,
        expected_acquisition_sha256=_sha(acquisition),
        expected_ortho_sha256=_sha(ortho),
    )
    assert len(validated.scenes) == 2

    validated.scenes[0].basic_path.write_bytes(b"changed")
    with pytest.raises(ProtocolError, match="byte size|SHA-256"):
        validate_frozen_inputs(
            acquisition,
            ortho,
            root=tmp_path,
            expected_acquisition_sha256=_sha(acquisition),
            expected_ortho_sha256=_sha(ortho),
        )

    partial_root = tmp_path / "partial"
    partial_acquisition, partial_ortho = _write_input_manifests(partial_root, partial_basic=True)
    with pytest.raises(ProtocolError, match="partial download"):
        validate_frozen_inputs(
            partial_acquisition,
            partial_ortho,
            root=partial_root,
            expected_acquisition_sha256=_sha(partial_acquisition),
            expected_ortho_sha256=_sha(partial_ortho),
        )


def test_input_validation_rejects_ortho_manifest_drift(tmp_path: Path):
    acquisition, ortho = _write_input_manifests(tmp_path)
    expected_ortho = _sha(ortho)
    payload = json.loads(ortho.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    ortho.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="ortho manifest SHA-256"):
        validate_frozen_inputs(
            acquisition,
            ortho,
            root=tmp_path,
            expected_acquisition_sha256=_sha(acquisition),
            expected_ortho_sha256=expected_ortho,
        )


def test_manifest_hash_parse_swap_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    acquisition, ortho = _write_input_manifests(tmp_path)
    expected_acquisition = _sha(acquisition)
    original_parse = basic_ortho._load_json_bytes
    swapped = False

    def swap_after_seal(payload: bytes, *, path: Path) -> dict[str, object]:
        nonlocal swapped
        if path == acquisition and not swapped:
            replacement = acquisition.with_name("replacement-acquisition.json")
            replacement.write_text('{"tampered": true}\n', encoding="utf-8")
            os.replace(replacement, acquisition)
            swapped = True
        return original_parse(payload, path=path)

    monkeypatch.setattr(basic_ortho, "_load_json_bytes", swap_after_seal)
    with pytest.raises(ProtocolError, match="changed between sealing and parsing"):
        validate_frozen_inputs(
            acquisition,
            ortho,
            root=tmp_path,
            expected_acquisition_sha256=expected_acquisition,
            expected_ortho_sha256=_sha(ortho),
        )
    assert swapped


def test_protocol_and_execution_identity_fail_closed(tmp_path: Path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    expected = _sha(protocol)
    assert validate_protocol_file(protocol, expected_sha256=expected) == expected
    protocol.write_text("changed protocol\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="hash mismatch"):
        validate_protocol_file(protocol, expected_sha256=expected)

    payload = {"mode": "design-only", "protocol": expected, "inputs": [1, 2]}
    assert execution_identity(payload) == execution_identity(payload)
    assert execution_identity(payload) != execution_identity({**payload, "inputs": [2, 1]})


def test_frozen_non_endpoint_constants_match_repository_files():
    root = Path(__file__).resolve().parents[1]
    preregistration = root / "docs" / "m1b_basic_ortho_sensitivity_preregistration.md"
    acquisition = root / "docs" / "basic_ortho_acquisition_manifest.json"
    ortho = root / "docs" / "input_manifest.json"
    assert sha256_file(preregistration) == FROZEN_PREREGISTRATION_SHA256
    assert sha256_file(acquisition) == FROZEN_ACQUISITION_MANIFEST_SHA256
    assert sha256_file(ortho) == FROZEN_ORTHO_MANIFEST_SHA256
    assert runner.DEFAULT_PREREGISTRATION == preregistration
    assert runner.DEFAULT_ACQUISITION_MANIFEST == acquisition
    assert runner.DEFAULT_ORTHO_MANIFEST == ortho
    assert basic_ortho.GOVERNING_FILE_KEYS == {
        "../tanager-spec/src/tanager_spec/__init__.py",
        "../tanager-spec/src/tanager_spec/bands.py",
        "../tanager-spec/src/tanager_spec/config.py",
        "../tanager-spec/src/tanager_spec/data/S2A_SRF.csv",
        "../tanager-spec/src/tanager_spec/data/S2B_SRF.csv",
        "../tanager-spec/src/tanager_spec/data/SOURCE.md",
        "../tanager-spec/src/tanager_spec/io.py",
        "../tanager-spec/src/tanager_spec/mask.py",
        "../tanager-spec/src/tanager_spec/sample.py",
        "../tanager-spec/src/tanager_spec/srf.py",
        "../tanager-spec/src/tanager_spec/stac.py",
        "scripts/run_basic_ortho_sensitivity.py",
        "src/tanager_rocks/__init__.py",
        "src/tanager_rocks/basic_ortho.py",
        "src/tanager_rocks/config.py",
        "src/tanager_rocks/features.py",
        "src/tanager_rocks/quality.py",
        "src/tanager_rocks/speclib.py",
        "src/tanager_rocks/unmix.py",
        "src/tanager_rocks/viz.py",
    }
    assert "tests/test_basic_ortho.py" not in runner.GOVERNING_FILES
    assert set(_RUNTIME_BINDING.loaded_local_files) == set(runner._PROJECT_MODULE_FILES.values())
    assert set(_RUNTIME_BINDING.loaded_dependency_files) == set(
        runner._TANAGER_SPEC_MODULE_FILES.values()
    )
    dependency_trust = _RUNTIME_BINDING.residual_dependency_trust["tanager_spec"]
    assert dependency_trust["hash_bound"] is True
    assert dependency_trust["classification"] == "captured_hash_bound_editable_dependency"
    assert dependency_trust["python_source_files"] == {
        path: _RUNTIME_BINDING.governing_hashes[path]
        for path in sorted(runner._TANAGER_SPEC_MODULE_FILES.values())
    }
    assert dependency_trust["package_data_files"] == {
        path: _RUNTIME_BINDING.governing_hashes[path]
        for path in sorted(runner._TANAGER_SPEC_PACKAGE_DATA_FILES)
    }

    acquisition_payload = json.loads(acquisition.read_text(encoding="utf-8"))
    observed_scenes = {item["site"]: item["scene_id"] for item in acquisition_payload["assets"]}
    assert observed_scenes == FROZEN_SCENES


def _copy_governing_sources(destination: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for logical_path in runner.GOVERNING_FILES:
        target = destination / logical_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / logical_path, target)


def _capsule_logical_path(capsule_root: Path, logical_path: str) -> Path:
    return Path(os.path.abspath(capsule_root / logical_path))


def _teardown_test_capsule(
    binding: object,
    *,
    expected_error: str | None = None,
) -> None:
    if expected_error is None:
        runner._teardown_runtime(binding)
        return
    with pytest.raises(runner.BootstrapError, match=expected_error):
        runner._teardown_runtime(binding)


def test_bootstrap_mutate_launch_restore_executes_only_bound_source(tmp_path: Path):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    capture = runner._capture_runtime_sources(capsule_root)
    basic_path = capsule_root / "src/tanager_rocks/basic_ortho.py"
    original = basic_path.read_bytes()
    tampered = original + b"\nCAPSULE_MUTATE_LAUNCH_SENTINEL = True\n"
    mutation_seen = False

    def mutate_before_exec(fullname: str) -> None:
        nonlocal mutation_seen
        if fullname.endswith(".basic_ortho"):
            basic_path.write_bytes(tampered)
            mutation_seen = True

    def restore_after_exec(fullname: str) -> None:
        if fullname.endswith(".basic_ortho"):
            basic_path.write_bytes(original)

    binding = runner._load_runtime_capsule(
        capture,
        before_exec=mutate_before_exec,
        after_exec=restore_after_exec,
    )
    try:
        assert mutation_seen
        assert not hasattr(binding.basic_ortho, "CAPSULE_MUTATE_LAUNCH_SENTINEL")
        assert runner._validate_runtime_binding(binding) == capture.governing_hashes
    finally:
        _teardown_test_capsule(binding)


def test_tanager_spec_source_replacement_after_capture_executes_no_replacement(
    tmp_path: Path,
):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    capture = runner._capture_runtime_sources(capsule_root)
    config_path = _capsule_logical_path(
        capsule_root,
        runner._TANAGER_SPEC_MODULE_FILES["tanager_spec.config"],
    )
    replacement = config_path.with_name("replacement-config.py")
    replacement.write_bytes(
        config_path.read_bytes()
        + b"\nraise AssertionError('replacement tanager_spec source executed')\n"
    )
    os.replace(replacement, config_path)

    with pytest.raises(runner.BootstrapError, match="governing source changed"):
        runner._load_runtime_capsule(capture)


def test_tanager_spec_deferred_import_and_package_data_use_captured_bytes(tmp_path: Path):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    basic_path = capsule_root / "src/tanager_rocks/basic_ortho.py"
    basic_path.write_bytes(
        basic_path.read_bytes()
        + b"\ndef _deferred_dependency_probe():\n"
        + b"    import importlib as _dependency_importlib\n"
        + b"    from importlib.resources import files as _dependency_files\n"
        + b'    module = _dependency_importlib.import_module("tanager_spec.srf")\n'
        + b'    payload = (_dependency_files("tanager_spec.data") / "SOURCE.md").read_bytes()\n'
        + b"    return module, payload\n"
    )
    capture = runner._capture_runtime_sources(capsule_root)
    binding = runner._load_runtime_capsule(capture)
    source_path = _capsule_logical_path(
        capsule_root,
        runner._TANAGER_SPEC_MODULE_FILES["tanager_spec.srf"],
    )
    data_logical_path = "../tanager-spec/src/tanager_spec/data/SOURCE.md"
    data_path = _capsule_logical_path(capsule_root, data_logical_path)
    source_bytes = source_path.read_bytes()
    data_bytes = data_path.read_bytes()
    try:
        source_path.write_bytes(
            source_bytes + b"\nraise AssertionError('deferred replacement source executed')\n"
        )
        data_path.write_bytes(b"replacement package data\n")
        sys.modules.pop("tanager_spec.srf")
        module, observed_data = binding.basic_ortho._deferred_dependency_probe()
        assert binding.finder._is_managed_dependency_module("tanager_spec.srf", module)
        assert observed_data == capture.source_bytes[data_logical_path]
    finally:
        source_path.write_bytes(source_bytes)
        data_path.write_bytes(data_bytes)
    try:
        assert runner._validate_runtime_binding(binding) == capture.governing_hashes
    finally:
        _teardown_test_capsule(binding)


@pytest.mark.parametrize("mutation", ["added", "removed"])
def test_tanager_spec_exact_inventory_drift_is_rejected(
    tmp_path: Path,
    mutation: str,
):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    capture = runner._capture_runtime_sources(capsule_root)
    binding = runner._load_runtime_capsule(capture)
    if mutation == "added":
        changed_path = _capsule_logical_path(
            capsule_root,
            "../tanager-spec/src/tanager_spec/deferred_escape.py",
        )
        changed_path.write_text("ESCAPED = True\n", encoding="utf-8")
    else:
        changed_path = _capsule_logical_path(
            capsule_root,
            runner._TANAGER_SPEC_MODULE_FILES["tanager_spec.mask"],
        )
        changed_path.unlink()
    with pytest.raises(runner.BootstrapError, match="package inventory differs"):
        runner._validate_runtime_binding(binding)
    _teardown_test_capsule(binding, expected_error="package inventory differs")


def test_tanager_spec_package_data_drift_is_rejected(tmp_path: Path):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    capture = runner._capture_runtime_sources(capsule_root)
    binding = runner._load_runtime_capsule(capture)
    data_path = _capsule_logical_path(
        capsule_root,
        "../tanager-spec/src/tanager_spec/data/S2A_SRF.csv",
    )
    data_path.write_bytes(data_path.read_bytes() + b"\n")
    with pytest.raises(runner.BootstrapError, match="governing source changed"):
        runner._validate_runtime_binding(binding)
    _teardown_test_capsule(binding, expected_error="governing source changed")


def test_tanager_spec_rejects_new_sibling_local_module_origin(tmp_path: Path):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    binding = runner._load_runtime_capsule(runner._capture_runtime_sources(capsule_root))
    escape_path = _capsule_logical_path(
        capsule_root,
        "../tanager-spec/src/tanager_spec/runtime_escape.py",
    )
    escape_path.write_text("ESCAPED = True\n", encoding="utf-8")
    escape_name = "_m1b_tanager_spec_sibling_escape"
    escape_spec = importlib.util.spec_from_file_location(escape_name, escape_path)
    assert escape_spec is not None and escape_spec.loader is not None
    escape_module = importlib.util.module_from_spec(escape_spec)
    sys.modules[escape_name] = escape_module
    escape_spec.loader.exec_module(escape_module)

    with pytest.raises(
        runner.BootstrapError,
        match="newly loaded module origins under the tanager-spec sibling",
    ):
        runner._teardown_runtime(binding)
    assert escape_name not in sys.modules


def test_tanager_spec_rejects_out_of_capsule_canonical_module(tmp_path: Path):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    binding = runner._load_runtime_capsule(runner._capture_runtime_sources(capsule_root))
    intruder_name = "tanager_spec.runtime_escape"
    intruder = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(intruder_name, loader=None)
    )
    sys.modules[intruder_name] = intruder
    with pytest.raises(ModuleNotFoundError, match="out-of-capsule tanager_spec module blocked"):
        binding.finder.guarded_import("tanager_spec.config", {}, {}, (), 0)
    _teardown_test_capsule(
        binding,
        expected_error="out-of-capsule tanager_spec modules were observed",
    )
    assert intruder_name not in sys.modules


def test_tanager_spec_preloaded_canonical_modules_are_restored_exactly(tmp_path: Path):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    capture = runner._capture_runtime_sources(capsule_root)
    outer_modules = runner._canonical_dependency_modules()
    for name in tuple(outer_modules):
        del sys.modules[name]
    fake_names = ("tanager_spec", "tanager_spec.config", "tanager_spec.preexisting")
    fake_modules = {
        name: importlib.util.module_from_spec(importlib.machinery.ModuleSpec(name, loader=None))
        for name in fake_names
    }
    sys.modules.update(fake_modules)
    try:
        binding = runner._load_runtime_capsule(capture)
        try:
            assert set(runner._canonical_dependency_modules()) == set(
                binding.finder.dependency_module_records
            )
            assert all(sys.modules.get(name) is not module for name, module in fake_modules.items())
        finally:
            runner._teardown_runtime(binding)
        assert all(sys.modules[name] is module for name, module in fake_modules.items())
    finally:
        for name in tuple(runner._canonical_dependency_modules()):
            del sys.modules[name]
        sys.modules.update(outer_modules)


def test_bootstrap_blocks_and_records_ungoverned_local_import(tmp_path: Path):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    ungoverned = capsule_root / "src/tanager_rocks/endpoint_sentinel.py"
    ungoverned.write_text("raise AssertionError('ungoverned module executed')\n", encoding="utf-8")
    binding = runner._load_runtime_capsule(runner._capture_runtime_sources(capsule_root))
    try:
        with pytest.raises(ModuleNotFoundError, match="ungoverned capsule-local import blocked"):
            importlib.import_module(f"{binding.prefix}.endpoint_sentinel")
        with pytest.raises(runner.BootstrapError, match="ungoverned capsule-local imports"):
            runner._validate_runtime_binding(binding)
    finally:
        _teardown_test_capsule(
            binding,
            expected_error="ungoverned capsule-local imports",
        )


def test_bootstrap_rejects_preloaded_canonical_project_import(tmp_path: Path):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    basic_path = capsule_root / "src/tanager_rocks/basic_ortho.py"
    basic_path.write_bytes(
        basic_path.read_bytes()
        + b"\nimport importlib as _canonical_importlib\n"
        + b'_canonical_importlib.import_module("tanager_rocks.escape")\n'
    )
    preloaded = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("tanager_rocks.escape", loader=None)
    )
    assert "tanager_rocks.escape" not in sys.modules
    sys.modules["tanager_rocks.escape"] = preloaded
    try:
        with pytest.raises(ModuleNotFoundError, match="canonical tanager_rocks import blocked"):
            runner._load_runtime_capsule(runner._capture_runtime_sources(capsule_root))
        assert sys.modules["tanager_rocks.escape"] is preloaded
    finally:
        sys.modules.pop("tanager_rocks.escape", None)


def test_bootstrap_rejects_new_repo_local_module_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    (capsule_root / "repo_escape.py").write_text("ESCAPED = True\n", encoding="utf-8")
    basic_path = capsule_root / "src/tanager_rocks/basic_ortho.py"
    basic_path.write_bytes(
        basic_path.read_bytes()
        + b"\nimport importlib as _escape_importlib\n"
        + b'_escape_importlib.import_module("repo_escape")\n'
    )
    monkeypatch.syspath_prepend(str(capsule_root))

    with pytest.raises(runner.BootstrapError, match="newly loaded module origins"):
        runner._load_runtime_capsule(runner._capture_runtime_sources(capsule_root))
    assert "repo_escape" not in sys.modules


def test_launcher_binds_compiled_bytes_across_pathname_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    temporary_launcher = capsule_root / "scripts/run_basic_ortho_sensitivity_launcher.py"
    temporary_runner = capsule_root / "scripts/run_basic_ortho_sensitivity.py"
    shutil.copy2(_LAUNCHER_PATH, temporary_launcher)

    original = temporary_runner.read_bytes()
    replacement = temporary_runner.with_name("replacement-runner.py")
    replacement.write_bytes(original + b"\nPATHNAME_REPLACEMENT_SENTINEL = True\n")
    temporary_launcher_name = "_m1b_path_replacement_launcher"
    temporary_runner_name = "_m1b_path_replacement_runner"
    launcher_spec = importlib.util.spec_from_file_location(
        temporary_launcher_name,
        temporary_launcher,
    )
    assert launcher_spec is not None and launcher_spec.loader is not None
    temporary_launcher_module = importlib.util.module_from_spec(launcher_spec)
    sys.modules[temporary_launcher_name] = temporary_launcher_module
    launcher_spec.loader.exec_module(temporary_launcher_module)

    compiled_payloads: list[bytes] = []
    original_compile = builtins.compile
    replaced = False

    def replace_path_then_compile(source, filename, mode, *args, **kwargs):
        nonlocal replaced
        if filename == str(temporary_runner):
            assert isinstance(source, bytes)
            compiled_payloads.append(source)
            if not replaced:
                os.replace(replacement, temporary_runner)
                replaced = True
        return original_compile(source, filename, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "compile", replace_path_then_compile)
    try:
        launched = temporary_launcher_module.load_runner_module(
            temporary_runner,
            module_name=temporary_runner_name,
        )
        assert replaced
        assert compiled_payloads == [original]
        assert compiled_payloads[0] is launched._BOOTSTRAP_RUNNER_SOURCE
        assert launched._BOOTSTRAP_RUNNER_SHA256 == hashlib.sha256(original).hexdigest()
        assert not hasattr(launched, "PATHNAME_REPLACEMENT_SENTINEL")
        assert launched.LAUNCHER_RESIDUAL_TRUST == {
            "classification": "residual_execution_bootstrap_trust",
            "hash_bound": False,
            "path": str(temporary_launcher),
            "risk": "Python loads the minimal launcher before descriptor-bound runner handoff",
        }
        with pytest.raises(
            launched.BootstrapError,
            match="runner source pathname changed after descriptor-bound launch",
        ):
            launched._capture_runtime_sources()
    finally:
        sys.modules.pop(temporary_runner_name, None)
        sys.modules.pop(temporary_launcher_name, None)


def test_capsule_blocks_deferred_canonical_import_until_explicit_teardown(tmp_path: Path):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    basic_path = capsule_root / "src/tanager_rocks/basic_ortho.py"
    basic_path.write_bytes(
        basic_path.read_bytes()
        + b"\nimport importlib as _deferred_importlib\n"
        + b"def _deferred_canonical_import():\n"
        + b'    return _deferred_importlib.import_module("tanager_rocks.escape")\n'
    )
    preloaded = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("tanager_rocks.escape", loader=None)
    )
    assert "tanager_rocks.escape" not in sys.modules
    sys.modules["tanager_rocks.escape"] = preloaded
    binding = runner._load_runtime_capsule(runner._capture_runtime_sources(capsule_root))
    try:
        assert "tanager_rocks.escape" not in sys.modules
        with pytest.raises(ModuleNotFoundError, match="canonical tanager_rocks import blocked"):
            binding.basic_ortho._deferred_canonical_import()
        with pytest.raises(runner.BootstrapError, match="canonical tanager_rocks imports"):
            runner._validate_runtime_binding(binding)
    finally:
        _teardown_test_capsule(
            binding,
            expected_error="canonical tanager_rocks imports",
        )
    try:
        assert sys.modules["tanager_rocks.escape"] is preloaded
        assert binding.finder not in sys.meta_path
        assert not binding.finder.active
    finally:
        sys.modules.pop("tanager_rocks.escape", None)


def test_teardown_rejects_post_run_canonical_inventory_and_restores_prior_state(
    tmp_path: Path,
):
    capsule_root = tmp_path / "capsule-root"
    _copy_governing_sources(capsule_root)
    preexisting_name = "tanager_rocks.preexisting"
    intruder_name = "tanager_rocks.post_run_escape"
    preexisting = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(preexisting_name, loader=None)
    )
    intruder = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(intruder_name, loader=None)
    )
    assert preexisting_name not in sys.modules
    assert intruder_name not in sys.modules
    sys.modules[preexisting_name] = preexisting
    binding = runner._load_runtime_capsule(runner._capture_runtime_sources(capsule_root))
    try:
        assert preexisting_name not in sys.modules
        sys.modules[intruder_name] = intruder
        with pytest.raises(
            runner.BootstrapError,
            match="canonical tanager_rocks module inventory was populated",
        ):
            runner._teardown_runtime(binding)
        assert sys.modules[preexisting_name] is preexisting
        assert intruder_name not in sys.modules
        assert binding.finder not in sys.meta_path
        assert not binding.finder.active
        assert not any(
            name == binding.prefix or name.startswith(f"{binding.prefix}.") for name in sys.modules
        )
    finally:
        if binding.finder.active:
            _teardown_test_capsule(binding)
        sys.modules.pop(preexisting_name, None)
        sys.modules.pop(intruder_name, None)


def test_design_document_declares_no_threshold_or_interpolation():
    design = design_document(
        protocol_sha256="protocol",
        acquisition_manifest_sha256="acquisition",
        ortho_manifest_sha256="ortho",
        governing_files=GOVERNING_HASHES,
    )
    assert design["mode"] == "design-only"
    assert design["scientific_endpoint_values_inspected"] is False
    assert design["mapping_contract"]["mapping_distance_threshold"] is None
    assert design["mapping_contract"]["interpolation"] is None
    assert design["mapping_contract"]["projectable_rank"] == 2
    assert "nearest geolocated basic source" in design["mapping_contract"]["assignment"]
    assert "source QA is applied after geometric selection" in design["mapping_contract"]["support"]
    assert "mapping_status.tif" in design["planned_artifacts"]


def test_mapping_bundle_is_atomic_identity_scoped_and_traceable(tmp_path: Path):
    mapping, grid = _mapping_fixture()
    scene = FrozenSceneInput(
        site="goldfield",
        scene_id="scene",
        basic_path=tmp_path / "basic.h5",
        basic_size_bytes=1,
        basic_sha256="basic",
        ortho_path=tmp_path / "ortho.h5",
        ortho_size_bytes=1,
        ortho_sha256="ortho",
        basic_stac_shape=(2, 4),
        basic_stac_crs="EPSG:4326",
        basic_stac_resolution_m=40.0,
        ortho_stac_shape=(2, 3),
        ortho_stac_crs="EPSG:32611",
        ortho_stac_resolution_m=30.0,
    )
    spectral_copy = SpectrumCopyAudit(2, 6, 5, 4, 4 / 5, 4, 3, 3 / 4)
    manifest = write_mapping_bundle(
        tmp_path / "outputs",
        scene=scene,
        grid=grid,
        mapping=mapping,
        protocol_sha256="protocol",
        acquisition_manifest_sha256="acquisition",
        ortho_manifest_sha256="ortho-manifest",
        governing_files=GOVERNING_HASHES,
        qa_invalid_counts={"beta_cloud_mask": 1},
        spectral_copy_audit=spectral_copy,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["scientific_endpoint_values_inspected"] is False
    assert payload["mapping_distance_threshold_m"] is None
    assert set(payload["artifacts"]) == {
        "source_index.tif",
        "mapping_distance_m.tif",
        "source_multiplicity.tif",
        "mapping_status.tif",
    }
    with rasterio.open(manifest.parent / "source_index.tif") as dataset:
        assert dataset.descriptions == ("basic_source_row", "basic_source_col")
        assert dataset.tags()["execution_id"] == payload["execution_id"]
        np.testing.assert_array_equal(dataset.read(1), mapping.source_row)
        np.testing.assert_array_equal(dataset.read(2), mapping.source_col)
    with rasterio.open(manifest.parent / "mapping_status.tif") as dataset:
        assert dataset.descriptions == ("mapping_status_code",)
        np.testing.assert_array_equal(dataset.read(1), mapping.target_status)
    assert not list(manifest.parent.glob(".*.tmp.tif"))

    changed = replace(scene, basic_sha256="different")
    second = write_mapping_bundle(
        tmp_path / "outputs",
        scene=changed,
        grid=grid,
        mapping=mapping,
        protocol_sha256="protocol",
        acquisition_manifest_sha256="acquisition",
        ortho_manifest_sha256="ortho-manifest",
        governing_files=GOVERNING_HASHES,
        qa_invalid_counts={},
        spectral_copy_audit=spectral_copy,
    )
    assert second.parent != manifest.parent


def test_strict_json_dump_is_atomic_and_rejects_nonstandard_nan(tmp_path: Path):
    path = tmp_path / "manifest.json"
    strict_json_dump(path, {"finite": 1.0, "missing": np.nan})
    assert json.loads(path.read_text(encoding="utf-8")) == {"finite": 1.0, "missing": None}
    assert sha256_file(path) == _sha(path)
    assert not list(tmp_path.glob(".*.tmp"))


def _runner_fixture(
    tmp_path: Path,
) -> tuple[
    ValidatedInputs,
    dict[str, tuple[basic_ortho.ProductSchema, basic_ortho.ProductSchema]],
    dict[str, OrthoGrid],
]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    scenes = []
    schemas = {}
    grids = {}
    for site, scene_id in FROZEN_SCENES.items():
        scene_root = tmp_path / site
        scene_root.mkdir()
        basic_path = scene_root / "basic.h5"
        ortho_path = scene_root / "ortho.h5"
        _write_basic(basic_path, shape=(2, 4), bands=5)
        _write_ortho(ortho_path, shape=(2, 3), bands=5)
        basic_schema = inspect_basic_schema(basic_path, expected_shape=(2, 4))
        ortho_schema = inspect_ortho_schema(ortho_path, expected_shape=(2, 3))
        grid = load_ortho_grid(
            ortho_path,
            expected_shape=(2, 3),
            expected_crs="EPSG:32611",
            expected_resolution_m=30.0,
        )
        scenes.append(
            FrozenSceneInput(
                site=site,
                scene_id=scene_id,
                basic_path=basic_path,
                basic_size_bytes=basic_path.stat().st_size,
                basic_sha256=sha256_file(basic_path),
                ortho_path=ortho_path,
                ortho_size_bytes=ortho_path.stat().st_size,
                ortho_sha256=sha256_file(ortho_path),
                basic_stac_shape=(2, 4),
                basic_stac_crs="EPSG:4326",
                basic_stac_resolution_m=40.0,
                ortho_stac_shape=(2, 3),
                ortho_stac_crs="EPSG:32611",
                ortho_stac_resolution_m=30.0,
            )
        )
        schemas[site] = (basic_schema, ortho_schema)
        grids[site] = grid
    preregistration = tmp_path / "protocol.md"
    acquisition_manifest = tmp_path / "acquisition.json"
    ortho_manifest = tmp_path / "ortho.json"
    preregistration.write_text("synthetic sealed protocol\n", encoding="utf-8")
    acquisition_manifest.write_text('{"fixture": "acquisition"}\n', encoding="utf-8")
    ortho_manifest.write_text('{"fixture": "ortho"}\n', encoding="utf-8")
    return (
        ValidatedInputs(
            acquisition_manifest_path=acquisition_manifest,
            acquisition_manifest_sha256=sha256_file(acquisition_manifest),
            ortho_manifest_path=ortho_manifest,
            ortho_manifest_sha256=sha256_file(ortho_manifest),
            scenes=tuple(scenes),
        ),
        schemas,
        grids,
    )


def _configure_runner(
    monkeypatch: pytest.MonkeyPatch,
    inputs: ValidatedInputs,
    schemas: dict[str, tuple[basic_ortho.ProductSchema, basic_ortho.ProductSchema]],
    grids: dict[str, OrthoGrid],
) -> None:
    def validate_protocol_snapshot(_path: Path, **kwargs) -> str:
        payload = kwargs["snapshot_bytes"]
        return hashlib.sha256(payload).hexdigest()

    def validate_input_snapshots(*_args, **kwargs) -> ValidatedInputs:
        assert hashlib.sha256(kwargs["acquisition_snapshot_bytes"]).hexdigest() == (
            inputs.acquisition_manifest_sha256
        )
        assert hashlib.sha256(kwargs["ortho_snapshot_bytes"]).hexdigest() == (
            inputs.ortho_manifest_sha256
        )
        return inputs

    monkeypatch.setattr(basic_ortho, "validate_protocol_file", validate_protocol_snapshot)
    monkeypatch.setattr(
        basic_ortho,
        "validate_frozen_inputs",
        validate_input_snapshots,
    )


def _run_mapping_fixture(output: Path, inputs: ValidatedInputs) -> Path:
    preregistration = inputs.acquisition_manifest_path.with_name("protocol.md")
    return runner.run_mapping_only(
        preregistration=preregistration,
        acquisition_manifest=inputs.acquisition_manifest_path,
        ortho_manifest=inputs.ortho_manifest_path,
        output_dir=output,
    )


def _coordinated_one_index_remap(mapping: NativeToOrthoMapping) -> NativeToOrthoMapping:
    selected = np.isin(
        mapping.target_status,
        [int(TARGET_BASIC_QA_INVALID), int(TARGET_MAPPED)],
    )
    source_total = mapping.source_status.size
    shifted_flat = mapping.source_flat_index.copy()
    shifted_flat[selected] = (shifted_flat[selected] + 1) % source_total
    shifted_row = mapping.source_row.copy()
    shifted_col = mapping.source_col.copy()
    shifted_row[selected], shifted_col[selected] = np.unravel_index(
        shifted_flat[selected],
        mapping.source_status.shape,
    )
    counts_by_source = np.bincount(
        shifted_flat[selected],
        minlength=source_total,
    ).astype(np.uint32)
    multiplicity = mapping.source_multiplicity.copy()
    multiplicity[selected] = counts_by_source[shifted_flat[selected]]
    target_count_per_source = counts_by_source.reshape(mapping.source_status.shape)
    source_status = np.where(
        mapping.source_status == basic_ortho.SOURCE_INVALID_GEOLOCATION,
        basic_ortho.SOURCE_INVALID_GEOLOCATION,
        basic_ortho.SOURCE_UNUSED,
    ).astype(np.uint8)
    source_status[target_count_per_source > 0] = basic_ortho.SOURCE_USED
    used = int((counts_by_source > 0).sum())
    counts = replace(
        mapping.counts,
        used_source_samples=used,
        unused_source_samples=source_total - used,
        sources_with_multiple_target_cells=int((counts_by_source > 1).sum()),
        duplicate_target_assignments=int((counts_by_source[counts_by_source > 0] - 1).sum()),
    )
    return replace(
        mapping,
        source_row=shifted_row,
        source_col=shifted_col,
        source_flat_index=shifted_flat,
        source_multiplicity=multiplicity,
        source_status=source_status,
        target_count_per_source=target_count_per_source,
        counts=counts,
    )


def test_false_bundle_accepted_regression_rejects_pre_generation_one_index_remap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    original_map = basic_ortho.map_native_to_ortho
    production_map_calls = 0

    def compromised_map(*args, **kwargs):
        nonlocal production_map_calls
        production_map_calls += 1
        return _coordinated_one_index_remap(original_map(*args, **kwargs))

    monkeypatch.setattr(basic_ortho, "map_native_to_ortho", compromised_map)
    output = tmp_path / "outputs"
    with pytest.raises(ProtocolError, match="externally attested mapping semantics differ"):
        _run_mapping_fixture(output, inputs)
    assert production_map_calls == len(FROZEN_SCENES)
    assert list(output.iterdir()) == []
    assert not list(output.glob(".*.staging"))


def test_independent_spectral_copy_oracle_rejects_tampered_production_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    production_audit = basic_ortho.exact_spectrum_copy_audit
    audit_calls = 0

    def disabled_copy_matches(*args, **kwargs):
        nonlocal audit_calls
        audit_calls += 1
        observed = production_audit(*args, **kwargs)
        return replace(
            observed,
            ortho_exact_match_to_any_basic=0,
            ortho_exact_match_to_any_basic_fraction=(0.0 if observed.valid_ortho_spectra else None),
            mapped_exact_match_to_selected_basic=0,
            mapped_exact_match_to_selected_basic_fraction=(
                0.0 if observed.mapped_valid_ortho_spectra else None
            ),
        )

    monkeypatch.setattr(basic_ortho, "exact_spectrum_copy_audit", disabled_copy_matches)
    output = tmp_path / "outputs"
    with pytest.raises(ProtocolError, match="externally attested spectral-copy audit"):
        _run_mapping_fixture(output, inputs)
    assert audit_calls == len(FROZEN_SCENES)
    assert list(output.iterdir()) == []
    assert not list(output.glob(".*.staging"))


@pytest.mark.parametrize(
    "failure_call",
    range(1, len(FROZEN_SCENES) * len(MAPPING_ARTIFACT_NAMES) + 1),
)
def test_mapping_run_raster_failures_leave_no_final_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    original_write = basic_ortho._atomic_write_raster
    calls = 0

    def injected_failure(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError(f"injected raster failure {failure_call}")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(basic_ortho, "_atomic_write_raster", injected_failure)
    output = tmp_path / "outputs"
    with pytest.raises(OSError, match="injected raster failure"):
        _run_mapping_fixture(output, inputs)
    assert calls == failure_call
    assert list(output.iterdir()) == []
    assert not list(output.glob(".*.staging"))


def test_mapping_run_failure_between_scenes_leaves_no_final_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    original_write = basic_ortho.write_mapping_bundle
    calls = 0

    def injected_failure(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected between scenes")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(basic_ortho, "write_mapping_bundle", injected_failure)
    output = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="injected between scenes"):
        _run_mapping_fixture(output, inputs)
    assert calls == 2
    assert list(output.iterdir()) == []
    assert not list(output.glob(".*.staging"))


def test_existing_accepted_mapping_bundle_cannot_be_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    output = tmp_path / "outputs"
    manifest = _run_mapping_fixture(output, inputs)
    run_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert run_payload["residual_dependency_trust"]["tanager_spec"]["hash_bound"] is True
    assert (
        run_payload["residual_dependency_trust"]["tanager_spec"]["classification"]
        == "captured_hash_bound_editable_dependency"
    )
    assert (
        run_payload["execution_identity_inputs"]["residual_dependency_trust"]
        == (run_payload["residual_dependency_trust"])
    )
    snapshot_evidence = run_payload["input_snapshot_evidence"]
    assert set(snapshot_evidence) == {
        "preregistration",
        "acquisition_manifest",
        "ortho_manifest",
        "scenes",
    }
    evidence_records = [
        snapshot_evidence["preregistration"],
        snapshot_evidence["acquisition_manifest"],
        snapshot_evidence["ortho_manifest"],
        *(
            snapshot_evidence["scenes"][site][product]
            for site in FROZEN_SCENES
            for product in ("basic", "ortho")
        ),
    ]
    for record in evidence_records:
        assert record["source_pre"] == record["source_post"]
        assert record["copied_byte_count"] == record["source_pre"]["size_bytes"]
        assert len(record["sha256"]) == 64
    accepted_files = {
        path.relative_to(manifest.parent).as_posix(): sha256_file(path)
        for path in manifest.parent.rglob("*")
        if path.is_file()
    }
    expected_file_count = 1 + len(FROZEN_SCENES) * (1 + len(MAPPING_ARTIFACT_NAMES))
    assert len(accepted_files) == expected_file_count

    with pytest.raises(ProtocolError, match="refusing to replace existing accepted bundle"):
        _run_mapping_fixture(output, inputs)
    observed_files = {
        path.relative_to(manifest.parent).as_posix(): sha256_file(path)
        for path in manifest.parent.rglob("*")
        if path.is_file()
    }
    assert observed_files == accepted_files
    assert not list(output.glob(".*.staging"))


def test_resource_pilot_selector_identity_and_outputs_are_non_promotable(tmp_path: Path):
    inputs, _, _ = _runner_fixture(tmp_path / "inputs")
    scene = select_resource_pilot_scene(inputs)
    assert scene.site == RESOURCE_PILOT_DEFAULT_SITE
    assert RESOURCE_PILOT_DEFAULT_BRANCH == "B"

    first = resource_pilot_document(
        scene,
        branch=RESOURCE_PILOT_DEFAULT_BRANCH,
        protocol_sha256="protocol",
        acquisition_manifest_sha256="acquisition",
        ortho_manifest_sha256="ortho",
        governing_files=GOVERNING_HASHES,
        telemetry=ResourcePilotTelemetry(1.0, 2.0, 3.0, 4, 5, 6),
    )
    second = resource_pilot_document(
        scene,
        branch=RESOURCE_PILOT_DEFAULT_BRANCH,
        protocol_sha256="protocol",
        acquisition_manifest_sha256="acquisition",
        ortho_manifest_sha256="ortho",
        governing_files=GOVERNING_HASHES,
        telemetry=ResourcePilotTelemetry(9.0, 8.0, 7.0, 6, 5, 4),
    )
    assert first["execution_id"] == second["execution_id"]
    assert str(first["execution_id"]).startswith("resource-pilot-non-promotable-")
    assert first["scientific_execution_promotable"] is False
    assert first["scientific_endpoint_values_inspected"] is False
    assert first["scientific_outputs_produced"] is False
    assert first["output_contract"] == {
        "allowed_files": ["resource_pilot_manifest.json"],
        "scientific_metrics": [],
        "scientific_maps": [],
        "reflectance_values": [],
    }
    with pytest.raises(ProtocolError, match="site is not frozen"):
        select_resource_pilot_scene(inputs, site="unfrozen", branch="B")
    with pytest.raises(ProtocolError, match="branch is not declared"):
        select_resource_pilot_scene(inputs, site=RESOURCE_PILOT_DEFAULT_SITE, branch="X")


def test_schema_mapping_and_resource_pilot_do_not_call_scientific_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)

    def forbidden_endpoint(*_args, **_kwargs):
        raise AssertionError("scientific endpoint invoked")

    monkeypatch.setattr(basic_ortho, "project_scalar_nearest", forbidden_endpoint)
    schema_manifest = runner.run_schema_only(
        preregistration=inputs.acquisition_manifest_path.with_name("protocol.md"),
        acquisition_manifest=inputs.acquisition_manifest_path,
        ortho_manifest=inputs.ortho_manifest_path,
        output_dir=tmp_path / "schema",
    )
    mapping_manifest = _run_mapping_fixture(tmp_path / "mapping", inputs)
    pilot_manifest = runner.run_resource_pilot(
        preregistration=inputs.acquisition_manifest_path.with_name("protocol.md"),
        acquisition_manifest=inputs.acquisition_manifest_path,
        ortho_manifest=inputs.ortho_manifest_path,
        output_dir=tmp_path / "pilot",
    )
    assert schema_manifest.name == "schema_manifest.json"
    assert mapping_manifest.name == "mapping_run_manifest.json"
    assert pilot_manifest.name == "resource_pilot_manifest.json"
    assert [path.name for path in pilot_manifest.parent.iterdir()] == [
        "resource_pilot_manifest.json"
    ]
    pilot_payload = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    assert set(pilot_payload["telemetry"]) == {
        "loaded_array_bytes",
        "process_max_rss_after_bytes",
        "process_max_rss_before_bytes",
        "system_cpu_seconds",
        "user_cpu_seconds",
        "wall_seconds",
    }


@pytest.mark.parametrize("mutation", ["content", "symlink"])
def test_post_verification_mutation_is_not_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    output = tmp_path / "outputs"
    run_id = "bounded-test-run"
    external = tmp_path / "external.json"
    external.write_text('{"state": "mutated"}\n', encoding="utf-8")

    def writer(staging_root: Path) -> Path:
        manifest = staging_root / "manifest.json"
        manifest.write_text('{"state": "verified"}\n', encoding="utf-8")
        return manifest

    def verifier(bundle_root: Path) -> None:
        manifest = bundle_root / "manifest.json"
        if manifest.is_symlink():
            raise ProtocolError("manifest became a symlink")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload != {"state": "verified"}:
            raise ProtocolError("manifest changed after verification")

    original_promote = basic_ortho._rename_directory_noreplace

    def mutate_then_promote(source: Path, destination: Path) -> None:
        manifest = source / "manifest.json"
        if mutation == "content":
            manifest.write_text('{"state": "mutated"}\n', encoding="utf-8")
        else:
            manifest.unlink()
            manifest.symlink_to(external)
        original_promote(source, destination)

    monkeypatch.setattr(basic_ortho, "_rename_directory_noreplace", mutate_then_promote)
    with pytest.raises(ProtocolError, match="changed after verification|became a symlink"):
        basic_ortho.atomic_write_run_bundle(
            output,
            run_id,
            writer=writer,
            verifier=verifier,
        )
    assert not (output / run_id).exists()
    assert not list(output.glob(".*.staging"))
    assert external.is_file()


def test_post_promotion_verifier_failure_removes_owned_bundle(tmp_path: Path):
    output = tmp_path / "outputs"
    verification_calls = 0

    def writer(staging_root: Path) -> Path:
        manifest = staging_root / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return manifest

    def verifier(_bundle_root: Path) -> None:
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 2:
            raise ProtocolError("injected post-promotion verifier failure")

    with pytest.raises(ProtocolError, match="injected post-promotion verifier failure"):
        basic_ortho.atomic_write_run_bundle(
            output,
            "verifier-failure-run",
            writer=writer,
            verifier=verifier,
        )
    assert verification_calls == 2
    assert list(output.iterdir()) == []


def test_mapping_dependency_source_drift_between_verification_passes_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    initial = _RUNTIME_BINDING.governing_hashes
    drifted = dict(initial)
    drifted["../tanager-spec/src/tanager_spec/config.py"] = "f" * 64
    hash_calls = 0

    def hashes_with_post_promotion_drift(_binding: object) -> dict[str, str]:
        nonlocal hash_calls
        hash_calls += 1
        return dict(drifted if hash_calls >= 3 else initial)

    monkeypatch.setattr(runner, "_observed_governing_hashes", hashes_with_post_promotion_drift)
    output = tmp_path / "outputs"
    with pytest.raises(ProtocolError, match="governing source changed after capsule binding"):
        _run_mapping_fixture(output, inputs)
    assert hash_calls == 3
    assert list(output.iterdir()) == []
    assert not list(output.glob(".*.staging"))


def test_preregistration_mutation_after_precheck_is_rejected_post_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    preregistration = inputs.acquisition_manifest_path.with_name("protocol.md")
    output = tmp_path / "outputs"
    original_promote = basic_ortho._rename_directory_noreplace
    mutated = False

    def mutate_preregistration_then_promote(source: Path, destination: Path) -> None:
        nonlocal mutated
        if destination.parent == output and not destination.name.startswith("."):
            preregistration.write_text("mutated after pre-promotion check\n", encoding="utf-8")
            mutated = True
        original_promote(source, destination)

    monkeypatch.setattr(
        basic_ortho,
        "_rename_directory_noreplace",
        mutate_preregistration_then_promote,
    )
    with pytest.raises(
        runner.IndependentVerificationError,
        match="source metadata changed after snapshot sealing",
    ):
        _run_mapping_fixture(output, inputs)
    assert mutated
    assert list(output.iterdir()) == []
    assert not list(output.glob(".*.staging"))


def test_destination_race_is_exclusive_and_does_not_replace_racer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "outputs"
    run_id = "destination-race-run"

    def writer(staging_root: Path) -> Path:
        manifest = staging_root / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return manifest

    original_promote = basic_ortho._rename_directory_noreplace

    def race_then_promote(source: Path, destination: Path) -> None:
        if destination == output / run_id:
            destination.mkdir()
            (destination / "accepted.txt").write_text("preserve\n", encoding="utf-8")
        original_promote(source, destination)

    monkeypatch.setattr(basic_ortho, "_rename_directory_noreplace", race_then_promote)
    with pytest.raises(ProtocolError, match="refusing to replace existing accepted bundle"):
        basic_ortho.atomic_write_run_bundle(
            output,
            run_id,
            writer=writer,
            verifier=lambda _root: None,
        )
    assert (output / run_id / "accepted.txt").read_text(encoding="utf-8") == "preserve\n"
    assert not list(output.glob(".*.staging"))


def test_failed_staging_cleanup_preserves_replacement_and_reports_owned_inode(
    tmp_path: Path,
):
    output = tmp_path / "outputs"
    moved_owned = tmp_path / "moved-owned-staging"
    replacement_path: Path | None = None

    def writer(staging_root: Path) -> Path:
        nonlocal replacement_path
        (staging_root / "owned.txt").write_text("owned\n", encoding="utf-8")
        staging_root.rename(moved_owned)
        staging_root.mkdir()
        (staging_root / "replacement.txt").write_text("preserve\n", encoding="utf-8")
        replacement_path = staging_root
        raise RuntimeError("CLEANUP_DELETED_REPLACEMENT_AND_LEFT_OWNED_STAGING")

    with pytest.raises(CleanupResidueError, match="replacement was restored.*owned residue"):
        basic_ortho.atomic_write_run_bundle(
            output,
            "cleanup-path-swap",
            writer=writer,
            verifier=lambda _root: None,
        )
    assert replacement_path is not None
    assert (replacement_path / "replacement.txt").read_text(encoding="utf-8") == "preserve\n"
    assert (moved_owned / "owned.txt").read_text(encoding="utf-8") == "owned\n"
    assert not (output / "cleanup-path-swap").exists()


def test_cleanup_swap_after_identity_check_never_deletes_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "outputs"
    moved_owned = tmp_path / "moved-after-check-owned"
    replacement_path: Path | None = None
    original_rename = basic_ortho._rename_directory_noreplace

    def writer(staging_root: Path) -> Path:
        (staging_root / "owned.txt").write_text("owned\n", encoding="utf-8")
        raise RuntimeError("trigger descriptor-safe cleanup")

    def swap_when_cleanup_detaches(source: Path, destination: Path) -> None:
        nonlocal replacement_path
        if destination.name == "detached" and destination.parent.name.endswith(".quarantine"):
            source.rename(moved_owned)
            source.mkdir()
            (source / "replacement.txt").write_text("preserve\n", encoding="utf-8")
            replacement_path = source
        original_rename(source, destination)

    monkeypatch.setattr(basic_ortho, "_rename_directory_noreplace", swap_when_cleanup_detaches)
    with pytest.raises(CleanupResidueError, match="replacement was restored.*owned residue"):
        basic_ortho.atomic_write_run_bundle(
            output,
            "cleanup-swap-after-check",
            writer=writer,
            verifier=lambda _root: None,
        )
    assert replacement_path is not None
    assert (replacement_path / "replacement.txt").read_text(encoding="utf-8") == "preserve\n"
    assert (moved_owned / "owned.txt").read_text(encoding="utf-8") == "owned\n"


@pytest.mark.parametrize(
    "nested_path",
    [
        ("selector",),
        ("input",),
        ("telemetry",),
        ("output_contract",),
        ("execution_identity_inputs", "selector"),
        ("execution_identity_inputs", "input"),
    ],
)
def test_resource_pilot_rejects_unknown_nested_endpoint_sentinel(
    tmp_path: Path,
    nested_path: tuple[str, ...],
):
    inputs, _, _ = _runner_fixture(tmp_path / "inputs")
    manifest = resource_pilot_document(
        select_resource_pilot_scene(inputs),
        branch=RESOURCE_PILOT_DEFAULT_BRANCH,
        protocol_sha256="protocol",
        acquisition_manifest_sha256="acquisition",
        ortho_manifest_sha256="ortho",
        governing_files=GOVERNING_HASHES,
        telemetry=ResourcePilotTelemetry(1.0, 2.0, 3.0, 4, 5, 6),
    )
    expected_run_id = str(manifest["execution_id"])
    cursor = manifest
    for key in nested_path:
        cursor = cursor[key]
    cursor["endpoint_sentinel"] = True
    bundle = tmp_path / "pilot-bundle"
    strict_json_dump(bundle / "resource_pilot_manifest.json", manifest)
    with pytest.raises(ProtocolError, match="fields|identity|output contract"):
        basic_ortho.verify_resource_pilot_bundle(
            bundle,
            expected_run_id=expected_run_id,
            expected_protocol_sha256="protocol",
        )


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_resource_pilot_rejects_nonexact_nested_governing_keys_with_recomputed_id(
    tmp_path: Path,
    mutation: str,
):
    inputs, _, _ = _runner_fixture(tmp_path / "inputs")
    manifest = resource_pilot_document(
        select_resource_pilot_scene(inputs),
        branch=RESOURCE_PILOT_DEFAULT_BRANCH,
        protocol_sha256="protocol",
        acquisition_manifest_sha256="acquisition",
        ortho_manifest_sha256="ortho",
        governing_files=GOVERNING_HASHES,
        telemetry=ResourcePilotTelemetry(1.0, 2.0, 3.0, 4, 5, 6),
    )
    governing = manifest["execution_identity_inputs"]["governing_files"]
    if mutation == "extra":
        governing["tests/test_basic_ortho.py"] = "e" * 64
    else:
        governing.pop("src/tanager_rocks/quality.py")
    expected_run_id = basic_ortho.resource_pilot_execution_identity(
        manifest["execution_identity_inputs"]
    )
    manifest["execution_id"] = expected_run_id
    bundle = tmp_path / f"pilot-{mutation}"
    strict_json_dump(bundle / "resource_pilot_manifest.json", manifest)
    with pytest.raises(ProtocolError, match="governing files fields"):
        basic_ortho.verify_resource_pilot_bundle(
            bundle,
            expected_run_id=expected_run_id,
            expected_protocol_sha256="protocol",
        )


def test_resource_pilot_rejects_hash_bound_dependency_identity_tampering(
    tmp_path: Path,
):
    inputs, _, _ = _runner_fixture(tmp_path / "inputs")
    manifest = resource_pilot_document(
        select_resource_pilot_scene(inputs),
        branch=RESOURCE_PILOT_DEFAULT_BRANCH,
        protocol_sha256="protocol",
        acquisition_manifest_sha256="acquisition",
        ortho_manifest_sha256="ortho",
        governing_files=GOVERNING_HASHES,
        telemetry=ResourcePilotTelemetry(1.0, 2.0, 3.0, 4, 5, 6),
    )
    dependency = manifest["execution_identity_inputs"]["residual_dependency_trust"]["tanager_spec"]
    source_path = next(iter(sorted(dependency["python_source_files"])))
    dependency["python_source_files"][source_path] = "0" * 64
    inventory_payload = {
        "python_source_files": dependency["python_source_files"],
        "package_data_files": dependency["package_data_files"],
        "module_origins": dependency["module_origins"],
    }
    dependency["inventory_sha256"] = execution_identity(inventory_payload)
    manifest["residual_dependency_trust"] = copy.deepcopy(
        manifest["execution_identity_inputs"]["residual_dependency_trust"]
    )
    expected_run_id = basic_ortho.resource_pilot_execution_identity(
        manifest["execution_identity_inputs"]
    )
    manifest["execution_id"] = expected_run_id
    bundle = tmp_path / "pilot-dependency-tamper"
    strict_json_dump(bundle / "resource_pilot_manifest.json", manifest)

    with pytest.raises(ProtocolError, match="source hashes linked to governing identity"):
        basic_ortho.verify_resource_pilot_bundle(
            bundle,
            expected_run_id=expected_run_id,
            expected_protocol_sha256="protocol",
        )


def test_resource_pilot_rejects_source_replacement_after_descriptor_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    selected = select_resource_pilot_scene(inputs)
    original_path = selected.basic_path
    replacement = tmp_path / "replacement.h5"
    _write_basic(replacement, shape=selected.basic_stac_shape, bands=5)
    original_h5_open = basic_ortho.h5py.File
    hdf_open_calls = 0

    def swap_source_on_hdf_open(*args, **kwargs):
        nonlocal hdf_open_calls
        hdf_open_calls += 1
        if hdf_open_calls == 1:
            preserved = original_path.with_name("preserved-basic.h5")
            original_path.rename(preserved)
            original_path.symlink_to(replacement)
        return original_h5_open(*args, **kwargs)

    monkeypatch.setattr(basic_ortho.h5py, "File", swap_source_on_hdf_open)
    output = tmp_path / "pilot"
    with pytest.raises(runner.BootstrapError, match="without following symlinks"):
        runner.run_resource_pilot(
            preregistration=inputs.acquisition_manifest_path.with_name("protocol.md"),
            acquisition_manifest=inputs.acquisition_manifest_path,
            ortho_manifest=inputs.ortho_manifest_path,
            output_dir=output,
        )
    assert hdf_open_calls == 1
    assert list(output.iterdir()) == []
    assert not list(output.glob(".*.staging"))


def test_resource_pilot_rejects_preexisting_input_symlink_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    selected = select_resource_pilot_scene(inputs)
    preserved = selected.basic_path.with_name("preserved-basic.h5")
    selected.basic_path.rename(preserved)
    selected.basic_path.symlink_to(preserved)
    output = tmp_path / "pilot"
    with pytest.raises(runner.BootstrapError, match="without following symlinks"):
        runner.run_resource_pilot(
            preregistration=inputs.acquisition_manifest_path.with_name("protocol.md"),
            acquisition_manifest=inputs.acquisition_manifest_path,
            ortho_manifest=inputs.ortho_manifest_path,
            output_dir=output,
        )
    assert not output.exists() or list(output.iterdir()) == []
    assert not list(output.glob(".*.staging"))


@pytest.mark.parametrize("mutation", ["in-place-restore", "replacement"])
def test_independent_oracle_reads_stable_snapshot_and_source_drift_rejects(
    tmp_path: Path,
    mutation: str,
):
    inputs, _, _ = _runner_fixture(tmp_path / "inputs")
    source_path = inputs.scenes[0].basic_path
    with runner._snapshot_hdf_inputs(inputs) as snapshots:
        baseline = runner._independent_mapping_attestations(inputs, snapshots)
        if mutation == "in-place-restore":
            original_prefix = source_path.read_bytes()[:1]
            with source_path.open("r+b") as source:
                source.write(bytes([original_prefix[0] ^ 0xFF]))
                source.flush()
                os.fsync(source.fileno())
                source.seek(0)
                source.write(original_prefix)
                source.flush()
                os.fsync(source.fileno())
            assert sha256_file(source_path) == inputs.scenes[0].basic_sha256
        else:
            preserved = source_path.with_name("preserved-basic.h5")
            source_path.rename(preserved)
            _write_basic(source_path, shape=inputs.scenes[0].basic_stac_shape, bands=5)

        observed = runner._independent_mapping_attestations(inputs, snapshots)
        for site in FROZEN_SCENES:
            assert observed[site]["source_accounting"] == baseline[site]["source_accounting"]
            assert observed[site]["spectral_copy_audit"] == baseline[site]["spectral_copy_audit"]
            for artifact_name in MAPPING_ARTIFACT_NAMES:
                np.testing.assert_array_equal(
                    observed[site]["arrays"][artifact_name],
                    baseline[site]["arrays"][artifact_name],
                )
        with pytest.raises(
            runner.IndependentVerificationError,
            match="source metadata changed",
        ):
            runner._revalidate_snapshot_source(
                snapshots[(inputs.scenes[0].site, "basic")],
                label="mutated synthetic HDF5",
            )


def _rewrite_status_raster(path: Path, mutation: str) -> None:
    with rasterio.open(path) as source:
        values = source.read()
        profile = source.profile.copy()
        descriptions = list(source.descriptions)
        tags = source.tags()
    if mutation == "shape":
        values = values[:, :, :-1]
        profile["width"] = values.shape[2]
    elif mutation == "transform":
        profile["transform"] = from_origin(500030.0, 4100000.0, 30.0, 30.0)
    elif mutation == "crs":
        profile["crs"] = CRS.from_epsg(32612)
    elif mutation == "count":
        values = np.concatenate((values, values), axis=0)
        profile["count"] = 2
        descriptions.append("extra_status")
    elif mutation == "dtype":
        values = values.astype(np.uint16)
        profile["dtype"] = "uint16"
    elif mutation == "nodata":
        profile["nodata"] = 255
    elif mutation == "description":
        descriptions[0] = "endpoint_status"
    elif mutation == "tags":
        tags["endpoint_sentinel"] = "true"
    elif mutation == "status-domain":
        values[0, 0, 0] = 9
    else:
        raise AssertionError(f"unknown test mutation: {mutation}")
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(values)
        for band, description in enumerate(descriptions, 1):
            destination.set_band_description(band, description)
        destination.update_tags(**tags)


def _rewrite_raster_values(path: Path, values: np.ndarray) -> None:
    with rasterio.open(path) as source:
        profile = source.profile.copy()
        descriptions = source.descriptions
        tags = source.tags()
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(values)
        for band, description in enumerate(descriptions, 1):
            destination.set_band_description(band, description)
        destination.update_tags(**tags)


def test_mapping_verifier_rejects_semantic_tampering_with_refreshed_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    run_manifest_path = _run_mapping_fixture(tmp_path / "mapping", inputs)
    with runner._snapshot_hdf_inputs(inputs) as snapshots:
        expected_semantics = runner._independent_mapping_attestations(inputs, snapshots)
    original_run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    site = next(iter(FROZEN_SCENES))
    scene_manifest_path = run_manifest_path.parent / original_run["scene_manifests"][site]["path"]
    original_scene = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    status_path = scene_manifest_path.parent / "mapping_status.tif"
    original_status = status_path.read_bytes()

    for mutation in (
        "shape",
        "transform",
        "crs",
        "count",
        "dtype",
        "nodata",
        "description",
        "tags",
        "status-domain",
    ):
        status_path.write_bytes(original_status)
        scene_manifest = json.loads(json.dumps(original_scene))
        run_manifest = json.loads(json.dumps(original_run))
        _rewrite_status_raster(status_path, mutation)
        scene_manifest["artifacts"]["mapping_status.tif"] = {
            "sha256": sha256_file(status_path),
            "size_bytes": status_path.stat().st_size,
        }
        strict_json_dump(scene_manifest_path, scene_manifest)
        run_manifest["scene_manifests"][site]["sha256"] = sha256_file(scene_manifest_path)
        strict_json_dump(run_manifest_path, run_manifest)
        with pytest.raises(ProtocolError):
            basic_ortho.verify_mapping_run_bundle(
                run_manifest_path.parent,
                expected_run_id=str(original_run["execution_id"]),
                expected_scenes=FROZEN_SCENES,
                expected_semantics=expected_semantics,
                expected_protocol_sha256=original_run["execution_identity_inputs"][
                    "protocol_sha256"
                ],
                expected_snapshot_evidence=original_run["input_snapshot_evidence"],
                expected_residual_dependency_trust=original_run["residual_dependency_trust"],
            )


def test_mapping_verifier_rejects_coordinated_remap_with_refreshed_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, schemas, grids = _runner_fixture(tmp_path / "inputs")
    _configure_runner(monkeypatch, inputs, schemas, grids)
    run_manifest_path = _run_mapping_fixture(tmp_path / "mapping", inputs)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    with runner._snapshot_hdf_inputs(inputs) as snapshots:
        expected_semantics = runner._independent_mapping_attestations(inputs, snapshots)
    site = next(iter(FROZEN_SCENES))
    scene_manifest_path = run_manifest_path.parent / run_manifest["scene_manifests"][site]["path"]
    scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {name: scene_manifest_path.parent / name for name in MAPPING_ARTIFACT_NAMES}
    with rasterio.open(artifact_paths["source_index.tif"]) as dataset:
        source_index = dataset.read()
    with rasterio.open(artifact_paths["mapping_distance_m.tif"]) as dataset:
        distance = dataset.read()
    with rasterio.open(artifact_paths["source_multiplicity.tif"]) as dataset:
        multiplicity = dataset.read()
    with rasterio.open(artifact_paths["mapping_status.tif"]) as dataset:
        status = dataset.read()

    selected = np.isin(status[0], [int(TARGET_BASIC_QA_INVALID), int(TARGET_MAPPED)])
    basic_rows, basic_cols = scene_manifest["frozen_input"]["basic_stac_shape"]
    flat_source = source_index[0, selected] * basic_cols + source_index[1, selected]
    reassigned = (flat_source + 1) % (basic_rows * basic_cols)
    source_index[0, selected] = reassigned // basic_cols
    source_index[1, selected] = reassigned % basic_cols
    distance[0, selected] += 1.0
    unique_source, target_counts = np.unique(reassigned, return_counts=True)
    count_by_source = dict(zip(unique_source.tolist(), target_counts.tolist(), strict=True))
    multiplicity[0, selected] = np.asarray(
        [count_by_source[int(value)] for value in reassigned],
        dtype=multiplicity.dtype,
    )

    arrays = {
        "source_index.tif": source_index,
        "mapping_distance_m.tif": distance,
        "source_multiplicity.tif": multiplicity,
        "mapping_status.tif": status,
    }
    for name, values in arrays.items():
        _rewrite_raster_values(artifact_paths[name], values)
        scene_manifest["artifacts"][name] = {
            "sha256": sha256_file(artifact_paths[name]),
            "size_bytes": artifact_paths[name].stat().st_size,
        }

    accounting = scene_manifest["source_accounting"]
    total_sources = basic_rows * basic_cols
    accounting.update(
        {
            "used_source_samples": int(unique_source.size),
            "unused_source_samples": int(total_sources - unique_source.size),
            "sources_with_multiple_target_cells": int((target_counts > 1).sum()),
            "duplicate_target_assignments": int((target_counts - 1).sum()),
        }
    )
    strict_json_dump(scene_manifest_path, scene_manifest)
    run_manifest["scene_manifests"][site]["sha256"] = sha256_file(scene_manifest_path)
    strict_json_dump(run_manifest_path, run_manifest)

    with pytest.raises(ProtocolError, match="externally attested mapping semantics"):
        basic_ortho.verify_mapping_run_bundle(
            run_manifest_path.parent,
            expected_run_id=str(run_manifest["execution_id"]),
            expected_scenes=FROZEN_SCENES,
            expected_semantics=expected_semantics,
            expected_protocol_sha256=run_manifest["execution_identity_inputs"]["protocol_sha256"],
            expected_snapshot_evidence=run_manifest["input_snapshot_evidence"],
            expected_residual_dependency_trust=run_manifest["residual_dependency_trust"],
        )
