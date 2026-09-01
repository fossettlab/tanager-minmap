"""Synthetic tests for the EMIT L2B independent-product packet."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import runpy
import sys
import types
import zipfile
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio
from affine import Affine

from tanager_rocks.emit_l2b import (
    BootstrapDraw,
    M2BlockScale,
    OntologyEntry,
    ProductIdentity,
    ProductMismatchError,
    RasterGeometry,
    SourceMineral,
    SpatialNullDraw,
    block_footprint_support,
    compute_endpoint_metrics,
    l2b_identity_evidence,
    load_emit_l2b_pair,
    load_m2_block_scales,
    load_pinned_emit_l2a_input,
    parse_l2a_product_identity,
    parse_product_identity,
    permute_complete_block_packet,
    reproject_categorical_nearest,
    sha256_file,
    summarize_bootstrap_interval,
    summarize_spatial_null,
    validate_emit_l2b_source_pair,
    validate_exchangeable_block_packets,
    validate_l2b_identity_against_l2a,
    validate_ontology_crosswalk,
    whole_block_spatial_nulls,
    write_strict_json,
)

ACQUISITION = "20230804T191650"
ORBIT = "2321613"
SCENE = "007"
VERSION = "001"


def _attrs(
    kind: str,
    *,
    version: str = VERSION,
    acquisition: str = ACQUISITION,
    orbit: str = ORBIT,
) -> dict[str, str]:
    del kind
    timestamp = acquisition.replace("T", "t")
    iso_timestamp = (
        f"{acquisition[:4]}-{acquisition[4:6]}-{acquisition[6:8]}T"
        f"{acquisition[9:11]}:{acquisition[11:13]}:{acquisition[13:15]}Z"
    )
    return {
        "time_coverage_start": iso_timestamp,
        "time_coverage_end": iso_timestamp,
        "flight_line": f"emit{timestamp}_o{orbit[2:]}_s000",
        "product_version": f"V{version}",
        "geotransform": np.asarray([100.0, 1.0, 0.0, 200.0, 0.0, -1.0]),
        "spatial_ref": "EPSG:4326",
    }


def _dataset(group: h5py.Group, name: str, values: np.ndarray, fill: float | int) -> None:
    dataset = group.create_dataset(name, data=values)
    dataset.attrs["_FillValue"] = fill


def _write_product(
    directory: Path,
    kind: str,
    *,
    version: str = VERSION,
    acquisition: str = ACQUISITION,
    orbit: str = ORBIT,
    scene: str = SCENE,
    attrs: dict[str, object] | None = None,
    reverse_uncertainty_glt: bool = False,
    glt_x: np.ndarray | None = None,
    glt_y: np.ndarray | None = None,
    raw_ids: dict[int, np.ndarray] | None = None,
    raw_depths: dict[int, np.ndarray] | None = None,
    mineral_records: tuple[SourceMineral, ...] | None = None,
    uncertainty_raw: np.ndarray | None = None,
) -> Path:
    path = directory / f"EMIT_L2B_{kind}_{version}_{acquisition}_{orbit}_{scene}.nc"
    with h5py.File(path, "w") as handle:
        for key, value in (
            attrs or _attrs(kind, version=version, acquisition=acquisition, orbit=orbit)
        ).items():
            handle.attrs[key] = value
        location = handle.create_group("location")
        if glt_x is None:
            glt_x = np.asarray([[3, 2, 1], [1, 2, 3]], dtype=np.int32)
        if glt_y is None:
            glt_y = np.asarray([[2, 2, 2], [1, 1, 1]], dtype=np.int32)
        if reverse_uncertainty_glt:
            glt_x = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.int32)
        location.create_dataset("glt_x", data=glt_x)
        location.create_dataset("glt_y", data=glt_y)

        if kind == "MIN":
            default_id = np.asarray([[11, 12, 13], [21, 22, 23]], dtype=np.int16)
            ids = raw_ids or {
                1: default_id,
                2: default_id + 100,
            }
            depths = raw_depths or {
                group_number: values.astype(np.float32) / 100.0
                for group_number, values in ids.items()
            }
            for group_number in (1, 2):
                _dataset(
                    handle,
                    f"group_{group_number}_mineral_id",
                    ids[group_number],
                    -9999,
                )
                _dataset(
                    handle,
                    f"group_{group_number}_band_depth",
                    depths[group_number],
                    -9999.0,
                )
            metadata = handle.create_group("mineral_metadata")
            if mineral_records is None:
                group_1_ids = [11, 12, 13, 21, 22, 23]
                group_2_ids = [value + 100 for value in group_1_ids]
                mineral_records = tuple(
                    SourceMineral(value, f"mineral_{value}", group, "splib07")
                    for group, values in ((1, group_1_ids), (2, group_2_ids))
                    for value in values
                )
            metadata.create_dataset(
                "index", data=np.asarray([record.index for record in mineral_records])
            )
            metadata.create_dataset(
                "name", data=np.asarray([record.name.encode() for record in mineral_records])
            )
            metadata.create_dataset(
                "group", data=np.asarray([record.group for record in mineral_records])
            )
            metadata.create_dataset(
                "library", data=np.asarray([record.library.encode() for record in mineral_records])
            )
        else:
            raw = (
                np.asarray(uncertainty_raw, dtype=np.float32)
                if uncertainty_raw is not None
                else np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            )
            for group_number in (1, 2):
                _dataset(
                    handle,
                    f"group_{group_number}_band_depth_unc",
                    raw / 10.0 + group_number,
                    -9999.0,
                )
                _dataset(
                    handle,
                    f"group_{group_number}_fit",
                    raw / 100.0 + group_number,
                    -9999.0,
                )
    return path


def _pair(tmp_path: Path, **uncertainty_options: object):
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", **uncertainty_options)
    return load_emit_l2b_pair(min_path, uncertainty_path)


def test_each_products_glt_controls_orientation(tmp_path: Path):
    pair = _pair(tmp_path, reverse_uncertainty_glt=True)

    assert pair.geometry.transform == Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
    np.testing.assert_array_equal(
        pair.groups[1].mineral_id,
        np.asarray([[23.0, 22.0, 21.0], [11.0, 12.0, 13.0]]),
    )
    np.testing.assert_allclose(
        pair.groups[1].uncertainty,
        np.asarray([[1.4, 1.5, 1.6], [1.3, 1.2, 1.1]]),
    )
    assert not np.array_equal(pair.min_glt_x, pair.minuncert_glt_x)


def test_delivered_v001_metadata_without_global_orbit_or_scene_is_accepted(tmp_path: Path):
    pair = _pair(tmp_path)

    assert "orbit" not in pair.min_metadata
    assert "scene" not in pair.min_metadata
    assert pair.identity.scene == "007"
    assert pair.min_metadata["flight_line"] == "emit20230804t191650_o21613_s000"


def test_flight_line_s_token_is_not_delivered_scene(tmp_path: Path):
    pair = _pair(tmp_path)
    evidence = l2b_identity_evidence(pair)

    assert pair.identity.scene == "007"
    assert evidence["flight_line_s_token"] == "000"
    assert evidence["delivered_scene_source"] == "filename_and_cmr"
    assert evidence["flight_line_s_token_is_delivered_scene"] is False


@pytest.mark.parametrize("orbit", ["232161", "23216130"])
def test_filename_orbit_requires_seven_digits(orbit: str):
    l2b_name = f"EMIT_L2B_MIN_{VERSION}_{ACQUISITION}_{orbit}_{SCENE}.nc"
    l2a_name = f"EMIT_L2A_RFL_{VERSION}_{ACQUISITION}_{orbit}_{SCENE}.nc"

    with pytest.raises(ProductMismatchError, match="invalid EMIT L2B"):
        parse_product_identity(l2b_name)
    with pytest.raises(ProductMismatchError, match="invalid EMIT L2A"):
        parse_l2a_product_identity(l2a_name)


@pytest.mark.parametrize(
    ("parser", "prefix"),
    [
        (parse_product_identity, "EMIT_L2B_MIN"),
        (parse_l2a_product_identity, "EMIT_L2A_RFL"),
    ],
)
def test_filename_orbit_year_must_match_acquisition(
    parser: Callable[[str], ProductIdentity], prefix: str
):
    name = f"{prefix}_{VERSION}_{ACQUISITION}_2421613_{SCENE}.nc"
    with pytest.raises(ProductMismatchError, match="orbit year"):
        parser(name)


@pytest.mark.parametrize(
    ("parser", "prefix"),
    [
        (parse_product_identity, "EMIT_L2B_MIN"),
        (parse_l2a_product_identity, "EMIT_L2A_RFL"),
    ],
)
def test_filename_orbit_doy_must_match_acquisition(
    parser: Callable[[str], ProductIdentity], prefix: str
):
    name = f"{prefix}_{VERSION}_{ACQUISITION}_2321513_{SCENE}.nc"
    with pytest.raises(ProductMismatchError, match="day of year"):
        parser(name)


@pytest.mark.parametrize(
    "missing",
    ["product_version", "time_coverage_start", "time_coverage_end", "flight_line"],
)
def test_missing_required_product_metadata_is_refused(tmp_path: Path, missing: str):
    attrs = _attrs("MINUNCERT")
    del attrs[missing]
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", attrs=attrs)

    with pytest.raises(ProductMismatchError, match=missing):
        load_emit_l2b_pair(min_path, uncertainty_path)


def test_version_or_acquisition_mismatch_is_refused(tmp_path: Path):
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", version="002")
    with pytest.raises(ProductMismatchError, match="version"):
        load_emit_l2b_pair(min_path, uncertainty_path)

    uncertainty_path = _write_product(
        tmp_path,
        "MINUNCERT",
        acquisition="20230805T191650",
        orbit="2321713",
    )
    with pytest.raises(ProductMismatchError, match="acquisition"):
        load_emit_l2b_pair(min_path, uncertainty_path)


def test_missing_pair_member_is_refused(tmp_path: Path):
    min_path = _write_product(tmp_path, "MIN")
    missing = tmp_path / f"EMIT_L2B_MINUNCERT_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"

    with pytest.raises(FileNotFoundError, match="pair member missing"):
        load_emit_l2b_pair(min_path, missing)


def test_filename_orbit_mismatch_is_refused(tmp_path: Path):
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", orbit="2321614")

    with pytest.raises(ProductMismatchError, match="orbit"):
        load_emit_l2b_pair(min_path, uncertainty_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("product_version", "V002", "product_version metadata"),
        ("time_coverage_start", "2023-08-04T19:16:51Z", "time_coverage_start metadata"),
    ],
)
def test_filename_product_and_acquisition_metadata_are_binding(
    tmp_path: Path, field: str, value: str, match: str
):
    attrs = _attrs("MINUNCERT")
    attrs[field] = value
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", attrs=attrs)

    with pytest.raises(ProductMismatchError, match=match):
        load_emit_l2b_pair(min_path, uncertainty_path)


def test_min_minuncert_time_coverage_must_match(tmp_path: Path):
    attrs = _attrs("MINUNCERT")
    attrs["time_coverage_end"] = "2023-08-04T19:16:51Z"
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", attrs=attrs)

    with pytest.raises(ProductMismatchError, match="time_coverage_end"):
        load_emit_l2b_pair(min_path, uncertainty_path)


@pytest.mark.parametrize(
    "flight_line",
    [
        None,
        "EMIT20230804t191650_o21613_s000",
        "emit20230804T191650_o21613_s000",
        "emit20230804t191650_o21x13_s000",
        "emit20230804t1916_o21613_s000",
    ],
)
def test_missing_or_malformed_flight_line_is_refused(tmp_path: Path, flight_line: str | None):
    attrs = _attrs("MINUNCERT")
    if flight_line is None:
        del attrs["flight_line"]
    else:
        attrs["flight_line"] = flight_line
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", attrs=attrs)

    with pytest.raises(ProductMismatchError, match="flight_line"):
        load_emit_l2b_pair(min_path, uncertainty_path)


@pytest.mark.parametrize(
    ("flight_line", "match"),
    [
        ("emit20230804t191651_o21613_s000", "flight_line acquisition"),
        ("emit20230804t191650_o21614_s000", "flight_line short orbit"),
    ],
)
def test_flight_line_acquisition_or_short_orbit_mismatch_is_refused(
    tmp_path: Path, flight_line: str, match: str
):
    attrs = _attrs("MINUNCERT")
    attrs["flight_line"] = flight_line
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", attrs=attrs)

    with pytest.raises(ProductMismatchError, match=match):
        load_emit_l2b_pair(min_path, uncertainty_path)


@pytest.mark.parametrize(
    ("flight_line", "match"),
    [
        ("emit20230804t191651_o21613_s000", "flight_line acquisition"),
        ("emit20230804t191650_o21614_s000", "flight_line short orbit"),
        ("emit20230804t191650_o21613_s001", "MIN/MINUNCERT flight_line mismatch"),
    ],
)
def test_min_minuncert_flight_lines_must_match(tmp_path: Path, flight_line: str, match: str):
    attrs = _attrs("MINUNCERT")
    attrs["flight_line"] = flight_line
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", attrs=attrs)

    with pytest.raises(ProductMismatchError, match=match):
        load_emit_l2b_pair(min_path, uncertainty_path)


def test_filename_scene_mismatch_is_refused_when_flight_lines_match(tmp_path: Path):
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", scene="008")

    with pytest.raises(ProductMismatchError, match="scene"):
        load_emit_l2b_pair(min_path, uncertainty_path)


def test_l2b_scene_must_match_pinned_l2a_when_flight_lines_match(tmp_path: Path):
    pair = _pair(tmp_path)
    pinned_l2a = ProductIdentity("L2A_RFL", VERSION, ACQUISITION, ORBIT, "008")

    with pytest.raises(ProductMismatchError, match="scene"):
        validate_l2b_identity_against_l2a(pair.identity, pinned_l2a)


def test_crs_must_be_explicitly_verified_from_both_products(tmp_path: Path):
    attrs = _attrs("MIN")
    del attrs["spatial_ref"]
    min_path = _write_product(tmp_path, "MIN", attrs=attrs)
    uncertainty_path = _write_product(tmp_path, "MINUNCERT")
    with pytest.raises(ProductMismatchError, match="CRS cannot be verified"):
        load_emit_l2b_pair(min_path, uncertainty_path)

    min_path.unlink()
    attrs = _attrs("MIN")
    attrs["spatial_ref"] = "EPSG:3857"
    min_path = _write_product(tmp_path, "MIN", attrs=attrs)
    with pytest.raises(ProductMismatchError, match="must be EPSG:4326"):
        load_emit_l2b_pair(min_path, uncertainty_path)


def test_min_minuncert_geotransforms_must_match(tmp_path: Path):
    attrs = _attrs("MINUNCERT")
    attrs["geotransform"] = np.asarray([101.0, 1.0, 0.0, 200.0, 0.0, -1.0])
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", attrs=attrs)

    with pytest.raises(ProductMismatchError, match="GLT output geometry mismatch"):
        load_emit_l2b_pair(min_path, uncertainty_path)


@pytest.mark.parametrize(
    ("glt_x", "glt_y", "match"),
    [
        (
            np.zeros((2, 3), dtype=np.int32),
            np.zeros((2, 3), dtype=np.int32),
            "no valid mapped cells",
        ),
        (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            "cannot be empty",
        ),
        (
            np.asarray([[0, 2, 3]], dtype=np.int32),
            np.asarray([[1, 2, 3]], dtype=np.int32),
            "fill locations must match",
        ),
    ],
)
def test_empty_all_zero_or_misaligned_glt_is_refused(
    tmp_path: Path, glt_x: np.ndarray, glt_y: np.ndarray, match: str
):
    min_path = _write_product(tmp_path, "MIN", glt_x=glt_x, glt_y=glt_y)
    uncertainty_path = _write_product(tmp_path, "MINUNCERT", glt_x=glt_x, glt_y=glt_y)
    with pytest.raises(ProductMismatchError, match=match):
        load_emit_l2b_pair(min_path, uncertainty_path)


def test_negative_unknown_ids_and_duplicate_group_index_metadata_are_refused(tmp_path: Path):
    min_path = _write_product(tmp_path, "MIN")
    uncertainty_path = _write_product(tmp_path, "MINUNCERT")
    with h5py.File(min_path, "r+") as handle:
        handle["group_1_mineral_id"][0, 0] = -1
    with pytest.raises(ProductMismatchError, match="negative mineral IDs"):
        load_emit_l2b_pair(min_path, uncertainty_path)

    min_path.unlink()
    min_path = _write_product(tmp_path, "MIN")
    with h5py.File(min_path, "r+") as handle:
        handle["group_1_mineral_id"][0, 0] = 999
    with pytest.raises(ProductMismatchError, match="absent from mineral metadata"):
        load_emit_l2b_pair(min_path, uncertainty_path)

    min_path.unlink()
    min_path = _write_product(tmp_path, "MIN")
    with h5py.File(min_path, "r+") as handle:
        handle["mineral_metadata/index"][1] = handle["mineral_metadata/index"][0]
        handle["mineral_metadata/group"][1] = handle["mineral_metadata/group"][0]
    with pytest.raises(ProductMismatchError, match=r"duplicate \(group, index\)"):
        load_emit_l2b_pair(min_path, uncertainty_path)


def test_categorical_reprojection_is_nearest_and_never_interpolates():
    source = np.asarray([[1, 2], [3, 4]], dtype=np.int16)
    projected = reproject_categorical_nearest(
        source,
        source_transform=Affine(1, 0, 0, 0, -1, 2),
        source_crs="EPSG:4326",
        destination_shape=(4, 4),
        destination_transform=Affine(0.5, 0, 0, 0, -0.5, 2),
        destination_crs="EPSG:4326",
        nodata=0,
    )

    assert set(np.unique(projected)) == {1, 2, 3, 4}
    np.testing.assert_array_equal(projected[:2, :2], np.ones((2, 2), dtype=np.int16))


def test_catalog_pair_resolution_ignores_alternate_s3_access_links():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "fetch_emit_l2b.py")
    )
    resolve_pair_links = namespace["resolve_pair_links"]
    min_name = f"EMIT_L2B_MIN_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"
    uncertainty_name = f"EMIT_L2B_MINUNCERT_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"

    class Result(dict):
        def __init__(self):
            super().__init__(
                meta={
                    "concept-id": "G123456-TEST",
                    "revision-id": 7,
                    "collection-concept-id": "C123456-TEST",
                },
                umm={"GranuleUR": Path(min_name).stem},
            )

        def data_links(self):
            return (
                f"s3://lp-prod-protected/example/{min_name}",
                f"https://example.test/{min_name}?token=secret",
                f"s3://lp-prod-protected/example/{uncertainty_name}",
                f"https://example.test/{uncertainty_name}?token=secret",
            )

    l2a_identity = ProductIdentity("L2A_RFL", VERSION, ACQUISITION, ORBIT, SCENE)
    pair = resolve_pair_links([Result()], l2a_identity)

    assert pair.min_url.startswith("https://")
    assert pair.minuncert_url.startswith("https://")
    assert pair.granule_concept_id == "G123456-TEST"
    assert pair.granule_revision_id == 7
    assert pair.collection_concept_id == "C123456-TEST"
    assert pair.granule_ur == Path(min_name).stem


def test_cmr_granule_ur_must_equal_min_stem():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "fetch_emit_l2b.py")
    )
    resolve_pair_links = namespace["resolve_pair_links"]
    min_name = f"EMIT_L2B_MIN_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"
    uncertainty_name = f"EMIT_L2B_MINUNCERT_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"

    class Result(dict):
        def __init__(self):
            super().__init__(
                meta={
                    "concept-id": "G123456-TEST",
                    "revision-id": 7,
                    "collection-concept-id": "C123456-TEST",
                },
                umm={"GranuleUR": "different-granule"},
            )

        def data_links(self):
            return (
                f"https://example.test/{min_name}",
                f"https://example.test/{uncertainty_name}",
            )

    l2a_identity = ProductIdentity("L2A_RFL", VERSION, ACQUISITION, ORBIT, SCENE)
    with pytest.raises(RuntimeError, match="GranuleUR"):
        resolve_pair_links([Result()], l2a_identity)


def test_catalog_pair_cannot_be_assembled_across_cmr_results():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "fetch_emit_l2b.py")
    )
    resolve_pair_links = namespace["resolve_pair_links"]
    min_name = f"EMIT_L2B_MIN_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"
    uncertainty_name = f"EMIT_L2B_MINUNCERT_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"

    class Result(dict):
        def __init__(self, link: str, concept: str):
            super().__init__(
                meta={"concept-id": concept, "revision-id": 1},
                umm={"GranuleUR": concept},
            )
            self.link = link

        def data_links(self):
            return (self.link,)

    results = [
        Result(f"https://example.test/{min_name}", "G1-TEST"),
        Result(f"https://example.test/{uncertainty_name}", "G2-TEST"),
    ]
    l2a_identity = ProductIdentity("L2A_RFL", VERSION, ACQUISITION, ORBIT, SCENE)
    with pytest.raises(RuntimeError, match="incomplete"):
        resolve_pair_links(results, l2a_identity)


def _pinned_l2a_fixture(tmp_path: Path):
    repository = tmp_path / "synthetic_repo"
    docs = repository / "docs"
    emit_l2a_dir = repository / "data" / "raw" / "emit"
    docs.mkdir(parents=True)
    emit_l2a_dir.mkdir(parents=True)
    l2a = emit_l2a_dir / f"EMIT_L2A_RFL_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"
    l2a.write_bytes(b"synthetic-pinned-l2a")
    manifest = docs / "input_manifest.json"
    write_strict_json(
        manifest,
        {
            "schema_version": "1.0",
            "hash_algorithm": "sha256",
            "inputs": [
                {
                    "id": "emit-goldfield-rfl",
                    "logical_path": str(l2a.relative_to(repository)),
                    "size_bytes": l2a.stat().st_size,
                    "sha256": sha256_file(l2a),
                }
            ],
        },
    )
    return manifest, load_pinned_emit_l2a_input(manifest)


def _valid_v4_fetch_manifest(tmp_path: Path):
    repository = Path(__file__).resolve().parents[1]
    fetcher = runpy.run_path(str(repository / "scripts" / "fetch_emit_l2b.py"))
    runner = runpy.run_path(str(repository / "scripts" / "run_emit_l2b_validation.py"))
    manifest, pinned_l2a = _pinned_l2a_fixture(tmp_path)
    product_dir = tmp_path / "products"
    product_dir.mkdir()
    min_path = _write_product(product_dir, "MIN")
    minuncert_path = _write_product(product_dir, "MINUNCERT")
    pair = validate_emit_l2b_source_pair(min_path, minuncert_path)
    links = fetcher["PairLinks"](
        min_url=f"https://example.test/{min_path.name}",
        minuncert_url=f"https://example.test/{minuncert_path.name}",
        min_name=min_path.name,
        minuncert_name=minuncert_path.name,
        granule_concept_id="G123456-TEST",
        granule_revision_id=7,
        collection_concept_id="C123456-TEST",
        granule_ur=min_path.stem,
    )
    prefix = f"{ACQUISITION}_{ORBIT}_{SCENE}"
    payload = fetcher["_fetch_manifest_payload"](
        pair,
        pinned_l2a,
        input_manifest=manifest,
        granule_prefix=prefix,
        links=links,
        catalog_resolved_at_utc="2026-08-09T00:00:00+00:00",
        retrieval_mode="verified_existing_pair",
        downloaded_at_utc=None,
    )
    manifest_path = product_dir / "download_manifest.json"
    write_strict_json(manifest_path, payload)
    return (
        pair,
        pinned_l2a,
        manifest,
        manifest_path,
        payload,
        runner["_validate_fetch_manifest"],
    )


def test_fetch_manifest_v4_accepts_exact_recomputed_evidence(tmp_path: Path):
    pair, pinned, input_manifest, path, payload, validate = _valid_v4_fetch_manifest(tmp_path)

    observed_path, observed = validate(
        pair,
        pinned,
        input_manifest_sha256=sha256_file(input_manifest),
    )

    assert observed_path == path
    assert observed == json.loads(path.read_text(encoding="utf-8"))
    assert observed["identity"]["scene"] == payload["identity"].scene


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("filename_orbit_year", "24"),
        ("filename_orbit_doy", "215"),
        ("filename_orbit_daily_sequence", "12"),
        ("min_flight_line", "emit20230804t191650_o21613_s001"),
        ("minuncert_flight_line", "emit20230804t191650_o21613_s001"),
        ("flight_line_acquisition", "20230804T191651"),
        ("flight_line_short_orbit", "21614"),
        ("flight_line_s_token", "001"),
        ("delivered_scene_source", "flight_line"),
        ("flight_line_s_token_is_delivered_scene", True),
    ],
)
def test_fetch_manifest_v4_rejects_tampered_identity_evidence(
    tmp_path: Path, field: str, tampered: object
):
    pair, pinned, input_manifest, path, payload, validate = _valid_v4_fetch_manifest(tmp_path)
    changed = copy.deepcopy(payload)
    changed["identity_evidence"][field] = tampered
    write_strict_json(path, changed)

    with pytest.raises(ValueError, match="identity_evidence"):
        validate(pair, pinned, input_manifest_sha256=sha256_file(input_manifest))


@pytest.mark.parametrize(
    ("role", "field"),
    [
        ("MIN", "size_bytes"),
        ("MIN", "sha256"),
        ("MINUNCERT", "size_bytes"),
        ("MINUNCERT", "sha256"),
    ],
)
def test_fetch_manifest_v4_rejects_input_size_or_hash_mismatch(
    tmp_path: Path, role: str, field: str
):
    pair, pinned, input_manifest, path, payload, validate = _valid_v4_fetch_manifest(tmp_path)
    changed = copy.deepcopy(payload)
    record = next(item for item in changed["inputs"] if item["role"] == role)
    record[field] = record[field] + 1 if field == "size_bytes" else "0" * 64
    write_strict_json(path, changed)

    with pytest.raises(ValueError, match="filename, size, hash, or metadata"):
        validate(pair, pinned, input_manifest_sha256=sha256_file(input_manifest))


def test_verified_existing_pair_does_not_claim_download_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository = Path(__file__).resolve().parents[1]
    fetcher = runpy.run_path(str(repository / "scripts" / "fetch_emit_l2b.py"))
    fetch_pair = fetcher["fetch_pair"]
    input_manifest, _ = _pinned_l2a_fixture(tmp_path)
    output_dir = tmp_path / "existing_pair"
    output_dir.mkdir()
    min_path = _write_product(output_dir, "MIN")
    minuncert_path = _write_product(output_dir, "MINUNCERT")

    class Result(dict):
        def __init__(self):
            super().__init__(
                meta={
                    "concept-id": "G123456-TEST",
                    "revision-id": 7,
                    "collection-concept-id": "C123456-TEST",
                },
                umm={"GranuleUR": min_path.stem},
            )

        def data_links(self):
            return (
                f"https://example.test/{min_path.name}",
                f"https://example.test/{minuncert_path.name}",
            )

    def refuse_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("verified existing pair must not be downloaded")

    fake_earthaccess = types.SimpleNamespace(
        login=lambda **_kwargs: None,
        search_data=lambda **_kwargs: [Result()],
        download=refuse_download,
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)
    fetch_pair.__globals__["DEFAULT_INPUT_MANIFEST"] = input_manifest
    prefix = f"{ACQUISITION}_{ORBIT}_{SCENE}"
    fetch_pair(prefix, output_dir, input_manifest=input_manifest)

    payload = json.loads((output_dir / "download_manifest.json").read_text(encoding="utf-8"))
    assert payload["catalog_resolved_at_utc"]
    assert payload["retrieval_mode"] == "verified_existing_pair"
    assert payload["downloaded_at_utc"] is None

    pair = load_emit_l2b_pair(min_path, minuncert_path)
    pinned = load_pinned_emit_l2a_input(input_manifest)
    runner = runpy.run_path(str(repository / "scripts" / "run_emit_l2b_validation.py"))
    runner["_validate_fetch_manifest"](
        pair,
        pinned,
        input_manifest_sha256=sha256_file(input_manifest),
    )

    payload["downloaded_at_utc"] = "2026-08-09T00:00:01+00:00"
    write_strict_json(output_dir / "download_manifest.json", payload)
    with pytest.raises(ValueError, match="must not claim downloaded_at_utc"):
        runner["_validate_fetch_manifest"](
            pair,
            pinned,
            input_manifest_sha256=sha256_file(input_manifest),
        )


def test_metric_masks_use_one_joint_complete_case_support():
    metrics = compute_endpoint_metrics(
        score=np.asarray([0.1, 0.5, 0.8, 0.4, 100.0]),
        mineral_id=np.asarray([11.0, 12.0, 11.0, 12.0, 11.0]),
        band_depth=np.asarray([0.1, np.nan, 0.3, 0.4, 100.0]),
        target_ids=frozenset({11}),
        block_ids=np.asarray(["a", "a", "b", "b", 0], dtype=object),
    )

    assert metrics.auc_n == 3
    assert metrics.joint_support_n == 3
    assert metrics.spearman_n == 2
    assert metrics.auc == pytest.approx(0.5)
    assert metrics.spearman == pytest.approx(1.0)


def test_complete_block_permutation_moves_one_l2b_packet_and_missingness():
    fields = {
        "mineral_id": np.asarray([11.0, np.nan, 21.0, 22.0]),
        "band_depth": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "uncertainty": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "fit": np.asarray([5.0, 6.0, 7.0, 8.0]),
        "x": np.asarray([100.0, 101.0, 200.0, 201.0]),
        "y": np.asarray([50.0, 50.0, 75.0, 75.0]),
    }
    permuted = permute_complete_block_packet(
        fields,
        np.asarray(["a", "a", "b", "b"], dtype=object),
        permutation=np.asarray([1, 0]),
    )

    np.testing.assert_array_equal(permuted["band_depth"], [0.3, 0.4, 0.1, 0.2])
    np.testing.assert_array_equal(permuted["fit"], [7.0, 8.0, 5.0, 6.0])
    np.testing.assert_array_equal(permuted["x"], [200.0, 201.0, 100.0, 101.0])
    np.testing.assert_array_equal(permuted["y"], [75.0, 75.0, 50.0, 50.0])
    assert np.isnan(permuted["mineral_id"][3])


def test_full_l2b_footprint_excludes_block_boundaries_and_incomplete_support(tmp_path: Path):
    base = {
        "scale": "L",
        "complete_block_ids": (1, 2),
        "transform": Affine(1, 0, 0, 0, -1, 1),
        "crs": "EPSG:4326",
        "source_path": tmp_path / "unused.tif",
        "source_sha256": "0" * 64,
        "block_side_pixels": 1,
        "halo_pixels": 0,
    }
    geometry = RasterGeometry(
        shape=(1, 2),
        transform=Affine(2, 0, 0, 0, -1, 1),
        crs="EPSG:4326",
    )
    invalid = block_footprint_support(
        M2BlockScale(values=np.asarray([[1, 2, 2, 0]], dtype=np.int32), **base),
        geometry,
    )
    np.testing.assert_array_equal(invalid.block_ids, [[0, 0]])
    np.testing.assert_array_equal(invalid.crosses_block_boundary, [[True, False]])
    np.testing.assert_array_equal(invalid.incomplete_or_halo_support, [[False, True]])

    valid = block_footprint_support(
        M2BlockScale(values=np.asarray([[1, 1, 2, 2]], dtype=np.int32), **base),
        geometry,
    )
    np.testing.assert_array_equal(valid.block_ids, [[1, 2]])
    assert not np.any(valid.crosses_block_boundary)
    assert not np.any(valid.incomplete_or_halo_support)


def test_exchangeability_is_prevalidated_before_null_permutation():
    fields = {
        "mineral_id": np.asarray([11.0, 12.0, 11.0]),
        "band_depth": np.asarray([0.1, 0.2, 0.3]),
        "x": np.asarray([0.0, 1.0, 2.0]),
        "y": np.asarray([0.0, 0.0, 0.0]),
    }
    with pytest.raises(ValueError, match="equal, nonzero observation count"):
        validate_exchangeable_block_packets(
            fields,
            np.asarray(["a", "a", "b"], dtype=object),
        )


@pytest.mark.parametrize(
    ("second_x", "second_y"),
    [
        ([10.0, 11.0, 12.0], [0.0, 0.0, 0.0]),
        ([10.0, 10.0, 11.0], [0.0, 1.0, 0.0]),
        ([10.0, 10.0, 11.0], [0.0, 0.0, 0.0]),
    ],
)
def test_equal_size_noncongruent_misaligned_or_duplicate_packets_fail(
    second_x: list[float], second_y: list[float]
):
    fields = {
        "mineral_id": np.arange(6, dtype=float),
        "band_depth": np.linspace(0.1, 0.6, 6),
        "x": np.asarray([0.0, 1.0, 0.0, *second_x]),
        "y": np.asarray([0.0, 0.0, 1.0, *second_y]),
    }
    with pytest.raises(ValueError, match="relative footprint|duplicate coordinates"):
        validate_exchangeable_block_packets(
            fields,
            np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object),
        )


def test_whole_block_null_rejects_unvalidated_packet_geometry():
    packet = {
        "mineral_id": np.asarray([11.0, 12.0, 11.0, 12.0]),
        "band_depth": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "uncertainty": np.asarray([1.0, np.nan, 3.0, 4.0]),
        "fit": np.asarray([5.0, 6.0, np.nan, 8.0]),
        "l2b_valid": np.ones(4, dtype=bool),
        "x": np.asarray([0.0, 1.0, 10.0, 12.0]),
        "y": np.asarray([0.0, 0.0, 0.0, 0.0]),
    }
    with pytest.raises(ValueError, match="relative footprint"):
        whole_block_spatial_nulls(
            score=np.asarray([0.1, 0.2, 0.3, 0.4]),
            l2b_fields=packet,
            target_ids=frozenset({11}),
            block_ids=np.asarray(["a", "a", "b", "b"], dtype=object),
            permutations=2,
            seed=1,
        )


def test_resampling_summaries_require_at_least_95_percent_finite_draws():
    passing_bootstrap = tuple(
        BootstrapDraw(index, "auc", None if index == 0 else index / 20, None) for index in range(20)
    )
    interval = summarize_bootstrap_interval(
        passing_bootstrap,
        metric="auc",
        scheduled_replicates=20,
    )
    assert interval.gate_eligible
    assert interval.valid_replicates == 19
    assert interval.lower_95 is not None

    failing_bootstrap = tuple(
        BootstrapDraw(index, "auc", None if index < 2 else index / 20, None) for index in range(20)
    )
    failed = summarize_bootstrap_interval(
        failing_bootstrap,
        metric="auc",
        scheduled_replicates=20,
    )
    assert not failed.gate_eligible
    assert failed.lower_95 is None
    assert failed.unavailable_reason == "fewer_than_95_percent_finite_bootstrap_replicates"

    null_draws = tuple(
        SpatialNullDraw(index, "auc", None if index == 0 else 0.4, None) for index in range(20)
    )
    null = summarize_spatial_null(
        null_draws,
        metric="auc",
        observed=0.8,
        scheduled_permutations=20,
    )
    assert null.gate_eligible
    assert null.p_value == pytest.approx(1 / 20)
    unavailable = summarize_spatial_null(
        null_draws,
        metric="auc",
        observed=None,
        scheduled_permutations=20,
    )
    assert not unavailable.gate_eligible
    assert unavailable.p_value is None


def test_missing_null_blocks_confirmatory_inference():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_emit_l2b_validation.py")
    )
    row = {
        "metric": "rank_auc",
        "support_status": "confirmatory_eligible",
        "mapping": "exact",
        "bootstrap_interval_gate_eligible": True,
        "null_gate_eligible": False,
        "value": 0.9,
        "bh_adjusted_p_value": None,
        "bootstrap_lower_95": 0.8,
        "bootstrap_upper_95": 1.0,
        "null_direction": 0.5,
    }
    namespace["_finalize_endpoint_status"](row)

    assert row["inference_status"] == "inference_unavailable"
    assert row["claim_status"] == "no_confirmatory_claim"


def test_m2_manifest_handoff_loads_both_scales_and_refuses_stale_hash(tmp_path: Path):
    records = {}
    for scale, values in {
        "L": np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32),
        "2L": np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32),
    }.items():
        raster_path = tmp_path / f"blocks_{scale}.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=2,
            width=4,
            count=1,
            dtype="int32",
            crs="EPSG:4326",
            transform=Affine(1, 0, 0, 0, -1, 2),
            nodata=0,
        ) as dataset:
            dataset.write(values, 1)
        records[scale] = {
            "block_raster": raster_path.name,
            "block_raster_sha256": sha256_file(raster_path),
            "complete_block_ids": sorted(int(value) for value in np.unique(values)),
            "block_side_pixels": 2,
            "halo_pixels": 1,
        }

    repository = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "block_manifest.json"
    manifest = {
        "protocol": {
            "sha256": sha256_file(repository / "docs" / "m2_spatial_validation_preregistration.md"),
            "protocol_compliant": True,
        },
        "sites": {
            "goldfield": {
                "scene_id": "synthetic-anchor",
                "grid": {
                    "shape": [2, 4],
                    "crs": "EPSG:4326",
                    "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 2.0],
                },
                "scales": records,
            }
        },
    }
    write_strict_json(manifest_path, manifest)

    scales = load_m2_block_scales(manifest_path, site="goldfield", scene_id="synthetic-anchor")
    assert set(scales) == {"L", "2L"}
    assert scales["L"].complete_block_ids == (1, 2)

    manifest["sites"]["goldfield"]["scales"]["2L"]["block_raster_sha256"] = "0" * 64
    write_strict_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="stale hash"):
        load_m2_block_scales(manifest_path, site="goldfield", scene_id="synthetic-anchor")


def test_pinned_l2a_hash_and_filename_identity_bind_the_l2b_pair(tmp_path: Path):
    docs = tmp_path / "docs"
    emit_dir = tmp_path / "data" / "raw" / "emit"
    docs.mkdir()
    emit_dir.mkdir(parents=True)
    l2a = emit_dir / f"EMIT_L2A_RFL_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"
    l2a.write_bytes(b"synthetic-pinned-l2a")
    manifest = docs / "input_manifest.json"
    record = {
        "id": "emit-goldfield-rfl",
        "logical_path": str(l2a.relative_to(tmp_path)),
        "size_bytes": l2a.stat().st_size,
        "sha256": "0" * 64,
    }
    write_strict_json(
        manifest,
        {"hash_algorithm": "sha256", "inputs": [record]},
    )
    with pytest.raises(ValueError, match="size or SHA-256 differs"):
        load_pinned_emit_l2a_input(manifest)

    record["sha256"] = sha256_file(l2a)
    write_strict_json(
        manifest,
        {"hash_algorithm": "sha256", "inputs": [record]},
    )
    pinned = load_pinned_emit_l2a_input(manifest)
    matching = ProductIdentity("MIN", VERSION, ACQUISITION, ORBIT, SCENE)
    validate_l2b_identity_against_l2a(matching, pinned.identity)

    internally_consistent_but_wrong = ProductIdentity(
        "MIN", VERSION, "20230805T191650", ORBIT, SCENE
    )
    with pytest.raises(ProductMismatchError, match="pinned L2A"):
        validate_l2b_identity_against_l2a(
            internally_consistent_but_wrong,
            pinned.identity,
        )


def _write_ontology_decision(
    evidence: Path,
    entry: OntologyEntry,
    *,
    assertion: str,
    authority_content_path: str | None = None,
    authority_content_sha256: str | None = None,
    authority_field_path: str | None = None,
) -> OntologyEntry:
    decision = _ontology_decision_payload(
        entry,
        assertion=assertion,
        authority_content_path=authority_content_path,
        authority_content_sha256=authority_content_sha256,
        authority_field_path=authority_field_path,
    )
    write_strict_json(
        evidence,
        {
            "schema_version": "emit-l2b-ontology-evidence/v3",
            "decisions": [decision],
        },
    )
    return OntologyEntry(**{**entry.__dict__, "source_sha256": sha256_file(evidence)})


def _ontology_decision_payload(
    entry: OntologyEntry,
    *,
    assertion: str,
    authority_content_path: str | None = None,
    authority_content_sha256: str | None = None,
    authority_field_path: str | None = None,
) -> dict[str, object]:
    return {
        "authority_content_path": authority_content_path,
        "authority_content_sha256": authority_content_sha256,
        "authority_field_path": authority_field_path,
        "evidence_id": entry.evidence_id,
        "evidence_type": entry.evidence_type,
        "evidence_locator": entry.evidence_locator,
        "evidence_assertion": assertion,
        "index": entry.index,
        "name": entry.name,
        "group": entry.group,
        "library": entry.library,
        "mapping": entry.mapping,
        "target": entry.target,
        "tanager_score": entry.tanager_score,
        "unavailable_reason": entry.unavailable_reason,
    }


def _ontology_entry(evidence: Path, **overrides: object) -> OntologyEntry:
    fields: dict[str, object] = {
        "ontology_version": "emit-e4-test-v2",
        "index": 11,
        "name": "alpha",
        "group": 1,
        "library": "splib07",
        "mapping": "broader",
        "target": "alunite",
        "tanager_score": "mtmf:alunite",
        "source_path": str(evidence),
        "source_sha256": "0" * 64,
        "evidence_id": "row-11",
        "evidence_type": "explicit_broader_mapping",
        "evidence_locator": ("https://lpdaac.usgs.gov/documents/1660/EMITL2BMIN_User_Guide_V1.pdf"),
        "unavailable_reason": None,
    }
    fields.update(overrides)
    return OntologyEntry(**fields)


def _write_authority_capture(
    path: Path,
    source_locator: str,
    decisions: list[dict[str, str]],
) -> None:
    write_strict_json(
        path,
        {
            "schema_version": "emit-l2b-authority-capture/v1",
            "source_locator": source_locator,
            "decisions": decisions,
        },
    )


def _authority_decision(
    field_path: str = "mineral_metadata[index=11,name=alpha]",
    *,
    source: str = "alpha",
    relation: str = "maps_to",
    target: str = "alunite",
) -> dict[str, str]:
    return {
        "field_path": field_path,
        "source": source,
        "relation": relation,
        "target": target,
    }


def test_ontology_rejects_nonliteral_exact_mapping(tmp_path: Path):
    evidence = tmp_path / "ontology_evidence.json"
    entry = _ontology_entry(
        evidence,
        mapping="exact",
        evidence_type="exact_name_equality",
        evidence_locator="mechanical:source_name_equals_target",
    )
    entry = _write_ontology_decision(
        evidence,
        entry,
        assertion="normalized_source_name == normalized_target",
    )
    source = (SourceMineral(index=11, name="alpha", group=1, library="splib07"),)

    with pytest.raises(ValueError, match="normalized literal"):
        validate_ontology_crosswalk((entry,), source)


def test_ontology_rejects_synthetic_and_unpinned_broader_evidence(tmp_path: Path):
    evidence = tmp_path / "ontology_evidence.json"
    authority = tmp_path / "authority_capture.json"
    locator = _ontology_entry(evidence).evidence_locator
    _write_authority_capture(authority, locator, [_authority_decision()])
    source = (SourceMineral(index=11, name="alpha", group=1, library="splib07"),)

    synthetic = _ontology_entry(
        evidence,
        evidence_locator="synthetic-authority:claim-1",
    )
    synthetic = _write_ontology_decision(
        evidence,
        synthetic,
        assertion="alpha maps_to alunite",
        authority_content_path=authority.name,
        authority_content_sha256=sha256_file(authority),
        authority_field_path="mineral_metadata[index=11,name=alpha]",
    )
    with pytest.raises(ValueError, match="synthetic or a placeholder"):
        validate_ontology_crosswalk((synthetic,), source)

    unpinned = _ontology_entry(evidence)
    unpinned = _write_ontology_decision(
        evidence,
        unpinned,
        assertion="alpha maps_to alunite",
        authority_content_path=authority.name,
        authority_field_path="mineral_metadata[index=11,name=alpha]",
    )
    with pytest.raises(ValueError, match="pinned authority content SHA-256"):
        validate_ontology_crosswalk((unpinned,), source)


def test_ontology_accepts_pinned_external_row_specific_citation(tmp_path: Path):
    evidence = tmp_path / "ontology_evidence.json"
    authority = tmp_path / "authority_capture.json"
    entry = _ontology_entry(evidence)
    _write_authority_capture(
        authority,
        entry.evidence_locator,
        [_authority_decision()],
    )
    entry = _write_ontology_decision(
        evidence,
        entry,
        assertion="alpha maps_to alunite",
        authority_content_path=authority.name,
        authority_content_sha256=sha256_file(authority),
        authority_field_path="mineral_metadata[index=11,name=alpha]",
    )
    source = (SourceMineral(index=11, name="alpha", group=1, library="splib07"),)

    assert validate_ontology_crosswalk((entry,), source) == (entry,)

    unsupported = (OntologyEntry(**{**entry.__dict__, "source_sha256": "0" * 64}),)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_ontology_crosswalk(unsupported, source)


@pytest.mark.parametrize(
    "locator",
    (
        "https://10.0.0.1/authority.txt",
        "https://192.168.1.1/authority.txt",
        "https://127.1/authority.txt",
        "https://0177.0.0.1/authority.txt",
        "https://0x7f.0.0.1/authority.txt",
        "https://localhost./authority.txt",
        "https://localhost.localdomain/authority.txt",
    ),
)
def test_ontology_rejects_nonpublic_authority_hosts(tmp_path: Path, locator: str):
    evidence = tmp_path / "ontology_evidence.json"
    authority = tmp_path / "authority_capture.json"
    _write_authority_capture(authority, locator, [_authority_decision()])
    entry = _ontology_entry(evidence, evidence_locator=locator)
    entry = _write_ontology_decision(
        evidence,
        entry,
        assertion="alpha maps_to alunite",
        authority_content_path=authority.name,
        authority_content_sha256=sha256_file(authority),
        authority_field_path="mineral_metadata[index=11,name=alpha]",
    )
    source = (SourceMineral(index=11, name="alpha", group=1, library="splib07"),)

    with pytest.raises(ValueError, match="public|non-placeholder|numeric"):
        validate_ontology_crosswalk((entry,), source)


def test_ontology_rejects_unbound_content_and_negative_assertion(tmp_path: Path):
    evidence = tmp_path / "ontology_evidence.json"
    authority = tmp_path / "authority_capture.json"
    entry = _ontology_entry(evidence)
    _write_authority_capture(
        authority,
        entry.evidence_locator,
        [_authority_decision("mineral_metadata[index=12,name=beta]", source="beta")],
    )
    unbound = _write_ontology_decision(
        evidence,
        entry,
        assertion="alpha maps_to alunite",
        authority_content_path=authority.name,
        authority_content_sha256=sha256_file(authority),
        authority_field_path="mineral_metadata[index=11,name=alpha]",
    )
    source = (SourceMineral(index=11, name="alpha", group=1, library="splib07"),)
    with pytest.raises(ValueError, match="exact authority field path"):
        validate_ontology_crosswalk((unbound,), source)

    _write_authority_capture(
        authority,
        entry.evidence_locator,
        [
            _authority_decision(
                source="NOT(alpha maps_to alunite)",
                relation="does_not_map_to",
                target="",
            )
        ],
    )
    negated = _write_ontology_decision(
        evidence,
        entry,
        assertion="alpha maps_to alunite",
        authority_content_path=authority.name,
        authority_content_sha256=sha256_file(authority),
        authority_field_path="mineral_metadata[index=11,name=alpha]",
    )
    with pytest.raises(ValueError, match="exactly bind"):
        validate_ontology_crosswalk((negated,), source)


def test_ontology_rejects_negated_unmapped_assertion(tmp_path: Path):
    evidence = tmp_path / "ontology_evidence.json"
    entry = _ontology_entry(
        evidence,
        mapping="unmapped",
        target="",
        tanager_score="",
        evidence_type="unmapped_decision",
        evidence_locator="schema-audit:unmapped:row-11",
        unavailable_reason="not_a_target",
    )
    source = (SourceMineral(index=11, name="alpha", group=1, library="splib07"),)
    negated = _write_ontology_decision(
        evidence,
        entry,
        assertion="NOT(alpha is_unmapped)",
    )
    with pytest.raises(ValueError, match="positive canonical"):
        validate_ontology_crosswalk((negated,), source)

    positive = _write_ontology_decision(
        evidence,
        entry,
        assertion="alpha is_unmapped",
    )
    assert validate_ontology_crosswalk((positive,), source) == (positive,)


def test_ontology_rejects_malformed_extra_missing_and_duplicate_authority_capture(
    tmp_path: Path,
):
    evidence = tmp_path / "ontology_evidence.json"
    authority = tmp_path / "authority_capture.json"
    entry = _ontology_entry(evidence)
    source = (SourceMineral(index=11, name="alpha", group=1, library="splib07"),)
    decision_options = {
        "assertion": "alpha maps_to alunite",
        "authority_content_path": authority.name,
        "authority_field_path": "mineral_metadata[index=11,name=alpha]",
    }

    authority.write_text("{malformed JSON\n", encoding="utf-8")
    malformed = _write_ontology_decision(
        evidence,
        entry,
        authority_content_sha256=sha256_file(authority),
        **decision_options,
    )
    with pytest.raises(ValueError, match="structured JSON"):
        validate_ontology_crosswalk((malformed,), source)

    write_strict_json(
        authority,
        {
            "schema_version": "emit-l2b-authority-capture/v1",
            "source_locator": entry.evidence_locator,
            "decisions": [_authority_decision()],
            "comment": "NOT(alpha maps_to alunite)",
        },
    )
    extra = _write_ontology_decision(
        evidence,
        entry,
        authority_content_sha256=sha256_file(authority),
        **decision_options,
    )
    with pytest.raises(ValueError, match="exact frozen schema"):
        validate_ontology_crosswalk((extra,), source)

    write_strict_json(
        authority,
        {
            "schema_version": "emit-l2b-authority-capture/v1",
            "source_locator": entry.evidence_locator,
            "decisions": [
                {
                    "field_path": "mineral_metadata[index=11,name=alpha]",
                    "source": "alpha",
                    "relation": "maps_to",
                }
            ],
        },
    )
    missing = _write_ontology_decision(
        evidence,
        entry,
        authority_content_sha256=sha256_file(authority),
        **decision_options,
    )
    with pytest.raises(ValueError, match="exact frozen schema"):
        validate_ontology_crosswalk((missing,), source)

    _write_authority_capture(
        authority,
        entry.evidence_locator,
        [_authority_decision(), _authority_decision(target="kaolinite")],
    )
    duplicate = _write_ontology_decision(
        evidence,
        entry,
        authority_content_sha256=sha256_file(authority),
        **decision_options,
    )
    with pytest.raises(ValueError, match="non-empty and unique"):
        validate_ontology_crosswalk((duplicate,), source)


def test_ontology_rejects_authority_capture_escape_hash_mismatch_and_non_utf8(
    tmp_path: Path,
):
    packet = tmp_path / "packet"
    packet.mkdir()
    evidence = packet / "ontology_evidence.json"
    authority = packet / "authority_capture.json"
    outside = tmp_path / "outside.json"
    entry = _ontology_entry(evidence)
    source = (SourceMineral(index=11, name="alpha", group=1, library="splib07"),)
    _write_authority_capture(outside, entry.evidence_locator, [_authority_decision()])

    escaped = _write_ontology_decision(
        evidence,
        entry,
        assertion="alpha maps_to alunite",
        authority_content_path="../outside.json",
        authority_content_sha256=sha256_file(outside),
        authority_field_path="mineral_metadata[index=11,name=alpha]",
    )
    with pytest.raises(ValueError, match="stay beside"):
        validate_ontology_crosswalk((escaped,), source)

    _write_authority_capture(authority, entry.evidence_locator, [_authority_decision()])
    mismatched = _write_ontology_decision(
        evidence,
        entry,
        assertion="alpha maps_to alunite",
        authority_content_path=authority.name,
        authority_content_sha256="0" * 64,
        authority_field_path="mineral_metadata[index=11,name=alpha]",
    )
    with pytest.raises(ValueError, match="authority content SHA-256 mismatch"):
        validate_ontology_crosswalk((mismatched,), source)

    authority.write_bytes(b"\xff\xfe")
    non_utf8 = _write_ontology_decision(
        evidence,
        entry,
        assertion="alpha maps_to alunite",
        authority_content_path=authority.name,
        authority_content_sha256=sha256_file(authority),
        authority_field_path="mineral_metadata[index=11,name=alpha]",
    )
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        validate_ontology_crosswalk((non_utf8,), source)


def test_ontology_accepts_multi_row_structured_authority_capture(tmp_path: Path):
    evidence = tmp_path / "ontology_evidence.json"
    authority = tmp_path / "authority_capture.json"
    first = _ontology_entry(evidence)
    second = _ontology_entry(
        evidence,
        index=12,
        name="beta",
        target="kaolinite",
        tanager_score="mtmf:kaolinite",
        evidence_id="row-12",
    )
    first_path = "mineral_metadata[index=11,name=alpha]"
    second_path = "mineral_metadata[index=12,name=beta]"
    _write_authority_capture(
        authority,
        first.evidence_locator,
        [
            _authority_decision(first_path),
            _authority_decision(second_path, source="beta", target="kaolinite"),
        ],
    )
    authority_sha256 = sha256_file(authority)
    evidence_decisions = [
        _ontology_decision_payload(
            first,
            assertion="alpha maps_to alunite",
            authority_content_path=authority.name,
            authority_content_sha256=authority_sha256,
            authority_field_path=first_path,
        ),
        _ontology_decision_payload(
            second,
            assertion="beta maps_to kaolinite",
            authority_content_path=authority.name,
            authority_content_sha256=authority_sha256,
            authority_field_path=second_path,
        ),
    ]
    write_strict_json(
        evidence,
        {
            "schema_version": "emit-l2b-ontology-evidence/v3",
            "decisions": evidence_decisions,
        },
    )
    evidence_sha256 = sha256_file(evidence)
    entries = tuple(
        OntologyEntry(**{**entry.__dict__, "source_sha256": evidence_sha256})
        for entry in (first, second)
    )
    source = (
        SourceMineral(index=11, name="alpha", group=1, library="splib07"),
        SourceMineral(index=12, name="beta", group=1, library="splib07"),
    )

    assert validate_ontology_crosswalk(entries, source) == entries


def test_confirmatory_plan_and_ontology_expected_digest_mismatch(tmp_path: Path):
    repository = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(repository / "scripts" / "run_emit_l2b_validation.py"))
    require_expected_sha256 = runner["_require_expected_sha256"]
    plan = tmp_path / "plan.md"
    ontology = tmp_path / "ontology.csv"
    plan.write_text("frozen plan\n", encoding="utf-8")
    ontology.write_text("frozen ontology\n", encoding="utf-8")

    with pytest.raises(ValueError, match="E4 plan SHA-256 mismatch"):
        require_expected_sha256(plan, "0" * 64, label="E4 plan")
    with pytest.raises(ValueError, match="ontology crosswalk SHA-256 mismatch"):
        require_expected_sha256(ontology, "f" * 64, label="ontology crosswalk")
    assert require_expected_sha256(plan, sha256_file(plan), label="E4 plan") == sha256_file(plan)


def test_driver_rejects_noncanonical_m2_manifest_before_analysis(tmp_path: Path):
    repository = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(repository / "scripts" / "run_emit_l2b_validation.py"))
    require_canonical = runner["_require_canonical_m2_manifest"]
    canonical = tmp_path / "canonical.json"
    substitute = tmp_path / "substitute.json"
    canonical.write_text("{}\n", encoding="utf-8")
    substitute.write_text("{}\n", encoding="utf-8")
    require_canonical.__globals__["CANONICAL_M2_BLOCK_MANIFEST"] = canonical

    with pytest.raises(ValueError, match="canonical accepted M2 block manifest"):
        require_canonical(substitute, sha256_file(substitute))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        require_canonical(canonical, "0" * 64)
    assert require_canonical(canonical, sha256_file(canonical)) == (
        canonical.resolve(),
        sha256_file(canonical),
    )


def test_synthetic_end_to_end_driver_reports_governed_endpoint_scale_results(
    tmp_path: Path,
):
    repository = Path(__file__).resolve().parents[1]
    synthetic_root = tmp_path / "synthetic_repo"
    docs = synthetic_root / "docs"
    raw = synthetic_root / "data" / "raw"
    emit_l2a_dir = raw / "emit"
    emit_dir = raw / "emit_l2b"
    speclib_parent = synthetic_root / "data" / "speclib"
    speclib = speclib_parent / "ASCIIdata_splib07a"
    for directory in (docs, raw, emit_l2a_dir, emit_dir, speclib):
        directory.mkdir(parents=True, exist_ok=True)

    scene = raw / "20240925_185504_87_4001_ortho_sr_hdf5.h5"
    scene.write_bytes(b"synthetic-tanager-scene")
    l2a = emit_l2a_dir / f"EMIT_L2A_RFL_{VERSION}_{ACQUISITION}_{ORBIT}_{SCENE}.nc"
    l2a.write_bytes(b"synthetic-pinned-l2a")
    library_member = speclib / "synthetic_library.txt"
    library_member.write_text("synthetic spectral library\n", encoding="utf-8")
    archive = speclib_parent / "ASCIIdata_splib07a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(
            library_member,
            arcname="ASCIIdata_splib07a/synthetic_library.txt",
        )
    input_manifest = docs / "input_manifest.json"
    write_strict_json(
        input_manifest,
        {
            "schema_version": "1.0",
            "hash_algorithm": "sha256",
            "inputs": [
                {
                    "id": "emit-goldfield-rfl",
                    "logical_path": str(l2a.relative_to(synthetic_root)),
                    "size_bytes": l2a.stat().st_size,
                    "sha256": sha256_file(l2a),
                },
                {
                    "id": "tanager-goldfield-1",
                    "logical_path": str(scene.relative_to(synthetic_root)),
                    "size_bytes": scene.stat().st_size,
                    "sha256": sha256_file(scene),
                },
                {
                    "id": "usgs-splib07a-archive",
                    "logical_path": str(archive.relative_to(synthetic_root)),
                    "size_bytes": archive.stat().st_size,
                    "sha256": sha256_file(archive),
                },
            ],
        },
    )

    shape = (4, 5)
    glt_x = np.tile(np.arange(1, shape[1] + 1, dtype=np.int32), (shape[0], 1))
    glt_y = np.tile(
        np.arange(1, shape[0] + 1, dtype=np.int32)[:, np.newaxis],
        (1, shape[1]),
    )
    target = np.arange(np.prod(shape)).reshape(shape) < 10
    raw_ids = {
        1: np.where(target, 11, 12).astype(np.int16),
        2: np.where(target, 111, 112).astype(np.int16),
    }
    base_depth = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100 + 0.1
    raw_depths = {1: base_depth, 2: base_depth + 1.0}
    mineral_records = (
        SourceMineral(11, "alunite", 1, "splib07"),
        SourceMineral(12, "other_group_1", 1, "splib07"),
        SourceMineral(111, "acid_sulfate", 2, "splib07"),
        SourceMineral(112, "other_group_2", 2, "splib07"),
    )
    min_attrs = _attrs("MIN")
    uncertainty_attrs = _attrs("MINUNCERT")
    for attrs in (min_attrs, uncertainty_attrs):
        attrs["geotransform"] = np.asarray([0.0, 1.0, 0.0, 4.0, 0.0, -1.0])
    min_path = _write_product(
        emit_dir,
        "MIN",
        attrs=min_attrs,
        glt_x=glt_x,
        glt_y=glt_y,
        raw_ids=raw_ids,
        raw_depths=raw_depths,
        mineral_records=mineral_records,
    )
    uncertainty_path = _write_product(
        emit_dir,
        "MINUNCERT",
        attrs=uncertainty_attrs,
        glt_x=glt_x,
        glt_y=glt_y,
        uncertainty_raw=np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1,
    )
    pair_for_manifest = load_emit_l2b_pair(min_path, uncertainty_path)
    write_strict_json(
        emit_dir / "download_manifest.json",
        {
            "schema_version": "emit-l2b-fetch/v4",
            "catalog_resolved_at_utc": "2026-08-09T00:00:00+00:00",
            "retrieval_mode": "verified_existing_pair",
            "downloaded_at_utc": None,
            "collection": "EMITL2BMIN",
            "granule_prefix": f"{ACQUISITION}_{ORBIT}_{SCENE}",
            "identity": {
                "kind": "MIN",
                "version": VERSION,
                "acquisition": ACQUISITION,
                "orbit": ORBIT,
                "scene": SCENE,
            },
            "identity_evidence": l2b_identity_evidence(pair_for_manifest),
            "pinned_l2a": {
                "input_manifest_id": "emit-goldfield-rfl",
                "input_manifest_sha256": sha256_file(input_manifest),
                "logical_path": str(l2a.relative_to(synthetic_root)),
                "filename": l2a.name,
                "size_bytes": l2a.stat().st_size,
                "sha256": sha256_file(l2a),
                "identity": {
                    "kind": "L2A_RFL",
                    "version": VERSION,
                    "acquisition": ACQUISITION,
                    "orbit": ORBIT,
                    "scene": SCENE,
                },
            },
            "cmr_granule": {
                "concept_id": "G123456-TEST",
                "revision_id": 7,
                "collection_concept_id": "C123456-TEST",
                "granule_ur": min_path.stem,
                "single_result_pair": True,
            },
            "inputs": [
                {
                    "role": "MIN",
                    "filename": min_path.name,
                    "size_bytes": min_path.stat().st_size,
                    "sha256": sha256_file(min_path),
                    "catalog_url": f"https://example.test/{min_path.name}",
                    "global_metadata": pair_for_manifest.min_metadata,
                },
                {
                    "role": "MINUNCERT",
                    "filename": uncertainty_path.name,
                    "size_bytes": uncertainty_path.stat().st_size,
                    "sha256": sha256_file(uncertainty_path),
                    "catalog_url": f"https://example.test/{uncertainty_path.name}",
                    "global_metadata": pair_for_manifest.minuncert_metadata,
                },
            ],
            "unavailable_reason": None,
        },
    )

    block_values = np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    scale_records = {}
    for scale in ("L", "2L"):
        raster_path = tmp_path / f"blocks_{scale}.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=shape[0],
            width=shape[1],
            count=1,
            dtype="int32",
            crs="EPSG:4326",
            transform=Affine(1, 0, 0, 0, -1, 4),
            nodata=0,
        ) as dataset:
            dataset.write(block_values, 1)
        scale_records[scale] = {
            "block_raster": raster_path.name,
            "block_raster_sha256": sha256_file(raster_path),
            "complete_block_ids": list(range(1, np.prod(shape) + 1)),
            "block_side_pixels": 1,
            "halo_pixels": 0,
        }
    block_manifest = tmp_path / "block_manifest.json"
    write_strict_json(
        block_manifest,
        {
            "protocol": {
                "sha256": sha256_file(
                    repository / "docs" / "m2_spatial_validation_preregistration.md"
                ),
                "protocol_compliant": True,
            },
            "sites": {
                "goldfield": {
                    "scene_id": "20240925_185504_87_4001",
                    "grid": {
                        "shape": list(shape),
                        "crs": "EPSG:4326",
                        "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 4.0],
                    },
                    "scales": scale_records,
                }
            },
        },
    )

    evidence = docs / "ontology_evidence.json"
    authority = docs / "emit_l2b_user_guide_rows.json"
    authority_locator = "https://lpdaac.usgs.gov/documents/1660/EMITL2BMIN_User_Guide_V1.pdf"
    _write_authority_capture(
        authority,
        authority_locator,
        [
            _authority_decision(
                "mineral_metadata[index=111,name=acid_sulfate]",
                source="acid_sulfate",
            )
        ],
    )
    ontology = docs / "ontology.csv"
    columns = (
        "ontology_version",
        "index",
        "name",
        "group",
        "library",
        "mapping",
        "target",
        "tanager_score",
        "source_path",
        "source_sha256",
        "evidence_id",
        "evidence_type",
        "evidence_locator",
        "unavailable_reason",
    )
    ontology_rows = (
        (11, "alunite", 1, "exact", "alunite", "mtmf:alunite", ""),
        (12, "other_group_1", 1, "unmapped", "", "", "not_a_target"),
        (111, "acid_sulfate", 2, "broader", "alunite", "mtmf:alunite", ""),
        (112, "other_group_2", 2, "unmapped", "", "", "not_a_target"),
    )
    evidence_decisions = []
    evidence_fields = []
    for index, name, group, mapping, mapped_target, score_name, reason in ontology_rows:
        evidence_id = f"group-{group}-index-{index}"
        if mapping == "exact":
            evidence_type = "exact_name_equality"
            evidence_locator = "mechanical:source_name_equals_target"
            evidence_assertion = "normalized_source_name == normalized_target"
            authority_content_path = None
            authority_content_sha256 = None
            authority_field_path = None
        elif mapping == "broader":
            evidence_type = "explicit_broader_mapping"
            evidence_locator = authority_locator
            evidence_assertion = "acid_sulfate maps_to alunite"
            authority_content_path = authority.name
            authority_content_sha256 = sha256_file(authority)
            authority_field_path = "mineral_metadata[index=111,name=acid_sulfate]"
        else:
            evidence_type = "unmapped_decision"
            evidence_locator = f"schema-audit:unmapped:{evidence_id}"
            evidence_assertion = f"{name} is_unmapped"
            authority_content_path = None
            authority_content_sha256 = None
            authority_field_path = None
        evidence_fields.append((evidence_id, evidence_type, evidence_locator))
        evidence_decisions.append(
            {
                "authority_content_path": authority_content_path,
                "authority_content_sha256": authority_content_sha256,
                "authority_field_path": authority_field_path,
                "evidence_id": evidence_id,
                "evidence_type": evidence_type,
                "evidence_locator": evidence_locator,
                "evidence_assertion": evidence_assertion,
                "index": index,
                "name": name,
                "group": group,
                "library": "splib07",
                "mapping": mapping,
                "target": mapped_target,
                "tanager_score": score_name,
                "unavailable_reason": reason or None,
            }
        )
    write_strict_json(
        evidence,
        {
            "schema_version": "emit-l2b-ontology-evidence/v3",
            "decisions": evidence_decisions,
        },
    )
    with ontology.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row, evidence_row in zip(ontology_rows, evidence_fields, strict=True):
            index, name, group, mapping, mapped_target, score_name, reason = row
            evidence_id, evidence_type, evidence_locator = evidence_row
            writer.writerow(
                {
                    "ontology_version": "synthetic-e4-v1",
                    "index": index,
                    "name": name,
                    "group": group,
                    "library": "splib07",
                    "mapping": mapping,
                    "target": mapped_target,
                    "tanager_score": score_name,
                    "source_path": evidence.name,
                    "source_sha256": sha256_file(evidence),
                    "evidence_id": evidence_id,
                    "evidence_type": evidence_type,
                    "evidence_locator": evidence_locator,
                    "unavailable_reason": reason,
                }
            )

    runner = runpy.run_path(str(repository / "scripts" / "run_emit_l2b_validation.py"))
    run = runner["run"]
    score = np.concatenate((np.linspace(0.6, 1.5, 10), np.linspace(0.0, 0.45, 10))).reshape(shape)

    def synthetic_scores(_scene: Path, _speclib: Path):
        return (
            {"mtmf:alunite": score},
            np.ones(shape, dtype=bool),
            Affine(1, 0, 0, 0, -1, 4),
            "EPSG:4326",
            {"synthetic": True},
        )

    run.__globals__["_tanager_scores"] = synthetic_scores
    run.__globals__["DEFAULT_INPUT_MANIFEST"] = input_manifest
    run.__globals__["CANONICAL_M2_BLOCK_MANIFEST"] = block_manifest
    run.__globals__["BOOTSTRAP_REPLICATES"] = 40
    run.__globals__["PERMUTATION_REPLICATES"] = 39
    output = tmp_path / "output"
    run(
        argparse.Namespace(
            tanager_scene=scene,
            emit_min=min_path,
            emit_minuncert=uncertainty_path,
            block_manifest=block_manifest,
            ontology_crosswalk=ontology,
            expected_e4_plan_sha256=sha256_file(
                repository / "docs" / "m3_external_validation_execution_plan.md"
            ),
            expected_ontology_sha256=sha256_file(ontology),
            expected_m2_block_manifest_sha256=sha256_file(block_manifest),
            input_manifest=input_manifest,
            speclib=speclib,
            output=output,
        )
    )

    required = {
        "input_manifest.json",
        "ontology_crosswalk.csv",
        "support_and_exclusions.csv",
        "metrics.csv",
        "endpoint_scale_results.csv",
        "bootstrap.csv",
        "spatial_nulls.csv",
        "spatial_null_summary.csv",
        "failure_map.tif",
        "summary.json",
        "report.md",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["execution_status"] == "complete"
    assert summary["inference_status"] == "confirmatory_result_available"
    assert summary["claim_status"] == "confirmatory_concordance_supported"
    assert summary["cmr_granule"]["concept_id"] == "G123456-TEST"

    with (output / "endpoint_scale_results.csv").open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle))
    exact = [
        row
        for row in results
        if row["mapping"] == "exact" and row["metric"] in {"rank_auc", "spearman_band_depth"}
    ]
    broader = [
        row
        for row in results
        if row["mapping"] == "broader" and row["metric"] in {"rank_auc", "spearman_band_depth"}
    ]
    assert len(exact) == 4
    assert {row["inference_status"] for row in exact} == {"confirmatory_eligible"}
    assert all(not row["bh_adjusted_p_value"] for row in exact)
    assert {row["claim_status"] for row in exact} == {"confirmatory_concordance_supported"}
    assert len(broader) == 4
    assert {row["support_status"] for row in broader} == {"secondary_only"}
    assert {row["claim_status"] for row in broader} == {"secondary_endpoint_no_confirmatory_claim"}
    assert all(row["bh_family"] == "compatible_mineral_secondary" for row in broader)
    assert all(row["bh_adjusted_p_value"] for row in broader)
    assert all(float(row["bootstrap_finite_fraction"]) >= 0.95 for row in exact)
    assert all(float(row["null_finite_fraction"]) >= 0.95 for row in exact)

    with (output / "support_and_exclusions.csv").open(newline="", encoding="utf-8") as handle:
        support = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in support:
        grouped.setdefault((row["endpoint"], row["geometry"]), []).append(row)
    for rows in grouped.values():
        assert sum(int(row["count"]) for row in rows) == int(rows[0]["denominator"])


def test_strict_json_replaces_nonfinite_numbers_and_rejects_unknown_types(tmp_path: Path):
    path = tmp_path / "summary.json"
    write_strict_json(path, {"metric": np.nan, "count": np.int64(2)})
    raw = path.read_text(encoding="utf-8")

    assert "NaN" not in raw
    assert json.loads(raw) == {"count": 2, "metric": None}
    with pytest.raises(TypeError):
        write_strict_json(path, {"bad": object()})
