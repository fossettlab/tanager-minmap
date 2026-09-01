"""Synthetic tests for frozen-scene repeatability metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from tanager_rocks.repeatability import (
    PRIMARY_PAIRS,
    SECONDARY_PAIRS,
    RepeatabilityPaths,
    _attach_grid_metadata,
    _boundary_block_resampling,
    _distribution_summary,
    _evaluated_layer_keys,
    _extract_paired_blocks,
    _json_safe,
    _load_repeatability_handoff,
    _reproject_continuous,
    _reproject_mask,
    _resampling_result,
    _rockwell_block_bootstrap,
    _safe_spearman,
    binary_overlap_metrics,
    classify_goldfield_repeatability,
    combined_public_gate,
    fixed_threshold_reference_metrics,
    goldfield_pair_gate,
    paired_block_bootstrap,
    paired_block_null,
    registration_sensitivity,
    resample_frozen_endmembers,
    site_scene_order,
    symmetric_boundary_distance_m,
)
from tanager_rocks.speclib import Endmember


def test_frozen_pairs_cover_every_declared_comparison_once():
    assert len(PRIMARY_PAIRS) == 5
    assert len(SECONDARY_PAIRS) == 6
    assert {pair.site_id for pair in PRIMARY_PAIRS} == {"bingham", "goldfield"}
    assert {pair.site_id for pair in SECONDARY_PAIRS} == {"goldfield"}


def test_site_anchor_is_processed_before_its_declared_repeats():
    orders = {site_id: site_scene_order(site_id) for site_id in ("bingham", "goldfield")}
    for site_id, ordered_scenes in orders.items():
        assert ordered_scenes[0] == next(
            pair.anchor_scene_id for pair in PRIMARY_PAIRS if pair.site_id == site_id
        )
        assert len(ordered_scenes) == len(set(ordered_scenes))


def test_qa_mask_keeps_crs_through_uint8_cast_and_reprojection():
    source_transform = from_origin(100.0, 200.0, 30.0, 30.0)
    raw_source = xr.DataArray(
        np.array([[1, 0], [1, 1]], dtype=np.uint8),
        dims=("y", "x"),
        coords={"y": [185.0, 155.0], "x": [115.0, 145.0]},
    )
    source_template = raw_source.rio.write_crs("EPSG:32611").rio.write_transform(source_transform)
    source = _attach_grid_metadata(raw_source, source_template)
    anchor = (
        xr.DataArray(
            np.zeros((4, 4), dtype=float),
            dims=("y", "x"),
            coords={
                "y": [192.5, 177.5, 162.5, 147.5],
                "x": [107.5, 122.5, 137.5, 152.5],
            },
        )
        .rio.write_crs("EPSG:32611")
        .rio.write_transform(from_origin(100.0, 200.0, 15.0, 15.0))
    )

    observed = _reproject_mask(source.astype("uint8"), anchor)

    assert observed.rio.crs == anchor.rio.crs
    assert observed.shape == anchor.shape
    assert observed.dtype == np.bool_
    assert np.count_nonzero(observed.values) == 12

    continuous = _reproject_continuous(source.astype(float), anchor)
    assert continuous.rio.crs == anchor.rio.crs
    assert continuous.shape == anchor.shape
    assert np.isfinite(continuous.values).all()


def test_repeatability_evaluates_only_manifest_declared_threshold_layers():
    declared = {"feature:al_oh_doublet": object(), "mtmf:alunite": object()}
    left = {**declared, "mtmf:gypsum": object()}
    right = {**declared, "mtmf:gypsum": object()}

    assert _evaluated_layer_keys(left, right, declared) == (
        "feature:al_oh_doublet",
        "mtmf:alunite",
    )

    with pytest.raises(ValueError, match="missing from one or both scenes"):
        _evaluated_layer_keys(left, {"feature:al_oh_doublet": object()}, declared)


def test_resampling_keeps_each_sites_anchor_sample_identity_on_native_axis():
    native_wavelengths = np.array([1000.0, 1000.2])
    library = [
        Endmember("alunite", "bingham_anchor.txt", "ASD", native_wavelengths, np.array([1.0, 1.1])),
        Endmember(
            "alunite", "goldfield_anchor.txt", "ASD", native_wavelengths, np.array([2.0, 2.1])
        ),
    ]

    bingham = resample_frozen_endmembers(library, {"alunite": "bingham_anchor.txt"})
    goldfield = resample_frozen_endmembers(library, {"alunite": "goldfield_anchor.txt"})

    assert bingham["alunite"].sample == "bingham_anchor.txt"
    assert goldfield["alunite"].sample == "goldfield_anchor.txt"
    np.testing.assert_allclose(goldfield["alunite"].wavelengths_nm, native_wavelengths)


def test_fixed_threshold_reference_metrics_uses_the_transferred_anchor_threshold():
    scores = np.array([[0.90, 0.80], [0.20, 0.10]])
    reference = np.array([[3, 3], [5, 5]])

    result = fixed_threshold_reference_metrics(scores, reference, frozenset({3}), threshold=0.50)

    assert result["available"] is True
    assert result["n_usable"] == 4
    assert result["n_pos"] == 2
    assert result["n_neg"] == 2
    assert result["auc"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_reference_metrics_mark_missing_class_coverage_unavailable():
    result = fixed_threshold_reference_metrics(
        np.array([[0.9, 0.1]]), np.array([[5, 5]]), frozenset({3}), threshold=0.5
    )

    assert result["available"] is False
    assert result["n_pos"] == 0


def test_reference_threshold_metrics_are_unavailable_without_transfer_training_support():
    result = fixed_threshold_reference_metrics(
        np.array([[0.9, 0.1]]),
        np.array([[3, 5]]),
        frozenset({3}),
        threshold=None,
    )

    assert result["auc"] == 1.0
    assert result["threshold"] is None
    assert result["threshold_metrics_available"] is False
    assert result["reason"] == "transferred_threshold_unavailable"


def test_binary_overlap_metrics_and_prevalence_are_hand_calculable():
    anchor = np.array([[True, True], [False, False]])
    repeat = np.array([[True, False], [True, False]])
    result = binary_overlap_metrics(anchor, repeat)

    assert result.intersection_count == 1
    assert result.union_count == 3
    assert result.anchor_count == 2
    assert result.repeat_count == 2
    assert result.iou == 1 / 3
    assert result.dice == 1 / 2
    assert result.prevalence_ratio == 1.0


def test_symmetric_boundary_distance_uses_projected_pixel_spacing():
    anchor = np.zeros((5, 5), dtype=bool)
    repeat = np.zeros((5, 5), dtype=bool)
    anchor[:, 2] = True
    repeat[:, 3] = True

    assert symmetric_boundary_distance_m(anchor, repeat, xres_m=10.0, yres_m=10.0) == 10.0


def test_missing_support_and_raster_edges_do_not_create_boundaries():
    all_detected = np.ones((4, 4), dtype=float)
    with_missing = all_detected.copy()
    with_missing[1:3, 1:3] = np.nan

    assert np.isnan(
        symmetric_boundary_distance_m(all_detected, all_detected, xres_m=10.0, yres_m=10.0)
    )
    assert np.isnan(
        symmetric_boundary_distance_m(with_missing, all_detected, xres_m=10.0, yres_m=10.0)
    )


def test_block_seams_and_omitted_cells_do_not_create_boundaries():
    block_ids = np.array(
        [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
        ]
    )
    anchor = np.array(
        [
            [1.0, 1.0, np.nan, 0.0, 0.0],
            [1.0, 1.0, np.nan, 0.0, 0.0],
        ]
    )

    result = _boundary_block_resampling(
        anchor,
        anchor,
        block_ids,
        xres_m=10.0,
        yres_m=10.0,
        n_bootstrap=20,
        n_null=10,
    )

    assert np.isnan(result["bootstrap"]["samples"]).all()
    assert np.isnan(result["spatial_null"]["samples"]).all()


def test_registration_sensitivity_reports_all_fixed_neighboring_shifts():
    anchor_scores = np.arange(25, dtype=float).reshape(5, 5)
    repeat_scores = anchor_scores.copy()
    anchor_mask = anchor_scores >= 12
    repeat_mask = anchor_mask.copy()
    valid = np.ones((5, 5), dtype=bool)

    result = registration_sensitivity(
        anchor_scores,
        repeat_scores,
        anchor_mask,
        repeat_mask,
        anchor_mask,
        repeat_mask,
        valid,
        valid,
        xres_m=10.0,
        yres_m=10.0,
    )

    assert len(result.shift_metrics) == 9
    assert result.unshifted.shift_y == 0
    assert result.unshifted.shift_x == 0
    assert result.unshifted.spearman == 1.0
    assert result.ranges["spearman"]["max"] == 1.0


def test_paired_block_bootstrap_is_seeded_and_uses_given_block_ids_only():
    anchor = np.array([[0.0, 1.0, 2.0, 3.0]])
    repeat = np.array([[0.0, 1.0, 3.0, 2.0]])
    block_ids = np.array([[10, 10, 20, 20]])

    first = paired_block_bootstrap(anchor, repeat, block_ids, n_reps=20, seed=42)
    second = paired_block_bootstrap(anchor, repeat, block_ids, n_reps=20, seed=42)

    np.testing.assert_allclose(first["spearman"], second["spearman"], equal_nan=True)
    assert first["n_blocks"] == 2
    assert first["scheduled_replicates"] == 20


def test_paired_block_null_is_seeded_and_preserves_complete_block_count():
    anchor = np.array([[0.0, 1.0, 2.0, 3.0]])
    repeat = np.array([[0.0, 1.0, 3.0, 2.0]])
    block_ids = np.array([[10, 10, 20, 20]])

    first = paired_block_null(anchor, repeat, block_ids, n_reps=20, seed=42)
    second = paired_block_null(anchor, repeat, block_ids, n_reps=20, seed=42)

    np.testing.assert_allclose(first["spearman"], second["spearman"], equal_nan=True)
    assert first["n_blocks"] == 2
    assert first["scheduled_replicates"] == 2
    assert first["enumerated_all_unique"] is True


def test_small_block_null_enumerates_all_unique_permutations():
    anchor = np.arange(6, dtype=float).reshape(1, 6)
    repeat = anchor.copy()
    block_ids = np.array([[1, 1, 2, 2, 3, 3]])

    result = paired_block_null(anchor, repeat, block_ids, n_reps=2, seed=42)

    assert result["scheduled_replicates"] == math.factorial(3)
    assert result["enumerated_all_unique"] is True


def test_block_null_does_not_carry_anchor_missingness_with_permuted_repeat_block():
    """A repeat block must move with only its own missingness pattern."""
    anchor = np.array([[1.0, np.nan, 2.0, 3.0]])
    repeat = np.array([[10.0, 11.0, np.nan, 13.0]])
    block_ids = np.array([[1, 1, 2, 2]])

    result = paired_block_null(
        anchor,
        repeat,
        block_ids,
        metric=lambda left, right: float(np.count_nonzero(np.isfinite(left) & np.isfinite(right))),
    )

    # Pre-permutation joint masking would make the swapped ordering have zero
    # finite pairs by attaching the anchor's original mask to the repeat date.
    np.testing.assert_array_equal(result["samples"], np.array([2.0, 2.0]))


def test_block_extraction_preserves_coordinates_and_missingness():
    block_ids = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    anchor = np.array([[1.0, np.nan, 10.0, 11.0], [3.0, 4.0, 12.0, 13.0]])
    repeat = np.array([[np.nan, 2.0, 20.0, 21.0], [5.0, 6.0, 22.0, np.nan]])

    blocks = _extract_paired_blocks(anchor, repeat, block_ids)

    assert [block.block_id for block in blocks] == [1, 2]
    assert all(block.anchor.shape == (2, 2) for block in blocks)
    np.testing.assert_allclose(blocks[0].anchor, anchor[:, :2], equal_nan=True)
    np.testing.assert_allclose(blocks[0].repeat, repeat[:, :2], equal_nan=True)
    np.testing.assert_allclose(blocks[1].anchor, anchor[:, 2:], equal_nan=True)
    np.testing.assert_allclose(blocks[1].repeat, repeat[:, 2:], equal_nan=True)


def test_nan_constant_and_empty_metrics_remain_undefined():
    empty_nan = binary_overlap_metrics(np.array([np.nan]), np.array([np.nan]))
    empty_zero = binary_overlap_metrics(np.zeros(4), np.zeros(4))
    assert np.isnan(empty_nan.iou) and np.isnan(empty_nan.dice)
    assert np.isnan(empty_zero.iou) and np.isnan(empty_zero.dice)
    assert np.isnan(_safe_spearman(np.ones(4), np.ones(4)))
    assert _safe_spearman(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(
        1.0
    )
    encoded = json.dumps(_json_safe({"undefined": float("nan")}), allow_nan=False)
    assert encoded == '{"undefined": null}'


def test_finite_replicate_counts_control_gate_eligibility():
    anchor = np.array([[1.0, 1.0, 2.0, 2.0]])
    repeat = anchor.copy()
    block_ids = np.array([[1, 1, 2, 2]])

    result = paired_block_bootstrap(
        anchor,
        repeat,
        block_ids,
        n_reps=40,
        seed=42,
        metric=lambda _left, _right: float("nan"),
    )

    assert result["scheduled_replicates"] == 40
    assert result["finite_replicates"] == 0
    assert result["finite_fraction"] == 0.0
    assert result["gate_eligible"] is False


def test_exactly_95_percent_finite_is_eligible_but_below_is_unavailable():
    exactly = _resampling_result(np.array([*np.arange(19, dtype=float), np.nan]), n_blocks=2)
    below = _resampling_result(np.array([*np.arange(18, dtype=float), np.nan, np.nan]), n_blocks=2)

    exactly_summary = _distribution_summary(exactly, interval=True)
    below_summary = _distribution_summary(below, interval=True)

    assert exactly_summary["finite_fraction"] == 0.95
    assert exactly_summary["status"] == "available"
    assert exactly_summary["gate_eligible"] is True
    assert below_summary["status"] == "unavailable"
    assert below_summary["lower_95"] is None
    assert below_summary["upper_95"] is None


def test_boundary_null_summary_uses_the_lower_fifth_percentile():
    raw = _resampling_result(np.arange(20, dtype=float), n_blocks=2)

    summary = _distribution_summary(raw, interval=False, tail="lower")

    assert summary["lower_5"] == pytest.approx(np.percentile(np.arange(20), 5))
    assert "upper_95" not in summary


def _gate(*, passed: bool, evaluable: bool = True) -> dict[str, bool]:
    return {"passed": passed, "evaluable": evaluable}


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        ([_gate(passed=True)] * 4, "strong"),
        ([_gate(passed=True), *[_gate(passed=False)] * 3], "date-dependent"),
        ([_gate(passed=False)] * 4, "unsupported"),
        ([_gate(passed=False)] * 3 + [_gate(passed=False, evaluable=False)], "unavailable"),
    ],
)
def test_goldfield_overall_classifications(gates, expected):
    assert classify_goldfield_repeatability(gates) == expected


def test_goldfield_pair_gate_uses_all_three_unshifted_components():
    gate = goldfield_pair_gate(
        spearman_bootstrap={"gate_eligible": True, "lower_95": 0.01},
        rockwell_balanced_accuracy={"gate_eligible": True, "lower_95": 0.51},
        observed_transferred_iou=0.4,
        transferred_iou_null={"gate_eligible": True, "upper_95": 0.39},
    )

    assert gate["grid"] == "unshifted_only"
    assert gate["registration_sensitivity_can_rescue"] is False
    assert gate["evaluable"] is True
    assert gate["passed"] is True


@pytest.mark.parametrize(
    ("external_gate", "repeatability", "classification", "wording", "passed"),
    [
        (
            {"passed": True, "evaluable": True},
            "strong",
            "validated_and_repeatable",
            "validated and repeatable",
            True,
        ),
        (
            {"passed": False, "evaluable": True},
            "strong",
            "stable_only",
            "stable",
            False,
        ),
        (
            {"passed": True, "evaluable": True},
            "date-dependent",
            "acquisition_specific",
            "acquisition-specific",
            False,
        ),
        (
            {"passed": False, "evaluable": True},
            "unsupported",
            "failed",
            "failed",
            False,
        ),
        (
            {"passed": False, "evaluable": False},
            "strong",
            "unavailable",
            "unavailable",
            False,
        ),
        (
            {"passed": True, "evaluable": True},
            "unavailable",
            "unavailable",
            "unavailable",
            False,
        ),
    ],
)
def test_combined_public_gate_uses_frozen_claim_wording(
    external_gate, repeatability, classification, wording, passed
):
    result = combined_public_gate(external_gate, repeatability)

    assert result["status"] == classification
    assert result["classification"] == classification
    assert result["frozen_wording"] == wording
    assert result["passed"] is passed


def test_repeat_rockwell_bootstrap_requires_ten_bearing_blocks_per_class():
    scores = np.tile(np.array([0.9, 0.1]), (10, 1))
    reference = np.tile(np.array([1.0, 0.0]), (10, 1))
    block_ids = np.repeat(np.arange(1, 11)[:, None], 2, axis=1)

    eligible = _rockwell_block_bootstrap(scores, reference, block_ids, threshold=0.5, n_reps=100)
    ineligible = _rockwell_block_bootstrap(
        scores[:9], reference[:9], block_ids[:9], threshold=0.5, n_reps=100
    )

    assert eligible["positive_bearing_blocks"] == 10
    assert eligible["negative_bearing_blocks"] == 10
    assert eligible["gate_eligible"] is True
    assert eligible["lower_95"] == 1.0
    assert ineligible["gate_eligible"] is False
    assert ineligible["lower_95"] is None
    assert ineligible["unavailable_reason"] == ("fewer_than_10_positive_or_negative_bearing_blocks")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_handoff_fixture(tmp_path: Path) -> tuple[RepeatabilityPaths, Path, Path]:
    root = tmp_path / "repo"
    spatial_dir = root / "data" / "processed" / "spatial_validation"
    spatial_dir.mkdir(parents=True)
    preregistration = root / "docs" / "m2_spatial_validation_preregistration.md"
    preregistration.parent.mkdir(parents=True)
    preregistration.write_text("frozen synthetic protocol\n", encoding="utf-8")
    transform = from_origin(100.0, 200.0, 30.0, 30.0)
    crs = "EPSG:32611"
    manifest_sites = {}
    source_rows = []
    for site_id, scene_id in (
        ("goldfield", "20240925_185504_87_4001"),
        ("bingham", "20250911_191523_58_4001"),
    ):
        raster_path = spatial_dir / f"block_ids_{site_id}_L.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint32",
            crs=crs,
            transform=transform,
            nodata=0,
        ) as dataset:
            dataset.write(np.ones((2, 2), dtype=np.uint32), 1)
        score_path = root / "data" / "intermediate" / "maps" / f"{site_id}_score.tif"
        reference_path = root / "data" / "reference" / f"{site_id}_reference.tif"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_bytes(f"{site_id}-score".encode())
        reference_path.write_bytes(f"{site_id}-reference".encode())
        raster_sha = _sha256(raster_path)
        scale_record = {
            "block_raster": raster_path.name,
            "block_raster_sha256": raster_sha,
            "complete_block_ids": [1],
        }
        manifest_sites[site_id] = {
            "scene_id": scene_id,
            "block_raster": raster_path.name,
            "complete_block_ids": [1],
            "grid": {
                "shape": [2, 2],
                "crs": crs,
                "transform": list(transform)[:6],
            },
            "scales": {"L": scale_record},
        }
        available = site_id == "goldfield"
        source_rows.append(
            {
                "site": site_id,
                "scene_id": scene_id,
                "scale": "L",
                "family": "feature",
                "layer": "al_oh_doublet",
                "governance_status": "exploratory_only" if available else "counts_and_maps_only",
                "positive_bearing_blocks": "5" if available else "0",
                "negative_bearing_blocks": "5" if available else "1",
                "threshold_status": "available" if available else "unavailable",
                "threshold": "0.75" if available else "",
                "unavailable_reason": "" if available else "counts_and_maps_only_support",
                "threshold_method": ("block_balanced_youden_all_usable_complete_primary_L_blocks"),
                "spatial_prereg_sha256": _sha256(preregistration),
                "source_score_path": str(score_path.relative_to(root)),
                "source_score_sha256": _sha256(score_path),
                "source_reference_path": str(reference_path.relative_to(root)),
                "source_reference_sha256": _sha256(reference_path),
                "block_manifest_path": "block_manifest.json",
                "block_manifest_sha256": "",
                "block_raster_path": raster_path.name,
                "block_raster_sha256": raster_sha,
                "block_shape_rows": "2",
                "block_shape_cols": "2",
                "block_crs": crs,
                "block_transform": json.dumps(list(transform)[:6], separators=(",", ":")),
            }
        )
    manifest_path = spatial_dir / "block_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "protocol": {
                    "path": "docs/m2_spatial_validation_preregistration.md",
                    "sha256": _sha256(preregistration),
                    "parameters": {
                        "max_pairs_per_field_lag": 200_000,
                        "bootstrap_replicates": 10_000,
                        "permutation_replicates": 9_999,
                        "seed": 42,
                    },
                },
                "sites": manifest_sites,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha = _sha256(manifest_path)
    for row in source_rows:
        row["block_manifest_sha256"] = manifest_sha
    threshold_path = spatial_dir / "transfer_thresholds.csv"
    with threshold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    summary_path = spatial_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "protocol": {
                    "path": "docs/m2_spatial_validation_preregistration.md",
                    "sha256": _sha256(preregistration),
                    "parameters": {
                        "max_pairs_per_field_lag": 200_000,
                        "bootstrap_replicates": 10_000,
                        "permutation_replicates": 9_999,
                        "seed": 42,
                    },
                    "protocol_compliant": True,
                },
                "block_manifest_sha256": manifest_sha,
                "external_reference_gate": {"passed": True, "evaluable": True},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths = RepeatabilityPaths.repo_default(root)
    return paths, manifest_path, threshold_path


def test_transfer_handoff_preserves_exact_threshold_provenance_and_zero_support(tmp_path):
    paths, manifest_path, threshold_path = _write_handoff_fixture(tmp_path)

    handoffs, thresholds, provenance = _load_repeatability_handoff(
        paths, manifest_path, threshold_path
    )

    assert thresholds["goldfield"]["feature:al_oh_doublet"].threshold == 0.75
    assert thresholds["bingham"]["feature:al_oh_doublet"].threshold is None
    assert thresholds["bingham"]["feature:al_oh_doublet"].unavailable_reason == (
        "counts_and_maps_only_support"
    )
    assert provenance["block_manifest_sha256"] == _sha256(manifest_path)
    assert provenance["transfer_thresholds_sha256"] == _sha256(threshold_path)
    assert provenance["spatial_summary_sha256"] == _sha256(manifest_path.with_name("summary.json"))
    assert provenance["external_reference_gate"] == {"passed": True, "evaluable": True}
    assert handoffs["goldfield"].shape == (2, 2)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda protocol: protocol.pop("parameters"), "protocol parameters mismatch"),
        (
            lambda protocol: protocol["parameters"].update({"bootstrap_replicates": 100}),
            "protocol parameters mismatch",
        ),
    ],
)
def test_block_manifest_rejects_absent_or_nondefault_protocol_parameters(tmp_path, mutation, match):
    paths, manifest_path, threshold_path = _write_handoff_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest["protocol"])
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _load_repeatability_handoff(paths, manifest_path, threshold_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda summary: summary["protocol"].update({"sha256": "0" * 64}),
            "preregistration SHA mismatch",
        ),
        (
            lambda summary: summary["protocol"].update({"protocol_compliant": False}),
            "protocol_compliant must be true",
        ),
        (
            lambda summary: summary["protocol"]["parameters"].update(
                {"permutation_replicates": 100}
            ),
            "protocol parameters mismatch",
        ),
        (
            lambda summary: summary.update({"block_manifest_sha256": "0" * 64}),
            "block manifest SHA mismatch",
        ),
        (
            lambda summary: summary.pop("external_reference_gate"),
            "no external_reference_gate record",
        ),
        (
            lambda summary: summary.update(
                {"external_reference_gate": {"passed": "yes", "evaluable": True}}
            ),
            "passed and evaluable must be booleans",
        ),
    ],
)
def test_spatial_summary_handoff_rejects_stale_or_invalid_gate_records(tmp_path, mutation, match):
    paths, manifest_path, threshold_path = _write_handoff_fixture(tmp_path)
    summary_path = manifest_path.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mutation(summary)
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _load_repeatability_handoff(paths, manifest_path, threshold_path)


def test_spatial_summary_is_required_beside_block_manifest_by_default(tmp_path):
    paths, manifest_path, threshold_path = _write_handoff_fixture(tmp_path)
    manifest_path.with_name("summary.json").unlink()

    with pytest.raises(FileNotFoundError, match="spatial-validation summary does not exist"):
        _load_repeatability_handoff(paths, manifest_path, threshold_path)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("scene_id", "wrong-anchor"),
        ("spatial_prereg_sha256", "0" * 64),
        ("block_manifest_sha256", "0" * 64),
        ("block_raster_sha256", "0" * 64),
        ("block_shape_rows", "3"),
        ("block_crs", "EPSG:4326"),
        ("block_transform", "[1,0,0,0,-1,0]"),
    ],
)
def test_transfer_handoff_mismatches_are_hard_failures(tmp_path, field, bad_value):
    paths, manifest_path, threshold_path = _write_handoff_fixture(tmp_path)
    with threshold_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0][field] = bad_value
    with threshold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="mismatch"):
        _load_repeatability_handoff(paths, manifest_path, threshold_path)
