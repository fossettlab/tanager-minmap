"""Synthetic, analytically checkable tests for spatially blocked validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rioxarray
from rasterio.transform import from_origin

from tanager_minmap.repeatability import BlockHandoff, _complete_overlap_block_ids
from tanager_minmap.spatial_validation import (
    FINITE_REPLICATE_FRACTION,
    BlockSample,
    VariogramPoint,
    _governed_confidence_interval,
    bearing_block_counts,
    benjamini_hochberg,
    block_balanced_youden,
    block_bootstrap_intervals,
    block_dimensions,
    complete_blocks,
    empirical_semivariogram,
    fit_exponential_variogram,
    governance_status,
    pooled_metrics,
    rank_auc,
    sample_blocks,
    spatial_cross_fit,
    whole_block_permutation_test,
)


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_spatial_validation.py"
    spec = importlib.util.spec_from_file_location("_spatial_validation_cli_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load CLI module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_spatial_cli_workers_is_positive_and_defaults_to_serial():
    cli = _load_cli_module()
    assert cli._parser().parse_args([]).workers == 1
    assert cli._parser().parse_args(["--workers", "4"]).workers == 4
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["--workers", "0"])


def test_metric_csv_schema_names_rank_and_threshold_denominators():
    cli = _load_cli_module()
    metrics = pooled_metrics(spatial_cross_fit(_perfect_block_samples(), halo_pixels=0))

    assert {
        "rank_n_pos",
        "rank_n_neg",
        "threshold_n_pos",
        "threshold_n_neg",
        "n_pos",
        "n_neg",
    }.issubset(cli.METRIC_CSV_FIELDS)
    assert set(vars(metrics)).issubset(cli.METRIC_CSV_FIELDS)


def test_spatial_runner_forwards_workers_to_exact_permutation_core(monkeypatch):
    cli = _load_cli_module()
    captured = {}

    def fake(samples, **kwargs):
        captured["samples"] = samples
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(cli, "whole_block_permutation_test", fake)
    samples = (object(),)
    result = cli._run_permutation_test(
        samples,
        halo_pixels=3,
        permutations=99,
        seed=42,
        workers=4,
    )

    assert result == "sentinel"
    assert captured == {
        "samples": samples,
        "halo_pixels": 3,
        "permutations": 99,
        "seed": 42,
        "workers": 4,
    }


def test_empirical_semivariogram_is_deterministic_and_bounded():
    field = np.arange(64, dtype=float).reshape(8, 8)
    first = empirical_semivariogram(field, pixel_size=30.0, lags=(1, 2), max_pairs=10)
    second = empirical_semivariogram(field, pixel_size=30.0, lags=(1, 2), max_pairs=10)

    assert first == second
    assert [point.lag_pixels for point in first] == [1, 2]
    assert [point.distance for point in first] == [30.0, 60.0]
    assert all(point.available_pairs > point.used_pairs == 10 for point in first)


def test_empirical_semivariogram_constant_field_is_zero():
    points = empirical_semivariogram(np.ones((8, 8)), pixel_size=1.0, lags=(1, 2))
    assert [point.semivariance for point in points] == [0.0, 0.0]


def test_within_support_exponential_variogram_fit_remains_exponential():
    nugget, sill, scale = 0.2, 1.5, 2.0
    distances = np.arange(1.0, 11.0)
    gamma = nugget + sill * (1.0 - np.exp(-distances / scale))
    points = tuple(
        VariogramPoint(i, float(distance), float(value), 100, 100)
        for i, (distance, value) in enumerate(zip(distances, gamma, strict=True), start=1)
    )
    fit = fit_exponential_variogram(points, field_variance=nugget + sill)

    assert fit.method == "exponential_bounded_least_squares"
    assert fit.nugget == pytest.approx(nugget, rel=1e-5)
    assert fit.sill == pytest.approx(sill, rel=1e-5)
    assert fit.scale == pytest.approx(scale, rel=1e-5)
    assert fit.practical_range == pytest.approx(-scale * np.log(0.05), rel=1e-5)


def test_finite_fit_beyond_observed_support_uses_empirical_fallback():
    nugget, sill, scale = 0.2, 1.5, 4.0
    distances = np.arange(1.0, 11.0)
    gamma = nugget + sill * (1.0 - np.exp(-distances / scale))
    points = tuple(
        VariogramPoint(i, float(distance), float(value), 100, 100)
        for i, (distance, value) in enumerate(zip(distances, gamma, strict=True), start=1)
    )
    fit = fit_exponential_variogram(points, field_variance=nugget + sill)

    assert -scale * np.log(0.05) > distances[-1]
    assert fit.method == "empirical_fallback_largest_evaluated_lag"
    assert fit.practical_range == distances[-1]
    assert fit.fallback_reason == "fitted_practical_range_beyond_largest_evaluated_lag"


def test_variogram_fallback_uses_first_lag_reaching_variance_target():
    points = (
        VariogramPoint(1, 30.0, 0.2, 10, 10),
        VariogramPoint(2, 60.0, 0.96, 10, 10),
    )
    fit = fit_exponential_variogram(points, field_variance=1.0)
    assert fit.practical_range == 60.0
    assert fit.method == "empirical_fallback_first_lag_at_95pct_variance"


def test_block_geometry_retains_only_complete_blocks():
    side, halo = block_dimensions((45.0, 60.0), pixel_size=30.0)
    blocks = complete_blocks((5, 7), block_side_pixels=2)

    assert (side, halo) == (4, 2)
    assert len(blocks) == 6
    assert blocks[-1].block_id == "r0001_c0002"
    assert blocks[-1].row_stop == 4
    assert blocks[-1].col_stop == 6


def test_block_raster_and_json_manifest_round_trip(tmp_path):
    cli = _load_cli_module()
    grid = cli.RasterGrid(
        shape=(5, 7),
        crs="EPSG:32611",
        transform=from_origin(100.0, 200.0, 30.0, 30.0),
        pixel_size=30.0,
    )
    primary_blocks = complete_blocks(grid.shape, 2)
    sensitivity_blocks = complete_blocks(grid.shape, 4)
    primary_record = cli._write_block_raster(
        tmp_path / "block_ids_goldfield_L.tif",
        grid,
        primary_blocks,
        site_id="goldfield",
        scene_id="anchor",
        scale="L",
    )
    sensitivity_record = cli._write_block_raster(
        tmp_path / "block_ids_goldfield_2L.tif",
        grid,
        sensitivity_blocks,
        site_id="goldfield",
        scene_id="anchor",
        scale="2L",
    )

    with rasterio.open(tmp_path / primary_record["block_raster"]) as dataset:
        values = dataset.read(1)
        assert dataset.shape == grid.shape
        assert dataset.crs.to_string() == grid.crs
        assert dataset.transform == grid.transform
        assert dataset.nodata == 0
        assert dataset.dtypes == ("uint32",)
    assert set(np.unique(values)) == {0, *primary_record["complete_block_ids"]}
    assert np.all(values[-1, :] == 0)
    assert np.all(values[:, -1] == 0)
    assert primary_record["complete_block_ids"] == list(range(1, 7))
    assert primary_record["numeric_to_string_block_ids"]["1"] == "r0000_c0000"

    result = {
        "site_summary": {
            "site": "goldfield",
            "scene_id": "anchor",
            "shape": list(grid.shape),
            "crs": grid.crs,
            "transform": [
                grid.transform.a,
                grid.transform.b,
                grid.transform.c,
                grid.transform.d,
                grid.transform.e,
                grid.transform.f,
            ],
            "pixel_size_metres": grid.pixel_size,
            "scores": [{"path": "anchor.tif", "sha256": "0" * 64}],
        },
        "block_handoff": {"L": primary_record, "2L": sensitivity_record},
        "blocks": [],
    }
    manifest_path = tmp_path / "block_manifest.json"
    protocol_parameters = cli._protocol_parameters(
        max_pairs=cli.MAX_PAIRS_PER_LAG,
        bootstrap_replicates=cli.BOOTSTRAP_REPLICATES,
        permutations=cli.PERMUTATION_REPLICATES,
        seed=cli.SEED,
    )
    cli._write_json(
        manifest_path,
        cli._block_manifest_payload([result], protocol_parameters=protocol_parameters),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    site = payload["sites"]["goldfield"]

    assert site["block_raster"] == "block_ids_goldfield_L.tif"
    assert site["complete_block_ids"] == list(range(1, 7))
    assert site["scales"]["2L"]["block_raster"] == "block_ids_goldfield_2L.tif"
    assert site["scales"]["2L"]["block_raster_sha256"] == sensitivity_record["block_raster_sha256"]
    assert (manifest_path.parent / site["block_raster"]).is_file()
    assert payload["protocol"]["parameters"] == protocol_parameters
    assert payload["protocol"]["protocol_compliant"] is True
    assert payload["strict_inductive_covariance"]["status"] == "deferred"

    with rioxarray.open_rasterio(tmp_path / site["block_raster"], masked=True) as raster:
        anchor_template = raster.squeeze("band", drop=True).load()
    handoff = BlockHandoff(
        site_id="goldfield",
        anchor_scene_id="anchor",
        raster_path=tmp_path / site["block_raster"],
        raster_sha256=primary_record["block_raster_sha256"],
        complete_block_ids=tuple(site["complete_block_ids"]),
        shape=grid.shape,
        crs=grid.crs,
        transform=grid.transform,
    )
    anchor_values = np.ones(grid.shape, dtype=float)
    repeat_values = np.ones(grid.shape, dtype=float)
    repeat_values[0, 0] = np.nan  # Partial QA must not discard this geometric block.
    repeat_values[0:2, 2:4] = np.nan  # No observed pair excludes this block.
    block_values, retained_count = _complete_overlap_block_ids(
        handoff,
        anchor_template,
        anchor_values,
        repeat_values,
    )
    assert 1 in np.unique(block_values)
    assert 2 not in np.unique(block_values)
    assert retained_count == 5
    assert np.all(block_values[-1, :] == 0)
    assert np.all(block_values[:, -1] == 0)


def test_block_balanced_youden_uses_equal_block_weight():
    blocks = complete_blocks((1, 2), 1)
    samples = (
        BlockSample(
            blocks[0],
            score=np.array([[0.9, 0.8, 0.7, 0.1]]),
            reference=np.array([[1.0, 1.0, 0.0, 0.0]]),
        ),
        BlockSample(
            blocks[1],
            score=np.array([[0.6, 0.5, 0.4, 0.3]]),
            reference=np.array([[1.0, 0.0, 0.0, 0.0]]),
        ),
    )
    assert block_balanced_youden(samples) == pytest.approx(0.6)


def test_block_balanced_youden_breaks_ties_at_highest_threshold():
    block = complete_blocks((1, 1), 1)[0]
    sample = BlockSample(
        block,
        score=np.array([[3.0, 2.0, 1.0, 0.0]]),
        reference=np.array([[1.0, 0.0, 1.0, 0.0]]),
    )
    assert block_balanced_youden((sample,)) == 3.0


def _perfect_block_samples() -> tuple[BlockSample, ...]:
    blocks = complete_blocks((2, 2), 1)
    score = np.array([[4.0, 4.0], [1.0, 1.0]])
    reference = np.array([[1.0, 1.0], [0.0, 0.0]])
    return sample_blocks(score, reference, blocks)


def test_spatial_cross_fit_and_pooled_metrics_are_perfect():
    result = spatial_cross_fit(_perfect_block_samples(), halo_pixels=0)
    metrics = pooled_metrics(result)

    assert len(result.folds) == 4
    assert result.skipped_blocks == ()
    assert {fold.threshold for fold in result.folds} == {4.0}
    assert metrics.auc == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.positive_f1 == 1.0
    assert metrics.negative_f1 == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.tpr == 1.0
    assert metrics.fpr == 0.0
    assert metrics.prevalence == 0.5
    assert (metrics.rank_n_pos, metrics.rank_n_neg) == (2, 2)
    assert (metrics.threshold_n_pos, metrics.threshold_n_neg) == (2, 2)
    assert (metrics.n_pos, metrics.n_neg) == (2, 2)


def test_halo_excludes_intersecting_training_blocks():
    blocks = complete_blocks((1, 5), 1)
    score = np.array([[4.0, 4.0, 1.0, 1.0, 1.0]])
    reference = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]])
    result = spatial_cross_fit(sample_blocks(score, reference, blocks), halo_pixels=1)

    # The first positive block has no remaining positive training block after
    # its adjacent block is excluded by the halo, so that fold is skipped.
    assert "r0000_c0000" in result.skipped_blocks


def test_rank_metrics_retain_blocks_when_threshold_calibration_fails():
    blocks = complete_blocks((1, 5), 1)
    samples = sample_blocks(
        np.array([[4.0, 4.0, 1.0, 1.0, 1.0]]),
        np.array([[1.0, 1.0, 0.0, 0.0, 0.0]]),
        blocks,
    )

    result = spatial_cross_fit(samples, halo_pixels=1)
    metrics = pooled_metrics(result)
    intervals = {
        interval.metric: interval
        for interval in block_bootstrap_intervals(result, replicates=200, seed=42)
    }

    assert "r0000_c0000" in result.skipped_blocks
    assert set(result.auc_block_ids) == {block.block_id for block in blocks}
    assert result.references.tolist() == [0, 0]
    assert metrics.auc == 1.0
    assert np.isnan(metrics.balanced_accuracy)
    assert (metrics.rank_n_pos, metrics.rank_n_neg) == (2, 3)
    assert (metrics.threshold_n_pos, metrics.threshold_n_neg) == (0, 2)
    assert (metrics.n_pos, metrics.n_neg) == (2, 3)
    assert intervals["auc"].valid_replicates > 0
    assert intervals["auc"].scheduled_replicates == 200
    assert intervals["auc"].gate_eligible is False
    assert np.isnan(intervals["auc"].lower)
    assert np.isnan(intervals["auc"].upper)
    assert intervals["auc"].unavailable_reason == "fewer_than_95_percent_finite_replicates"
    assert intervals["balanced_accuracy"].valid_replicates == 0


def test_rank_auc_counts_ties_as_half():
    scores = np.array([2.0, 1.0, 2.0, 0.0])
    references = np.array([1, 1, 0, 0])
    # Positive-negative comparisons: 0.5, 1, 0, 1 -> 2.5 / 4.
    assert rank_auc(scores, references) == pytest.approx(0.625)


def test_nonbinary_reference_is_rejected_instead_of_coerced():
    block = complete_blocks((1, 1), 1)[0]
    sample = BlockSample(block, np.array([[1.0]]), np.array([[0.5]]))
    with pytest.raises(ValueError, match="not binary"):
        sample.paired_values()


def test_block_bootstrap_is_seeded_and_withholds_small_sample_intervals():
    result = spatial_cross_fit(_perfect_block_samples(), halo_pixels=0)
    first = block_bootstrap_intervals(result, replicates=200, seed=42)
    second = block_bootstrap_intervals(result, replicates=200, seed=42)

    for first_interval, second_interval in zip(first, second, strict=True):
        assert first_interval.metric == second_interval.metric
        assert first_interval.scheduled_replicates == second_interval.scheduled_replicates
        assert first_interval.valid_replicates == second_interval.valid_replicates
        assert first_interval.finite_fraction == second_interval.finite_fraction
        assert first_interval.gate_eligible == second_interval.gate_eligible
        assert first_interval.unavailable_reason == second_interval.unavailable_reason
        assert np.isnan(first_interval.lower) == np.isnan(second_interval.lower)
        assert np.isnan(first_interval.upper) == np.isnan(second_interval.upper)
    interval = {item.metric: item for item in first}
    for metric in ("auc", "balanced_accuracy", "positive_f1", "negative_f1", "macro_f1"):
        assert interval[metric].scheduled_replicates == 200
        assert interval[metric].finite_fraction == pytest.approx(
            interval[metric].valid_replicates / 200
        )
        expected_eligible = interval[metric].valid_replicates >= 190
        assert interval[metric].gate_eligible is expected_eligible
        if expected_eligible:
            assert interval[metric].lower == 1.0
            assert interval[metric].upper == 1.0
        else:
            assert np.isnan(interval[metric].lower)
            assert np.isnan(interval[metric].upper)


def test_confidence_interval_requires_95_percent_finite_replicates():
    below = _governed_confidence_interval(
        "auc",
        np.array([0.8] * 18 + [np.nan] * 2),
        scheduled_replicates=20,
    )

    assert FINITE_REPLICATE_FRACTION == 0.95
    assert below.valid_replicates == 18
    assert below.finite_fraction == 0.9
    assert below.gate_eligible is False
    assert np.isnan(below.lower)
    assert np.isnan(below.upper)
    assert below.unavailable_reason == "fewer_than_95_percent_finite_replicates"


def test_confidence_interval_accepts_exact_95_percent_finite_floor():
    at_floor = _governed_confidence_interval(
        "auc",
        np.array([0.8] * 19 + [np.nan]),
        scheduled_replicates=20,
    )

    assert at_floor.valid_replicates == 19
    assert at_floor.finite_fraction == 0.95
    assert at_floor.gate_eligible is True
    assert at_floor.lower == pytest.approx(0.8)
    assert at_floor.upper == pytest.approx(0.8)
    assert at_floor.unavailable_reason is None


def test_whole_block_permutation_is_deterministic():
    samples = _perfect_block_samples()
    first = whole_block_permutation_test(samples, halo_pixels=0, permutations=99, seed=42)
    second = whole_block_permutation_test(samples, halo_pixels=0, permutations=99, seed=42)

    assert first == second
    assert 0.0 < first.auc_p_value <= 1.0
    assert 0.0 < first.balanced_accuracy_p_value <= 1.0
    assert first.valid_auc_permutations == 99


def test_governance_and_block_counts_follow_frozen_thresholds():
    samples = _perfect_block_samples()
    assert bearing_block_counts(samples) == (2, 2)
    assert governance_status(4, 20) == "counts_and_maps_only"
    assert governance_status(5, 20) == "exploratory_only"
    assert governance_status(10, 10) == "confirmatory_eligible"


def test_benjamini_hochberg_restores_original_order_and_nan():
    adjusted = benjamini_hochberg([0.04, 0.01, np.nan, 0.03])
    assert adjusted[:2] == pytest.approx([0.04, 0.03])
    assert np.isnan(adjusted[2])
    assert adjusted[3] == pytest.approx(0.04)


def test_transfer_threshold_uses_all_primary_l_blocks_and_exact_provenance(tmp_path):
    cli = _load_cli_module()
    score_path = tmp_path / "score.tif"
    reference_path = tmp_path / "reference.tif"
    score_path.write_bytes(b"score-source")
    reference_path.write_bytes(b"reference-source")
    score = np.array([[0.9] * 5 + [0.1] * 5])
    reference = np.array([[1.0] * 5 + [0.0] * 5])
    blocks = complete_blocks(score.shape, 1)
    endpoint = cli.EndpointInput(
        spec=cli.EndpointSpec("feature", "al_oh_doublet", "al_oh_doublet", frozenset({3})),
        score=score,
        score_path=score_path,
        binary_reference=reference,
    )
    grid = cli.RasterGrid(
        shape=score.shape,
        crs="EPSG:32611",
        transform=from_origin(0.0, 30.0, 30.0, 30.0),
        pixel_size=30.0,
    )

    row = cli._transfer_threshold_row(
        "goldfield",
        "anchor",
        endpoint,
        sample_blocks(score, reference, blocks),
        governance="exploratory_only",
        positive_blocks=5,
        negative_blocks=5,
        complete_block_count=10,
        reference_path=reference_path,
        raster_record={"block_raster": "blocks.tif", "block_raster_sha256": "a" * 64},
        grid=grid,
    )

    assert row["scale"] == "L"
    assert row["threshold"] == pytest.approx(0.9)
    assert row["threshold_status"] == "available"
    assert row["threshold_method"] == ("block_balanced_youden_all_usable_complete_primary_L_blocks")
    assert row["source_score_sha256"] == cli._sha256(score_path)
    assert row["source_reference_sha256"] == cli._sha256(reference_path)
    assert row["spatial_prereg_sha256"] == cli._sha256(cli.PREREGISTRATION_PATH)
    manifest_path = tmp_path / "block_manifest.json"
    manifest_path.write_text('{"frozen":true}\n', encoding="utf-8")
    finalized = cli._finalize_transfer_threshold_rows([row], manifest_path)
    assert row["block_manifest_sha256"] is None
    assert finalized[0]["block_manifest_path"] == "block_manifest.json"
    assert finalized[0]["block_manifest_sha256"] == cli._sha256(manifest_path)


def test_transfer_threshold_zero_positive_support_is_explicitly_unavailable(tmp_path):
    cli = _load_cli_module()
    score_path = tmp_path / "score.tif"
    reference_path = tmp_path / "reference.tif"
    score_path.write_bytes(b"score-source")
    reference_path.write_bytes(b"reference-source")
    score = np.array([[0.9, 0.1]])
    reference = np.array([[0.0, 0.0]])
    endpoint = cli.EndpointInput(
        spec=cli.EndpointSpec("feature", "al_oh_doublet", "al_oh_doublet", frozenset({3})),
        score=score,
        score_path=score_path,
        binary_reference=reference,
    )
    grid = cli.RasterGrid(
        shape=score.shape,
        crs="EPSG:32611",
        transform=from_origin(0.0, 30.0, 30.0, 30.0),
        pixel_size=30.0,
    )

    row = cli._transfer_threshold_row(
        "goldfield",
        "anchor",
        endpoint,
        sample_blocks(score, reference, complete_blocks(score.shape, 1)),
        governance="counts_and_maps_only",
        positive_blocks=0,
        negative_blocks=2,
        complete_block_count=2,
        reference_path=reference_path,
        raster_record={"block_raster": "blocks.tif", "block_raster_sha256": "a" * 64},
        grid=grid,
    )

    assert row["threshold_status"] == "unavailable"
    assert row["threshold"] is None
    assert row["unavailable_reason"] == "counts_and_maps_only_support"


def _gate_rows(
    *,
    positive_blocks: int = 10,
    negative_blocks: int = 10,
    auc_2l: float = 0.6,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    common = {
        "site": "goldfield",
        "family": "feature",
        "layer": "al_oh_doublet",
        "positive_bearing_blocks": positive_blocks,
        "negative_bearing_blocks": negative_blocks,
    }
    metrics = [
        {**common, "scale": "L", "auc": 0.8},
        {**common, "scale": "2L", "auc": auc_2l},
    ]
    intervals = [
        {
            **common,
            "scale": "L",
            "metric": "auc",
            "lower": 0.6,
            "upper": 0.9,
            "scheduled_replicates": 10_000,
            "valid_replicates": 10_000,
            "finite_fraction": 1.0,
            "gate_eligible": True,
            "unavailable_reason": None,
        },
        {
            **common,
            "scale": "L",
            "metric": "balanced_accuracy",
            "lower": 0.55,
            "upper": 0.8,
            "scheduled_replicates": 10_000,
            "valid_replicates": 10_000,
            "finite_fraction": 1.0,
            "gate_eligible": True,
            "unavailable_reason": None,
        },
    ]
    return metrics, intervals


@pytest.mark.parametrize(
    ("positive_blocks", "auc_2l", "balanced_lower", "expected"),
    [
        (10, 0.6, 0.55, "pass"),
        (10, 0.6, 0.50, "ranking_discrimination_only"),
        (10, 0.49, 0.55, "fail"),
        (9, 0.6, 0.55, "unavailable"),
    ],
)
def test_external_reference_gate_applies_frozen_decision_rule(
    positive_blocks: int,
    auc_2l: float,
    balanced_lower: float,
    expected: str,
):
    cli = _load_cli_module()
    metrics, intervals = _gate_rows(positive_blocks=positive_blocks, auc_2l=auc_2l)
    intervals[1]["lower"] = balanced_lower

    gate = cli._external_reference_gate(
        metrics,
        intervals,
        protocol_compliant=True,
    )

    assert gate["classification"] == expected
    assert gate["status"] == expected
    assert gate["evaluable"] is (expected != "unavailable")
    assert gate["passed"] is (expected == "pass")
    assert gate["conditions"]["support_eligible_at_L"]["positive_bearing_blocks"] == (
        positive_blocks
    )
    assert gate["conditions"]["auc_lower_95_above_half_at_L"]["threshold"] == 0.5
    assert gate["conditions"]["auc_direction_positive_at_2L"]["observed_auc"] == auc_2l


def test_external_reference_gate_retains_ranking_claim_when_threshold_ci_is_unavailable():
    cli = _load_cli_module()
    metrics, intervals = _gate_rows()
    intervals = [row for row in intervals if row["metric"] != "balanced_accuracy"]

    gate = cli._external_reference_gate(metrics, intervals, protocol_compliant=True)

    assert gate["classification"] == "ranking_discrimination_only"
    assert not gate["conditions"]["balanced_accuracy_lower_95_above_half_at_L"]["available"]


def test_external_reference_gate_rejects_ineligible_auc_interval():
    cli = _load_cli_module()
    metrics, intervals = _gate_rows()
    auc_interval = next(row for row in intervals if row["metric"] == "auc")
    auc_interval.update(
        {
            "valid_replicates": 9_499,
            "finite_fraction": 0.9499,
            "gate_eligible": False,
            "unavailable_reason": "fewer_than_95_percent_finite_replicates",
        }
    )

    gate = cli._external_reference_gate(metrics, intervals, protocol_compliant=True)

    condition = gate["conditions"]["auc_lower_95_above_half_at_L"]
    assert gate["classification"] == "unavailable"
    assert condition["available"] is False
    assert condition["passed"] is False
    assert condition["observed_lower_95"] is None
    assert condition["gate_eligible"] is False
    assert condition["valid_replicates"] == 9_499
    assert condition["reason"] == "fewer_than_95_percent_finite_replicates"


def test_nondefault_protocol_is_bound_into_manifest_and_cannot_be_compliant(tmp_path):
    cli = _load_cli_module()
    grid = cli.RasterGrid(
        shape=(2, 2),
        crs="EPSG:32611",
        transform=from_origin(0.0, 60.0, 30.0, 30.0),
        pixel_size=30.0,
    )
    blocks = complete_blocks(grid.shape, 1)
    raster_record = cli._write_block_raster(
        tmp_path / "block_ids_goldfield_L.tif",
        grid,
        blocks,
        site_id="goldfield",
        scene_id="anchor",
        scale="L",
    )
    result = {
        "site_summary": {
            "site": "goldfield",
            "scene_id": "anchor",
            "shape": list(grid.shape),
            "crs": grid.crs,
            "transform": list(grid.transform)[:6],
            "pixel_size_metres": grid.pixel_size,
            "scores": [{"path": "anchor.tif", "sha256": "0" * 64}],
        },
        "block_handoff": {"L": raster_record, "2L": raster_record},
        "blocks": [],
    }
    parameters = cli._protocol_parameters(
        max_pairs=cli.MAX_PAIRS_PER_LAG,
        bootstrap_replicates=99,
        permutations=cli.PERMUTATION_REPLICATES,
        seed=cli.SEED,
    )

    payload = cli._block_manifest_payload([result], protocol_parameters=parameters)

    assert payload["protocol"]["parameters"] == parameters
    assert payload["protocol"]["protocol_compliant"] is False


def test_summary_contract_links_manifest_and_keeps_public_gate_pending(tmp_path):
    cli = _load_cli_module()
    manifest_path = tmp_path / "block_manifest.json"
    cli._write_json(manifest_path, {"protocol": {"protocol_compliant": True}})
    metrics, intervals = _gate_rows()
    gate = cli._external_reference_gate(metrics, intervals, protocol_compliant=True)
    summary = {
        "block_manifest": cli._block_manifest_link(manifest_path),
        "block_manifest_sha256": cli._sha256(manifest_path),
        "external_reference_gate": gate,
        "combined_public_gate": cli._combined_public_gate(gate),
    }
    summary_path = tmp_path / "summary.json"

    cli._write_json(summary_path, summary)
    raw = summary_path.read_text(encoding="utf-8")
    written = json.loads(raw)

    assert written["block_manifest"]["sha256"] == cli._sha256(manifest_path)
    assert written["block_manifest_sha256"] == cli._sha256(manifest_path)
    assert written["combined_public_gate"]["status"] == "pending_repeatability"
    assert written["combined_public_gate"]["classification"] == "pending_repeatability"
    assert written["combined_public_gate"]["passed"] is False
    assert "NaN" not in raw
