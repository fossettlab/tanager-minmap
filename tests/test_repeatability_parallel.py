"""Exact-equivalence and fail-closed tests for repeatability acceleration."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import tanager_minmap.repeatability as repeatability_module
from tanager_minmap.repeatability import (
    _RESAMPLED_METRIC_COMPONENTS,
    BOOTSTRAP_REPLICATES,
    EXECUTION_SCHEMA_VERSION,
    NULL_REPLICATES,
    PROGRESS_SCHEMA_VERSION,
    SCIENTIFIC_EXECUTION_IDENTITY,
    RepeatabilityPaths,
    _atomic_write_json,
    _binary_block_resampling,
    _bootstrap_choices,
    _boundary_block_resampling,
    _boundary_distance_from_coordinates,
    _distribution_summary,
    _exclusive_output_lock,
    _execution_lock_path,
    _extract_paired_blocks,
    _load_resumable_results,
    _paired_boundary_coordinates,
    _resampling_summary,
    _rockwell_block_resampling,
    _rockwell_metric_values,
    _sha256,
    _task_order,
    _timing_pilot_branch_schedule,
    _validate_cached_raster,
    _validate_expected_input_identity,
    _validate_timing_pilot_admission,
    binary_overlap_metrics,
    paired_block_bootstrap,
    paired_block_null,
    run_repeatability_packet,
)


def _binary_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_ids = np.repeat(np.arange(1, 4)[:, None], 3, axis=1)
    anchor = np.array(
        [
            [1.0, 0.0, np.nan],
            [0.0, 1.0, 1.0],
            [np.nan, 0.0, 0.0],
        ]
    )
    repeat = np.array(
        [
            [1.0, np.nan, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, np.nan],
        ]
    )
    return anchor, repeat, block_ids


@pytest.mark.parametrize(("metric", "attribute"), [("iou", "iou"), ("dice", "dice")])
def test_binary_sufficient_statistics_match_reference_draws_with_moving_missingness(
    metric, attribute
):
    anchor, repeat, block_ids = _binary_fixture()
    accelerated = _binary_block_resampling(
        anchor,
        repeat,
        block_ids,
        n_bootstrap=80,
        n_null=20,
        seed=42,
    )[metric]

    def metric_fn(left, right):
        return getattr(binary_overlap_metrics(left, right), attribute)

    reference_bootstrap = paired_block_bootstrap(
        anchor, repeat, block_ids, n_reps=80, seed=42, metric=metric_fn
    )
    reference_null = paired_block_null(
        anchor, repeat, block_ids, n_reps=20, seed=42, metric=metric_fn
    )

    np.testing.assert_array_equal(
        accelerated["bootstrap"]["samples"], reference_bootstrap["samples"]
    )
    np.testing.assert_array_equal(accelerated["spatial_null"]["samples"], reference_null["samples"])
    assert _distribution_summary(accelerated["bootstrap"], interval=True) == (
        _distribution_summary(reference_bootstrap, interval=True)
    )
    assert _distribution_summary(accelerated["spatial_null"], interval=False) == (
        _distribution_summary(reference_null, interval=False)
    )


def test_binary_sufficient_statistics_preserve_empty_unavailable_governance():
    block_ids = np.repeat(np.arange(1, 4)[:, None], 2, axis=1)
    empty = np.zeros(block_ids.shape, dtype=float)

    accelerated = _binary_block_resampling(
        empty, empty, block_ids, n_bootstrap=40, n_null=10, seed=42
    )
    reference = paired_block_bootstrap(
        empty,
        empty,
        block_ids,
        n_reps=40,
        seed=42,
        metric=lambda left, right: binary_overlap_metrics(left, right).iou,
    )

    np.testing.assert_array_equal(accelerated["iou"]["bootstrap"]["samples"], reference["samples"])
    assert accelerated["iou"]["bootstrap"]["finite_replicates"] == 0
    assert accelerated["iou"]["bootstrap"]["gate_eligible"] is False
    assert accelerated["dice"]["spatial_null"]["finite_replicates"] == 0
    assert accelerated["dice"]["spatial_null"]["gate_eligible"] is False


def test_prevalence_bootstrap_matches_direct_draws_including_zero_anchor_prevalence():
    block_ids = np.repeat(np.arange(1, 4)[:, None], 3, axis=1)
    anchor = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, np.nan],
        ]
    )
    repeat = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )

    accelerated = _binary_block_resampling(
        anchor, repeat, block_ids, n_bootstrap=80, n_null=20, seed=42
    )["prevalence_ratio"]
    direct = paired_block_bootstrap(
        anchor,
        repeat,
        block_ids,
        n_reps=80,
        seed=42,
        metric=lambda left, right: binary_overlap_metrics(left, right).prevalence_ratio,
    )

    np.testing.assert_allclose(
        accelerated["bootstrap"]["samples"], direct["samples"], equal_nan=True
    )
    assert np.isnan(accelerated["bootstrap"]["samples"]).any()
    assert accelerated["spatial_null"] == {
        "status": "not_applicable",
        "reason": "whole_block_null_is_not_defined_for_detection_prevalence_ratio",
    }


def test_threaded_spearman_preserves_exact_seeded_draw_order_and_counts():
    block_ids = np.repeat(np.arange(1, 5)[:, None], 4, axis=1)
    anchor = np.arange(16, dtype=float).reshape(4, 4)
    repeat = np.flip(anchor, axis=1).copy()
    anchor[0, 1] = np.nan
    repeat[2, 3] = np.nan

    serial_bootstrap = paired_block_bootstrap(
        anchor, repeat, block_ids, n_reps=120, seed=42, workers=1
    )
    threaded_bootstrap = paired_block_bootstrap(
        anchor, repeat, block_ids, n_reps=120, seed=42, workers=3
    )
    serial_null = paired_block_null(anchor, repeat, block_ids, n_reps=30, seed=42, workers=1)
    threaded_null = paired_block_null(anchor, repeat, block_ids, n_reps=30, seed=42, workers=3)

    np.testing.assert_array_equal(serial_bootstrap["samples"], threaded_bootstrap["samples"])
    np.testing.assert_array_equal(serial_null["samples"], threaded_null["samples"])
    for key in (
        "scheduled_replicates",
        "finite_replicates",
        "finite_fraction",
        "gate_eligible",
    ):
        assert serial_bootstrap[key] == threaded_bootstrap[key]
        assert serial_null[key] == threaded_null[key]


def test_boundary_bootstrap_preserves_duplicated_and_omitted_block_coordinates():
    block_ids = np.zeros((3, 8), dtype=int)
    block_ids[:, :3] = 1
    block_ids[:, 5:] = 2
    anchor = np.full(block_ids.shape, np.nan)
    repeat = np.full(block_ids.shape, np.nan)
    anchor[:, 0] = 1.0
    anchor[:, 1:3] = 0.0
    repeat[:, 1] = 1.0
    repeat[:, (0, 2)] = 0.0
    anchor[:, 5] = 1.0
    anchor[:, 6:8] = 0.0
    repeat[:, 7] = 1.0
    repeat[:, 5:7] = 0.0

    result = _boundary_block_resampling(
        anchor,
        repeat,
        block_ids,
        xres_m=10.0,
        yres_m=10.0,
        n_bootstrap=80,
        n_null=20,
        seed=42,
    )
    threaded = _boundary_block_resampling(
        anchor,
        repeat,
        block_ids,
        xres_m=10.0,
        yres_m=10.0,
        n_bootstrap=80,
        n_null=20,
        seed=42,
        workers=3,
    )
    choices = _bootstrap_choices(2, n_reps=80, seed=42)
    expected = np.where(np.all(choices == 0, axis=1), 10.0, 20.0)

    np.testing.assert_array_equal(result["bootstrap"]["samples"], expected)
    np.testing.assert_array_equal(result["bootstrap"]["samples"], threaded["bootstrap"]["samples"])
    np.testing.assert_array_equal(
        result["spatial_null"]["samples"], threaded["spatial_null"]["samples"]
    )
    assert np.any(np.all(choices == 0, axis=1))
    assert np.any(np.all(choices == 1, axis=1))


def test_boundary_null_enumerates_all_permutations_and_moves_repeat_missingness():
    block_ids = np.zeros((3, 8), dtype=int)
    block_ids[:, :3] = 1
    block_ids[:, 5:] = 2
    anchor = np.full(block_ids.shape, np.nan)
    repeat = np.full(block_ids.shape, np.nan)
    anchor[:, (0, 5)] = 1.0
    anchor[:, 1:3] = 0.0
    anchor[:, 6:8] = 0.0
    repeat[:, 1] = 1.0
    repeat[:, (0, 2)] = 0.0
    repeat[:, 7] = 1.0
    repeat[:, 5:7] = 0.0
    repeat[0, 1] = np.nan
    repeat[2, 7] = np.nan

    observed = _boundary_block_resampling(
        anchor,
        repeat,
        block_ids,
        xres_m=10.0,
        yres_m=10.0,
        n_bootstrap=20,
        n_null=20,
        seed=42,
    )["spatial_null"]
    blocks = _extract_paired_blocks(anchor, repeat, block_ids)
    expected = []
    for ordering in ((0, 1), (1, 0)):
        coordinates = [
            _paired_boundary_coordinates(
                blocks[destination].anchor,
                blocks[source].repeat,
                row_start=blocks[destination].row_start,
                column_start=blocks[destination].column_start,
                xres_m=10.0,
                yres_m=10.0,
            )
            for destination, source in enumerate(ordering)
        ]
        left = np.concatenate([item[0] for item in coordinates if item[0].size])
        right = np.concatenate([item[1] for item in coordinates if item[1].size])
        expected.append(_boundary_distance_from_coordinates(left, right))

    np.testing.assert_allclose(observed["samples"], expected, equal_nan=True)
    assert observed["scheduled_replicates"] == 2
    assert observed["enumerated_all_unique"] is True


def test_rockwell_bootstrap_and_null_match_direct_fixed_threshold_resampling():
    block_ids = np.repeat(np.arange(1, 4)[:, None], 4, axis=1)
    scores = np.array(
        [
            [0.90, 0.80, 0.40, 0.10],
            [np.nan, 0.60, 0.45, 0.20],
            [0.95, 0.51, 0.49, np.nan],
        ]
    )
    reference = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), (3, 1))
    threshold = 0.50

    observed = _rockwell_block_resampling(
        scores,
        reference,
        block_ids,
        threshold=threshold,
        n_bootstrap=80,
        n_null=20,
        seed=42,
        workers=1,
    )
    threaded = _rockwell_block_resampling(
        scores,
        reference,
        block_ids,
        threshold=threshold,
        n_bootstrap=80,
        n_null=20,
        seed=42,
        workers=3,
    )

    for metric in ("auc", "balanced_accuracy", "macro_f1"):
        direct_bootstrap = paired_block_bootstrap(
            scores,
            reference,
            block_ids,
            n_reps=80,
            seed=42,
            metric=lambda sample_scores, fixed_reference, name=metric: _rockwell_metric_values(
                sample_scores, fixed_reference, threshold=threshold
            )[name],
        )
        direct_null = paired_block_null(
            reference,
            scores,
            block_ids,
            n_reps=20,
            seed=42,
            metric=lambda fixed_reference, permuted_scores, name=metric: _rockwell_metric_values(
                permuted_scores, fixed_reference, threshold=threshold
            )[name],
        )
        np.testing.assert_array_equal(
            observed["metrics"][metric]["bootstrap"]["samples"],
            direct_bootstrap["samples"],
        )
        np.testing.assert_array_equal(
            observed["metrics"][metric]["spatial_null"]["samples"],
            direct_null["samples"],
        )
        np.testing.assert_array_equal(
            observed["metrics"][metric]["bootstrap"]["samples"],
            threaded["metrics"][metric]["bootstrap"]["samples"],
        )
        np.testing.assert_array_equal(
            observed["metrics"][metric]["spatial_null"]["samples"],
            threaded["metrics"][metric]["spatial_null"]["samples"],
        )


def test_rockwell_threshold_unavailable_schema_keeps_auc_resampling_explicit():
    block_ids = np.repeat(np.arange(1, 3)[:, None], 4, axis=1)
    scores = np.array([[0.9, 0.8, 0.2, 0.1], [0.8, 0.7, 0.3, 0.2]])
    reference = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), (2, 1))

    result = _rockwell_block_resampling(
        scores,
        reference,
        block_ids,
        threshold=None,
        n_bootstrap=20,
        n_null=10,
    )

    assert result["metrics"]["auc"]["bootstrap"]["scheduled_replicates"] == 20
    for metric in ("balanced_accuracy", "macro_f1"):
        assert result["metrics"][metric]["bootstrap"] == {
            "status": "unavailable",
            "reason": "transferred_threshold_unavailable",
        }
        assert result["metrics"][metric]["spatial_null"] == {
            "status": "unavailable",
            "reason": "transferred_threshold_unavailable",
        }


def test_accelerated_summary_matches_worker_independent_gates_and_unavailable_cases():
    anchor, repeat, block_ids = _binary_fixture()
    scores_left = np.arange(anchor.size, dtype=float).reshape(anchor.shape)
    scores_right = np.flip(scores_left, axis=1)
    scores_left[np.isnan(anchor)] = np.nan
    scores_right[np.isnan(repeat)] = np.nan

    serial = _resampling_summary(
        scores_left,
        scores_right,
        anchor,
        repeat,
        anchor,
        repeat,
        block_ids,
        xres_m=10.0,
        yres_m=10.0,
        rockwell_reference=anchor,
        transferred_threshold=4.0,
        n_bootstrap=80,
        n_null=20,
        workers=1,
    )
    threaded = _resampling_summary(
        scores_left,
        scores_right,
        anchor,
        repeat,
        anchor,
        repeat,
        block_ids,
        xres_m=10.0,
        yres_m=10.0,
        rockwell_reference=anchor,
        transferred_threshold=4.0,
        n_bootstrap=80,
        n_null=20,
        workers=3,
    )

    assert serial == threaded
    assert tuple(serial["metrics"]) == _RESAMPLED_METRIC_COMPONENTS
    for metric, components in serial["metrics"].items():
        assert set(components) == {"bootstrap", "spatial_null"}
        assert components["bootstrap"]["scheduled_replicates"] == 80
        if metric.endswith("prevalence_ratio"):
            assert components["spatial_null"]["status"] == "not_applicable"
            assert components["spatial_null"]["scheduled_replicates"] is None
        else:
            assert components["spatial_null"]["scheduled_replicates"] == math.factorial(3)
    assert "lower_5" in serial["metrics"]["transferred_boundary_distance_m"]["spatial_null"]
    unavailable = _resampling_summary(
        scores_left[:1],
        scores_right[:1],
        anchor[:1],
        repeat[:1],
        anchor[:1],
        repeat[:1],
        np.ones((1, 3), dtype=int),
        xres_m=10.0,
        yres_m=10.0,
        n_bootstrap=20,
        n_null=10,
        workers=2,
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["n_complete_paired_overlap_blocks"] == 1
    assert tuple(unavailable["metrics"]) == _RESAMPLED_METRIC_COMPONENTS
    for metric, components in unavailable["metrics"].items():
        assert components["bootstrap"]["status"] == "unavailable"
        assert components["bootstrap"]["scheduled_replicates"] is None
        expected_null_status = (
            "not_applicable" if metric.endswith("prevalence_ratio") else "unavailable"
        )
        assert components["spatial_null"]["status"] == expected_null_status


def test_thread_worker_failure_propagates_without_returning_partial_samples():
    anchor, repeat, block_ids = _binary_fixture()

    def fail_metric(_left, _right):
        raise RuntimeError("synthetic worker failure")

    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        paired_block_bootstrap(
            anchor,
            repeat,
            block_ids,
            n_reps=20,
            metric=fail_metric,
            workers=2,
        )


def test_packet_failure_leaves_progress_failed_and_no_final_manifest(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    paths = RepeatabilityPaths(
        raw_dir=tmp_path / "raw",
        speclib_dir=tmp_path / "speclib",
        validation_dir=tmp_path / "validation",
        output_dir=output_dir,
        reference_dir=tmp_path / "reference",
    )
    handoffs = {
        site_id: SimpleNamespace(
            raster_path=tmp_path / f"{site_id}.tif",
            raster_sha256="a" * 64,
            shape=(2, 2),
            crs="EPSG:32611",
            transform=from_origin(0.0, 60.0, 30.0, 30.0),
            complete_block_ids=(1, 2),
        )
        for site_id in ("bingham", "goldfield")
    }
    thresholds = {
        site_id: {"feature:al_oh_doublet": SimpleNamespace(threshold=0.75)}
        for site_id in ("bingham", "goldfield")
    }
    provenance = {
        "block_manifest": "block_manifest.json",
        "block_manifest_sha256": "b" * 64,
        "transfer_thresholds": "transfer_thresholds.csv",
        "transfer_thresholds_sha256": "c" * 64,
        "spatial_summary": "summary.json",
        "spatial_summary_sha256": "d" * 64,
        "spatial_prereg_sha256": "e" * 64,
        "external_reference_gate": {"passed": False, "evaluable": False},
    }
    raw_scenes = [
        {"site_id": site_id, "scene_id": scene_id, "sha256": "f" * 64}
        for site_id in ("bingham", "goldfield")
        for scene_id in repeatability_module.site_scene_order(site_id)
    ]
    inventory = {
        "raw_scenes": raw_scenes,
        "reference_rasters": [],
        "spectral_library": {"tree_sha256": "1" * 64},
        "code_bytes": [{"sha256": "2" * 64}],
    }
    monkeypatch.setattr(
        repeatability_module,
        "_load_repeatability_handoff",
        lambda *_args, **_kwargs: (handoffs, thresholds, provenance),
    )
    monkeypatch.setattr(repeatability_module, "_repo_root", lambda _paths: tmp_path)
    monkeypatch.setattr(
        repeatability_module, "_execution_source_inventory", lambda *_args: inventory
    )
    monkeypatch.setattr(
        repeatability_module,
        "_load_scene_products",
        lambda *_args, **_kwargs: (object(), (), {}),
    )
    monkeypatch.setattr(
        repeatability_module, "_load_anchor_reference", lambda *_args: (None, "missing")
    )

    def fail_pair(pair, *_args, progress_callback, **_kwargs):
        progress_callback(pair, "feature:al_oh_doublet", "running", None, None)
        raise RuntimeError("synthetic pair failure")

    monkeypatch.setattr(repeatability_module, "_pair_result", fail_pair)

    with pytest.raises(RuntimeError, match="synthetic pair failure"):
        run_repeatability_packet(
            paths,
            block_manifest=tmp_path / "block_manifest.json",
            workers=2,
            timing_pilot=True,
        )

    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["run_status"] == "failed"
    assert progress["accepted_final_manifest"] is False
    assert progress["tasks"][0]["status"] == "failed"
    assert not (output_dir / "manifest.json").exists()


def test_task_order_is_frozen_independent_of_worker_count():
    thresholds = {
        "bingham": {"mtmf:z": object(), "feature:a": object()},
        "goldfield": {"feature:b": object(), "feature:a": object()},
    }

    tasks = _task_order(thresholds)

    assert [task["index"] for task in tasks] == list(range(len(tasks)))
    assert [task["layer"] for task in tasks[:2]] == ["feature:a", "mtmf:z"]
    assert tasks[0]["comparison_role"] == "primary"
    assert tasks[0]["site_id"] == "bingham"


def test_cache_raster_hash_and_grid_are_both_required(tmp_path):
    path = tmp_path / "score.tif"
    transform = from_origin(100.0, 200.0, 30.0, 30.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=transform,
    ) as dataset:
        dataset.write(np.arange(4, dtype=np.float32).reshape(2, 2), 1)
    record = {"sha256": _sha256(path)}
    grid = {"shape": [2, 2], "crs": "EPSG:32611", "transform": list(transform)[:6]}

    _validate_cached_raster(path, record, grid)
    with pytest.raises(ValueError, match="SHA mismatch"):
        _validate_cached_raster(path, {"sha256": "0" * 64}, grid)
    with pytest.raises(ValueError, match="grid mismatch"):
        _validate_cached_raster(path, record, {**grid, "shape": [1, 4]})


def test_resume_validates_execution_bytes_and_completed_result_hashes(tmp_path):
    output_dir = tmp_path / "run"
    execution = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
        "source_inventory": {"code_bytes": [{"sha256": "a" * 64}]},
        "member_order": {"tasks": ["task-0"]},
        "seed": 42,
    }
    tasks = [{"index": 0, "task_id": "task-0", "layer": "feature:a"}]
    _, progress_path, progress, completed = _load_resumable_results(
        output_dir, execution, tasks, resume=False
    )
    assert completed == {}
    assert progress["schema_version"] == PROGRESS_SCHEMA_VERSION
    assert not (output_dir / "manifest.json").exists()

    result_path = output_dir / "task_results" / "00000-attempt-001.json"
    _atomic_write_json(result_path, {"endpoint": {"gate_eligible": False}})
    progress["tasks"][0].update(
        {
            "status": "completed",
            "attempts": 1,
            "result_path": str(result_path.relative_to(output_dir)),
            "result_sha256": _sha256(result_path),
        }
    )
    _atomic_write_json(progress_path, progress)

    _, _, _, completed = _load_resumable_results(output_dir, execution, tasks, resume=True)
    assert completed == {"task-0": {"endpoint": {"gate_eligible": False}}}
    with pytest.raises(ValueError, match="execution manifest mismatch"):
        _load_resumable_results(
            output_dir,
            {**execution, "source_inventory": {"code_bytes": [{"sha256": "b" * 64}]}},
            tasks,
            resume=True,
        )

    result_path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="result SHA mismatch"):
        _load_resumable_results(output_dir, execution, tasks, resume=True)
    assert not (output_dir / "manifest.json").exists()


def test_old_scientific_execution_identity_cannot_resume(tmp_path):
    execution = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "scientific_execution_identity": "partial-metric-contract-v1",
        "source_inventory": {"code_bytes": [{"sha256": "a" * 64}]},
        "member_order": {"tasks": ["task-0"]},
    }

    with pytest.raises(ValueError, match="scientific execution identity mismatch"):
        _load_resumable_results(
            tmp_path / "old-run",
            execution,
            [{"index": 0, "task_id": "task-0", "layer": "feature:a"}],
            resume=True,
        )


def test_timing_pilot_schedule_covers_every_branch_without_endpoint_values():
    metrics = {}
    for metric in _RESAMPLED_METRIC_COMPONENTS:
        spatial_null = (
            {"status": "not_applicable", "scheduled_replicates": None}
            if metric.endswith("prevalence_ratio")
            else {"status": "available", "scheduled_replicates": NULL_REPLICATES}
        )
        metrics[metric] = {
            "bootstrap": {
                "status": "available",
                "scheduled_replicates": BOOTSTRAP_REPLICATES,
            },
            "spatial_null": spatial_null,
        }
    pair_result = {
        "layers": {
            "feature:a": {
                "uncertainty_and_nulls": {
                    "n_complete_paired_overlap_blocks": 10,
                    "metrics": metrics,
                }
            }
        }
    }

    schedule = _timing_pilot_branch_schedule(pair_result, "feature:a")

    assert schedule["contains_endpoint_values"] is False
    assert tuple(schedule["components"]) == _RESAMPLED_METRIC_COMPONENTS
    assert not any(key in json.dumps(schedule) for key in ("lower_95", "upper_95", "lower_5"))


def _write_expected_input_fixture(tmp_path: Path) -> tuple[RepeatabilityPaths, Path, Path, Path]:
    root = tmp_path / "repo"
    raw_dir = root / "data" / "raw"
    library_dir = root / "data" / "speclib" / "ASCIIdata_splib07a"
    docs_dir = root / "docs"
    for directory in (raw_dir, library_dir, docs_dir):
        directory.mkdir(parents=True)

    inputs = []
    first_scene_path: Path | None = None
    for site_id in sorted(repeatability_module.SITES):
        site = repeatability_module.SITES[site_id]
        for index, scene_id in enumerate(site.scene_ids, start=1):
            filename = f"{scene_id}_{repeatability_module.TANAGER_SR_ASSET}.h5"
            path = raw_dir / filename
            path.write_bytes(f"synthetic-{scene_id}".encode())
            first_scene_path = first_scene_path or path
            inputs.append(
                {
                    "id": f"tanager-{site_id}-{index}",
                    "logical_path": f"data/raw/{filename}",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

    library_members = {
        "ChapterM_Minerals/mineral.txt": b"synthetic-mineral\n",
        "ASD/ChapterV_Vegetation/wavelengths.txt": b"synthetic-wavelengths\n",
    }
    archive_path = library_dir.parent / "ASCIIdata_splib07a.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for relative, content in library_members.items():
            archive.writestr(f"ASCIIdata_splib07a/{relative}", content)
            extracted = library_dir / relative
            extracted.parent.mkdir(parents=True, exist_ok=True)
            extracted.write_bytes(content)
    inputs.append(
        {
            "id": "usgs-splib07a-archive",
            "logical_path": "data/speclib/ASCIIdata_splib07a.zip",
            "size_bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        }
    )
    manifest_path = docs_dir / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "hash_algorithm": "sha256",
                "inputs": inputs,
            }
        ),
        encoding="utf-8",
    )
    paths = RepeatabilityPaths(
        raw_dir=raw_dir,
        speclib_dir=library_dir,
        validation_dir=root / "data" / "intermediate" / "validation",
        output_dir=root / "data" / "processed" / "repeatability-test",
        reference_dir=root / "data" / "reference",
    )
    assert first_scene_path is not None
    return paths, manifest_path, first_scene_path, library_dir / "ChapterM_Minerals/mineral.txt"


def test_expected_input_admission_accepts_exact_frozen_scene_and_library_closures(tmp_path):
    paths, manifest_path, _, _ = _write_expected_input_fixture(tmp_path)

    admitted = _validate_expected_input_identity(paths, manifest_path.parents[1], manifest_path)

    assert len(admitted["raw_scenes"]) == 7
    assert admitted["spectral_library"]["file_count"] == 2
    assert (
        admitted["spectral_library"]["tree_sha256"]
        == (admitted["spectral_library"]["expected_tree_sha256"])
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("changed_scene", "raw scene .* (size|SHA-256) mismatch"),
        ("missing_scene", "raw-scene closure is missing"),
        ("extra_scene", "raw-scene closure has extra"),
        ("changed_library", "spectral-library member .* SHA-256 mismatch"),
        ("missing_library", "spectral-library closure is missing"),
        ("extra_library", "spectral-library closure has extra"),
    ],
)
def test_expected_input_admission_fails_closed_on_changed_missing_or_extra_closure(
    tmp_path, mutation, match
):
    paths, manifest_path, scene_path, library_path = _write_expected_input_fixture(tmp_path)
    if mutation == "changed_scene":
        scene_path.write_bytes(b"changed-scene-bytes")
    elif mutation == "missing_scene":
        scene_path.unlink()
    elif mutation == "extra_scene":
        (paths.raw_dir / f"extra_{repeatability_module.TANAGER_SR_ASSET}.h5").write_bytes(b"extra")
    elif mutation == "changed_library":
        original = library_path.read_bytes()
        library_path.write_bytes(b"X" + original[1:])
    elif mutation == "missing_library":
        library_path.unlink()
    else:
        (paths.speclib_dir / "extra.txt").write_bytes(b"extra")

    with pytest.raises((FileNotFoundError, ValueError), match=match):
        _validate_expected_input_identity(paths, manifest_path.parents[1], manifest_path)


def _write_valid_timing_admission_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, object], Path]:
    output_dir = tmp_path / "timing-run"
    pilot_task = {"task_id": "primary:synthetic:anchor:repeat:feature:a"}
    other_task = {"task_id": "primary:synthetic:anchor:repeat:feature:b"}
    execution = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
        "compute_controls": {"workers": 4},
        "member_order": {"tasks": [pilot_task, other_task]},
    }
    execution_path = output_dir / "execution_manifest.json"
    _atomic_write_json(execution_path, execution)
    n_blocks = 3
    components = {}
    for metric in _RESAMPLED_METRIC_COMPONENTS:
        spatial_null = (
            {"status": "not_applicable"}
            if metric.endswith("prevalence_ratio")
            else {"status": "scheduled", "scheduled_replicates": math.factorial(n_blocks)}
        )
        components[metric] = {
            "bootstrap": {"scheduled_replicates": BOOTSTRAP_REPLICATES},
            "spatial_null": spatial_null,
        }
    timing = {
        "schema_version": repeatability_module.TIMING_PILOT_SCHEMA_VERSION,
        "mode": "timing",
        "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
        "accepted_scientific_result": False,
        "contains_endpoint_values": False,
        "execution_manifest": str(execution_path.resolve()),
        "execution_manifest_sha256": _sha256(execution_path),
        "task_id": pilot_task["task_id"],
        "workers": 4,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "null_replicates_maximum": NULL_REPLICATES,
        "resampling_branch_schedule": {
            "contains_endpoint_values": False,
            "n_complete_paired_overlap_blocks": n_blocks,
            "components": components,
        },
        "elapsed_seconds": 1.25,
        "result_sha256": "a" * 64,
    }
    timing_path = output_dir / "timing_pilot.json"
    _atomic_write_json(timing_path, timing)
    _atomic_write_json(
        output_dir / "progress.json",
        {
            "schema_version": repeatability_module.PROGRESS_SCHEMA_VERSION,
            "execution_manifest_sha256": repeatability_module._stable_json_sha256(execution),
            "run_status": "timing_pilot_complete",
            "accepted_final_manifest": False,
            "tasks": [
                {
                    "task_id": pilot_task["task_id"],
                    "status": "completed",
                    "attempts": 1,
                    "elapsed_seconds": timing["elapsed_seconds"],
                    "result_path": None,
                    "result_sha256": timing["result_sha256"],
                },
                {
                    "task_id": other_task["task_id"],
                    "status": "pending",
                    "attempts": 0,
                    "elapsed_seconds": None,
                    "result_path": None,
                    "result_sha256": None,
                },
            ],
        },
    )
    return output_dir, execution, pilot_task, timing_path


def test_full_admits_exact_reviewed_timing_artifact_success_path(tmp_path):
    output_dir, execution, pilot_task, timing_path = _write_valid_timing_admission_fixture(tmp_path)

    admitted = _validate_timing_pilot_admission(
        output_dir,
        expected_sha256=_sha256(timing_path),
        execution_manifest=execution,
        pilot_task=pilot_task,
    )

    assert admitted == timing_path


def test_full_rejects_timing_artifact_digest_mismatch_before_schema_admission(tmp_path):
    output_dir, execution, pilot_task, _ = _write_valid_timing_admission_fixture(tmp_path)

    with pytest.raises(ValueError, match="timing-pilot SHA-256 mismatch"):
        _validate_timing_pilot_admission(
            output_dir,
            expected_sha256="0" * 64,
            execution_manifest=execution,
            pilot_task=pilot_task,
        )


def test_full_rejects_timing_result_hash_not_proven_by_progress_ledger(tmp_path):
    output_dir, execution, pilot_task, timing_path = _write_valid_timing_admission_fixture(tmp_path)
    progress_path = output_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["tasks"][0]["result_sha256"] = "b" * 64
    _atomic_write_json(progress_path, progress)

    with pytest.raises(ValueError, match="result SHA-256 provenance mismatch"):
        _validate_timing_pilot_admission(
            output_dir,
            expected_sha256=_sha256(timing_path),
            execution_manifest=execution,
            pilot_task=pilot_task,
        )


def test_full_rejects_missing_timing_progress_ledger(tmp_path):
    output_dir, execution, pilot_task, timing_path = _write_valid_timing_admission_fixture(tmp_path)
    (output_dir / "progress.json").unlink()

    with pytest.raises(FileNotFoundError, match="progress ledger is missing"):
        _validate_timing_pilot_admission(
            output_dir,
            expected_sha256=_sha256(timing_path),
            execution_manifest=execution,
            pilot_task=pilot_task,
        )


def test_full_rejects_second_completed_timing_task(tmp_path):
    output_dir, execution, pilot_task, timing_path = _write_valid_timing_admission_fixture(tmp_path)
    result_path = output_dir / "task_results" / "other.json"
    _atomic_write_json(result_path, {"task_id": "other"})
    progress_path = output_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["tasks"][1].update(
        {
            "status": "completed",
            "attempts": 1,
            "elapsed_seconds": 0.5,
            "result_path": str(result_path.relative_to(output_dir)),
            "result_sha256": _sha256(result_path),
        }
    )
    _atomic_write_json(progress_path, progress)

    with pytest.raises(ValueError, match="feature:b status mismatch"):
        _validate_timing_pilot_admission(
            output_dir,
            expected_sha256=_sha256(timing_path),
            execution_manifest=execution,
            pilot_task=pilot_task,
        )


def test_full_rejects_boolean_timing_progress_elapsed_seconds(tmp_path):
    output_dir, execution, pilot_task, timing_path = _write_valid_timing_admission_fixture(tmp_path)
    progress_path = output_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["tasks"][0]["elapsed_seconds"] = True
    _atomic_write_json(progress_path, progress)

    with pytest.raises(ValueError, match="finite, non-negative, and non-boolean"):
        _validate_timing_pilot_admission(
            output_dir,
            expected_sha256=_sha256(timing_path),
            execution_manifest=execution,
            pilot_task=pilot_task,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda timing: timing.update({"schema_version": "stale"}), "schema mismatch"),
        (
            lambda timing: timing.update({"scientific_execution_identity": "wrong"}),
            "scientific execution identity mismatch",
        ),
        (lambda timing: timing.update({"mode": "full"}), "mode mismatch"),
        (
            lambda timing: timing["resampling_branch_schedule"]["components"]["spearman"][
                "bootstrap"
            ].update({"scheduled_replicates": 1}),
            "spearman bootstrap schedule mismatch",
        ),
    ],
)
def test_full_rejects_invalid_timing_schema_identity_mode_or_schedule(tmp_path, mutation, match):
    output_dir, execution, pilot_task, timing_path = _write_valid_timing_admission_fixture(tmp_path)
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    mutation(timing)
    _atomic_write_json(timing_path, timing)

    with pytest.raises(ValueError, match=match):
        _validate_timing_pilot_admission(
            output_dir,
            expected_sha256=_sha256(timing_path),
            execution_manifest=execution,
            pilot_task=pilot_task,
        )


def test_timing_to_full_resume_resets_pilot_row_for_recomputation(tmp_path):
    output_dir = tmp_path / "run"
    execution = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
    }
    tasks = [{"index": 0, "task_id": "pilot-task", "layer": "feature:a"}]
    _, progress_path, progress, _ = _load_resumable_results(
        output_dir, execution, tasks, resume=False
    )
    progress["tasks"][0].update(
        {
            "status": "completed",
            "attempts": 1,
            "elapsed_seconds": 1.0,
            "result_path": None,
            "result_sha256": "a" * 64,
        }
    )
    progress["run_status"] = "timing_pilot_complete"
    _atomic_write_json(progress_path, progress)

    _, _, resumed, completed = _load_resumable_results(output_dir, execution, tasks, resume=True)
    recomputed_task_ids = [
        row["task_id"] for row in resumed["tasks"] if row["task_id"] not in completed
    ]

    assert completed == {}
    assert resumed["tasks"][0]["status"] == "pending"
    assert resumed["tasks"][0]["result_sha256"] is None
    assert recomputed_task_ids == ["pilot-task"]


def test_synthetic_timing_then_full_recomputes_pilot_task_end_to_end(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "processed" / "repeatability"
    paths = RepeatabilityPaths(
        raw_dir=tmp_path / "raw",
        speclib_dir=tmp_path / "speclib",
        validation_dir=tmp_path / "validation",
        output_dir=output_dir,
        reference_dir=tmp_path / "reference",
    )
    handoffs = {
        site_id: SimpleNamespace(
            raster_path=tmp_path / f"{site_id}.tif",
            raster_sha256="a" * 64,
            shape=(2, 2),
            crs="EPSG:32611",
            transform=from_origin(0.0, 60.0, 30.0, 30.0),
            complete_block_ids=(1, 2, 3),
        )
        for site_id in ("bingham", "goldfield")
    }
    thresholds = {
        site_id: {"feature:al_oh_doublet": SimpleNamespace(threshold=0.75)}
        for site_id in ("bingham", "goldfield")
    }
    provenance = {
        "block_manifest": "block_manifest.json",
        "block_manifest_sha256": "b" * 64,
        "transfer_thresholds": "transfer_thresholds.csv",
        "transfer_thresholds_sha256": "c" * 64,
        "spatial_summary": "summary.json",
        "spatial_summary_sha256": "d" * 64,
        "spatial_prereg_sha256": "e" * 64,
        "external_reference_gate": {"passed": False, "evaluable": False},
    }
    inventory = {
        "expected_input_admission": {
            "input_manifest": str(tmp_path / "input_manifest.json"),
            "input_manifest_sha256": "f" * 64,
            "status": "admitted_before_computation",
        },
        "raw_scenes": [
            {"site_id": site_id, "scene_id": scene_id, "sha256": "1" * 64}
            for site_id in ("bingham", "goldfield")
            for scene_id in repeatability_module.site_scene_order(site_id)
        ],
        "reference_rasters": [],
        "spectral_library": {"tree_sha256": "2" * 64},
        "code_bytes": [{"sha256": "3" * 64}],
    }
    monkeypatch.setattr(
        repeatability_module,
        "_load_repeatability_handoff",
        lambda *_args, **_kwargs: (handoffs, thresholds, provenance),
    )
    monkeypatch.setattr(repeatability_module, "_repo_root", lambda _paths: tmp_path)
    monkeypatch.setattr(
        repeatability_module, "_execution_source_inventory", lambda *_args: inventory
    )

    def synthetic_scene(site_id, scene_id, *_args, **_kwargs):
        product = SimpleNamespace(
            site_id=site_id,
            scene_id=scene_id,
            feature_definitions=(),
            endmember_samples={},
            scores={"feature:al_oh_doublet": None},
        )
        return product, (), {}

    monkeypatch.setattr(repeatability_module, "_load_scene_products", synthetic_scene)
    monkeypatch.setattr(
        repeatability_module, "_load_anchor_reference", lambda *_args: (None, "missing")
    )
    monkeypatch.setattr(
        repeatability_module, "classify_goldfield_repeatability", lambda _gates: "synthetic"
    )
    monkeypatch.setattr(
        repeatability_module,
        "combined_public_gate",
        lambda _external, _classification: {"status": "synthetic"},
    )
    resource_admission_path = tmp_path / "resource_admission.json"
    resource_admission_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        repeatability_module,
        "validate_resource_admission",
        lambda **_kwargs: resource_admission_path,
    )
    null_replicates = math.factorial(3)
    metrics = {
        metric: {
            "bootstrap": {"scheduled_replicates": BOOTSTRAP_REPLICATES},
            "spatial_null": (
                {"status": "not_applicable"}
                if metric.endswith("prevalence_ratio")
                else {"status": "available", "scheduled_replicates": null_replicates}
            ),
        }
        for metric in _RESAMPLED_METRIC_COMPONENTS
    }
    timing_calls: list[str] = []
    full_calls: list[str] = []

    def synthetic_pair_result(
        pair,
        *_args,
        selected_layer_keys,
        precomputed_layers,
        progress_callback,
        **_kwargs,
    ):
        layer = "feature:al_oh_doublet"
        task_id = repeatability_module._pair_task_id(pair, layer)
        if layer not in precomputed_layers:
            calls = timing_calls if selected_layer_keys is not None else full_calls
            calls.append(task_id)
            progress_callback(pair, layer, "running", None, None)
            progress_callback(pair, layer, "completed", 0.01, {"task_id": task_id})
        return {
            "site_id": pair.site_id,
            "comparison_role": pair.role,
            "layers": {
                layer: {
                    "uncertainty_and_nulls": {
                        "n_complete_paired_overlap_blocks": 3,
                        "metrics": metrics,
                    },
                    "goldfield_al_oh_doublet_pair_gate": {"status": "synthetic"},
                }
            },
        }

    monkeypatch.setattr(repeatability_module, "_pair_result", synthetic_pair_result)

    timing_path = run_repeatability_packet(
        paths,
        block_manifest=tmp_path / "block_manifest.json",
        workers=4,
        timing_pilot=True,
    )
    timing_task_id = timing_calls[0]
    full_path = run_repeatability_packet(
        paths,
        block_manifest=tmp_path / "block_manifest.json",
        workers=4,
        resume=True,
        expected_timing_pilot_sha256=_sha256(timing_path),
        expected_resource_admission_sha256="b" * 64,
        resource_admission_path=resource_admission_path,
    )
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    timing_row = next(row for row in progress["tasks"] if row["task_id"] == timing_task_id)

    assert timing_task_id in full_calls
    assert len(full_calls) == len(repeatability_module.PRIMARY_PAIRS) + len(
        repeatability_module.SECONDARY_PAIRS
    )
    assert timing_row["attempts"] == 2
    assert timing_row["result_path"] is not None
    assert full_path == output_dir / "manifest.json"


def test_output_identity_lock_rejects_contention_and_releases_owned_lock(tmp_path):
    output_dir = tmp_path / "processed" / "repeatability"
    with _exclusive_output_lock(output_dir, mode="timing") as lock_path:
        with pytest.raises(RuntimeError, match="already held or stale"):
            with _exclusive_output_lock(output_dir, mode="full"):
                pass
        assert lock_path == _execution_lock_path(output_dir)

    assert not _execution_lock_path(output_dir).exists()


def test_output_identity_lock_does_not_remove_replaced_owner_record(tmp_path):
    output_dir = tmp_path / "processed" / "repeatability"
    lock_path = _execution_lock_path(output_dir)

    with pytest.raises(ValueError, match="lock ownership mismatch"):
        with _exclusive_output_lock(output_dir, mode="timing") as acquired:
            owner_path = acquired / "owner.json"
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            owner["owner_id"] = "replaced-owner"
            owner_path.write_text(json.dumps(owner), encoding="utf-8")

    assert lock_path.is_dir()
    assert (lock_path / "owner.json").is_file()
