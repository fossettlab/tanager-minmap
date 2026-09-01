"""Synthetic tests for the preregistered scene-level sensor ablation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from tanager_spec.srf import gaussian_srf

from tanager_rocks.sensor_ablation import (
    BOOTSTRAP_REPLICATES,
    MIN_COVERAGE,
    NEGATIVE_CLASS,
    PERMUTATION_REPLICATES,
    POSITIVE_CLASS,
    RIDGE,
    SEED,
    apply_robust_margin,
    benjamini_hochberg,
    benjamini_hochberg_by_family,
    block_designs_from_frame,
    compute_sensor_mtmf,
    confirmatory_bh_by_family,
    evaluate_sensor_pair,
    fit_robust_margin,
    governed_metric_summary,
    inference_status,
    paired_block_bootstrap,
    paired_sensor_auc_randomization,
    percentile_interval,
    support_governance,
)
from tanager_rocks.speclib import Endmember


@pytest.fixture(scope="module")
def scene_ablation_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_scene_ablation.py"
    spec = importlib.util.spec_from_file_location("run_scene_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _m2_manifest_payload(scene_ablation_runner, directory: Path) -> dict:
    raster_contents = (
        ("block_ids_goldfield_L.tif", b"L"),
        ("block_ids_goldfield_2L.tif", b"2L"),
    )
    for name, contents in raster_contents:
        (directory / name).write_bytes(contents)
    return {
        "protocol": {
            "sha256": scene_ablation_runner.sha256_file(scene_ablation_runner.M2_PREREGISTRATION)
        },
        "sites": {
            "goldfield": {
                "scene_id": scene_ablation_runner.SCENE_ID,
                "grid": {
                    "shape": [2, 3],
                    "crs": "EPSG:32611",
                    "transform": [30.0, 0.0, 1.0, 0.0, -30.0, 2.0],
                },
                "scales": {
                    scale: {
                        "block_raster": name,
                        "block_raster_sha256": scene_ablation_runner.sha256_file(directory / name),
                    }
                    for scale, name in (
                        ("L", "block_ids_goldfield_L.tif"),
                        ("2L", "block_ids_goldfield_2L.tif"),
                    )
                },
            }
        },
    }


def test_m3_manifest_validation_accepts_exact_frozen_handoff(scene_ablation_runner, tmp_path: Path):
    path = tmp_path / "block_manifest.json"
    path.write_text(
        json.dumps(_m2_manifest_payload(scene_ablation_runner, tmp_path)), encoding="utf-8"
    )

    scene_ablation_runner._validate_m2_manifest(
        path,
        site="goldfield",
        scene_id=scene_ablation_runner.SCENE_ID,
        shape=(2, 3),
        crs="EPSG:32611",
        transform=(30.0, 0.0, 1.0, 0.0, -30.0, 2.0),
    )


def test_m3_manifest_validation_rejects_stale_protocol_and_raster(
    scene_ablation_runner, tmp_path: Path
):
    payload = _m2_manifest_payload(scene_ablation_runner, tmp_path)
    path = tmp_path / "block_manifest.json"
    payload["protocol"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol hash"):
        scene_ablation_runner._validate_m2_manifest(
            path,
            site="goldfield",
            scene_id=scene_ablation_runner.SCENE_ID,
            shape=(2, 3),
            crs="EPSG:32611",
            transform=(30.0, 0.0, 1.0, 0.0, -30.0, 2.0),
        )

    payload = _m2_manifest_payload(scene_ablation_runner, tmp_path)
    path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "block_ids_goldfield_L.tif").write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale hash"):
        scene_ablation_runner._validate_m2_manifest(
            path,
            site="goldfield",
            scene_id=scene_ablation_runner.SCENE_ID,
            shape=(2, 3),
            crs="EPSG:32611",
            transform=(30.0, 0.0, 1.0, 0.0, -30.0, 2.0),
        )


def _four_block_design(shape: tuple[int, int] = (4, 8)):
    records = pd.DataFrame(
        [
            {
                "geometry": "L",
                "block_id": f"b{index}",
                "row_start": row,
                "row_stop": row + 2,
                "col_start": col,
                "col_stop": col + 4,
                "complete": True,
                "halo_pixels": 0,
            }
            for index, (row, col) in enumerate(((0, 0), (0, 4), (2, 0), (2, 4)))
        ]
    )
    return block_designs_from_frame(records, shape)["L"]


def _eight_block_design():
    records = pd.DataFrame(
        [
            {
                "geometry": "L",
                "block_id": f"b{index}",
                "row_start": 0,
                "row_stop": 1,
                "col_start": 4 * index,
                "col_stop": 4 * (index + 1),
                "complete": True,
                "halo_pixels": 0,
            }
            for index in range(8)
        ]
    )
    return block_designs_from_frame(records, (1, 32))["L"]


def test_preregistered_constants_are_frozen():
    assert RIDGE == 1e-2
    assert MIN_COVERAGE == 0.5
    assert BOOTSTRAP_REPLICATES == 10_000
    assert PERMUTATION_REPLICATES == 9_999
    assert SEED == 42
    assert POSITIVE_CLASS == 3
    assert NEGATIVE_CLASS == 4


def test_block_manifest_builds_complete_test_blocks_and_exact_halo():
    records = pd.DataFrame(
        [
            {
                "geometry": "L",
                "block_id": "left",
                "row_start": 0,
                "row_stop": 4,
                "col_start": 0,
                "col_stop": 4,
                "complete": True,
                "halo_pixels": 1,
            },
            {
                "geometry": "L",
                "block_id": "right",
                "row_start": 0,
                "row_stop": 4,
                "col_start": 4,
                "col_stop": 8,
                "complete": True,
                "halo_pixels": 1,
            },
            {
                "geometry": "L",
                "block_id": "edge",
                "row_start": 0,
                "row_stop": 2,
                "col_start": 0,
                "col_stop": 2,
                "complete": False,
                "halo_pixels": 1,
            },
        ]
    )

    design = block_designs_from_frame(records, (4, 8))["L"]

    assert [fold.block_id for fold in design.folds] == ["left", "right"]
    left = design.folds[0]
    assert left.test_mask[:, :4].all()
    assert not left.test_mask[:, 4:].any()
    assert not left.train_mask.any()


def test_block_manifest_requires_frozen_halo_and_complete_flags():
    records = pd.DataFrame(
        [
            {
                "geometry": "L",
                "block_id": "b",
                "row_start": 0,
                "row_stop": 2,
                "col_start": 0,
                "col_stop": 2,
            }
        ]
    )
    with pytest.raises(ValueError, match="complete"):
        block_designs_from_frame(records, (2, 2))


def test_block_manifest_consumes_m2_complete_block_contract_with_summary_halo():
    records = pd.DataFrame(
        [
            {
                "site": "goldfield",
                "scene_id": "20240925_185504_87_4001",
                "scale": "L",
                "block_id": "r0000_c0000",
                "row_start": 0,
                "row_stop": 2,
                "col_start": 0,
                "col_stop": 2,
            }
        ]
    )

    design = block_designs_from_frame(
        records,
        (2, 2),
        halo_pixels=1,
        manifest_contains_only_complete_blocks=True,
    )["L"]

    assert len(design.folds) == 1
    assert design.folds[0].test_mask.all()
    assert not design.folds[0].train_mask.any()


def test_robust_margin_uses_only_training_values_and_rejects_zero_iqr():
    alunite = np.array([[0.0, 2.0, 100.0, 200.0]])
    kaolinite = np.array([[10.0, 14.0, -100.0, -200.0]])
    train = np.array([[True, True, False, False]])

    fitted = fit_robust_margin(alunite, kaolinite, train)

    assert fitted.available
    assert fitted.alunite_median == 1.0
    assert fitted.alunite_iqr == 1.0
    assert fitted.kaolinite_median == 12.0
    assert fitted.kaolinite_iqr == 2.0
    margin = apply_robust_margin(alunite, kaolinite, fitted)
    assert margin[0, 2] == 155.0

    unavailable = fit_robust_margin(np.ones((1, 4)), kaolinite, train)
    assert not unavailable.available
    assert unavailable.reason == "zero_alunite_iqr"


def test_sensor_pair_is_cross_fitted_on_class_3_vs_4_only():
    design = _four_block_design()
    reference = np.tile(np.array([3, 3, 4, 4, 3, 3, 4, 4]), (4, 1))
    reference[0, 0] = 5  # must not enter class-3-versus-4 support
    jitter = np.arange(reference.size, dtype=float).reshape(reference.shape) / 100.0
    native_alunite = np.where(reference == 3, 3.0, 1.0) + jitter
    native_kaolinite = np.where(reference == 4, 3.0, 1.0) - jitter
    degraded_alunite = np.where(reference == 3, 2.0, 1.5) + jitter
    degraded_kaolinite = np.where(reference == 4, 2.0, 1.5) - jitter

    paired = evaluate_sensor_pair(
        native_alunite,
        native_kaolinite,
        degraded_alunite,
        degraded_kaolinite,
        reference,
        design,
    )

    assert paired.native.support_pixels == reference.size - 1
    assert paired.degraded.support_pixels == reference.size - 1
    assert set(np.unique(paired.native.labels)) == {3, 4}
    assert paired.native.metrics["auc"] >= paired.degraded.metrics["auc"]
    assert len(paired.native.thresholds) == 4


def test_zero_iqr_marks_folds_unavailable_without_rescue():
    design = _four_block_design()
    reference = np.tile(np.array([3, 3, 4, 4, 3, 3, 4, 4]), (4, 1))
    varying = np.arange(reference.size, dtype=float).reshape(reference.shape)

    paired = evaluate_sensor_pair(
        np.ones(reference.shape),
        varying,
        np.ones(reference.shape),
        varying,
        reference,
        design,
    )

    assert paired.support_pixels == reference.size
    assert paired.native.support_pixels == 0
    assert len(paired.unavailable_folds) == 4
    assert {fold["reason"] for fold in paired.unavailable_folds} == {"zero_alunite_iqr"}


def test_paired_block_bootstrap_is_reproducible_and_keeps_sensor_pairing():
    design = _four_block_design()
    reference = np.tile(np.array([3, 3, 4, 4, 3, 3, 4, 4]), (4, 1))
    jitter = np.arange(reference.size, dtype=float).reshape(reference.shape) / 100.0
    paired = evaluate_sensor_pair(
        np.where(reference == 3, 3.0, 1.0) + jitter,
        np.where(reference == 4, 3.0, 1.0) - jitter,
        np.where(reference == 3, 2.0, 1.5) + jitter,
        np.where(reference == 4, 2.0, 1.5) - jitter,
        reference,
        design,
    )

    first = paired_block_bootstrap(paired, n_boot=100, seed=SEED)
    second = paired_block_bootstrap(paired, n_boot=100, seed=SEED)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 100
    assert {"native_auc", "degraded_auc", "delta_auc"}.issubset(first.columns)
    np.testing.assert_allclose(first["delta_auc"], first["native_auc"] - first["degraded_auc"])


def test_benjamini_hochberg_corrects_one_family_and_preserves_nan():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))
    np.testing.assert_allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_paired_sensor_randomization_is_deterministic_and_null_is_one():
    design = _eight_block_design()
    reference = np.tile(np.array([3, 3, 4, 4]), 8)[None, :]
    score = np.tile(np.array([3.0, 2.0, 0.0, 1.0]), 8)[None, :]

    first = paired_sensor_auc_randomization(
        score,
        score,
        reference,
        design,
        positive_classes=frozenset({3}),
        randomizations=255,
        seed=SEED,
    )
    second = paired_sensor_auc_randomization(
        score,
        score,
        reference,
        design,
        positive_classes=frozenset({3}),
        randomizations=255,
        seed=SEED,
    )

    assert first.p_value == 1.0
    assert first.exceedances == 255
    np.testing.assert_array_equal(first.permuted_deltas, second.permuted_deltas)


def test_paired_sensor_randomization_detects_effect_with_add_one_bounds():
    design = _eight_block_design()
    reference = np.tile(np.array([3, 3, 4, 4]), 8)[None, :]
    native = np.tile(np.array([3.0, 2.0, 0.0, 1.0]), 8)[None, :]
    degraded = np.tile(np.array([0.0, 1.0, 3.0, 2.0]), 8)[None, :]
    randomizations = 999

    result = paired_sensor_auc_randomization(
        native,
        degraded,
        reference,
        design,
        positive_classes=frozenset({3}),
        randomizations=randomizations,
        seed=SEED,
    )

    assert result.observed_delta == pytest.approx(1.0)
    assert result.p_value == (1 + result.exceedances) / (randomizations + 1)
    assert 1 / (randomizations + 1) <= result.p_value <= 1.0
    assert result.p_value < 0.05


def test_bh_adjustment_keeps_s2a_and_s2b_as_separate_families():
    p_values = np.array([0.01, 0.04, 0.04, 0.90])
    families = np.array(["S2A", "S2A", "S2B", "S2B"])

    adjusted = benjamini_hochberg_by_family(p_values, families)

    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.08, 0.90])


@pytest.mark.parametrize(
    ("positive_blocks", "negative_blocks", "expected"),
    [
        (10, 10, "confirmatory"),
        (10, 9, "exploratory"),
        (5, 100, "exploratory"),
        (100, 5, "exploratory"),
        (4, 100, "counts_maps_only"),
        (100, 4, "counts_maps_only"),
    ],
)
def test_support_governance_uses_the_limiting_complete_block_class(
    positive_blocks,
    negative_blocks,
    expected,
):
    governance = support_governance(positive_blocks, negative_blocks)

    assert inference_status(positive_blocks, negative_blocks) == expected
    assert governance.status == expected
    assert governance.effect_estimates == (expected != "counts_maps_only")
    assert governance.bootstrap_cis == (expected != "counts_maps_only")
    assert governance.permutation_inference == (expected == "confirmatory")
    assert governance.bh_adjustment == (expected == "confirmatory")


def test_counts_maps_only_masks_effects_and_bootstrap_intervals():
    bootstrap = np.array([-0.2, 0.1, 0.4])
    exploratory = support_governance(5, 9)
    counts_only = support_governance(4, 100)

    exploratory_summary = governed_metric_summary(0.25, bootstrap, exploratory)
    counts_summary = governed_metric_summary(0.25, bootstrap, counts_only)

    assert exploratory_summary == (0.25, *percentile_interval(bootstrap))
    assert all(np.isnan(value) for value in counts_summary)


def test_counts_maps_only_rows_and_distributions_have_no_inferential_leakage(
    scene_ablation_runner,
):
    design = _four_block_design()
    reference = np.tile(np.array([3, 3, 4, 4, 3, 3, 4, 4]), (4, 1))
    jitter = np.arange(reference.size, dtype=float).reshape(reference.shape) / 100.0
    paired = evaluate_sensor_pair(
        np.where(reference == 3, 3.0, 1.0) + jitter,
        np.where(reference == 4, 3.0, 1.0) - jitter,
        np.where(reference == 3, 2.0, 1.5) + jitter,
        np.where(reference == 4, 2.0, 1.5) - jitter,
        reference,
        design,
    )
    bootstrap = paired_block_bootstrap(paired, n_boot=20, seed=SEED)

    rows = scene_ablation_runner._metric_rows(
        paired,
        bootstrap,
        endpoint="primary_margin",
        layer="alunite_minus_kaolinite",
        geometry="L",
        comparator="S2A",
    )

    assert {row["inference_status"] for row in rows} == {"counts_maps_only"}
    assert all(np.isnan(row["value"]) for row in rows)
    assert all(np.isnan(row["bootstrap_ci_lower"]) for row in rows)
    assert all(np.isnan(row["bootstrap_ci_upper"]) for row in rows)
    assert scene_ablation_runner._bootstrap_if_permitted(paired) is None
    assert (
        scene_ablation_runner._fold_rows(
            paired,
            endpoint="primary_margin",
            layer="alunite_minus_kaolinite",
            geometry="L",
            comparator="S2A",
        )
        == []
    )


def test_bh_excludes_nonconfirmatory_endpoints_from_each_sensor_family():
    adjusted = confirmatory_bh_by_family(
        np.array([0.03, 0.04, 0.02, 0.04]),
        np.array(["S2A", "S2A", "S2B", "S2B"]),
        np.array(["confirmatory", "exploratory", "confirmatory", "counts_maps_only"]),
    )

    np.testing.assert_allclose(adjusted[[0, 2]], [0.03, 0.02])
    assert np.isnan(adjusted[[1, 3]]).all()


def test_primary_gate_rejects_positive_exploratory_sensitivity(
    scene_ablation_runner,
):
    rows = pd.DataFrame(
        [
            {
                "comparator": "S2A",
                "geometry": "L",
                "sensor": "native_minus_degraded",
                "metric": "auc",
                "value": 0.2,
                "bootstrap_ci_lower": 0.1,
                "inference_status": "confirmatory",
            },
            {
                "comparator": "S2A",
                "geometry": "L",
                "sensor": "native_minus_degraded",
                "metric": "balanced_accuracy",
                "value": 0.2,
                "bootstrap_ci_lower": 0.1,
                "inference_status": "confirmatory",
            },
            {
                "comparator": "S2A",
                "geometry": "2L",
                "sensor": "native_minus_degraded",
                "metric": "auc",
                "value": 0.2,
                "bootstrap_ci_lower": 0.1,
                "inference_status": "confirmatory",
            },
            {
                "comparator": "S2B",
                "geometry": "L",
                "sensor": "native_minus_degraded",
                "metric": "auc",
                "value": 0.2,
                "bootstrap_ci_lower": 0.1,
                "inference_status": "exploratory",
            },
        ]
    )

    decision = scene_ablation_runner._decision_summary(rows)

    assert not decision["gate_passed"]
    assert not decision["checks"]["s2b_auc_direction_positive"]


def test_compute_sensor_mtmf_degrades_spectra_without_spatial_change():
    rng = np.random.default_rng(7)
    wavelengths = np.arange(900.0, 1501.0, 100.0)
    data = rng.normal(0.4, 0.02, size=(wavelengths.size, 3, 5))
    cube = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": wavelengths, "y": [3.0, 2.0, 1.0], "x": np.arange(5)},
    ).rio.write_crs("EPSG:32611")
    endmembers = {
        "alunite": Endmember(
            "alunite", "alunite_sample", "ASD", wavelengths, np.linspace(0.3, 0.5, 7)
        ),
        "kaolinite": Endmember(
            "kaolinite", "kaolinite_sample", "BECK", wavelengths, np.linspace(0.5, 0.3, 7)
        ),
    }
    grid = np.arange(850.0, 1551.0, 1.0)
    srf = gaussian_srf(
        ["b1", "b2", "b3"],
        np.array([1000.0, 1200.0, 1400.0]),
        np.array([80.0, 80.0, 80.0]),
        grid,
    )

    result = compute_sensor_mtmf(
        cube,
        wavelengths,
        endmembers,
        {"S2A": srf},
        ridge=RIDGE,
        min_coverage=MIN_COVERAGE,
    )

    assert set(result) == {"native", "S2A"}
    assert result["native"].sizes == {"y": 3, "x": 5}
    assert result["S2A"].sizes == {"y": 3, "x": 5}
    np.testing.assert_array_equal(result["S2A"].y, cube.y)
    np.testing.assert_array_equal(result["S2A"].x, cube.x)
    assert {"alunite_mf", "kaolinite_mf"}.issubset(result["S2A"].data_vars)
