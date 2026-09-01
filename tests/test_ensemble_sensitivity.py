"""Synthetic tests for the preregistered E6 MTMF sensitivity design."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tanager_rocks.config import TARGET_MINERALS
from tanager_rocks.ensemble_sensitivity import (
    BASELINE_ENDMEMBERS,
    DOMINANT_NODATA,
    EXPECTED_CANDIDATE_COUNTS,
    FROZEN_BOOTSTRAP_REPLICATES,
    FROZEN_GATES,
    FROZEN_QUANTILES,
    FROZEN_RIDGES,
    FROZEN_SEED,
    FROZEN_STOCHASTIC_REPLICATES,
    CrossFittedMetrics,
    MapAccumulator,
    MemberLedger,
    ProtocolError,
    _load_goldfield_reference,
    build_design,
    calibration_diagnostic,
    classify_permitted_claim,
    confidence_classes,
    evaluate_goldfield_claim_gate,
    governing_file_provenance,
    nested_block_bootstrap,
    nested_external_metric_intervals,
    nested_ratio_bootstrap,
    nested_spearman_bootstrap,
    operational_detection,
    paired_factor_effect_rows,
    scientific_design_sha256,
    sha256_file,
    strict_covariance_cross_fitted_threshold_evaluation,
    strict_json_dump,
    summarize_dominant_classes,
    timing_pilot_fit_ids,
    validate_m2_manifest,
    validate_protocol_amendment,
    validate_protocol_arguments,
    validate_protocol_file,
)

ROOT = Path(__file__).resolve().parents[1]


def test_goldfield_integer_reference_converts_masked_nodata_to_nan(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    reference_dir = tmp_path / "data" / "reference"
    reference_dir.mkdir(parents=True)
    path = reference_dir / "rockwell_goldfield_20240925_185504_87_4001.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint8",
        nodata=255,
        transform=rasterio.transform.from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.array([[1, 255], [2, 3]], dtype=np.uint8), 1)

    loaded = _load_goldfield_reference(tmp_path, (2, 2))

    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded[[0, 1, 1], [0, 0, 1]], [1.0, 2.0, 3.0])
    assert np.isnan(loaded[0, 1])


def _write_amendment(path: Path, changes: dict, **overrides) -> None:
    payload = {
        "schema_version": "1.0",
        "amendment_type": "e6_pre_result_protocol_amendment",
        "amendment_date": "2026-08-09",
        "authorized": True,
        "authorized_by": "E6 scientific lead",
        "authorization_basis": "Written pre-result authorization",
        "pre_result": True,
        "results_seen": False,
        "rationale": "Authorize only the explicitly enumerated change.",
        "changes": changes,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate_population() -> dict[str, tuple[str, ...]]:
    population: dict[str, tuple[str, ...]] = {}
    for mineral in TARGET_MINERALS:
        medoid = BASELINE_ENDMEMBERS[mineral]
        extras = tuple(
            f"splib07a_{mineral.title()}_candidate_{index:02d}_ASDFRa_AREF.txt"
            for index in range(EXPECTED_CANDIDATE_COUNTS[mineral] - 1)
        )
        population[mineral] = tuple(sorted((medoid, *extras)))
    return population


def _design():
    return build_design(
        candidates=_candidate_population(),
        complete_blocks={"goldfield": (1, 2, 3), "bingham": (10, 20, 30, 40)},
        sites=("goldfield", "bingham"),
        ridges=FROZEN_RIDGES,
        quantiles=FROZEN_QUANTILES,
        gates=FROZEN_GATES,
        stochastic_replicates=FROZEN_STOCHASTIC_REPLICATES,
        bootstrap_replicates=FROZEN_BOOTSTRAP_REPLICATES,
        seed=FROZEN_SEED,
    )


def test_design_has_frozen_counts_fit_reuse_and_balanced_schedules():
    design, members = _design()

    assert design["analytical_cells"] == 18
    assert design["recorded_variants_per_scene"] == 355
    assert design["unique_mtmf_fits_per_scene"] == 83
    assert len(members) == 710

    for site in ("goldfield", "bingham"):
        site_rows = [row for row in members if row["site"] == site]
        assert Counter(row["member_class"] for row in site_rows) == {
            "baseline": 1,
            "endmember_only": 16,
            "covariance_only": 16,
            "calibration_only": 16,
            "analytical_grid": 18,
            "joint": 288,
        }
        assert len({row["fit_id"] for row in site_rows}) == 83

    schedules = design["endmember_schedules"]
    assert len(schedules) == 16
    for mineral, candidates in _candidate_population().items():
        counts = Counter(schedule[mineral] for schedule in schedules)
        assert set(counts) == set(candidates)
        assert max(counts.values()) - min(counts.values()) <= 1
        assert set(counts.values()).issubset(
            {16 // len(candidates), int(np.ceil(16 / len(candidates)))}
        )


def test_endmember_schedule_uses_seeded_random_remainder_allocation():
    design, _ = _design()
    schedules = design["endmember_schedules"]

    for mineral_index, mineral in enumerate(TARGET_MINERALS):
        candidates = _candidate_population()[mineral]
        quotient, remainder = divmod(16, len(candidates))
        counts = Counter(schedule[mineral] for schedule in schedules)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([42, mineral_index])))
        expected_extra = {
            candidates[int(index)] for index in rng.permutation(len(candidates))[:remainder]
        }
        actual_extra = {candidate for candidate, count in counts.items() if count > quotient}
        assert actual_extra == expected_extra


def test_design_seeds_are_frozen_paired_and_site_specific():
    design, members = _design()
    assert design["seed_derivations"]["endmember"] == [
        [42, index] for index in range(len(TARGET_MINERALS))
    ]

    goldfield = [
        row for row in members if row["site"] == "goldfield" and row["member_class"] == "joint"
    ]
    first_cell = [row for row in goldfield if row["stochastic_replicate"] == 0]
    assert {row["covariance_seed_entropy"] for row in first_cell} == {"[42,1000,0]"}
    assert {row["calibration_seed_entropy"] for row in first_cell} == {"[42,2000,0]"}
    assert len({row["covariance_draw"] for row in first_cell}) == 1
    assert len({row["calibration_draw"] for row in first_cell}) == 1

    bingham = next(
        row
        for row in members
        if row["site"] == "bingham"
        and row["member_class"] == "joint"
        and row["stochastic_replicate"] == 0
    )
    assert bingham["covariance_seed_entropy"] == "[42,1001,0]"
    assert bingham["calibration_seed_entropy"] == "[42,2001,0]"


def test_design_contains_no_pseudo_binomial_inference_fields():
    design, members = _design()
    serialized = json.dumps({"design": design, "members": members}).lower()
    for forbidden in ("wilson", "binomial", "standard_error", "pixelwise_p_value"):
        assert forbidden not in serialized
    assert design["frequency_estimand"] == "finite_design_empirical_frequency"


def test_protocol_hash_mismatch_refuses_without_amendment(tmp_path):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("changed preregistration\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="preregistration hash"):
        validate_protocol_file(prereg, expected_sha256="0" * 64)

    amendment = tmp_path / "amendment.json"
    _write_amendment(
        amendment,
        {
            "preregistration_sha256": {
                "expected": "0" * 64,
                "observed": sha256_file(prereg),
            }
        },
    )
    record = validate_protocol_file(prereg, expected_sha256="0" * 64, protocol_amendment=amendment)
    assert record["protocol_compliant"] is False
    assert record["amendment"]["path"] == str(amendment)
    assert len(record["amendment"]["sha256"]) == 64


def test_protocol_amendment_rejects_unexecuted_extra_change(tmp_path):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("changed preregistration\n", encoding="utf-8")
    amendment = tmp_path / "amendment.json"
    _write_amendment(
        amendment,
        {
            "preregistration_sha256": {
                "expected": "0" * 64,
                "observed": sha256_file(prereg),
            },
            "unrelated_change": {"expected": "frozen", "observed": "altered"},
        },
    )

    with pytest.raises(ProtocolError, match="unexecuted changes"):
        validate_protocol_file(
            prereg,
            expected_sha256="0" * 64,
            protocol_amendment=amendment,
        )


def test_protocol_amendment_rejected_when_frozen_preregistration_matches(tmp_path):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("frozen preregistration\n", encoding="utf-8")
    amendment = tmp_path / "amendment.json"
    _write_amendment(amendment, {"arbitrary_change": {"expected": 1, "observed": 2}})

    with pytest.raises(ProtocolError, match="no governed changes are expected"):
        validate_protocol_file(
            prereg,
            expected_sha256=sha256_file(prereg),
            protocol_amendment=amendment,
        )


def test_protocol_amendment_change_comparison_is_json_type_strict(tmp_path):
    amendment = tmp_path / "amendment.json"
    _write_amendment(amendment, {"governed_integer": True})

    with pytest.raises(ProtocolError, match="does not match execution"):
        validate_protocol_amendment(
            amendment,
            expected_changes={"governed_integer": 1},
        )


@pytest.mark.parametrize(
    ("field", "numeric_value"),
    [("authorized", 1), ("pre_result", 1), ("results_seen", 0)],
)
def test_protocol_amendment_boolean_controls_reject_numeric_values(
    tmp_path,
    field,
    numeric_value,
):
    amendment = tmp_path / "amendment.json"
    _write_amendment(
        amendment,
        {"governed_integer": 1},
        **{field: numeric_value},
    )

    with pytest.raises(ProtocolError, match=field):
        validate_protocol_amendment(
            amendment,
            expected_changes={"governed_integer": 1},
        )


def test_frozen_preregistration_hash_matches_current_document():
    from tanager_rocks.ensemble_sensitivity import FROZEN_PREREGISTRATION_SHA256

    assert sha256_file(ROOT / "docs/m2_ensemble_sensitivity_preregistration.md") == (
        FROZEN_PREREGISTRATION_SHA256
    )


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("Dated but free-form amendment.\n", "valid UTF-8 JSON"),
        (json.dumps({"authorized": True}), "schema_version"),
    ],
)
def test_protocol_amendment_fails_closed_on_free_form_or_incomplete_content(
    tmp_path, content, match
):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("changed preregistration\n", encoding="utf-8")
    amendment = tmp_path / "amendment.json"
    amendment.write_text(content, encoding="utf-8")
    with pytest.raises(ProtocolError, match=match):
        validate_protocol_file(
            prereg,
            expected_sha256="0" * 64,
            protocol_amendment=amendment,
        )


def test_protocol_amendment_requires_pre_result_authorization(tmp_path):
    prereg = tmp_path / "prereg.md"
    prereg.write_text("changed preregistration\n", encoding="utf-8")
    change = {
        "preregistration_sha256": {
            "expected": "0" * 64,
            "observed": sha256_file(prereg),
        }
    }
    amendment = tmp_path / "amendment.json"
    _write_amendment(amendment, change, results_seen=True)
    with pytest.raises(ProtocolError, match="results_seen"):
        validate_protocol_file(
            prereg,
            expected_sha256="0" * 64,
            protocol_amendment=amendment,
        )


def test_m2_manifest_stale_protocol_hash_is_refused_before_raster_use(tmp_path):
    preregistration = tmp_path / "m2.md"
    preregistration.write_text("current M2 protocol\n", encoding="utf-8")
    manifest = tmp_path / "block_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_type": "spatial_validation_complete_blocks",
                "protocol": {
                    "path": "docs/m2_spatial_validation_preregistration.md",
                    "sha256": "0" * 64,
                },
                "sites": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="M2 manifest protocol hash"):
        validate_m2_manifest(
            manifest,
            m2_preregistration=preregistration,
        )


def test_zero_eligible_calibration_blocks_is_unavailable_not_negative():
    scores = np.array([[0.9, 0.1], [0.8, np.nan]])
    infeasibility = np.zeros_like(scores)
    block_ids = np.array([[1, 1], [2, 0]])

    result = operational_detection(
        scores,
        infeasibility,
        block_ids,
        calibration_draw=(),
        quantile=0.90,
        max_infeasibility=1.0,
    )

    assert result.status == "unavailable"
    assert result.reason == "zero_eligible_calibration_blocks"
    assert result.threshold is None
    assert result.detections is None


def test_infeasibility_gate_keeps_finite_pixels_as_valid_nondetections():
    result = operational_detection(
        np.array([[0.9, 0.8], [0.7, 0.6]]),
        np.array([[0.1, 1.5], [0.2, 1.2]]),
        np.ones((2, 2), dtype=np.uint8),
        calibration_draw=(1,),
        quantile=0.5,
        max_infeasibility=1.0,
    )

    assert result.status == "complete"
    np.testing.assert_array_equal(result.valid_support, np.ones((2, 2), dtype=bool))
    assert result.detections is not None
    assert not result.detections[0, 1]
    assert not result.detections[1, 1]
    accumulator = MapAccumulator((2, 2))
    accumulator.add(result.detections, result.valid_support)
    np.testing.assert_array_equal(accumulator.valid_count, np.ones((2, 2), dtype=np.uint16))


def test_nested_bootstrap_uses_one_shared_draw_then_summarizes_members():
    values = {
        "m1": {1: 1.0, 2: 3.0, 3: 5.0},
        "m2": {1: 10.0, 2: 30.0, 3: 50.0},
    }
    result = nested_block_bootstrap(values, replicates=8, seed=42)

    assert result.draws.shape == (8, 3)
    assert result.member_values.shape == (8, 2)
    for replicate, draw in enumerate(result.draws):
        expected_1 = np.mean([values["m1"][int(block)] for block in draw])
        expected_2 = np.mean([values["m2"][int(block)] for block in draw])
        np.testing.assert_allclose(result.member_values[replicate], (expected_1, expected_2))
        assert result.replicate_summaries[replicate] == pytest.approx(
            np.median((expected_1, expected_2))
        )
    assert result.interval_available
    assert result.valid_replicates == 8


def test_nested_bootstrap_enforces_finite_replicate_and_structural_block_rules():
    sparse = nested_block_bootstrap(
        {"member": {1: 1.0, 2: np.nan}},
        replicates=200,
        seed=42,
    )
    assert sparse.valid_replicates < 190
    assert not sparse.interval_available
    assert sparse.unavailable_reason == "fewer_than_95_percent_finite_replicates"
    assert sparse.lower_95 is None

    one_block = nested_block_bootstrap({"member": {1: 0.7}}, replicates=20, seed=42)
    assert one_block.valid_replicates == 20
    assert not one_block.interval_available
    assert one_block.unavailable_reason == "fewer_than_2_complete_blocks"

    zero_blocks = nested_block_bootstrap({"member": {}}, replicates=20, seed=42)
    assert zero_blocks.valid_replicates == 0
    assert zero_blocks.unavailable_reason == "zero_complete_blocks"


def test_nested_ratio_and_rank_metrics_are_recomputed_after_each_block_draw():
    ratio = nested_ratio_bootstrap(
        {"m0": {1: (1.0, 1.0), 2: (0.0, 9.0)}},
        replicates=20,
        seed=42,
    )
    mixed = next(index for index, draw in enumerate(ratio.draws) if set(draw.tolist()) == {1, 2})
    assert ratio.member_values[mixed, 0] == pytest.approx(0.1)

    pairs = {
        "m0": {
            1: (np.array([1.0, 2.0]), np.array([1.0, 2.0])),
            2: (np.array([3.0, 4.0]), np.array([4.0, 3.0])),
        }
    }
    rank = nested_spearman_bootstrap(pairs, replicates=20, seed=42)
    mixed = next(index for index, draw in enumerate(rank.draws) if set(draw.tolist()) == {1, 2})
    from scipy.stats import spearmanr

    expected = spearmanr([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0]).statistic
    assert rank.member_values[mixed, 0] == pytest.approx(expected)


def _synthetic_crossfit_result(block_count: int) -> CrossFittedMetrics:
    blocks = tuple(
        {
            "block_id": block_id,
            "scores": np.array([0.1, 0.9]),
            "references": np.array([0, 1], dtype=np.int8),
            "predictions": np.array([False, True]),
        }
        for block_id in range(1, block_count + 1)
    )
    return CrossFittedMetrics(
        status="complete",
        reason=None,
        covariance_scope="full_scene_covariance",
        auc=1.0,
        balanced_accuracy=1.0,
        evaluated_blocks=block_count,
        unavailable_blocks=0,
        n_pixels=2 * block_count,
        block_results=blocks,
    )


def test_external_nested_intervals_cover_all_declared_metrics_and_support_tier():
    result = _synthetic_crossfit_result(20)
    rows = nested_external_metric_intervals(
        {"m0": result, "m1": result},
        scale="L",
        replicates=40,
        seed=42,
    )

    assert {row["metric"] for row in rows} == {
        "auc",
        "balanced_accuracy",
        "positive_f1",
        "negative_f1",
        "macro_f1",
        "tpr",
        "fpr",
        "prevalence",
    }
    assert all(row["interval_available"] for row in rows)
    assert all(row["confirmatory_support"] for row in rows)
    assert all(row["valid_replicates"] == 40 for row in rows)


def test_goldfield_2l_single_block_is_structurally_nonconfirmatory():
    rows = nested_external_metric_intervals(
        {"m0": _synthetic_crossfit_result(1)},
        scale="2L",
        replicates=20,
        seed=42,
    )

    assert all(row["complete_blocks"] == 1 for row in rows)
    assert all(not row["interval_available"] for row in rows)
    assert all(not row["confirmatory_support"] for row in rows)
    assert all(row["unavailable_reason"] == "fewer_than_two_complete_blocks" for row in rows)


def test_strict_covariance_evaluation_uses_fold_specific_full_score_maps():
    reference = np.array([[0.0, 1.0], [0.0, 1.0]])
    block_ids = np.array([[1, 1], [2, 2]], dtype=np.uint32)
    records = [
        {
            "numeric_block_id": 1,
            "block_id": "b1",
            "block_row": 0,
            "block_col": 0,
            "row_start": 0,
            "row_stop": 1,
            "col_start": 0,
            "col_stop": 2,
        },
        {
            "numeric_block_id": 2,
            "block_id": "b2",
            "block_row": 1,
            "block_col": 0,
            "row_start": 1,
            "row_stop": 2,
            "col_start": 0,
            "col_stop": 2,
        },
    ]
    fold_scores = {
        1: np.array([[0.1, 0.9], [0.3, 0.7]]),
        2: np.array([[0.2, 0.8], [0.4, 0.9]]),
    }
    result = strict_covariance_cross_fitted_threshold_evaluation(
        fold_scores,
        reference,
        block_ids,
        records,
        halo_pixels=0,
        site_index=0,
        stochastic_replicate=0,
    )

    assert result.covariance_scope == "strict_covariance_exclusion"
    assert result.status == "complete"
    assert result.auc == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert [block["scores"].tolist() for block in result.block_results] == [
        [0.1, 0.9],
        [0.4, 0.9],
    ]


def test_calibration_endpoints_have_governed_block_intervals():
    frequency = np.array([[0.1, 0.9], [0.2, 0.8]])
    reference = np.array([[0.0, 1.0], [0.0, 1.0]])
    blocks = np.array([[1, 1], [2, 2]], dtype=np.uint32)
    rows = calibration_diagnostic(
        frequency,
        reference,
        blocks,
        site="goldfield",
        mineral="alunite",
        bootstrap_replicates=20,
    )

    assert all(row["brier_interval_available"] for row in rows)
    assert all(row["ece_interval_available"] for row in rows)
    assert all(row["brier_valid_replicates"] == 20 for row in rows)
    assert all(row["ece_valid_replicates"] == 20 for row in rows)


def test_empirical_confidence_classes_use_exact_fixed_boundaries():
    frequency = np.array([[0.0, 0.2, 0.20001], [0.79999, 0.8, 1.0], [np.nan, 0.5, 0.9]])
    classes = confidence_classes(frequency)

    np.testing.assert_array_equal(
        classes,
        np.array([[0, 0, 1], [1, 2, 2], [-1, 1, 2]], dtype=np.int8),
    )


def test_dominant_summary_distinguishes_nodata_from_valid_no_detection():
    baseline = np.array([[-1, 0, DOMINANT_NODATA]])
    members = (
        np.array([[-1, 0, DOMINANT_NODATA]]),
        np.array([[0, 0, DOMINANT_NODATA]]),
    )
    summary = summarize_dominant_classes(members, baseline)

    np.testing.assert_array_equal(summary.valid_count, np.array([[2, 2, 0]]))
    assert summary.modal_class[0, 2] == DOMINANT_NODATA
    assert np.isnan(summary.modal_frequency[0, 2])
    assert summary.switch_frequency[0, 0] == pytest.approx(0.5)


def test_failed_members_are_ledgered_and_excluded_from_valid_frequency(tmp_path):
    rows = [
        {"member_id": "m0", "site": "goldfield", "status": "pending", "failure_reason": None},
        {"member_id": "m1", "site": "goldfield", "status": "pending", "failure_reason": None},
    ]
    ledger = MemberLedger.initialize(tmp_path / "members.csv", rows, design_sha256="a" * 64)
    ledger.update("m0", status="complete")
    ledger.update("m1", status="failed", failure_reason="singular_covariance")

    resumed = MemberLedger.initialize(
        tmp_path / "members.csv", rows, design_sha256="a" * 64, resume=True
    )
    assert [row["member_id"] for row in resumed.rows] == ["m0", "m1"]
    assert resumed.status_counts() == {"complete": 1, "failed": 1}

    accumulator = MapAccumulator((1, 2))
    accumulator.add(np.array([[True, False]]), np.array([[True, True]]))
    accumulator.record_failure("m1", "singular_covariance")
    np.testing.assert_allclose(accumulator.frequency(), np.array([[1.0, 0.0]]))
    assert accumulator.failures == [{"member_id": "m1", "reason": "singular_covariance"}]


def test_factor_effects_are_member_and_block_paired_deltas():
    rows = []
    for block_id, baseline, treatment in ((1, 0.1, 0.3), (2, 0.2, 0.5)):
        common = {
            "site": "goldfield",
            "mineral": "alunite",
            "aggregation": "block",
            "block_scale": "L",
            "block_id": block_id,
            "ridge": 0.01,
            "detection_quantile": 0.9,
            "infeasibility_gate": "1",
            "common_support_pixels": 10,
        }
        rows.append(
            {
                **common,
                "member_id": "baseline",
                "member_class": "baseline",
                "stochastic_replicate": None,
                "detection_prevalence": baseline,
            }
        )
        rows.append(
            {
                **common,
                "member_id": "endmember-0",
                "member_class": "endmember_only",
                "stochastic_replicate": 0,
                "detection_prevalence": treatment,
            }
        )

    effects = paired_factor_effect_rows(rows, bootstrap_replicates=30)
    effect = next(
        row
        for row in effects
        if row["factor"] == "axis"
        and row["level"] == "endmember_only"
        and row["endpoint"] == "detection_prevalence"
    )
    assert effect["paired_delta_median"] == pytest.approx(0.25)
    assert effect["n_pairs"] == 1
    assert effect["complete_blocks"] == 2
    assert effect["interval_available"]
    assert "median" not in effect


def test_claim_gate_requires_operational_strict_and_support_evidence():
    l_rows = nested_external_metric_intervals(
        {"m0": _synthetic_crossfit_result(20)},
        scale="L",
        replicates=30,
    )
    two_l_rows = nested_external_metric_intervals(
        {"m0": _synthetic_crossfit_result(20)},
        scale="2L",
        replicates=30,
    )
    switch = nested_block_bootstrap(
        {"m0": {block_id: 0.1 for block_id in range(1, 21)}},
        replicates=30,
    )
    passed = evaluate_goldfield_claim_gate(
        analytical_cells_complete=True,
        stable_core_retention=0.9,
        median_rank_correlation=0.9,
        rank_correlation_5th_percentile=0.7,
        switch_interval=switch,
        operational_intervals=[*l_rows, *two_l_rows],
        strict_intervals=[*l_rows, *two_l_rows],
    )
    assert passed["confirmatory_gate_available"]
    assert passed["confirmatory_gate_pass"]
    assert passed["permitted_claim_classification"].startswith("validated_")

    structural_2l = nested_external_metric_intervals(
        {"m0": _synthetic_crossfit_result(1)},
        scale="2L",
        replicates=30,
    )
    unavailable = evaluate_goldfield_claim_gate(
        analytical_cells_complete=True,
        stable_core_retention=0.9,
        median_rank_correlation=0.9,
        rank_correlation_5th_percentile=0.7,
        switch_interval=switch,
        operational_intervals=[*l_rows, *structural_2l],
        strict_intervals=[*l_rows, *structural_2l],
    )
    assert not unavailable["confirmatory_gate_available"]
    assert unavailable["confirmatory_gate_pass"] is None
    assert unavailable["permitted_claim_classification"] == "unavailable_required_evidence"


def test_incomplete_analytical_cells_cannot_receive_validated_claim():
    l_rows = nested_external_metric_intervals(
        {"m0": _synthetic_crossfit_result(20)},
        scale="L",
        replicates=30,
    )
    two_l_rows = nested_external_metric_intervals(
        {"m0": _synthetic_crossfit_result(20)},
        scale="2L",
        replicates=30,
    )
    switch = nested_block_bootstrap(
        {"m0": {block_id: 0.1 for block_id in range(1, 21)}},
        replicates=30,
    )

    result = evaluate_goldfield_claim_gate(
        analytical_cells_complete=False,
        stable_core_retention=0.9,
        median_rank_correlation=0.9,
        rank_correlation_5th_percentile=0.7,
        switch_interval=switch,
        operational_intervals=[*l_rows, *two_l_rows],
        strict_intervals=[*l_rows, *two_l_rows],
    )

    assert result["confirmatory_gate_pass"] is None
    assert result["permitted_claim_classification"] == "unavailable_required_evidence"
    assert result["permitted_claim_classification"] != (
        "validated_analytically_robust_alteration_zone_discrimination"
    )


@pytest.mark.parametrize(
    ("stability", "external", "strict", "expected"),
    [
        (True, False, True, "analytically_stable_spatial_pattern_only"),
        (False, True, True, "discriminative_but_analytically_sensitive"),
        (False, False, False, "negative_or_unstable_result"),
        (True, True, None, "unavailable_required_evidence"),
    ],
)
def test_permitted_claim_classification_covers_negative_unstable_and_unavailable(
    stability, external, strict, expected
):
    assert (
        classify_permitted_claim(
            stability_pass=stability,
            external_pass=external,
            strict_covariance_pass=strict,
        )
        == expected
    )


def test_governing_provenance_hashes_current_bytes_and_reports_git_state():
    records = {record["path"]: record for record in governing_file_provenance(ROOT)}
    for relative in (
        "src/tanager_rocks/ensemble_sensitivity.py",
        "scripts/run_ensemble_sensitivity.py",
        "tests/test_ensemble_sensitivity.py",
        "docs/m2_ensemble_sensitivity_preregistration.md",
    ):
        record = records[relative]
        assert record["sha256"] == sha256_file(ROOT / relative)
        assert isinstance(record["dirty"], bool)
        assert isinstance(record["tracked"], bool)
        assert len(record["git_status"]) == 2


def test_compute_controls_are_inert_but_scientific_resume_identity_is_strict():
    first = {
        "protocol": {"sha256": "a" * 64},
        "members": ["m0"],
        "compute_controls": {"batch_size": 1, "storage_layout": "disk"},
    }
    rescued = {
        **first,
        "compute_controls": {"batch_size": 8, "storage_layout": "memory"},
    }
    changed = {**first, "members": ["m1"]}
    assert scientific_design_sha256(first) == scientific_design_sha256(rescued)
    assert scientific_design_sha256(first) != scientific_design_sha256(changed)


def test_resume_rejects_order_or_design_changes_and_checkpoint_is_deterministic(tmp_path):
    rows = [
        {"member_id": "m0", "site": "goldfield", "status": "pending"},
        {"member_id": "m1", "site": "goldfield", "status": "pending"},
    ]
    path = tmp_path / "members.csv"
    ledger = MemberLedger.initialize(path, rows, design_sha256="b" * 64)
    ledger.update("m0", status="complete", contributing_pixels=12)
    first_bytes = path.read_bytes()
    MemberLedger.initialize(path, rows, design_sha256="b" * 64, resume=True)
    assert path.read_bytes() == first_bytes

    with pytest.raises(ProtocolError, match="member order"):
        MemberLedger.initialize(path, list(reversed(rows)), design_sha256="b" * 64, resume=True)
    with pytest.raises(ProtocolError, match="design hash"):
        MemberLedger.initialize(path, rows, design_sha256="c" * 64, resume=True)


def test_strict_json_converts_nonfinite_values_to_null(tmp_path):
    path = tmp_path / "strict.json"
    strict_json_dump(path, {"finite": 1.0, "nan": np.nan, "inf": np.inf})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "inf": None,
        "nan": None,
    }
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_timing_pilot_is_exactly_baseline_plus_replicate_zero_per_site():
    _, members = _design()
    selected = timing_pilot_fit_ids(members)
    assert set(selected) == {"goldfield", "bingham"}
    assert all(len(fit_ids) == 2 for fit_ids in selected.values())
    for site, fit_ids in selected.items():
        site_rows = [row for row in members if row["site"] == site]
        expected = {
            next(row["fit_id"] for row in site_rows if row["member_class"] == "baseline"),
            next(
                row["fit_id"]
                for row in site_rows
                if row["member_class"] == "joint"
                and row["stochastic_replicate"] == 0
                and row["ridge"] == 0.01
            ),
        }
        assert set(fit_ids) == expected


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ensemble_sensitivity.py"
    spec = importlib.util.spec_from_file_location("_ensemble_sensitivity_cli_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load CLI module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_cli_refuses_deviation_without_explicit_amendment(tmp_path):
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--preregistration",
            "docs/m2_ensemble_sensitivity_preregistration.md",
            "--block-manifest",
            "data/processed/spatial_validation/block_manifest.json",
            "--sites",
            "goldfield",
            "bingham",
            "--ridge",
            "0.001",
            "0.01",
            "0.1",
            "--detection-quantiles",
            "0.85",
            "0.90",
            "0.95",
            "--infeasibility-gates",
            "none",
            "1.0",
            "--stochastic-replicates",
            "16",
            "--bootstrap-replicates",
            "10000",
            "--seed",
            "41",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ProtocolError, match="--seed"):
        validate_protocol_arguments(args)

    amendment = tmp_path / "amendment.json"
    _write_amendment(
        amendment,
        {"scientific_cli": {"seed": {"expected": 42, "observed": 41}}},
    )
    args.protocol_amendment = amendment
    deviations = validate_protocol_arguments(args)
    assert deviations == {"seed": {"expected": 42, "observed": 41}}
