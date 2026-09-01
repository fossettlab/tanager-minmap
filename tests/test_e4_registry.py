from __future__ import annotations

import copy
import math

import pytest

from tanager_rocks.e4_registry import (
    E4RegistryValidationError,
    registry_row_sort_key,
    validate_e4_registry,
)

NA = "not_applicable"


def _source_key(**changes: object) -> dict[str, object]:
    key: dict[str, object] = {
        "group": 1,
        "index": 7,
        "name": "alunite",
        "library": "splib07",
    }
    key.update(changes)
    return key


def _row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "row_kind": "metric",
        "endpoint_id": "exact:alunite:group_1",
        "mapping_class": "exact",
        "target": "alunite",
        "tanager_score": "mtmf:alunite",
        "l2b_group": 1,
        "source_mineral_keys": [_source_key()],
        "covariance_branch": "operational",
        "scale": "L",
        "metric": "rank_auc",
        "component": "estimate",
        "category": NA,
        "field": NA,
        "artifact_path": "tables/metrics.csv",
        "planned_status": "primary",
        "multiplicity_family": NA,
        "null_direction": 0.5,
        "allowed_terminal_statuses": ["complete", "unavailable"],
        "unavailable_reason": NA,
    }
    row.update(changes)
    return row


def _artifact_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "row_kind": "structural_artifact",
        "endpoint_id": NA,
        "mapping_class": NA,
        "target": NA,
        "tanager_score": NA,
        "l2b_group": NA,
        "source_mineral_keys": [],
        "covariance_branch": NA,
        "scale": NA,
        "metric": NA,
        "component": "manifest",
        "category": NA,
        "field": NA,
        "artifact_path": "input_manifest.json",
        "planned_status": "descriptive",
        "multiplicity_family": NA,
        "null_direction": NA,
        "allowed_terminal_statuses": ["complete", "unavailable"],
        "unavailable_reason": NA,
    }
    row.update(changes)
    return row


def _support_rows() -> list[dict[str, object]]:
    categories = (
        "included_joint_support",
        "incomplete_or_halo_m2_footprint",
        "footprint_crosses_m2_block_boundary",
        "invalid_l2b_glt_support",
        "invalid_tanager_qa_support",
        "nonfinite_tanager_score",
        "invalid_l2b_identity",
        "invalid_l2b_band_depth",
    )
    return [
        _row(
            row_kind="support",
            metric=NA,
            component="joint_support",
            category=category,
            planned_status="descriptive",
            null_direction=NA,
            artifact_path="tables/support_and_exclusions.csv",
        )
        for category in categories
    ]


def _count_rows(*, component: str = "cell_counts") -> list[dict[str, object]]:
    return [
        _row(
            row_kind="counts",
            metric=NA,
            component=component,
            category=category,
            field="mineral_identity",
            planned_status="counts_and_maps_only",
            null_direction=NA,
            artifact_path="tables/counts.csv",
        )
        for category in ("matched", "unmatched")
    ]


def _distribution_rows(*, field: str = "fit") -> list[dict[str, object]]:
    return [
        _row(
            row_kind="descriptive",
            metric=NA,
            component="distribution_summary",
            category=category,
            field=field,
            planned_status="descriptive",
            null_direction=NA,
            artifact_path="tables/fit_uncertainty_distributions.csv",
        )
        for category in ("matched", "unmatched", "tanager_no_call")
    ]


def _prevalence_row(**changes: object) -> dict[str, object]:
    row = _row(
        row_kind="descriptive",
        metric="l2b_id_prevalence",
        component="estimate",
        field="mineral_identity",
        planned_status="descriptive",
        null_direction=NA,
        artifact_path="tables/prevalence.csv",
    )
    row.update(changes)
    return row


def _map_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "row_kind": "map",
        "endpoint_id": NA,
        "mapping_class": NA,
        "target": NA,
        "tanager_score": NA,
        "l2b_group": 1,
        "source_mineral_keys": [],
        "covariance_branch": NA,
        "scale": NA,
        "metric": NA,
        "component": "map",
        "category": NA,
        "field": "mineral_identity",
        "artifact_path": "maps/group_1_mineral_identity.tif",
        "planned_status": "descriptive",
        "multiplicity_family": NA,
        "null_direction": NA,
        "allowed_terminal_statuses": ["complete", "unavailable"],
        "unavailable_reason": NA,
    }
    row.update(changes)
    return row


def _unresolved_failure_map_row(**changes: object) -> dict[str, object]:
    row = _artifact_row(
        component="map",
        field=NA,
        artifact_path="maps/failure_map.tif",
        planned_status="unavailable",
        allowed_terminal_statuses=["unavailable"],
        unavailable_reason="failure_map_scope_unresolved",
    )
    row.update(changes)
    return row


def _expand_inferential(row: dict[str, object]) -> list[dict[str, object]]:
    if not (
        row.get("row_kind") == "metric"
        and row.get("metric") in {"rank_auc", "spearman_band_depth"}
        and row.get("component") == "estimate"
    ):
        return [row]
    interval = copy.deepcopy(row)
    interval.update(
        row_kind="interval",
        component="paired_block_bootstrap_95_interval",
        artifact_path="tables/intervals.csv",
    )
    null = copy.deepcopy(row)
    null.update(
        row_kind="null",
        component="whole_block_spatial_null",
        artifact_path="tables/spatial_null_summary.csv",
    )
    if row["planned_status"] == "counts_and_maps_only":
        for companion in (interval, null):
            companion.update(
                planned_status="unavailable",
                allowed_terminal_statuses=["unavailable"],
                unavailable_reason="support_governance_counts_and_maps_only",
            )
    return [row, interval, null]


def _registry(*rows: dict[str, object], mode: str = "exact_primary") -> dict[str, object]:
    seeds = list(rows) if rows else [_row(), _artifact_row()]
    selected = [expanded for row in seeds for expanded in _expand_inferential(row)]
    selected.sort(key=registry_row_sort_key)
    return {
        "schema_version": "e4-scientific-output-registry/v2",
        "registry_id": "e4-goldfield-frozen",
        "ontology_crosswalk_sha256": "a" * 64,
        "source_inventory_sha256": "b" * 64,
        "decision_record_sha256": "c" * 64,
        "mode": mode,
        "bh_alpha": 0.05,
        "expected_files": sorted({str(row["artifact_path"]) for row in selected}),
        "rows": selected,
    }


def _assert_code(payload: dict[str, object], code: str) -> None:
    with pytest.raises(E4RegistryValidationError) as captured:
        validate_e4_registry(payload)
    assert captured.value.code == code
    assert captured.value.path.startswith("$")


def test_valid_exact_primary_registry_returns_structural_receipt():
    bh = _row(
        endpoint_id="broader:acid_sulfate:group_2",
        mapping_class="broader",
        l2b_group=2,
        source_mineral_keys=[
            _source_key(group=2, index=19, name="acid sulfate", library="splib07")
        ],
        covariance_branch="strict_inductive",
        scale="2L",
        metric="spearman_band_depth",
        planned_status="bh_secondary",
        multiplicity_family="compatible_mineral_secondary",
        null_direction=0.0,
    )
    payload = _registry(_row(), bh, _artifact_row())

    result = validate_e4_registry(payload)

    assert result.registry_id == "e4-goldfield-frozen"
    assert result.row_count == 7
    assert result.primary_row_count == 1
    assert result.bh_row_count == 1


def test_valid_all_exploratory_registry_has_no_selected_roles():
    exploratory = _row(
        planned_status="exploratory",
        covariance_branch="strict_inductive",
        scale="2L",
    )
    payload = _registry(exploratory, _artifact_row(), mode="all_exploratory")

    result = validate_e4_registry(payload)

    assert result.mode == "all_exploratory"
    assert result.primary_row_count == 0
    assert result.bh_row_count == 0


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda payload: payload.update(schema_version="e4-scientific-output-registry/v1"),
            "unsupported_schema_version",
        ),
        (lambda payload: payload.update(extra=True), "field_closure_violation"),
        (lambda payload: payload.pop("rows"), "field_closure_violation"),
        (lambda payload: payload.update(registry_id="registry-*"), "unsafe_or_dynamic_identifier"),
        (
            lambda payload: payload.update(ontology_crosswalk_sha256="A" * 64),
            "invalid_lowercase_sha256",
        ),
        (
            lambda payload: payload.update(source_inventory_sha256="a" * 63),
            "invalid_lowercase_sha256",
        ),
        (lambda payload: payload.update(bh_alpha=True), "bh_alpha_must_equal_0_05"),
        (lambda payload: payload.update(bh_alpha=0.0500001), "bh_alpha_must_equal_0_05"),
        (
            lambda payload: payload.update(
                expected_files=list(reversed(payload["expected_files"]))
            ),
            "unsorted_expected_files",
        ),
        (
            lambda payload: payload.update(
                expected_files=["input_manifest.json", "input_manifest.json"]
            ),
            "duplicate_expected_file",
        ),
        (
            lambda payload: payload.update(
                expected_files=["../metrics.csv", "input_manifest.json"]
            ),
            "unsafe_relative_posix_path",
        ),
        (
            lambda payload: payload.update(expected_files=["tables/*.csv", "input_manifest.json"]),
            "unsafe_relative_posix_path",
        ),
    ],
)
def test_top_level_contract_fails_closed(mutation, code: str):
    payload = _registry()
    mutation(payload)
    _assert_code(payload, code)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("mapping_class", "EXACT", "unsupported_vocabulary_value"),
        ("target", "any", "unsupported_vocabulary_value"),
        ("tanager_score", "mtmf:*", "unsupported_vocabulary_value"),
        ("covariance_branch", "both_primary", "unsupported_vocabulary_value"),
        ("scale", "l", "unsupported_vocabulary_value"),
        ("metric", "auc", "unsupported_vocabulary_value"),
        ("component", "dynamic", "unsupported_vocabulary_value"),
        ("category", "all", "unsupported_vocabulary_value"),
        ("field", "fit_*", "unsupported_vocabulary_value"),
        ("artifact_path", "/metrics.csv", "unsafe_relative_posix_path"),
        ("artifact_path", "tables//metrics.csv", "unsafe_relative_posix_path"),
        ("artifact_path", "tables\\metrics.csv", "unsafe_relative_posix_path"),
        ("null_direction", math.inf, "nonfinite_numeric_value"),
    ],
)
def test_row_vocabularies_paths_and_numbers_fail_closed(field: str, value: object, code: str):
    row = _row(**{field: value})
    payload = _registry(row, _artifact_row())
    if field == "artifact_path":
        payload["expected_files"] = sorted({str(item["artifact_path"]) for item in payload["rows"]})
    _assert_code(payload, code)


def test_unhashable_vocabulary_values_use_the_registry_exception_contract():
    _assert_code(
        _registry(_row(mapping_class=["exact"]), _artifact_row()),
        "unsupported_vocabulary_value",
    )
    _assert_code(
        _registry(_row(l2b_group=[1]), _artifact_row()),
        "unsupported_vocabulary_value",
    )


def test_row_and_source_object_key_closure_is_exact():
    row = _row()
    row["extra"] = "forbidden"
    _assert_code(_registry(row, _artifact_row()), "field_closure_violation")

    row = _row(source_mineral_keys=[{**_source_key(), "alias": "alunite"}])
    _assert_code(_registry(row, _artifact_row()), "field_closure_violation")


def test_source_keys_must_be_explicit_unique_sorted_and_same_group():
    first = _source_key(index=7, name="alunite")
    second = _source_key(index=8, name="alunite-2")
    payload = _registry(
        _row(source_mineral_keys=[second, first]),
        _artifact_row(),
    )
    _assert_code(payload, "unsorted_source_mineral_keys")

    payload = _registry(_row(source_mineral_keys=[first, copy.deepcopy(first)]), _artifact_row())
    _assert_code(payload, "duplicate_source_mineral_key")

    payload = _registry(_row(source_mineral_keys=[_source_key(group=2)]), _artifact_row())
    _assert_code(payload, "source_group_mismatch")

    payload = _registry(_row(source_mineral_keys=[_source_key(group=1.0)]), _artifact_row())
    _assert_code(payload, "invalid_source_mineral_group")

    payload = _registry(_row(source_mineral_keys=[_source_key(name="mineral_*")]), _artifact_row())
    _assert_code(payload, "wildcard_or_placeholder_forbidden")


def test_only_ontology_independent_artifacts_may_have_empty_source_keys():
    _assert_code(
        _registry(_row(source_mineral_keys=[]), _artifact_row()),
        "endpoint_row_requires_explicit_identity",
    )
    _assert_code(
        _registry(_artifact_row(source_mineral_keys=[_source_key()]), _row()),
        "ontology_independent_row_has_source_keys",
    )


def test_unmapped_rows_forbid_inferred_targets_and_inferential_status():
    unmapped = _row(
        row_kind="descriptive",
        endpoint_id="unmapped:group_1:index_7",
        mapping_class="unmapped",
        target=NA,
        tanager_score=NA,
        covariance_branch=NA,
        scale=NA,
        planned_status="descriptive",
        metric=NA,
        component="mineral_identity",
        field="mineral_identity",
        null_direction=NA,
    )
    validate_e4_registry(_registry(_row(), unmapped, _artifact_row()))

    inferred = copy.deepcopy(unmapped)
    inferred["target"] = "alunite"
    _assert_code(
        _registry(_row(), inferred, _artifact_row()), "unmapped_row_forbids_target_and_score"
    )

    inferential = copy.deepcopy(unmapped)
    inferential.update(
        row_kind="metric",
        target=NA,
        tanager_score=NA,
        covariance_branch="operational",
        scale="L",
        planned_status="exploratory",
        metric="rank_auc",
        component="estimate",
        field=NA,
        null_direction=0.5,
    )
    _assert_code(
        _registry(_row(), inferential, _artifact_row()), "inferential_status_requires_mapped_row"
    )


def test_bh_family_is_present_only_on_bh_rows():
    missing = _row(
        endpoint_id="exact:kaolinite:group_1",
        target="kaolinite",
        tanager_score="mtmf:kaolinite",
        source_mineral_keys=[_source_key(index=8, name="kaolinite")],
        planned_status="bh_secondary",
        multiplicity_family=NA,
    )
    _assert_code(_registry(_row(), missing, _artifact_row()), "unsupported_vocabulary_value")

    unknown = copy.deepcopy(missing)
    unknown["multiplicity_family"] = "another_secondary_family"
    _assert_code(_registry(_row(), unknown, _artifact_row()), "unsupported_vocabulary_value")

    forbidden = _row(multiplicity_family="compatible_mineral_secondary")
    _assert_code(_registry(forbidden, _artifact_row()), "multiplicity_family_forbidden")


def test_unavailable_rows_and_terminal_fallback_cannot_be_omitted():
    missing_fallback = _row(allowed_terminal_statuses=["complete"])
    _assert_code(
        _registry(missing_fallback, _artifact_row()),
        "unavailable_terminal_must_be_explicit",
    )

    unavailable = _row(
        planned_status="unavailable",
        allowed_terminal_statuses=["unavailable"],
        unavailable_reason="ontology_mapping_unavailable",
    )
    validate_e4_registry(_registry(unavailable, _artifact_row(), mode="all_exploratory"))

    unavailable["unavailable_reason"] = NA
    _assert_code(
        _registry(unavailable, _artifact_row(), mode="all_exploratory"),
        "unsafe_or_dynamic_identifier",
    )


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"mapping_class": "broader"}, "invalid_primary_row"),
        ({"covariance_branch": "strict_inductive"}, "invalid_primary_row"),
        ({"scale": "2L"}, "invalid_primary_row"),
        ({"metric": "spearman_band_depth", "null_direction": 0.0}, "invalid_primary_row"),
        ({"component": "whole_block_spatial_null"}, "inferential_row_shape_incompatible"),
        ({"row_kind": "interval"}, "inferential_row_shape_incompatible"),
    ],
)
def test_exact_primary_mode_requires_one_exact_operational_l_rank_auc_estimate(changes, code):
    _assert_code(_registry(_row(**changes), _artifact_row()), code)


def test_mode_cardinality_and_role_rules_fail_closed():
    second_primary = _row(
        endpoint_id="exact:kaolinite:group_1",
        target="kaolinite",
        tanager_score="mtmf:kaolinite",
        source_mineral_keys=[_source_key(index=8, name="kaolinite")],
    )
    _assert_code(
        _registry(_row(), second_primary, _artifact_row()),
        "exact_primary_mode_requires_one_primary",
    )
    _assert_code(
        _registry(_row(), _artifact_row(), mode="all_exploratory"),
        "all_exploratory_forbids_primary",
    )
    bh = _row(
        planned_status="bh_secondary",
        multiplicity_family="compatible_mineral_secondary",
    )
    result = validate_e4_registry(_registry(bh, _artifact_row(), mode="all_exploratory"))
    assert result.primary_row_count == 0
    assert result.bh_row_count == 1


def test_rows_must_be_sorted_and_scientific_identities_unique():
    descriptive = _row(
        row_kind="descriptive",
        metric="l2b_id_prevalence",
        component="estimate",
        field="mineral_identity",
        planned_status="descriptive",
        null_direction=NA,
    )
    payload = _registry(_row(), descriptive, _artifact_row())
    payload["rows"] = list(reversed(payload["rows"]))
    _assert_code(payload, "unsorted_registry_rows")

    duplicate = copy.deepcopy(_row())
    duplicate.update(
        artifact_path="tables/duplicate.csv",
        planned_status="unavailable",
        allowed_terminal_statuses=["unavailable"],
        unavailable_reason="duplicate_not_allowed",
    )
    _assert_code(
        _registry(_row(), duplicate, _artifact_row()),
        "duplicate_scientific_identity",
    )


def test_expected_files_and_row_artifacts_have_exact_bidirectional_closure():
    payload = _registry()
    payload["expected_files"].append("report.md")
    payload["expected_files"].sort()
    _assert_code(payload, "expected_file_without_registry_row")

    payload = _registry()
    payload["expected_files"].remove("tables/metrics.csv")
    _assert_code(payload, "row_artifact_not_declared")


def test_neutral_cross_field_matrix_accepts_closed_summary_families():
    rows = [
        _row(),
        *_support_rows(),
        *_count_rows(),
        *_count_rows(component="block_counts"),
        *_distribution_rows(field="fit"),
        *_distribution_rows(field="band_depth_uncertainty"),
        _prevalence_row(),
        _map_row(),
        _unresolved_failure_map_row(),
        _artifact_row(),
    ]

    result = validate_e4_registry(_registry(*rows))

    assert result.primary_row_count == 1
    assert result.bh_row_count == 0
    assert result.row_count == 25


@pytest.mark.parametrize(
    ("row", "code"),
    [
        (
            _row(
                row_kind="interval",
                component="whole_block_spatial_null",
                planned_status="exploratory",
            ),
            "inferential_row_shape_incompatible",
        ),
        (
            _row(
                row_kind="support",
                metric=NA,
                component="joint_support",
                category="matched",
                planned_status="descriptive",
                null_direction=NA,
            ),
            "support_row_shape_incompatible",
        ),
        (
            _row(
                row_kind="counts",
                metric=NA,
                component="cell_counts",
                category="tanager_no_call",
                field="mineral_identity",
                planned_status="counts_and_maps_only",
                null_direction=NA,
            ),
            "counts_row_shape_incompatible",
        ),
        (
            _row(
                row_kind="descriptive",
                metric=NA,
                component="distribution_summary",
                category="matched",
                field="band_depth",
                planned_status="descriptive",
                null_direction=NA,
            ),
            "descriptive_row_shape_incompatible",
        ),
        (
            _map_row(field="tanager_score"),
            "product_map_row_shape_incompatible",
        ),
        (
            _unresolved_failure_map_row(
                planned_status="descriptive",
                allowed_terminal_statuses=["complete", "unavailable"],
                unavailable_reason=NA,
            ),
            "structural_artifact_row_shape_incompatible",
        ),
    ],
)
def test_cross_field_matrix_rejects_semantically_impossible_rows(row, code: str):
    _assert_code(_registry(_row(), row, _artifact_row()), code)


def test_product_maps_are_group_specific_and_endpoint_free():
    _assert_code(
        _registry(_row(), _map_row(l2b_group=NA), _artifact_row()),
        "product_map_requires_group",
    )
    _assert_code(
        _registry(
            _row(),
            _map_row(endpoint_id="exact:alunite:group_1"),
            _artifact_row(),
        ),
        "endpoint_free_row_has_endpoint_dimensions",
    )


def test_inferential_rows_require_exact_estimate_interval_null_triplets():
    orphan = _row(
        row_kind="interval",
        endpoint_id="exact:kaolinite:group_1",
        target="kaolinite",
        tanager_score="mtmf:kaolinite",
        source_mineral_keys=[_source_key(index=8, name="kaolinite")],
        component="paired_block_bootstrap_95_interval",
        planned_status="exploratory",
        artifact_path="tables/orphan_interval.csv",
    )
    _assert_code(
        _registry(_row(), orphan, _artifact_row()),
        "inferential_triplet_incomplete",
    )

    payload = _registry(
        _row(),
        _row(
            endpoint_id="exact:kaolinite:group_1",
            target="kaolinite",
            tanager_score="mtmf:kaolinite",
            source_mineral_keys=[_source_key(index=8, name="kaolinite")],
            planned_status="exploratory",
        ),
        _artifact_row(),
    )
    companion = next(
        row
        for row in payload["rows"]
        if row["endpoint_id"] == "exact:kaolinite:group_1" and row["row_kind"] == "interval"
    )
    companion["planned_status"] = "bh_secondary"
    companion["multiplicity_family"] = "compatible_mineral_secondary"
    payload["rows"].sort(key=registry_row_sort_key)
    _assert_code(payload, "inferential_triplet_status_incompatible")


def test_counts_only_estimate_requires_explicit_unavailable_companions():
    counts_only = _row(
        endpoint_id="exact:kaolinite:group_1",
        target="kaolinite",
        tanager_score="mtmf:kaolinite",
        source_mineral_keys=[_source_key(index=8, name="kaolinite")],
        planned_status="counts_and_maps_only",
    )
    validate_e4_registry(_registry(_row(), counts_only, _artifact_row()))

    payload = _registry(_row(), counts_only, _artifact_row())
    companion = next(
        row
        for row in payload["rows"]
        if row["endpoint_id"] == "exact:kaolinite:group_1" and row["row_kind"] == "interval"
    )
    companion.update(
        planned_status="exploratory",
        allowed_terminal_statuses=["complete", "unavailable"],
        unavailable_reason=NA,
    )
    payload["rows"].sort(key=registry_row_sort_key)
    _assert_code(payload, "inferential_triplet_status_incompatible")


def test_support_count_and_distribution_summary_families_are_closed():
    _assert_code(
        _registry(_row(), *_support_rows()[:-1], _artifact_row()),
        "support_category_family_incomplete",
    )
    _assert_code(
        _registry(_row(), _count_rows()[0], _artifact_row()),
        "count_category_family_incomplete",
    )
    _assert_code(
        _registry(_row(), *_distribution_rows()[:-1], _artifact_row()),
        "distribution_category_family_incomplete",
    )


def test_raw_draw_files_are_structural_artifacts_not_inferential_rows():
    bootstrap_trace = _artifact_row(
        component="table",
        artifact_path="traces/bootstrap.csv",
    )
    null_trace = _artifact_row(
        component="table",
        artifact_path="traces/spatial_nulls.csv",
    )
    result = validate_e4_registry(_registry(_row(), bootstrap_trace, null_trace, _artifact_row()))
    assert result.primary_row_count == 1


def test_failure_map_reserved_path_and_unresolved_role_are_exclusive():
    completed_descriptive = _artifact_row(
        row_kind="descriptive_artifact",
        component="map",
        artifact_path="maps/failure_map.tif",
    )
    _assert_code(
        _registry(_row(), completed_descriptive, _artifact_row()),
        "failure_map_artifact_role_mismatch",
    )

    wrong_path = _unresolved_failure_map_row(artifact_path="maps/another_map.tif")
    _assert_code(
        _registry(_row(), wrong_path, _artifact_row()),
        "failure_map_scope_path_mismatch",
    )


@pytest.mark.parametrize("artifact_path", ["traces/bootstrap.csv", "traces/spatial_nulls.csv"])
def test_raw_trace_paths_reject_descriptive_artifact_roles(artifact_path: str):
    disguised = _artifact_row(
        row_kind="descriptive_artifact",
        component="table",
        artifact_path=artifact_path,
    )

    _assert_code(
        _registry(_row(), disguised, _artifact_row()),
        "raw_trace_artifact_role_mismatch",
    )
