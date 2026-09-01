"""Endpoint-free structural validation for the future E4 output registry.

This module validates only the closed grammar of a caller-supplied registry. It
does not generate registry rows, read source or result data, infer ontology
mappings, select endpoints, or assign primary/family roles. A successful result
therefore does not prove crosswalk completeness, scientific-identity
completeness, multiplicity-family completeness, or compatibility with the
bound ontology and decision artifacts. External crosswalk/decision compatibility
validation remains mandatory before preflight admission.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = "e4-scientific-output-registry/v2"
BH_ALPHA = 0.05
NOT_APPLICABLE = "not_applicable"

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "registry_id",
        "ontology_crosswalk_sha256",
        "source_inventory_sha256",
        "decision_record_sha256",
        "mode",
        "bh_alpha",
        "expected_files",
        "rows",
    }
)
ROW_FIELDS = frozenset(
    {
        "row_kind",
        "endpoint_id",
        "mapping_class",
        "target",
        "tanager_score",
        "l2b_group",
        "source_mineral_keys",
        "covariance_branch",
        "scale",
        "metric",
        "component",
        "category",
        "field",
        "artifact_path",
        "planned_status",
        "multiplicity_family",
        "null_direction",
        "allowed_terminal_statuses",
        "unavailable_reason",
    }
)
SOURCE_MINERAL_KEY_FIELDS = frozenset({"group", "index", "name", "library"})

MODES = frozenset({"exact_primary", "all_exploratory"})
ROW_KINDS = frozenset(
    {
        "metric",
        "interval",
        "null",
        "support",
        "descriptive",
        "counts",
        "map",
        "structural_artifact",
        "descriptive_artifact",
    }
)
ONTOLOGY_INDEPENDENT_ROW_KINDS = frozenset({"structural_artifact", "descriptive_artifact"})
ENDPOINT_FREE_ROW_KINDS = ONTOLOGY_INDEPENDENT_ROW_KINDS | {"map"}
ENDPOINT_ROW_KINDS = frozenset({"metric", "interval", "null", "support", "counts", "descriptive"})
MAPPING_CLASSES = frozenset({"exact", "broader", "unmapped", NOT_APPLICABLE})
TARGETS = frozenset(
    {
        "alunite",
        "kaolinite",
        "dickite",
        "jarosite",
        "hematite",
        "goethite",
        "gypsum",
        "muscovite",
    }
)
TANAGER_SCORES = frozenset(
    {
        *(f"mtmf:{target}" for target in TARGETS),
        "feature:al_oh_doublet",
        "feature:jarosite",
        "feature:gypsum_carbonate",
        "feature:fe_oxide",
    }
)
L2B_GROUPS = frozenset({1, 2})
COVARIANCE_BRANCHES = frozenset({"operational", "strict_inductive", NOT_APPLICABLE})
SCALES = frozenset({"L", "2L", NOT_APPLICABLE})
METRICS = frozenset({"rank_auc", "spearman_band_depth", "l2b_id_prevalence", NOT_APPLICABLE})
INFERENTIAL_METRICS = frozenset({"rank_auc", "spearman_band_depth"})
BH_FAMILIES = frozenset({"compatible_mineral_secondary"})
COMPONENTS = frozenset(
    {
        "estimate",
        "paired_block_bootstrap_95_interval",
        "whole_block_spatial_null",
        "joint_support",
        "cell_counts",
        "block_counts",
        "distribution_summary",
        "mineral_identity",
        "band_depth",
        "fit",
        "band_depth_uncertainty",
        "failure_class",
        "manifest",
        "table",
        "report",
        "map",
        NOT_APPLICABLE,
    }
)
CATEGORIES = frozenset(
    {
        "matched",
        "unmatched",
        "tanager_no_call",
        "included_joint_support",
        "incomplete_or_halo_m2_footprint",
        "footprint_crosses_m2_block_boundary",
        "invalid_l2b_glt_support",
        "invalid_tanager_qa_support",
        "nonfinite_tanager_score",
        "invalid_l2b_identity",
        "invalid_l2b_band_depth",
        NOT_APPLICABLE,
    }
)
FIELDS = frozenset(
    {
        "mineral_identity",
        "band_depth",
        "fit",
        "band_depth_uncertainty",
        "tanager_score",
        "failure_class",
        NOT_APPLICABLE,
    }
)
PLANNED_STATUSES = frozenset(
    {
        "primary",
        "bh_secondary",
        "exploratory",
        "descriptive",
        "counts_and_maps_only",
        "unavailable",
    }
)
TERMINAL_STATUSES = frozenset({"complete", "unavailable"})
INFERENTIAL_ROW_KINDS = frozenset({"metric", "interval", "null"})
SUPPORT_CATEGORIES = frozenset(
    {
        "included_joint_support",
        "incomplete_or_halo_m2_footprint",
        "footprint_crosses_m2_block_boundary",
        "invalid_l2b_glt_support",
        "invalid_tanager_qa_support",
        "nonfinite_tanager_score",
        "invalid_l2b_identity",
        "invalid_l2b_band_depth",
    }
)
DISTRIBUTION_CATEGORIES = frozenset({"matched", "unmatched", "tanager_no_call"})
PRODUCT_MAP_FIELDS = frozenset({"mineral_identity", "band_depth", "fit", "band_depth_uncertainty"})
INFERENTIAL_COMPONENTS = {
    "metric": "estimate",
    "interval": "paired_block_bootstrap_95_interval",
    "null": "whole_block_spatial_null",
}
INFERENTIAL_ACTIVE_STATUSES = frozenset({"primary", "bh_secondary", "exploratory"})
ESTIMATE_PLANNED_STATUSES = INFERENTIAL_ACTIVE_STATUSES | {
    "counts_and_maps_only",
    "unavailable",
}
COMPANION_PLANNED_STATUSES = INFERENTIAL_ACTIVE_STATUSES | {"unavailable"}
DESCRIPTIVE_PLANNED_STATUSES = frozenset({"descriptive", "unavailable"})
COUNT_MAP_PLANNED_STATUSES = frozenset({"descriptive", "counts_and_maps_only", "unavailable"})
COUNT_CATEGORIES = frozenset({"matched", "unmatched"})
COUNT_COMPONENTS = frozenset({"cell_counts", "block_counts"})
SUMMARY_DISTRIBUTION_FIELDS = frozenset({"fit", "band_depth_uncertainty"})
FAILURE_MAP_ARTIFACT_PATH = "maps/failure_map.tif"
RAW_TRACE_ARTIFACT_PATHS = frozenset(
    {
        "traces/bootstrap.csv",
        "traces/spatial_nulls.csv",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_PATH_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DYNAMIC_MARKERS = re.compile(r"[*?\[\]{}<>\\]|\$\{|\.\.\.")
_PLACEHOLDER_WORDS = frozenset(
    {
        "all",
        "any",
        "contains",
        "dynamic",
        "each",
        "endswith",
        "glob",
        "placeholder",
        "regex",
        "startswith",
        "substring",
        "tbd",
        "template",
        "todo",
        "unknown",
        "wildcard",
    }
)


class E4RegistryValidationError(ValueError):
    """Raised when an E4 registry violates the closed structural contract."""

    def __init__(self, code: str, path: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code} at {path}")


@dataclass(frozen=True)
class E4RegistryValidationResult:
    """A structural-validation receipt; it is not scientific admission."""

    schema_version: str
    registry_id: str
    mode: str
    expected_file_count: int
    row_count: int
    primary_row_count: int
    bh_row_count: int


def _fail(code: str, path: str) -> None:
    raise E4RegistryValidationError(code, path)


def _reject_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("nonfinite_numeric_value", path)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _require_exact_fields(value: Any, expected: frozenset[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("object_required", path)
    observed = set(value)
    if observed != expected:
        _fail("field_closure_violation", path)
    return value


def _require_controlled(value: Any, allowed: frozenset[Any], path: str) -> Any:
    try:
        accepted = value in allowed
    except TypeError:
        accepted = False
    if not accepted:
        _fail("unsupported_vocabulary_value", path)
    return value


def _require_identifier(value: Any, path: str, *, allow_not_applicable: bool = False) -> str:
    if not isinstance(value, str):
        _fail("string_required", path)
    if allow_not_applicable and value == NOT_APPLICABLE:
        return value
    if value == NOT_APPLICABLE or not _IDENTIFIER.fullmatch(value) or _contains_placeholder(value):
        _fail("unsafe_or_dynamic_identifier", path)
    return value


def _contains_placeholder(value: str) -> bool:
    if _DYNAMIC_MARKERS.search(value) or value.startswith("^") or value.endswith("$"):
        return True
    words = {word for word in re.split(r"[^a-z0-9]+", value.casefold()) if word}
    return bool(words & _PLACEHOLDER_WORDS)


def _require_exact_source_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail("nonempty_exact_source_text_required", path)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail("control_character_forbidden", path)
    if _contains_placeholder(value):
        _fail("wildcard_or_placeholder_forbidden", path)
    return value


def _require_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail("unsafe_relative_posix_path", path)
    normalized = PurePosixPath(value)
    if normalized.is_absolute() or str(normalized) != value:
        _fail("unsafe_relative_posix_path", path)
    if any(
        part in {"", ".", ".."} or not _PATH_SEGMENT.fullmatch(part) for part in normalized.parts
    ):
        _fail("unsafe_relative_posix_path", path)
    if _contains_placeholder(value):
        _fail("unsafe_relative_posix_path", path)
    return value


def _source_key_sort_key(value: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (value["group"], value["index"], value["name"], value["library"])


def _validate_source_keys(value: Any, path: str) -> tuple[tuple[int, int, str, str], ...]:
    if not isinstance(value, list):
        _fail("source_mineral_keys_list_required", path)
    keys: list[tuple[int, int, str, str]] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        key = _require_exact_fields(item, SOURCE_MINERAL_KEY_FIELDS, item_path)
        group = key["group"]
        mineral_index = key["index"]
        if isinstance(group, bool) or not isinstance(group, int) or group not in L2B_GROUPS:
            _fail("invalid_source_mineral_group", f"{item_path}.group")
        if (
            isinstance(mineral_index, bool)
            or not isinstance(mineral_index, int)
            or mineral_index <= 0
        ):
            _fail("invalid_source_mineral_index", f"{item_path}.index")
        name = _require_exact_source_text(key["name"], f"{item_path}.name")
        library = _require_exact_source_text(key["library"], f"{item_path}.library")
        keys.append((group, mineral_index, name, library))
    if len(set(keys)) != len(keys):
        _fail("duplicate_source_mineral_key", path)
    if keys != sorted(keys):
        _fail("unsorted_source_mineral_keys", path)
    return tuple(keys)


def _scientific_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    if row["row_kind"] in ONTOLOGY_INDEPENDENT_ROW_KINDS:
        return (
            row["row_kind"],
            row["component"],
            row["category"],
            row["field"],
            row["artifact_path"],
        )
    return (
        row["row_kind"],
        row["endpoint_id"],
        row["mapping_class"],
        row["target"],
        row["tanager_score"],
        row["l2b_group"],
        tuple(_source_key_sort_key(item) for item in row["source_mineral_keys"]),
        row["covariance_branch"],
        row["scale"],
        row["metric"],
        row["component"],
        row["category"],
        row["field"],
    )


def registry_row_sort_key(row: Mapping[str, Any]) -> str:
    """Return the documented canonical ordering key for already-shaped rows."""
    return json.dumps(
        (_scientific_identity(row), row["artifact_path"], row["planned_status"]),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _validate_cross_field_shape(record: Mapping[str, Any], path: str) -> None:
    """Enforce the neutral row-kind compatibility matrix.

    This closes structural combinations only. It does not decide which source
    minerals map to a target, which endpoint is primary, or which rows belong
    to the externally frozen BH family.
    """
    row_kind = record["row_kind"]
    metric = record["metric"]
    component = record["component"]
    category = record["category"]
    field = record["field"]
    planned = record["planned_status"]
    unmapped_inventory = (
        row_kind == "descriptive"
        and record["mapping_class"] == "unmapped"
        and metric == NOT_APPLICABLE
        and component == "mineral_identity"
        and category == NOT_APPLICABLE
        and field == "mineral_identity"
    )

    if row_kind in ENDPOINT_ROW_KINDS and not unmapped_inventory:
        if record["covariance_branch"] == NOT_APPLICABLE or record["scale"] == NOT_APPLICABLE:
            _fail("endpoint_row_requires_branch_and_scale", path)
    if unmapped_inventory and not (
        record["covariance_branch"] == record["scale"] == NOT_APPLICABLE
    ):
        _fail("unmapped_inventory_forbids_branch_and_scale", path)

    if row_kind in INFERENTIAL_ROW_KINDS:
        allowed_statuses = (
            ESTIMATE_PLANNED_STATUSES if row_kind == "metric" else COMPANION_PLANNED_STATUSES
        )
        if not (
            record["mapping_class"] in {"exact", "broader"}
            and metric in INFERENTIAL_METRICS
            and component == INFERENTIAL_COMPONENTS[row_kind]
            and category == field == NOT_APPLICABLE
            and planned in allowed_statuses
        ):
            _fail("inferential_row_shape_incompatible", path)
        return

    if row_kind == "support":
        if not (
            record["mapping_class"] in {"exact", "broader"}
            and metric == NOT_APPLICABLE
            and component == "joint_support"
            and category in SUPPORT_CATEGORIES
            and field == NOT_APPLICABLE
            and planned in COUNT_MAP_PLANNED_STATUSES
        ):
            _fail("support_row_shape_incompatible", path)
        return

    if row_kind == "counts":
        if not (
            record["mapping_class"] in {"exact", "broader"}
            and metric == NOT_APPLICABLE
            and component in COUNT_COMPONENTS
            and category in COUNT_CATEGORIES
            and field == "mineral_identity"
            and planned in COUNT_MAP_PLANNED_STATUSES
        ):
            _fail("counts_row_shape_incompatible", path)
        return

    if row_kind == "descriptive":
        prevalence = (
            record["mapping_class"] in {"exact", "broader"}
            and metric == "l2b_id_prevalence"
            and component == "estimate"
            and category == NOT_APPLICABLE
            and field == "mineral_identity"
        )
        distribution = (
            record["mapping_class"] in {"exact", "broader"}
            and metric == NOT_APPLICABLE
            and component == "distribution_summary"
            and category in DISTRIBUTION_CATEGORIES
            and field in SUMMARY_DISTRIBUTION_FIELDS
        )
        if planned not in DESCRIPTIVE_PLANNED_STATUSES or not (
            prevalence or distribution or unmapped_inventory
        ):
            _fail("descriptive_row_shape_incompatible", path)
        return

    if row_kind == "map":
        if not (
            record["l2b_group"] in L2B_GROUPS
            and component == "map"
            and category == NOT_APPLICABLE
            and field in PRODUCT_MAP_FIELDS
            and planned in DESCRIPTIVE_PLANNED_STATUSES
        ):
            _fail("product_map_row_shape_incompatible", path)
        return

    if row_kind == "structural_artifact":
        ordinary = (
            component in {"manifest", "table", "report"}
            and category == field == NOT_APPLICABLE
            and planned in DESCRIPTIVE_PLANNED_STATUSES
        )
        unresolved_failure_map = (
            component == "map"
            and category == field == NOT_APPLICABLE
            and planned == "unavailable"
            and record["unavailable_reason"] == "failure_map_scope_unresolved"
        )
        if not (ordinary or unresolved_failure_map):
            _fail("structural_artifact_row_shape_incompatible", path)
        return

    if row_kind == "descriptive_artifact":
        if not (
            component in {"table", "report", "map"}
            and category == field == NOT_APPLICABLE
            and planned in DESCRIPTIVE_PLANNED_STATUSES
        ):
            _fail("descriptive_artifact_row_shape_incompatible", path)
        return

    _fail("row_kind_shape_unhandled", path)


def _validate_reserved_artifact_roles(record: Mapping[str, Any], path: str) -> None:
    """Bind reserved artifact paths to their only admissible structural roles."""
    artifact = record["artifact_path"]
    unresolved_failure_map = (
        record["row_kind"] == "structural_artifact"
        and record["component"] == "map"
        and record["planned_status"] == "unavailable"
        and record["unavailable_reason"] == "failure_map_scope_unresolved"
    )
    if artifact == FAILURE_MAP_ARTIFACT_PATH and not unresolved_failure_map:
        _fail("failure_map_artifact_role_mismatch", f"{path}.artifact_path")
    if unresolved_failure_map and artifact != FAILURE_MAP_ARTIFACT_PATH:
        _fail("failure_map_scope_path_mismatch", f"{path}.artifact_path")
    if artifact in RAW_TRACE_ARTIFACT_PATHS and not (
        record["row_kind"] == "structural_artifact" and record["component"] == "table"
    ):
        _fail("raw_trace_artifact_role_mismatch", f"{path}.artifact_path")


def _closure_key(row: Mapping[str, Any], omitted: frozenset[str]) -> str:
    value = {key: row[key] for key in sorted(ROW_FIELDS - omitted)}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_summary_family_closure(rows: list[Mapping[str, Any]]) -> None:
    """Require closed summary families without registering raw draws as rows."""
    inferential: dict[str, list[Mapping[str, Any]]] = {}
    support: dict[str, set[str]] = {}
    counts: dict[str, set[str]] = {}
    distributions: dict[str, set[str]] = {}

    for row in rows:
        row_kind = row["row_kind"]
        if row_kind in INFERENTIAL_ROW_KINDS:
            key = _closure_key(
                row,
                frozenset(
                    {
                        "row_kind",
                        "component",
                        "artifact_path",
                        "planned_status",
                        "multiplicity_family",
                        "allowed_terminal_statuses",
                        "unavailable_reason",
                    }
                ),
            )
            inferential.setdefault(key, []).append(row)
        elif row_kind == "support":
            key = _closure_key(row, frozenset({"category", "artifact_path"}))
            support.setdefault(key, set()).add(row["category"])
        elif row_kind == "counts":
            key = _closure_key(row, frozenset({"category", "artifact_path"}))
            counts.setdefault(key, set()).add(row["category"])
        elif row_kind == "descriptive" and row["component"] == "distribution_summary":
            key = _closure_key(row, frozenset({"category", "artifact_path"}))
            distributions.setdefault(key, set()).add(row["category"])

    expected_triplet = set(INFERENTIAL_COMPONENTS.items())
    for members in inferential.values():
        observed = {(row["row_kind"], row["component"]) for row in members}
        if len(members) != 3 or observed != expected_triplet:
            _fail("inferential_triplet_incomplete", "$.rows")
        by_kind = {row["row_kind"]: row for row in members}
        estimate_status = by_kind["metric"]["planned_status"]
        companion_statuses = {
            by_kind["interval"]["planned_status"],
            by_kind["null"]["planned_status"],
        }
        if estimate_status in INFERENTIAL_ACTIVE_STATUSES:
            if companion_statuses != {estimate_status}:
                _fail("inferential_triplet_status_incompatible", "$.rows")
            if any(
                row["multiplicity_family"] != by_kind["metric"]["multiplicity_family"]
                for row in members
            ):
                _fail("inferential_triplet_family_incompatible", "$.rows")
        elif estimate_status in {"counts_and_maps_only", "unavailable"}:
            if companion_statuses != {"unavailable"}:
                _fail("inferential_triplet_status_incompatible", "$.rows")
        else:  # pragma: no cover - row-level matrix rejects this first
            _fail("inferential_triplet_status_incompatible", "$.rows")
    if any(categories != SUPPORT_CATEGORIES for categories in support.values()):
        _fail("support_category_family_incomplete", "$.rows")
    if any(categories != COUNT_CATEGORIES for categories in counts.values()):
        _fail("count_category_family_incomplete", "$.rows")
    if any(categories != DISTRIBUTION_CATEGORIES for categories in distributions.values()):
        _fail("distribution_category_family_incomplete", "$.rows")


def _validate_row(row: Any, index: int, expected_files: frozenset[str]) -> None:
    path = f"$.rows[{index}]"
    record = _require_exact_fields(row, ROW_FIELDS, path)
    row_kind = _require_controlled(record["row_kind"], ROW_KINDS, f"{path}.row_kind")
    independent = row_kind in ENDPOINT_FREE_ROW_KINDS
    endpoint_id = _require_identifier(
        record["endpoint_id"], f"{path}.endpoint_id", allow_not_applicable=independent
    )
    mapping = _require_controlled(record["mapping_class"], MAPPING_CLASSES, f"{path}.mapping_class")
    target = _require_controlled(record["target"], TARGETS | {NOT_APPLICABLE}, f"{path}.target")
    score = _require_controlled(
        record["tanager_score"], TANAGER_SCORES | {NOT_APPLICABLE}, f"{path}.tanager_score"
    )
    group = record["l2b_group"]
    if not (
        (isinstance(group, int) and not isinstance(group, bool) and group in L2B_GROUPS)
        or group == NOT_APPLICABLE
    ):
        _fail("unsupported_vocabulary_value", f"{path}.l2b_group")
    source_keys = _validate_source_keys(
        record["source_mineral_keys"], f"{path}.source_mineral_keys"
    )
    branch = _require_controlled(
        record["covariance_branch"], COVARIANCE_BRANCHES, f"{path}.covariance_branch"
    )
    scale = _require_controlled(record["scale"], SCALES, f"{path}.scale")
    metric = _require_controlled(record["metric"], METRICS, f"{path}.metric")
    _require_controlled(record["component"], COMPONENTS, f"{path}.component")
    _require_controlled(record["category"], CATEGORIES, f"{path}.category")
    _require_controlled(record["field"], FIELDS, f"{path}.field")
    artifact = _require_path(record["artifact_path"], f"{path}.artifact_path")
    if artifact not in expected_files:
        _fail("row_artifact_not_declared", f"{path}.artifact_path")
    planned = _require_controlled(
        record["planned_status"], PLANNED_STATUSES, f"{path}.planned_status"
    )

    if independent:
        if not (
            endpoint_id == mapping == target == score == branch == scale == metric == NOT_APPLICABLE
        ):
            _fail("endpoint_free_row_has_endpoint_dimensions", path)
        if row_kind == "map":
            if group not in L2B_GROUPS:
                _fail("product_map_requires_group", f"{path}.l2b_group")
        elif group != NOT_APPLICABLE:
            _fail("ontology_independent_row_has_group", f"{path}.l2b_group")
        if source_keys:
            _fail("ontology_independent_row_has_source_keys", f"{path}.source_mineral_keys")
    else:
        if endpoint_id == NOT_APPLICABLE or not source_keys:
            _fail("endpoint_row_requires_explicit_identity", path)
        if mapping == NOT_APPLICABLE or group == NOT_APPLICABLE:
            _fail("endpoint_row_requires_mapping_and_group", path)
        if any(key[0] != group for key in source_keys):
            _fail("source_group_mismatch", f"{path}.source_mineral_keys")
        if mapping in {"exact", "broader"}:
            if target == NOT_APPLICABLE or score == NOT_APPLICABLE:
                _fail("mapped_row_requires_target_and_score", path)
        elif target != NOT_APPLICABLE or score != NOT_APPLICABLE:
            _fail("unmapped_row_forbids_target_and_score", path)

    if planned in {"primary", "bh_secondary", "exploratory"}:
        if row_kind not in INFERENTIAL_ROW_KINDS or metric not in INFERENTIAL_METRICS:
            _fail("inferential_status_incompatible_with_row", path)
        if mapping not in {"exact", "broader"}:
            _fail("inferential_status_requires_mapped_row", path)
    if mapping == "unmapped" and planned not in {
        "descriptive",
        "counts_and_maps_only",
        "unavailable",
    }:
        _fail("unmapped_status_incompatibility", f"{path}.planned_status")

    family = record["multiplicity_family"]
    if planned == "bh_secondary":
        _require_controlled(family, BH_FAMILIES, f"{path}.multiplicity_family")
    elif family != NOT_APPLICABLE:
        _fail("multiplicity_family_forbidden", f"{path}.multiplicity_family")

    null_direction = record["null_direction"]
    if metric == "rank_auc":
        if isinstance(null_direction, bool) or null_direction != 0.5:
            _fail("invalid_null_direction", f"{path}.null_direction")
    elif metric == "spearman_band_depth":
        if isinstance(null_direction, bool) or null_direction != 0.0:
            _fail("invalid_null_direction", f"{path}.null_direction")
    elif null_direction != NOT_APPLICABLE:
        _fail("null_direction_forbidden", f"{path}.null_direction")

    terminal = record["allowed_terminal_statuses"]
    if not isinstance(terminal, list) or not terminal:
        _fail("terminal_status_list_required", f"{path}.allowed_terminal_statuses")
    if any(value not in TERMINAL_STATUSES for value in terminal):
        _fail("unsupported_terminal_status", f"{path}.allowed_terminal_statuses")
    if len(set(terminal)) != len(terminal):
        _fail("duplicate_terminal_status", f"{path}.allowed_terminal_statuses")
    if terminal != sorted(terminal):
        _fail("unsorted_terminal_statuses", f"{path}.allowed_terminal_statuses")
    expected_terminal = ["unavailable"] if planned == "unavailable" else ["complete", "unavailable"]
    if terminal != expected_terminal:
        _fail("unavailable_terminal_must_be_explicit", f"{path}.allowed_terminal_statuses")

    reason = record["unavailable_reason"]
    if planned == "unavailable":
        _require_identifier(reason, f"{path}.unavailable_reason")
    elif reason != NOT_APPLICABLE:
        _fail("unavailable_reason_forbidden", f"{path}.unavailable_reason")

    _validate_cross_field_shape(record, path)
    _validate_reserved_artifact_roles(record, path)


def validate_e4_registry(payload: Mapping[str, Any]) -> E4RegistryValidationResult:
    """Validate an endpoint-free E4 registry grammar and return a receipt.

    The input must already contain every externally chosen row and role. This
    function neither repairs nor completes it. Passing this check cannot prove
    compatibility or completeness against the ontology crosswalk, source
    inventory, decision record, or BH family; those hash-bound checks remain a
    mandatory external admission step.

    Parameters
    ----------
    payload
        In-memory registry object. No path or scientific source is accepted.

    Returns
    -------
    E4RegistryValidationResult
        Counts and identity for the structurally valid object.

    Raises
    ------
    E4RegistryValidationError
        If any closed-schema, vocabulary, ordering, identity, or compatibility
        rule fails. ``code`` and ``path`` identify the failure class/location.
    """
    _reject_nonfinite(payload)
    registry = _require_exact_fields(payload, TOP_LEVEL_FIELDS, "$")
    if registry["schema_version"] != SCHEMA_VERSION:
        _fail("unsupported_schema_version", "$.schema_version")
    registry_id = _require_identifier(registry["registry_id"], "$.registry_id")
    for field in (
        "ontology_crosswalk_sha256",
        "source_inventory_sha256",
        "decision_record_sha256",
    ):
        if not isinstance(registry[field], str) or not _SHA256.fullmatch(registry[field]):
            _fail("invalid_lowercase_sha256", f"$.{field}")
    mode = _require_controlled(registry["mode"], MODES, "$.mode")
    alpha = registry["bh_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha != BH_ALPHA:
        _fail("bh_alpha_must_equal_0_05", "$.bh_alpha")

    expected_files = registry["expected_files"]
    if not isinstance(expected_files, list) or not expected_files:
        _fail("nonempty_expected_files_required", "$.expected_files")
    paths = [
        _require_path(value, f"$.expected_files[{index}]")
        for index, value in enumerate(expected_files)
    ]
    if len(set(paths)) != len(paths):
        _fail("duplicate_expected_file", "$.expected_files")
    if paths != sorted(paths):
        _fail("unsorted_expected_files", "$.expected_files")
    expected_set = frozenset(paths)

    rows = registry["rows"]
    if not isinstance(rows, list) or not rows:
        _fail("nonempty_rows_required", "$.rows")
    for index, row in enumerate(rows):
        _validate_row(row, index, expected_set)
    referenced_artifacts = {row["artifact_path"] for row in rows}
    if referenced_artifacts != expected_set:
        _fail("expected_file_without_registry_row", "$.expected_files")

    identities = [_scientific_identity(row) for row in rows]
    encoded_identities = [
        json.dumps(identity, ensure_ascii=False, separators=(",", ":")) for identity in identities
    ]
    if len(set(encoded_identities)) != len(encoded_identities):
        _fail("duplicate_scientific_identity", "$.rows")
    row_keys = [registry_row_sort_key(row) for row in rows]
    if row_keys != sorted(row_keys):
        _fail("unsorted_registry_rows", "$.rows")

    _validate_summary_family_closure(rows)

    primary_rows = [row for row in rows if row["planned_status"] == "primary"]
    primary_tests = [
        row
        for row in primary_rows
        if row["row_kind"] == "metric" and row["component"] == "estimate"
    ]
    bh_tests = [
        row
        for row in rows
        if row["planned_status"] == "bh_secondary"
        and row["row_kind"] == "metric"
        and row["component"] == "estimate"
    ]
    if mode == "exact_primary":
        if len(primary_tests) != 1:
            _fail("exact_primary_mode_requires_one_primary", "$.rows")
        primary = primary_tests[0]
        if not (
            primary["row_kind"] == "metric"
            and primary["mapping_class"] == "exact"
            and primary["covariance_branch"] == "operational"
            and primary["scale"] == "L"
            and primary["metric"] == "rank_auc"
            and primary["component"] == "estimate"
        ):
            _fail("invalid_primary_row", "$.rows")
    else:
        if primary_rows:
            _fail("all_exploratory_forbids_primary", "$.rows")

    # BH-family membership in all-exploratory mode is an unresolved external
    # method decision.  Structural validation therefore records a caller-
    # supplied closed family but does not choose whether that mode has one.

    return E4RegistryValidationResult(
        schema_version=SCHEMA_VERSION,
        registry_id=registry_id,
        mode=mode,
        expected_file_count=len(paths),
        row_count=len(rows),
        primary_row_count=len(primary_tests),
        bh_row_count=len(bh_tests),
    )


__all__ = [
    "BH_ALPHA",
    "E4RegistryValidationError",
    "E4RegistryValidationResult",
    "NOT_APPLICABLE",
    "SCHEMA_VERSION",
    "registry_row_sort_key",
    "validate_e4_registry",
]
