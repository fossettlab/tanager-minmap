"""Endpoint-blind resource admission for the frozen repeatability full run.

This module deliberately operates only on redacted timing metadata, scheduler
telemetry, and file sizes.  It neither opens a repeatability result nor admits
scientific conclusions.  A ``PASS-64`` means only that the fixed 64-GiB,
four-CPU, 24-hour allocation is conservatively supported by the timing pilot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

GIB = 1024**3
RESOURCE_ADMISSION_SCHEMA_VERSION = "1.0"
RESOURCE_ADMISSION_IDENTITY = "repeatability-resource-admission-v1"
FOOTPRINT_SCHEMA_VERSION = "1.0"
SCHEDULER_RECORD_SCHEMA_VERSION = "1.0"
RULE_RELATIVE_PATH = Path("docs/m2_repeatability_resource_admission_rule_v1.json")
SOURCE_MANIFEST_RELATIVE_PATH = Path("docs/m2_repeatability_bigmem_source_manifest.sha256")
VERIFIER_SCRIPT_RELATIVE_PATH = Path("scripts/admit_repeatability_resources.py")
VERIFIER_MODULE_RELATIVE_PATH = Path("src/tanager_rocks/repeatability_resources.py")
MAX_UINT63 = 2**63 - 1
SOURCE_MANIFEST_ENTRY_COUNT = 47
BOOTSTRAP_REPLICATES = 10_000
NULL_REPLICATES = 9_999
TIMING_SCHEMA_VERSION = "2.0"
SCIENTIFIC_EXECUTION_IDENTITY = "paired-complete-block-metric-contract-v2"
RESAMPLED_METRIC_COMPONENTS = (
    "spearman",
    "transferred_iou",
    "transferred_dice",
    "transferred_prevalence_ratio",
    "transferred_boundary_distance_m",
    "rank_relative_iou",
    "rank_relative_dice",
    "rank_relative_prevalence_ratio",
    "rank_relative_boundary_distance_m",
    "rockwell_auc",
    "rockwell_balanced_accuracy",
    "rockwell_macro_f1",
)
FROZEN_LAYERS = (
    "feature:al_oh_doublet",
    "feature:fe_oxide",
    "feature:gypsum_carbonate",
    "feature:jarosite",
    "mtmf:alunite",
    "mtmf:dickite",
    "mtmf:goethite",
    "mtmf:hematite",
    "mtmf:jarosite",
    "mtmf:kaolinite",
    "mtmf:muscovite",
)
FROZEN_PAIRS = (
    ("primary", "bingham", "20250911_191523_58_4001", "20250911_191547_88_4001"),
    ("primary", "goldfield", "20240925_185504_87_4001", "20240925_185509_74_4001"),
    ("primary", "goldfield", "20240925_185504_87_4001", "20250222_190233_00_4001"),
    ("primary", "goldfield", "20240925_185504_87_4001", "20250222_190237_16_4001"),
    ("primary", "goldfield", "20240925_185504_87_4001", "20250222_190241_32_4001"),
    ("secondary", "goldfield", "20240925_185509_74_4001", "20250222_190233_00_4001"),
    ("secondary", "goldfield", "20240925_185509_74_4001", "20250222_190237_16_4001"),
    ("secondary", "goldfield", "20240925_185509_74_4001", "20250222_190241_32_4001"),
    ("secondary", "goldfield", "20250222_190233_00_4001", "20250222_190237_16_4001"),
    ("secondary", "goldfield", "20250222_190233_00_4001", "20250222_190241_32_4001"),
    ("secondary", "goldfield", "20250222_190237_16_4001", "20250222_190241_32_4001"),
)

_RULE = {
    "schema_version": RESOURCE_ADMISSION_SCHEMA_VERSION,
    "resource_admission_identity": RESOURCE_ADMISSION_IDENTITY,
    "allocation": {"cpus": 4, "memory_bytes": 64 * GIB, "wall_seconds": 24 * 3600},
    "full_task_count": 121,
    "source_manifest_entry_count": SOURCE_MANIFEST_ENTRY_COUNT,
    "bounds": {
        "memory_multiplier": 4,
        "memory_upper_bound_bytes": 48 * GIB,
        "wall_non_task_multiplier": 4,
        "wall_task_multiplier": 2,
        "wall_upper_bound_seconds": 18 * 3600,
        "disk_footprint_multiplier": 4,
        "disk_fixed_overhead_bytes": 2 * GIB,
        "disk_free_minimum_bytes": 16 * GIB,
    },
    "decision_values": ["PASS-64", "HOLD", "FAIL"],
    "endpoint_seal": {"contains_endpoint_values": False, "accepted_scientific_result": False},
    "schemas": {
        "scheduler_record": {
            "schema_version": SCHEDULER_RECORD_SCHEMA_VERSION,
            "required_fields": [
                "schema_version",
                "job_id",
                "state",
                "exit_code",
                "oom_killed",
                "elapsed_raw_seconds",
                "timelimit_raw_seconds",
                "alloc_cpus",
                "req_mem_bytes",
                "max_rss_bytes",
                "max_vm_bytes",
                "max_disk_read_bytes",
                "max_disk_write_bytes",
                "node",
            ],
        },
        "footprint_manifest": {
            "schema_version": FOOTPRINT_SCHEMA_VERSION,
            "required_fields": ["schema_version", "files"],
            "file_fields": ["root", "path", "byte_size"],
        },
        "resource_admission": {
            "schema_version": RESOURCE_ADMISSION_SCHEMA_VERSION,
            "required_fields": [
                "schema_version",
                "resource_admission_identity",
                "accepted_scientific_result",
                "contains_endpoint_values",
                "input_digests",
                "status",
                "full_mode_admitted",
                "rule_path",
                "rule_sha256",
                "source_manifest_path",
                "source_manifest_sha256",
                "verifier_script_path",
                "verifier_script_sha256",
                "verifier_module_path",
                "verifier_module_sha256",
                "timing_pilot_sha256",
                "execution_manifest_sha256",
                "scheduler_record_sha256",
                "footprint_manifest_sha256",
                "telemetry",
                "computed_bounds",
                "failure_codes",
            ],
        },
    },
}


class ResourceAdmissionError(ValueError):
    """A fail-closed resource-admission validation error with a safe code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code.replace("_", " "))


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _open_directory_nofollow(path: Path, *, create: bool = False) -> int:
    """Open a directory through no-follow component traversal."""
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor or os.sep, flags)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, flags | nofollow, dir_fd=descriptor)
            except NotADirectoryError:
                # macOS firmlinks (not symbolic links) reject O_NOFOLLOW. Bind
                # the fallback open to an lstat/fstat identity comparison.
                before = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                    raise
                child = os.open(part, flags, dir_fd=descriptor)
                after = os.fstat(child)
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    os.close(child)
                    raise ResourceAdmissionError("directory_changed")
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                child = os.open(part, flags | nofollow, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_bound_regular(path: Path) -> tuple[bytes, str, _FileIdentity]:
    """Read one regular single-link file without following any path component."""
    try:
        parent_fd = _open_directory_nofollow(path.parent)
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        finally:
            os.close(parent_fd)
    except OSError as error:
        raise ResourceAdmissionError("nonregular_input") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ResourceAdmissionError("nonregular_input")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        before_identity = _FileIdentity(
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = _FileIdentity(
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise ResourceAdmissionError("input_changed_during_read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ResourceAdmissionError("input_changed_during_read")
        return payload, hashlib.sha256(payload).hexdigest(), before_identity
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    """Return a SHA-256 for one regular, non-symlink file."""
    return _read_bound_regular(path)[1]


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ResourceAdmissionError(f"malformed_{label}") from error
    if not isinstance(value, dict):
        raise ResourceAdmissionError(f"malformed_{label}")
    return value


def read_strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read a finite, duplicate-key-free JSON object from a regular file."""
    return _parse_strict_json_object(_read_bound_regular(path)[0], label=label)


def _require_exact(value: Any, expected: Any, code: str) -> None:
    if value != expected:
        raise ResourceAdmissionError(code)


def _require_uint(value: Any, code: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_UINT63:
        raise ResourceAdmissionError(code)
    if positive and value == 0:
        raise ResourceAdmissionError(code)
    return value


def _require_sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ResourceAdmissionError(code)
    try:
        int(value, 16)
    except ValueError as error:
        raise ResourceAdmissionError(code) from error
    return value.lower()


def _require_safe_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or any(ord(character) < 32 for character in value):
        raise ResourceAdmissionError(code)
    return value


def _require_relative_path(value: Any, code: str) -> str:
    text = _require_safe_text(value, code)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or text in {"", "."}
        or path.as_posix() != text
        or "//" in text
    ):
        raise ResourceAdmissionError(code)
    return text


def load_rule(rule_path: Path) -> tuple[dict[str, Any], str]:
    """Validate the canonical frozen rule and return it with its byte digest."""
    payload, digest, _ = _read_bound_regular(rule_path)
    rule = _parse_strict_json_object(payload, label="rule")
    _require_exact(rule, _RULE, "rule_contract_mismatch")
    expected_bytes = (json.dumps(_RULE, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    _require_exact(payload, expected_bytes, "rule_not_byte_canonical")
    return rule, digest


def _expected_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for role, site, anchor, repeat in FROZEN_PAIRS:
        for layer in FROZEN_LAYERS:
            tasks.append(
                {
                    "index": len(tasks),
                    "task_id": ":".join((role, site, anchor, repeat, layer)),
                    "comparison_role": role,
                    "site_id": site,
                    "anchor_scene_id": anchor,
                    "repeat_scene_id": repeat,
                    "layer": layer,
                }
            )
    return tasks


def _validate_resampling_schedule(value: Any) -> None:
    if not isinstance(value, dict):
        raise ResourceAdmissionError("timing_schedule")
    _require_exact(
        set(value),
        {"contains_endpoint_values", "n_complete_paired_overlap_blocks", "components"},
        "timing_schedule_field_closure",
    )
    _require_exact(value.get("contains_endpoint_values"), False, "timing_schedule_endpoint_seal")
    n_blocks = _require_uint(value.get("n_complete_paired_overlap_blocks"), "timing_blocks")
    if n_blocks < 2:
        raise ResourceAdmissionError("timing_blocks")
    components = value.get("components")
    if not isinstance(components, dict):
        raise ResourceAdmissionError("timing_schedule_components")
    _require_exact(
        set(components), set(RESAMPLED_METRIC_COMPONENTS), "timing_schedule_component_closure"
    )
    expected_null = math.factorial(n_blocks) if n_blocks < 8 else NULL_REPLICATES
    expected_null = min(expected_null, NULL_REPLICATES)
    for metric in RESAMPLED_METRIC_COMPONENTS:
        record = components.get(metric)
        if not isinstance(record, dict):
            raise ResourceAdmissionError("timing_schedule_component")
        _require_exact(set(record), {"bootstrap", "spatial_null"}, "timing_schedule_component")
        _require_exact(
            record.get("bootstrap"),
            {"scheduled_replicates": BOOTSTRAP_REPLICATES},
            "timing_schedule_bootstrap",
        )
        expected_spatial_null = (
            {"status": "not_applicable"}
            if metric.endswith("prevalence_ratio")
            else {"status": "scheduled", "scheduled_replicates": expected_null}
        )
        _require_exact(record.get("spatial_null"), expected_spatial_null, "timing_schedule_null")


def _validate_timing_capsule(
    value: Mapping[str, Any], timing_output_root: Path, full_task_count: int
) -> tuple[float, str]:
    expected_fields = {
        "schema_version",
        "mode",
        "scientific_execution_identity",
        "accepted_scientific_result",
        "contains_endpoint_values",
        "execution_manifest",
        "execution_manifest_sha256",
        "task_id",
        "workers",
        "bootstrap_replicates",
        "null_replicates_maximum",
        "resampling_branch_schedule",
        "elapsed_seconds",
        "result_sha256",
    }
    _require_exact(set(value), expected_fields, "timing_capsule_field_closure")
    _require_exact(value.get("schema_version"), TIMING_SCHEMA_VERSION, "timing_capsule_schema")
    _require_exact(value.get("mode"), "timing", "timing_capsule_mode")
    _require_exact(
        value.get("scientific_execution_identity"),
        SCIENTIFIC_EXECUTION_IDENTITY,
        "timing_capsule_identity",
    )
    _require_exact(value.get("accepted_scientific_result"), False, "timing_capsule_result_seal")
    _require_exact(value.get("contains_endpoint_values"), False, "timing_capsule_endpoint_seal")
    _require_exact(value.get("workers"), 4, "timing_workers")
    _require_exact(
        value.get("bootstrap_replicates"), BOOTSTRAP_REPLICATES, "timing_bootstrap_replicates"
    )
    _require_exact(value.get("null_replicates_maximum"), NULL_REPLICATES, "timing_null_replicates")
    _validate_resampling_schedule(value.get("resampling_branch_schedule"))
    elapsed = value.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ResourceAdmissionError("timing_elapsed_seconds")
    execution_sha = _require_sha256(
        value.get("execution_manifest_sha256"), "timing_execution_manifest_sha256"
    )
    result_sha = _require_sha256(value.get("result_sha256"), "timing_result_sha256")
    execution_path = timing_output_root / "execution_manifest.json"
    progress_path = timing_output_root / "progress.json"
    recorded_execution_path = value.get("execution_manifest")
    if not isinstance(recorded_execution_path, str):
        raise ResourceAdmissionError("timing_execution_manifest_path")
    _require_exact(
        Path(recorded_execution_path).resolve(),
        execution_path.resolve(),
        "timing_execution_manifest_path",
    )
    _require_exact(sha256_file(execution_path), execution_sha, "timing_execution_manifest_sha256")
    execution = read_strict_json_object(execution_path, label="execution_manifest")
    expected_execution_fields = {
        "schema_version",
        "scientific_execution_identity",
        "permitted_actions",
        "accepted_scientific_result",
        "resume_validates_completed_results",
        "protocol_artifacts",
        "block_handoffs",
        "source_inventory",
        "member_order",
        "resampling",
        "compute_controls",
    }
    _require_exact(set(execution), expected_execution_fields, "timing_execution_field_closure")
    _require_exact(
        execution.get("schema_version"), TIMING_SCHEMA_VERSION, "timing_execution_schema"
    )
    _require_exact(
        execution.get("scientific_execution_identity"),
        SCIENTIFIC_EXECUTION_IDENTITY,
        "timing_execution_identity",
    )
    _require_exact(execution.get("accepted_scientific_result"), False, "timing_execution_seal")
    _require_exact(execution.get("permitted_actions"), ["timing_pilot", "full"], "timing_actions")
    _require_exact(
        execution.get("resume_validates_completed_results"), True, "timing_resume_contract"
    )
    compute_controls = execution.get("compute_controls")
    if not isinstance(compute_controls, dict):
        raise ResourceAdmissionError("timing_compute_controls")
    _require_exact(compute_controls.get("workers"), 4, "timing_workers")
    _require_exact(
        compute_controls.get("task_result_order"),
        "frozen pair order then sorted manifest-declared layer",
        "timing_task_order_contract",
    )
    resampling = execution.get("resampling")
    if not isinstance(resampling, dict):
        raise ResourceAdmissionError("timing_resampling_contract")
    _require_exact(
        resampling.get("bootstrap_replicates"), BOOTSTRAP_REPLICATES, "timing_bootstrap_replicates"
    )
    _require_exact(
        resampling.get("null_replicates_maximum"), NULL_REPLICATES, "timing_null_replicates"
    )
    _require_exact(
        resampling.get("metric_components"),
        list(RESAMPLED_METRIC_COMPONENTS),
        "timing_metric_components",
    )
    member_order = execution.get("member_order")
    if not isinstance(member_order, dict) or set(member_order) != {
        "scenes",
        "complete_block_ids",
        "tasks",
    }:
        raise ResourceAdmissionError("timing_member_order")
    _require_exact(
        member_order.get("scenes"),
        {
            "bingham": ["20250911_191523_58_4001", "20250911_191547_88_4001"],
            "goldfield": [
                "20240925_185504_87_4001",
                "20240925_185509_74_4001",
                "20250222_190233_00_4001",
                "20250222_190237_16_4001",
                "20250222_190241_32_4001",
            ],
        },
        "timing_scene_order",
    )
    complete_blocks = member_order.get("complete_block_ids")
    if not isinstance(complete_blocks, dict) or set(complete_blocks) != {"bingham", "goldfield"}:
        raise ResourceAdmissionError("timing_block_order")
    for block_ids in complete_blocks.values():
        if (
            not isinstance(block_ids, list)
            or not block_ids
            or any(
                _require_uint(block_id, "timing_block_id", positive=True) < 1
                for block_id in block_ids
            )
            or block_ids != sorted(set(block_ids))
        ):
            raise ResourceAdmissionError("timing_block_order")
    expected_tasks = member_order.get("tasks")
    frozen_tasks = _expected_tasks()
    _require_exact(full_task_count, len(frozen_tasks), "timing_frozen_task_count")
    _require_exact(expected_tasks, frozen_tasks, "timing_task_order")
    task_ids = [task.get("task_id") for task in expected_tasks]
    if not all(isinstance(task_id, str) and task_id for task_id in task_ids):
        raise ResourceAdmissionError("timing_task_ids")
    if len(set(task_ids)) != full_task_count:
        raise ResourceAdmissionError("timing_task_ids")
    progress = read_strict_json_object(progress_path, label="progress_ledger")
    expected_progress_fields = {
        "schema_version",
        "execution_manifest_sha256",
        "run_status",
        "accepted_final_manifest",
        "tasks",
    }
    _require_exact(set(progress), expected_progress_fields, "timing_progress_field_closure")
    _require_exact(progress.get("schema_version"), TIMING_SCHEMA_VERSION, "timing_progress_schema")
    stable_execution_sha = hashlib.sha256(
        json.dumps(execution, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    _require_exact(
        progress.get("execution_manifest_sha256"),
        stable_execution_sha,
        "timing_progress_execution_sha256",
    )
    _require_exact(progress.get("run_status"), "timing_pilot_complete", "timing_progress_status")
    _require_exact(
        progress.get("accepted_final_manifest"), False, "timing_progress_final_acceptance"
    )
    progress_tasks = progress.get("tasks")
    if not isinstance(progress_tasks, list) or len(progress_tasks) != full_task_count:
        raise ResourceAdmissionError("timing_progress_task_count")
    _require_exact(
        [row.get("task_id") if isinstance(row, dict) else None for row in progress_tasks],
        task_ids,
        "timing_progress_task_order",
    )
    completed_count = 0
    metadata_fields = {"status", "attempts", "elapsed_seconds", "result_path", "result_sha256"}
    for task, row in zip(expected_tasks, progress_tasks, strict=True):
        if not isinstance(row, dict) or set(row) != set(task) | metadata_fields:
            raise ResourceAdmissionError("timing_progress_row_closure")
        for key, expected in task.items():
            _require_exact(row.get(key), expected, "timing_progress_task_metadata")
        attempts = _require_uint(row.get("attempts"), "timing_progress_attempts")
        if row.get("task_id") == value.get("task_id"):
            completed_count += 1
            _require_exact(row.get("status"), "completed", "timing_progress_pilot_status")
            _require_exact(attempts, 1, "timing_progress_pilot_attempts")
            _require_exact(row.get("result_path"), None, "timing_progress_pilot_result_path")
            _require_exact(row.get("result_sha256"), result_sha, "timing_progress_result_sha256")
            _require_exact(row.get("elapsed_seconds"), elapsed, "timing_progress_elapsed_seconds")
        else:
            _require_exact(row.get("status"), "pending", "timing_progress_nonpilot_status")
            _require_exact(attempts, 0, "timing_progress_nonpilot_attempts")
            for key in ("elapsed_seconds", "result_path", "result_sha256"):
                _require_exact(row.get(key), None, "timing_progress_nonpilot_metadata")
    _require_exact(completed_count, 1, "timing_progress_pilot_count")
    return float(elapsed), execution_sha


def _validate_scheduler_record(
    value: Mapping[str, Any], rule: Mapping[str, Any]
) -> tuple[dict[str, int], list[str]]:
    expected_fields = {
        "schema_version",
        "job_id",
        "state",
        "exit_code",
        "oom_killed",
        "elapsed_raw_seconds",
        "timelimit_raw_seconds",
        "alloc_cpus",
        "req_mem_bytes",
        "max_rss_bytes",
        "max_vm_bytes",
        "max_disk_read_bytes",
        "max_disk_write_bytes",
        "node",
    }
    _require_exact(set(value), expected_fields, "scheduler_field_closure")
    _require_exact(value.get("schema_version"), SCHEDULER_RECORD_SCHEMA_VERSION, "scheduler_schema")
    _require_safe_text(value.get("job_id"), "scheduler_job_id")
    _require_safe_text(value.get("node"), "scheduler_node")
    _require_safe_text(value.get("state"), "scheduler_state_value")
    _require_safe_text(value.get("exit_code"), "scheduler_exit_code_value")
    if not isinstance(value.get("oom_killed"), bool):
        raise ResourceAdmissionError("scheduler_oom_flag")
    allocation = rule["allocation"]
    telemetry = {
        "elapsed_raw_seconds": _require_uint(value.get("elapsed_raw_seconds"), "scheduler_elapsed"),
        "timelimit_raw_seconds": _require_uint(
            value.get("timelimit_raw_seconds"), "scheduler_timelimit", positive=True
        ),
        "alloc_cpus": _require_uint(value.get("alloc_cpus"), "scheduler_cpus", positive=True),
        "req_mem_bytes": _require_uint(
            value.get("req_mem_bytes"), "scheduler_memory", positive=True
        ),
        "max_rss_bytes": _require_uint(
            value.get("max_rss_bytes"), "scheduler_max_rss", positive=True
        ),
        "max_vm_bytes": _require_uint(value.get("max_vm_bytes"), "scheduler_max_vm"),
        "max_disk_read_bytes": _require_uint(
            value.get("max_disk_read_bytes"), "scheduler_disk_read"
        ),
        "max_disk_write_bytes": _require_uint(
            value.get("max_disk_write_bytes"), "scheduler_disk_write"
        ),
    }
    failures: list[str] = []
    if value.get("state") != "COMPLETED":
        failures.append("scheduler_state")
    if value.get("exit_code") != "0:0":
        failures.append("scheduler_exit_code")
    if value.get("oom_killed") is not False:
        failures.append("scheduler_oom")
    if telemetry["timelimit_raw_seconds"] != allocation["wall_seconds"]:
        failures.append("scheduler_timelimit")
    if telemetry["alloc_cpus"] != allocation["cpus"]:
        failures.append("scheduler_cpus")
    if telemetry["req_mem_bytes"] != allocation["memory_bytes"]:
        failures.append("scheduler_memory")
    if telemetry["elapsed_raw_seconds"] > telemetry["timelimit_raw_seconds"]:
        failures.append("scheduler_elapsed_exceeds_limit")
    if telemetry["max_rss_bytes"] > telemetry["req_mem_bytes"]:
        failures.append("scheduler_rss_exceeds_request")
    return telemetry, failures


def _paths_overlap(left: Path, right: Path) -> bool:
    left_abs = Path(os.path.abspath(left))
    right_abs = Path(os.path.abspath(right))
    return left_abs == right_abs or left_abs in right_abs.parents or right_abs in left_abs.parents


def ensure_output_outside_roots(output: Path, *roots: Path) -> None:
    """Reject evidence output paths that would mutate an inventoried root."""
    if any(_paths_overlap(output, root) for root in roots):
        raise ResourceAdmissionError("output_inside_footprint_root")


def _snapshot_tree(
    root_name: str, root: Path
) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]]]:
    files: list[dict[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    try:
        root_fd = _open_directory_nofollow(root)
    except OSError as error:
        raise ResourceAdmissionError("footprint_root") from error
    try:
        root_stat = os.fstat(root_fd)
        identities.append(
            (
                root_name,
                ".",
                root_stat.st_dev,
                root_stat.st_ino,
                root_stat.st_mtime_ns,
                root_stat.st_ctime_ns,
            )
        )

        def visit(directory_fd: int, relative_parts: tuple[str, ...]) -> None:
            try:
                with os.scandir(directory_fd) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
            except OSError as error:
                raise ResourceAdmissionError("footprint_scan") from error
            for entry in entries:
                if entry.name in {".", ".."} or "/" in entry.name:
                    raise ResourceAdmissionError("footprint_path")
                relative = PurePosixPath(*relative_parts, entry.name).as_posix()
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise ResourceAdmissionError("footprint_scan") from error
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ResourceAdmissionError("footprint_symlink")
                if stat.S_ISDIR(entry_stat.st_mode):
                    try:
                        child_fd = os.open(
                            entry.name,
                            os.O_RDONLY
                            | os.O_DIRECTORY
                            | os.O_CLOEXEC
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=directory_fd,
                        )
                    except OSError as error:
                        raise ResourceAdmissionError("footprint_scan") from error
                    try:
                        child_stat = os.fstat(child_fd)
                        if (child_stat.st_dev, child_stat.st_ino) != (
                            entry_stat.st_dev,
                            entry_stat.st_ino,
                        ):
                            raise ResourceAdmissionError("footprint_changed")
                        identities.append(
                            (
                                root_name,
                                relative,
                                child_stat.st_dev,
                                child_stat.st_ino,
                                child_stat.st_mtime_ns,
                                child_stat.st_ctime_ns,
                            )
                        )
                        visit(child_fd, (*relative_parts, entry.name))
                    finally:
                        os.close(child_fd)
                    continue
                if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                    raise ResourceAdmissionError("footprint_nonregular")
                try:
                    file_fd = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                except OSError as error:
                    raise ResourceAdmissionError("footprint_scan") from error
                try:
                    bound_stat = os.fstat(file_fd)
                finally:
                    os.close(file_fd)
                identity = (
                    root_name,
                    relative,
                    bound_stat.st_dev,
                    bound_stat.st_ino,
                    bound_stat.st_size,
                    bound_stat.st_mtime_ns,
                    bound_stat.st_ctime_ns,
                )
                if identity[2:] != (
                    entry_stat.st_dev,
                    entry_stat.st_ino,
                    entry_stat.st_size,
                    entry_stat.st_mtime_ns,
                    entry_stat.st_ctime_ns,
                ):
                    raise ResourceAdmissionError("footprint_changed")
                identities.append(identity)
                files.append({"root": root_name, "path": relative, "byte_size": bound_stat.st_size})

        visit(root_fd, ())
    finally:
        os.close(root_fd)
    return files, identities


def _snapshot_footprint(
    timing_root: Path, runtime_root: Path
) -> tuple[dict[str, Any], tuple[tuple[Any, ...], ...]]:
    if _paths_overlap(timing_root, runtime_root):
        raise ResourceAdmissionError("footprint_roots_overlap")
    files: list[dict[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    for root_name, root in (("timing_output", timing_root), ("runtime", runtime_root)):
        root_files, root_identities = _snapshot_tree(root_name, root)
        files.extend(root_files)
        identities.extend(root_identities)
    return (
        {"schema_version": FOOTPRINT_SCHEMA_VERSION, "files": files},
        tuple(identities),
    )


def build_footprint_manifest(timing_root: Path, runtime_root: Path) -> dict[str, Any]:
    """Return a deterministic, no-follow footprint of the redacted timing roots."""
    return _snapshot_footprint(timing_root, runtime_root)[0]


def _validate_footprint_manifest(
    value: Mapping[str, Any], timing_root: Path, runtime_root: Path
) -> tuple[int, tuple[tuple[Any, ...], ...]]:
    _require_exact(set(value), {"schema_version", "files"}, "footprint_field_closure")
    _require_exact(value.get("schema_version"), FOOTPRINT_SCHEMA_VERSION, "footprint_schema")
    files = value.get("files")
    if not isinstance(files, list):
        raise ResourceAdmissionError("footprint_files")
    observed, identities = _snapshot_footprint(timing_root, runtime_root)
    _require_exact(files, observed["files"], "footprint_mismatch")
    return sum(row["byte_size"] for row in files), identities


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize canonical resource evidence without a timestamp or host state."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create one admission document without overwrite or link following."""
    if not path.name or path.name in {".", ".."}:
        raise ResourceAdmissionError("resource_output_path")
    try:
        parent_fd = _open_directory_nofollow(path.parent, create=True)
    except OSError as error:
        raise ResourceAdmissionError("resource_output_parent") from error
    parent_identity = os.fstat(parent_fd)
    expected_bytes = canonical_json_bytes(value)
    temporary = f".{path.name}.{secrets.token_hex(12)}.partial"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            temporary,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
        rebound_parent = _open_directory_nofollow(path.parent)
        try:
            rebound_identity = os.fstat(rebound_parent)
            if (parent_identity.st_dev, parent_identity.st_ino) != (
                rebound_identity.st_dev,
                rebound_identity.st_ino,
            ):
                raise ResourceAdmissionError("resource_output_parent_changed")
        finally:
            os.close(rebound_parent)
        observed_bytes, _, _ = _read_bound_regular(path)
        _require_exact(observed_bytes, expected_bytes, "resource_output_changed")
    except FileExistsError as error:
        raise ResourceAdmissionError("resource_output_exists") from error
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(parent_fd)


def _input_digest(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except ResourceAdmissionError:
        return None


_MANIFEST_LINE = re.compile(r"^(?P<sha>[0-9a-f]{64})  (?P<path>[^\x00-\x1f\x7f]+)$")


def _manifest_target(repo_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts:
        raise ResourceAdmissionError("source_manifest_path")
    if pure.parts[0] == "..":
        if len(pure.parts) < 3 or pure.parts[1] != "tanager-spec" or ".." in pure.parts[2:]:
            raise ResourceAdmissionError("source_manifest_path")
        return repo_root.parent.joinpath(*pure.parts[1:])
    if ".." in pure.parts or pure.as_posix() != relative:
        raise ResourceAdmissionError("source_manifest_path")
    return repo_root.joinpath(*pure.parts)


def verify_source_manifest(
    source_manifest_path: Path, *, expected_count: int
) -> tuple[str, tuple[tuple[str, str, _FileIdentity], ...]]:
    """Verify the exact sorted source closure without following links."""
    payload, manifest_sha, _ = _read_bound_regular(source_manifest_path)
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise ResourceAdmissionError("source_manifest_encoding") from error
    if not text.endswith("\n") or "\r" in text:
        raise ResourceAdmissionError("source_manifest_format")
    lines = text.splitlines()
    if len(lines) != expected_count or lines != sorted(lines):
        raise ResourceAdmissionError("source_manifest_closure")
    repo_root = source_manifest_path.parent.parent
    observed: list[tuple[str, str, _FileIdentity]] = []
    seen: set[str] = set()
    for line in lines:
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ResourceAdmissionError("source_manifest_format")
        relative = match.group("path")
        if relative in seen:
            raise ResourceAdmissionError("source_manifest_duplicate")
        seen.add(relative)
        target = _manifest_target(repo_root, relative)
        _, digest, identity = _read_bound_regular(target)
        _require_exact(digest, match.group("sha"), "source_manifest_entry_digest")
        observed.append((relative, digest, identity))
    return manifest_sha, tuple(observed)


def produce_resource_admission(
    *,
    timing_pilot_path: Path,
    scheduler_record_path: Path,
    footprint_manifest_path: Path,
    timing_output_root: Path,
    runtime_root: Path,
    free_bytes: Any,
    source_manifest_path: Path,
    rule_path: Path,
    verifier_script_path: Path,
    verifier_module_path: Path,
) -> dict[str, Any]:
    """Build a sanitized PASS-64, HOLD, or FAIL admission document.

    Structural evidence failures return ``FAIL`` with a closed failure code.  A
    valid timing artifact that exceeds a resource bound returns ``HOLD``.
    Neither outcome accepts a scientific result.
    """
    input_digests = {
        "timing_pilot_sha256": _input_digest(timing_pilot_path),
        "scheduler_record_sha256": _input_digest(scheduler_record_path),
        "footprint_manifest_sha256": _input_digest(footprint_manifest_path),
        "source_manifest_sha256": _input_digest(source_manifest_path),
        "rule_sha256": _input_digest(rule_path),
        "verifier_script_sha256": _input_digest(verifier_script_path),
        "verifier_module_sha256": _input_digest(verifier_module_path),
    }
    base: dict[str, Any] = {
        "schema_version": RESOURCE_ADMISSION_SCHEMA_VERSION,
        "resource_admission_identity": RESOURCE_ADMISSION_IDENTITY,
        "accepted_scientific_result": False,
        "contains_endpoint_values": False,
        "input_digests": input_digests,
    }
    try:
        if timing_pilot_path != timing_output_root / "timing_pilot.json":
            raise ResourceAdmissionError("timing_capsule_path")
        for external in (
            scheduler_record_path,
            footprint_manifest_path,
            source_manifest_path,
            rule_path,
            verifier_script_path,
            verifier_module_path,
        ):
            if _paths_overlap(external, timing_output_root) or _paths_overlap(
                external, runtime_root
            ):
                raise ResourceAdmissionError("control_input_inside_footprint_root")
        rule, rule_sha = load_rule(rule_path)
        if input_digests["rule_sha256"] != rule_sha:
            raise ResourceAdmissionError("rule_digest")
        source_sha, source_entries = verify_source_manifest(
            source_manifest_path, expected_count=rule["source_manifest_entry_count"]
        )
        script_sha = sha256_file(verifier_script_path)
        module_sha = sha256_file(verifier_module_path)
        timing = read_strict_json_object(timing_pilot_path, label="timing_capsule")
        pilot_seconds, execution_manifest_sha = _validate_timing_capsule(
            timing, timing_output_root, rule["full_task_count"]
        )
        scheduler = read_strict_json_object(scheduler_record_path, label="scheduler_record")
        telemetry, scheduler_failures = _validate_scheduler_record(scheduler, rule)
        footprint = read_strict_json_object(footprint_manifest_path, label="footprint_manifest")
        footprint_bytes, footprint_identities = _validate_footprint_manifest(
            footprint, timing_output_root, runtime_root
        )
        free = _require_uint(free_bytes, "free_bytes")
        bounds = rule["bounds"]
        overhead_seconds = max(telemetry["elapsed_raw_seconds"] - pilot_seconds, 0.0)
        memory_bound = bounds["memory_multiplier"] * telemetry["max_rss_bytes"]
        wall_bound = (
            bounds["wall_non_task_multiplier"] * overhead_seconds
            + bounds["wall_task_multiplier"] * rule["full_task_count"] * pilot_seconds
        )
        disk_bound = (
            bounds["disk_footprint_multiplier"] * footprint_bytes
            + bounds["disk_fixed_overhead_bytes"]
        )
        required_free = max(bounds["disk_free_minimum_bytes"], 2 * disk_bound)
        resource_failures: list[str] = []
        if memory_bound > bounds["memory_upper_bound_bytes"]:
            resource_failures.append("memory_bound")
        if wall_bound > bounds["wall_upper_bound_seconds"]:
            resource_failures.append("wall_bound")
        if free < required_free:
            resource_failures.append("disk_free")
        if pilot_seconds > telemetry["elapsed_raw_seconds"]:
            raise ResourceAdmissionError("timing_exceeds_scheduler_elapsed")
        if not all(
            math.isfinite(value)
            for value in (overhead_seconds, memory_bound, wall_bound, disk_bound, required_free)
        ):
            raise ResourceAdmissionError("computed_bound_nonfinite")
        final_footprint, final_footprint_identities = _snapshot_footprint(
            timing_output_root, runtime_root
        )
        _require_exact(final_footprint, footprint, "footprint_changed")
        _require_exact(final_footprint_identities, footprint_identities, "footprint_changed")
        final_source_sha, final_source_entries = verify_source_manifest(
            source_manifest_path, expected_count=rule["source_manifest_entry_count"]
        )
        _require_exact(final_source_sha, source_sha, "source_manifest_changed")
        _require_exact(final_source_entries, source_entries, "source_manifest_changed")
        for key, path in (
            ("scheduler_record_sha256", scheduler_record_path),
            ("footprint_manifest_sha256", footprint_manifest_path),
            ("rule_sha256", rule_path),
            ("verifier_script_sha256", verifier_script_path),
            ("verifier_module_sha256", verifier_module_path),
        ):
            _require_exact(sha256_file(path), input_digests[key], "control_input_changed")
        status = "FAIL" if scheduler_failures else ("HOLD" if resource_failures else "PASS-64")
        return {
            **base,
            "status": status,
            "full_mode_admitted": status == "PASS-64",
            "rule_path": RULE_RELATIVE_PATH.as_posix(),
            "rule_sha256": rule_sha,
            "source_manifest_path": SOURCE_MANIFEST_RELATIVE_PATH.as_posix(),
            "source_manifest_sha256": source_sha,
            "verifier_script_path": VERIFIER_SCRIPT_RELATIVE_PATH.as_posix(),
            "verifier_script_sha256": script_sha,
            "verifier_module_path": VERIFIER_MODULE_RELATIVE_PATH.as_posix(),
            "verifier_module_sha256": module_sha,
            "timing_pilot_sha256": sha256_file(timing_pilot_path),
            "execution_manifest_sha256": execution_manifest_sha,
            "scheduler_record_sha256": sha256_file(scheduler_record_path),
            "footprint_manifest_sha256": sha256_file(footprint_manifest_path),
            "telemetry": {
                "pilot_elapsed_seconds": pilot_seconds,
                "scheduler_elapsed_seconds": telemetry["elapsed_raw_seconds"],
                "non_task_elapsed_seconds": overhead_seconds,
                "max_rss_bytes": telemetry["max_rss_bytes"],
                "footprint_bytes": footprint_bytes,
                "free_bytes_before_full": free,
            },
            "computed_bounds": {
                "memory_bound_bytes": memory_bound,
                "wall_bound_seconds": wall_bound,
                "disk_bound_bytes": disk_bound,
                "required_free_bytes": required_free,
            },
            "failure_codes": scheduler_failures + resource_failures,
        }
    except ResourceAdmissionError as error:
        return {
            **base,
            "status": "FAIL",
            "full_mode_admitted": False,
            "rule_path": RULE_RELATIVE_PATH.as_posix(),
            "rule_sha256": input_digests["rule_sha256"],
            "source_manifest_path": SOURCE_MANIFEST_RELATIVE_PATH.as_posix(),
            "source_manifest_sha256": input_digests["source_manifest_sha256"],
            "verifier_script_path": VERIFIER_SCRIPT_RELATIVE_PATH.as_posix(),
            "verifier_script_sha256": input_digests["verifier_script_sha256"],
            "verifier_module_path": VERIFIER_MODULE_RELATIVE_PATH.as_posix(),
            "verifier_module_sha256": input_digests["verifier_module_sha256"],
            "timing_pilot_sha256": input_digests["timing_pilot_sha256"],
            "execution_manifest_sha256": None,
            "scheduler_record_sha256": input_digests["scheduler_record_sha256"],
            "footprint_manifest_sha256": input_digests["footprint_manifest_sha256"],
            "telemetry": {},
            "computed_bounds": {},
            "failure_codes": [error.code],
        }


def validate_resource_admission(
    *,
    resource_admission_path: Path,
    expected_sha256: str,
    expected_timing_pilot_sha256: str,
    expected_execution_manifest_sha256: str,
    rule_path: Path,
    source_manifest_path: Path,
    verifier_script_path: Path,
    verifier_module_path: Path,
) -> Path:
    """Admit exactly one reviewed PASS-64 record before full computation."""
    _require_sha256(expected_sha256, "resource_admission_sha256")
    _require_sha256(expected_timing_pilot_sha256, "timing_pilot_sha256")
    _require_sha256(expected_execution_manifest_sha256, "execution_manifest_sha256")
    try:
        admission_bytes, admission_sha256, _ = _read_bound_regular(resource_admission_path)
    except ResourceAdmissionError as error:
        raise ResourceAdmissionError("missing_resource_admission") from error
    _require_exact(admission_sha256, expected_sha256, "resource_admission_digest")
    value = _parse_strict_json_object(admission_bytes, label="resource_admission")
    required = {
        "schema_version",
        "resource_admission_identity",
        "accepted_scientific_result",
        "contains_endpoint_values",
        "input_digests",
        "status",
        "full_mode_admitted",
        "rule_path",
        "rule_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "verifier_script_path",
        "verifier_script_sha256",
        "verifier_module_path",
        "verifier_module_sha256",
        "timing_pilot_sha256",
        "execution_manifest_sha256",
        "scheduler_record_sha256",
        "footprint_manifest_sha256",
        "telemetry",
        "computed_bounds",
        "failure_codes",
    }
    _require_exact(set(value), required, "resource_admission_field_closure")
    _require_exact(
        value.get("schema_version"), RESOURCE_ADMISSION_SCHEMA_VERSION, "resource_schema"
    )
    _require_exact(
        value.get("resource_admission_identity"), RESOURCE_ADMISSION_IDENTITY, "resource_identity"
    )
    _require_exact(value.get("accepted_scientific_result"), False, "resource_result_seal")
    _require_exact(value.get("contains_endpoint_values"), False, "resource_endpoint_seal")
    _require_exact(value.get("status"), "PASS-64", "resource_status")
    _require_exact(value.get("full_mode_admitted"), True, "resource_full_admission")
    _require_exact(value.get("failure_codes"), [], "resource_failure_codes")
    input_digests = value.get("input_digests")
    expected_input_digest_keys = {
        "timing_pilot_sha256",
        "scheduler_record_sha256",
        "footprint_manifest_sha256",
        "source_manifest_sha256",
        "rule_sha256",
        "verifier_script_sha256",
        "verifier_module_sha256",
    }
    if not isinstance(input_digests, dict):
        raise ResourceAdmissionError("resource_input_digests")
    _require_exact(set(input_digests), expected_input_digest_keys, "resource_input_digest_closure")
    _, rule_sha = load_rule(rule_path)
    _require_exact(value.get("rule_path"), RULE_RELATIVE_PATH.as_posix(), "resource_rule_path")
    _require_exact(value.get("rule_sha256"), rule_sha, "resource_rule_sha256")
    _require_exact(
        value.get("source_manifest_path"),
        SOURCE_MANIFEST_RELATIVE_PATH.as_posix(),
        "resource_source_path",
    )
    _require_exact(
        value.get("source_manifest_sha256"),
        verify_source_manifest(
            source_manifest_path, expected_count=_RULE["source_manifest_entry_count"]
        )[0],
        "resource_source_sha256",
    )
    _require_exact(
        value.get("verifier_script_path"),
        VERIFIER_SCRIPT_RELATIVE_PATH.as_posix(),
        "resource_verifier_script_path",
    )
    _require_exact(
        value.get("verifier_script_sha256"),
        sha256_file(verifier_script_path),
        "resource_verifier_script_sha256",
    )
    _require_exact(
        value.get("verifier_module_path"),
        VERIFIER_MODULE_RELATIVE_PATH.as_posix(),
        "resource_verifier_module_path",
    )
    _require_exact(
        value.get("verifier_module_sha256"),
        sha256_file(verifier_module_path),
        "resource_verifier_module_sha256",
    )
    _require_exact(
        value.get("timing_pilot_sha256"), expected_timing_pilot_sha256, "resource_timing_sha256"
    )
    _require_exact(
        value.get("execution_manifest_sha256"),
        expected_execution_manifest_sha256,
        "resource_execution_sha256",
    )
    for key in ("scheduler_record_sha256", "footprint_manifest_sha256"):
        _require_sha256(value.get(key), f"resource_{key}")
    for key in expected_input_digest_keys:
        _require_exact(input_digests.get(key), value.get(key), f"resource_input_{key}")
    telemetry = value.get("telemetry")
    computed_bounds = value.get("computed_bounds")
    if not isinstance(telemetry, dict) or not isinstance(computed_bounds, dict):
        raise ResourceAdmissionError("resource_telemetry")
    expected_telemetry = {
        "pilot_elapsed_seconds",
        "scheduler_elapsed_seconds",
        "non_task_elapsed_seconds",
        "max_rss_bytes",
        "footprint_bytes",
        "free_bytes_before_full",
    }
    expected_bounds = {
        "memory_bound_bytes",
        "wall_bound_seconds",
        "disk_bound_bytes",
        "required_free_bytes",
    }
    _require_exact(set(telemetry), expected_telemetry, "resource_telemetry_closure")
    _require_exact(set(computed_bounds), expected_bounds, "resource_bounds_closure")
    integer_telemetry = {
        "scheduler_elapsed_seconds",
        "max_rss_bytes",
        "footprint_bytes",
        "free_bytes_before_full",
    }
    for key, item in telemetry.items():
        if key in integer_telemetry:
            _require_uint(item, f"resource_{key}")
        elif (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item < 0
        ):
            raise ResourceAdmissionError(f"resource_{key}")
    for key, item in computed_bounds.items():
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item < 0
        ):
            raise ResourceAdmissionError(f"resource_{key}")
    bounds = _RULE["bounds"]
    pilot = telemetry["pilot_elapsed_seconds"]
    scheduler_elapsed = telemetry["scheduler_elapsed_seconds"]
    expected_overhead = max(scheduler_elapsed - pilot, 0.0)
    expected_memory = bounds["memory_multiplier"] * telemetry["max_rss_bytes"]
    expected_wall = (
        bounds["wall_non_task_multiplier"] * expected_overhead
        + bounds["wall_task_multiplier"] * _RULE["full_task_count"] * pilot
    )
    expected_disk = (
        bounds["disk_footprint_multiplier"] * telemetry["footprint_bytes"]
        + bounds["disk_fixed_overhead_bytes"]
    )
    expected_required_free = max(bounds["disk_free_minimum_bytes"], 2 * expected_disk)
    _require_exact(
        telemetry["non_task_elapsed_seconds"], expected_overhead, "resource_non_task_elapsed"
    )
    _require_exact(computed_bounds["memory_bound_bytes"], expected_memory, "resource_memory_bound")
    _require_exact(computed_bounds["wall_bound_seconds"], expected_wall, "resource_wall_bound")
    _require_exact(computed_bounds["disk_bound_bytes"], expected_disk, "resource_disk_bound")
    _require_exact(
        computed_bounds["required_free_bytes"],
        expected_required_free,
        "resource_required_free",
    )
    if expected_memory > bounds["memory_upper_bound_bytes"]:
        raise ResourceAdmissionError("resource_memory_admission")
    if expected_wall > bounds["wall_upper_bound_seconds"]:
        raise ResourceAdmissionError("resource_wall_admission")
    if telemetry["free_bytes_before_full"] < expected_required_free:
        raise ResourceAdmissionError("resource_disk_admission")
    return resource_admission_path


__all__ = [
    "FOOTPRINT_SCHEMA_VERSION",
    "RESOURCE_ADMISSION_IDENTITY",
    "RESOURCE_ADMISSION_SCHEMA_VERSION",
    "ResourceAdmissionError",
    "atomic_write_json",
    "build_footprint_manifest",
    "canonical_json_bytes",
    "ensure_output_outside_roots",
    "load_rule",
    "produce_resource_admission",
    "read_strict_json_object",
    "sha256_file",
    "verify_source_manifest",
    "validate_resource_admission",
]
