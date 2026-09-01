"""Endpoint-sealed E4 non-result schemas, bundles, and verification helpers.

This module intentionally contains no scientific metric, score, map, or
result-field reader.  It provides only the operational gates that must be
admitted before a separately authorized E4 scientific endpoint can exist.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


class NonResultError(ValueError):
    """Raised when an E4 non-result control or bundle fails closed."""


BUNDLE_SCHEMAS = {
    "mapping": "e4-nonresult-mapping/v1",
    "resource_pilot": "e4-nonresult-resource-pilot/v1",
    "resource_admission": "e4-nonresult-resource-admission/v2",
    "preflight": "e4-nonresult-preflight/v1",
}
MANIFEST_NAMES = {
    "mapping": "mapping_manifest.json",
    "resource_pilot": "pilot_manifest.json",
    "resource_admission": "resource_admission_manifest.json",
    "preflight": "preflight_manifest.json",
}
REQUIRED_BUNDLE_FILES = {
    "mapping": frozenset(
        {
            "mapping_manifest.json",
            "source_pair_identity.json",
            "source_mineral_inventory.csv",
            "geometry_contract.json",
            "glt_validation.json",
            "m2_mapping_contract.json",
            "code_manifest.json",
            "output_checksums.sha256",
        }
    ),
    "resource_pilot": frozenset(
        {
            "pilot_manifest.json",
            "input_bindings.json",
            "stage_telemetry.csv",
            "resource_summary.json",
            "forbidden_output_audit.json",
            "code_manifest.json",
            "output_checksums.sha256",
        }
    ),
    "resource_admission": frozenset(),
    "preflight": frozenset(
        {
            "preflight_manifest.json",
            "decision_record.json",
            "input_manifest.json",
            "ontology_crosswalk.csv",
            "ontology_evidence_manifest.json",
            "mapping_admission.json",
            "resource_admission.json",
            "expected_scientific_output_registry.json",
            "code_manifest.json",
            "preflight_summary.json",
            "output_checksums.sha256",
        }
    ),
}
RESOURCE_TELEMETRY_COLUMNS = (
    "stage",
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_bytes",
    "input_bytes",
    "scratch_bytes",
    "exit_status",
)
RESOURCE_STAGES_V2 = (
    "source_snapshot_and_verify",
    "l2b_metadata_glt_kernel",
    "tanager_score_and_footprint_block_kernel",
    "synthetic_bootstrap_kernel",
    "synthetic_spatial_null_kernel",
    "closure_and_forbidden_output_audit",
)
RESOURCE_ATTESTATION_FILENAMES_V2 = tuple(
    f"measurement_attestation_{index:02d}_{stage}.json"
    for index, stage in enumerate(RESOURCE_STAGES_V2)
)
RESOURCE_BINDING_STAGE_FILENAMES_V2 = {
    stage: {
        "contract": f"policy_bindings/{index:02d}_{stage}_contract.json",
        "runner": f"policy_bindings/{index:02d}_{stage}_runner.bin",
        "arguments": f"policy_bindings/{index:02d}_{stage}_arguments.json",
        "input_manifest": f"policy_bindings/{index:02d}_{stage}_input_manifest.json",
        "member_attestation": (
            f"policy_bindings/{index:02d}_{stage}_input_member_attestation.json"
        ),
    }
    for index, stage in enumerate(RESOURCE_STAGES_V2)
}
RESOURCE_SOURCE_MEMBER_ATTESTATION_FILENAME_V2 = "policy_bindings/source_member_attestation.json"
RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2 = frozenset(
    {
        "policy_bindings/scheduler_snapshot.json",
        "policy_bindings/source_capsule.json",
        RESOURCE_SOURCE_MEMBER_ATTESTATION_FILENAME_V2,
        "policy_bindings/synthetic_workload_registry.json",
        *(
            name
            for files in RESOURCE_BINDING_STAGE_FILENAMES_V2.values()
            for name in files.values()
        ),
    }
)
RESOURCE_ADMISSION_EVIDENCE_FILES_V2 = frozenset(
    {
        "resource_policy.json",
        "resource_admission.json",
        "stage_telemetry.json",
        "forbidden_output_audit.json",
        *RESOURCE_ATTESTATION_FILENAMES_V2,
        *RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2,
    }
)
RESOURCE_ADMISSION_PROVENANCE_FILES_V2 = frozenset(
    {
        "resource_admission_bundle_manifest.json",
        "resource_admission_bundle_checksums.sha256",
    }
)
REQUIRED_BUNDLE_FILES["resource_admission"] = frozenset(
    {
        "resource_admission_manifest.json",
        "output_checksums.sha256",
        *RESOURCE_ADMISSION_EVIDENCE_FILES_V2,
    }
)
REQUIRED_BUNDLE_FILES["preflight"] = (
    REQUIRED_BUNDLE_FILES["preflight"]
    | RESOURCE_ADMISSION_EVIDENCE_FILES_V2
    | RESOURCE_ADMISSION_PROVENANCE_FILES_V2
)
RESOURCE_TELEMETRY_COLUMNS_V2 = (
    "stage",
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_bytes",
    "input_bytes",
    "scratch_bytes",
    "exit_status",
    "input_manifest_sha256",
    "stage_contract_sha256",
    "measurement_attestation_sha256",
)
RESOURCE_POLICY_CLASS_V2 = "real_nonresult_resource_pilot"
RESOURCE_STAGE_INPUT_MANIFEST_SCHEMA_V2 = "e4-resource-stage-input-manifest/v2"
RESOURCE_STAGE_CONTRACT_SCHEMA_V2 = "e4-resource-stage-contract/v2"
RESOURCE_MEASUREMENT_ATTESTATION_SCHEMA_V2 = "e4-resource-measurement-attestation/v2"
RESOURCE_SOURCE_CAPSULE_SCHEMA_V2 = "e4-resource-source-capsule/v2"
RESOURCE_WORKLOAD_REGISTRY_SCHEMA_V2 = "e4-resource-synthetic-workload-registry/v2"
RESOURCE_STAGE_ARGUMENTS_SCHEMA_V2 = "e4-resource-stage-arguments/v2"
RESOURCE_SOURCE_MEMBER_ATTESTATION_SCHEMA_V2 = "e4-resource-source-member-attestation/v2"
RESOURCE_STAGE_MEMBER_ATTESTATION_SCHEMA_V2 = "e4-resource-stage-member-attestation/v2"
RESOURCE_MEASUREMENT_CONTRACT_V2 = {
    "wall_seconds_mechanism": "monotonic_ns_elapsed",
    "cpu_seconds_mechanism": "getrusage_self_plus_children",
    "peak_rss_bytes_mechanism": "linux_cgroup_v2_memory.peak",
    "input_bytes_mechanism": "manifest_bound_regular_file_size_bytes",
    "scratch_bytes_mechanism": (
        "recursive_st_blocks_512_allocated_bytes_whole_run_directory_including_hidden_files"
    ),
    "scratch_sampler_cadence_ns": 1_000_000_000,
    "scratch_sampler_availability": "required_fail_closed",
    "scheduler_cross_check": "slurm_independent_cross_check_only",
}
_RESOURCE_POLICY_V2_KEYS = frozenset(
    {
        "schema_version",
        "policy_class",
        "account",
        "qos",
        "partition",
        "allocation_cpus",
        "numerical_threads",
        "allocation_memory_bytes",
        "memory_reserve_bytes",
        "allocation_wall_seconds",
        "wall_reserve_seconds",
        "scratch_budget_bytes",
        "scratch_reserve_bytes",
        "allowed_stages",
        "stage_bindings",
        "bootstrap_count",
        "permutation_count",
        "seed",
        "scheduler_snapshot_sha256",
        "source_capsule_sha256",
        "synthetic_workload_registry_sha256",
        "measurement_contract",
    }
)
_RESOURCE_STAGE_BINDING_KEYS_V2 = frozenset(
    {
        "stage",
        "stage_contract_sha256",
        "input_manifest_sha256",
        "expected_input_bytes",
        "wall_limit_seconds",
        "cpu_limit_seconds",
        "peak_rss_limit_bytes",
        "scratch_limit_bytes",
    }
)
_RESOURCE_STAGE_INPUT_MANIFEST_KEYS_V2 = frozenset({"schema_version", "stage", "files"})
_RESOURCE_STAGE_INPUT_FILE_KEYS_V2 = frozenset({"path", "sha256", "size_bytes"})
_RESOURCE_STAGE_CONTRACT_KEYS_V2 = frozenset(
    {
        "schema_version",
        "stage",
        "runner_sha256",
        "arguments_sha256",
        "measurement_contract_sha256",
    }
)
_RESOURCE_SOURCE_CAPSULE_KEYS_V2 = frozenset({"schema_version", "members"})
_RESOURCE_SOURCE_MEMBER_KEYS_V2 = frozenset({"path", "sha256", "size_bytes"})
_RESOURCE_ATTESTED_MEMBER_KEYS_V2 = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "regular_file_verified",
        "single_link_verified",
        "no_follow_verified",
    }
)
_RESOURCE_SOURCE_MEMBER_ATTESTATION_KEYS_V2 = frozenset(
    {
        "schema_version",
        "scope",
        "verified_at_utc",
        "source_capsule_sha256",
        "members",
    }
)
_RESOURCE_STAGE_MEMBER_ATTESTATION_KEYS_V2 = frozenset(
    {
        "schema_version",
        "scope",
        "stage",
        "verified_at_utc",
        "stage_contract_sha256",
        "runner_sha256",
        "arguments_sha256",
        "input_manifest_sha256",
        "members",
    }
)
_RESOURCE_WORKLOAD_REGISTRY_KEYS_V2 = frozenset({"schema_version", "workloads"})
_RESOURCE_WORKLOAD_KEYS_V2 = frozenset(
    {
        "stage",
        "workload_id",
        "arrays",
        "block_count",
        "covariance_branches",
        "scheduled_iterations",
        "seed",
        "numerical_threads",
        "process_count",
        "generator_sha256",
        "arguments_sha256",
    }
)
_RESOURCE_WORKLOAD_ARRAY_KEYS_V2 = frozenset({"name", "shape", "dtype"})
_RESOURCE_STAGE_ARGUMENTS_KEYS_V2 = frozenset({"schema_version", "stage", "arguments"})
_RESOURCE_FORBIDDEN_OUTPUT_AUDIT_KEYS_V2 = frozenset(
    {
        "schema_version",
        "scientific_endpoint_called",
        "scientific_output_count",
        "audit_completed",
    }
)
_RESOURCE_SCHEDULER_SNAPSHOT_KEYS_V2 = frozenset(
    {
        "schema_version",
        "captured_at_utc",
        "account",
        "qos",
        "partition",
        "allocation_cpus",
        "allocation_memory_bytes",
        "allocation_wall_seconds",
    }
)
_RESOURCE_MEASUREMENT_ATTESTATION_KEYS_V2 = frozenset(
    {
        "schema_version",
        "stage",
        "stage_contract_sha256",
        "input_manifest_sha256",
        "measurement_contract_sha256",
        "wall_start_monotonic_ns",
        "wall_end_monotonic_ns",
        "wall_seconds",
        "cpu_self_user_before_ns",
        "cpu_self_user_after_ns",
        "cpu_self_system_before_ns",
        "cpu_self_system_after_ns",
        "cpu_children_user_before_ns",
        "cpu_children_user_after_ns",
        "cpu_children_system_before_ns",
        "cpu_children_system_after_ns",
        "cpu_seconds",
        "cgroup_identity_sha256",
        "cgroup_v2_memory_peak_available",
        "peak_rss_bytes",
        "scratch_root_identity_sha256",
        "scratch_sampler_cadence_ns",
        "scratch_first_sample_monotonic_ns",
        "scratch_last_sample_monotonic_ns",
        "scratch_max_sample_gap_ns",
        "scratch_missed_samples",
        "scratch_final_scan_completed",
        "scratch_sampler_failed",
        "scratch_escape_detected",
        "scratch_peak_allocated_bytes",
        "exit_status",
    }
)
_RESOURCE_ADMISSION_KEYS_V2 = frozenset(
    {
        "schema_version",
        "admission_status",
        "policy_class",
        "resource_policy_sha256",
        "scheduler_snapshot_sha256",
        "source_capsule_sha256",
        "synthetic_workload_registry_sha256",
        "telemetry_sha256",
        "measurement_attestation_closure_sha256",
        "binding_evidence_closure_sha256",
        "synthetic_fixture_only",
    }
)
_SCHEDULER_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ALLOWED_ESTIMANDS = frozenset(
    {
        "operational_transductive_primary",
        "strict_inductive_primary",
        "operational_primary_strict_inductive_sensitivity",
        "both_primary",
    }
)
ALLOWED_CLAIM_CLASSES = frozenset(
    {
        "operational_association_only",
        "held_block_association_only",
        "separate_operational_and_held_block_association",
        "both_primary_with_multiplicity",
    }
)
SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,95}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DARWIN_AT_FDCWD = -2
_DARWIN_RENAME_EXCL = 0x00000004
_DARWIN_RENAME_NOFOLLOW_ANY = 0x00000010
_LINUX_AT_FDCWD = -100
_LINUX_RENAME_NOREPLACE = 1


@dataclass(frozen=True)
class BundleReceipt:
    """Identity returned after an exact-closure bundle promotion."""

    bundle_path: Path
    bundle_type: str
    bundle_id: str
    closure_sha256: str


@dataclass(frozen=True)
class ResourceAdmissionEvidence:
    """Verified raw controls carried by a real resource-admission bundle."""

    bundle_receipt: BundleReceipt | None
    policy: dict[str, Any]
    admission: dict[str, Any]
    payloads: dict[str, bytes]


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise NonResultError("nonfinite_json_value")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite(item)


def canonical_json_bytes(payload: Any) -> bytes:
    """Return deterministic strict JSON bytes without NaN or Infinity."""
    _reject_nonfinite(payload)
    try:
        text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise NonResultError("noncanonical_json_payload") from error
    return (text + "\n").encode("utf-8")


def strict_json_load_bytes(data: bytes) -> Any:
    """Parse strict UTF-8 JSON while rejecting duplicate keys and constants."""

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in values:
            if key in output:
                raise NonResultError("duplicate_json_key")
            output[key] = value
        return output

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(NonResultError("json_constant")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NonResultError("malformed_json") from error
    _reject_nonfinite(value)
    return value


def _safe_identifier(value: str) -> str:
    if not isinstance(value, str) or not SAFE_IDENTIFIER.fullmatch(value):
        raise NonResultError("invalid_bundle_identifier")
    return value


def _safe_relative(path: str) -> str:
    value = PurePosixPath(path)
    if (
        not path
        or value.is_absolute()
        or ".." in value.parts
        or "." in value.parts
        or not value.parts
        or str(value) != path
    ):
        raise NonResultError("unsafe_relative_path")
    return path


def _open_regular_no_follow(path: Path) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise NonResultError("unsafe_or_missing_file") from error
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise NonResultError("nonregular_or_linked_file")
    return descriptor


def _descriptor_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_regular_descriptor(descriptor: int) -> bytes:
    before = os.fstat(descriptor)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if _descriptor_identity(before) != _descriptor_identity(after):
        raise NonResultError("file_changed_during_read")
    return b"".join(chunks)


def _hash_regular_descriptor(descriptor: int) -> tuple[os.stat_result, str]:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(descriptor)
    if _descriptor_identity(before) != _descriptor_identity(after):
        raise NonResultError("file_changed_during_hash")
    return after, digest.hexdigest()


def read_regular_bytes(path: str | Path) -> bytes:
    """Read one regular, unlinked file through a no-follow descriptor."""
    descriptor = _open_regular_no_follow(Path(path))
    try:
        return _read_regular_descriptor(descriptor)
    finally:
        os.close(descriptor)


def sha256_regular_file(path: str | Path) -> str:
    """Hash one regular, unlinked file without following links."""
    descriptor = _open_regular_no_follow(Path(path))
    try:
        _, digest = _hash_regular_descriptor(descriptor)
    finally:
        os.close(descriptor)
    return digest


def _hash_relative_regular_file(root_descriptor: int, relative_path: str) -> tuple[int, str]:
    """Hash one manifest member through no-follow directory descriptors."""
    parts = PurePosixPath(_safe_relative(relative_path)).parts
    current = os.dup(root_descriptor)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            info = os.fstat(next_descriptor)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_descriptor)
                raise NonResultError("nonregular_stage_input_parent")
            os.close(current)
            current = next_descriptor
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise NonResultError("nonregular_or_linked_stage_input")
            final, digest = _hash_regular_descriptor(descriptor)
            return final.st_size, digest
        finally:
            os.close(descriptor)
    except OSError as error:
        raise NonResultError("unsafe_or_missing_stage_input") from error
    finally:
        os.close(current)


def _inventory(root: Path) -> dict[str, tuple[int, int, int, str]]:
    if not root.is_dir() or root.is_symlink():
        raise NonResultError("unsafe_bundle_root")
    files: dict[str, tuple[int, int, int, str]] = {}
    for base, directories, names in os.walk(root, followlinks=False):
        base_path = Path(base)
        for directory in directories:
            info = os.lstat(base_path / directory)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise NonResultError("unsafe_bundle_directory")
        for name in names:
            path = base_path / name
            descriptor = _open_regular_no_follow(path)
            try:
                info, digest = _hash_regular_descriptor(descriptor)
            finally:
                os.close(descriptor)
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative)
            files[relative] = (info.st_dev, info.st_ino, info.st_size, digest)
    return files


def _checksum_text(inventory: Mapping[str, tuple[int, int, int, str]]) -> bytes:
    return "".join(
        f"{inventory[name][3]}  {name}\n"
        for name in sorted(inventory)
        if name != "output_checksums.sha256"
    ).encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically promote a directory without replacement or link following."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            _DARWIN_AT_FDCWD,
            source_bytes,
            _DARWIN_AT_FDCWD,
            destination_bytes,
            _DARWIN_RENAME_EXCL | _DARWIN_RENAME_NOFOLLOW_ANY,
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise NonResultError("atomic_noreplace_unavailable") from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            _LINUX_AT_FDCWD,
            source_bytes,
            _LINUX_AT_FDCWD,
            destination_bytes,
            _LINUX_RENAME_NOREPLACE,
        )
    else:
        raise NonResultError("atomic_noreplace_unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise NonResultError("bundle_target_exists")
    raise OSError(error_number, os.strerror(error_number), destination)


def atomic_write_bundle(
    parent: str | Path,
    *,
    bundle_type: str,
    bundle_id: str,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    fault_after: int | None = None,
) -> BundleReceipt:
    """Write, close, verify, and promote a fixed E4 non-result bundle.

    The caller supplies every non-checksum payload.  The helper derives the
    manifest closure and checksum file, rejects extra/missing paths, and never
    replaces an existing target.  ``fault_after`` is test-only fault injection.
    """
    if bundle_type not in BUNDLE_SCHEMAS:
        raise NonResultError("unknown_bundle_type")
    _safe_identifier(bundle_id)
    expected = REQUIRED_BUNDLE_FILES[bundle_type]
    manifest_name = MANIFEST_NAMES[bundle_type]
    if set(payloads) != expected - {manifest_name, "output_checksums.sha256"}:
        raise NonResultError("bundle_payload_closure_mismatch")
    parent_path = Path(parent)
    parent_path.mkdir(parents=True, exist_ok=True)
    if parent_path.is_symlink():
        raise NonResultError("unsafe_bundle_parent")
    final = parent_path / bundle_id
    if final.exists() or final.is_symlink():
        raise NonResultError("bundle_target_exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.staging-", dir=parent_path))
    try:
        staged_manifest = dict(manifest)
        staged_manifest.update(
            {
                "schema_version": BUNDLE_SCHEMAS[bundle_type],
                "bundle_type": bundle_type,
                "bundle_id": bundle_id,
                "expected_files": sorted(expected),
            }
        )
        all_payloads = {**payloads, manifest_name: canonical_json_bytes(staged_manifest)}
        for index, relative in enumerate(sorted(all_payloads), start=1):
            _safe_relative(relative)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_exclusive(destination, all_payloads[relative])
            if fault_after == index:
                raise RuntimeError("injected_staging_failure")
        before = _inventory(staging)
        if set(before) != expected - {"output_checksums.sha256"}:
            raise NonResultError("staging_file_closure_mismatch")
        _write_exclusive(staging / "output_checksums.sha256", _checksum_text(before))
        after = _inventory(staging)
        if set(after) != expected:
            raise NonResultError("staging_checksum_closure_mismatch")
        if final.exists() or final.is_symlink():
            raise NonResultError("bundle_target_exists")
        _rename_directory_noreplace(staging, final)
        final_inventory = _inventory(final)
        if final_inventory != after:
            raise NonResultError("bundle_changed_during_promotion")
        receipt = verify_nonresult_bundle(final, expected_type=bundle_type)
        return receipt
    except Exception:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()
        raise


def _parse_checksum_file(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if match is None:
            raise NonResultError("malformed_checksum_manifest")
        digest, relative = match.groups()
        _safe_relative(relative)
        if relative == "output_checksums.sha256" or relative in result:
            raise NonResultError("invalid_checksum_member")
        result[relative] = digest
    return result


def verify_nonresult_bundle(
    bundle_path: str | Path,
    *,
    expected_type: str | None = None,
) -> BundleReceipt:
    """Independently validate exact closure, checksums, and manifest semantics."""
    root = Path(bundle_path)
    inventory_before = _inventory(root)
    manifests = [name for name in inventory_before if name in MANIFEST_NAMES.values()]
    if len(manifests) != 1:
        raise NonResultError("ambiguous_bundle_manifest")
    manifest_name = manifests[0]
    manifest = strict_json_load_bytes(read_regular_bytes(root / manifest_name))
    if not isinstance(manifest, dict):
        raise NonResultError("invalid_bundle_manifest")
    bundle_type = manifest.get("bundle_type")
    if bundle_type not in BUNDLE_SCHEMAS or MANIFEST_NAMES[bundle_type] != manifest_name:
        raise NonResultError("bundle_type_manifest_mismatch")
    if expected_type is not None and bundle_type != expected_type:
        raise NonResultError("unexpected_bundle_type")
    bundle_id = _safe_identifier(manifest.get("bundle_id"))
    if root.name != bundle_id or manifest.get("schema_version") != BUNDLE_SCHEMAS[bundle_type]:
        raise NonResultError("bundle_identity_mismatch")
    expected = REQUIRED_BUNDLE_FILES[bundle_type]
    declared = manifest.get("expected_files")
    if (
        not isinstance(declared, list)
        or set(declared) != expected
        or len(declared) != len(expected)
    ):
        raise NonResultError("declared_file_closure_mismatch")
    if set(inventory_before) != expected:
        raise NonResultError("actual_file_closure_mismatch")
    checksum_bytes = read_regular_bytes(root / "output_checksums.sha256")
    checksum_rows = _parse_checksum_file(checksum_bytes)
    if set(checksum_rows) != expected - {"output_checksums.sha256"}:
        raise NonResultError("checksum_file_closure_mismatch")
    for relative, digest in checksum_rows.items():
        if inventory_before[relative][3] != digest:
            raise NonResultError("checksum_mismatch")
    inventory_after = _inventory(root)
    if inventory_before != inventory_after:
        raise NonResultError("bundle_changed_during_verification")
    closure = hashlib.sha256(checksum_bytes).hexdigest()
    return BundleReceipt(root, bundle_type, bundle_id, closure)


def _require_mapping(value: Any, *, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NonResultError(code)
    return value


def _require_nonblank(value: Any, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().casefold() == "tbd":
        raise NonResultError(code)
    return value.strip()


def _normalized_identity(value: str) -> str:
    return " ".join(value.casefold().split())


def _validate_ontology_decision(value: Any) -> None:
    ontology = _require_mapping(value, code="invalid_ontology_decision")
    crosswalk = _require_nonblank(ontology.get("crosswalk_sha256"), code="unbound_ontology")
    if not SHA256.fullmatch(crosswalk.casefold()):
        raise NonResultError("invalid_ontology_hash")
    entries = ontology.get("entries")
    if not isinstance(entries, list) or not entries:
        raise NonResultError("unresolved_ontology_entries")
    seen: set[tuple[int, int, str, str]] = set()
    for entry in entries:
        row = _require_mapping(entry, code="invalid_ontology_entry")
        group = row.get("group")
        index = row.get("index")
        name = _require_nonblank(row.get("name"), code="invalid_ontology_entry")
        library = _require_nonblank(row.get("library"), code="invalid_ontology_entry")
        mapping = _require_nonblank(row.get("mapping"), code="unresolved_ontology_entry")
        if isinstance(group, bool) or not isinstance(group, int) or group not in {1, 2}:
            raise NonResultError("invalid_ontology_group")
        if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
            raise NonResultError("invalid_ontology_index")
        identity = (group, index, name, library)
        if identity in seen:
            raise NonResultError("duplicate_ontology_entry")
        seen.add(identity)
        if mapping == "exact":
            target = _require_nonblank(row.get("target"), code="invalid_exact_ontology")
            if _normalized_identity(name) != _normalized_identity(target):
                raise NonResultError("exact_ontology_name_mismatch")
        elif mapping == "broader":
            target = _require_nonblank(row.get("target"), code="invalid_broader_ontology")
            del target
            locator = _require_nonblank(
                row.get("authority_locator"),
                code="invalid_broader_ontology",
            )
            authority = _require_nonblank(
                row.get("authority_sha256"),
                code="invalid_broader_ontology",
            )
            if not SHA256.fullmatch(authority.casefold()) or not locator.startswith(
                ("https://", "doi:")
            ):
                raise NonResultError("invalid_broader_authority")
        elif mapping == "unmapped":
            if row.get("target") not in {None, ""}:
                raise NonResultError("unmapped_ontology_has_target")
            _require_nonblank(
                row.get("unavailable_reason"),
                code="unmapped_ontology_reason_missing",
            )
        else:
            raise NonResultError("unknown_ontology_mapping")


def validate_decision_record(payload: Any) -> dict[str, Any]:
    """Validate an explicit E4 pre-result decision contract without choosing it."""
    record = dict(_require_mapping(payload, code="invalid_decision_record"))
    if record.get("schema_version") != "e4-decision-record/v1":
        raise NonResultError("invalid_decision_schema")
    for name in (
        "ontology",
        "support",
        "covariance_estimand",
        "negative_control",
        "claim_class",
        "output_registry",
    ):
        if name not in record:
            raise NonResultError("missing_required_decision")
    _validate_ontology_decision(record["ontology"])
    support = _require_mapping(record["support"], code="invalid_support_decision")
    if support.get("primary_geometry") != "L" or support.get("sensitivity_geometry") != "2L":
        raise NonResultError("invalid_support_geometry")
    estimand = _require_nonblank(record["covariance_estimand"], code="unresolved_estimand")
    claim = _require_nonblank(record["claim_class"], code="unresolved_claim_class")
    if estimand not in ALLOWED_ESTIMANDS or claim not in ALLOWED_CLAIM_CLASSES:
        raise NonResultError("unknown_estimand_or_claim_class")
    compatible = {
        "operational_transductive_primary": {"operational_association_only"},
        "strict_inductive_primary": {"held_block_association_only"},
        "operational_primary_strict_inductive_sensitivity": {
            "separate_operational_and_held_block_association"
        },
        "both_primary": {"both_primary_with_multiplicity"},
    }
    if claim not in compatible[estimand]:
        raise NonResultError("estimand_claim_incompatibility")
    negative = _require_mapping(record["negative_control"], code="invalid_negative_control")
    negative_kind = _require_nonblank(negative.get("kind"), code="unresolved_negative_control")
    if negative_kind == "none":
        if negative.get("specificity_claims") is not False:
            raise NonResultError("negative_control_claim_incompatibility")
    elif negative_kind == "predeclared_mineral_identity_mismatch":
        for key in ("authority_sha256", "authority_locator"):
            _require_nonblank(negative.get(key), code="unresolved_negative_control")
        if not SHA256.fullmatch(str(negative["authority_sha256"]).casefold()):
            raise NonResultError("invalid_negative_control_hash")
    else:
        raise NonResultError("unknown_negative_control")
    output_registry = _require_mapping(record["output_registry"], code="invalid_output_registry")
    if not output_registry.get("sha256") or not SHA256.fullmatch(
        str(output_registry["sha256"]).casefold()
    ):
        raise NonResultError("unresolved_output_registry")
    if estimand == "both_primary" and not _require_nonblank(
        record.get("multiplicity_policy"), code="missing_multiplicity_policy"
    ):
        raise NonResultError("missing_multiplicity_policy")
    return record


def _require_nonnegative_integer(value: Any, *, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NonResultError(code)
    return value


def _require_positive_integer(value: Any, *, code: str) -> int:
    result = _require_nonnegative_integer(value, code=code)
    if result == 0:
        raise NonResultError(code)
    return result


def _require_canonical_control_json(data: bytes, payload: Any, *, code: str) -> None:
    if canonical_json_bytes(payload) != data:
        raise NonResultError(code)


def _validate_source_capsule_v2(data: bytes) -> dict[str, Any]:
    capsule = dict(
        _require_mapping(strict_json_load_bytes(data), code="invalid_resource_source_capsule")
    )
    _require_canonical_control_json(data, capsule, code="noncanonical_resource_source_capsule")
    if (
        set(capsule) != _RESOURCE_SOURCE_CAPSULE_KEYS_V2
        or capsule["schema_version"] != RESOURCE_SOURCE_CAPSULE_SCHEMA_V2
        or not isinstance(capsule["members"], list)
        or not capsule["members"]
    ):
        raise NonResultError("resource_source_capsule_schema_mismatch")
    paths: list[str] = []
    for raw_member in capsule["members"]:
        member = _require_mapping(raw_member, code="invalid_resource_source_member")
        if set(member) != _RESOURCE_SOURCE_MEMBER_KEYS_V2:
            raise NonResultError("resource_source_member_schema_mismatch")
        path = _safe_relative(member["path"])
        digest = member["sha256"]
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise NonResultError("invalid_resource_source_member_hash")
        _require_nonnegative_integer(
            member["size_bytes"], code="invalid_resource_source_member_size"
        )
        paths.append(path)
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise NonResultError("noncanonical_resource_source_member_order")
    return capsule


def _validate_workload_registry_v2(
    data: bytes,
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    registry = dict(
        _require_mapping(strict_json_load_bytes(data), code="invalid_synthetic_workload_registry")
    )
    _require_canonical_control_json(data, registry, code="noncanonical_synthetic_workload_registry")
    if (
        set(registry) != _RESOURCE_WORKLOAD_REGISTRY_KEYS_V2
        or registry["schema_version"] != RESOURCE_WORKLOAD_REGISTRY_SCHEMA_V2
        or not isinstance(registry["workloads"], list)
    ):
        raise NonResultError("synthetic_workload_registry_schema_mismatch")
    expected_stages = ("synthetic_bootstrap_kernel", "synthetic_spatial_null_kernel")
    workloads = registry["workloads"]
    if len(workloads) != len(expected_stages):
        raise NonResultError("synthetic_workload_registry_stage_mismatch")
    for stage, raw_workload in zip(expected_stages, workloads, strict=True):
        workload = _require_mapping(raw_workload, code="invalid_synthetic_workload")
        if set(workload) != _RESOURCE_WORKLOAD_KEYS_V2 or workload["stage"] != stage:
            raise NonResultError("synthetic_workload_schema_mismatch")
        _safe_identifier(workload["workload_id"])
        arrays = workload["arrays"]
        if not isinstance(arrays, list) or not arrays:
            raise NonResultError("synthetic_workload_arrays_required")
        array_names: list[str] = []
        for raw_array in arrays:
            array = _require_mapping(raw_array, code="invalid_synthetic_workload_array")
            if set(array) != _RESOURCE_WORKLOAD_ARRAY_KEYS_V2:
                raise NonResultError("synthetic_workload_array_schema_mismatch")
            name = _safe_identifier(array["name"])
            shape = array["shape"]
            if not isinstance(shape, list) or not shape:
                raise NonResultError("invalid_synthetic_workload_shape")
            for dimension in shape:
                _require_positive_integer(dimension, code="invalid_synthetic_workload_shape")
            if array["dtype"] not in {
                "bool",
                "float32",
                "float64",
                "int16",
                "int32",
                "int64",
            }:
                raise NonResultError("invalid_synthetic_workload_dtype")
            array_names.append(name)
        if array_names != sorted(array_names) or len(set(array_names)) != len(array_names):
            raise NonResultError("noncanonical_synthetic_workload_array_order")
        _require_positive_integer(workload["block_count"], code="invalid_workload_block_count")
        if workload["covariance_branches"] != ["operational", "strict_inductive"]:
            raise NonResultError("invalid_workload_covariance_branches")
        _require_positive_integer(
            workload["scheduled_iterations"], code="invalid_workload_iteration_count"
        )
        _require_nonnegative_integer(workload["seed"], code="invalid_workload_seed")
        _require_positive_integer(
            workload["numerical_threads"], code="invalid_workload_thread_count"
        )
        _require_positive_integer(workload["process_count"], code="invalid_workload_process_count")
        expected_iterations = (
            policy["bootstrap_count"]
            if stage == "synthetic_bootstrap_kernel"
            else policy["permutation_count"]
        )
        if (
            workload["scheduled_iterations"] != expected_iterations
            or workload["seed"] != policy["seed"]
            or workload["numerical_threads"] != policy["numerical_threads"]
            or workload["process_count"] != 1
        ):
            raise NonResultError("synthetic_workload_policy_mismatch")
        for key in ("generator_sha256", "arguments_sha256"):
            value = workload[key]
            if not isinstance(value, str) or SHA256.fullmatch(value) is None:
                raise NonResultError("invalid_synthetic_workload_hash")
    return registry


def _validate_stage_arguments_v2(data: bytes, *, stage: str) -> dict[str, Any]:
    payload = dict(
        _require_mapping(strict_json_load_bytes(data), code="invalid_resource_stage_arguments")
    )
    _require_canonical_control_json(data, payload, code="noncanonical_resource_stage_arguments")
    if (
        set(payload) != _RESOURCE_STAGE_ARGUMENTS_KEYS_V2
        or payload["schema_version"] != RESOURCE_STAGE_ARGUMENTS_SCHEMA_V2
        or payload["stage"] != stage
        or not isinstance(payload["arguments"], dict)
        or not payload["arguments"]
    ):
        raise NonResultError("resource_stage_arguments_schema_mismatch")
    return payload


def _validate_utc_timestamp(value: Any, *, code: str) -> str:
    if not isinstance(value, str):
        raise NonResultError(code)
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise NonResultError(code) from error
    return value


def _attest_manifest_members(
    root: str | Path,
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Hash declared members through no-follow descriptors for an admission receipt."""
    try:
        root_descriptor = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise NonResultError("unsafe_or_missing_resource_binding_root") from error
    try:
        root_info = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_info.st_mode):
            raise NonResultError("invalid_resource_binding_root")
        result: list[dict[str, Any]] = []
        for member in members:
            relative = _safe_relative(member["path"])
            observed_size, observed_digest = _hash_relative_regular_file(root_descriptor, relative)
            if observed_size != member["size_bytes"] or observed_digest != member["sha256"]:
                raise NonResultError("resource_binding_member_identity_mismatch")
            result.append(
                {
                    "path": relative,
                    "sha256": observed_digest,
                    "size_bytes": observed_size,
                    "regular_file_verified": True,
                    "single_link_verified": True,
                    "no_follow_verified": True,
                }
            )
        return result
    finally:
        os.close(root_descriptor)


def _validate_member_attestation_v2(
    data: bytes,
    *,
    schema_version: str,
    scope: str,
    expected_bindings: Mapping[str, str],
    declared_members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attestation = dict(
        _require_mapping(strict_json_load_bytes(data), code="invalid_resource_member_attestation")
    )
    _require_canonical_control_json(
        data, attestation, code="noncanonical_resource_member_attestation"
    )
    expected_keys = (
        _RESOURCE_SOURCE_MEMBER_ATTESTATION_KEYS_V2
        if scope == "source_capsule"
        else _RESOURCE_STAGE_MEMBER_ATTESTATION_KEYS_V2
    )
    if (
        set(attestation) != expected_keys
        or attestation["schema_version"] != schema_version
        or attestation["scope"] != scope
    ):
        raise NonResultError("resource_member_attestation_schema_mismatch")
    _validate_utc_timestamp(
        attestation["verified_at_utc"], code="invalid_resource_member_attestation_timestamp"
    )
    for key, expected in expected_bindings.items():
        if attestation.get(key) != expected:
            raise NonResultError("resource_member_attestation_binding_mismatch")
    members = attestation["members"]
    if not isinstance(members, list) or not members:
        raise NonResultError("resource_member_attestation_members_required")
    observed: list[dict[str, Any]] = []
    for raw_member in members:
        member = dict(_require_mapping(raw_member, code="invalid_attested_resource_member"))
        if set(member) != _RESOURCE_ATTESTED_MEMBER_KEYS_V2:
            raise NonResultError("attested_resource_member_schema_mismatch")
        member["path"] = _safe_relative(member["path"])
        if not isinstance(member["sha256"], str) or SHA256.fullmatch(member["sha256"]) is None:
            raise NonResultError("invalid_attested_resource_member_hash")
        _require_nonnegative_integer(
            member["size_bytes"], code="invalid_attested_resource_member_size"
        )
        if any(
            member[key] is not True
            for key in (
                "regular_file_verified",
                "single_link_verified",
                "no_follow_verified",
            )
        ):
            raise NonResultError("resource_member_attestation_verification_failed")
        observed.append(member)
    if [member["path"] for member in observed] != sorted(
        member["path"] for member in observed
    ) or len({member["path"] for member in observed}) != len(observed):
        raise NonResultError("noncanonical_resource_member_attestation_order")
    expected_members = [
        {
            "path": _safe_relative(member["path"]),
            "sha256": member["sha256"],
            "size_bytes": member["size_bytes"],
            "regular_file_verified": True,
            "single_link_verified": True,
            "no_follow_verified": True,
        }
        for member in declared_members
    ]
    if observed != expected_members:
        raise NonResultError("resource_member_attestation_manifest_mismatch")
    return attestation


def _binding_evidence_closure_sha256(payloads: Mapping[str, bytes]) -> str:
    if set(payloads) != RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2:
        raise NonResultError("resource_binding_evidence_closure_mismatch")
    closure = b"".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n".encode()
        for name in sorted(RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2)
    )
    return hashlib.sha256(closure).hexdigest()


def _effective_resource_limits(policy: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        policy["allocation_wall_seconds"] - policy["wall_reserve_seconds"],
        policy["allocation_memory_bytes"] - policy["memory_reserve_bytes"],
        policy["scratch_budget_bytes"] - policy["scratch_reserve_bytes"],
    )


def _validate_resource_policy_v2(policy: dict[str, Any]) -> dict[str, Any]:
    if set(policy) != _RESOURCE_POLICY_V2_KEYS:
        raise NonResultError("resource_policy_v2_schema_mismatch")
    if policy["policy_class"] != RESOURCE_POLICY_CLASS_V2:
        raise NonResultError("invalid_resource_policy_class")

    for key in ("account", "qos", "partition"):
        value = policy[key]
        if (
            not isinstance(value, str)
            or value != value.strip()
            or value.casefold() == "tbd"
            or _SCHEDULER_IDENTIFIER.fullmatch(value) is None
        ):
            raise NonResultError("invalid_scheduler_binding")

    for key in (
        "allocation_cpus",
        "numerical_threads",
        "allocation_memory_bytes",
        "memory_reserve_bytes",
        "allocation_wall_seconds",
        "wall_reserve_seconds",
        "scratch_budget_bytes",
        "scratch_reserve_bytes",
        "bootstrap_count",
        "permutation_count",
        "seed",
    ):
        _require_nonnegative_integer(policy[key], code="invalid_resource_policy_v2_value")
    for key in (
        "allocation_cpus",
        "numerical_threads",
        "allocation_memory_bytes",
        "allocation_wall_seconds",
        "scratch_budget_bytes",
        "bootstrap_count",
        "permutation_count",
    ):
        if policy[key] == 0:
            raise NonResultError("invalid_resource_policy_v2_value")
    if policy["numerical_threads"] > policy["allocation_cpus"]:
        raise NonResultError("numerical_threads_exceed_allocation")
    for capacity, reserve in (
        ("allocation_memory_bytes", "memory_reserve_bytes"),
        ("allocation_wall_seconds", "wall_reserve_seconds"),
        ("scratch_budget_bytes", "scratch_reserve_bytes"),
    ):
        if policy[reserve] >= policy[capacity]:
            raise NonResultError("resource_reserve_exceeds_capacity")

    stages = policy["allowed_stages"]
    if not isinstance(stages, list) or tuple(stages) != RESOURCE_STAGES_V2:
        raise NonResultError("resource_policy_v2_stage_sequence_mismatch")

    stage_bindings = policy["stage_bindings"]
    if not isinstance(stage_bindings, list) or len(stage_bindings) != len(RESOURCE_STAGES_V2):
        raise NonResultError("resource_stage_bindings_mismatch")
    for expected_stage, raw_binding in zip(RESOURCE_STAGES_V2, stage_bindings, strict=True):
        binding = _require_mapping(raw_binding, code="invalid_resource_stage_binding")
        if set(binding) != _RESOURCE_STAGE_BINDING_KEYS_V2 or binding["stage"] != expected_stage:
            raise NonResultError("resource_stage_bindings_mismatch")
        for key in (
            "expected_input_bytes",
            "wall_limit_seconds",
            "cpu_limit_seconds",
            "peak_rss_limit_bytes",
            "scratch_limit_bytes",
        ):
            _require_nonnegative_integer(binding[key], code="invalid_resource_stage_binding")
        for key in ("input_manifest_sha256", "stage_contract_sha256"):
            value = binding[key]
            if not isinstance(value, str) or SHA256.fullmatch(value) is None:
                raise NonResultError("invalid_resource_stage_binding_hash")

    effective_wall, effective_memory, effective_scratch = _effective_resource_limits(policy)
    for binding in stage_bindings:
        if (
            binding["wall_limit_seconds"] > effective_wall
            or binding["cpu_limit_seconds"]
            > policy["allocation_cpus"] * binding["wall_limit_seconds"]
            or binding["peak_rss_limit_bytes"] > effective_memory
            or binding["scratch_limit_bytes"] > effective_scratch
        ):
            raise NonResultError("resource_stage_limit_exceeds_effective_allocation")
    if (
        sum(binding["wall_limit_seconds"] for binding in stage_bindings) > effective_wall
        or sum(binding["cpu_limit_seconds"] for binding in stage_bindings)
        > policy["allocation_cpus"] * effective_wall
    ):
        raise NonResultError("resource_stage_limits_exceed_aggregate_allocation")

    for key in (
        "scheduler_snapshot_sha256",
        "source_capsule_sha256",
        "synthetic_workload_registry_sha256",
    ):
        value = policy[key]
        if not isinstance(value, str) or SHA256.fullmatch(value) is None:
            raise NonResultError("invalid_resource_policy_v2_hash")

    measurement = _require_mapping(
        policy["measurement_contract"], code="invalid_measurement_contract"
    )
    if dict(measurement) != RESOURCE_MEASUREMENT_CONTRACT_V2:
        raise NonResultError("invalid_measurement_contract")
    return policy


def validate_resource_policy(payload: Any) -> dict[str, Any]:
    """Validate the only policy schema admissible to real-pilot preflight.

    Version 2 scratch values are accounting budgets only.  They neither name nor
    authorize a filesystem endpoint.
    """
    policy = dict(_require_mapping(payload, code="invalid_resource_policy"))
    if policy.get("schema_version") != "e4-resource-policy/v2":
        raise NonResultError("invalid_resource_policy_schema")
    return _validate_resource_policy_v2(policy)


def verify_resource_policy_bindings(
    policy: Mapping[str, Any],
    *,
    scheduler_snapshot: str | Path,
    source_capsule: str | Path,
    source_root: str | Path | None = None,
    source_member_attestation: str | Path | None = None,
    synthetic_workload_registry: str | Path,
    stage_contracts: Mapping[str, str | Path],
    stage_runners: Mapping[str, str | Path],
    stage_arguments: Mapping[str, str | Path],
    input_manifests: Mapping[str, str | Path],
    input_roots: Mapping[str, str | Path] | None = None,
    input_member_attestations: Mapping[str, str | Path] | None = None,
) -> None:
    """No-follow verify every artifact referenced by a v2 policy.

    This is an operational binding check only. It reads scheduler/control
    metadata and declared input bytes, never scientific result fields.
    """
    validated = _validate_resource_policy_v2(dict(policy))
    expected_stages = set(RESOURCE_STAGES_V2)
    live_mode = source_root is not None or input_roots is not None
    attested_mode = source_member_attestation is not None or input_member_attestations is not None
    if live_mode == attested_mode:
        raise NonResultError("resource_binding_evidence_mode_mismatch")
    if live_mode and (source_root is None or input_roots is None):
        raise NonResultError("resource_binding_evidence_mode_mismatch")
    if attested_mode and (source_member_attestation is None or input_member_attestations is None):
        raise NonResultError("resource_binding_evidence_mode_mismatch")
    for mapping in (
        stage_contracts,
        stage_runners,
        stage_arguments,
        input_manifests,
    ):
        if not isinstance(mapping, Mapping) or set(mapping) != expected_stages:
            raise NonResultError("resource_binding_artifact_stage_mismatch")
    selected_member_mapping = input_roots if live_mode else input_member_attestations
    if (
        not isinstance(selected_member_mapping, Mapping)
        or set(selected_member_mapping) != expected_stages
    ):
        raise NonResultError("resource_binding_artifact_stage_mismatch")

    scheduler_bytes = read_regular_bytes(scheduler_snapshot)
    if hashlib.sha256(scheduler_bytes).hexdigest() != validated["scheduler_snapshot_sha256"]:
        raise NonResultError("scheduler_snapshot_hash_mismatch")
    scheduler = _require_mapping(
        strict_json_load_bytes(scheduler_bytes), code="invalid_scheduler_snapshot"
    )
    _require_canonical_control_json(
        scheduler_bytes, scheduler, code="noncanonical_scheduler_snapshot"
    )
    if set(scheduler) != _RESOURCE_SCHEDULER_SNAPSHOT_KEYS_V2:
        raise NonResultError("scheduler_snapshot_schema_mismatch")
    if scheduler["schema_version"] != "e4-scheduler-snapshot/v2":
        raise NonResultError("invalid_scheduler_snapshot")
    _validate_utc_timestamp(
        scheduler["captured_at_utc"], code="invalid_scheduler_snapshot_timestamp"
    )
    for key in ("account", "qos", "partition"):
        if scheduler[key] != validated[key]:
            raise NonResultError("scheduler_snapshot_policy_mismatch")
    for key in (
        "allocation_cpus",
        "allocation_memory_bytes",
        "allocation_wall_seconds",
    ):
        if (
            isinstance(scheduler[key], bool)
            or not isinstance(scheduler[key], int)
            or scheduler[key] != validated[key]
        ):
            raise NonResultError("scheduler_snapshot_policy_mismatch")

    source_capsule_bytes = read_regular_bytes(source_capsule)
    if hashlib.sha256(source_capsule_bytes).hexdigest() != validated["source_capsule_sha256"]:
        raise NonResultError("source_capsule_hash_mismatch")
    source_capsule_payload = _validate_source_capsule_v2(source_capsule_bytes)
    if live_mode:
        _attest_manifest_members(source_root, source_capsule_payload["members"])
    else:
        source_attestation_bytes = read_regular_bytes(source_member_attestation)
        _validate_member_attestation_v2(
            source_attestation_bytes,
            schema_version=RESOURCE_SOURCE_MEMBER_ATTESTATION_SCHEMA_V2,
            scope="source_capsule",
            expected_bindings={
                "source_capsule_sha256": hashlib.sha256(source_capsule_bytes).hexdigest()
            },
            declared_members=source_capsule_payload["members"],
        )

    workload_bytes = read_regular_bytes(synthetic_workload_registry)
    if (
        hashlib.sha256(workload_bytes).hexdigest()
        != validated["synthetic_workload_registry_sha256"]
    ):
        raise NonResultError("synthetic_workload_registry_hash_mismatch")
    workload_registry = _validate_workload_registry_v2(workload_bytes, policy=validated)
    workloads_by_stage = {
        workload["stage"]: workload for workload in workload_registry["workloads"]
    }

    measurement_digest = hashlib.sha256(
        canonical_json_bytes(validated["measurement_contract"])
    ).hexdigest()
    bindings = {binding["stage"]: binding for binding in validated["stage_bindings"]}
    for stage in RESOURCE_STAGES_V2:
        binding = bindings[stage]
        contract_bytes = read_regular_bytes(stage_contracts[stage])
        if hashlib.sha256(contract_bytes).hexdigest() != binding["stage_contract_sha256"]:
            raise NonResultError("stage_contract_hash_mismatch")
        contract = _require_mapping(
            strict_json_load_bytes(contract_bytes), code="invalid_resource_stage_contract"
        )
        _require_canonical_control_json(
            contract_bytes, contract, code="noncanonical_resource_stage_contract"
        )
        if set(contract) != _RESOURCE_STAGE_CONTRACT_KEYS_V2:
            raise NonResultError("resource_stage_contract_schema_mismatch")
        if (
            contract["schema_version"] != RESOURCE_STAGE_CONTRACT_SCHEMA_V2
            or contract["stage"] != stage
            or contract["measurement_contract_sha256"] != measurement_digest
        ):
            raise NonResultError("resource_stage_contract_binding_mismatch")
        for key in (
            "runner_sha256",
            "arguments_sha256",
            "measurement_contract_sha256",
        ):
            value = contract[key]
            if not isinstance(value, str) or SHA256.fullmatch(value) is None:
                raise NonResultError("invalid_resource_stage_contract_hash")
        if stage in workloads_by_stage:
            workload = workloads_by_stage[stage]
            if (
                workload["generator_sha256"] != contract["runner_sha256"]
                or workload["arguments_sha256"] != contract["arguments_sha256"]
            ):
                raise NonResultError("synthetic_workload_stage_contract_mismatch")

        runner_bytes = read_regular_bytes(stage_runners[stage])
        if (
            not runner_bytes
            or hashlib.sha256(runner_bytes).hexdigest() != contract["runner_sha256"]
        ):
            raise NonResultError("stage_runner_hash_mismatch")
        argument_bytes = read_regular_bytes(stage_arguments[stage])
        if hashlib.sha256(argument_bytes).hexdigest() != contract["arguments_sha256"]:
            raise NonResultError("stage_arguments_hash_mismatch")
        _validate_stage_arguments_v2(argument_bytes, stage=stage)

        manifest_bytes = read_regular_bytes(input_manifests[stage])
        if hashlib.sha256(manifest_bytes).hexdigest() != binding["input_manifest_sha256"]:
            raise NonResultError("stage_input_manifest_hash_mismatch")
        manifest = _require_mapping(
            strict_json_load_bytes(manifest_bytes), code="invalid_stage_input_manifest"
        )
        _require_canonical_control_json(
            manifest_bytes, manifest, code="noncanonical_stage_input_manifest"
        )
        if set(manifest) != _RESOURCE_STAGE_INPUT_MANIFEST_KEYS_V2:
            raise NonResultError("stage_input_manifest_schema_mismatch")
        if (
            manifest["schema_version"] != RESOURCE_STAGE_INPUT_MANIFEST_SCHEMA_V2
            or manifest["stage"] != stage
            or not isinstance(manifest["files"], list)
            or not manifest["files"]
        ):
            raise NonResultError("invalid_stage_input_manifest")

        paths: list[str] = []
        total_bytes = 0
        declared_files: list[dict[str, Any]] = []
        for raw_file in manifest["files"]:
            record = dict(_require_mapping(raw_file, code="invalid_stage_input_record"))
            if set(record) != _RESOURCE_STAGE_INPUT_FILE_KEYS_V2:
                raise NonResultError("stage_input_record_schema_mismatch")
            relative = _safe_relative(record["path"])
            expected_size = _require_nonnegative_integer(
                record["size_bytes"], code="invalid_stage_input_size"
            )
            expected_digest = record["sha256"]
            if not isinstance(expected_digest, str) or SHA256.fullmatch(expected_digest) is None:
                raise NonResultError("invalid_stage_input_hash")
            record["path"] = relative
            declared_files.append(record)
            paths.append(relative)
            total_bytes += expected_size
        if paths != sorted(paths) or len(set(paths)) != len(paths):
            raise NonResultError("noncanonical_stage_input_manifest_order")
        if total_bytes != binding["expected_input_bytes"]:
            raise NonResultError("stage_input_bytes_mismatch")
        if live_mode:
            _attest_manifest_members(input_roots[stage], declared_files)
        else:
            _validate_member_attestation_v2(
                read_regular_bytes(input_member_attestations[stage]),
                schema_version=RESOURCE_STAGE_MEMBER_ATTESTATION_SCHEMA_V2,
                scope="stage_input",
                expected_bindings={
                    "stage": stage,
                    "stage_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
                    "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
                    "arguments_sha256": hashlib.sha256(argument_bytes).hexdigest(),
                    "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                },
                declared_members=declared_files,
            )


def capture_resource_policy_binding_evidence(
    policy: Mapping[str, Any],
    *,
    verified_at_utc: str,
    scheduler_snapshot: str | Path,
    source_capsule: str | Path,
    source_root: str | Path,
    synthetic_workload_registry: str | Path,
    stage_contracts: Mapping[str, str | Path],
    stage_runners: Mapping[str, str | Path],
    stage_arguments: Mapping[str, str | Path],
    input_manifests: Mapping[str, str | Path],
    input_roots: Mapping[str, str | Path],
) -> dict[str, bytes]:
    """Capture hash-only policy-binding evidence after two live no-follow passes.

    Scientific input bytes are hashed in place but are not copied into the
    admission bundle. The returned payloads contain only control artifacts,
    code/configuration, manifests, and per-member identity attestations.
    """
    timestamp = _validate_utc_timestamp(
        verified_at_utc, code="invalid_resource_member_attestation_timestamp"
    )
    live_bindings = {
        "scheduler_snapshot": scheduler_snapshot,
        "source_capsule": source_capsule,
        "source_root": source_root,
        "synthetic_workload_registry": synthetic_workload_registry,
        "stage_contracts": stage_contracts,
        "stage_runners": stage_runners,
        "stage_arguments": stage_arguments,
        "input_manifests": input_manifests,
        "input_roots": input_roots,
    }
    verify_resource_policy_bindings(policy, **live_bindings)

    payloads: dict[str, bytes] = {
        "policy_bindings/scheduler_snapshot.json": read_regular_bytes(scheduler_snapshot),
        "policy_bindings/source_capsule.json": read_regular_bytes(source_capsule),
        "policy_bindings/synthetic_workload_registry.json": read_regular_bytes(
            synthetic_workload_registry
        ),
    }
    source_capsule_bytes = payloads["policy_bindings/source_capsule.json"]
    source_payload = _validate_source_capsule_v2(source_capsule_bytes)
    source_members = _attest_manifest_members(source_root, source_payload["members"])
    payloads[RESOURCE_SOURCE_MEMBER_ATTESTATION_FILENAME_V2] = canonical_json_bytes(
        {
            "schema_version": RESOURCE_SOURCE_MEMBER_ATTESTATION_SCHEMA_V2,
            "scope": "source_capsule",
            "verified_at_utc": timestamp,
            "source_capsule_sha256": hashlib.sha256(source_capsule_bytes).hexdigest(),
            "members": source_members,
        }
    )

    for stage in RESOURCE_STAGES_V2:
        names = RESOURCE_BINDING_STAGE_FILENAMES_V2[stage]
        contract_bytes = read_regular_bytes(stage_contracts[stage])
        runner_bytes = read_regular_bytes(stage_runners[stage])
        argument_bytes = read_regular_bytes(stage_arguments[stage])
        manifest_bytes = read_regular_bytes(input_manifests[stage])
        payloads[names["contract"]] = contract_bytes
        payloads[names["runner"]] = runner_bytes
        payloads[names["arguments"]] = argument_bytes
        payloads[names["input_manifest"]] = manifest_bytes
        manifest = dict(
            _require_mapping(
                strict_json_load_bytes(manifest_bytes), code="invalid_stage_input_manifest"
            )
        )
        members = _attest_manifest_members(input_roots[stage], manifest["files"])
        payloads[names["member_attestation"]] = canonical_json_bytes(
            {
                "schema_version": RESOURCE_STAGE_MEMBER_ATTESTATION_SCHEMA_V2,
                "scope": "stage_input",
                "stage": stage,
                "verified_at_utc": timestamp,
                "stage_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
                "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
                "arguments_sha256": hashlib.sha256(argument_bytes).hexdigest(),
                "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "members": members,
            }
        )

    _binding_evidence_closure_sha256(payloads)
    verify_resource_policy_bindings(policy, **live_bindings)
    return payloads


def validate_resource_admission_receipt(
    payload: Any,
    *,
    policy: Mapping[str, Any],
    resource_policy_sha256: str,
) -> dict[str, Any]:
    """Validate receipt structure and policy hashes, not operational evidence."""
    validated_policy = _validate_resource_policy_v2(dict(policy))
    admission = dict(_require_mapping(payload, code="invalid_resource_admission"))
    if set(admission) != _RESOURCE_ADMISSION_KEYS_V2:
        raise NonResultError("resource_admission_schema_mismatch")
    if (
        admission["schema_version"] != "e4-resource-admission/v2"
        or admission["admission_status"] != "PASS"
        or admission["policy_class"] != RESOURCE_POLICY_CLASS_V2
        or admission["synthetic_fixture_only"] is not False
    ):
        raise NonResultError("invalid_resource_admission")
    if (
        not isinstance(resource_policy_sha256, str)
        or SHA256.fullmatch(resource_policy_sha256) is None
    ):
        raise NonResultError("invalid_resource_policy_binding")
    if admission["resource_policy_sha256"] != resource_policy_sha256:
        raise NonResultError("resource_policy_admission_mismatch")
    for key in (
        "scheduler_snapshot_sha256",
        "source_capsule_sha256",
        "synthetic_workload_registry_sha256",
    ):
        if admission[key] != validated_policy[key]:
            raise NonResultError("resource_admission_governing_hash_mismatch")
    for key in (
        "resource_policy_sha256",
        "scheduler_snapshot_sha256",
        "source_capsule_sha256",
        "synthetic_workload_registry_sha256",
        "telemetry_sha256",
        "measurement_attestation_closure_sha256",
        "binding_evidence_closure_sha256",
    ):
        value = admission[key]
        if not isinstance(value, str) or SHA256.fullmatch(value) is None:
            raise NonResultError("invalid_resource_admission_hash")
    return admission


def _measurement_attestation_closure_sha256(attestations: Sequence[bytes]) -> str:
    if len(attestations) != len(RESOURCE_ATTESTATION_FILENAMES_V2):
        raise NonResultError("measurement_attestation_count_mismatch")
    closure = b"".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode()
        for name, data in zip(RESOURCE_ATTESTATION_FILENAMES_V2, attestations, strict=True)
    )
    return hashlib.sha256(closure).hexdigest()


def validate_resource_admission_evidence_files(
    root: str | Path,
    *,
    expected_policy_sha256: str,
) -> ResourceAdmissionEvidence:
    """Recompute a real-pilot admission from its exact raw control artifacts."""
    if (
        not isinstance(expected_policy_sha256, str)
        or SHA256.fullmatch(expected_policy_sha256) is None
    ):
        raise NonResultError("invalid_resource_policy_binding")
    base = Path(root)
    payloads = {
        name: read_regular_bytes(base / name) for name in RESOURCE_ADMISSION_EVIDENCE_FILES_V2
    }

    policy_bytes = payloads["resource_policy.json"]
    if hashlib.sha256(policy_bytes).hexdigest() != expected_policy_sha256:
        raise NonResultError("resource_policy_admission_mismatch")
    policy_payload = strict_json_load_bytes(policy_bytes)
    _require_canonical_control_json(
        policy_bytes, policy_payload, code="noncanonical_resource_policy"
    )
    policy = validate_resource_policy(policy_payload)

    admission_bytes = payloads["resource_admission.json"]
    admission_payload = strict_json_load_bytes(admission_bytes)
    _require_canonical_control_json(
        admission_bytes, admission_payload, code="noncanonical_resource_admission"
    )
    admission = validate_resource_admission_receipt(
        admission_payload,
        policy=policy,
        resource_policy_sha256=expected_policy_sha256,
    )

    binding_payloads = {name: payloads[name] for name in RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2}
    if (
        _binding_evidence_closure_sha256(binding_payloads)
        != admission["binding_evidence_closure_sha256"]
    ):
        raise NonResultError("resource_binding_evidence_digest_mismatch")
    verify_resource_policy_bindings(
        policy,
        scheduler_snapshot=base / "policy_bindings/scheduler_snapshot.json",
        source_capsule=base / "policy_bindings/source_capsule.json",
        source_member_attestation=base / RESOURCE_SOURCE_MEMBER_ATTESTATION_FILENAME_V2,
        synthetic_workload_registry=(base / "policy_bindings/synthetic_workload_registry.json"),
        stage_contracts={
            stage: base / RESOURCE_BINDING_STAGE_FILENAMES_V2[stage]["contract"]
            for stage in RESOURCE_STAGES_V2
        },
        stage_runners={
            stage: base / RESOURCE_BINDING_STAGE_FILENAMES_V2[stage]["runner"]
            for stage in RESOURCE_STAGES_V2
        },
        stage_arguments={
            stage: base / RESOURCE_BINDING_STAGE_FILENAMES_V2[stage]["arguments"]
            for stage in RESOURCE_STAGES_V2
        },
        input_manifests={
            stage: base / RESOURCE_BINDING_STAGE_FILENAMES_V2[stage]["input_manifest"]
            for stage in RESOURCE_STAGES_V2
        },
        input_member_attestations={
            stage: base / RESOURCE_BINDING_STAGE_FILENAMES_V2[stage]["member_attestation"]
            for stage in RESOURCE_STAGES_V2
        },
    )

    telemetry_bytes = payloads["stage_telemetry.json"]
    if hashlib.sha256(telemetry_bytes).hexdigest() != admission["telemetry_sha256"]:
        raise NonResultError("resource_telemetry_hash_mismatch")
    telemetry_payload = strict_json_load_bytes(telemetry_bytes)
    _require_canonical_control_json(
        telemetry_bytes, telemetry_payload, code="noncanonical_resource_telemetry"
    )
    if not isinstance(telemetry_payload, list):
        raise NonResultError("invalid_resource_telemetry")

    attestations = [payloads[name] for name in RESOURCE_ATTESTATION_FILENAMES_V2]
    for data in attestations:
        attestation_payload = strict_json_load_bytes(data)
        _require_canonical_control_json(
            data,
            attestation_payload,
            code="noncanonical_resource_measurement_attestation",
        )
    if (
        _measurement_attestation_closure_sha256(attestations)
        != admission["measurement_attestation_closure_sha256"]
    ):
        raise NonResultError("measurement_attestation_closure_mismatch")
    validate_resource_telemetry(
        telemetry_payload,
        policy,
        measurement_attestation_bytes=attestations,
    )

    audit_bytes = payloads["forbidden_output_audit.json"]
    audit = _require_mapping(
        strict_json_load_bytes(audit_bytes), code="invalid_forbidden_output_audit"
    )
    _require_canonical_control_json(audit_bytes, audit, code="noncanonical_forbidden_output_audit")
    if (
        set(audit) != _RESOURCE_FORBIDDEN_OUTPUT_AUDIT_KEYS_V2
        or audit["schema_version"] != "e4-resource-forbidden-output-audit/v2"
        or audit["scientific_endpoint_called"] is not False
        or isinstance(audit["scientific_output_count"], bool)
        or audit["scientific_output_count"] != 0
        or audit["audit_completed"] is not True
    ):
        raise NonResultError("forbidden_output_audit_failed")
    return ResourceAdmissionEvidence(
        bundle_receipt=None,
        policy=policy,
        admission=admission,
        payloads=payloads,
    )


def verify_resource_admission_bundle(
    bundle_path: str | Path,
    *,
    expected_policy_sha256: str,
) -> ResourceAdmissionEvidence:
    """Verify exact closure and raw evidence for a real resource admission."""
    receipt = verify_nonresult_bundle(bundle_path, expected_type="resource_admission")
    evidence = validate_resource_admission_evidence_files(
        receipt.bundle_path,
        expected_policy_sha256=expected_policy_sha256,
    )
    final_receipt = verify_nonresult_bundle(receipt.bundle_path, expected_type="resource_admission")
    if final_receipt != receipt:
        raise NonResultError("bundle_changed_during_semantic_verification")
    return ResourceAdmissionEvidence(
        bundle_receipt=receipt,
        policy=evidence.policy,
        admission=evidence.admission,
        payloads=evidence.payloads,
    )


def verify_embedded_resource_admission_provenance(
    root: str | Path,
    *,
    expected_closure_sha256: str,
) -> None:
    """Reconstruct the original admission closure from preflight copies."""
    if (
        not isinstance(expected_closure_sha256, str)
        or SHA256.fullmatch(expected_closure_sha256) is None
    ):
        raise NonResultError("invalid_resource_admission_closure")
    base = Path(root)
    manifest_bytes = read_regular_bytes(base / "resource_admission_bundle_manifest.json")
    checksum_bytes = read_regular_bytes(base / "resource_admission_bundle_checksums.sha256")
    if hashlib.sha256(checksum_bytes).hexdigest() != expected_closure_sha256:
        raise NonResultError("embedded_resource_admission_closure_mismatch")
    manifest = dict(
        _require_mapping(
            strict_json_load_bytes(manifest_bytes), code="invalid_embedded_admission_manifest"
        )
    )
    _require_canonical_control_json(
        manifest_bytes, manifest, code="noncanonical_embedded_admission_manifest"
    )
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMAS["resource_admission"]
        or manifest.get("bundle_type") != "resource_admission"
        or not isinstance(manifest.get("bundle_id"), str)
        or _safe_identifier(manifest["bundle_id"]) != manifest["bundle_id"]
        or manifest.get("expected_files") != sorted(REQUIRED_BUNDLE_FILES["resource_admission"])
    ):
        raise NonResultError("invalid_embedded_admission_manifest")
    checksums = _parse_checksum_file(checksum_bytes)
    expected = REQUIRED_BUNDLE_FILES["resource_admission"] - {"output_checksums.sha256"}
    if set(checksums) != expected:
        raise NonResultError("embedded_resource_admission_checksum_closure_mismatch")
    for name, expected_digest in checksums.items():
        data = (
            manifest_bytes
            if name == "resource_admission_manifest.json"
            else read_regular_bytes(base / name)
        )
        if hashlib.sha256(data).hexdigest() != expected_digest:
            raise NonResultError("embedded_resource_admission_checksum_mismatch")


def validate_legacy_synthetic_resource_policy(payload: Any) -> dict[str, Any]:
    """Validate the migration-only v1 policy used by synthetic guard bundles.

    This function is intentionally separate from ``validate_resource_policy``
    so a v1 fixture can never enter real-pilot preflight by schema fallback.
    """
    policy = dict(_require_mapping(payload, code="invalid_resource_policy"))
    if policy.get("schema_version") != "e4-resource-policy/v1":
        raise NonResultError("invalid_legacy_resource_policy_schema")
    keys = (
        "allocation_memory_bytes",
        "memory_reserve_bytes",
        "allocation_wall_seconds",
        "wall_reserve_seconds",
        "scratch_capacity_bytes",
        "scratch_reserve_bytes",
    )
    if set(policy) != {"schema_version", *keys}:
        raise NonResultError("legacy_resource_policy_schema_mismatch")
    for key in keys:
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NonResultError("invalid_resource_policy_value")
    for capacity, reserve in (
        ("allocation_memory_bytes", "memory_reserve_bytes"),
        ("allocation_wall_seconds", "wall_reserve_seconds"),
        ("scratch_capacity_bytes", "scratch_reserve_bytes"),
    ):
        if policy[reserve] >= policy[capacity]:
            raise NonResultError("resource_reserve_exceeds_capacity")
    return policy


def validate_resource_telemetry(
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    measurement_attestation_bytes: Sequence[bytes],
) -> None:
    """Validate v2 telemetry and its raw, hash-bound stage attestations."""
    validated_policy = _validate_resource_policy_v2(dict(policy))
    _validate_resource_telemetry_v2(
        rows,
        validated_policy,
        measurement_attestation_bytes=measurement_attestation_bytes,
    )


def validate_legacy_synthetic_resource_telemetry(
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> None:
    """Validate migration-only telemetry from the synthetic v1 guard."""
    policy = validate_legacy_synthetic_resource_policy(policy)
    if not rows:
        raise NonResultError("missing_resource_telemetry")
    for row in rows:
        if set(row) != set(RESOURCE_TELEMETRY_COLUMNS):
            raise NonResultError("resource_telemetry_schema_mismatch")
        if _require_nonblank(row["stage"], code="invalid_resource_stage") == "scientific":
            raise NonResultError("forbidden_resource_stage")
        if row["exit_status"] != 0:
            raise NonResultError("resource_pilot_nonzero_exit")
        for key in RESOURCE_TELEMETRY_COLUMNS[1:-1]:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise NonResultError("invalid_resource_telemetry_value")
        if row["peak_rss_bytes"] > (
            policy["allocation_memory_bytes"] - policy["memory_reserve_bytes"]
        ):
            raise NonResultError("resource_memory_limit_exceeded")
        if row["wall_seconds"] > (
            policy["allocation_wall_seconds"] - policy["wall_reserve_seconds"]
        ):
            raise NonResultError("resource_wall_limit_exceeded")
        if row["scratch_bytes"] > (
            policy["scratch_capacity_bytes"] - policy["scratch_reserve_bytes"]
        ):
            raise NonResultError("resource_scratch_limit_exceeded")


def _ceil_nanoseconds_to_seconds(value: int) -> int:
    return (value + 999_999_999) // 1_000_000_000


def _require_nonnegative_attestation_integer(value: Any) -> int:
    return _require_nonnegative_integer(value, code="invalid_measurement_attestation_value")


def _validate_measurement_attestation_v2(
    payload: Any,
    *,
    row: Mapping[str, Any],
    binding: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    attestation = _require_mapping(payload, code="invalid_measurement_attestation")
    if set(attestation) != _RESOURCE_MEASUREMENT_ATTESTATION_KEYS_V2:
        raise NonResultError("measurement_attestation_schema_mismatch")
    if (
        attestation["schema_version"] != RESOURCE_MEASUREMENT_ATTESTATION_SCHEMA_V2
        or attestation["stage"] != row["stage"]
        or attestation["stage_contract_sha256"] != binding["stage_contract_sha256"]
        or attestation["input_manifest_sha256"] != binding["input_manifest_sha256"]
    ):
        raise NonResultError("measurement_attestation_binding_mismatch")
    measurement_digest = hashlib.sha256(
        canonical_json_bytes(policy["measurement_contract"])
    ).hexdigest()
    if attestation["measurement_contract_sha256"] != measurement_digest:
        raise NonResultError("measurement_attestation_contract_mismatch")
    for key in (
        "stage_contract_sha256",
        "input_manifest_sha256",
        "measurement_contract_sha256",
        "cgroup_identity_sha256",
        "scratch_root_identity_sha256",
    ):
        value = attestation[key]
        if not isinstance(value, str) or SHA256.fullmatch(value) is None:
            raise NonResultError("invalid_measurement_attestation_hash")

    integer_fields = (
        "wall_start_monotonic_ns",
        "wall_end_monotonic_ns",
        "wall_seconds",
        "cpu_self_user_before_ns",
        "cpu_self_user_after_ns",
        "cpu_self_system_before_ns",
        "cpu_self_system_after_ns",
        "cpu_children_user_before_ns",
        "cpu_children_user_after_ns",
        "cpu_children_system_before_ns",
        "cpu_children_system_after_ns",
        "cpu_seconds",
        "peak_rss_bytes",
        "scratch_sampler_cadence_ns",
        "scratch_first_sample_monotonic_ns",
        "scratch_last_sample_monotonic_ns",
        "scratch_max_sample_gap_ns",
        "scratch_missed_samples",
        "scratch_peak_allocated_bytes",
        "exit_status",
    )
    for key in integer_fields:
        _require_nonnegative_attestation_integer(attestation[key])
    if attestation["wall_end_monotonic_ns"] <= attestation["wall_start_monotonic_ns"]:
        raise NonResultError("invalid_measurement_attestation_wall_interval")
    wall_seconds = _ceil_nanoseconds_to_seconds(
        attestation["wall_end_monotonic_ns"] - attestation["wall_start_monotonic_ns"]
    )
    if attestation["wall_seconds"] != wall_seconds or row["wall_seconds"] != wall_seconds:
        raise NonResultError("measurement_attestation_wall_mismatch")

    cpu_pairs = (
        ("cpu_self_user_before_ns", "cpu_self_user_after_ns"),
        ("cpu_self_system_before_ns", "cpu_self_system_after_ns"),
        ("cpu_children_user_before_ns", "cpu_children_user_after_ns"),
        ("cpu_children_system_before_ns", "cpu_children_system_after_ns"),
    )
    cpu_delta_ns = 0
    for before_key, after_key in cpu_pairs:
        before = attestation[before_key]
        after = attestation[after_key]
        if after < before:
            raise NonResultError("invalid_measurement_attestation_cpu_interval")
        cpu_delta_ns += after - before
    cpu_seconds = _ceil_nanoseconds_to_seconds(cpu_delta_ns)
    if attestation["cpu_seconds"] != cpu_seconds or row["cpu_seconds"] != cpu_seconds:
        raise NonResultError("measurement_attestation_cpu_mismatch")

    cadence = policy["measurement_contract"]["scratch_sampler_cadence_ns"]
    if (
        attestation["cgroup_v2_memory_peak_available"] is not True
        or attestation["scratch_sampler_cadence_ns"] != cadence
        or attestation["scratch_first_sample_monotonic_ns"] > attestation["wall_start_monotonic_ns"]
        or attestation["scratch_last_sample_monotonic_ns"] < attestation["wall_end_monotonic_ns"]
        or attestation["scratch_max_sample_gap_ns"] > cadence
        or attestation["scratch_missed_samples"] != 0
        or attestation["scratch_final_scan_completed"] is not True
        or attestation["scratch_sampler_failed"] is not False
        or attestation["scratch_escape_detected"] is not False
    ):
        raise NonResultError("resource_measurement_attestation_failed")
    if (
        attestation["peak_rss_bytes"] != row["peak_rss_bytes"]
        or attestation["scratch_peak_allocated_bytes"] != row["scratch_bytes"]
        or attestation["exit_status"] != row["exit_status"]
    ):
        raise NonResultError("measurement_attestation_telemetry_mismatch")


def _validate_resource_telemetry_v2(
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    measurement_attestation_bytes: Sequence[bytes],
) -> None:
    if len(rows) != len(RESOURCE_STAGES_V2):
        raise NonResultError("resource_telemetry_v2_stage_sequence_mismatch")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(RESOURCE_TELEMETRY_COLUMNS_V2):
            raise NonResultError("resource_telemetry_v2_schema_mismatch")
    if tuple(row["stage"] for row in rows) != RESOURCE_STAGES_V2:
        raise NonResultError("resource_telemetry_v2_stage_sequence_mismatch")
    if len(measurement_attestation_bytes) != len(RESOURCE_STAGES_V2):
        raise NonResultError("measurement_attestation_count_mismatch")

    metric_keys = (
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "input_bytes",
        "scratch_bytes",
    )
    stage_bindings = policy["stage_bindings"]
    for row, binding, attestation_bytes in zip(
        rows, stage_bindings, measurement_attestation_bytes, strict=True
    ):
        if (
            isinstance(row["exit_status"], bool)
            or not isinstance(row["exit_status"], int)
            or row["exit_status"] != 0
        ):
            raise NonResultError("resource_pilot_nonzero_exit")
        for key in metric_keys:
            _require_nonnegative_integer(row[key], code="invalid_resource_telemetry_v2_value")
        for key in (
            "input_manifest_sha256",
            "stage_contract_sha256",
            "measurement_attestation_sha256",
        ):
            value = row[key]
            if not isinstance(value, str) or SHA256.fullmatch(value) is None:
                raise NonResultError("invalid_resource_telemetry_v2_hash")
        if (
            row["input_manifest_sha256"] != binding["input_manifest_sha256"]
            or row["stage_contract_sha256"] != binding["stage_contract_sha256"]
            or row["input_bytes"] != binding["expected_input_bytes"]
        ):
            raise NonResultError("resource_stage_binding_mismatch")
        if row["cpu_seconds"] > policy["allocation_cpus"] * row["wall_seconds"]:
            raise NonResultError("resource_stage_cpu_plausibility_failed")
        for observed, limit_key in (
            ("wall_seconds", "wall_limit_seconds"),
            ("cpu_seconds", "cpu_limit_seconds"),
            ("peak_rss_bytes", "peak_rss_limit_bytes"),
            ("scratch_bytes", "scratch_limit_bytes"),
        ):
            if row[observed] > binding[limit_key]:
                raise NonResultError("resource_stage_limit_exceeded")
        if hashlib.sha256(attestation_bytes).hexdigest() != row["measurement_attestation_sha256"]:
            raise NonResultError("measurement_attestation_hash_mismatch")
        attestation = strict_json_load_bytes(attestation_bytes)
        _validate_measurement_attestation_v2(
            attestation,
            row=row,
            binding=binding,
            policy=policy,
        )

    effective_wall, effective_memory, effective_scratch = _effective_resource_limits(policy)
    if sum(row["wall_seconds"] for row in rows) > effective_wall:
        raise NonResultError("resource_aggregate_wall_limit_exceeded")
    if sum(row["cpu_seconds"] for row in rows) > policy["allocation_cpus"] * effective_wall:
        raise NonResultError("resource_aggregate_cpu_limit_exceeded")
    if max(row["peak_rss_bytes"] for row in rows) > effective_memory:
        raise NonResultError("resource_memory_limit_exceeded")
    if max(row["scratch_bytes"] for row in rows) > effective_scratch:
        raise NonResultError("resource_scratch_limit_exceeded")


def csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize a fixed-column CSV with no implicit fields or nonfinite values."""
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="raise")
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise NonResultError("csv_schema_mismatch")
        _reject_nonfinite(row)
        writer.writerow(dict(row))
    return buffer.getvalue().encode("utf-8")


def source_identity_payload(path: str | Path) -> dict[str, Any]:
    """Return safe file identity data for a predeclared non-result input."""
    candidate = Path(path)
    descriptor = _open_regular_no_follow(candidate)
    try:
        info, digest = _hash_regular_descriptor(descriptor)
    finally:
        os.close(descriptor)
    return {
        "sha256": digest,
        "size_bytes": info.st_size,
        "filename": candidate.name,
    }


def source_mineral_rows(records: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert metadata records to the fixed mapping-only inventory schema."""
    output = []
    for record in records:
        row = asdict(record)
        if set(row) != {"index", "name", "group", "library"}:
            raise NonResultError("invalid_source_mineral_record")
        output.append(row)
    return output


__all__ = [
    "BUNDLE_SCHEMAS",
    "MANIFEST_NAMES",
    "REQUIRED_BUNDLE_FILES",
    "RESOURCE_ADMISSION_EVIDENCE_FILES_V2",
    "RESOURCE_ADMISSION_PROVENANCE_FILES_V2",
    "RESOURCE_ATTESTATION_FILENAMES_V2",
    "RESOURCE_BINDING_STAGE_FILENAMES_V2",
    "RESOURCE_MEASUREMENT_CONTRACT_V2",
    "RESOURCE_MEASUREMENT_ATTESTATION_SCHEMA_V2",
    "RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2",
    "RESOURCE_POLICY_CLASS_V2",
    "RESOURCE_STAGES_V2",
    "RESOURCE_STAGE_CONTRACT_SCHEMA_V2",
    "RESOURCE_STAGE_INPUT_MANIFEST_SCHEMA_V2",
    "RESOURCE_SOURCE_CAPSULE_SCHEMA_V2",
    "RESOURCE_SOURCE_MEMBER_ATTESTATION_FILENAME_V2",
    "RESOURCE_SOURCE_MEMBER_ATTESTATION_SCHEMA_V2",
    "RESOURCE_STAGE_ARGUMENTS_SCHEMA_V2",
    "RESOURCE_STAGE_MEMBER_ATTESTATION_SCHEMA_V2",
    "RESOURCE_TELEMETRY_COLUMNS",
    "RESOURCE_TELEMETRY_COLUMNS_V2",
    "RESOURCE_WORKLOAD_REGISTRY_SCHEMA_V2",
    "BundleReceipt",
    "NonResultError",
    "ResourceAdmissionEvidence",
    "atomic_write_bundle",
    "capture_resource_policy_binding_evidence",
    "canonical_json_bytes",
    "csv_bytes",
    "read_regular_bytes",
    "sha256_regular_file",
    "source_identity_payload",
    "source_mineral_rows",
    "strict_json_load_bytes",
    "validate_decision_record",
    "validate_legacy_synthetic_resource_policy",
    "validate_legacy_synthetic_resource_telemetry",
    "validate_resource_admission_evidence_files",
    "validate_resource_admission_receipt",
    "validate_resource_policy",
    "validate_resource_telemetry",
    "verify_resource_policy_bindings",
    "verify_embedded_resource_admission_provenance",
    "verify_nonresult_bundle",
    "verify_resource_admission_bundle",
]
