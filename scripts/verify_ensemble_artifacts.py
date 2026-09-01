"""Verify a completed E6 ensemble run without disclosing scientific results.

The verifier is deliberately post-run and closed-world. It requires an explicit
run directory, never resolves artifact-declared paths, rejects links and special
files, and emits only operational conformance records. Scientific values are
used only for internal gate-consistency checks and are never included in output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

FROZEN_PREREGISTRATION_SHA256 = "4c228fac93828d039c36b535331dce36411717e38c68ef2fc1a355b73fdacb22"
FROZEN_SITES = ("goldfield", "bingham")
ANCHOR_SCENES = {
    "goldfield": "20240925_185504_87_4001",
    "bingham": "20250911_191523_58_4001",
}
TARGET_MINERALS = (
    "alunite",
    "kaolinite",
    "dickite",
    "jarosite",
    "hematite",
    "goethite",
    "gypsum",
    "muscovite",
)
ROCKWELL_MINERALS = tuple(mineral for mineral in TARGET_MINERALS if mineral != "gypsum")
CANDIDATE_COUNTS = {
    "alunite": 11,
    "kaolinite": 8,
    "dickite": 2,
    "jarosite": 9,
    "hematite": 12,
    "goethite": 8,
    "gypsum": 6,
    "muscovite": 16,
}
BASELINE_ENDMEMBERS = {
    "alunite": "splib07a_Alunite_SUSTDA-20_BECKb_AREF.txt",
    "kaolinite": "splib07a_Kaolinite_CM5_BECKb_AREF.txt",
    "dickite": "splib07a_Dickite_NMNH106242_BECKb_AREF.txt",
    "jarosite": "splib07a_Jarosite_GDS101_Na_200C_Syn_BECKa_AREF.txt",
    "hematite": "splib07a_Hematite_GDS69.e_20-30um_BECKb_AREF.txt",
    "goethite": "splib07a_Goethite_MPCMA2-C_M-Crsgrad2_BECKb_AREF.txt",
    "gypsum": "splib07a_Gypsum_HS333.2B_(Selenite)_ASDFRa_AREF.txt",
    "muscovite": "splib07a_Muscovite_GDS118_Capitan_BECKa_AREF.txt",
}
GOVERNING_FILES = (
    "src/tanager_rocks/ensemble_sensitivity.py",
    "scripts/run_ensemble_sensitivity.py",
    "tests/test_ensemble_sensitivity.py",
    "docs/m2_ensemble_sensitivity_preregistration.md",
    "docs/m2_spatial_validation_preregistration.md",
    "docs/tanager_quality_mask_policy.md",
    "src/tanager_rocks/spatial_validation.py",
    "src/tanager_rocks/strict_inductive.py",
    "src/tanager_rocks/unmix.py",
    "src/tanager_rocks/quality.py",
    "src/tanager_rocks/speclib.py",
    "src/tanager_rocks/reference.py",
    "src/tanager_rocks/config.py",
    "src/tanager_rocks/pipeline.py",
    "src/tanager_rocks/viz.py",
)

RIDGES = ("0.001", "0.01", "0.1")
QUANTILES = ("0.85", "0.9", "0.95")
GATES = ("none", "1")
STOCHASTIC_REPLICATES = 16
BOOTSTRAP_REPLICATES = 10_000
RETAINED_BANDS = 363
ANALYTICAL_CELLS = 18
VARIANTS_PER_SITE = 355
TOTAL_VARIANTS = 710
UNIQUE_FITS_PER_SITE = 83
TOTAL_UNIQUE_FITS = 166
JOINT_MEMBERS_PER_SITE = 288
EXPECTED_SOURCE_MANIFEST_ENTRIES = 49

FIT_CACHE_KEYS = frozenset(
    {
        "valid_support",
        "contributing_pixels",
        "retained_bands",
        *(f"mf_{mineral}" for mineral in TARGET_MINERALS),
        *(f"infeas_{mineral}" for mineral in TARGET_MINERALS),
    }
)

HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
SOURCE_LINE = re.compile(r"(?P<sha256>[0-9a-f]{64})  (?P<path>[^\x00-\x1f\x7f]+)\Z")
SOURCE_SIBLING_PREFIX = "../tanager-spec/"
LEGACY_SOURCE_PREFIXES = ("tanager-rocks/", "tanager-spec/")

MEMBER_FIELDS = (
    "scene",
    "site",
    "member_id",
    "member_class",
    "fit_id",
    "stochastic_replicate",
    "covariance_mode",
    "calibration_mode",
    "covariance_draw",
    "calibration_draw",
    "covariance_seed_entropy",
    "calibration_seed_entropy",
    "ridge",
    "detection_quantile",
    "infeasibility_gate",
    *(f"endmember_{mineral}" for mineral in TARGET_MINERALS),
    "contributing_pixels",
    "retained_bands",
    "status",
    "failure_reason",
    "output_checksum",
    "wall_time_seconds",
    "peak_memory_bytes",
    "design_sha256",
)
METRIC_FIELDS = (
    "site",
    "scene",
    "mineral",
    "member_id",
    "member_class",
    "stochastic_replicate",
    "ridge",
    "detection_quantile",
    "infeasibility_gate",
    "aggregation",
    "block_scale",
    "block_id",
    "common_support_pixels",
    "common_support_loss_fraction",
    "detection_prevalence",
    "rank_correlation",
    "dominant_class_switch_frequency",
    "auc",
    "balanced_accuracy",
    "positive_f1",
    "negative_f1",
    "macro_f1",
    "tpr",
    "fpr",
    "prevalence",
    "external_status",
    "covariance_scope",
    "strict_covariance_exclusion_status",
)
FACTOR_FIELDS = (
    "site",
    "mineral",
    "block_scale",
    "factor",
    "level",
    "reference_level",
    "endpoint",
    "paired_delta_median",
    "interval_lower",
    "interval_upper",
    "scheduled_replicates",
    "valid_replicates",
    "finite_fraction",
    "interval_available",
    "unavailable_reason",
    "n_pairs",
    "complete_blocks",
    "paired_support_pixels",
    "contrast_status",
)
CALIBRATION_FIELDS = (
    "site",
    "mineral",
    "confidence_bin",
    "support_blocks",
    "support_pixels",
    "compatible_positive_rate",
    "interval_lower",
    "interval_upper",
    "scheduled_replicates",
    "valid_replicates",
    "finite_fraction",
    "interval_available",
    "unavailable_reason",
    "brier_score",
    "brier_interval_lower",
    "brier_interval_upper",
    "brier_interval_available",
    "brier_valid_replicates",
    "brier_finite_fraction",
    "expected_calibration_error",
    "ece_interval_lower",
    "ece_interval_upper",
    "ece_interval_available",
    "ece_valid_replicates",
    "ece_finite_fraction",
    "status",
)

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


class SealViolation(RuntimeError):
    """A sanitized verifier failure that never carries artifact content."""

    def __init__(
        self,
        check: str,
        discrepancy: str,
        *,
        filename: str | None = None,
        count: int | None = None,
    ) -> None:
        super().__init__(discrepancy)
        self.check = check
        self.discrepancy = discrepancy
        self.filename = filename
        self.count = count


@dataclass(frozen=True)
class FileRecord:
    """Link-safe identity captured during closed-world inventory."""

    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class CheckRecord:
    """One result-safe operational conformance record."""

    passed: bool
    check: str
    count: int | None = None
    filename: str | None = None
    digest: str | None = None
    discrepancy: str | None = None

    def render(self) -> str:
        fields = ["PASS" if self.passed else "FAIL", f"check={self.check}"]
        if self.count is not None:
            fields.append(f"count={self.count}")
        if self.filename is not None:
            fields.append(f"filename={json.dumps(self.filename, ensure_ascii=True)}")
        if self.digest is not None:
            fields.append(f"sha256={self.digest}")
        if self.discrepancy is not None:
            fields.append(f"discrepancy={self.discrepancy}")
        return " ".join(fields)


@dataclass
class VerificationReport:
    """Operational verifier result with deliberately constrained rendering."""

    checks: list[CheckRecord] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(record.passed for record in self.checks)

    def render(self) -> str:
        lines = [record.render() for record in self.checks]
        failures = sum(not record.passed for record in self.checks)
        lines.append(f"{'PASS' if self.passed else 'FAIL'} check=overall count={failures}")
        return "\n".join(lines)


def _safe_relative(value: str, *, check: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise SealViolation(check, "unsafe_relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SealViolation(check, "unsafe_relative_path")
    return path.as_posix()


def _open_directory_chain(path: Path) -> int:
    if any(part == ".." for part in path.parts):
        raise SealViolation("run_directory", "parent_component_rejected")
    current = os.open(os.sep if path.is_absolute() else ".", DIR_FLAGS)
    parts = path.parts[1:] if path.is_absolute() else path.parts
    try:
        for part in parts:
            if part in {"", ".", os.sep}:
                continue
            following = os.open(part, DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = following
        return current
    except Exception:
        os.close(current)
        raise


class SealedDirectory:
    """Descriptor-relative reader that never follows artifact links."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._root_fd: int | None = None
        self.records: dict[str, FileRecord] = {}

    def __enter__(self) -> SealedDirectory:
        try:
            self._root_fd = _open_directory_chain(self.path)
        except (OSError, SealViolation) as error:
            if isinstance(error, SealViolation):
                raise
            raise SealViolation("run_directory", "not_nofollow_directory") from error
        return self

    def __exit__(self, *_args: object) -> None:
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None

    @property
    def root_fd(self) -> int:
        if self._root_fd is None:
            raise RuntimeError("sealed directory is not open")
        return self._root_fd

    @contextmanager
    def open_file(self, relative: str) -> Iterator[int]:
        relative = _safe_relative(relative, check="artifact_path")
        parts = PurePosixPath(relative).parts
        current = os.dup(self.root_fd)
        opened: int | None = None
        try:
            prefix: list[str] = []
            for part in parts[:-1]:
                prefix.append(part)
                following = os.open(part, DIR_FLAGS, dir_fd=current)
                self._verify_open_identity("/".join(prefix), following, directory=True)
                os.close(current)
                current = following
            opened = os.open(parts[-1], FILE_FLAGS, dir_fd=current)
            self._verify_open_identity(relative, opened, directory=False)
            try:
                yield opened
            finally:
                self._verify_open_identity(relative, opened, directory=False)
        except OSError as error:
            raise SealViolation(
                "artifact_open", "nofollow_open_failed", filename=relative
            ) from error
        finally:
            if opened is not None:
                os.close(opened)
            os.close(current)

    def _verify_open_identity(self, relative: str, descriptor: int, *, directory: bool) -> None:
        observed = os.fstat(descriptor)
        expected = self.records.get(relative)
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(observed.st_mode):
            raise SealViolation("artifact_open", "special_file_rejected", filename=relative)
        if not directory and observed.st_nlink != 1:
            raise SealViolation("artifact_open", "hardlink_rejected", filename=relative)
        if expected is not None and (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        ) != (
            expected.device,
            expected.inode,
            expected.size,
            expected.modified_ns,
            expected.changed_ns,
        ):
            raise SealViolation("artifact_open", "artifact_changed_during_verification")

    def inventory(self, *, allowed_directories: set[str]) -> set[str]:
        self.records = {}
        self._walk(self.root_fd, "", allowed_directories)
        return {path for path, record in self.records.items() if stat.S_ISREG(record.mode)}

    def root_record(self) -> FileRecord:
        """Return the current identity of the descriptor-pinned run root."""
        metadata = os.fstat(self.root_fd)
        return FileRecord(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            links=metadata.st_nlink,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
            changed_ns=metadata.st_ctime_ns,
        )

    def _walk(self, descriptor: int, prefix: str, allowed_directories: set[str]) -> None:
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise SealViolation("artifact_closure", "directory_listing_failed") from error
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise SealViolation(
                    "artifact_closure", "entry_stat_failed", filename=relative
                ) from error
            record = FileRecord(
                device=metadata.st_dev,
                inode=metadata.st_ino,
                mode=metadata.st_mode,
                links=metadata.st_nlink,
                size=metadata.st_size,
                modified_ns=metadata.st_mtime_ns,
                changed_ns=metadata.st_ctime_ns,
            )
            self.records[relative] = record
            if stat.S_ISLNK(metadata.st_mode):
                raise SealViolation("artifact_closure", "symlink_rejected", filename=relative)
            if stat.S_ISDIR(metadata.st_mode):
                if relative not in allowed_directories:
                    raise SealViolation(
                        "artifact_closure", "unexpected_directory", filename=relative
                    )
                try:
                    child = os.open(name, DIR_FLAGS, dir_fd=descriptor)
                except OSError as error:
                    raise SealViolation(
                        "artifact_closure", "directory_nofollow_open_failed", filename=relative
                    ) from error
                try:
                    self._verify_open_identity(relative, child, directory=True)
                    self._walk(child, relative, allowed_directories)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise SealViolation("artifact_closure", "hardlink_rejected", filename=relative)
            else:
                raise SealViolation("artifact_closure", "special_file_rejected", filename=relative)

    def read_bytes(self, relative: str) -> bytes:
        with self.open_file(relative) as descriptor:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                return handle.read()

    def sha256(self, relative: str) -> str:
        digest = hashlib.sha256()
        with self.open_file(relative) as descriptor:
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()


@contextmanager
def _open_external_file(path: Path, *, check: str) -> Iterator[BinaryIO]:
    if not path.name or any(part == ".." for part in path.parts):
        raise SealViolation(check, "unsafe_external_path")
    parent = path.parent if str(path.parent) else Path(".")
    parent_fd: int | None = None
    descriptor: int | None = None
    try:
        parent_fd = _open_directory_chain(parent)
        descriptor = os.open(path.name, FILE_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        if parent_fd is not None:
            os.close(parent_fd)
        raise SealViolation(check, "nofollow_open_failed") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SealViolation(check, "link_or_special_file_rejected")
        file_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        parent_metadata = os.fstat(parent_fd)
        parent_identity = (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
            parent_metadata.st_mode,
            parent_metadata.st_nlink,
            parent_metadata.st_size,
            parent_metadata.st_mtime_ns,
            parent_metadata.st_ctime_ns,
        )
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            yield handle
        observed = os.fstat(descriptor)
        observed_parent = os.fstat(parent_fd)
        _require(
            (
                observed.st_dev,
                observed.st_ino,
                observed.st_mode,
                observed.st_nlink,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )
            == file_identity
            and (
                observed_parent.st_dev,
                observed_parent.st_ino,
                observed_parent.st_mode,
                observed_parent.st_nlink,
                observed_parent.st_size,
                observed_parent.st_mtime_ns,
                observed_parent.st_ctime_ns,
            )
            == parent_identity,
            check,
            "external_file_changed_during_verification",
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SealViolation("json_structure", "duplicate_json_key")
        result[key] = value
    return result


def _reject_nonfinite_json(_value: str) -> None:
    raise SealViolation("json_structure", "nonfinite_json_number")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SealViolation("json_structure", "nonfinite_json_number")
    return parsed


def _decode_json(payload_bytes: bytes, relative: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json,
            parse_float=_parse_finite_json_float,
        )
    except SealViolation:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealViolation("json_structure", "invalid_utf8_json", filename=relative) from error
    if not isinstance(payload, dict):
        raise SealViolation("json_structure", "top_level_object_required", filename=relative)
    return payload


def _load_json(sealed: SealedDirectory, relative: str) -> dict[str, Any]:
    return _decode_json(sealed.read_bytes(relative), relative)


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require(condition: bool, check: str, discrepancy: str, filename: str | None = None) -> None:
    if not condition:
        raise SealViolation(check, discrepancy, filename=filename)


def _require_exact_keys(value: Any, expected: set[str], *, check: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), check, "object_required")
    assert isinstance(value, dict)
    _require(set(value) == expected, check, "field_closure_mismatch")
    return value


def _exact_scalar(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return value is expected
    if type(expected) is int:
        return type(value) is int and value == expected
    if expected is None:
        return value is None
    return value == expected


def _validate_optional_csv_numbers(
    row: Mapping[str, str | None], fields: Sequence[str], *, check: str
) -> None:
    for field_name in fields:
        value = row.get(field_name)
        _require(value is not None, check, "row_width_mismatch")
        if value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise SealViolation(check, "numeric_field_invalid") from error
        _require(math.isfinite(parsed), check, "numeric_field_invalid")


def _validate_csv_booleans(
    row: Mapping[str, str | None], fields: Sequence[str], *, check: str
) -> None:
    for field_name in fields:
        _require(row.get(field_name) in {"True", "False"}, check, "boolean_field_invalid")


def _validate_csv_row_width(row: Mapping[str | None, str | None], *, check: str) -> None:
    _require(
        None not in row and all(value is not None for value in row.values()),
        check,
        "row_width_mismatch",
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and HEX_64.fullmatch(value) is not None


def _is_positive_integer_text(value: str) -> bool:
    return value.isascii() and value.isdigit() and int(value) > 0


def _validate_design(design: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "frequency_estimand",
        "sites",
        "anchor_scenes",
        "target_minerals",
        "candidate_populations",
        "baseline_endmembers",
        "endmember_schedules",
        "schedule_algorithm",
        "seed_derivations",
        "ridges",
        "detection_quantiles",
        "infeasibility_gates",
        "stochastic_replicates",
        "bootstrap_replicates",
        "seed",
        "analytical_cells",
        "recorded_variants_per_scene",
        "recorded_variants_total",
        "unique_mtmf_fits_per_scene",
        "unique_mtmf_fits_total",
        "axis_contrasts",
        "covariance_terminology",
        "protocol",
        "protocol_deviations",
        "protocol_amendment",
        "code_commit",
        "governing_files",
        "lockfile_sha256",
        "input_manifest",
        "rockwell_reference",
        "quality_policy",
        "block_manifest",
        "software",
        "compute_controls",
        "scientific_design_sha256",
    }
    _require(set(design) == expected_top, "design", "top_level_field_closure_mismatch")
    expected_values = {
        "schema_version": "1.0",
        "frequency_estimand": "finite_design_empirical_frequency",
        "sites": list(FROZEN_SITES),
        "anchor_scenes": ANCHOR_SCENES,
        "target_minerals": list(TARGET_MINERALS),
        "baseline_endmembers": BASELINE_ENDMEMBERS,
        "schedule_algorithm": "balanced_resize_then_pcg64_shuffle",
        "ridges": [0.001, 0.01, 0.1],
        "detection_quantiles": [0.85, 0.9, 0.95],
        "infeasibility_gates": ["none", "1"],
        "stochastic_replicates": STOCHASTIC_REPLICATES,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": 42,
        "analytical_cells": ANALYTICAL_CELLS,
        "recorded_variants_per_scene": VARIANTS_PER_SITE,
        "recorded_variants_total": TOTAL_VARIANTS,
        "unique_mtmf_fits_per_scene": UNIQUE_FITS_PER_SITE,
        "unique_mtmf_fits_total": TOTAL_UNIQUE_FITS,
        "axis_contrasts": "descriptive_paired_only",
        "protocol_deviations": {},
        "protocol_amendment": None,
    }
    for key, expected in expected_values.items():
        _require(design.get(key) == expected, "design", f"frozen_{key}_mismatch")
    _validate_candidate_design(design)
    _validate_design_provenance(design)
    identity = {
        key: value
        for key, value in design.items()
        if key
        not in {
            "compute_controls",
            "scientific_design_sha256",
        }
    }
    observed_identity = hashlib.sha256(_compact_json(identity).encode("utf-8")).hexdigest()
    _require(
        design.get("scientific_design_sha256") == observed_identity,
        "design",
        "scientific_design_hash_mismatch",
    )


def _validate_candidate_design(design: Mapping[str, Any]) -> None:
    populations = _require_exact_keys(
        design.get("candidate_populations"), set(TARGET_MINERALS), check="design_candidates"
    )
    for mineral in TARGET_MINERALS:
        candidates = populations[mineral]
        _require(isinstance(candidates, list), "design_candidates", "candidate_list_required")
        _require(
            len(candidates) == CANDIDATE_COUNTS[mineral]
            and len(set(candidates)) == len(candidates)
            and candidates == sorted(candidates),
            "design_candidates",
            "candidate_population_mismatch",
        )
        _require(
            all(
                isinstance(candidate, str)
                and candidate not in {"", ".", ".."}
                and "/" not in candidate
                and "\\" not in candidate
                for candidate in candidates
            ),
            "design_candidates",
            "unsafe_candidate_filename",
        )
        _require(
            BASELINE_ENDMEMBERS[mineral] in candidates,
            "design_candidates",
            "baseline_endmember_missing",
        )
    schedules = design.get("endmember_schedules")
    _require(
        isinstance(schedules, list) and len(schedules) == STOCHASTIC_REPLICATES,
        "design_candidates",
        "schedule_count_mismatch",
    )
    for schedule in schedules:
        schedule_map = _require_exact_keys(
            schedule, set(TARGET_MINERALS), check="design_candidates"
        )
        for mineral in TARGET_MINERALS:
            _require(
                schedule_map[mineral] in populations[mineral],
                "design_candidates",
                "schedule_candidate_mismatch",
            )
    for mineral in TARGET_MINERALS:
        counts = Counter(schedule[mineral] for schedule in schedules)
        quotient, remainder = divmod(STOCHASTIC_REPLICATES, CANDIDATE_COUNTS[mineral])
        expected_frequencies = {quotient, quotient + int(remainder > 0)}
        _require(
            set(counts) == set(populations[mineral])
            and set(counts.values()).issubset(expected_frequencies),
            "design_candidates",
            "schedule_balance_mismatch",
        )
    expected_seeds = {
        "endmember": [[42, index] for index in range(len(TARGET_MINERALS))],
        "covariance": {
            site: [[42, 1000 + index, replicate] for replicate in range(16)]
            for index, site in enumerate(FROZEN_SITES)
        },
        "calibration": {
            site: [[42, 2000 + index, replicate] for replicate in range(16)]
            for index, site in enumerate(FROZEN_SITES)
        },
    }
    _require(design.get("seed_derivations") == expected_seeds, "design", "seed_mismatch")


def _validate_design_provenance(design: Mapping[str, Any]) -> None:
    protocol = _require_exact_keys(
        design.get("protocol"),
        {"path", "sha256", "expected_sha256", "protocol_compliant", "amendment"},
        check="preregistration",
    )
    _require(
        protocol.get("sha256") == FROZEN_PREREGISTRATION_SHA256
        and protocol.get("expected_sha256") == FROZEN_PREREGISTRATION_SHA256
        and protocol.get("protocol_compliant") is True
        and protocol.get("amendment") is None,
        "preregistration",
        "frozen_hash_or_identity_mismatch",
    )
    _require(
        isinstance(protocol.get("path"), str)
        and protocol["path"]
        .replace("\\", "/")
        .endswith("/docs/m2_ensemble_sensitivity_preregistration.md")
        or protocol.get("path") == "docs/m2_ensemble_sensitivity_preregistration.md",
        "preregistration",
        "protocol_path_mismatch",
    )
    _require(
        isinstance(design.get("code_commit"), str)
        and HEX_40.fullmatch(design["code_commit"]) is not None,
        "source_provenance",
        "code_commit_invalid",
    )
    records = design.get("governing_files")
    _require(isinstance(records, list), "source_provenance", "governing_list_required")
    by_path: dict[str, Mapping[str, Any]] = {}
    for record in records:
        item = _require_exact_keys(
            record,
            {"path", "sha256", "git_status", "tracked", "dirty"},
            check="source_provenance",
        )
        path = item.get("path")
        _require(isinstance(path, str), "source_provenance", "governing_path_invalid")
        assert isinstance(path, str)
        _safe_relative(path, check="source_provenance")
        _require(path not in by_path, "source_provenance", "duplicate_governing_path")
        _require(_is_sha256(item.get("sha256")), "source_provenance", "source_hash_invalid")
        status = item.get("git_status")
        _require(
            isinstance(status, str)
            and len(status) == 2
            and isinstance(item.get("tracked"), bool)
            and isinstance(item.get("dirty"), bool),
            "source_provenance",
            "git_provenance_invalid",
        )
        by_path[path] = item
    _require(set(by_path) == set(GOVERNING_FILES), "source_provenance", "source_file_closure")
    _require(
        by_path["docs/m2_ensemble_sensitivity_preregistration.md"]["sha256"]
        == FROZEN_PREREGISTRATION_SHA256,
        "preregistration",
        "governing_preregistration_hash_mismatch",
    )
    quality = _require_exact_keys(
        design.get("quality_policy"),
        {"path", "sha256", "retained_bands"},
        check="source_provenance",
    )
    _require(
        quality.get("path") == "docs/tanager_quality_mask_policy.md"
        and quality.get("retained_bands") == RETAINED_BANDS
        and quality.get("sha256") == by_path["docs/tanager_quality_mask_policy.md"]["sha256"],
        "source_provenance",
        "quality_policy_binding_mismatch",
    )
    _require(_is_sha256(design.get("lockfile_sha256")), "source_provenance", "lock_hash_invalid")
    controls = _require_exact_keys(
        design.get("compute_controls"),
        {"device", "batch_size", "storage_layout", "numpy_reference", "accelerator_backend"},
        check="compute_controls",
    )
    _require(
        controls.get("device") == "cpu"
        and type(controls.get("batch_size")) is int
        and controls.get("batch_size") == 1
        and controls.get("storage_layout") == "disk"
        and controls.get("numpy_reference") is True
        and controls.get("accelerator_backend") is None,
        "compute_controls",
        "bigmem_compute_contract_mismatch",
    )
    for key in ("input_manifest", "block_manifest"):
        value = design.get(key)
        _require(
            isinstance(value, dict) and _is_sha256(value.get("sha256")),
            "source_provenance",
            f"{key}_hash_invalid",
        )


def _expected_member_specs(design: Mapping[str, Any]) -> list[dict[str, str]]:
    schedules = design["endmember_schedules"]
    baseline = design["baseline_endmembers"]
    specs: list[dict[str, str]] = []
    for site_index, site in enumerate(FROZEN_SITES):
        baseline_fit = f"{site}:fit:baseline:r0.01"
        specs.append(
            _member_spec(
                site,
                "baseline",
                "baseline",
                baseline_fit,
                "",
                "0.01",
                "0.9",
                "1",
                baseline,
                "full_scene",
                "full_scene",
                site_index,
            )
        )
        for replicate in range(STOCHASTIC_REPLICATES):
            specs.append(
                _member_spec(
                    site,
                    f"endmember_only:r{replicate:02d}",
                    "endmember_only",
                    f"{site}:fit:endmember:r{replicate:02d}:ridge0.01",
                    str(replicate),
                    "0.01",
                    "0.9",
                    "1",
                    schedules[replicate],
                    "full_scene",
                    "full_scene",
                    site_index,
                )
            )
        for replicate in range(STOCHASTIC_REPLICATES):
            specs.append(
                _member_spec(
                    site,
                    f"covariance_only:r{replicate:02d}",
                    "covariance_only",
                    f"{site}:fit:covariance:r{replicate:02d}:ridge0.01",
                    str(replicate),
                    "0.01",
                    "0.9",
                    "1",
                    baseline,
                    "bootstrap_blocks",
                    "full_scene",
                    site_index,
                )
            )
        for replicate in range(STOCHASTIC_REPLICATES):
            specs.append(
                _member_spec(
                    site,
                    f"calibration_only:r{replicate:02d}",
                    "calibration_only",
                    baseline_fit,
                    str(replicate),
                    "0.01",
                    "0.9",
                    "1",
                    baseline,
                    "full_scene",
                    "bootstrap_blocks",
                    site_index,
                )
            )
        for ridge in RIDGES:
            fit_id = baseline_fit if ridge == "0.01" else f"{site}:fit:analytical:ridge{ridge}"
            for quantile in QUANTILES:
                for gate in GATES:
                    suffix = f"analytical:ridge{ridge}:q{quantile}:gate{gate}"
                    specs.append(
                        _member_spec(
                            site,
                            suffix,
                            "analytical_grid",
                            fit_id,
                            "",
                            ridge,
                            quantile,
                            gate,
                            baseline,
                            "full_scene",
                            "full_scene",
                            site_index,
                        )
                    )
        for replicate in range(STOCHASTIC_REPLICATES):
            for ridge in RIDGES:
                fit_id = f"{site}:fit:joint:r{replicate:02d}:ridge{ridge}"
                for quantile in QUANTILES:
                    for gate in GATES:
                        suffix = f"joint:r{replicate:02d}:ridge{ridge}:q{quantile}:gate{gate}"
                        specs.append(
                            _member_spec(
                                site,
                                suffix,
                                "joint",
                                fit_id,
                                str(replicate),
                                ridge,
                                quantile,
                                gate,
                                schedules[replicate],
                                "bootstrap_blocks",
                                "bootstrap_blocks",
                                site_index,
                            )
                        )
    return specs


def _member_spec(
    site: str,
    suffix: str,
    member_class: str,
    fit_id: str,
    replicate: str,
    ridge: str,
    quantile: str,
    gate: str,
    endmembers: Mapping[str, str],
    covariance_mode: str,
    calibration_mode: str,
    site_index: int,
) -> dict[str, str]:
    member_id = f"{site}:{suffix}" if suffix != "baseline" else f"{site}:baseline"
    record = {
        "scene": ANCHOR_SCENES[site],
        "site": site,
        "member_id": member_id,
        "member_class": member_class,
        "fit_id": fit_id,
        "stochastic_replicate": replicate,
        "covariance_mode": covariance_mode,
        "calibration_mode": calibration_mode,
        "ridge": ridge,
        "detection_quantile": quantile,
        "infeasibility_gate": gate,
    }
    record.update({f"endmember_{mineral}": endmembers[mineral] for mineral in TARGET_MINERALS})
    if replicate and covariance_mode == "bootstrap_blocks":
        record["covariance_seed_entropy"] = f"[42,{1000 + site_index},{replicate}]"
    else:
        record["covariance_seed_entropy"] = ""
    if replicate and calibration_mode == "bootstrap_blocks":
        record["calibration_seed_entropy"] = f"[42,{2000 + site_index},{replicate}]"
    else:
        record["calibration_seed_entropy"] = ""
    return record


def _csv_reader(
    sealed: SealedDirectory, relative: str, fields: Sequence[str]
) -> tuple[csv.DictReader, io.TextIOWrapper, BinaryIO]:
    context = sealed.open_file(relative)
    descriptor = context.__enter__()
    raw = os.fdopen(os.dup(descriptor), "rb")
    text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
    reader = csv.DictReader(text)
    if tuple(reader.fieldnames or ()) != tuple(fields):
        text.close()
        context.__exit__(None, None, None)
        raise SealViolation("csv_structure", "header_closure_mismatch", filename=relative)
    setattr(reader, "_sealed_context", context)
    return reader, text, raw


def _close_csv(reader: csv.DictReader, text: io.TextIOWrapper) -> None:
    context = getattr(reader, "_sealed_context")
    text.close()
    context.__exit__(None, None, None)


def _load_members(
    sealed: SealedDirectory, design: Mapping[str, Any], design_sha: str
) -> tuple[list[dict[str, str]], dict[str, str], set[str]]:
    reader, text, _raw = _csv_reader(sealed, "members.csv", MEMBER_FIELDS)
    try:
        rows = list(reader)
    except (csv.Error, UnicodeDecodeError) as error:
        raise SealViolation("members", "invalid_csv", filename="members.csv") from error
    finally:
        _close_csv(reader, text)
    _require(len(rows) == TOTAL_VARIANTS, "members", "row_count_mismatch", "members.csv")
    specs = _expected_member_specs(design)
    _require(
        [row.get("member_id") for row in rows] == [spec["member_id"] for spec in specs],
        "members",
        "member_order_or_identity_mismatch",
        "members.csv",
    )
    fit_checksums: dict[str, str] = {}
    fit_provenance: dict[str, tuple[str, str]] = {}
    covariance_draws: dict[tuple[str, str], str] = {}
    calibration_draws: dict[tuple[str, str], str] = {}
    for row, spec in zip(rows, specs, strict=True):
        _validate_csv_row_width(row, check="members")
        for field_name, expected in spec.items():
            _require(
                row.get(field_name) == expected,
                "members",
                "member_definition_mismatch",
                "members.csv",
            )
        _validate_member_draws(row, covariance_draws, calibration_draws)
        _require(
            row.get("status") == "complete", "members", "nonconforming_status_rows", "members.csv"
        )
        _require(
            row.get("failure_reason") == "", "members", "failure_reason_present", "members.csv"
        )
        _require(
            _is_positive_integer_text(row.get("contributing_pixels", "")),
            "members",
            "contributing_pixel_identity_missing",
            "members.csv",
        )
        _require(
            row.get("retained_bands") == str(RETAINED_BANDS),
            "members",
            "retained_band_mismatch",
            "members.csv",
        )
        checksum = row.get("output_checksum")
        _require(_is_sha256(checksum), "members", "fit_checksum_invalid", "members.csv")
        _require(
            row.get("design_sha256") == design_sha,
            "members",
            "design_hash_binding_mismatch",
            "members.csv",
        )
        _require(
            row.get("wall_time_seconds") == "" and row.get("peak_memory_bytes") == "",
            "members",
            "unexpected_member_runtime_fields",
            "members.csv",
        )
        fit_id = row["fit_id"]
        if fit_id in fit_checksums:
            _require(
                fit_checksums[fit_id] == checksum,
                "members",
                "fit_checksum_identity_conflict",
                "members.csv",
            )
            _require(
                fit_provenance[fit_id] == (row["contributing_pixels"], row["retained_bands"]),
                "members",
                "fit_provenance_identity_conflict",
                "members.csv",
            )
        else:
            assert isinstance(checksum, str)
            fit_checksums[fit_id] = checksum
            fit_provenance[fit_id] = (row["contributing_pixels"], row["retained_bands"])
    _require(
        len(fit_checksums) == TOTAL_UNIQUE_FITS,
        "members",
        "unique_fit_count_mismatch",
        "members.csv",
    )
    cache_paths = {
        f".score_cache/{row['site']}/{hashlib.sha256(row['fit_id'].encode('utf-8')).hexdigest()}.npz"
        for row in rows
    }
    _require(len(cache_paths) == TOTAL_UNIQUE_FITS, "cache_identity", "cache_name_collision")
    return rows, fit_checksums, cache_paths


def _npz_array(archive: Any, key: str) -> np.ndarray:
    try:
        value = archive[key]
    except Exception as error:
        raise SealViolation("fit_cache_semantics", "array_load_rejected") from error
    _require(isinstance(value, np.ndarray), "fit_cache_semantics", "array_required")
    _require(not value.dtype.hasobject, "fit_cache_semantics", "object_dtype_rejected")
    return value


def _integer_scalar(archive: Any, key: str) -> int:
    value = _npz_array(archive, key)
    _require(
        value.shape == () and np.issubdtype(value.dtype, np.integer),
        "fit_cache_semantics",
        "metadata_scalar_invalid",
    )
    return int(value.item())


def _validate_score_array(
    archive: Any,
    key: str,
    valid_support: np.ndarray,
    *,
    infeasibility: bool,
) -> np.ndarray:
    """Load one producer score map and enforce its frozen semantic contract."""
    value = _npz_array(archive, key)
    _require(
        value.ndim == 2 and value.shape == valid_support.shape,
        "fit_cache_semantics",
        "score_shape_mismatch",
    )
    _require(
        value.dtype == np.dtype(np.float64),
        "fit_cache_semantics",
        "score_dtype_mismatch",
    )
    _require(
        bool(np.any(valid_support))
        and bool(np.all(np.isfinite(value[valid_support])))
        and bool(np.all(np.isnan(value[~valid_support]))),
        "fit_cache_semantics",
        "score_support_mismatch",
    )
    if infeasibility:
        _require(
            bool(np.all(value[valid_support] >= 0.0)),
            "fit_cache_semantics",
            "infeasibility_value_invalid",
        )
    return value


def _validate_fit_cache(
    sealed: SealedDirectory,
    relative: str,
    rows: Sequence[Mapping[str, str]],
    expected_checksum: str,
) -> None:
    try:
        with sealed.open_file(relative) as descriptor:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                loaded = np.load(handle, allow_pickle=False)
                _require(
                    isinstance(loaded, np.lib.npyio.NpzFile),
                    "fit_cache_semantics",
                    "npz_archive_required",
                )
                with loaded as archive:
                    _require(
                        len(archive.files) == len(FIT_CACHE_KEYS)
                        and set(archive.files) == FIT_CACHE_KEYS,
                        "fit_cache_semantics",
                        "member_key_closure_mismatch",
                    )
                    contributing_pixels = _integer_scalar(archive, "contributing_pixels")
                    retained_bands = _integer_scalar(archive, "retained_bands")
                    _require(
                        all(
                            row["contributing_pixels"] == str(contributing_pixels)
                            and row["retained_bands"] == str(retained_bands)
                            for row in rows
                        ),
                        "fit_cache_semantics",
                        "ledger_metadata_mismatch",
                    )
                    valid_support = _npz_array(archive, "valid_support")
                    _require(
                        valid_support.ndim == 2 and valid_support.dtype == np.dtype(np.bool_),
                        "fit_cache_semantics",
                        "valid_support_invalid",
                    )
                    checksum = hashlib.sha256()
                    for mineral in TARGET_MINERALS:
                        matched_filter = _validate_score_array(
                            archive,
                            f"mf_{mineral}",
                            valid_support,
                            infeasibility=False,
                        )
                        infeasibility = _validate_score_array(
                            archive,
                            f"infeas_{mineral}",
                            valid_support,
                            infeasibility=True,
                        )
                        checksum.update(memoryview(np.ascontiguousarray(matched_filter)).cast("B"))
                        checksum.update(memoryview(np.ascontiguousarray(infeasibility)).cast("B"))
                    _require(
                        checksum.hexdigest() == expected_checksum,
                        "fit_cache_semantics",
                        "logical_checksum_mismatch",
                    )
    except SealViolation:
        raise
    except Exception as error:
        raise SealViolation("fit_cache_semantics", "invalid_npz") from error


def _validate_fit_caches(
    sealed: SealedDirectory,
    members: Sequence[Mapping[str, str]],
    fit_checksums: Mapping[str, str],
    cache_paths: set[str],
) -> int:
    rows_by_fit: dict[str, list[Mapping[str, str]]] = {}
    for row in members:
        rows_by_fit.setdefault(row["fit_id"], []).append(row)
    _require(
        len(rows_by_fit) == TOTAL_UNIQUE_FITS,
        "fit_cache_semantics",
        "fit_identity_count_mismatch",
    )
    validated_paths: set[str] = set()
    for fit_id, rows in rows_by_fit.items():
        sites = {row["site"] for row in rows}
        _require(len(sites) == 1, "fit_cache_semantics", "fit_site_identity_mismatch")
        relative = (
            f".score_cache/{rows[0]['site']}/"
            f"{hashlib.sha256(fit_id.encode('utf-8')).hexdigest()}.npz"
        )
        validated_paths.add(relative)
        _validate_fit_cache(sealed, relative, rows, fit_checksums[fit_id])
    _require(
        validated_paths == cache_paths,
        "fit_cache_semantics",
        "fit_cache_path_closure_mismatch",
    )
    return len(validated_paths)


def _validate_member_draws(
    row: Mapping[str, str],
    covariance_draws: dict[tuple[str, str], str],
    calibration_draws: dict[tuple[str, str], str],
) -> None:
    site = row["site"]
    replicate = row["stochastic_replicate"]
    for mode_field, draw_field, store in (
        ("covariance_mode", "covariance_draw", covariance_draws),
        ("calibration_mode", "calibration_draw", calibration_draws),
    ):
        draw = row.get(draw_field, "")
        if row[mode_field] == "full_scene":
            _require(draw == "[]", "members", "full_scene_draw_not_empty", "members.csv")
            continue
        _require(bool(replicate), "members", "bootstrap_replicate_missing", "members.csv")
        try:
            records = json.loads(draw, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, SealViolation) as error:
            raise SealViolation("members", "invalid_block_draw", filename="members.csv") from error
        _require(
            isinstance(records, list) and len(records) >= 1,
            "members",
            "invalid_block_draw",
            "members.csv",
        )
        block_ids: list[int] = []
        total = 0
        for record in records:
            item = _require_exact_keys(record, {"block_id", "multiplicity"}, check="members")
            block_id = item.get("block_id")
            multiplicity = item.get("multiplicity")
            _require(
                type(block_id) is int
                and block_id > 0
                and type(multiplicity) is int
                and multiplicity > 0,
                "members",
                "invalid_block_draw",
                "members.csv",
            )
            block_ids.append(block_id)
            total += multiplicity
        _require(
            block_ids == sorted(set(block_ids)) and total >= 2,
            "members",
            "invalid_block_draw",
            "members.csv",
        )
        key = (site, replicate)
        if key in store:
            _require(store[key] == draw, "members", "paired_draw_mismatch", "members.csv")
        else:
            store[key] = draw


def _map_paths() -> set[str]:
    paths = {
        f"maps/{site}_{mineral}_{suffix}.tif"
        for site in FROZEN_SITES
        for mineral in TARGET_MINERALS
        for suffix in ("n_valid", "detection_frequency", "confidence_class")
    }
    paths.update(
        f"maps/{site}_{suffix}.tif"
        for site in FROZEN_SITES
        for suffix in ("modal_class", "modal_frequency", "class_entropy", "switch_frequency")
    )
    return paths


def _validate_source_manifest(
    path: Path,
    expected_sha256: str,
    design: Mapping[str, Any],
) -> tuple[str, int]:
    _require(_is_sha256(expected_sha256), "source_manifest", "expected_hash_invalid")
    with _open_external_file(path, check="source_manifest") as handle:
        manifest_bytes = handle.read()
    observed_sha = hashlib.sha256(manifest_bytes).hexdigest()
    _require(observed_sha == expected_sha256, "source_manifest", "detached_hash_mismatch")
    entries = _parse_source_manifest(manifest_bytes)
    _require(
        len(entries) == EXPECTED_SOURCE_MANIFEST_ENTRIES,
        "source_manifest",
        "entry_count_mismatch",
    )
    _bind_source_records(entries, design)
    return observed_sha, len(entries)


def _is_normalized_source_remainder(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and "\\" not in value
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in value.split("/"))
        and path.as_posix() == value
    )


def _normalize_source_manifest_path(value: str) -> str:
    if value.startswith(SOURCE_SIBLING_PREFIX):
        remainder = value.removeprefix(SOURCE_SIBLING_PREFIX)
        _require(
            _is_normalized_source_remainder(remainder),
            "source_manifest",
            "unsafe_relative_path",
        )
        return value
    _require(
        _is_normalized_source_remainder(value) and not value.startswith(LEGACY_SOURCE_PREFIXES),
        "source_manifest",
        "unsafe_relative_path",
    )
    return value


def _parse_source_manifest(manifest_bytes: bytes) -> dict[str, str]:
    try:
        text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SealViolation("source_manifest", "invalid_utf8") from error
    _require(
        "\r" not in text and text.endswith("\n"),
        "source_manifest",
        "noncanonical_newline",
    )
    lines = text[:-1].split("\n")
    entries: dict[str, str] = {}
    paths: list[str] = []
    for line in lines:
        match = SOURCE_LINE.fullmatch(line)
        _require(match is not None, "source_manifest", "invalid_sha256sum_record")
        assert match is not None
        digest = match.group("sha256")
        normalized = _normalize_source_manifest_path(match.group("path"))
        _require(normalized not in entries, "source_manifest", "duplicate_manifest_path")
        entries[normalized] = digest
        paths.append(normalized)
    _require(
        paths == sorted(paths),
        "source_manifest",
        "noncanonical_order",
    )
    return entries


def _validate_block_manifest(
    path: Path, design: Mapping[str, Any]
) -> tuple[dict[str, tuple[tuple[int, int], str, tuple[float, ...]]], str]:
    with _open_external_file(path, check="block_manifest") as handle:
        manifest_bytes = handle.read()
    observed_sha = hashlib.sha256(manifest_bytes).hexdigest()
    design_record = design.get("block_manifest")
    _require(
        isinstance(design_record, dict) and observed_sha == design_record.get("sha256"),
        "block_manifest",
        "design_hash_binding_mismatch",
    )
    rockwell = design.get("rockwell_reference")
    rockwell_manifest = rockwell.get("m2_block_manifest") if isinstance(rockwell, dict) else None
    rockwell_rasters = rockwell.get("m2_block_rasters") if isinstance(rockwell, dict) else None
    _require(
        isinstance(rockwell_manifest, dict)
        and rockwell_manifest.get("sha256") == observed_sha
        and isinstance(rockwell_rasters, dict),
        "block_manifest",
        "rockwell_manifest_binding_mismatch",
    )
    payload = _decode_json(manifest_bytes, path.name)
    _require(
        payload.get("manifest_type") == "spatial_validation_complete_blocks",
        "block_manifest",
        "manifest_type_mismatch",
    )
    protocol = payload.get("protocol")
    _require(
        isinstance(protocol, dict)
        and protocol.get("path") == "docs/m2_spatial_validation_preregistration.md"
        and _is_sha256(protocol.get("sha256")),
        "block_manifest",
        "protocol_binding_invalid",
    )
    sites = payload.get("sites")
    _require(
        isinstance(sites, dict) and set(sites) == set(FROZEN_SITES),
        "block_manifest",
        "site_closure_mismatch",
    )
    grids: dict[str, tuple[tuple[int, int], str, tuple[float, ...]]] = {}
    for site in FROZEN_SITES:
        entry = sites[site]
        _require(
            isinstance(entry, dict) and entry.get("scene_id") == ANCHOR_SCENES[site],
            "block_manifest",
            "anchor_scene_mismatch",
        )
        grid = entry.get("grid")
        _require(isinstance(grid, dict), "block_manifest", "grid_record_missing")
        shape = grid.get("shape")
        transform = grid.get("transform")
        crs = grid.get("crs")
        _require(
            isinstance(shape, list)
            and len(shape) == 2
            and all(type(value) is int and value > 0 for value in shape)
            and isinstance(transform, list)
            and len(transform) == 6
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in transform
            )
            and isinstance(crs, str)
            and bool(crs),
            "block_manifest",
            "grid_record_invalid",
        )
        grids[site] = (
            (shape[0], shape[1]),
            crs,
            tuple(float(value) for value in transform),
        )
        scales = entry.get("scales")
        _require(
            isinstance(scales, dict) and set(scales) == {"L", "2L"},
            "block_manifest",
            "scale_closure_mismatch",
        )
        for scale in ("L", "2L"):
            scale_record = scales[scale]
            _require(isinstance(scale_record, dict), "block_manifest", "scale_record_invalid")
            block_ids = scale_record.get("complete_block_ids")
            _require(
                scale_record.get("anchor_scene_id") == ANCHOR_SCENES[site]
                and isinstance(block_ids, list)
                and len(block_ids) == len(set(block_ids))
                and all(type(value) is int and value > 0 for value in block_ids)
                and scale_record.get("complete_blocks") == len(block_ids)
                and isinstance(scale_record.get("block_raster"), str)
                and _is_sha256(scale_record.get("block_raster_sha256")),
                "block_manifest",
                "scale_record_invalid",
            )
            _safe_relative(scale_record["block_raster"], check="block_manifest")
            if site == "goldfield":
                _require(
                    isinstance(rockwell_rasters.get(scale), dict)
                    and rockwell_rasters[scale].get("sha256")
                    == scale_record.get("block_raster_sha256"),
                    "block_manifest",
                    "rockwell_raster_binding_mismatch",
                )
            if scale == "L":
                _require(
                    len(block_ids) >= 2,
                    "block_manifest",
                    "stochastic_block_support_invalid",
                )
    return grids, observed_sha


def _manifest_lookup(entries: Mapping[str, str], relative: str) -> str | None:
    return entries.get(relative)


def _bind_source_records(entries: Mapping[str, str], design: Mapping[str, Any]) -> None:
    governing = {record["path"]: record["sha256"] for record in design["governing_files"]}
    required = {**governing, "uv.lock": design["lockfile_sha256"]}
    input_manifest = design["input_manifest"]
    required["docs/input_manifest.json"] = input_manifest["sha256"]
    block_manifest = design["block_manifest"]
    required["data/processed/spatial_validation/block_manifest.json"] = block_manifest["sha256"]
    rockwell = design.get("rockwell_reference")
    _require(isinstance(rockwell, dict), "source_manifest", "rockwell_binding_missing")
    rockwell_path = rockwell.get("path")
    _require(
        isinstance(rockwell_path, str) and _is_sha256(rockwell.get("sha256")),
        "source_manifest",
        "rockwell_binding_invalid",
    )
    assert isinstance(rockwell_path, str)
    required[_safe_relative(rockwell_path, check="source_manifest")] = rockwell["sha256"]
    for relative, expected in required.items():
        _require(
            _manifest_lookup(entries, relative) == expected,
            "source_manifest",
            "source_record_binding_mismatch",
        )
    raster_records = rockwell.get("m2_block_rasters")
    _require(
        isinstance(raster_records, dict) and set(raster_records) == {"L", "2L"},
        "source_manifest",
        "block_raster_binding_missing",
    )
    manifest_hashes = set(entries.values())
    for record in raster_records.values():
        _require(
            isinstance(record, dict) and _is_sha256(record.get("sha256")),
            "source_manifest",
            "block_raster_hash_invalid",
        )
        _require(
            record["sha256"] in manifest_hashes, "source_manifest", "block_raster_hash_unbound"
        )


def _verify_declared_hashes(
    sealed: SealedDirectory,
    summary: Mapping[str, Any],
    artifact_paths: set[str],
) -> dict[str, str]:
    declared = summary.get("artifact_sha256")
    _require(isinstance(declared, dict), "artifact_sha256", "hash_mapping_required")
    _require(
        set(declared) == artifact_paths, "artifact_sha256", "declared_artifact_closure_mismatch"
    )
    observed: dict[str, str] = {}
    for relative in sorted(artifact_paths):
        expected = declared.get(relative)
        _require(_is_sha256(expected), "artifact_sha256", "declared_hash_invalid", relative)
        digest = sealed.sha256(relative)
        _require(digest == expected, "artifact_sha256", "declared_hash_mismatch", relative)
        observed[relative] = digest
    return observed


def _scan_metrics(sealed: SealedDirectory, members: Mapping[str, Mapping[str, str]]) -> int:
    reader, text, _raw = _csv_reader(sealed, "member_metrics.csv", METRIC_FIELDS)
    count = 0
    scene_pairs: set[tuple[str, str]] = set()
    identities: set[tuple[str, str, str, str, str]] = set()
    try:
        for row in reader:
            count += 1
            _validate_csv_row_width(row, check="member_metrics")
            member = members.get(row["member_id"])
            _require(member is not None, "member_metrics", "unknown_member_identity")
            assert member is not None
            for field_name in (
                "site",
                "scene",
                "member_class",
                "stochastic_replicate",
                "ridge",
                "detection_quantile",
                "infeasibility_gate",
            ):
                _require(
                    row[field_name] == member[field_name],
                    "member_metrics",
                    "member_definition_mismatch",
                )
            _require(row["mineral"] in TARGET_MINERALS, "member_metrics", "unknown_mineral")
            _require(
                row["aggregation"] in {"scene", "block"} and row["block_scale"] in {"L", "2L"},
                "member_metrics",
                "aggregation_identity_invalid",
            )
            block_id = row["block_id"]
            _require(block_id.isdigit(), "member_metrics", "block_identity_invalid")
            identity = (
                row["member_id"],
                row["mineral"],
                row["aggregation"],
                row["block_scale"],
                block_id,
            )
            _require(identity not in identities, "member_metrics", "duplicate_metric_identity")
            identities.add(identity)
            if row["aggregation"] == "scene":
                _require(
                    row["block_scale"] == "L" and block_id == "0",
                    "member_metrics",
                    "scene_identity_invalid",
                )
                scene_pairs.add((row["member_id"], row["mineral"]))
            else:
                _require(int(block_id) > 0, "member_metrics", "block_identity_invalid")
            _require(
                row["external_status"]
                in {"not_applicable", "complete", "unavailable", "block_recorded"}
                and row["covariance_scope"] in {"not_applicable", "full_scene_covariance"}
                and row["strict_covariance_exclusion_status"]
                in {"not_applicable", "complete", "unavailable"},
                "member_metrics",
                "status_identity_invalid",
            )
            _validate_optional_csv_numbers(
                row,
                (
                    "common_support_pixels",
                    "common_support_loss_fraction",
                    "detection_prevalence",
                    "rank_correlation",
                    "dominant_class_switch_frequency",
                    "auc",
                    "balanced_accuracy",
                    "positive_f1",
                    "negative_f1",
                    "macro_f1",
                    "tpr",
                    "fpr",
                    "prevalence",
                ),
                check="member_metrics",
            )
    except (csv.Error, UnicodeDecodeError) as error:
        raise SealViolation("member_metrics", "invalid_csv") from error
    finally:
        _close_csv(reader, text)
    expected_pairs = {(member_id, mineral) for member_id in members for mineral in TARGET_MINERALS}
    _require(scene_pairs == expected_pairs, "member_metrics", "scene_metric_closure_mismatch")
    return count


def _scan_factor_effects(sealed: SealedDirectory) -> int:
    reader, text, _raw = _csv_reader(sealed, "factor_effects.csv", FACTOR_FIELDS)
    count = 0
    identities: set[tuple[str, ...]] = set()
    endpoints = {
        "detection_prevalence",
        "common_support_loss_fraction",
        "rank_correlation",
        "dominant_class_switch_frequency",
        "auc",
        "balanced_accuracy",
        "positive_f1",
        "negative_f1",
        "macro_f1",
        "tpr",
        "fpr",
        "prevalence",
    }
    levels = {
        "axis": {"endmember_only", "covariance_only", "calibration_only"},
        "ridge": {"0.001", "0.1"},
        "quantile": {"0.85", "0.95"},
        "gate": {"none"},
    }
    reference_levels = {
        "axis": "baseline",
        "ridge": "0.01",
        "quantile": "0.9",
        "gate": "1",
    }
    try:
        for row in reader:
            count += 1
            _validate_csv_row_width(row, check="factor_effects")
            _require(
                row["site"] in FROZEN_SITES
                and row["mineral"] in TARGET_MINERALS
                and row["block_scale"] in {"L", "2L"},
                "factor_effects",
                "context_identity_invalid",
            )
            _require(
                row["factor"] in levels and row["level"] in levels[row["factor"]],
                "factor_effects",
                "factor_identity_invalid",
            )
            _require(
                row["reference_level"] == reference_levels[row["factor"]],
                "factor_effects",
                "reference_level_identity_invalid",
            )
            _require(row["endpoint"] in endpoints, "factor_effects", "endpoint_identity_invalid")
            _require(
                row["scheduled_replicates"] == str(BOOTSTRAP_REPLICATES),
                "factor_effects",
                "bootstrap_identity_mismatch",
            )
            _require(
                row["contrast_status"] == "descriptive_paired_complete_block",
                "factor_effects",
                "contrast_status_mismatch",
            )
            identity = tuple(
                row[field_name]
                for field_name in ("site", "mineral", "block_scale", "factor", "level", "endpoint")
            )
            _require(identity not in identities, "factor_effects", "duplicate_factor_identity")
            identities.add(identity)
            _validate_optional_csv_numbers(
                row,
                (
                    "paired_delta_median",
                    "interval_lower",
                    "interval_upper",
                    "valid_replicates",
                    "finite_fraction",
                    "n_pairs",
                    "complete_blocks",
                    "paired_support_pixels",
                ),
                check="factor_effects",
            )
            _validate_csv_booleans(row, ("interval_available",), check="factor_effects")
    except (csv.Error, UnicodeDecodeError) as error:
        raise SealViolation("factor_effects", "invalid_csv") from error
    finally:
        _close_csv(reader, text)
    expected_identities = {
        (site, mineral, scale, factor, level, endpoint)
        for site in FROZEN_SITES
        for mineral in TARGET_MINERALS
        for scale in ("L", "2L")
        for factor, factor_levels in levels.items()
        for level in factor_levels
        for endpoint in endpoints
    }
    _require(
        identities == expected_identities,
        "factor_effects",
        "factor_identity_closure_mismatch",
    )
    return count


def _scan_calibration(sealed: SealedDirectory) -> int:
    reader, text, _raw = _csv_reader(sealed, "calibration.csv", CALIBRATION_FIELDS)
    expected_bins = tuple(
        f"[{index / 10:.1f},{(index + 1) / 10:.1f}" + ("]" if index == 9 else ")")
        for index in range(10)
    )
    identities: set[tuple[str, str]] = set()
    minerals: set[str] = set()
    count = 0
    try:
        for row in reader:
            count += 1
            _validate_csv_row_width(row, check="calibration")
            _require(
                row["site"] == "goldfield" and row["mineral"] in TARGET_MINERALS,
                "calibration",
                "context_identity_invalid",
            )
            _require(
                row["confidence_bin"] in expected_bins, "calibration", "fixed_bin_identity_invalid"
            )
            _require(
                row["scheduled_replicates"] == str(BOOTSTRAP_REPLICATES),
                "calibration",
                "bootstrap_identity_mismatch",
            )
            _require(
                row["status"]
                in {"complete", "empty_fixed_bin", "unavailable_zero_complete_blocks"},
                "calibration",
                "status_identity_invalid",
            )
            identity = (row["mineral"], row["confidence_bin"])
            _require(identity not in identities, "calibration", "duplicate_bin_identity")
            identities.add(identity)
            minerals.add(row["mineral"])
            _validate_optional_csv_numbers(
                row,
                (
                    "support_blocks",
                    "support_pixels",
                    "compatible_positive_rate",
                    "interval_lower",
                    "interval_upper",
                    "valid_replicates",
                    "finite_fraction",
                    "brier_score",
                    "brier_interval_lower",
                    "brier_interval_upper",
                    "brier_valid_replicates",
                    "brier_finite_fraction",
                    "expected_calibration_error",
                    "ece_interval_lower",
                    "ece_interval_upper",
                    "ece_valid_replicates",
                    "ece_finite_fraction",
                ),
                check="calibration",
            )
            _validate_csv_booleans(
                row,
                (
                    "interval_available",
                    "brier_interval_available",
                    "ece_interval_available",
                ),
                check="calibration",
            )
    except (csv.Error, UnicodeDecodeError) as error:
        raise SealViolation("calibration", "invalid_csv") from error
    finally:
        _close_csv(reader, text)
    _require(
        minerals == set(ROCKWELL_MINERALS),
        "calibration",
        "calibration_identity_closure_mismatch",
    )
    for mineral in minerals:
        _require(
            {bin_name for item_mineral, bin_name in identities if item_mineral == mineral}
            == set(expected_bins),
            "calibration",
            "fixed_bin_closure_mismatch",
        )
    return count


def _optional_finite_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SealViolation("gate_logic", "numeric_gate_component_invalid")
    candidate = float(value)
    _require(math.isfinite(candidate), "gate_logic", "numeric_gate_component_invalid")
    return candidate


def _classification(stability: bool | None, external: bool | None, strict: bool | None) -> str:
    if None in {stability, external, strict}:
        return "unavailable_required_evidence"
    if stability and external and strict:
        return "validated_analytically_robust_alteration_zone_discrimination"
    if stability and not external:
        return "analytically_stable_spatial_pattern_only"
    if external and not strict:
        return "operational_discrimination_not_strictly_held_out"
    if external and not stability:
        return "discriminative_but_analytically_sensitive"
    return "negative_or_unstable_result"


def _external_gate(rows: Sequence[Mapping[str, Any]], scope: str) -> tuple[bool, bool | None]:
    def one(scale: str, metric: str) -> Mapping[str, Any] | None:
        matches = [
            row
            for row in rows
            if row.get("covariance_scope") == scope
            and row.get("mineral") == "alunite"
            and row.get("scale") == scale
            and row.get("metric") == metric
        ]
        return matches[0] if len(matches) == 1 else None

    auc_l = one("L", "auc")
    balanced_l = one("L", "balanced_accuracy")
    auc_2l = one("2L", "auc")
    required = (auc_l, balanced_l, auc_2l)
    available = all(
        row is not None
        and row.get("interval_available") is True
        and row.get("confirmatory_support") is True
        for row in required
    )
    if not available:
        return False, None
    assert auc_l is not None and balanced_l is not None and auc_2l is not None
    lower_auc = _optional_finite_number(auc_l.get("lower_95"))
    lower_balanced = _optional_finite_number(balanced_l.get("lower_95"))
    point_auc_2l = _optional_finite_number(auc_2l.get("point_estimate"))
    _require(
        None not in {lower_auc, lower_balanced, point_auc_2l},
        "gate_logic",
        "available_external_value_missing",
    )
    assert lower_auc is not None and lower_balanced is not None and point_auc_2l is not None
    return True, lower_auc > 0.5 and lower_balanced > 0.5 and point_auc_2l > 0.5


def _validate_summary(summary: Mapping[str, Any], members: Sequence[Mapping[str, str]]) -> None:
    _require(
        set(summary)
        == {
            "schema_version",
            "frequency_estimand",
            "sites",
            "counts",
            "artifact_sha256",
            "permitted_claim_classification",
            "axis_contrasts",
            "compute_controls",
        },
        "summary",
        "top_level_field_closure_mismatch",
    )
    _require(
        summary.get("schema_version") == "1.0"
        and summary.get("frequency_estimand") == "finite_design_empirical_frequency"
        and summary.get("axis_contrasts") == "descriptive_paired_only",
        "summary",
        "frozen_identity_mismatch",
    )
    counts = _require_exact_keys(
        summary.get("counts"),
        {"recorded_variants", "unique_mtmf_fits", "failed_members"},
        check="summary",
    )
    expected_counts = {
        "recorded_variants": TOTAL_VARIANTS,
        "unique_mtmf_fits": TOTAL_UNIQUE_FITS,
        "failed_members": 0,
    }
    _require(
        all(_exact_scalar(counts.get(key), value) for key, value in expected_counts.items()),
        "summary",
        "count_mismatch",
    )
    controls = _require_exact_keys(
        summary.get("compute_controls"),
        {"device", "batch_size", "storage_layout", "scientifically_inert"},
        check="summary",
    )
    _require(
        controls.get("device") == "cpu"
        and type(controls.get("batch_size")) is int
        and controls.get("batch_size") == 1
        and controls.get("storage_layout") == "disk"
        and controls.get("scientifically_inert") is True,
        "summary",
        "compute_control_mismatch",
    )
    site_rows = summary.get("sites")
    _require(
        isinstance(site_rows, list) and len(site_rows) == 2, "summary", "site_closure_mismatch"
    )
    by_site = {row.get("site"): row for row in site_rows if isinstance(row, dict)}
    _require(set(by_site) == set(FROZEN_SITES), "summary", "site_closure_mismatch")
    for site in FROZEN_SITES:
        _validate_site_summary(by_site[site], site)
    goldfield = by_site["goldfield"]
    _validate_goldfield_gate(goldfield)
    _require(
        summary.get("permitted_claim_classification")
        == goldfield.get("permitted_claim_classification"),
        "gate_logic",
        "top_level_classification_mismatch",
    )
    _require(len(members) == TOTAL_VARIANTS, "summary", "member_count_crosscheck_failed")


def _validate_site_summary(site_summary: Mapping[str, Any], site: str) -> None:
    expected_fields = {
        "site",
        "recorded_variants",
        "unique_mtmf_fits",
        "joint_valid_members",
        "failed_members",
        "analytical_cell_valid_member_counts",
        "confirmatory_gate_available",
        "confirmatory_gate_pass",
        "permitted_claim_classification",
        "external_covariance_estimand",
        "goldfield_alunite_gate_components",
        "nested_block_bootstrap",
        "strict_covariance_exclusion",
    }
    if site == "goldfield":
        expected_fields.update(
            {
                "analytical_cells_complete",
                "stability_available",
                "stability_pass",
                "external_interval_available",
                "external_pass",
                "strict_covariance_interval_available",
                "strict_covariance_pass",
            }
        )
    _require(set(site_summary) == expected_fields, "summary", "site_field_closure_mismatch")
    _require(
        site_summary.get("recorded_variants") == VARIANTS_PER_SITE
        and site_summary.get("unique_mtmf_fits") == UNIQUE_FITS_PER_SITE
        and site_summary.get("joint_valid_members") == JOINT_MEMBERS_PER_SITE,
        "summary",
        "site_count_mismatch",
    )
    _require(site_summary.get("failed_members") == [], "summary", "failed_member_records_present")
    cells = site_summary.get("analytical_cell_valid_member_counts")
    _require(
        isinstance(cells, dict)
        and len(cells) == ANALYTICAL_CELLS
        and set(cells.values()) == {STOCHASTIC_REPLICATES},
        "summary",
        "analytical_cell_closure_mismatch",
    )
    parsed_cells: set[tuple[str, str, str]] = set()
    for key in cells:
        try:
            cell = json.loads(key, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, SealViolation) as error:
            raise SealViolation("summary", "analytical_cell_identity_invalid") from error
        _require(
            isinstance(cell, dict) and set(cell) == {"ridge", "quantile", "gate"},
            "summary",
            "analytical_cell_identity_invalid",
        )
        parsed_cells.add((str(cell["ridge"]), str(cell["quantile"]), str(cell["gate"])))
    expected_cells = {
        (ridge, quantile, gate) for ridge in RIDGES for quantile in QUANTILES for gate in GATES
    }
    _require(parsed_cells == expected_cells, "summary", "analytical_cell_identity_mismatch")
    nested = site_summary.get("nested_block_bootstrap")
    _require(
        isinstance(nested, dict)
        and set(nested)
        == {
            "shared_draw_per_replicate",
            "member_summary_within_replicate",
            "replicates",
            "finite_replicate_fraction_required",
            "dominant_class_switch_lower_95",
            "dominant_class_switch_upper_95",
            "endpoint_intervals",
            "external_intervals",
        }
        and nested.get("shared_draw_per_replicate") is True
        and nested.get("member_summary_within_replicate") == "median"
        and nested.get("replicates") == BOOTSTRAP_REPLICATES
        and nested.get("finite_replicate_fraction_required") == 0.95,
        "summary",
        "nested_bootstrap_identity_mismatch",
    )
    strict = site_summary.get("strict_covariance_exclusion")
    _require(
        isinstance(strict, dict)
        and set(strict) == {"status", "pooled_with_operational", "fold_failures"}
        and strict.get("pooled_with_operational") is False
        and strict.get("fold_failures") == {},
        "summary",
        "strict_covariance_identity_mismatch",
    )
    expected_status = "complete" if site == "goldfield" else "not_applicable"
    _require(
        strict.get("status") == expected_status, "summary", "strict_covariance_status_mismatch"
    )
    _require(
        site_summary.get("external_covariance_estimand")
        == "full_scene_covariance_operational_transductive",
        "summary",
        "external_covariance_estimand_mismatch",
    )
    components = site_summary.get("goldfield_alunite_gate_components")
    _require(
        isinstance(components, dict)
        and set(components)
        == {
            "stable_core_retention",
            "median_rank_correlation",
            "rank_correlation_5th_percentile",
            "dominant_class_switch_nested_bootstrap_upper_95",
            "external_interval_gate",
        },
        "summary",
        "gate_component_field_closure_mismatch",
    )
    if site == "bingham":
        _require(
            site_summary.get("confirmatory_gate_available") is False
            and site_summary.get("confirmatory_gate_pass") is None
            and site_summary.get("permitted_claim_classification")
            == "map_stability_only_no_external_reference",
            "gate_logic",
            "bingham_classification_mismatch",
        )


def _validate_goldfield_gate(site_summary: Mapping[str, Any]) -> None:
    nested = site_summary["nested_block_bootstrap"]
    endpoint_rows = nested.get("endpoint_intervals")
    external_rows = nested.get("external_intervals")
    _require(
        isinstance(endpoint_rows, list) and isinstance(external_rows, list),
        "gate_logic",
        "interval_records_missing",
    )
    switch_matches = [
        row
        for row in endpoint_rows
        if isinstance(row, dict)
        and row.get("scale") == "L"
        and row.get("mineral") == "alunite"
        and row.get("metric") == "dominant_class_switch_frequency"
    ]
    _require(len(switch_matches) == 1, "gate_logic", "switch_interval_identity_mismatch")
    switch = switch_matches[0]
    components = site_summary.get("goldfield_alunite_gate_components")
    _require(isinstance(components, dict), "gate_logic", "gate_components_missing")
    stable_core = _optional_finite_number(components.get("stable_core_retention"))
    median_rank = _optional_finite_number(components.get("median_rank_correlation"))
    rank_fifth = _optional_finite_number(components.get("rank_correlation_5th_percentile"))
    switch_upper = _optional_finite_number(switch.get("upper_95"))
    stability_available = (
        None not in {stable_core, median_rank, rank_fifth, switch_upper}
        and switch.get("interval_available") is True
    )
    stability_pass: bool | None = None
    if stability_available:
        assert (
            stable_core is not None
            and median_rank is not None
            and rank_fifth is not None
            and switch_upper is not None
        )
        stability_pass = (
            stable_core >= 0.80
            and median_rank >= 0.80
            and rank_fifth > 0.50
            and switch_upper <= 0.20
        )
    operational_available, operational_pass = _external_gate(external_rows, "full_scene_covariance")
    strict_available, strict_pass = _external_gate(external_rows, "strict_covariance_exclusion")
    analytical_complete = site_summary.get("analytical_cells_complete") is True
    confirmatory_available = (
        analytical_complete and stability_available and operational_available and strict_available
    )
    confirmatory_pass = (
        bool(stability_pass and operational_pass and strict_pass)
        if confirmatory_available
        else None
    )
    expected = {
        "stability_available": stability_available,
        "stability_pass": stability_pass,
        "external_interval_available": operational_available,
        "external_pass": operational_pass,
        "strict_covariance_interval_available": strict_available,
        "strict_covariance_pass": strict_pass,
        "confirmatory_gate_available": confirmatory_available,
        "confirmatory_gate_pass": confirmatory_pass,
        "permitted_claim_classification": _classification(
            stability_pass, operational_pass, strict_pass
        )
        if analytical_complete
        else "unavailable_required_evidence",
    }
    for key, value in expected.items():
        _require(
            _exact_scalar(site_summary.get(key), value),
            "gate_logic",
            "gate_or_classification_mismatch",
        )
    _require(
        components.get("dominant_class_switch_nested_bootstrap_upper_95") == switch.get("upper_95")
        and components.get("external_interval_gate")
        == site_summary.get("external_interval_available"),
        "gate_logic",
        "gate_component_binding_mismatch",
    )


def _expected_report(summary: Mapping[str, Any]) -> bytes:
    lines = [
        "# E6 MTMF ensemble sensitivity",
        "",
        "This report is generated from the frozen finite sensitivity design.",
        "Detection frequencies are empirical design frequencies, not probabilities.",
        "Axis contrasts are descriptive paired contrasts and do not identify variance components.",
        "",
        "## Site status",
        "",
    ]
    for site in summary["sites"]:
        lines.append(
            f"- {site['site']}: {site['joint_valid_members']} valid joint members; "
            f"confirmatory gate available = {site['confirmatory_gate_available']}."
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _validate_timing_pilot(
    sealed: SealedDirectory,
    expected_sha256: str,
    fit_checksums: Mapping[str, str],
) -> str:
    _require(_is_sha256(expected_sha256), "timing_pilot", "expected_hash_invalid")
    timing_bytes = sealed.read_bytes("timing_pilot.json")
    observed = hashlib.sha256(timing_bytes).hexdigest()
    _require(
        observed == expected_sha256, "timing_pilot", "detached_hash_mismatch", "timing_pilot.json"
    )
    payload = _decode_json(timing_bytes, "timing_pilot.json")
    _require(
        set(payload) == {"schema_version", "mode", "fit_count", "records"}
        and payload.get("schema_version") == "1.0"
        and payload.get("mode") == "timing_pilot_only"
        and payload.get("fit_count") == 4,
        "timing_pilot",
        "identity_mismatch",
    )
    records = payload.get("records")
    _require(
        isinstance(records, list) and len(records) == 4, "timing_pilot", "record_count_mismatch"
    )
    expected_fits = {f"{site}:fit:baseline:r0.01" for site in FROZEN_SITES} | {
        f"{site}:fit:joint:r00:ridge0.01" for site in FROZEN_SITES
    }
    observed_fits: set[str] = set()
    for record in records:
        _require(
            isinstance(record, dict)
            and set(record)
            == {
                "site",
                "scene",
                "fit_id",
                "member_class",
                "stochastic_replicate",
                "wall_time_seconds",
                "peak_memory_bytes",
                "output_sha256",
                "device",
                "scientific_outputs_retained",
            },
            "timing_pilot",
            "record_field_closure_mismatch",
        )
        site = record.get("site")
        fit_id = record.get("fit_id")
        _require(
            site in FROZEN_SITES
            and record.get("scene") == ANCHOR_SCENES[site]
            and fit_id in expected_fits,
            "timing_pilot",
            "fit_identity_mismatch",
        )
        _require(
            record.get("device") == "cpu" and record.get("scientific_outputs_retained") is False,
            "timing_pilot",
            "retention_or_device_mismatch",
        )
        _require(
            record.get("output_sha256") == fit_checksums.get(fit_id),
            "timing_pilot",
            "fit_checksum_binding_mismatch",
        )
        _require(
            type(record.get("wall_time_seconds")) in {int, float}
            and record["wall_time_seconds"] >= 0
            and type(record.get("peak_memory_bytes")) is int
            and record["peak_memory_bytes"] >= 0,
            "timing_pilot",
            "operational_metadata_invalid",
        )
        assert isinstance(fit_id, str)
        observed_fits.add(fit_id)
    _require(observed_fits == expected_fits, "timing_pilot", "fit_closure_mismatch")
    return observed


def _validate_map_metadata(
    sealed: SealedDirectory,
    design: Mapping[str, Any],
    expected_grids: Mapping[str, tuple[tuple[int, int], str, tuple[float, ...]]],
) -> int:
    try:
        import rasterio
    except ImportError as error:
        raise SealViolation("map_metadata", "rasterio_unavailable") from error
    goldfield_reference = design["rockwell_reference"]
    goldfield_grid = (
        tuple(goldfield_reference["shape"]),
        str(goldfield_reference["crs"]),
        tuple(float(value) for value in goldfield_reference["transform"]),
    )
    _require(
        expected_grids.get("goldfield") == goldfield_grid,
        "map_metadata",
        "goldfield_grid_binding_mismatch",
    )
    site_grids: dict[str, tuple[tuple[int, int], str, tuple[float, ...]]] = {}
    expected_metadata = {
        "n_valid": ("uint16", 0.0),
        "detection_frequency": ("float32", math.nan),
        "confidence_class": ("int8", -1.0),
        "modal_class": ("int16", -2.0),
        "modal_frequency": ("float32", math.nan),
        "class_entropy": ("float32", math.nan),
        "switch_frequency": ("float32", math.nan),
    }
    for relative in sorted(_map_paths()):
        suffix = next(name for name in expected_metadata if relative.endswith(f"_{name}.tif"))
        site = next(
            site_name for site_name in FROZEN_SITES if relative.startswith(f"maps/{site_name}_")
        )
        with sealed.open_file(relative) as descriptor:
            try:
                with rasterio.open(f"/dev/fd/{descriptor}") as dataset:
                    _require(
                        dataset.driver == "GTiff"
                        and dataset.count == 1
                        and dataset.width > 0
                        and dataset.height > 0
                        and dataset.crs is not None,
                        "map_metadata",
                        "raster_structure_mismatch",
                        relative,
                    )
                    dtype, nodata = expected_metadata[suffix]
                    _require(
                        dataset.dtypes == (dtype,),
                        "map_metadata",
                        "raster_dtype_mismatch",
                        relative,
                    )
                    if math.isnan(nodata):
                        _require(
                            dataset.nodata is not None and math.isnan(dataset.nodata),
                            "map_metadata",
                            "raster_nodata_mismatch",
                            relative,
                        )
                    else:
                        _require(
                            dataset.nodata == nodata,
                            "map_metadata",
                            "raster_nodata_mismatch",
                            relative,
                        )
                    affine = dataset.transform
                    grid = (
                        (dataset.height, dataset.width),
                        dataset.crs.to_string(),
                        (affine.a, affine.b, affine.c, affine.d, affine.e, affine.f),
                    )
            except SealViolation:
                raise
            except Exception as error:
                raise SealViolation(
                    "map_metadata", "raster_open_failed", filename=relative
                ) from error
        if site in site_grids:
            _require(
                site_grids[site] == grid, "map_metadata", "within_site_grid_mismatch", relative
            )
        else:
            site_grids[site] = grid
    _require(site_grids == expected_grids, "map_metadata", "provenance_grid_mismatch")
    return len(_map_paths())


def _closure_digest(sealed: SealedDirectory, files: set[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(sealed.sha256(relative).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_run(
    *,
    run_dir: Path,
    source_manifest: Path,
    block_manifest: Path,
    expected_source_manifest_sha256: str,
    expected_timing_pilot_sha256: str,
) -> VerificationReport:
    """Verify one explicitly supplied completed run directory."""
    report = VerificationReport()
    try:
        with SealedDirectory(run_dir) as sealed:
            allowed_directories = {
                "maps",
                ".score_cache",
                *(f".score_cache/{site}" for site in FROZEN_SITES),
            }
            observed_files = sealed.inventory(allowed_directories=allowed_directories)
            initial_records = dict(sealed.records)
            initial_root = sealed.root_record()
            design = _load_json(sealed, "design.json")
            _validate_design(design)
            design_sha = sealed.sha256("design.json")
            report.checks.append(
                CheckRecord(True, "design_identity", filename="design.json", digest=design_sha)
            )
            members, fit_checksums, cache_paths = _load_members(sealed, design, design_sha)
            report.checks.append(
                CheckRecord(
                    True, "member_status_closure", count=len(members), filename="members.csv"
                )
            )
            report.checks.append(
                CheckRecord(True, "fit_checksum_identities", count=len(fit_checksums))
            )
            report.checks.append(
                CheckRecord(True, "fit_cache_path_identities", count=len(cache_paths))
            )

            map_paths = _map_paths()
            root_artifacts = {
                "design.json",
                "members.csv",
                "member_metrics.csv",
                "factor_effects.csv",
                "calibration.csv",
                "summary.json",
                "report.md",
                "timing_pilot.json",
            }
            expected_files = root_artifacts | map_paths | cache_paths
            missing = expected_files - observed_files
            unexpected = observed_files - expected_files
            if missing:
                raise SealViolation(
                    "artifact_closure",
                    "required_file_missing",
                    filename=sorted(missing)[0],
                    count=len(missing),
                )
            if unexpected:
                raise SealViolation(
                    "artifact_closure",
                    "unexpected_file",
                    filename=sorted(unexpected)[0],
                    count=len(unexpected),
                )
            report.checks.append(CheckRecord(True, "artifact_closure", count=len(observed_files)))

            fit_cache_count = _validate_fit_caches(sealed, members, fit_checksums, cache_paths)
            report.checks.append(CheckRecord(True, "fit_cache_semantics", count=fit_cache_count))

            source_sha, source_count = _validate_source_manifest(
                source_manifest, expected_source_manifest_sha256, design
            )
            report.checks.append(
                CheckRecord(
                    True,
                    "source_manifest_binding",
                    count=source_count,
                    filename="source_manifest.sha256",
                    digest=source_sha,
                )
            )
            expected_grids, block_manifest_sha = _validate_block_manifest(block_manifest, design)
            report.checks.append(
                CheckRecord(
                    True,
                    "block_manifest_binding",
                    count=len(expected_grids),
                    filename="block_manifest.json",
                    digest=block_manifest_sha,
                )
            )

            summary = _load_json(sealed, "summary.json")
            artifact_paths = {
                "design.json",
                "members.csv",
                "member_metrics.csv",
                "factor_effects.csv",
                "calibration.csv",
            } | map_paths
            declared_hashes = _verify_declared_hashes(sealed, summary, artifact_paths)
            report.checks.append(
                CheckRecord(True, "declared_artifact_sha256", count=len(declared_hashes))
            )

            member_by_id = {row["member_id"]: row for row in members}
            metric_count = _scan_metrics(sealed, member_by_id)
            factor_count = _scan_factor_effects(sealed)
            calibration_count = _scan_calibration(sealed)
            report.checks.extend(
                (
                    CheckRecord(
                        True,
                        "member_metrics_structure",
                        count=metric_count,
                        filename="member_metrics.csv",
                    ),
                    CheckRecord(
                        True,
                        "factor_effects_structure",
                        count=factor_count,
                        filename="factor_effects.csv",
                    ),
                    CheckRecord(
                        True,
                        "calibration_structure",
                        count=calibration_count,
                        filename="calibration.csv",
                    ),
                )
            )
            _validate_summary(summary, members)
            report.checks.append(CheckRecord(True, "gate_classification_logic"))
            _require(
                sealed.read_bytes("report.md") == _expected_report(summary),
                "sealed_report",
                "generated_report_mismatch",
                "report.md",
            )
            report.checks.append(
                CheckRecord(
                    True, "sealed_report", filename="report.md", digest=sealed.sha256("report.md")
                )
            )
            timing_sha = _validate_timing_pilot(sealed, expected_timing_pilot_sha256, fit_checksums)
            report.checks.append(
                CheckRecord(
                    True,
                    "timing_pilot_binding",
                    count=4,
                    filename="timing_pilot.json",
                    digest=timing_sha,
                )
            )
            map_count = _validate_map_metadata(sealed, design, expected_grids)
            report.checks.append(CheckRecord(True, "map_metadata", count=map_count))
            closure_sha = _closure_digest(sealed, expected_files)
            final_files = sealed.inventory(allowed_directories=allowed_directories)
            _require(
                final_files == observed_files
                and sealed.records == initial_records
                and sealed.root_record() == initial_root,
                "artifact_closure",
                "directory_or_artifact_changed_during_verification",
            )
            report.checks.append(
                CheckRecord(True, "sealed_closure", count=len(expected_files), digest=closure_sha)
            )
    except SealViolation as error:
        report.checks.append(
            CheckRecord(
                False,
                error.check,
                count=error.count,
                discrepancy=error.discrepancy,
            )
        )
    except Exception:
        report.checks.append(
            CheckRecord(False, "internal_validation", discrepancy="sealed_validation_error")
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit post-run verifier interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="explicit path to one completed, locally available E6 run directory",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="detached exact 49-entry v2 source SHA-256 manifest admitted for the run",
    )
    parser.add_argument(
        "--block-manifest",
        type=Path,
        required=True,
        help="explicit SHA-bound M2 block manifest used by the completed run",
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        required=True,
        help="out-of-band SHA-256 recorded when the source capsule was admitted",
    )
    parser.add_argument(
        "--expected-timing-pilot-sha256",
        required=True,
        help="out-of-band SHA-256 recorded for timing_pilot.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sealed verifier and return zero only for full conformance."""
    args = build_parser().parse_args(argv)
    report = verify_run(
        run_dir=args.run_dir,
        source_manifest=args.source_manifest,
        block_manifest=args.block_manifest,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_timing_pilot_sha256=args.expected_timing_pilot_sha256,
    )
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
