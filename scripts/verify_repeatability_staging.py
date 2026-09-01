#!/usr/bin/env python3
"""Verify the endpoint-blind repeatability Stage A staged-root boundary."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).absolute().parent
if os.fspath(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPT_DIR))

import verify_source_capsule as source_capsule  # noqa: E402

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_VERSION = "1.0"
PROJECT_RELATIVE_PATH = Path("Tanager/tanager-rocks")
SIBLING_RELATIVE_PATH = Path("Tanager/tanager-spec")
SOURCE_MANIFEST_RELATIVE_PATH = Path("docs/m2_repeatability_bigmem_source_manifest.sha256")
INPUT_MANIFEST_RELATIVE_PATH = Path("docs/input_manifest.json")
OUTPUT_RELATIVE_PATH = Path("data/processed/repeatability_metric_contract_v2_20260811")
SCIENTIFIC_EXECUTION_IDENTITY = "paired-complete-block-metric-contract-v2"
WRAPPER_RELATIVE_PATH = Path("scripts/run_repeatability_bigmem.sbatch")
WRAPPER_RUNTIME_RELATIVE_PATH = Path("runtime/repeatability_metric_contract_v2_20260811")
CACHE_RELATIVE_PATH = Path("uv-cache")
LOG_RELATIVE_PATH = Path("slurm_logs")
SPECTRAL_ARCHIVE_RELATIVE_PATH = Path("data/speclib/ASCIIdata_splib07a.zip")
SPECTRAL_TREE_RELATIVE_PATH = Path("data/speclib/ASCIIdata_splib07a")
PROJECT_SUPPORT_FILES = frozenset({"README.md"})
SIBLING_SUPPORT_FILES = frozenset({"README.md"})
OPAQUE_PROJECT_DIRECTORIES = frozenset({".git"})
EXPECTED_RAW_SCENE_COUNT = 7
REQUIRED_SOURCE_MEMBERS = frozenset(
    {
        "docs/input_manifest.json",
        "scripts/run_repeatability_bigmem.sbatch",
        "src/tanager_rocks/repeatability.py",
        "uv.lock",
    }
)


class StagingVerificationError(Exception):
    """A deliberately low-disclosure Stage A verification failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

    def render(self) -> str:
        """Return the closed failure record."""
        return f"FAIL code={self.code}"


@dataclass(frozen=True)
class StagingConfig:
    """Explicit control-plane inputs for one staged-root verification."""

    actual_root: Path
    expected_root: Path
    e6_root: Path
    proposal: Path
    expected_proposal_sha256: str
    source_manifest: Path
    expected_source_manifest_sha256: str
    expected_source_member_count: int
    evidence_output: Path


@dataclass(frozen=True)
class InputClosurePaths:
    """Staged paths recorded for identity-only admission and fault tests."""

    raw_dir: Path
    speclib_dir: Path
    validation_dir: Path
    output_dir: Path
    reference_dir: Path


@dataclass(frozen=True)
class InputAdmissionSummary:
    """Sanitized output from independent identity-only input admission."""

    input_manifest_sha256: str
    raw_scene_count: int
    raw_scene_hashes_sha256: str
    spectral_archive_sha256: str
    spectral_library_member_count: int
    spectral_library_tree_sha256: str


@dataclass(frozen=True)
class FileSnapshot:
    """Descriptor-bound size and digest for one regular single-link file."""

    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class InputClosureSnapshot:
    """Descriptor-bound identity of every raw and spectral-library input."""

    file_count: int
    sha256: str
    spectral_library_member_count: int
    spectral_library_tree_sha256: str


@dataclass(frozen=True)
class InputAdmissionContext:
    """Descriptor-bound facts supplied to an identity-only admission adapter."""

    paths: InputClosurePaths
    root: Path
    input_manifest_path: Path
    input_manifest_sha256: str
    expectations: Mapping[str, FileSnapshot]
    snapshot: InputClosureSnapshot


@dataclass(frozen=True)
class ClosedLayoutSummary:
    """Sanitized counts for the exact staged file-and-directory boundary."""

    project_directory_count: int
    project_file_count: int
    sibling_directory_count: int
    sibling_file_count: int
    support_files_sha256: str


@dataclass(frozen=True)
class BoundDirectory:
    """One no-follow directory descriptor and its parent binding."""

    parent_fd: int
    name: str
    descriptor: int
    identity: source_capsule.DirectoryIdentity
    display: str


InputAdmitter = Callable[[InputAdmissionContext], Mapping[str, Any]]


def _fail(code: str) -> None:
    raise StagingVerificationError(code)


def _validate_sha256(value: str, *, code: str) -> None:
    if SHA256_RE.fullmatch(value) is None:
        _fail(code)


def _canonical_absolute(path: Path, *, code: str) -> Path:
    raw = os.fspath(path)
    if not raw or not path.is_absolute():
        _fail(code)
    normalized = Path(os.path.abspath(raw))
    if normalized != path:
        _fail(code)
    return normalized


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        _fail("FILESYSTEM_UNAVAILABLE")
    return True


def _open_bound_directory(parent_fd: int, name: str, *, display: str) -> BoundDirectory:
    try:
        descriptor = source_capsule._open_directory_at(
            parent_fd,
            name,
            path=display,
            check="staging_layout",
        )
    except source_capsule.VerificationError as error:
        raise StagingVerificationError("ROOT_OR_LAYOUT_UNSAFE") from error
    return BoundDirectory(
        parent_fd=parent_fd,
        name=name,
        descriptor=descriptor,
        identity=source_capsule.DirectoryIdentity.from_stat(os.fstat(descriptor)),
        display=display,
    )


def _verify_bound_directory(bound: BoundDirectory) -> None:
    try:
        source_capsule._verify_directory_binding(
            bound.parent_fd,
            bound.name,
            bound.descriptor,
            bound.identity,
            path=bound.display,
            check="staging_layout",
        )
    except source_capsule.VerificationError as error:
        raise StagingVerificationError("ROOT_OR_LAYOUT_CHANGED") from error


def _open_staging_layout(root: Path) -> tuple[int, tuple[BoundDirectory, ...]]:
    try:
        parent_fd, name = source_capsule._open_parent(root, check="staging_root")
    except source_capsule.VerificationError as error:
        raise StagingVerificationError("ROOT_OR_LAYOUT_UNSAFE") from error

    opened: list[BoundDirectory] = []
    try:
        root_bound = _open_bound_directory(parent_fd, name, display=os.fspath(root))
        opened.append(root_bound)
        workspace = _open_bound_directory(
            root_bound.descriptor,
            "Tanager",
            display=os.fspath(root / "Tanager"),
        )
        opened.append(workspace)
        project = _open_bound_directory(
            workspace.descriptor,
            "tanager-rocks",
            display=os.fspath(root / PROJECT_RELATIVE_PATH),
        )
        opened.append(project)
        sibling = _open_bound_directory(
            workspace.descriptor,
            "tanager-spec",
            display=os.fspath(root / SIBLING_RELATIVE_PATH),
        )
        opened.append(sibling)
        logs = _open_bound_directory(
            root_bound.descriptor,
            LOG_RELATIVE_PATH.name,
            display=os.fspath(root / LOG_RELATIVE_PATH),
        )
        opened.append(logs)
        return parent_fd, tuple(opened)
    except BaseException:
        for bound in reversed(opened):
            os.close(bound.descriptor)
        os.close(parent_fd)
        raise


def _close_staging_layout(parent_fd: int, bounds: Sequence[BoundDirectory]) -> None:
    for bound in reversed(bounds):
        os.close(bound.descriptor)
    os.close(parent_fd)


def _snapshot_single_link_file(
    root_fd: int,
    parts: Sequence[str],
    *,
    display: str,
    unsafe_code: str,
    hardlink_code: str,
    changed_code: str,
    content_digest: Any | None = None,
    payload_chunks: list[bytes] | None = None,
) -> FileSnapshot:
    descriptors = [os.dup(root_fd)]
    bindings: list[tuple[int, str, int, source_capsule.DirectoryIdentity]] = []
    file_descriptor: int | None = None
    try:
        current = descriptors[0]
        for part in parts[:-1]:
            following = source_capsule._open_directory_at(
                current,
                part,
                path=display,
                check="single_link_file",
            )
            identity = source_capsule.DirectoryIdentity.from_stat(os.fstat(following))
            bindings.append((current, part, following, identity))
            descriptors.append(following)
            current = following

        name = parts[-1]
        before = source_capsule._lstat_at(
            current,
            name,
            path=display,
            check="single_link_file",
        )
        if stat.S_ISLNK(before.st_mode):
            _fail(unsafe_code)
        if not stat.S_ISREG(before.st_mode):
            _fail(unsafe_code)
        if before.st_nlink != 1:
            _fail(hardlink_code)
        expected = source_capsule.FileIdentity.from_stat(before)
        file_descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
        if source_capsule.FileIdentity.from_stat(os.fstat(file_descriptor)) != expected:
            _fail(changed_code)
        digest = hashlib.sha256()
        while chunk := os.read(file_descriptor, source_capsule.READ_SIZE):
            digest.update(chunk)
            if content_digest is not None:
                content_digest.update(chunk)
            if payload_chunks is not None:
                payload_chunks.append(chunk)
        rebound = source_capsule._lstat_at(
            current,
            name,
            path=display,
            check="single_link_file",
        )
        if (
            source_capsule.FileIdentity.from_stat(os.fstat(file_descriptor)) != expected
            or source_capsule.FileIdentity.from_stat(rebound) != expected
        ):
            _fail(changed_code)
        for parent_fd, component, descriptor, identity in reversed(bindings):
            source_capsule._verify_directory_binding(
                parent_fd,
                component,
                descriptor,
                identity,
                path=display,
                check="single_link_file",
            )
        return FileSnapshot(size_bytes=expected.size, sha256=digest.hexdigest())
    except StagingVerificationError:
        raise
    except (OSError, source_capsule.VerificationError) as error:
        raise StagingVerificationError(unsafe_code) from error
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _verify_single_link_member(
    root_fd: int,
    entry: source_capsule.ManifestEntry,
) -> None:
    snapshot = _snapshot_single_link_file(
        root_fd,
        entry.parts,
        display=entry.path,
        unsafe_code="SOURCE_MEMBER_UNSAFE",
        hardlink_code="SOURCE_MEMBER_HARDLINK",
        changed_code="SOURCE_MEMBER_CHANGED",
    )
    if snapshot.sha256 != entry.sha256:
        _fail("SOURCE_MEMBER_CHANGED")


def _verify_source_single_links(
    entries: Sequence[source_capsule.ManifestEntry],
    *,
    project_fd: int,
    sibling_fd: int,
) -> None:
    for entry in entries:
        root_fd = project_fd if entry.root == "project" else sibling_fd
        _verify_single_link_member(root_fd, entry)


def _read_control(path: Path, *, code: str) -> tuple[bytes, str]:
    try:
        return source_capsule._read_control_file(
            path,
            check="staging_control",
            after_read_hook=None,
        )
    except source_capsule.VerificationError as error:
        raise StagingVerificationError(code) from error


def _parse_digest_bound_manifest(
    payload: bytes,
    *,
    expected_count: int,
) -> tuple[source_capsule.ManifestEntry, ...]:
    """Validate an approved manifest without changing its governed byte order.

    The frozen repeatability manifest is sorted by complete checksum records,
    while the reusable source-capsule parser has a legacy path-order
    precondition.  The detached digest binds the original bytes.  Reordering a
    copy in memory only satisfies that parser precondition; it does not replace,
    rewrite, or weaken the approved manifest identity.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise source_capsule.VerificationError(
            path="<manifest>",
            check="manifest_format",
            reason="invalid_utf8",
        ) from error
    if "\r" in text or (text and not text.endswith("\n")):
        raise source_capsule.VerificationError(
            path="<manifest>",
            check="manifest_format",
            reason="noncanonical_newline",
        )
    lines = text[:-1].split("\n") if text else []
    keyed_lines: list[tuple[str, str]] = []
    for line in lines:
        match = source_capsule.MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise source_capsule.VerificationError(
                path="<manifest>",
                check="manifest_format",
                reason="malformed_line",
            )
        keyed_lines.append((match.group("path"), line))
    path_sorted_payload = "".join(
        f"{line}\n" for _path, line in sorted(keyed_lines, key=lambda item: item[0])
    ).encode()
    return source_capsule._parse_manifest(
        path_sorted_payload,
        expected_count=expected_count,
    )


def _verify_source_members(
    entries: Sequence[source_capsule.ManifestEntry],
    *,
    project_fd: int,
    sibling_fd: int,
) -> None:
    for entry in entries:
        root_fd = project_fd if entry.root == "project" else sibling_fd
        try:
            observed_sha256 = source_capsule._read_member(
                root_fd,
                entry,
                after_read_hook=None,
            )
        except source_capsule.VerificationError as error:
            raise StagingVerificationError("SOURCE_CAPSULE_FAILED") from error
        if observed_sha256 != entry.sha256:
            _fail("SOURCE_CAPSULE_FAILED")


def _verify_source_capsule(
    config: StagingConfig,
    *,
    project_fd: int,
    sibling_fd: int,
) -> tuple[str, int, tuple[source_capsule.ManifestEntry, ...]]:
    manifest_payload, manifest_sha256 = _read_control(
        config.source_manifest,
        code="SOURCE_MANIFEST_UNSAFE",
    )
    if manifest_sha256 != config.expected_source_manifest_sha256:
        _fail("SOURCE_MANIFEST_DRIFT")
    try:
        entries = _parse_digest_bound_manifest(
            manifest_payload,
            expected_count=config.expected_source_member_count,
        )
    except source_capsule.VerificationError as error:
        raise StagingVerificationError("SOURCE_MANIFEST_INVALID") from error
    if not REQUIRED_SOURCE_MEMBERS.issubset(entry.path for entry in entries):
        _fail("SOURCE_MANIFEST_CLOSURE")
    _verify_source_members(entries, project_fd=project_fd, sibling_fd=sibling_fd)
    _verify_source_single_links(entries, project_fd=project_fd, sibling_fd=sibling_fd)
    return manifest_sha256, len(entries), entries


def _safe_relative_path(value: str, *, code: str) -> tuple[str, ...]:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
    ):
        _fail(code)
    return tuple(pure.parts)


def _parse_input_expectations(payload: bytes) -> dict[str, FileSnapshot]:
    def reject_constant(_value: str) -> None:
        _fail("INPUT_MANIFEST_INVALID")

    try:
        parsed = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StagingVerificationError("INPUT_MANIFEST_INVALID") from error
    if not isinstance(parsed, dict) or parsed.get("schema_version") != "1.0":
        _fail("INPUT_MANIFEST_INVALID")
    records = parsed.get("inputs")
    if not isinstance(records, list):
        _fail("INPUT_MANIFEST_INVALID")

    selected: dict[str, FileSnapshot] = {}
    raw_count = 0
    archive_seen = False
    for record in records:
        if not isinstance(record, dict):
            _fail("INPUT_MANIFEST_INVALID")
        input_id = record.get("id")
        logical_path = record.get("logical_path")
        if not isinstance(input_id, str) or not isinstance(logical_path, str):
            _fail("INPUT_MANIFEST_INVALID")
        is_raw = logical_path.startswith("data/raw/") and logical_path.endswith("_ortho_sr_hdf5.h5")
        is_archive = input_id == "usgs-splib07a-archive"
        if not is_raw and not is_archive:
            continue
        parts = _safe_relative_path(logical_path, code="INPUT_MANIFEST_INVALID")
        if is_raw and (len(parts) != 3 or parts[:2] != ("data", "raw")):
            _fail("INPUT_MANIFEST_INVALID")
        if is_archive and logical_path != SPECTRAL_ARCHIVE_RELATIVE_PATH.as_posix():
            _fail("INPUT_MANIFEST_INVALID")
        size_bytes = record.get("size_bytes")
        sha256 = record.get("sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(sha256, str)
        ):
            _fail("INPUT_MANIFEST_INVALID")
        _validate_sha256(sha256, code="INPUT_MANIFEST_INVALID")
        if logical_path in selected:
            _fail("INPUT_MANIFEST_INVALID")
        selected[logical_path] = FileSnapshot(size_bytes=size_bytes, sha256=sha256)
        raw_count += int(is_raw)
        archive_seen = archive_seen or is_archive
    if raw_count != EXPECTED_RAW_SCENE_COUNT or not archive_seen:
        _fail("INPUT_MANIFEST_INVALID")
    return selected


def _collect_inventory(
    root_fd: int,
    *,
    skip_directories: frozenset[str] = frozenset(),
) -> tuple[frozenset[str], frozenset[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def walk(directory_fd: int, prefix: tuple[str, ...]) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as error:
            raise StagingVerificationError("CLOSED_LAYOUT_UNSAFE") from error
        for name in names:
            relative_parts = (*prefix, name)
            relative = PurePosixPath(*relative_parts).as_posix()
            try:
                metadata = source_capsule._lstat_at(
                    directory_fd,
                    name,
                    path=relative,
                    check="closed_layout",
                )
            except source_capsule.VerificationError as error:
                raise StagingVerificationError("CLOSED_LAYOUT_UNSAFE") from error
            if stat.S_ISLNK(metadata.st_mode):
                _fail("CLOSED_LAYOUT_UNSAFE")
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                if relative in skip_directories:
                    continue
                try:
                    child_fd = source_capsule._open_directory_at(
                        directory_fd,
                        name,
                        path=relative,
                        check="closed_layout",
                    )
                    identity = source_capsule.DirectoryIdentity.from_stat(os.fstat(child_fd))
                    try:
                        walk(child_fd, relative_parts)
                        source_capsule._verify_directory_binding(
                            directory_fd,
                            name,
                            child_fd,
                            identity,
                            path=relative,
                            check="closed_layout",
                        )
                    finally:
                        os.close(child_fd)
                except source_capsule.VerificationError as error:
                    raise StagingVerificationError("CLOSED_LAYOUT_CHANGED") from error
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    _fail("CLOSED_LAYOUT_HARDLINK")
                files.add(relative)
            else:
                _fail("CLOSED_LAYOUT_UNSAFE")

    descriptor = os.dup(root_fd)
    try:
        walk(descriptor, ())
    finally:
        os.close(descriptor)
    return frozenset(files), frozenset(directories)


def _required_directories(
    files: set[str],
    *,
    explicit: set[str] | None = None,
) -> frozenset[str]:
    directories = set() if explicit is None else set(explicit)
    for value in files:
        parts = PurePosixPath(value).parts
        for end in range(1, len(parts)):
            directories.add(PurePosixPath(*parts[:end]).as_posix())
    return frozenset(directories)


def _verify_closed_layout(
    bounds: Sequence[BoundDirectory],
    *,
    entries: Sequence[source_capsule.ManifestEntry],
    input_expectations: Mapping[str, FileSnapshot],
    source_manifest_sha256: str,
) -> ClosedLayoutSummary:
    root_bound, workspace, project, sibling, logs = bounds
    expected_root_names = {"Tanager", LOG_RELATIVE_PATH.name}
    expected_workspace_names = {"tanager-rocks", "tanager-spec"}
    try:
        if set(os.listdir(root_bound.descriptor)) != expected_root_names:
            _fail("CLOSED_LAYOUT_MISMATCH")
        if set(os.listdir(workspace.descriptor)) != expected_workspace_names:
            _fail("CLOSED_LAYOUT_MISMATCH")
        if os.listdir(logs.descriptor):
            _fail("E6_LOG_CONTAMINATION")
    except OSError as error:
        raise StagingVerificationError("CLOSED_LAYOUT_UNSAFE") from error

    project_source = {entry.path for entry in entries if entry.root == "project"}
    sibling_source = {
        PurePosixPath(*entry.parts).as_posix() for entry in entries if entry.root == "sibling"
    }
    project_files = (
        project_source
        | set(PROJECT_SUPPORT_FILES)
        | set(input_expectations)
        | {SOURCE_MANIFEST_RELATIVE_PATH.as_posix()}
    )
    sibling_files = sibling_source | set(SIBLING_SUPPORT_FILES)
    project_special_dirs = set(OPAQUE_PROJECT_DIRECTORIES) | {
        SPECTRAL_TREE_RELATIVE_PATH.as_posix()
    }
    expected_project_dirs = _required_directories(
        project_files,
        explicit=project_special_dirs,
    )
    expected_sibling_dirs = _required_directories(sibling_files)
    observed_project_files, observed_project_dirs = _collect_inventory(
        project.descriptor,
        skip_directories=frozenset(project_special_dirs),
    )
    observed_sibling_files, observed_sibling_dirs = _collect_inventory(sibling.descriptor)
    if (
        observed_project_files != project_files
        or observed_project_dirs != expected_project_dirs
        or observed_sibling_files != sibling_files
        or observed_sibling_dirs != expected_sibling_dirs
    ):
        _fail("CLOSED_LAYOUT_MISMATCH")

    support_records: list[dict[str, Any]] = [
        {
            "root": "project",
            "path": SOURCE_MANIFEST_RELATIVE_PATH.as_posix(),
            "sha256": source_manifest_sha256,
        }
    ]
    for root_name, root_fd, members in (
        ("project", project.descriptor, PROJECT_SUPPORT_FILES),
        ("sibling", sibling.descriptor, SIBLING_SUPPORT_FILES),
    ):
        for member in sorted(members):
            snapshot = _snapshot_single_link_file(
                root_fd,
                _safe_relative_path(member, code="CLOSED_LAYOUT_MISMATCH"),
                display=f"{root_name}:{member}",
                unsafe_code="CLOSED_LAYOUT_UNSAFE",
                hardlink_code="CLOSED_LAYOUT_HARDLINK",
                changed_code="CLOSED_LAYOUT_CHANGED",
            )
            support_records.append(
                {
                    "root": root_name,
                    "path": member,
                    "sha256": snapshot.sha256,
                    "size_bytes": snapshot.size_bytes,
                }
            )
    return ClosedLayoutSummary(
        project_directory_count=len(observed_project_dirs),
        project_file_count=len(observed_project_files),
        sibling_directory_count=len(observed_sibling_dirs),
        sibling_file_count=len(observed_sibling_files),
        support_files_sha256=hashlib.sha256(canonical_json_bytes(support_records)).hexdigest(),
    )


def _snapshot_spectral_tree(project_fd: int) -> tuple[list[dict[str, Any]], str]:
    descriptors = [os.dup(project_fd)]
    bindings: list[tuple[int, str, int, source_capsule.DirectoryIdentity]] = []
    try:
        current = descriptors[0]
        for part in SPECTRAL_TREE_RELATIVE_PATH.parts:
            following = source_capsule._open_directory_at(
                current,
                part,
                path=SPECTRAL_TREE_RELATIVE_PATH.as_posix(),
                check="spectral_tree",
            )
            identity = source_capsule.DirectoryIdentity.from_stat(os.fstat(following))
            bindings.append((current, part, following, identity))
            descriptors.append(following)
            current = following
        files, _directories = _collect_inventory(current)
        if not files:
            _fail("INPUT_TREE_INVALID")
        tree_digest = hashlib.sha256()
        records: list[dict[str, Any]] = []
        for relative in sorted(files):
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(b"\0")
            snapshot = _snapshot_single_link_file(
                current,
                _safe_relative_path(relative, code="INPUT_TREE_INVALID"),
                display=f"spectral:{relative}",
                unsafe_code="INPUT_FILE_UNSAFE",
                hardlink_code="INPUT_FILE_HARDLINK",
                changed_code="INPUT_FILE_CHANGED",
                content_digest=tree_digest,
            )
            records.append(
                {
                    "path": relative,
                    "sha256": snapshot.sha256,
                    "size_bytes": snapshot.size_bytes,
                }
            )
        for parent_fd, component, descriptor, identity in reversed(bindings):
            source_capsule._verify_directory_binding(
                parent_fd,
                component,
                descriptor,
                identity,
                path=SPECTRAL_TREE_RELATIVE_PATH.as_posix(),
                check="spectral_tree",
            )
        return records, tree_digest.hexdigest()
    except StagingVerificationError:
        raise
    except (OSError, source_capsule.VerificationError) as error:
        raise StagingVerificationError("INPUT_FILE_UNSAFE") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _archive_tree_identity(payload: bytes) -> tuple[int, str]:
    """Derive the pinned library closure from captured archive bytes only."""
    expected_root = SPECTRAL_TREE_RELATIVE_PATH.name
    seen: set[str] = set()
    digest = hashlib.sha256()
    count = 0
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = sorted(
                (info for info in archive.infolist() if not info.is_dir()),
                key=lambda info: info.filename,
            )
            for info in members:
                member = PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or len(member.parts) < 2
                    or member.parts[0] != expected_root
                    or any(part in {"", ".", ".."} for part in member.parts)
                ):
                    _fail("INPUT_ARCHIVE_INVALID")
                file_type = (info.external_attr >> 16) & 0o170000
                if file_type and not stat.S_ISREG(file_type):
                    _fail("INPUT_ARCHIVE_INVALID")
                relative = PurePosixPath(*member.parts[1:]).as_posix()
                if relative in seen:
                    _fail("INPUT_ARCHIVE_INVALID")
                seen.add(relative)
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                with archive.open(info) as handle:
                    for chunk in iter(lambda: handle.read(source_capsule.READ_SIZE), b""):
                        digest.update(chunk)
                count += 1
    except StagingVerificationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise StagingVerificationError("INPUT_ARCHIVE_INVALID") from error
    if count == 0:
        _fail("INPUT_ARCHIVE_INVALID")
    return count, digest.hexdigest()


def _snapshot_input_closure(
    project_fd: int,
    expectations: Mapping[str, FileSnapshot],
) -> InputClosureSnapshot:
    records: list[dict[str, Any]] = []
    archive_payload_chunks: list[bytes] = []
    for relative, expected in sorted(expectations.items()):
        snapshot = _snapshot_single_link_file(
            project_fd,
            _safe_relative_path(relative, code="INPUT_MANIFEST_INVALID"),
            display=f"input:{relative}",
            unsafe_code="INPUT_FILE_UNSAFE",
            hardlink_code="INPUT_FILE_HARDLINK",
            changed_code="INPUT_FILE_CHANGED",
            payload_chunks=(
                archive_payload_chunks
                if relative == SPECTRAL_ARCHIVE_RELATIVE_PATH.as_posix()
                else None
            ),
        )
        if snapshot != expected:
            _fail("INPUT_FILE_MISMATCH")
        records.append(
            {
                "path": relative,
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            }
        )
    spectral_records, spectral_tree_sha256 = _snapshot_spectral_tree(project_fd)
    archive_member_count, archive_tree_sha256 = _archive_tree_identity(
        b"".join(archive_payload_chunks)
    )
    if archive_member_count != len(spectral_records) or archive_tree_sha256 != spectral_tree_sha256:
        _fail("INPUT_ARCHIVE_TREE_MISMATCH")
    aggregate = {
        "files": records,
        "spectral_library": spectral_records,
    }
    return InputClosureSnapshot(
        file_count=len(records) + len(spectral_records),
        sha256=hashlib.sha256(canonical_json_bytes(aggregate)).hexdigest(),
        spectral_library_member_count=len(spectral_records),
        spectral_library_tree_sha256=spectral_tree_sha256,
    )


def _input_paths(project_root: Path) -> InputClosurePaths:
    return InputClosurePaths(
        raw_dir=project_root / "data/raw",
        speclib_dir=project_root / "data/speclib/ASCIIdata_splib07a",
        validation_dir=project_root / "data/intermediate/validation",
        output_dir=project_root / OUTPUT_RELATIVE_PATH,
        reference_dir=project_root / "data/reference",
    )


def independent_input_admitter(context: InputAdmissionContext) -> Mapping[str, Any]:
    """Reproduce the frozen identity contract without executing staged code."""
    raw_scenes = [
        {"sha256": expected.sha256}
        for path, expected in sorted(context.expectations.items())
        if path.startswith("data/raw/")
    ]
    archive = context.expectations.get(SPECTRAL_ARCHIVE_RELATIVE_PATH.as_posix())
    if archive is None:
        _fail("INPUT_MANIFEST_INVALID")
    return {
        "input_manifest_sha256": context.input_manifest_sha256,
        "raw_scenes": raw_scenes,
        "spectral_library": {
            "archive_sha256": archive.sha256,
            "expected_tree_sha256": context.snapshot.spectral_library_tree_sha256,
            "file_count": context.snapshot.spectral_library_member_count,
            "tree_sha256": context.snapshot.spectral_library_tree_sha256,
        },
    }


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("INPUT_ADMISSION_MALFORMED")
    return value


def _sanitize_input_admission(
    admission: Mapping[str, Any],
    *,
    observed_input_manifest_sha256: str,
) -> InputAdmissionSummary:
    input_manifest_sha256 = admission.get("input_manifest_sha256")
    if not isinstance(input_manifest_sha256, str):
        _fail("INPUT_ADMISSION_MALFORMED")
    _validate_sha256(input_manifest_sha256, code="INPUT_ADMISSION_MALFORMED")
    if input_manifest_sha256 != observed_input_manifest_sha256:
        _fail("INPUT_MANIFEST_DRIFT")

    raw_scenes = admission.get("raw_scenes")
    if not isinstance(raw_scenes, list) or len(raw_scenes) != EXPECTED_RAW_SCENE_COUNT:
        _fail("INPUT_ADMISSION_MALFORMED")
    raw_hashes: list[str] = []
    for row in raw_scenes:
        record = _require_mapping(row)
        digest = record.get("sha256")
        if not isinstance(digest, str):
            _fail("INPUT_ADMISSION_MALFORMED")
        _validate_sha256(digest, code="INPUT_ADMISSION_MALFORMED")
        raw_hashes.append(digest)

    spectral = _require_mapping(admission.get("spectral_library"))
    archive_sha256 = spectral.get("archive_sha256")
    tree_sha256 = spectral.get("tree_sha256")
    expected_tree_sha256 = spectral.get("expected_tree_sha256")
    file_count = spectral.get("file_count")
    for digest in (archive_sha256, tree_sha256, expected_tree_sha256):
        if not isinstance(digest, str):
            _fail("INPUT_ADMISSION_MALFORMED")
        _validate_sha256(digest, code="INPUT_ADMISSION_MALFORMED")
    if tree_sha256 != expected_tree_sha256:
        _fail("INPUT_ADMISSION_MALFORMED")
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count <= 0:
        _fail("INPUT_ADMISSION_MALFORMED")

    raw_hashes_bytes = canonical_json_bytes(raw_hashes)
    return InputAdmissionSummary(
        input_manifest_sha256=input_manifest_sha256,
        raw_scene_count=len(raw_hashes),
        raw_scene_hashes_sha256=hashlib.sha256(raw_hashes_bytes).hexdigest(),
        spectral_archive_sha256=archive_sha256,
        spectral_library_member_count=file_count,
        spectral_library_tree_sha256=tree_sha256,
    )


def _execution_lock_path(output_dir: Path) -> Path:
    resolved_output = output_dir.resolve()
    identity = f"{SCIENTIFIC_EXECUTION_IDENTITY}\0{resolved_output}".encode()
    suffix = hashlib.sha256(identity).hexdigest()[:16]
    return resolved_output.parent / f".repeatability-{suffix}.lock"


def _verify_absence_and_logs(root: Path, project_root: Path, logs_fd: int) -> None:
    output_dir = project_root / OUTPUT_RELATIVE_PATH
    python_lock = _execution_lock_path(output_dir)
    checks = (
        (output_dir, "REPEATABILITY_OUTPUT_EXISTS"),
        (python_lock, "PYTHON_LOCK_EXISTS"),
        (root / WRAPPER_RUNTIME_RELATIVE_PATH, "WRAPPER_RUNTIME_EXISTS"),
        (root / "runtime", "E6_RUNTIME_CONTAMINATION"),
        (root / CACHE_RELATIVE_PATH, "E6_CACHE_CONTAMINATION"),
    )
    for path, code in checks:
        if _lexists(path):
            _fail(code)
    try:
        if os.listdir(logs_fd):
            _fail("E6_LOG_CONTAMINATION")
    except OSError as error:
        raise StagingVerificationError("ROOT_OR_LAYOUT_UNSAFE") from error


def build_timing_argv(root: Path, source_manifest_sha256: str) -> tuple[str, ...]:
    """Build the proposal-bound timing command without executing it.

    Parameters
    ----------
    root
        Exact dedicated repeatability root.
    source_manifest_sha256
        Verified frozen source-manifest digest.

    Returns
    -------
    tuple[str, ...]
        Exact ``sbatch`` argument vector specified by the proposal.
    """
    project_root = root / PROJECT_RELATIVE_PATH
    log_root = root / LOG_RELATIVE_PATH
    exports = ",".join(
        (
            "ALL",
            f"TANAGER_BIGMEM_ROOT={root}",
            "TANAGER_MODE=timing",
            f"TANAGER_SOURCE_MANIFEST_SHA256={source_manifest_sha256}",
        )
    )
    return (
        "sbatch",
        f"--chdir={project_root}",
        f"--output={log_root}/%x-%j.out",
        f"--error={log_root}/%x-%j.err",
        f"--export={exports}",
        os.fspath(WRAPPER_RELATIVE_PATH),
    )


def verify_timing_argv(
    argv: Sequence[str],
    *,
    root: Path,
    source_manifest_sha256: str,
) -> tuple[str, ...]:
    """Verify exact root, log, export, wrapper, and argument-order agreement.

    Parameters
    ----------
    argv
        Candidate timing command argument vector.
    root
        Exact dedicated repeatability root.
    source_manifest_sha256
        Verified frozen source-manifest digest.

    Returns
    -------
    tuple[str, ...]
        The validated immutable argument vector.
    """
    candidate = tuple(argv)
    expected = build_timing_argv(root, source_manifest_sha256)
    if candidate != expected:
        _fail("TIMING_COMMAND_DISAGREEMENT")

    export_prefix = "--export="
    export_value = candidate[4].removeprefix(export_prefix)
    expected_exports = {
        "ALL": None,
        "TANAGER_BIGMEM_ROOT": os.fspath(root),
        "TANAGER_MODE": "timing",
        "TANAGER_SOURCE_MANIFEST_SHA256": source_manifest_sha256,
    }
    parsed_exports: dict[str, str | None] = {}
    for item in export_value.split(","):
        key, separator, value = item.partition("=")
        if not key or key in parsed_exports:
            _fail("TIMING_COMMAND_DISAGREEMENT")
        parsed_exports[key] = value if separator else None
    if parsed_exports != expected_exports:
        _fail("TIMING_COMMAND_DISAGREEMENT")
    return candidate


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically with no non-finite values.

    Parameters
    ----------
    value
        JSON-serializable value.

    Returns
    -------
    bytes
        Canonical UTF-8 JSON terminated by one newline.
    """
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _open_evidence_parent(
    path: Path,
) -> tuple[int, str, int, source_capsule.DirectoryIdentity]:
    try:
        grandparent_fd, parent_name = source_capsule._open_parent(
            path,
            check="evidence_output_parent",
        )
        try:
            parent_fd = source_capsule._open_directory_at(
                grandparent_fd,
                parent_name,
                path=os.fspath(path),
                check="evidence_output_parent",
            )
        except BaseException:
            os.close(grandparent_fd)
            raise
    except source_capsule.VerificationError as error:
        raise StagingVerificationError("EVIDENCE_OUTPUT_PARENT_UNSAFE") from error
    return (
        grandparent_fd,
        parent_name,
        parent_fd,
        source_capsule.DirectoryIdentity.from_stat(os.fstat(parent_fd)),
    )


def write_evidence_exclusive(path: Path, evidence: Mapping[str, Any]) -> None:
    """Create canonical evidence exclusively without following links.

    Parameters
    ----------
    path
        New evidence path whose parent already exists.
    evidence
        Complete endpoint-blind PASS record.
    """
    if not path.name or path.name in {".", ".."}:
        _fail("EVIDENCE_OUTPUT_INVALID")
    grandparent_fd, parent_name, parent_fd, parent_identity = _open_evidence_parent(path.parent)
    descriptor: int | None = None
    expected_bytes = canonical_json_bytes(evidence)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(expected_bytes)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("EVIDENCE_WRITE_FAILED")
            view = view[written:]
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            _fail("EVIDENCE_OUTPUT_UNSAFE")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        if b"".join(chunks) != expected_bytes:
            _fail("EVIDENCE_WRITE_FAILED")
        rebound = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (rebound.st_dev, rebound.st_ino, rebound.st_mode, rebound.st_nlink) != (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_nlink,
        ):
            _fail("EVIDENCE_OUTPUT_CHANGED")
        try:
            observed_parent = os.fstat(parent_fd)
            rebound_parent = source_capsule._lstat_at(
                grandparent_fd,
                parent_name,
                path=os.fspath(path.parent),
                check="evidence_output_parent",
            )
        except (OSError, source_capsule.VerificationError) as error:
            raise StagingVerificationError("EVIDENCE_OUTPUT_PARENT_CHANGED") from error
        expected_parent_binding = (
            parent_identity.device,
            parent_identity.inode,
            parent_identity.mode,
        )
        if (
            stat.S_ISLNK(rebound_parent.st_mode)
            or not stat.S_ISDIR(rebound_parent.st_mode)
            or (
                observed_parent.st_dev,
                observed_parent.st_ino,
                observed_parent.st_mode,
            )
            != expected_parent_binding
            or (
                rebound_parent.st_dev,
                rebound_parent.st_ino,
                rebound_parent.st_mode,
            )
            != expected_parent_binding
        ):
            _fail("EVIDENCE_OUTPUT_PARENT_CHANGED")
        os.fsync(parent_fd)
    except FileExistsError as error:
        raise StagingVerificationError("EVIDENCE_OUTPUT_EXISTS") from error
    except StagingVerificationError:
        raise
    except OSError as error:
        raise StagingVerificationError("EVIDENCE_WRITE_FAILED") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
        os.close(grandparent_fd)


def _verifier_sha256() -> str:
    _payload, digest = _read_control(Path(__file__).absolute(), code="VERIFIER_IDENTITY_UNSAFE")
    return digest


def _source_capsule_verifier_sha256() -> str:
    _payload, digest = _read_control(
        SCRIPT_DIR / "verify_source_capsule.py",
        code="VERIFIER_IDENTITY_UNSAFE",
    )
    return digest


def verify_repeatability_staging(
    config: StagingConfig,
    *,
    input_admitter: InputAdmitter = independent_input_admitter,
    timing_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify one staged root and create endpoint-blind PASS evidence.

    Parameters
    ----------
    config
        Explicit roots, detached identities, and exclusive evidence path.
    input_admitter
        Identity-only admission adapter over descriptor-bound facts; injectable
        for endpoint-free fault tests. Staged Python code is never imported or
        executed.
    timing_argv
        Optional candidate command vector for mechanical disagreement tests.

    Returns
    -------
    dict[str, Any]
        The exact PASS record written to ``config.evidence_output``.

    Raises
    ------
    StagingVerificationError
        If any control, filesystem, source, input, residue, or command gate fails.
    """
    _validate_sha256(config.expected_proposal_sha256, code="ARGUMENT_INVALID")
    _validate_sha256(config.expected_source_manifest_sha256, code="ARGUMENT_INVALID")
    if config.expected_source_member_count <= 0:
        _fail("ARGUMENT_INVALID")

    actual_root = _canonical_absolute(config.actual_root, code="ACTUAL_ROOT_INVALID")
    expected_root = _canonical_absolute(config.expected_root, code="EXPECTED_ROOT_INVALID")
    e6_root = _canonical_absolute(config.e6_root, code="E6_ROOT_INVALID")
    proposal = _canonical_absolute(config.proposal, code="PROPOSAL_PATH_INVALID")
    evidence_output = _canonical_absolute(
        config.evidence_output,
        code="EVIDENCE_OUTPUT_INVALID",
    )
    if actual_root != expected_root:
        _fail("ROOT_IDENTITY_MISMATCH")
    if _paths_overlap(actual_root, e6_root):
        _fail("ROOT_E6_OVERLAP")
    if _paths_overlap(evidence_output, actual_root) or _paths_overlap(evidence_output, e6_root):
        _fail("EVIDENCE_OUTPUT_SCOPE")

    project_root = actual_root / PROJECT_RELATIVE_PATH
    sibling_root = actual_root / SIBLING_RELATIVE_PATH
    expected_manifest_path = project_root / SOURCE_MANIFEST_RELATIVE_PATH
    if config.source_manifest != expected_manifest_path:
        _fail("SOURCE_MANIFEST_PATH_MISMATCH")

    proposal_payload, proposal_sha256 = _read_control(
        proposal,
        code="PROPOSAL_UNSAFE",
    )
    del proposal_payload
    if proposal_sha256 != config.expected_proposal_sha256:
        _fail("PROPOSAL_DRIFT")

    parent_fd, bounds = _open_staging_layout(actual_root)
    try:
        root_bound, _workspace, project_bound, sibling_bound, logs_bound = bounds
        _verify_absence_and_logs(actual_root, project_root, logs_bound.descriptor)
        source_manifest_sha256, source_member_count, entries = _verify_source_capsule(
            config,
            project_fd=project_bound.descriptor,
            sibling_fd=sibling_bound.descriptor,
        )

        input_manifest_path = project_root / INPUT_MANIFEST_RELATIVE_PATH
        input_payload, observed_input_manifest_sha256 = _read_control(
            input_manifest_path,
            code="INPUT_MANIFEST_UNSAFE",
        )
        input_expectations = _parse_input_expectations(input_payload)
        layout_before = _verify_closed_layout(
            bounds,
            entries=entries,
            input_expectations=input_expectations,
            source_manifest_sha256=source_manifest_sha256,
        )
        input_snapshot_before = _snapshot_input_closure(
            project_bound.descriptor,
            input_expectations,
        )
        try:
            raw_admission = input_admitter(
                InputAdmissionContext(
                    paths=_input_paths(project_root),
                    root=project_root,
                    input_manifest_path=input_manifest_path,
                    input_manifest_sha256=observed_input_manifest_sha256,
                    expectations=input_expectations,
                    snapshot=input_snapshot_before,
                )
            )
        except StagingVerificationError:
            raise
        except Exception as error:
            raise StagingVerificationError("INPUT_ADMISSION_FAILED") from error
        input_summary = _sanitize_input_admission(
            _require_mapping(raw_admission),
            observed_input_manifest_sha256=observed_input_manifest_sha256,
        )
        raw_hashes = [
            expected.sha256
            for path, expected in sorted(input_expectations.items())
            if path.startswith("data/raw/")
        ]
        expected_raw_hashes_sha256 = hashlib.sha256(canonical_json_bytes(raw_hashes)).hexdigest()
        archive_expected = input_expectations[SPECTRAL_ARCHIVE_RELATIVE_PATH.as_posix()]
        if (
            expected_raw_hashes_sha256 != input_summary.raw_scene_hashes_sha256
            or archive_expected.sha256 != input_summary.spectral_archive_sha256
            or input_snapshot_before.spectral_library_member_count
            != input_summary.spectral_library_member_count
            or input_snapshot_before.spectral_library_tree_sha256
            != input_summary.spectral_library_tree_sha256
        ):
            _fail("INPUT_ADMISSION_DISAGREEMENT")

        candidate_argv = (
            build_timing_argv(actual_root, source_manifest_sha256)
            if timing_argv is None
            else tuple(timing_argv)
        )
        command = verify_timing_argv(
            candidate_argv,
            root=actual_root,
            source_manifest_sha256=source_manifest_sha256,
        )
        command_sha256 = hashlib.sha256(canonical_json_bytes(list(command))).hexdigest()

        _verify_absence_and_logs(actual_root, project_root, logs_bound.descriptor)
        _source_sha256_after, _source_count_after, entries_after = _verify_source_capsule(
            config,
            project_fd=project_bound.descriptor,
            sibling_fd=sibling_bound.descriptor,
        )
        layout_after = _verify_closed_layout(
            bounds,
            entries=entries_after,
            input_expectations=input_expectations,
            source_manifest_sha256=source_manifest_sha256,
        )
        input_snapshot_after = _snapshot_input_closure(
            project_bound.descriptor,
            input_expectations,
        )
        if layout_after != layout_before:
            _fail("CLOSED_LAYOUT_CHANGED")
        if input_snapshot_after != input_snapshot_before:
            _fail("INPUT_CLOSURE_CHANGED")
        for bound in reversed(bounds):
            _verify_bound_directory(bound)

        root_identity = source_capsule.DirectoryIdentity.from_stat(os.fstat(root_bound.descriptor))
        evidence: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "authority": "staged_root_verification_only",
            "closed_checks": [
                "e6_disjoint",
                "input_closure",
                "input_descriptor_snapshot_rechecked",
                "layout_closed_world",
                "python_lock_absent",
                "repeatability_output_absent",
                "root_identity",
                "slurm_logs_empty",
                "source_capsule",
                "source_members_single_link",
                "timing_command_bound_not_executed",
                "uv_cache_absent",
                "wrapper_runtime_absent",
            ],
            "closed_layout": {
                "opaque_project_directories": sorted(OPAQUE_PROJECT_DIRECTORIES),
                "project_directory_count": layout_before.project_directory_count,
                "project_file_count": layout_before.project_file_count,
                "sibling_directory_count": layout_before.sibling_directory_count,
                "sibling_file_count": layout_before.sibling_file_count,
                "support_files_sha256": layout_before.support_files_sha256,
            },
            "input_closure": {
                "status": "PASS",
                "descriptor_file_count": input_snapshot_before.file_count,
                "descriptor_snapshot_sha256": input_snapshot_before.sha256,
                "input_manifest_sha256": input_summary.input_manifest_sha256,
                "raw_scene_count": input_summary.raw_scene_count,
                "raw_scene_hashes_sha256": input_summary.raw_scene_hashes_sha256,
                "spectral_archive_sha256": input_summary.spectral_archive_sha256,
                "spectral_library_member_count": (input_summary.spectral_library_member_count),
                "spectral_library_tree_sha256": (input_summary.spectral_library_tree_sha256),
            },
            "proposal": {"sha256": proposal_sha256},
            "root_identity": {
                "actual_path": os.fspath(actual_root),
                "device": root_identity.device,
                "expected_path": os.fspath(expected_root),
                "inode": root_identity.inode,
                "repository_path": os.fspath(project_root),
                "sibling_path": os.fspath(sibling_root),
            },
            "source_manifest": {
                "member_count": source_member_count,
                "sha256": source_manifest_sha256,
            },
            "timing_command": {
                "argv": list(command),
                "executed": False,
                "sha256": command_sha256,
            },
            "verifier": {
                "sha256": _verifier_sha256(),
                "source_capsule_sha256": _source_capsule_verifier_sha256(),
            },
        }
        write_evidence_exclusive(evidence_output, evidence)
        return evidence
    finally:
        _close_staging_layout(parent_fd, bounds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-root", type=Path, required=True)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--e6-root", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--expected-proposal-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-source-member-count", type=int, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone staged-root verifier CLI.

    Parameters
    ----------
    argv
        Optional command-line arguments excluding the program name.

    Returns
    -------
    int
        Zero for PASS and one for any closed failure.
    """
    args = _parser().parse_args(argv)
    config = StagingConfig(
        actual_root=args.actual_root,
        expected_root=args.expected_root,
        e6_root=args.e6_root,
        proposal=args.proposal,
        expected_proposal_sha256=args.expected_proposal_sha256,
        source_manifest=args.source_manifest,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_source_member_count=args.expected_source_member_count,
        evidence_output=args.evidence_output,
    )
    try:
        evidence = verify_repeatability_staging(config)
    except StagingVerificationError as error:
        print(error.render(), file=sys.stderr)
        return 1
    except Exception:
        print("FAIL code=INTERNAL_ERROR", file=sys.stderr)
        return 1
    print(
        "PASS check=repeatability_staging "
        f"evidence={json.dumps(os.fspath(args.evidence_output), ensure_ascii=True)} "
        f"source_member_count={evidence['source_manifest']['member_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
