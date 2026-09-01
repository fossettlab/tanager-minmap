#!/usr/bin/env python3
"""Verify a bounded E6 v2 source capsule without following filesystem links."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_LINE_RE = re.compile(r"^(?P<sha256>[0-9a-f]{64})  (?P<path>[^\x00-\x1f\x7f]+)$")
SIBLING_PREFIX = "../tanager-spec/"
READ_SIZE = 1024 * 1024


class VerificationError(Exception):
    """A deliberately low-disclosure capsule verification failure."""

    def __init__(self, *, path: str, check: str, reason: str) -> None:
        super().__init__(reason)
        self.path = path
        self.check = check
        self.reason = reason

    def render(self) -> str:
        """Render only the bounded diagnostic fields."""
        return (
            f"FAIL path={json.dumps(self.path, ensure_ascii=True)} "
            f"check={self.check} reason={self.reason}"
        )


@dataclass(frozen=True)
class FileIdentity:
    """Metadata used to detect drift or pathname replacement during a read."""

    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> FileIdentity:
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            links=metadata.st_nlink,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
            changed_ns=metadata.st_ctime_ns,
        )


@dataclass(frozen=True)
class DirectoryIdentity:
    """Stable identity for a descriptor-bound directory component."""

    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> DirectoryIdentity:
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            links=metadata.st_nlink,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
            changed_ns=metadata.st_ctime_ns,
        )


@dataclass(frozen=True)
class ManifestEntry:
    """One validated canonical manifest entry."""

    sha256: str
    path: str
    root: str
    parts: tuple[str, ...]


AfterReadHook = Callable[[str], None]


def _flags(*, directory: bool) -> int:
    required = ("O_NOFOLLOW", "O_DIRECTORY")
    if any(not hasattr(os, name) for name in required):
        raise VerificationError(path=".", check="platform", reason="nofollow_unsupported")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= os.O_DIRECTORY
    return flags


def _lstat_at(parent_fd: int, name: str, *, path: str, check: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise VerificationError(path=path, check=check, reason="path_unavailable") from error


def _open_directory_at(parent_fd: int, name: str, *, path: str, check: str) -> int:
    before = _lstat_at(parent_fd, name, path=path, check=check)
    if stat.S_ISLNK(before.st_mode):
        raise VerificationError(path=path, check=check, reason="symlink_component")
    if not stat.S_ISDIR(before.st_mode):
        raise VerificationError(path=path, check=check, reason="non_directory_component")
    try:
        descriptor = os.open(name, _flags(directory=True), dir_fd=parent_fd)
    except OSError as error:
        raise VerificationError(path=path, check=check, reason="component_open_failed") from error
    if DirectoryIdentity.from_stat(os.fstat(descriptor)) != DirectoryIdentity.from_stat(before):
        os.close(descriptor)
        raise VerificationError(path=path, check=check, reason="component_changed")
    return descriptor


def _absolute_parts(path: Path, *, check: str) -> tuple[str, ...]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if not parts or parts[0] != os.sep:
        raise VerificationError(path=str(path), check=check, reason="invalid_control_path")
    return tuple(part for part in parts[1:] if part)


def _open_directory_path(path: Path, *, check: str) -> int:
    display = str(path)
    parts = _absolute_parts(path, check=check)
    try:
        current = os.open(os.sep, _flags(directory=True))
    except OSError as error:
        raise VerificationError(path=display, check=check, reason="root_open_failed") from error
    try:
        for index, part in enumerate(parts):
            component_path = os.sep + os.path.join(*parts[: index + 1])
            following = _open_directory_at(
                current,
                part,
                path=component_path,
                check=check,
            )
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _open_parent(path: Path, *, check: str) -> tuple[int, str]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute == Path(os.sep) or not absolute.name:
        raise VerificationError(path=str(path), check=check, reason="invalid_control_path")
    return _open_directory_path(absolute.parent, check=check), absolute.name


def _verify_directory_binding(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: DirectoryIdentity,
    *,
    path: str,
    check: str,
) -> None:
    observed = DirectoryIdentity.from_stat(os.fstat(descriptor))
    rebound = _lstat_at(parent_fd, name, path=path, check=check)
    if stat.S_ISLNK(rebound.st_mode):
        raise VerificationError(path=path, check=check, reason="symlink_component")
    if observed != expected or DirectoryIdentity.from_stat(rebound) != expected:
        raise VerificationError(path=path, check=check, reason="component_changed")


def _read_regular_at(
    parent_fd: int,
    name: str,
    *,
    path: str,
    check: str,
    after_read_hook: AfterReadHook | None,
    capture_payload: bool,
) -> tuple[bytes, str]:
    before = _lstat_at(parent_fd, name, path=path, check=check)
    if stat.S_ISLNK(before.st_mode):
        raise VerificationError(path=path, check=check, reason="symlink_final")
    if not stat.S_ISREG(before.st_mode):
        raise VerificationError(path=path, check=check, reason="non_regular_final")
    expected = FileIdentity.from_stat(before)
    try:
        descriptor = os.open(name, _flags(directory=False), dir_fd=parent_fd)
    except OSError as error:
        raise VerificationError(path=path, check=check, reason="file_open_failed") from error
    try:
        if FileIdentity.from_stat(os.fstat(descriptor)) != expected:
            raise VerificationError(path=path, check=check, reason="file_changed")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, READ_SIZE):
            digest.update(chunk)
            if capture_payload:
                chunks.append(chunk)
        if after_read_hook is not None:
            after_read_hook(path)
        rebound = _lstat_at(parent_fd, name, path=path, check=check)
        if (
            FileIdentity.from_stat(os.fstat(descriptor)) != expected
            or FileIdentity.from_stat(rebound) != expected
        ):
            raise VerificationError(path=path, check=check, reason="file_changed")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_control_file(
    path: Path,
    *,
    check: str,
    after_read_hook: AfterReadHook | None,
) -> tuple[bytes, str]:
    parent_fd, name = _open_parent(path, check=check)
    try:
        return _read_regular_at(
            parent_fd,
            name,
            path=str(path),
            check=check,
            after_read_hook=after_read_hook,
            capture_payload=True,
        )
    finally:
        os.close(parent_fd)


def _normalize_member_path(value: str) -> tuple[str, tuple[str, ...]]:
    if "\\" in value:
        raise VerificationError(path=value, check="member_path", reason="backslash_rejected")
    pure = PurePosixPath(value)
    if pure.is_absolute():
        raise VerificationError(path=value, check="member_path", reason="absolute_path_rejected")
    parts = pure.parts
    if value.startswith(SIBLING_PREFIX):
        member_parts = parts[2:]
        if (
            len(parts) < 3
            or parts[:2] != ("..", "tanager-spec")
            or any(part in {"", ".", ".."} for part in member_parts)
            or pure.as_posix() != value
        ):
            raise VerificationError(path=value, check="member_path", reason="path_not_normalized")
        return "sibling", tuple(member_parts)
    if not parts or any(part in {"", ".", ".."} for part in parts) or pure.as_posix() != value:
        reason = "parent_path_rejected" if ".." in parts else "path_not_normalized"
        raise VerificationError(path=value, check="member_path", reason=reason)
    return "project", tuple(parts)


def _parse_manifest(payload: bytes, *, expected_count: int) -> tuple[ManifestEntry, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(
            path="<manifest>", check="manifest_format", reason="invalid_utf8"
        ) from error
    if "\r" in text or (text and not text.endswith("\n")):
        raise VerificationError(
            path="<manifest>", check="manifest_format", reason="noncanonical_newline"
        )
    lines = text[:-1].split("\n") if text else []
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for line in lines:
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise VerificationError(
                path="<manifest>", check="manifest_format", reason="malformed_line"
            )
        member_path = match.group("path")
        root, parts = _normalize_member_path(member_path)
        if member_path in seen:
            raise VerificationError(
                path=member_path, check="manifest_entries", reason="duplicate_path"
            )
        seen.add(member_path)
        entries.append(
            ManifestEntry(
                sha256=match.group("sha256"),
                path=member_path,
                root=root,
                parts=parts,
            )
        )
    if len(entries) != expected_count:
        raise VerificationError(path="<manifest>", check="entry_count", reason="count_mismatch")
    paths = [entry.path for entry in entries]
    if paths != sorted(paths):
        raise VerificationError(path="<manifest>", check="entry_order", reason="not_sorted")
    return tuple(entries)


def _read_member(
    root_fd: int,
    entry: ManifestEntry,
    *,
    after_read_hook: AfterReadHook | None,
) -> str:
    current = os.dup(root_fd)
    bindings: list[tuple[int, str, int, DirectoryIdentity, str]] = []
    try:
        prefix: list[str] = []
        for part in entry.parts[:-1]:
            prefix.append(part)
            component_path = "/".join(prefix)
            following = _open_directory_at(
                current,
                part,
                path=entry.path,
                check="member_component",
            )
            identity = DirectoryIdentity.from_stat(os.fstat(following))
            bindings.append((current, part, following, identity, component_path))
            current = following
        _, digest = _read_regular_at(
            current,
            entry.parts[-1],
            path=entry.path,
            check="member_file",
            after_read_hook=after_read_hook,
            capture_payload=False,
        )
        for parent_fd, name, descriptor, identity, _component_path in reversed(bindings):
            _verify_directory_binding(
                parent_fd,
                name,
                descriptor,
                identity,
                path=entry.path,
                check="member_component",
            )
        return digest
    finally:
        for parent_fd, _name, _descriptor, _identity, _path in bindings:
            if parent_fd != root_fd:
                os.close(parent_fd)
        os.close(current)


def verify_source_capsule(
    *,
    manifest: Path,
    expected_manifest_sha256: str,
    expected_entry_count: int,
    project_root: Path,
    after_read_hook: AfterReadHook | None = None,
) -> tuple[str, int]:
    """Verify the detached manifest identity and every descriptor-bound member."""
    if SHA256_RE.fullmatch(expected_manifest_sha256) is None:
        raise VerificationError(
            path="<expected-manifest-sha256>",
            check="arguments",
            reason="invalid_sha256",
        )
    if expected_entry_count < 0:
        raise VerificationError(
            path="<expected-entry-count>", check="arguments", reason="negative_count"
        )

    payload, manifest_sha256 = _read_control_file(
        manifest,
        check="manifest_read",
        after_read_hook=after_read_hook,
    )
    if manifest_sha256 != expected_manifest_sha256:
        raise VerificationError(
            path=str(manifest), check="manifest_digest", reason="digest_mismatch"
        )
    entries = _parse_manifest(payload, expected_count=expected_entry_count)

    project_parent_fd, project_name = _open_parent(project_root, check="project_root")
    project_fd: int | None = None
    sibling_fd: int | None = None
    sibling_identity: DirectoryIdentity | None = None
    try:
        project_fd = _open_directory_at(
            project_parent_fd,
            project_name,
            path=str(project_root),
            check="project_root",
        )
        project_identity = DirectoryIdentity.from_stat(os.fstat(project_fd))
        if any(entry.root == "sibling" for entry in entries):
            sibling_fd = _open_directory_at(
                project_parent_fd,
                "tanager-spec",
                path=SIBLING_PREFIX.removesuffix("/"),
                check="sibling_root",
            )
            sibling_identity = DirectoryIdentity.from_stat(os.fstat(sibling_fd))
        for entry in entries:
            root_fd = project_fd if entry.root == "project" else sibling_fd
            if root_fd is None:  # pragma: no cover - guarded by sibling opening above.
                raise VerificationError(
                    path=entry.path, check="sibling_root", reason="path_unavailable"
                )
            observed = _read_member(root_fd, entry, after_read_hook=after_read_hook)
            if observed != entry.sha256:
                raise VerificationError(
                    path=entry.path, check="member_digest", reason="digest_mismatch"
                )
        if sibling_fd is not None and sibling_identity is not None:
            _verify_directory_binding(
                project_parent_fd,
                "tanager-spec",
                sibling_fd,
                sibling_identity,
                path=SIBLING_PREFIX.removesuffix("/"),
                check="sibling_root",
            )
        _verify_directory_binding(
            project_parent_fd,
            project_name,
            project_fd,
            project_identity,
            path=str(project_root),
            check="project_root",
        )
    finally:
        if sibling_fd is not None:
            os.close(sibling_fd)
        if project_fd is not None:
            os.close(project_fd)
        os.close(project_parent_fd)
    return manifest_sha256, len(entries)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-entry-count", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone verifier CLI."""
    args = _parser().parse_args(argv)
    try:
        manifest_sha256, entry_count = verify_source_capsule(
            manifest=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_entry_count=args.expected_entry_count,
            project_root=args.project_root,
        )
    except VerificationError as error:
        print(error.render(), file=sys.stderr)
        return 1
    print(f"PASS check=source_capsule manifest_sha256={manifest_sha256} entry_count={entry_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
