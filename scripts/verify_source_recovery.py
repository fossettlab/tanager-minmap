#!/usr/bin/env python3
"""Verify or install a bounded source-capsule recovery without following links."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_LINE_RE = re.compile(r"^(?P<sha256>[0-9a-f]{64})  (?P<path>[^\x00-\x1f\x7f]+)$")
SIBLING_PREFIX = "../tanager-spec/"
READ_SIZE = 1024 * 1024


class RecoveryError(Exception):
    """A deliberately low-disclosure recovery failure."""

    def __init__(self, *, path: str, check: str, reason: str) -> None:
        super().__init__(reason)
        self.path = path
        self.check = check
        self.reason = reason

    def render(self) -> str:
        """Render only bounded diagnostic fields."""
        return (
            f"FAIL path={json.dumps(self.path, ensure_ascii=True)} "
            f"check={self.check} reason={self.reason}"
        )


@dataclass(frozen=True)
class Identity:
    """Stable filesystem identity used for drift detection."""

    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> Identity:
        """Construct an identity from one stat result."""
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
    """Stable directory binding that permits expected child-file creation."""

    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> DirectoryIdentity:
        """Construct a directory identity from one stat result."""
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
        )


@dataclass(frozen=True)
class ManifestEntry:
    """One canonical recovery-manifest member."""

    sha256: str
    path: str
    namespace: str
    parts: tuple[str, ...]


@dataclass
class DirectoryChain:
    """An open descriptor chain from the filesystem root to one directory."""

    descriptors: list[int]
    names: list[str]
    identities: list[DirectoryIdentity]
    display: str

    @property
    def descriptor(self) -> int:
        """Return the final directory descriptor."""
        return self.descriptors[-1]

    def verify(self, *, check: str) -> None:
        """Verify every open component is still bound to its pathname."""
        for index, identity in enumerate(self.identities):
            descriptor = self.descriptors[index + 1]
            parent = self.descriptors[index]
            name = self.names[index]
            observed = DirectoryIdentity.from_stat(os.fstat(descriptor))
            rebound = _lstat_at(
                parent,
                name,
                path=self.display,
                check=check,
            )
            if (
                not stat.S_ISDIR(rebound.st_mode)
                or observed != identity
                or DirectoryIdentity.from_stat(rebound) != identity
            ):
                raise RecoveryError(
                    path=self.display,
                    check=check,
                    reason="directory_changed",
                )

    def close(self) -> None:
        """Close every descriptor in reverse order."""
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.descriptors.clear()


@dataclass
class SourceHandle:
    """One preflighted source file and its open parent chain."""

    entry: ManifestEntry
    parent: DirectoryChain
    descriptor: int
    identity: Identity

    def close(self) -> None:
        """Close the file and parent descriptors."""
        os.close(self.descriptor)
        self.parent.close()


@dataclass
class DestinationHandle:
    """One preflighted absent destination and its open parent chain."""

    entry: ManifestEntry
    parent: DirectoryChain

    def close(self) -> None:
        """Close the parent descriptors."""
        self.parent.close()


AfterSourceReadHook = Callable[[str], None]
BeforeDestinationCreateHook = Callable[[str], None]
AfterInstallHook = Callable[[str, int], None]


def _flags(*, directory: bool) -> int:
    required = ("O_NOFOLLOW", "O_DIRECTORY")
    if any(not hasattr(os, name) for name in required):
        raise RecoveryError(path=".", check="platform", reason="nofollow_unsupported")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= os.O_DIRECTORY
    return flags


def _lstat_at(parent_fd: int, name: str, *, path: str, check: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise RecoveryError(path=path, check=check, reason="path_unavailable") from error


def _absolute_parts(path: Path, *, check: str) -> tuple[str, ...]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if not parts or parts[0] != os.sep:
        raise RecoveryError(path=str(path), check=check, reason="invalid_control_path")
    return tuple(part for part in parts[1:] if part)


def _open_absolute_directory(path: Path, *, check: str) -> DirectoryChain:
    display = str(Path(os.path.abspath(os.fspath(path))))
    descriptors = [os.open(os.sep, _flags(directory=True))]
    names: list[str] = []
    identities: list[DirectoryIdentity] = []
    try:
        for part in _absolute_parts(path, check=check):
            parent = descriptors[-1]
            before = _lstat_at(parent, part, path=display, check=check)
            if stat.S_ISLNK(before.st_mode):
                raise RecoveryError(path=display, check=check, reason="symlink_component")
            if not stat.S_ISDIR(before.st_mode):
                raise RecoveryError(path=display, check=check, reason="non_directory_component")
            try:
                descriptor = os.open(part, _flags(directory=True), dir_fd=parent)
            except OSError as error:
                raise RecoveryError(
                    path=display,
                    check=check,
                    reason="component_open_failed",
                ) from error
            identity = DirectoryIdentity.from_stat(before)
            if DirectoryIdentity.from_stat(os.fstat(descriptor)) != identity:
                os.close(descriptor)
                raise RecoveryError(path=display, check=check, reason="component_changed")
            descriptors.append(descriptor)
            names.append(part)
            identities.append(identity)
        chain = DirectoryChain(descriptors, names, identities, display)
        chain.verify(check=check)
        return chain
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _open_relative_parent(
    root: DirectoryChain,
    parts: tuple[str, ...],
    *,
    path: str,
    check: str,
) -> DirectoryChain:
    root.verify(check=check)
    descriptors = [os.dup(root.descriptor)]
    names: list[str] = []
    identities: list[DirectoryIdentity] = []
    try:
        for part in parts:
            parent = descriptors[-1]
            before = _lstat_at(parent, part, path=path, check=check)
            if stat.S_ISLNK(before.st_mode):
                raise RecoveryError(path=path, check=check, reason="symlink_component")
            if not stat.S_ISDIR(before.st_mode):
                raise RecoveryError(path=path, check=check, reason="non_directory_component")
            descriptor = os.open(part, _flags(directory=True), dir_fd=parent)
            identity = DirectoryIdentity.from_stat(before)
            if DirectoryIdentity.from_stat(os.fstat(descriptor)) != identity:
                os.close(descriptor)
                raise RecoveryError(path=path, check=check, reason="component_changed")
            descriptors.append(descriptor)
            names.append(part)
            identities.append(identity)
        chain = DirectoryChain(descriptors, names, identities, path)
        chain.verify(check=check)
        return chain
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _open_control_file(path: Path, *, check: str) -> tuple[int, DirectoryChain, str, Identity]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parent = _open_absolute_directory(absolute.parent, check=check)
    name = absolute.name
    before = _lstat_at(parent.descriptor, name, path=str(path), check=check)
    if stat.S_ISLNK(before.st_mode):
        parent.close()
        raise RecoveryError(path=str(path), check=check, reason="symlink_final")
    if not stat.S_ISREG(before.st_mode):
        parent.close()
        raise RecoveryError(path=str(path), check=check, reason="non_regular_final")
    if before.st_nlink != 1:
        parent.close()
        raise RecoveryError(path=str(path), check=check, reason="multiple_links")
    try:
        descriptor = os.open(name, _flags(directory=False), dir_fd=parent.descriptor)
    except OSError as error:
        parent.close()
        raise RecoveryError(path=str(path), check=check, reason="file_open_failed") from error
    identity = Identity.from_stat(before)
    if Identity.from_stat(os.fstat(descriptor)) != identity:
        os.close(descriptor)
        parent.close()
        raise RecoveryError(path=str(path), check=check, reason="file_changed")
    return descriptor, parent, name, identity


def _read_descriptor(
    descriptor: int,
    *,
    expected: Identity,
    path: str,
    check: str,
    parent: DirectoryChain,
    name: str,
    after_read_hook: AfterSourceReadHook | None = None,
) -> tuple[bytes, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, READ_SIZE):
        digest.update(chunk)
        chunks.append(chunk)
    if after_read_hook is not None:
        after_read_hook(path)
    parent.verify(check=check)
    rebound = _lstat_at(parent.descriptor, name, path=path, check=check)
    if (
        rebound.st_nlink != 1
        or Identity.from_stat(os.fstat(descriptor)) != expected
        or Identity.from_stat(rebound) != expected
    ):
        raise RecoveryError(path=path, check=check, reason="file_changed")
    return b"".join(chunks), digest.hexdigest()


def _read_manifest(
    manifest: Path,
    *,
    expected_manifest_sha256: str,
    expected_entry_count: int,
) -> tuple[list[ManifestEntry], str]:
    if not SHA256_RE.fullmatch(expected_manifest_sha256):
        raise RecoveryError(
            path="<expected-manifest-sha256>",
            check="arguments",
            reason="invalid_sha256",
        )
    if expected_entry_count < 0:
        raise RecoveryError(
            path="<expected-entry-count>",
            check="arguments",
            reason="negative_count",
        )
    descriptor, parent, name, identity = _open_control_file(
        manifest,
        check="manifest_read",
    )
    try:
        payload, observed_sha256 = _read_descriptor(
            descriptor,
            expected=identity,
            path=str(manifest),
            check="manifest_read",
            parent=parent,
            name=name,
        )
    finally:
        os.close(descriptor)
        parent.close()
    if observed_sha256 != expected_manifest_sha256:
        raise RecoveryError(
            path=str(manifest),
            check="manifest_digest",
            reason="digest_mismatch",
        )
    entries = _parse_manifest(payload, expected_count=expected_entry_count)
    return entries, observed_sha256


def _parse_manifest(payload: bytes, *, expected_count: int) -> list[ManifestEntry]:
    if b"\r" in payload:
        raise RecoveryError(path="<manifest>", check="manifest_format", reason="cr_rejected")
    if payload and not payload.endswith(b"\n"):
        raise RecoveryError(
            path="<manifest>",
            check="manifest_format",
            reason="missing_final_lf",
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RecoveryError(
            path="<manifest>",
            check="manifest_format",
            reason="invalid_utf8",
        ) from error
    raw_lines = text.splitlines()
    if len(raw_lines) != expected_count:
        raise RecoveryError(
            path="<manifest>",
            check="manifest_count",
            reason="count_mismatch",
        )
    entries: list[ManifestEntry] = []
    for line in raw_lines:
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise RecoveryError(
                path="<manifest>",
                check="manifest_format",
                reason="malformed_record",
            )
        path = match.group("path")
        namespace, parts = _normalize_member_path(path)
        entries.append(
            ManifestEntry(
                sha256=match.group("sha256"),
                path=path,
                namespace=namespace,
                parts=parts,
            )
        )
    paths = [entry.path for entry in entries]
    if len(set(paths)) != len(paths):
        raise RecoveryError(
            path="<manifest>",
            check="manifest_order",
            reason="duplicate_path",
        )
    if paths != sorted(paths):
        raise RecoveryError(
            path="<manifest>",
            check="manifest_order",
            reason="unsorted_paths",
        )
    return entries


def _normalize_member_path(value: str) -> tuple[str, tuple[str, ...]]:
    if not value or "\\" in value or value.startswith("/"):
        raise RecoveryError(path=value, check="member_path", reason="invalid_path")
    if value.startswith(SIBLING_PREFIX):
        namespace = "sibling"
        remainder = value.removeprefix(SIBLING_PREFIX)
    else:
        namespace = "project"
        remainder = value
    parts = tuple(remainder.split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise RecoveryError(path=value, check="member_path", reason="invalid_component")
    if namespace == "project" and value.startswith("../"):
        raise RecoveryError(path=value, check="member_path", reason="parent_escape")
    canonical = SIBLING_PREFIX + "/".join(parts) if namespace == "sibling" else "/".join(parts)
    if canonical != value:
        raise RecoveryError(path=value, check="member_path", reason="noncanonical_path")
    return namespace, parts


def _open_roots(project_root: Path, *, check: str) -> dict[str, DirectoryChain]:
    absolute = Path(os.path.abspath(os.fspath(project_root)))
    project = _open_absolute_directory(absolute, check=check)
    try:
        sibling = _open_absolute_directory(
            absolute.parent / "tanager-spec",
            check=check,
        )
    except BaseException:
        project.close()
        raise
    return {"project": project, "sibling": sibling}


def _close_roots(roots: dict[str, DirectoryChain]) -> None:
    for root in roots.values():
        root.close()


def _open_source(
    entry: ManifestEntry,
    roots: dict[str, DirectoryChain],
    *,
    check: str,
    after_read_hook: AfterSourceReadHook | None,
) -> SourceHandle:
    root = roots[entry.namespace]
    parent = _open_relative_parent(
        root,
        entry.parts[:-1],
        path=entry.path,
        check=check,
    )
    name = entry.parts[-1]
    before = _lstat_at(parent.descriptor, name, path=entry.path, check=check)
    if stat.S_ISLNK(before.st_mode):
        parent.close()
        raise RecoveryError(path=entry.path, check=check, reason="symlink_final")
    if not stat.S_ISREG(before.st_mode):
        parent.close()
        raise RecoveryError(path=entry.path, check=check, reason="non_regular_final")
    if before.st_nlink != 1:
        parent.close()
        raise RecoveryError(path=entry.path, check=check, reason="multiple_links")
    try:
        descriptor = os.open(name, _flags(directory=False), dir_fd=parent.descriptor)
    except OSError as error:
        parent.close()
        raise RecoveryError(path=entry.path, check=check, reason="file_open_failed") from error
    identity = Identity.from_stat(before)
    if Identity.from_stat(os.fstat(descriptor)) != identity:
        os.close(descriptor)
        parent.close()
        raise RecoveryError(path=entry.path, check=check, reason="file_changed")
    handle = SourceHandle(entry, parent, descriptor, identity)
    try:
        _, digest = _read_descriptor(
            descriptor,
            expected=identity,
            path=entry.path,
            check=check,
            parent=parent,
            name=name,
            after_read_hook=after_read_hook,
        )
    except BaseException:
        handle.close()
        raise
    if digest != entry.sha256:
        handle.close()
        raise RecoveryError(path=entry.path, check=check, reason="digest_mismatch")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return handle


def _destination_absent(parent: DirectoryChain, name: str, *, path: str, check: str) -> None:
    parent.verify(check=check)
    try:
        os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        if error.errno == errno.ENOENT:
            return
        raise RecoveryError(path=path, check=check, reason="destination_probe_failed") from error
    raise RecoveryError(path=path, check=check, reason="destination_exists")


def _open_destination(
    entry: ManifestEntry,
    roots: dict[str, DirectoryChain],
    *,
    check: str,
) -> DestinationHandle:
    parent = _open_relative_parent(
        roots[entry.namespace],
        entry.parts[:-1],
        path=entry.path,
        check=check,
    )
    try:
        _destination_absent(
            parent,
            entry.parts[-1],
            path=entry.path,
            check=check,
        )
    except BaseException:
        parent.close()
        raise
    return DestinationHandle(entry, parent)


def _revalidate_source(handle: SourceHandle, *, check: str) -> None:
    handle.parent.verify(check=check)
    rebound = _lstat_at(
        handle.parent.descriptor,
        handle.entry.parts[-1],
        path=handle.entry.path,
        check=check,
    )
    if (
        rebound.st_nlink != 1
        or Identity.from_stat(os.fstat(handle.descriptor)) != handle.identity
        or Identity.from_stat(rebound) != handle.identity
    ):
        raise RecoveryError(
            path=handle.entry.path,
            check=check,
            reason="file_changed",
        )


def _copy_one(
    source: SourceHandle,
    destination: DestinationHandle,
    *,
    before_destination_create_hook: BeforeDestinationCreateHook | None,
) -> None:
    entry = source.entry
    _revalidate_source(source, check="install_source")
    _destination_absent(
        destination.parent,
        entry.parts[-1],
        path=entry.path,
        check="install_destination",
    )
    if before_destination_create_hook is not None:
        before_destination_create_hook(entry.path)
    destination.parent.verify(check="install_destination")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        output_fd = os.open(
            entry.parts[-1],
            flags,
            0o600,
            dir_fd=destination.parent.descriptor,
        )
    except OSError as error:
        raise RecoveryError(
            path=entry.path,
            check="install_destination",
            reason="exclusive_create_failed",
        ) from error
    try:
        initial = os.fstat(output_fd)
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise RecoveryError(
                path=entry.path,
                check="install_destination",
                reason="invalid_created_file",
            )
        os.lseek(source.descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(source.descriptor, READ_SIZE):
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output_fd, view)
                if written <= 0:
                    raise RecoveryError(
                        path=entry.path,
                        check="install_destination",
                        reason="short_write",
                    )
                view = view[written:]
        os.fsync(output_fd)
        if digest.hexdigest() != entry.sha256:
            raise RecoveryError(
                path=entry.path,
                check="install_source",
                reason="digest_mismatch",
            )
        _revalidate_source(source, check="install_source")
        destination.parent.verify(check="install_destination")
        final = Identity.from_stat(os.fstat(output_fd))
        rebound = _lstat_at(
            destination.parent.descriptor,
            entry.parts[-1],
            path=entry.path,
            check="install_destination",
        )
        if not stat.S_ISREG(final.mode) or final.links != 1 or Identity.from_stat(rebound) != final:
            raise RecoveryError(
                path=entry.path,
                check="install_destination",
                reason="created_file_changed",
            )
    finally:
        os.close(output_fd)


def verify_source_recovery(
    *,
    manifest: Path,
    expected_manifest_sha256: str,
    expected_entry_count: int,
    project_root: Path,
    after_source_read_hook: AfterSourceReadHook | None = None,
) -> tuple[str, int]:
    """Verify every recovery member using stable, single-link file identities."""
    entries, manifest_sha256 = _read_manifest(
        manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_entry_count=expected_entry_count,
    )
    roots = _open_roots(project_root, check="source_root")
    try:
        for entry in entries:
            handle = _open_source(
                entry,
                roots,
                check="source_member",
                after_read_hook=after_source_read_hook,
            )
            handle.close()
        for root in roots.values():
            root.verify(check="source_root")
    finally:
        _close_roots(roots)
    return manifest_sha256, len(entries)


def install_source_recovery(
    *,
    manifest: Path,
    expected_manifest_sha256: str,
    expected_entry_count: int,
    source_project_root: Path,
    destination_project_root: Path,
    after_source_read_hook: AfterSourceReadHook | None = None,
    before_destination_create_hook: BeforeDestinationCreateHook | None = None,
    after_install_hook: AfterInstallHook | None = None,
) -> tuple[str, int]:
    """Install only absent recovery members and verify the complete destination."""
    entries, manifest_sha256 = _read_manifest(
        manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_entry_count=expected_entry_count,
    )
    source_roots = _open_roots(source_project_root, check="source_root")
    try:
        destination_roots = _open_roots(
            destination_project_root,
            check="destination_root",
        )
    except BaseException:
        _close_roots(source_roots)
        raise
    sources: list[SourceHandle] = []
    destinations: list[DestinationHandle] = []
    try:
        for entry in entries:
            sources.append(
                _open_source(
                    entry,
                    source_roots,
                    check="install_source",
                    after_read_hook=after_source_read_hook,
                )
            )
            destinations.append(
                _open_destination(
                    entry,
                    destination_roots,
                    check="install_destination",
                )
            )
        for source in sources:
            _revalidate_source(source, check="install_source")
        for destination in destinations:
            _destination_absent(
                destination.parent,
                destination.entry.parts[-1],
                path=destination.entry.path,
                check="install_destination",
            )
        for index, (source, destination) in enumerate(
            zip(sources, destinations, strict=True),
            start=1,
        ):
            _copy_one(
                source,
                destination,
                before_destination_create_hook=before_destination_create_hook,
            )
            if after_install_hook is not None:
                after_install_hook(source.entry.path, index)
    finally:
        for source in sources:
            source.close()
        for destination in destinations:
            destination.close()
        _close_roots(source_roots)
        _close_roots(destination_roots)
    verified_sha256, verified_count = verify_source_recovery(
        manifest=manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_entry_count=expected_entry_count,
        project_root=destination_project_root,
    )
    if verified_sha256 != manifest_sha256 or verified_count != len(entries):
        raise RecoveryError(
            path="<destination>",
            check="post_install",
            reason="verification_mismatch",
        )
    return manifest_sha256, len(entries)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("verify", "install"):
        command = subparsers.add_parser(mode)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--expected-manifest-sha256", required=True)
        command.add_argument("--expected-entry-count", type=int, required=True)
        if mode == "verify":
            command.add_argument("--project-root", type=Path, required=True)
        else:
            command.add_argument("--source-project-root", type=Path, required=True)
            command.add_argument("--destination-project-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded recovery CLI."""
    args = _parser().parse_args(argv)
    try:
        if args.mode == "verify":
            manifest_sha256, entry_count = verify_source_recovery(
                manifest=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_entry_count=args.expected_entry_count,
                project_root=args.project_root,
            )
        else:
            manifest_sha256, entry_count = install_source_recovery(
                manifest=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_entry_count=args.expected_entry_count,
                source_project_root=args.source_project_root,
                destination_project_root=args.destination_project_root,
            )
    except RecoveryError as error:
        print(error.render(), file=sys.stderr)
        return 1
    print(
        f"PASS check=source_recovery mode={args.mode} "
        f"manifest_sha256={manifest_sha256} entry_count={entry_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
