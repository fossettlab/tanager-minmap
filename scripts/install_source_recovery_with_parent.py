#!/usr/bin/env python3
"""Create one missing project parent and install a bounded recovery atomically."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

REVIEWED_HELPER_NAME = "verify_source_recovery.py"
REVIEWED_HELPER_SHA256 = "e7fdca673f170cff1828366400cec0559384f5e20fc7d72d23a8a55da11a1dd4"
MODULE_NAME = "_tanager_reviewed_source_recovery_transaction"
READ_SIZE = 1024 * 1024

AfterParentCreateHook = Callable[[str], None]
BeforeDestinationCreateHook = Callable[[str], None]


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return the stable file identity fields used by the reviewed helper."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _load_reviewed_helper(path: Path) -> ModuleType:
    """Load the exact reviewed helper from captured, no-follow bytes."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    parent_fd = os.open(absolute.parent, parent_flags)
    try:
        before = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError("reviewed helper is not a regular single-link file")
        descriptor = os.open(absolute.name, file_flags, dir_fd=parent_fd)
        try:
            opened = os.fstat(descriptor)
            if _identity(opened) != _identity(before):
                raise RuntimeError("reviewed helper identity changed before read")
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, READ_SIZE):
                digest.update(chunk)
                chunks.append(chunk)
            rebound = os.stat(
                absolute.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            after = os.fstat(descriptor)
            if _identity(after) != _identity(opened) or _identity(rebound) != _identity(opened):
                raise RuntimeError("reviewed helper identity changed after read")
            if digest.hexdigest() != REVIEWED_HELPER_SHA256:
                raise RuntimeError("reviewed helper digest mismatch")
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)

    module = ModuleType(MODULE_NAME)
    module.__file__ = os.fspath(absolute)
    module.__package__ = None
    sys.modules[MODULE_NAME] = module
    exec(compile(payload, os.fspath(absolute), "exec"), module.__dict__)
    return module


def _parent_parts(helper: ModuleType, value: str, entries: Sequence[object]) -> tuple[str, ...]:
    """Validate one project parent used directly by at least one manifest member."""
    namespace, parts = helper._normalize_member_path(value)
    if namespace != "project" or not parts:
        raise helper.RecoveryError(
            path=value,
            check="create_project_parent",
            reason="invalid_project_parent",
        )
    direct_children = [
        entry for entry in entries if entry.namespace == "project" and entry.parts[:-1] == parts
    ]
    nested_children = [
        entry
        for entry in entries
        if entry.namespace == "project"
        and entry.parts[: len(parts)] == parts
        and entry.parts[:-1] != parts
    ]
    if not direct_children or nested_children:
        raise helper.RecoveryError(
            path=value,
            check="create_project_parent",
            reason="unsupported_project_parent",
        )
    return parts


def _create_project_parent(
    helper: ModuleType,
    destination_roots: dict[str, object],
    parts: tuple[str, ...],
    *,
    path: str,
    after_create_hook: AfterParentCreateHook | None,
) -> tuple[object, object]:
    """Exclusively create and retain a descriptor chain to one missing parent."""
    chain = helper._open_relative_parent(
        destination_roots["project"],
        parts[:-1],
        path=path,
        check="create_project_parent",
    )
    descriptor: int | None = None
    name = parts[-1]
    try:
        helper._destination_absent(
            chain,
            name,
            path=path,
            check="create_project_parent",
        )
        chain.verify(check="create_project_parent")
        try:
            os.mkdir(name, 0o700, dir_fd=chain.descriptor)
            os.fsync(chain.descriptor)
        except OSError as error:
            raise helper.RecoveryError(
                path=path,
                check="create_project_parent",
                reason="exclusive_parent_create_failed",
            ) from error
        before = helper._lstat_at(
            chain.descriptor,
            name,
            path=path,
            check="create_project_parent",
        )
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise helper.RecoveryError(
                path=path,
                check="create_project_parent",
                reason="invalid_created_parent",
            )
        descriptor = os.open(
            name,
            helper._flags(directory=True),
            dir_fd=chain.descriptor,
        )
        identity = helper.DirectoryIdentity.from_stat(before)
        if helper.DirectoryIdentity.from_stat(os.fstat(descriptor)) != identity:
            raise helper.RecoveryError(
                path=path,
                check="create_project_parent",
                reason="created_parent_changed",
            )
        chain.descriptors.append(descriptor)
        descriptor = None
        chain.names.append(name)
        chain.identities.append(identity)
        chain.display = path
        if after_create_hook is not None:
            after_create_hook(path)
        chain.verify(check="create_project_parent")
        return chain, identity
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        chain.close()
        raise


def install_source_recovery_with_parent(
    *,
    manifest: Path,
    expected_manifest_sha256: str,
    expected_entry_count: int,
    source_project_root: Path,
    destination_project_root: Path,
    create_project_parent: str,
    reviewed_helper: Path | None = None,
    after_parent_create_hook: AfterParentCreateHook | None = None,
    before_destination_create_hook: BeforeDestinationCreateHook | None = None,
) -> tuple[str, int]:
    """Create one absent parent and install all members in one pinned process."""
    helper_path = reviewed_helper or Path(__file__).with_name(REVIEWED_HELPER_NAME)
    helper = _load_reviewed_helper(helper_path)
    entries, manifest_sha256 = helper._read_manifest(
        manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_entry_count=expected_entry_count,
    )
    parent_parts = _parent_parts(helper, create_project_parent, entries)
    source_roots = helper._open_roots(source_project_root, check="source_root")
    try:
        destination_roots = helper._open_roots(
            destination_project_root,
            check="destination_root",
        )
    except BaseException:
        helper._close_roots(source_roots)
        raise

    sources: list[object] = []
    destinations: list[object] = []
    created_parent: object | None = None
    created_identity: object | None = None
    try:
        for entry in entries:
            sources.append(
                helper._open_source(
                    entry,
                    source_roots,
                    check="install_source",
                    after_read_hook=None,
                )
            )
        for source in sources:
            helper._revalidate_source(source, check="install_source")

        created_parent, created_identity = _create_project_parent(
            helper,
            destination_roots,
            parent_parts,
            path=create_project_parent,
            after_create_hook=after_parent_create_hook,
        )
        for entry in entries:
            destination = helper._open_destination(
                entry,
                destination_roots,
                check="install_destination",
            )
            if entry.namespace == "project" and entry.parts[:-1] == parent_parts:
                observed = helper.DirectoryIdentity.from_stat(
                    os.fstat(destination.parent.descriptor)
                )
                if observed != created_identity:
                    destination.close()
                    raise helper.RecoveryError(
                        path=entry.path,
                        check="install_destination",
                        reason="created_parent_changed",
                    )
            destinations.append(destination)

        for destination in destinations:
            helper._destination_absent(
                destination.parent,
                destination.entry.parts[-1],
                path=destination.entry.path,
                check="install_destination",
            )
        for source, destination in zip(sources, destinations, strict=True):
            created_parent.verify(check="install_destination")
            helper._copy_one(
                source,
                destination,
                before_destination_create_hook=before_destination_create_hook,
            )
            created_parent.verify(check="install_destination")

        created_parent.verify(check="post_install")
        verified_sha256, verified_count = helper.verify_source_recovery(
            manifest=manifest,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_entry_count=expected_entry_count,
            project_root=destination_project_root,
        )
        created_parent.verify(check="post_install")
        if verified_sha256 != manifest_sha256 or verified_count != len(entries):
            raise helper.RecoveryError(
                path="<destination>",
                check="post_install",
                reason="verification_mismatch",
            )
        return manifest_sha256, len(entries)
    finally:
        for source in sources:
            source.close()
        for destination in destinations:
            destination.close()
        if created_parent is not None:
            created_parent.close()
        helper._close_roots(source_roots)
        helper._close_roots(destination_roots)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-entry-count", type=int, required=True)
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--destination-project-root", type=Path, required=True)
    parser.add_argument("--create-project-parent", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded parent-creation recovery transaction."""
    args = _parser().parse_args(argv)
    try:
        manifest_sha256, entry_count = install_source_recovery_with_parent(
            manifest=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_entry_count=args.expected_entry_count,
            source_project_root=args.source_project_root,
            destination_project_root=args.destination_project_root,
            create_project_parent=args.create_project_parent,
        )
    except RuntimeError:
        print("FAIL check=reviewed_helper reason=authentication_failed", file=sys.stderr)
        return 1
    except Exception as error:  # RecoveryError belongs to the authenticated module.
        if hasattr(error, "render"):
            print(error.render(), file=sys.stderr)
            return 1
        raise
    print(
        "PASS check=source_recovery_parent_transaction "
        f"manifest_sha256={manifest_sha256} entry_count={entry_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
