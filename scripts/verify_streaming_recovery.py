#!/usr/bin/env python3
"""Run the reviewed recovery helper with streaming-only member hashing."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

REVIEWED_HELPER_NAME = "verify_source_recovery.py"
REVIEWED_HELPER_SHA256 = "e7fdca673f170cff1828366400cec0559384f5e20fc7d72d23a8a55da11a1dd4"
MODULE_NAME = "_tanager_reviewed_source_recovery"


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return the file identity fields used by the reviewed helper."""
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
    parent_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(
            os,
            "O_CLOEXEC",
            0,
        )
    )
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
            while chunk := os.read(descriptor, 1024 * 1024):
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


def _enable_streaming_member_reads(helper: ModuleType) -> None:
    """Capture only manifest bytes; stream every recovery member."""

    def read_descriptor(
        descriptor: int,
        *,
        expected: object,
        path: str,
        check: str,
        parent: object,
        name: str,
        after_read_hook: object | None = None,
    ) -> tuple[bytes | None, str]:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        capture_manifest = check == "manifest_read"
        chunks: list[bytes] | None = [] if capture_manifest else None
        while chunk := os.read(descriptor, helper.READ_SIZE):
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        if after_read_hook is not None:
            after_read_hook(path)
        parent.verify(check=check)
        rebound = helper._lstat_at(
            parent.descriptor,
            name,
            path=path,
            check=check,
        )
        if (
            rebound.st_nlink != 1
            or helper.Identity.from_stat(os.fstat(descriptor)) != expected
            or helper.Identity.from_stat(rebound) != expected
        ):
            raise helper.RecoveryError(
                path=path,
                check=check,
                reason="file_changed",
            )
        payload = b"".join(chunks) if chunks is not None else None
        return payload, digest.hexdigest()

    helper._read_descriptor = read_descriptor


def load_streaming_helper(path: Path | None = None) -> ModuleType:
    """Load and adapt the exact reviewed helper for bounded streaming reads."""
    helper_path = path or Path(__file__).with_name(REVIEWED_HELPER_NAME)
    helper = _load_reviewed_helper(helper_path)
    _enable_streaming_member_reads(helper)
    return helper


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate the CLI to the reviewed helper after the streaming adaptation."""
    helper = load_streaming_helper()
    return helper.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
