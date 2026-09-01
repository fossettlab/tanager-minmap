#!/usr/bin/env python3
"""Verify one detached Slurm timing receipt without disclosing its values.

The receipt preserves the exact output row from this field order:
``JobIDRaw,State,ExitCode,Elapsed,MaxRSS`` under ``sacct --parsable2``.
The raw strings remain unconverted; this verifier does not infer Slurm units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CHECK_NAME = "ensemble_timing_scheduler_receipt"
READ_SIZE = 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
JOB_ID_RE = re.compile(r"[1-9][0-9]{0,19}")

QUERY_FIELDS = ("JobIDRaw", "State", "ExitCode", "Elapsed", "MaxRSS")
TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "source", "query_fields", "raw_rows", "record_count", "records"}
)
RECORD_FIELDS = frozenset(
    {
        "job_id",
        "step_id",
        "state",
        "exit_code",
        "elapsed_raw",
        "max_rss_raw",
        "elapsed_scope",
        "max_rss_scope",
        "separate_from_per_fit_python_telemetry",
        "accelerator_memory_measured",
        "unit_conversion_applied",
    }
)

AfterReadHook = Callable[[], None]


class VerificationError(RuntimeError):
    """A low-disclosure failure carrying only a fixed reason code."""

    def __init__(self, reason: str) -> None:
        super().__init__("scheduler receipt verification failed")
        self.reason = reason

    def render(self) -> str:
        """Render a value-free failure line."""
        return f"FAIL check={CHECK_NAME} reason={self.reason}"


@dataclass(frozen=True)
class EntryIdentity:
    """Metadata used to detect replacement or mutation during verification."""

    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> EntryIdentity:
        """Capture stable identity fields from one stat result."""
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            links=metadata.st_nlink,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
            changed_ns=metadata.st_ctime_ns,
        )


def _flags(*, directory: bool) -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise VerificationError("platform_nofollow_unsupported")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= os.O_DIRECTORY
    else:
        flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _stat_at(parent_fd: int, name: str, *, reason: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise VerificationError(reason) from error


def _open_directory_component(parent_fd: int, name: str) -> int:
    before = _stat_at(parent_fd, name, reason="receipt_path_invalid")
    if stat.S_ISLNK(before.st_mode):
        raise VerificationError("link_rejected")
    if not stat.S_ISDIR(before.st_mode):
        raise VerificationError("receipt_path_invalid")
    try:
        descriptor = os.open(name, _flags(directory=True), dir_fd=parent_fd)
    except OSError as error:
        raise VerificationError("receipt_path_invalid") from error
    if EntryIdentity.from_stat(os.fstat(descriptor)) != EntryIdentity.from_stat(before):
        os.close(descriptor)
        raise VerificationError("directory_replaced")
    return descriptor


@dataclass(frozen=True)
class DirectoryIdentity:
    """Stable identity for one opened directory component."""

    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> DirectoryIdentity:
        return cls(device=metadata.st_dev, inode=metadata.st_ino, mode=metadata.st_mode)


@dataclass(frozen=True)
class DirectoryLink:
    """One parent-to-child binding retained for the complete path chain."""

    parent_fd: int
    name: str
    child_fd: int
    identity: DirectoryIdentity


class DirectoryChain:
    """All no-follow descriptors needed to re-prove an absolute directory path."""

    def __init__(self, descriptors: list[int], links: list[DirectoryLink]) -> None:
        self.descriptors = descriptors
        self.links = links

    @property
    def leaf_fd(self) -> int:
        return self.descriptors[-1]

    def verify(self) -> None:
        for link in self.links:
            try:
                opened = os.fstat(link.child_fd)
                rebound = os.stat(link.name, dir_fd=link.parent_fd, follow_symlinks=False)
            except OSError as error:
                raise VerificationError("directory_replaced") from error
            if (
                stat.S_ISLNK(rebound.st_mode)
                or not stat.S_ISDIR(rebound.st_mode)
                or DirectoryIdentity.from_stat(opened) != link.identity
                or DirectoryIdentity.from_stat(rebound) != link.identity
            ):
                raise VerificationError("directory_replaced")

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.descriptors.clear()


def _open_directory_chain(path: Path) -> DirectoryChain:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        root_fd = os.open(os.sep, _flags(directory=True))
    except OSError as error:
        raise VerificationError("receipt_path_invalid") from error
    descriptors = [root_fd]
    links: list[DirectoryLink] = []
    current = root_fd
    try:
        for part in absolute.parts[1:]:
            following = _open_directory_component(current, part)
            identity = DirectoryIdentity.from_stat(os.fstat(following))
            links.append(
                DirectoryLink(
                    parent_fd=current,
                    name=part,
                    child_fd=following,
                    identity=identity,
                )
            )
            descriptors.append(following)
            current = following
        return DirectoryChain(descriptors, links)
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


class BoundDirectory:
    """A directory pinned to no-follow parent and directory descriptors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.chain: DirectoryChain | None = None
        self.root_fd: int | None = None

    def __enter__(self) -> BoundDirectory:
        absolute = Path(os.path.abspath(os.fspath(self.path)))
        if absolute == Path(os.sep) or not absolute.name:
            raise VerificationError("receipt_path_invalid")
        chain = _open_directory_chain(absolute)
        try:
            chain.verify()
        except BaseException:
            chain.close()
            raise
        self.chain = chain
        self.root_fd = chain.leaf_fd
        return self

    def __exit__(self, *_args: object) -> None:
        self.root_fd = None
        if self.chain is not None:
            self.chain.close()
            self.chain = None

    def verify_binding(self) -> None:
        """Require the opened directory and parent entry to remain identical."""
        if self.chain is None or self.root_fd is None:
            raise VerificationError("internal_error")
        self.chain.verify()


def _read_receipt(
    path: Path,
    *,
    after_read_hook: AfterReadHook | None,
) -> tuple[bytes, str]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.name:
        raise VerificationError("receipt_path_invalid")
    with BoundDirectory(absolute.parent) as bound:
        if bound.root_fd is None:
            raise VerificationError("internal_error")
        before = _stat_at(bound.root_fd, absolute.name, reason="receipt_unavailable")
        if stat.S_ISLNK(before.st_mode):
            raise VerificationError("link_rejected")
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError("special_receipt_rejected")
        if before.st_nlink != 1:
            raise VerificationError("hardlink_rejected")
        expected = EntryIdentity.from_stat(before)
        try:
            descriptor = os.open(absolute.name, _flags(directory=False), dir_fd=bound.root_fd)
        except OSError as error:
            raise VerificationError("receipt_unavailable") from error
        try:
            if EntryIdentity.from_stat(os.fstat(descriptor)) != expected:
                raise VerificationError("receipt_replaced")
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            try:
                while chunk := os.read(descriptor, READ_SIZE):
                    digest.update(chunk)
                    chunks.append(chunk)
            except OSError as error:
                raise VerificationError("receipt_replaced") from error
            if after_read_hook is not None:
                try:
                    after_read_hook()
                except VerificationError:
                    raise
                except Exception as error:
                    raise VerificationError("verification_interrupted") from error
            try:
                opened = EntryIdentity.from_stat(os.fstat(descriptor))
                rebound = os.stat(absolute.name, dir_fd=bound.root_fd, follow_symlinks=False)
            except OSError as error:
                raise VerificationError("receipt_replaced") from error
            if opened != expected or EntryIdentity.from_stat(rebound) != expected:
                raise VerificationError("receipt_replaced")
            bound.verify_binding()
            return b"".join(chunks), digest.hexdigest()
        finally:
            os.close(descriptor)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError("duplicate_json_key")
        result[key] = value
    return result


def _reject_nonfinite_constant(_value: str) -> None:
    raise VerificationError("nonfinite_number")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise VerificationError("nonfinite_number")
    return parsed


def _decode(payload: bytes) -> dict[str, object]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except VerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise VerificationError("invalid_json") from error
    if type(decoded) is not dict:
        raise VerificationError("top_level_fields")
    return decoded


def _canonical_sha256(value: object) -> bool:
    return type(value) is str and SHA256_RE.fullmatch(value) is not None


def _canonical_job_id(value: object) -> bool:
    return type(value) is str and JOB_ID_RE.fullmatch(value) is not None


def _raw_value(value: object) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= 128
        and value == value.strip()
        and all(character.isprintable() for character in value)
        and "|" not in value
    )


def _validate(payload: Mapping[str, object], expected_job_id: str) -> None:
    if set(payload) != TOP_LEVEL_FIELDS:
        raise VerificationError("top_level_fields")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("source") != "slurm_sacct_parsable2"
        or payload.get("query_fields") != list(QUERY_FIELDS)
        or type(payload.get("record_count")) is not int
        or payload.get("record_count") != 1
    ):
        raise VerificationError("top_level_identity")

    raw_rows = payload.get("raw_rows")
    records = payload.get("records")
    if type(raw_rows) is not list or len(raw_rows) != 1:
        raise VerificationError("raw_row_count")
    if type(records) is not list or len(records) != 1:
        raise VerificationError("record_count")
    raw_row = raw_rows[0]
    record = records[0]
    if type(raw_row) is not str or "\n" in raw_row or "\r" in raw_row:
        raise VerificationError("raw_row_format")
    if type(record) is not dict or set(record) != RECORD_FIELDS:
        raise VerificationError("record_fields")

    fields = raw_row.split("|")
    if len(fields) != len(QUERY_FIELDS) + 1 or fields[-1] != "":
        raise VerificationError("raw_row_format")
    job_id_raw, state, exit_code, elapsed_raw, max_rss_raw = fields[:-1]
    if job_id_raw != f"{expected_job_id}.batch":
        raise VerificationError("job_identity")
    if record["job_id"] != expected_job_id or record["step_id"] != "batch":
        raise VerificationError("job_identity")
    if record["state"] != state or state != "COMPLETED":
        raise VerificationError("state_identity")
    if record["exit_code"] != exit_code or exit_code != "0:0":
        raise VerificationError("exit_identity")
    if record["elapsed_raw"] != elapsed_raw or not _raw_value(elapsed_raw):
        raise VerificationError("elapsed_raw")
    if record["max_rss_raw"] != max_rss_raw or not _raw_value(max_rss_raw):
        raise VerificationError("max_rss_raw")
    if (
        record["elapsed_scope"] != "slurm_batch_step_elapsed"
        or record["max_rss_scope"] != "slurm_batch_step_host_memory"
        or record["separate_from_per_fit_python_telemetry"] is not True
        or record["accelerator_memory_measured"] is not False
        or record["unit_conversion_applied"] is not False
    ):
        raise VerificationError("telemetry_semantics")


def verify_scheduler_receipt(
    receipt: Path,
    *,
    expected_receipt_sha256: str,
    expected_job_id: str,
    after_read_hook: AfterReadHook | None = None,
) -> None:
    """Verify one exact, closed Slurm batch-step receipt."""
    if not _canonical_sha256(expected_receipt_sha256):
        raise VerificationError("expected_sha256")
    if not _canonical_job_id(expected_job_id):
        raise VerificationError("expected_job_id")
    payload, observed_sha256 = _read_receipt(receipt, after_read_hook=after_read_hook)
    if observed_sha256 != expected_receipt_sha256:
        raise VerificationError("detached_hash_mismatch")
    _validate(_decode(payload), expected_job_id)


class LowDisclosureArgumentParser(argparse.ArgumentParser):
    """Raise a fixed reason code instead of echoing malformed CLI values."""

    def error(self, _message: str) -> None:
        raise VerificationError("cli_arguments")


def _parser() -> argparse.ArgumentParser:
    parser = LowDisclosureArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the low-disclosure verifier CLI."""
    try:
        args = _parser().parse_args(argv)
        verify_scheduler_receipt(
            args.receipt,
            expected_receipt_sha256=args.expected_receipt_sha256,
            expected_job_id=args.expected_job_id,
        )
    except VerificationError as error:
        print(error.render(), file=sys.stderr)
        return 1
    except Exception:
        print(f"FAIL check={CHECK_NAME} reason=internal_error", file=sys.stderr)
        return 1
    print(f"PASS check={CHECK_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
