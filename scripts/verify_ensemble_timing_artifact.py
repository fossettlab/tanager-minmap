#!/usr/bin/env python3
"""Verify the frozen E6-v6 timing-only artifact without exposing its values."""

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

CHECK_NAME = "ensemble_timing_artifact"
READ_SIZE = 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")

EXPECTED_MEMBERS = frozenset({"design.json", "members.csv", "timing_pilot.json"})
EXPECTED_TOP_LEVEL_FIELDS = frozenset({"schema_version", "mode", "fit_count", "records"})
EXPECTED_RECORD_FIELDS = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class ExpectedRecord:
    """Frozen non-operational identity for one required timing record."""

    site: str
    scene: str
    member_class: str
    stochastic_replicate: int | None


EXPECTED_RECORDS = {
    "goldfield:fit:baseline:r0.01": ExpectedRecord(
        site="goldfield",
        scene="20240925_185504_87_4001",
        member_class="baseline",
        stochastic_replicate=None,
    ),
    "goldfield:fit:joint:r00:ridge0.01": ExpectedRecord(
        site="goldfield",
        scene="20240925_185504_87_4001",
        member_class="joint",
        stochastic_replicate=0,
    ),
    "bingham:fit:baseline:r0.01": ExpectedRecord(
        site="bingham",
        scene="20250911_191523_58_4001",
        member_class="baseline",
        stochastic_replicate=None,
    ),
    "bingham:fit:joint:r00:ridge0.01": ExpectedRecord(
        site="bingham",
        scene="20250911_191523_58_4001",
        member_class="joint",
        stochastic_replicate=0,
    ),
}

AfterReadHook = Callable[[str], None]


class VerificationError(RuntimeError):
    """A low-disclosure failure carrying only a fixed reason code."""

    def __init__(self, reason: str) -> None:
        super().__init__("timing artifact verification failed")
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
        """Capture the stable identity fields from one stat result."""
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
    before = _stat_at(parent_fd, name, reason="run_directory_invalid")
    if stat.S_ISLNK(before.st_mode):
        raise VerificationError("link_rejected")
    if not stat.S_ISDIR(before.st_mode):
        raise VerificationError("run_directory_invalid")
    try:
        descriptor = os.open(name, _flags(directory=True), dir_fd=parent_fd)
    except OSError as error:
        raise VerificationError("run_directory_invalid") from error
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
        raise VerificationError("run_directory_invalid") from error
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
    """A run directory pinned to no-follow parent and directory descriptors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.chain: DirectoryChain | None = None
        self.root_fd: int | None = None

    def __enter__(self) -> BoundDirectory:
        absolute = Path(os.path.abspath(os.fspath(self.path)))
        if absolute == Path(os.sep) or not absolute.name:
            raise VerificationError("run_directory_invalid")
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
        """Require the opened directory and its parent entry to remain identical."""
        if self.chain is None or self.root_fd is None:
            raise VerificationError("internal_error")
        self.chain.verify()


def _inventory(root_fd: int) -> dict[str, EntryIdentity]:
    try:
        names = os.listdir(root_fd)
    except OSError as error:
        raise VerificationError("membership_unavailable") from error
    if len(names) != len(EXPECTED_MEMBERS) or set(names) != EXPECTED_MEMBERS:
        raise VerificationError("membership_mismatch")
    records: dict[str, EntryIdentity] = {}
    for name in sorted(EXPECTED_MEMBERS):
        metadata = _stat_at(root_fd, name, reason="member_unavailable")
        if stat.S_ISLNK(metadata.st_mode):
            raise VerificationError("link_rejected")
        if not stat.S_ISREG(metadata.st_mode):
            raise VerificationError("special_member_rejected")
        if metadata.st_nlink != 1:
            raise VerificationError("hardlink_rejected")
        records[name] = EntryIdentity.from_stat(metadata)
    return records


def _verify_inventory_unchanged(
    root_fd: int,
    expected: Mapping[str, EntryIdentity],
) -> None:
    try:
        names = os.listdir(root_fd)
    except OSError as error:
        raise VerificationError("member_replaced") from error
    if len(names) != len(EXPECTED_MEMBERS) or set(names) != EXPECTED_MEMBERS:
        raise VerificationError("member_replaced")
    for name, identity in expected.items():
        metadata = _stat_at(root_fd, name, reason="member_replaced")
        if EntryIdentity.from_stat(metadata) != identity:
            raise VerificationError("member_replaced")


def _read_member(
    root_fd: int,
    name: str,
    expected: EntryIdentity,
    *,
    capture_payload: bool,
    after_read_hook: AfterReadHook | None,
) -> tuple[bytes | None, str]:
    before = _stat_at(root_fd, name, reason="member_replaced")
    if EntryIdentity.from_stat(before) != expected:
        raise VerificationError("member_replaced")
    try:
        descriptor = os.open(name, _flags(directory=False), dir_fd=root_fd)
    except OSError as error:
        raise VerificationError("member_replaced") from error
    try:
        if EntryIdentity.from_stat(os.fstat(descriptor)) != expected:
            raise VerificationError("member_replaced")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        try:
            while chunk := os.read(descriptor, READ_SIZE):
                digest.update(chunk)
                if capture_payload:
                    chunks.append(chunk)
        except OSError as error:
            raise VerificationError("member_replaced") from error
        if after_read_hook is not None:
            try:
                after_read_hook(name)
            except VerificationError:
                raise
            except Exception as error:
                raise VerificationError("verification_interrupted") from error
        try:
            opened = EntryIdentity.from_stat(os.fstat(descriptor))
            rebound = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except OSError as error:
            raise VerificationError("member_replaced") from error
        if opened != expected or EntryIdentity.from_stat(rebound) != expected:
            raise VerificationError("member_replaced")
        return (b"".join(chunks) if capture_payload else None), digest.hexdigest()
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


def _decode_timing_payload(payload: bytes) -> dict[str, object]:
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


def _valid_wall_time(value: object) -> bool:
    if type(value) is int:
        return value >= 0
    if type(value) is float:
        return math.isfinite(value) and value >= 0
    return False


def _canonical_sha256(value: object) -> bool:
    return type(value) is str and SHA256_RE.fullmatch(value) is not None


def _validate_payload(payload: Mapping[str, object]) -> None:
    if set(payload) != EXPECTED_TOP_LEVEL_FIELDS:
        raise VerificationError("top_level_fields")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("mode") != "timing_pilot_only"
        or type(payload.get("fit_count")) is not int
        or payload.get("fit_count") != len(EXPECTED_RECORDS)
    ):
        raise VerificationError("top_level_identity")
    records = payload.get("records")
    if type(records) is not list or len(records) != len(EXPECTED_RECORDS):
        raise VerificationError("record_count")

    observed_fits: set[str] = set()
    for record in records:
        if type(record) is not dict or set(record) != EXPECTED_RECORD_FIELDS:
            raise VerificationError("record_fields")
        fit_id = record["fit_id"]
        if type(fit_id) is not str or fit_id not in EXPECTED_RECORDS or fit_id in observed_fits:
            raise VerificationError("fit_identity")
        expected = EXPECTED_RECORDS[fit_id]
        replicate = record["stochastic_replicate"]
        replicate_matches = (
            replicate is None
            if expected.stochastic_replicate is None
            else type(replicate) is int and replicate == expected.stochastic_replicate
        )
        if (
            type(record["site"]) is not str
            or record["site"] != expected.site
            or type(record["scene"]) is not str
            or record["scene"] != expected.scene
            or type(record["member_class"]) is not str
            or record["member_class"] != expected.member_class
            or not replicate_matches
        ):
            raise VerificationError("record_identity")
        if not _valid_wall_time(record["wall_time_seconds"]):
            raise VerificationError("wall_time_type_or_range")
        peak_memory = record["peak_memory_bytes"]
        if type(peak_memory) is not int or peak_memory < 0:
            raise VerificationError("peak_memory_type_or_range")
        if not _canonical_sha256(record["output_sha256"]):
            raise VerificationError("output_sha256")
        if record["device"] != "cpu" or type(record["device"]) is not str:
            raise VerificationError("device_identity")
        if record["scientific_outputs_retained"] is not False:
            raise VerificationError("retention_identity")
        observed_fits.add(fit_id)
    if observed_fits != set(EXPECTED_RECORDS):
        raise VerificationError("fit_identity")


def verify_ensemble_timing_artifact(
    run_dir: Path,
    *,
    expected_design_sha256: str,
    expected_members_sha256: str,
    expected_timing_sha256: str,
    after_read_hook: AfterReadHook | None = None,
) -> None:
    """Verify the closed v6 timing artifact through descriptor-bound reads."""
    expected_hashes = {
        "design.json": expected_design_sha256,
        "members.csv": expected_members_sha256,
        "timing_pilot.json": expected_timing_sha256,
    }
    if not all(_canonical_sha256(value) for value in expected_hashes.values()):
        raise VerificationError("expected_sha256")

    timing_payload: bytes | None = None
    with BoundDirectory(run_dir) as bound:
        if bound.root_fd is None:
            raise VerificationError("internal_error")
        identities = _inventory(bound.root_fd)
        bound.verify_binding()
        for name in ("design.json", "members.csv", "timing_pilot.json"):
            payload, observed_sha256 = _read_member(
                bound.root_fd,
                name,
                identities[name],
                capture_payload=name == "timing_pilot.json",
                after_read_hook=after_read_hook,
            )
            if observed_sha256 != expected_hashes[name]:
                raise VerificationError("detached_hash_mismatch")
            if payload is not None:
                timing_payload = payload
            bound.verify_binding()
        _verify_inventory_unchanged(bound.root_fd, identities)
        bound.verify_binding()
        if timing_payload is None:
            raise VerificationError("internal_error")
        _validate_payload(_decode_timing_payload(timing_payload))
        _verify_inventory_unchanged(bound.root_fd, identities)
        bound.verify_binding()


def verify_timing_artifact(
    run_dir: Path,
    *,
    expected_design_sha256: str,
    expected_members_sha256: str,
    expected_timing_sha256: str,
    after_read_hook: AfterReadHook | None = None,
) -> None:
    """Compatibility entry point for the standalone timing verifier."""
    verify_ensemble_timing_artifact(
        run_dir,
        expected_design_sha256=expected_design_sha256,
        expected_members_sha256=expected_members_sha256,
        expected_timing_sha256=expected_timing_sha256,
        after_read_hook=after_read_hook,
    )


class LowDisclosureArgumentParser(argparse.ArgumentParser):
    """Raise a fixed reason code instead of echoing malformed CLI values."""

    def error(self, _message: str) -> None:
        raise VerificationError("cli_arguments")


def _parser() -> argparse.ArgumentParser:
    parser = LowDisclosureArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument(
        "--expected-members-sha256",
        "--expected-member-sha256",
        dest="expected_members_sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-timing-sha256",
        "--expected-timing-pilot-sha256",
        dest="expected_timing_sha256",
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the low-disclosure verifier CLI."""
    try:
        args = _parser().parse_args(argv)
        verify_ensemble_timing_artifact(
            args.run_dir,
            expected_design_sha256=args.expected_design_sha256,
            expected_members_sha256=args.expected_members_sha256,
            expected_timing_sha256=args.expected_timing_sha256,
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
