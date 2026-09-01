#!/usr/bin/env python3
"""Convert one exact raw ``sacct --parsable2`` row into a closed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

import verify_ensemble_timing_scheduler_receipt as receipt_verifier

CHECK_NAME = "capture_ensemble_timing_scheduler_receipt"


class CollectionError(RuntimeError):
    """A low-disclosure collection failure carrying a fixed reason code."""

    def __init__(self, reason: str) -> None:
        super().__init__("scheduler receipt collection failed")
        self.reason = reason

    def render(self) -> str:
        return f"FAIL check={CHECK_NAME} reason={self.reason}"


class LowDisclosureArgumentParser(argparse.ArgumentParser):
    """Raise a fixed reason code instead of echoing malformed CLI values."""

    def error(self, _message: str) -> None:
        raise CollectionError("cli_arguments")


def _read_raw(path: Path) -> bytes:
    try:
        payload, _digest = receipt_verifier._read_receipt(path, after_read_hook=None)
    except receipt_verifier.VerificationError as error:
        raise CollectionError("raw_input_invalid") from error
    return payload


def _parse_raw_row(payload: bytes, expected_job_id: str) -> tuple[str, str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CollectionError("raw_input_format") from error
    if not text.endswith("\n") or "\r" in text or text.count("\n") != 1:
        raise CollectionError("raw_input_format")
    row = text[:-1]
    fields = row.split("|")
    if len(fields) != len(receipt_verifier.QUERY_FIELDS) + 1 or fields[-1] != "":
        raise CollectionError("raw_input_format")
    job_id_raw, state, exit_code, elapsed_raw, max_rss_raw = fields[:-1]
    if job_id_raw != f"{expected_job_id}.batch":
        raise CollectionError("job_identity")
    if state != "COMPLETED":
        raise CollectionError("state_identity")
    if exit_code != "0:0":
        raise CollectionError("exit_identity")
    if not receipt_verifier._raw_value(elapsed_raw):
        raise CollectionError("elapsed_raw")
    if not receipt_verifier._raw_value(max_rss_raw):
        raise CollectionError("max_rss_raw")
    return row, elapsed_raw, max_rss_raw


def _receipt_payload(
    *,
    expected_job_id: str,
    raw_row: str,
    elapsed_raw: str,
    max_rss_raw: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "slurm_sacct_parsable2",
        "query_fields": list(receipt_verifier.QUERY_FIELDS),
        "raw_rows": [raw_row],
        "record_count": 1,
        "records": [
            {
                "job_id": expected_job_id,
                "step_id": "batch",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "elapsed_raw": elapsed_raw,
                "max_rss_raw": max_rss_raw,
                "elapsed_scope": "slurm_batch_step_elapsed",
                "max_rss_scope": "slurm_batch_step_host_memory",
                "separate_from_per_fit_python_telemetry": True,
                "accelerator_memory_measured": False,
                "unit_conversion_applied": False,
            }
        ],
    }


def _encode(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.name:
        raise CollectionError("output_path_invalid")
    try:
        bound_context = receipt_verifier.BoundDirectory(absolute.parent)
        with bound_context as bound:
            if bound.root_fd is None:
                raise CollectionError("internal_error")
            temporary_name = f".{absolute.name}.part-{os.getpid()}-{secrets.token_hex(8)}"
            flags = (
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                descriptor = os.open(temporary_name, flags, 0o600, dir_fd=bound.root_fd)
            except OSError as error:
                raise CollectionError("output_create_failed") from error
            temporary_identity = receipt_verifier.EntryIdentity.from_stat(os.fstat(descriptor))
            published = False
            completed = False

            def unlink_if_owned(name: str) -> None:
                try:
                    current = os.stat(name, dir_fd=bound.root_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return
                if (
                    current.st_dev == temporary_identity.device
                    and current.st_ino == temporary_identity.inode
                ):
                    os.unlink(name, dir_fd=bound.root_fd)

            try:
                if not stat.S_ISREG(temporary_identity.mode) or temporary_identity.links != 1:
                    raise CollectionError("output_identity")
                view = memoryview(payload)
                while view:
                    try:
                        written = os.write(descriptor, view)
                    except OSError as error:
                        raise CollectionError("output_write_failed") from error
                    if written <= 0:
                        raise CollectionError("output_write_failed")
                    view = view[written:]
                try:
                    os.fsync(descriptor)
                except OSError as error:
                    raise CollectionError("output_write_failed") from error
                opened = receipt_verifier.EntryIdentity.from_stat(os.fstat(descriptor))
                rebound = receipt_verifier.EntryIdentity.from_stat(
                    os.stat(temporary_name, dir_fd=bound.root_fd, follow_symlinks=False)
                )
                if opened != rebound or opened.links != 1 or opened.size != len(payload):
                    raise CollectionError("output_identity")
                bound.verify_binding()
                try:
                    os.link(
                        temporary_name,
                        absolute.name,
                        src_dir_fd=bound.root_fd,
                        dst_dir_fd=bound.root_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError as error:
                    raise CollectionError("output_exists") from error
                except OSError as error:
                    raise CollectionError("output_publish_failed") from error
                published = True
                try:
                    os.unlink(temporary_name, dir_fd=bound.root_fd)
                    final = os.stat(
                        absolute.name,
                        dir_fd=bound.root_fd,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise CollectionError("output_publish_failed") from error
                if (
                    not stat.S_ISREG(final.st_mode)
                    or final.st_nlink != 1
                    or final.st_dev != temporary_identity.device
                    or final.st_ino != temporary_identity.inode
                    or final.st_size != len(payload)
                ):
                    raise CollectionError("output_identity")
                bound.verify_binding()
                completed = True
            finally:
                os.close(descriptor)
                unlink_if_owned(temporary_name)
                if published and not completed:
                    unlink_if_owned(absolute.name)
    except receipt_verifier.VerificationError as error:
        raise CollectionError("output_path_invalid") from error


def capture_scheduler_receipt(
    raw_sacct: Path,
    output: Path,
    *,
    expected_job_id: str,
) -> str:
    """Capture and verify a deterministic receipt; return its SHA-256."""
    if not receipt_verifier._canonical_job_id(expected_job_id):
        raise CollectionError("expected_job_id")
    raw_row, elapsed_raw, max_rss_raw = _parse_raw_row(_read_raw(raw_sacct), expected_job_id)
    encoded = _encode(
        _receipt_payload(
            expected_job_id=expected_job_id,
            raw_row=raw_row,
            elapsed_raw=elapsed_raw,
            max_rss_raw=max_rss_raw,
        )
    )
    _write_exclusive(output, encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        receipt_verifier.verify_scheduler_receipt(
            output,
            expected_receipt_sha256=digest,
            expected_job_id=expected_job_id,
        )
    except receipt_verifier.VerificationError as error:
        raise CollectionError("generated_receipt_invalid") from error
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = LowDisclosureArgumentParser(description=__doc__)
    parser.add_argument("--raw-sacct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the low-disclosure receipt collector."""
    try:
        args = _parser().parse_args(argv)
        capture_scheduler_receipt(
            args.raw_sacct,
            args.output,
            expected_job_id=args.expected_job_id,
        )
    except CollectionError as error:
        print(error.render(), file=sys.stderr)
        return 1
    except Exception:
        print(f"FAIL check={CHECK_NAME} reason=internal_error", file=sys.stderr)
        return 1
    print(f"PASS check={CHECK_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
