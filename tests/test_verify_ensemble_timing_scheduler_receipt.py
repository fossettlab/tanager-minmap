"""Tests for the endpoint-blind Slurm timing-receipt verifier."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_ensemble_timing_scheduler_receipt.py"
JOB_ID = "2770999"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scheduler_receipt_verifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()


@dataclass(frozen=True)
class Receipt:
    path: Path
    sha256: str
    payload: dict[str, Any]
    job_id: str = JOB_ID


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_payload(job_id: str = JOB_ID) -> dict[str, Any]:
    elapsed = "00:01:44"
    max_rss = "22854592K"
    return {
        "schema_version": "1.0",
        "source": "slurm_sacct_parsable2",
        "query_fields": ["JobIDRaw", "State", "ExitCode", "Elapsed", "MaxRSS"],
        "raw_rows": [f"{job_id}.batch|COMPLETED|0:0|{elapsed}|{max_rss}|"],
        "record_count": 1,
        "records": [
            {
                "job_id": job_id,
                "step_id": "batch",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "elapsed_raw": elapsed,
                "max_rss_raw": max_rss,
                "elapsed_scope": "slurm_batch_step_elapsed",
                "max_rss_scope": "slurm_batch_step_host_memory",
                "separate_from_per_fit_python_telemetry": True,
                "accelerator_memory_measured": False,
                "unit_conversion_applied": False,
            }
        ],
    }


def _encode(payload: dict[str, Any], *, allow_nan: bool = False) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=allow_nan,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _make_receipt(tmp_path: Path) -> Receipt:
    path = tmp_path / "scheduler_receipt.json"
    payload = _valid_payload()
    encoded = _encode(payload)
    path.write_bytes(encoded)
    return Receipt(path=path, sha256=_sha256(encoded), payload=payload)


def _replace_payload(
    receipt: Receipt,
    payload: dict[str, Any],
    *,
    allow_nan: bool = False,
) -> Receipt:
    encoded = _encode(payload, allow_nan=allow_nan)
    receipt.path.write_bytes(encoded)
    return replace(receipt, sha256=_sha256(encoded), payload=payload)


def _replace_bytes(receipt: Receipt, payload: bytes) -> Receipt:
    receipt.path.write_bytes(payload)
    return replace(receipt, sha256=_sha256(payload))


def _verify(receipt: Receipt, *, after_read_hook=None) -> None:
    VERIFIER.verify_scheduler_receipt(
        receipt.path,
        expected_receipt_sha256=receipt.sha256,
        expected_job_id=receipt.job_id,
        after_read_hook=after_read_hook,
    )


def _run(receipt: Receipt) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            os.fspath(SCRIPT),
            "--receipt",
            os.fspath(receipt.path),
            "--expected-receipt-sha256",
            receipt.sha256,
            "--expected-job-id",
            receipt.job_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_reason(receipt: Receipt, reason: str) -> None:
    with pytest.raises(VERIFIER.VerificationError) as caught:
        _verify(receipt)
    assert caught.value.reason == reason


def test_valid_receipt_passes(tmp_path: Path):
    receipt = _make_receipt(tmp_path)

    _verify(receipt)
    result = _run(receipt)

    assert result.returncode == 0
    assert result.stdout == "PASS check=ensemble_timing_scheduler_receipt\n"
    assert result.stderr == ""


def test_imports_only_standard_library(tmp_path: Path):
    del tmp_path
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".", maxsplit=1)[0])
    assert imported <= {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "stat",
        "sys",
    }
    assert "tanager_rocks" not in SCRIPT.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "field",
    ["schema_version", "source", "query_fields", "raw_rows", "record_count", "records"],
)
def test_requires_all_top_level_fields(tmp_path: Path, field: str):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    del payload[field]
    _assert_reason(_replace_payload(receipt, payload), "top_level_fields")


def test_rejects_extra_top_level_field(tmp_path: Path):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["extra"] = None
    _assert_reason(_replace_payload(receipt, payload), "top_level_fields")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "2.0"),
        ("source", "other"),
        ("query_fields", ["JobIDRaw"]),
        ("record_count", 2),
        ("record_count", True),
    ],
)
def test_rejects_top_level_identity_drift(
    tmp_path: Path,
    field: str,
    value: Any,
):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload[field] = value
    _assert_reason(_replace_payload(receipt, payload), "top_level_identity")


@pytest.mark.parametrize("rows", [[], ["a", "b"], {}, [None]])
def test_rejects_missing_or_ambiguous_raw_rows(tmp_path: Path, rows: Any):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["raw_rows"] = rows
    reason = "raw_row_format" if rows == [None] else "raw_row_count"
    _assert_reason(_replace_payload(receipt, payload), reason)


@pytest.mark.parametrize("records", [[], [{}, {}], {}, [None]])
def test_rejects_missing_or_ambiguous_records(tmp_path: Path, records: Any):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["records"] = records
    reason = "record_fields" if records == [None] else "record_count"
    _assert_reason(_replace_payload(receipt, payload), reason)


@pytest.mark.parametrize("field", sorted(VERIFIER.RECORD_FIELDS))
def test_requires_all_record_fields(tmp_path: Path, field: str):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    del payload["records"][0][field]
    _assert_reason(_replace_payload(receipt, payload), "record_fields")


def test_rejects_extra_record_field(tmp_path: Path):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["records"][0]["extra"] = None
    _assert_reason(_replace_payload(receipt, payload), "record_fields")


@pytest.mark.parametrize(
    ("raw_job", "record_job", "step"),
    [
        ("999.batch", JOB_ID, "batch"),
        (f"{JOB_ID}.extern", JOB_ID, "extern"),
        (f"{JOB_ID}.batch", "999", "batch"),
    ],
)
def test_rejects_wrong_job_or_step(
    tmp_path: Path,
    raw_job: str,
    record_job: str,
    step: str,
):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["raw_rows"][0] = payload["raw_rows"][0].replace(f"{JOB_ID}.batch", raw_job)
    payload["records"][0]["job_id"] = record_job
    payload["records"][0]["step_id"] = step
    _assert_reason(_replace_payload(receipt, payload), "job_identity")


@pytest.mark.parametrize(("raw", "record"), [("FAILED", "FAILED"), ("COMPLETED", "FAILED")])
def test_rejects_noncompleted_or_mismatched_state(tmp_path: Path, raw: str, record: str):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["raw_rows"][0] = payload["raw_rows"][0].replace("COMPLETED", raw)
    payload["records"][0]["state"] = record
    _assert_reason(_replace_payload(receipt, payload), "state_identity")


@pytest.mark.parametrize(("raw", "record"), [("1:0", "1:0"), ("0:0", "1:0")])
def test_rejects_nonzero_or_mismatched_exit(tmp_path: Path, raw: str, record: str):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["raw_rows"][0] = payload["raw_rows"][0].replace("|0:0|", f"|{raw}|")
    payload["records"][0]["exit_code"] = record
    _assert_reason(_replace_payload(receipt, payload), "exit_identity")


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        ("elapsed_raw", "", "elapsed_raw"),
        ("elapsed_raw", " 00:01:44", "elapsed_raw"),
        ("max_rss_raw", "", "max_rss_raw"),
        ("max_rss_raw", "22854592K ", "max_rss_raw"),
    ],
)
def test_rejects_blank_or_noncanonical_raw_telemetry(
    tmp_path: Path,
    field: str,
    replacement: str,
    reason: str,
):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    original = payload["records"][0][field]
    payload["records"][0][field] = replacement
    payload["raw_rows"][0] = payload["raw_rows"][0].replace(original, replacement)
    _assert_reason(_replace_payload(receipt, payload), reason)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("elapsed_scope", "per_fit"),
        ("max_rss_scope", "accelerator"),
        ("separate_from_per_fit_python_telemetry", False),
        ("accelerator_memory_measured", True),
        ("unit_conversion_applied", True),
    ],
)
def test_rejects_telemetry_semantic_drift(tmp_path: Path, field: str, value: Any):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["records"][0][field] = value
    _assert_reason(_replace_payload(receipt, payload), "telemetry_semantics")


@pytest.mark.parametrize("row", ["bad", f"{JOB_ID}.batch|COMPLETED|0:0|00:01:44||", "x\n"])
def test_rejects_malformed_raw_row(tmp_path: Path, row: str):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["raw_rows"][0] = row
    reason = "max_rss_raw" if row.endswith("||") else "raw_row_format"
    _assert_reason(_replace_payload(receipt, payload), reason)


def test_rejects_duplicate_json_key(tmp_path: Path):
    receipt = _make_receipt(tmp_path)
    encoded = _encode(receipt.payload)
    duplicate = encoded.replace(
        b'"state":"COMPLETED"',
        b'"state":"COMPLETED","state":"COMPLETED"',
        1,
    )
    _assert_reason(_replace_bytes(receipt, duplicate), "duplicate_json_key")


def test_rejects_nonfinite_json_number(tmp_path: Path):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["extra"] = float("nan")
    _assert_reason(_replace_payload(receipt, payload, allow_nan=True), "nonfinite_number")


@pytest.mark.parametrize("digest", ["0" * 64, "A" * 64, "0" * 63])
def test_rejects_bad_or_mismatched_detached_hash(tmp_path: Path, digest: str):
    receipt = _make_receipt(tmp_path)
    receipt = replace(receipt, sha256=digest)
    reason = "detached_hash_mismatch" if digest == "0" * 64 else "expected_sha256"
    _assert_reason(receipt, reason)


@pytest.mark.parametrize("job_id", ["0", "01", "abc", "", "1.2"])
def test_rejects_noncanonical_expected_job_id(tmp_path: Path, job_id: str):
    receipt = replace(_make_receipt(tmp_path), job_id=job_id)
    _assert_reason(receipt, "expected_job_id")


def test_rejects_symlink(tmp_path: Path):
    receipt = _make_receipt(tmp_path)
    target = tmp_path / "target.json"
    target.write_bytes(receipt.path.read_bytes())
    receipt.path.unlink()
    receipt.path.symlink_to(target)
    _assert_reason(receipt, "link_rejected")


def test_rejects_hardlink(tmp_path: Path):
    receipt = _make_receipt(tmp_path)
    os.link(receipt.path, tmp_path / "second.json")
    _assert_reason(receipt, "hardlink_rejected")


def test_rejects_special_member(tmp_path: Path):
    receipt = _make_receipt(tmp_path)
    receipt.path.unlink()
    receipt.path.mkdir()
    _assert_reason(receipt, "special_receipt_rejected")


def test_rejects_receipt_replacement_during_read(tmp_path: Path):
    receipt = _make_receipt(tmp_path)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(receipt.path.read_bytes())

    def replace_receipt() -> None:
        replacement.replace(receipt.path)

    with pytest.raises(VERIFIER.VerificationError) as caught:
        _verify(receipt, after_read_hook=replace_receipt)
    assert caught.value.reason == "receipt_replaced"


def test_fifo_swap_between_stat_and_open_does_not_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    receipt = _make_receipt(tmp_path)
    real_open = os.open
    swapped = False

    def swap_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == receipt.path.name and not swapped:
            swapped = True
            receipt.path.unlink()
            os.mkfifo(receipt.path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(VERIFIER.os, "open", swap_then_open)
    _assert_reason(receipt, "receipt_replaced")
    assert swapped


def test_rejects_parent_directory_replacement_during_read(tmp_path: Path):
    parent = tmp_path / "receipt-root"
    parent.mkdir()
    receipt = _make_receipt(parent)
    detached = tmp_path / "detached-root"

    def replace_parent() -> None:
        parent.rename(detached)
        parent.mkdir()

    with pytest.raises(VERIFIER.VerificationError) as caught:
        _verify(receipt, after_read_hook=replace_parent)
    assert caught.value.reason == "directory_replaced"


def test_rejects_higher_ancestor_replacement_during_read(tmp_path: Path):
    outer = tmp_path / "outer"
    outer.mkdir()
    receipt = _make_receipt(outer)
    detached = tmp_path / "detached-outer"

    def replace_ancestor() -> None:
        outer.rename(detached)
        outer.mkdir()

    with pytest.raises(VERIFIER.VerificationError) as caught:
        _verify(receipt, after_read_hook=replace_ancestor)
    assert caught.value.reason == "directory_replaced"


def test_malformed_cli_output_never_echoes_supplied_value(tmp_path: Path):
    del tmp_path
    secret = "secret-cli-value-must-not-print"
    result = subprocess.run(
        [sys.executable, os.fspath(SCRIPT), "--unexpected", secret],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == ("FAIL check=ensemble_timing_scheduler_receipt reason=cli_arguments\n")
    assert secret not in result.stderr


def test_failure_output_discloses_no_values(tmp_path: Path):
    receipt = _make_receipt(tmp_path)
    payload = copy.deepcopy(receipt.payload)
    payload["records"][0]["state"] = "SECRET_STATE"
    receipt = _replace_payload(receipt, payload)

    result = _run(receipt)

    rendered = result.stdout + result.stderr
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "FAIL check=ensemble_timing_scheduler_receipt reason=state_identity\n"
    for forbidden in (
        os.fspath(receipt.path),
        receipt.sha256,
        receipt.job_id,
        "00:01:44",
        "22854592K",
        "SECRET_STATE",
    ):
        assert forbidden not in rendered
