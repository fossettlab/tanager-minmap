"""Tests for deterministic Slurm timing-receipt collection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "capture_ensemble_timing_scheduler_receipt.py"
VERIFIER_SCRIPT = SCRIPTS / "verify_ensemble_timing_scheduler_receipt.py"
JOB_ID = "2770999"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, os.fspath(SCRIPTS))
try:
    COLLECTOR = _load(SCRIPT, "capture_scheduler_receipt")
    VERIFIER = _load(VERIFIER_SCRIPT, "verify_scheduler_receipt_for_capture_test")
finally:
    sys.path.remove(os.fspath(SCRIPTS))


def _raw(
    *,
    job_id: str = JOB_ID,
    state: str = "COMPLETED",
    exit_code: str = "0:0",
    elapsed: str = "00:01:44",
    max_rss: str = "22854592K",
) -> bytes:
    return f"{job_id}.batch|{state}|{exit_code}|{elapsed}|{max_rss}|\n".encode()


def _make_raw(tmp_path: Path, payload: bytes | None = None) -> Path:
    path = tmp_path / "sacct.raw"
    path.write_bytes(_raw() if payload is None else payload)
    return path


def _capture(tmp_path: Path, payload: bytes | None = None):
    raw = _make_raw(tmp_path, payload)
    output = tmp_path / "receipt.json"
    digest = COLLECTOR.capture_scheduler_receipt(raw, output, expected_job_id=JOB_ID)
    return raw, output, digest


def test_capture_is_deterministic_and_verifier_accepts_it(tmp_path: Path):
    _raw_path, output, digest = _capture(tmp_path)

    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    VERIFIER.verify_scheduler_receipt(
        output,
        expected_receipt_sha256=digest,
        expected_job_id=JOB_ID,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["raw_rows"] == [_raw().decode().removesuffix("\n")]
    assert payload["records"][0]["elapsed_raw"] == "00:01:44"
    assert payload["records"][0]["max_rss_raw"] == "22854592K"
    assert payload["records"][0]["unit_conversion_applied"] is False


def test_same_raw_row_produces_same_bytes(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _a_raw, a_output, a_digest = _capture(first)
    _b_raw, b_output, b_digest = _capture(second)

    assert a_output.read_bytes() == b_output.read_bytes()
    assert a_digest == b_digest


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_raw() + _raw(), "raw_input_format"),
        (_raw().rstrip(b"\n"), "raw_input_format"),
        (_raw(job_id="999"), "job_identity"),
        (_raw(state="FAILED"), "state_identity"),
        (_raw(exit_code="1:0"), "exit_identity"),
        (_raw(elapsed=""), "elapsed_raw"),
        (_raw(max_rss=""), "max_rss_raw"),
    ],
)
def test_rejects_ambiguous_or_invalid_raw_rows(
    tmp_path: Path,
    payload: bytes,
    reason: str,
):
    raw = _make_raw(tmp_path, payload)

    with pytest.raises(COLLECTOR.CollectionError) as caught:
        COLLECTOR.capture_scheduler_receipt(
            raw,
            tmp_path / "receipt.json",
            expected_job_id=JOB_ID,
        )
    assert caught.value.reason == reason


@pytest.mark.parametrize("job_id", ["", "0", "01", "abc", "1.2"])
def test_rejects_noncanonical_expected_job_id(tmp_path: Path, job_id: str):
    raw = _make_raw(tmp_path)

    with pytest.raises(COLLECTOR.CollectionError) as caught:
        COLLECTOR.capture_scheduler_receipt(
            raw,
            tmp_path / "receipt.json",
            expected_job_id=job_id,
        )
    assert caught.value.reason == "expected_job_id"


def test_never_overwrites_existing_output(tmp_path: Path):
    raw = _make_raw(tmp_path)
    output = tmp_path / "receipt.json"
    output.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(COLLECTOR.CollectionError) as caught:
        COLLECTOR.capture_scheduler_receipt(raw, output, expected_job_id=JOB_ID)
    assert caught.value.reason == "output_exists"
    assert output.read_text(encoding="utf-8") == "preserve me\n"


def test_write_failure_leaves_no_final_or_temporary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = _make_raw(tmp_path)
    output = tmp_path / "receipt.json"

    def fail_write(_descriptor, _payload):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(COLLECTOR.os, "write", fail_write)
    with pytest.raises(COLLECTOR.CollectionError) as caught:
        COLLECTOR.capture_scheduler_receipt(raw, output, expected_job_id=JOB_ID)

    assert caught.value.reason == "output_write_failed"
    assert not output.exists()
    assert not list(tmp_path.glob(".receipt.json.part-*"))


def test_rejects_symlinked_raw_input(tmp_path: Path):
    target = _make_raw(tmp_path)
    link = tmp_path / "linked.raw"
    link.symlink_to(target)

    with pytest.raises(COLLECTOR.CollectionError) as caught:
        COLLECTOR.capture_scheduler_receipt(
            link,
            tmp_path / "receipt.json",
            expected_job_id=JOB_ID,
        )
    assert caught.value.reason == "raw_input_invalid"


def test_rejects_hardlinked_raw_input(tmp_path: Path):
    raw = _make_raw(tmp_path)
    os.link(raw, tmp_path / "second.raw")

    with pytest.raises(COLLECTOR.CollectionError) as caught:
        COLLECTOR.capture_scheduler_receipt(
            raw,
            tmp_path / "receipt.json",
            expected_job_id=JOB_ID,
        )
    assert caught.value.reason == "raw_input_invalid"


def test_cli_success_is_low_disclosure(tmp_path: Path):
    raw = _make_raw(tmp_path)
    output = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            os.fspath(SCRIPT),
            "--raw-sacct",
            os.fspath(raw),
            "--output",
            os.fspath(output),
            "--expected-job-id",
            JOB_ID,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "PASS check=capture_ensemble_timing_scheduler_receipt\n"
    assert result.stderr == ""
    assert JOB_ID not in result.stdout
    assert "22854592K" not in result.stdout


def test_malformed_cli_never_echoes_value(tmp_path: Path):
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
    assert result.stderr == (
        "FAIL check=capture_ensemble_timing_scheduler_receipt reason=cli_arguments\n"
    )
    assert secret not in result.stderr
