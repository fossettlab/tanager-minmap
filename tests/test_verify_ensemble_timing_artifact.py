"""Tests for the standalone endpoint-blind E6-v6 timing verifier."""

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
SCRIPT = ROOT / "scripts" / "verify_ensemble_timing_artifact.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_ensemble_timing_artifact", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()


@dataclass(frozen=True)
class Capsule:
    run_dir: Path
    design_sha256: str
    members_sha256: str
    timing_sha256: str
    payload: dict[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(
    *,
    site: str,
    scene: str,
    fit_id: str,
    member_class: str,
    replicate: int | None,
) -> dict[str, Any]:
    return {
        "site": site,
        "scene": scene,
        "fit_id": fit_id,
        "member_class": member_class,
        "stochastic_replicate": replicate,
        "wall_time_seconds": 0,
        "peak_memory_bytes": 0,
        "output_sha256": _sha256(fit_id.encode("utf-8")),
        "device": "cpu",
        "scientific_outputs_retained": False,
    }


def _valid_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "mode": "timing_pilot_only",
        "fit_count": 4,
        "records": [
            _record(
                site="goldfield",
                scene="20240925_185504_87_4001",
                fit_id="goldfield:fit:baseline:r0.01",
                member_class="baseline",
                replicate=None,
            ),
            _record(
                site="goldfield",
                scene="20240925_185504_87_4001",
                fit_id="goldfield:fit:joint:r00:ridge0.01",
                member_class="joint",
                replicate=0,
            ),
            _record(
                site="bingham",
                scene="20250911_191523_58_4001",
                fit_id="bingham:fit:baseline:r0.01",
                member_class="baseline",
                replicate=None,
            ),
            _record(
                site="bingham",
                scene="20250911_191523_58_4001",
                fit_id="bingham:fit:joint:r00:ridge0.01",
                member_class="joint",
                replicate=0,
            ),
        ],
    }


def _timing_bytes(payload: dict[str, Any], *, allow_nan: bool = False) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=allow_nan,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _make_capsule(tmp_path: Path) -> Capsule:
    run_dir = tmp_path / "v6-output"
    run_dir.mkdir()
    design = b"opaque design content\n"
    members = b"opaque member content\n"
    payload = _valid_payload()
    timing = _timing_bytes(payload)
    (run_dir / "design.json").write_bytes(design)
    (run_dir / "members.csv").write_bytes(members)
    (run_dir / "timing_pilot.json").write_bytes(timing)
    return Capsule(
        run_dir=run_dir,
        design_sha256=_sha256(design),
        members_sha256=_sha256(members),
        timing_sha256=_sha256(timing),
        payload=payload,
    )


def _replace_timing(
    capsule: Capsule,
    payload: dict[str, Any],
    *,
    allow_nan: bool = False,
) -> Capsule:
    timing = _timing_bytes(payload, allow_nan=allow_nan)
    (capsule.run_dir / "timing_pilot.json").write_bytes(timing)
    return replace(capsule, timing_sha256=_sha256(timing), payload=payload)


def _replace_timing_bytes(capsule: Capsule, timing: bytes) -> Capsule:
    (capsule.run_dir / "timing_pilot.json").write_bytes(timing)
    return replace(capsule, timing_sha256=_sha256(timing))


def _verify(capsule: Capsule, *, after_read_hook=None) -> None:
    VERIFIER.verify_ensemble_timing_artifact(
        capsule.run_dir,
        expected_design_sha256=capsule.design_sha256,
        expected_members_sha256=capsule.members_sha256,
        expected_timing_sha256=capsule.timing_sha256,
        after_read_hook=after_read_hook,
    )


def _run(capsule: Capsule) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            os.fspath(SCRIPT),
            "--run-dir",
            os.fspath(capsule.run_dir),
            "--expected-design-sha256",
            capsule.design_sha256,
            "--expected-members-sha256",
            capsule.members_sha256,
            "--expected-timing-sha256",
            capsule.timing_sha256,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_reason(capsule: Capsule, reason: str) -> None:
    with pytest.raises(VERIFIER.VerificationError) as caught:
        _verify(capsule)
    assert caught.value.reason == reason


def test_valid_closed_timing_artifact_passes(tmp_path: Path):
    capsule = _make_capsule(tmp_path)

    _verify(capsule)
    result = _run(capsule)

    assert result.returncode == 0
    assert result.stdout == "PASS check=ensemble_timing_artifact\n"
    assert result.stderr == ""


def test_verifier_imports_only_the_standard_library():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_roots.add((node.module or "").split(".", maxsplit=1)[0])

    assert imported_roots <= {
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


def test_rejects_unexpected_shallow_member(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    (capsule.run_dir / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

    _assert_reason(capsule, "membership_mismatch")


def test_rejects_missing_shallow_member(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    (capsule.run_dir / "members.csv").unlink()

    _assert_reason(capsule, "membership_mismatch")


def test_rejects_symlinked_member(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    target = tmp_path / "timing-target.json"
    target.write_bytes((capsule.run_dir / "timing_pilot.json").read_bytes())
    (capsule.run_dir / "timing_pilot.json").unlink()
    (capsule.run_dir / "timing_pilot.json").symlink_to(target)

    _assert_reason(capsule, "link_rejected")


def test_rejects_symlinked_run_directory(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    link = tmp_path / "linked-output"
    link.symlink_to(capsule.run_dir, target_is_directory=True)
    linked = replace(capsule, run_dir=link)

    _assert_reason(linked, "link_rejected")


def test_rejects_hardlinked_member(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    os.link(capsule.run_dir / "members.csv", tmp_path / "second-members-link.csv")

    _assert_reason(capsule, "hardlink_rejected")


def test_rejects_directory_in_place_of_member(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    (capsule.run_dir / "timing_pilot.json").unlink()
    (capsule.run_dir / "timing_pilot.json").mkdir()

    _assert_reason(capsule, "special_member_rejected")


@pytest.mark.parametrize("field", ["schema_version", "mode", "fit_count", "records"])
def test_requires_every_exact_top_level_field(tmp_path: Path, field: str):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    del payload[field]
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "top_level_fields")


def test_rejects_extra_top_level_field(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["extra"] = None
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "top_level_fields")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "1"),
        ("mode", "timing"),
        ("fit_count", 3),
        ("fit_count", True),
    ],
)
def test_rejects_wrong_top_level_identity(tmp_path: Path, field: str, value: Any):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload[field] = value
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "top_level_identity")


@pytest.mark.parametrize("field", sorted(VERIFIER.EXPECTED_RECORD_FIELDS))
def test_requires_every_exact_record_field(tmp_path: Path, field: str):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    del payload["records"][0][field]
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "record_fields")


def test_rejects_extra_record_field(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"][0]["extra"] = None
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "record_fields")


@pytest.mark.parametrize("records", [[], {}, [None, None, None, None]])
def test_rejects_record_count_or_container_type(tmp_path: Path, records: Any):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"] = records
    capsule = _replace_timing(capsule, payload)

    reason = "record_fields" if type(records) is list and len(records) == 4 else "record_count"
    _assert_reason(capsule, reason)


def test_rejects_unknown_fit_identity(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"][0]["fit_id"] = "unknown:fit"
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "fit_identity")


def test_rejects_duplicate_fit_identity(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"][-1] = copy.deepcopy(payload["records"][0])
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "fit_identity")


@pytest.mark.parametrize(
    ("record_index", "field", "value"),
    [
        (0, "site", "bingham"),
        (0, "scene", "wrong-scene"),
        (0, "member_class", "joint"),
        (0, "stochastic_replicate", 0),
        (1, "stochastic_replicate", 0.0),
        (1, "stochastic_replicate", "0"),
        (1, "stochastic_replicate", False),
        (1, "stochastic_replicate", None),
    ],
)
def test_rejects_wrong_record_identity_or_noninteger_replicate(
    tmp_path: Path,
    record_index: int,
    field: str,
    value: Any,
):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"][record_index][field] = value
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "record_identity")


@pytest.mark.parametrize("value", [True, "0", -1])
def test_rejects_wall_time_bool_type_or_negative(tmp_path: Path, value: Any):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"][0]["wall_time_seconds"] = value
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "wall_time_type_or_range")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_json_numbers(tmp_path: Path, value: float):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"][0]["wall_time_seconds"] = value
    capsule = _replace_timing(capsule, payload, allow_nan=True)

    _assert_reason(capsule, "nonfinite_number")


@pytest.mark.parametrize("value", [True, 0.0, "0", -1])
def test_rejects_peak_memory_bool_noninteger_or_negative(tmp_path: Path, value: Any):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"][0]["peak_memory_bytes"] = value
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "peak_memory_type_or_range")


@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "g" * 64, None])
def test_rejects_noncanonical_output_sha256(tmp_path: Path, value: Any):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"][0]["output_sha256"] = value
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, "output_sha256")


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("device", "gpu", "device_identity"),
        ("device", None, "device_identity"),
        ("scientific_outputs_retained", True, "retention_identity"),
        ("scientific_outputs_retained", 0, "retention_identity"),
    ],
)
def test_rejects_device_or_retention_drift(
    tmp_path: Path,
    field: str,
    value: Any,
    reason: str,
):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    payload["records"][0][field] = value
    capsule = _replace_timing(capsule, payload)

    _assert_reason(capsule, reason)


def test_rejects_duplicate_json_keys(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    timing = _timing_bytes(capsule.payload)
    duplicate = timing.replace(b'"device":"cpu"', b'"device":"cpu","device":"cpu"', 1)
    assert duplicate != timing
    capsule = _replace_timing_bytes(capsule, duplicate)

    _assert_reason(capsule, "duplicate_json_key")


def test_rejects_invalid_json(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    capsule = _replace_timing_bytes(capsule, b"not-json\n")

    _assert_reason(capsule, "invalid_json")


@pytest.mark.parametrize("digest_name", ["design_sha256", "members_sha256", "timing_sha256"])
def test_rejects_each_detached_hash_mismatch(tmp_path: Path, digest_name: str):
    capsule = _make_capsule(tmp_path)
    capsule = replace(capsule, **{digest_name: "0" * 64})

    _assert_reason(capsule, "detached_hash_mismatch")


@pytest.mark.parametrize("bad_digest", ["A" * 64, "0" * 63, "g" * 64])
def test_rejects_noncanonical_expected_sha256(tmp_path: Path, bad_digest: str):
    capsule = _make_capsule(tmp_path)
    capsule = replace(capsule, timing_sha256=bad_digest)

    _assert_reason(capsule, "expected_sha256")


def test_rejects_member_replacement_during_read(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    replacement_path = tmp_path / "replacement-design.json"
    replacement_path.write_bytes((capsule.run_dir / "design.json").read_bytes())

    def replace_member(name: str) -> None:
        if name == "design.json":
            replacement_path.replace(capsule.run_dir / name)

    with pytest.raises(VERIFIER.VerificationError) as caught:
        _verify(capsule, after_read_hook=replace_member)
    assert caught.value.reason == "member_replaced"


def test_fifo_swap_between_stat_and_open_does_not_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    capsule = _make_capsule(tmp_path)
    real_open = os.open
    swapped = False

    def swap_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "timing_pilot.json" and not swapped:
            swapped = True
            target = capsule.run_dir / "timing_pilot.json"
            target.unlink()
            os.mkfifo(target)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(VERIFIER.os, "open", swap_then_open)
    _assert_reason(capsule, "member_replaced")
    assert swapped


def test_rejects_run_directory_replacement_during_read(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    detached = tmp_path / "detached-output"

    def replace_directory(name: str) -> None:
        if name == "design.json":
            capsule.run_dir.rename(detached)
            capsule.run_dir.mkdir()

    with pytest.raises(VERIFIER.VerificationError) as caught:
        _verify(capsule, after_read_hook=replace_directory)
    assert caught.value.reason == "directory_replaced"


def test_rejects_higher_ancestor_replacement_during_read(tmp_path: Path):
    outer = tmp_path / "outer"
    outer.mkdir()
    capsule = _make_capsule(outer)
    detached = tmp_path / "detached-outer"

    def replace_ancestor(name: str) -> None:
        if name == "design.json":
            outer.rename(detached)
            outer.mkdir()

    with pytest.raises(VERIFIER.VerificationError) as caught:
        _verify(capsule, after_read_hook=replace_ancestor)
    assert caught.value.reason == "directory_replaced"


def test_malformed_cli_output_never_echoes_supplied_value(tmp_path: Path):
    secret = "secret-cli-value-must-not-print"
    result = subprocess.run(
        [sys.executable, os.fspath(SCRIPT), "--unexpected", secret],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "FAIL check=ensemble_timing_artifact reason=cli_arguments\n"
    assert secret not in result.stderr


def test_failure_output_never_discloses_values_hashes_fit_ids_or_content(tmp_path: Path):
    capsule = _make_capsule(tmp_path)
    payload = copy.deepcopy(capsule.payload)
    secret_fit = "operational-fit-identity-must-not-print"
    payload["records"][0]["fit_id"] = secret_fit
    capsule = _replace_timing(capsule, payload)

    result = _run(capsule)

    rendered = result.stdout + result.stderr
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "FAIL check=ensemble_timing_artifact reason=fit_identity\n"
    for forbidden in (
        os.fspath(capsule.run_dir),
        capsule.design_sha256,
        capsule.members_sha256,
        capsule.timing_sha256,
        secret_fit,
        "opaque design content",
        "opaque member content",
    ):
        assert forbidden not in rendered
