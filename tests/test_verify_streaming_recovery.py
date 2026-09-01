"""Tests for the hash-bound streaming recovery adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_streaming_recovery.py"
REVIEWED_HELPER = ROOT / "scripts" / "verify_source_recovery.py"


def _load_adapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_streaming_recovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load_adapter()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _workspace(base: Path) -> tuple[Path, Path]:
    project = base / "tanager-rocks"
    sibling = base / "tanager-spec"
    project.mkdir(parents=True)
    sibling.mkdir()
    return project, sibling


def _manifest(path: Path, relative: str, payload: bytes) -> str:
    content = f"{_sha(payload)}  {relative}\n".encode()
    path.write_bytes(content)
    return _sha(content)


def test_member_reads_are_streamed_but_manifest_is_captured(tmp_path: Path):
    project, _sibling = _workspace(tmp_path / "source")
    member_payload = b"streamed-member\n"
    (project / "member.bin").write_bytes(member_payload)
    manifest = tmp_path / "manifest.sha256"
    manifest_sha256 = _manifest(manifest, "member.bin", member_payload)
    helper = ADAPTER.load_streaming_helper(REVIEWED_HELPER)
    observed: list[tuple[str, bytes | None]] = []
    streaming_read = helper._read_descriptor

    def spy(*args: object, **kwargs: object) -> tuple[bytes | None, str]:
        payload, digest = streaming_read(*args, **kwargs)
        observed.append((str(kwargs["check"]), payload))
        return payload, digest

    helper._read_descriptor = spy
    helper.verify_source_recovery(
        manifest=manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_entry_count=1,
        project_root=project,
    )

    assert observed[0] == ("manifest_read", manifest.read_bytes())
    assert observed[1] == ("source_member", None)


def test_streaming_adapter_preserves_install_and_postverify(tmp_path: Path):
    source, _source_sibling = _workspace(tmp_path / "source")
    destination, _destination_sibling = _workspace(tmp_path / "destination")
    (source / "data" / "raw").mkdir(parents=True)
    (destination / "data" / "raw").mkdir(parents=True)
    payload = b"input-bytes\n"
    relative = "data/raw/input.bin"
    (source / relative).write_bytes(payload)
    manifest = tmp_path / "manifest.sha256"
    manifest_sha256 = _manifest(manifest, relative, payload)
    helper = ADAPTER.load_streaming_helper(REVIEWED_HELPER)

    observed_sha256, count = helper.install_source_recovery(
        manifest=manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_entry_count=1,
        source_project_root=source,
        destination_project_root=destination,
    )

    assert observed_sha256 == manifest_sha256
    assert count == 1
    assert (destination / relative).read_bytes() == payload


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "wrong_digest"])
def test_reviewed_helper_identity_attacks_are_rejected(
    tmp_path: Path,
    attack: str,
):
    copied = tmp_path / "verify_source_recovery.py"
    copied.write_bytes(REVIEWED_HELPER.read_bytes())
    candidate = copied
    if attack == "symlink":
        candidate = tmp_path / "linked.py"
        candidate.symlink_to(copied)
    elif attack == "hardlink":
        candidate = tmp_path / "hardlinked.py"
        os.link(copied, candidate)
    else:
        copied.write_bytes(copied.read_bytes() + b"\n")

    with pytest.raises(RuntimeError):
        ADAPTER.load_streaming_helper(candidate)
