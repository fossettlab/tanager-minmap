"""Tests for atomic parent creation plus bounded source recovery."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_source_recovery_with_parent.py"
REVIEWED_HELPER = ROOT / "scripts" / "verify_source_recovery.py"


def _load_transaction() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "install_source_recovery_with_parent",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRANSACTION = _load_transaction()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _workspace(base: Path) -> tuple[Path, Path]:
    project = base / "tanager-rocks"
    sibling = base / "tanager-spec"
    project.mkdir(parents=True)
    sibling.mkdir()
    return project, sibling


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    source, _source_sibling = _workspace(tmp_path / "source")
    destination, _destination_sibling = _workspace(tmp_path / "destination")
    (source / "docs" / "operational").mkdir(parents=True)
    (source / "docs" / "operational" / "bundle.bin").write_bytes(b"bundle\n")
    (source / "docs" / "plan.md").write_bytes(b"plan\n")
    (destination / "docs").mkdir()
    entries = [
        ("docs/operational/bundle.bin", _sha(b"bundle\n")),
        ("docs/plan.md", _sha(b"plan\n")),
    ]
    payload = "".join(f"{digest}  {path}\n" for path, digest in entries).encode()
    manifest = tmp_path / "manifest.sha256"
    manifest.write_bytes(payload)
    return source, destination, manifest, _sha(payload)


def test_parent_creation_and_install_are_one_verified_transaction(tmp_path: Path):
    source, destination, manifest, manifest_sha256 = _prepare(tmp_path)

    observed_sha256, count = TRANSACTION.install_source_recovery_with_parent(
        manifest=manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_entry_count=2,
        source_project_root=source,
        destination_project_root=destination,
        create_project_parent="docs/operational",
        reviewed_helper=REVIEWED_HELPER,
    )

    assert observed_sha256 == manifest_sha256
    assert count == 2
    assert (destination / "docs" / "operational" / "bundle.bin").read_bytes() == b"bundle\n"
    assert (destination / "docs" / "plan.md").read_bytes() == b"plan\n"


def test_parent_replacement_immediately_after_creation_is_rejected(tmp_path: Path):
    source, destination, manifest, manifest_sha256 = _prepare(tmp_path)

    def replace_parent(path: str) -> None:
        assert path == "docs/operational"
        created = destination / path
        created.rename(destination / "docs" / "operational-original")
        created.mkdir()

    with pytest.raises(Exception, match="directory_changed"):
        TRANSACTION.install_source_recovery_with_parent(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=2,
            source_project_root=source,
            destination_project_root=destination,
            create_project_parent="docs/operational",
            reviewed_helper=REVIEWED_HELPER,
            after_parent_create_hook=replace_parent,
        )

    assert not (destination / "docs" / "operational" / "bundle.bin").exists()
    assert not (destination / "docs" / "operational-original" / "bundle.bin").exists()
    assert not (destination / "docs" / "plan.md").exists()


def test_parent_replacement_before_first_file_create_is_rejected(tmp_path: Path):
    source, destination, manifest, manifest_sha256 = _prepare(tmp_path)

    def replace_parent(path: str) -> None:
        if path == "docs/operational/bundle.bin":
            created = destination / "docs" / "operational"
            created.rename(destination / "docs" / "operational-original")
            created.mkdir()

    with pytest.raises(Exception, match="directory_changed"):
        TRANSACTION.install_source_recovery_with_parent(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=2,
            source_project_root=source,
            destination_project_root=destination,
            create_project_parent="docs/operational",
            reviewed_helper=REVIEWED_HELPER,
            before_destination_create_hook=replace_parent,
        )

    assert not (destination / "docs" / "operational" / "bundle.bin").exists()
    assert not (destination / "docs" / "operational-original" / "bundle.bin").exists()
    assert not (destination / "docs" / "plan.md").exists()


@pytest.mark.parametrize(
    "parent",
    ["../tanager-spec/src", "docs", "docs/operational/nested"],
)
def test_parent_must_be_one_direct_project_parent(tmp_path: Path, parent: str):
    source, destination, manifest, manifest_sha256 = _prepare(tmp_path)

    with pytest.raises(Exception):
        TRANSACTION.install_source_recovery_with_parent(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=2,
            source_project_root=source,
            destination_project_root=destination,
            create_project_parent=parent,
            reviewed_helper=REVIEWED_HELPER,
        )


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "wrong_digest"])
def test_reviewed_helper_identity_attacks_are_rejected(tmp_path: Path, attack: str):
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
        TRANSACTION._load_reviewed_helper(candidate)
