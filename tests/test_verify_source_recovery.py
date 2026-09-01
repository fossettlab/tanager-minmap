"""Tests for the bounded source-capsule recovery helper."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_source_recovery.py"


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_source_recovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPER = _load_helper()


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_manifest(manifest: Path, entries: Iterable[tuple[str, str]]) -> str:
    payload = "".join(f"{digest}  {path}\n" for path, digest in entries)
    manifest.write_text(payload, encoding="utf-8")
    return _sha_bytes(payload.encode())


def _workspace(base: Path) -> tuple[Path, Path]:
    project = base / "tanager-rocks"
    sibling = base / "tanager-spec"
    project.mkdir(parents=True)
    sibling.mkdir()
    return project, sibling


def _member(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _prepare_pair(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, str, list[tuple[str, str]]]:
    source_project, source_sibling = _workspace(tmp_path / "source")
    destination_project, destination_sibling = _workspace(tmp_path / "destination")
    project_member = _member(source_project, "scripts/operation.py", b"project\n")
    sibling_member = _member(
        source_sibling,
        "src/tanager_spec/contract.py",
        b"sibling\n",
    )
    (destination_project / "scripts").mkdir()
    (destination_sibling / "src" / "tanager_spec").mkdir(parents=True)
    entries = [
        (
            "../tanager-spec/src/tanager_spec/contract.py",
            _sha_file(sibling_member),
        ),
        ("scripts/operation.py", _sha_file(project_member)),
    ]
    manifest = tmp_path / "recovery.sha256"
    manifest_sha256 = _write_manifest(manifest, entries)
    return (
        source_project,
        source_sibling,
        destination_project,
        destination_sibling,
        manifest,
        manifest_sha256,
        entries,
    )


def _run_verify(
    manifest: Path,
    manifest_sha256: str,
    count: int,
    project_root: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            os.fspath(SCRIPT),
            "verify",
            "--manifest",
            os.fspath(manifest),
            "--expected-manifest-sha256",
            manifest_sha256,
            "--expected-entry-count",
            str(count),
            "--project-root",
            os.fspath(project_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_verify_and_install_across_both_namespaces(tmp_path: Path):
    (
        source_project,
        _source_sibling,
        destination_project,
        destination_sibling,
        manifest,
        manifest_sha256,
        _entries,
    ) = _prepare_pair(tmp_path)

    verify_result = _run_verify(manifest, manifest_sha256, 2, source_project)
    installed_sha256, installed_count = HELPER.install_source_recovery(
        manifest=manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_entry_count=2,
        source_project_root=source_project,
        destination_project_root=destination_project,
    )

    assert verify_result.returncode == 0
    assert "mode=verify" in verify_result.stdout
    assert installed_sha256 == manifest_sha256
    assert installed_count == 2
    assert (destination_project / "scripts" / "operation.py").read_bytes() == b"project\n"
    assert (
        destination_sibling / "src" / "tanager_spec" / "contract.py"
    ).read_bytes() == b"sibling\n"


@pytest.mark.parametrize(
    "payload",
    [
        b"not a record\n",
        f"{'0' * 64}  /absolute.py\n".encode(),
        f"{'0' * 64}  ../outside.py\n".encode(),
        f"{'0' * 64}  nested/../outside.py\n".encode(),
        f"{'0' * 64}  folder\\member.py\n".encode(),
        f"{'0' * 64}  ./member.py\n".encode(),
        f"{'0' * 64}  a.py\r\n".encode(),
        f"{'0' * 64}  a.py".encode(),
        (f"{'0' * 64}  a.py\n" * 2).encode(),
        (f"{'0' * 64}  b.py\n{'0' * 64}  a.py\n").encode(),
    ],
)
def test_noncanonical_manifests_are_rejected(tmp_path: Path, payload: bytes):
    project, _sibling = _workspace(tmp_path / "source")
    manifest = tmp_path / "bad.sha256"
    manifest.write_bytes(payload)

    result = _run_verify(
        manifest,
        _sha_bytes(payload),
        len(payload.splitlines()),
        project,
    )

    assert result.returncode != 0
    assert result.stdout == ""


@pytest.mark.parametrize("fault", ["manifest_digest", "count", "member_digest"])
def test_digest_and_count_mismatches_are_rejected(tmp_path: Path, fault: str):
    project, _sibling = _workspace(tmp_path / "source")
    member = _member(project, "member.py", b"member\n")
    digest = "0" * 64 if fault == "member_digest" else _sha_file(member)
    manifest = tmp_path / "manifest.sha256"
    manifest_sha256 = _write_manifest(manifest, [("member.py", digest)])

    result = _run_verify(
        manifest,
        "f" * 64 if fault == "manifest_digest" else manifest_sha256,
        2 if fault == "count" else 1,
        project,
    )

    assert result.returncode != 0


@pytest.mark.parametrize("attack", ["final_symlink", "hardlink", "fifo", "parent_symlink"])
def test_source_link_and_type_attacks_are_rejected(tmp_path: Path, attack: str):
    project, _sibling = _workspace(tmp_path / "source")
    real = _member(project, "real/member.py", b"source\n")
    logical = "member.py"
    if attack == "final_symlink":
        (project / logical).symlink_to(real)
    elif attack == "hardlink":
        os.link(real, project / logical)
    elif attack == "fifo":
        os.mkfifo(project / logical)
    else:
        (project / "linked").symlink_to(real.parent, target_is_directory=True)
        logical = "linked/member.py"
    manifest = tmp_path / "manifest.sha256"
    manifest_sha256 = _write_manifest(manifest, [(logical, _sha_file(real))])

    result = _run_verify(manifest, manifest_sha256, 1, project)

    assert result.returncode != 0
    if attack == "hardlink":
        assert "reason=multiple_links" in result.stderr


@pytest.mark.parametrize("attack", ["file", "symlink", "parent_symlink"])
def test_existing_or_linked_destination_is_rejected(tmp_path: Path, attack: str):
    (
        source_project,
        _source_sibling,
        destination_project,
        _destination_sibling,
        manifest,
        manifest_sha256,
        _entries,
    ) = _prepare_pair(tmp_path)
    destination = destination_project / "scripts" / "operation.py"
    if attack == "file":
        destination.write_bytes(b"existing\n")
    elif attack == "symlink":
        destination.symlink_to(source_project / "scripts" / "operation.py")
    else:
        (destination_project / "scripts").rmdir()
        (destination_project / "scripts").symlink_to(
            source_project / "scripts",
            target_is_directory=True,
        )

    with pytest.raises(HELPER.RecoveryError):
        HELPER.install_source_recovery(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=2,
            source_project_root=source_project,
            destination_project_root=destination_project,
        )


def test_source_mutation_during_verify_is_rejected(tmp_path: Path):
    project, _sibling = _workspace(tmp_path / "source")
    member = _member(project, "member.py", b"source\n")
    manifest = tmp_path / "manifest.sha256"
    manifest_sha256 = _write_manifest(
        manifest,
        [("member.py", _sha_file(member))],
    )

    def mutate(path: str) -> None:
        if path == "member.py":
            member.write_bytes(b"changed\n")

    with pytest.raises(HELPER.RecoveryError, match="file_changed"):
        HELPER.verify_source_recovery(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=1,
            project_root=project,
            after_source_read_hook=mutate,
        )


def test_bound_source_directory_replacement_after_read_is_rejected(tmp_path: Path):
    project, _sibling = _workspace(tmp_path / "source")
    member = _member(project, "nested/member.py", b"source\n")
    manifest = tmp_path / "manifest.sha256"
    manifest_sha256 = _write_manifest(
        manifest,
        [("nested/member.py", _sha_file(member))],
    )

    def replace_directory(path: str) -> None:
        if path == "nested/member.py":
            (project / "nested").rename(project / "nested-original")
            replacement = project / "nested"
            replacement.mkdir()
            (replacement / "member.py").write_bytes(b"source\n")

    with pytest.raises(HELPER.RecoveryError):
        HELPER.verify_source_recovery(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=1,
            project_root=project,
            after_source_read_hook=replace_directory,
        )


def test_source_mutation_between_preflight_and_copy_is_rejected(tmp_path: Path):
    (
        source_project,
        _source_sibling,
        destination_project,
        _destination_sibling,
        manifest,
        manifest_sha256,
        _entries,
    ) = _prepare_pair(tmp_path)
    member = source_project / "scripts" / "operation.py"

    def mutate(path: str) -> None:
        if path == "scripts/operation.py":
            member.write_bytes(b"changed\n")

    with pytest.raises(HELPER.RecoveryError):
        HELPER.install_source_recovery(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=2,
            source_project_root=source_project,
            destination_project_root=destination_project,
            before_destination_create_hook=mutate,
        )


def test_destination_race_is_rejected_by_exclusive_create(tmp_path: Path):
    (
        source_project,
        _source_sibling,
        destination_project,
        _destination_sibling,
        manifest,
        manifest_sha256,
        _entries,
    ) = _prepare_pair(tmp_path)
    raced = destination_project / "scripts" / "operation.py"

    def create_race(path: str) -> None:
        if path == "scripts/operation.py":
            raced.write_bytes(b"raced\n")

    with pytest.raises(HELPER.RecoveryError, match="exclusive_create_failed"):
        HELPER.install_source_recovery(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=2,
            source_project_root=source_project,
            destination_project_root=destination_project,
            before_destination_create_hook=create_race,
        )
    assert raced.read_bytes() == b"raced\n"


def test_destination_tampering_before_post_install_verify_is_rejected(tmp_path: Path):
    (
        source_project,
        _source_sibling,
        destination_project,
        _destination_sibling,
        manifest,
        manifest_sha256,
        _entries,
    ) = _prepare_pair(tmp_path)
    installed = destination_project / "scripts" / "operation.py"

    def tamper(path: str, index: int) -> None:
        if path == "scripts/operation.py" and index == 2:
            installed.write_bytes(b"tampered\n")

    with pytest.raises(HELPER.RecoveryError, match="digest_mismatch"):
        HELPER.install_source_recovery(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=2,
            source_project_root=source_project,
            destination_project_root=destination_project,
            after_install_hook=tamper,
        )


def test_partial_failure_preserves_prior_correct_member(tmp_path: Path):
    source_project, _source_sibling = _workspace(tmp_path / "source")
    destination_project, _destination_sibling = _workspace(tmp_path / "destination")
    first = _member(source_project, "a.py", b"a\n")
    second = _member(source_project, "b.py", b"b\n")
    manifest = tmp_path / "manifest.sha256"
    entries = [("a.py", _sha_file(first)), ("b.py", _sha_file(second))]
    manifest_sha256 = _write_manifest(manifest, entries)

    def race_second(path: str) -> None:
        if path == "b.py":
            (destination_project / "b.py").write_bytes(b"raced\n")

    with pytest.raises(HELPER.RecoveryError, match="exclusive_create_failed"):
        HELPER.install_source_recovery(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=2,
            source_project_root=source_project,
            destination_project_root=destination_project,
            before_destination_create_hook=race_second,
        )
    assert (destination_project / "a.py").read_bytes() == b"a\n"
    assert (destination_project / "b.py").read_bytes() == b"raced\n"


def test_source_code_is_hashed_but_never_imported_or_executed(tmp_path: Path):
    project, _sibling = _workspace(tmp_path / "source")
    destination_project, _destination_sibling = _workspace(tmp_path / "destination")
    (destination_project / "tanager_rocks").mkdir()
    marker = tmp_path / "executed"
    payload = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n".encode()
    member = _member(project, "tanager_rocks/__init__.py", payload)
    manifest = tmp_path / "manifest.sha256"
    manifest_sha256 = _write_manifest(
        manifest,
        [("tanager_rocks/__init__.py", _sha_file(member))],
    )

    observed_sha256, count = HELPER.verify_source_recovery(
        manifest=manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_entry_count=1,
        project_root=project,
    )
    installed_sha256, installed_count = HELPER.install_source_recovery(
        manifest=manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_entry_count=1,
        source_project_root=project,
        destination_project_root=destination_project,
    )

    assert observed_sha256 == manifest_sha256
    assert count == 1
    assert installed_sha256 == manifest_sha256
    assert installed_count == 1
    assert not marker.exists()
