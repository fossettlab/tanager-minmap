"""Tests for the standalone E6 v2 source-capsule verifier."""

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
SCRIPT = ROOT / "scripts" / "verify_source_capsule.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_source_capsule", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _manifest_line(path: str, digest: str) -> str:
    return f"{digest}  {path}"


def _write_manifest(manifest: Path, entries: Iterable[tuple[str, str]]) -> str:
    payload = "".join(f"{_manifest_line(path, digest)}\n" for path, digest in entries)
    manifest.write_text(payload, encoding="utf-8")
    return _sha_bytes(payload.encode())


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    project = workspace / "tanager-rocks"
    sibling = workspace / "tanager-spec"
    project.mkdir(parents=True)
    sibling.mkdir()
    return project, sibling, workspace / "source_capsule.sha256"


def _run(
    manifest: Path,
    manifest_sha256: str,
    count: int,
    project_root: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            os.fspath(SCRIPT),
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


def test_happy_project_and_exact_sibling_capsule(tmp_path: Path):
    project, sibling, manifest = _workspace(tmp_path)
    project_member = project / "scripts" / "operation.py"
    sibling_member = sibling / "src" / "tanager_spec" / "contract.py"
    project_member.parent.mkdir(parents=True)
    sibling_member.parent.mkdir(parents=True)
    project_member.write_bytes(b"project source\n")
    sibling_member.write_bytes(b"sibling source\n")
    entries = [
        ("../tanager-spec/src/tanager_spec/contract.py", _sha_file(sibling_member)),
        ("scripts/operation.py", _sha_file(project_member)),
    ]
    manifest_sha256 = _write_manifest(manifest, entries)

    result = _run(manifest, manifest_sha256, len(entries), project)

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == (
        f"PASS check=source_capsule manifest_sha256={manifest_sha256} entry_count=2\n"
    )


@pytest.mark.parametrize(
    ("fault", "expected_check"),
    [
        ("digest", "manifest_digest"),
        ("count", "entry_count"),
        ("order", "entry_order"),
    ],
)
def test_detached_digest_count_and_order_failures(
    tmp_path: Path,
    fault: str,
    expected_check: str,
):
    project, _sibling, manifest = _workspace(tmp_path)
    first = project / "a.py"
    second = project / "b.py"
    first.write_bytes(b"a\n")
    second.write_bytes(b"b\n")
    entries = [("a.py", _sha_file(first)), ("b.py", _sha_file(second))]
    if fault == "order":
        entries.reverse()
    manifest_sha256 = _write_manifest(manifest, entries)
    expected_sha256 = "0" * 64 if fault == "digest" else manifest_sha256
    expected_count = 3 if fault == "count" else 2

    result = _run(manifest, expected_sha256, expected_count, project)

    assert result.returncode != 0
    assert f"check={expected_check}" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "member_path",
    [
        "/absolute.py",
        "../outside.py",
        "nested/../../outside.py",
        "../tanager-spec/../outside.py",
        "../other/file.py",
        "../tanager-specification/file.py",
        "folder\\member.py",
        "./member.py",
    ],
)
def test_path_escape_and_noncanonical_paths_are_rejected(tmp_path: Path, member_path: str):
    project, _sibling, manifest = _workspace(tmp_path)
    manifest_sha256 = _write_manifest(manifest, [(member_path, "0" * 64)])

    result = _run(manifest, manifest_sha256, 1, project)

    assert result.returncode != 0
    assert "check=member_path" in result.stderr


@pytest.mark.parametrize("attack", ["component", "final"])
def test_symlink_component_and_final_are_rejected(tmp_path: Path, attack: str):
    project, _sibling, manifest = _workspace(tmp_path)
    target_directory = project / "target"
    target_directory.mkdir()
    target = target_directory / "member.py"
    target.write_bytes(b"target\n")
    if attack == "component":
        (project / "linked").symlink_to(target_directory, target_is_directory=True)
        member_path = "linked/member.py"
    else:
        (project / "member.py").symlink_to(target)
        member_path = "member.py"
    manifest_sha256 = _write_manifest(manifest, [(member_path, _sha_file(target))])

    result = _run(manifest, manifest_sha256, 1, project)

    assert result.returncode != 0
    assert f"reason=symlink_{attack}" in result.stderr


def test_non_directory_component_and_non_regular_final_are_rejected(tmp_path: Path):
    project, _sibling, manifest = _workspace(tmp_path)
    (project / "component").write_bytes(b"not a directory\n")
    component_sha = _write_manifest(manifest, [("component/member.py", "0" * 64)])

    component_result = _run(manifest, component_sha, 1, project)

    assert component_result.returncode != 0
    assert "reason=non_directory_component" in component_result.stderr

    final_directory = project / "directory.py"
    final_directory.mkdir()
    final_sha = _write_manifest(manifest, [("directory.py", "0" * 64)])

    final_result = _run(manifest, final_sha, 1, project)

    assert final_result.returncode != 0
    assert "reason=non_regular_final" in final_result.stderr


@pytest.mark.parametrize("mutation", ["replace", "modify"])
def test_member_replacement_or_metadata_drift_during_read_is_rejected(
    tmp_path: Path,
    mutation: str,
):
    project, _sibling, manifest = _workspace(tmp_path)
    member = project / "member.py"
    member.write_bytes(b"original\n")
    manifest_sha256 = _write_manifest(manifest, [("member.py", _sha_file(member))])

    def mutate_after_read(logical_path: str) -> None:
        if logical_path != "member.py":
            return
        if mutation == "replace":
            replacement = project / "replacement.py"
            replacement.write_bytes(b"replacement\n")
            os.replace(replacement, member)
        else:
            member.write_bytes(b"modified\n")

    with pytest.raises(VERIFIER.VerificationError) as caught:
        VERIFIER.verify_source_capsule(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=1,
            project_root=project,
            after_read_hook=mutate_after_read,
        )

    assert caught.value.path == "member.py"
    assert caught.value.check == "member_file"
    assert caught.value.reason == "file_changed"


def test_manifest_replacement_during_read_is_rejected(tmp_path: Path):
    project, _sibling, manifest = _workspace(tmp_path)
    member = project / "member.py"
    member.write_bytes(b"member\n")
    manifest_sha256 = _write_manifest(manifest, [("member.py", _sha_file(member))])

    def replace_manifest(logical_path: str) -> None:
        if logical_path != str(manifest):
            return
        replacement = manifest.with_suffix(".replacement")
        replacement.write_bytes(manifest.read_bytes())
        os.replace(replacement, manifest)

    with pytest.raises(VERIFIER.VerificationError) as caught:
        VERIFIER.verify_source_capsule(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            expected_entry_count=1,
            project_root=project,
            after_read_hook=replace_manifest,
        )

    assert caught.value.check == "manifest_read"
    assert caught.value.reason == "file_changed"


def test_member_digest_mismatch_is_rejected_without_content_leakage(tmp_path: Path):
    project, _sibling, manifest = _workspace(tmp_path)
    member = project / "member.py"
    secret_content = "private-capsule-content"
    member.write_text(secret_content, encoding="utf-8")
    manifest_sha256 = _write_manifest(manifest, [("member.py", "0" * 64)])

    result = _run(manifest, manifest_sha256, 1, project)

    assert result.returncode != 0
    assert 'path="member.py" check=member_digest reason=digest_mismatch' in result.stderr
    assert secret_content not in result.stderr


@pytest.mark.parametrize(
    "payload",
    [
        b"\n",
        b"not-a-sha  member.py\n",
        f"{'0' * 64} member.py\n".encode(),
        f"{'0' * 64}  member.py".encode(),
        f"{'0' * 64}  member.py\r\n".encode(),
        f"{'0' * 64}  member.py\n{'0' * 64}  member.py\n".encode(),
    ],
)
def test_blank_malformed_newline_and_duplicate_lines_are_rejected(
    tmp_path: Path,
    payload: bytes,
):
    project, _sibling, manifest = _workspace(tmp_path)
    manifest.write_bytes(payload)

    result = _run(manifest, _sha_bytes(payload), payload.count(b"\n"), project)

    assert result.returncode != 0
    assert "PASS" not in result.stdout
