"""Synthetic endpoint-free tests for the repeatability staging verifier."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_repeatability_staging.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_repeatability_staging", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()


@dataclass(frozen=True)
class SyntheticStage:
    """Paths and immutable control identities for one synthetic staged root."""

    config: Any
    root: Path
    project: Path
    sibling: Path
    proposal_sha256: str
    manifest_sha256: str


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha_bytes(payload)


def _manifest_payload(entries: list[tuple[str, str]]) -> bytes:
    return "".join(f"{digest}  {path}\n" for path, digest in sorted(entries)).encode()


def _record_sorted_manifest_payload(entries: list[tuple[str, str]]) -> bytes:
    return "".join(sorted(f"{digest}  {path}\n" for path, digest in entries)).encode()


@pytest.fixture
def staged(tmp_path: Path) -> SyntheticStage:
    root = tmp_path / "repeatability-stage"
    project = root / "Tanager" / "tanager-rocks"
    sibling = root / "Tanager" / "tanager-spec"
    (root / "slurm_logs").mkdir(parents=True)
    sibling.mkdir(parents=True)
    (project / ".git").mkdir(parents=True)
    _write(project / "README.md", b"synthetic project readme\n")
    _write(sibling / "README.md", b"synthetic sibling readme\n")

    input_records: list[dict[str, Any]] = []
    for index in range(VERIFIER.EXPECTED_RAW_SCENE_COUNT):
        logical_path = f"data/raw/scene-{index:02d}_ortho_sr_hdf5.h5"
        payload = f"raw-scene-{index}".encode()
        _write(project / logical_path, payload)
        input_records.append(
            {
                "id": f"tanager-synthetic-{index + 1}",
                "logical_path": logical_path,
                "size_bytes": len(payload),
                "sha256": _sha_bytes(payload),
            }
        )
    spectral_root = project / VERIFIER.SPECTRAL_TREE_RELATIVE_PATH
    spectral_members = {f"member-{index}.txt": f"member-{index}\n".encode() for index in range(3)}
    archive_path = project / VERIFIER.SPECTRAL_ARCHIVE_RELATIVE_PATH
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        for relative, payload in spectral_members.items():
            archive.writestr(f"{VERIFIER.SPECTRAL_TREE_RELATIVE_PATH.name}/{relative}", payload)
            _write(spectral_root / relative, payload)
    archive_payload = archive_path.read_bytes()
    input_records.append(
        {
            "id": "usgs-splib07a-archive",
            "logical_path": VERIFIER.SPECTRAL_ARCHIVE_RELATIVE_PATH.as_posix(),
            "size_bytes": len(archive_payload),
            "sha256": _sha_bytes(archive_payload),
        }
    )
    input_manifest = (
        json.dumps(
            {
                "schema_version": "1.0",
                "hash_algorithm": "sha256",
                "inputs": input_records,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )

    members = {
        "docs/input_manifest.json": input_manifest,
        "scripts/run_repeatability_bigmem.sbatch": b"#!/usr/bin/env bash\nexit 99\n",
        "src/tanager_rocks/repeatability.py": b"SCIENTIFIC_EXECUTION_IDENTITY = 'synthetic'\n",
        "uv.lock": b"version = 1\n",
    }
    entries = [(path, _write(project / path, payload)) for path, payload in members.items()]
    sibling_member = sibling / "src" / "tanager_spec" / "io.py"
    entries.append(("../tanager-spec/src/tanager_spec/io.py", _write(sibling_member, b"pass\n")))

    manifest = project / "docs" / "m2_repeatability_bigmem_source_manifest.sha256"
    manifest_sha256 = _write(manifest, _manifest_payload(entries))
    proposal = tmp_path / "proposal.md"
    proposal_sha256 = _write(proposal, b"proposal-only synthetic control\n")
    config = VERIFIER.StagingConfig(
        actual_root=root,
        expected_root=root,
        e6_root=tmp_path / "e6-root",
        proposal=proposal,
        expected_proposal_sha256=proposal_sha256,
        source_manifest=manifest,
        expected_source_manifest_sha256=manifest_sha256,
        expected_source_member_count=len(entries),
        evidence_output=tmp_path / "evidence.json",
    )
    return SyntheticStage(
        config=config,
        root=root,
        project=project,
        sibling=sibling,
        proposal_sha256=proposal_sha256,
        manifest_sha256=manifest_sha256,
    )


def _synthetic_admission(context: Any) -> dict[str, Any]:
    return dict(VERIFIER.independent_input_admitter(context))


def _verify(stage: SyntheticStage, **kwargs: Any) -> dict[str, Any]:
    return VERIFIER.verify_repeatability_staging(stage.config, **kwargs)


def _assert_failure(code: str, callable_: Any) -> None:
    with pytest.raises(VERIFIER.StagingVerificationError) as caught:
        callable_()
    assert caught.value.code == code


def test_success_writes_canonical_endpoint_blind_evidence(staged: SyntheticStage):
    evidence = _verify(staged)

    payload = staged.config.evidence_output.read_bytes()
    assert payload == VERIFIER.canonical_json_bytes(evidence)
    assert evidence["status"] == "PASS"
    assert evidence["authority"] == "staged_root_verification_only"
    assert evidence["source_manifest"] == {
        "member_count": staged.config.expected_source_member_count,
        "sha256": staged.manifest_sha256,
    }
    assert evidence["verifier"] == {
        "sha256": VERIFIER._verifier_sha256(),
        "source_capsule_sha256": VERIFIER._source_capsule_verifier_sha256(),
    }
    assert evidence["input_closure"]["raw_scene_count"] == 7
    assert evidence["timing_command"]["executed"] is False
    assert evidence["timing_command"]["argv"] == list(
        VERIFIER.build_timing_argv(staged.root, staged.manifest_sha256)
    )
    assert "endpoint" not in json.dumps(evidence).lower()
    assert stat.S_IMODE(staged.config.evidence_output.stat().st_mode) == 0o600


def test_literal_frozen_manifest_record_order_is_admitted():
    payload = (ROOT / "docs" / "m2_repeatability_bigmem_source_manifest.sha256").read_bytes()
    raw_paths = [line.split("  ", maxsplit=1)[1] for line in payload.decode("utf-8").splitlines()]

    assert raw_paths != sorted(raw_paths)
    entries = VERIFIER._parse_digest_bound_manifest(payload, expected_count=47)

    assert len(entries) == 47
    assert [entry.path for entry in entries] == sorted(raw_paths)


def test_cli_uses_real_staged_frozen_input_admission(tmp_path: Path):
    root = tmp_path / "production-stage"
    project = root / "Tanager" / "tanager-rocks"
    sibling = root / "Tanager" / "tanager-spec"
    (root / "slurm_logs").mkdir(parents=True)
    (project / ".git").mkdir(parents=True)
    sibling.mkdir(parents=True)
    shutil.copyfile(ROOT / "README.md", project / "README.md")
    shutil.copyfile(ROOT.parent / "tanager-spec" / "README.md", sibling / "README.md")

    current_manifest = ROOT / "docs" / "m2_repeatability_bigmem_source_manifest.sha256"
    entries: list[tuple[str, str]] = []
    for line in current_manifest.read_text().splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        if relative == "docs/input_manifest.json":
            continue
        if relative.startswith("../tanager-spec/"):
            source = ROOT / relative
            destination = sibling / relative.removeprefix("../tanager-spec/")
        else:
            source = ROOT / relative
            destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert _sha_bytes(destination.read_bytes()) == digest
        entries.append((relative, digest))

    current_inputs = json.loads((ROOT / "docs" / "input_manifest.json").read_text())
    input_records: list[dict[str, Any]] = []
    for record in current_inputs["inputs"]:
        logical_path = record["logical_path"]
        if not (
            logical_path.startswith("data/raw/") and logical_path.endswith("_ortho_sr_hdf5.h5")
        ):
            continue
        payload = f"synthetic-{record['id']}\n".encode()
        _write(project / logical_path, payload)
        input_records.append(
            {
                "id": record["id"],
                "logical_path": logical_path,
                "size_bytes": len(payload),
                "sha256": _sha_bytes(payload),
            }
        )

    archive_path = project / VERIFIER.SPECTRAL_ARCHIVE_RELATIVE_PATH
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    spectral_members = {
        f"member-{index}.txt": f"spectral-member-{index}\n".encode() for index in range(3)
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        for relative, payload in spectral_members.items():
            archive.writestr(f"ASCIIdata_splib07a/{relative}", payload)
            _write(project / VERIFIER.SPECTRAL_TREE_RELATIVE_PATH / relative, payload)
    archive_payload = archive_path.read_bytes()
    input_records.append(
        {
            "id": "usgs-splib07a-archive",
            "logical_path": VERIFIER.SPECTRAL_ARCHIVE_RELATIVE_PATH.as_posix(),
            "size_bytes": len(archive_payload),
            "sha256": _sha_bytes(archive_payload),
        }
    )
    input_manifest_payload = (
        json.dumps(
            {
                "schema_version": "1.0",
                "hash_algorithm": "sha256",
                "inputs": input_records,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    input_manifest_sha256 = _write(
        project / VERIFIER.INPUT_MANIFEST_RELATIVE_PATH,
        input_manifest_payload,
    )
    entries.append((VERIFIER.INPUT_MANIFEST_RELATIVE_PATH.as_posix(), input_manifest_sha256))
    manifest = project / VERIFIER.SOURCE_MANIFEST_RELATIVE_PATH
    manifest_sha256 = _write(manifest, _record_sorted_manifest_payload(entries))
    assert [path for path, _digest in entries] != sorted(path for path, _digest in entries)
    proposal = tmp_path / "proposal.md"
    proposal_sha256 = _write(proposal, b"production-path synthetic proposal\n")
    evidence = tmp_path / "production-evidence.json"

    command = [
        sys.executable,
        os.fspath(SCRIPT),
        "--actual-root",
        os.fspath(root),
        "--expected-root",
        os.fspath(root),
        "--e6-root",
        os.fspath(tmp_path / "e6-root"),
        "--proposal",
        os.fspath(proposal),
        "--expected-proposal-sha256",
        proposal_sha256,
        "--source-manifest",
        os.fspath(manifest),
        "--expected-source-manifest-sha256",
        manifest_sha256,
        "--expected-source-member-count",
        str(len(entries)),
        "--evidence-output",
        os.fspath(evidence),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("PASS check=repeatability_staging ")
    assert json.loads(evidence.read_text())["status"] == "PASS"


@pytest.mark.parametrize("relation", ["same", "ancestor", "descendant"])
def test_root_overlap_with_e6_fails_closed(staged: SyntheticStage, relation: str):
    e6_root = {
        "same": staged.root,
        "ancestor": staged.root.parent,
        "descendant": staged.root / "e6-child",
    }[relation]
    config = replace(staged.config, e6_root=e6_root)

    _assert_failure(
        "ROOT_E6_OVERLAP",
        lambda: VERIFIER.verify_repeatability_staging(
            config,
            input_admitter=_synthetic_admission,
        ),
    )


def test_actual_and_expected_root_mismatch_fails_closed(staged: SyntheticStage):
    config = replace(staged.config, expected_root=staged.root.parent / "other-stage")

    _assert_failure(
        "ROOT_IDENTITY_MISMATCH",
        lambda: VERIFIER.verify_repeatability_staging(
            config,
            input_admitter=_synthetic_admission,
        ),
    )


@pytest.mark.parametrize("attack", ["component", "final"])
def test_symlink_component_or_final_root_fails_closed(
    staged: SyntheticStage,
    attack: str,
):
    if attack == "final":
        attacked_root = staged.root.parent / "linked-stage"
        attacked_root.symlink_to(staged.root, target_is_directory=True)
    else:
        linked_parent = staged.root.parent / "linked-parent"
        linked_parent.symlink_to(staged.root.parent, target_is_directory=True)
        attacked_root = linked_parent / staged.root.name
    config = replace(
        staged.config,
        actual_root=attacked_root,
        expected_root=attacked_root,
        source_manifest=attacked_root
        / VERIFIER.PROJECT_RELATIVE_PATH
        / (VERIFIER.SOURCE_MANIFEST_RELATIVE_PATH),
    )

    _assert_failure(
        "ROOT_OR_LAYOUT_UNSAFE",
        lambda: VERIFIER.verify_repeatability_staging(
            config,
            input_admitter=_synthetic_admission,
        ),
    )


def test_source_member_final_symlink_fails_closed(staged: SyntheticStage):
    member = staged.project / "src" / "tanager_rocks" / "repeatability.py"
    target = staged.project / "repeatability-target.py"
    target.write_bytes(member.read_bytes())
    member.unlink()
    member.symlink_to(target)

    _assert_failure("SOURCE_CAPSULE_FAILED", lambda: _verify(staged))


def test_source_member_hard_link_fails_closed(staged: SyntheticStage):
    member = staged.project / "uv.lock"
    os.link(member, staged.project / "uv-copy.lock")

    _assert_failure("SOURCE_MEMBER_HARDLINK", lambda: _verify(staged))


@pytest.mark.parametrize("drift", ["manifest", "member"])
def test_manifest_or_member_drift_fails_closed(staged: SyntheticStage, drift: str):
    if drift == "manifest":
        staged.config.source_manifest.write_bytes(
            staged.config.source_manifest.read_bytes() + b"\n"
        )
        expected = "SOURCE_MANIFEST_DRIFT"
    else:
        staged.project.joinpath("uv.lock").write_bytes(b"drifted\n")
        expected = "SOURCE_CAPSULE_FAILED"

    _assert_failure(expected, lambda: _verify(staged))


def test_proposal_drift_fails_closed(staged: SyntheticStage):
    staged.config.proposal.write_bytes(b"changed proposal\n")

    _assert_failure("PROPOSAL_DRIFT", lambda: _verify(staged))


def test_noncanonical_proposal_path_fails_closed(staged: SyntheticStage):
    proposal = Path(f"{staged.config.proposal.parent}/unused/../proposal.md")
    config = replace(staged.config, proposal=proposal)

    _assert_failure(
        "PROPOSAL_PATH_INVALID",
        lambda: VERIFIER.verify_repeatability_staging(
            config,
            input_admitter=_synthetic_admission,
        ),
    )


@pytest.mark.parametrize(
    ("residue", "expected"),
    [
        ("output", "REPEATABILITY_OUTPUT_EXISTS"),
        ("python-lock", "PYTHON_LOCK_EXISTS"),
        ("wrapper-runtime", "WRAPPER_RUNTIME_EXISTS"),
    ],
)
def test_output_lock_or_runtime_residue_fails_closed(
    staged: SyntheticStage,
    residue: str,
    expected: str,
):
    output = staged.project / VERIFIER.OUTPUT_RELATIVE_PATH
    if residue == "output":
        output.mkdir(parents=True)
    elif residue == "python-lock":
        VERIFIER._execution_lock_path(output).mkdir(parents=True)
    else:
        (staged.root / VERIFIER.WRAPPER_RUNTIME_RELATIVE_PATH).mkdir(parents=True)

    _assert_failure(expected, lambda: _verify(staged))


@pytest.mark.parametrize("contamination", ["cache", "log"])
def test_e6_cache_or_log_contamination_fails_closed(
    staged: SyntheticStage,
    contamination: str,
):
    if contamination == "cache":
        (staged.root / VERIFIER.CACHE_RELATIVE_PATH).mkdir()
        expected = "E6_CACHE_CONTAMINATION"
    else:
        (staged.root / VERIFIER.LOG_RELATIVE_PATH / "copied.out").write_bytes(b"")
        expected = "E6_LOG_CONTAMINATION"

    _assert_failure(expected, lambda: _verify(staged))


def test_unrelated_runtime_residue_fails_closed(staged: SyntheticStage):
    (staged.root / "runtime").mkdir()

    _assert_failure("E6_RUNTIME_CONTAMINATION", lambda: _verify(staged))


def test_unexpected_staged_file_fails_closed(staged: SyntheticStage):
    (staged.project / "copied-e6-artifact.bin").write_bytes(b"unexpected\n")

    _assert_failure("CLOSED_LAYOUT_MISMATCH", lambda: _verify(staged))


def test_staged_bytecode_cache_fails_closed(staged: SyntheticStage):
    cached = staged.project / "src" / "tanager_rocks" / "__pycache__"
    cached.mkdir()
    (cached / "repeatability.cpython-311.pyc").write_bytes(b"untrusted cache\n")

    _assert_failure("CLOSED_LAYOUT_MISMATCH", lambda: _verify(staged))


def test_input_hard_link_fails_closed(staged: SyntheticStage):
    raw = next((staged.project / "data" / "raw").iterdir())
    os.link(raw, staged.root.parent / "raw-hardlink.h5")

    _assert_failure("CLOSED_LAYOUT_HARDLINK", lambda: _verify(staged))


def test_input_parent_symlink_fails_closed(staged: SyntheticStage):
    raw = staged.project / "data" / "raw"
    moved = staged.root.parent / "moved-raw"
    raw.rename(moved)
    raw.symlink_to(moved, target_is_directory=True)

    _assert_failure("CLOSED_LAYOUT_UNSAFE", lambda: _verify(staged))


def test_post_admission_input_drift_fails_closed(staged: SyntheticStage):
    def mutate_after_admission(context: Any) -> dict[str, Any]:
        admission = _synthetic_admission(context)
        raw = next(context.paths.raw_dir.iterdir())
        raw.write_bytes(b"changed after admission\n")
        return admission

    _assert_failure(
        "INPUT_FILE_MISMATCH",
        lambda: VERIFIER.verify_repeatability_staging(
            staged.config,
            input_admitter=mutate_after_admission,
        ),
    )


def test_extracted_spectral_tree_must_match_captured_archive(staged: SyntheticStage):
    member = next(
        candidate
        for candidate in (staged.project / VERIFIER.SPECTRAL_TREE_RELATIVE_PATH).rglob("*")
        if candidate.is_file()
    )
    member.write_bytes(b"not the archived member\n")

    _assert_failure("INPUT_ARCHIVE_TREE_MISMATCH", lambda: _verify(staged))


def test_command_disagreement_fails_closed(staged: SyntheticStage):
    command = list(VERIFIER.build_timing_argv(staged.root, staged.manifest_sha256))
    command[1] = f"--chdir={staged.sibling}"

    _assert_failure(
        "TIMING_COMMAND_DISAGREEMENT",
        lambda: _verify(staged, timing_argv=command),
    )


def test_input_admission_failure_is_redacted(staged: SyntheticStage):
    secret = "synthetic-private-value"

    def reject_inputs(_context: Any) -> dict[str, Any]:
        raise ValueError(secret)

    with pytest.raises(VERIFIER.StagingVerificationError) as caught:
        VERIFIER.verify_repeatability_staging(
            staged.config,
            input_admitter=reject_inputs,
        )

    assert caught.value.code == "INPUT_ADMISSION_FAILED"
    assert secret not in caught.value.render()


def test_default_admitter_never_imports_staged_packages(
    staged: SyntheticStage,
    monkeypatch: pytest.MonkeyPatch,
):
    original_import = builtins.__import__

    def reject_staged_import(
        name: str,
        globals_: Any = None,
        locals_: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "tanager_rocks" or name.startswith("tanager_rocks."):
            raise AssertionError("staged tanager_rocks import attempted")
        if name == "tanager_spec" or name.startswith("tanager_spec."):
            raise AssertionError("staged tanager_spec import attempted")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_staged_import)

    evidence = VERIFIER.verify_repeatability_staging(staged.config)

    assert evidence["status"] == "PASS"


def test_existing_evidence_is_never_overwritten(staged: SyntheticStage):
    original = _verify(staged)
    original_bytes = staged.config.evidence_output.read_bytes()

    _assert_failure("EVIDENCE_OUTPUT_EXISTS", lambda: _verify(staged))

    assert staged.config.evidence_output.read_bytes() == original_bytes
    assert original["status"] == "PASS"


def test_evidence_final_symlink_is_not_followed(staged: SyntheticStage):
    target = staged.config.evidence_output.with_name("target.json")
    target.write_bytes(b"unchanged\n")
    staged.config.evidence_output.symlink_to(target)

    _assert_failure("EVIDENCE_OUTPUT_EXISTS", lambda: _verify(staged))

    assert target.read_bytes() == b"unchanged\n"


def test_evidence_parent_symlink_is_not_followed(staged: SyntheticStage):
    real_parent = staged.config.evidence_output.parent / "real-evidence-parent"
    real_parent.mkdir()
    linked_parent = staged.config.evidence_output.parent / "linked-evidence-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    config = replace(
        staged.config,
        evidence_output=linked_parent / "evidence.json",
    )

    _assert_failure(
        "EVIDENCE_OUTPUT_PARENT_UNSAFE",
        lambda: VERIFIER.verify_repeatability_staging(
            config,
            input_admitter=_synthetic_admission,
        ),
    )
    assert list(real_parent.iterdir()) == []
