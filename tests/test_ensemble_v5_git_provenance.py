"""Regression tests for the E6-v5 hash-bound Git provenance capsule."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs" / "operational" / "tanager-rocks-e061439-v5.bundle"
STATUS_LEDGER = ROOT / "docs" / "operational" / "m2_ensemble_v5_expected_git_status.txt"
WRAPPER = ROOT / "scripts" / "run_ensemble_bigmem_v5.sbatch"
EXPECTED_HEAD = "e06143909af7651eaec0ecbe8e84b191970c2398"
EXPECTED_BUNDLE_SHA256 = "9caed40ffaa805632bddb282d324fb9fc0f856e4b6c6b437618ddd59c9c8cc6e"
EXPECTED_STATUS_SHA256 = "8d076a9b9f22351b1f788e672275d4b064a88d46f5d3be88e3c0dba60f344ef4"
GOVERNING_PATHS = (
    "src/tanager_rocks/ensemble_sensitivity.py",
    "scripts/run_ensemble_sensitivity.py",
    "tests/test_ensemble_sensitivity.py",
    "docs/m2_ensemble_sensitivity_preregistration.md",
    "docs/m2_spatial_validation_preregistration.md",
    "docs/tanager_quality_mask_policy.md",
    "src/tanager_rocks/spatial_validation.py",
    "src/tanager_rocks/strict_inductive.py",
    "src/tanager_rocks/unmix.py",
    "src/tanager_rocks/quality.py",
    "src/tanager_rocks/speclib.py",
    "src/tanager_rocks/reference.py",
    "src/tanager_rocks/config.py",
    "src/tanager_rocks/pipeline.py",
    "src/tanager_rocks/viz.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_bundle_and_status_ledger_match_frozen_identities():
    assert _sha256(BUNDLE) == EXPECTED_BUNDLE_SHA256
    assert _sha256(STATUS_LEDGER) == EXPECTED_STATUS_SHA256
    verification = _git("bundle", "verify", os.fspath(BUNDLE))
    assert EXPECTED_HEAD in verification.stdout

    observed = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *GOVERNING_PATHS,
    ).stdout
    assert observed == STATUS_LEDGER.read_text(encoding="utf-8")


def test_bundle_restores_git_status_with_empty_primary_object_store(tmp_path: Path):
    bare = tmp_path / "bare"
    template = tmp_path / "template"
    empty_objects = tmp_path / "empty-objects"
    template.mkdir()
    empty_objects.mkdir()
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--bare",
            "--no-hardlinks",
            f"--template={template}",
            os.fspath(BUNDLE),
            os.fspath(bare),
        ],
        check=True,
    )
    environment = {
        **os.environ,
        "GIT_OBJECT_DIRECTORY": os.fspath(empty_objects),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": os.fspath(bare / "objects"),
        "GIT_OPTIONAL_LOCKS": "0",
    }

    _git("cat-file", "-e", f"{EXPECTED_HEAD}^{{commit}}", env=environment)
    assert _git("rev-parse", "HEAD", env=environment).stdout.strip() == EXPECTED_HEAD
    observed = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *GOVERNING_PATHS,
        env=environment,
    ).stdout
    assert observed == STATUS_LEDGER.read_text(encoding="utf-8")


def test_wrapper_binds_alternate_object_store_and_frozen_status():
    payload = WRAPPER.read_text(encoding="utf-8")
    assert f'EXPECTED_GIT_BUNDLE_SHA256="{EXPECTED_BUNDLE_SHA256}"' in payload
    assert f'EXPECTED_GIT_STATUS_SHA256="{EXPECTED_STATUS_SHA256}"' in payload
    assert f'EXPECTED_GIT_HEAD="{EXPECTED_HEAD}"' in payload
    assert 'export GIT_ALTERNATE_OBJECT_DIRECTORIES="${GIT_OBJECT_REPO}/objects"' in payload
    assert "export GIT_OPTIONAL_LOCKS=0" in payload
    assert 'cmp -s "${GIT_STATUS_LEDGER}" "${GIT_STATUS_OBSERVED}"' in payload
