"""Regression tests for the E6-v4 wrapper's persistent execution lock."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_ensemble_bigmem_v4.sbatch"


def _sandboxed_wrapper(
    tmp_path: Path, *, interpreter: Path | None = None
) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "run"
    project = run_root / "Tanager" / "tanager-rocks"
    script = project / "scripts" / WRAPPER.name
    script.parent.mkdir(parents=True)
    payload = WRAPPER.read_text(encoding="utf-8")
    payload = payload.replace(
        'RUN_ROOT="${TANAGER_BIGMEM_ROOT:-/scratch2/fs1/alexander.s.bradley/'
        'tanager-rocks-bigmem-20260810}"',
        f'RUN_ROOT="${{TANAGER_BIGMEM_ROOT:-{run_root}}}"',
    )
    payload = payload.replace(
        'PYTHON_311="/home/alexander.s.bradley/.local/share/uv/python/'
        'cpython-3.11-linux-x86_64-gnu/bin/python3.11"',
        f'PYTHON_311="{interpreter or Path(sys.executable)}"',
    )
    payload = payload.replace(
        'UV_BIN="/home/alexander.s.bradley/.local/bin/uv"',
        'UV_BIN="/usr/bin/true"',
    )
    script.write_text(payload, encoding="utf-8")
    script.chmod(0o700)
    lock = run_root / "runtime" / "v4" / "ensemble_v4.lock"
    lock.parent.parent.mkdir(parents=True)
    return script, project, lock


def _run(script: Path, project: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", os.fspath(script)],
        cwd=project,
        env={
            **os.environ,
            "TANAGER_MODE": "design",
            "TANAGER_SOURCE_MANIFEST_SHA256": "0" * 64,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_retained_lock_is_reused_without_truncation(tmp_path: Path):
    script, project, lock = _sandboxed_wrapper(tmp_path)
    lock.parent.mkdir()
    lock.write_bytes(b"")
    before = lock.stat()

    result = _run(script, project)

    after = lock.stat()
    assert result.returncode != 0  # the synthetic capsule has no source verifier
    assert "PASS check=execution_lock" in result.stdout
    assert (after.st_dev, after.st_ino, after.st_size) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
    )
    assert after.st_mtime_ns == before.st_mtime_ns


def test_absent_lock_is_created_as_regular_empty_file(tmp_path: Path):
    script, project, lock = _sandboxed_wrapper(tmp_path)

    result = _run(script, project)

    assert result.returncode != 0
    assert "PASS check=execution_lock" in result.stdout
    metadata = lock.lstat()
    assert lock.is_file()
    assert not lock.is_symlink()
    assert metadata.st_nlink == 1
    assert metadata.st_size == 0


def test_symlink_lock_is_rejected_without_touching_target(tmp_path: Path):
    script, project, lock = _sandboxed_wrapper(tmp_path)
    lock.parent.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"preserve me")
    lock.symlink_to(target)

    result = _run(script, project)

    assert result.returncode != 0
    assert "reason=invalid_lock_file" in result.stderr
    assert target.read_bytes() == b"preserve me"


def test_contended_lock_is_rejected(tmp_path: Path):
    script, project, lock = _sandboxed_wrapper(tmp_path)
    lock.parent.mkdir()
    lock.write_bytes(b"")
    descriptor = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(script, project)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert result.returncode != 0
    assert "reason=lock_contended" in result.stderr


def test_symlink_runtime_directory_is_rejected(tmp_path: Path):
    script, project, lock = _sandboxed_wrapper(tmp_path)
    target = tmp_path / "runtime-target"
    target.mkdir()
    lock.parent.symlink_to(target, target_is_directory=True)

    result = _run(script, project)

    assert result.returncode != 0
    assert "reason=invalid_parent" in result.stderr
    assert not (target / lock.name).exists()


def test_parent_replacement_between_execs_is_rejected(tmp_path: Path):
    runtime = tmp_path / "run" / "runtime" / "v4"
    moved = tmp_path / "run" / "runtime" / "v4-original"
    target = tmp_path / "replacement"
    target.mkdir()
    interpreter = tmp_path / "python-race"
    interpreter.write_text(
        "#!/bin/sh\n"
        'if [ "${TANAGER_LOCK_HELD:-}" = 1 ]; then\n'
        f"  mv '{runtime}' '{moved}'\n"
        f"  ln -s '{target}' '{runtime}'\n"
        "fi\n"
        f"exec '{sys.executable}' \"$@\"\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o700)
    script, project, lock = _sandboxed_wrapper(tmp_path, interpreter=interpreter)

    result = _run(script, project)

    assert result.returncode != 0
    assert "reason=post_exec_rebind" in result.stderr
    assert not (target / "jobs").exists()
    assert (moved / lock.name).is_file()
