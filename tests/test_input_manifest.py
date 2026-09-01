"""Tests for deterministic source-input hashing."""

from __future__ import annotations

from pathlib import Path
from runpy import run_path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_input_manifest.py"
sha256_file = run_path(str(SCRIPT))["sha256_file"]


def test_sha256_file_is_deterministic(tmp_path: Path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"tanager\x00rocks\n")

    assert sha256_file(path, chunk_size=3) == sha256_file(path, chunk_size=8)
    assert sha256_file(path) == "8e6c24fd9662d1a7ad256f1654b2a46aceeb162d1f556513fbed0eb78a8b6aef"
