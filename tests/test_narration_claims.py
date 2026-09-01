"""Regression checks for the release-bound narration claim contract."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "video" / "narration_script_v2.md"
SEGMENT_DIR = ROOT / "video" / "segments_v2"
SEGMENTS = (
    "00_title",
    "01_hook",
    "02_stakes",
    "03_data",
    "04_ablation",
    "05_livemap",
    "06_validation",
    "07_amd",
    "08_close",
)


def _spoken_blocks() -> tuple[str, ...]:
    """Return the nine spoken paragraphs from the narration master."""
    master = MASTER.read_text(encoding="utf-8")
    blocks = re.findall(
        r"^### \d+ — .*?\n\n(.+?)(?=\n\n### |\Z)",
        master,
        flags=re.MULTILINE | re.DOTALL,
    )
    return tuple(block.strip() for block in blocks)


def test_master_spoken_text_matches_release_segments_exactly() -> None:
    """The editorial master and TTS inputs must remain one source of truth."""
    expected = tuple(
        (SEGMENT_DIR / f"{segment}.txt").read_text(encoding="utf-8").strip() for segment in SEGMENTS
    )

    assert _spoken_blocks() == expected


def test_release_narration_retains_scientific_scope_boundaries() -> None:
    """Required caveats must survive editorial and TTS-source revisions."""
    spoken = "\n".join(_spoken_blocks())

    required = (
        "Four hundred and twenty-six colors of light.",
        "Tanager's delivered ortho product contains 426 contiguous bands",
        "on a thirty-meter grid",
        "Jarosite is associated with acidic sulfate conditions",
        "gypsum without acidic iron phases can indicate more buffered conditions",
        "show features in those regions",
        "scene-relative, library-matched candidate",
        "not field truth",
        "jarosite at zero point five eight",
        "candidate screening layer",
        "relative within the scene",
        "it does not measure pH or replace sampling",
    )
    for phrase in required:
        assert phrase in spoken

    forbidden = (
        "Four hundred and twenty-five",
        "Tanager measures 425",
        "gypsum is benign",
        "jarosite at zero point five nine",
        "replace field sampling",
    )
    for phrase in forbidden:
        assert phrase not in spoken
