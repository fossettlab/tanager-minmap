"""Shared paths, render constants, and EDL math for the v2 video pipeline.

Every other module under scripts/video/ imports from here so there is exactly
one place that knows the repo layout, the master render settings
(docs/edit_plan.md "Master render settings"), and the duration-preserving
assembly math (docs/edit_plan.md "Transition assembly").
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

STRICT_BUILD_ENV = "TANAGER_VIDEO_STRICT_BUILD_V2"
STRICT_SNAPSHOT_ENV = "TANAGER_VIDEO_STRICT_INPUT_SNAPSHOT"
STRICT_WORKER_ENV = "TANAGER_VIDEO_STRICT_WORKER"
STRICT_STAGING_ENV = "TANAGER_VIDEO_STRICT_STAGING_ROOT"
CODE_ROOT = Path(__file__).resolve().parents[2]


def _runtime_root() -> Path:
    if os.environ.get(STRICT_WORKER_ENV) != "1":
        return CODE_ROOT
    staging = os.environ.get(STRICT_STAGING_ENV)
    if staging is None:
        raise RuntimeError(f"{STRICT_STAGING_ENV} is required for strict workers")
    staging_root = Path(staging)
    if not staging_root.is_absolute() or len(staging_root.parents) < 3:
        raise RuntimeError(f"{STRICT_STAGING_ENV} is not a valid release staging path")
    return staging_root.parents[2]


ROOT = _runtime_root()


def _strict_path_override(name: str, *, expected_leaf: str) -> Path | None:
    override = os.environ.get(name)
    if override is None:
        return None
    if os.environ.get(STRICT_WORKER_ENV) != "1":
        raise RuntimeError(f"{name} is reserved for strict release workers")
    candidate = Path(override)
    if not candidate.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path")
    staging_value = os.environ.get(STRICT_STAGING_ENV)
    if staging_value is None:
        raise RuntimeError(f"{STRICT_STAGING_ENV} is required for strict workers")
    staging_root = Path(staging_value)
    if not staging_root.is_absolute() or staging_root.is_symlink():
        raise RuntimeError(f"{STRICT_STAGING_ENV} must identify an absolute regular directory")
    staging_resolved = staging_root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if staging_resolved != resolved and staging_resolved not in resolved.parents:
        raise RuntimeError(f"{name} must be beneath {staging_resolved}")
    if resolved.name != expected_leaf or not resolved.is_dir() or candidate.is_symlink():
        raise RuntimeError(f"{name} must identify a regular strict {expected_leaf} directory")
    return candidate


INPUT_ROOT = _strict_path_override(STRICT_SNAPSHOT_ENV, expected_leaf="capsule") or ROOT
SUBMISSION_FIGURES = INPUT_ROOT / "submission" / "figures"
AUDIO_V2 = INPUT_ROOT / "video" / "audio_v2"
SEGMENTS_V2 = INPUT_ROOT / "video" / "segments_v2"
BUILD = INPUT_ROOT / "video" / "build"


def _build_v2_path() -> Path:
    """Return the draft workspace or the isolated strict-worker workspace.

    The override is intentionally accepted only beneath ``output/releases``.
    ``render_v2.py`` sets it only in the strict child process after creating a
    symlink-safe staging root. Ordinary draft imports therefore retain the
    historical ``video/build/v2`` path unchanged.
    """
    override = os.environ.get(STRICT_BUILD_ENV)
    if override is None:
        return BUILD / "v2"
    candidate = _strict_path_override(STRICT_BUILD_ENV, expected_leaf="v2")
    assert candidate is not None
    return candidate


BUILD_V2 = _build_v2_path()
CLIPS_V2 = BUILD_V2 / "clips"
LOGS_V2 = BUILD_V2 / "logs"
QC_V2 = BUILD_V2 / "qc"
UPGRADES_V2 = BUILD_V2 / "upgrades"  # swappable-input drop zone; see beats.py
OUTPUT = ROOT / "output"

FPS = 30
WIDTH, HEIGHT = 1920, 1080
BG_HEX = "#0a0e1a"
BG_FFMPEG = "0x0a0e1a"
DISSOLVE_D = 0.33  # s, 10 frames @ 30 fps

_MATPLOTLIB_SPEC = find_spec("matplotlib")
if _MATPLOTLIB_SPEC is None or _MATPLOTLIB_SPEC.origin is None:
    raise RuntimeError("matplotlib is required by the video rendering pipeline")
FONT_DIR = Path(_MATPLOTLIB_SPEC.origin).parent / "mpl-data" / "fonts" / "ttf"
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

GF = "goldfield_20240925_185504_87_4001"
BH = "bingham_20250911_191523_58_4001"

ENCODE_ARGS = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]

# Beat -> narration mp3. 08 uses take3 per the 2026-07-04 re-TTS decision
# (edit_plan.md "What is locked"); 00/03 use the _426 re-TTS per the
# 2026-07-07 band-count correction (425 -> 426, narration text changed to
# match); everything else is the existing v2 take.
SEGMENT_FILES = {
    "00_title": "00_title_426.mp3",
    "01_hook": "01_hook.mp3",
    "02_stakes": "02_stakes.mp3",
    "03_data": "03_data_426.mp3",
    "04_ablation": "04_ablation.mp3",
    "05_livemap": "05_livemap.mp3",
    "06_validation": "06_validation.mp3",
    "07_amd": "07_amd.mp3",
    "08_close": "08_close_take3.mp3",
}
BEAT_06A_FRACTION = 0.52  # d6a = round(dur*0.52, 2), per edit_plan.md


def ffprobe_dur(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    return float(out.strip())


def ffprobe_video_info(path: Path) -> dict[str, str]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ]
    ).decode()
    return dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)


def run(cmd: list[str], log: Path | None = None) -> None:
    """Run a command, raising on failure. Noisy stdout/stderr goes to `log`.

    RTK truncates long Bash output, so ffmpeg/ffprobe chatter is redirected to
    disk; only the exit code is trusted at the call site (per team-lead brief).
    On failure the log tail is surfaced in the raised error.
    """
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = log.read_text()[-2000:]
            raise RuntimeError(
                f"command failed ({proc.returncode}): {cmd}\n--- log tail ---\n{tail}"
            )
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@dataclass
class Beat:
    """One picture clip's slot in the timeline (docs/edit_plan.md EDL table)."""

    name: str  # clip id, e.g. "06a" -- matches CLIPS_V2/{name}.mp4
    vo_dur: float  # narration duration this beat must match
    render_dur: float  # actual rendered clip length (vo_dur, or +D if it carries a dissolve tail)
    abs_start: float  # cumulative narration-driven start time (assembly offsets use this)
    abs_end: float
    dissolve_out: bool  # True if this beat crossfades into the next (00, 06a, 07)


def default_segment_paths() -> dict[str, Path]:
    """Selected narration masters for ordinary draft renders."""
    return {name: AUDIO_V2 / fname for name, fname in SEGMENT_FILES.items()}


def probe_segment_durations(segment_paths: Mapping[str, Path] | None = None) -> dict[str, float]:
    """Probe the selected narration masters.

    Draft mode uses :data:`SEGMENT_FILES`. Strict release mode supplies the
    hash-verified mapping from its frozen release contract instead, so a
    different take cannot be selected merely because it exists on disk.
    """
    paths = default_segment_paths() if segment_paths is None else dict(segment_paths)
    if set(paths) != set(SEGMENT_FILES):
        missing = sorted(set(SEGMENT_FILES) - set(paths))
        extra = sorted(set(paths) - set(SEGMENT_FILES))
        raise ValueError(f"narration segment mapping mismatch: missing={missing}, extra={extra}")
    return {name: ffprobe_dur(paths[name]) for name in SEGMENT_FILES}


def build_edl(durations: dict[str, float]) -> list[Beat]:
    """Recompute the EDL from probed durations -- never hard-code timings.

    Mirrors docs/edit_plan.md: beat 06 splits 52/48 into 06a/06b, and 00/06a/07
    render at VOdur+D because they carry a dissolve tail consumed by the xfade
    overlap in assemble.py.
    """
    d6a = round(durations["06_validation"] * BEAT_06A_FRACTION, 2)
    d6b = round(durations["06_validation"] - d6a, 2)
    order = [
        ("00", durations["00_title"], True),
        ("01", durations["01_hook"], False),
        ("02", durations["02_stakes"], False),
        ("03", durations["03_data"], False),
        ("04", durations["04_ablation"], False),
        ("05", durations["05_livemap"], False),
        ("06a", d6a, True),
        ("06b", d6b, False),
        ("07", durations["07_amd"], True),
        ("08", durations["08_close"], False),
    ]
    beats = []
    t = 0.0
    for name, vo_dur, dissolve_out in order:
        render_dur = round(vo_dur + DISSOLVE_D, 3) if dissolve_out else vo_dur
        beats.append(Beat(name, vo_dur, render_dur, t, t + vo_dur, dissolve_out))
        t += vo_dur
    return beats


def beat_by_name(edl: list[Beat], name: str) -> Beat:
    for b in edl:
        if b.name == name:
            return b
    raise KeyError(name)


# --- zoompan / overlay geometry -------------------------------------------
#
# One clamp formula shared by every zoompan filter string AND by any overlay
# that must land on a specific point of the zoomed picture (CALL-03a/b/c):
# both have to agree on where the crop window actually ends up, including
# when the requested centre gets pulled inward to stay in-bounds.


def clamped_crop_window(
    cx: float, cy: float, zoom: float, canvas_w: float = WIDTH, canvas_h: float = HEIGHT
) -> tuple[float, float, float, float]:
    """Top-left (x0, y0) and size (w, h) of the crop window a zoompan/crop at
    fractional centre (cx, cy) and the given zoom actually produces, clamped
    to stay inside the canvas."""
    win_w, win_h = canvas_w / zoom, canvas_h / zoom
    x0 = min(max(cx * canvas_w - win_w / 2, 0.0), canvas_w - win_w)
    y0 = min(max(cy * canvas_h - win_h / 2, 0.0), canvas_h - win_h)
    return x0, y0, win_w, win_h


def canvas_point_to_screen(
    px: float, py: float, crop_window: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Where a point at canvas coords (px, py) lands on screen once the given
    crop window is scaled back up to fill the output frame."""
    x0, y0, win_w, win_h = crop_window
    zoom_x, zoom_y = WIDTH / win_w, HEIGHT / win_h
    return (px - x0) * zoom_x, (py - y0) * zoom_y
