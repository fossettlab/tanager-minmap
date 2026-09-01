"""Animated re-render of beat 06b ("EMIT cross-check") for the v2 submission video.

Unlike beats 03/04, this beat's storyboard/edit-plan treatment is a plain
settle-zoom + one caption -- no before/after data reveal is specified (contrast
with 04's explicit two-state animation). Re-running `scripts/compare_emit.py`
would mean re-orthorectifying the 1.7 GB cached EMIT granule and re-running MTMF
on both sensors for a result that would look identical to the committed static
panel, since nothing about the display state changes. So this beat animates the
existing `submission/figures/goldfield_..._emit_comparison.png` (itself the real
`run_emit()` output) with motion + a composited caption, per the "animate the
existing PNG" fallback that docs/edit_plan.md sanctions whenever a re-render adds
compute but no visual difference.

Treatment (docs/edit_plan.md beat 06b + the CAP-06b overlay row):
- Letterbox->information band: the 3.21:1 source scales to width 1920
  (height ~597 px) seated high on the navy canvas, leaving the lower band for
  the caption (same technique as beat 04's letterbox, generalised).
- Slight 1.00->1.04x settle over the full clip.
- One caption, verbatim: "EMIT detects the same minerals - alunite r = 0.55 *
  jarosite r = 0.58" (source: data/intermediate/emit/emit_comparison_goldfield_*.csv
  -- alunite detection r=0.5501, jarosite r=0.5838), fading in at +4.0s (beat-
  relative) and held to the clip's end. Built as an RGBA sprite and alpha-faded
  in ffmpeg, per the master "Overlays - consolidated spec" compositing method.

Run: uv run python scripts/video/anim_06b_emit.py
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SRC_PNG = ROOT / "submission" / "figures" / "goldfield_20240925_185504_87_4001_emit_comparison.png"
CACHE_DIR = ROOT / "video" / "build" / "v2" / "cache"
CAPTION_PNG = CACHE_DIR / "06b_caption_sprite.png"
OUT = ROOT / "video" / "build" / "v2" / "upgrades" / "06b.mp4"

FPS = 30
DUR_S = 12.076  # beat 06b VO dur, docs/edit_plan.md EDL table (06b itself carries no +D pad)
W, H = 1920, 1080
NAVY = "#0a0e1a"
NAVY_HEX = "0x0a0e1a"

PANEL_W = 1920
PANEL_H = 598  # nearest even height at width 1920 for the source's 3.21:1 aspect
TOP_MARGIN = 40  # "seat high" -- small top margin, rest of the height is the lower band
ZOOM_END = 1.04

# CAP-06b caption, verbatim text + timing from docs/edit_plan.md's overlays table.
CAPTION_TEXT = "EMIT detects the same minerals — alunite r = 0.55 · jarosite r = 0.58"
CAPTION_IN_T = 4.0
FADE_S = 0.3


def _build_caption_sprite() -> None:
    """RGBA caption pill on a transparent 1920x1080 canvas (master overlay method)."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    # fig.text uses bottom-up figure fractions; the lower navy band is defined
    # in top-down pixel space (it's what's left below the letterboxed panel),
    # so its vertical centre has to be flipped before use here.
    band_top = TOP_MARGIN + PANEL_H
    center_from_top = band_top + (H - band_top) / 2
    y_frac = 1.0 - center_from_top / H
    fig.text(
        0.5,
        y_frac,
        CAPTION_TEXT,
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.4", "fc": NAVY, "ec": "none"},
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CAPTION_PNG, transparent=True)
    plt.close(fig)
    logger.info("wrote %s", CAPTION_PNG)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    if not SRC_PNG.exists():
        raise FileNotFoundError(f"source figure missing: {SRC_PNG}")
    _build_caption_sprite()

    n_frames = round(DUR_S * FPS)
    inc = (ZOOM_END - 1.0) / n_frames
    # Panel: 2x supersample -> slow centred zoompan -> pad onto the navy canvas,
    # seated high (small top margin, caption lives in the resulting lower band).
    panel_vf = (
        f"scale={PANEL_W * 2}:{PANEL_H * 2},"
        f"zoompan=z='min(zoom+{inc:.8f},{ZOOM_END})':d={n_frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={PANEL_W}x{PANEL_H}:fps={FPS},"
        f"pad={W}:{H}:0:{TOP_MARGIN}:color={NAVY_HEX},format=yuv420p"
    )
    # Caption sprite is drawn on its own full 1920x1080 transparent canvas (text
    # already at its correct absolute position within it), so it overlays flush
    # at (0,0) -- no x/y offset math. Fades alpha in at CAPTION_IN_T over FADE_S,
    # then holds (no fade-out -- CAP-06b runs "to clip end" per the overlays
    # table).
    overlay_vf = (
        f"[1:v]format=rgba,"
        f"fade=t=in:st={CAPTION_IN_T}:d={FADE_S}:alpha=1[ovf];"
        f"[base][ovf]overlay=x=0:y=0:enable='gte(t,{CAPTION_IN_T})'"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(FPS),
        "-i",
        str(SRC_PNG),
        "-loop",
        "1",
        "-framerate",
        str(FPS),
        "-i",
        str(CAPTION_PNG),
        "-filter_complex",
        f"[0:v]{panel_vf}[base];{overlay_vf}",
        "-t",
        f"{DUR_S}",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(OUT),
    ]
    _run(cmd)
    logger.info("wrote %s (%.3fs target)", OUT, DUR_S)


if __name__ == "__main__":
    main()
