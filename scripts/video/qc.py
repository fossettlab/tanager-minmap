"""Automated checks + the exact frame-at-T acceptance table from
docs/edit_plan.md "Acceptance checks -- per work packet (frame-at-T)".

Duration, stream params, loudness, and a few text checks are objectively
checkable here. Everything else in that table (legends readable, callouts on
the right marker, no black frames at a seam, end-card text) needs eyes on a
frame. Acceptance checks are stored as beat-relative offsets and resolved
against the live EDL, so a replacement narration take cannot silently move a
check to the wrong picture.
"""

from __future__ import annotations

import json
from pathlib import Path

from common import (
    BUILD_V2,
    DISSOLVE_D,
    FPS,
    HEIGHT,
    QC_V2,
    WIDTH,
    Beat,
    beat_by_name,
    build_edl,
    ffprobe_dur,
    ffprobe_video_info,
    probe_segment_durations,
    run,
)

# WP, beat, beat-relative offset (s), description. The offsets preserve the
# accepted editorial moments from edit_plan.md while the absolute time is
# always recomputed from the selected narration masters.
ACCEPTANCE_TABLE: list[tuple[str, str, float, str]] = [
    ("WP-00", "00", 0.15, "fade-up in progress (dim/near-black), spectral bars emerging left"),
    (
        "WP-00",
        "00",
        6.90,
        'title "The color you can\'t see" + sub "Planet Tanager...426 bands...376-2499 nm"',
    ),
    ("WP-01", "01", 2.842, "Goldfield RGB with LT-01 lower-left"),
    ("WP-01", "01", 14.842, "RGB pushed in off-centre onto the alteration ground; no lower-third"),
    ("WP-02", "02", 7.915, "two-up locator, both baked headers"),
    ("WP-03", "03", 10.536, "SWIR region framed; CALL-03a on the 2200 nm marker"),
    ("WP-03", "03", 14.536, "all three callouts present (2200 / 2265 / 2340 nm)"),
    ("WP-04", "04", 2.305, "Tanager full-VSWIR before state; S2 markers not yet in"),
    ("WP-04", "04", 13.305, "Al-OH doublet after state; CALL-04 arrow on native (50% loss) box"),
    ("WP-05", "05", 1.258, "LT-05; full 8-class legend + no detection readable"),
    ("WP-05", "05", 18.258, "framed on alunite/kaolinite cluster; legend still readable"),
    ("WP-06a", "06a", 2.864, "validation pair (Tanager alunite MTMF | Rockwell ASTER)"),
    ("WP-06a", "06a", 10.864, "matched highlight rectangles in both panels"),
    ("WP-06b", "06b", 5.784, "EMIT comparison; CAP-06b caption legible in lower band"),
    ("WP-07", "07", 1.708, "LT-07; full 4-class AGP legend readable"),
    ("WP-07", "07", 13.708, "framed on tailings high-AGP (red) zones"),
    ("WP-08", "08", 4.624, "end card: headline + result + repo line + archive + disclosure"),
    ("WP-08", "08", 14.524, "fading to black"),
    ("WP-ASM", "01", DISSOLVE_D / 2, "mid-dissolve motif<->hook"),
    ("WP-ASM", "06b", DISSOLVE_D / 2, "mid-dissolve 06a<->06b"),
    ("WP-ASM", "08", DISSOLVE_D / 2, "mid-dissolve 07<->end-card"),
]
# Fallback only; render_v2.py passes the live-probed total. v4: the 426-band
# re-TTS of beats 00/03 shifted this +0.317 s vs the plan's 170.136.
EXPECTED_TOTAL = 170.453


def check_total_duration(picture: Path, expected: float = EXPECTED_TOTAL) -> tuple[bool, str]:
    actual = ffprobe_dur(picture)
    ok = abs(actual - expected) <= (1 / FPS) + 0.02
    return ok, f"actual={actual:.3f}s expected={expected:.3f}s diff={actual - expected:+.3f}s"


def check_stream_params(picture: Path) -> tuple[bool, str]:
    info = ffprobe_video_info(picture)
    ok = (
        info.get("width") == str(WIDTH)
        and info.get("height") == str(HEIGHT)
        and info.get("r_frame_rate") == f"{FPS}/1"
        and info.get("pix_fmt") == "yuv420p"
    )
    return ok, str(info)


def check_loudness(final_video: Path) -> tuple[bool, str]:
    log = QC_V2 / "loudness_check.log"
    run(
        [
            "ffmpeg",
            "-i",
            str(final_video),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        log,
    )
    text = log.read_text()
    stats = json.loads(text[text.rindex("{") : text.rindex("}") + 1])
    i_lufs, tp, lra = float(stats["input_i"]), float(stats["input_tp"]), float(stats["input_lra"])
    ok = loudness_within_spec(i_lufs, tp)
    message = (
        f"I={i_lufs:.2f} LUFS (target -16 +/-1), TP={tp:.2f} dBTP (target <=-1.5), LRA={lra:.2f} LU"
    )
    return ok, message


def loudness_within_spec(integrated_lufs: float, true_peak_dbtp: float) -> bool:
    """Apply the documented release loudness limits exactly."""
    return abs(integrated_lufs - (-16.0)) <= 1.0 and true_peak_dbtp <= -1.5


def check_av_equal_length(final_video: Path) -> tuple[bool, str]:
    v = ffprobe_video_info(final_video)
    import subprocess

    a_dur = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "csv=p=0",
                str(final_video),
            ]
        ).strip()
    )
    v_dur = ffprobe_dur(final_video)
    ok = abs(v_dur - a_dur) <= (1 / FPS) + 0.05
    message = (
        f"video={v_dur:.3f}s audio={a_dur:.3f}s "
        f"diff={v_dur - a_dur:+.3f}s (pix_fmt={v.get('pix_fmt')})"
    )
    return ok, message


def check_vo_master_duration(expected: float = EXPECTED_TOTAL) -> tuple[bool, str]:
    vo = BUILD_V2 / "vo_master.wav"
    if not vo.exists():
        return False, "vo_master.wav not found"
    dur = ffprobe_dur(vo)
    ok = abs(dur - expected) <= 0.05
    return ok, f"vo_master.wav={dur:.3f}s expected={expected:.3f}s (+/-0.05)"


def check_srt_content(srt_path: Path) -> tuple[bool, str]:
    if not srt_path.exists():
        return False, f"{srt_path} not found"
    text = srt_path.read_text()
    cues = text.strip().split("\n\n")
    has_stakes_cue = any(
        "00:00:2" in cue
        and ("USGS" in cue or "BLM" in cue or "stakes" in cue.lower() or "matters" in cue.lower())
        for cue in cues
    )
    last_cue = cues[-1]
    ends_right = "open data community" in last_cue
    ok = has_stakes_cue and ends_right
    message = (
        f"stakes cue near 00:24 present={has_stakes_cue}; "
        f"final cue ends '...open data community'={ends_right}"
    )
    return ok, message


def run_qc(
    final_video: Path, srt_path: Path, expected_total: float = EXPECTED_TOTAL
) -> list[tuple[str, bool, str]]:
    checks = [
        (
            "WP-ASM total duration == VO length (+/-1 frame)",
            *check_total_duration(final_video, expected_total),
        ),
        ("WP-MUX stream params (1920x1080/30fps/yuv420p)", *check_stream_params(final_video)),
        ("WP-MUX loudness (-16 LUFS / -1.5 dBTP)", *check_loudness(final_video)),
        ("WP-MUX A/V equal length", *check_av_equal_length(final_video)),
        ("WP-AUD vo_master.wav == VO length (+/-0.05)", *check_vo_master_duration(expected_total)),
        ("WP-SRT stakes cue + final cue text", *check_srt_content(srt_path)),
    ]
    return checks


def extract_frame(video: Path, t: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["ffmpeg", "-y", "-i", str(video), "-ss", f"{t}", "-frames:v", "1", str(out)],
        QC_V2 / f"{out.stem}_extract.log",
    )
    return out


def acceptance_times(edl: list[Beat]) -> list[tuple[str, float, str]]:
    """Resolve beat-relative acceptance offsets against the live EDL."""
    return [
        (wp, beat_by_name(edl, beat_name).abs_start + offset, description)
        for wp, beat_name, offset, description in ACCEPTANCE_TABLE
    ]


def extract_acceptance_frames(
    final_video: Path,
    edl: list[Beat] | None = None,
) -> dict[str, Path]:
    """One frame per editorial acceptance check. `-ss` is placed
    after `-i` (accurate/slow seek) since several checks are dissolve
    midpoints where fast keyframe-seek slop of even a couple frames visibly
    changes the blend fraction."""
    if edl is None:
        edl = build_edl(probe_segment_durations())
    out: dict[str, Path] = {}
    for wp, t, _desc in acceptance_times(edl):
        label = f"{wp}_{t:.2f}".replace(".", "_")
        out[label] = extract_frame(final_video, t, QC_V2 / f"{label}.png")
    return out
