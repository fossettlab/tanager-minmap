"""Duration-preserving xfade assembly (docs/edit_plan.md "Transition assembly").

Concats within hard-cut runs (stream copy), then three xfade seams --
00->01, {06a}->{06b,07} (at 06b's start), and {...,07}->08. Offsets are the
abs_start of the *incoming* beat, recomputed from the probed EDL every run,
never hard-coded (the offset math is the same identity edit_plan.md derives:
each +D-padded clip's tail is exactly consumed by its dissolve overlap).
"""

from __future__ import annotations

from pathlib import Path

from common import (
    BUILD_V2,
    DISSOLVE_D,
    ENCODE_ARGS,
    FPS,
    HEIGHT,
    LOGS_V2,
    WIDTH,
    Beat,
    beat_by_name,
    run,
)

NORM = f"format=yuv420p,fps={FPS},scale={WIDTH}:{HEIGHT},setsar=1"


def _concat(paths: list[Path], out: Path, log: Path) -> None:
    listfile = out.with_suffix(".txt")
    listfile.write_text("".join(f"file '{p}'\n" for p in paths))
    run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)],
        log,
    )


def _xfade(a: Path, b: Path, offset: float, out: Path, log: Path) -> None:
    filt = (
        f"[0:v]{NORM}[a];[1:v]{NORM}[b];"
        f"[a][b]xfade=transition=fade:duration={DISSOLVE_D}:offset={offset}[v]"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(a),
            "-i",
            str(b),
            "-filter_complex",
            filt,
            "-map",
            "[v]",
            *ENCODE_ARGS,
            "-r",
            str(FPS),
            str(out),
        ],
        log,
    )


def assemble_picture(edl: list[Beat], clips: dict[str, tuple[Path, str]]) -> Path:
    """R1=[00], R2=concat[01..06a], R3=concat[06b,07], R4=[08];
    xfade at each seam per the plan's assembly graph."""
    LOGS_V2.mkdir(parents=True, exist_ok=True)
    r1 = clips["00"][0]
    r2 = BUILD_V2 / "r2.mp4"
    _concat(
        [clips[n][0] for n in ("01", "02", "03", "04", "05", "06a")], r2, LOGS_V2 / "concat_r2.log"
    )
    r3 = BUILD_V2 / "r3.mp4"
    _concat([clips[n][0] for n in ("06b", "07")], r3, LOGS_V2 / "concat_r3.log")
    r4 = clips["08"][0]

    t1 = BUILD_V2 / "t1.mp4"
    _xfade(r1, r2, beat_by_name(edl, "01").abs_start, t1, LOGS_V2 / "xfade_t1.log")
    t2 = BUILD_V2 / "t2.mp4"
    _xfade(t1, r3, beat_by_name(edl, "06b").abs_start, t2, LOGS_V2 / "xfade_t2.log")
    picture = BUILD_V2 / "picture.mp4"
    _xfade(t2, r4, beat_by_name(edl, "08").abs_start, picture, LOGS_V2 / "xfade_t3.log")
    return picture
