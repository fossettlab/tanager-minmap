"""Audio backbone: per-segment VO concat, music-bed gain envelope + duck,
two-pass loudnorm to -16 LUFS / -1.5 dBTP (docs/edit_plan.md "Audio mix + loudness").

Canonical bed is video/audio_v2/music_bed_v2a.mp3 (steady texture, LRA 1.8 --
a deliberate automation substrate, not a flat mistake: the composition_plan
regeneration came back level-flat per-beat, so team-lead's round-2 call was
to implement the intended dynamic arc as explicit pre-duck gain automation
in code (BED_GAIN_BREAKPOINTS below) rather than chase more bed regens.
music_bed_v1a.mp3 is the named fallback if v2a is missing; music_bed_v1b.mp3
and music_bed_v2b.mp3 are deliberately excluded (v1b has a ~1.6s dropout,
v2b was the rejected composition_plan candidate). If no bed is found at all,
the mix stage is a no-op passthrough of VO alone -- draft renders are never
blocked on the music generation step.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from common import (
    AUDIO_V2,
    BUILD_V2,
    LOGS_V2,
    QC_V2,
    SEGMENT_FILES,
    Beat,
    beat_by_name,
    ffprobe_dur,
    run,
)

VO_ORDER = [
    "00_title",
    "01_hook",
    "02_stakes",
    "03_data",
    "04_ablation",
    "05_livemap",
    "06_validation",
    "07_amd",
    "08_close",
]
MUSIC_BED_CANDIDATES = [
    AUDIO_V2 / "music_bed_v2a.mp3",
    AUDIO_V2 / "music_bed_v1a.mp3",
    AUDIO_V2 / "music_bed.wav",
    BUILD_V2 / "music_bed.wav",
]
BED_TAIL_FADE_S = 1.0  # bed is longer than the picture; fade it out at the cut
# rather than let amix's hard trim (duration=first) clip it mid-note -- the
# bed's own natural fade (172.2-174.0s per team-lead) starts after our 170.1s
# cutoff, so that fade never plays without this.

# Pre-duck gain envelope (team-lead's round-2 call, docs/edit_plan.md's R1-C
# fix): dB offsets from the base bed level (volume=0.14 below, ~-17 dBFS),
# ramped RAMP_S-wide at each EDL boundary. 03 sparse under the callouts, 04
# the one deliberate crest (holding through the ~1:20 mark the critique
# wanted), everything else at the reference level -- the bed's own soft head
# and natural tail fade cover 00 and 08 without any extra ramp there.
RAMP_S = 2.0
BED_GAIN_03_DB = -2.5
BED_GAIN_04_DB = 1.5


def bed_gain_breakpoints(edl: list[Beat]) -> list[tuple[float, float]]:
    """(time, dB) breakpoints of the piecewise-linear envelope, in absolute
    EDL time. The 04 exit ramp is explicitly "the last 2s of 04" (not
    centered on the 04->05 boundary); every other transition ramps
    symmetrically across the boundary."""
    b03, b04 = beat_by_name(edl, "03"), beat_by_name(edl, "04")
    half = RAMP_S / 2
    return [
        (0.0, 0.0),
        (b03.abs_start - half, 0.0),
        (b03.abs_start + half, BED_GAIN_03_DB),
        (b04.abs_start - half, BED_GAIN_03_DB),
        (b04.abs_start + half, BED_GAIN_04_DB),
        (b04.abs_end - RAMP_S, BED_GAIN_04_DB),  # "ramp down over the last 2s" of 04
        (b04.abs_end, 0.0),
        (edl[-1].abs_end, 0.0),
    ]


def _piecewise_gain_expr(breakpoints: list[tuple[float, float]]) -> str:
    """ffmpeg time-expression for a piecewise-linear dB(t) envelope through
    the given (time, dB) breakpoints -- linear ramp between consecutive
    points, flat before the first / after the last."""
    expr = f"{breakpoints[-1][1]}"
    for i in range(len(breakpoints) - 1, 0, -1):
        t0, v0 = breakpoints[i - 1]
        t1, v1 = breakpoints[i]
        interp = f"({v0}+({v1}-{v0})*(t-{t0})/({t1}-{t0}))"
        expr = f"if(lt(t,{t1}),{interp},{expr})"
    return expr


def build_vo_master(segment_paths: Mapping[str, Path] | None = None) -> Path:
    """Concat the nine segments, upmixed mono->stereo center (master render
    setting: "VO mono upmixed to stereo center") so the output is stereo even
    when the music-bed mix stage below is a no-op passthrough.

    Each segment is decoded individually and padded/trimmed to exactly its
    own ffprobe-measured duration before concatenating. Feeding the mp3s
    straight into the concat demuxer (as docs/edit_plan.md's literal command
    does) loses ~30-50ms per file to MP3 encoder-delay/padding that ffmpeg's
    decode doesn't correct for -- summed over 9 segments that cost ~0.38s and
    silently desynced the result, since every video clip's length downstream
    is driven by these same ffprobe durations.
    """
    selected = (
        {name: AUDIO_V2 / filename for name, filename in SEGMENT_FILES.items()}
        if segment_paths is None
        else dict(segment_paths)
    )
    if set(selected) != set(VO_ORDER):
        missing = sorted(set(VO_ORDER) - set(selected))
        extra = sorted(set(selected) - set(VO_ORDER))
        raise ValueError(f"narration segment mapping mismatch: missing={missing}, extra={extra}")

    seg_dir = BUILD_V2 / "vo_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_paths = []
    for name in VO_ORDER:
        src = selected[name]
        dur = ffprobe_dur(src)
        seg_out = seg_dir / f"{name}.wav"
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-af",
                "aformat=channel_layouts=stereo,apad",
                "-t",
                f"{dur}",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(seg_out),
            ],
            LOGS_V2 / f"vo_seg_{name}.log",
        )
        seg_paths.append(seg_out)

    listfile = BUILD_V2 / "vo_list.txt"
    listfile.write_text("".join(f"file '{p}'\n" for p in seg_paths))
    out = BUILD_V2 / "vo_master.wav"
    run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)],
        LOGS_V2 / "vo_concat.log",
    )
    return out


def _find_music_bed() -> Path | None:
    return next((p for p in MUSIC_BED_CANDIDATES if p.exists()), None)


def mix_with_music(
    vo: Path,
    edl: list[Beat],
    *,
    music_bed: Path | None = None,
    strict: bool = False,
) -> tuple[Path, bool]:
    """Pre-duck gain envelope (bed_gain_breakpoints) then sidechain-duck the
    bed under VO, trimmed/faded to the VO's own length. Returns (mix_path,
    had_music); with no bed found, mix_path == vo unchanged (the stub)."""
    bed = music_bed if strict else (music_bed if music_bed is not None else _find_music_bed())
    if bed is None:
        if strict:
            raise FileNotFoundError("strict release requires the contract-selected music bed")
        return vo, False
    if not bed.is_file():
        raise FileNotFoundError(f"music bed not found: {bed}")
    vo_dur = ffprobe_dur(vo)
    fade_start = vo_dur - BED_TAIL_FADE_S
    gain_expr = _piecewise_gain_expr(bed_gain_breakpoints(edl))
    out = BUILD_V2 / "mix.wav"
    filt = (
        "[0:a]aformat=cl=stereo:sample_rates=44100[vo];"
        f"[1:a]aformat=cl=stereo:sample_rates=44100,"
        f"volume=volume='0.14*pow(10,({gain_expr})/20)':eval=frame,"
        f"afade=t=out:st={fade_start}:d={BED_TAIL_FADE_S}[mus];"
        "[mus][vo]sidechaincompress=threshold=0.05:ratio=8:attack=5:release=250[mduck];"
        "[vo][mduck]amix=inputs=2:normalize=0:duration=first[mix]"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(vo),
            "-i",
            str(bed),
            "-filter_complex",
            filt,
            "-map",
            "[mix]",
            str(out),
        ],
        LOGS_V2 / "mix.log",
    )
    return out, True


def render_bed_stem(edl: list[Beat]) -> Path | None:
    """The bed alone with the base level + gain envelope + tail fade applied,
    but BEFORE ducking -- the artifact the R1-C acceptance check measures
    per-beat energy against (measure_bed_per_beat)."""
    bed = _find_music_bed()
    if bed is None:
        return None
    vo_end = edl[-1].abs_end
    fade_start = vo_end - BED_TAIL_FADE_S
    gain_expr = _piecewise_gain_expr(bed_gain_breakpoints(edl))
    out = BUILD_V2 / "bed_stem.wav"
    af = (
        "aformat=cl=stereo:sample_rates=44100,"
        f"volume=volume='0.14*pow(10,({gain_expr})/20)':eval=frame,"
        f"afade=t=out:st={fade_start}:d={BED_TAIL_FADE_S}"
    )
    run(
        ["ffmpeg", "-y", "-i", str(bed), "-af", af, "-t", f"{vo_end}", str(out)],
        LOGS_V2 / "bed_stem.log",
    )
    return out


def measure_bed_per_beat(stem: Path, edl: list[Beat]) -> list[tuple[str, float]]:
    """Mean volume (dBFS) of the pre-duck bed stem within each beat's abs
    span -- docs/edit_plan.md's R1-C acceptance measurement: 04 mean should
    be >=2 dB above 03; 03 sparsest interior; soft bookends at 00/08.

    Trims each beat to its own file before measuring, in two separate ffmpeg
    calls. Combining -ss/-t with -af volumedetect -f null - in one command
    does NOT restrict what volumedetect sees (verified: it reports the same
    mean as the untrimmed whole file) -- only a real trim-then-measure
    two-step gives per-slice numbers.
    """
    QC_V2.mkdir(parents=True, exist_ok=True)
    slice_dir = QC_V2 / "bed_energy_slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for beat in edl:
        sliced = slice_dir / f"{beat.name}.wav"
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(stem),
                "-ss",
                f"{beat.abs_start}",
                "-t",
                f"{beat.vo_dur}",
                "-c:a",
                "pcm_s16le",
                str(sliced),
            ],
            QC_V2 / f"bed_energy_{beat.name}_trim.log",
        )
        log = QC_V2 / f"bed_energy_{beat.name}.log"
        run(["ffmpeg", "-i", str(sliced), "-af", "volumedetect", "-f", "null", "-"], log)
        text = log.read_text()
        mean = next(
            (
                float(line.split("mean_volume:")[1].split("dB")[0].strip())
                for line in text.splitlines()
                if "mean_volume:" in line
            ),
            float("nan"),
        )
        rows.append((beat.name, mean))
    return rows


def _parse_loudnorm_json(log_text: str) -> dict[str, str]:
    start, end = log_text.rindex("{"), log_text.rindex("}") + 1
    return json.loads(log_text[start:end])


def loudnorm_two_pass(mix: Path) -> Path:
    measure_log = LOGS_V2 / "loudnorm_measure.log"
    run(
        [
            "ffmpeg",
            "-i",
            str(mix),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        measure_log,
    )
    stats = _parse_loudnorm_json(measure_log.read_text())
    out = BUILD_V2 / "audio_final.wav"
    af = (
        f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={stats['input_i']}:"
        f"measured_TP={stats['input_tp']}:measured_LRA={stats['input_lra']}:"
        f"measured_thresh={stats['input_thresh']}:offset={stats['target_offset']}:linear=true"
    )
    run(
        ["ffmpeg", "-y", "-i", str(mix), "-af", af, "-ar", "44100", str(out)],
        LOGS_V2 / "loudnorm_apply.log",
    )
    return out


def build_audio(
    edl: list[Beat],
    *,
    segment_paths: Mapping[str, Path] | None = None,
    music_bed: Path | None = None,
    strict: bool = False,
) -> Path:
    """Build the final audio mix.

    Draft mode preserves the historical candidate search and VO-only escape
    hatch. Strict release mode receives one hash-verified bed and one exact
    mapping of narration masters; absence is fatal.
    """
    vo = build_vo_master(segment_paths)
    mix, had_music = mix_with_music(vo, edl, music_bed=music_bed, strict=strict)
    music_status = "found, ducked + gain envelope" if had_music else "absent -- VO-only mix (stub)"
    print(f"  music bed: {music_status}")
    return loudnorm_two_pass(mix)
