"""Generate the draft .srt sidecar (docs/edit_plan.md "Captions (.srt)" + Appendix D).

Cue text is verbatim from video/segments_v2/*.txt, split at sentence breaks
into 1-3 cues per beat and spread proportionally across the beat's abs
start/end (a duration-weighted estimate -- the TTS gives no per-word
timestamps). Text is never trimmed or paraphrased to fit the 2-line/42-char
guideline; a cue that runs long stays long rather than dropping words.
"""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

from common import SEGMENTS_V2, Beat, beat_by_name

MAX_CHARS_PER_LINE = 42
TARGET_CHARS_PER_CUE = 75  # keeps most cues to <=2 lines at 42 chars/line
FORCED_ONE_CUE = {"00", "08"}  # Appendix D: "One-beat beats (00, 08) = one cue each."
FORCED_CUES = {"03": 3, "06": 3}  # Appendix D: beat 03 -> 3 cues, beat 06 -> 2-3 cues

SEGMENT_STEM = {
    "00": "00_title",
    "01": "01_hook",
    "02": "02_stakes",
    "03": "03_data",
    "04": "04_ablation",
    "05": "05_livemap",
    "07": "07_amd",
    "08": "08_close",
}


def _segment_text(seg_stem: str) -> str:
    return (SEGMENTS_V2 / f"{seg_stem}.txt").read_text().strip()


def _split_sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


def _n_cues_for(beat_name: str, seg_stem: str) -> int:
    if beat_name in FORCED_ONE_CUE:
        return 1
    if beat_name in FORCED_CUES:
        return FORCED_CUES[beat_name]
    text = _segment_text(seg_stem)
    n_sentences = len(_split_sentences(text))
    return max(1, min(n_sentences, math.ceil(len(text) / TARGET_CHARS_PER_CUE)))


def _chunk_sentences(sentences: list[str], n_cues: int) -> list[str]:
    n_cues = max(1, min(n_cues, len(sentences)))
    per = len(sentences) / n_cues
    return [" ".join(sentences[round(i * per) : round((i + 1) * per)]) for i in range(n_cues)]


def _wrap(text: str) -> str:
    return "\n".join(
        textwrap.wrap(text, width=MAX_CHARS_PER_LINE, break_long_words=False) or [text]
    )


def _fmt_ts(t: float) -> str:
    if not math.isfinite(t) or t < 0:
        raise ValueError("caption timestamps must be finite and non-negative")
    total_ms = round(t * 1000)
    total_seconds, ms = divmod(total_ms, 1000)
    total_minutes, s = divmod(total_seconds, 60)
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _cues_for_span(
    seg_stem: str, start: float, end: float, n_cues: int
) -> list[tuple[float, float, str]]:
    """Split one narration segment's text into n_cues at sentence breaks,
    spread across [start, end] weighted by character count -- we don't have
    per-word timestamps from the TTS, but a short cue ("Two independent
    checks.") reading for as long as a much longer one reads oddly on screen,
    so length-weighting is a better duration estimate than an equal split."""
    sentences = _split_sentences(_segment_text(seg_stem))
    chunks = _chunk_sentences(sentences, n_cues)
    weights = [len(c) for c in chunks]
    total_weight = sum(weights)
    dur = end - start
    cues, t = [], start
    for chunk, w in zip(chunks, weights):
        c_dur = dur * w / total_weight
        cues.append((t, t + c_dur, chunk))
        t += c_dur
    return cues


def build_srt(edl: list[Beat], out: Path) -> Path:
    b06b = beat_by_name(edl, "06b")
    cue_i = 1
    lines: list[str] = []
    for beat in edl:
        if beat.name == "06b":
            continue  # folded into the 06a span below (one narration segment)
        if beat.name == "06a":
            seg_stem, span_start, span_end = "06_validation", beat.abs_start, b06b.abs_end
        else:
            seg_stem, span_start, span_end = SEGMENT_STEM[beat.name], beat.abs_start, beat.abs_end
        n_cues = _n_cues_for("06" if beat.name == "06a" else beat.name, seg_stem)
        for c_start, c_end, text in _cues_for_span(seg_stem, span_start, span_end, n_cues):
            lines.append(f"{cue_i}\n{_fmt_ts(c_start)} --> {_fmt_ts(c_end)}\n{_wrap(text)}\n")
            cue_i += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out
