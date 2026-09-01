"""Per-beat clip builders for the v2 submission video (docs/edit_plan.md).

Each build_NN(beat, log_dir) returns (clip_path, tier). `tier` is one of:
  "designed"           -- not a swappable beat; built directly from the plan.
  "upgrade"            -- another worker's pre-rendered animation / live
                          capture, found at UPGRADES_V2/{beat}.mp4.
  "fallback-still"      -- an improved static composite another worker built
                          (05/07's legend-safe reframe), consumed with our own
                          zoompan + lower-third.
  "fallback"            -- built here from the plan's stated fallback.
  "fallback-emergency"  -- the source figure with no legend fix; used only if
                          neither an upgrade nor a fallback-still exists, so
                          the pipeline never hard-fails on a missing asset.
  "tanager-still"       -- strict-release-only 05/07 path using the frozen,
                          redistribution-safe Tanager-derived composites.
render_v2.py prints the tier render_v2 actually used for each swappable beat.
Draft mode retains the convenience fallback chain above. Strict release mode
passes one explicit tier and asset per swappable beat; no discovery or fallback
is performed there.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import overlays  # noqa: E402
from common import (  # noqa: E402
    BG_FFMPEG,
    BG_HEX,
    BH,
    BUILD,
    BUILD_V2,
    CLIPS_V2,
    ENCODE_ARGS,
    FONT_BOLD,
    FONT_REGULAR,
    FPS,
    GF,
    HEIGHT,
    LOGS_V2,
    ROOT,
    SUBMISSION_FIGURES,
    UPGRADES_V2,
    WIDTH,
    Beat,
    canvas_point_to_screen,
    clamped_crop_window,
    ffprobe_dur,
    run,
)
from matplotlib.font_manager import FontProperties  # noqa: E402

# needs-user (edit_plan.md "Remaining needs-user"): confirm the repo slug and
# mint the DOI before final export. Draft renders show the literal placeholder.
REPO_SLUG = "github.com/bradleylab/tanager-rocks"
DOI_LINE_DRAFT = "Archive: DOI pending"


# --------------------------------------------------------------------------
# Generic image/video treatments
# --------------------------------------------------------------------------


def resolve_upgrade(beat_id: str) -> Path | None:
    p = UPGRADES_V2 / f"{beat_id}.mp4"
    return p if p.exists() else None


def normalize_upgrade(src: Path, dur: float, out: Path, log: Path) -> None:
    """Trim/pad an externally-produced clip to exactly `dur` and normalize to
    the master render settings so it splices cleanly into assembly.

    Beats that carry a dissolve tail (07, among the swappable ones) need
    render_dur = VOdur + D; an upgrade built by a worker who only targeted
    VOdur comes in short. Extend by holding the last frame rather than
    leaving it short -- a hard-cut concat/xfade offset computed from the
    plan's EDL would otherwise drift out of sync with the narration.
    """
    src_dur = ffprobe_dur(src)
    vf = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:-1:-1:color={BG_FFMPEG},fps={FPS},format=yuv420p"
    )
    if src_dur < dur:
        vf += f",tpad=stop_mode=clone:stop_duration={dur - src_dur:.6f}"
        run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, *ENCODE_ARGS, str(out)], log)
    else:
        run(
            ["ffmpeg", "-y", "-i", str(src), "-t", f"{dur}", "-vf", vf, *ENCODE_ARGS, str(out)], log
        )


def image_clip(
    img: Path,
    dur: float,
    out: Path,
    log: Path,
    *,
    fit: str = "cover",
    zoom_end: float = 1.0,
    cx: float = 0.5,
    cy: float = 0.5,
) -> None:
    """Render a still image to a `dur`-second clip with a zoompan push.

    fit="cover"   -- 2x-supersample + crop to fill 1920x1080 (photos; some
                     edge content is cropped, matches v1's clip_photo).
    fit="contain" -- scale to fit *inside* 1920x1080, navy-pad the rest
                     (charts; nothing is cropped, matches v1's clip_chart).
    fit="none"    -- input is already exactly 1920x1080 (pre-built canvases).
    zoom_end=1.0 means no motion; zoompan's own `d=` frame count provides the
    hold, so no `-loop` flag is needed for any of the three modes.
    """
    frames = round(dur * FPS)
    inc = (zoom_end - 1.0) / max(frames, 1)
    pre = {
        "cover": "scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160,",
        "contain": f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:-1:-1:color={BG_FFMPEG},",
        "none": "",
    }[fit]
    vf = (
        f"{pre}zoompan=z='min(zoom+{inc:.6f},{zoom_end})':d={frames}:"
        f"x='min(max({cx}*iw-(iw/zoom/2),0),iw-iw/zoom)':"
        f"y='min(max({cy}*ih-(ih/zoom/2),0),ih-ih/zoom)':s={WIDTH}x{HEIGHT}:fps={FPS}"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(img),
            "-vf",
            vf,
            "-t",
            f"{dur}",
            "-r",
            str(FPS),
            *ENCODE_ARGS,
            str(out),
        ],
        log,
    )


def seat_high(img: Path, dur: float, out: Path, log: Path, zoom_end: float = 1.0) -> None:
    """Scale to width=1920 and seat at the TOP of the canvas; the navy band
    below becomes an information band (used for the 2.6:1 / 3.2:1 letterboxed
    figures -- 04, 06b -- instead of dead pillarbox)."""
    frames = round(dur * FPS)
    inc = (zoom_end - 1.0) / max(frames, 1)
    vf = (
        f"scale={WIDTH}:-2,pad={WIDTH}:{HEIGHT}:0:0:color={BG_FFMPEG},"
        f"zoompan=z='min(zoom+{inc:.6f},{zoom_end})':d={frames}:x=0:y=0:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(img),
            "-vf",
            vf,
            "-t",
            f"{dur}",
            "-r",
            str(FPS),
            *ENCODE_ARGS,
            str(out),
        ],
        log,
    )


def composite_overlays(
    base: Path,
    layers: list[tuple[Path, float, float, bool]],
    out: Path,
    log: Path,
    fade: float = 0.3,
) -> None:
    """Composite N full-frame RGBA sprites onto `base` in one ffmpeg call.

    Each layer is (sprite_path, t_in, t_out, rise). `rise` adds the 12px
    rise-in used for lower-thirds (Appendix A); callouts/captions just fade.
    Filter template verbatim from docs/edit_plan.md's overlays spec.
    """
    if not layers:
        run(["ffmpeg", "-y", "-i", str(base), "-c", "copy", str(out)], log)
        return
    base_dur = ffprobe_dur(base)
    # Sprites are single-frame PNGs; -loop 1 makes each an infinite stream so
    # the timed fade/enable expressions below have a continuous per-frame
    # timestamp to evaluate against (a bare `-i sprite.png` decodes to exactly
    # one frame at pts=0, so `fade=t=in:st=<t_in>` never fires for t_in > 0).
    inputs: list[str] = ["-i", str(base)]
    parts: list[str] = []
    prev = "0:v"
    for i, (sprite, t_in, t_out, rise) in enumerate(layers, start=1):
        inputs += ["-loop", "1", "-i", str(sprite)]
        y_expr = f"12*(1-min(max((t-{t_in})/0.3,0),1))" if rise else "0"
        parts.append(
            f"[{i}:v]format=rgba,fade=t=in:st={t_in}:d={fade}:alpha=1,"
            f"fade=t=out:st={t_out - fade}:d={fade}:alpha=1[ov{i}];"
        )
        nxt = f"stage{i}" if i < len(layers) else "vout"
        parts.append(
            f"[{prev}][ov{i}]overlay=x=0:y='{y_expr}':enable='between(t,{t_in},{t_out})'[{nxt}];"
        )
        prev = nxt
    filt = "".join(parts).rstrip(";")
    run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filt,
            "-map",
            f"[{prev}]",
            "-t",
            f"{base_dur}",
            *ENCODE_ARGS,
            "-r",
            str(FPS),
            str(out),
        ],
        log,
    )


def _try_upgrade(beat_id: str, dur: float, out: Path, log_dir: Path) -> bool:
    src = resolve_upgrade(beat_id)
    if src is None:
        return False
    normalize_upgrade(src, dur, out, log_dir / f"{beat_id}_upgrade.log")
    return True


# --------------------------------------------------------------------------
# 00 -- title / motif
# --------------------------------------------------------------------------


def build_00(beat: Beat, log_dir: Path) -> tuple[Path, str]:
    motif = BUILD / "motif.mp4"
    motif_dur = ffprobe_dur(motif)
    out = CLIPS_V2 / "00.mp4"
    log = log_dir / "00.log"
    trim_start, needed = 0.5, beat.render_dur
    available = motif_dur - trim_start
    base_vf = f"scale={WIDTH}:{HEIGHT},fade=t=in:st=0:d=0.3"
    if available >= needed:
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{trim_start}",
                "-i",
                str(motif),
                "-t",
                f"{needed}",
                "-vf",
                base_vf,
                "-r",
                str(FPS),
                *ENCODE_ARGS,
                str(out),
            ],
            log,
        )
    else:
        # edit_plan.md's trim (t=0.5, dur=VOdur+D=7.488s) asks for 0.488s more
        # than motif.mp4 contains (7.5s total, confirmed via ffprobe). The
        # tail is already a held title card (render_motif.py plateaus title
        # alpha by ~t=6.6s), so padding by cloning the last frame is visually
        # identical to a longer source -- not a fabricated frame, a hold.
        hold = needed - available
        # -t must precede -i here (an input-read limit): placed after -i it
        # becomes an output-duration cap and truncates tpad's extension away.
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{trim_start}",
                "-t",
                f"{available}",
                "-i",
                str(motif),
                "-vf",
                f"{base_vf},tpad=stop_mode=clone:stop_duration={hold}",
                "-r",
                str(FPS),
                *ENCODE_ARGS,
                str(out),
            ],
            log,
        )
    return out, "designed"


# --------------------------------------------------------------------------
# 01 -- hook
# --------------------------------------------------------------------------


def build_01(beat: Beat, log_dir: Path) -> tuple[Path, str]:
    img = SUBMISSION_FIGURES / "goldfield_rgb.png"
    base = CLIPS_V2 / "01_base.mp4"
    image_clip(
        img, beat.render_dur, base, log_dir / "01.log", fit="cover", zoom_end=1.25, cx=0.40, cy=0.62
    )
    sprite = overlays.lower_third(
        "Goldfield district, Nevada", "Planet Tanager true color", BUILD_V2 / "sprites" / "lt01.png"
    )
    out = CLIPS_V2 / "01.mp4"
    composite_overlays(base, [(sprite, 1.0, 5.5, True)], out, log_dir / "01_overlay.log")
    return out, "designed"


# --------------------------------------------------------------------------
# 02 -- stakes (two-up locator)
# --------------------------------------------------------------------------


def build_twoup() -> Path:
    """Own copy of v1's make_twoup(), written under build/v2 (never touches
    the existing video/build/twoup.png)."""
    out = BUILD_V2 / "twoup_v2.png"
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.0), facecolor=BG_HEX)
    for ax, img, label in (
        (axes[0], SUBMISSION_FIGURES / "bingham_rgb.png", "Bingham Canyon, Utah"),
        (axes[1], SUBMISSION_FIGURES / "goldfield_rgb.png", "Goldfield district, Nevada"),
    ):
        ax.imshow(mpimg.imread(img))
        ax.set_title(label, color="white", fontsize=30, pad=16)
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.06)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=BG_HEX)
    plt.close(fig)
    return out


def build_02(beat: Beat, log_dir: Path) -> tuple[Path, str]:
    twoup = build_twoup()
    out = CLIPS_V2 / "02.mp4"
    image_clip(
        twoup, beat.render_dur, out, log_dir / "02.log", fit="cover", zoom_end=1.08, cx=0.5, cy=0.5
    )
    return out, "designed"


# --------------------------------------------------------------------------
# 03 -- the data (animated upgrade preferred; static push-pan + callouts fallback)
# --------------------------------------------------------------------------

SPECTRA_ZOOM_START, SPECTRA_ZOOM_END = 1.0, 1.35
SPECTRA_CX_START, SPECTRA_CY_START = 0.5, 0.5
SPECTRA_CX_END, SPECTRA_CY_END = 0.72, 0.30
SPECTRA_ARRIVE_S = 9.0
# (label, wavelength_nm, t_in, t_out) -- verbatim from the overlays table.
SPECTRA_CALLOUTS = [
    ("Al–OH · 2200 nm", 2200.0, 10.0, 17.0),
    ("jarosite · 2265 nm", 2265.0, 12.0, 17.0),
    ("gypsum · 2340 nm", 2340.0, 14.0, 17.0),
]


def _find_marker_columns(img_path: Path, n: int = 3) -> list[int]:
    """Column positions of the dotted vertical reference lines in
    goldfield_spectra.png, found by scanning for gray (not white, not
    curve-colored) pixel density in the right ~20% of the chart -- avoids
    hardcoding figure-specific pixel offsets that would silently drift if the
    source figure regenerates."""
    a = np.array(Image.open(img_path).convert("RGB"))
    h, w, _ = a.shape
    y0, y1 = int(0.12 * h), int(0.92 * h)
    x0, x1 = int(0.8 * w), int(0.98 * w)
    band = a[y0:y1, x0:x1, :].astype(int)
    gray = (
        (np.abs(band[:, :, 0] - band[:, :, 1]) < 8)
        & (np.abs(band[:, :, 1] - band[:, :, 2]) < 8)
        & (band[:, :, 0] < 210)
        & (band[:, :, 0] > 100)
    )
    frac = gray.mean(axis=0)
    cols = np.where(frac > 0.015)[0] + x0
    clusters: list[tuple[int, int]] = []
    start = prev = None
    for c in cols:
        if start is None:
            start = c
        elif c - prev > 15:
            clusters.append((start, prev))
            start = c
        prev = c
    if start is not None:
        clusters.append((start, prev))
    mids = sorted((s + e) // 2 for s, e in clusters)
    if len(mids) != n:
        raise RuntimeError(f"expected {n} marker lines in {img_path}, found {len(mids)}: {mids}")
    return mids


def _eased_frac(arrive_s: float) -> str:
    """ffmpeg-expression smoothstep of time/arrive_s, clamped to [0, 1]."""
    p = f"min(max(time/{arrive_s},0),1)"
    return f"(3*({p})*({p})-2*({p})*({p})*({p}))"


def spectra_push_fallback(beat: Beat, out: Path, log_dir: Path) -> Path:
    """Static goldfield_spectra.png, eased push-pan front-loaded to arrive by
    +9s (edit_plan.md #03 fallback), then three timed callout sprites at the
    settled framing's actual on-screen marker positions."""
    img = SUBMISSION_FIGURES / "goldfield_spectra.png"
    src_w, src_h = Image.open(img).size
    scale = HEIGHT / src_h  # height-constrained fit (chart aspect < 16:9)
    x_off = (WIDTH - src_w * scale) / 2

    frac = _eased_frac(SPECTRA_ARRIVE_S)
    z = f"({SPECTRA_ZOOM_START}+{frac}*({SPECTRA_ZOOM_END}-{SPECTRA_ZOOM_START}))"
    cxe = f"({SPECTRA_CX_START}+{frac}*({SPECTRA_CX_END}-{SPECTRA_CX_START}))"
    cye = f"({SPECTRA_CY_START}+{frac}*({SPECTRA_CY_END}-{SPECTRA_CY_START}))"
    x_expr = f"min(max({cxe}*iw-(iw/{z}/2),0),iw-iw/{z})"
    y_expr = f"min(max({cye}*ih-(ih/{z}/2),0),ih-ih/{z})"
    frames = round(beat.render_dur * FPS)
    vf = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:-1:-1:color={BG_FFMPEG},"
        f"zoompan=z='{z}':d={frames}:x='{x_expr}':y='{y_expr}':s={WIDTH}x{HEIGHT}:fps={FPS}"
    )
    base = CLIPS_V2 / "03_base.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(img),
            "-vf",
            vf,
            "-t",
            f"{beat.render_dur}",
            "-r",
            str(FPS),
            *ENCODE_ARGS,
            str(base),
        ],
        log_dir / "03.log",
    )

    mids = _find_marker_columns(img)
    end_window = clamped_crop_window(SPECTRA_CX_END, SPECTRA_CY_END, SPECTRA_ZOOM_END)
    tip_canvas_y = 500.0  # comfortably inside the settled crop; on the dotted lines
    layers = []
    for idx, ((label, nm, t_in, t_out), mid_px) in enumerate(zip(SPECTRA_CALLOUTS, mids)):
        canvas_x = mid_px * scale + x_off
        tip = canvas_point_to_screen(canvas_x, tip_canvas_y, end_window)
        label_xy = (tip[0], tip[1] - 90 - 55 * idx)  # stagger to avoid collision
        sprite = overlays.chart_callout(
            label, tip, label_xy, BUILD_V2 / "sprites" / f"call03_{nm:.0f}.png"
        )
        layers.append((sprite, t_in, t_out, False))
    composite_overlays(base, layers, out, log_dir / "03_overlay.log")
    return out


def build_03(
    beat: Beat,
    log_dir: Path,
    *,
    strict_tier: str | None = None,
    strict_asset: Path | None = None,
) -> tuple[Path, str]:
    out = CLIPS_V2 / "03.mp4"
    if strict_tier == "upgrade":
        if strict_asset is None:
            raise ValueError("strict beat 03 upgrade requires an exact asset")
        normalize_upgrade(strict_asset, beat.render_dur, out, log_dir / "03_upgrade.log")
        return out, "upgrade"
    if strict_tier == "fallback":
        spectra_push_fallback(beat, out, log_dir)
        return out, "fallback"
    if strict_tier is not None:
        raise ValueError(f"unsupported strict tier for beat 03: {strict_tier}")
    if _try_upgrade("03", beat.render_dur, out, log_dir):
        return out, "upgrade"
    spectra_push_fallback(beat, out, log_dir)
    return out, "fallback"


# --------------------------------------------------------------------------
# 04 -- central result (animated upgrade preferred; static + native-box arrow fallback)
# --------------------------------------------------------------------------

# Centre of the figure's own "(50% loss)" callout box, measured on
# bingham_..._band_ablation.png after scale-to-width=1920 (seat_high). No new
# number is shown -- the arrow just points at text the figure already prints.
CALL04_BOX_CANVAS = (1419.0, 158.0)


def build_04(
    beat: Beat,
    log_dir: Path,
    *,
    strict_tier: str | None = None,
    strict_asset: Path | None = None,
) -> tuple[Path, str]:
    out = CLIPS_V2 / "04.mp4"
    if strict_tier == "upgrade":
        if strict_asset is None:
            raise ValueError("strict beat 04 upgrade requires an exact asset")
        normalize_upgrade(strict_asset, beat.render_dur, out, log_dir / "04_upgrade.log")
        return out, "upgrade"
    if strict_tier not in (None, "fallback"):
        raise ValueError(f"unsupported strict tier for beat 04: {strict_tier}")
    if strict_tier is None and _try_upgrade("04", beat.render_dur, out, log_dir):
        return out, "upgrade"
    # Deviation from edit_plan.md's stated fallback: a two-state cross-dissolve
    # needs a "Tanager-only" render of the ablation figure, which doesn't exist
    # as a committed asset (only the combined Tanager+S2 figure is in
    # submission/figures/). Falling back further to a single static seat-high
    # frame -- still correct, just without the before/after staging.
    img = SUBMISSION_FIGURES / f"{BH}_band_ablation.png"
    base = CLIPS_V2 / "04_base.mp4"
    seat_high(img, beat.render_dur, base, log_dir / "04.log")
    tip = CALL04_BOX_CANVAS
    tail = (tip[0], tip[1] + 110)
    sprite = overlays.arrow_only(tip, tail, BUILD_V2 / "sprites" / "call04.png")
    composite_overlays(base, [(sprite, 11.0, 16.0, False)], out, log_dir / "04_overlay.log")
    return out, "fallback"


# --------------------------------------------------------------------------
# 05 -- deliverable (live capture preferred; legend-safe still fallback)
# --------------------------------------------------------------------------

# From video/build/v2/logs/stills_05_07.log (scripts/video/stills_05_07.py),
# which also composited the native-color legend into the safe area of
# fallback_05.png so it survives this push (fixes the review's crop flag).
LIVEMAP_ZOOM_END, LIVEMAP_CX, LIVEMAP_CY = 1.08, 0.265, 0.590


def build_05(
    beat: Beat,
    log_dir: Path,
    *,
    strict_tier: str | None = None,
    strict_asset: Path | None = None,
) -> tuple[Path, str]:
    """LT-05 must be composited regardless of tier: unlike 03/04/06b's
    animated upgrades (which bake their own overlays in), 05's upgrade is a
    raw browser-capture recording with no lower-third of its own."""
    out = CLIPS_V2 / "05.mp4"
    base = CLIPS_V2 / "05_base.mp4"
    if strict_tier == "tanager-still":
        if strict_asset is None:
            raise ValueError("strict beat 05 requires the exact Tanager-derived still")
        image_clip(
            strict_asset,
            beat.render_dur,
            base,
            log_dir / "05.log",
            fit="none",
            zoom_end=LIVEMAP_ZOOM_END,
            cx=LIVEMAP_CX,
            cy=LIVEMAP_CY,
        )
        tier = "tanager-still"
    elif strict_tier is not None:
        raise ValueError(f"unsupported strict tier for beat 05: {strict_tier}")
    elif _try_upgrade("05", beat.render_dur, base, log_dir):
        tier = "upgrade"
    else:
        fallback_png = BUILD_V2 / "fallback_05.png"
        if fallback_png.exists():
            image_clip(
                fallback_png,
                beat.render_dur,
                base,
                log_dir / "05.log",
                fit="none",
                zoom_end=LIVEMAP_ZOOM_END,
                cx=LIVEMAP_CX,
                cy=LIVEMAP_CY,
            )
            tier = "fallback-still"
        else:
            img = SUBMISSION_FIGURES / f"{GF}_hero_mineral_map.png"
            image_clip(
                img,
                beat.render_dur,
                base,
                log_dir / "05.log",
                fit="cover",
                zoom_end=1.08,
                cx=0.38,
                cy=0.60,
            )
            tier = "fallback-emergency"  # legend not guaranteed readable -- see qc.py
    sprite = overlays.lower_third(
        "Goldfield district, Nevada",
        "Dominant alteration mineral · Tanager MTMF",
        BUILD_V2 / "sprites" / "lt05.png",
    )
    composite_overlays(base, [(sprite, 1.0, 6.0, True)], out, log_dir / "05_overlay.log")
    return out, tier


# --------------------------------------------------------------------------
# 06a -- validation
# --------------------------------------------------------------------------

# Matched alteration-zone rectangles (canvas px), one per panel, over
# goldfield_validation_pair.png's dense alunite cluster (visible in both the
# Tanager MTMF panel and the ASTER reference panel). Measured by eye; tune
# against a rendered frame per edit_plan.md's general tuning allowance.
HL06A_RECTS = [
    (144, 583, 192, 130),  # left panel (Tanager alunite MTMF)
    (1008, 544, 192, 130),  # right panel (ASTER reference)
]


def build_06a(beat: Beat, log_dir: Path) -> tuple[Path, str]:
    img = SUBMISSION_FIGURES / "goldfield_validation_pair.png"
    base = CLIPS_V2 / "06a_base.mp4"
    image_clip(img, beat.render_dur, base, log_dir / "06a.log", fit="contain", zoom_end=1.05)
    sprite = overlays.matched_highlight(HL06A_RECTS, BUILD_V2 / "sprites" / "hl06a.png")
    out = CLIPS_V2 / "06a.mp4"
    composite_overlays(base, [(sprite, 6.0, 12.5, False)], out, log_dir / "06a_overlay.log")
    return out, "designed"


# --------------------------------------------------------------------------
# 06b -- EMIT cross-check
# --------------------------------------------------------------------------

CAP06B_TEXT = "EMIT detects the same minerals — alunite r = 0.55 · jarosite r = 0.58"
# Source: EMIT comparison CSV, alunite detection r=0.545 and corrected
# jarosite detection r=0.5838; values are displayed to two decimal places.
# Centered on the frame (not biased toward the EMIT-60m panel): at this
# string length, a right-biased anchor pushed the pill past the safe margin.
CAP06B_CENTER = (WIDTH / 2, 900.0)


def build_06b(
    beat: Beat,
    log_dir: Path,
    *,
    strict_tier: str | None = None,
    strict_asset: Path | None = None,
) -> tuple[Path, str]:
    out = CLIPS_V2 / "06b.mp4"
    if strict_tier == "upgrade":
        if strict_asset is None:
            raise ValueError("strict beat 06b upgrade requires an exact asset")
        normalize_upgrade(strict_asset, beat.render_dur, out, log_dir / "06b_upgrade.log")
        return out, "upgrade"
    if strict_tier not in (None, "fallback"):
        raise ValueError(f"unsupported strict tier for beat 06b: {strict_tier}")
    if strict_tier is None and _try_upgrade("06b", beat.render_dur, out, log_dir):
        return out, "upgrade"
    img = SUBMISSION_FIGURES / f"{GF}_emit_comparison.png"
    base = CLIPS_V2 / "06b_base.mp4"
    seat_high(img, beat.render_dur, base, log_dir / "06b.log", zoom_end=1.04)
    sprite = overlays.caption_strip(
        CAP06B_TEXT, CAP06B_CENTER, BUILD_V2 / "sprites" / "cap06b.png", fontsize=27
    )
    composite_overlays(base, [(sprite, 4.0, beat.vo_dur, False)], out, log_dir / "06b_overlay.log")
    return out, "fallback"


# --------------------------------------------------------------------------
# 07 -- AMD payoff (live capture preferred; legend-safe still fallback)
# --------------------------------------------------------------------------

AMD_ZOOM_END, AMD_CX, AMD_CY = 1.06, 0.413, 0.369  # see LIVEMAP_* provenance note


def build_07(
    beat: Beat,
    log_dir: Path,
    *,
    strict_tier: str | None = None,
    strict_asset: Path | None = None,
) -> tuple[Path, str]:
    """LT-07 must be composited regardless of tier -- same reasoning as
    build_05 (the upgrade here is a raw browser capture, not an animation
    with its own baked-in overlay)."""
    out = CLIPS_V2 / "07.mp4"
    base = CLIPS_V2 / "07_base.mp4"
    if strict_tier == "tanager-still":
        if strict_asset is None:
            raise ValueError("strict beat 07 requires the exact Tanager-derived still")
        image_clip(
            strict_asset,
            beat.render_dur,
            base,
            log_dir / "07.log",
            fit="none",
            zoom_end=AMD_ZOOM_END,
            cx=AMD_CX,
            cy=AMD_CY,
        )
        tier = "tanager-still"
    elif strict_tier is not None:
        raise ValueError(f"unsupported strict tier for beat 07: {strict_tier}")
    elif _try_upgrade("07", beat.render_dur, base, log_dir):
        tier = "upgrade"
    else:
        fallback_png = BUILD_V2 / "fallback_07.png"
        if fallback_png.exists():
            image_clip(
                fallback_png,
                beat.render_dur,
                base,
                log_dir / "07.log",
                fit="none",
                zoom_end=AMD_ZOOM_END,
                cx=AMD_CX,
                cy=AMD_CY,
            )
            tier = "fallback-still"
        else:
            img = SUBMISSION_FIGURES / f"{BH}_amd_agp.png"
            image_clip(
                img,
                beat.render_dur,
                base,
                log_dir / "07.log",
                fit="cover",
                zoom_end=1.06,
                cx=0.45,
                cy=0.55,
            )
            tier = "fallback-emergency"
    sprite = overlays.lower_third(
        "Bingham Canyon, Utah",
        "Acid-generating potential · Tanager MTMF (ordinal)",
        BUILD_V2 / "sprites" / "lt07.png",
    )
    composite_overlays(base, [(sprite, 1.0, 6.0, True)], out, log_dir / "07_overlay.log")
    return out, tier


# --------------------------------------------------------------------------
# 08 -- close / end card
# --------------------------------------------------------------------------

# Ported verbatim from video/build/render_motif.py's wl_to_rgb (approximate
# visible-spectrum colour 400-700 nm, dim infrared ember 700-2500 nm) so the
# end-card bookend uses the identical ramp -- same synthetic element already
# covered by the motif disclosure, not a new one. render_motif.py isn't an
# importable package (it's a standalone script under video/build/, off
# sys.path for scripts/video/), so the small pure function is duplicated
# here rather than reached for across that boundary.
_FAN_VIS_MAX_NM = 700.0
# 426 bands / 376.44-2499.16 nm from the SR files' own dataset attributes
# (matches render_motif.py and data/processed/hard_pairs_dataset/wavelengths.csv).
_FAN_N_BANDS = 426
_FAN_WL = np.linspace(376.44, 2499.16, _FAN_N_BANDS)


def _wl_to_rgb(wl: float) -> np.ndarray:
    if wl < _FAN_VIS_MAX_NM:
        if wl < 440:
            r, g, b = -(wl - 440) / 60, 0.0, 1.0
        elif wl < 490:
            r, g, b = 0.0, (wl - 440) / 50, 1.0
        elif wl < 510:
            r, g, b = 0.0, 1.0, -(wl - 510) / 20
        elif wl < 580:
            r, g, b = (wl - 510) / 70, 1.0, 0.0
        elif wl < 645:
            r, g, b = 1.0, -(wl - 645) / 65, 0.0
        else:
            r, g, b = 1.0, 0.0, 0.0
        if wl < 420:
            f = 0.3 + 0.7 * (wl - 380) / 40
        elif wl > 645:
            f = 0.5 + 0.5 * (700 - wl) / 55
        else:
            f = 1.0
        return np.array([r, g, b]) * max(f, 0.0)
    decay = (2500 - wl) / (2500 - _FAN_VIS_MAX_NM)
    base = 0.18 + 0.42 * decay
    return np.array([base, 0.05 * decay + 0.02, 0.09 * decay + 0.02])


def _spectral_fan_strip(px_width: int, dim: float = 0.4) -> np.ndarray:
    """Thin horizontal bookend strip: the same 426-band barcode as the
    opening motif, dimmed further so it reads as a quiet accent, not a
    second focal point (Appendix B: "faint spectral-fan edge... as a
    bookend"; round-1 critique: fill the card's dead mid-frame gap)."""
    band_rgb = np.array([_wl_to_rgb(w) for w in _FAN_WL])  # (426, 3)
    band_of_px = np.floor(np.linspace(0, _FAN_N_BANDS - 1e-6, px_width)).astype(int)
    strip = band_rgb[band_of_px] * dim
    sep = np.zeros(px_width, bool)
    sep[1:] = band_of_px[1:] != band_of_px[:-1]
    strip[sep] *= 0.5
    return strip


def render_end_card(
    out_png: Path,
    doi_line: str = DOI_LINE_DRAFT,
    repository: str = REPO_SLUG,
) -> Path:
    """Appendix B layout: navy field, centered column, headline/result/links/
    spectral-fan bookend/disclosure. Vertical rhythm redistributed across the
    frame per round-1 critique ("top blocks cluster high, dead gap above the
    disclosure") -- blocks now step down at a roughly even pace instead of
    clustering in the top third."""
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor(BG_HEX)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    cx = WIDTH / 2
    ax.text(
        cx,
        380,
        "The color you can't see",
        color="white",
        ha="center",
        va="center",
        fontproperties=FontProperties(fname=str(FONT_BOLD), size=64),
    )
    ax.text(
        cx,
        475,
        "426 bands · 30 m · Goldfield NV + Bingham Canyon UT",
        color="white",
        alpha=0.90,
        ha="center",
        va="center",
        fontproperties=FontProperties(fname=str(FONT_REGULAR), size=34),
    )
    ax.text(
        cx,
        570,
        f"Code & data: {repository}",
        color="white",
        alpha=0.80,
        ha="center",
        va="center",
        fontproperties=FontProperties(fname=str(FONT_REGULAR), size=28),
    )
    ax.text(
        cx,
        608,
        doi_line,
        color="white",
        alpha=0.80,
        ha="center",
        va="center",
        fontproperties=FontProperties(fname=str(FONT_REGULAR), size=28),
    )

    fan_w, fan_h, fan_top = round(WIDTH * 0.5), 28, 725
    fan = _spectral_fan_strip(fan_w)
    ax.imshow(
        np.tile(fan[None, :, :], (fan_h, 1, 1)),
        extent=(cx - fan_w / 2, cx + fan_w / 2, fan_top + fan_h, fan_top),
    )

    disclosure = (
        "Narration: ElevenLabs. Music: Eleven Music v2, subsequently edited and mixed.\n"
        "Opening motif: procedural graphic, not imagery or scientific data.\n"
        "Credits and media terms: see CREDITS.md in the release bundle."
    )
    ax.text(
        cx,
        925,
        disclosure,
        color="white",
        alpha=0.65,
        ha="center",
        va="center",
        fontproperties=FontProperties(fname=str(FONT_REGULAR), size=20),
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100, facecolor=BG_HEX)
    plt.close(fig)
    return out_png


def build_08(
    beat: Beat,
    log_dir: Path,
    *,
    doi_line: str = DOI_LINE_DRAFT,
    repository: str = REPO_SLUG,
) -> tuple[Path, str]:
    card = render_end_card(BUILD_V2 / "end_card.png", doi_line=doi_line, repository=repository)
    out = CLIPS_V2 / "08.mp4"
    fade_start = beat.render_dur - 0.6
    vf = f"fade=t=out:st={fade_start}:d=0.6"
    run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-i",
            str(card),
            "-t",
            f"{beat.render_dur}",
            "-vf",
            vf,
            "-r",
            str(FPS),
            *ENCODE_ARGS,
            str(out),
        ],
        log_dir / "08.log",
    )
    return out, "designed"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

BUILDERS = {
    "00": build_00,
    "01": build_01,
    "02": build_02,
    "03": build_03,
    "04": build_04,
    "05": build_05,
    "06a": build_06a,
    "06b": build_06b,
    "07": build_07,
    "08": build_08,
}


def build_all_clips(
    edl: list[Beat],
    strict_sources: Mapping[str, tuple[str, Path | None]] | None = None,
    *,
    release_archive_doi: str | None = None,
    release_repository: str | None = None,
) -> dict[str, tuple[Path, str]]:
    CLIPS_V2.mkdir(parents=True, exist_ok=True)
    LOGS_V2.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[Path, str]] = {}
    for beat in edl:
        if strict_sources is None:
            path, tier = BUILDERS[beat.name](beat, LOGS_V2)
        else:
            expected_tier, expected_asset = strict_sources[beat.name]
            if beat.name in {"03", "04", "05", "06b", "07"}:
                path, tier = BUILDERS[beat.name](
                    beat,
                    LOGS_V2,
                    strict_tier=expected_tier,
                    strict_asset=expected_asset,
                )
            else:
                if expected_tier != "designed":
                    raise ValueError(
                        f"strict beat {beat.name} must use tier 'designed', got {expected_tier!r}"
                    )
                if beat.name == "08":
                    if release_archive_doi is None or release_repository is None:
                        raise ValueError(
                            "strict beat 08 requires the frozen DOI and repository URL"
                        )
                    path, tier = build_08(
                        beat,
                        LOGS_V2,
                        doi_line=f"Archive: {release_archive_doi}",
                        repository=release_repository.removeprefix("https://"),
                    )
                else:
                    path, tier = BUILDERS[beat.name](beat, LOGS_V2)
            if tier != expected_tier:
                raise RuntimeError(
                    f"strict beat {beat.name} produced tier {tier!r}; expected {expected_tier!r}"
                )
        results[beat.name] = (path, tier)
        print(f"  {beat.name}: {tier:<20s} -> {path.relative_to(ROOT)} ({beat.render_dur:.3f}s)")
    return results
