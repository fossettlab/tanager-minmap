"""Animated re-render of beat 03 ("the data") for the v2 submission video.

Re-derives the exact same library-vs-scene spectra shown in
`submission/figures/goldfield_spectra.png` (`figures.py:spectra_story`, invoked
by `scripts/build_submission.py`) -- same minerals, same normalisation, same
absorption markers -- then animates it: a data-space xlim push into the SWIR
diagnostic region, three in-plot callouts on the Al-OH/jarosite/gypsum markers
timed to the narration, and a thicken+brighten/dim-the-rest emphasis moment on
the alunite pair when the VO says the library and scene spectra "line up".
Durations and cue times are from docs/edit_plan.md (beat 03 section + the
CALL-03a/b/c rows of the overlays table).

The reflectance cube and speclib load in ~10 s (the MTMF maps are read from
the already-computed `data/intermediate/maps/*_mf_*.tif` / `*_infeas_*.tif`
GeoTIFFs rather than re-run), so this re-renders from source data rather than
falling back to animating the static PNG.

Run: uv run python scripts/video/anim_03_spectra.py
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import rioxarray
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
from tanager_spec.io import load_tanager_sr_hdf5  # noqa: E402
from tanager_spec.mask import mask_absorption_bands  # noqa: E402

from tanager_rocks.config import SITES, TANAGER_SR_ASSET  # noqa: E402
from tanager_rocks.figures import _normalize, representative_spectra  # noqa: E402
from tanager_rocks.speclib import load_library, select_endmembers  # noqa: E402
from tanager_rocks.viz import MINERAL_COLORS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
MAPS_DIR = ROOT / "data" / "intermediate" / "maps"
OUT = ROOT / "video" / "build" / "v2" / "upgrades" / "03.mp4"

FPS = 30
DUR_S = 25.731  # beat 03 VO dur, docs/edit_plan.md EDL table (hard cut both sides, no +D pad)
N_FRAMES = round(DUR_S * FPS)
W, H = 1920, 1080
NAVY = "#0a0e1a"
# figures.py:spectra_story's own figsize is 9.0x7.0in; round-1 critique found
# that pillarboxed small in the 1920x1080 frame, so this is scaled up ~10%
# (aspect ratio unchanged) for the video.
CHART_W_IN, CHART_H_IN = 9.9, 7.7

# Same story constants as scripts/build_submission.py (the script that produced
# the committed goldfield_spectra.png) -- reused verbatim, not re-derived.
STORY_MINERALS = ["alunite", "kaolinite", "jarosite", "muscovite", "hematite"]
ABSORPTIONS = {
    "Al-OH 2200": 2200.0,
    "jarosite 2265": 2265.0,
    "gypsum/carb 2340": 2340.0,
    "Fe³⁺ ~900": 900.0,
}
STEP = 1.25  # per-mineral vertical offset, matches spectra_story
# spectra_story's own headroom is `+0.55` above the mineral stack (fits the
# rotated absorption labels only); the animated version reserves more so the
# staggered CALL-03a/b/c pills have room without spilling past the axes box.
CALLOUT_HEADROOM = 2.15
CALLOUT_Y0, CALLOUT_DY = 0.75, 0.5

# Beat-relative timings (seconds from beat start), verbatim from
# docs/edit_plan.md: the "03" per-clip section (xlim push, emphasis) and the
# CALL-03a/b/c rows of the consolidated overlays table (in/out).
XLIM_START_T, XLIM_END_T = 4.0, 11.0
XLIM_END = (1900.0, 2400.0)
FADE_S = 0.3  # matches the master fade-up/fade-out convention
CALLOUTS = [
    # (label, wavelength_nm, in_t, out_t, stagger_rank)
    ("Al–OH · 2200 nm", 2200.0, 10.0, 17.0, 0),
    ("jarosite · 2265 nm", 2265.0, 12.0, 17.0, 1),
    ("gypsum · 2340 nm", 2340.0, 14.0, 17.0, 2),
]
EMPHASIS_MINERAL = "alunite"
# storyboard.md 03: '"...line up" (~+0:18)'. Round-1 critique: the original
# thicken-only pulse read too subtle as an "emphasis moment" -- widened to a
# proper 3 s window (0.5 s in, 2 s hold spanning the +18 s VO cue, 0.5 s out)
# and paired with a brighten-alunite / dim-everyone-else contrast.
EMPHASIS_IN_T, EMPHASIS_OUT_T = 17.0, 20.0
EMPHASIS_FADE_S = 0.5
EMPHASIS_LW_BOOST_SOLID, EMPHASIS_LW_BOOST_DASHED = 1.8, 1.6
EMPHASIS_DIM_FRAC = 0.3  # non-emphasised minerals dim to 70% alpha at full emphasis
EMPHASIS_GLOW_COLOR = "#ffd54a"


def _ease(x: float) -> float:
    """Smoothstep ease-in/out, clamped to [0, 1]."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _fade(t: float, t_in: float, t_out: float, fade: float = FADE_S) -> float:
    """Alpha envelope: fades in over `fade` s at t_in, out over `fade` s at t_out."""
    if t < t_in or t > t_out:
        return 0.0
    if t < t_in + fade:
        return (t - t_in) / fade
    if t > t_out - fade:
        return max(0.0, (t_out - t) / fade)
    return 1.0


def _load_data() -> tuple[np.ndarray, dict, dict]:
    """Real endmember + top-detected-pixel spectra, identical inputs to spectra_story."""
    site = SITES["goldfield"]
    scene_id = site.scene_ids[0]
    cube_raw, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    cube = mask_absorption_bands(cube_raw, wl)

    ds = xr.Dataset(
        {
            **{
                f"{m}_mf": rioxarray.open_rasterio(
                    MAPS_DIR / f"{site.site_id}_{scene_id}_mf_{m}.tif", masked=True
                ).squeeze("band", drop=True)
                for m in STORY_MINERALS
            },
            **{
                f"{m}_infeas": rioxarray.open_rasterio(
                    MAPS_DIR / f"{site.site_id}_{scene_id}_infeas_{m}.tif", masked=True
                ).squeeze("band", drop=True)
                for m in STORY_MINERALS
            },
        }
    )
    pixel_spectra = representative_spectra(cube, ds, STORY_MINERALS)
    endmembers = select_endmembers(load_library(SPECLIB_DIR, wl))
    return wl, endmembers, pixel_spectra


def _build_static(fig, wl: np.ndarray, endmembers: dict, pixel_spectra: dict):
    """Draw the unchanging chart content (same layout as figures.py:spectra_story)
    into a white inset axes, letterboxed on the navy 1920x1080 canvas."""
    left = (19.2 - CHART_W_IN) / 2 / 19.2
    bottom = (10.8 - CHART_H_IN) / 2 / 10.8
    ax = fig.add_axes((left, bottom, CHART_W_IN / 19.2, CHART_H_IN / 10.8), facecolor="white")

    lines: dict[str, tuple] = {}  # mineral -> (solid_line, dashed_line_or_None, label_text)
    for i, mineral in enumerate(STORY_MINERALS):
        offset = i * STEP
        color = MINERAL_COLORS.get(mineral, f"C{i}")
        solid = None
        if mineral in endmembers:
            (solid,) = ax.plot(
                wl, _normalize(endmembers[mineral].reflectance) + offset, color=color, lw=1.4
            )
        dashed = None
        if mineral in pixel_spectra:
            (dashed,) = ax.plot(
                wl,
                _normalize(pixel_spectra[mineral]) + offset,
                color=color,
                lw=1.2,
                ls="--",
                alpha=0.9,
            )
        label = ax.text(
            wl[-1], offset + 0.55, mineral, color=color, ha="right", va="center", fontsize=10
        )
        lines[mineral] = (solid, dashed, label, offset)

    top = len(STORY_MINERALS) * STEP
    for absorb_label, nm in ABSORPTIONS.items():
        ax.axvline(nm, color="0.5", lw=0.8, ls=":")
        ax.text(
            nm,
            top + 0.05,
            absorb_label,
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
            c="0.4",
        )

    ax.set_ylim(-0.15, top + CALLOUT_HEADROOM)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("normalised reflectance (offset per mineral)")
    ax.set_yticks([])
    # Bold + explicit size/colour: round-1 critique found the default title
    # faint at the enlarged chart size.
    ax.set_title(
        "Diagnostic spectra: library vs. Tanager scene",
        fontsize=16,
        fontweight="bold",
        color="black",
    )
    handles = [
        plt.Line2D([], [], color="0.2", lw=1.4, label="USGS splib07a library"),
        plt.Line2D([], [], color="0.2", lw=1.2, ls="--", label="Tanager scene (top pixels)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    return ax, lines, top


def _build_callouts(ax, top: float):
    """Staggered navy-pill callouts (Appendix A style), one per CALLOUTS entry."""
    anns = []
    for label, nm, t_in, t_out, rank in CALLOUTS:
        y_text = top + CALLOUT_Y0 + CALLOUT_DY * rank
        ann = ax.annotate(
            label,
            xy=(nm, top + 0.05),
            xytext=(nm, y_text),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.35", "fc": NAVY, "ec": "none"},
            arrowprops={
                "arrowstyle": "-|>",
                "color": "white",
                "lw": 1.6,
                "path_effects": [withStroke(linewidth=3, foreground=NAVY)],
            },
            annotation_clip=False,
        )
        ann.set_alpha(0.0)
        anns.append((ann, t_in, t_out))
    return anns


def _set_annotation_alpha(ann, alpha: float) -> None:
    """Fade an Annotation's text, pill background, and arrow together.

    Annotation.set_alpha only reliably drives the text; the bbox patch and
    arrow_patch are separate artists that need their own alpha set (observed
    empirically: without this, the arrow/pill stayed visible at alpha=0).
    """
    ann.set_alpha(alpha)
    bbox_patch = ann.get_bbox_patch()
    if bbox_patch is not None:
        bbox_patch.set_alpha(alpha)
    if ann.arrow_patch is not None:
        ann.arrow_patch.set_alpha(alpha)


def main() -> None:
    plt.rcParams["animation.ffmpeg_path"] = shutil.which("ffmpeg") or "ffmpeg"
    wl, endmembers, pixel_spectra = _load_data()
    assert wl.min() <= XLIM_END[0] and wl.max() >= XLIM_END[1], (
        f"XLIM_END {XLIM_END} outside data range [{wl.min():.1f}, {wl.max():.1f}]"
    )
    full_xlim = (float(wl.min()), float(wl.max()))

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(NAVY)
    ax, lines, top = _build_static(fig, wl, endmembers, pixel_spectra)
    ax.set_xlim(*full_xlim)
    callout_anns = _build_callouts(ax, top)

    em_solid, em_dashed, _, em_offset = lines[EMPHASIS_MINERAL]
    base_lw_solid = em_solid.get_linewidth() if em_solid else 1.4
    base_lw_dashed = em_dashed.get_linewidth() if em_dashed else 1.2

    # "Brighten": a warm glow duplicate drawn just behind the real alunite
    # curves, faded in/out with the emphasis envelope (no colour/data change to
    # the real lines -- purely a highlight halo).
    em_glow_solid = em_glow_dashed = None
    if em_solid:
        (em_glow_solid,) = ax.plot(
            em_solid.get_xdata(),
            em_solid.get_ydata(),
            color=EMPHASIS_GLOW_COLOR,
            lw=4.5,
            alpha=0.0,
            zorder=em_solid.get_zorder() - 0.1,
            solid_capstyle="round",
        )
    if em_dashed:
        (em_glow_dashed,) = ax.plot(
            em_dashed.get_xdata(),
            em_dashed.get_ydata(),
            color=EMPHASIS_GLOW_COLOR,
            lw=3.8,
            alpha=0.0,
            zorder=em_dashed.get_zorder() - 0.1,
            solid_capstyle="round",
        )

    # "Dim the others": remember each non-emphasised mineral's own base alpha
    # (the dashed "scene" lines already carry alpha=0.9) so dimming multiplies
    # it rather than clobbering it.
    dim_targets = []
    for mineral in STORY_MINERALS:
        if mineral == EMPHASIS_MINERAL:
            continue
        solid, dashed, label, _ = lines[mineral]
        for artist in (solid, dashed, label):
            if artist is not None:
                dim_targets.append((artist, artist.get_alpha() or 1.0))

    def update(i: int):
        t = i / FPS

        # 1. Data-space xlim push into the SWIR diagnostic region.
        frac = _ease((t - XLIM_START_T) / (XLIM_END_T - XLIM_START_T))
        lo = full_xlim[0] + frac * (XLIM_END[0] - full_xlim[0])
        hi = full_xlim[1] + frac * (XLIM_END[1] - full_xlim[1])
        ax.set_xlim(lo, hi)

        # Keep each mineral's name label pinned near the current right edge
        # (its original position, wl[-1], scrolls out of view once we zoom).
        margin = 0.02 * (hi - lo)
        for mineral in STORY_MINERALS:
            _, _, label, offset = lines[mineral]
            label.set_position((hi - margin, offset + 0.55))

        # 2. Three staged callouts, fading in/out on the narration cues.
        for ann, t_in, t_out in callout_anns:
            _set_annotation_alpha(ann, _fade(t, t_in, t_out))

        # 3. Alunite emphasis on "...line up": thicken + brighten (glow) the
        # emphasised pair while dimming every other mineral ~30%.
        em = _fade(t, EMPHASIS_IN_T, EMPHASIS_OUT_T, fade=EMPHASIS_FADE_S)
        if em_solid:
            em_solid.set_linewidth(base_lw_solid + EMPHASIS_LW_BOOST_SOLID * em)
        if em_dashed:
            em_dashed.set_linewidth(base_lw_dashed + EMPHASIS_LW_BOOST_DASHED * em)
        if em_glow_solid:
            em_glow_solid.set_alpha(0.6 * em)
        if em_glow_dashed:
            em_glow_dashed.set_alpha(0.5 * em)
        dim = 1.0 - EMPHASIS_DIM_FRAC * em
        for artist, base_alpha in dim_targets:
            artist.set_alpha(base_alpha * dim)

        artists = (
            [ax, em_solid, em_dashed, em_glow_solid, em_glow_dashed]
            + [a for a, _, _ in callout_anns]
            + [a for a, _ in dim_targets]
        )
        return [a for a in artists if a is not None]

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        extra_args=["-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"],
    )
    anim.save(OUT, writer=writer)
    plt.close(fig)
    logger.info("wrote %s (%d frames, %.3fs)", OUT, N_FRAMES, N_FRAMES / FPS)


if __name__ == "__main__":
    main()
