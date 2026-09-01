"""Animated re-render of beat 04 ("central result" / band ablation) for the v2
submission video.

Re-derives the exact inputs behind `submission/figures/bingham_..._band_ablation.png`
(`pipeline.py:run_ablate`, `viz.py:band_ablation_panel`) -- same endmember pair
(alunite/kaolinite), same Sentinel-2 SRF degradation, same spectral-angle
separability numbers -- but stages the two-panel static figure as one
continuous before->after reveal: open on the full-VSWIR Tanager curves alone,
fade in the 13 degraded Sentinel-2 band markers, then push the x-axis into the
Al-OH doublet (2000-2350 nm) where the native "(50% loss)" annotation and the
S2 band-FWHM shading appear. Durations and cue times are from
docs/edit_plan.md (beat 04 section + the CALL-04 row of the overlays table).

Only `wl` (the wavelength grid) is read from the Bingham scene -- no cube data
or MTMF is needed for this beat, so this is cheap to re-run.

Run: uv run python scripts/video/anim_04_ablation.py
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
from tanager_spec.io import load_tanager_sr_hdf5  # noqa: E402
from tanager_spec.srf import load_s2_srf  # noqa: E402

from tanager_rocks.config import SITES, TANAGER_SR_ASSET  # noqa: E402
from tanager_rocks.degrade import degrade_endmembers, separability, srf_band_stats  # noqa: E402
from tanager_rocks.pipeline import ABLATION_HEADLINE  # noqa: E402
from tanager_rocks.speclib import load_library, select_endmembers  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
OUT = ROOT / "video" / "build" / "v2" / "upgrades" / "04.mp4"

FPS = 30
DUR_S = 22.047  # beat 04 VO dur, docs/edit_plan.md EDL table (hard cut both sides, no +D pad)
N_FRAMES = round(DUR_S * FPS)
NAVY = "#0a0e1a"
CHART_W_IN, CHART_H_IN = 11.5, 7.0

# band_ablation_panel's own local colour mapping (viz.py), not the global
# MINERAL_COLORS palette -- reused verbatim so this matches the committed
# figure rather than the story-page palette.
COLORS = {ABLATION_HEADLINE[0]: "#1b9e77", ABLATION_HEADLINE[1]: "#d95f02"}
AL_OH_LO, AL_OH_HI = 2000.0, 2350.0  # same zoom window as viz.py's right panel

FADE_S = 0.3
S2_FADE_T = 3.0  # storyboard/edit_plan 04: "...Sentinel-2's broad bands" (~+0:03)
PUSH_START_T, PUSH_END_T = 9.0, 13.0  # edit_plan 04: xlim push "...wash out" (~+0:09)
LOSSBOX_FADE_T = PUSH_START_T  # native annotation + FWHM shading arrive with the push
# CALL-04 in/out, beat-relative seconds, verbatim from docs/edit_plan.md overlays table.
CALL04_IN, CALL04_OUT = 11.0, 16.0


def _ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _fade(t: float, t_in: float, t_out: float, fade: float = FADE_S) -> float:
    if t < t_in or t > t_out:
        return 0.0
    if t < t_in + fade:
        return (t - t_in) / fade
    if t > t_out - fade:
        return max(0.0, (t_out - t) / fade)
    return 1.0


def _ramp(t: float, t0: float, dur: float = 1.0) -> float:
    """0->1 ramp starting at t0 over `dur` seconds, held at 1 after."""
    return _ease((t - t0) / dur)


def _set_annotation_alpha(ann, alpha: float) -> None:
    """Fade an Annotation's text + pill background together.

    Annotation.set_alpha only reliably drives the text; the bbox patch is a
    separate artist that stays fully opaque unless its own alpha is set too
    (observed empirically: the native loss-box's grey-bordered white pill
    stayed visible at alpha=0 without this).
    """
    ann.set_alpha(alpha)
    bbox_patch = ann.get_bbox_patch()
    if bbox_patch is not None:
        bbox_patch.set_alpha(alpha)


def _load_data():
    site = SITES["bingham"]
    scene_id = site.scene_ids[0]
    _, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    endmembers = select_endmembers(load_library(SPECLIB_DIR, wl))
    srf = load_s2_srf()
    centers, fwhm = srf_band_stats(srf)
    sep = separability(endmembers, wl, srf, [ABLATION_HEADLINE])
    full_deg, s2_deg = (np.degrees(x) for x in sep[ABLATION_HEADLINE])
    degraded = degrade_endmembers(endmembers, wl, srf)
    return site, wl, endmembers, degraded, centers, fwhm, full_deg, s2_deg


def main() -> None:
    plt.rcParams["animation.ffmpeg_path"] = shutil.which("ffmpeg") or "ffmpeg"
    site, wl, endmembers, degraded, s2_centers, s2_fwhm, full_deg, s2_deg = _load_data()
    wl = np.asarray(wl, float)
    full_xlim = (float(wl.min()), float(wl.max()))
    m0, m1 = ABLATION_HEADLINE

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(NAVY)
    left = (19.2 - CHART_W_IN) / 2 / 19.2
    bottom = (10.8 - CHART_H_IN) / 2 / 10.8
    ax = fig.add_axes((left, bottom, CHART_W_IN / 19.2, CHART_H_IN / 10.8), facecolor="white")

    tanager_lines = {}
    s2_lines = {}
    for m in ABLATION_HEADLINE:
        (tanager_lines[m],) = ax.plot(
            wl, endmembers[m].reflectance, color=COLORS[m], lw=1.6, label=f"{m} (Tanager)"
        )
        (s2_lines[m],) = ax.plot(
            s2_centers,
            degraded[m],
            "o--",
            color=COLORS[m],
            ms=6,
            lw=1.2,
            label=f"{m} (Sentinel-2)",
        )
        s2_lines[m].set_alpha(0.0)

    # S2 band-FWHM shading + centre lines within the Al-OH window (mirrors
    # viz.py's right-hand panel), faded in with the zoom.
    in_win = (s2_centers >= AL_OH_LO) & (s2_centers <= AL_OH_HI)
    fwhm_artists = []
    for c, w in zip(s2_centers[in_win], s2_fwhm[in_win], strict=False):
        span = ax.axvspan(c - w / 2, c + w / 2, color="0.6", alpha=0.0)
        vline = ax.axvline(c, color="0.4", lw=0.8, ls=":", alpha=0.0)
        fwhm_artists.append((span, vline, c, w))

    ax.set_xlim(*full_xlim)
    all_refl = np.concatenate(
        [endmembers[m].reflectance for m in ABLATION_HEADLINE]
        + [degraded[m] for m in ABLATION_HEADLINE]
    )
    lo_y, hi_y = np.nanmin(all_refl), np.nanmax(all_refl)
    pad = 0.08 * (hi_y - lo_y)
    ax.set_ylim(lo_y - pad, hi_y + pad)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.set_title(f"{site.name}: Tanager vs Sentinel-2 — Al-OH doublet")

    # Two legends, not one: the Tanager-curve legend is present from frame 0;
    # the Sentinel-2 legend fades in with its lines. A single shared legend
    # doesn't work here -- Legend.set_alpha does not reliably cascade to its
    # child text artists (observed empirically: text stayed opaque while the
    # line swatch respected alpha), so the two series need independent legend
    # objects with their own per-artist alpha control.
    legend_tanager = ax.legend(
        [tanager_lines[m0], tanager_lines[m1]],
        [f"{m0} (Tanager)", f"{m1} (Tanager)"],
        fontsize=9,
        frameon=False,
        loc="upper left",
    )
    ax.add_artist(legend_tanager)  # otherwise the next ax.legend() call replaces it
    legend_s2 = ax.legend(
        [s2_lines[m0], s2_lines[m1]],
        [f"{m0} (Sentinel-2)", f"{m1} (Sentinel-2)"],
        fontsize=9,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.82),
    )

    # Native loss-callout, verbatim text/format from viz.py:band_ablation_panel.
    loss_pct = 100 * (1 - s2_deg / full_deg)
    lossbox = ax.annotate(
        f"{m0}–{m1} spectral angle\n"
        f"Tanager {full_deg:.1f}°  →  S2 {s2_deg:.1f}°  ({loss_pct:.0f}% loss)",
        xy=(0.5, 0.97),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.5"},
        annotation_clip=False,
    )
    _set_annotation_alpha(lossbox, 0.0)
    # CALL-04: an arrow pointing at the native box's bottom edge, plus a soft
    # glow behind it (storyboard.md's "arrow or one soft pulse" -- both, since
    # the acceptance check names the arrow specifically). No new text/number.
    glow = FancyBboxPatch(
        (0.30, 0.865),
        0.40,
        0.115,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02",
        fc="#ffd54a",
        ec="none",
        alpha=0.0,
        zorder=lossbox.zorder - 1,
    )
    ax.add_patch(glow)
    call04_arrow = ax.annotate(
        "",
        xy=(0.5, 0.865),
        xycoords="axes fraction",
        xytext=(0.5, 0.68),
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "white",
            "lw": 2.4,
            "path_effects": [withStroke(linewidth=4.5, foreground=NAVY)],
        },
        annotation_clip=False,
    )
    call04_arrow.arrow_patch.set_alpha(0.0)

    def update(i: int):
        t = i / FPS

        s2_alpha = 0.85 * _ramp(t, S2_FADE_T)
        for m in ABLATION_HEADLINE:
            s2_lines[m].set_alpha(s2_alpha)
        for txt in legend_s2.get_texts():
            txt.set_alpha(s2_alpha)
        for handle in legend_s2.legend_handles:
            handle.set_alpha(s2_alpha)

        frac = _ease((t - PUSH_START_T) / (PUSH_END_T - PUSH_START_T))
        lo = full_xlim[0] + frac * (AL_OH_LO - full_xlim[0])
        hi = full_xlim[1] + frac * (AL_OH_HI - full_xlim[1])
        ax.set_xlim(lo, hi)

        fwhm_alpha = _ramp(t, LOSSBOX_FADE_T)
        for span, vline, _, _ in fwhm_artists:
            span.set_alpha(0.18 * fwhm_alpha)
            vline.set_alpha(0.8 * fwhm_alpha)
        _set_annotation_alpha(lossbox, fwhm_alpha)

        call04_alpha = _fade(t, CALL04_IN, CALL04_OUT)
        glow.set_alpha(0.55 * call04_alpha)
        call04_arrow.arrow_patch.set_alpha(call04_alpha)

        return (
            [ax, lossbox, glow, call04_arrow, legend_tanager, legend_s2]
            + list(tanager_lines.values())
            + list(s2_lines.values())
            + [a for pair in fwhm_artists for a in pair[:2]]
        )

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
