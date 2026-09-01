"""Procedural opening motif for the submission video (no AI, no external assets).

Renders `video/build/motif.mp4` (1920x1080, 30 fps, ~7.5 s): 426 spectral bands
dispersed as a barcode. The visible window (400-700 nm) blooms in approximate
true colour; the ~85% of bands in the shortwave infrared (700-2500 nm) glow as a
dim ember gradient -- "the color you can't see". Bands reveal left-to-right, a
visible/SWIR divider fades in, then the title. Pure matplotlib + ffmpeg, fully
controllable and reproducible.

Run: uv run python video/build/render_motif.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402

FPS = 30
DUR_S = 7.5
N_FRAMES = int(FPS * DUR_S)
W, H = 1920, 1080
# Band count + range from the SR files' own dataset attributes (see
# data/processed/hard_pairs_dataset/wavelengths.csv): 426 bands, 376.44-2499.16 nm.
N_BANDS = 426
WL_MIN_NM = 376.44
WL_MAX_NM = 2499.16
WL = np.linspace(WL_MIN_NM, WL_MAX_NM, N_BANDS)  # nm, actual Tanager VSWIR axis
VIS_MAX_NM = 700.0
REVEAL_S = 4.2  # left-to-right fill completes here
DIM_START_S = 4.6  # field dims so the title reads
TITLE_START_S = 5.0


def wl_to_rgb(wl: float) -> np.ndarray:
    """Approximate visible-spectrum colour (Bruton) for 400-700 nm; dim infrared
    ember for the SWIR (700-2500 nm), decaying with wavelength."""
    if wl < VIS_MAX_NM:
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
    decay = (2500 - wl) / (2500 - VIS_MAX_NM)  # 1 at 700 nm -> 0 at 2500 nm
    base = 0.18 + 0.42 * decay  # ember red, 0.60 -> 0.18
    return np.array([base, 0.05 * decay + 0.02, 0.09 * decay + 0.02])


def build_strip() -> np.ndarray:
    """Full 1920-wide colour strip with thin dark separators at band edges."""
    band_rgb = np.array([wl_to_rgb(w) for w in WL])  # (426, 3)
    band_of_px = np.floor(np.linspace(0, N_BANDS - 1e-6, W)).astype(int)
    strip = band_rgb[band_of_px]  # (1920, 3)
    sep = np.zeros(W, bool)
    sep[1:] = band_of_px[1:] != band_of_px[:-1]
    strip[sep] *= 0.25  # dark gaps -> "426 discrete bands" texture
    return strip


def main() -> None:
    plt.rcParams["animation.ffmpeg_path"] = shutil.which("ffmpeg") or "ffmpeg"
    strip = build_strip()
    full = np.tile(strip[None, :, :], (H, 1, 1))  # (1080, 1920, 3)
    vis_x = int(W * (VIS_MAX_NM - WL_MIN_NM) / (WL_MAX_NM - WL_MIN_NM))

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    im = ax.imshow(np.zeros((H, W, 3)), extent=(0, W, 0, H), origin="lower", aspect="auto")

    divider = ax.plot([vis_x, vis_x], [0, H], color="white", lw=1.2, alpha=0.0)[0]
    lbl_vis = ax.text(
        vis_x / 2, H * 0.92, "VISIBLE", color="white", ha="center", fontsize=20, alpha=0.0
    )
    lbl_swir = ax.text(
        (vis_x + W) / 2,
        H * 0.92,
        "SHORTWAVE INFRARED",
        color="white",
        ha="center",
        fontsize=20,
        alpha=0.0,
    )
    lbl_l = ax.text(24, 24, "376 nm", color="white", ha="left", fontsize=16, alpha=0.0)
    lbl_r = ax.text(W - 24, 24, "2499 nm", color="white", ha="right", fontsize=16, alpha=0.0)
    title = ax.text(
        W / 2,
        H * 0.56,
        "The color you can't see",
        color="white",
        ha="center",
        va="center",
        fontsize=66,
        fontweight="bold",
        alpha=0.0,
    )
    sub = ax.text(
        W / 2,
        H * 0.45,
        "Planet Tanager  ·  426 contiguous bands  ·  376–2499 nm",
        color="white",
        ha="center",
        va="center",
        fontsize=26,
        alpha=0.0,
    )

    def update(i: int):
        t = i / FPS
        img = full.copy()
        rev = float(np.clip(t / REVEAL_S, 0, 1))
        img[:, int(W * rev) :, :] = 0.0
        if t < 0.35:  # opening white flash
            img = np.clip(img + (0.35 - t) / 0.35 * 0.8, 0, 1)
        if t > DIM_START_S:  # dim the field for the title
            img *= 1 - 0.55 * float(np.clip((t - DIM_START_S) / 1.0, 0, 1))
        im.set_data(img)

        region_a = float(np.clip((rev - 0.3) / 0.6, 0, 1)) * 0.7
        if t > TITLE_START_S:
            region_a *= 1 - 0.5 * float(np.clip((t - TITLE_START_S) / 1.6, 0, 1))
        for artist in (divider, lbl_vis, lbl_swir, lbl_l, lbl_r):
            artist.set_alpha(region_a)

        title_a = float(np.clip((t - TITLE_START_S) / 1.6, 0, 1))
        title.set_alpha(title_a)
        sub.set_alpha(title_a)
        return im, divider, lbl_vis, lbl_swir, lbl_l, lbl_r, title, sub

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=False)
    out = Path("video/build/motif.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=FFMpegWriter(fps=FPS, bitrate=9000, codec="libx264"))
    plt.close(fig)
    print(f"wrote {out} ({N_FRAMES} frames, {DUR_S}s)")


if __name__ == "__main__":
    main()
