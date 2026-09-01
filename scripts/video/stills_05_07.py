"""Reframed fallback stills for beats 05 (Goldfield mineral map) and 07 (Bingham AMD map).

Pipeline v2 (docs/edit_plan.md ##05, ##07): the rough cut's `clip_photo()` push-in
crops symmetrically around a fixed centre, which truncates both source figures'
legends -- both are anchored outside the axis box (`bbox_to_anchor=(1.0, 0.5)` in
viz.py's `mineral_map`/`amd_map`). This strips the legend and title from the
source PNG, keeping only the axis-box data (map + scale bar), and composites it
with a native-rendered legend (same colors/labels as viz.py, white text) onto a
fresh 1920x1080 navy canvas. The legend is inset from the frame edges by enough
margin to survive the push-in zoom the edit plan specifies (zoom_end 1.08 for 05,
1.06 for 07) -- verified below by cropping to the tightest (max-zoom) window and
saving it for inspection.

Run under the repo venv: uv run python scripts/video/stills_05_07.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from tanager_rocks.hazard import AGP_LABELS
from tanager_rocks.viz import AGP_TIER_COLORS, MINERAL_COLORS

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "submission" / "figures"
OUT = ROOT / "video" / "build" / "v2"
GF_HERO = FIG / "goldfield_20240925_185504_87_4001_hero_mineral_map.png"
BH_AGP = FIG / "bingham_20250911_191523_58_4001_amd_agp.png"

NAVY_HEX = "#0a0e1a"
NAVY_RGB = (10, 14, 26)
CANVAS_W, CANVAS_H = 1920, 1080
BORDER_TRIM = 4  # px trimmed inside the detected axis border (~3px stroke) to drop it fully


def _axis_box(
    rgb: np.ndarray, black_max: int = 60, run_frac: float = 0.5
) -> tuple[int, int, int, int]:
    """Locate the plot axes' black border (top, bottom, left, right), in pixels.

    Both source figures (viz.py `mineral_map`/`amd_map`) draw one square axes
    with default spines; the border is the only content forming a long
    contiguous near-black run in a row or column, so a density scan finds it
    without hardcoding a figure-specific offset.
    """
    black = rgb.max(axis=2) < black_max
    h, w = black.shape
    cols = np.where(black.sum(axis=0) > run_frac * h)[0]
    rows = np.where(black.sum(axis=1) > run_frac * w)[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def _legend_card(
    items: list[tuple[str, tuple[float, float, float] | str]], fontsize: float
) -> Image.Image:
    """Render a navy-background legend (native colors/labels, white text)."""
    handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in items]
    fig = plt.figure(figsize=(6, 6), dpi=200, facecolor=NAVY_HEX)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=fontsize,
        labelcolor="white",
        handlelength=1.4,
        handleheight=1.4,
        borderaxespad=0,
    )
    tmp = OUT / "_legend_tmp.png"
    fig.savefig(tmp, facecolor=NAVY_HEX, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    card = Image.open(tmp).convert("RGB")
    tmp.unlink()
    return card


def _compose(
    map_crop: Image.Image,
    legend: Image.Image,
    cx_box: float,
    cy_box: float,
    *,
    legend_right_margin_px: int,
    legend_max_h_frac: float,
) -> tuple[Image.Image, float, float]:
    """Paste map (fills canvas height) + legend (right, inset) onto the navy canvas.

    Returns the canvas and the point-of-interest re-expressed as (cx, cy)
    fractions of the *canvas* -- what the edit plan's zoompan notation expects.
    """
    scale = CANVAS_H / map_crop.height
    map_w = round(map_crop.width * scale)
    map_scaled = map_crop.resize((map_w, CANVAS_H), Image.LANCZOS)

    room = CANVAS_W - map_w
    lscale = min(legend_max_h_frac * CANVAS_H / legend.height, 0.9 * room / legend.width)
    legend_scaled = legend.resize(
        (round(legend.width * lscale), round(legend.height * lscale)), Image.LANCZOS
    )
    lx = CANVAS_W - legend_right_margin_px - legend_scaled.width
    ly = (CANVAS_H - legend_scaled.height) // 2

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), NAVY_RGB)
    canvas.paste(map_scaled, (0, 0))
    canvas.paste(legend_scaled, (lx, ly))

    cx = cx_box * map_w / CANVAS_W  # map fills x in [0, map_w); canvas is x in [0, CANVAS_W)
    cy = cy_box  # map fills the full canvas height, so the box fraction carries over as-is
    return canvas, cx, cy


def _max_zoom_window(cx: float, cy: float, zoom_end: float) -> tuple[int, int, int, int]:
    """The tightest crop `clip_photo`-style zoompan reaches, centred at (cx, cy).

    Mirrors the clamped centre-zoom crop math (edit_plan.md ## Per-clip build
    specs, Notation): window shrinks from the full frame (zoom=1.0) to
    `frame/zoom_end`, centred at (cx, cy) but clamped so it never runs off the
    source. Content inside this window is visible for the clip's entire push-in.
    """
    win_w, win_h = CANVAS_W / zoom_end, CANVAS_H / zoom_end
    x0 = min(max(cx * CANVAS_W - win_w / 2, 0), CANVAS_W - win_w)
    y0 = min(max(cy * CANVAS_H - win_h / 2, 0), CANVAS_H - win_h)
    return round(x0), round(y0), round(x0 + win_w), round(y0 + win_h)


def build_goldfield() -> None:
    rgb = np.array(Image.open(GF_HERO).convert("RGB"))
    top, bottom, left, right = _axis_box(rgb)
    map_crop = (
        Image.open(GF_HERO)
        .convert("RGB")
        .crop((left + BORDER_TRIM, top + BORDER_TRIM, right - BORDER_TRIM, bottom - BORDER_TRIM))
    )
    items = [(m, c) for m, c in MINERAL_COLORS.items()] + [("no detection", (0.92, 0.92, 0.92))]
    legend = _legend_card(items, fontsize=30)

    # edit_plan.md #05 fallback start value: cx~0.38, cy~0.60 of the ORIGINAL
    # (legend-inclusive) image -- the dense alunite/kaolinite/dickite cluster
    # left-of-centre. Re-expressed as a fraction of the axis box alone (the
    # frame this composite actually re-centres against).
    px, py = 0.38 * rgb.shape[1], 0.60 * rgb.shape[0]
    cx_box = (px - left) / (right - left)
    cy_box = (py - top) / (bottom - top)

    canvas, cx, cy = _compose(
        map_crop, legend, cx_box, cy_box, legend_right_margin_px=90, legend_max_h_frac=0.62
    )
    canvas.save(OUT / "fallback_05.png")
    zoom_end = 1.08
    canvas.crop(_max_zoom_window(cx, cy, zoom_end)).save(OUT / "verify_05_maxzoom.png")
    print(f"05: cx={cx:.3f} cy={cy:.3f} zoom_end={zoom_end}")


def build_bingham() -> None:
    rgb = np.array(Image.open(BH_AGP).convert("RGB"))
    top, bottom, left, right = _axis_box(rgb)
    map_crop = (
        Image.open(BH_AGP)
        .convert("RGB")
        .crop((left + BORDER_TRIM, top + BORDER_TRIM, right - BORDER_TRIM, bottom - BORDER_TRIM))
    )
    items = [(AGP_LABELS[c], AGP_TIER_COLORS[c]) for c in sorted(AGP_TIER_COLORS)]
    legend = _legend_card(items, fontsize=30)

    # edit_plan.md #07 fallback gives no explicit cx/cy ("centre to the
    # pit/tailings"). The storyboard ties the red high-AGP (jarosite) zones to
    # ground near the tailings, so use the largest contiguous "high" tier
    # blob within the axis box as a code-grounded stand-in -- a plain
    # pixel-weighted centroid over ALL "high" pixels lands between two
    # disjoint hotspots (in empty ground), and the single largest blob overall
    # turns out to hug the scene's rotated no-data edge (likely an edge
    # artifact, not a real feature), so components within 3% of the axis-box
    # border are excluded before ranking by size.
    high_rgb = np.array([round(c * 255) for c in to_rgb(AGP_TIER_COLORS[3])])
    box = rgb[top:bottom, left:right].astype(int)
    mask = np.abs(box - high_rgb).sum(axis=2) < 30
    bh, bw = mask.shape
    edge_margin = 0.03
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    best_size, cx_box, cy_box = 0, 0.5, 0.5
    for comp in range(1, n + 1):
        ys, xs = np.nonzero(labels == comp)
        if xs.min() < edge_margin * bw or xs.max() > (1 - edge_margin) * bw:
            continue
        if ys.min() < edge_margin * bh or ys.max() > (1 - edge_margin) * bh:
            continue
        if xs.size > best_size:
            best_size = xs.size
            cx_box, cy_box = float(xs.mean() / bw), float(ys.mean() / bh)

    canvas, cx, cy = _compose(
        map_crop, legend, cx_box, cy_box, legend_right_margin_px=90, legend_max_h_frac=0.5
    )
    canvas.save(OUT / "fallback_07.png")
    zoom_end = 1.06
    canvas.crop(_max_zoom_window(cx, cy, zoom_end)).save(OUT / "verify_07_maxzoom.png")
    print(f"07: cx={cx:.3f} cy={cy:.3f} zoom_end={zoom_end} (from {mask.sum()} high-tier px)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_goldfield()
    build_bingham()


if __name__ == "__main__":
    main()
