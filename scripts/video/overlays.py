"""RGBA sprite renderers for lower-thirds, callouts, highlights, and captions.

Per docs/edit_plan.md "Overlays -- consolidated spec": each overlay is a
1920x1080 matplotlib RGBA PNG (transparent background) with the navy pill/
scrim baked in, composited onto a beat's picture clip in ffmpeg with a timed
alpha fade + 12px rise-in (that compositing lives in beats.py, using the exact
filter template the plan gives). This module only draws the static sprite --
one PNG per overlay, at its fully-visible state.

Coordinate convention: (0, 0) is the top-left of the frame, y grows downward,
matching screen/video coordinates (not matplotlib's default bottom-left).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from common import BG_HEX, FONT_BOLD, FONT_REGULAR, HEIGHT, WIDTH  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

MARGIN = 96  # px, title-safe margin (Appendix A: "96 px (5%) from left/bottom")
PAD = 16
PILL_ALPHA = 0.72
PILL_RADIUS = 8


def _blank_canvas() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)  # y=0 at top
    ax.axis("off")
    return fig, ax


def _extent(ax: plt.Axes, artists: list, renderer) -> tuple[float, float, float, float]:
    """Bounding box of one or more text artists, in this axes' data coords."""
    xs, ys = [], []
    for a in artists:
        bbox = a.get_window_extent(renderer=renderer)
        pts = ax.transData.inverted().transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        xs += [pts[0][0], pts[1][0]]
        ys += [pts[0][1], pts[1][1]]
    return min(xs), min(ys), max(xs), max(ys)


def _pill_behind(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0 - PAD, y0 - PAD),
            (x1 - x0) + 2 * PAD,
            (y1 - y0) + 2 * PAD,
            boxstyle=f"round,pad=0,rounding_size={PILL_RADIUS}",
            fc=BG_HEX,
            ec="none",
            alpha=PILL_ALPHA,
            zorder=1,
        )
    )


def lower_third(label: str, sublabel: str, out: Path) -> Path:
    """LT-01/04/05/06a/07: bold label + regular sublabel, navy pill, lower-left."""
    fig, ax = _blank_canvas()
    sub_y = HEIGHT - MARGIN - PAD  # bottom of sublabel text
    label_y = sub_y - 44  # gap above the sublabel
    sub = ax.text(
        MARGIN + PAD,
        sub_y,
        sublabel,
        color="white",
        ha="left",
        va="bottom",
        fontproperties=FontProperties(fname=str(FONT_REGULAR), size=32),
    )
    label_a = ax.text(
        MARGIN + PAD,
        label_y,
        label,
        color="white",
        ha="left",
        va="bottom",
        fontproperties=FontProperties(fname=str(FONT_BOLD), size=46),
    )
    fig.canvas.draw()
    x0, y0, x1, y1 = _extent(ax, [sub, label_a], fig.canvas.get_renderer())
    _pill_behind(ax, x0, y0, x1, y1)
    sub.set_zorder(2)
    label_a.set_zorder(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, transparent=True)
    plt.close(fig)
    return out


def chart_callout(
    label: str, tip_xy: tuple[float, float], label_xy: tuple[float, float], out: Path
) -> Path:
    """CALL-03a/b/c (fallback path) / CALL-04: white arrow + thin dark stroke,
    label in a navy pill, sub-size text (Appendix A callout style)."""
    fig, ax = _blank_canvas()
    ax.annotate(
        "",
        xy=tip_xy,
        xytext=label_xy,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "white",
            "lw": 1.6,
            "path_effects": [patheffects.withStroke(linewidth=3, foreground=BG_HEX)],
        },
    )
    label_a = ax.text(
        label_xy[0],
        label_xy[1],
        label,
        color="white",
        ha="center",
        va="center",
        fontproperties=FontProperties(fname=str(FONT_BOLD), size=29),
    )
    fig.canvas.draw()
    x0, y0, x1, y1 = _extent(ax, [label_a], fig.canvas.get_renderer())
    _pill_behind(ax, x0, y0, x1, y1)
    label_a.set_zorder(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, transparent=True)
    plt.close(fig)
    return out


def arrow_only(tip_xy: tuple[float, float], tail_xy: tuple[float, float], out: Path) -> Path:
    """CALL-04: white arrow + thin dark stroke pointing at the figure's own
    existing annotation box -- no new label (the text is native to the figure)."""
    fig, ax = _blank_canvas()
    ax.annotate(
        "",
        xy=tip_xy,
        xytext=tail_xy,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "white",
            "lw": 1.8,
            "path_effects": [patheffects.withStroke(linewidth=3.5, foreground=BG_HEX)],
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, transparent=True)
    plt.close(fig)
    return out


def matched_highlight(rects: list[tuple[float, float, float, float]], out: Path) -> Path:
    """HL-06a: matched zone rectangles (both panels), white stroke, no text."""
    fig, ax = _blank_canvas()
    for x, y, w, h in rects:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="white", lw=2.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, transparent=True)
    plt.close(fig)
    return out


def caption_strip(
    text: str, center_xy: tuple[float, float], out: Path, fontsize: float = 30
) -> Path:
    """CAP-06b: one caption strip in a navy pill, centered at `center_xy`."""
    fig, ax = _blank_canvas()
    txt = ax.text(
        center_xy[0],
        center_xy[1],
        text,
        color="white",
        ha="center",
        va="center",
        fontproperties=FontProperties(fname=str(FONT_BOLD), size=fontsize),
    )
    fig.canvas.draw()
    x0, y0, x1, y1 = _extent(ax, [txt], fig.canvas.get_renderer())
    _pill_behind(ax, x0, y0, x1, y1)
    txt.set_zorder(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, transparent=True)
    plt.close(fig)
    return out
