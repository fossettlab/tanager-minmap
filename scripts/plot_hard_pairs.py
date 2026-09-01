"""Render the hard-pairs figure: RGB-ambiguous, SWIR-separable patch pairs.

Reads ``data/processed/hard_pairs/pairs.csv`` and ``summary.json`` (written by
``scripts/find_hard_pairs.py``) and renders the top ``--top-n`` pairs by SWIR
separability -- the mineralogical analog of the headline figure in the
Sentinel-2 "Similar-but-Different" post (Robinson & Corley 2026). Each row
shows the two patches' true-color chips side by side (rendered with the exact
pooled stretch that defined "RGB-ambiguous" during mining) plus their
overlaid continuum-removed SWIR spectra with the diagnostic absorptions
marked.

Run::

    uv run python scripts/plot_hard_pairs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tanager_spec.bands import indices_in_windows
from tanager_spec.io import load_tanager_sr_hdf5

from tanager_rocks.config import DIAGNOSTIC_NM, SITES, TANAGER_SR_ASSET
from tanager_rocks.figures import RGB_NM, _nearest
from tanager_rocks.pairs import SWIR_WINDOW_NM, continuum_removed, stretch_to_uint8
from tanager_rocks.quality import mask_tanager_scene
from tanager_rocks.viz import MINERAL_COLORS, setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
IN_DIR = ROOT / "data" / "processed" / "hard_pairs"
FIGURES_DIR = ROOT / "figures"


def _load_pairs_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _load_site_cube(site_id: str):
    """Quality-masked cube and wavelength axis for one site's lead scene."""
    site = SITES[site_id]
    scene_id = site.scene_ids[0]
    path = RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5"
    cube_raw, wl = load_tanager_sr_hdf5(path)
    cube_masked, _ = mask_tanager_scene(cube_raw, wl, path)
    return cube_masked, cube_masked, wl


def _patch_slice(row: int, col: int, patch_size: int) -> tuple[slice, slice]:
    y0, x0 = row * patch_size, col * patch_size
    return slice(y0, y0 + patch_size), slice(x0, x0 + patch_size)


def _rgb_chip(cube_raw, wl: np.ndarray, ys: slice, xs: slice, lo: np.ndarray, hi: np.ndarray):
    idx = [_nearest(wl, t) for t in RGB_NM]
    patch = cube_raw.isel(band=idx, y=ys, x=xs).values  # (3, size, size)
    invalid = ~np.isfinite(patch).all(axis=0)
    return stretch_to_uint8(patch, invalid, lo, hi)


def _swir_spectrum(
    cube_masked, wl: np.ndarray, ys: slice, xs: slice
) -> tuple[np.ndarray, np.ndarray]:
    win_idx = np.flatnonzero(indices_in_windows(wl, [SWIR_WINDOW_NM]))
    patch = cube_masked.isel(band=win_idx, y=ys, x=xs).values  # (n_win, size, size)
    mean_spectrum = np.nanmean(patch.reshape(patch.shape[0], -1), axis=1)
    return wl[win_idx], mean_spectrum


def _upsample(chip: np.ndarray, factor: int = 8) -> np.ndarray:
    """Nearest-neighbor upsample a small RGB chip so it reads at figure scale."""
    return np.kron(chip, np.ones((factor, factor, 1), dtype=chip.dtype))


def render(rows: list[dict[str, str]], lo: np.ndarray, hi: np.ndarray, patch_size: int):
    setup_style()
    site_cubes: dict[str, tuple] = {}

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.3 * n), squeeze=False)
    footprint_m = patch_size * 30.0  # Tanager GSD

    for r, row in enumerate(rows):
        for tag in ("a", "b"):
            site_id = row[f"site_{tag}"]
            if site_id not in site_cubes:
                logger.info("loading %s lead scene", site_id)
                site_cubes[site_id] = _load_site_cube(site_id)

        ax_a, ax_b, ax_spec = axes[r]
        specs = {}
        for tag, ax in (("a", ax_a), ("b", ax_b)):
            site_id = row[f"site_{tag}"]
            cube_raw, cube_masked, wl = site_cubes[site_id]
            ys, xs = _patch_slice(int(row[f"row_{tag}"]), int(row[f"col_{tag}"]), patch_size)
            chip = _rgb_chip(cube_raw, wl, ys, xs, lo, hi)
            ax.imshow(_upsample(chip), interpolation="nearest")
            label = row[f"label_{tag}"]
            ax.set_title(
                f"{site_id} — {label}", color=MINERAL_COLORS.get(label, "black"), fontsize=10
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(MINERAL_COLORS.get(label, "0.3"))
                spine.set_linewidth(2.5)
            win_wl, mean_spectrum = _swir_spectrum(cube_masked, wl, ys, xs)
            specs[tag] = (win_wl, continuum_removed(win_wl, mean_spectrum), label)

        for tag in ("a", "b"):
            win_wl, cr, label = specs[tag]
            ax_spec.plot(win_wl, cr, color=MINERAL_COLORS.get(label, "black"), lw=1.6, label=label)
        for feature_label, nm in DIAGNOSTIC_NM.items():
            if win_wl.min() <= nm <= win_wl.max():
                ax_spec.axvline(nm, color="0.6", lw=0.8, ls=":")
                ax_spec.text(
                    nm,
                    ax_spec.get_ylim()[1],
                    feature_label,
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=7,
                    c="0.4",
                )
        ax_spec.set_xlabel("wavelength (nm)")
        ax_spec.set_ylabel("continuum-removed R")
        ax_spec.legend(fontsize=8, frameon=False, loc="lower left")
        ax_spec.set_title(f"SWIR spectral angle {row['swir_angle_deg']}°", fontsize=10)

    fig.suptitle(
        f"Similar-but-different: RGB-ambiguous, SWIR-separable mineral patch pairs "
        f"({patch_size}×{patch_size} px, {footprint_m:.0f} m footprint)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / "hard_pairs.png")
    args = parser.parse_args(argv)

    with open(IN_DIR / "summary.json") as fh:
        summary = json.load(fh)
    lo = np.asarray(summary["rgb_stretch_lo"])
    hi = np.asarray(summary["rgb_stretch_hi"])
    patch_size = int(summary["patch_size_px"])

    pairs = _load_pairs_csv(IN_DIR / "pairs.csv")
    if not pairs:
        raise RuntimeError(
            f"{IN_DIR / 'pairs.csv'} has no rows; run scripts/find_hard_pairs.py first"
        )
    rows = pairs[: args.top_n]
    logger.info("rendering top %d of %d hard pairs", len(rows), len(pairs))

    fig = render(rows, lo, hi, patch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
