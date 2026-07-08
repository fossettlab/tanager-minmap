"""Mine RGB-ambiguous, SWIR-separable hard patch pairs across both sites.

The mineralogical analog of the Sentinel-2 "Similar-but-Different" benchmark
(Robinson & Corley 2026, cited in METHODS.md): tiles both sites' lead scenes
into non-overlapping patches, labels each by its dominant MTMF mineral (the
same infeasibility-gated, per-mineral-quantile product behind the hero map),
and keeps patch pairs that are near-identical in true-color statistics but
carry different labels and pull apart in the SWIR. Reuses the cached
``data/intermediate/maps/*_mf_*.tif`` / ``*_infeas_*.tif`` GeoTIFFs rather
than re-running MTMF.

Writes ``data/processed/hard_pairs/patches.csv`` (every surviving labeled
patch), ``pairs.csv`` (every hard pair, sorted by SWIR separability), and
``summary.json`` (discard counts and derived thresholds), so every number in
the METHODS.md writeup traces back to this script.

Run::

    uv run python scripts/find_hard_pairs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from tanager_spec.bands import indices_in_windows
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import invalid_pixel_mask, mask_absorption_bands

from tanager_rocks.config import SITES, TANAGER_SR_ASSET, TARGET_MINERALS
from tanager_rocks.figures import RGB_NM, RGB_VALID_RANGE, _nearest
from tanager_rocks.pairs import (
    PATCH_SIZE_PX,
    PURITY_FLOOR,
    RGB_CANDIDATE_QUANTILE,
    SWIR_NULL_QUANTILE,
    SWIR_WINDOW_NM,
    Patch,
    pooled_rgb_percentiles,
    rgb_ambiguous_pairs,
    stretch_to_uint8,
    swir_separable_pairs,
    tile_and_label,
)
from tanager_rocks.viz import dominant_mineral_class

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MAPS_DIR = ROOT / "data" / "intermediate" / "maps"
OUT_DIR = ROOT / "data" / "processed" / "hard_pairs"


def _dominant_code(site_id: str, scene_id: str, max_infeas: float, quantile: float):
    """Rebuild the hero map's dominant-mineral code array from the cached GeoTIFFs.

    Reads the ``mf_<mineral>.tif`` / ``infeas_<mineral>.tif`` products already
    written by ``scripts/unmix_site.py`` and gates/composites them with the
    IDENTICAL function the hero map uses (:func:`tanager_rocks.viz.
    dominant_mineral_class`), so patch labels agree with the published map by
    construction rather than by re-deriving the logic.
    """
    mf = {
        m: rioxarray.open_rasterio(
            MAPS_DIR / f"{site_id}_{scene_id}_mf_{m}.tif", masked=True
        ).squeeze("band", drop=True)
        for m in TARGET_MINERALS
    }
    infeas = {
        m: rioxarray.open_rasterio(
            MAPS_DIR / f"{site_id}_{scene_id}_infeas_{m}.tif", masked=True
        ).squeeze("band", drop=True)
        for m in TARGET_MINERALS
    }
    gated = xr.Dataset({m: mf[m].where(infeas[m] < max_infeas) for m in TARGET_MINERALS})
    code_da, minerals = dominant_mineral_class(gated, per_mineral_quantile=quantile)
    return code_da.values, minerals


def _site_products(site_id: str, max_infeas: float, quantile: float):
    """Load one site's lead scene and reduce it to the arrays the miner needs.

    Loads the full 426-band cube only inside this function's scope so it can
    be garbage-collected before the second site is loaded -- ``pairs.py``'s
    cross-scene stretch needs both sites' RGB/SWIR products in memory at
    once, but never both full cubes.
    """
    site = SITES[site_id]
    scene_id = site.scene_ids[0]
    cube_raw, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")

    rgb_idx = [_nearest(wl, t) for t in RGB_NM]
    rgb_sub = cube_raw.isel(band=rgb_idx)
    invalid = invalid_pixel_mask(rgb_sub, valid_range=RGB_VALID_RANGE).values
    rgb_raw = rgb_sub.values.astype(float)  # (3, ny, nx)

    cube_masked = mask_absorption_bands(cube_raw, wl)
    win_idx = indices_in_windows(wl, [SWIR_WINDOW_NM])
    swir_cube = cube_masked.isel(band=np.flatnonzero(win_idx)).values  # (n_win, ny, nx)
    logger.info(
        "%s (%s): %d SWIR-window bands (%.0f-%.0f nm)",
        site_id,
        scene_id,
        int(win_idx.sum()),
        SWIR_WINDOW_NM[0],
        SWIR_WINDOW_NM[1],
    )

    code, minerals = _dominant_code(site.site_id, scene_id, max_infeas, quantile)
    if code.shape != invalid.shape:
        raise ValueError(f"{site_id}: dominant-code grid {code.shape} != cube grid {invalid.shape}")

    return site, scene_id, rgb_raw, invalid, swir_cube, code, minerals


def _write_patches_csv(path: Path, patches: list[Patch]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "site_id",
                "scene_id",
                "row",
                "col",
                "y0",
                "x0",
                "label",
                "purity",
                "rgb_mean_r",
                "rgb_mean_g",
                "rgb_mean_b",
                "rgb_std_r",
                "rgb_std_g",
                "rgb_std_b",
            ]
        )
        for p in patches:
            writer.writerow(
                [
                    p.site_id,
                    p.scene_id,
                    p.row,
                    p.col,
                    p.y0,
                    p.x0,
                    p.label,
                    f"{p.purity:.4f}",
                    # Full precision (not display-rounded): scripts/build_hard_pairs_dataset.py
                    # re-derives the RGB-ambiguity graph from this CSV alone (no cube reload),
                    # and cross-checks the result against summary.json's cached thresholds --
                    # 2-decimal DN rounding was enough to shift that threshold measurably.
                    *[f"{v:.6f}" for v in p.rgb_mean],
                    *[f"{v:.6f}" for v in p.rgb_std],
                ]
            )


def _write_pairs_csv(path: Path, pairs) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "rank",
                "site_a",
                "row_a",
                "col_a",
                "label_a",
                "site_b",
                "row_b",
                "col_b",
                "label_b",
                "rgb_mean_l2",
                "rgb_std_l2",
                "swir_angle_deg",
            ]
        )
        for rank, hp in enumerate(pairs, start=1):
            writer.writerow(
                [
                    rank,
                    hp.a.site_id,
                    hp.a.row,
                    hp.a.col,
                    hp.a.label,
                    hp.b.site_id,
                    hp.b.row,
                    hp.b.col,
                    hp.b.label,
                    f"{hp.rgb_mean_l2:.3f}",
                    f"{hp.rgb_std_l2:.3f}",
                    f"{hp.swir_angle_deg:.3f}",
                ]
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE_PX)
    parser.add_argument("--purity-floor", type=float, default=PURITY_FLOOR)
    parser.add_argument("--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.90,
        help="per-mineral detection quantile (hero-map default)",
    )
    parser.add_argument("--rgb-quantile", type=float, default=RGB_CANDIDATE_QUANTILE)
    parser.add_argument("--swir-null-quantile", type=float, default=SWIR_NULL_QUANTILE)
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_patches: list[Patch] = []
    per_site_counts: dict[str, dict[str, int]] = {}
    channel_stacks = []
    site_products = []
    for site_id in SITES:
        prod = _site_products(site_id, args.max_infeas, args.quantile)
        site_products.append(prod)
        _, _, rgb_raw, invalid, _, _, _ = prod
        channel_stacks.append((rgb_raw, invalid))

    lo, hi = pooled_rgb_percentiles(channel_stacks)
    logger.info("pooled RGB stretch bounds: lo=%s hi=%s (reflectance)", lo, hi)

    for site, scene_id, rgb_raw, invalid, swir_cube, code, minerals in site_products:
        rgb_uint8 = stretch_to_uint8(rgb_raw, invalid, lo, hi)
        patches, counts = tile_and_label(
            code,
            minerals,
            invalid,
            rgb_uint8,
            swir_cube,
            site_id=site.site_id,
            scene_id=scene_id,
            patch_size=args.patch_size,
            purity_floor=args.purity_floor,
        )
        per_site_counts[site.site_id] = counts
        logger.info("%s: %s", site.site_id, counts)
        all_patches.extend(patches)

    logger.info("total labeled patches across both sites: %d", len(all_patches))
    if len(all_patches) < 2:
        raise RuntimeError("fewer than 2 labeled patches survived; cannot mine pairs")

    rgb_result = rgb_ambiguous_pairs(all_patches, quantile=args.rgb_quantile)
    logger.info(
        "RGB-ambiguous candidates: %d (mean_thr=%.2f, std_thr=%.2f, from %d cross-label pairs, "
        "distance range mean=[%.2f, %.2f] std=[%.2f, %.2f])",
        len(rgb_result.candidates),
        rgb_result.mean_threshold,
        rgb_result.std_threshold,
        rgb_result.cross_mean_distances.size,
        float(rgb_result.cross_mean_distances.min()),
        float(rgb_result.cross_mean_distances.max()),
        float(rgb_result.cross_std_distances.min()),
        float(rgb_result.cross_std_distances.max()),
    )

    swir_result = swir_separable_pairs(
        all_patches, rgb_result.candidates, null_quantile=args.swir_null_quantile
    )
    logger.info(
        "SWIR-separable hard pairs: %d (threshold=%.2f deg, from %d same-label null pairs, "
        "range=[%.2f, %.2f] deg)",
        len(swir_result.pairs),
        swir_result.threshold_deg,
        swir_result.same_label_angles_deg.size,
        float(swir_result.same_label_angles_deg.min()),
        float(swir_result.same_label_angles_deg.max()),
    )

    _write_patches_csv(OUT_DIR / "patches.csv", all_patches)
    _write_pairs_csv(OUT_DIR / "pairs.csv", swir_result.pairs)

    summary = {
        "patch_size_px": args.patch_size,
        "purity_floor": args.purity_floor,
        "max_infeas": args.max_infeas,
        "detection_quantile": args.quantile,
        # Reflectance-space bounds of the pooled true-color stretch, so
        # scripts/plot_hard_pairs.py can re-render RGB chips with the exact
        # stretch that defined "close" during mining, without reloading both
        # full cubes just to recompute percentiles.
        "rgb_stretch_lo": lo.tolist(),
        "rgb_stretch_hi": hi.tolist(),
        "per_site_counts": per_site_counts,
        "total_labeled_patches": len(all_patches),
        "rgb_ambiguous_candidates": len(rgb_result.candidates),
        "rgb_mean_threshold": rgb_result.mean_threshold,
        "rgb_std_threshold": rgb_result.std_threshold,
        "rgb_quantile": args.rgb_quantile,
        "n_cross_label_pairs": int(rgb_result.cross_mean_distances.size),
        "swir_separable_pairs": len(swir_result.pairs),
        "swir_threshold_deg": swir_result.threshold_deg,
        "swir_null_quantile": args.swir_null_quantile,
        "n_same_label_null_pairs": int(swir_result.same_label_angles_deg.size),
    }
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("wrote %s", OUT_DIR)


if __name__ == "__main__":
    main()
