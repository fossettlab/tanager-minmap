"""Validate a site's mineral maps against the Rockwell ASTER reference.

spec.md step 4-5 ("validated maps"). Recomputes the diagnostic band-depth and
MTMF abundance maps for a site's lead scene, aligns the published Rockwell
alteration raster to the scene grid (run ``download_reference.py`` first), and
reports, per layer, how well the score separates its published alteration
zone(s) from the other classified ground (rank AUC + Mann-Whitney p) plus the
Youden-J-optimal threshold that calibrates detection to the external map.

Run::

    uv run python scripts/download_reference.py --site goldfield
    uv run python scripts/validate_site.py --site goldfield
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from pathlib import Path

import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import mask_absorption_bands

from tanager_rocks.config import SITES, TANAGER_SR_ASSET
from tanager_rocks.features import build_feature_defs, diagnostic_feature_maps
from tanager_rocks.reference import (
    FEATURE_TO_ROCKWELL,
    MINERAL_TO_ROCKWELL,
    align_reference,
)
from tanager_rocks.speclib import load_library, select_endmembers
from tanager_rocks.unmix import mtmf
from tanager_rocks.validate import Discrimination, validate_scores
from tanager_rocks.viz import setup_style, zone_discrimination_panel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("validate_site")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
REF_DIR = ROOT / "data" / "reference"
OUT_DIR = ROOT / "data" / "intermediate" / "validation"
FIGURES_DIR = ROOT / "figures"


def _write_csv(path: Path, results: dict[str, Discrimination], kind: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "kind",
                "layer",
                "positive_classes",
                "n_pos",
                "n_neg",
                "auc",
                "p_value",
                "median_in",
                "median_out",
                "threshold",
                "tpr",
                "fpr",
                "youden_j",
            ]
        )
        for layer, d in results.items():
            w.writerow(
                [
                    kind,
                    layer,
                    " ".join(map(str, d.positive_classes)),
                    d.n_pos,
                    d.n_neg,
                    f"{d.auc:.4f}",
                    f"{d.p_value:.3e}",
                    f"{d.median_pos:.5f}",
                    f"{d.median_neg:.5f}",
                    f"{d.threshold:.5f}",
                    f"{d.tpr:.4f}",
                    f"{d.fpr:.4f}",
                    f"{d.youden_j:.4f}",
                ]
            )


def _log_table(results: dict[str, Discrimination], kind: str) -> None:
    logger.info("--- %s discrimination vs Rockwell zones ---", kind)
    for layer, d in results.items():
        logger.info(
            "%-16s AUC=%.3f p=%.1e n+=%d n-=%d thr=%.4f (TPR=%.2f FPR=%.2f)",
            layer,
            d.auc,
            d.p_value,
            d.n_pos,
            d.n_neg,
            d.threshold,
            d.tpr,
            d.fpr,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="goldfield", choices=tuple(SITES))
    args = parser.parse_args(argv)
    site = SITES[args.site]
    scene_id = site.scene_ids[0]

    ref_path = REF_DIR / f"rockwell_{args.site}_{scene_id}.tif"
    if not ref_path.exists():
        raise SystemExit(
            f"reference clip {ref_path} missing — run "
            f"`uv run python scripts/download_reference.py --site {args.site}` first"
        )

    cube, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    cube = mask_absorption_bands(cube, wl)
    reference = align_reference(
        rioxarray.open_rasterio(ref_path, masked=False).squeeze("band", drop=True),
        cube.isel(band=0),
    )

    depths = diagnostic_feature_maps(cube, wl, build_feature_defs(wl, SPECLIB_DIR))
    ds = mtmf(cube, select_endmembers(load_library(SPECLIB_DIR, wl)))
    minerals = [v[:-3] for v in ds.data_vars if v.endswith("_mf")]
    mf = xr.Dataset({m: ds[f"{m}_mf"] for m in minerals})

    feat_results = validate_scores(depths, reference, FEATURE_TO_ROCKWELL)
    mineral_results = validate_scores(mf, reference, MINERAL_TO_ROCKWELL)
    _log_table(feat_results, "band-depth feature")
    _log_table(mineral_results, "MTMF abundance")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_DIR / f"validation_{args.site}_{scene_id}.csv", feat_results, "feature")
    with open(OUT_DIR / f"validation_{args.site}_{scene_id}.csv", "a", newline="") as fh:
        w = csv.writer(fh)
        for layer, d in mineral_results.items():
            w.writerow(
                [
                    "mtmf",
                    layer,
                    " ".join(map(str, d.positive_classes)),
                    d.n_pos,
                    d.n_neg,
                    f"{d.auc:.4f}",
                    f"{d.p_value:.3e}",
                    f"{d.median_pos:.5f}",
                    f"{d.median_neg:.5f}",
                    f"{d.threshold:.5f}",
                    f"{d.tpr:.4f}",
                    f"{d.fpr:.4f}",
                    f"{d.youden_j:.4f}",
                ]
            )

    setup_style()
    base = f"{site.name} ({scene_id})"
    zone_discrimination_panel(
        depths,
        reference,
        FEATURE_TO_ROCKWELL,
        feat_results,
        title=f"{base} — band depth by Rockwell zone",
    ).savefig(FIGURES_DIR / f"{args.site}_{scene_id}_validation_features.png")
    zone_discrimination_panel(
        mf,
        reference,
        MINERAL_TO_ROCKWELL,
        mineral_results,
        title=f"{base} — MTMF abundance by Rockwell zone",
    ).savefig(FIGURES_DIR / f"{args.site}_{scene_id}_validation_mtmf.png")
    logger.info("wrote validation CSV + figures for %s", args.site)


if __name__ == "__main__":
    main()
