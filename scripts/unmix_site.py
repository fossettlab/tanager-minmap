"""Unmixing for a site's scene (spec.md step 4): SAM baseline + MTMF.

Loads the SR cube, masks absorption bands, and selects one medoid endmember per
target mineral from splib07. Runs (1) the SAM baseline — best-match
classification within an angle threshold — and (2) MTMF — covariance-aware
matched-filter abundance plus the mixture-tuned infeasibility, gated to keep
abundance only where the pixel is spectrally feasible. Writes class/abundance/
infeasibility GeoTIFFs and PNG panels.

Run::

    uv run python scripts/unmix_site.py --site bingham
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import xarray as xr
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import mask_absorption_bands

from tanager_rocks.config import SITES, TANAGER_SR_ASSET
from tanager_rocks.speclib import load_library, select_endmembers
from tanager_rocks.unmix import mtmf, sam_classify, spectral_angle
from tanager_rocks.viz import classification_map, score_panel, setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("unmix_site")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
MAPS_DIR = ROOT / "data" / "intermediate" / "maps"
FIGURES_DIR = ROOT / "figures"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="bingham", choices=tuple(SITES))
    # 0.15 rad ~ the 5th percentile of best-match angles on the Bingham scene:
    # full-spectrum SAM vs pure endmembers rarely beats ~0.14 rad on mixed 30 m
    # pixels, so this keeps only the most spectrally pure matches. SAM is a
    # coarse baseline here (not ground-truth-calibrated); MTMF is the primary
    # method. Tune with --max-angle.
    parser.add_argument(
        "--max-angle", type=float, default=0.15, help="SAM acceptance threshold (radians)"
    )
    # Infeasibility gate for MTMF detections. Background sits near ~0.2 and the
    # anomalous false-positive tail runs well above ~2 on Bingham; 1.0 keeps the
    # feasible high-abundance "nose" while dropping the worst anomalies. Coarse
    # and not ground-truth-calibrated (that comes with USGS-map validation).
    parser.add_argument("--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate")
    args = parser.parse_args(argv)
    site = SITES[args.site]

    scene_id = site.scene_ids[0]
    cube, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    cube = mask_absorption_bands(cube, wl)

    endmembers = select_endmembers(load_library(SPECLIB_DIR, wl))
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    crs, transform = cube.rio.crs, cube.rio.transform()

    # --- SAM baseline ---
    angles = spectral_angle(cube, endmembers)
    classes, labels = sam_classify(angles, max_angle_rad=args.max_angle)
    counts = {labels[i]: int((classes.values == i).sum()) for i in range(len(labels))}
    counts["unclassified"] = int((classes.values == -1).sum())
    logger.info("SAM class counts (max_angle=%.3f rad): %s", args.max_angle, counts)
    classes.rio.write_crs(crs).rio.write_transform(transform).astype("int16").rio.to_raster(
        MAPS_DIR / f"{args.site}_{scene_id}_sam_class.tif"
    )
    classification_map(
        classes, labels, title=f"{site.name} ({scene_id}) — SAM classification"
    ).savefig(FIGURES_DIR / f"{args.site}_{scene_id}_sam_class.png")

    # --- MTMF: matched-filter abundance + mixture-tuned infeasibility ---
    ds = mtmf(cube, endmembers)
    minerals = [v[:-3] for v in ds.data_vars if v.endswith("_mf")]
    mf = xr.Dataset({m: ds[f"{m}_mf"] for m in minerals})
    infeas = xr.Dataset({m: ds[f"{m}_infeas"] for m in minerals})
    # Mixture-tuned: keep abundance only where the pixel is spectrally feasible.
    gated = xr.Dataset(
        {m: ds[f"{m}_mf"].where(ds[f"{m}_infeas"] < args.max_infeas) for m in minerals}
    )

    for mineral in minerals:
        for kind, da in (("mf", ds[f"{mineral}_mf"]), ("infeas", ds[f"{mineral}_infeas"])):
            geo = da.rio.write_crs(crs).rio.write_transform(transform)
            geo.rio.to_raster(MAPS_DIR / f"{args.site}_{scene_id}_{kind}_{mineral}.tif")

    base = f"{site.name} ({scene_id})"
    score_panel(mf, f"{base} — matched-filter abundance", cbar_label="MF score").savefig(
        FIGURES_DIR / f"{args.site}_{scene_id}_mf.png"
    )
    score_panel(infeas, f"{base} — MTMF infeasibility", cbar_label="infeasibility").savefig(
        FIGURES_DIR / f"{args.site}_{scene_id}_infeas.png"
    )
    out_png = FIGURES_DIR / f"{args.site}_{scene_id}_mtmf_gated.png"
    score_panel(
        gated, f"{base} — MTMF abundance (infeas < {args.max_infeas})", cbar_label="MF score"
    ).savefig(out_png)
    logger.info("wrote %s", out_png)


if __name__ == "__main__":
    main()
