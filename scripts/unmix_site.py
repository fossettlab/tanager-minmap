"""SAM mineral classification for a site's scene (spec.md step 4, SAM half).

Loads the SR cube, masks absorption bands, selects one medoid endmember per
target mineral from splib07, computes the per-mineral spectral angle, and
assigns each pixel to its best match within an acceptance threshold. Writes a
class GeoTIFF and a PNG. MTMF (the covariance-aware primary method) is the next
increment; this establishes the SAM baseline.

Run::

    uv run python scripts/unmix_site.py --site bingham
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import mask_absorption_bands

from tanager_rocks.config import SITES, TANAGER_SR_ASSET
from tanager_rocks.speclib import load_library, select_endmembers
from tanager_rocks.unmix import sam_classify, spectral_angle
from tanager_rocks.viz import classification_map, setup_style

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
    args = parser.parse_args(argv)
    site = SITES[args.site]

    scene_id = site.scene_ids[0]
    cube, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    cube = mask_absorption_bands(cube, wl)

    endmembers = select_endmembers(load_library(SPECLIB_DIR, wl))
    angles = spectral_angle(cube, endmembers)
    classes, labels = sam_classify(angles, max_angle_rad=args.max_angle)

    counts = {labels[i]: int((classes.values == i).sum()) for i in range(len(labels))}
    counts["unclassified"] = int((classes.values == -1).sum())
    logger.info("class pixel counts (max_angle=%.3f rad): %s", args.max_angle, counts)

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    geo = classes.rio.write_crs(cube.rio.crs).rio.write_transform(cube.rio.transform())
    geo.astype("int16").rio.to_raster(MAPS_DIR / f"{args.site}_{scene_id}_sam_class.tif")

    setup_style()
    fig = classification_map(
        classes, labels, title=f"{site.name} ({scene_id}) — SAM classification"
    )
    out_png = FIGURES_DIR / f"{args.site}_{scene_id}_sam_class.png"
    fig.savefig(out_png)
    logger.info("wrote %s", out_png)


if __name__ == "__main__":
    main()
