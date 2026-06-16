"""Produce continuum-removed diagnostic band-depth maps for a site's scene.

Pipeline for one Tanager scene (spec.md steps 2-3): load the SR cube, mask the
O2/H2O absorption bands, derive each diagnostic feature's continuum shoulders
from the splib07 endmembers (data-driven, not hand-picked), compute the
band-depth maps, and write GeoTIFFs + a PNG panel. This is the "one real map"
deliverable; SAM/MTMF unmixing (unmix.py) comes later.

Run::

    uv run python scripts/download_speclib.py   # once
    uv run python scripts/download_scenes.py --site bingham
    uv run python scripts/map_site.py --site bingham
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import mask_absorption_bands

from tanager_rocks.config import (
    DIAGNOSTIC_NM,
    FEATURE_DIAGNOSTIC_MINERAL,
    SITES,
    TANAGER_SR_ASSET,
)
from tanager_rocks.features import FeatureDef, diagnostic_feature_maps, shoulders_from_endmember
from tanager_rocks.speclib import by_mineral, load_library
from tanager_rocks.viz import band_depth_panel, setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("map_site")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
MAPS_DIR = ROOT / "data" / "intermediate" / "maps"
FIGURES_DIR = ROOT / "figures"


def build_feature_defs(wavelengths: np.ndarray) -> list[FeatureDef]:
    """Derive FeatureDefs whose shoulders come from the splib07 endmembers."""
    grouped = by_mineral(load_library(SPECLIB_DIR, wavelengths))
    defs: list[FeatureDef] = []
    for name, center in DIAGNOSTIC_NM.items():
        mineral = FEATURE_DIAGNOSTIC_MINERAL[name]
        samples = grouped[mineral]
        median = np.nanmedian(np.vstack([e.reflectance for e in samples]), axis=0)
        lo, hi = shoulders_from_endmember(wavelengths, median, center)
        defs.append(
            FeatureDef(name, center, lo, hi, source=f"splib07a {mineral} median (n={len(samples)})")
        )
        logger.info("%s: center %.0f, shoulders %.0f / %.0f nm (%s)", name, center, lo, hi, mineral)
    return defs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="bingham", choices=tuple(SITES))
    args = parser.parse_args(argv)
    site = SITES[args.site]

    scene_id = site.scene_ids[0]
    scene_path = RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5"
    logger.info("loading %s", scene_path)
    cube, wl = load_tanager_sr_hdf5(scene_path)
    cube = mask_absorption_bands(cube, wl)

    depths = diagnostic_feature_maps(cube, wl, build_feature_defs(wl))

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name in depths.data_vars:
        da = depths[name].rio.write_crs(cube.rio.crs).rio.write_transform(cube.rio.transform())
        da.rio.to_raster(MAPS_DIR / f"{args.site}_{scene_id}_{name}.tif")

    setup_style()
    fig = band_depth_panel(depths, title=f"{site.name} ({scene_id}) — continuum-removed band depth")
    out_png = FIGURES_DIR / f"{args.site}_{scene_id}_band_depth.png"
    fig.savefig(out_png)
    logger.info("wrote %s", out_png)


if __name__ == "__main__":
    main()
