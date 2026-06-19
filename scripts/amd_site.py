"""AMD acid-generating-potential proxy for a site's scene (spec.md step 7).

Runs the MTMF pipeline, reduces the secondary AMD-indicator assemblage
(jarosite / Fe-oxides / gypsum) to an ordinal acid-generating-potential map via
:func:`tanager_rocks.hazard.acid_generating_potential`, and writes the tier
GeoTIFF + a categorical PNG. Site-agnostic; the default is the headline
narrative site (Bingham / Kennecott), but it runs identically on Goldfield.

Run::

    uv run python scripts/amd_site.py --site bingham
    uv run python scripts/amd_site.py --site goldfield
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import mask_absorption_bands

from tanager_rocks.config import SITES, TANAGER_SR_ASSET
from tanager_rocks.hazard import AGP_LABELS, acid_generating_potential
from tanager_rocks.speclib import load_library, select_endmembers
from tanager_rocks.unmix import mtmf
from tanager_rocks.viz import amd_map, setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("amd_site")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
MAPS_DIR = ROOT / "data" / "intermediate" / "maps"
FIGURES_DIR = ROOT / "figures"

# Tier nodata sentinel for the int16 raster (NaN off-domain pixels).
TIER_NODATA = -1


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Bingham is the headline AMD site (mine-waste / tailings narrative); the
    # proxy runs identically on Goldfield, where jarosite validated strongest.
    parser.add_argument("--site", default="bingham", choices=tuple(SITES))
    parser.add_argument("--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate")
    parser.add_argument(
        "--quantile", type=float, default=0.90, help="per-mineral detection floor (upper tail)"
    )
    args = parser.parse_args(argv)
    site = SITES[args.site]
    scene_id = site.scene_ids[0]

    setup_style()
    cube, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    cube = mask_absorption_bands(cube, wl)
    crs, transform = cube.rio.crs, cube.rio.transform()

    ds = mtmf(cube, select_endmembers(load_library(SPECLIB_DIR, wl)))
    result = acid_generating_potential(ds, max_infeas=args.max_infeas, quantile=args.quantile)

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # GeoTIFF: int16 ordinal tiers, NaN -> nodata sentinel.
    tier_int = np.where(np.isfinite(result.tiers.values), result.tiers.values, TIER_NODATA)
    raster = result.tiers.copy(data=tier_int.astype("int16"))
    raster.rio.write_crs(crs).rio.write_transform(transform).rio.write_nodata(
        TIER_NODATA
    ).rio.to_raster(MAPS_DIR / f"{args.site}_{scene_id}_amd_agp.tif")

    fig = amd_map(
        result.tiers,
        title=f"{site.name} — acid-generating-potential proxy (Tanager MTMF assemblage)",
        labels=AGP_LABELS,
    )
    out = FIGURES_DIR / f"{args.site}_{scene_id}_amd_agp.png"
    fig.savefig(out)
    in_domain = int(result.domain.sum())
    logger.info(
        "wrote %s — %d in-scene px, tiers %s",
        out,
        in_domain,
        {AGP_LABELS[c]: result.counts[c] for c in result.counts},
    )


if __name__ == "__main__":
    main()
