"""Bingham hero mineral map (spec.md step 9, Visualization & Storytelling).

Runs the MTMF pipeline on the hero site's lead scene, gates abundance by the
mixture-tuned infeasibility, and composites the per-mineral layers into a single
dominant-mineral map — the submission's "readable in 10 s" figure. Defaults
match `unmix_site.py` so the hero is consistent with the per-mineral panels.

Run::

    uv run python scripts/hero_map.py --site bingham
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
from tanager_rocks.unmix import mtmf
from tanager_rocks.viz import mineral_map, setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hero_map")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
FIGURES_DIR = ROOT / "figures"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Goldfield/Cuprite is the hero: it is the site whose maps validate cleanly
    # against the Rockwell ASTER map (alunite advanced-argillic centre at Cuprite).
    parser.add_argument("--site", default="goldfield", choices=tuple(SITES))
    parser.add_argument(
        "--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate (see unmix_site.py)"
    )
    parser.add_argument(
        "--quantile", type=float, default=0.90, help="per-mineral detection quantile"
    )
    args = parser.parse_args(argv)
    site = SITES[args.site]
    scene_id = site.scene_ids[0]

    setup_style()
    cube, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    cube = mask_absorption_bands(cube, wl)
    ds = mtmf(cube, select_endmembers(load_library(SPECLIB_DIR, wl)))
    minerals = [v[:-3] for v in ds.data_vars if v.endswith("_mf")]
    gated = xr.Dataset(
        {m: ds[f"{m}_mf"].where(ds[f"{m}_infeas"] < args.max_infeas) for m in minerals}
    )

    fig = mineral_map(
        gated,
        title=f"{site.name} — dominant alteration mineral (Tanager MTMF)",
        per_mineral_quantile=args.quantile,
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{args.site}_{scene_id}_hero_mineral_map.png"
    fig.savefig(out)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
