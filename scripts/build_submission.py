"""Build the submission story-page figures + interactive maps.

Generates the presentation composites unique to the story page — true-color
context, the diagnostic-spectra figure, the validation side-by-side — plus the
two interactive folium maps (Goldfield dominant minerals, Bingham AMD), from the
same products the pipeline computes. The per-stage analytical panels
(band-ablation, EMIT, hero, AMD) come from the ``tanager-minmap`` CLI.

Run::

    uv run python scripts/build_submission.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor; used via open_rasterio)
import xarray as xr
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import mask_absorption_bands

from tanager_rocks.config import SITES, TANAGER_SR_ASSET
from tanager_rocks.figures import (
    representative_spectra,
    rgb_context,
    spectra_story,
    validation_pair,
)
from tanager_rocks.hazard import acid_generating_potential
from tanager_rocks.interactive import class_rgba, reproject_classes_4326, story_map
from tanager_rocks.pipeline import PipelinePaths, run_ablate, run_amd, run_emit, run_hero
from tanager_rocks.reference import MINERAL_TO_ROCKWELL, ROCKWELL_EXCLUDED, align_reference
from tanager_rocks.speclib import load_library, select_endmembers
from tanager_rocks.unmix import mtmf
from tanager_rocks.viz import AGP_TIER_COLORS, MINERAL_COLORS, dominant_mineral_class, setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_submission")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
REF_DIR = ROOT / "data" / "reference"
SUBMISSION = ROOT / "submission"
FIG_DIR = SUBMISSION / "figures"

MAX_INFEAS = 1.0  # MTMF infeasibility gate (matches the pipeline default)
STORY_MINERALS = ["alunite", "kaolinite", "jarosite", "muscovite", "hematite"]
ABSORPTIONS = {
    "Al-OH 2200": 2200.0,
    "jarosite 2265": 2265.0,
    "gypsum/carb 2340": 2340.0,
    "Fe³⁺ ~900": 900.0,
}


def _load(site_id: str):
    site = SITES[site_id]
    scene_id = site.scene_ids[0]
    cube_raw, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    return site, scene_id, cube_raw, wl


def _save(fig, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out)
    logger.info("wrote %s", out)


def _gated(ds: xr.Dataset) -> xr.Dataset:
    minerals = [v[:-3] for v in ds.data_vars if str(v).endswith("_mf")]
    return xr.Dataset({m: ds[f"{m}_mf"].where(ds[f"{m}_infeas"] < MAX_INFEAS) for m in minerals})


def _mineral_map_html(site_id: str, cube: xr.DataArray, ds: xr.Dataset) -> None:
    code, minerals = dominant_mineral_class(_gated(ds))
    arr, bounds = reproject_classes_4326(code, cube.rio.crs, cube.rio.transform())
    colors = {i: MINERAL_COLORS.get(m, "#888888") for i, m in enumerate(minerals)}
    rgba = class_rgba(arr, colors, transparent=frozenset({-1}))
    out = SUBMISSION / f"{site_id}_map.html"
    story_map(rgba, bounds, layer_name="Dominant alteration mineral (Tanager MTMF)").save(str(out))
    logger.info("wrote %s", out)


def _amd_map_html(site_id: str, cube: xr.DataArray, ds: xr.Dataset) -> None:
    tiers = acid_generating_potential(ds).tiers
    tier_int = xr.where(np.isfinite(tiers), tiers, -1).astype(int)
    arr, bounds = reproject_classes_4326(tier_int, cube.rio.crs, cube.rio.transform())
    rgba = class_rgba(arr, AGP_TIER_COLORS, transparent=frozenset({0}))  # background transparent
    out = SUBMISSION / f"{site_id}_map.html"
    fmap = story_map(rgba, bounds, layer_name="Acid-generating-potential proxy (Tanager MTMF)")
    fmap.save(str(out))
    logger.info("wrote %s", out)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Goldfield (showcase): RGB + spectra + validation + dominant-mineral map.
    site, scene_id, cube_raw, wl = _load("goldfield")
    _save(rgb_context(cube_raw, wl, title=f"{site.name} — Tanager true color"), "goldfield_rgb.png")

    cube = mask_absorption_bands(cube_raw, wl)
    endmembers = select_endmembers(load_library(SPECLIB_DIR, wl))
    ds = mtmf(cube, endmembers)

    pixel_spectra = representative_spectra(cube, ds, STORY_MINERALS)
    _save(
        spectra_story(
            endmembers,
            pixel_spectra,
            wl,
            STORY_MINERALS,
            absorptions=ABSORPTIONS,
            title=f"{site.name}: USGS library vs. Tanager scene spectra",
        ),
        "goldfield_spectra.png",
    )

    ref_raster = rioxarray.open_rasterio(
        REF_DIR / f"rockwell_goldfield_{scene_id}.tif", masked=False
    ).squeeze("band", drop=True)
    reference = align_reference(ref_raster, cube.isel(band=0))
    _save(
        validation_pair(
            ds["alunite_mf"],
            reference,
            MINERAL_TO_ROCKWELL["alunite"],
            mineral="alunite",
            title=f"{site.name}: Tanager alunite vs. Rockwell ASTER alteration zone",
            excluded=ROCKWELL_EXCLUDED,
        ),
        "goldfield_validation_pair.png",
    )
    _mineral_map_html("goldfield", cube, ds)

    # Bingham (mine-waste site): RGB context + AMD interactive map.
    site_b, _, bingham_raw, wl_b = _load("bingham")
    rgb_b = rgb_context(bingham_raw, wl_b, title=f"{site_b.name} — Tanager true color")
    _save(rgb_b, "bingham_rgb.png")
    cube_b = mask_absorption_bands(bingham_raw, wl_b)
    ds_b = mtmf(cube_b, select_endmembers(load_library(SPECLIB_DIR, wl_b)))
    _amd_map_html("bingham", cube_b, ds_b)

    # Reused analytical panels: regenerate into submission/figures via the
    # pipeline (the DRY home for these), so the story page is self-contained.
    panels = replace(PipelinePaths.repo_default(ROOT), figures_dir=FIG_DIR)
    run_ablate(SITES["bingham"], panels)  # band-ablation panel
    run_hero(SITES["goldfield"], panels)  # dominant-mineral hero map
    run_amd(SITES["bingham"], panels)  # AMD acid-generating-potential map
    if os.environ.get("EARTHDATA_USERNAME"):
        run_emit(SITES["goldfield"], panels)  # EMIT cross-sensor panel (needs creds)
    else:
        logger.warning("EMIT panel skipped (no EARTHDATA_USERNAME); run under doppler to include it")


if __name__ == "__main__":
    main()
