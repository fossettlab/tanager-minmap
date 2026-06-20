"""Build the submission story-page figures into ``submission/figures/``.

Generates the presentation composites unique to the story page — true-color
context, the diagnostic-spectra figure, and the validation side-by-side — from
the same products the pipeline computes. The per-stage analytical panels
(band-ablation, EMIT, hero, AMD) come from the ``tanager-minmap`` CLI.

Run::

    uv run python scripts/build_submission.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import rioxarray  # noqa: F401  (registers the .rio accessor; used via open_rasterio)
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import mask_absorption_bands

from tanager_rocks.config import SITES, TANAGER_SR_ASSET
from tanager_rocks.figures import (
    representative_spectra,
    rgb_context,
    spectra_story,
    validation_pair,
)
from tanager_rocks.reference import MINERAL_TO_ROCKWELL, ROCKWELL_EXCLUDED, align_reference
from tanager_rocks.speclib import load_library, select_endmembers
from tanager_rocks.unmix import mtmf
from tanager_rocks.viz import setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_submission")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
REF_DIR = ROOT / "data" / "reference"
FIG_DIR = ROOT / "submission" / "figures"

# Minerals stacked in the diagnostic-spectra story (distinct diagnostic features).
STORY_MINERALS = ["alunite", "kaolinite", "jarosite", "muscovite", "hematite"]
# Diagnostic absorptions to mark (the SWIR features plus the VNIR ferric band).
ABSORPTIONS = {
    "Al-OH 2200": 2200.0,
    "jarosite 2265": 2265.0,
    "gypsum/carb 2340": 2340.0,
    "Fe³⁺ ~900": 900.0,
}


def _load_raw(site_id: str):
    site = SITES[site_id]
    scene_id = site.scene_ids[0]
    cube, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    return site, scene_id, cube, wl


def _rgb(site_id: str, cube, wl) -> None:
    site = SITES[site_id]
    fig = rgb_context(cube, wl, title=f"{site.name} — Tanager true color")
    out = FIG_DIR / f"{site_id}_rgb.png"
    fig.savefig(out)
    logger.info("wrote %s", out)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Goldfield (showcase): RGB + diagnostic spectra + validation side-by-side.
    site, scene_id, cube_raw, wl = _load_raw("goldfield")
    _rgb("goldfield", cube_raw, wl)

    cube = mask_absorption_bands(cube_raw, wl)
    endmembers = select_endmembers(load_library(SPECLIB_DIR, wl))
    ds = mtmf(cube, endmembers)

    pixel_spectra = representative_spectra(cube, ds, STORY_MINERALS)
    spectra_story(
        endmembers,
        pixel_spectra,
        wl,
        STORY_MINERALS,
        absorptions=ABSORPTIONS,
        title=f"{site.name}: USGS library vs. Tanager scene spectra",
    ).savefig(FIG_DIR / "goldfield_spectra.png")
    logger.info("wrote %s", FIG_DIR / "goldfield_spectra.png")

    ref_raster = rioxarray.open_rasterio(
        REF_DIR / f"rockwell_goldfield_{scene_id}.tif", masked=False
    ).squeeze("band", drop=True)
    reference = align_reference(ref_raster, cube.isel(band=0))
    validation_pair(
        ds["alunite_mf"],
        reference,
        MINERAL_TO_ROCKWELL["alunite"],
        mineral="alunite",
        title=f"{site.name}: Tanager alunite vs. Rockwell ASTER alteration zone",
        excluded=ROCKWELL_EXCLUDED,
    ).savefig(FIG_DIR / "goldfield_validation_pair.png")
    logger.info("wrote %s", FIG_DIR / "goldfield_validation_pair.png")

    # Bingham (mine-waste site): true-color context.
    _, _, bingham_raw, bingham_wl = _load_raw("bingham")
    _rgb("bingham", bingham_raw, bingham_wl)


if __name__ == "__main__":
    main()
