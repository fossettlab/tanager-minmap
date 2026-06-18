"""Cross-sensor comparison: Tanager vs EMIT at a shared site (spec.md step 6).

Runs the identical alteration-mapping pipeline (diagnostic band depths + MTMF)
on a site's Tanager lead scene and an overlapping EMIT L2A scene, then reports
spectral agreement (scene-mean), per-mineral detection agreement (MTMF maps on
the common grid), and the resolution ratio.

The EMIT granule is queried, selected, and downloaded reproducibly: the clearest
fully-overlapping scene is chosen and its identifier logged. Network steps need
NASA Earthdata credentials in the environment, so run under doppler::

    doppler run --project mac --config dev -- \\
        uv run python scripts/compare_emit.py --site goldfield

If the reflectance file is already present it is reused (no re-download).
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

from tanager_rocks.compare import detection_agreement, reproject_crs, spectral_agreement
from tanager_rocks.config import SITES, TANAGER_SR_ASSET
from tanager_rocks.emit import (
    EMIT_L2A_SHORT_NAME,
    box,
    footprint_bbox,
    load_emit_reflectance,
    rank_granules,
    rfl_path,
    select_granule,
)
from tanager_rocks.features import build_feature_defs, diagnostic_feature_maps
from tanager_rocks.speclib import load_library, select_endmembers
from tanager_rocks.unmix import mtmf
from tanager_rocks.viz import emit_comparison_panel, setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("compare_emit")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
EMIT_DIR = RAW_DIR / "emit"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
OUT_DIR = ROOT / "data" / "intermediate" / "emit"
FIGURES_DIR = ROOT / "figures"

# Minerals correlated across sensors; alunite is the panel headline (it validated
# at Goldfield/Cuprite as the advanced-argillic signature).
COMPARE_MINERALS = ("alunite", "kaolinite", "muscovite", "jarosite", "hematite", "goethite")
HEADLINE_MINERAL = "alunite"


def _ensure_emit_granule(bbox: list[float]) -> Path:
    """Return a local EMIT RFL path for ``bbox``, downloading the best scene once."""
    import earthaccess

    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name=EMIT_L2A_SHORT_NAME, bounding_box=tuple(bbox), count=100
    )
    ranked = rank_granules(results, box(*bbox))
    chosen = select_granule(ranked)
    dest = rfl_path(EMIT_DIR, chosen.granule_ur)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("EMIT reflectance already present: %s", dest.name)
        return dest
    EMIT_DIR.mkdir(parents=True, exist_ok=True)
    rfl_links = [u for u in chosen.data_links if u.endswith(".nc") and "_RFLUNCERT_" not in u]
    rfl_only = [u for u in rfl_links if "_RFL_" in u and "_MASK_" not in u]
    logger.info("downloading EMIT reflectance for %s", chosen.granule_ur)
    earthaccess.download(rfl_only, str(EMIT_DIR))
    return dest


def _map_sensor(cube, wl):
    """Diagnostic band depths + MTMF for one sensor (the shared pipeline).

    The masking/MTMF steps build fresh objects that drop the rio CRS, so it is
    written back from the input cube — the maps must stay georeferenced for the
    cross-sensor reprojection.
    """
    crs = cube.rio.crs
    cube = mask_absorption_bands(cube, wl)
    depths = diagnostic_feature_maps(cube, wl, build_feature_defs(wl, SPECLIB_DIR))
    ds = mtmf(cube, select_endmembers(load_library(SPECLIB_DIR, wl)))
    minerals = [v[:-3] for v in ds.data_vars if v.endswith("_mf")]
    mf = xr.Dataset({m: ds[f"{m}_mf"] for m in minerals}).rio.write_crs(crs).rio.write_transform()
    return cube.rio.write_crs(crs), depths.rio.write_crs(crs), mf


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="goldfield", choices=tuple(SITES))
    args = parser.parse_args(argv)
    site = SITES[args.site]
    scene_id = site.scene_ids[0]

    setup_style()
    tan_cube, tan_wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    bbox = footprint_bbox(tan_cube)

    emit_path = _ensure_emit_granule(bbox)
    emit_cube_raw, emit_wl = load_emit_reflectance(emit_path, bbox=bbox)

    tan_masked, _, tan_mf = _map_sensor(tan_cube, tan_wl)
    emit_masked, _, emit_mf = _map_sensor(emit_cube_raw, emit_wl)

    spec, common_nm, tan_mean, emit_mean = spectral_agreement(
        tan_masked, tan_wl, emit_masked, emit_wl
    )
    logger.info(
        "spectral agreement (scene-mean): Pearson r=%.3f, angle=%.2f deg, n_bands=%d",
        spec.pearson_r,
        spec.spectral_angle_deg,
        spec.n_bands,
    )
    detect = detection_agreement(tan_mf, emit_mf, list(COMPARE_MINERALS))
    logger.info("--- per-mineral MTMF detection agreement (Tanager reprojected to EMIT) ---")
    for m, d in detect.items():
        logger.info("%-10s detection r=%+.3f  n=%d", m, d.pearson_r, d.n_pixels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"emit_comparison_{args.site}_{scene_id}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "mineral", "value", "n"])
        w.writerow(["spectral_pearson_r", "", f"{spec.pearson_r:.4f}", spec.n_bands])
        w.writerow(["spectral_angle_deg", "", f"{spec.spectral_angle_deg:.4f}", spec.n_bands])
        for m, d in detect.items():
            w.writerow(["detection_pearson_r", m, f"{d.pearson_r:.4f}", d.n_pixels])

    head = detect.get(HEADLINE_MINERAL)
    # Put the headline Tanager map in EMIT's geographic CRS at ~30 m so the two
    # maps share an extent for side-by-side display while Tanager keeps its grain.
    emit_head = emit_mf[HEADLINE_MINERAL]
    tan_head = reproject_crs(
        tan_mf[HEADLINE_MINERAL], emit_head.rio.crs, resolution=0.00027
    ).rio.clip_box(*emit_head.rio.bounds())
    fig = emit_comparison_panel(
        common_nm,
        tan_mean,
        emit_mean,
        spec.pearson_r,
        spec.spectral_angle_deg,
        tan_head,
        emit_head,
        HEADLINE_MINERAL,
        head.pearson_r if head else float("nan"),
        title=f"Tanager (30 m) vs EMIT (60 m) — {site.name}",
    )
    fig_path = FIGURES_DIR / f"{args.site}_{scene_id}_emit_comparison.png"
    fig.savefig(fig_path)
    logger.info("wrote %s and %s", csv_path.name, fig_path.name)


if __name__ == "__main__":
    main()
