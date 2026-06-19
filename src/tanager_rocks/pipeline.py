"""Pipeline orchestration for the tanager-rocks mineral-mapping stages.

Each ``run_*`` function executes one ``spec.md`` pipeline stage end-to-end for a
site — load the SR cube, mask absorption bands, compute, and write the
GeoTIFF / PNG / CSV products — parameterised by a :class:`PipelinePaths` so the
*same* logic backs both the repo-relative dev scripts (``scripts/*_site.py``)
and the installed ``tanager-minmap`` CLI. The analysis lives in the
feature / unmix / hazard / degrade / viz modules; this module only sequences
those calls and handles I/O, so the pipeline ships in the wheel rather than in
``scripts/`` (which is not packaged).

This module covers the offline stages (a local Tanager scene + the splib07
library, no network): ``map``, ``unmix``, ``ablate``, ``amd``, ``hero``. The
EMIT cross-sensor comparison and the USGS-map validation live in their own
drivers because they need network access / a reference-map download.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import mask_absorption_bands
from tanager_spec.srf import load_s2_srf

from .config import TANAGER_SR_ASSET, SiteSpec
from .degrade import degrade_endmembers, separability, srf_band_stats
from .features import build_feature_defs, diagnostic_feature_maps
from .hazard import AGP_LABELS, acid_generating_potential
from .speclib import load_library, select_endmembers
from .unmix import mtmf, sam_classify, spectral_angle
from .viz import (
    amd_map,
    band_ablation_panel,
    band_depth_panel,
    classification_map,
    mineral_map,
    score_panel,
    setup_style,
)

logger = logging.getLogger(__name__)

# Int16 tier-raster sentinel for off-domain (NaN) AMD pixels.
TIER_NODATA = -1

# Alteration-relevant mineral contrasts for band-ablation. alunite-kaolinite is
# the headline (advanced argillic vs argillic — the discrimination S2 cannot
# make); the rest show the loss is specific to the SWIR Al-OH region (the
# VNIR-driven jarosite-goethite contrast survives degradation).
ABLATION_PAIRS: list[tuple[str, str]] = [
    ("alunite", "kaolinite"),
    ("alunite", "muscovite"),
    ("kaolinite", "muscovite"),
    ("kaolinite", "dickite"),
    ("jarosite", "goethite"),
]
ABLATION_HEADLINE = ("alunite", "kaolinite")


@dataclass(frozen=True)
class PipelinePaths:
    """Input and output directories for a pipeline run.

    The dev scripts use the repo layout (:meth:`repo_default`); the CLI builds
    its directories from ``--data-root`` / ``--output`` (:meth:`from_cli`).
    """

    raw_dir: Path  # Tanager SR scenes (<scene>_<asset>.h5)
    speclib_dir: Path  # extracted splib07a ASCIIdata directory
    maps_dir: Path  # GeoTIFF outputs
    figures_dir: Path  # PNG outputs
    tables_dir: Path  # CSV outputs (band-ablation angles)

    @classmethod
    def repo_default(cls, root: Path) -> PipelinePaths:
        """The historical repo layout used by ``scripts/*_site.py``."""
        data = root / "data"
        return cls(
            raw_dir=data / "raw",
            speclib_dir=data / "speclib" / "ASCIIdata_splib07a",
            maps_dir=data / "intermediate" / "maps",
            figures_dir=root / "figures",
            tables_dir=data / "intermediate" / "ablation",
        )

    @classmethod
    def from_cli(cls, data_root: Path, output: Path) -> PipelinePaths:
        """CLI layout: inputs under ``data_root``, all products under ``output``."""
        return cls(
            raw_dir=data_root / "raw",
            speclib_dir=data_root / "speclib" / "ASCIIdata_splib07a",
            maps_dir=output / "maps",
            figures_dir=output / "figures",
            tables_dir=output / "tables",
        )

    def ensure_outputs(self) -> None:
        for directory in (self.maps_dir, self.figures_dir, self.tables_dir):
            directory.mkdir(parents=True, exist_ok=True)


def _scene_path(site: SiteSpec, paths: PipelinePaths) -> Path:
    """Path to the site's lead-scene SR HDF5."""
    return paths.raw_dir / f"{site.scene_ids[0]}_{TANAGER_SR_ASSET}.h5"


def _load_masked_cube(site: SiteSpec, paths: PipelinePaths) -> tuple[xr.DataArray, np.ndarray]:
    """Load the lead scene's SR cube and mask the O2/H2O absorption bands."""
    cube, wl = load_tanager_sr_hdf5(_scene_path(site, paths))
    return mask_absorption_bands(cube, wl), wl


def _endmembers(wl: np.ndarray, paths: PipelinePaths):
    """One medoid splib07 endmember per target mineral, resampled to ``wl``."""
    return select_endmembers(load_library(paths.speclib_dir, wl))


def _write_raster(da: xr.DataArray, crs, transform, path: Path) -> None:
    """Write a 2-D array as a GeoTIFF with the cube's CRS and transform."""
    da.rio.write_crs(crs).rio.write_transform(transform).rio.to_raster(path)


def run_map(site: SiteSpec, paths: PipelinePaths) -> Path:
    """Diagnostic continuum-removed band-depth maps (spec steps 2-3)."""
    paths.ensure_outputs()
    setup_style()
    cube, wl = _load_masked_cube(site, paths)
    scene_id = site.scene_ids[0]
    depths = diagnostic_feature_maps(cube, wl, build_feature_defs(wl, paths.speclib_dir))

    crs, transform = cube.rio.crs, cube.rio.transform()
    for name in depths.data_vars:
        _write_raster(
            depths[name], crs, transform, paths.maps_dir / f"{site.site_id}_{scene_id}_{name}.tif"
        )

    fig = band_depth_panel(depths, title=f"{site.name} ({scene_id}) — continuum-removed band depth")
    out = paths.figures_dir / f"{site.site_id}_{scene_id}_band_depth.png"
    fig.savefig(out)
    logger.info("wrote %s", out)
    return out


def run_unmix(
    site: SiteSpec,
    paths: PipelinePaths,
    *,
    max_angle: float = 0.15,
    max_infeas: float = 1.0,
) -> Path:
    """SAM baseline + MTMF abundance/infeasibility (spec step 4).

    ``max_angle`` is the SAM acceptance threshold (radians); ``max_infeas`` gates
    MTMF abundance to spectrally feasible pixels. Both are distribution-informed
    defaults, not ground-truth-calibrated.
    """
    paths.ensure_outputs()
    setup_style()
    cube, wl = _load_masked_cube(site, paths)
    scene_id = site.scene_ids[0]
    endmembers = _endmembers(wl, paths)
    crs, transform = cube.rio.crs, cube.rio.transform()

    # SAM baseline.
    angles = spectral_angle(cube, endmembers)
    classes, labels = sam_classify(angles, max_angle_rad=max_angle)
    counts = {labels[i]: int((classes.values == i).sum()) for i in range(len(labels))}
    counts["unclassified"] = int((classes.values == -1).sum())
    logger.info("SAM class counts (max_angle=%.3f rad): %s", max_angle, counts)
    classes.rio.write_crs(crs).rio.write_transform(transform).astype("int16").rio.to_raster(
        paths.maps_dir / f"{site.site_id}_{scene_id}_sam_class.tif"
    )
    classification_map(
        classes, labels, title=f"{site.name} ({scene_id}) — SAM classification"
    ).savefig(paths.figures_dir / f"{site.site_id}_{scene_id}_sam_class.png")

    # MTMF: matched-filter abundance + mixture-tuned infeasibility.
    ds = mtmf(cube, endmembers)
    minerals = [v[:-3] for v in ds.data_vars if str(v).endswith("_mf")]
    mf = xr.Dataset({m: ds[f"{m}_mf"] for m in minerals})
    infeas = xr.Dataset({m: ds[f"{m}_infeas"] for m in minerals})
    gated = xr.Dataset({m: ds[f"{m}_mf"].where(ds[f"{m}_infeas"] < max_infeas) for m in minerals})
    for mineral in minerals:
        for kind, da in (("mf", ds[f"{mineral}_mf"]), ("infeas", ds[f"{mineral}_infeas"])):
            tif = paths.maps_dir / f"{site.site_id}_{scene_id}_{kind}_{mineral}.tif"
            _write_raster(da, crs, transform, tif)

    base = f"{site.name} ({scene_id})"
    score_panel(mf, f"{base} — matched-filter abundance", cbar_label="MF score").savefig(
        paths.figures_dir / f"{site.site_id}_{scene_id}_mf.png"
    )
    score_panel(infeas, f"{base} — MTMF infeasibility", cbar_label="infeasibility").savefig(
        paths.figures_dir / f"{site.site_id}_{scene_id}_infeas.png"
    )
    out = paths.figures_dir / f"{site.site_id}_{scene_id}_mtmf_gated.png"
    score_panel(
        gated, f"{base} — MTMF abundance (infeas < {max_infeas})", cbar_label="MF score"
    ).savefig(out)
    logger.info("wrote %s", out)
    return out


def run_ablate(site: SiteSpec, paths: PipelinePaths) -> Path:
    """Sentinel-2 band-ablation separability comparison (spec step 5)."""
    paths.ensure_outputs()
    setup_style()
    _, wl = load_tanager_sr_hdf5(_scene_path(site, paths))  # only the wavelength axis is needed
    scene_id = site.scene_ids[0]
    endmembers = _endmembers(wl, paths)
    srf = load_s2_srf()
    centers, fwhm = srf_band_stats(srf)

    sep = separability(endmembers, wl, srf, ABLATION_PAIRS)
    logger.info("--- alunite/kaolinite-type separability (spectral angle, deg) ---")
    for (a, b), (full, coarse) in sep.items():
        loss = 100 * (1 - coarse / full) if full else float("nan")
        logger.info(
            "%-22s Tanager %5.2f  S2 %5.2f  (%+.0f%%)",
            f"{a}-{b}",
            np.degrees(full),
            np.degrees(coarse),
            loss,
        )

    with open(paths.tables_dir / f"ablation_{site.site_id}_{scene_id}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair", "tanager_angle_deg", "s2_angle_deg", "loss_pct"])
        for (a, b), (full, coarse) in sep.items():
            writer.writerow(
                [
                    f"{a}-{b}",
                    f"{np.degrees(full):.3f}",
                    f"{np.degrees(coarse):.3f}",
                    f"{100 * (1 - coarse / full):.1f}",
                ]
            )

    degraded = degrade_endmembers(endmembers, wl, srf)
    full_deg, s2_deg = (np.degrees(x) for x in sep[ABLATION_HEADLINE])
    fig = band_ablation_panel(
        np.asarray(wl, float),
        {m: endmembers[m].reflectance for m in ABLATION_HEADLINE},
        {m: degraded[m] for m in ABLATION_HEADLINE},
        centers,
        fwhm,
        full_deg,
        s2_deg,
        minerals=ABLATION_HEADLINE,
        title=f"{site.name}: Tanager vs Sentinel-2 — Al-OH doublet",
    )
    out = paths.figures_dir / f"{site.site_id}_{scene_id}_band_ablation.png"
    fig.savefig(out)
    logger.info("wrote %s", out)
    return out


def run_amd(
    site: SiteSpec,
    paths: PipelinePaths,
    *,
    max_infeas: float = 1.0,
    quantile: float = 0.90,
) -> Path:
    """Ordinal acid-generating-potential proxy (spec step 7)."""
    paths.ensure_outputs()
    setup_style()
    cube, wl = _load_masked_cube(site, paths)
    scene_id = site.scene_ids[0]
    crs, transform = cube.rio.crs, cube.rio.transform()

    ds = mtmf(cube, _endmembers(wl, paths))
    result = acid_generating_potential(ds, max_infeas=max_infeas, quantile=quantile)

    tier_int = np.where(np.isfinite(result.tiers.values), result.tiers.values, TIER_NODATA)
    raster = result.tiers.copy(data=tier_int.astype("int16"))
    raster.rio.write_crs(crs).rio.write_transform(transform).rio.write_nodata(
        TIER_NODATA
    ).rio.to_raster(paths.maps_dir / f"{site.site_id}_{scene_id}_amd_agp.tif")

    fig = amd_map(
        result.tiers,
        title=f"{site.name} — acid-generating-potential proxy (Tanager MTMF assemblage)",
        labels=AGP_LABELS,
    )
    out = paths.figures_dir / f"{site.site_id}_{scene_id}_amd_agp.png"
    fig.savefig(out)
    logger.info(
        "wrote %s — %d in-scene px, tiers %s",
        out,
        int(result.domain.sum()),
        {AGP_LABELS[c]: result.counts[c] for c in result.counts},
    )
    return out


def run_hero(
    site: SiteSpec,
    paths: PipelinePaths,
    *,
    max_infeas: float = 1.0,
    quantile: float = 0.90,
) -> Path:
    """Dominant-mineral hero map (spec step 9)."""
    paths.ensure_outputs()
    setup_style()
    cube, wl = _load_masked_cube(site, paths)
    scene_id = site.scene_ids[0]
    ds = mtmf(cube, _endmembers(wl, paths))
    minerals = [v[:-3] for v in ds.data_vars if str(v).endswith("_mf")]
    gated = xr.Dataset({m: ds[f"{m}_mf"].where(ds[f"{m}_infeas"] < max_infeas) for m in minerals})

    fig = mineral_map(
        gated,
        title=f"{site.name} — dominant alteration mineral (Tanager MTMF)",
        per_mineral_quantile=quantile,
    )
    out = paths.figures_dir / f"{site.site_id}_{scene_id}_hero_mineral_map.png"
    fig.savefig(out)
    logger.info("wrote %s", out)
    return out
