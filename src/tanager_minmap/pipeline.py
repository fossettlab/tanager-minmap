"""Pipeline orchestration for the tanager-minmap mineral-mapping stages.

Each ``run_*`` function executes one ``spec.md`` pipeline stage end-to-end for a
site — load the SR cube, mask absorption bands, compute, and write the
GeoTIFF / PNG / CSV products — parameterised by a :class:`PipelinePaths` so the
*same* logic backs both the repo-relative dev scripts (``scripts/*_site.py``)
and the installed ``tanager-minmap`` CLI. The analysis lives in the
feature / unmix / hazard / degrade / viz modules; this module only sequences
those calls and handles I/O, so the pipeline ships in the wheel rather than in
``scripts/`` (which is not packaged).

It covers every pipeline stage: the offline ones (a local Tanager scene + the
splib07 library) — ``map``, ``unmix``, ``ablate``, ``amd``, ``hero`` — plus the
two that reach outside the repo: ``emit`` (downloads an overlapping EMIT L2A
granule via Earthdata) and ``validate`` (reads a pre-downloaded Rockwell ASTER
reference clip). The latter two need network / a reference download, so the
caller is responsible for credentials (Earthdata) and for fetching the
reference (``scripts/download_reference.py``) first.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.mask import mask_absorption_bands
from tanager_spec.srf import load_s2_srf

from .compare import detection_agreement, reproject_crs, spectral_agreement
from .config import TANAGER_SR_ASSET, SiteSpec
from .degrade import degrade_endmembers, separability, srf_band_stats
from .emit import (
    EMIT_L2A_SHORT_NAME,
    box,
    footprint_bbox,
    load_emit_reflectance,
    rank_granules,
    rfl_path,
    select_granule,
    validate_emit_reflectance_file,
)
from .features import build_feature_defs, diagnostic_feature_maps
from .hazard import AGP_LABELS, acid_generating_potential
from .quality import mask_tanager_scene
from .reference import FEATURE_TO_ROCKWELL, MINERAL_TO_ROCKWELL, align_reference
from .speclib import load_library, select_endmembers
from .unmix import mtmf, sam_classify, spectral_angle
from .validate import Discrimination, validate_scores
from .viz import (
    amd_map,
    band_ablation_panel,
    band_depth_panel,
    classification_map,
    emit_comparison_panel,
    mineral_map,
    score_panel,
    setup_style,
    zone_discrimination_panel,
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

# Minerals correlated across sensors in the EMIT comparison; alunite is the
# panel headline (it validated at Goldfield/Cuprite as the advanced-argillic
# signature). The headline Tanager map is reprojected to EMIT's geographic CRS
# at ~30 m (0.00027 deg) so both maps share an extent while Tanager keeps grain.
COMPARE_MINERALS = ("alunite", "kaolinite", "muscovite", "jarosite", "hematite", "goethite")
COMPARE_HEADLINE = "alunite"
COMPARE_TANAGER_DEG = 0.00027

# Exact granule selected by the documented overlap/cloud-ranking procedure for
# the current Goldfield comparison. Pinning the selected input lets a cached
# reproduction run without repeating a mutable catalog query. If it is absent,
# the authenticated query below must still recover this same granule.
EMIT_GRANULE_URS: dict[str, str] = {
    "goldfield": "EMIT_L2A_RFL_001_20230804T191650_2321613_007",
}


@dataclass(frozen=True)
class PipelinePaths:
    """Input and output directories for a pipeline run.

    The dev scripts use the repo layout (:meth:`repo_default`); the CLI builds
    its directories from ``--data-root`` / ``--output`` (:meth:`from_cli`).
    """

    raw_dir: Path  # Tanager SR scenes (<scene>_<asset>.h5)
    speclib_dir: Path  # extracted splib07a ASCIIdata directory
    reference_dir: Path  # Rockwell ASTER reference clips (validate)
    emit_dir: Path  # downloaded EMIT L2A reflectance (.nc)
    maps_dir: Path  # GeoTIFF outputs
    figures_dir: Path  # PNG outputs
    intermediate_dir: Path  # base for CSV outputs; per-stage subdirs underneath

    @classmethod
    def repo_default(cls, root: Path) -> PipelinePaths:
        """The historical repo layout used by ``scripts/*_site.py``."""
        data = root / "data"
        return cls(
            raw_dir=data / "raw",
            speclib_dir=data / "speclib" / "ASCIIdata_splib07a",
            reference_dir=data / "reference",
            emit_dir=data / "raw" / "emit",
            maps_dir=data / "intermediate" / "maps",
            figures_dir=root / "figures",
            intermediate_dir=data / "intermediate",
        )

    @classmethod
    def from_cli(cls, data_root: Path, output: Path) -> PipelinePaths:
        """CLI layout: inputs under ``data_root``, all products under ``output``."""
        return cls(
            raw_dir=data_root / "raw",
            speclib_dir=data_root / "speclib" / "ASCIIdata_splib07a",
            reference_dir=data_root / "reference",
            emit_dir=data_root / "raw" / "emit",
            maps_dir=output / "maps",
            figures_dir=output / "figures",
            intermediate_dir=output / "intermediate",
        )

    def ensure_outputs(self) -> None:
        for directory in (self.maps_dir, self.figures_dir, self.intermediate_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def stage_tables(self, stage: str) -> Path:
        """Per-stage CSV output dir (``intermediate_dir/<stage>``), created."""
        out = self.intermediate_dir / stage
        out.mkdir(parents=True, exist_ok=True)
        return out


def _scene_path(site: SiteSpec, paths: PipelinePaths) -> Path:
    """Path to the site's lead-scene SR HDF5."""
    path = paths.raw_dir / f"{site.scene_ids[0]}_{TANAGER_SR_ASSET}.h5"
    if not path.is_file():
        raise FileNotFoundError(
            f"Tanager scene {path} is missing — run "
            f"`uv run python scripts/download_scenes.py --site {site.site_id}` first"
        )
    return path


def _load_masked_cube(site: SiteSpec, paths: PipelinePaths) -> tuple[xr.DataArray, np.ndarray]:
    """Load the lead scene and apply the authoritative Tanager quality policy."""
    path = _scene_path(site, paths)
    cube, wl = load_tanager_sr_hdf5(path)
    masked, _ = mask_tanager_scene(cube, wl, path)
    return masked, wl


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

    csv_path = paths.stage_tables("ablation") / f"ablation_{site.site_id}_{scene_id}.csv"
    with open(csv_path, "w", newline="") as fh:
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

    counts_path = paths.stage_tables("amd") / f"amd_counts_{site.site_id}_{scene_id}.csv"
    _write_amd_counts_csv(
        counts_path,
        result.counts,
        in_scene_pixels=int(result.domain.sum()),
        max_infeas=max_infeas,
        quantile=quantile,
    )

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


def _write_amd_counts_csv(
    path: Path,
    counts: dict[int, int],
    *,
    in_scene_pixels: int,
    max_infeas: float,
    quantile: float,
) -> None:
    """Write the exact tier counts and analytical gates behind an AMD raster."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "tier_code",
                "tier_label",
                "pixel_count",
                "in_scene_pixels",
                "max_infeas",
                "detection_quantile",
            ]
        )
        for code in sorted(counts):
            writer.writerow(
                [
                    code,
                    AGP_LABELS[code],
                    counts[code],
                    in_scene_pixels,
                    max_infeas,
                    quantile,
                ]
            )


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


def _ensure_emit_granule(
    bbox: list[float],
    emit_dir: Path,
    *,
    expected_granule_ur: str | None = None,
) -> Path:
    """Return a local EMIT RFL path for ``bbox``, downloading the best scene once.

    Needs NASA Earthdata credentials in the environment
    (``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``); earthaccess is imported
    lazily so the rest of the pipeline does not depend on network access.
    """
    if expected_granule_ur is not None:
        expected = rfl_path(emit_dir, expected_granule_ur)
        if expected.exists():
            try:
                validate_emit_reflectance_file(expected)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"cached EMIT reflectance {expected} is incomplete or invalid; "
                    "move it aside and rerun the command to download a clean copy"
                ) from exc
            logger.info("using pinned cached EMIT reflectance: %s", expected.name)
            return expected

    has_password_login = bool(
        os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")
    )
    if not (os.environ.get("EARTHDATA_TOKEN") or has_password_login):
        raise RuntimeError(
            "EMIT cache is absent and Earthdata credentials are not configured; "
            "run under `doppler run --project mac --config dev -- ...` with "
            "EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN"
        )

    import earthaccess

    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name=EMIT_L2A_SHORT_NAME, bounding_box=tuple(bbox), count=100
    )
    chosen = select_granule(rank_granules(results, box(*bbox)))
    if expected_granule_ur is not None and chosen.granule_ur != expected_granule_ur:
        raise RuntimeError(
            "current EMIT catalog ranking selected "
            f"{chosen.granule_ur}, expected pinned {expected_granule_ur}"
        )
    dest = rfl_path(emit_dir, chosen.granule_ur)
    if dest.exists():
        try:
            validate_emit_reflectance_file(dest)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"cached EMIT reflectance {dest} is incomplete or invalid; "
                "move it aside and rerun the command to download a clean copy"
            ) from exc
        logger.info("EMIT reflectance already present: %s", dest.name)
        return dest
    emit_dir.mkdir(parents=True, exist_ok=True)
    rfl_links = [u for u in chosen.data_links if u.endswith(".nc") and "_RFLUNCERT_" not in u]
    rfl_only = [u for u in rfl_links if "_RFL_" in u and "_MASK_" not in u]
    if len(rfl_only) != 1:
        raise RuntimeError(
            f"expected exactly one EMIT L2A reflectance link for {chosen.granule_ur}; "
            f"found {len(rfl_only)}"
        )
    logger.info("downloading EMIT reflectance for %s", chosen.granule_ur)
    downloaded = earthaccess.download(rfl_only, str(emit_dir))
    if len(downloaded) != 1:
        raise RuntimeError(
            f"Earthdata returned {len(downloaded)} paths for one requested reflectance file"
        )
    actual = Path(downloaded[0])
    if not actual.is_absolute():
        actual = emit_dir / actual
    actual = validate_emit_reflectance_file(actual)
    logger.info("validated downloaded EMIT reflectance: %s", actual.name)
    return actual


def _map_for_compare(cube, wl, speclib_dir: Path):
    """Diagnostic band depths + MTMF for one sensor (the shared compare pipeline).

    Masking/MTMF build fresh objects that drop the rio CRS, so it is written back
    from the input cube — the maps must stay georeferenced for the cross-sensor
    reprojection.
    """
    crs = cube.rio.crs
    cube = mask_absorption_bands(cube, wl)
    ds = mtmf(cube, select_endmembers(load_library(speclib_dir, wl)))
    minerals = [v[:-3] for v in ds.data_vars if str(v).endswith("_mf")]
    mf = xr.Dataset({m: ds[f"{m}_mf"] for m in minerals}).rio.write_crs(crs).rio.write_transform()
    return cube.rio.write_crs(crs), mf


def run_emit(site: SiteSpec, paths: PipelinePaths) -> Path:
    """Tanager vs EMIT cross-sensor comparison (spec step 6).

    Runs the identical band-depth + MTMF pipeline on the site's Tanager lead
    scene and the clearest fully-overlapping EMIT L2A granule, and reports
    scene-mean spectral agreement + per-mineral detection agreement. Needs
    Earthdata credentials (see :func:`_ensure_emit_granule`); the granule is
    reused if already downloaded.
    """
    paths.ensure_outputs()
    setup_style()
    scene_id = site.scene_ids[0]
    tan_path = _scene_path(site, paths)
    tan_cube, tan_wl = load_tanager_sr_hdf5(tan_path)
    bbox = footprint_bbox(tan_cube)
    tan_cube, _ = mask_tanager_scene(tan_cube, tan_wl, tan_path)

    emit_path = _ensure_emit_granule(
        bbox,
        paths.emit_dir,
        expected_granule_ur=EMIT_GRANULE_URS.get(site.site_id),
    )
    emit_cube_raw, emit_wl = load_emit_reflectance(emit_path, bbox=bbox)
    tan_masked, tan_mf = _map_for_compare(tan_cube, tan_wl, paths.speclib_dir)
    emit_masked, emit_mf = _map_for_compare(emit_cube_raw, emit_wl, paths.speclib_dir)

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
    for mineral, d in detect.items():
        logger.info("%-10s detection r=%+.3f  n=%d", mineral, d.pearson_r, d.n_pixels)

    csv_path = paths.stage_tables("emit") / f"emit_comparison_{site.site_id}_{scene_id}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "mineral", "value", "n"])
        writer.writerow(["spectral_pearson_r", "", f"{spec.pearson_r:.4f}", spec.n_bands])
        writer.writerow(["spectral_angle_deg", "", f"{spec.spectral_angle_deg:.4f}", spec.n_bands])
        for mineral, d in detect.items():
            writer.writerow(["detection_pearson_r", mineral, f"{d.pearson_r:.4f}", d.n_pixels])

    head = detect.get(COMPARE_HEADLINE)
    emit_head = emit_mf[COMPARE_HEADLINE]
    tan_head = reproject_crs(
        tan_mf[COMPARE_HEADLINE], emit_head.rio.crs, resolution=COMPARE_TANAGER_DEG
    ).rio.clip_box(*emit_head.rio.bounds())
    fig = emit_comparison_panel(
        common_nm,
        tan_mean,
        emit_mean,
        spec.pearson_r,
        spec.spectral_angle_deg,
        tan_head,
        emit_head,
        COMPARE_HEADLINE,
        head.pearson_r if head else float("nan"),
        title=f"Tanager (30 m) vs EMIT (60 m) — {site.name}",
    )
    out = paths.figures_dir / f"{site.site_id}_{scene_id}_emit_comparison.png"
    fig.savefig(out)
    logger.info("wrote %s and %s", csv_path.name, out.name)
    return out


def _write_validation_csv(
    path: Path,
    feature_results: dict[str, Discrimination],
    mineral_results: dict[str, Discrimination],
) -> None:
    """Write the band-depth-feature and MTMF discriminations to one CSV."""
    header = [
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

    def _row(kind: str, layer: str, d: Discrimination) -> list:
        return [
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

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for layer, d in feature_results.items():
            writer.writerow(_row("feature", layer, d))
        for layer, d in mineral_results.items():
            writer.writerow(_row("mtmf", layer, d))


def _log_validation(results: dict[str, Discrimination], kind: str) -> None:
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


def run_validate(site: SiteSpec, paths: PipelinePaths) -> Path:
    """Validate band-depth + MTMF maps against the Rockwell ASTER reference (step 4b).

    Reads a pre-downloaded reference clip (``scripts/download_reference.py``),
    aligns it to the scene grid, and reports per-layer rank-AUC discrimination of
    each score against its published alteration zone(s). Raises if the clip is
    missing.
    """
    paths.ensure_outputs()
    scene_id = site.scene_ids[0]
    ref_path = paths.reference_dir / f"rockwell_{site.site_id}_{scene_id}.tif"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"reference clip {ref_path} missing — run "
            f"`python scripts/download_reference.py --site {site.site_id}` first"
        )

    cube, wl = _load_masked_cube(site, paths)
    reference = align_reference(
        rioxarray.open_rasterio(ref_path, masked=False).squeeze("band", drop=True),
        cube.isel(band=0),
    )

    depths = diagnostic_feature_maps(cube, wl, build_feature_defs(wl, paths.speclib_dir))
    ds = mtmf(cube, _endmembers(wl, paths))
    minerals = [v[:-3] for v in ds.data_vars if str(v).endswith("_mf")]
    mf = xr.Dataset({m: ds[f"{m}_mf"] for m in minerals})

    feature_results = validate_scores(depths, reference, FEATURE_TO_ROCKWELL)
    mineral_results = validate_scores(mf, reference, MINERAL_TO_ROCKWELL)
    _log_validation(feature_results, "band-depth feature")
    _log_validation(mineral_results, "MTMF abundance")

    csv_path = paths.stage_tables("validation") / f"validation_{site.site_id}_{scene_id}.csv"
    _write_validation_csv(csv_path, feature_results, mineral_results)

    setup_style()
    base = f"{site.name} ({scene_id})"
    zone_discrimination_panel(
        depths,
        reference,
        FEATURE_TO_ROCKWELL,
        feature_results,
        title=f"{base} — band depth by Rockwell zone",
    ).savefig(paths.figures_dir / f"{site.site_id}_{scene_id}_validation_features.png")
    out = paths.figures_dir / f"{site.site_id}_{scene_id}_validation_mtmf.png"
    zone_discrimination_panel(
        mf,
        reference,
        MINERAL_TO_ROCKWELL,
        mineral_results,
        title=f"{base} — MTMF abundance by Rockwell zone",
    ).savefig(out)
    logger.info("wrote %s and %s", csv_path.name, out.name)
    return out
