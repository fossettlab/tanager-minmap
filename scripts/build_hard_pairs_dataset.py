"""Build the local hard-pairs eval-only probe dataset (chips + manifests + card).

Extends the hard-pair mining (``scripts/find_hard_pairs.py``) into a small,
standalone, evaluation-only dataset: reads the existing
``data/processed/hard_pairs/{patches,pairs,summary}`` outputs (no re-mining,
no re-derivation of labels or RGB/SWIR thresholds, no re-running MTMF) and
exports:

- one full-spectral GeoTIFF chip per labeled patch (``chips/<scene_id>/<patch_id>.tif``)
- ``patches.csv`` -- the full labeled-patch manifest, self-contained
- ``pairs.csv`` -- the 29 SWIR-separable hard pairs, joined to chip patch_ids
- ``clusters.csv`` -- connected components of the RGB-ambiguity graph
  spanning >=2 labels (the blog's "hard clusters" analog), re-derived
  deterministically from ``patches.csv``'s own RGB statistics -- no cube
  reload needed for this step
- ``wavelengths.csv`` -- per-scene band-center wavelengths (the two scenes'
  axes differ by up to 0.22 nm; recorded separately, never assumed shared)
- ``DATASET_CARD.md`` -- written by hand alongside this script (not
  regenerated here); this script's docstring and METHODS.md both point to it

This is a LOCAL build only. Publishing to Hugging Face Hub (or anywhere else)
is a separate, explicit operator decision -- this script never uploads
anything. After building, it round-trip-verifies ONE seeded-random chip
against its source cube window (band count, CRS, and exact pixel values --
no resampling tolerance) so a georeferencing or slicing regression would be
caught immediately rather than discovered downstream.

Run::

    uv run python scripts/build_hard_pairs_dataset.py
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import pyproj
import rioxarray
from tanager_spec.io import load_tanager_sr_hdf5

from tanager_rocks.config import SEED, SITES, TANAGER_SR_ASSET
from tanager_rocks.pairs import (
    Patch,
    rgb_ambiguity_clusters,
    rgb_ambiguous_pairs,
    write_chip_geotiff,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
HARD_PAIRS_DIR = ROOT / "data" / "processed" / "hard_pairs"
OUT_DIR = ROOT / "data" / "processed" / "hard_pairs_dataset"
CHIPS_DIR = OUT_DIR / "chips"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _patch_id(site_id: str, row: int, col: int) -> str:
    return f"{site_id}_r{row}_c{col}"


def _patch_from_row(r: dict[str, str]) -> Patch:
    """Reconstruct a Patch from patches.csv for RGB-graph re-derivation.

    ``swir_mean`` is not stored in patches.csv (it is large and only used
    transiently during mining); it is not needed here -- clustering and the
    RGB-ambiguity thresholds are computed from ``rgb_mean``/``rgb_std``/
    ``label`` alone, so an empty placeholder is safe.
    """
    return Patch(
        site_id=r["site_id"],
        scene_id=r["scene_id"],
        row=int(r["row"]),
        col=int(r["col"]),
        y0=int(r["y0"]),
        x0=int(r["x0"]),
        label=r["label"],
        purity=float(r["purity"]),
        rgb_mean=np.array([float(r["rgb_mean_r"]), float(r["rgb_mean_g"]), float(r["rgb_mean_b"])]),
        rgb_std=np.array([float(r["rgb_std_r"]), float(r["rgb_std_g"]), float(r["rgb_std_b"])]),
        swir_mean=np.array([]),
    )


def _centroid_lonlat(cube_raw, y0: int, x0: int, size: int, to_wgs84) -> tuple[float, float]:
    yc, xc = int(y0 + size // 2), int(x0 + size // 2)
    x_native = float(cube_raw.x.values[xc])
    y_native = float(cube_raw.y.values[yc])
    lon, lat = to_wgs84.transform(x_native, y_native)
    return lon, lat


def _write_wavelengths_csv(path: Path, wl_by_site: dict[str, np.ndarray]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["site_id", "band_index", "wavelength_nm"])
        for site_id, wl in wl_by_site.items():
            for i, w in enumerate(wl):
                writer.writerow([site_id, i, f"{w:.4f}"])


def _export_chips_and_manifest(
    patch_rows: list[dict[str, str]], patch_size: int
) -> tuple[list[dict[str, str]], dict[str, np.ndarray], dict[tuple[str, int, int], str]]:
    """Write every patch's GeoTIFF chip and build the self-contained manifest rows."""
    wgs84 = pyproj.CRS.from_epsg(4326)
    manifest_rows: list[dict[str, str]] = []
    wl_by_site: dict[str, np.ndarray] = {}
    id_lookup: dict[tuple[str, int, int], str] = {}

    by_site: dict[str, list[dict[str, str]]] = {}
    for row in patch_rows:
        by_site.setdefault(row["site_id"], []).append(row)

    for site_id, rows in by_site.items():
        site = SITES[site_id]
        scene_id = site.scene_ids[0]
        logger.info(
            "loading %s (%s) lead scene for chip export (%d patches)", site_id, scene_id, len(rows)
        )
        cube_raw, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
        wl_by_site[site_id] = wl
        to_wgs84 = pyproj.Transformer.from_crs(cube_raw.rio.crs, wgs84, always_xy=True)
        (CHIPS_DIR / scene_id).mkdir(parents=True, exist_ok=True)

        for r in rows:
            row_i, col_i, y0, x0 = int(r["row"]), int(r["col"]), int(r["y0"]), int(r["x0"])
            patch_id = _patch_id(site_id, row_i, col_i)
            id_lookup[(site_id, row_i, col_i)] = patch_id
            chip_rel = Path("chips") / scene_id / f"{patch_id}.tif"
            write_chip_geotiff(cube_raw, y0, x0, patch_size, OUT_DIR / chip_rel)
            lon, lat = _centroid_lonlat(cube_raw, y0, x0, patch_size, to_wgs84)

            manifest_rows.append(
                {
                    "patch_id": patch_id,
                    "chip_path": str(chip_rel),
                    "site_id": site_id,
                    "scene_id": scene_id,
                    "row": row_i,
                    "col": col_i,
                    "y0": y0,
                    "x0": x0,
                    "patch_size_px": patch_size,
                    "footprint_m": patch_size * 30.0,
                    "label": r["label"],
                    "purity": r["purity"],
                    "rgb_mean_r": r["rgb_mean_r"],
                    "rgb_mean_g": r["rgb_mean_g"],
                    "rgb_mean_b": r["rgb_mean_b"],
                    "rgb_std_r": r["rgb_std_r"],
                    "rgb_std_g": r["rgb_std_g"],
                    "rgb_std_b": r["rgb_std_b"],
                    "centroid_lon": f"{lon:.6f}",
                    "centroid_lat": f"{lat:.6f}",
                    "crs": str(cube_raw.rio.crs),
                    "n_bands": len(wl),
                }
            )
        del cube_raw  # free the ~1 GB cube before the next site

    return manifest_rows, wl_by_site, id_lookup


def _write_pairs_csv(
    path: Path, hard_pairs_rows: list[dict[str, str]], id_lookup: dict[tuple[str, int, int], str]
) -> int:
    out_rows = []
    for p in hard_pairs_rows:
        id_a = id_lookup[(p["site_a"], int(p["row_a"]), int(p["col_a"]))]
        id_b = id_lookup[(p["site_b"], int(p["row_b"]), int(p["col_b"]))]
        out_rows.append(
            {
                "rank": p["rank"],
                "patch_id_a": id_a,
                "label_a": p["label_a"],
                "patch_id_b": id_b,
                "label_b": p["label_b"],
                "rgb_mean_l2": p["rgb_mean_l2"],
                "rgb_std_l2": p["rgb_std_l2"],
                "swir_angle_deg": p["swir_angle_deg"],
            }
        )
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    return len(out_rows)


def _write_clusters_csv(path: Path, patches: list[Patch], candidates, id_lookup) -> list:
    clusters = rgb_ambiguity_clusters(patches, candidates)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cluster_id", "patch_id", "label", "cluster_size", "n_labels_in_cluster"])
        for c in clusters:
            for p in c.patches:
                patch_id = id_lookup[(p.site_id, p.row, p.col)]
                writer.writerow([c.cluster_id, patch_id, p.label, c.size, len(c.labels)])
    return clusters


def _verify_round_trip(row: dict[str, str]) -> bool:
    """Re-load the source scene and confirm one chip round-trips exactly.

    Checks band count, CRS, and bit-exact pixel values (NaN-aware) against
    the source cube's own window -- no resampling tolerance, since a raw
    slice-and-write should never introduce drift.
    """
    site = SITES[row["site_id"]]
    scene_id = site.scene_ids[0]
    cube_raw, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    y0, x0, size = int(row["y0"]), int(row["x0"]), int(row["patch_size_px"])
    expected = cube_raw.isel(y=slice(y0, y0 + size), x=slice(x0, x0 + size)).values

    written = rioxarray.open_rasterio(OUT_DIR / row["chip_path"])
    band_ok = written.shape[0] == len(wl) == expected.shape[0]
    crs_ok = str(written.rio.crs) == str(cube_raw.rio.crs)
    values_ok = bool(np.array_equal(written.values, expected, equal_nan=True))

    logger.info(
        "round-trip check on %s: bands %s (%d==%d==%d), crs %s, exact pixel match %s",
        row["patch_id"],
        band_ok,
        written.shape[0],
        len(wl),
        expected.shape[0],
        crs_ok,
        values_ok,
    )
    return band_ok and crs_ok and values_ok


def main() -> None:
    patch_rows = _read_csv(HARD_PAIRS_DIR / "patches.csv")
    hard_pairs_rows = _read_csv(HARD_PAIRS_DIR / "pairs.csv")
    if not patch_rows:
        raise RuntimeError(
            f"{HARD_PAIRS_DIR / 'patches.csv'} has no rows; run scripts/find_hard_pairs.py first"
        )
    with open(HARD_PAIRS_DIR / "summary.json") as fh:
        summary = json.load(fh)
    patch_size = int(summary["patch_size_px"])

    CHIPS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows, wl_by_site, id_lookup = _export_chips_and_manifest(patch_rows, patch_size)
    with open(OUT_DIR / "patches.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_pairs = _write_pairs_csv(OUT_DIR / "pairs.csv", hard_pairs_rows, id_lookup)

    # Re-derive the RGB-ambiguity candidate graph from patches.csv's own RGB
    # stats -- deterministic, no cube reload, no MTMF re-run. Cross-checked
    # against task 8's cached thresholds as a cheap consistency guard.
    patches = [_patch_from_row(r) for r in manifest_rows]
    rgb_result = rgb_ambiguous_pairs(patches, quantile=summary["rgb_quantile"])
    if not np.isclose(rgb_result.mean_threshold, summary["rgb_mean_threshold"], rtol=1e-6):
        raise AssertionError(
            f"re-derived RGB mean threshold {rgb_result.mean_threshold} != "
            f"cached {summary['rgb_mean_threshold']} -- patches.csv may be stale"
        )
    clusters = _write_clusters_csv(
        OUT_DIR / "clusters.csv", patches, rgb_result.candidates, id_lookup
    )

    _write_wavelengths_csv(OUT_DIR / "wavelengths.csv", wl_by_site)

    total_chip_bytes = sum(f.stat().st_size for f in CHIPS_DIR.rglob("*.tif"))
    logger.info(
        "wrote %d chips (%.1f MB), patches.csv (%d rows), pairs.csv (%d rows), "
        "clusters.csv (%d clusters, %d member rows), wavelengths.csv to %s",
        len(manifest_rows),
        total_chip_bytes / 1e6,
        len(manifest_rows),
        n_pairs,
        len(clusters),
        sum(c.size for c in clusters),
        OUT_DIR,
    )

    verify_row = random.Random(SEED).choice(manifest_rows)
    ok = _verify_round_trip(verify_row)
    if not ok:
        raise RuntimeError(f"round-trip verification FAILED for {verify_row['patch_id']}")
    logger.info("round-trip verification PASSED for %s", verify_row["patch_id"])


if __name__ == "__main__":
    main()
