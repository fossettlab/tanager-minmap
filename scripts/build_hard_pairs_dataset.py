"""Build the local hard-pairs eval-only probe dataset (chips + manifests + card).

Extends the hard-pair mining (``scripts/find_hard_pairs.py``) into a small,
standalone, evaluation-only dataset: reads the existing
``data/processed/hard_pairs/{patches,pairs,summary}`` outputs (no re-mining,
no re-derivation of labels or RGB/SWIR thresholds, no re-running MTMF) and
exports:

- one full-spectral GeoTIFF chip per labeled patch (``chips/<scene_id>/<patch_id>.tif``)
- ``patches.csv`` -- the full labeled-patch manifest, self-contained
- ``pairs.csv`` -- the frozen SWIR-separable hard pairs, joined to chip patch_ids
- ``clusters.csv`` -- connected components of the RGB-ambiguity graph
  spanning >=2 labels (the blog's "hard clusters" analog), re-derived
  deterministically from ``patches.csv``'s own RGB statistics -- no cube
  reload needed for this step
- ``wavelengths.csv`` -- per-scene band-center wavelengths (the two scenes'
  axes differ by up to 0.22 nm; recorded separately, never assumed shared)
- ``chips.sha256`` -- one sorted SHA-256 entry for every and only every chip
  referenced by ``patches.csv``
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
    uv run python scripts/build_hard_pairs_dataset.py --check

The build is fail-closed: it writes a sibling staging directory, checks the
frozen row-count contract, exact manifest-to-chip set, and every chip digest,
then promotes the validated directory with same-filesystem renames. The
``--check`` path performs the same structural and checksum checks without
writing or deleting anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pyproj
import rioxarray
from tanager_spec.io import load_tanager_sr_hdf5

from tanager_rocks.config import SEED, SITES, TANAGER_SR_ASSET
from tanager_rocks.pairs import (
    Patch,
    promote_staged_dataset,
    rgb_ambiguity_clusters,
    rgb_ambiguous_pairs,
    validate_chip_dataset,
    write_chip_checksum_manifest,
    write_chip_geotiff,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
HARD_PAIRS_DIR = ROOT / "data" / "processed" / "hard_pairs"
OUT_DIR = ROOT / "data" / "processed" / "hard_pairs_dataset"
CHECKSUM_FILENAME = "chips.sha256"

# Frozen release contract from the governed upstream hard-pair selection.
# Updating these counts requires a separately approved scientific rebuild;
# this packaging script must never silently accept selection drift.
EXPECTED_N_PATCHES = 268
EXPECTED_N_PAIRS = 18
EXPECTED_N_CLUSTERS = 14
EXPECTED_N_CLUSTER_MEMBERSHIPS = 175


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
    patch_rows: list[dict[str, str]], patch_size: int, output_dir: Path
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
        (output_dir / "chips" / scene_id).mkdir(parents=True, exist_ok=True)

        for r in rows:
            row_i, col_i, y0, x0 = int(r["row"]), int(r["col"]), int(r["y0"]), int(r["x0"])
            patch_id = _patch_id(site_id, row_i, col_i)
            if (site_id, row_i, col_i) in id_lookup:
                raise RuntimeError(f"duplicate patch identity in upstream manifest: {patch_id}")
            id_lookup[(site_id, row_i, col_i)] = patch_id
            chip_rel = Path("chips") / scene_id / f"{patch_id}.tif"
            write_chip_geotiff(cube_raw, y0, x0, patch_size, output_dir / chip_rel)
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


def _verify_round_trip(row: dict[str, str], output_dir: Path) -> bool:
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

    written = rioxarray.open_rasterio(output_dir / row["chip_path"])
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


def _validate_contract(
    patch_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    cluster_rows: list[dict[str, str]],
) -> None:
    """Reject structural drift from the frozen release manifest contract."""
    cluster_ids = {row["cluster_id"] for row in cluster_rows}
    actual = {
        "patches": len(patch_rows),
        "pairs": len(pair_rows),
        "clusters": len(cluster_ids),
        "cluster_memberships": len(cluster_rows),
    }
    expected = {
        "patches": EXPECTED_N_PATCHES,
        "pairs": EXPECTED_N_PAIRS,
        "clusters": EXPECTED_N_CLUSTERS,
        "cluster_memberships": EXPECTED_N_CLUSTER_MEMBERSHIPS,
    }
    if actual != expected:
        raise RuntimeError(
            f"hard-pairs release contract drifted: expected={expected}, actual={actual}"
        )

    patch_ids = [row["patch_id"] for row in patch_rows]
    if len(set(patch_ids)) != len(patch_ids):
        raise RuntimeError("patches.csv contains duplicate patch_id values")
    known = set(patch_ids)
    for row in pair_rows:
        for field in ("patch_id_a", "patch_id_b"):
            if row[field] not in known:
                raise RuntimeError(f"pairs.csv references unknown {field}: {row[field]}")
    memberships: set[tuple[str, str]] = set()
    for row in cluster_rows:
        patch_id = row["patch_id"]
        if patch_id not in known:
            raise RuntimeError(f"clusters.csv references unknown patch_id: {patch_id}")
        membership = (row["cluster_id"], patch_id)
        if membership in memberships:
            raise RuntimeError(f"clusters.csv contains duplicate membership: {membership}")
        memberships.add(membership)


def _read_source_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    patch_rows = _read_csv(HARD_PAIRS_DIR / "patches.csv")
    hard_pairs_rows = _read_csv(HARD_PAIRS_DIR / "pairs.csv")
    if not patch_rows:
        raise RuntimeError(
            f"{HARD_PAIRS_DIR / 'patches.csv'} has no rows; run scripts/find_hard_pairs.py first"
        )
    with open(HARD_PAIRS_DIR / "summary.json") as fh:
        summary = json.load(fh)
    if len(patch_rows) != EXPECTED_N_PATCHES or len(hard_pairs_rows) != EXPECTED_N_PAIRS:
        raise RuntimeError(
            "governed hard-pair inputs do not match the frozen release contract: "
            f"patches={len(patch_rows)} (expected {EXPECTED_N_PATCHES}), "
            f"pairs={len(hard_pairs_rows)} (expected {EXPECTED_N_PAIRS})"
        )
    return patch_rows, hard_pairs_rows, summary


def _build_dataset() -> None:
    patch_rows, hard_pairs_rows, summary = _read_source_inputs()
    patch_size = int(summary["patch_size_px"])

    if not (OUT_DIR / "DATASET_CARD.md").is_file():
        raise RuntimeError(f"tracked dataset card is missing: {OUT_DIR / 'DATASET_CARD.md'}")
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(mkdtemp(prefix=f".{OUT_DIR.name}.staging-", dir=OUT_DIR.parent))
    try:
        shutil.copy2(OUT_DIR / "DATASET_CARD.md", staging_dir / "DATASET_CARD.md")
        manifest_rows, wl_by_site, id_lookup = _export_chips_and_manifest(
            patch_rows, patch_size, staging_dir
        )
        with open(staging_dir / "patches.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        _write_pairs_csv(staging_dir / "pairs.csv", hard_pairs_rows, id_lookup)

        # Re-derive only the already-governed RGB graph from the persisted
        # full-precision statistics. Thresholds and labels are unchanged.
        patches = [_patch_from_row(row) for row in manifest_rows]
        rgb_result = rgb_ambiguous_pairs(patches, quantile=summary["rgb_quantile"])
        if not np.isclose(rgb_result.mean_threshold, summary["rgb_mean_threshold"], rtol=1e-6):
            raise AssertionError(
                f"re-derived RGB mean threshold {rgb_result.mean_threshold} != "
                f"cached {summary['rgb_mean_threshold']} -- patches.csv may be stale"
            )
        _write_clusters_csv(staging_dir / "clusters.csv", patches, rgb_result.candidates, id_lookup)
        _write_wavelengths_csv(staging_dir / "wavelengths.csv", wl_by_site)

        staged_patches = _read_csv(staging_dir / "patches.csv")
        staged_pairs = _read_csv(staging_dir / "pairs.csv")
        staged_clusters = _read_csv(staging_dir / "clusters.csv")
        _validate_contract(staged_patches, staged_pairs, staged_clusters)
        write_chip_checksum_manifest(staging_dir, staged_patches, filename=CHECKSUM_FILENAME)
        report = validate_chip_dataset(
            staging_dir, staged_patches, checksum_filename=CHECKSUM_FILENAME
        )

        verify_row = random.Random(SEED).choice(staged_patches)
        if not _verify_round_trip(verify_row, staging_dir):
            raise RuntimeError(f"round-trip verification FAILED for {verify_row['patch_id']}")

        promote_staged_dataset(staging_dir, OUT_DIR)
        logger.info(
            "promoted validated dataset: %d chips (%.1f MB), %d pairs, "
            "%d clusters/%d memberships; chips.sha256 digest %s",
            report.n_chips,
            report.total_bytes / 1e6,
            len(staged_pairs),
            len({row["cluster_id"] for row in staged_clusters}),
            len(staged_clusters),
            report.checksum_manifest_sha256,
        )
        logger.info("round-trip verification PASSED for %s", verify_row["patch_id"])
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _check_dataset() -> None:
    patch_rows = _read_csv(OUT_DIR / "patches.csv")
    pair_rows = _read_csv(OUT_DIR / "pairs.csv")
    cluster_rows = _read_csv(OUT_DIR / "clusters.csv")
    _validate_contract(patch_rows, pair_rows, cluster_rows)
    report = validate_chip_dataset(OUT_DIR, patch_rows, checksum_filename=CHECKSUM_FILENAME)
    logger.info(
        "dataset check PASSED: %d manifest-bound chips (%.1f MB); chips.sha256 digest %s",
        report.n_chips,
        report.total_bytes / 1e6,
        report.checksum_manifest_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the existing dataset without writing or deleting files",
    )
    args = parser.parse_args()
    if args.check:
        _check_dataset()
    else:
        _build_dataset()


if __name__ == "__main__":
    main()
