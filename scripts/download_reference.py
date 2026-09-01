"""Acquire the USGS Rockwell ASTER alteration map and clip it to a site.

Validation reference for spec.md step 4-5. Downloads the western-US ASTER
hydrothermal-alteration raster (Rockwell & Bonham 2017, USGS data release
doi:10.5066/F7CR5RK7, public domain) and clips/reprojects it onto a site's lead
Tanager scene grid as a small categorical GeoTIFF that the validation reads.

The download URLs are resolved from the ScienceBase item API at run time (never
hand-built), the same pattern as ``download_scenes.py``. The full mosaic is an
ERDAS Imagine pair (``.img`` header + attribute table, ``.ige`` raster); both
are needed for GDAL to read pixel values. The raster is reprojected with
nearest-neighbour resampling to preserve the integer class codes.

Run::

    uv run python scripts/download_reference.py --site goldfield
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import rioxarray  # noqa: F401  (registers the .rio accessor)
from tanager_spec.io import load_tanager_sr_hdf5

from tanager_minmap.config import SITES, TANAGER_SR_ASSET, site_search_bbox
from tanager_minmap.reference import align_reference

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("download_reference")

USER_AGENT = "tanager-minmap/0.1 (research; abradley@wustl.edu)"
# Parent collection of the western-US ASTER alteration data release (doi
# 10.5066/F7CR5RK7). The southwestern-US child tile (build AA61, v8) spans the
# whole product extent and covers both study sites.
SCIENCEBASE_ITEM = "58cc1f95e4b0849ce97dce60"
SW_TILE_TOKEN = "southwest"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "reference" / "raw"
OUT_DIR = ROOT / "data" / "reference"
# Buffer (deg) added around the scene footprint before clipping the reference,
# so the reprojected grid has margin at the edges.
CLIP_BUFFER_DEG = 0.05


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def resolve_tile_files() -> dict[str, str]:
    """Return ``{ '.img': url, '.ige': url }`` for the southwestern tile."""
    api = (
        f"https://www.sciencebase.gov/catalog/items?parentId={SCIENCEBASE_ITEM}"
        "&format=json&max=30&fields=title,files"
    )
    items = json.loads(_get(api)).get("items", [])
    for it in items:
        if SW_TILE_TOKEN not in (it.get("title") or "").lower():
            continue
        urls: dict[str, str] = {}
        for f in it.get("files", []) or []:
            name = f.get("name", "")
            for ext in (".img", ".ige"):
                if name.endswith(ext):
                    urls[ext] = f["downloadUri"]
        if {".img", ".ige"} <= urls.keys():
            return urls
    raise RuntimeError("southwestern ASTER tile (.img + .ige) not found in ScienceBase item")


def download_raw() -> Path:
    """Download the .img + .ige pair if absent or size-mismatched; return the .img path.

    Streams to a ``.part`` file and atomically replaces it, and skips only a file
    whose size already matches the server's ``Content-Length`` — so an interrupted
    download never leaves a truncated file at the canonical name (the ``.ige`` is
    ~3 GB). Same pattern as ``download_speclib.py``.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urls = resolve_tile_files()
    img_path = RAW_DIR / "aster_southwest_aa61_v8_1-17-17.img"
    ige_path = img_path.with_suffix(".ige")
    for ext, dest in ((".img", img_path), (".ige", ige_path)):
        req = urllib.request.Request(urls[ext], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=600) as resp:
            remote_size = int(resp.headers.get("Content-Length", -1))
            if dest.exists() and dest.stat().st_size == remote_size:
                logger.info("have %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
                continue
            logger.info("downloading %s", dest.name)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
        tmp.replace(dest)
        logger.info("wrote %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return img_path


def clip_to_site(img_path: Path, site_id: str) -> Path:
    """Clip + reproject the reference onto the site's lead-scene grid."""
    site = SITES[site_id]
    scene_id = site.scene_ids[0]
    cube, _ = load_tanager_sr_hdf5(ROOT / "data" / "raw" / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    like = cube.isel(band=0)

    reference = rioxarray.open_rasterio(img_path, masked=False).squeeze("band", drop=True)
    lon0, lat0, lon1, lat1 = site_search_bbox(site, buffer_deg=CLIP_BUFFER_DEG)
    # Windowed read in the reference CRS (EPSG:4326), then match the cube grid.
    clipped = reference.rio.clip_box(lon0, lat0, lon1, lat1)
    aligned = align_reference(clipped, like)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"rockwell_{site_id}_{scene_id}.tif"
    aligned = aligned.astype("uint8")
    aligned.rio.write_crs(like.rio.crs).rio.write_transform(like.rio.transform())
    aligned.rio.to_raster(out_path)
    logger.info("wrote %s (%s)", out_path, dict(zip(("y", "x"), aligned.shape, strict=False)))
    return out_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="goldfield", choices=tuple(SITES))
    args = parser.parse_args(argv)
    img_path = download_raw()
    clip_to_site(img_path, args.site)


if __name__ == "__main__":
    main()
