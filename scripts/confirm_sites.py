"""Week-1 gate: confirm the Bingham and Goldfield study sites against the data.

The Planet Tanager open data is a static STAC catalog with no search endpoint,
so it is walked once into a full scene inventory; each site is then matched
locally by footprint intersection. For each site this reports the matching
scenes, their category, cloud cover, and whether they carry the
``ortho_sr_hdf5`` asset, then queries the NASA LP DAAC STAC for overlapping
EMIT L2A granules.

It asserts nothing about site *identity* — confirming that a scene really is
Bingham or Goldfield (vs USGS USMIN/MRDS and a basemap) is a human judgement
made from this output and the saved footprints.

Outputs (all under ``data/intermediate/``):

- ``tanager_scene_inventory.csv`` — the full catalog inventory.
- ``scenes_<site>.csv`` — the scenes matched to each site.

Run::

    uv run python scripts/confirm_sites.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from tanager_spec.config import TANAGER_SR_ASSET
from tanager_spec.stac import (
    build_scene_inventory,
    query_emit_scenes,
    query_tanager_scenes,
    save_inventory,
)

from tanager_rocks.config import SITES, SiteSpec

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("confirm_sites")

# Search-box half-width (degrees) around the site centroids. The spec records
# these as *scene centroids*, so the footprints extend well beyond them; a
# modest buffer catches the overlapping scenes without pulling in unrelated
# districts. ~0.15 deg ~= 12-17 km at these latitudes.
BUFFER_DEG = 0.15

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "intermediate"


def site_bbox(site: SiteSpec, buffer_deg: float = BUFFER_DEG) -> list[float]:
    """Build a WGS84 ``[xmin, ymin, xmax, ymax]`` search box around a site."""
    lats = [lat for lat, _ in site.centroids]
    lons = [lon for _, lon in site.centroids]
    return [
        min(lons) - buffer_deg,
        min(lats) - buffer_deg,
        max(lons) + buffer_deg,
        max(lats) + buffer_deg,
    ]


def report_site(site: SiteSpec, inventory: gpd.GeoDataFrame) -> None:
    """Match one site against the full inventory + EMIT, then print/save."""
    bbox = site_bbox(site)
    search_geom = box(*bbox)
    matched = inventory[inventory.geometry.intersects(search_geom)]

    # Scenes are cross-listed under multiple catalog categories, so a scene_id
    # appears once per category; collapse to the unique scene (the real unit).
    unique = matched.drop_duplicates(subset="scene_id").sort_values("scene_id")

    print(f"\n--- {site.site_id}: {site.name} ({site.role}) ---")
    print(f"search bbox {bbox}")
    print(f"spec expects {site.n_scenes} scene(s); {len(unique)} unique scene(s) intersect")
    for _, row in unique.iterrows():
        cats = sorted(
            matched.loc[matched["scene_id"] == row["scene_id"], "category"].dropna().unique()
        )
        has_sr = TANAGER_SR_ASSET in (row["assets"] or {})
        print(
            f"  {row['scene_id']}  cats={cats}  "
            f"date={row['datetime']}  cloud={row['cloud_percent']}  "
            f"{TANAGER_SR_ASSET}={'yes' if has_sr else 'NO'}"
        )

    save_inventory(unique, OUTPUT_DIR / f"scenes_{site.site_id}.csv")

    emit = query_emit_scenes(bbox=bbox)
    print(f"  EMIT L2A granules overlapping bbox: {len(emit)}")
    for g in emit[:5]:
        print(f"    {g.get('id')}  {g.get('properties', {}).get('datetime')}")


def main() -> None:
    logger.info("walking the Tanager static catalog (all categories)...")
    items = query_tanager_scenes()
    inventory = build_scene_inventory(items)
    save_inventory(inventory, OUTPUT_DIR / "tanager_scene_inventory.csv")
    print(f"full catalog inventory: {len(inventory)} scenes")

    for site in SITES.values():
        report_site(site, inventory)


if __name__ == "__main__":
    main()
