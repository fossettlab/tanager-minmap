"""Week-1 gate: confirm site *identity* against the USGS MRDS.

`confirm_sites.py` shows that Tanager scenes (with the SR asset and EMIT
overlap) sit over the spec coordinates. This script closes the remaining
data-integrity step: it queries the USGS Mineral Resources Data System (MRDS)
WFS for mineral deposits inside each site's search box and checks that the
expected named, correctly-commoditied deposit is present — i.e. that the
footprint really is Bingham Canyon (a producing Cu porphyry) and Goldfield
(an Au district), not merely the right coordinates.

MRDS is queried live (mrdata.usgs.gov WFS, GML output read with geopandas).
A site PASSES when a developed deposit (Producer / Past Producer / Plant)
whose name contains the site keyword and whose commodity list includes the
expected element is found in the box.

Run::

    uv run python scripts/confirm_site_identity.py
"""

from __future__ import annotations

import logging
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd

from tanager_rocks.config import SITES, SiteSpec, site_search_bbox

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("confirm_site_identity")

MRDS_WFS = "https://mrdata.usgs.gov/wfs/mrds"
# mrdata.usgs.gov rejects the default urllib User-Agent (403); identify the client.
USER_AGENT = "tanager-rocks/0.1 (research; abradley@wustl.edu)"
# Development stages that denote a real, developed deposit (vs a bare occurrence).
DEVELOPED = {"Producer", "Past Producer", "Plant"}
# Per-site identity expectation: name keyword + the MRDS commodity code that
# defines the district (Bingham = Cu porphyry, Goldfield = Au epithermal).
IDENTITY = {
    "bingham": {"keyword": "Bingham", "commodity": "CU"},
    "goldfield": {"keyword": "Goldfield", "commodity": "AU"},
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "intermediate"


def fetch_mrds(bbox: list[float]) -> gpd.GeoDataFrame:
    """Fetch MRDS deposits within a WGS84 bbox via the USGS WFS.

    Parameters
    ----------
    bbox : list of float
        ``[lon_min, lat_min, lon_max, lat_max]``.

    Returns
    -------
    gpd.GeoDataFrame
        MRDS point records (``site_name``, ``dev_stat``, ``code_list``, ...).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    # WFS 2.0 with an EPSG urn uses lat,lon axis order.
    wfs_bbox = f"{lat_min},{lon_min},{lat_max},{lon_max},urn:ogc:def:crs:EPSG::4326"
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": "mrds",
        "count": "5000",
        "srsName": "EPSG:4326",
        "bbox": wfs_bbox,
    }
    url = f"{MRDS_WFS}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tmp:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (trusted USGS host)
            tmp.write(resp.read())
        gml_path = tmp.name
    gdf = gpd.read_file(gml_path)
    logger.info("MRDS returned %d records for bbox %s", len(gdf), bbox)
    return gdf


def confirm_identity(site: SiteSpec) -> bool:
    """Query MRDS for one site and print an identity verdict; return PASS bool."""
    spec = IDENTITY[site.site_id]
    keyword, commodity = spec["keyword"], spec["commodity"]
    gdf = fetch_mrds(site_search_bbox(site))

    named = gdf[gdf["site_name"].str.contains(keyword, case=False, na=False)].copy()
    named["commodities"] = named["code_list"].fillna("").str.split()
    matched = named[named["commodities"].apply(lambda codes: commodity in codes)]
    developed = matched[matched["dev_stat"].isin(DEVELOPED)]

    print(f"\n--- {site.site_id}: {site.name} ---")
    print(
        f"MRDS deposits named ~'{keyword}': {len(named)}; "
        f"with {commodity}: {len(matched)}; developed: {len(developed)}"
    )
    for _, row in developed.head(8).iterrows():
        pt = row.geometry
        print(
            f"  {row['site_name']}  [{row['dev_stat']}]  {row['code_list'].strip()}  "
            f"({pt.y:.4f}, {pt.x:.4f})"
        )

    if len(named):
        named.drop(columns="commodities").to_file(
            OUTPUT_DIR / f"mrds_{site.site_id}.gpkg", driver="GPKG"
        )

    verdict = len(developed) > 0
    print(f"  identity verdict: {'PASS' if verdict else 'REVIEW (no developed match)'}")
    return verdict


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {site.site_id: confirm_identity(site) for site in SITES.values()}
    print("\n=== summary ===")
    for site_id, ok in results.items():
        print(f"  {site_id}: {'confirmed' if ok else 'needs review'}")


if __name__ == "__main__":
    main()
