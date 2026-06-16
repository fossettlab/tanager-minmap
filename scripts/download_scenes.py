"""Download the surface-reflectance HDF5 assets for a site's confirmed scenes.

Resolves each scene's asset href from the STAC catalog (never hand-built — the
href comes from the item's ``assets`` dict) and streams the file to
``data/raw/``. The full catalog inventory is cached at
``data/intermediate/tanager_scene_inventory.csv`` by ``confirm_sites.py``; this
script reuses that cache and walks the catalog only if it is absent. Existing
files of the right size are skipped, so re-runs are cheap.

Run::

    uv run python scripts/download_scenes.py --site bingham
    uv run python scripts/download_scenes.py --site all
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tanager_rocks.config import SITES, TANAGER_SR_ASSET

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("download_scenes")

USER_AGENT = "tanager-rocks/0.1 (research; abradley@wustl.edu)"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INVENTORY_CSV = ROOT / "data" / "intermediate" / "tanager_scene_inventory.csv"


def load_inventory() -> pd.DataFrame:
    """Return the catalog inventory, from cache if present else a fresh walk."""
    if INVENTORY_CSV.exists():
        logger.info("using cached inventory %s", INVENTORY_CSV)
        return pd.read_csv(INVENTORY_CSV)
    logger.info("no cached inventory; walking the Tanager catalog...")
    from tanager_spec.stac import build_scene_inventory, query_tanager_scenes, save_inventory

    inventory = build_scene_inventory(query_tanager_scenes())
    save_inventory(inventory, INVENTORY_CSV)
    return pd.read_csv(INVENTORY_CSV)


def asset_href(inventory: pd.DataFrame, scene_id: str, asset: str) -> str:
    """Resolve a scene's asset href from the inventory's assets dict."""
    rows = inventory[inventory["scene_id"] == scene_id]
    if rows.empty:
        raise KeyError(f"scene {scene_id} not in inventory")
    assets = json.loads(rows.iloc[0]["assets"])
    if asset not in assets:
        raise KeyError(f"scene {scene_id} has no asset {asset!r}")
    return assets[asset]["href"]


def download(href: str, dest: Path) -> None:
    """Stream a URL to ``dest``, skipping if a same-size file already exists."""
    req = urllib.request.Request(href, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (trusted GCS host)
        remote_size = int(resp.headers.get("Content-Length", -1))
        if dest.exists() and dest.stat().st_size == remote_size:
            logger.info("skip %s (already present, %d bytes)", dest.name, remote_size)
            return
        logger.info("downloading %s (%.2f GB)", dest.name, remote_size / 1e9)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    tmp.replace(dest)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=(*SITES, "all"))
    parser.add_argument("--asset", default=TANAGER_SR_ASSET)
    args = parser.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory()
    site_ids = list(SITES) if args.site == "all" else [args.site]
    for site_id in site_ids:
        for scene_id in SITES[site_id].scene_ids:
            href = asset_href(inventory, scene_id, args.asset)
            download(href, RAW_DIR / Path(href).name)


if __name__ == "__main__":
    main()
