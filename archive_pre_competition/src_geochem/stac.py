"""STAC catalog query helpers for Tanager-1 hyperspectral imagery.

Functions for searching the Tanager STAC catalog, filtering scenes by
geometry/date, and downloading assets (surface reflectance or radiance).

Typical usage::

    from src.stac import search_scenes, load_scene

    items = search_scenes(bbox=(-118.5, 35.0, -117.5, 36.0))
    ds = load_scene(items[0], bands="reflectance")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import pystac_client
import stackstac
import xarray as xr

if TYPE_CHECKING:
    from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

# Tanager STAC catalog endpoint — update when public catalog URL is confirmed
TANAGER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
TANAGER_COLLECTION = "tanager-1"


def search_scenes(
    bbox: tuple[float, float, float, float] | None = None,
    geometry: Polygon | None = None,
    datetime_range: str | None = None,
    max_cloud_cover: float = 20.0,
) -> list:
    """Search Tanager STAC catalog for scenes matching spatial/temporal criteria.

    Parameters
    ----------
    bbox : tuple of float, optional
        Bounding box as (west, south, east, north) in EPSG:4326.
    geometry : Polygon, optional
        Shapely polygon to intersect with scene footprints.
    datetime_range : str, optional
        ISO 8601 datetime range, e.g. ``"2024-01/2024-12"``.
    max_cloud_cover : float
        Maximum cloud cover percentage (0–100).

    Returns
    -------
    list of pystac.Item
        Matching STAC items sorted by date.
    """
    # TODO: implement STAC search
    # 1. Open catalog with pystac_client.Client.open(TANAGER_STAC_URL)
    # 2. Search with bbox/geometry, datetime, and cloud cover filter
    # 3. Return sorted list of items
    raise NotImplementedError


def load_scene(
    item,
    resolution: float = 30.0,
    epsg: int = 4326,
) -> xr.DataArray:
    """Load a Tanager scene as an xarray DataArray via stackstac.

    Parameters
    ----------
    item : pystac.Item
        STAC item to load.
    resolution : float
        Target resolution in CRS units (default 30 m).
    epsg : int
        Target CRS EPSG code.

    Returns
    -------
    xr.DataArray
        Hyperspectral data cube with dims (band, y, x).
    """
    # TODO: implement scene loading
    # 1. stackstac.stack([item], resolution=resolution, epsg=epsg)
    # 2. Return computed DataArray
    raise NotImplementedError


def load_footprints(
    geojson_path: str = "~/Desktop/EDC/tanager_footprints.geojson",
) -> gpd.GeoDataFrame:
    """Load pre-computed Tanager scene footprints.

    Parameters
    ----------
    geojson_path : str
        Path to the footprints GeoJSON file.

    Returns
    -------
    gpd.GeoDataFrame
        Scene footprints with geometry and date columns.
    """
    # TODO: implement footprint loading
    # 1. gpd.read_file(geojson_path)
    # 2. Parse dates, validate CRS
    raise NotImplementedError
