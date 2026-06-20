"""Interactive slippy-map overlays for the submission story page.

Reprojects a categorical class raster (dominant mineral, or AMD tier) to
geographic coordinates and renders it as a translucent RGBA overlay on a
satellite basemap with :mod:`folium`, so a reader can pan and zoom the Tanager
results over the real terrain. The colours match the static figures
(:data:`tanager_rocks.viz.MINERAL_COLORS`, :data:`tanager_rocks.viz.AGP_TIER_COLORS`).
"""

from __future__ import annotations

import folium
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from rasterio.enums import Resampling

# Esri World Imagery XYZ basemap (public, no key) — satellite context.
_ESRI_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_ESRI_ATTR = "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"

# Opaque alpha for a shown class pixel (0-255); transparent classes get 0.
_SHOWN_ALPHA = 255


def reproject_classes_4326(
    classes: xr.DataArray, crs, transform, nodata: int = -1
) -> tuple[np.ndarray, list[list[float]]]:
    """Reproject a class raster to EPSG:4326 for a folium ImageOverlay.

    Returns the north-up class array and the lat/lon bounds in folium order
    ``[[south, west], [north, east]]``. Nearest-neighbour keeps class codes
    intact; reprojection fill is set to ``nodata``.
    """
    geo = classes.rio.write_crs(crs).rio.write_transform(transform)
    rep = geo.rio.reproject("EPSG:4326", resampling=Resampling.nearest, nodata=nodata)
    west, south, east, north = rep.rio.bounds()
    return rep.values.astype(int), [[south, west], [north, east]]


def class_rgba(
    arr: np.ndarray,
    colors: dict[int, str | tuple[float, float, float]],
    transparent: frozenset[int] | set[int] = frozenset(),
) -> np.ndarray:
    """Colorize an integer class array to an ``(H, W, 4)`` uint8 RGBA overlay.

    Classes in ``colors`` are drawn opaque; classes in ``transparent`` (and any
    code not in ``colors``) are fully transparent, so background / nodata shows
    the basemap.
    """
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    for code, color in colors.items():
        sel = arr == code
        if not sel.any():
            continue
        r, g, b = (int(255 * c) for c in mcolors.to_rgb(color))
        rgba[sel, 0], rgba[sel, 1], rgba[sel, 2] = r, g, b
        rgba[sel, 3] = 0 if code in transparent else _SHOWN_ALPHA
    return rgba


def _center(bounds: list[list[float]]) -> list[float]:
    (south, west), (north, east) = bounds
    return [(south + north) / 2.0, (west + east) / 2.0]


def story_map(
    rgba: np.ndarray,
    bounds: list[list[float]],
    *,
    layer_name: str,
    zoom: int = 12,
    opacity: float = 0.75,
) -> folium.Map:
    """A folium map: satellite basemap + one toggleable raster overlay, fit to it."""
    fmap = folium.Map(location=_center(bounds), zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(tiles=_ESRI_IMAGERY, attr=_ESRI_ATTR, name="Satellite").add_to(fmap)
    folium.raster_layers.ImageOverlay(
        rgba, bounds=bounds, name=layer_name, opacity=opacity, interactive=False, cross_origin=False
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(bounds)
    return fmap
