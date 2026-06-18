"""EMIT L2A reflectance access + orthorectification (spec.md step 6).

The cross-sensor tie-breaker. NASA's EMIT is the only other spaceborne imaging
spectrometer with comparable VSWIR coverage (285 bands, 381-2493 nm, ~60 m),
so running the *same* alteration-mapping pipeline on an EMIT scene over a shared
site is the cleanest external check on Tanager's mineral maps.

This module handles the EMIT-specific I/O the shared `tanager_spec` layer does
not: querying the NASA Earthdata STAC for overlapping L2A granules, downloading
the reflectance file (auth via `earthaccess`; credentials come from the
environment, never inlined), and orthorectifying the raw `(downtrack,
crosstrack, band)` array onto its WGS84 grid with the granule's geometry lookup
table (GLT). The orthorectified cube is returned as a ``(band, y, x)``
:class:`xarray.DataArray` — the same contract the diagnostic-feature and MTMF
steps already consume — so the comparison reuses the existing pipeline verbatim.

Authentication
--------------
``earthaccess.login(strategy="environment")`` reads ``EARTHDATA_USERNAME`` and
``EARTHDATA_PASSWORD`` from the environment. Run any function that touches the
network under ``doppler run --project mac --config dev -- ...`` so the
credentials are injected as env vars and never appear in a command line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from shapely.geometry import Polygon, box

logger = logging.getLogger(__name__)

EMIT_L2A_SHORT_NAME = "EMITL2ARFL"
GLT_FILL = 0  # GLT nodata: a 0 index means "no raw pixel maps here"
REFL_FILL = -9999.0  # EMIT reflectance no-data sentinel


@dataclass(frozen=True)
class EmitGranule:
    """A ranked EMIT L2A candidate over a site footprint."""

    granule_ur: str
    date: str
    cloud_cover: float | None
    overlap: float  # fraction of the site footprint the granule covers
    data_links: tuple[str, ...]


def _granule_polygon(umm: dict) -> Polygon | None:
    """WGS84 footprint polygon from a granule's UMM, or None if absent."""
    try:
        pts = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0][
            "Boundary"
        ]["Points"]
        return Polygon([(p["Longitude"], p["Latitude"]) for p in pts])
    except (KeyError, IndexError):
        return None


def rank_granules(results: list, footprint: Polygon) -> list[EmitGranule]:
    """Rank Earthdata search results by footprint overlap (descending).

    Parameters
    ----------
    results : list
        ``earthaccess.search_data`` results (each exposes ``["umm"]`` and
        ``.data_links()``).
    footprint : shapely.geometry.Polygon
        The site footprint (WGS84) to score coverage against.

    Returns
    -------
    list of EmitGranule
        Sorted by overlap fraction, highest first.
    """
    ranked: list[EmitGranule] = []
    for g in results:
        umm = g["umm"]
        poly = _granule_polygon(umm)
        if poly is None:
            continue
        overlap = poly.intersection(footprint).area / footprint.area
        date = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10]
        cloud = umm.get("CloudCover")
        ranked.append(
            EmitGranule(
                granule_ur=umm["GranuleUR"],
                date=date,
                cloud_cover=float(cloud) if cloud is not None else None,
                overlap=float(overlap),
                data_links=tuple(g.data_links()),
            )
        )
    ranked.sort(key=lambda e: e.overlap, reverse=True)
    return ranked


def select_granule(
    ranked: list[EmitGranule],
    max_cloud: float = 10.0,
    min_overlap: float = 0.99,
) -> EmitGranule:
    """Pick the clearest fully-overlapping granule.

    Among granules covering at least ``min_overlap`` of the footprint, return the
    one with the lowest cloud cover at or below ``max_cloud``. Selection is
    explicit and logged so the chosen scene is reproducible, not arbitrary.
    """
    covering = [g for g in ranked if g.overlap >= min_overlap]
    if not covering:
        raise ValueError(f"no EMIT granule covers >= {min_overlap:.0%} of the footprint")
    clear = [g for g in covering if g.cloud_cover is not None and g.cloud_cover <= max_cloud]
    pool = clear or covering  # fall back to coverage-only if no cloud metadata
    chosen = min(pool, key=lambda e: (e.cloud_cover if e.cloud_cover is not None else 1e3, e.date))
    logger.info(
        "selected EMIT granule %s (%s, cloud=%s, overlap=%.2f)",
        chosen.granule_ur,
        chosen.date,
        chosen.cloud_cover,
        chosen.overlap,
    )
    return chosen


def rfl_path(dest_dir: Path, granule_ur: str) -> Path:
    """Local path of a granule's reflectance file under ``dest_dir``."""
    return Path(dest_dir) / f"{granule_ur}.nc"


def _ortho_window(geotransform: np.ndarray, bbox: list[float], shape: tuple[int, int]):
    """Row/col slices of the ortho grid covering a WGS84 ``bbox``.

    ``geotransform`` is GDAL order ``[x0, dx, 0, y0, 0, dy]`` (dy < 0). Returns
    ``(row0, row1, col0, col1)`` clamped to the grid, so only the lead-scene
    sized window is materialised rather than the full ~75 km swath.
    """
    x0, dx, _, y0, _, dy = geotransform
    ny, nx = shape
    xmin, ymin, xmax, ymax = bbox
    cols = [(xmin - x0) / dx, (xmax - x0) / dx]
    rows = [(ymax - y0) / dy, (ymin - y0) / dy]  # dy < 0 => ymax maps to the smaller row
    col0 = max(0, int(np.floor(min(cols))))
    col1 = min(nx, int(np.ceil(max(cols))) + 1)
    row0 = max(0, int(np.floor(min(rows))))
    row1 = min(ny, int(np.ceil(max(rows))) + 1)
    return row0, row1, col0, col1


def load_emit_reflectance(
    path: str | Path,
    bbox: list[float] | None = None,
) -> tuple[xr.DataArray, np.ndarray]:
    """Orthorectified EMIT L2A reflectance as a ``(band, y, x)`` cube (EPSG:4326).

    The raw ``(downtrack, crosstrack, band)`` reflectance is mapped onto its
    regular WGS84 grid with the granule's GLT (``location/glt_x``,
    ``location/glt_y``; 1-based indices, 0 = fill). EMIT-flagged bad bands
    (``sensor_band_parameters/good_wavelengths == 0``) and the reflectance
    no-data sentinel are set to NaN, matching the masking contract the rest of
    the pipeline expects.

    Parameters
    ----------
    path : str or Path
        Local ``EMIT_L2A_RFL_*.nc`` file.
    bbox : list of float, optional
        WGS84 ``[xmin, ymin, xmax, ymax]``; if given, only the ortho window
        covering it is materialised (keeps memory to the lead-scene footprint).

    Returns
    -------
    cube : xr.DataArray
        Dims ``("band", "y", "x")`` with a WGS84 CRS and transform; the ``band``
        coordinate carries the wavelength (nm).
    wavelengths : np.ndarray
        Band-center wavelengths (nm), length 285.
    """
    with h5py.File(path, "r") as f:
        wl = f["sensor_band_parameters/wavelengths"][:].astype(float)
        good = f["sensor_band_parameters/good_wavelengths"][:].astype(bool)
        geotransform = np.asarray(f.attrs["geotransform"], dtype=float)
        glt_x = f["location/glt_x"][:]
        glt_y = f["location/glt_y"][:]
        if bbox is not None:
            r0, r1, c0, c1 = _ortho_window(geotransform, bbox, glt_x.shape)
            glt_x = glt_x[r0:r1, c0:c1]
            glt_y = glt_y[r0:r1, c0:c1]
        else:
            r0, c0 = 0, 0
            r1, c1 = glt_x.shape
        refl = f["reflectance"]  # (downtrack, crosstrack, band), read lazily below

        valid = (glt_x != GLT_FILL) & (glt_y != GLT_FILL)
        ny, nx = glt_x.shape
        nb = wl.size
        out = np.full((ny, nx, nb), np.nan, dtype="float32")
        # Gather only the raw pixels the window references (1-based -> 0-based).
        dt = glt_y[valid] - 1
        ct = glt_x[valid] - 1
        # h5py fancy-indexing needs sorted unique rows; pull the needed downtrack
        # rows once, then index crosstrack in-memory to keep the read bounded.
        rows_needed = np.unique(dt)
        sub = refl[rows_needed.min() : rows_needed.max() + 1, :, :]
        out[valid] = sub[dt - rows_needed.min(), ct, :]

    out[out == REFL_FILL] = np.nan
    out[:, :, ~good] = np.nan

    x0, dx, _, y0, _, dy = geotransform
    xs = x0 + (c0 + np.arange(nx) + 0.5) * dx
    ys = y0 + (r0 + np.arange(ny) + 0.5) * dy
    cube = xr.DataArray(
        np.moveaxis(out, 2, 0),  # (band, y, x)
        dims=("band", "y", "x"),
        coords={"band": wl, "y": ys, "x": xs},
    )
    cube = cube.rio.write_crs("EPSG:4326").rio.write_transform()
    logger.info(
        "loaded EMIT reflectance %s -> ortho (%d, %d), %d bands",
        Path(path).name,
        ny,
        nx,
        nb,
    )
    return cube, wl


def footprint_bbox(cube_or_bounds) -> list[float]:
    """WGS84 ``[xmin, ymin, xmax, ymax]`` from a rio-aware cube (helper)."""
    b = cube_or_bounds.rio.reproject("EPSG:4326").rio.bounds()
    return list(b)


__all__ = [
    "EmitGranule",
    "EMIT_L2A_SHORT_NAME",
    "rank_granules",
    "select_granule",
    "rfl_path",
    "load_emit_reflectance",
    "footprint_bbox",
    "box",  # re-export for callers building footprints
]
