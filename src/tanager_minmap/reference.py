"""USGS Rockwell ASTER alteration map as an independent validation reference.

Validation for spec.md pipeline step 4-5: the per-mineral diagnostic-feature and
MTMF maps are compared against a published, independent product — the USGS
*Digital map of hydrothermal alteration type, key mineral groups, and green
vegetation of the western United States derived from automated analysis of ASTER
satellite data* (Rockwell & Bonham 2017, USGS data release,
doi:10.5066/F7CR5RK7, public domain / CC0). It is a *categorical* mineral-group /
alteration-type raster (~30 m, EPSG:4326), so the comparison is necessarily
*zone agreement*: does our continuous score for a mineral separate the published
alteration class(es) that contain that mineral group from the other classified
ground?

The class table below is transcribed verbatim from the data release's FGDC
metadata ``Enumerated_Domain`` entries (``aster_southwest_aa61_v8``); nothing is
invented. The mineral/feature -> class mappings are derived from those published
class *definitions* (a class is "positive" for a mineral only when its
definition names that mineral group), with the reasoning recorded inline.
"""

from __future__ import annotations

import xarray as xr
from rasterio.enums import Resampling

# Rockwell & Bonham (2017) pixel value -> material / alteration-type class,
# transcribed from the data-release FGDC metadata Enumerated_Domain. Pixel
# value 0 is also the raster's nodata value (so "no materials classified" and
# "off image" are indistinguishable in the data).
ROCKWELL_CLASSES: dict[int, str] = {
    0: "No materials classified",
    1: "Minor ferric iron",
    2: "Major ferric iron",
    3: "Advanced argillic +/- ferric iron",
    4: "Argillic (kandite clay +/- sericite +/- smectite) +/- minor advanced argillic",
    5: "Sericite and (or) smectite",
    6: "Carbonate-propylitic",
    7: "Dolomite",
    8: "Jarosite +/- sericite +/- smectite, or hydrous silica + ferric iron",
    9: "Hydrous silica",
    10: "Sericite + chlorite or Fe/Mg sericite",
    11: "Argillic + ferric iron or weathered phyllic +/- minor advanced argillic",
    12: "Sericite and (or) smectite + ferric iron",
    13: "Carbonate-propylitic + sericite and (or) smectite",
    14: "Green vegetation",
    15: "Carbonate-propylitic + ferric iron",
    16: "Sericite + chlorite or Fe/Mg sericite + ferric iron",
    17: "Carbonate-propylitic + sericite and (or) smectite + ferric iron",
    18: "Dolomite + ferric iron",
    19: "Carbonate-propylitic +/- hydrous silica",
    20: "Carbonate-propylitic +/- hydrous silica + ferric iron",
    21: "Ferrous or coarse-grained ferric iron",
    22: "Carbonate-propylitic + ferrous or coarse-grained ferric iron",
    23: "Dolomite + ferrous or coarse-grained ferric iron",
    45: "Dry vegetation in fallow agricultural fields - deep 2.17 and 2.20 micron",
    46: "Dry yellow/brown vegetation in fallow agricultural fields - 2.17 and 2.20 micron",
    47: "Dry vegetation in fallow agricultural fields - deep 2.17 micron",
    48: "No data (clouds, cloud shadow, smoke, haze, data errors)",
    49: "Semi-corrupted SWIR band 5 (advanced argillic / argillic unreliable)",
    50: "Semi-corrupted SWIR band 7 (dolomite / hydrous silica / jarosite unreliable)",
}

# Classes excluded from validation entirely: unclassified / nodata (0, 48),
# vegetation (14, 45-47), and the two semi-corrupted-SWIR flags (49, 50) whose
# alteration calls the metadata itself marks unreliable. Discrimination is then
# tested only among *classified, reliable* ground — the honest comparison.
# Including bare / unclassified ground as negatives would inflate separability
# (altered-vs-bare is trivial; altered-vs-other-alteration is the real test).
ROCKWELL_EXCLUDED: frozenset[int] = frozenset({0, 14, 45, 46, 47, 48, 49, 50})

# Target mineral -> Rockwell class value(s) whose definition names that mineral
# group as a primary constituent. Conservative: only classes that *name* the
# group are positive; all other non-excluded classes are negatives.
#   - alunite: advanced argillic (3) is the alunite-bearing class.
#   - kaolinite / dickite: kandite-group clays in advanced argillic (3) and
#     argillic (4).
#   - jarosite: the jarosite class (8).
#   - hematite / goethite: ferric iron (1, 2). ASTER's VNIR ferric class does
#     not speciate hematite vs goethite, so both validate against the same
#     classes and the reference cannot tell them apart (documented limitation).
#   - muscovite: sericite (fine-grained muscovite) classes (5, 10, 12, 16).
#   - gypsum: ASTER's scheme has no sulfate-evaporite class, so gypsum has no
#     Rockwell positive class and is NOT validatable here. The 2340 nm
#     gypsum/carbonate diagnostic feature is instead checked against the
#     carbonate classes via FEATURE_TO_ROCKWELL.
MINERAL_TO_ROCKWELL: dict[str, frozenset[int]] = {
    "alunite": frozenset({3}),
    "kaolinite": frozenset({3, 4}),
    "dickite": frozenset({3, 4}),
    "jarosite": frozenset({8}),
    "hematite": frozenset({1, 2}),
    "goethite": frozenset({1, 2}),
    "muscovite": frozenset({5, 10, 12, 16}),
}

# Diagnostic band-depth features (features.py / map_site.py) -> Rockwell classes.
# These are spectral features, not single minerals, so the positive sets are the
# alteration types that produce the absorption:
#   - al_oh_doublet (2200 nm Al-OH): every Al-OH clay/sericite alteration type
#     (advanced argillic, argillic, sericite, and their ferric-iron variants).
#   - jarosite (2265 nm): the jarosite class (8).
#   - gypsum_carbonate (2340 nm): the carbonate/dolomite classes (the 2340 nm
#     feature is Mg-OH/carbonate; Rockwell has no gypsum class).
#   - fe_oxide (VNIR ferric): the ferric-iron classes (1, 2).
FEATURE_TO_ROCKWELL: dict[str, frozenset[int]] = {
    "al_oh_doublet": frozenset({3, 4, 5, 10, 11, 12, 16}),
    "jarosite": frozenset({8}),
    "gypsum_carbonate": frozenset({6, 7, 13, 15, 17, 18, 19, 20, 22, 23}),
    "fe_oxide": frozenset({1, 2}),
}


def align_reference(reference: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Reproject the categorical reference onto a target cube's grid.

    Nearest-neighbour resampling preserves the integer class codes — any
    averaging would invent class values that do not exist. ``like`` supplies the
    target CRS, transform and shape (e.g. a 2-D band slice of the Tanager SR
    cube); both arrays must carry rioxarray spatial metadata.

    Parameters
    ----------
    reference : xr.DataArray
        Categorical Rockwell class raster (2-D), with ``rio`` CRS/transform.
    like : xr.DataArray
        Target grid (2-D), with ``rio`` CRS/transform.

    Returns
    -------
    xr.DataArray
        The reference resampled onto ``like``'s grid (integer class codes).
    """
    return reference.rio.reproject_match(like, resampling=Resampling.nearest)
