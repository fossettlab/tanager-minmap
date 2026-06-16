"""Shared constants for the tanager-rocks mineral-mapping flagship.

Values here are sourced from ``spec.md`` (sites, diagnostic absorptions,
target assemblage) or re-exported from :mod:`tanager_spec` (the SR asset
name, absorption-band masks). Nothing is invented: where the spec does not
fix a numeric value (e.g. exact Fe-oxide band centres), the value is
resolved at runtime from the reference spectral library, not hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass

# The Tanager surface-reflectance asset and absorption masks are owned by the
# shared data layer; re-export so this project has a single source of truth.
from tanager_spec.config import TANAGER_SR_ASSET

SEED = 42

# Half-width (degrees) of the search box drawn around a site's scene centroids
# for STAC/MRDS queries. The spec records centroids, not footprints, so a
# modest buffer catches the overlapping scenes; ~0.15 deg ~= 12-17 km here.
SEARCH_BUFFER_DEG = 0.15

__all__ = [
    "TANAGER_SR_ASSET",
    "SEED",
    "SEARCH_BUFFER_DEG",
    "SITES",
    "SiteSpec",
    "site_search_bbox",
    "TARGET_MINERALS",
    "DIAGNOSTIC_NM",
    "FEATURE_DIAGNOSTIC_MINERAL",
]


@dataclass(frozen=True)
class SiteSpec:
    """A study site.

    Coordinates are *approximate scene-centroid* values taken from ``spec.md``.

    ``scene_ids`` are the Tanager scenes whose footprints intersect the site,
    confirmed 2026-06-15 by ``scripts/confirm_sites.py`` walking the open STAC
    catalog (all carry the ``ortho_sr_hdf5`` asset; EMIT L2A overlaps both
    sites). ``confirmed=True`` means the site *identity* was verified the same
    day against the USGS MRDS by ``scripts/confirm_site_identity.py``: a
    developed deposit of the expected commodity and name is present in the box
    (Bingham Open Pit Mine, Producer, Cu-Mo; Goldfield District Gold Deposits,
    Producer, Au). Downstream code that asserts a named site must check
    ``confirmed``.
    """

    site_id: str
    name: str
    role: str  # "hero" | "alteration_showcase"
    n_scenes: int
    centroids: tuple[tuple[float, float], ...]  # (lat, lon), approximate
    scene_ids: tuple[str, ...] = ()
    confirmed: bool = False


# Sites from spec.md "Sites" table; scene_ids from scripts/confirm_sites.py.
# Goldfield centroids are a coarse range in the spec, recorded as the range
# corners. Scene counts match the spec (Bingham 2, Goldfield 5).
SITES: dict[str, SiteSpec] = {
    "bingham": SiteSpec(
        site_id="bingham",
        name="Bingham Canyon / Kennecott, UT",
        role="hero",
        n_scenes=2,
        centroids=((40.56, -112.08), (40.78, -112.01)),
        scene_ids=(
            "20250911_191523_58_4001",
            "20250911_191547_88_4001",
        ),
        confirmed=True,
    ),
    "goldfield": SiteSpec(
        site_id="goldfield",
        name="Goldfield district, NV",
        role="alteration_showcase",
        n_scenes=5,
        centroids=((37.4, -117.2), (37.7, -117.1)),
        scene_ids=(
            "20240925_185504_87_4001",
            "20240925_185509_74_4001",
            "20250222_190233_00_4001",
            "20250222_190237_16_4001",
            "20250222_190241_32_4001",
        ),
        confirmed=True,
    ),
}


def site_search_bbox(site: SiteSpec, buffer_deg: float = SEARCH_BUFFER_DEG) -> list[float]:
    """WGS84 ``[lon_min, lat_min, lon_max, lat_max]`` search box around a site.

    The box bounds the site's scene centroids, expanded by ``buffer_deg`` on
    each side, for STAC and MRDS queries.
    """
    lats = [lat for lat, _ in site.centroids]
    lons = [lon for _, lon in site.centroids]
    return [
        min(lons) - buffer_deg,
        min(lats) - buffer_deg,
        max(lons) + buffer_deg,
        max(lats) + buffer_deg,
    ]


# Target hydrothermal-alteration / mine-waste assemblage (spec.md "one sharp
# question" + pipeline step 3). These drive both diagnostic-feature mapping
# and the unmixing endmember set.
TARGET_MINERALS: tuple[str, ...] = (
    "alunite",
    "kaolinite",
    "dickite",
    "jarosite",
    "hematite",
    "goethite",
    "gypsum",
    "muscovite",
)

# Diagnostic absorption positions EXPLICITLY given in spec.md step 3 (nm).
# Fe-oxide VNIR centres (hematite/goethite) are deliberately absent: the spec
# names the feature but not a wavelength, so they are located from the
# reference library at runtime rather than fabricated here.
DIAGNOSTIC_NM: dict[str, float] = {
    "al_oh_doublet": 2200.0,  # alunite vs kaolinite/dickite
    "jarosite": 2265.0,
    "gypsum_carbonate": 2340.0,
}

# Which target mineral's library spectrum defines each feature's continuum
# (and from which the band-depth shoulders are derived). The Al-OH continuum is
# taken from kaolinite, the canonical 2200 nm Al-OH mineral.
FEATURE_DIAGNOSTIC_MINERAL: dict[str, str] = {
    "al_oh_doublet": "kaolinite",
    "jarosite": "jarosite",
    "gypsum_carbonate": "gypsum",
}
