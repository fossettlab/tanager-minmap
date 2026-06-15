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

__all__ = ["TANAGER_SR_ASSET", "SEED", "SITES", "SiteSpec", "TARGET_MINERALS", "DIAGNOSTIC_NM"]


@dataclass(frozen=True)
class SiteSpec:
    """A study site.

    Coordinates are *approximate scene-centroid* values taken from ``spec.md``.
    Per the data-integrity rule they are NOT authoritative site footprints:
    ``confirmed=False`` until verified against USGS USMIN/MRDS and a basemap
    in Week 1 (see ``spec.md`` open items). Downstream code that asserts a
    named site must check ``confirmed``.
    """

    site_id: str
    name: str
    role: str  # "hero" | "alteration_showcase"
    n_scenes: int
    centroids: tuple[tuple[float, float], ...]  # (lat, lon), approximate
    confirmed: bool = False


# Sites from spec.md "Sites" table. Goldfield centroids are a coarse range in
# the spec; recorded here as the range corners pending Week-1 confirmation.
SITES: dict[str, SiteSpec] = {
    "bingham": SiteSpec(
        site_id="bingham",
        name="Bingham Canyon / Kennecott, UT",
        role="hero",
        n_scenes=2,
        centroids=((40.56, -112.08), (40.78, -112.01)),
    ),
    "goldfield": SiteSpec(
        site_id="goldfield",
        name="Goldfield district, NV",
        role="alteration_showcase",
        n_scenes=5,
        centroids=((37.4, -117.2), (37.7, -117.1)),
    ),
}

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
