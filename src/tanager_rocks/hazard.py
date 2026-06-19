"""Acid-mine-drainage hazard proxy (spec.md step 7).

A qualitative *acid-generating-potential* (AGP) layer built from the
infeasibility-gated MTMF abundance of the secondary AMD-indicator assemblage —
jarosite, the Fe-oxyhydroxides (hematite / goethite), and gypsum. This is a
spectral indicator, **not** a measured pH, acid flux, or net-acid-generation
test; it flags where the *surface mineralogy* is consistent with acid
generation, at 30 m and surface-only.

The ordinal tiers follow the iron-mineral pH zonation of supergene weathering
over sulfide-bearing ground (Swayze et al. 2000, USGS OFR 2000-0205; Williams &
Hauff 2007): jarosite ``KFe3(SO4)2(OH)6`` is stable only in acidic (pH ~2-4),
oxidising, sulfate-rich conditions and is the diagnostic active-acid indicator;
the Fe-oxyhydroxides (goethite, then hematite) are the higher-pH, partly
neutralised oxidation products; gypsum ``CaSO4.2H2O`` is a secondary sulfate
that, *absent* the acidic iron phases, points to a buffered / neutralised
setting (e.g. sulfuric acid consumed against carbonate).

MTMF abundances are **not** summed across minerals (their matched-filter scores
are not on a common scale — see the EMIT comparison's per-map stretch). Instead
each mineral is reduced to a per-pixel presence call using the *same*
per-mineral upper-tail detection floor as the hero map
(:func:`tanager_rocks.viz.mineral_map`), and the AGP tier is assigned by the
most acidic indicator present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Ordinal acid-generating-potential tiers (higher = stronger spectral indication
# of acid generation). BACKGROUND is reserved for in-scene pixels carrying none
# of the indicator minerals; off-scene / nodata pixels are left NaN, not 0.
AGP_BACKGROUND = 0
AGP_LOW = 1  # gypsum only: secondary sulfate, no acidic iron phase -> buffered
AGP_MODERATE = 2  # Fe-oxide present, no jarosite: oxidised / partly neutralised
AGP_HIGH = 3  # jarosite present: acidic (pH ~2-4), sulfate-rich, active

AGP_LABELS: dict[int, str] = {
    AGP_BACKGROUND: "background (no indicator)",
    AGP_LOW: "low / neutralised (gypsum)",
    AGP_MODERATE: "moderate (Fe-oxide)",
    AGP_HIGH: "high (jarosite)",
}

# The AMD indicator assemblage. Jarosite is the diagnostic active-acid phase;
# the Fe-oxyhydroxides set the moderate tier when jarosite is absent; gypsum
# sets the low tier when neither acidic-iron phase is present.
JAROSITE = "jarosite"
FE_OXIDE_MINERALS = ("hematite", "goethite")
GYPSUM = "gypsum"


@dataclass(frozen=True)
class AmdResult:
    """Outputs of :func:`acid_generating_potential`.

    Attributes
    ----------
    tiers : xr.DataArray
        Ordinal AGP code per pixel (``AGP_*``), ``NaN`` off the in-scene domain.
    presence : dict
        Mineral -> boolean presence map (the per-pixel detection calls).
    counts : dict
        AGP tier -> in-domain pixel count.
    domain : np.ndarray
        Boolean mask of analysable (in-scene, valid) pixels.
    """

    tiers: xr.DataArray
    presence: dict[str, np.ndarray]
    counts: dict[int, int]
    domain: np.ndarray


def _present(gated: np.ndarray, quantile: float) -> np.ndarray:
    """Per-pixel presence: gated abundance at or above its own upper-tail floor.

    Mirrors the per-mineral detection floor in :func:`tanager_rocks.viz.mineral_map`
    so "detection" means the same thing everywhere: the threshold is the
    ``quantile`` of the mineral's own positive (infeasibility-gated) abundances.
    Infeasibility-gated (``NaN``) pixels are never present.
    """
    out = np.zeros(gated.shape, dtype=bool)
    pos = gated[np.isfinite(gated) & (gated > 0)]
    if pos.size == 0:
        return out
    thr = float(np.quantile(pos, quantile))
    if thr <= 0:
        return out
    return np.isfinite(gated) & (gated >= thr)


def acid_generating_potential(
    mtmf_ds: xr.Dataset,
    *,
    max_infeas: float = 1.0,
    quantile: float = 0.90,
) -> AmdResult:
    """Ordinal acid-generating-potential proxy from an MTMF dataset.

    Each indicator mineral's abundance is infeasibility-gated (``infeas <
    max_infeas``, the same gate as the unmixing/hero steps), reduced to a
    per-pixel presence call against its own upper-tail floor (``quantile``), and
    the tier is assigned by the most acidic indicator present:
    jarosite -> HIGH, else Fe-oxide -> MODERATE, else gypsum -> LOW, else
    BACKGROUND. Off-scene / nodata pixels (non-finite raw matched-filter score)
    are ``NaN``.

    Parameters
    ----------
    mtmf_ds : xr.Dataset
        Output of :func:`tanager_rocks.unmix.mtmf` (``<mineral>_mf`` abundance +
        ``<mineral>_infeas`` per mineral). Must include jarosite.
    max_infeas : float
        Infeasibility gate applied per mineral before the presence call.
    quantile : float
        Upper quantile of each mineral's positive abundance used as its
        detection floor (0.90 keeps the top decile, as in the hero map).

    Returns
    -------
    AmdResult
    """
    available = {v[:-3] for v in mtmf_ds.data_vars if str(v).endswith("_mf")}
    needed = (JAROSITE, *FE_OXIDE_MINERALS, GYPSUM)
    if JAROSITE not in available:
        raise ValueError("AMD proxy requires a jarosite MTMF layer (the acid indicator)")
    for mineral in needed:
        if mineral not in available:
            logger.warning("AMD indicator %s absent from MTMF dataset; tier unreachable", mineral)

    template = mtmf_ds[f"{JAROSITE}_mf"]
    raw = template.values
    domain = np.isfinite(raw)  # in-scene valid pixels (all minerals share validity)

    presence: dict[str, np.ndarray] = {}
    for mineral in needed:
        if mineral not in available:
            presence[mineral] = np.zeros(raw.shape, dtype=bool)
            continue
        gated = mtmf_ds[f"{mineral}_mf"].where(mtmf_ds[f"{mineral}_infeas"] < max_infeas).values
        presence[mineral] = _present(gated, quantile)

    fe_oxide = np.zeros(raw.shape, dtype=bool)
    for mineral in FE_OXIDE_MINERALS:
        fe_oxide |= presence[mineral]

    # Assign in ascending priority so the most acidic present indicator wins.
    tier = np.full(raw.shape, float(AGP_BACKGROUND))
    tier[presence[GYPSUM]] = AGP_LOW
    tier[fe_oxide] = AGP_MODERATE
    tier[presence[JAROSITE]] = AGP_HIGH
    tier[~domain] = np.nan

    tiers = xr.DataArray(tier, dims=template.dims, coords=template.coords, name="agp")
    counts = {code: int(np.sum(tier == code)) for code in AGP_LABELS}
    logger.info(
        "AGP tiers (max_infeas=%.2f, q=%.2f): %s",
        max_infeas,
        quantile,
        {AGP_LABELS[c]: counts[c] for c in counts},
    )
    return AmdResult(tiers=tiers, presence=presence, counts=counts, domain=domain)
