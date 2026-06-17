"""Band-ablation: quantify what Sentinel-2 loses vs Tanager (spec.md step 5).

Degrades the splib07 alteration endmembers (resampled to a scene's Tanager
wavelength grid) to Sentinel-2 bands via published SRFs, and reports the
pairwise spectral-angle separability in each sensor's band space. The headline
is alunite vs kaolinite: S2's single broad ~2200 nm band (B12) spans the whole
Al-OH doublet, roughly halving their separability. Writes a CSV of the pair
angles and the band-ablation figure.

Run::

    uv run python scripts/ablate_site.py --site bingham
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.srf import load_s2_srf

from tanager_rocks.config import SITES, TANAGER_SR_ASSET
from tanager_rocks.degrade import degrade_endmembers, separability, srf_band_stats
from tanager_rocks.speclib import load_library, select_endmembers
from tanager_rocks.viz import band_ablation_panel, setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ablate_site")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPECLIB_DIR = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
OUT_DIR = ROOT / "data" / "intermediate" / "ablation"
FIGURES_DIR = ROOT / "figures"

# Alteration-relevant mineral contrasts. alunite-kaolinite is the headline
# (advanced argillic vs argillic — the discrimination S2 cannot make); the rest
# show the loss is specific to the SWIR Al-OH region, not universal (the
# VNIR-driven jarosite-goethite contrast survives degradation).
PAIRS: list[tuple[str, str]] = [
    ("alunite", "kaolinite"),
    ("alunite", "muscovite"),
    ("kaolinite", "muscovite"),
    ("kaolinite", "dickite"),
    ("jarosite", "goethite"),
]
HEADLINE = ("alunite", "kaolinite")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="bingham", choices=tuple(SITES))
    args = parser.parse_args(argv)
    scene_id = SITES[args.site].scene_ids[0]

    _, wl = load_tanager_sr_hdf5(RAW_DIR / f"{scene_id}_{TANAGER_SR_ASSET}.h5")
    endmembers = select_endmembers(load_library(SPECLIB_DIR, wl))
    srf = load_s2_srf()
    centers, fwhm = srf_band_stats(srf)

    sep = separability(endmembers, wl, srf, PAIRS)
    logger.info("--- alunite/kaolinite-type separability (spectral angle, deg) ---")
    for (a, b), (full, coarse) in sep.items():
        loss = 100 * (1 - coarse / full) if full else float("nan")
        logger.info(
            "%-22s Tanager %5.2f  S2 %5.2f  (%+.0f%%)",
            f"{a}-{b}",
            np.degrees(full),
            np.degrees(coarse),
            loss,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"ablation_{args.site}_{scene_id}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "tanager_angle_deg", "s2_angle_deg", "loss_pct"])
        for (a, b), (full, coarse) in sep.items():
            w.writerow(
                [
                    f"{a}-{b}",
                    f"{np.degrees(full):.3f}",
                    f"{np.degrees(coarse):.3f}",
                    f"{100 * (1 - coarse / full):.1f}",
                ]
            )

    degraded = degrade_endmembers(endmembers, wl, srf)
    full_deg, s2_deg = (np.degrees(x) for x in sep[HEADLINE])
    setup_style()
    fig = band_ablation_panel(
        np.asarray(wl, float),
        {m: endmembers[m].reflectance for m in HEADLINE},
        {m: degraded[m] for m in HEADLINE},
        centers,
        fwhm,
        full_deg,
        s2_deg,
        minerals=HEADLINE,
        title=f"{SITES[args.site].name}: Tanager vs Sentinel-2 — Al-OH doublet",
    )
    out_png = FIGURES_DIR / f"{args.site}_{scene_id}_band_ablation.png"
    fig.savefig(out_png)
    logger.info("wrote %s", out_png)


if __name__ == "__main__":
    main()
