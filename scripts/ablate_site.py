"""Band-ablation: quantify what Sentinel-2 loses vs Tanager (spec.md step 5).

Degrades the splib07 alteration endmembers (resampled to a scene's Tanager
wavelength grid) to Sentinel-2 bands via published SRFs, and reports the
pairwise spectral-angle separability in each sensor's band space. The headline
is alunite vs kaolinite: S2's single broad ~2200 nm band (B12) spans the whole
Al-OH doublet, roughly halving their separability. Writes a CSV of the pair
angles and the band-ablation figure.

Thin wrapper over :func:`tanager_rocks.pipeline.run_ablate`; the installed
``tanager-minmap ablate`` runs the same logic.

Run::

    uv run python scripts/ablate_site.py --site bingham
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tanager_rocks.config import SITES
from tanager_rocks.pipeline import PipelinePaths, run_ablate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="bingham", choices=tuple(SITES))
    args = parser.parse_args(argv)
    run_ablate(SITES[args.site], PipelinePaths.repo_default(ROOT))


if __name__ == "__main__":
    main()
