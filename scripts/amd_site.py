"""AMD acid-generating-potential proxy for a site's scene (spec.md step 7).

Runs the MTMF pipeline, reduces the secondary AMD-indicator assemblage
(jarosite / Fe-oxides / gypsum) to an ordinal acid-generating-potential map via
:func:`tanager_minmap.hazard.acid_generating_potential`, and writes the tier
GeoTIFF + a categorical PNG. Site-agnostic; the default is the headline
narrative site (Bingham / Kennecott), but it runs identically on Goldfield.

Thin wrapper over :func:`tanager_minmap.pipeline.run_amd`; the installed
``tanager-minmap amd`` runs the same logic.

Run::

    uv run python scripts/amd_site.py --site bingham
    uv run python scripts/amd_site.py --site goldfield
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tanager_minmap.config import SITES
from tanager_minmap.pipeline import PipelinePaths, run_amd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Bingham is the headline AMD site (mine-waste / tailings narrative); the
    # proxy runs identically on Goldfield, where jarosite validated strongest.
    parser.add_argument("--site", default="bingham", choices=tuple(SITES))
    parser.add_argument("--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate")
    parser.add_argument(
        "--quantile", type=float, default=0.90, help="per-mineral detection floor (upper tail)"
    )
    args = parser.parse_args(argv)
    run_amd(
        SITES[args.site],
        PipelinePaths.repo_default(ROOT),
        max_infeas=args.max_infeas,
        quantile=args.quantile,
    )


if __name__ == "__main__":
    main()
