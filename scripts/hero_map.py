"""Hero mineral map (spec.md step 9, Visualization & Storytelling).

Runs the MTMF pipeline on the hero site's lead scene, gates abundance by the
mixture-tuned infeasibility, and composites the per-mineral layers into a single
dominant-mineral map — the submission's "readable in 10 s" figure. Defaults
match unmix so the hero is consistent with the per-mineral panels.

Thin wrapper over :func:`tanager_minmap.pipeline.run_hero`; the installed
``tanager-minmap hero`` runs the same logic.

Run::

    uv run python scripts/hero_map.py --site goldfield
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tanager_minmap.config import SITES
from tanager_minmap.pipeline import PipelinePaths, run_hero

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Goldfield/Cuprite is the hero: it is the site whose maps validate cleanly
    # against the Rockwell ASTER map (alunite advanced-argillic centre at Cuprite).
    parser.add_argument("--site", default="goldfield", choices=tuple(SITES))
    parser.add_argument("--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate")
    parser.add_argument(
        "--quantile", type=float, default=0.90, help="per-mineral detection quantile"
    )
    args = parser.parse_args(argv)
    run_hero(
        SITES[args.site],
        PipelinePaths.repo_default(ROOT),
        max_infeas=args.max_infeas,
        quantile=args.quantile,
    )


if __name__ == "__main__":
    main()
