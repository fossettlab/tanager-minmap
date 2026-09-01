"""Produce continuum-removed diagnostic band-depth maps for a site's scene.

Pipeline for one Tanager scene (spec.md steps 2-3): load the SR cube, mask the
O2/H2O absorption bands, derive each diagnostic feature's continuum shoulders
from the splib07 endmembers (data-driven, not hand-picked), compute the
band-depth maps, and write GeoTIFFs + a PNG panel. This is the "one real map"
deliverable; SAM/MTMF unmixing (unmix.py) comes later.

Thin wrapper over :func:`tanager_minmap.pipeline.run_map`; the installed
``tanager-minmap map`` runs the same logic.

Run::

    uv run python scripts/download_speclib.py   # once
    uv run python scripts/download_scenes.py --site bingham
    uv run python scripts/map_site.py --site bingham
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tanager_minmap.config import SITES
from tanager_minmap.pipeline import PipelinePaths, run_map

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="bingham", choices=tuple(SITES))
    args = parser.parse_args(argv)
    run_map(SITES[args.site], PipelinePaths.repo_default(ROOT))


if __name__ == "__main__":
    main()
