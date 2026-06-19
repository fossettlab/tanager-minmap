"""Validate a site's mineral maps against the Rockwell ASTER reference.

spec.md step 4-5 ("validated maps"). Recomputes the diagnostic band-depth and
MTMF abundance maps for a site's lead scene, aligns the published Rockwell
alteration raster to the scene grid (run ``download_reference.py`` first), and
reports, per layer, how well the score separates its published alteration
zone(s) from the other classified ground (rank AUC + Mann-Whitney p) plus the
Youden-J-optimal threshold that calibrates detection to the external map.

Thin wrapper over :func:`tanager_rocks.pipeline.run_validate`; the installed
``tanager-minmap validate`` runs the same logic.

Run::

    uv run python scripts/download_reference.py --site goldfield
    uv run python scripts/validate_site.py --site goldfield
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tanager_rocks.config import SITES
from tanager_rocks.pipeline import PipelinePaths, run_validate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="goldfield", choices=tuple(SITES))
    args = parser.parse_args(argv)
    run_validate(SITES[args.site], PipelinePaths.repo_default(ROOT))


if __name__ == "__main__":
    main()
