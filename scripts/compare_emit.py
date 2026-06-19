"""Cross-sensor comparison: Tanager vs EMIT at a shared site (spec.md step 6).

Runs the identical alteration-mapping pipeline (diagnostic band depths + MTMF)
on a site's Tanager lead scene and an overlapping EMIT L2A scene, then reports
spectral agreement (scene-mean), per-mineral detection agreement (MTMF maps on
the common grid), and the resolution ratio.

The EMIT granule is queried, selected, and downloaded reproducibly: the clearest
fully-overlapping scene is chosen and its identifier logged. Network steps need
NASA Earthdata credentials in the environment, so run under doppler::

    doppler run --project mac --config dev -- \\
        uv run python scripts/compare_emit.py --site goldfield

Thin wrapper over :func:`tanager_rocks.pipeline.run_emit`; the installed
``tanager-minmap emit`` runs the same logic. If the reflectance file is already
present it is reused (no re-download).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tanager_rocks.config import SITES
from tanager_rocks.pipeline import PipelinePaths, run_emit

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="goldfield", choices=tuple(SITES))
    args = parser.parse_args(argv)
    run_emit(SITES[args.site], PipelinePaths.repo_default(ROOT))


if __name__ == "__main__":
    main()
