"""Unmixing for a site's scene (spec.md step 4): SAM baseline + MTMF.

Loads the SR cube, masks absorption bands, and selects one medoid endmember per
target mineral from splib07. Runs (1) the SAM baseline — best-match
classification within an angle threshold — and (2) MTMF — covariance-aware
matched-filter abundance plus the mixture-tuned infeasibility, gated to keep
abundance only where the pixel is spectrally feasible. Writes class/abundance/
infeasibility GeoTIFFs and PNG panels.

Thin wrapper over :func:`tanager_minmap.pipeline.run_unmix`; the installed
``tanager-minmap unmix`` runs the same logic.

Run::

    uv run python scripts/unmix_site.py --site bingham
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tanager_minmap.config import SITES
from tanager_minmap.pipeline import PipelinePaths, run_unmix

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="bingham", choices=tuple(SITES))
    # 0.15 rad ~ the 5th percentile of best-match angles on the Bingham scene:
    # full-spectrum SAM vs pure endmembers rarely beats ~0.14 rad on mixed 30 m
    # pixels, so this keeps only the most spectrally pure matches. SAM is a
    # coarse baseline here (not ground-truth-calibrated); MTMF is primary.
    parser.add_argument(
        "--max-angle", type=float, default=0.15, help="SAM acceptance threshold (radians)"
    )
    # Infeasibility gate for MTMF detections. Background sits near ~0.2 and the
    # anomalous false-positive tail runs well above ~2; 1.0 keeps the feasible
    # high-abundance nose while dropping the worst anomalies.
    parser.add_argument("--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate")
    args = parser.parse_args(argv)
    run_unmix(
        SITES[args.site],
        PipelinePaths.repo_default(ROOT),
        max_angle=args.max_angle,
        max_infeas=args.max_infeas,
    )


if __name__ == "__main__":
    main()
