"""``tanager-minmap`` command-line interface.

A thin entry point over :mod:`tanager_rocks.pipeline` so the whole offline
pipeline runs from one installed command (spec.md "Workflow & Tool").
Subcommands map onto the pipeline stages:

- ``map``    : continuum-removed diagnostic band-depth maps (steps 2-3)
- ``unmix``  : SAM baseline + MTMF abundance/infeasibility (step 4)
- ``ablate`` : SRF-degrade to Sentinel-2 and quantify the separability loss (step 5)
- ``amd``    : ordinal acid-generating-potential proxy (step 7)
- ``hero``   : dominant-mineral hero map (step 9)
- ``emit``   : Tanager vs EMIT cross-sensor comparison (step 6)
- ``validate``: discrimination vs the Rockwell ASTER reference (step 4b)

Inputs are read from ``--data-root`` (``<root>/raw`` scenes, ``<root>/speclib``
library, ``<root>/reference`` Rockwell clips); all products are written under
``--output``. ``emit`` needs NASA Earthdata credentials in the environment (run
under ``doppler``); ``validate`` needs the reference clip fetched first
(``scripts/download_reference.py``).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .config import SITES
from .pipeline import (
    PipelinePaths,
    run_ablate,
    run_amd,
    run_emit,
    run_hero,
    run_map,
    run_unmix,
    run_validate,
)

_SITE_CHOICES = tuple(SITES)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tanager-minmap",
        description="Mineral / alteration mapping from Tanager VSWIR surface reflectance.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared options inherited by every subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--site", required=True, choices=_SITE_CHOICES)
    common.add_argument(
        "--data-root", type=Path, default=Path("data"), help="inputs root (<root>/raw + /speclib)"
    )
    common.add_argument(
        "--output", type=Path, default=Path("out"), help="output dir for maps/figures/tables"
    )

    sub.add_parser("map", parents=[common], help="diagnostic band-depth maps")

    p_unmix = sub.add_parser("unmix", parents=[common], help="SAM + MTMF unmixing")
    p_unmix.add_argument("--max-angle", type=float, default=0.15, help="SAM acceptance (radians)")
    p_unmix.add_argument("--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate")

    sub.add_parser("ablate", parents=[common], help="Sentinel-2 band-ablation comparison")

    p_amd = sub.add_parser("amd", parents=[common], help="acid-generating-potential proxy")
    p_amd.add_argument("--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate")
    p_amd.add_argument("--quantile", type=float, default=0.90, help="per-mineral detection floor")

    p_hero = sub.add_parser("hero", parents=[common], help="dominant-mineral hero map")
    p_hero.add_argument("--max-infeas", type=float, default=1.0, help="MTMF infeasibility gate")
    p_hero.add_argument("--quantile", type=float, default=0.90, help="per-mineral detection floor")

    sub.add_parser("emit", parents=[common], help="Tanager vs EMIT comparison (needs Earthdata)")
    sub.add_parser("validate", parents=[common], help="discrimination vs Rockwell reference")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``tanager-minmap`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    site = SITES[args.site]
    paths = PipelinePaths.from_cli(args.data_root, args.output)

    if args.command == "map":
        run_map(site, paths)
    elif args.command == "unmix":
        run_unmix(site, paths, max_angle=args.max_angle, max_infeas=args.max_infeas)
    elif args.command == "ablate":
        run_ablate(site, paths)
    elif args.command == "amd":
        run_amd(site, paths, max_infeas=args.max_infeas, quantile=args.quantile)
    elif args.command == "hero":
        run_hero(site, paths, max_infeas=args.max_infeas, quantile=args.quantile)
    elif args.command == "emit":
        run_emit(site, paths)
    elif args.command == "validate":
        run_validate(site, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
