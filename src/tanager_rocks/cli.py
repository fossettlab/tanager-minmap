"""``tanager-minmap`` command-line interface.

A thin, STAC-driven entry point over the analysis modules so the whole
pipeline runs from one command (spec.md "Workflow & Tool"). Subcommands map
onto the pipeline stages:

- ``map``    : diagnostic-feature + SAM/MTMF mineral map for a site
- ``ablate`` : SRF-degrade to Sentinel-2 and quantify the loss
- ``emit``   : EMIT cross-sensor comparison at the overlapping site

The command bodies are not yet implemented; argument parsing is wired so the
interface and ``--help`` are stable from the start.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import SITES

_SITE_CHOICES = tuple(SITES)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tanager-minmap",
        description="Mineral / alteration mapping from Tanager VSWIR surface reflectance.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_map = sub.add_parser("map", help="mineral map for a site")
    p_map.add_argument("--site", required=True, choices=_SITE_CHOICES)
    p_map.add_argument("--output", required=True, help="output directory")

    p_ablate = sub.add_parser("ablate", help="Sentinel-2 band-ablation comparison")
    p_ablate.add_argument("--site", required=True, choices=_SITE_CHOICES)
    p_ablate.add_argument("--output", required=True, help="output directory")

    p_emit = sub.add_parser("emit", help="EMIT cross-sensor comparison")
    p_emit.add_argument("--site", required=True, choices=_SITE_CHOICES)
    p_emit.add_argument("--output", required=True, help="output directory")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``tanager-minmap`` console script."""
    args = _build_parser().parse_args(argv)
    # TODO: dispatch args.command to the pipeline (features -> unmix -> viz).
    raise NotImplementedError(f"command {args.command!r} not yet implemented")


if __name__ == "__main__":
    raise SystemExit(main())
