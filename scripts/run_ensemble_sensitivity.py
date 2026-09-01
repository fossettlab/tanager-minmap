"""Run the preregistered E6 MTMF ensemble sensitivity analysis.

The default command is frozen by ``docs/m2_ensemble_sensitivity_preregistration.md``.
Scientific argument changes require an explicit, recorded protocol amendment.
Only the deterministic NumPy/CPU reference path is implemented; batching and
storage options are compute controls and do not alter member order or methods.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tanager_minmap.ensemble_sensitivity import (
    FROZEN_BOOTSTRAP_REPLICATES,
    FROZEN_QUANTILES,
    FROZEN_RIDGES,
    FROZEN_SEED,
    FROZEN_SITES,
    FROZEN_STOCHASTIC_REPLICATES,
    ProtocolError,
    run_ensemble_sensitivity,
    validate_protocol_arguments,
)

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the exact preregistered CLI plus compute-only controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--block-manifest", type=Path, required=True)
    parser.add_argument("--sites", nargs="+", default=list(FROZEN_SITES))
    parser.add_argument("--ridge", nargs="+", type=float, default=list(FROZEN_RIDGES))
    parser.add_argument(
        "--detection-quantiles", nargs="+", type=float, default=list(FROZEN_QUANTILES)
    )
    parser.add_argument("--infeasibility-gates", nargs="+", default=["none", "1.0"])
    parser.add_argument("--stochastic-replicates", type=int, default=FROZEN_STOCHASTIC_REPLICATES)
    parser.add_argument("--bootstrap-replicates", type=int, default=FROZEN_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-amendment",
        type=Path,
        default=None,
        help=(
            "authorized pre-result JSON amendment matching the E6 amendment schema and "
            "exact scientific deviations"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="compute-only control; only the deterministic NumPy reference is implemented",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of score fits staged at once; does not alter arithmetic or member order",
    )
    parser.add_argument(
        "--storage-layout",
        choices=("disk", "memory"),
        default="disk",
        help="compute-only score-cache layout",
    )
    parser.add_argument("--resume", action="store_true", help="resume the exact ordered design")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--timing-pilot",
        action="store_true",
        help="run only baseline and stochastic replicate 0 per site",
    )
    mode.add_argument(
        "--design-only",
        action="store_true",
        help="stop after preflight and deterministic design materialization",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        deviations = validate_protocol_arguments(args)
        outputs = run_ensemble_sensitivity(args, root=ROOT, deviations=deviations)
    except (FileNotFoundError, ProtocolError, ValueError) as error:
        parser.error(str(error))
    for label, path in outputs.items():
        logging.info("%s: %s", label, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
