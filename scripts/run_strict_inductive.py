#!/usr/bin/env python3
"""Preflight or run the mandatory strict-inductive MTMF sensitivity.

Preflight is the safe default and does not open source cubes or write outputs.
A real cube-level run requires the explicit ``--execute`` flag.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tanager_rocks.strict_inductive import (
    failure_payload,
    preflight_strict_inductive,
    run_strict_inductive,
    strict_json_dumps,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPATIAL_DIR = ROOT / "data" / "processed" / "spatial_validation"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without exposing scientific overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block-manifest",
        type=Path,
        default=DEFAULT_SPATIAL_DIR / "block_manifest.json",
    )
    parser.add_argument(
        "--m2-summary",
        type=Path,
        default=DEFAULT_SPATIAL_DIR / "summary.json",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=ROOT / "docs" / "input_manifest.json",
    )
    parser.add_argument("--maps-dir", type=Path, default=ROOT / "data" / "intermediate" / "maps")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data" / "reference")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--speclib-dir",
        type=Path,
        default=ROOT / "data" / "speclib" / "ASCIIdata_splib07a",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "strict_inductive",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate frozen artifacts and print strict JSON without opening source cubes (default)"
        ),
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="open source cubes and execute all frozen folds",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=None,
        help="parallel workers for deterministic confirmatory permutations only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.execute:
            _, payload = preflight_strict_inductive(
                root=ROOT,
                block_manifest_path=args.block_manifest,
                input_manifest_path=args.input_manifest,
                summary_path=args.m2_summary,
                maps_dir=args.maps_dir,
                reference_dir=args.reference_dir,
            )
            print(strict_json_dumps(payload), end="")
            return 0
        outputs = run_strict_inductive(
            root=ROOT,
            block_manifest_path=args.block_manifest,
            input_manifest_path=args.input_manifest,
            summary_path=args.m2_summary,
            maps_dir=args.maps_dir,
            reference_dir=args.reference_dir,
            raw_dir=args.raw_dir,
            speclib_dir=args.speclib_dir,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(strict_json_dumps(failure_payload(error)), end="")
        return 2
    print(
        strict_json_dumps(
            {
                "status": "complete",
                "outputs": {key: str(path) for key, path in sorted(outputs.items())},
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
