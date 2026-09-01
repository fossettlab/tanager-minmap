"""Run the frozen all-seven-scene M2 repeatability packet.

This command writes per-scene feature/MTMF GeoTIFFs beneath
``data/processed/repeatability/scenes`` and a JSON manifest containing all
five primary and six secondary frozen comparisons.  It never chooses a best
registration shift.  The spatial-validation block manifest is required and is
validated exactly; categorical block IDs are never reprojected or rebuilt.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from tanager_minmap.repeatability import RepeatabilityPaths, run_repeatability_packet

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block-manifest",
        type=Path,
        required=True,
        help="required spatial-validation block_manifest.json; no geometry is made here",
    )
    parser.add_argument(
        "--transfer-thresholds",
        type=Path,
        default=None,
        help="primary-L transfer threshold CSV (default: sibling transfer_thresholds.csv)",
    )
    parser.add_argument(
        "--spatial-summary",
        type=Path,
        default=None,
        help="spatial-validation summary JSON (default: sibling summary.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="fresh output directory (default: data/processed/repeatability)",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=ROOT / "docs" / "input_manifest.json",
        help="frozen expected source-input identities",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="compute-only thread workers; seeded task and result order are unchanged",
    )
    parser.add_argument(
        "--timing-pilot",
        action="store_true",
        help=(
            "run one eligible frozen pair/layer through every resampling branch at full "
            "replicate counts; write no endpoint values or final manifest"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume matching execution state after validating all completed task hashes",
    )
    parser.add_argument(
        "--expected-timing-pilot-sha256",
        default=None,
        help="required verifier-supplied digest of the exact reviewed pilot in full mode",
    )
    parser.add_argument(
        "--resource-admission",
        type=Path,
        default=None,
        help="required reviewed PASS-64 resource_admission.json path in full mode",
    )
    parser.add_argument(
        "--expected-resource-admission-sha256",
        default=None,
        help="required verifier-supplied digest of the exact reviewed resource admission",
    )
    args = parser.parse_args(argv)
    paths = RepeatabilityPaths.repo_default(ROOT)
    if args.output_dir is not None:
        paths = replace(paths, output_dir=args.output_dir)
    manifest = run_repeatability_packet(
        paths,
        block_manifest=args.block_manifest,
        transfer_thresholds=args.transfer_thresholds,
        spatial_summary=args.spatial_summary,
        input_manifest=args.input_manifest,
        workers=args.workers,
        timing_pilot=args.timing_pilot,
        resume=args.resume,
        expected_timing_pilot_sha256=args.expected_timing_pilot_sha256,
        expected_resource_admission_sha256=args.expected_resource_admission_sha256,
        resource_admission_path=args.resource_admission,
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
