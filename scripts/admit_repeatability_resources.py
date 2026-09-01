#!/usr/bin/env python3
"""Create endpoint-blind repeatability resource-admission evidence.

This command reads only a redacted timing capsule, normalized scheduler
telemetry, a byte-size footprint manifest, and current free-space telemetry.
It never opens a task result, metric, interval, gate, or scientific manifest.
``PASS-64`` is an operational capacity decision, not scientific admission.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tanager_rocks.repeatability_resources import (
    RULE_RELATIVE_PATH,
    SOURCE_MANIFEST_RELATIVE_PATH,
    VERIFIER_MODULE_RELATIVE_PATH,
    ResourceAdmissionError,
    atomic_write_json,
    build_footprint_manifest,
    ensure_output_outside_roots,
    produce_resource_admission,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_footprint(args: argparse.Namespace) -> int:
    ensure_output_outside_roots(args.output, args.timing_output_root, args.runtime_root)
    footprint = build_footprint_manifest(args.timing_output_root, args.runtime_root)
    atomic_write_json(args.output, footprint)
    print(f"wrote footprint manifest sha256={sha256_file(args.output)}")
    return 0


def _admit(args: argparse.Namespace) -> int:
    ensure_output_outside_roots(args.output, args.timing_output_root, args.runtime_root)
    admission = produce_resource_admission(
        timing_pilot_path=args.timing_pilot,
        scheduler_record_path=args.scheduler_record,
        footprint_manifest_path=args.footprint_manifest,
        timing_output_root=args.timing_output_root,
        runtime_root=args.runtime_root,
        free_bytes=args.free_bytes,
        source_manifest_path=args.source_manifest,
        rule_path=args.rule,
        verifier_script_path=Path(__file__).resolve(),
        verifier_module_path=ROOT / VERIFIER_MODULE_RELATIVE_PATH,
    )
    atomic_write_json(args.output, admission)
    print(
        "resource admission "
        f"status={admission['status']} sha256={sha256_file(args.output)} "
        f"endpoint_values=false"
    )
    return 0 if admission["status"] == "PASS-64" else 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    footprint = subparsers.add_parser(
        "footprint", help="write a no-follow exact file-size manifest for timing/runtime roots"
    )
    footprint.add_argument("--timing-output-root", type=Path, required=True)
    footprint.add_argument("--runtime-root", type=Path, required=True)
    footprint.add_argument("--output", type=Path, required=True)
    footprint.set_defaults(handler=_write_footprint)

    admit = subparsers.add_parser(
        "admit", help="write PASS-64, HOLD, or FAIL resource-admission evidence"
    )
    admit.add_argument("--timing-pilot", type=Path, required=True)
    admit.add_argument("--scheduler-record", type=Path, required=True)
    admit.add_argument("--footprint-manifest", type=Path, required=True)
    admit.add_argument("--timing-output-root", type=Path, required=True)
    admit.add_argument("--runtime-root", type=Path, required=True)
    admit.add_argument("--free-bytes", type=int, required=True)
    admit.add_argument("--source-manifest", type=Path, default=ROOT / SOURCE_MANIFEST_RELATIVE_PATH)
    admit.add_argument("--rule", type=Path, default=ROOT / RULE_RELATIVE_PATH)
    admit.add_argument("--output", type=Path, required=True)
    admit.set_defaults(handler=_admit)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ResourceAdmissionError as error:
        print(f"resource admission failed code={error.code}", file=sys.stderr)
        return 2
    except OSError:
        print("resource admission failed code=io_error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
