"""Build a deterministic SHA-256 manifest of scientific source inputs.

The manifest records bytes consumed by the current study, not generated maps
or figures. Package resources are resolved through ``importlib.resources`` so
the same command works with an installed public ``tanager-spec`` distribution.

Run::

    uv run python scripts/build_input_manifest.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from tanager_rocks.config import SITES, TANAGER_SR_ASSET
from tanager_rocks.pipeline import EMIT_GRANULE_URS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "input_manifest.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of ``path`` using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_inputs() -> list[tuple[str, str, Path]]:
    raw = ROOT / "data" / "raw"
    records = [
        (
            f"tanager-{site.site_id}-{index + 1}",
            f"data/raw/{scene_id}_{TANAGER_SR_ASSET}.h5",
            raw / f"{scene_id}_{TANAGER_SR_ASSET}.h5",
        )
        for site in SITES.values()
        for index, scene_id in enumerate(site.scene_ids)
    ]
    records.extend(
        [
            (
                "emit-goldfield-rfl",
                f"data/raw/emit/{EMIT_GRANULE_URS['goldfield']}.nc",
                raw / "emit" / f"{EMIT_GRANULE_URS['goldfield']}.nc",
            ),
            (
                "usgs-splib07a-archive",
                "data/speclib/ASCIIdata_splib07a.zip",
                ROOT / "data" / "speclib" / "ASCIIdata_splib07a.zip",
            ),
            (
                "rockwell-aster-img",
                "data/reference/raw/aster_southwest_aa61_v8_1-17-17.img",
                ROOT / "data" / "reference" / "raw" / "aster_southwest_aa61_v8_1-17-17.img",
            ),
            (
                "rockwell-aster-ige",
                "data/reference/raw/aster_southwest_aa61_v8_1-17-17.ige",
                ROOT / "data" / "reference" / "raw" / "aster_southwest_aa61_v8_1-17-17.ige",
            ),
            ("dependency-lock", "uv.lock", ROOT / "uv.lock"),
        ]
    )
    return records


def _package_inputs() -> list[tuple[str, str, Path]]:
    package_root = files("tanager_spec")
    return [
        (
            f"tanager-spec-{sensor.lower()}-srf",
            f"package:tanager_spec/data/{sensor}_SRF.csv",
            Path(str(package_root.joinpath("data", f"{sensor}_SRF.csv"))),
        )
        for sensor in ("S2A", "S2B")
    ]


def build_manifest() -> dict[str, object]:
    """Hash every declared source input and return a sorted manifest."""
    inputs = []
    for input_id, logical_path, path in [*_repo_inputs(), *_package_inputs()]:
        if not path.is_file():
            raise FileNotFoundError(f"missing required input {input_id}: {path}")
        print(f"hashing {logical_path}", flush=True)
        inputs.append(
            {
                "id": input_id,
                "logical_path": logical_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "1.0",
        "hash_algorithm": "sha256",
        "scope": "Scientific source inputs used by the current Tanager Rocks study",
        "inputs": sorted(inputs, key=lambda item: str(item["id"])),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
