#!/usr/bin/env python3
"""Fetch one exact EMIT L2B MIN/MINUNCERT pair from NASA Earthdata.

Authentication is read by ``earthaccess`` from the environment.  Run this
script under the repository's Doppler command shown in the E4 execution plan.
The resolver refuses missing, duplicate, cross-version, or cross-acquisition
pairs and validates the downloaded NetCDF metadata before writing its manifest.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tanager_minmap.emit_l2b import (
    EMIT_L2B_SHORT_NAME,
    EmitL2BSourcePair,
    PinnedEmitL2AInput,
    ProductIdentity,
    l2b_identity_evidence,
    load_pinned_emit_l2a_input,
    parse_product_identity,
    sha256_file,
    validate_emit_l2b_source_pair,
    validate_l2b_identity_against_l2a,
    write_strict_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_MANIFEST = ROOT / "docs" / "input_manifest.json"


@dataclass(frozen=True)
class PairLinks:
    """Exact download links for one catalog-resolved product pair."""

    min_url: str
    minuncert_url: str
    min_name: str
    minuncert_name: str
    granule_concept_id: str
    granule_revision_id: int
    collection_concept_id: str
    granule_ur: str


def _result_links(result: Any) -> tuple[str, ...]:
    links = result.data_links() if callable(getattr(result, "data_links", None)) else ()
    return tuple(str(link) for link in links)


def _safe_url(url: str) -> str:
    """Remove query strings and fragments before recording catalog provenance."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _result_catalog_identity(result: Any) -> tuple[str, int, str, str]:
    try:
        meta = result["meta"]
        umm = result["umm"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("CMR granule result lacks meta/umm provenance") from error
    if not isinstance(meta, dict) or not isinstance(umm, dict):
        raise RuntimeError("CMR granule result meta/umm provenance is invalid")
    concept_id = str(meta.get("concept-id", "")).strip()
    revision_text = str(meta.get("revision-id", "")).strip()
    granule_ur = str(umm.get("GranuleUR", "")).strip()
    collection_concept_id = str(meta.get("collection-concept-id", "")).strip()
    if (
        not concept_id
        or not revision_text.isdigit()
        or int(revision_text) <= 0
        or not collection_concept_id
        or not granule_ur
    ):
        raise RuntimeError(
            "CMR granule result lacks concept, revision, collection, or GranuleUR identity"
        )
    return concept_id, int(revision_text), collection_concept_id, granule_ur


def resolve_pair_links(results: Sequence[Any], expected_l2a_identity: ProductIdentity) -> PairLinks:
    """Resolve both assets from exactly one versioned CMR granule result."""
    granule_prefix = (
        f"{expected_l2a_identity.acquisition}_{expected_l2a_identity.orbit}_"
        f"{expected_l2a_identity.scene}"
    )
    complete: list[PairLinks] = []
    incomplete: list[tuple[int, tuple[str, str, str, str], tuple[str, ...]]] = []
    for result_index, result in enumerate(results):
        candidates: dict[tuple[str, str, str, str], dict[str, tuple[str, str]]] = {}
        for url in _result_links(result):
            parsed_url = urlsplit(url)
            if parsed_url.scheme not in {"http", "https"}:
                continue
            name = Path(parsed_url.path).name
            try:
                identity = parse_product_identity(name)
            except ValueError:
                continue
            observed_prefix = f"{identity.acquisition}_{identity.orbit}_{identity.scene}"
            if observed_prefix != granule_prefix:
                continue
            key = (identity.version, identity.acquisition, identity.orbit, identity.scene)
            packet = candidates.setdefault(key, {})
            if identity.kind in packet and packet[identity.kind][0] != url:
                raise RuntimeError(f"duplicate {identity.kind} links for EMIT L2B pair {key}")
            packet[identity.kind] = (url, name)
        for identity, packet in candidates.items():
            if set(packet) != {"MIN", "MINUNCERT"}:
                incomplete.append((result_index, identity, tuple(sorted(packet))))
                continue
            concept_id, revision_id, collection_id, granule_ur = _result_catalog_identity(result)
            if granule_ur != Path(packet["MIN"][1]).stem:
                raise RuntimeError(
                    "CMR GranuleUR does not equal the resolved MIN filename stem: "
                    f"{granule_ur!r} != {Path(packet['MIN'][1]).stem!r}"
                )
            complete.append(
                PairLinks(
                    min_url=packet["MIN"][0],
                    minuncert_url=packet["MINUNCERT"][0],
                    min_name=packet["MIN"][1],
                    minuncert_name=packet["MINUNCERT"][1],
                    granule_concept_id=concept_id,
                    granule_revision_id=revision_id,
                    collection_concept_id=collection_id,
                    granule_ur=granule_ur,
                )
            )
    if incomplete:
        raise RuntimeError(f"catalog returned incomplete EMIT L2B pair candidates: {incomplete!r}")
    if len(complete) != 1:
        versions = [parse_product_identity(pair.min_name).version for pair in complete]
        raise RuntimeError(
            "catalog must resolve both assets from exactly one CMR granule result for "
            f"{granule_prefix}; found versions={versions!r}"
        )
    return complete[0]


def _fetch_manifest_payload(
    pair: EmitL2BSourcePair,
    pinned_l2a: PinnedEmitL2AInput,
    *,
    input_manifest: Path,
    granule_prefix: str,
    links: PairLinks,
    catalog_resolved_at_utc: str,
    retrieval_mode: str,
    downloaded_at_utc: str | None,
) -> dict[str, Any]:
    """Build the strict v4 fetch manifest after all identity checks pass."""
    if retrieval_mode not in {"fresh_download", "verified_existing_pair"}:
        raise ValueError(f"unsupported L2B retrieval mode: {retrieval_mode!r}")
    if (retrieval_mode == "fresh_download") != (downloaded_at_utc is not None):
        raise ValueError("downloaded_at_utc must be set only for a fresh download")
    return {
        "schema_version": "emit-l2b-fetch/v4",
        "catalog_resolved_at_utc": catalog_resolved_at_utc,
        "retrieval_mode": retrieval_mode,
        "downloaded_at_utc": downloaded_at_utc,
        "collection": EMIT_L2B_SHORT_NAME,
        "granule_prefix": granule_prefix,
        "identity": pair.identity,
        "identity_evidence": l2b_identity_evidence(pair),
        "pinned_l2a": {
            "input_manifest_id": pinned_l2a.input_id,
            "input_manifest_sha256": sha256_file(input_manifest),
            "logical_path": pinned_l2a.logical_path,
            "filename": pinned_l2a.path.name,
            "size_bytes": pinned_l2a.size_bytes,
            "sha256": pinned_l2a.sha256,
            "identity": pinned_l2a.identity,
        },
        "cmr_granule": {
            "concept_id": links.granule_concept_id,
            "revision_id": links.granule_revision_id,
            "collection_concept_id": links.collection_concept_id,
            "granule_ur": links.granule_ur,
            "single_result_pair": True,
        },
        "inputs": [
            {
                "role": "MIN",
                "filename": pair.min_path.name,
                "size_bytes": pair.min_path.stat().st_size,
                "sha256": pair.min_sha256,
                "catalog_url": _safe_url(links.min_url),
                "global_metadata": pair.min_metadata,
            },
            {
                "role": "MINUNCERT",
                "filename": pair.minuncert_path.name,
                "size_bytes": pair.minuncert_path.stat().st_size,
                "sha256": pair.minuncert_sha256,
                "catalog_url": _safe_url(links.minuncert_url),
                "global_metadata": pair.minuncert_metadata,
            },
        ],
        "unavailable_reason": None,
    }


def fetch_pair(
    granule_prefix: str,
    output_dir: Path,
    *,
    input_manifest: Path = DEFAULT_INPUT_MANIFEST,
) -> tuple[Path, Path]:
    """Resolve, download when absent, validate, and manifest one L2B pair."""
    if input_manifest.resolve() != DEFAULT_INPUT_MANIFEST.resolve():
        raise ValueError("E4 fetch must be bound to repository docs/input_manifest.json")
    pinned_l2a = load_pinned_emit_l2a_input(input_manifest)
    expected_prefix = (
        f"{pinned_l2a.identity.acquisition}_{pinned_l2a.identity.orbit}_{pinned_l2a.identity.scene}"
    )
    if granule_prefix != expected_prefix:
        raise ValueError(
            "requested L2B granule identity differs from the byte-verified pinned L2A filename"
        )
    import earthaccess

    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name=EMIT_L2B_SHORT_NAME,
        granule_name=f"*{granule_prefix}*",
        count=100,
    )
    links = resolve_pair_links(results, pinned_l2a.identity)
    catalog_resolved_at_utc = datetime.now(UTC).isoformat()
    min_path = output_dir / links.min_name
    minuncert_path = output_dir / links.minuncert_name
    existence = (min_path.is_file(), minuncert_path.is_file())
    if any(existence) and not all(existence):
        raise RuntimeError("refusing to combine a cached partial pair with a fresh download")
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_mode = "verified_existing_pair" if all(existence) else "fresh_download"
    downloaded_at_utc = None
    if not all(existence):
        earthaccess.download([links.min_url, links.minuncert_url], str(output_dir))
        downloaded_at_utc = datetime.now(UTC).isoformat()
    pair = validate_emit_l2b_source_pair(min_path, minuncert_path)
    validate_l2b_identity_against_l2a(pair.identity, pinned_l2a.identity)
    write_strict_json(
        output_dir / "download_manifest.json",
        _fetch_manifest_payload(
            pair,
            pinned_l2a,
            input_manifest=input_manifest,
            granule_prefix=granule_prefix,
            links=links,
            catalog_resolved_at_utc=catalog_resolved_at_utc,
            retrieval_mode=retrieval_mode,
            downloaded_at_utc=downloaded_at_utc,
        ),
    )
    return min_path, minuncert_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--granule-prefix",
        required=True,
        help="exact YYYYMMDDTHHMMSS_orbit_scene suffix, without product version",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    min_path, minuncert_path = fetch_pair(
        args.granule_prefix,
        args.output_dir,
        input_manifest=args.input_manifest,
    )
    print(f"validated {min_path.name} + {minuncert_path.name}")


if __name__ == "__main__":
    main()
