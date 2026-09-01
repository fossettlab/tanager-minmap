#!/usr/bin/env python3
"""Run the preregistered E4 EMIT L2B cross-product concordance packet.

The command requires a fetched, metadata-validated MIN/MINUNCERT pair, an
authoritative M2 block handoff, and a versioned ontology CSV.  It derives the
unchanged Tanager feature-depth and MTMF score fields, area-averages continuous
scores to native L2B support, and keeps all L2B categorical fields on the GLT
grid.  Fit and uncertainty are descriptive only; this driver defines no
quality threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import rasterio
from tanager_spec.io import load_tanager_sr_hdf5

from tanager_rocks.emit_l2b import (
    GROUPS,
    EmitL2BPair,
    EmitL2BSourcePair,
    OntologyEntry,
    PinnedEmitL2AInput,
    ProductMismatchError,
    RasterGeometry,
    area_average_continuous,
    block_footprint_support,
    compute_endpoint_metrics,
    joint_support_mask,
    l2b_identity_evidence,
    load_emit_l2b_metadata,
    load_emit_l2b_pair,
    load_m2_block_scales,
    load_pinned_emit_l2a_input,
    paired_block_bootstrap,
    read_ontology_crosswalk,
    sha256_file,
    sha256_tree,
    summarize_bootstrap_interval,
    summarize_spatial_null,
    validate_emit_l2b_source_pair,
    validate_exchangeable_block_packets,
    validate_l2b_identity_against_l2a,
    validate_ontology_crosswalk,
    whole_block_spatial_nulls,
    write_strict_json,
)
from tanager_rocks.emit_l2b_nonresult import (
    NonResultError,
    atomic_write_bundle,
    canonical_json_bytes,
    csv_bytes,
    read_regular_bytes,
    sha256_regular_file,
    source_identity_payload,
    source_mineral_rows,
    strict_json_load_bytes,
    validate_decision_record,
    validate_legacy_synthetic_resource_policy,
    validate_resource_policy,
    verify_nonresult_bundle,
    verify_resource_admission_bundle,
)
from tanager_rocks.features import build_feature_defs, diagnostic_feature_maps
from tanager_rocks.quality import mask_tanager_scene
from tanager_rocks.sensor_ablation import FDR_ALPHA, support_governance
from tanager_rocks.spatial_validation import (
    BOOTSTRAP_REPLICATES,
    FINITE_REPLICATE_FRACTION,
    PERMUTATION_REPLICATES,
    SEED,
    benjamini_hochberg,
)
from tanager_rocks.speclib import load_library, select_endmembers
from tanager_rocks.unmix import mtmf

ROOT = Path(__file__).resolve().parents[1]
M2_PROTOCOL = ROOT / "docs" / "m2_spatial_validation_preregistration.md"
E4_PLAN = ROOT / "docs" / "m3_external_validation_execution_plan.md"
CANONICAL_M2_BLOCK_MANIFEST = (
    ROOT / "data" / "processed" / "spatial_validation" / "block_manifest.json"
)
DEFAULT_INPUT_MANIFEST = ROOT / "docs" / "input_manifest.json"
DEFAULT_SPECLIB = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
TANAGER_SCENE_ID = "20240925_185504_87_4001"
SITE_ID = "goldfield"
IDENTITY_NODATA = np.iinfo(np.int32).min
SECONDARY_BH_FAMILY = "compatible_mineral_secondary"
INFERENTIAL_METRICS = ("rank_auc", "spearman_band_depth")
DRAW_METRIC_NAMES = {"auc": "rank_auc", "spearman": "spearman_band_depth"}
NULL_DIRECTIONS = {"rank_auc": 0.5, "spearman_band_depth": 0.0}
FAILURE_CODES = {
    "included_joint_support": 0,
    "incomplete_or_halo_m2_footprint": 1,
    "footprint_crosses_m2_block_boundary": 2,
    "invalid_l2b_glt_support": 3,
    "invalid_tanager_qa_support": 4,
    "nonfinite_tanager_score": 5,
    "invalid_l2b_identity": 6,
    "invalid_l2b_band_depth": 7,
}


class OutputDirectoryError(ValueError):
    """Raised before execution when an output directory contains prior artifacts."""


def _require_expected_sha256(path: Path, expected: str | None, *, label: str) -> str:
    """Verify a caller-pinned protocol input before any analytical input is loaded."""
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"{label} requires a caller-supplied expected SHA-256")
    observed = sha256_file(path)
    if observed != expected.casefold():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected.casefold()} observed={observed}"
        )
    return observed


def _require_canonical_m2_manifest(path: Path, expected: str | None) -> tuple[Path, str]:
    """Bind E4 to the accepted M2 handoff before loading analytical inputs."""
    supplied = path.resolve()
    canonical = CANONICAL_M2_BLOCK_MANIFEST.resolve()
    if supplied != canonical:
        raise ValueError(f"E4 requires the canonical accepted M2 block manifest: {canonical}")
    observed = _require_expected_sha256(
        canonical,
        expected,
        label="canonical M2 block manifest",
    )
    return canonical, observed


def _strict_json_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant in {path}: {value}")
        ),
    )


def _require_utc_timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"download manifest {field} must be a UTC timestamp")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"download manifest {field} is not an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"download manifest {field} must include a UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"download manifest {field} must be UTC")
    return text


def _require_catalog_url(value: Any, *, role: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"download manifest {role} catalog URL is missing")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"download manifest {role} catalog URL is invalid")
    if parts.query or parts.fragment:
        raise ValueError(f"download manifest {role} catalog URL is not sanitized")
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if value is None or isinstance(value, float) and not np.isfinite(value)
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _git_revision() -> dict[str, str | None]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return {"revision": result.stdout.strip(), "unavailable_reason": None}
    return {
        "revision": None,
        "unavailable_reason": f"git_rev_parse_failed_exit_{result.returncode}",
    }


def _validate_fetch_manifest(
    pair: EmitL2BSourcePair | EmitL2BPair,
    pinned_l2a: PinnedEmitL2AInput,
    *,
    input_manifest_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    if pair.min_path.parent != pair.minuncert_path.parent:
        raise ValueError("MIN and MINUNCERT must share the fetch-manifest directory")
    path = pair.min_path.parent / "download_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"required L2B access-date manifest is missing: {path}")
    payload = _strict_json_load(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "emit-l2b-fetch/v4"
        or payload.get("collection") != "EMITL2BMIN"
    ):
        raise ValueError("download manifest lacks the required v4 schema or collection")
    _require_utc_timestamp(payload.get("catalog_resolved_at_utc"), field="catalog_resolved_at_utc")
    retrieval_mode = payload.get("retrieval_mode")
    if retrieval_mode not in {"fresh_download", "verified_existing_pair"}:
        raise ValueError("download manifest retrieval_mode is invalid")
    downloaded_at_utc = payload.get("downloaded_at_utc")
    if retrieval_mode == "fresh_download":
        _require_utc_timestamp(downloaded_at_utc, field="downloaded_at_utc")
    elif downloaded_at_utc is not None:
        raise ValueError("verified existing pair must not claim downloaded_at_utc")
    cmr = payload.get("cmr_granule")
    if (
        not isinstance(cmr, dict)
        or not str(cmr.get("concept_id", "")).strip()
        or isinstance(cmr.get("revision_id"), bool)
        or not isinstance(cmr.get("revision_id"), int)
        or cmr["revision_id"] <= 0
        or not str(cmr.get("collection_concept_id", "")).strip()
        or cmr.get("granule_ur") != pair.min_path.stem
        or cmr.get("single_result_pair") is not True
    ):
        raise ValueError(
            "download manifest lacks exact single-result CMR collection/GranuleUR evidence"
        )
    expected_identity = asdict(pair.identity)
    if payload.get("identity") != expected_identity:
        raise ValueError("download manifest identity differs from the L2B pair")
    if payload.get("identity_evidence") != l2b_identity_evidence(pair):
        raise ValueError("download manifest identity_evidence differs from validated files")
    expected_prefix = f"{pair.identity.acquisition}_{pair.identity.orbit}_{pair.identity.scene}"
    if payload.get("granule_prefix") != expected_prefix:
        raise ValueError("download manifest granule prefix differs from the L2B pair")
    expected_l2a = {
        "input_manifest_id": pinned_l2a.input_id,
        "input_manifest_sha256": input_manifest_sha256,
        "logical_path": pinned_l2a.logical_path,
        "filename": pinned_l2a.path.name,
        "size_bytes": pinned_l2a.size_bytes,
        "sha256": pinned_l2a.sha256,
        "identity": asdict(pinned_l2a.identity),
    }
    if payload.get("pinned_l2a") != expected_l2a:
        raise ValueError("download manifest is not bound to the byte-verified pinned L2A")
    inputs = payload.get("inputs")
    if (
        not isinstance(inputs, list)
        or len(inputs) != 2
        or not all(isinstance(record, dict) for record in inputs)
    ):
        raise ValueError("download manifest must contain exactly two product inputs")
    records_by_role = {record.get("role"): record for record in inputs}
    if len(records_by_role) != 2 or set(records_by_role) != {"MIN", "MINUNCERT"}:
        raise ValueError("download manifest must contain unique MIN and MINUNCERT records")
    expected_inputs = {
        "MIN": {
            "filename": pair.min_path.name,
            "size_bytes": pair.min_path.stat().st_size,
            "sha256": pair.min_sha256,
            "global_metadata": pair.min_metadata,
        },
        "MINUNCERT": {
            "filename": pair.minuncert_path.name,
            "size_bytes": pair.minuncert_path.stat().st_size,
            "sha256": pair.minuncert_sha256,
            "global_metadata": pair.minuncert_metadata,
        },
    }
    for role, expected in expected_inputs.items():
        record = records_by_role[role]
        if any(record.get(field) != value for field, value in expected.items()):
            raise ValueError(f"download manifest {role} filename, size, hash, or metadata differs")
        _require_catalog_url(record.get("catalog_url"), role=role)
    return path, payload


def _manifest_record(payload: Mapping[str, Any], input_id: str) -> dict[str, Any]:
    records = payload.get("inputs")
    if not isinstance(records, list):
        raise ValueError("scientific input manifest has no input records")
    matches = [
        record for record in records if isinstance(record, dict) and record.get("id") == input_id
    ]
    if len(matches) != 1:
        raise ValueError(f"scientific input manifest must contain exactly one {input_id!r}")
    return matches[0]


def _resolve_manifest_path(manifest_path: Path, logical_path: Any) -> Path:
    if not isinstance(logical_path, str) or not logical_path or Path(logical_path).is_absolute():
        raise ValueError(
            "scientific input logical paths must be non-empty repository-relative paths"
        )
    root = manifest_path.resolve().parent.parent
    resolved = (root / logical_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"scientific input path escapes repository root: {logical_path}"
        ) from error
    return resolved


def _verify_manifest_file(
    manifest_path: Path, record: Mapping[str, Any], *, expected_path: Path | None = None
) -> dict[str, Any]:
    path = _resolve_manifest_path(manifest_path, record.get("logical_path"))
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"declared input path does not match requested path: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"declared scientific input is missing: {path}")
    size = path.stat().st_size
    expected_size = record.get("size_bytes")
    expected_hash = record.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or size != expected_size
        or not isinstance(expected_hash, str)
        or sha256_file(path) != expected_hash
    ):
        raise ValueError(f"declared scientific input size or SHA-256 differs: {path}")
    return {
        "id": record.get("id"),
        "logical_path": record.get("logical_path"),
        "resolved_path": str(path),
        "size_bytes": size,
        "sha256": expected_hash,
    }


def _validate_speclib_extraction(archive_path: Path, library_dir: Path) -> dict[str, Any]:
    """Verify the complete extracted library byte-for-byte against its archive."""
    if not library_dir.is_dir():
        raise FileNotFoundError(f"spectral-library extraction is missing: {library_dir}")
    with zipfile.ZipFile(archive_path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        roots = {Path(info.filename).parts[0] for info in files if Path(info.filename).parts}
        if len(roots) != 1 or not files:
            raise ValueError("spectral-library archive lacks one non-empty extraction root")
        archive_root = next(iter(roots))
        members: dict[Path, zipfile.ZipInfo] = {}
        for info in files:
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts or member.parts[0] != archive_root:
                raise ValueError(f"unsafe spectral-library archive member: {info.filename}")
            relative = Path(*member.parts[1:])
            if not relative.parts or relative in members:
                raise ValueError("spectral-library archive contains duplicate or root-only files")
            members[relative] = info
        extracted = {
            path.relative_to(library_dir) for path in library_dir.rglob("*") if path.is_file()
        }
        if extracted != set(members):
            missing = sorted(str(path) for path in set(members) - extracted)
            extra = sorted(str(path) for path in extracted - set(members))
            raise ValueError(
                "spectral-library extraction differs from pinned archive members; "
                f"missing={missing!r}, extra={extra!r}"
            )
        for relative, info in members.items():
            archive_digest = hashlib.sha256()
            with archive.open(info) as source:
                while chunk := source.read(1024 * 1024):
                    archive_digest.update(chunk)
            if archive_digest.hexdigest() != sha256_file(library_dir / relative):
                raise ValueError(f"spectral-library extraction differs from archive: {relative}")
    return {
        "path": str(library_dir),
        "sha256_tree": sha256_tree(library_dir),
        "verified_archive_members": len(members),
    }


def _validate_frozen_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], PinnedEmitL2AInput]:
    manifest_path = args.input_manifest.resolve()
    if manifest_path != DEFAULT_INPUT_MANIFEST.resolve():
        raise ValueError("E4 inputs must be bound to the repository docs/input_manifest.json")
    payload = _strict_json_load(manifest_path)
    if not isinstance(payload, dict) or payload.get("hash_algorithm") != "sha256":
        raise ValueError("scientific input manifest must declare SHA-256")
    records = payload.get("inputs")
    if not isinstance(records, list) or not records:
        raise ValueError("scientific input manifest has no inputs")
    ids = [record.get("id") for record in records if isinstance(record, dict)]
    if len(ids) != len(records) or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("scientific input manifest IDs must be non-empty and unique")
    pinned_l2a = load_pinned_emit_l2a_input(manifest_path)
    scene = _verify_manifest_file(
        manifest_path,
        _manifest_record(payload, "tanager-goldfield-1"),
        expected_path=args.tanager_scene,
    )
    archive = _verify_manifest_file(
        manifest_path,
        _manifest_record(payload, "usgs-splib07a-archive"),
    )
    archive_path = Path(archive["resolved_path"])
    if args.speclib.resolve() != archive_path.with_suffix("").resolve():
        raise ValueError("spectral-library path is not the pinned archive extraction")
    extraction = _validate_speclib_extraction(archive_path, args.speclib.resolve())
    return {
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "emit_l2a": {
            "id": pinned_l2a.input_id,
            "logical_path": pinned_l2a.logical_path,
            "resolved_path": str(pinned_l2a.path),
            "size_bytes": pinned_l2a.size_bytes,
            "sha256": pinned_l2a.sha256,
            "identity": asdict(pinned_l2a.identity),
        },
        "tanager_scene": scene,
        "speclib_archive": archive,
        "speclib_extraction": extraction,
    }, pinned_l2a


def _tanager_scores(
    scene_path: Path, speclib_path: Path
) -> tuple[dict[str, np.ndarray], np.ndarray, Any, str, dict[str, Any]]:
    cube, wavelengths = load_tanager_sr_hdf5(scene_path)
    masked, quality = mask_tanager_scene(cube, wavelengths, scene_path)
    endmembers = select_endmembers(load_library(speclib_path, wavelengths))
    depths = diagnostic_feature_maps(
        masked, wavelengths, build_feature_defs(wavelengths, speclib_path)
    )
    matched = mtmf(masked, endmembers)
    scores = {
        **{f"feature:{name}": np.asarray(depths[name].values, dtype=float) for name in depths},
        **{
            f"mtmf:{name[:-3]}": np.asarray(matched[name].values, dtype=float)
            for name in matched
            if str(name).endswith("_mf")
        },
    }
    qa_valid = np.any(np.isfinite(masked.values), axis=0)
    crs = masked.rio.crs
    if crs is None:
        raise ValueError("Tanager scene has no CRS")
    return scores, qa_valid, masked.rio.transform(), crs.to_string(), asdict(quality)


def _write_field(
    path: Path, values: np.ndarray, geometry: RasterGeometry, *, identity: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    source = np.asarray(values)
    if identity:
        finite = source[np.isfinite(source)]
        if np.any(finite != np.floor(finite)):
            raise ValueError("mineral identity contains non-integer values")
        if np.any(finite < np.iinfo(np.int32).min + 1) or np.any(finite > np.iinfo(np.int32).max):
            raise ValueError("mineral identity exceeds int32 GeoTIFF storage")
        raster = np.where(np.isfinite(source), source, IDENTITY_NODATA).astype(np.int32)
        dtype = "int32"
        nodata: int | float = IDENTITY_NODATA
    else:
        raster = source.astype(np.float32)
        dtype = "float32"
        nodata = float("nan")
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        height=geometry.shape[0],
        width=geometry.shape[1],
        count=1,
        dtype=dtype,
        crs=geometry.crs,
        transform=geometry.transform,
        nodata=nodata,
        compress="deflate",
    ) as dataset:
        dataset.write(raster, 1)
    temporary.replace(path)


def _endpoint_groups(
    entries: Sequence[OntologyEntry],
) -> list[tuple[tuple[str, str, str, int], list[OntologyEntry]]]:
    grouped: dict[tuple[str, str, str, int], list[OntologyEntry]] = defaultdict(list)
    for entry in entries:
        if entry.mapping != "unmapped":
            grouped[(entry.mapping, entry.target, entry.tanager_score, entry.group)].append(entry)
    return list(grouped.items())


def _support_ledger(
    *,
    block_ids: np.ndarray,
    footprint_boundary: np.ndarray,
    incomplete_footprint: np.ndarray,
    glt_valid: np.ndarray,
    qa_valid: np.ndarray,
    score: np.ndarray,
    identity: np.ndarray,
    depth: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    masks = (
        ("incomplete_or_halo_m2_footprint", incomplete_footprint),
        ("footprint_crosses_m2_block_boundary", footprint_boundary),
        ("invalid_l2b_glt_support", ~glt_valid),
        ("invalid_tanager_qa_support", ~qa_valid),
        ("nonfinite_tanager_score", ~np.isfinite(score)),
        ("invalid_l2b_identity", ~np.isfinite(identity)),
        ("invalid_l2b_band_depth", ~np.isfinite(depth)),
    )
    denominator = int(np.size(block_ids))
    remaining = np.ones(block_ids.shape, dtype=bool)
    failure = np.full(block_ids.shape, FAILURE_CODES["included_joint_support"], dtype=np.int16)
    rows = []
    for reason, excluded in masks:
        selected = remaining & excluded
        failure[selected] = FAILURE_CODES[reason]
        count = int(np.count_nonzero(selected))
        rows.append(
            {
                "reason": reason,
                "count": count,
                "denominator": denominator,
                "fraction": count / denominator if denominator else None,
            }
        )
        remaining &= ~excluded
    joint = joint_support_mask(
        score=score,
        mineral_id=identity,
        band_depth=depth,
        qa_valid=qa_valid,
        glt_valid=glt_valid,
        block_ids=block_ids,
    )
    if not np.array_equal(remaining, joint):
        raise ValueError("support ledger does not equal the governed joint-support mask")
    count = int(np.count_nonzero(joint))
    rows.append(
        {
            "reason": "included_joint_support",
            "count": count,
            "denominator": denominator,
            "fraction": count / denominator if denominator else None,
        }
    )
    if sum(int(row["count"]) for row in rows) != denominator:
        raise ValueError("support exclusion categories are not exhaustive")
    return rows, failure, joint


def _distribution_rows(
    *,
    identity: np.ndarray,
    uncertainty: np.ndarray,
    fit: np.ndarray,
    target_ids: frozenset[int],
    support: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    joint_denominator = int(np.count_nonzero(support))
    categories = {
        "matched": support & np.isin(identity, tuple(target_ids)),
        "unmatched": support & ~np.isin(identity, tuple(target_ids)),
    }
    for category, use in categories.items():
        category_count = int(np.count_nonzero(use))
        for field, values in (("fit", fit), ("band_depth_uncertainty", uncertainty)):
            finite = np.asarray(values, dtype=float)[use & np.isfinite(values)]
            missing_count = category_count - int(finite.size)
            rows.append(
                {
                    "category": category,
                    "field": field,
                    "joint_support_denominator": joint_denominator,
                    "category_count": category_count,
                    "finite_count": int(finite.size),
                    "missing_count": missing_count,
                    "missing_fraction": (
                        missing_count / category_count if category_count else None
                    ),
                    "minimum": float(np.min(finite)) if finite.size else None,
                    "median": float(np.median(finite)) if finite.size else None,
                    "maximum": float(np.max(finite)) if finite.size else None,
                    "unavailable_reason": None if finite.size else "no_finite_values_in_category",
                }
            )
    for field in ("fit", "band_depth_uncertainty"):
        rows.append(
            {
                "category": "tanager_no_call",
                "field": field,
                "joint_support_denominator": joint_denominator,
                "category_count": None,
                "finite_count": None,
                "missing_count": None,
                "missing_fraction": None,
                "minimum": None,
                "median": None,
                "maximum": None,
                "unavailable_reason": "tanager_no_call_rule_not_defined_for_e4",
            }
        )
    return rows


def _failure_summary(output: Path, reason: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_strict_json(
        output / "summary.json",
        {
            "schema_version": "emit-l2b-validation/v1",
            "status": "unavailable",
            "execution_status": "failed_closed",
            "inference_status": "unavailable",
            "claim_status": "no_claim",
            "unavailable_reason": reason,
            "scientific_scope": "cross-product_cross-acquisition_concordance_not_ground_truth",
        },
    )
    (output / "report.md").write_text(
        "# E4 EMIT L2B validation\n\n"
        f"Status: unavailable. Reason: `{reason}`.\n\n"
        "No endpoint, threshold, or scientific result was produced.\n",
        encoding="utf-8",
    )


def _unavailable_interval(metric: str, reason: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "lower_95": None,
        "upper_95": None,
        "scheduled_replicates": BOOTSTRAP_REPLICATES,
        "valid_replicates": 0,
        "finite_fraction": 0.0,
        "gate_eligible": False,
        "unavailable_reason": reason,
    }


def _unavailable_null(metric: str, observed: float | None, reason: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "observed": observed,
        "null_lower_95": None,
        "null_median": None,
        "null_upper_95": None,
        "p_value": None,
        "exceedances": None,
        "scheduled_permutations": PERMUTATION_REPLICATES,
        "valid_permutations": 0,
        "finite_fraction": 0.0,
        "gate_eligible": False,
        "unavailable_reason": reason,
    }


def _apply_secondary_bh(rows: list[dict[str, Any]]) -> None:
    secondary: list[int] = []
    for index, row in enumerate(rows):
        row["bh_family"] = None
        row["bh_adjusted_p_value"] = None
        row["bh_reject_at_fdr_0_05"] = None
        if row["metric"] not in INFERENTIAL_METRICS:
            row["bh_unavailable_reason"] = "descriptive_metric_not_in_bh_family"
        elif row["mapping"] != "broader":
            row["bh_unavailable_reason"] = "primary_exact_endpoint_not_in_secondary_bh_family"
        else:
            row["bh_family"] = SECONDARY_BH_FAMILY
            secondary.append(index)
    if not secondary:
        return
    if any(
        rows[index].get("null_p_value") is None or not np.isfinite(rows[index]["null_p_value"])
        for index in secondary
    ):
        for index in secondary:
            rows[index]["bh_unavailable_reason"] = (
                "secondary_bh_family_incomplete_due_to_unavailable_null"
            )
        return
    adjusted = benjamini_hochberg([float(rows[index]["null_p_value"]) for index in secondary])
    for index, value in zip(secondary, adjusted, strict=True):
        rows[index]["bh_adjusted_p_value"] = float(value)
        rows[index]["bh_reject_at_fdr_0_05"] = bool(value <= FDR_ALPHA)
        rows[index]["bh_unavailable_reason"] = None


def _finalize_endpoint_status(row: dict[str, Any]) -> None:
    metric = row["metric"]
    support_status = row["support_status"]
    mapping = row["mapping"]
    if metric == "l2b_id_prevalence":
        row["inference_status"] = "descriptive_only"
        row["claim_status"] = "descriptive_prevalence"
        row["claim_reason"] = None
        return
    if support_status == "counts_and_maps_only":
        row["inference_status"] = "counts_and_maps_only"
        row["claim_status"] = "no_inferential_claim"
        row["claim_reason"] = "fewer_than_five_positive_or_negative_bearing_blocks"
        return
    if support_status == "exploratory_only":
        row["inference_status"] = "exploratory_estimate_only"
        row["claim_status"] = "no_confirmatory_claim"
        row["claim_reason"] = "five_to_nine_positive_or_negative_bearing_blocks"
        return
    interval_available = bool(row["bootstrap_interval_gate_eligible"])
    null_available = bool(row["null_gate_eligible"])
    observed_available = row["value"] is not None and np.isfinite(row["value"])
    if mapping == "broader":
        secondary_available = (
            interval_available
            and null_available
            and row["bh_adjusted_p_value"] is not None
            and observed_available
        )
        row["inference_status"] = (
            "secondary_inference_available"
            if secondary_available
            else "secondary_inference_unavailable"
        )
        row["claim_status"] = "secondary_endpoint_no_confirmatory_claim"
        row["claim_reason"] = (
            None if secondary_available else "interval_null_or_bh_adjustment_unavailable"
        )
        return
    if not (interval_available and null_available and observed_available):
        row["inference_status"] = "inference_unavailable"
        row["claim_status"] = "no_confirmatory_claim"
        row["claim_reason"] = "observed_metric_interval_or_whole_block_null_unavailable"
        return
    row["inference_status"] = "confirmatory_eligible"
    lower = row["bootstrap_lower_95"]
    upper = row["bootstrap_upper_95"]
    null_direction = row["null_direction"]
    if lower is not None and lower > null_direction:
        row["claim_status"] = "confirmatory_concordance_supported"
        row["claim_reason"] = None
    elif upper is not None and upper < null_direction:
        row["claim_status"] = "confirmatory_reversed_result"
        row["claim_reason"] = "bootstrap_interval_lies_below_the_null_direction"
    else:
        row["claim_status"] = "confirmatory_concordance_not_supported"
        row["claim_reason"] = "bootstrap_interval_does_not_exclude_the_null_direction"


def _global_result_status(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    claims = {str(row.get("claim_status")) for row in rows}
    inference = {str(row.get("inference_status")) for row in rows}
    if "confirmatory_concordance_supported" in claims:
        return "confirmatory_result_available", "confirmatory_concordance_supported"
    if "confirmatory_reversed_result" in claims:
        return "confirmatory_result_available", "confirmatory_reversed_result"
    if "confirmatory_eligible" in inference:
        return "confirmatory_result_available", "confirmatory_concordance_not_supported"
    if "secondary_inference_available" in inference:
        return "secondary_only", "no_confirmatory_claim"
    if "exploratory_estimate_only" in inference:
        return "exploratory_only", "no_confirmatory_claim"
    if "counts_and_maps_only" in inference:
        return "counts_and_maps_only", "no_inferential_claim"
    return "unavailable", "no_inferential_claim"


def run(args: argparse.Namespace) -> None:
    if not args.tanager_scene.name.startswith(TANAGER_SCENE_ID):
        raise ValueError(f"Tanager scene must be the frozen Goldfield anchor {TANAGER_SCENE_ID}")
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise OutputDirectoryError(
            "output directory must be absent or empty to prevent stale products"
        )

    crosswalk_path = args.ontology_crosswalk.resolve()
    e4_plan_sha256 = _require_expected_sha256(
        E4_PLAN,
        getattr(args, "expected_e4_plan_sha256", None),
        label="E4 plan",
    )
    ontology_sha256 = _require_expected_sha256(
        crosswalk_path,
        getattr(args, "expected_ontology_sha256", None),
        label="ontology crosswalk",
    )
    m2_block_manifest_path, m2_block_manifest_sha256 = _require_canonical_m2_manifest(
        args.block_manifest,
        getattr(args, "expected_m2_block_manifest_sha256", None),
    )

    frozen_inputs, pinned_l2a = _validate_frozen_inputs(args)
    source_pair = validate_emit_l2b_source_pair(args.emit_min, args.emit_minuncert)
    validate_l2b_identity_against_l2a(source_pair.identity, pinned_l2a.identity)
    fetch_manifest_path, fetch_manifest = _validate_fetch_manifest(
        source_pair,
        pinned_l2a,
        input_manifest_sha256=frozen_inputs["manifest"]["sha256"],
    )
    pair = load_emit_l2b_pair(args.emit_min, args.emit_minuncert)
    entries = validate_ontology_crosswalk(
        read_ontology_crosswalk(crosswalk_path),
        pair.mineral_metadata,
        source_root=crosswalk_path.parent,
    )
    endpoint_groups = _endpoint_groups(entries)
    if not endpoint_groups:
        raise ValueError("ontology contains no source-supported compatible endpoint")
    scales = load_m2_block_scales(
        m2_block_manifest_path,
        site=SITE_ID,
        scene_id=TANAGER_SCENE_ID,
    )
    scores, qa_pixels, score_transform, score_crs, quality = _tanager_scores(
        args.tanager_scene, args.speclib
    )
    requested_scores = {key[2] for key, _ in endpoint_groups}
    missing_scores = sorted(requested_scores - set(scores))
    if missing_scores:
        raise ValueError(
            f"ontology names unavailable exact Tanager score fields: {missing_scores!r}"
        )

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "ontology_crosswalk.csv",
        [asdict(entry) for entry in entries],
        tuple(asdict(entries[0])),
    )
    for group in GROUPS:
        fields = pair.groups[group]
        _write_field(
            output / f"group_{group}_mineral_id.tif",
            fields.mineral_id,
            pair.geometry,
            identity=True,
        )
        _write_field(
            output / f"group_{group}_band_depth.tif",
            fields.band_depth,
            pair.geometry,
            identity=False,
        )
        _write_field(
            output / f"group_{group}_band_depth_uncertainty.tif",
            fields.uncertainty,
            pair.geometry,
            identity=False,
        )
        _write_field(output / f"group_{group}_fit.tif", fields.fit, pair.geometry, identity=False)

    qa_fraction = area_average_continuous(
        qa_pixels.astype(float),
        source_transform=score_transform,
        source_crs=score_crs,
        destination=pair.geometry,
    )
    complete_qa = np.isfinite(qa_fraction) & (qa_fraction == 1.0)
    min_geometry_valid = (pair.min_glt_x != 0) & (pair.min_glt_y != 0)
    uncertainty_geometry_valid = (pair.minuncert_glt_x != 0) & (pair.minuncert_glt_y != 0)
    complete_glt = min_geometry_valid & uncertainty_geometry_valid
    projected_scores = {
        name: area_average_continuous(
            scores[name],
            source_transform=score_transform,
            source_crs=score_crs,
            destination=pair.geometry,
        )
        for name in requested_scores
    }
    coordinate_rows, coordinate_columns = np.indices(pair.geometry.shape)
    l2b_x = pair.geometry.transform.c + (coordinate_columns + 0.5) * pair.geometry.transform.a
    l2b_y = pair.geometry.transform.f + (coordinate_rows + 0.5) * pair.geometry.transform.e

    support_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    scale_null_design: dict[str, dict[str, Any]] = {}
    first_failure: np.ndarray | None = None
    failure_map_endpoint: str | None = None

    for scale_name, scale in scales.items():
        footprint = block_footprint_support(scale, pair.geometry)
        block_grid = footprint.block_ids
        packet_use = block_grid.reshape(-1) != 0
        try:
            design = validate_exchangeable_block_packets(
                {
                    "x": l2b_x.reshape(-1)[packet_use],
                    "y": l2b_y.reshape(-1)[packet_use],
                },
                block_grid.reshape(-1)[packet_use].astype(object),
            )
        except ValueError as error:
            null_design_reason = f"nonexchangeable_complete_block_packets:{error}"
            scale_null_design[scale_name] = {
                "status": "unavailable",
                "reason": null_design_reason,
            }
        else:
            null_design_reason = None
            scale_null_design[scale_name] = {
                "status": "available",
                "block_count": len(design.block_ids),
                "packet_size": design.packet_size,
            }

        for (mapping, target, score_name, group), source_entries in endpoint_groups:
            endpoint = f"{mapping}:{target}:group_{group}"
            score = projected_scores[score_name]
            fields = pair.groups[group]
            ledger, failure, joint = _support_ledger(
                block_ids=block_grid,
                footprint_boundary=footprint.crosses_block_boundary,
                incomplete_footprint=footprint.incomplete_or_halo_support,
                glt_valid=complete_glt,
                qa_valid=complete_qa,
                score=score,
                identity=fields.mineral_id,
                depth=fields.band_depth,
            )
            if first_failure is None and scale_name == "L":
                first_failure = failure
                failure_map_endpoint = endpoint
            for row in ledger:
                support_rows.append(
                    {
                        "endpoint": endpoint,
                        "mapping": mapping,
                        "geometry": scale_name,
                        **row,
                    }
                )

            target_ids = frozenset(entry.index for entry in source_entries)
            metrics = compute_endpoint_metrics(
                score=score,
                mineral_id=fields.mineral_id,
                band_depth=fields.band_depth,
                target_ids=target_ids,
                block_ids=block_grid.astype(object),
                joint_support=joint,
            )
            governance = support_governance(metrics.positive_blocks, metrics.negative_blocks)
            distribution_rows.extend(
                {
                    "endpoint": endpoint,
                    "mapping": mapping,
                    "geometry": scale_name,
                    **row,
                }
                for row in _distribution_rows(
                    identity=fields.mineral_id,
                    uncertainty=fields.uncertainty,
                    fit=fields.fit,
                    target_ids=target_ids,
                    support=joint,
                )
            )

            interval_by_metric: dict[str, dict[str, Any]] = {}
            if governance.bootstrap_cis:
                draws = paired_block_bootstrap(
                    score=score,
                    mineral_id=fields.mineral_id,
                    band_depth=fields.band_depth,
                    target_ids=target_ids,
                    block_ids=block_grid.astype(object),
                    joint_support=joint,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=SEED,
                )
                bootstrap_rows.extend(
                    {
                        "endpoint": endpoint,
                        "mapping": mapping,
                        "geometry": scale_name,
                        **asdict(draw),
                        "metric": DRAW_METRIC_NAMES[draw.metric],
                    }
                    for draw in draws
                )
                interval_by_metric = {
                    "rank_auc": asdict(
                        summarize_bootstrap_interval(
                            draws,
                            metric="auc",
                            scheduled_replicates=BOOTSTRAP_REPLICATES,
                        )
                    ),
                    "spearman_band_depth": asdict(
                        summarize_bootstrap_interval(
                            draws,
                            metric="spearman",
                            scheduled_replicates=BOOTSTRAP_REPLICATES,
                        )
                    ),
                }
            else:
                reason = "m2_support_tier_does_not_permit_bootstrap_intervals"
                interval_by_metric = {
                    metric: _unavailable_interval(metric, reason) for metric in INFERENTIAL_METRICS
                }
                for metric in INFERENTIAL_METRICS:
                    bootstrap_rows.append(
                        {
                            "endpoint": endpoint,
                            "mapping": mapping,
                            "geometry": scale_name,
                            "replicate": None,
                            "metric": metric,
                            "value": None,
                            "unavailable_reason": reason,
                        }
                    )

            observed_by_metric = {
                "rank_auc": metrics.auc,
                "spearman_band_depth": metrics.spearman,
            }
            null_by_metric: dict[str, dict[str, Any]] = {}
            if governance.permutation_inference and null_design_reason is None:
                tanager_valid = (complete_qa & np.isfinite(score)).reshape(-1)[packet_use]
                l2b_valid = (
                    complete_glt & np.isfinite(fields.mineral_id) & np.isfinite(fields.band_depth)
                ).reshape(-1)[packet_use]
                packet = {
                    "mineral_id": fields.mineral_id.reshape(-1)[packet_use],
                    "band_depth": fields.band_depth.reshape(-1)[packet_use],
                    "uncertainty": fields.uncertainty.reshape(-1)[packet_use],
                    "fit": fields.fit.reshape(-1)[packet_use],
                    "l2b_valid": l2b_valid,
                    "x": l2b_x.reshape(-1)[packet_use],
                    "y": l2b_y.reshape(-1)[packet_use],
                }
                try:
                    draws = whole_block_spatial_nulls(
                        score=score.reshape(-1)[packet_use],
                        l2b_fields=packet,
                        target_ids=target_ids,
                        block_ids=block_grid.reshape(-1)[packet_use].astype(object),
                        tanager_valid=tanager_valid,
                        observed_joint_support=joint.reshape(-1)[packet_use],
                        permutations=PERMUTATION_REPLICATES,
                        seed=SEED,
                    )
                except ValueError as error:
                    null_reason = f"whole_block_null_design_invalid:{error}"
                else:
                    null_reason = None
                    null_rows.extend(
                        {
                            "endpoint": endpoint,
                            "mapping": mapping,
                            "geometry": scale_name,
                            **asdict(draw),
                            "metric": DRAW_METRIC_NAMES[draw.metric],
                        }
                        for draw in draws
                    )
                    null_by_metric = {
                        "rank_auc": asdict(
                            summarize_spatial_null(
                                draws,
                                metric="auc",
                                observed=metrics.auc,
                                scheduled_permutations=PERMUTATION_REPLICATES,
                            )
                        ),
                        "spearman_band_depth": asdict(
                            summarize_spatial_null(
                                draws,
                                metric="spearman",
                                observed=metrics.spearman,
                                scheduled_permutations=PERMUTATION_REPLICATES,
                            )
                        ),
                    }
                if null_reason is not None:
                    null_by_metric = {
                        metric: _unavailable_null(metric, observed_by_metric[metric], null_reason)
                        for metric in INFERENTIAL_METRICS
                    }
            else:
                null_reason = (
                    null_design_reason
                    if governance.permutation_inference
                    else "m2_support_tier_does_not_permit_spatial_null_inference"
                )
                null_by_metric = {
                    metric: _unavailable_null(metric, observed_by_metric[metric], null_reason)
                    for metric in INFERENTIAL_METRICS
                }
            if not any(
                row["endpoint"] == endpoint
                and row["geometry"] == scale_name
                and row["mapping"] == mapping
                for row in null_rows
            ):
                for metric in INFERENTIAL_METRICS:
                    null_rows.append(
                        {
                            "endpoint": endpoint,
                            "mapping": mapping,
                            "geometry": scale_name,
                            "permutation": None,
                            "metric": metric,
                            "value": None,
                            "unavailable_reason": null_by_metric[metric]["unavailable_reason"],
                        }
                    )

            metric_specs = (
                (
                    "rank_auc",
                    metrics.auc,
                    metrics.auc_n,
                    metrics.auc_unavailable_reason,
                ),
                (
                    "spearman_band_depth",
                    metrics.spearman,
                    metrics.spearman_n,
                    metrics.spearman_unavailable_reason,
                ),
                ("l2b_id_prevalence", metrics.prevalence, metrics.auc_n, None),
            )
            for metric, raw_value, metric_n, metric_reason in metric_specs:
                value = raw_value if governance.effect_estimates else None
                unavailable_reason = (
                    metric_reason
                    if governance.effect_estimates
                    else "insufficient_complete_block_support_for_effect_estimates"
                )
                if metric in INFERENTIAL_METRICS:
                    interval = interval_by_metric[metric]
                    null = null_by_metric[metric]
                    interval_reason = interval["unavailable_reason"]
                    null_reason = null["unavailable_reason"]
                else:
                    interval = _unavailable_interval(
                        metric, "descriptive_prevalence_has_no_bootstrap_interval"
                    )
                    null = _unavailable_null(
                        metric,
                        raw_value,
                        "descriptive_prevalence_has_no_spatial_null_test",
                    )
                    interval_reason = interval["unavailable_reason"]
                    null_reason = null["unavailable_reason"]
                metric_rows.append(
                    {
                        "endpoint": endpoint,
                        "mapping": mapping,
                        "target": target,
                        "tanager_score": score_name,
                        "l2b_group": group,
                        "geometry": scale_name,
                        "metric": metric,
                        "value": value,
                        "metric_n": metric_n,
                        "joint_support_cells": metrics.joint_support_n,
                        "grid_cells": int(np.size(block_grid)),
                        "positive_cells": metrics.auc_positive,
                        "negative_cells": metrics.auc_negative,
                        "positive_blocks": metrics.positive_blocks,
                        "negative_blocks": metrics.negative_blocks,
                        "block_support_tier": governance.status,
                        "support_status": (
                            "secondary_only"
                            if mapping == "broader"
                            and metrics.governance == "confirmatory_eligible"
                            else metrics.governance
                        ),
                        "metric_unavailable_reason": unavailable_reason,
                        "bootstrap_lower_95": interval["lower_95"],
                        "bootstrap_upper_95": interval["upper_95"],
                        "bootstrap_scheduled_replicates": interval["scheduled_replicates"],
                        "bootstrap_valid_replicates": interval["valid_replicates"],
                        "bootstrap_finite_fraction": interval["finite_fraction"],
                        "bootstrap_interval_gate_eligible": interval["gate_eligible"],
                        "bootstrap_unavailable_reason": interval_reason,
                        "null_direction": NULL_DIRECTIONS.get(metric),
                        "null_lower_95": null["null_lower_95"],
                        "null_median": null["null_median"],
                        "null_upper_95": null["null_upper_95"],
                        "null_p_value": null["p_value"],
                        "null_exceedances": null["exceedances"],
                        "null_scheduled_permutations": null["scheduled_permutations"],
                        "null_valid_permutations": null["valid_permutations"],
                        "null_finite_fraction": null["finite_fraction"],
                        "null_gate_eligible": null["gate_eligible"],
                        "null_unavailable_reason": null_reason,
                        "execution_status": "complete",
                    }
                )

    if first_failure is None:
        raise ValueError("no primary L geometry was available for failure mapping")

    _apply_secondary_bh(metric_rows)
    for row in metric_rows:
        _finalize_endpoint_status(row)
    inference_status, claim_status = _global_result_status(metric_rows)

    _write_field(output / "failure_map.tif", first_failure, pair.geometry, identity=True)
    _write_csv(
        output / "support_and_exclusions.csv",
        support_rows,
        ("endpoint", "mapping", "geometry", "reason", "count", "denominator", "fraction"),
    )
    metric_columns = tuple(metric_rows[0])
    _write_csv(output / "metrics.csv", metric_rows, metric_columns)
    _write_csv(output / "endpoint_scale_results.csv", metric_rows, metric_columns)
    _write_csv(
        output / "bootstrap.csv",
        bootstrap_rows,
        (
            "endpoint",
            "mapping",
            "geometry",
            "replicate",
            "metric",
            "value",
            "unavailable_reason",
        ),
    )
    _write_csv(
        output / "spatial_nulls.csv",
        null_rows,
        (
            "endpoint",
            "mapping",
            "geometry",
            "permutation",
            "metric",
            "value",
            "unavailable_reason",
        ),
    )
    null_summary_rows = [
        {
            key: row[key]
            for key in (
                "endpoint",
                "mapping",
                "geometry",
                "metric",
                "null_direction",
                "null_lower_95",
                "null_median",
                "null_upper_95",
                "null_p_value",
                "null_exceedances",
                "null_scheduled_permutations",
                "null_valid_permutations",
                "null_finite_fraction",
                "null_gate_eligible",
                "null_unavailable_reason",
                "bh_family",
                "bh_adjusted_p_value",
                "bh_reject_at_fdr_0_05",
                "bh_unavailable_reason",
            )
        }
        for row in metric_rows
        if row["metric"] in INFERENTIAL_METRICS
    ]
    _write_csv(
        output / "spatial_null_summary.csv",
        null_summary_rows,
        tuple(null_summary_rows[0]),
    )
    _write_csv(
        output / "fit_uncertainty_distributions.csv",
        distribution_rows,
        (
            "endpoint",
            "mapping",
            "geometry",
            "category",
            "field",
            "joint_support_denominator",
            "category_count",
            "finite_count",
            "missing_count",
            "missing_fraction",
            "minimum",
            "median",
            "maximum",
            "unavailable_reason",
        ),
    )

    ontology_evidence = sorted(
        {
            (
                str(
                    (
                        Path(entry.source_path)
                        if Path(entry.source_path).is_absolute()
                        else crosswalk_path.parent / entry.source_path
                    ).resolve()
                ),
                entry.source_sha256,
            )
            for entry in entries
        }
    )
    input_manifest = {
        "schema_version": "emit-l2b-input-manifest/v3",
        "frozen_scientific_inputs": frozen_inputs,
        "inputs": {
            "emit_l2a": frozen_inputs["emit_l2a"],
            "tanager_scene": {
                "path": str(args.tanager_scene),
                "sha256": frozen_inputs["tanager_scene"]["sha256"],
                "input_manifest_id": "tanager-goldfield-1",
            },
            "emit_min": {
                "path": str(pair.min_path),
                "sha256": pair.min_sha256,
                "global_metadata": pair.min_metadata,
            },
            "emit_minuncert": {
                "path": str(pair.minuncert_path),
                "sha256": pair.minuncert_sha256,
                "global_metadata": pair.minuncert_metadata,
            },
            "fetch_manifest": {
                "path": str(fetch_manifest_path),
                "sha256": sha256_file(fetch_manifest_path),
                "catalog_resolved_at_utc": fetch_manifest["catalog_resolved_at_utc"],
                "retrieval_mode": fetch_manifest["retrieval_mode"],
                "downloaded_at_utc": fetch_manifest["downloaded_at_utc"],
                "cmr_granule": fetch_manifest["cmr_granule"],
                "identity_evidence": fetch_manifest["identity_evidence"],
            },
            "ontology": {
                "path": str(crosswalk_path),
                "sha256": ontology_sha256,
                "caller_expected_sha256": args.expected_ontology_sha256.casefold(),
                "version": entries[0].ontology_version,
                "source_evidence": [
                    {"path": path, "sha256": digest} for path, digest in ontology_evidence
                ],
            },
            "m2_block_manifest": {
                "path": str(m2_block_manifest_path),
                "sha256": m2_block_manifest_sha256,
                "caller_expected_sha256": args.expected_m2_block_manifest_sha256.casefold(),
                "scales": {
                    name: {
                        "raster_path": str(scale.source_path),
                        "raster_sha256": scale.source_sha256,
                        "block_side_pixels": scale.block_side_pixels,
                        "halo_pixels": scale.halo_pixels,
                    }
                    for name, scale in scales.items()
                },
            },
            "m2_protocol": {"path": str(M2_PROTOCOL), "sha256": sha256_file(M2_PROTOCOL)},
            "e4_plan": {
                "path": str(E4_PLAN),
                "sha256": e4_plan_sha256,
                "caller_expected_sha256": args.expected_e4_plan_sha256.casefold(),
            },
            "speclib": {
                "path": str(args.speclib),
                "archive_sha256": frozen_inputs["speclib_archive"]["sha256"],
                "sha256_tree": frozen_inputs["speclib_extraction"]["sha256_tree"],
                "input_manifest_id": "usgs-splib07a-archive",
            },
        },
        "code": {
            "git": _git_revision(),
            "emit_l2b_module_sha256": sha256_file(ROOT / "src" / "tanager_rocks" / "emit_l2b.py"),
            "driver_sha256": sha256_file(Path(__file__)),
            "pyproject_sha256": sha256_file(ROOT / "pyproject.toml"),
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        },
        "geometry": {
            "crs": pair.geometry.crs,
            "crs_verified_from_product_metadata": True,
            "transform": tuple(pair.geometry.transform)[:6],
            "shape": pair.geometry.shape,
        },
        "resampling": {
            "tanager_continuous_scores": "area_average",
            "l2b_fields": "own_glt_no_interpolation",
            "m2_blocks": "full_area_footprint_single_complete_block",
        },
        "support": (
            "one endpoint-scale intersection of finite score, valid identity and band depth, "
            "complete Tanager QA, both product GLTs, and one full complete M2 block footprint"
        ),
        "scale_null_design": scale_null_design,
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "spatial_permutations": PERMUTATION_REPLICATES,
        "bootstrap_and_null_minimum_finite_fraction": FINITE_REPLICATE_FRACTION,
        "secondary_bh_family": SECONDARY_BH_FAMILY,
        "secondary_bh_fdr": FDR_ALPHA,
        "fit_uncertainty_threshold": None,
        "fit_uncertainty_threshold_unavailable_reason": (
            "exact product documentation has not supplied an approved E4 threshold contract"
        ),
        "failure_map_endpoint": failure_map_endpoint,
        "pilot_resource_estimate": {
            "status": "unavailable",
            "reason": "synthetic-only implementation; no real EMIT L2B scene run was authorized",
        },
        "unavailable_reason": None,
        "quality_report": quality,
    }
    write_strict_json(output / "input_manifest.json", input_manifest)
    (output / "report.md").write_text(
        "# E4 EMIT L2B validation\n\n"
        f"Execution status: complete. Inference status: {inference_status}. "
        f"Claim status: {claim_status}.\n\n"
        "This packet reports cross-product, cross-acquisition concordance; "
        "EMIT L2B is not field mineral truth.\n\n"
        "MIN and MINUNCERT were resolved from one CMR granule result, matched by "
        "filename and global metadata, and orthorectified with their own verified "
        "WGS84 GLTs. Tanager scores were area-averaged only where each complete "
        "footprint remained inside one M2 block. The same endpoint-scale joint "
        "support governed observations, counts, bootstrap intervals, nulls, and "
        "exclusions.\n\n"
        "Only exact ontology mappings can be confirmatory-eligible. Broader mappings "
        "remain secondary and use the single frozen compatible-mineral BH family. "
        "Unavailable intervals or whole-block nulls block confirmatory inference.\n",
        encoding="utf-8",
    )
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "summary.json"
    }
    write_strict_json(
        output / "summary.json",
        {
            "schema_version": "emit-l2b-validation/v2",
            "status": "complete",
            "execution_status": "complete",
            "inference_status": inference_status,
            "claim_status": claim_status,
            "scientific_scope": "cross-product_cross-acquisition_concordance_not_ground_truth",
            "identity": pair.identity,
            "cmr_granule": fetch_manifest["cmr_granule"],
            "ontology_version": entries[0].ontology_version,
            "endpoint_count": len(endpoint_groups),
            "endpoint_scale_metric_rows": len(metric_rows),
            "support_rows": len(support_rows),
            "support_statuses": sorted({row["support_status"] for row in metric_rows}),
            "inference_statuses": sorted({row["inference_status"] for row in metric_rows}),
            "claim_statuses": sorted({row["claim_status"] for row in metric_rows}),
            "failure_map_endpoint": failure_map_endpoint,
            "effect_estimates_unavailable_reason": (
                None
                if any(row["value"] is not None for row in metric_rows)
                else "no_endpoint_met_m2_effect_estimate_support"
            ),
            "output_sha256": output_hashes,
            "summary_self_hash": None,
            "summary_self_hash_unavailable_reason": (
                "a file cannot contain its own stable content hash"
            ),
            "unavailable_reason": None,
        },
    )


def _snapshot_pinned_input(source: Path, destination: Path, expected: str | None) -> str:
    """Copy one hash-pinned regular input through a stable no-follow descriptor."""
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise NonResultError("missing_or_invalid_expected_hash")
    try:
        source_descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise NonResultError("unsafe_or_missing_file") from error
    destination_descriptor = -1
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise NonResultError("nonregular_or_linked_file")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        digest = hashlib.sha256()
        while chunk := os.read(source_descriptor, 1024 * 1024):
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in identity_fields):
            raise NonResultError("source_changed_during_snapshot")
        observed = digest.hexdigest()
        if observed != expected:
            raise NonResultError("pinned_input_hash_mismatch")
        return observed
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)


def _m2_mapping_contract(path: Path) -> tuple[dict[str, Any], str]:
    data = read_regular_bytes(path)
    payload = strict_json_load_bytes(data)
    if not isinstance(payload, dict):
        raise NonResultError("invalid_m2_mapping_contract")
    if payload.get("schema_version") != "e4-m2-mapping-contract/v1":
        raise NonResultError("invalid_m2_mapping_contract")
    digest = payload.get("block_manifest_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise NonResultError("unbound_m2_mapping_contract")
    return payload, hashlib.sha256(data).hexdigest()


def _code_manifest() -> dict[str, Any]:
    files = {
        "run_emit_l2b_validation.py": Path(__file__),
        "emit_l2b.py": ROOT / "src" / "tanager_rocks" / "emit_l2b.py",
        "emit_l2b_nonresult.py": ROOT / "src" / "tanager_rocks" / "emit_l2b_nonresult.py",
    }
    return {
        "schema_version": "e4-nonresult-code-manifest/v1",
        "files": {name: sha256_regular_file(path) for name, path in sorted(files.items())},
    }


def _bundle_id(prefix: str, *digests: str) -> str:
    joined = "".join(digests).encode("ascii")
    return f"{prefix}-{hashlib.sha256(joined).hexdigest()[:20]}"


def run_mapping_only(args: argparse.Namespace) -> Path:
    """Write a metadata/GLT-only mapping bundle with no L2B result reads."""
    contract, contract_digest = _m2_mapping_contract(args.m2_mapping_contract)
    output_parent = Path(args.output)
    output_parent.mkdir(parents=True, exist_ok=True)
    if output_parent.is_symlink():
        raise NonResultError("unsafe_bundle_parent")
    source_names = (Path(args.emit_min).name, Path(args.emit_minuncert).name)
    if not all(source_names) or source_names[0] == source_names[1]:
        raise NonResultError("invalid_source_filenames")
    with tempfile.TemporaryDirectory(prefix=".e4-source-snapshot-", dir=output_parent) as raw:
        snapshot = Path(raw)
        min_path = snapshot / source_names[0]
        minuncert_path = snapshot / source_names[1]
        min_digest = _snapshot_pinned_input(
            args.emit_min,
            min_path,
            args.expected_emit_min_sha256,
        )
        minuncert_digest = _snapshot_pinned_input(
            args.emit_minuncert,
            minuncert_path,
            args.expected_emit_minuncert_sha256,
        )
        packet = load_emit_l2b_metadata(min_path, minuncert_path)
        if (
            packet.source.min_sha256 != min_digest
            or packet.source.minuncert_sha256 != minuncert_digest
        ):
            raise NonResultError("source_changed_during_mapping")
        bundle_id = _bundle_id("mapping", min_digest, minuncert_digest, contract_digest)
        source_identity = {
            "identity": asdict(packet.source.identity),
            "min": source_identity_payload(packet.source.min_path),
            "minuncert": source_identity_payload(packet.source.minuncert_path),
        }
        transform = tuple(packet.source.geometry.transform)[:6]
        geometry = {
            "shape": list(packet.source.geometry.shape),
            "transform": list(transform),
            "crs": packet.source.geometry.crs,
        }
        glt = {
            "min_shape": list(packet.min_glt_x.shape),
            "min_fill_locations_agree": bool(
                np.array_equal(packet.min_glt_x == 0, packet.min_glt_y == 0)
            ),
            "minuncert_shape": list(packet.minuncert_glt_x.shape),
            "minuncert_fill_locations_agree": bool(
                np.array_equal(packet.minuncert_glt_x == 0, packet.minuncert_glt_y == 0)
            ),
        }
        payloads = {
            "source_pair_identity.json": canonical_json_bytes(source_identity),
            "source_mineral_inventory.csv": csv_bytes(
                ("index", "name", "group", "library"),
                source_mineral_rows(packet.mineral_metadata),
            ),
            "geometry_contract.json": canonical_json_bytes(geometry),
            "glt_validation.json": canonical_json_bytes(glt),
            "m2_mapping_contract.json": canonical_json_bytes(contract),
            "code_manifest.json": canonical_json_bytes(_code_manifest()),
        }
        receipt = atomic_write_bundle(
            output_parent,
            bundle_type="mapping",
            bundle_id=bundle_id,
            manifest={
                "operation": "mapping_only",
                "endpoint_execution": "forbidden",
                "source_pair_sha256": {"min": min_digest, "minuncert": minuncert_digest},
                "m2_mapping_contract_sha256": contract_digest,
            },
            payloads=payloads,
        )
    return receipt.bundle_path


def run_resource_pilot(args: argparse.Namespace) -> Path:
    """Write an explicitly non-admissible synthetic-only resource pilot bundle.

    This command deliberately has no real I/O or score adapter.  A future,
    separately reviewed adapter must be supplied before a real resource pilot
    can be requested; this prevents a policy-only command from becoming an
    accidental scientific endpoint.
    """
    if not args.synthetic_only:
        raise NonResultError("real_resource_pilot_not_implemented")
    policy_path = args.resource_policy
    policy_bytes = read_regular_bytes(policy_path)
    policy = validate_legacy_synthetic_resource_policy(strict_json_load_bytes(policy_bytes))
    mapping = verify_nonresult_bundle(args.mapping_admission, expected_type="mapping")
    fixture = source_identity_payload(args.synthetic_fixture)
    policy_digest = hashlib.sha256(policy_bytes).hexdigest()
    bundle_id = _bundle_id(
        "resource-pilot",
        mapping.closure_sha256,
        policy_digest,
        fixture["sha256"],
    )
    telemetry = {
        "stage": "synthetic_fixture_guard",
        "wall_seconds": 0,
        "cpu_seconds": 0,
        "peak_rss_bytes": 0,
        "input_bytes": fixture["size_bytes"],
        "scratch_bytes": 0,
        "exit_status": 0,
    }
    payloads = {
        "input_bindings.json": canonical_json_bytes(
            {
                "mapping_closure_sha256": mapping.closure_sha256,
                "resource_policy_sha256": policy_digest,
                "synthetic_fixture": fixture,
            }
        ),
        "stage_telemetry.csv": csv_bytes(tuple(telemetry), [telemetry]),
        "resource_summary.json": canonical_json_bytes(
            {
                "synthetic_fixture_only": True,
                "admission_status": "not_admissible",
                "reason": "no_real_resource_adapter_is_implemented",
                "policy": policy,
            }
        ),
        "forbidden_output_audit.json": canonical_json_bytes(
            {
                "scientific_endpoint_called": False,
                "scientific_output_count": 0,
                "admission_status": "not_admissible",
            }
        ),
        "code_manifest.json": canonical_json_bytes(_code_manifest()),
    }
    receipt = atomic_write_bundle(
        args.output,
        bundle_type="resource_pilot",
        bundle_id=bundle_id,
        manifest={
            "operation": "resource_pilot",
            "endpoint_execution": "forbidden",
            "synthetic_fixture_only": True,
            "resource_policy_sha256": policy_digest,
        },
        payloads=payloads,
    )
    return receipt.bundle_path


def _validate_output_registry(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "e4-scientific-output-registry/v1":
        raise NonResultError("invalid_output_registry_schema")
    paths = payload.get("expected_files")
    if not isinstance(paths, list) or not paths:
        raise NonResultError("unresolved_output_registry")
    if len(set(paths)) != len(paths):
        raise NonResultError("duplicate_output_registry_path")
    for path in paths:
        if not isinstance(path, str):
            raise NonResultError("invalid_output_registry_path")
        normalized = PurePosixPath(path)
        if (
            not path
            or normalized.is_absolute()
            or ".." in normalized.parts
            or str(normalized) != path
        ):
            raise NonResultError("invalid_output_registry_path")


def run_preflight(args: argparse.Namespace) -> Path:
    """Validate sealed controls only; this mode cannot open E4 arrays or scenes."""
    decision_bytes = read_regular_bytes(args.decision_record)
    decision = validate_decision_record(strict_json_load_bytes(decision_bytes))
    policy_bytes = read_regular_bytes(args.resource_policy)
    validate_resource_policy(strict_json_load_bytes(policy_bytes))
    policy_digest = hashlib.sha256(policy_bytes).hexdigest()
    mapping = verify_nonresult_bundle(args.mapping_admission, expected_type="mapping")
    resource_evidence = verify_resource_admission_bundle(
        args.resource_admission_bundle,
        expected_policy_sha256=policy_digest,
    )
    if resource_evidence.bundle_receipt is None:  # pragma: no cover
        raise NonResultError("invalid_resource_admission_bundle")
    input_manifest_bytes = read_regular_bytes(args.input_manifest)
    input_manifest = strict_json_load_bytes(input_manifest_bytes)
    if not isinstance(input_manifest, dict) or not input_manifest.get("schema_version"):
        raise NonResultError("invalid_input_manifest")
    ontology_bytes = read_regular_bytes(args.ontology_crosswalk)
    if not ontology_bytes.strip():
        raise NonResultError("empty_ontology_crosswalk")
    if decision["ontology"]["crosswalk_sha256"] != hashlib.sha256(ontology_bytes).hexdigest():
        raise NonResultError("decision_ontology_crosswalk_mismatch")
    ontology_evidence_bytes = read_regular_bytes(args.ontology_evidence_manifest)
    ontology_evidence = strict_json_load_bytes(ontology_evidence_bytes)
    if not isinstance(ontology_evidence, dict) or not ontology_evidence.get("schema_version"):
        raise NonResultError("invalid_ontology_evidence_manifest")
    registry_bytes = read_regular_bytes(args.output_registry)
    registry = strict_json_load_bytes(registry_bytes)
    if not isinstance(registry, dict):
        raise NonResultError("invalid_output_registry")
    _validate_output_registry(registry)
    registry_digest = hashlib.sha256(registry_bytes).hexdigest()
    if decision["output_registry"]["sha256"] != registry_digest:
        raise NonResultError("decision_output_registry_mismatch")
    bundle_id = _bundle_id(
        "preflight",
        mapping.closure_sha256,
        resource_evidence.bundle_receipt.closure_sha256,
        policy_digest,
        hashlib.sha256(decision_bytes).hexdigest(),
        registry_digest,
    )
    payloads = {
        "decision_record.json": decision_bytes,
        "input_manifest.json": input_manifest_bytes,
        "ontology_crosswalk.csv": ontology_bytes,
        "ontology_evidence_manifest.json": ontology_evidence_bytes,
        "mapping_admission.json": canonical_json_bytes(
            {"mapping_closure_sha256": mapping.closure_sha256, "bundle_id": mapping.bundle_id}
        ),
        "expected_scientific_output_registry.json": registry_bytes,
        "code_manifest.json": canonical_json_bytes(_code_manifest()),
        "preflight_summary.json": canonical_json_bytes(
            {
                "endpoint_execution": "forbidden",
                "scientific_run_command": "absent",
                "resource_policy_sha256": policy_digest,
                "resource_admission_closure_sha256": (
                    resource_evidence.bundle_receipt.closure_sha256
                ),
                "mapping_closure_sha256": mapping.closure_sha256,
            }
        ),
        **resource_evidence.payloads,
        "resource_admission_bundle_manifest.json": read_regular_bytes(
            resource_evidence.bundle_receipt.bundle_path / "resource_admission_manifest.json"
        ),
        "resource_admission_bundle_checksums.sha256": read_regular_bytes(
            resource_evidence.bundle_receipt.bundle_path / "output_checksums.sha256"
        ),
    }
    receipt = atomic_write_bundle(
        args.output,
        bundle_type="preflight",
        bundle_id=bundle_id,
        manifest={
            "operation": "preflight",
            "endpoint_execution": "forbidden",
            "mapping_closure_sha256": mapping.closure_sha256,
            "resource_policy_sha256": policy_digest,
            "resource_admission_closure_sha256": (resource_evidence.bundle_receipt.closure_sha256),
            "decision_record_sha256": hashlib.sha256(decision_bytes).hexdigest(),
            "output_registry_sha256": registry_digest,
        },
        payloads=payloads,
    )
    return receipt.bundle_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Endpoint-sealed E4 non-result controls. Scientific execution is absent."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    mapping = subparsers.add_parser(
        "mapping-only",
        help="validate L2B metadata, GLTs, and inventory",
    )
    mapping.add_argument("--emit-min", required=True, type=Path)
    mapping.add_argument("--emit-minuncert", required=True, type=Path)
    mapping.add_argument("--expected-emit-min-sha256", required=True)
    mapping.add_argument("--expected-emit-minuncert-sha256", required=True)
    mapping.add_argument("--m2-mapping-contract", required=True, type=Path)
    mapping.add_argument("--output", required=True, type=Path)

    resource = subparsers.add_parser(
        "resource-pilot",
        help="exercise only the synthetic resource-pilot guard; real pilot adapter absent",
    )
    resource.add_argument("--mapping-admission", required=True, type=Path)
    resource.add_argument("--resource-policy", required=True, type=Path)
    resource.add_argument("--synthetic-fixture", required=True, type=Path)
    resource.add_argument("--synthetic-only", action="store_true")
    resource.add_argument("--output", required=True, type=Path)

    preflight = subparsers.add_parser("preflight", help="validate admitted controls without arrays")
    preflight.add_argument("--decision-record", required=True, type=Path)
    preflight.add_argument("--resource-policy", required=True, type=Path)
    preflight.add_argument("--mapping-admission", required=True, type=Path)
    preflight.add_argument("--resource-admission-bundle", required=True, type=Path)
    preflight.add_argument("--input-manifest", required=True, type=Path)
    preflight.add_argument("--ontology-crosswalk", required=True, type=Path)
    preflight.add_argument("--ontology-evidence-manifest", required=True, type=Path)
    preflight.add_argument("--output-registry", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "mapping-only":
            output = run_mapping_only(args)
        elif args.mode == "resource-pilot":
            output = run_resource_pilot(args)
        elif args.mode == "preflight":
            output = run_preflight(args)
        else:  # pragma: no cover - argparse keeps this unreachable.
            raise NonResultError("unknown_nonresult_mode")
    except (FileNotFoundError, OSError, ProductMismatchError, NonResultError, ValueError) as error:
        # Do not echo attacker-controlled paths, JSON fields, or sentinel values.
        raise SystemExit(f"E4_NONRESULT_FAILED:{type(error).__name__}") from error
    print(f"E4_NONRESULT_OK:{args.mode}:{output.name}")


if __name__ == "__main__":
    main()
