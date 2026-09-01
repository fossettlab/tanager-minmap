"""Preregistered strict-inductive MTMF covariance sensitivity.

This module consumes the frozen M2 block handoff. It never derives replacement
blocks: both ``L`` and ``2L`` rasters must match the manifest byte-for-byte and
grid-for-grid before the primary-``L`` covariance folds can run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from affine import Affine

from .config import SITES, TANAGER_SR_ASSET, TARGET_MINERALS
from .quality import mask_tanager_scene
from .reference import MINERAL_TO_ROCKWELL, ROCKWELL_EXCLUDED
from .spatial_validation import (
    BOOTSTRAP_REPLICATES,
    FINITE_REPLICATE_FRACTION,
    PERMUTATION_REPLICATES,
    SEED,
    Block,
    bearing_block_counts,
    benjamini_hochberg,
    block_bootstrap_intervals,
    governance_status,
    pooled_metrics,
    rank_auc,
    sample_blocks,
    spatial_cross_fit,
    whole_block_permutation_test,
)
from .speclib import Endmember, load_library, select_endmembers
from .unmix import fit_mtmf_background, score_mtmf_background

M2_PROTOCOL_RELATIVE_PATH = "docs/m2_spatial_validation_preregistration.md"
FROZEN_RIDGE = 1e-2
FROZEN_RETAINED_BANDS = 363
SENSITIVITY_STATEMENT = (
    "This is the preregistered mandatory strict-inductive MTMF covariance "
    "sensitivity, not a new tuned model."
)
ANCHOR_SCENES = {site_id: site.scene_ids[0] for site_id, site in SITES.items()}
PROTOCOL_PARAMETERS = {
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "max_pairs_per_field_lag": 200_000,
    "permutation_replicates": PERMUTATION_REPLICATES,
    "seed": SEED,
}


class StrictInductiveError(ValueError):
    """Raised when a frozen input or analysis invariant is violated."""


@dataclass(frozen=True)
class GridSpec:
    """Exact raster grid encoded by the M2 handoff."""

    shape: tuple[int, int]
    crs: str
    transform: Affine


@dataclass(frozen=True)
class ScaleHandoff:
    """One validated block scale and its categorical raster."""

    scale: str
    raster_path: Path
    raster_sha256: str
    block_ids: tuple[int, ...]
    block_names: Mapping[int, str]
    block_side_pixels: int
    halo_pixels: int
    values: np.ndarray
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class SiteHandoff:
    """Validated M2 block handoff for one anchor site."""

    site_id: str
    scene_id: str
    grid: GridSpec
    scales: Mapping[str, ScaleHandoff]


@dataclass(frozen=True)
class ValidatedHandoff:
    """Current protocol-compliant M2 handoff."""

    manifest_path: Path
    manifest_sha256: str
    summary_path: Path
    summary_sha256: str
    protocol_path: Path
    protocol_sha256: str
    sites: Mapping[str, SiteHandoff]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise StrictInductiveError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise StrictInductiveError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise StrictInductiveError(f"{label} must contain a JSON object")
    return payload


def _safe_child(parent: Path, recorded: str, label: str) -> Path:
    path = (parent / recorded).resolve()
    try:
        path.relative_to(parent.resolve())
    except ValueError as error:
        raise StrictInductiveError(f"{label} escapes its artifact directory: {recorded}") from error
    return path


def _blocks_from_raster(
    values: np.ndarray,
    block_ids: tuple[int, ...],
    names: Mapping[int, str],
    block_side_pixels: int,
    *,
    label: str,
) -> tuple[Block, ...]:
    raster_ids = tuple(sorted(int(value) for value in np.unique(values) if int(value) > 0))
    _require_equal(f"{label} raster IDs", raster_ids, tuple(sorted(block_ids)))
    blocks: list[Block] = []
    for numeric_id in block_ids:
        rows, cols = np.nonzero(values == numeric_id)
        if rows.size == 0:
            raise StrictInductiveError(f"{label} block {numeric_id} has no raster cells")
        row_start, row_stop = int(rows.min()), int(rows.max()) + 1
        col_start, col_stop = int(cols.min()), int(cols.max()) + 1
        if row_stop - row_start != block_side_pixels or col_stop - col_start != block_side_pixels:
            raise StrictInductiveError(f"{label} block {numeric_id} was shrunk or is not square")
        window = values[row_start:row_stop, col_start:col_stop]
        if window.size != rows.size or not np.all(window == numeric_id):
            raise StrictInductiveError(f"{label} block {numeric_id} is not a complete rectangle")
        blocks.append(
            Block(
                block_id=names[numeric_id],
                block_row=row_start // block_side_pixels,
                block_col=col_start // block_side_pixels,
                row_start=row_start,
                row_stop=row_stop,
                col_start=col_start,
                col_stop=col_stop,
            )
        )
    return tuple(blocks)


def _validate_scale(
    manifest_path: Path,
    site_id: str,
    scene_id: str,
    scale: str,
    record: Mapping[str, Any],
    grid: GridSpec,
) -> ScaleHandoff:
    _require_equal(f"{site_id}/{scale} scale", record.get("scale"), scale)
    _require_equal(f"{site_id}/{scale} anchor", record.get("anchor_scene_id"), scene_id)
    raster_name = record.get("block_raster")
    raster_hash = record.get("block_raster_sha256")
    if not isinstance(raster_name, str) or not isinstance(raster_hash, str):
        raise StrictInductiveError(f"{site_id}/{scale} lacks block-raster provenance")
    raster_path = _safe_child(manifest_path.parent, raster_name, f"{site_id}/{scale} raster")
    if not raster_path.is_file():
        raise FileNotFoundError(
            f"declared {site_id}/{scale} block raster is missing: {raster_path}"
        )
    observed_hash = sha256_file(raster_path)
    _require_equal(f"{site_id}/{scale} block raster SHA-256", observed_hash, raster_hash)

    ids = tuple(int(value) for value in record.get("complete_block_ids", ()))
    if len(ids) != len(set(ids)) or any(value <= 0 for value in ids):
        raise StrictInductiveError(f"{site_id}/{scale} block IDs must be unique positive integers")
    _require_equal(f"{site_id}/{scale} block count", record.get("complete_blocks"), len(ids))
    raw_names = record.get("numeric_to_string_block_ids")
    if not isinstance(raw_names, dict):
        raise StrictInductiveError(f"{site_id}/{scale} block-name mapping is missing")
    names = {int(key): str(value) for key, value in raw_names.items()}
    _require_equal(f"{site_id}/{scale} block-name IDs", set(names), set(ids))
    if len(set(names.values())) != len(names):
        raise StrictInductiveError(f"{site_id}/{scale} block names are not unique")

    block_side_pixels = int(record.get("block_side_pixels", 0))
    halo_pixels = int(record.get("halo_pixels", -1))
    if block_side_pixels <= 0 or halo_pixels < 0:
        raise StrictInductiveError(f"{site_id}/{scale} block side or halo is invalid")
    with rasterio.open(raster_path) as dataset:
        _require_equal(f"{site_id}/{scale} band count", dataset.count, 1)
        _require_equal(f"{site_id}/{scale} raster shape", dataset.shape, grid.shape)
        if dataset.crs is None:
            raise StrictInductiveError(f"{site_id}/{scale} raster has no CRS")
        _require_equal(f"{site_id}/{scale} raster CRS", dataset.crs.to_string(), grid.crs)
        _require_equal(f"{site_id}/{scale} raster transform", dataset.transform, grid.transform)
        _require_equal(f"{site_id}/{scale} raster nodata", dataset.nodata, 0.0)
        _require_equal(f"{site_id}/{scale} raster dtype", dataset.dtypes, ("uint32",))
        values = dataset.read(1, masked=False)
    blocks = _blocks_from_raster(
        values,
        ids,
        names,
        block_side_pixels,
        label=f"{site_id}/{scale}",
    )
    return ScaleHandoff(
        scale=scale,
        raster_path=raster_path,
        raster_sha256=observed_hash,
        block_ids=ids,
        block_names=names,
        block_side_pixels=block_side_pixels,
        halo_pixels=halo_pixels,
        values=values,
        blocks=blocks,
    )


def _validate_block_rows(
    rows: Any,
    site: SiteHandoff,
) -> None:
    if not isinstance(rows, list):
        raise StrictInductiveError("block manifest has no blocks list")
    for scale, handoff in site.scales.items():
        matches = [
            row
            for row in rows
            if isinstance(row, dict)
            and row.get("site") == site.site_id
            and row.get("scene_id") == site.scene_id
            and row.get("scale") == scale
        ]
        _require_equal(f"{site.site_id}/{scale} block-row count", len(matches), len(handoff.blocks))
        by_numeric = {int(row["numeric_block_id"]): row for row in matches}
        _require_equal(
            f"{site.site_id}/{scale} block-row IDs", set(by_numeric), set(handoff.block_ids)
        )
        for numeric_id, block in zip(handoff.block_ids, handoff.blocks, strict=True):
            row = by_numeric[numeric_id]
            expected = {
                "block_id": block.block_id,
                "complete": True,
                "halo_pixels": handoff.halo_pixels,
                "row_start": block.row_start,
                "row_stop": block.row_stop,
                "col_start": block.col_start,
                "col_stop": block.col_stop,
                "crs": site.grid.crs,
            }
            for key, value in expected.items():
                _require_equal(f"{site.site_id}/{scale}/{numeric_id} {key}", row.get(key), value)


def validate_block_handoff(
    manifest_path: Path,
    *,
    root: Path,
    protocol_path: Path | None = None,
    summary_path: Path | None = None,
) -> ValidatedHandoff:
    """Validate current protocol, compliant M2 summary, and exact L/2L rasters."""
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    protocol_path = (protocol_path or root / M2_PROTOCOL_RELATIVE_PATH).resolve()
    summary_path = (summary_path or manifest_path.with_name("summary.json")).resolve()
    protocol_hash = sha256_file(protocol_path)
    manifest = _load_json(manifest_path, "M2 block manifest")
    manifest_hash = sha256_file(manifest_path)
    _require_equal(
        "M2 manifest type", manifest.get("manifest_type"), "spatial_validation_complete_blocks"
    )
    _require_equal("M2 manifest schema", manifest.get("schema_version"), "1.0")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise StrictInductiveError("M2 block manifest has no protocol record")
    _require_equal("M2 protocol path", protocol.get("path"), M2_PROTOCOL_RELATIVE_PATH)
    _require_equal("M2 protocol SHA-256", protocol.get("sha256"), protocol_hash)
    _require_equal(
        "M2 manifest protocol parameters", protocol.get("parameters"), PROTOCOL_PARAMETERS
    )
    _require_equal("M2 manifest protocol compliance", protocol.get("protocol_compliant"), True)

    summary = _load_json(summary_path, "M2 summary")
    _require_equal(
        "M2 summary top-level block-manifest SHA-256",
        summary.get("block_manifest_sha256"),
        manifest_hash,
    )
    summary_manifest = summary.get("block_manifest")
    if not isinstance(summary_manifest, dict):
        raise StrictInductiveError("M2 summary has no block_manifest link")
    _require_equal(
        "M2 summary nested block-manifest SHA-256",
        summary_manifest.get("sha256"),
        manifest_hash,
    )
    recorded_manifest_path = summary_manifest.get("path")
    if not isinstance(recorded_manifest_path, str) or not recorded_manifest_path:
        raise StrictInductiveError("M2 summary block_manifest path is missing or invalid")
    linked_manifest_path = _safe_child(
        summary_path.parent,
        recorded_manifest_path,
        "M2 summary block_manifest link",
    )
    _require_equal("M2 summary block-manifest path", linked_manifest_path, manifest_path)
    summary_protocol = summary.get("protocol")
    if not isinstance(summary_protocol, dict):
        raise StrictInductiveError("M2 summary has no protocol record")
    _require_equal(
        "M2 summary protocol path",
        summary_protocol.get("path"),
        M2_PROTOCOL_RELATIVE_PATH,
    )
    _require_equal("M2 summary protocol SHA-256", summary_protocol.get("sha256"), protocol_hash)
    _require_equal(
        "M2 summary protocol compliance", summary_protocol.get("protocol_compliant"), True
    )
    _require_equal(
        "M2 summary protocol parameters",
        summary_protocol.get("parameters"),
        PROTOCOL_PARAMETERS,
    )

    site_records = manifest.get("sites")
    if not isinstance(site_records, dict):
        raise StrictInductiveError("M2 block manifest has no sites object")
    sites: dict[str, SiteHandoff] = {}
    for site_id, scene_id in ANCHOR_SCENES.items():
        entry = site_records.get(site_id)
        if not isinstance(entry, dict):
            raise StrictInductiveError(f"M2 block manifest has no {site_id} entry")
        _require_equal(f"{site_id} anchor", entry.get("scene_id"), scene_id)
        _require_equal(f"{site_id} primary scale", entry.get("primary_scale"), "L")
        grid_record = entry.get("grid")
        if not isinstance(grid_record, dict):
            raise StrictInductiveError(f"{site_id} grid record is missing")
        shape_raw = grid_record.get("shape")
        transform_raw = grid_record.get("transform")
        if not isinstance(shape_raw, list) or len(shape_raw) != 2:
            raise StrictInductiveError(f"{site_id} grid shape is invalid")
        if not isinstance(transform_raw, list) or len(transform_raw) != 6:
            raise StrictInductiveError(f"{site_id} grid transform is invalid")
        grid = GridSpec(
            shape=(int(shape_raw[0]), int(shape_raw[1])),
            crs=str(grid_record.get("crs")),
            transform=Affine(*(float(value) for value in transform_raw)),
        )
        scale_records = entry.get("scales")
        if not isinstance(scale_records, dict) or set(scale_records) != {"L", "2L"}:
            raise StrictInductiveError(f"{site_id} must contain exactly L and 2L")
        scales = {
            scale: _validate_scale(
                manifest_path,
                site_id,
                scene_id,
                scale,
                scale_records[scale],
                grid,
            )
            for scale in ("L", "2L")
        }
        _require_equal(
            f"{site_id} top-level primary raster",
            entry.get("block_raster"),
            scales["L"].raster_path.name,
        )
        _require_equal(
            f"{site_id} top-level primary IDs",
            tuple(int(value) for value in entry.get("complete_block_ids", ())),
            scales["L"].block_ids,
        )
        _require_equal(
            f"{site_id} 2L side", scales["2L"].block_side_pixels, 2 * scales["L"].block_side_pixels
        )
        _require_equal(f"{site_id} scale halos", scales["2L"].halo_pixels, scales["L"].halo_pixels)
        sites[site_id] = SiteHandoff(site_id, scene_id, grid, scales)
    for site in sites.values():
        _validate_block_rows(manifest.get("blocks"), site)
    return ValidatedHandoff(
        manifest_path=manifest_path,
        manifest_sha256=manifest_hash,
        summary_path=summary_path,
        summary_sha256=sha256_file(summary_path),
        protocol_path=protocol_path,
        protocol_sha256=protocol_hash,
        sites=sites,
    )


def _manifest_records(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    payload = _load_json(path, "scientific input manifest")
    _require_equal("input-manifest hash algorithm", payload.get("hash_algorithm"), "sha256")
    records = payload.get("inputs")
    if not isinstance(records, list):
        raise StrictInductiveError("scientific input manifest has no inputs list")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise StrictInductiveError("scientific input manifest contains an invalid record")
        if record["id"] in indexed:
            raise StrictInductiveError(f"duplicate scientific input ID: {record['id']}")
        indexed[record["id"]] = record
    return indexed, sha256_file(path)


def _validate_declared_input(
    record: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    logical_path = str(record.get("logical_path", ""))
    path = _safe_child(root, logical_path, str(record.get("id", "input")))
    if not path.is_file():
        raise FileNotFoundError(f"declared scientific input is missing: {logical_path}")
    _require_equal(f"{record['id']} size", path.stat().st_size, int(record.get("size_bytes", -1)))
    observed_hash = sha256_file(path)
    _require_equal(f"{record['id']} SHA-256", observed_hash, record.get("sha256"))
    return {
        "id": record["id"],
        "logical_path": logical_path,
        "resolved_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": observed_hash,
    }


def _validate_grid_raster(path: Path, grid: GridSpec, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    with rasterio.open(path) as dataset:
        _require_equal(f"{label} band count", dataset.count, 1)
        _require_equal(f"{label} shape", dataset.shape, grid.shape)
        if dataset.crs is None:
            raise StrictInductiveError(f"{label} has no CRS")
        _require_equal(f"{label} CRS", dataset.crs.to_string(), grid.crs)
        _require_equal(f"{label} transform", dataset.transform, grid.transform)
    return {"path": str(path), "sha256": sha256_file(path)}


def preflight_strict_inductive(
    *,
    root: Path,
    block_manifest_path: Path,
    input_manifest_path: Path,
    summary_path: Path | None,
    maps_dir: Path,
    reference_dir: Path,
) -> tuple[ValidatedHandoff, dict[str, Any]]:
    """Perform byte and raster-grid validation without opening source cubes."""
    root = root.resolve()
    handoff = validate_block_handoff(
        block_manifest_path,
        root=root,
        summary_path=summary_path,
    )
    input_records, input_manifest_hash = _manifest_records(input_manifest_path)
    required_ids = [f"tanager-{site_id}-1" for site_id in ANCHOR_SCENES]
    required_ids.append("usgs-splib07a-archive")
    declared_inputs = []
    for input_id in required_ids:
        if input_id not in input_records:
            raise StrictInductiveError(f"scientific input manifest lacks {input_id}")
        declared_inputs.append(_validate_declared_input(input_records[input_id], root=root))

    raster_inputs: list[dict[str, Any]] = []
    for site_id, site in handoff.sites.items():
        scene_id = site.scene_id
        reference_path = reference_dir / f"rockwell_{site_id}_{scene_id}.tif"
        raster_inputs.append(
            _validate_grid_raster(reference_path, site.grid, f"{site_id} reference")
        )
        for mineral in MINERAL_TO_ROCKWELL:
            full_path = maps_dir / f"{site_id}_{scene_id}_mf_{mineral}.tif"
            if full_path.is_file():
                raster_inputs.append(
                    _validate_grid_raster(full_path, site.grid, f"{site_id}/{mineral} full score")
                )
    return handoff, {
        "status": "available",
        "analysis_scale": "L",
        "two_l_role": "validated frozen sensitivity geometry; not used to replace primary-L folds",
        "sensitivity_statement": SENSITIVITY_STATEMENT,
        "protocol": {
            "path": str(handoff.protocol_path),
            "sha256": handoff.protocol_sha256,
            "parameters": PROTOCOL_PARAMETERS,
        },
        "block_manifest": {
            "path": str(handoff.manifest_path),
            "sha256": handoff.manifest_sha256,
        },
        "m2_summary": {"path": str(handoff.summary_path), "sha256": handoff.summary_sha256},
        "input_manifest": {"path": str(input_manifest_path), "sha256": input_manifest_hash},
        "declared_inputs": declared_inputs,
        "grid_validated_rasters": sorted(raster_inputs, key=lambda item: item["path"]),
        "sites": [
            {
                "site": site_id,
                "scene_id": site.scene_id,
                "complete_L_blocks": len(site.scales["L"].blocks),
                "complete_2L_blocks": len(site.scales["2L"].blocks),
                "halo_pixels": site.scales["L"].halo_pixels,
                "block_rasters": {
                    scale: {
                        "path": str(site.scales[scale].raster_path),
                        "sha256": site.scales[scale].raster_sha256,
                    }
                    for scale in ("L", "2L")
                },
            }
            for site_id, site in sorted(handoff.sites.items())
        ],
        "source_cubes_opened": False,
    }


def held_block_halo_mask(
    block_ids: np.ndarray,
    held_block_id: int,
    halo_pixels: int,
) -> np.ndarray:
    """Return the clipped held-block-plus-halo exclusion mask."""
    values = np.asarray(block_ids)
    if values.ndim != 2:
        raise ValueError("block_ids must be two-dimensional")
    if halo_pixels < 0:
        raise ValueError("halo_pixels cannot be negative")
    rows, cols = np.nonzero(values == held_block_id)
    if rows.size == 0:
        raise ValueError(f"held block ID {held_block_id} is absent")
    row_start = max(0, int(rows.min()) - halo_pixels)
    row_stop = min(values.shape[0], int(rows.max()) + 1 + halo_pixels)
    col_start = max(0, int(cols.min()) - halo_pixels)
    col_stop = min(values.shape[1], int(cols.max()) + 1 + halo_pixels)
    exclusion = np.zeros(values.shape, dtype=bool)
    exclusion[row_start:row_stop, col_start:col_stop] = True
    return exclusion


def strict_fold_scores(
    cube: xr.DataArray,
    endmembers: dict[str, Endmember],
    block_ids: np.ndarray,
    held_block_id: int,
    halo_pixels: int,
    *,
    ridge: float = FROZEN_RIDGE,
) -> tuple[xr.Dataset, dict[str, int]]:
    """Fit one held-block-excluded background and score only that block."""
    exclusion = held_block_halo_mask(block_ids, held_block_id, halo_pixels)
    held = np.asarray(block_ids) == held_block_id
    background = fit_mtmf_background(
        cube,
        endmembers,
        ridge=ridge,
        fit_mask=~exclusion,
    )
    scores = score_mtmf_background(cube, endmembers, background, score_mask=held)
    return scores, {
        "excluded_geometric_pixels": int(np.count_nonzero(exclusion)),
        "held_geometric_pixels": int(np.count_nonzero(held)),
        "covariance_training_pixels": background.sample_count,
        "covariance_bands": int(np.count_nonzero(background.valid_bands)),
    }


def _binary_reference(reference: np.ndarray, positive_classes: frozenset[int]) -> np.ndarray:
    domain = np.isfinite(reference)
    for class_value in ROCKWELL_EXCLUDED:
        domain &= reference != class_value
    binary = np.full(reference.shape, np.nan, dtype=float)
    binary[domain] = np.isin(reference[domain], tuple(sorted(positive_classes))).astype(float)
    return binary


def _read_raster_values(path: Path, grid: GridSpec) -> np.ndarray:
    _validate_grid_raster(path, grid, str(path))
    with rasterio.open(path) as dataset:
        values = dataset.read(1, masked=True).astype(float)
        return values.filled(np.nan)


def _classification(site_id: str, mineral: str) -> str:
    if site_id == "goldfield" and mineral in {"alunite", "muscovite"}:
        return "key_secondary"
    if mineral == "jarosite":
        return "descriptive"
    return "exploratory"


def _direction(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "unavailable"
    if value > 0.5:
        return "above_chance"
    if value < 0.5:
        return "below_chance"
    return "at_chance"


def _metric_template(
    site_id: str,
    scene_id: str,
    mineral: str,
    positive_blocks: int,
    negative_blocks: int,
    status: str,
) -> dict[str, Any]:
    return {
        "site": site_id,
        "scene_id": scene_id,
        "scale": "L",
        "family": "mtmf",
        "layer": mineral,
        "positive_classes": ";".join(str(value) for value in sorted(MINERAL_TO_ROCKWELL[mineral])),
        "classification": _classification(site_id, mineral),
        "governance_status": status,
        "positive_bearing_blocks": positive_blocks,
        "negative_bearing_blocks": negative_blocks,
        "metric_status": "unavailable",
        "unavailable_reason": "",
        "rank_status": "unavailable",
        "rank_unavailable_reason": "",
        "rank_evaluated_blocks": 0,
        "rank_observations": 0,
        "rank_n_pos": 0,
        "rank_n_neg": 0,
        "threshold_status": "unavailable",
        "threshold_unavailable_reason": "",
        "threshold_evaluated_blocks": 0,
        "threshold_observations": 0,
        "threshold_n_pos": 0,
        "threshold_n_neg": 0,
        "evaluated_blocks": 0,
        "skipped_blocks": 0,
        "auc": None,
        "balanced_accuracy": None,
        "positive_f1": None,
        "negative_f1": None,
        "macro_f1": None,
        "tpr": None,
        "fpr": None,
        "prevalence": None,
        "n_pos": 0,
        "n_neg": 0,
        "threshold_min": None,
        "threshold_median": None,
        "threshold_max": None,
    }


def _evaluate_layer(
    *,
    site: SiteHandoff,
    mineral: str,
    strict_score: np.ndarray,
    reference: np.ndarray,
    full_score: np.ndarray | None,
    full_path: Path,
    workers: int | None,
) -> dict[str, list[dict[str, Any]]]:
    primary = site.scales["L"]
    samples = sample_blocks(strict_score, reference, primary.blocks)
    positive_blocks, negative_blocks = bearing_block_counts(samples)
    status = governance_status(positive_blocks, negative_blocks)
    metric_row = _metric_template(
        site.site_id,
        site.scene_id,
        mineral,
        positive_blocks,
        negative_blocks,
        status,
    )
    support_rows: list[dict[str, Any]] = []
    for sample in samples:
        score, labels = sample.paired_values()
        support_rows.append(
            {
                "site": site.site_id,
                "scene_id": site.scene_id,
                "scale": "L",
                "layer": mineral,
                "block_id": sample.block.block_id,
                "pairwise_finite": int(score.size),
                "n_pos": int(np.count_nonzero(labels == 1)),
                "n_neg": int(np.count_nonzero(labels == 0)),
            }
        )

    interval_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    strict_oof = None
    if status == "counts_and_maps_only":
        reason = "fewer_than_five_positive_or_negative_bearing_blocks"
        metric_row.update(
            {
                "unavailable_reason": reason,
                "rank_unavailable_reason": reason,
                "threshold_unavailable_reason": reason,
            }
        )
    else:
        strict_oof = spatial_cross_fit(samples, halo_pixels=primary.halo_pixels)
        auc_references = (
            strict_oof.auc_references
            if strict_oof.auc_references is not None
            else strict_oof.references
        )
        auc_block_ids = (
            strict_oof.auc_block_ids
            if strict_oof.auc_block_ids is not None
            else strict_oof.block_ids
        )
        rank_n_pos = int(np.count_nonzero(auc_references == 1))
        rank_n_neg = int(np.count_nonzero(auc_references == 0))
        threshold_n_pos = int(np.count_nonzero(strict_oof.references == 1))
        threshold_n_neg = int(np.count_nonzero(strict_oof.references == 0))
        metric_row.update(
            {
                "rank_evaluated_blocks": len(set(auc_block_ids.tolist())),
                "rank_observations": int(auc_references.size),
                "rank_n_pos": rank_n_pos,
                "rank_n_neg": rank_n_neg,
                "threshold_evaluated_blocks": len(strict_oof.folds),
                "threshold_observations": int(strict_oof.references.size),
                "threshold_n_pos": threshold_n_pos,
                "threshold_n_neg": threshold_n_neg,
                "evaluated_blocks": len(strict_oof.folds),
                "skipped_blocks": len(strict_oof.skipped_blocks),
            }
        )
        if auc_references.size == 0 or len(np.unique(auc_references)) < 2:
            reason = "rank_evaluation_lacks_two_reference_classes"
            metric_row.update(
                {
                    "unavailable_reason": reason,
                    "rank_unavailable_reason": reason,
                    "threshold_unavailable_reason": "rank_evaluation_unavailable",
                }
            )
        else:
            metrics = pooled_metrics(strict_oof)
            thresholds = np.asarray([fold.threshold for fold in strict_oof.folds], dtype=float)
            threshold_available = (
                strict_oof.references.size > 0 and len(np.unique(strict_oof.references)) == 2
            )
            if strict_oof.references.size == 0:
                threshold_reason = "no_successful_threshold_folds"
            elif not threshold_available:
                threshold_reason = "successful_threshold_folds_lack_two_reference_classes"
            else:
                threshold_reason = ""
            metric_row.update(
                {
                    "metric_status": (
                        "available"
                        if threshold_available
                        else "rank_available_threshold_unavailable"
                    ),
                    "unavailable_reason": threshold_reason,
                    "rank_status": "available",
                    "rank_unavailable_reason": "",
                    "threshold_status": "available" if threshold_available else "unavailable",
                    "threshold_unavailable_reason": threshold_reason,
                    "auc": metrics.auc,
                    "prevalence": metrics.prevalence,
                    "n_pos": metrics.n_pos,
                    "n_neg": metrics.n_neg,
                    "balanced_accuracy": (
                        metrics.balanced_accuracy if threshold_available else None
                    ),
                    "positive_f1": metrics.positive_f1 if threshold_available else None,
                    "negative_f1": metrics.negative_f1 if threshold_available else None,
                    "macro_f1": metrics.macro_f1 if threshold_available else None,
                    "tpr": metrics.tpr if threshold_available else None,
                    "fpr": metrics.fpr if threshold_available else None,
                    "threshold_min": float(np.min(thresholds)) if thresholds.size else None,
                    "threshold_median": float(np.median(thresholds)) if thresholds.size else None,
                    "threshold_max": float(np.max(thresholds)) if thresholds.size else None,
                }
            )
            threshold_rows.extend(
                {
                    "site": site.site_id,
                    "scene_id": site.scene_id,
                    "scale": "L",
                    "layer": mineral,
                    **asdict(fold),
                }
                for fold in strict_oof.folds
            )
            minimum_finite = math.ceil(FINITE_REPLICATE_FRACTION * BOOTSTRAP_REPLICATES)
            for interval in block_bootstrap_intervals(
                strict_oof,
                replicates=BOOTSTRAP_REPLICATES,
                seed=SEED,
            ):
                available = interval.gate_eligible
                interval_rows.append(
                    {
                        "site": site.site_id,
                        "scene_id": site.scene_id,
                        "scale": "L",
                        "layer": mineral,
                        "metric": interval.metric,
                        "interval_status": "available" if available else "unavailable",
                        "lower": interval.lower if available else None,
                        "upper": interval.upper if available else None,
                        "scheduled_replicates": interval.scheduled_replicates,
                        "finite_replicates": interval.valid_replicates,
                        "minimum_finite_replicates": minimum_finite,
                        "unavailable_reason": interval.unavailable_reason or "",
                    }
                )
            if status == "confirmatory_eligible":
                permutation = whole_block_permutation_test(
                    samples,
                    halo_pixels=primary.halo_pixels,
                    permutations=PERMUTATION_REPLICATES,
                    seed=SEED,
                    workers=workers,
                )
                permutation_rows.append(
                    {
                        "site": site.site_id,
                        "scene_id": site.scene_id,
                        "scale": "L",
                        "layer": mineral,
                        "scheduled_permutations": PERMUTATION_REPLICATES,
                        **asdict(permutation),
                        "auc_q_value": None,
                        "balanced_accuracy_q_value": None,
                    }
                )

    comparison = {
        "site": site.site_id,
        "scene_id": site.scene_id,
        "scale": "L",
        "layer": mineral,
        "comparison_status": "unavailable",
        "unavailable_reason": "full_scene_score_raster_missing" if full_score is None else "",
        "full_score_path": str(full_path),
        "full_score_sha256": sha256_file(full_path) if full_score is not None else None,
        "common_pairwise_finite": 0,
        "common_n_pos": 0,
        "common_n_neg": 0,
        "full_auc_common_support": None,
        "strict_auc_common_support": None,
        "strict_minus_full_auc": None,
        "full_direction": "unavailable",
        "strict_direction": "unavailable",
        "direction_preserved": None,
        "full_cross_fitted_balanced_accuracy": None,
        "strict_cross_fitted_balanced_accuracy": metric_row["balanced_accuracy"],
    }
    if full_score is not None:
        complete_domain = primary.values > 0
        common = (
            complete_domain
            & np.isfinite(reference)
            & np.isfinite(strict_score)
            & np.isfinite(full_score)
        )
        common_labels = reference[common].astype(np.int8)
        comparison["common_pairwise_finite"] = int(np.count_nonzero(common))
        comparison["common_n_pos"] = int(np.count_nonzero(common_labels == 1))
        comparison["common_n_neg"] = int(np.count_nonzero(common_labels == 0))
        if status == "counts_and_maps_only":
            comparison["unavailable_reason"] = "external_support_governed_counts_and_maps_only"
        elif common_labels.size == 0 or len(np.unique(common_labels)) < 2:
            comparison["unavailable_reason"] = "common_support_lacks_two_reference_classes"
        else:
            full_auc = rank_auc(full_score[common], common_labels)
            strict_auc = rank_auc(strict_score[common], common_labels)
            full_direction = _direction(full_auc)
            strict_direction = _direction(strict_auc)
            comparison.update(
                {
                    "comparison_status": "available",
                    "full_auc_common_support": full_auc,
                    "strict_auc_common_support": strict_auc,
                    "strict_minus_full_auc": strict_auc - full_auc,
                    "full_direction": full_direction,
                    "strict_direction": strict_direction,
                    "direction_preserved": full_direction == strict_direction,
                    "unavailable_reason": "",
                }
            )
            full_samples = sample_blocks(full_score, reference, primary.blocks)
            full_oof = spatial_cross_fit(full_samples, halo_pixels=primary.halo_pixels)
            if full_oof.references.size and len(np.unique(full_oof.references)) == 2:
                comparison["full_cross_fitted_balanced_accuracy"] = pooled_metrics(
                    full_oof
                ).balanced_accuracy
    failures: list[dict[str, Any]] = []
    if metric_row["rank_status"] == "unavailable":
        failures.append(
            {
                "site": site.site_id,
                "scene_id": site.scene_id,
                "stage": "external_support",
                "block_id": "",
                "layer": mineral,
                "status": "unavailable",
                "reason": metric_row["unavailable_reason"],
            }
        )
    elif metric_row["threshold_status"] == "unavailable":
        failures.append(
            {
                "site": site.site_id,
                "scene_id": site.scene_id,
                "stage": "threshold_calibration",
                "block_id": "",
                "layer": mineral,
                "status": "unavailable",
                "reason": metric_row["threshold_unavailable_reason"],
            }
        )
    return {
        "metrics": [metric_row],
        "support": support_rows,
        "intervals": interval_rows,
        "thresholds": threshold_rows,
        "permutations": permutation_rows,
        "comparisons": [comparison],
        "failures": failures,
    }


def _validate_speclib_extraction(archive_path: Path, library_dir: Path) -> dict[str, str]:
    """Verify every candidate frozen target spectrum against the pinned archive."""
    prefixes = tuple(f"splib07a_{mineral}_".lower() for mineral in TARGET_MINERALS)
    required: list[zipfile.ZipInfo] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = Path(info.filename)
            lower = name.name.lower()
            if info.is_dir():
                continue
            if name.parent.name == "ChapterM_Minerals" and lower.endswith("_aref.txt"):
                if lower.startswith(prefixes):
                    required.append(info)
            elif name.name in {
                "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt",
                "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt",
            }:
                required.append(info)
        if not required:
            raise StrictInductiveError("pinned spectral-library archive has no target spectra")
        hashes: dict[str, str] = {}
        for info in sorted(required, key=lambda item: item.filename):
            relative = Path(info.filename).relative_to(Path(info.filename).parts[0])
            extracted = library_dir / relative
            if not extracted.is_file():
                raise FileNotFoundError(f"frozen spectral-library member is missing: {extracted}")
            archive_bytes = archive.read(info)
            extracted_bytes = extracted.read_bytes()
            if archive_bytes != extracted_bytes:
                raise StrictInductiveError(
                    f"extracted spectral-library member differs from pinned archive: {relative}"
                )
            hashes[str(relative)] = hashlib.sha256(extracted_bytes).hexdigest()
    return hashes


def _cube_grid(cube: xr.DataArray) -> GridSpec:
    if cube.rio.crs is None:
        raise StrictInductiveError("masked anchor cube has no CRS")
    transform = cube.rio.transform()
    return GridSpec(
        shape=(cube.sizes["y"], cube.sizes["x"]),
        crs=cube.rio.crs.to_string(),
        transform=transform,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def strict_json_dumps(payload: Mapping[str, Any]) -> str:
    """Serialize deterministic JSON with no non-standard NaN tokens."""
    return json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(strict_json_dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fields})
    temporary.replace(path)


def _fields(rows: Sequence[Mapping[str, Any]], fallback: Sequence[str]) -> list[str]:
    if not rows:
        return list(fallback)
    first = list(rows[0])
    extras = sorted(set().union(*(row.keys() for row in rows)) - set(first))
    return [*first, *extras]


def _write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Path]:
    table_names = {
        "metrics": "strict_inductive_metrics.csv",
        "support": "strict_inductive_block_support.csv",
        "intervals": "strict_inductive_intervals.csv",
        "thresholds": "strict_inductive_threshold_folds.csv",
        "covariance_folds": "strict_inductive_covariance_folds.csv",
        "permutations": "strict_inductive_permutations.csv",
        "comparisons": "strict_inductive_full_comparison.csv",
        "failures": "strict_inductive_failures.csv",
    }
    written: dict[str, Path] = {}
    for key, filename in table_names.items():
        rows = payload.get(key, [])
        if not isinstance(rows, list):
            raise TypeError(f"{key} output must be a list")
        path = output_dir / filename
        _write_csv(path, rows, _fields(rows, ("site", "scene_id", "status", "reason")))
        written[key] = path
    summary_path = output_dir / "strict_inductive_summary.json"
    _write_json(summary_path, payload)
    written["summary"] = summary_path
    return written


def run_strict_inductive(
    *,
    root: Path,
    block_manifest_path: Path,
    input_manifest_path: Path,
    summary_path: Path | None,
    maps_dir: Path,
    reference_dir: Path,
    raw_dir: Path,
    speclib_dir: Path,
    output_dir: Path,
    workers: int | None = None,
) -> dict[str, Path]:
    """Execute the frozen sensitivity after complete preflight validation."""
    from tanager_spec.io import load_tanager_sr_hdf5

    handoff, preflight = preflight_strict_inductive(
        root=root,
        block_manifest_path=block_manifest_path,
        input_manifest_path=input_manifest_path,
        summary_path=summary_path,
        maps_dir=maps_dir,
        reference_dir=reference_dir,
    )
    archive_path = Path(
        next(
            record["resolved_path"]
            for record in preflight["declared_inputs"]
            if record["id"] == "usgs-splib07a-archive"
        )
    )
    library_hashes = _validate_speclib_extraction(archive_path, speclib_dir)
    tables: dict[str, list[dict[str, Any]]] = {
        key: []
        for key in (
            "metrics",
            "support",
            "intervals",
            "thresholds",
            "covariance_folds",
            "permutations",
            "comparisons",
            "failures",
        )
    }
    site_provenance: list[dict[str, Any]] = []
    source_cubes_opened = False
    for site_id, site in sorted(handoff.sites.items()):
        primary = site.scales["L"]
        if not primary.blocks:
            tables["failures"].append(
                {
                    "site": site_id,
                    "scene_id": site.scene_id,
                    "stage": "site",
                    "block_id": "",
                    "layer": "",
                    "status": "unavailable",
                    "reason": "no_complete_primary_L_blocks; blocks_were_not_shrunk",
                }
            )
            reference_path = reference_dir / f"rockwell_{site_id}_{site.scene_id}.tif"
            reference_classes = _read_raster_values(reference_path, site.grid)
            unavailable_scores = np.full(site.grid.shape, np.nan, dtype=float)
            for mineral, positive_classes in MINERAL_TO_ROCKWELL.items():
                binary = _binary_reference(reference_classes, positive_classes)
                full_path = maps_dir / f"{site_id}_{site.scene_id}_mf_{mineral}.tif"
                full_score = (
                    _read_raster_values(full_path, site.grid) if full_path.is_file() else None
                )
                evaluated = _evaluate_layer(
                    site=site,
                    mineral=mineral,
                    strict_score=unavailable_scores,
                    reference=binary,
                    full_score=full_score,
                    full_path=full_path,
                    workers=workers,
                )
                for key, rows in evaluated.items():
                    tables[key].extend(rows)
            continue
        declared_scene = next(
            record
            for record in preflight["declared_inputs"]
            if record["id"] == f"tanager-{site_id}-1"
        )
        scene_path = Path(declared_scene["resolved_path"])
        requested_scene = (raw_dir / f"{site.scene_id}_{TANAGER_SR_ASSET}.h5").resolve()
        _require_equal(f"{site_id} requested scene path", requested_scene, scene_path.resolve())
        cube_raw, wavelengths = load_tanager_sr_hdf5(scene_path)
        source_cubes_opened = True
        cube, quality = mask_tanager_scene(cube_raw, wavelengths, scene_path)
        _require_equal(f"{site_id} masked cube grid", _cube_grid(cube), site.grid)
        _require_equal(
            f"{site_id} retained QA bands", quality.retained_bands, FROZEN_RETAINED_BANDS
        )
        endmembers = select_endmembers(load_library(speclib_dir, wavelengths))
        missing = sorted(set(TARGET_MINERALS) - set(endmembers))
        if missing:
            raise StrictInductiveError(f"{site_id} lacks frozen endmembers: {', '.join(missing)}")
        strict_scores = {
            mineral: np.full(site.grid.shape, np.nan, dtype=float) for mineral in endmembers
        }
        for numeric_id, block in zip(primary.block_ids, primary.blocks, strict=True):
            try:
                fold_scores, fold_support = strict_fold_scores(
                    cube,
                    endmembers,
                    primary.values,
                    numeric_id,
                    primary.halo_pixels,
                    ridge=FROZEN_RIDGE,
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                tables["failures"].append(
                    {
                        "site": site_id,
                        "scene_id": site.scene_id,
                        "stage": "covariance_fold",
                        "block_id": block.block_id,
                        "layer": "all_minerals",
                        "status": "unavailable",
                        "reason": str(error),
                    }
                )
                continue
            tables["covariance_folds"].append(
                {
                    "site": site_id,
                    "scene_id": site.scene_id,
                    "scale": "L",
                    "block_id": block.block_id,
                    "numeric_block_id": numeric_id,
                    "halo_pixels": primary.halo_pixels,
                    "ridge": FROZEN_RIDGE,
                    **fold_support,
                }
            )
            held = primary.values == numeric_id
            for mineral in endmembers:
                strict_scores[mineral][held] = fold_scores[f"{mineral}_mf"].values[held]

        reference_path = reference_dir / f"rockwell_{site_id}_{site.scene_id}.tif"
        reference_classes = _read_raster_values(reference_path, site.grid)
        for mineral, positive_classes in MINERAL_TO_ROCKWELL.items():
            binary = _binary_reference(reference_classes, positive_classes)
            full_path = maps_dir / f"{site_id}_{site.scene_id}_mf_{mineral}.tif"
            full_score = _read_raster_values(full_path, site.grid) if full_path.is_file() else None
            evaluated = _evaluate_layer(
                site=site,
                mineral=mineral,
                strict_score=strict_scores[mineral],
                reference=binary,
                full_score=full_score,
                full_path=full_path,
                workers=workers,
            )
            for key, rows in evaluated.items():
                tables[key].extend(rows)
        site_provenance.append(
            {
                "site": site_id,
                "scene_id": site.scene_id,
                "scene_path": str(scene_path),
                "scene_sha256": declared_scene["sha256"],
                "quality_policy": "tanager_rocks.quality.mask_tanager_scene",
                "retained_bands": quality.retained_bands,
                "ridge": FROZEN_RIDGE,
                "endmembers": {
                    mineral: {
                        "sample": endmember.sample,
                        "extracted_sha256": library_hashes[
                            str(Path("ChapterM_Minerals") / endmember.sample)
                        ],
                    }
                    for mineral, endmember in sorted(endmembers.items())
                },
                "reference_path": str(reference_path),
                "reference_sha256": sha256_file(reference_path),
            }
        )

    for site_id in sorted(handoff.sites):
        family_rows = [row for row in tables["permutations"] if row["site"] == site_id]
        auc_adjusted = benjamini_hochberg([row["auc_p_value"] for row in family_rows])
        balanced_adjusted = benjamini_hochberg(
            [row["balanced_accuracy_p_value"] for row in family_rows]
        )
        for row, auc_q, balanced_q in zip(
            family_rows,
            auc_adjusted,
            balanced_adjusted,
            strict=True,
        ):
            row["auc_q_value"] = float(auc_q)
            row["balanced_accuracy_q_value"] = float(balanced_q)

    for rows in tables.values():
        rows.sort(
            key=lambda row: tuple(
                str(row.get(key, "")) for key in ("site", "scene_id", "layer", "block_id", "metric")
            )
        )
    payload = {
        "schema_version": "1.0",
        "analysis": "strict_inductive_mtmf_covariance_sensitivity",
        "status": "complete_with_unavailable_components" if tables["failures"] else "complete",
        "sensitivity_statement": SENSITIVITY_STATEMENT,
        "tuned_model": False,
        "blocks_shrunk": False,
        "preflight": {**preflight, "source_cubes_opened": source_cubes_opened},
        "site_provenance": site_provenance,
        **tables,
    }
    return _write_outputs(output_dir, payload)


def failure_payload(error: Exception) -> dict[str, Any]:
    """Return strict JSON-ready unavailability for a failed preflight."""
    return {
        "schema_version": "1.0",
        "analysis": "strict_inductive_mtmf_covariance_sensitivity",
        "status": "unavailable",
        "sensitivity_statement": SENSITIVITY_STATEMENT,
        "tuned_model": False,
        "blocks_shrunk": False,
        "source_cubes_opened": False,
        "failures": [
            {
                "stage": "preflight",
                "status": "unavailable",
                "error_type": type(error).__name__,
                "reason": str(error),
            }
        ],
    }


__all__ = [
    "ANCHOR_SCENES",
    "FROZEN_RETAINED_BANDS",
    "FROZEN_RIDGE",
    "SENSITIVITY_STATEMENT",
    "StrictInductiveError",
    "failure_payload",
    "held_block_halo_mask",
    "preflight_strict_inductive",
    "run_strict_inductive",
    "sha256_file",
    "strict_fold_scores",
    "strict_json_dumps",
    "validate_block_handoff",
]
