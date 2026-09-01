#!/usr/bin/env python3
"""Run the frozen lead-scene spatial-validation protocol.

This command consumes the corrected score GeoTIFFs and aligned Rockwell ASTER
reference rasters already produced by the repository.  It writes only generated
tables, manifests, and JSON metadata under ``data/processed/spatial_validation``.
The legacy pixelwise validation remains unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

from tanager_minmap.config import SEED, SITES
from tanager_minmap.reference import (
    FEATURE_TO_ROCKWELL,
    MINERAL_TO_ROCKWELL,
    ROCKWELL_EXCLUDED,
)
from tanager_minmap.spatial_validation import (
    BOOTSTRAP_REPLICATES,
    MAX_PAIRS_PER_LAG,
    PERMUTATION_REPLICATES,
    Block,
    bearing_block_counts,
    benjamini_hochberg,
    block_balanced_youden,
    block_bootstrap_intervals,
    block_dimensions,
    categorical_block_raster,
    complete_blocks,
    empirical_semivariogram,
    fit_exponential_variogram,
    governance_status,
    pooled_metrics,
    sample_blocks,
    spatial_cross_fit,
    whole_block_permutation_test,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPS_DIR = ROOT / "data" / "intermediate" / "maps"
DEFAULT_REFERENCE_DIR = ROOT / "data" / "reference"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "spatial_validation"
PREREGISTRATION_PATH = ROOT / "docs" / "m2_spatial_validation_preregistration.md"
CHANCE_LEVEL = 0.5
MIN_CONFIRMATORY_BLOCKS = 10


@dataclass(frozen=True)
class EndpointSpec:
    family: str
    layer: str
    raster_suffix: str
    positive_classes: frozenset[int]


@dataclass(frozen=True)
class RasterGrid:
    shape: tuple[int, int]
    crs: str
    transform: Affine
    pixel_size: float


@dataclass(frozen=True)
class EndpointInput:
    spec: EndpointSpec
    score: np.ndarray
    score_path: Path
    binary_reference: np.ndarray


ENDPOINTS: tuple[EndpointSpec, ...] = tuple(
    [
        EndpointSpec("feature", layer, layer, classes)
        for layer, classes in FEATURE_TO_ROCKWELL.items()
    ]
    + [
        EndpointSpec("mtmf", layer, f"mf_{layer}", classes)
        for layer, classes in MINERAL_TO_ROCKWELL.items()
    ]
)

METRIC_CSV_FIELDS: tuple[str, ...] = (
    "site",
    "scene_id",
    "scale",
    "family",
    "layer",
    "classification",
    "positive_classes",
    "governance_status",
    "complete_blocks",
    "positive_bearing_blocks",
    "negative_bearing_blocks",
    "rank_evaluated_blocks",
    "rank_observations",
    "evaluated_blocks",
    "skipped_blocks",
    "auc",
    "balanced_accuracy",
    "positive_f1",
    "negative_f1",
    "macro_f1",
    "tpr",
    "fpr",
    "prevalence",
    "n_pos",
    "n_neg",
    "rank_n_pos",
    "rank_n_neg",
    "threshold_n_pos",
    "threshold_n_neg",
    "threshold_min",
    "threshold_median",
    "threshold_max",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _read_score(path: Path) -> tuple[np.ndarray, RasterGrid]:
    if not path.exists():
        raise FileNotFoundError(f"score raster missing: {path}")
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"expected one-band score raster: {path}")
        if dataset.crs is None or not dataset.crs.is_projected:
            raise ValueError(f"score raster must use a projected CRS: {path}")
        if not math.isclose(dataset.transform.b, 0.0) or not math.isclose(dataset.transform.d, 0.0):
            raise ValueError(f"rotated score grids are not supported: {path}")
        x_size = abs(float(dataset.transform.a))
        y_size = abs(float(dataset.transform.e))
        if not math.isclose(x_size, y_size, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"score raster must have square pixels: {path}")
        values = dataset.read(1, masked=True).filled(np.nan).astype(float)
        grid = RasterGrid(
            shape=values.shape,
            crs=dataset.crs.to_string(),
            transform=dataset.transform,
            pixel_size=x_size,
        )
    return values, grid


def _read_reference(path: Path, expected: RasterGrid) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"aligned Rockwell reference missing: {path}")
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"expected one-band reference raster: {path}")
        if dataset.shape != expected.shape:
            raise ValueError(f"reference shape does not match score grid: {path}")
        if dataset.crs is None or dataset.crs.to_string() != expected.crs:
            raise ValueError(f"reference CRS does not match score grid: {path}")
        if dataset.transform != expected.transform:
            raise ValueError(f"reference transform does not match score grid: {path}")
        return dataset.read(1, masked=False).astype(float)


def _binary_reference(reference: np.ndarray, positive_classes: frozenset[int]) -> np.ndarray:
    domain = np.isfinite(reference)
    for class_value in ROCKWELL_EXCLUDED:
        domain &= reference != class_value
    binary = np.full(reference.shape, np.nan, dtype=float)
    binary[domain] = np.isin(reference[domain], tuple(sorted(positive_classes))).astype(float)
    return binary


def load_site_inputs(
    site_id: str, maps_dir: Path, reference_dir: Path
) -> tuple[str, RasterGrid, tuple[EndpointInput, ...], Path]:
    """Load all frozen lead-scene endpoints and their aligned reference."""
    site = SITES[site_id]
    scene_id = site.scene_ids[0]
    first_path = maps_dir / f"{site_id}_{scene_id}_{ENDPOINTS[0].raster_suffix}.tif"
    _, grid = _read_score(first_path)
    reference_path = reference_dir / f"rockwell_{site_id}_{scene_id}.tif"
    reference = _read_reference(reference_path, grid)
    inputs: list[EndpointInput] = []
    for spec in ENDPOINTS:
        score_path = maps_dir / f"{site_id}_{scene_id}_{spec.raster_suffix}.tif"
        score, score_grid = _read_score(score_path)
        if score_grid != grid:
            raise ValueError(f"score grid differs from the first endpoint: {score_path}")
        inputs.append(
            EndpointInput(
                spec=spec,
                score=score,
                score_path=score_path,
                binary_reference=_binary_reference(reference, spec.positive_classes),
            )
        )
    return scene_id, grid, tuple(inputs), reference_path


def _classification(site_id: str, spec: EndpointSpec) -> str:
    if site_id == "goldfield" and spec.family == "feature" and spec.layer == "al_oh_doublet":
        return "primary"
    if (
        site_id == "goldfield"
        and (spec.family, spec.layer)
        in {("feature", "gypsum_carbonate"), ("mtmf", "alunite"), ("mtmf", "muscovite")}
    ) or (site_id == "bingham" and (spec.family, spec.layer) == ("feature", "gypsum_carbonate")):
        return "key_secondary"
    if spec.layer == "jarosite":
        return "descriptive"
    return "exploratory"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _run_permutation_test(
    samples: tuple[Any, ...],
    *,
    halo_pixels: int,
    permutations: int,
    seed: int,
    workers: int,
) -> Any:
    """Forward computation controls without changing the seeded estimand."""
    return whole_block_permutation_test(
        samples,
        halo_pixels=halo_pixels,
        permutations=permutations,
        seed=seed,
        workers=workers,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _protocol_parameters(
    *,
    max_pairs: int,
    bootstrap_replicates: int,
    permutations: int,
    seed: int,
) -> dict[str, int]:
    """Return the protocol-affecting command parameters with stable keys."""
    return {
        "max_pairs_per_field_lag": max_pairs,
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutations,
        "seed": seed,
    }


def _protocol_compliant(parameters: dict[str, int]) -> bool:
    """Return whether parameters equal every frozen protocol default."""
    return parameters == _protocol_parameters(
        max_pairs=MAX_PAIRS_PER_LAG,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        permutations=PERMUTATION_REPLICATES,
        seed=SEED,
    )


def _block_manifest_link(path: Path) -> dict[str, str]:
    """Return the summary-to-manifest path and content-hash linkage."""
    return {"path": path.name, "sha256": _sha256(path)}


def _finalize_transfer_threshold_rows(
    rows: list[dict[str, Any]], block_manifest_path: Path
) -> list[dict[str, Any]]:
    """Attach the hash of the completed block handoff without mutating inputs."""
    manifest_hash = _sha256(block_manifest_path)
    return [
        {
            **row,
            "block_manifest_path": block_manifest_path.name,
            "block_manifest_sha256": manifest_hash,
        }
        for row in rows
    ]


def _write_block_raster(
    path: Path,
    grid: RasterGrid,
    blocks: tuple[Block, ...],
    *,
    site_id: str,
    scene_id: str,
    scale: str,
) -> dict[str, Any]:
    """Write one exact-anchor-grid categorical block raster atomically."""
    values, mapping = categorical_block_raster(grid.shape, blocks)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype="uint32",
        crs=grid.crs,
        transform=grid.transform,
        nodata=0,
        compress="lzw",
    ) as dataset:
        dataset.write(values, 1)
        dataset.update_tags(
            site_id=site_id,
            anchor_scene_id=scene_id,
            block_scale=scale,
            complete_blocks_only="true",
            numeric_id_zero="nodata",
        )
    temporary.replace(path)
    return {
        "block_raster": path.name,
        "block_raster_sha256": _sha256(path),
        "complete_block_ids": list(mapping),
        "numeric_to_string_block_ids": {str(key): value for key, value in mapping.items()},
        "nodata": 0,
        "dtype": "uint32",
    }


def _block_manifest_row(
    site_id: str,
    scene_id: str,
    scale: str,
    block: Block,
    numeric_block_id: int,
    halo_pixels: int,
    grid: RasterGrid,
) -> dict[str, Any]:
    left, bottom, right, top = window_bounds(
        Window(
            block.col_start,
            block.row_start,
            block.col_stop - block.col_start,
            block.row_stop - block.row_start,
        ),
        grid.transform,
    )
    return {
        "site": site_id,
        "scene_id": scene_id,
        "scale": scale,
        "geometry": scale,
        "block_id": block.block_id,
        "numeric_block_id": numeric_block_id,
        "complete": True,
        "halo_pixels": halo_pixels,
        "block_row": block.block_row,
        "block_col": block.block_col,
        "row_start": block.row_start,
        "row_stop": block.row_stop,
        "col_start": block.col_start,
        "col_stop": block.col_stop,
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
        "crs": grid.crs,
    }


def _base_endpoint_row(
    site_id: str,
    scene_id: str,
    scale: str,
    endpoint: EndpointInput,
    n_blocks: int,
    positive_blocks: int,
    negative_blocks: int,
    status: str,
) -> dict[str, Any]:
    return {
        "site": site_id,
        "scene_id": scene_id,
        "scale": scale,
        "family": endpoint.spec.family,
        "layer": endpoint.spec.layer,
        "classification": _classification(site_id, endpoint.spec),
        "positive_classes": ";".join(
            str(value) for value in sorted(endpoint.spec.positive_classes)
        ),
        "governance_status": status,
        "complete_blocks": n_blocks,
        "positive_bearing_blocks": positive_blocks,
        "negative_bearing_blocks": negative_blocks,
    }


def _transfer_threshold_row(
    site_id: str,
    scene_id: str,
    endpoint: EndpointInput,
    samples: tuple[Any, ...],
    *,
    governance: str,
    positive_blocks: int,
    negative_blocks: int,
    complete_block_count: int,
    reference_path: Path,
    raster_record: dict[str, Any],
    grid: RasterGrid,
) -> dict[str, Any]:
    """Build one primary-L threshold handoff row from all usable blocks."""
    n_usable = sum(sample.paired_values()[0].size for sample in samples)
    threshold = None
    reason = "counts_and_maps_only_support"
    if governance != "counts_and_maps_only":
        fitted = block_balanced_youden(samples)
        if fitted is not None and math.isfinite(fitted):
            threshold = float(fitted)
            reason = ""
        else:
            reason = "block_balanced_youden_unavailable"
    return {
        "site": site_id,
        "scene_id": scene_id,
        "scale": "L",
        "family": endpoint.spec.family,
        "layer": endpoint.spec.layer,
        "classification": _classification(site_id, endpoint.spec),
        "positive_classes": ";".join(
            str(value) for value in sorted(endpoint.spec.positive_classes)
        ),
        "governance_status": governance,
        "complete_blocks": complete_block_count,
        "positive_bearing_blocks": positive_blocks,
        "negative_bearing_blocks": negative_blocks,
        "n_usable": int(n_usable),
        "threshold_status": "available" if threshold is not None else "unavailable",
        "threshold": threshold,
        "unavailable_reason": reason,
        "threshold_method": "block_balanced_youden_all_usable_complete_primary_L_blocks",
        "spatial_prereg_sha256": _sha256(PREREGISTRATION_PATH),
        "source_score_path": _path_label(endpoint.score_path),
        "source_score_sha256": _sha256(endpoint.score_path),
        "source_reference_path": _path_label(reference_path),
        "source_reference_sha256": _sha256(reference_path),
        "block_manifest_path": None,
        "block_manifest_sha256": None,
        "block_raster_path": raster_record["block_raster"],
        "block_raster_sha256": raster_record["block_raster_sha256"],
        "block_shape_rows": grid.shape[0],
        "block_shape_cols": grid.shape[1],
        "block_crs": grid.crs,
        "block_transform": json.dumps(
            [
                grid.transform.a,
                grid.transform.b,
                grid.transform.c,
                grid.transform.d,
                grid.transform.e,
                grid.transform.f,
            ],
            separators=(",", ":"),
        ),
    }


def _apply_fdr(rows: list[dict[str, Any]]) -> None:
    for site_id in SITES:
        for scale in ("L", "2L"):
            for family in ("feature", "mtmf"):
                family_rows = [
                    row
                    for row in rows
                    if row["site"] == site_id
                    and row["scale"] == scale
                    and row["family"] == family
                    and row["classification"] != "primary"
                ]
                adjusted_auc = benjamini_hochberg([row["auc_p_value"] for row in family_rows])
                adjusted_balanced = benjamini_hochberg(
                    [row["balanced_accuracy_p_value"] for row in family_rows]
                )
                for row, auc_q, balanced_q in zip(
                    family_rows, adjusted_auc, adjusted_balanced, strict=True
                ):
                    row["auc_q_value"] = float(auc_q)
                    row["balanced_accuracy_q_value"] = float(balanced_q)


def _block_manifest_payload(
    results: list[dict[str, Any]], *, protocol_parameters: dict[str, int]
) -> dict[str, Any]:
    """Build the cross-stage JSON handoff without deriving new block geometry."""
    sites: dict[str, Any] = {}
    all_blocks: list[dict[str, Any]] = []
    for result in results:
        summary = result["site_summary"]
        handoff = result["block_handoff"]
        primary = handoff["L"]
        site_id = summary["site"]
        sites[site_id] = {
            "scene_id": summary["scene_id"],
            "primary_scale": "L",
            # These two keys are the stable repeatability-stage API.
            "block_raster": primary["block_raster"],
            "complete_block_ids": primary["complete_block_ids"],
            "numeric_to_string_block_ids": primary["numeric_to_string_block_ids"],
            "grid": {
                "shape": summary["shape"],
                "crs": summary["crs"],
                "transform": summary["transform"],
                "pixel_size_metres": summary["pixel_size_metres"],
            },
            "scales": handoff,
            "provenance": {
                "anchor_grid_source": summary["scores"][0],
                "complete_blocks_only": True,
                "raster_nodata": 0,
                "numeric_ids": "positive row-major integers; zero is nodata",
            },
        }
        all_blocks.extend(result["blocks"])
    return {
        "schema_version": "1.0",
        "manifest_type": "spatial_validation_complete_blocks",
        "protocol": {
            "path": str(PREREGISTRATION_PATH.relative_to(ROOT)),
            "sha256": _sha256(PREREGISTRATION_PATH),
            "parameters": dict(protocol_parameters),
            "protocol_compliant": _protocol_compliant(protocol_parameters),
        },
        "sites": sites,
        # The tabular records keep this JSON compatible with consumers that
        # use the same rows as block_manifest.csv.
        "blocks": all_blocks,
        "strict_inductive_covariance": {
            "status": "deferred",
            "reason": (
                "requires cube-level MTMF covariance recomputation outside this raster-only command"
            ),
        },
    }


def _matching_row(
    rows: list[dict[str, Any]],
    *,
    scale: str,
    metric: str | None = None,
) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if row.get("site") == "goldfield"
        and row.get("family") == "feature"
        and row.get("layer") == "al_oh_doublet"
        and row.get("scale") == scale
        and (metric is None or row.get("metric") == metric)
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple Goldfield Al-OH records for scale={scale!r}, metric={metric!r}")
    return matches[0] if matches else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _external_reference_gate(
    metric_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    *,
    protocol_compliant: bool,
) -> dict[str, Any]:
    """Classify the preregistered Goldfield Al-OH external-reference gate."""
    l_metric = _matching_row(metric_rows, scale="L")
    two_l_metric = _matching_row(metric_rows, scale="2L")
    auc_interval = _matching_row(interval_rows, scale="L", metric="auc")
    balanced_interval = _matching_row(interval_rows, scale="L", metric="balanced_accuracy")

    positive_blocks = _finite_number(l_metric.get("positive_bearing_blocks")) if l_metric else None
    negative_blocks = _finite_number(l_metric.get("negative_bearing_blocks")) if l_metric else None
    support_available = positive_blocks is not None and negative_blocks is not None
    support_passed = bool(
        support_available
        and positive_blocks >= MIN_CONFIRMATORY_BLOCKS
        and negative_blocks >= MIN_CONFIRMATORY_BLOCKS
    )
    auc_interval_eligible = bool(auc_interval and auc_interval.get("gate_eligible") is True)
    balanced_interval_eligible = bool(
        balanced_interval and balanced_interval.get("gate_eligible") is True
    )
    auc_lower = _finite_number(auc_interval.get("lower")) if auc_interval_eligible else None
    balanced_lower = (
        _finite_number(balanced_interval.get("lower")) if balanced_interval_eligible else None
    )
    auc_2l = _finite_number(two_l_metric.get("auc")) if two_l_metric else None

    def interval_reason(
        row: dict[str, Any] | None,
        *,
        available: bool,
        generic: str,
    ) -> str | None:
        if available:
            return None
        if row is None:
            return generic
        if row.get("gate_eligible") is not True:
            reason = row.get("unavailable_reason")
            return str(reason) if reason else "bootstrap_interval_not_gate_eligible"
        return generic

    conditions = {
        "support_eligible_at_L": {
            "available": support_available,
            "passed": support_passed,
            "positive_bearing_blocks": positive_blocks,
            "negative_bearing_blocks": negative_blocks,
            "minimum_required_each": MIN_CONFIRMATORY_BLOCKS,
            "reason": None if support_available else "L_support_record_unavailable",
        },
        "auc_lower_95_above_half_at_L": {
            "available": auc_lower is not None,
            "passed": bool(auc_lower is not None and auc_lower > CHANCE_LEVEL),
            "observed_lower_95": auc_lower,
            "scheduled_replicates": (
                auc_interval.get("scheduled_replicates") if auc_interval else None
            ),
            "valid_replicates": (auc_interval.get("valid_replicates") if auc_interval else None),
            "finite_fraction": auc_interval.get("finite_fraction") if auc_interval else None,
            "gate_eligible": auc_interval_eligible,
            "threshold": CHANCE_LEVEL,
            "comparison": "strictly_greater_than",
            "reason": interval_reason(
                auc_interval,
                available=auc_interval_eligible and auc_lower is not None,
                generic="L_auc_interval_unavailable",
            ),
        },
        "balanced_accuracy_lower_95_above_half_at_L": {
            "available": balanced_lower is not None,
            "passed": bool(balanced_lower is not None and balanced_lower > CHANCE_LEVEL),
            "observed_lower_95": balanced_lower,
            "scheduled_replicates": (
                balanced_interval.get("scheduled_replicates") if balanced_interval else None
            ),
            "valid_replicates": (
                balanced_interval.get("valid_replicates") if balanced_interval else None
            ),
            "finite_fraction": (
                balanced_interval.get("finite_fraction") if balanced_interval else None
            ),
            "gate_eligible": balanced_interval_eligible,
            "threshold": CHANCE_LEVEL,
            "comparison": "strictly_greater_than",
            "reason": interval_reason(
                balanced_interval,
                available=balanced_interval_eligible and balanced_lower is not None,
                generic="L_balanced_accuracy_interval_unavailable",
            ),
        },
        "auc_direction_positive_at_2L": {
            "available": auc_2l is not None,
            "passed": bool(auc_2l is not None and auc_2l > CHANCE_LEVEL),
            "observed_auc": auc_2l,
            "threshold": CHANCE_LEVEL,
            "comparison": "strictly_greater_than",
            "reason": None if auc_2l is not None else "2L_auc_point_estimate_unavailable",
        },
    }
    rank_conditions_available = bool(
        conditions["auc_lower_95_above_half_at_L"]["available"]
        and conditions["auc_direction_positive_at_2L"]["available"]
    )
    rank_conditions_passed = bool(
        conditions["auc_lower_95_above_half_at_L"]["passed"]
        and conditions["auc_direction_positive_at_2L"]["passed"]
    )
    if not protocol_compliant or not support_passed or not rank_conditions_available:
        classification = "unavailable"
    elif not rank_conditions_passed:
        classification = "fail"
    elif conditions["balanced_accuracy_lower_95_above_half_at_L"]["passed"]:
        classification = "pass"
    else:
        classification = "ranking_discrimination_only"

    unavailable_reasons = [
        condition["reason"]
        for condition in conditions.values()
        if not condition["available"] and condition["reason"] is not None
    ]
    if not protocol_compliant:
        unavailable_reasons.insert(0, "nondefault_protocol_parameters")
    if support_available and not support_passed:
        unavailable_reasons.append("insufficient_confirmatory_block_support_at_L")
    return {
        "target": "goldfield_feature_al_oh_doublet",
        "status": classification,
        "classification": classification,
        "evaluable": classification != "unavailable",
        "passed": classification == "pass",
        "protocol_compliant": protocol_compliant,
        "conditions": conditions,
        "unavailable_reasons": unavailable_reasons,
    }


def _combined_public_gate(external_reference_gate: dict[str, Any]) -> dict[str, Any]:
    """Keep the public claim pending until the separate repeatability stage."""
    return {
        "status": "pending_repeatability",
        "classification": "pending_repeatability",
        "passed": False,
        "external_reference_classification": external_reference_gate["classification"],
        "repeatability_status": "pending",
        "reason": "repeatability_gate_has_not_been_evaluated_by_this_command",
    }


def run_site(
    site_id: str,
    *,
    maps_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    max_pairs: int,
    bootstrap_replicates: int,
    permutations: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    """Execute one site's two frozen block scales and return generated rows."""
    scene_id, grid, endpoints, reference_path = load_site_inputs(site_id, maps_dir, reference_dir)
    variogram_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    fits = []
    reference_cache: dict[frozenset[int], tuple[Any, Any]] = {}

    fields: list[tuple[str, str, str, np.ndarray]] = []
    for endpoint in endpoints:
        fields.append((endpoint.spec.family, endpoint.spec.layer, "score", endpoint.score))
        classes = endpoint.spec.positive_classes
        if classes not in reference_cache:
            points = empirical_semivariogram(
                endpoint.binary_reference,
                pixel_size=grid.pixel_size,
                max_pairs=max_pairs,
            )
            finite_reference = endpoint.binary_reference[np.isfinite(endpoint.binary_reference)]
            fit = fit_exponential_variogram(points, field_variance=float(np.var(finite_reference)))
            reference_cache[classes] = (points, fit)
        fields.append(
            (
                endpoint.spec.family,
                endpoint.spec.layer,
                "reference_indicator",
                endpoint.binary_reference,
            )
        )

    for family, layer, field_kind, field in fields:
        if field_kind == "reference_indicator":
            endpoint = next(
                item
                for item in endpoints
                if item.spec.family == family and item.spec.layer == layer
            )
            points, fit = reference_cache[endpoint.spec.positive_classes]
        else:
            points = empirical_semivariogram(field, pixel_size=grid.pixel_size, max_pairs=max_pairs)
            finite_field = field[np.isfinite(field)]
            if finite_field.size == 0:
                raise ValueError(f"{site_id} {family}/{layer} score has no finite cells")
            fit = fit_exponential_variogram(points, field_variance=float(np.var(finite_field)))
        fits.append(fit)
        field_name = f"{family}:{layer}:{field_kind}"
        for point in points:
            variogram_rows.append(
                {
                    "site": site_id,
                    "scene_id": scene_id,
                    "field": field_name,
                    **asdict(point),
                }
            )
        fit_rows.append(
            {
                "site": site_id,
                "scene_id": scene_id,
                "field": field_name,
                **asdict(fit),
            }
        )
        print(f"{site_id}: {field_name} practical range={fit.practical_range:.1f} m ({fit.method})")

    primary_side, halo_pixels = block_dimensions(
        [fit.practical_range for fit in fits], grid.pixel_size
    )
    block_rows: list[dict[str, Any]] = []
    endpoint_block_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    transfer_threshold_rows: list[dict[str, Any]] = []
    scale_summaries: list[dict[str, Any]] = []
    block_handoff: dict[str, dict[str, Any]] = {}

    for scale, side in (("L", primary_side), ("2L", 2 * primary_side)):
        blocks = complete_blocks(grid.shape, side)
        raster_record = _write_block_raster(
            output_dir / f"block_ids_{site_id}_{scale}.tif",
            grid,
            blocks,
            site_id=site_id,
            scene_id=scene_id,
            scale=scale,
        )
        block_handoff[scale] = {
            **raster_record,
            "site_id": site_id,
            "anchor_scene_id": scene_id,
            "scale": scale,
            "block_side_pixels": side,
            "block_side_metres": side * grid.pixel_size,
            "halo_pixels": halo_pixels,
            "halo_metres": halo_pixels * grid.pixel_size,
            "complete_blocks": len(blocks),
        }
        for numeric_block_id, block in enumerate(blocks, start=1):
            block_rows.append(
                _block_manifest_row(
                    site_id,
                    scene_id,
                    scale,
                    block,
                    numeric_block_id,
                    halo_pixels,
                    grid,
                )
            )
        covered_rows = (grid.shape[0] // side) * side
        covered_cols = (grid.shape[1] // side) * side
        scale_summaries.append(
            {
                "scale": scale,
                "block_side_pixels": side,
                "block_side_metres": side * grid.pixel_size,
                "halo_pixels": halo_pixels,
                "halo_metres": halo_pixels * grid.pixel_size,
                "complete_blocks": len(blocks),
                "covered_pixels": covered_rows * covered_cols,
                "edge_excluded_pixels": grid.shape[0] * grid.shape[1] - covered_rows * covered_cols,
            }
        )
        for endpoint in endpoints:
            samples = sample_blocks(endpoint.score, endpoint.binary_reference, blocks)
            positive_blocks, negative_blocks = bearing_block_counts(samples)
            status = governance_status(positive_blocks, negative_blocks)
            base = _base_endpoint_row(
                site_id,
                scene_id,
                scale,
                endpoint,
                len(blocks),
                positive_blocks,
                negative_blocks,
                status,
            )
            if scale == "L":
                transfer_threshold_rows.append(
                    _transfer_threshold_row(
                        site_id,
                        scene_id,
                        endpoint,
                        samples,
                        governance=status,
                        positive_blocks=positive_blocks,
                        negative_blocks=negative_blocks,
                        complete_block_count=len(blocks),
                        reference_path=reference_path,
                        raster_record=raster_record,
                        grid=grid,
                    )
                )
            for sample in samples:
                score, reference = sample.paired_values()
                endpoint_block_rows.append(
                    {
                        **base,
                        "block_id": sample.block.block_id,
                        "n_usable": int(score.size),
                        "n_pos": int(np.count_nonzero(reference == 1)),
                        "n_neg": int(np.count_nonzero(reference == 0)),
                    }
                )

            metric_row = {
                **base,
                "rank_evaluated_blocks": 0,
                "rank_observations": 0,
                "evaluated_blocks": 0,
                "skipped_blocks": len(blocks),
                "auc": float("nan"),
                "balanced_accuracy": float("nan"),
                "positive_f1": float("nan"),
                "negative_f1": float("nan"),
                "macro_f1": float("nan"),
                "tpr": float("nan"),
                "fpr": float("nan"),
                "prevalence": float("nan"),
                "rank_n_pos": 0,
                "rank_n_neg": 0,
                "threshold_n_pos": 0,
                "threshold_n_neg": 0,
                "n_pos": 0,
                "n_neg": 0,
                "threshold_min": float("nan"),
                "threshold_median": float("nan"),
                "threshold_max": float("nan"),
            }
            if status != "counts_and_maps_only":
                oof = spatial_cross_fit(samples, halo_pixels=halo_pixels)
                auc_references = (
                    oof.auc_references if oof.auc_references is not None else oof.references
                )
                if auc_references.size and len(np.unique(auc_references)) == 2:
                    metrics = pooled_metrics(oof)
                    thresholds = np.asarray([fold.threshold for fold in oof.folds])
                    metric_row.update(
                        {
                            "rank_evaluated_blocks": len(set(oof.auc_block_ids.tolist())),
                            "rank_observations": int(auc_references.size),
                            "evaluated_blocks": len(oof.folds),
                            "skipped_blocks": len(oof.skipped_blocks),
                            **asdict(metrics),
                            "threshold_min": (
                                float(np.min(thresholds)) if thresholds.size else float("nan")
                            ),
                            "threshold_median": (
                                float(np.median(thresholds)) if thresholds.size else float("nan")
                            ),
                            "threshold_max": (
                                float(np.max(thresholds)) if thresholds.size else float("nan")
                            ),
                        }
                    )
                    for fold in oof.folds:
                        fold_rows.append({**base, **asdict(fold)})
                    for interval in block_bootstrap_intervals(
                        oof, replicates=bootstrap_replicates, seed=seed
                    ):
                        interval_rows.append({**base, **asdict(interval)})
                    if status == "confirmatory_eligible":
                        permutation = _run_permutation_test(
                            samples,
                            halo_pixels=halo_pixels,
                            permutations=permutations,
                            seed=seed,
                            workers=workers,
                        )
                        permutation_rows.append(
                            {
                                **base,
                                **asdict(permutation),
                                "auc_q_value": float("nan"),
                                "balanced_accuracy_q_value": float("nan"),
                            }
                        )
            metric_rows.append(metric_row)
            print(
                f"{site_id} {scale} {endpoint.spec.family}/{endpoint.spec.layer}: "
                f"{status}, positive blocks={positive_blocks}, negative blocks={negative_blocks}"
            )

    _apply_fdr(permutation_rows)
    return {
        "site_summary": {
            "site": site_id,
            "scene_id": scene_id,
            "shape": list(grid.shape),
            "crs": grid.crs,
            "transform": [
                grid.transform.a,
                grid.transform.b,
                grid.transform.c,
                grid.transform.d,
                grid.transform.e,
                grid.transform.f,
            ],
            "pixel_size_metres": grid.pixel_size,
            "site_practical_range_metres": max(fit.practical_range for fit in fits),
            "primary_block_side_pixels": primary_side,
            "halo_pixels": halo_pixels,
            "scales": scale_summaries,
            "reference": {
                "path": _path_label(reference_path),
                "sha256": _sha256(reference_path),
            },
            "scores": [
                {
                    "family": endpoint.spec.family,
                    "layer": endpoint.spec.layer,
                    "path": _path_label(endpoint.score_path),
                    "sha256": _sha256(endpoint.score_path),
                }
                for endpoint in endpoints
            ],
        },
        "variograms": variogram_rows,
        "fits": fit_rows,
        "blocks": block_rows,
        "endpoint_blocks": endpoint_block_rows,
        "metrics": metric_rows,
        "intervals": interval_rows,
        "folds": fold_rows,
        "permutations": permutation_rows,
        "transfer_thresholds": transfer_threshold_rows,
        "block_handoff": block_handoff,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=["all", *SITES], default="all")
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_LAG)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--permutations", type=int, default=PERMUTATION_REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="permutation worker processes; computation-only and seed-invariant",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    site_ids = list(SITES) if args.site == "all" else [args.site]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_site(
            site_id,
            maps_dir=args.maps_dir.resolve(),
            reference_dir=args.reference_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            max_pairs=args.max_pairs,
            bootstrap_replicates=args.bootstrap_replicates,
            permutations=args.permutations,
            seed=args.seed,
            workers=args.workers,
        )
        for site_id in site_ids
    ]
    combined = {
        key: [row for result in results for row in result[key]]
        for key in (
            "variograms",
            "fits",
            "blocks",
            "endpoint_blocks",
            "metrics",
            "intervals",
            "folds",
            "permutations",
            "transfer_thresholds",
        )
    }
    schemas = {
        "variograms": [
            "site",
            "scene_id",
            "field",
            "lag_pixels",
            "distance",
            "semivariance",
            "available_pairs",
            "used_pairs",
        ],
        "fits": [
            "site",
            "scene_id",
            "field",
            "nugget",
            "sill",
            "scale",
            "practical_range",
            "method",
            "fallback_reason",
        ],
        "blocks": [
            "site",
            "scene_id",
            "scale",
            "geometry",
            "block_id",
            "numeric_block_id",
            "complete",
            "halo_pixels",
            "block_row",
            "block_col",
            "row_start",
            "row_stop",
            "col_start",
            "col_stop",
            "left",
            "bottom",
            "right",
            "top",
            "crs",
        ],
        "endpoint_blocks": [
            "site",
            "scene_id",
            "scale",
            "family",
            "layer",
            "classification",
            "positive_classes",
            "governance_status",
            "complete_blocks",
            "positive_bearing_blocks",
            "negative_bearing_blocks",
            "block_id",
            "n_usable",
            "n_pos",
            "n_neg",
        ],
        "metrics": list(METRIC_CSV_FIELDS),
        "intervals": [
            "site",
            "scene_id",
            "scale",
            "family",
            "layer",
            "classification",
            "positive_classes",
            "governance_status",
            "complete_blocks",
            "positive_bearing_blocks",
            "negative_bearing_blocks",
            "metric",
            "lower",
            "upper",
            "scheduled_replicates",
            "valid_replicates",
            "finite_fraction",
            "gate_eligible",
            "unavailable_reason",
        ],
        "folds": [
            "site",
            "scene_id",
            "scale",
            "family",
            "layer",
            "classification",
            "positive_classes",
            "governance_status",
            "complete_blocks",
            "positive_bearing_blocks",
            "negative_bearing_blocks",
            "block_id",
            "threshold",
            "n_test",
            "n_pos",
            "n_neg",
            "n_training_blocks",
        ],
        "permutations": [
            "site",
            "scene_id",
            "scale",
            "family",
            "layer",
            "classification",
            "positive_classes",
            "governance_status",
            "complete_blocks",
            "positive_bearing_blocks",
            "negative_bearing_blocks",
            "auc_p_value",
            "balanced_accuracy_p_value",
            "valid_auc_permutations",
            "valid_balanced_accuracy_permutations",
            "auc_q_value",
            "balanced_accuracy_q_value",
        ],
        "transfer_thresholds": [
            "site",
            "scene_id",
            "scale",
            "family",
            "layer",
            "classification",
            "positive_classes",
            "governance_status",
            "complete_blocks",
            "positive_bearing_blocks",
            "negative_bearing_blocks",
            "n_usable",
            "threshold_status",
            "threshold",
            "unavailable_reason",
            "threshold_method",
            "spatial_prereg_sha256",
            "source_score_path",
            "source_score_sha256",
            "source_reference_path",
            "source_reference_sha256",
            "block_manifest_path",
            "block_manifest_sha256",
            "block_raster_path",
            "block_raster_sha256",
            "block_shape_rows",
            "block_shape_cols",
            "block_crs",
            "block_transform",
        ],
    }
    filenames = {
        "variograms": "empirical_variograms.csv",
        "fits": "variogram_fits.csv",
        "blocks": "block_manifest.csv",
        "endpoint_blocks": "endpoint_block_counts.csv",
        "metrics": "endpoint_metrics.csv",
        "intervals": "bootstrap_intervals.csv",
        "folds": "fold_thresholds.csv",
        "permutations": "permutation_tests.csv",
        "transfer_thresholds": "transfer_thresholds.csv",
    }
    for key, rows in combined.items():
        if key == "transfer_thresholds":
            continue
        _write_csv(args.output_dir / filenames[key], rows, schemas[key])
    protocol_parameters = _protocol_parameters(
        max_pairs=args.max_pairs,
        bootstrap_replicates=args.bootstrap_replicates,
        permutations=args.permutations,
        seed=args.seed,
    )
    protocol_compliant = _protocol_compliant(protocol_parameters)
    block_manifest_path = args.output_dir / "block_manifest.json"
    _write_json(
        block_manifest_path,
        _block_manifest_payload(results, protocol_parameters=protocol_parameters),
    )
    transfer_threshold_rows = _finalize_transfer_threshold_rows(
        combined["transfer_thresholds"], block_manifest_path
    )
    _write_csv(
        args.output_dir / filenames["transfer_thresholds"],
        transfer_threshold_rows,
        schemas["transfer_thresholds"],
    )

    external_reference_gate = _external_reference_gate(
        combined["metrics"],
        combined["intervals"],
        protocol_compliant=protocol_compliant,
    )
    _write_json(
        args.output_dir / "summary.json",
        {
            "schema_version": "1.0",
            "protocol": {
                "path": str(PREREGISTRATION_PATH.relative_to(ROOT)),
                "sha256": _sha256(PREREGISTRATION_PATH),
                "parameters": protocol_parameters,
                "protocol_compliant": protocol_compliant,
            },
            "computation": {
                "permutation_workers": args.workers,
                "affects_seeded_protocol": False,
            },
            "block_manifest": _block_manifest_link(block_manifest_path),
            "block_manifest_sha256": _sha256(block_manifest_path),
            "external_reference_gate": external_reference_gate,
            "combined_public_gate": _combined_public_gate(external_reference_gate),
            "sites": [result["site_summary"] for result in results],
            "interpretation_limits": [
                "Rockwell is alteration-zone context, not mineral-level truth.",
                (
                    "Existing MTMF rasters estimate covariance from the full scene and "
                    "represent the operational estimand."
                ),
                (
                    "Strict-inductive MTMF covariance sensitivity requires cube-level "
                    "recomputation and is not implemented by this raster-only command."
                ),
                (
                    "Counts-and-maps-only endpoints receive no effect estimate, interval, "
                    "or permutation test."
                ),
                (
                    "Exploratory-only endpoints receive estimates and block-bootstrap "
                    "intervals but no permutation decision test."
                ),
                (
                    "No class mapping, block size, scene, or registration is selected "
                    "from validation performance."
                ),
                (
                    "All-seven-scene repeatability and registration sensitivity are a "
                    "separate preregistered stage and are not implemented by this "
                    "lead-scene external-reference command."
                ),
            ],
        },
    )
    print(f"wrote spatial-validation artifacts to {args.output_dir}")
    if not protocol_compliant:
        print("WARNING: non-default replicate/thinning settings make this a non-confirmatory run")


if __name__ == "__main__":
    main()
