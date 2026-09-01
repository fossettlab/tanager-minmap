"""Frozen seven-scene repeatability processing and spatial agreement metrics.

This module implements the repeatability portion of
``docs/m2_spatial_validation_preregistration.md``.  It deliberately does not
choose a registration, create spatial blocks, or re-fit a threshold after a
comparison has started.  The reusable metric functions operate only on arrays;
the I/O layer supplies quality-masked, georeferenced scene products.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import stat
import zipfile
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import combinations, permutations
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

import numpy as np
import rasterio
import rioxarray  # noqa: F401 -- registers xarray's ``rio`` accessor.
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import Affine
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
from tanager_spec.io import load_tanager_sr_hdf5

from .config import SITES, TANAGER_SR_ASSET
from .features import FeatureDef, build_feature_defs, diagnostic_feature_maps
from .quality import mask_tanager_scene
from .reference import FEATURE_TO_ROCKWELL, MINERAL_TO_ROCKWELL, ROCKWELL_EXCLUDED
from .repeatability_resources import (
    RULE_RELATIVE_PATH as RESOURCE_RULE_RELATIVE_PATH,
)
from .repeatability_resources import (
    SOURCE_MANIFEST_RELATIVE_PATH as RESOURCE_SOURCE_MANIFEST_RELATIVE_PATH,
)
from .repeatability_resources import (
    VERIFIER_MODULE_RELATIVE_PATH,
    VERIFIER_SCRIPT_RELATIVE_PATH,
    validate_resource_admission,
)
from .repeatability_resources import (
    sha256_file as secure_sha256_file,
)
from .spatial_validation import (
    BOOTSTRAP_REPLICATES as SPATIAL_BOOTSTRAP_REPLICATES,
)
from .spatial_validation import (
    MAX_PAIRS_PER_LAG,
)
from .spatial_validation import (
    PERMUTATION_REPLICATES as SPATIAL_PERMUTATION_REPLICATES,
)
from .spatial_validation import (
    SEED as SPATIAL_SEED,
)
from .speclib import Endmember, load_library, select_endmembers
from .unmix import mtmf

logger = logging.getLogger(__name__)

MTMF_RIDGE = 1e-2
MAX_INFEASIBILITY = 1.0
UPPER_DECILE_QUANTILE = 0.90
BOOTSTRAP_REPLICATES = 10_000
NULL_REPLICATES = 9_999
SEED = 42
FINITE_REPLICATE_FRACTION = 0.95
STATISTICS_BATCH_BYTES = 8 * 1024 * 1024
SCIENTIFIC_EXECUTION_IDENTITY = "paired-complete-block-metric-contract-v2"
EXECUTION_SCHEMA_VERSION = "2.0"
PROGRESS_SCHEMA_VERSION = "2.0"
TIMING_PILOT_SCHEMA_VERSION = "2.0"
RESULT_SCHEMA_VERSION = "3.0"
PREREGISTRATION_RELATIVE_PATH = Path("docs/m2_spatial_validation_preregistration.md")
INPUT_MANIFEST_RELATIVE_PATH = Path("docs/input_manifest.json")
_RESAMPLED_METRIC_COMPONENTS = (
    "spearman",
    "transferred_iou",
    "transferred_dice",
    "transferred_prevalence_ratio",
    "transferred_boundary_distance_m",
    "rank_relative_iou",
    "rank_relative_dice",
    "rank_relative_prevalence_ratio",
    "rank_relative_boundary_distance_m",
    "rockwell_auc",
    "rockwell_balanced_accuracy",
    "rockwell_macro_f1",
)
SPATIAL_PROTOCOL_PARAMETERS = {
    "max_pairs_per_field_lag": MAX_PAIRS_PER_LAG,
    "bootstrap_replicates": SPATIAL_BOOTSTRAP_REPLICATES,
    "permutation_replicates": SPATIAL_PERMUTATION_REPLICATES,
    "seed": SPATIAL_SEED,
}

_ANCHORS = {
    "goldfield": "20240925_185504_87_4001",
    "bingham": "20250911_191523_58_4001",
}


@dataclass(frozen=True)
class PairSpec:
    """One frozen within-site comparison."""

    site_id: str
    anchor_scene_id: str
    repeat_scene_id: str
    role: str  # ``primary`` or ``secondary``


def _frozen_pairs() -> tuple[tuple[PairSpec, ...], tuple[PairSpec, ...]]:
    primary: list[PairSpec] = []
    secondary: list[PairSpec] = []
    for site_id, site in SITES.items():
        anchor = _ANCHORS[site_id]
        if anchor not in site.scene_ids:
            raise ValueError(f"frozen anchor {anchor} is not declared for {site_id}")
        repeats = tuple(scene_id for scene_id in site.scene_ids if scene_id != anchor)
        primary.extend(PairSpec(site_id, anchor, scene_id, "primary") for scene_id in repeats)
        if site_id == "goldfield":
            secondary.extend(
                PairSpec(site_id, left, right, "secondary")
                for left, right in combinations(repeats, 2)
            )
    return tuple(primary), tuple(secondary)


PRIMARY_PAIRS, SECONDARY_PAIRS = _frozen_pairs()


def site_scene_order(site_id: str) -> tuple[str, ...]:
    """Return one site's anchor first, followed by its declared repeat order."""
    site = SITES[site_id]
    anchor = _ANCHORS[site_id]
    if anchor not in site.scene_ids:
        raise ValueError(f"frozen anchor {anchor} is not declared for {site_id}")
    return (anchor, *(scene_id for scene_id in site.scene_ids if scene_id != anchor))


def resample_frozen_endmembers(
    library: list[Endmember], frozen_samples: Mapping[str, str]
) -> dict[str, Endmember]:
    """Select each anchor-frozen sample from a library resampled to one scene.

    ``load_library`` is intentionally called for every acquisition because its
    output is resampled to that acquisition's native wavelength centres.  Only
    the selected sample identities are frozen at the site's anchor.
    """
    candidates = {(endmember.mineral, endmember.sample): endmember for endmember in library}
    selected: dict[str, Endmember] = {}
    for mineral, sample in frozen_samples.items():
        key = (mineral, sample)
        if key not in candidates:
            raise ValueError(f"frozen endmember {sample!r} for {mineral} is absent from library")
        selected[mineral] = candidates[key]
    return selected


@dataclass(frozen=True)
class BinaryOverlap:
    """Agreement of two binary maps on their supplied joint domain."""

    intersection_count: int
    union_count: int
    anchor_count: int
    repeat_count: int
    iou: float
    dice: float
    prevalence_ratio: float


@dataclass(frozen=True)
class ShiftMetric:
    """All reported repeatability values for one predeclared pixel shift."""

    shift_y: int
    shift_x: int
    n_joint_finite: int
    spearman: float
    transferred_iou: float
    transferred_dice: float
    transferred_prevalence_ratio: float
    transferred_boundary_distance_m: float
    rank_relative_iou: float
    rank_relative_dice: float
    rank_relative_prevalence_ratio: float
    rank_relative_boundary_distance_m: float


@dataclass(frozen=True)
class RegistrationSensitivity:
    """Unshifted result plus the entire fixed nine-shift sensitivity set."""

    unshifted: ShiftMetric
    shift_metrics: tuple[ShiftMetric, ...]
    ranges: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RepeatabilityPaths:
    """Explicit input/output paths for a repeatability run."""

    raw_dir: Path
    speclib_dir: Path
    validation_dir: Path
    output_dir: Path
    reference_dir: Path

    @classmethod
    def repo_default(cls, root: Path) -> RepeatabilityPaths:
        return cls(
            raw_dir=root / "data" / "raw",
            speclib_dir=root / "data" / "speclib" / "ASCIIdata_splib07a",
            validation_dir=root / "data" / "intermediate" / "validation",
            output_dir=root / "data" / "processed" / "repeatability",
            reference_dir=root / "data" / "reference",
        )


@dataclass
class SceneProducts:
    """In-memory products for one scene; all maps retain scene grid metadata."""

    site_id: str
    scene_id: str
    scores: dict[str, xr.DataArray]
    qa_valid: xr.DataArray
    template: xr.DataArray
    crs: Any
    transform: Any
    feature_definitions: tuple[FeatureDef, ...]
    endmember_samples: dict[str, str]


@dataclass(frozen=True)
class TransferThreshold:
    """One externally calibrated primary-L threshold and its frozen support."""

    site_id: str
    scene_id: str
    family: str
    layer: str
    governance_status: str
    positive_bearing_blocks: int
    negative_bearing_blocks: int
    threshold: float | None
    unavailable_reason: str | None

    @property
    def key(self) -> str:
        return f"{self.family}:{self.layer}"


@dataclass(frozen=True)
class BlockHandoff:
    """Verified primary-L categorical block assignment for one anchor grid."""

    site_id: str
    anchor_scene_id: str
    raster_path: Path
    raster_sha256: str
    complete_block_ids: tuple[int, ...]
    shape: tuple[int, int]
    crs: str
    transform: Affine


@dataclass(frozen=True)
class PairedBlock:
    """Aligned block-shaped arrays with explicit missing cells retained."""

    block_id: int
    row_start: int
    column_start: int
    anchor: np.ndarray
    repeat: np.ndarray


@dataclass(frozen=True)
class BinarySufficientStatistics:
    """Integer counts that exactly determine binary overlap metrics."""

    intersection_count: int
    union_count: int
    anchor_count: int
    repeat_count: int


def binary_overlap_metrics(anchor: np.ndarray, repeat: np.ndarray) -> BinaryOverlap:
    """Return IoU, Dice, and repeat-to-anchor prevalence for two boolean maps.

    Pairwise-finite filtering is applied here, after any block resampling or
    permutation.  Missing binary cells therefore remain missing rather than
    being coerced to ``False``.  Empty-versus-empty overlap is undefined.
    """
    return _binary_overlap_from_statistics(_binary_sufficient_statistics(anchor, repeat))


def _binary_sufficient_statistics(
    anchor: np.ndarray, repeat: np.ndarray
) -> BinarySufficientStatistics:
    anchor_values = np.asarray(anchor, dtype=float)
    repeat_values = np.asarray(repeat, dtype=float)
    if anchor_values.shape != repeat_values.shape:
        raise ValueError("binary maps must share a shape")
    finite = np.isfinite(anchor_values) & np.isfinite(repeat_values)
    for name, values in (("anchor", anchor_values[finite]), ("repeat", repeat_values[finite])):
        if not np.all(np.isin(values, (0.0, 1.0))):
            raise ValueError(f"{name} binary map contains values other than 0, 1, or NaN")
    anchor_bool = anchor_values[finite].astype(bool)
    repeat_bool = repeat_values[finite].astype(bool)
    intersection = int(np.logical_and(anchor_bool, repeat_bool).sum())
    union = int(np.logical_or(anchor_bool, repeat_bool).sum())
    anchor_count = int(anchor_bool.sum())
    repeat_count = int(repeat_bool.sum())
    return BinarySufficientStatistics(
        intersection_count=intersection,
        union_count=union,
        anchor_count=anchor_count,
        repeat_count=repeat_count,
    )


def _binary_overlap_from_statistics(statistics: BinarySufficientStatistics) -> BinaryOverlap:
    intersection = statistics.intersection_count
    union = statistics.union_count
    anchor_count = statistics.anchor_count
    repeat_count = statistics.repeat_count
    return BinaryOverlap(
        intersection_count=intersection,
        union_count=union,
        anchor_count=anchor_count,
        repeat_count=repeat_count,
        iou=float(intersection / union) if union else float("nan"),
        dice=float(2 * intersection / (anchor_count + repeat_count))
        if anchor_count + repeat_count
        else float("nan"),
        prevalence_ratio=float(repeat_count / anchor_count) if anchor_count else float("nan"),
    )


def fixed_threshold_reference_metrics(
    scores: np.ndarray,
    reference: np.ndarray,
    positive_classes: frozenset[int],
    *,
    threshold: float | None,
) -> dict[str, Any]:
    """Evaluate a repeat scene against Rockwell at an unchanged anchor threshold.

    Rank AUC is tie-aware.  Coverage is unavailable when either positive or
    negative classified ground is absent; counts are still returned so a report
    cannot imply that an unavailable endpoint was estimated.
    """
    score_values = np.asarray(scores, dtype=float)
    reference_values = np.asarray(reference)
    if score_values.shape != reference_values.shape:
        raise ValueError("scores and reference must share a shape")
    domain = np.isfinite(score_values) & np.isfinite(reference_values)
    for value in ROCKWELL_EXCLUDED:
        domain &= reference_values != value
    used_scores = score_values[domain]
    used_reference = reference_values[domain]
    is_positive = np.isin(used_reference, tuple(positive_classes))
    n_pos = int(is_positive.sum())
    n_neg = int((~is_positive).sum())
    result: dict[str, Any] = {
        "available": bool(n_pos and n_neg),
        "n_usable": int(used_scores.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": float(threshold) if threshold is not None else None,
    }
    if not n_pos or not n_neg:
        result.update(
            {
                "auc": float("nan"),
                "threshold_metrics_available": False,
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
                "reason": "positive or negative Rockwell coverage is absent",
            }
        )
        return result
    ranks = rankdata(used_scores, method="average")
    auc = (ranks[is_positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    if threshold is None or not math.isfinite(threshold):
        result.update(
            {
                "auc": float(auc),
                "threshold_metrics_available": False,
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
                "reason": "transferred_threshold_unavailable",
            }
        )
        return result
    predicted = used_scores >= threshold
    true_positive = int(np.logical_and(predicted, is_positive).sum())
    false_positive = int(np.logical_and(predicted, ~is_positive).sum())
    true_negative = int(np.logical_and(~predicted, ~is_positive).sum())
    false_negative = int(np.logical_and(~predicted, is_positive).sum())
    tpr = true_positive / n_pos
    tnr = true_negative / n_neg
    positive_f1_denominator = 2 * true_positive + false_positive + false_negative
    negative_f1_denominator = 2 * true_negative + false_positive + false_negative
    positive_f1 = 2 * true_positive / positive_f1_denominator if positive_f1_denominator else 0.0
    negative_f1 = 2 * true_negative / negative_f1_denominator if negative_f1_denominator else 0.0
    result.update(
        {
            "auc": float(auc),
            "threshold_metrics_available": True,
            "balanced_accuracy": float((tpr + tnr) / 2),
            "macro_f1": float((positive_f1 + negative_f1) / 2),
            "tpr": float(tpr),
            "tnr": float(tnr),
            "positive_f1": float(positive_f1),
            "negative_f1": float(negative_f1),
        }
    )
    return result


def symmetric_boundary_distance_m(
    anchor: np.ndarray,
    repeat: np.ndarray,
    *,
    xres_m: float,
    yres_m: float,
) -> float:
    """Symmetric 95th-percentile boundary displacement in metres.

    The reported value is the larger directed 95th percentile, so neither map
    is privileged.  Empty detections have no boundary and return ``NaN``.
    """
    anchor_values = np.asarray(anchor, dtype=float)
    repeat_values = np.asarray(repeat, dtype=float)
    if anchor_values.shape != repeat_values.shape:
        raise ValueError("binary maps must share a shape")
    if xres_m <= 0 or yres_m <= 0:
        raise ValueError("pixel resolutions must be positive metres")
    anchor_boundary, repeat_boundary = _paired_boundary_coordinates(
        anchor_values,
        repeat_values,
        row_start=0,
        column_start=0,
        xres_m=xres_m,
        yres_m=yres_m,
    )
    return _boundary_distance_from_coordinates(anchor_boundary, repeat_boundary)


def _paired_boundary_coordinates(
    anchor: np.ndarray,
    repeat: np.ndarray,
    *,
    row_start: int,
    column_start: int,
    xres_m: float,
    yres_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract positive-side boundaries using only joint-finite neighbours."""
    anchor_values = np.asarray(anchor, dtype=float)
    repeat_values = np.asarray(repeat, dtype=float)
    if anchor_values.shape != repeat_values.shape:
        raise ValueError("binary maps must share a shape")
    if anchor_values.ndim != 2:
        raise ValueError("boundary maps must be two-dimensional")
    if xres_m <= 0 or yres_m <= 0:
        raise ValueError("pixel resolutions must be positive metres")
    finite = np.isfinite(anchor_values) & np.isfinite(repeat_values)
    for name, values in (("anchor", anchor_values[finite]), ("repeat", repeat_values[finite])):
        if not np.all(np.isin(values, (0.0, 1.0))):
            raise ValueError(f"{name} binary map contains values other than 0, 1, or NaN")
    return (
        _within_block_boundary_coordinates(
            anchor_values,
            finite,
            row_start=row_start,
            column_start=column_start,
            xres_m=xres_m,
            yres_m=yres_m,
        ),
        _within_block_boundary_coordinates(
            repeat_values,
            finite,
            row_start=row_start,
            column_start=column_start,
            xres_m=xres_m,
            yres_m=yres_m,
        ),
    )


def _within_block_boundary_coordinates(
    values: np.ndarray,
    finite: np.ndarray,
    *,
    row_start: int,
    column_start: int,
    xres_m: float,
    yres_m: float,
) -> np.ndarray:
    """Return boundary-cell centres without treating array edges as background."""
    boundary = np.zeros(values.shape, dtype=bool)
    height, width = values.shape
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            source_rows = slice(max(0, -row_offset), min(height, height - row_offset))
            source_columns = slice(max(0, -column_offset), min(width, width - column_offset))
            neighbour_rows = slice(max(0, row_offset), min(height, height + row_offset))
            neighbour_columns = slice(max(0, column_offset), min(width, width + column_offset))
            comparable = (
                finite[source_rows, source_columns] & finite[neighbour_rows, neighbour_columns]
            )
            boundary[source_rows, source_columns] |= (
                comparable
                & (values[source_rows, source_columns] == 1.0)
                & (values[neighbour_rows, neighbour_columns] == 0.0)
            )
    rows, columns = np.nonzero(boundary)
    if rows.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack(
        (
            (row_start + rows.astype(float) + 0.5) * yres_m,
            (column_start + columns.astype(float) + 0.5) * xres_m,
        )
    )


def _boundary_distance_from_coordinates(
    anchor_coordinates: np.ndarray, repeat_coordinates: np.ndarray
) -> float:
    if anchor_coordinates.size == 0 or repeat_coordinates.size == 0:
        return float("nan")
    anchor_to_repeat = cKDTree(repeat_coordinates).query(anchor_coordinates, k=1)[0]
    repeat_to_anchor = cKDTree(anchor_coordinates).query(repeat_coordinates, k=1)[0]
    return float(
        max(
            np.percentile(anchor_to_repeat, 95),
            np.percentile(repeat_to_anchor, 95),
        )
    )


def _safe_spearman(anchor_scores: np.ndarray, repeat_scores: np.ndarray) -> float:
    anchor_values = np.asarray(anchor_scores, dtype=float)
    repeat_values = np.asarray(repeat_scores, dtype=float)
    if anchor_values.shape != repeat_values.shape:
        raise ValueError("Spearman inputs must share a shape")
    finite = np.isfinite(anchor_values) & np.isfinite(repeat_values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    anchor_finite = anchor_values[finite]
    repeat_finite = repeat_values[finite]
    if np.ptp(anchor_finite) == 0 or np.ptp(repeat_finite) == 0:
        return float("nan")
    result = spearmanr(anchor_finite, repeat_finite)
    return float(result.statistic) if np.isfinite(result.statistic) else float("nan")


def _shift(array: np.ndarray, shift_y: int, shift_x: int, fill: float | bool) -> np.ndarray:
    """Translate an array without wraparound; positive shifts move it down/right."""
    source = np.asarray(array)
    out = np.full(source.shape, fill, dtype=source.dtype)
    ny, nx = source.shape
    src_y0, src_y1 = max(0, -shift_y), min(ny, ny - shift_y)
    src_x0, src_x1 = max(0, -shift_x), min(nx, nx - shift_x)
    dst_y0, dst_y1 = max(0, shift_y), min(ny, ny + shift_y)
    dst_x0, dst_x1 = max(0, shift_x), min(nx, nx + shift_x)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    return out


def _shift_metric(
    anchor_scores: np.ndarray,
    repeat_scores: np.ndarray,
    anchor_transferred: np.ndarray,
    repeat_transferred: np.ndarray,
    anchor_rank_relative: np.ndarray,
    repeat_rank_relative: np.ndarray,
    anchor_valid: np.ndarray,
    repeat_valid: np.ndarray,
    *,
    shift_y: int,
    shift_x: int,
    xres_m: float,
    yres_m: float,
) -> ShiftMetric:
    shifted_scores = _shift(repeat_scores, shift_y, shift_x, np.nan)
    shifted_transferred = _shift(repeat_transferred, shift_y, shift_x, np.nan)
    shifted_rank = _shift(repeat_rank_relative, shift_y, shift_x, np.nan)
    shifted_valid = _shift(repeat_valid, shift_y, shift_x, False)
    anchor_valid_values = np.asarray(anchor_valid, dtype=bool)
    repeat_valid_values = np.asarray(shifted_valid, dtype=bool)
    anchor_score_values = np.where(
        anchor_valid_values, np.asarray(anchor_scores, dtype=float), np.nan
    )
    repeat_score_values = np.where(
        repeat_valid_values, np.asarray(shifted_scores, dtype=float), np.nan
    )
    anchor_t = np.where(anchor_valid_values, np.asarray(anchor_transferred, dtype=float), np.nan)
    repeat_t = np.where(repeat_valid_values, np.asarray(shifted_transferred, dtype=float), np.nan)
    anchor_r = np.where(anchor_valid_values, np.asarray(anchor_rank_relative, dtype=float), np.nan)
    repeat_r = np.where(repeat_valid_values, np.asarray(shifted_rank, dtype=float), np.nan)
    transferred = binary_overlap_metrics(anchor_t, repeat_t)
    rank_relative = binary_overlap_metrics(anchor_r, repeat_r)
    joint_score_finite = np.isfinite(anchor_score_values) & np.isfinite(repeat_score_values)
    return ShiftMetric(
        shift_y=shift_y,
        shift_x=shift_x,
        n_joint_finite=int(joint_score_finite.sum()),
        spearman=_safe_spearman(anchor_score_values, repeat_score_values),
        transferred_iou=transferred.iou,
        transferred_dice=transferred.dice,
        transferred_prevalence_ratio=transferred.prevalence_ratio,
        transferred_boundary_distance_m=symmetric_boundary_distance_m(
            anchor_t,
            repeat_t,
            xres_m=xres_m,
            yres_m=yres_m,
        ),
        rank_relative_iou=rank_relative.iou,
        rank_relative_dice=rank_relative.dice,
        rank_relative_prevalence_ratio=rank_relative.prevalence_ratio,
        rank_relative_boundary_distance_m=symmetric_boundary_distance_m(
            anchor_r,
            repeat_r,
            xres_m=xres_m,
            yres_m=yres_m,
        ),
    )


def registration_sensitivity(
    anchor_scores: np.ndarray,
    repeat_scores: np.ndarray,
    anchor_transferred: np.ndarray,
    repeat_transferred: np.ndarray,
    anchor_rank_relative: np.ndarray,
    repeat_rank_relative: np.ndarray,
    anchor_valid: np.ndarray,
    repeat_valid: np.ndarray,
    *,
    xres_m: float,
    yres_m: float,
) -> RegistrationSensitivity:
    """Evaluate the unshifted grid and every predeclared ±1-pixel neighbour.

    No value is selected from this set.  ``ranges`` is an audit summary of all
    nine values, including the unshifted comparison.
    """
    shifts = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    rows = tuple(
        _shift_metric(
            anchor_scores,
            repeat_scores,
            anchor_transferred,
            repeat_transferred,
            anchor_rank_relative,
            repeat_rank_relative,
            anchor_valid,
            repeat_valid,
            shift_y=dy,
            shift_x=dx,
            xres_m=xres_m,
            yres_m=yres_m,
        )
        for dy, dx in shifts
    )
    metric_names = (
        "spearman",
        "transferred_iou",
        "transferred_dice",
        "transferred_prevalence_ratio",
        "transferred_boundary_distance_m",
        "rank_relative_iou",
        "rank_relative_dice",
        "rank_relative_prevalence_ratio",
        "rank_relative_boundary_distance_m",
    )
    ranges: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([getattr(row, name) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        ranges[name] = {
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
        }
    unshifted = next(row for row in rows if row.shift_y == 0 and row.shift_x == 0)
    return RegistrationSensitivity(unshifted=unshifted, shift_metrics=rows, ranges=ranges)


def paired_block_bootstrap(
    anchor: np.ndarray,
    repeat: np.ndarray,
    block_ids: np.ndarray,
    *,
    n_reps: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Resample paired complete blocks while retaining shape and missingness."""
    if n_reps <= 0:
        raise ValueError("n_reps must be positive")
    _validate_workers(workers)
    blocks = _extract_paired_blocks(anchor, repeat, block_ids)
    metric_fn = metric or _safe_spearman
    choices = _bootstrap_choices(len(blocks), n_reps=n_reps, seed=seed)

    def evaluate(start: int, stop: int) -> np.ndarray:
        chunk = np.empty(stop - start, dtype=float)
        for offset, chosen in enumerate(choices[start:stop]):
            left, right = _concatenate_blocks([blocks[int(item)] for item in chosen])
            chunk[offset] = metric_fn(left, right)
        return chunk

    samples = _evaluate_ordered_chunks(n_reps, workers=workers, evaluate=evaluate)
    return _resampling_result(samples, n_blocks=len(blocks))


def paired_block_null(
    anchor: np.ndarray,
    repeat: np.ndarray,
    block_ids: np.ndarray,
    *,
    n_reps: int = NULL_REPLICATES,
    seed: int = SEED,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Permute repeat-block identities without moving within-block cells."""
    if n_reps <= 0:
        raise ValueError("n_reps must be positive")
    _validate_workers(workers)
    blocks = _extract_paired_blocks(anchor, repeat, block_ids)
    shapes = {block.anchor.shape for block in blocks} | {block.repeat.shape for block in blocks}
    if len(shapes) != 1:
        raise ValueError("complete block arrays must have one common rectangular shape")
    orderings = _permutation_orderings(len(blocks), n_reps=n_reps, seed=seed)
    metric_fn = metric or _safe_spearman
    anchor_joined = np.concatenate([block.anchor.ravel() for block in blocks])

    def evaluate(start: int, stop: int) -> np.ndarray:
        chunk = np.empty(stop - start, dtype=float)
        for offset, ordering in enumerate(orderings[start:stop]):
            repeat_joined = np.concatenate([blocks[item].repeat.ravel() for item in ordering])
            chunk[offset] = metric_fn(anchor_joined, repeat_joined)
        return chunk

    samples = _evaluate_ordered_chunks(len(orderings), workers=workers, evaluate=evaluate)
    return _resampling_result(
        samples,
        n_blocks=len(blocks),
        enumerated_all_unique=math.factorial(len(blocks)) <= NULL_REPLICATES,
    )


def _validate_workers(workers: int) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")


def _bootstrap_choices(n_blocks: int, *, n_reps: int, seed: int) -> np.ndarray:
    """Materialize the reference RNG calls in their original replicate order."""
    rng = np.random.default_rng(seed)
    choices = np.empty((n_reps, n_blocks), dtype=np.intp)
    for index in range(n_reps):
        choices[index] = rng.integers(0, n_blocks, size=n_blocks)
    return choices


def _chunk_bounds(n_items: int, workers: int) -> tuple[tuple[int, int], ...]:
    n_chunks = min(n_items, workers)
    width, remainder = divmod(n_items, n_chunks)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(n_chunks):
        stop = start + width + int(index < remainder)
        bounds.append((start, stop))
        start = stop
    return tuple(bounds)


def _evaluate_ordered_chunks(
    n_items: int,
    *,
    workers: int,
    evaluate: Callable[[int, int], np.ndarray],
) -> np.ndarray:
    """Evaluate independent slices concurrently and restore exact input order."""
    bounds = _chunk_bounds(n_items, workers)
    if len(bounds) == 1:
        return evaluate(*bounds[0])
    samples = np.empty(n_items, dtype=float)
    with ThreadPoolExecutor(max_workers=len(bounds), thread_name_prefix="repeatability") as pool:
        futures = [pool.submit(evaluate, start, stop) for start, stop in bounds]
        for (start, stop), future in zip(bounds, futures, strict=True):
            samples[start:stop] = future.result()
            logger.info("completed resampling replicates %d:%d of %d", start, stop, n_items)
    return samples


def _resampling_result(
    samples: np.ndarray,
    *,
    n_blocks: int,
    enumerated_all_unique: bool | None = None,
) -> dict[str, Any]:
    finite_count = int(np.count_nonzero(np.isfinite(samples)))
    scheduled = int(samples.size)
    result: dict[str, Any] = {
        "n_blocks": n_blocks,
        "scheduled_replicates": scheduled,
        "finite_replicates": finite_count,
        "finite_fraction": finite_count / scheduled,
        "gate_eligible": finite_count >= math.ceil(FINITE_REPLICATE_FRACTION * scheduled),
        "samples": samples,
        # Backward-compatible key for callers written against the first draft.
        "spearman": samples,
    }
    if enumerated_all_unique is not None:
        result["enumerated_all_unique"] = enumerated_all_unique
    return result


def _extract_paired_blocks(
    anchor: np.ndarray, repeat: np.ndarray, block_ids: np.ndarray
) -> tuple[PairedBlock, ...]:
    """Extract full rectangular blocks without filtering unavailable cells."""
    anchor_values = np.asarray(anchor, dtype=float)
    repeat_values = np.asarray(repeat, dtype=float)
    ids = np.asarray(block_ids)
    if anchor_values.shape != repeat_values.shape or anchor_values.shape != ids.shape:
        raise ValueError("anchor, repeat, and block_ids must have matching shapes")
    if ids.ndim != 2:
        raise ValueError("block_ids must be a two-dimensional categorical raster")
    labels = sorted(int(value) for value in np.unique(ids[np.isfinite(ids)]) if int(value) > 0)
    if not labels:
        raise ValueError("at least one positive complete block ID is required")
    blocks: list[PairedBlock] = []
    for label in labels:
        rows, cols = np.nonzero(ids == label)
        row_slice = slice(int(rows.min()), int(rows.max()) + 1)
        col_slice = slice(int(cols.min()), int(cols.max()) + 1)
        block_id_values = ids[row_slice, col_slice]
        if not np.all(block_id_values == label):
            raise ValueError(f"block ID {label} is not one complete rectangular footprint")
        blocks.append(
            PairedBlock(
                block_id=label,
                row_start=int(rows.min()),
                column_start=int(cols.min()),
                anchor=anchor_values[row_slice, col_slice].copy(),
                repeat=repeat_values[row_slice, col_slice].copy(),
            )
        )
    return tuple(blocks)


def _concatenate_blocks(blocks: list[PairedBlock]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([block.anchor.ravel() for block in blocks]),
        np.concatenate([block.repeat.ravel() for block in blocks]),
    )


def _permutation_orderings(n_blocks: int, *, n_reps: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """Return every unique ordering when feasible, else seeded unique draws."""
    total = math.factorial(n_blocks)
    if total <= NULL_REPLICATES:
        return tuple(permutations(range(n_blocks)))
    rng = np.random.default_rng(seed)
    draws: set[tuple[int, ...]] = set()
    while len(draws) < n_reps:
        draws.add(tuple(int(value) for value in rng.permutation(n_blocks)))
    return tuple(sorted(draws))


def _statistics_array(statistics: BinarySufficientStatistics) -> np.ndarray:
    return np.asarray(
        [
            statistics.intersection_count,
            statistics.union_count,
            statistics.anchor_count,
            statistics.repeat_count,
        ],
        dtype=np.int64,
    )


def _binary_samples_from_totals(totals: np.ndarray, *, metric: str) -> np.ndarray:
    intersection = totals[:, 0].astype(float)
    if metric == "iou":
        denominator = totals[:, 1]
        numerator = intersection
    elif metric == "dice":
        denominator = totals[:, 2] + totals[:, 3]
        numerator = 2.0 * intersection
    elif metric == "prevalence_ratio":
        denominator = totals[:, 2]
        numerator = totals[:, 3].astype(float)
    else:
        raise ValueError(f"unknown binary metric {metric!r}")
    samples = np.full(totals.shape[0], np.nan, dtype=float)
    available = denominator != 0
    samples[available] = numerator[available] / denominator[available]
    return samples


def _statistics_batch_rows(n_blocks: int) -> int:
    bytes_per_row = max(1, n_blocks) * 4 * np.dtype(np.int64).itemsize
    return max(1, STATISTICS_BATCH_BYTES // bytes_per_row)


def _sum_bootstrap_statistics(statistics: np.ndarray, choices: np.ndarray) -> np.ndarray:
    totals = np.empty((choices.shape[0], 4), dtype=np.int64)
    batch_rows = _statistics_batch_rows(choices.shape[1])
    for start in range(0, choices.shape[0], batch_rows):
        stop = min(start + batch_rows, choices.shape[0])
        totals[start:stop] = statistics[choices[start:stop]].sum(axis=1, dtype=np.int64)
    return totals


def _sum_null_statistics(statistics: np.ndarray, orderings: np.ndarray) -> np.ndarray:
    totals = np.empty((orderings.shape[0], 4), dtype=np.int64)
    block_rows = np.arange(orderings.shape[1])
    batch_rows = _statistics_batch_rows(orderings.shape[1])
    for start in range(0, orderings.shape[0], batch_rows):
        stop = min(start + batch_rows, orderings.shape[0])
        totals[start:stop] = statistics[block_rows, orderings[start:stop]].sum(
            axis=1, dtype=np.int64
        )
    return totals


def _binary_block_resampling(
    anchor: np.ndarray,
    repeat: np.ndarray,
    block_ids: np.ndarray,
    *,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    n_null: int = NULL_REPLICATES,
    seed: int = SEED,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute exact binary-metric draws from block-level contingencies."""
    if n_bootstrap <= 0 or n_null <= 0:
        raise ValueError("resampling replicate counts must be positive")
    blocks = _extract_paired_blocks(anchor, repeat, block_ids)
    shapes = {block.anchor.shape for block in blocks} | {block.repeat.shape for block in blocks}
    if len(shapes) != 1:
        raise ValueError("complete block arrays must have one common rectangular shape")

    paired_statistics = np.stack(
        [
            _statistics_array(_binary_sufficient_statistics(block.anchor, block.repeat))
            for block in blocks
        ]
    )
    choices = _bootstrap_choices(len(blocks), n_reps=n_bootstrap, seed=seed)
    bootstrap_totals = _sum_bootstrap_statistics(paired_statistics, choices)

    orderings_tuple = _permutation_orderings(len(blocks), n_reps=n_null, seed=seed)
    orderings = np.asarray(orderings_tuple, dtype=np.intp)
    cross_statistics = np.empty((len(blocks), len(blocks), 4), dtype=np.int64)
    for anchor_index, anchor_block in enumerate(blocks):
        for repeat_index, repeat_block in enumerate(blocks):
            cross_statistics[anchor_index, repeat_index] = _statistics_array(
                _binary_sufficient_statistics(anchor_block.anchor, repeat_block.repeat)
            )
    null_totals = _sum_null_statistics(cross_statistics, orderings)
    enumerated = math.factorial(len(blocks)) <= NULL_REPLICATES

    output: dict[str, dict[str, dict[str, Any]]] = {}
    for metric in ("iou", "dice", "prevalence_ratio"):
        output[metric] = {
            "bootstrap": _resampling_result(
                _binary_samples_from_totals(bootstrap_totals, metric=metric),
                n_blocks=len(blocks),
            ),
        }
        if metric == "prevalence_ratio":
            output[metric]["spatial_null"] = {
                "status": "not_applicable",
                "reason": "whole_block_null_is_not_defined_for_detection_prevalence_ratio",
            }
        else:
            output[metric]["spatial_null"] = _resampling_result(
                _binary_samples_from_totals(null_totals, metric=metric),
                n_blocks=len(blocks),
                enumerated_all_unique=enumerated,
            )
    return output


def _concatenate_boundary_coordinates(coordinates: list[np.ndarray]) -> np.ndarray:
    nonempty = [values for values in coordinates if values.size]
    return np.concatenate(nonempty, axis=0) if nonempty else np.empty((0, 2), dtype=float)


def _boundary_block_resampling(
    anchor: np.ndarray,
    repeat: np.ndarray,
    block_ids: np.ndarray,
    *,
    xres_m: float,
    yres_m: float,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    n_null: int = NULL_REPLICATES,
    seed: int = SEED,
    workers: int = 1,
) -> dict[str, dict[str, Any]]:
    """Resample boundary distance without creating seams between blocks."""
    if n_bootstrap <= 0 or n_null <= 0:
        raise ValueError("resampling replicate counts must be positive")
    _validate_workers(workers)
    blocks = _extract_paired_blocks(anchor, repeat, block_ids)
    shapes = {block.anchor.shape for block in blocks} | {block.repeat.shape for block in blocks}
    if len(shapes) != 1:
        raise ValueError("complete block arrays must have one common rectangular shape")

    paired_coordinates = [
        _paired_boundary_coordinates(
            block.anchor,
            block.repeat,
            row_start=block.row_start,
            column_start=block.column_start,
            xres_m=xres_m,
            yres_m=yres_m,
        )
        for block in blocks
    ]
    choices = _bootstrap_choices(len(blocks), n_reps=n_bootstrap, seed=seed)

    def evaluate_bootstrap(start: int, stop: int) -> np.ndarray:
        chunk = np.empty(stop - start, dtype=float)
        for offset, chosen in enumerate(choices[start:stop]):
            selected = [paired_coordinates[int(index)] for index in chosen]
            anchor_coordinates = _concatenate_boundary_coordinates(
                [coordinates[0] for coordinates in selected]
            )
            repeat_coordinates = _concatenate_boundary_coordinates(
                [coordinates[1] for coordinates in selected]
            )
            chunk[offset] = _boundary_distance_from_coordinates(
                anchor_coordinates, repeat_coordinates
            )
        return chunk

    bootstrap_samples = _evaluate_ordered_chunks(
        n_bootstrap, workers=workers, evaluate=evaluate_bootstrap
    )

    orderings_tuple = _permutation_orderings(len(blocks), n_reps=n_null, seed=seed)
    cross_coordinates = [
        [
            _paired_boundary_coordinates(
                anchor_block.anchor,
                repeat_block.repeat,
                row_start=anchor_block.row_start,
                column_start=anchor_block.column_start,
                xres_m=xres_m,
                yres_m=yres_m,
            )
            for repeat_block in blocks
        ]
        for anchor_block in blocks
    ]

    def evaluate_null(start: int, stop: int) -> np.ndarray:
        chunk = np.empty(stop - start, dtype=float)
        for offset, ordering in enumerate(orderings_tuple[start:stop]):
            selected = [
                cross_coordinates[destination][source]
                for destination, source in enumerate(ordering)
            ]
            anchor_coordinates = _concatenate_boundary_coordinates(
                [coordinates[0] for coordinates in selected]
            )
            repeat_coordinates = _concatenate_boundary_coordinates(
                [coordinates[1] for coordinates in selected]
            )
            chunk[offset] = _boundary_distance_from_coordinates(
                anchor_coordinates, repeat_coordinates
            )
        return chunk

    null_samples = _evaluate_ordered_chunks(
        len(orderings_tuple), workers=workers, evaluate=evaluate_null
    )
    return {
        "bootstrap": _resampling_result(bootstrap_samples, n_blocks=len(blocks)),
        "spatial_null": _resampling_result(
            null_samples,
            n_blocks=len(blocks),
            enumerated_all_unique=math.factorial(len(blocks)) <= NULL_REPLICATES,
        ),
    }


def _source_mask(scores: xr.DataArray, qa_valid: xr.DataArray, threshold: float) -> xr.DataArray:
    """Return a binary float mask with unavailable cells preserved as NaN."""
    if not math.isfinite(threshold):
        return xr.full_like(scores, np.nan, dtype=float)
    valid = qa_valid.astype(bool) & np.isfinite(scores)
    return xr.where(valid, (scores >= threshold).astype(float), np.nan)


def _upper_decile_threshold(scores: xr.DataArray, qa_valid: xr.DataArray) -> float:
    values = np.asarray(scores.values)[np.asarray(qa_valid.values, dtype=bool)]
    values = values[np.isfinite(values)]
    return float(np.quantile(values, UPPER_DECILE_QUANTILE)) if values.size else float("nan")


def _require_projected_resolution(template: xr.DataArray) -> tuple[float, float]:
    crs = template.rio.crs
    if crs is None or not crs.is_projected:
        raise ValueError("repeatability boundary distance requires a projected anchor CRS")
    transform = template.rio.transform()
    xres, yres = abs(float(transform.a)), abs(float(transform.e))
    if xres <= 0 or yres <= 0:
        raise ValueError("anchor grid has invalid pixel resolution")
    try:
        _, metre_factor = crs.linear_units_factor
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("anchor CRS has no usable linear-unit conversion") from error
    if metre_factor is None or metre_factor <= 0:
        raise ValueError("anchor CRS has no positive linear-unit conversion")
    return xres * float(metre_factor), yres * float(metre_factor)


def _attach_grid_metadata(data: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    """Attach a scene template's CRS and affine transform to a derived field."""
    if template.rio.crs is None:
        raise ValueError("grid template has no CRS")
    return data.rio.write_crs(template.rio.crs).rio.write_transform(template.rio.transform())


def _reproject_continuous(score: xr.DataArray, anchor_template: xr.DataArray) -> xr.DataArray:
    return score.rio.reproject_match(anchor_template, resampling=Resampling.bilinear, nodata=np.nan)


def _reproject_mask(mask: xr.DataArray, anchor_template: xr.DataArray) -> xr.DataArray:
    reprojected = mask.rio.write_nodata(255).rio.reproject_match(
        anchor_template, resampling=Resampling.nearest, nodata=255
    )
    return reprojected == 1


def _reproject_binary(mask: xr.DataArray, anchor_template: xr.DataArray) -> xr.DataArray:
    """Nearest-neighbour a binary float field while retaining nodata as NaN."""
    return mask.rio.write_nodata(np.nan).rio.reproject_match(
        anchor_template, resampling=Resampling.nearest, nodata=np.nan
    )


def _write_raster(
    data: xr.DataArray,
    template: xr.DataArray,
    path: Path,
    *,
    nodata: float | int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.stem}.partial{path.suffix}")
    if path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to overwrite repeatability cache file: {path}")
    data.rio.write_crs(template.rio.crs).rio.write_transform(
        template.rio.transform()
    ).rio.write_nodata(nodata).rio.to_raster(partial_path, driver="GTiff", compress="LZW")
    partial_path.replace(path)


def _scene_output_dir(output_dir: Path, site_id: str, scene_id: str) -> Path:
    return output_dir / "scenes" / site_id / scene_id


def _load_scene_products(
    site_id: str,
    scene_id: str,
    paths: RepeatabilityPaths,
    *,
    frozen_features: tuple[FeatureDef, ...] | None,
    frozen_endmember_samples: Mapping[str, str] | None,
    source_sha256: str,
    implementation_sha256: str,
    preregistration_sha256: str,
    speclib_sha256: str,
) -> tuple[SceneProducts, tuple[FeatureDef, ...], dict[str, str]]:
    """Load, quality-mask, map, and write one scene using frozen definitions."""
    path = paths.raw_dir / f"{scene_id}_{TANAGER_SR_ASSET}.h5"
    scene_dir = _scene_output_dir(paths.output_dir, site_id, scene_id)
    cache_manifest_path = scene_dir / "cache_manifest.json"
    cache_inputs = {
        "source_path": str(path.resolve()),
        "source_sha256": source_sha256,
        "implementation_sha256": implementation_sha256,
        "preregistration_sha256": preregistration_sha256,
        "speclib_sha256": speclib_sha256,
        "parameters": {
            "mtmf_ridge": MTMF_RIDGE,
            "max_infeasibility": MAX_INFEASIBILITY,
        },
    }
    if cache_manifest_path.is_file():
        return _load_cached_scene_products(
            site_id,
            scene_id,
            scene_dir,
            cache_inputs=cache_inputs,
            frozen_features=frozen_features,
            frozen_endmember_samples=frozen_endmember_samples,
        )
    if scene_dir.exists() and any(scene_dir.iterdir()):
        raise FileExistsError(
            f"unvalidated partial scene cache exists at {scene_dir}; use a fresh output directory"
        )
    cube, wavelengths = load_tanager_sr_hdf5(path)
    masked, _ = mask_tanager_scene(cube, wavelengths, path)
    template = masked.isel(band=0, drop=True)
    if template.rio.crs is None:
        raise ValueError(f"{scene_id} has no spatial CRS")
    qa_valid = _attach_grid_metadata(
        xr.DataArray(
            np.isfinite(masked.values).any(axis=0),
            dims=("y", "x"),
            coords={"y": masked.y, "x": masked.x},
        ),
        template,
    )
    library = load_library(paths.speclib_dir, wavelengths)
    if frozen_features is None or frozen_endmember_samples is None:
        features = tuple(build_feature_defs(wavelengths, paths.speclib_dir))
        endmembers = select_endmembers(library)
        frozen_samples = {name: endmember.sample for name, endmember in endmembers.items()}
    else:
        features = frozen_features
        frozen_samples = dict(frozen_endmember_samples)
        endmembers = resample_frozen_endmembers(library, frozen_samples)
    feature_scores = diagnostic_feature_maps(masked, wavelengths, list(features))
    mtmf_scores = mtmf(masked, endmembers, ridge=MTMF_RIDGE)
    scores: dict[str, xr.DataArray] = {}
    for name, score in feature_scores.data_vars.items():
        scores[f"feature:{name}"] = score
    for mineral in endmembers:
        abundance = mtmf_scores[f"{mineral}_mf"]
        infeasibility = mtmf_scores[f"{mineral}_infeas"]
        scores[f"mtmf:{mineral}"] = abundance.where(infeasibility < MAX_INFEASIBILITY)
    scores = {key: _attach_grid_metadata(score, template) for key, score in scores.items()}

    for key, score in scores.items():
        kind, layer = key.split(":", maxsplit=1)
        _write_raster(score, template, scene_dir / f"{kind}_{layer}.tif", nodata=np.nan)
    _write_raster(qa_valid.astype("uint8"), template, scene_dir / "qa_valid_mask.tif", nodata=255)
    product = SceneProducts(
        site_id=site_id,
        scene_id=scene_id,
        scores=scores,
        qa_valid=qa_valid,
        template=template,
        crs=template.rio.crs,
        transform=template.rio.transform(),
        feature_definitions=features,
        endmember_samples=frozen_samples,
    )
    score_files = {
        key: {
            "path": f"{key.replace(':', '_')}.tif",
            "sha256": _sha256(scene_dir / f"{key.replace(':', '_')}.tif"),
        }
        for key in sorted(scores)
    }
    qa_path = scene_dir / "qa_valid_mask.tif"
    _atomic_write_json(
        cache_manifest_path,
        {
            "schema_version": "1.0",
            "site_id": site_id,
            "scene_id": scene_id,
            "cache_inputs": cache_inputs,
            "grid": _grid_record(template),
            "feature_definitions": [asdict(feature) for feature in features],
            "endmember_samples": dict(sorted(frozen_samples.items())),
            "score_files": score_files,
            "qa_valid_file": {"path": qa_path.name, "sha256": _sha256(qa_path)},
        },
    )
    return product, features, frozen_samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    """Hash relative names and bytes for every regular file in a directory."""
    if not path.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"source directory contains no files: {path}")
    for candidate in files:
        digest.update(candidate.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _grid_record(template: xr.DataArray) -> dict[str, Any]:
    if template.rio.crs is None:
        raise ValueError("cached raster has no CRS")
    return {
        "shape": [int(value) for value in template.shape],
        "crs": template.rio.crs.to_string(),
        "transform": list(template.rio.transform())[:6],
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.partial")
    if partial_path.exists():
        raise FileExistsError(f"stale partial JSON exists: {partial_path}")
    text = json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with partial_path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    partial_path.replace(path)


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_cache_path(scene_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or Path(value).name != value:
        raise ValueError(f"scene cache contains an invalid file name: {value!r}")
    return scene_dir / value


def _validate_cached_raster(path: Path, record: Mapping[str, Any], grid: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"cached raster does not exist: {path}")
    _require_equal(f"{path.name} SHA", _sha256(path), record.get("sha256"))
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"cached raster has no CRS: {path}")
        observed_grid = {
            "shape": [int(value) for value in dataset.shape],
            "crs": dataset.crs.to_string(),
            "transform": list(dataset.transform)[:6],
        }
    _require_equal(f"{path.name} grid", observed_grid, dict(grid))


def _load_cached_scene_products(
    site_id: str,
    scene_id: str,
    scene_dir: Path,
    *,
    cache_inputs: Mapping[str, Any],
    frozen_features: tuple[FeatureDef, ...] | None,
    frozen_endmember_samples: Mapping[str, str] | None,
) -> tuple[SceneProducts, tuple[FeatureDef, ...], dict[str, str]]:
    manifest = _read_strict_json_object(
        scene_dir / "cache_manifest.json", label=f"{site_id} {scene_id} scene cache"
    )
    _require_equal("scene cache schema", manifest.get("schema_version"), "1.0")
    _require_equal("scene cache site", manifest.get("site_id"), site_id)
    _require_equal("scene cache scene", manifest.get("scene_id"), scene_id)
    _require_equal("scene cache inputs", manifest.get("cache_inputs"), dict(cache_inputs))
    grid = manifest.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("scene cache has no grid record")
    feature_records = manifest.get("feature_definitions")
    if not isinstance(feature_records, list):
        raise ValueError("scene cache has no feature definitions")
    features = tuple(FeatureDef(**record) for record in feature_records)
    samples_value = manifest.get("endmember_samples")
    if not isinstance(samples_value, dict):
        raise ValueError("scene cache has no endmember sample record")
    samples = {str(key): str(value) for key, value in samples_value.items()}
    if frozen_features is not None:
        _require_equal("scene cache frozen feature definitions", features, frozen_features)
    if frozen_endmember_samples is not None:
        _require_equal(
            "scene cache frozen endmember samples", samples, dict(frozen_endmember_samples)
        )

    score_records = manifest.get("score_files")
    if not isinstance(score_records, dict) or not score_records:
        raise ValueError("scene cache has no score files")
    scores: dict[str, xr.DataArray] = {}
    for key in sorted(score_records):
        record = score_records[key]
        if not isinstance(record, dict):
            raise ValueError(f"scene cache score record is invalid for {key}")
        score_path = _validated_cache_path(scene_dir, record.get("path"))
        _validate_cached_raster(score_path, record, grid)
        with rioxarray.open_rasterio(score_path, masked=True) as opened:
            scores[key] = opened.squeeze("band", drop=True).load()
    template = next(iter(scores.values()))

    qa_record = manifest.get("qa_valid_file")
    if not isinstance(qa_record, dict):
        raise ValueError("scene cache has no QA-valid file")
    qa_path = _validated_cache_path(scene_dir, qa_record.get("path"))
    _validate_cached_raster(qa_path, qa_record, grid)
    with rioxarray.open_rasterio(qa_path, masked=False) as opened:
        qa_raw = opened.squeeze("band", drop=True).load()
    qa_valid = _attach_grid_metadata(qa_raw == 1, template)
    product = SceneProducts(
        site_id=site_id,
        scene_id=scene_id,
        scores=scores,
        qa_valid=qa_valid,
        template=template,
        crs=template.rio.crs,
        transform=template.rio.transform(),
        feature_definitions=features,
        endmember_samples=samples,
    )
    logger.info("reused validated scene cache %s", scene_dir)
    return product, features, samples


def _json_safe(value: Any) -> Any:
    """Replace non-finite numeric values with strict-JSON nulls recursively."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _repo_root(paths: RepeatabilityPaths) -> Path:
    try:
        root = paths.output_dir.resolve().parents[2]
    except IndexError as error:
        raise ValueError("repeatability output_dir does not identify a repository root") from error
    preregistration = root / PREREGISTRATION_RELATIVE_PATH
    if not preregistration.is_file():
        raise FileNotFoundError(f"missing frozen preregistration: {preregistration}")
    return root


def _resolve_recorded_path(value: str, *, artifact_dir: Path, root: Path) -> Path:
    recorded = Path(value)
    if recorded.is_absolute():
        return recorded
    root_candidate = root / recorded
    if root_candidate.exists():
        return root_candidate
    return artifact_dir / recorded


def _parse_transform(value: str | list[float]) -> Affine:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or len(parsed) != 6:
        raise ValueError("recorded transform must contain six affine coefficients")
    return Affine(*(float(item) for item in parsed))


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def _read_strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read a JSON object while rejecting non-standard numeric constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-standard JSON constant {value!r}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _input_manifest_records(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"frozen input manifest is missing or not a regular file: {path}")
    payload = _read_strict_json_object(path, label="frozen input manifest")
    _require_equal("input manifest schema", payload.get("schema_version"), "1.0")
    _require_equal("input manifest hash algorithm", payload.get("hash_algorithm"), "sha256")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("frozen input manifest has no input records")
    records: dict[str, dict[str, Any]] = {}
    logical_paths: set[str] = set()
    for record in inputs:
        if not isinstance(record, dict):
            raise ValueError("frozen input manifest contains a non-object input record")
        input_id = record.get("id")
        logical_path = record.get("logical_path")
        if not isinstance(input_id, str) or not input_id:
            raise ValueError("frozen input manifest contains an invalid input ID")
        if not isinstance(logical_path, str) or not logical_path:
            raise ValueError(f"frozen input manifest record {input_id!r} has no logical path")
        if input_id in records:
            raise ValueError(f"frozen input manifest contains duplicate input ID {input_id!r}")
        if logical_path in logical_paths:
            raise ValueError(
                f"frozen input manifest contains duplicate logical path {logical_path!r}"
            )
        size_bytes = record.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"frozen input manifest record {input_id!r} has invalid size_bytes")
        _require_sha256(record.get("sha256"), label=f"{input_id} expected SHA-256")
        records[input_id] = record
        logical_paths.add(logical_path)
    return records, _sha256(path)


def _admit_expected_file(path: Path, record: Mapping[str, Any], *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or not a regular file: {path}")
    _require_equal(f"{label} size", path.stat().st_size, record.get("size_bytes"))
    observed_sha = _sha256(path)
    _require_equal(f"{label} SHA-256", observed_sha, record.get("sha256"))
    return observed_sha


def _archive_library_members(
    archive_path: Path, *, library_name: str
) -> tuple[dict[str, dict[str, Any]], str]:
    expected: dict[str, dict[str, Any]] = {}
    tree_digest = hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        for info in sorted(members, key=lambda item: item.filename):
            member = PurePosixPath(info.filename)
            if (
                member.is_absolute()
                or len(member.parts) < 2
                or member.parts[0] != library_name
                or any(part in {"", ".", ".."} for part in member.parts)
            ):
                raise ValueError(
                    f"pinned spectral-library archive has an out-of-closure member: {info.filename}"
                )
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type and not stat.S_ISREG(file_type):
                raise ValueError(
                    f"pinned spectral-library archive member is not regular: {info.filename}"
                )
            relative = PurePosixPath(*member.parts[1:]).as_posix()
            if relative in expected:
                raise ValueError(
                    f"pinned spectral-library archive has duplicate member {relative!r}"
                )
            member_digest = hashlib.sha256()
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(b"\0")
            with archive.open(info) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    member_digest.update(chunk)
                    tree_digest.update(chunk)
            expected[relative] = {
                "size_bytes": info.file_size,
                "sha256": member_digest.hexdigest(),
            }
    if not expected:
        raise ValueError("pinned spectral-library archive contains no files")
    return expected, tree_digest.hexdigest()


def _validate_expected_input_identity(
    paths: RepeatabilityPaths, root: Path, input_manifest_path: Path
) -> dict[str, Any]:
    """Admit exact frozen raw scenes and the archive-derived library closure."""
    records, input_manifest_sha = _input_manifest_records(input_manifest_path)
    expected_scenes: dict[str, tuple[str, str, Path, Mapping[str, Any]]] = {}
    for site_id in sorted(SITES):
        for index, scene_id in enumerate(SITES[site_id].scene_ids, start=1):
            input_id = f"tanager-{site_id}-{index}"
            logical_path = f"data/raw/{scene_id}_{TANAGER_SR_ASSET}.h5"
            record = records.get(input_id)
            if record is None:
                raise ValueError(f"frozen input manifest lacks required scene record {input_id!r}")
            _require_equal(f"{input_id} logical path", record.get("logical_path"), logical_path)
            path = paths.raw_dir / f"{scene_id}_{TANAGER_SR_ASSET}.h5"
            expected_scenes[path.name] = (site_id, scene_id, input_id, record)

    actual_scene_names = {
        candidate.name
        for candidate in paths.raw_dir.glob(f"*_{TANAGER_SR_ASSET}.h5")
        if candidate.is_file() or candidate.is_symlink()
    }
    expected_scene_names = set(expected_scenes)
    missing_scenes = sorted(expected_scene_names - actual_scene_names)
    extra_scenes = sorted(actual_scene_names - expected_scene_names)
    if missing_scenes:
        raise FileNotFoundError(f"frozen raw-scene closure is missing files: {missing_scenes!r}")
    if extra_scenes:
        raise ValueError(f"frozen raw-scene closure has extra files: {extra_scenes!r}")

    raw_scenes: list[dict[str, str]] = []
    for filename in sorted(expected_scenes):
        site_id, scene_id, input_id, record = expected_scenes[filename]
        path = paths.raw_dir / filename
        observed_sha = _admit_expected_file(path, record, label=f"raw scene {input_id}")
        raw_scenes.append(
            {
                "site_id": site_id,
                "scene_id": scene_id,
                "path": str(path.resolve()),
                "sha256": observed_sha,
            }
        )

    archive_id = "usgs-splib07a-archive"
    archive_record = records.get(archive_id)
    if archive_record is None:
        raise ValueError(f"frozen input manifest lacks required record {archive_id!r}")
    archive_logical_path = "data/speclib/ASCIIdata_splib07a.zip"
    _require_equal(
        f"{archive_id} logical path", archive_record.get("logical_path"), archive_logical_path
    )
    archive_path = root / archive_logical_path
    archive_sha = _admit_expected_file(
        archive_path, archive_record, label="pinned spectral-library archive"
    )
    expected_members, expected_tree_sha = _archive_library_members(
        archive_path, library_name=paths.speclib_dir.name
    )

    if not paths.speclib_dir.is_dir() or paths.speclib_dir.is_symlink():
        raise FileNotFoundError(
            f"extracted spectral-library directory is missing or unsafe: {paths.speclib_dir}"
        )
    actual_members: dict[str, Path] = {}
    for candidate in paths.speclib_dir.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"extracted spectral-library closure contains a symlink: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(paths.speclib_dir).as_posix()
            actual_members[relative] = candidate
        elif not candidate.is_dir():
            raise ValueError(
                f"extracted spectral-library closure contains a non-regular entry: {candidate}"
            )
    missing_members = sorted(set(expected_members) - set(actual_members))
    extra_members = sorted(set(actual_members) - set(expected_members))
    if missing_members:
        raise FileNotFoundError(
            f"extracted spectral-library closure is missing files: {missing_members!r}"
        )
    if extra_members:
        raise ValueError(f"extracted spectral-library closure has extra files: {extra_members!r}")
    for relative, expected in expected_members.items():
        _admit_expected_file(
            actual_members[relative],
            expected,
            label=f"extracted spectral-library member {relative}",
        )
    observed_tree_sha = _directory_sha256(paths.speclib_dir)
    _require_equal("extracted spectral-library tree SHA-256", observed_tree_sha, expected_tree_sha)
    return {
        "input_manifest": str(input_manifest_path.resolve()),
        "input_manifest_sha256": input_manifest_sha,
        "raw_scenes": raw_scenes,
        "spectral_library": {
            "path": str(paths.speclib_dir.resolve()),
            "archive_path": str(archive_path.resolve()),
            "archive_sha256": archive_sha,
            "file_count": len(expected_members),
            "tree_sha256": observed_tree_sha,
            "expected_tree_sha256": expected_tree_sha,
        },
    }


def _execution_lock_path(output_dir: Path) -> Path:
    resolved_output = output_dir.resolve()
    identity = f"{SCIENTIFIC_EXECUTION_IDENTITY}\0{resolved_output}".encode()
    suffix = hashlib.sha256(identity).hexdigest()[:16]
    return resolved_output.parent / f".repeatability-{suffix}.lock"


@contextmanager
def _exclusive_output_lock(output_dir: Path, *, mode: str) -> Iterator[Path]:
    """Hold a fail-closed directory lock for one output identity."""
    lock_path = _execution_lock_path(output_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError(
            f"repeatability execution lock is already held or stale: {lock_path}"
        ) from error
    owner_path = lock_path / "owner.json"
    owner = {
        "schema_version": "1.0",
        "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
        "output_dir": str(output_dir.resolve()),
        "mode": mode,
        "pid": os.getpid(),
        "owner_id": os.urandom(32).hex(),
    }
    try:
        with owner_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(owner, sort_keys=True, allow_nan=False) + "\n")
    except BaseException:
        lock_path.rmdir()
        raise
    active_error: BaseException | None = None
    try:
        yield lock_path
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            if lock_path.is_symlink() or not lock_path.is_dir():
                raise RuntimeError(f"repeatability execution lock was replaced: {lock_path}")
            entries = list(lock_path.iterdir())
            if entries != [owner_path]:
                raise RuntimeError(f"repeatability execution lock contents changed: {lock_path}")
            observed_owner = _read_strict_json_object(
                owner_path, label="repeatability execution lock owner"
            )
            _require_equal("repeatability lock ownership", observed_owner, owner)
            owner_path.unlink()
            lock_path.rmdir()
        except BaseException:
            if active_error is None:
                raise
            logger.exception("repeatability lock could not be released safely")


def _validate_spatial_protocol(
    protocol: Any,
    *,
    preregistration_sha: str,
    label: str,
    require_compliance_flag: bool,
) -> None:
    if not isinstance(protocol, dict):
        raise ValueError(f"{label} has no protocol record")
    _require_equal(
        f"{label} preregistration path",
        protocol.get("path"),
        str(PREREGISTRATION_RELATIVE_PATH),
    )
    _require_equal(
        f"{label} preregistration SHA",
        protocol.get("sha256"),
        preregistration_sha,
    )
    _require_equal(
        f"{label} protocol parameters",
        protocol.get("parameters"),
        SPATIAL_PROTOCOL_PARAMETERS,
    )
    if require_compliance_flag and protocol.get("protocol_compliant") is not True:
        raise ValueError(f"{label} protocol_compliant must be true")


def _validate_external_reference_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("spatial summary has no external_reference_gate record")
    passed = value.get("passed")
    evaluable = value.get("evaluable")
    if not isinstance(passed, bool) or not isinstance(evaluable, bool):
        raise ValueError("external_reference_gate passed and evaluable must be booleans")
    if passed and not evaluable:
        raise ValueError("external_reference_gate cannot pass when it is not evaluable")
    return value


def _load_repeatability_handoff(
    paths: RepeatabilityPaths,
    block_manifest_path: Path,
    transfer_thresholds_path: Path,
    spatial_summary_path: Path | None = None,
) -> tuple[
    dict[str, BlockHandoff],
    dict[str, dict[str, TransferThreshold]],
    dict[str, Any],
]:
    """Load and strictly validate the spatial-validation handoff artifacts."""
    root = _repo_root(paths)
    preregistration_sha = _sha256(root / PREREGISTRATION_RELATIVE_PATH)
    summary_path = spatial_summary_path or block_manifest_path.with_name("summary.json")
    if not block_manifest_path.is_file():
        raise FileNotFoundError(f"block manifest does not exist: {block_manifest_path}")
    if not transfer_thresholds_path.is_file():
        raise FileNotFoundError(
            f"transfer-threshold artifact does not exist: {transfer_thresholds_path}"
        )
    if not summary_path.is_file():
        raise FileNotFoundError(f"spatial-validation summary does not exist: {summary_path}")
    manifest_sha = _sha256(block_manifest_path)
    manifest = _read_strict_json_object(block_manifest_path, label="block manifest")
    _validate_spatial_protocol(
        manifest.get("protocol"),
        preregistration_sha=preregistration_sha,
        label="block manifest",
        require_compliance_flag=False,
    )
    summary = _read_strict_json_object(summary_path, label="spatial summary")
    _validate_spatial_protocol(
        summary.get("protocol"),
        preregistration_sha=preregistration_sha,
        label="spatial summary",
        require_compliance_flag=True,
    )
    _require_equal(
        "spatial summary block manifest SHA",
        summary.get("block_manifest_sha256"),
        manifest_sha,
    )
    external_reference_gate = _validate_external_reference_gate(
        summary.get("external_reference_gate")
    )

    handoffs: dict[str, BlockHandoff] = {}
    for site_id, anchor_scene_id in _ANCHORS.items():
        site_entry = manifest.get("sites", {}).get(site_id)
        if not isinstance(site_entry, dict):
            raise ValueError(f"block manifest has no site entry for {site_id}")
        _require_equal(
            f"{site_id} block manifest anchor",
            site_entry.get("scene_id"),
            anchor_scene_id,
        )
        scale_entry = site_entry.get("scales", {}).get("L")
        if not isinstance(scale_entry, dict):
            raise ValueError(f"block manifest has no primary L record for {site_id}")
        _require_equal(
            f"{site_id} primary block raster",
            site_entry.get("block_raster"),
            scale_entry.get("block_raster"),
        )
        _require_equal(
            f"{site_id} complete block IDs",
            site_entry.get("complete_block_ids"),
            scale_entry.get("complete_block_ids"),
        )
        raster_path = _resolve_recorded_path(
            str(scale_entry["block_raster"]),
            artifact_dir=block_manifest_path.parent,
            root=root,
        )
        if not raster_path.is_file():
            raise FileNotFoundError(f"declared block raster does not exist: {raster_path}")
        raster_sha = _sha256(raster_path)
        _require_equal(
            f"{site_id} block raster SHA",
            raster_sha,
            scale_entry.get("block_raster_sha256"),
        )
        grid = site_entry.get("grid")
        if not isinstance(grid, dict):
            raise ValueError(f"block manifest has no grid record for {site_id}")
        shape_values = grid.get("shape")
        if not isinstance(shape_values, list) or len(shape_values) != 2:
            raise ValueError(f"block manifest shape is invalid for {site_id}")
        shape = (int(shape_values[0]), int(shape_values[1]))
        crs = str(grid.get("crs"))
        transform = _parse_transform(grid.get("transform"))
        complete_ids = tuple(int(value) for value in scale_entry["complete_block_ids"])
        if (
            not complete_ids
            or len(set(complete_ids)) != len(complete_ids)
            or min(complete_ids) <= 0
        ):
            raise ValueError(f"{site_id} complete block IDs must be unique positive integers")
        with rasterio.open(raster_path) as dataset:
            _require_equal(f"{site_id} block raster shape", dataset.shape, shape)
            if dataset.crs is None:
                raise ValueError(f"{site_id} block raster has no CRS")
            _require_equal(f"{site_id} block raster CRS", dataset.crs.to_string(), crs)
            _require_equal(f"{site_id} block raster transform", dataset.transform, transform)
            values = dataset.read(1, masked=False)
        positive_ids = set(int(value) for value in np.unique(values) if int(value) > 0)
        _require_equal(f"{site_id} raster complete block IDs", positive_ids, set(complete_ids))
        handoffs[site_id] = BlockHandoff(
            site_id=site_id,
            anchor_scene_id=anchor_scene_id,
            raster_path=raster_path,
            raster_sha256=raster_sha,
            complete_block_ids=complete_ids,
            shape=shape,
            crs=crs,
            transform=transform,
        )

    thresholds: dict[str, dict[str, TransferThreshold]] = {site_id: {} for site_id in _ANCHORS}
    with transfer_thresholds_path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            site_id = row.get("site", "")
            if site_id not in _ANCHORS:
                raise ValueError(
                    f"transfer threshold row {row_number} has unknown site {site_id!r}"
                )
            handoff = handoffs[site_id]
            _require_equal(f"row {row_number} scale", row.get("scale"), "L")
            _require_equal(
                f"row {row_number} anchor scene", row.get("scene_id"), handoff.anchor_scene_id
            )
            _require_equal(
                f"row {row_number} preregistration SHA",
                row.get("spatial_prereg_sha256"),
                preregistration_sha,
            )
            _require_equal(
                f"row {row_number} block manifest SHA",
                row.get("block_manifest_sha256"),
                manifest_sha,
            )
            recorded_manifest_path = _resolve_recorded_path(
                row["block_manifest_path"],
                artifact_dir=transfer_thresholds_path.parent,
                root=root,
            )
            _require_equal(
                f"row {row_number} block manifest path",
                recorded_manifest_path.resolve(),
                block_manifest_path.resolve(),
            )
            recorded_raster_path = _resolve_recorded_path(
                row["block_raster_path"],
                artifact_dir=block_manifest_path.parent,
                root=root,
            )
            _require_equal(
                f"row {row_number} block raster path",
                recorded_raster_path.resolve(),
                handoff.raster_path.resolve(),
            )
            _require_equal(
                f"row {row_number} block raster SHA",
                row.get("block_raster_sha256"),
                handoff.raster_sha256,
            )
            _require_equal(
                f"row {row_number} block shape",
                (int(row["block_shape_rows"]), int(row["block_shape_cols"])),
                handoff.shape,
            )
            _require_equal(f"row {row_number} block CRS", row.get("block_crs"), handoff.crs)
            _require_equal(
                f"row {row_number} block transform",
                _parse_transform(row["block_transform"]),
                handoff.transform,
            )
            for label in ("score", "reference"):
                source_path = _resolve_recorded_path(
                    row[f"source_{label}_path"],
                    artifact_dir=transfer_thresholds_path.parent,
                    root=root,
                )
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"row {row_number} source {label} is missing: {source_path}"
                    )
                _require_equal(
                    f"row {row_number} source {label} SHA",
                    _sha256(source_path),
                    row[f"source_{label}_sha256"],
                )
            status = row.get("threshold_status")
            threshold_text = row.get("threshold", "").strip()
            governance = row.get("governance_status", "")
            if status == "available":
                if governance == "counts_and_maps_only":
                    raise ValueError(f"row {row_number} exposes a counts-only threshold")
                if (
                    int(row["positive_bearing_blocks"]) < 5
                    or int(row["negative_bearing_blocks"]) < 5
                ):
                    raise ValueError(f"row {row_number} exposes a threshold without frozen support")
                _require_equal(
                    f"row {row_number} threshold method",
                    row.get("threshold_method"),
                    "block_balanced_youden_all_usable_complete_primary_L_blocks",
                )
                threshold = float(threshold_text)
                if not math.isfinite(threshold):
                    raise ValueError(f"row {row_number} threshold is not finite")
                unavailable_reason = None
            elif status == "unavailable":
                if threshold_text:
                    raise ValueError(f"row {row_number} unavailable threshold must be blank")
                threshold = None
                unavailable_reason = (
                    row.get("unavailable_reason") or "unspecified_threshold_failure"
                )
            else:
                raise ValueError(f"row {row_number} has invalid threshold_status {status!r}")
            record = TransferThreshold(
                site_id=site_id,
                scene_id=row["scene_id"],
                family=row["family"],
                layer=row["layer"],
                governance_status=governance,
                positive_bearing_blocks=int(row["positive_bearing_blocks"]),
                negative_bearing_blocks=int(row["negative_bearing_blocks"]),
                threshold=threshold,
                unavailable_reason=unavailable_reason,
            )
            if record.key in thresholds[site_id]:
                raise ValueError(f"duplicate transfer threshold for {site_id} {record.key}")
            thresholds[site_id][record.key] = record
    return (
        handoffs,
        thresholds,
        {
            "block_manifest": str(block_manifest_path),
            "block_manifest_sha256": manifest_sha,
            "transfer_thresholds": str(transfer_thresholds_path),
            "transfer_thresholds_sha256": _sha256(transfer_thresholds_path),
            "spatial_summary": str(summary_path),
            "spatial_summary_sha256": _sha256(summary_path),
            "spatial_prereg_sha256": preregistration_sha,
            "external_reference_gate": external_reference_gate,
        },
    )


def _load_anchor_reference(
    paths: RepeatabilityPaths, site_id: str
) -> tuple[xr.DataArray | None, str]:
    """Load the one external Rockwell clip aligned for a site's anchor scene."""
    path = paths.reference_dir / f"rockwell_{site_id}_{_ANCHORS[site_id]}.tif"
    if not path.exists():
        return None, f"anchor Rockwell clip is missing: {path}"
    with rioxarray.open_rasterio(path, masked=True) as opened:
        reference = opened.squeeze("band", drop=True).load()
    return reference, str(path)


def _reference_classes(key: str) -> frozenset[int] | None:
    kind, layer = key.split(":", maxsplit=1)
    if kind == "feature":
        return FEATURE_TO_ROCKWELL.get(layer)
    if kind == "mtmf":
        return MINERAL_TO_ROCKWELL.get(layer)
    raise ValueError(f"unknown score kind {kind!r}")


def _complete_overlap_block_ids(
    handoff: BlockHandoff,
    anchor_template: xr.DataArray,
    anchor_values: np.ndarray,
    repeat_values: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Retain blocks with overlap without applying a joint mask to either date."""
    anchor_shape = tuple(int(value) for value in anchor_template.shape)
    _require_equal(f"{handoff.site_id} anchor shape", anchor_shape, handoff.shape)
    if anchor_template.rio.crs is None:
        raise ValueError(f"{handoff.site_id} anchor has no CRS")
    _require_equal(
        f"{handoff.site_id} anchor CRS", anchor_template.rio.crs.to_string(), handoff.crs
    )
    _require_equal(
        f"{handoff.site_id} anchor transform", anchor_template.rio.transform(), handoff.transform
    )
    anchor_array = np.asarray(anchor_values, dtype=float)
    repeat_array = np.asarray(repeat_values, dtype=float)
    if anchor_array.shape != handoff.shape or repeat_array.shape != handoff.shape:
        raise ValueError(f"{handoff.site_id} metric arrays do not match block raster")
    with rasterio.open(handoff.raster_path) as dataset:
        block_values = dataset.read(1, masked=False)
    declared = np.asarray(handoff.complete_block_ids)
    in_declared = np.isin(block_values, declared)
    keep_ids: list[int] = []
    for block_id in declared:
        cells = in_declared & (block_values == block_id)
        if cells.any() and np.any(
            np.isfinite(anchor_array[cells]) & np.isfinite(repeat_array[cells])
        ):
            keep_ids.append(int(block_id))
    if not keep_ids:
        return np.zeros(block_values.shape, dtype=np.uint32), 0
    retained_raster = np.where(np.isin(block_values, keep_ids), block_values, 0).astype(np.uint32)
    return retained_raster, len(keep_ids)


def _resampling_summary(
    anchor_scores: np.ndarray,
    repeat_scores: np.ndarray,
    anchor_transferred: np.ndarray,
    repeat_transferred: np.ndarray,
    anchor_rank: np.ndarray,
    repeat_rank: np.ndarray,
    block_ids: np.ndarray,
    *,
    xres_m: float,
    yres_m: float,
    rockwell_reference: np.ndarray | None = None,
    transferred_threshold: float | None = None,
    rockwell_unavailable_reason: str | None = None,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    n_null: int = NULL_REPLICATES,
    workers: int = 1,
) -> dict[str, Any]:
    """Run bootstrap and null distributions with finite-replicate governance."""
    retained_ids = np.unique(block_ids[block_ids > 0])
    if retained_ids.size < 2:
        reason = "fewer than two complete paired overlap blocks are available"
        return {
            "status": "unavailable",
            "reason": reason,
            "n_complete_paired_overlap_blocks": int(retained_ids.size),
            "rockwell_block_support": {
                "positive_bearing_blocks": 0,
                "negative_bearing_blocks": 0,
            },
            "metrics": _unavailable_metric_contract(reason),
        }
    spearman_bootstrap = paired_block_bootstrap(
        anchor_scores,
        repeat_scores,
        block_ids,
        n_reps=n_bootstrap,
        metric=_safe_spearman,
        workers=workers,
    )
    spearman_null = paired_block_null(
        anchor_scores,
        repeat_scores,
        block_ids,
        n_reps=n_null,
        metric=_safe_spearman,
        workers=workers,
    )
    summaries: dict[str, Any] = {
        "spearman": {
            "bootstrap": _distribution_summary(spearman_bootstrap, interval=True),
            "spatial_null": _distribution_summary(spearman_null, interval=False),
        }
    }
    for prefix, left, right in (
        ("transferred", anchor_transferred, repeat_transferred),
        ("rank_relative", anchor_rank, repeat_rank),
    ):
        binary = _binary_block_resampling(
            left,
            right,
            block_ids,
            n_bootstrap=n_bootstrap,
            n_null=n_null,
        )
        for metric in ("iou", "dice", "prevalence_ratio"):
            spatial_null = (
                _not_applicable_summary(
                    "whole_block_null_is_not_defined_for_detection_prevalence_ratio",
                    tail="upper",
                )
                if metric == "prevalence_ratio"
                else _distribution_summary(binary[metric]["spatial_null"], interval=False)
            )
            summaries[f"{prefix}_{metric}"] = {
                "bootstrap": _distribution_summary(binary[metric]["bootstrap"], interval=True),
                "spatial_null": spatial_null,
            }
        boundary = _boundary_block_resampling(
            left,
            right,
            block_ids,
            xres_m=xres_m,
            yres_m=yres_m,
            n_bootstrap=n_bootstrap,
            n_null=n_null,
            workers=workers,
        )
        summaries[f"{prefix}_boundary_distance_m"] = {
            "bootstrap": _distribution_summary(boundary["bootstrap"], interval=True),
            "spatial_null": _distribution_summary(
                boundary["spatial_null"], interval=False, tail="lower"
            ),
        }

    rockwell_support: dict[str, Any] = {
        "positive_bearing_blocks": 0,
        "negative_bearing_blocks": 0,
    }
    if rockwell_reference is None:
        reason = rockwell_unavailable_reason or "Rockwell reference is unavailable"
        for metric in ("auc", "balanced_accuracy", "macro_f1"):
            summaries[f"rockwell_{metric}"] = {
                "bootstrap": _unavailable_summary(reason, interval=True),
                "spatial_null": _unavailable_summary(reason, interval=False, tail="upper"),
            }
    else:
        rockwell = _rockwell_block_resampling(
            repeat_scores,
            rockwell_reference,
            block_ids,
            threshold=transferred_threshold,
            n_bootstrap=n_bootstrap,
            n_null=n_null,
            workers=workers,
        )
        rockwell_support = {
            "positive_bearing_blocks": rockwell["positive_bearing_blocks"],
            "negative_bearing_blocks": rockwell["negative_bearing_blocks"],
        }
        for metric in ("auc", "balanced_accuracy", "macro_f1"):
            raw_components = rockwell["metrics"][metric]
            if raw_components["bootstrap"].get("status") == "unavailable":
                reason = str(raw_components["bootstrap"]["reason"])
                bootstrap_summary = _unavailable_summary(reason, interval=True)
                null_summary = _unavailable_summary(reason, interval=False, tail="upper")
            else:
                bootstrap_summary = _distribution_summary(
                    raw_components["bootstrap"], interval=True
                )
                null_summary = _distribution_summary(
                    raw_components["spatial_null"], interval=False, tail="upper"
                )
            summaries[f"rockwell_{metric}"] = {
                "bootstrap": bootstrap_summary,
                "spatial_null": null_summary,
            }
    if tuple(summaries) != _RESAMPLED_METRIC_COMPONENTS:
        raise RuntimeError("repeatability resampling metric contract is incomplete or reordered")
    return {
        "status": "available",
        "n_complete_paired_overlap_blocks": int(retained_ids.size),
        "rockwell_block_support": rockwell_support,
        "metrics": summaries,
    }


def _unavailable_metric_contract(reason: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in _RESAMPLED_METRIC_COMPONENTS:
        metrics[metric] = {
            "bootstrap": _unavailable_summary(reason, interval=True),
            "spatial_null": (
                _not_applicable_summary(
                    "whole_block_null_is_not_defined_for_detection_prevalence_ratio",
                    tail="upper",
                )
                if metric.endswith("prevalence_ratio")
                else _unavailable_summary(
                    reason,
                    interval=False,
                    tail="lower" if metric.endswith("boundary_distance_m") else "upper",
                )
            ),
        }
    return metrics


def _unavailable_summary(reason: str, *, interval: bool, tail: str = "upper") -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": "unavailable",
        "reason": reason,
        "scheduled_replicates": None,
        "finite_replicates": None,
        "finite_fraction": None,
        "gate_eligible": False,
        "unavailable_reason": reason,
    }
    if interval:
        summary.update({"lower_95": None, "upper_95": None})
    elif tail == "lower":
        summary["lower_5"] = None
    elif tail == "upper":
        summary["upper_95"] = None
    else:
        raise ValueError(f"unknown distribution tail {tail!r}")
    return summary


def _not_applicable_summary(reason: str, *, tail: str) -> dict[str, Any]:
    summary = _unavailable_summary(reason, interval=False, tail=tail)
    summary.update(
        {
            "status": "not_applicable",
            "reason": reason,
            "unavailable_reason": None,
        }
    )
    return summary


def _distribution_summary(
    result: Mapping[str, Any], *, interval: bool, tail: str = "upper"
) -> dict[str, Any]:
    samples = np.asarray(result["samples"], dtype=float)
    finite = samples[np.isfinite(samples)]
    gate_eligible = bool(result["gate_eligible"])
    summary: dict[str, Any] = {
        "status": "available" if gate_eligible else "unavailable",
        "scheduled_replicates": int(result["scheduled_replicates"]),
        "finite_replicates": int(result["finite_replicates"]),
        "finite_fraction": float(result["finite_fraction"]),
        "gate_eligible": gate_eligible,
        "unavailable_reason": None,
    }
    if not gate_eligible:
        summary["unavailable_reason"] = "fewer_than_95_percent_finite_replicates"
        if interval:
            summary.update({"lower_95": None, "upper_95": None})
        elif tail == "lower":
            summary["lower_5"] = None
        elif tail == "upper":
            summary["upper_95"] = None
        else:
            raise ValueError(f"unknown distribution tail {tail!r}")
        return summary
    if interval:
        lower, upper = np.percentile(finite, [2.5, 97.5])
        summary.update({"lower_95": float(lower), "upper_95": float(upper)})
    elif tail == "lower":
        summary["lower_5"] = float(np.percentile(finite, 5))
    elif tail == "upper":
        summary["upper_95"] = float(np.percentile(finite, 95))
    else:
        raise ValueError(f"unknown distribution tail {tail!r}")
    return summary


def _rockwell_metric_values(
    scores: np.ndarray,
    references: np.ndarray,
    *,
    threshold: float | None,
) -> dict[str, float]:
    """Return fixed-reference Rockwell metrics after pairwise-finite filtering."""
    score_values = np.asarray(scores, dtype=float)
    reference_values = np.asarray(references, dtype=float)
    if score_values.shape != reference_values.shape:
        raise ValueError("Rockwell scores and reference must share a shape")
    finite = np.isfinite(score_values) & np.isfinite(reference_values)
    score_values = score_values[finite]
    reference_values = reference_values[finite]
    if not np.all(np.isin(reference_values, (0.0, 1.0))):
        raise ValueError("Rockwell reference must be binary or NaN")
    positive = reference_values == 1
    negative = reference_values == 0
    if not positive.any() or not negative.any():
        return {
            "auc": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
        }
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(np.count_nonzero(negative))
    ranks = rankdata(score_values, method="average")
    auc = (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    if threshold is None or not math.isfinite(threshold):
        return {
            "auc": float(auc),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
        }
    predicted = score_values >= threshold
    true_positive = int(np.count_nonzero(predicted & positive))
    false_positive = int(np.count_nonzero(predicted & negative))
    true_negative = int(np.count_nonzero(~predicted & negative))
    false_negative = int(np.count_nonzero(~predicted & positive))
    tpr = true_positive / n_positive
    tnr = true_negative / n_negative
    positive_f1_denominator = 2 * true_positive + false_positive + false_negative
    negative_f1_denominator = 2 * true_negative + false_positive + false_negative
    positive_f1 = 2 * true_positive / positive_f1_denominator if positive_f1_denominator else 0.0
    negative_f1 = 2 * true_negative / negative_f1_denominator if negative_f1_denominator else 0.0
    return {
        "auc": float(auc),
        "balanced_accuracy": float((tpr + tnr) / 2),
        "macro_f1": float((positive_f1 + negative_f1) / 2),
    }


def _balanced_accuracy_at_threshold(
    scores: np.ndarray, references: np.ndarray, *, threshold: float
) -> float:
    return _rockwell_metric_values(scores, references, threshold=threshold)["balanced_accuracy"]


def _rockwell_block_support(
    scores: np.ndarray, binary_reference: np.ndarray, block_ids: np.ndarray
) -> tuple[int, int]:
    blocks = _extract_paired_blocks(scores, binary_reference, block_ids)
    positive_blocks = 0
    negative_blocks = 0
    for block in blocks:
        valid = np.isfinite(block.anchor) & np.isfinite(block.repeat)
        values = block.repeat[valid]
        positive_blocks += int(np.any(values == 1))
        negative_blocks += int(np.any(values == 0))
    return positive_blocks, negative_blocks


def _rockwell_block_resampling(
    scores: np.ndarray,
    binary_reference: np.ndarray,
    block_ids: np.ndarray,
    *,
    threshold: float | None,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    n_null: int = NULL_REPLICATES,
    seed: int = SEED,
    workers: int = 1,
) -> dict[str, Any]:
    """Bootstrap paired blocks and permute repeat scores against fixed Rockwell."""
    if n_bootstrap <= 0 or n_null <= 0:
        raise ValueError("resampling replicate counts must be positive")
    _validate_workers(workers)
    positive_blocks, negative_blocks = _rockwell_block_support(scores, binary_reference, block_ids)
    metrics: dict[str, Any] = {}
    for metric in ("auc", "balanced_accuracy", "macro_f1"):
        if metric != "auc" and (threshold is None or not math.isfinite(threshold)):
            metrics[metric] = {
                "bootstrap": {
                    "status": "unavailable",
                    "reason": "transferred_threshold_unavailable",
                },
                "spatial_null": {
                    "status": "unavailable",
                    "reason": "transferred_threshold_unavailable",
                },
            }
            continue

        def metric_value(
            sample_scores: np.ndarray,
            sample_reference: np.ndarray,
            *,
            metric_name: str = metric,
        ) -> float:
            return _rockwell_metric_values(sample_scores, sample_reference, threshold=threshold)[
                metric_name
            ]

        bootstrap = paired_block_bootstrap(
            scores,
            binary_reference,
            block_ids,
            n_reps=n_bootstrap,
            seed=seed,
            metric=metric_value,
            workers=workers,
        )
        spatial_null = paired_block_null(
            binary_reference,
            scores,
            block_ids,
            n_reps=n_null,
            seed=seed,
            metric=lambda fixed_reference, permuted_scores, metric_name=metric: (
                _rockwell_metric_values(permuted_scores, fixed_reference, threshold=threshold)[
                    metric_name
                ]
            ),
            workers=workers,
        )
        metrics[metric] = {"bootstrap": bootstrap, "spatial_null": spatial_null}
    return {
        "positive_bearing_blocks": positive_blocks,
        "negative_bearing_blocks": negative_blocks,
        "metrics": metrics,
    }


def _rockwell_block_bootstrap(
    scores: np.ndarray,
    binary_reference: np.ndarray,
    block_ids: np.ndarray,
    *,
    threshold: float,
    n_reps: int = BOOTSTRAP_REPLICATES,
    workers: int = 1,
) -> dict[str, Any]:
    positive_blocks, negative_blocks = _rockwell_block_support(scores, binary_reference, block_ids)
    support_eligible = positive_blocks >= 10 and negative_blocks >= 10
    bootstrap = paired_block_bootstrap(
        scores,
        binary_reference,
        block_ids,
        n_reps=n_reps,
        workers=workers,
        metric=lambda left, right: _balanced_accuracy_at_threshold(
            left, right, threshold=threshold
        ),
    )
    interval = _distribution_summary(bootstrap, interval=True)
    gate_eligible = support_eligible and bool(interval["gate_eligible"])
    if not support_eligible:
        reason = "fewer_than_10_positive_or_negative_bearing_blocks"
    else:
        reason = interval["unavailable_reason"]
    return {
        "positive_bearing_blocks": positive_blocks,
        "negative_bearing_blocks": negative_blocks,
        "support_eligible": support_eligible,
        "scheduled_replicates": interval["scheduled_replicates"],
        "finite_replicates": interval["finite_replicates"],
        "finite_fraction": interval["finite_fraction"],
        "lower_95": interval["lower_95"] if gate_eligible else None,
        "upper_95": interval["upper_95"] if gate_eligible else None,
        "gate_eligible": gate_eligible,
        "unavailable_reason": reason,
    }


def _rockwell_balanced_accuracy_gate_summary(
    uncertainty: Mapping[str, Any],
) -> dict[str, Any]:
    support = uncertainty.get("rockwell_block_support", {})
    positive_blocks = int(support.get("positive_bearing_blocks", 0))
    negative_blocks = int(support.get("negative_bearing_blocks", 0))
    support_eligible = positive_blocks >= 10 and negative_blocks >= 10
    component = (
        uncertainty.get("metrics", {}).get("rockwell_balanced_accuracy", {}).get("bootstrap")
    )
    if not isinstance(component, Mapping):
        component = _unavailable_summary("reference_or_block_support_unavailable", interval=True)
    distribution_eligible = bool(component.get("gate_eligible"))
    gate_eligible = support_eligible and distribution_eligible
    if not support_eligible:
        reason = "fewer_than_10_positive_or_negative_bearing_blocks"
    else:
        reason = component.get("unavailable_reason")
    return {
        "positive_bearing_blocks": positive_blocks,
        "negative_bearing_blocks": negative_blocks,
        "support_eligible": support_eligible,
        "scheduled_replicates": component.get("scheduled_replicates"),
        "finite_replicates": component.get("finite_replicates"),
        "finite_fraction": component.get("finite_fraction"),
        "lower_95": component.get("lower_95") if gate_eligible else None,
        "upper_95": component.get("upper_95") if gate_eligible else None,
        "gate_eligible": gate_eligible,
        "unavailable_reason": reason,
    }


def goldfield_pair_gate(
    *,
    spearman_bootstrap: Mapping[str, Any] | None,
    rockwell_balanced_accuracy: Mapping[str, Any] | None,
    observed_transferred_iou: float,
    transferred_iou_null: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the frozen unshifted Goldfield Al-OH three-part pair gate."""
    spearman_available = bool(
        spearman_bootstrap
        and spearman_bootstrap.get("gate_eligible")
        and spearman_bootstrap.get("lower_95") is not None
    )
    rockwell_available = bool(
        rockwell_balanced_accuracy
        and rockwell_balanced_accuracy.get("gate_eligible")
        and rockwell_balanced_accuracy.get("lower_95") is not None
    )
    iou_available = bool(
        transferred_iou_null
        and transferred_iou_null.get("gate_eligible")
        and transferred_iou_null.get("upper_95") is not None
        and math.isfinite(observed_transferred_iou)
    )
    components = {
        "spearman_bootstrap_lower_above_zero": {
            "available": spearman_available,
            "passed": bool(spearman_available and spearman_bootstrap["lower_95"] > 0),
            "reason": None if spearman_available else "spearman_bootstrap_interval_unavailable",
        },
        "repeat_rockwell_balanced_accuracy_lower_above_half": {
            "available": rockwell_available,
            "passed": bool(rockwell_available and rockwell_balanced_accuracy["lower_95"] > 0.5),
            "reason": None
            if rockwell_available
            else "rockwell_balanced_accuracy_interval_or_support_unavailable",
        },
        "transferred_iou_above_null_95": {
            "available": iou_available,
            "passed": bool(
                iou_available and observed_transferred_iou > transferred_iou_null["upper_95"]
            ),
            "reason": None if iou_available else "transferred_iou_null_unavailable",
        },
    }
    evaluable = all(component["available"] for component in components.values())
    passed = evaluable and all(component["passed"] for component in components.values())
    return {
        "grid": "unshifted_only",
        "registration_sensitivity_can_rescue": False,
        "evaluable": evaluable,
        "passed": passed,
        "components": components,
        "unavailable_reasons": [
            component["reason"]
            for component in components.values()
            if component["reason"] is not None
        ],
    }


def classify_goldfield_repeatability(pair_gates: list[Mapping[str, Any]]) -> str:
    """Apply the frozen four-comparison Goldfield classification."""
    if len(pair_gates) != 4:
        raise ValueError("Goldfield classification requires exactly four primary comparisons")
    passed = sum(bool(gate.get("passed")) for gate in pair_gates)
    if passed == 4:
        return "strong"
    if 1 <= passed <= 3:
        return "date-dependent"
    if all(bool(gate.get("evaluable")) for gate in pair_gates):
        return "unsupported"
    return "unavailable"


def combined_public_gate(
    external_reference_gate: Mapping[str, Any], goldfield_repeatability: str
) -> dict[str, Any]:
    """Combine the frozen external-reference and repeatability decisions."""
    external = _validate_external_reference_gate(dict(external_reference_gate))
    if goldfield_repeatability not in {
        "strong",
        "date-dependent",
        "unsupported",
        "unavailable",
    }:
        raise ValueError(
            f"unknown Goldfield repeatability classification {goldfield_repeatability!r}"
        )
    if not external["evaluable"] or goldfield_repeatability == "unavailable":
        classification = "unavailable"
        wording = "unavailable"
    elif external["passed"] and goldfield_repeatability == "strong":
        classification = "validated_and_repeatable"
        wording = "validated and repeatable"
    elif not external["passed"] and goldfield_repeatability == "strong":
        classification = "stable_only"
        wording = "stable"
    elif external["passed"]:
        classification = "acquisition_specific"
        wording = "acquisition-specific"
    else:
        classification = "failed"
        wording = "failed"
    return {
        "status": classification,
        "classification": classification,
        "passed": classification == "validated_and_repeatable",
        "frozen_wording": wording,
        "external_reference_gate_passed": external["passed"],
        "external_reference_gate_evaluable": external["evaluable"],
        "goldfield_repeatability_classification": goldfield_repeatability,
    }


def _evaluated_layer_keys(
    left_scores: Mapping[str, xr.DataArray],
    right_scores: Mapping[str, xr.DataArray],
    thresholds: Mapping[str, TransferThreshold],
) -> tuple[str, ...]:
    """Return only manifest-declared layers, while requiring each score pair."""
    declared = set(thresholds)
    paired = set(left_scores) & set(right_scores)
    missing = sorted(declared - paired)
    if missing:
        raise ValueError(
            "transfer-threshold layers are missing from one or both scenes: " + ", ".join(missing)
        )
    return tuple(sorted(declared))


def _pair_result(
    pair: PairSpec,
    left: SceneProducts,
    right: SceneProducts,
    grid_anchor: SceneProducts,
    thresholds: Mapping[str, TransferThreshold],
    block_handoff: BlockHandoff,
    anchor_reference: xr.DataArray | None,
    reference_source: str,
    *,
    workers: int = 1,
    selected_layer_keys: tuple[str, ...] | None = None,
    precomputed_layers: Mapping[str, Mapping[str, Any]] | None = None,
    progress_callback: Callable[[PairSpec, str, str, float | None, Mapping[str, Any] | None], None]
    | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen scene pair on the calibration anchor grid."""
    _validate_workers(workers)
    xres_m, yres_m = _require_projected_resolution(grid_anchor.template)
    left_valid = (
        left.qa_valid.astype(bool)
        if left.scene_id == grid_anchor.scene_id
        else _reproject_mask(left.qa_valid.astype("uint8"), grid_anchor.template)
    )
    right_valid = (
        right.qa_valid.astype(bool)
        if right.scene_id == grid_anchor.scene_id
        else _reproject_mask(right.qa_valid.astype("uint8"), grid_anchor.template)
    )
    evaluated_keys = _evaluated_layer_keys(left.scores, right.scores, thresholds)
    if selected_layer_keys is not None:
        missing_selected = set(selected_layer_keys) - set(evaluated_keys)
        if missing_selected:
            raise ValueError(f"selected repeatability layers are unavailable: {missing_selected}")
        evaluated_keys = tuple(key for key in evaluated_keys if key in selected_layer_keys)
    layers: dict[str, Any] = {}
    for key in evaluated_keys:
        if precomputed_layers is not None and key in precomputed_layers:
            layers[key] = dict(precomputed_layers[key])
            continue
        task_started = perf_counter()
        if progress_callback is not None:
            progress_callback(pair, key, "running", None, None)
        threshold_record = thresholds[key]
        transferred_threshold = threshold_record.threshold
        threshold_value = (
            float(transferred_threshold) if transferred_threshold is not None else float("nan")
        )
        left_score = (
            left.scores[key]
            if left.scene_id == grid_anchor.scene_id
            else _reproject_continuous(left.scores[key], grid_anchor.template)
        )
        right_score = (
            right.scores[key]
            if right.scene_id == grid_anchor.scene_id
            else _reproject_continuous(right.scores[key], grid_anchor.template)
        )
        left_score_values = np.where(
            np.asarray(left_valid.values, dtype=bool),
            np.asarray(left_score.values, dtype=float),
            np.nan,
        )
        right_score_values = np.where(
            np.asarray(right_valid.values, dtype=bool),
            np.asarray(right_score.values, dtype=float),
            np.nan,
        )
        left_transferred_native = _source_mask(left.scores[key], left.qa_valid, threshold_value)
        right_transferred_native = _source_mask(right.scores[key], right.qa_valid, threshold_value)
        left_transferred = (
            left_transferred_native
            if left.scene_id == grid_anchor.scene_id
            else _reproject_binary(left_transferred_native, grid_anchor.template)
        )
        right_transferred = (
            right_transferred_native
            if right.scene_id == grid_anchor.scene_id
            else _reproject_binary(right_transferred_native, grid_anchor.template)
        )
        left_rank_threshold = _upper_decile_threshold(left.scores[key], left.qa_valid)
        right_rank_threshold = _upper_decile_threshold(right.scores[key], right.qa_valid)
        left_rank_native = _source_mask(left.scores[key], left.qa_valid, left_rank_threshold)
        right_rank_native = _source_mask(right.scores[key], right.qa_valid, right_rank_threshold)
        left_rank = (
            left_rank_native
            if left.scene_id == grid_anchor.scene_id
            else _reproject_binary(left_rank_native, grid_anchor.template)
        )
        right_rank = (
            right_rank_native
            if right.scene_id == grid_anchor.scene_id
            else _reproject_binary(right_rank_native, grid_anchor.template)
        )
        left_transferred_values = np.where(
            np.asarray(left_valid.values, dtype=bool),
            np.asarray(left_transferred.values, dtype=float),
            np.nan,
        )
        right_transferred_values = np.where(
            np.asarray(right_valid.values, dtype=bool),
            np.asarray(right_transferred.values, dtype=float),
            np.nan,
        )
        left_rank_values = np.where(
            np.asarray(left_valid.values, dtype=bool),
            np.asarray(left_rank.values, dtype=float),
            np.nan,
        )
        right_rank_values = np.where(
            np.asarray(right_valid.values, dtype=bool),
            np.asarray(right_rank.values, dtype=float),
            np.nan,
        )
        sensitivity = registration_sensitivity(
            left_score_values,
            right_score_values,
            left_transferred_values,
            right_transferred_values,
            left_rank_values,
            right_rank_values,
            np.asarray(left_valid.values, dtype=bool),
            np.asarray(right_valid.values, dtype=bool),
            xres_m=xres_m,
            yres_m=yres_m,
        )
        unshifted = asdict(sensitivity.unshifted)
        joint_finite_count = int(
            np.count_nonzero(np.isfinite(left_score_values) & np.isfinite(right_score_values))
        )
        left_qa_count = int(np.asarray(left_valid.values, dtype=bool).sum())
        right_qa_count = int(np.asarray(right_valid.values, dtype=bool).sum())
        layers[key] = {
            "transferred_threshold": asdict(threshold_record),
            "left_rank_relative_upper_decile_threshold": left_rank_threshold,
            "right_rank_relative_upper_decile_threshold": right_rank_threshold,
            "unshifted": unshifted,
            "overlap": {
                "joint_finite_count": joint_finite_count,
                "left_qa_valid_count_on_anchor_grid": left_qa_count,
                "right_qa_valid_count_on_anchor_grid": right_qa_count,
                "joint_fraction_left_qa": float(joint_finite_count / left_qa_count)
                if left_qa_count
                else float("nan"),
                "joint_fraction_right_qa": float(joint_finite_count / right_qa_count)
                if right_qa_count
                else float("nan"),
                "joint_fraction_calibration_anchor_grid": float(
                    joint_finite_count / left_score_values.size
                ),
            },
            "registration_sensitivity": {
                "all_fixed_shifts": [asdict(row) for row in sensitivity.shift_metrics],
                "full_range_not_best_selected": sensitivity.ranges,
            },
        }
        positive_classes = _reference_classes(key)
        binary_reference: np.ndarray | None = None
        rockwell_unavailable_reason: str | None = None
        if anchor_reference is None:
            rockwell_unavailable_reason = reference_source
            layers[key]["repeat_scene_rockwell"] = {
                "available": False,
                "reason": reference_source,
                "n_usable": 0,
                "n_pos": 0,
                "n_neg": 0,
                "threshold": transferred_threshold,
                "auc": float("nan"),
                "threshold_metrics_available": False,
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
            }
        elif positive_classes is None:
            rockwell_unavailable_reason = "no Rockwell positive-class mapping for this layer"
            layers[key]["repeat_scene_rockwell"] = {
                "available": False,
                "reason": rockwell_unavailable_reason,
                "n_usable": 0,
                "n_pos": 0,
                "n_neg": 0,
                "threshold": transferred_threshold,
                "auc": float("nan"),
                "threshold_metrics_available": False,
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
            }
        else:
            layers[key]["repeat_scene_rockwell"] = {
                **fixed_threshold_reference_metrics(
                    right_score_values,
                    np.asarray(anchor_reference.values),
                    positive_classes,
                    threshold=transferred_threshold,
                ),
                "reference_source": reference_source,
                "positive_classes": sorted(positive_classes),
            }
            reference_values = np.asarray(anchor_reference.values, dtype=float)
            binary_reference = np.full(reference_values.shape, np.nan, dtype=float)
            reference_domain = np.isfinite(reference_values)
            for excluded in ROCKWELL_EXCLUDED:
                reference_domain &= reference_values != excluded
            binary_reference[reference_domain] = np.isin(
                reference_values[reference_domain], tuple(positive_classes)
            ).astype(float)

        block_grid, _ = _complete_overlap_block_ids(
            block_handoff,
            grid_anchor.template,
            left_score_values,
            right_score_values,
        )
        uncertainty = _resampling_summary(
            left_score_values,
            right_score_values,
            left_transferred_values,
            right_transferred_values,
            left_rank_values,
            right_rank_values,
            block_grid,
            xres_m=xres_m,
            yres_m=yres_m,
            rockwell_reference=binary_reference,
            transferred_threshold=transferred_threshold,
            rockwell_unavailable_reason=rockwell_unavailable_reason,
            workers=workers,
        )
        layers[key]["uncertainty_and_nulls"] = uncertainty
        rockwell_bootstrap = _rockwell_balanced_accuracy_gate_summary(uncertainty)
        layers[key]["repeat_scene_rockwell"]["balanced_accuracy_block_bootstrap"] = (
            rockwell_bootstrap
        )
        if (
            pair.site_id == "goldfield"
            and pair.role == "primary"
            and key == "feature:al_oh_doublet"
        ):
            metric_summaries = uncertainty.get("metrics", {})
            layers[key]["goldfield_al_oh_doublet_pair_gate"] = goldfield_pair_gate(
                spearman_bootstrap=metric_summaries.get("spearman", {}).get("bootstrap"),
                rockwell_balanced_accuracy=rockwell_bootstrap,
                observed_transferred_iou=float(unshifted["transferred_iou"]),
                transferred_iou_null=metric_summaries.get("transferred_iou", {}).get(
                    "spatial_null"
                ),
            )
        if progress_callback is not None:
            progress_callback(
                pair,
                key,
                "completed",
                perf_counter() - task_started,
                layers[key],
            )
    return {
        "site_id": pair.site_id,
        "anchor_scene_id": pair.anchor_scene_id,
        "repeat_scene_id": pair.repeat_scene_id,
        "comparison_role": pair.role,
        "layers": layers,
    }


def _pair_task_id(pair: PairSpec, layer: str) -> str:
    return ":".join(
        (
            pair.role,
            pair.site_id,
            pair.anchor_scene_id,
            pair.repeat_scene_id,
            layer,
        )
    )


def _task_order(
    thresholds: Mapping[str, Mapping[str, TransferThreshold]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for pair in (*PRIMARY_PAIRS, *SECONDARY_PAIRS):
        for layer in sorted(thresholds[pair.site_id]):
            tasks.append(
                {
                    "index": len(tasks),
                    "task_id": _pair_task_id(pair, layer),
                    "comparison_role": pair.role,
                    "site_id": pair.site_id,
                    "anchor_scene_id": pair.anchor_scene_id,
                    "repeat_scene_id": pair.repeat_scene_id,
                    "layer": layer,
                }
            )
    return tasks


def _pair_from_task(task: Mapping[str, Any]) -> PairSpec:
    for pair in (*PRIMARY_PAIRS, *SECONDARY_PAIRS):
        if (
            pair.role == task.get("comparison_role")
            and pair.site_id == task.get("site_id")
            and pair.anchor_scene_id == task.get("anchor_scene_id")
            and pair.repeat_scene_id == task.get("repeat_scene_id")
        ):
            return pair
    raise ValueError(f"task does not identify a frozen repeatability pair: {task.get('task_id')}")


def _timing_pilot_task(
    tasks: list[dict[str, Any]],
    thresholds: Mapping[str, Mapping[str, TransferThreshold]],
) -> tuple[dict[str, Any], PairSpec]:
    """Choose the first frozen task able to execute every resampling branch."""
    for task in tasks:
        layer = str(task["layer"])
        threshold = thresholds[str(task["site_id"])][layer]
        if _reference_classes(layer) is not None and threshold.threshold is not None:
            return task, _pair_from_task(task)
    raise ValueError(
        "timing pilot requires a Rockwell-mapped layer with an available transferred threshold"
    )


def _timing_pilot_branch_schedule(pair_result: Mapping[str, Any], layer: str) -> dict[str, Any]:
    """Validate full branch scheduling without returning any endpoint value."""
    uncertainty = pair_result.get("layers", {}).get(layer, {}).get("uncertainty_and_nulls", {})
    metrics = uncertainty.get("metrics", {})
    if tuple(metrics) != _RESAMPLED_METRIC_COMPONENTS:
        raise RuntimeError("timing pilot did not produce the complete metric contract")
    n_blocks = int(uncertainty.get("n_complete_paired_overlap_blocks", 0))
    if n_blocks < 2:
        raise RuntimeError("timing pilot requires at least two complete paired overlap blocks")
    total_permutations = math.factorial(n_blocks)
    expected_null = total_permutations if total_permutations <= NULL_REPLICATES else NULL_REPLICATES
    schedule: dict[str, Any] = {}
    for metric, components in metrics.items():
        bootstrap_count = components["bootstrap"].get("scheduled_replicates")
        if bootstrap_count != BOOTSTRAP_REPLICATES:
            raise RuntimeError(f"timing pilot did not fully schedule {metric} bootstrap")
        if metric.endswith("prevalence_ratio"):
            if components["spatial_null"].get("status") != "not_applicable":
                raise RuntimeError(f"timing pilot has an invalid {metric} null schema")
            null_record: dict[str, Any] = {"status": "not_applicable"}
        else:
            null_count = components["spatial_null"].get("scheduled_replicates")
            if null_count != expected_null:
                raise RuntimeError(f"timing pilot did not fully schedule {metric} null")
            null_record = {"status": "scheduled", "scheduled_replicates": null_count}
        schedule[metric] = {
            "bootstrap": {"scheduled_replicates": bootstrap_count},
            "spatial_null": null_record,
        }
    return {
        "contains_endpoint_values": False,
        "n_complete_paired_overlap_blocks": n_blocks,
        "components": schedule,
    }


def _validate_timing_pilot_schedule(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("timing pilot resampling schedule must be an object")
    _require_equal(
        "timing pilot schedule fields",
        set(value),
        {"contains_endpoint_values", "n_complete_paired_overlap_blocks", "components"},
    )
    _require_equal(
        "timing pilot schedule endpoint seal", value.get("contains_endpoint_values"), False
    )
    n_blocks = value.get("n_complete_paired_overlap_blocks")
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, int) or n_blocks < 2:
        raise ValueError("timing pilot schedule requires at least two complete blocks")
    components = value.get("components")
    if not isinstance(components, dict):
        raise ValueError("timing pilot schedule components must be an object")
    _require_equal(
        "timing pilot schedule component closure",
        set(components),
        set(_RESAMPLED_METRIC_COMPONENTS),
    )
    total_permutations = math.factorial(n_blocks)
    expected_null = total_permutations if total_permutations <= NULL_REPLICATES else NULL_REPLICATES
    for metric in _RESAMPLED_METRIC_COMPONENTS:
        metric_schedule = components.get(metric)
        if not isinstance(metric_schedule, dict):
            raise ValueError(f"timing pilot {metric} schedule must be an object")
        _require_equal(
            f"timing pilot {metric} schedule fields",
            set(metric_schedule),
            {"bootstrap", "spatial_null"},
        )
        _require_equal(
            f"timing pilot {metric} bootstrap schedule",
            metric_schedule.get("bootstrap"),
            {"scheduled_replicates": BOOTSTRAP_REPLICATES},
        )
        expected_spatial_null = (
            {"status": "not_applicable"}
            if metric.endswith("prevalence_ratio")
            else {"status": "scheduled", "scheduled_replicates": expected_null}
        )
        _require_equal(
            f"timing pilot {metric} null schedule",
            metric_schedule.get("spatial_null"),
            expected_spatial_null,
        )


def _validate_timing_pilot_admission(
    output_dir: Path,
    *,
    expected_sha256: str,
    execution_manifest: Mapping[str, Any],
    pilot_task: Mapping[str, Any],
) -> Path:
    """Admit only the exact reviewed, redacted timing artifact for full mode."""
    _require_sha256(expected_sha256, label="expected timing-pilot SHA-256")
    timing_path = output_dir / "timing_pilot.json"
    if timing_path.is_symlink() or not timing_path.is_file():
        raise FileNotFoundError(
            f"reviewed timing-pilot artifact is missing or not a regular file: {timing_path}"
        )
    _require_equal("timing-pilot SHA-256", _sha256(timing_path), expected_sha256)
    timing = _read_strict_json_object(timing_path, label="reviewed timing-pilot artifact")
    _require_equal(
        "timing pilot fields",
        set(timing),
        {
            "schema_version",
            "mode",
            "scientific_execution_identity",
            "accepted_scientific_result",
            "contains_endpoint_values",
            "execution_manifest",
            "execution_manifest_sha256",
            "task_id",
            "workers",
            "bootstrap_replicates",
            "null_replicates_maximum",
            "resampling_branch_schedule",
            "elapsed_seconds",
            "result_sha256",
        },
    )
    _require_equal("timing pilot schema", timing.get("schema_version"), TIMING_PILOT_SCHEMA_VERSION)
    _require_equal("timing pilot mode", timing.get("mode"), "timing")
    _require_equal(
        "timing pilot scientific execution identity",
        timing.get("scientific_execution_identity"),
        SCIENTIFIC_EXECUTION_IDENTITY,
    )
    _require_equal(
        "timing pilot accepted-scientific-result flag",
        timing.get("accepted_scientific_result"),
        False,
    )
    _require_equal("timing pilot endpoint seal", timing.get("contains_endpoint_values"), False)
    execution_path = output_dir / "execution_manifest.json"
    if execution_path.is_symlink() or not execution_path.is_file():
        raise FileNotFoundError(
            f"timing pilot execution manifest is missing or not regular: {execution_path}"
        )
    recorded_execution_path = timing.get("execution_manifest")
    if not isinstance(recorded_execution_path, str):
        raise ValueError("timing pilot execution_manifest must be a path string")
    _require_equal(
        "timing pilot execution-manifest path",
        Path(recorded_execution_path).resolve(),
        execution_path.resolve(),
    )
    recorded_execution_sha = _require_sha256(
        timing.get("execution_manifest_sha256"),
        label="timing pilot execution-manifest SHA-256",
    )
    _require_equal(
        "timing pilot execution-manifest SHA-256",
        _sha256(execution_path),
        recorded_execution_sha,
    )
    existing_execution = _read_strict_json_object(
        execution_path, label="timing pilot execution manifest"
    )
    _require_equal("timing pilot execution manifest", existing_execution, dict(execution_manifest))
    _require_equal("timing pilot task identity", timing.get("task_id"), pilot_task.get("task_id"))
    expected_workers = execution_manifest.get("compute_controls", {}).get("workers")
    _require_equal("timing pilot worker schedule", timing.get("workers"), expected_workers)
    _require_equal(
        "timing pilot bootstrap schedule",
        timing.get("bootstrap_replicates"),
        BOOTSTRAP_REPLICATES,
    )
    _require_equal(
        "timing pilot null schedule maximum",
        timing.get("null_replicates_maximum"),
        NULL_REPLICATES,
    )
    elapsed_seconds = timing.get("elapsed_seconds")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("timing pilot elapsed_seconds must be finite and non-negative")
    timing_result_sha = _require_sha256(
        timing.get("result_sha256"), label="timing pilot redacted result SHA-256"
    )
    progress_path = output_dir / "progress.json"
    if progress_path.is_symlink() or not progress_path.is_file():
        raise FileNotFoundError(
            f"timing pilot progress ledger is missing or not regular: {progress_path}"
        )
    progress = _read_strict_json_object(progress_path, label="timing pilot progress ledger")
    _require_equal(
        "timing pilot progress fields",
        set(progress),
        {
            "schema_version",
            "execution_manifest_sha256",
            "run_status",
            "accepted_final_manifest",
            "tasks",
        },
    )
    _require_equal(
        "timing pilot progress schema",
        progress.get("schema_version"),
        PROGRESS_SCHEMA_VERSION,
    )
    _require_equal(
        "timing pilot progress execution SHA-256",
        progress.get("execution_manifest_sha256"),
        _stable_json_sha256(execution_manifest),
    )
    _require_equal(
        "timing pilot progress status",
        progress.get("run_status"),
        "timing_pilot_complete",
    )
    if progress.get("accepted_final_manifest") is not False:
        raise ValueError("timing pilot progress accepted-final-manifest flag must be false")
    progress_tasks = progress.get("tasks")
    if not isinstance(progress_tasks, list):
        raise ValueError("timing pilot progress tasks must be a list")
    member_order = execution_manifest.get("member_order")
    if not isinstance(member_order, dict):
        raise ValueError("timing pilot execution member_order must be an object")
    expected_tasks = member_order.get("tasks")
    if not isinstance(expected_tasks, list) or not expected_tasks:
        raise ValueError("timing pilot execution tasks must be a non-empty list")
    if not all(isinstance(task, dict) for task in expected_tasks):
        raise ValueError("timing pilot execution task rows must be objects")
    if not all(isinstance(row, dict) for row in progress_tasks):
        raise ValueError("timing pilot progress task rows must be objects")
    _require_equal(
        "timing pilot progress task order",
        [row.get("task_id") for row in progress_tasks],
        [task.get("task_id") for task in expected_tasks],
    )
    metadata_fields = {
        "status",
        "attempts",
        "elapsed_seconds",
        "result_path",
        "result_sha256",
    }
    pilot_rows: list[dict[str, Any]] = []
    for expected_task, row in zip(expected_tasks, progress_tasks, strict=True):
        _require_equal(
            f"timing pilot progress {row.get('task_id')} fields",
            set(row),
            set(expected_task) | metadata_fields,
        )
        for key, expected_value in expected_task.items():
            _require_equal(
                f"timing pilot progress {row.get('task_id')} {key}",
                row.get(key),
                expected_value,
            )
        attempts = row.get("attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ValueError(
                f"timing pilot progress {row.get('task_id')} attempts must be "
                "a non-negative integer"
            )
        if row.get("task_id") == pilot_task.get("task_id"):
            pilot_rows.append(row)
            continue
        _require_equal(
            f"timing pilot progress {row.get('task_id')} status",
            row.get("status"),
            "pending",
        )
        _require_equal(
            f"timing pilot progress {row.get('task_id')} attempts",
            attempts,
            0,
        )
        for key in ("elapsed_seconds", "result_path", "result_sha256"):
            _require_equal(
                f"timing pilot progress {row.get('task_id')} {key}",
                row.get(key),
                None,
            )
    if len(pilot_rows) != 1:
        raise ValueError("timing pilot progress must contain exactly one admitted task row")
    pilot_row = pilot_rows[0]
    _require_equal("timing pilot progress task status", pilot_row.get("status"), "completed")
    _require_equal("timing pilot progress task attempts", pilot_row.get("attempts"), 1)
    _require_equal("timing pilot progress task result path", pilot_row.get("result_path"), None)
    progress_result_sha = _require_sha256(
        pilot_row.get("result_sha256"),
        label="timing pilot progress result SHA-256",
    )
    _require_equal(
        "timing pilot result SHA-256 provenance",
        progress_result_sha,
        timing_result_sha,
    )
    progress_elapsed_seconds = pilot_row.get("elapsed_seconds")
    if (
        isinstance(progress_elapsed_seconds, bool)
        or not isinstance(progress_elapsed_seconds, (int, float))
        or not math.isfinite(progress_elapsed_seconds)
        or progress_elapsed_seconds < 0
    ):
        raise ValueError(
            "timing pilot progress elapsed_seconds must be finite, non-negative, and non-boolean"
        )
    _require_equal(
        "timing pilot elapsed-seconds provenance",
        progress_elapsed_seconds,
        elapsed_seconds,
    )
    _validate_timing_pilot_schedule(timing.get("resampling_branch_schedule"))
    return timing_path


def _execution_source_inventory(
    paths: RepeatabilityPaths, root: Path, input_manifest_path: Path
) -> dict[str, Any]:
    input_admission = _validate_expected_input_identity(paths, root, input_manifest_path)
    references: list[dict[str, Any]] = []
    for site_id in sorted(SITES):
        path = paths.reference_dir / f"rockwell_{site_id}_{_ANCHORS[site_id]}.tif"
        references.append(
            {
                "site_id": site_id,
                "path": str(path.resolve()),
                "available": path.is_file(),
                "sha256": _sha256(path) if path.is_file() else None,
            }
        )
    code_paths = (Path(__file__).resolve(), root / "scripts" / "run_repeatability.py")
    code_bytes = []
    for path in code_paths:
        if not path.is_file():
            raise FileNotFoundError(f"repeatability code file is missing: {path}")
        code_bytes.append({"path": str(path), "sha256": _sha256(path)})
    return {
        "expected_input_admission": {
            "input_manifest": input_admission["input_manifest"],
            "input_manifest_sha256": input_admission["input_manifest_sha256"],
            "status": "admitted_before_computation",
        },
        "raw_scenes": input_admission["raw_scenes"],
        "reference_rasters": references,
        "spectral_library": input_admission["spectral_library"],
        "code_bytes": code_bytes,
    }


def _task_result_path(output_dir: Path, task: Mapping[str, Any], attempt: int) -> Path:
    return output_dir / "task_results" / f"{int(task['index']):05d}-attempt-{attempt:03d}.json"


def _load_resumable_results(
    output_dir: Path,
    execution_manifest: Mapping[str, Any],
    tasks: list[dict[str, Any]],
    *,
    resume: bool,
) -> tuple[Path, Path, dict[str, Any], dict[str, Mapping[str, Any]]]:
    _require_equal(
        "repeatability execution schema",
        execution_manifest.get("schema_version"),
        EXECUTION_SCHEMA_VERSION,
    )
    _require_equal(
        "repeatability scientific execution identity",
        execution_manifest.get("scientific_execution_identity"),
        SCIENTIFIC_EXECUTION_IDENTITY,
    )
    execution_path = output_dir / "execution_manifest.json"
    progress_path = output_dir / "progress.json"
    final_path = output_dir / "manifest.json"
    pilot_path = output_dir / "timing_pilot.json"
    if final_path.exists() or (pilot_path.exists() and not resume):
        raise FileExistsError(f"completed repeatability artifact already exists in {output_dir}")
    execution_sha = _stable_json_sha256(execution_manifest)
    if execution_path.exists() or progress_path.exists():
        if not resume:
            raise FileExistsError(
                "repeatability execution state already exists in "
                f"{output_dir}; use --resume or a fresh path"
            )
        existing_execution = _read_strict_json_object(
            execution_path, label="repeatability execution manifest"
        )
        _require_equal(
            "repeatability execution manifest", existing_execution, dict(execution_manifest)
        )
        progress = _read_strict_json_object(progress_path, label="repeatability progress ledger")
        _require_equal("progress schema", progress.get("schema_version"), PROGRESS_SCHEMA_VERSION)
        _require_equal(
            "progress execution SHA", progress.get("execution_manifest_sha256"), execution_sha
        )
        _require_equal(
            "progress task order",
            [row.get("task_id") for row in progress.get("tasks", [])],
            [task["task_id"] for task in tasks],
        )
    else:
        if resume:
            raise FileNotFoundError(
                f"no repeatability execution is available to resume in {output_dir}"
            )
        _atomic_write_json(execution_path, execution_manifest)
        progress = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "execution_manifest_sha256": execution_sha,
            "run_status": "running",
            "accepted_final_manifest": False,
            "tasks": [
                {
                    **task,
                    "status": "pending",
                    "attempts": 0,
                    "elapsed_seconds": None,
                    "result_path": None,
                    "result_sha256": None,
                }
                for task in tasks
            ],
        }
        _atomic_write_json(progress_path, progress)

    completed: dict[str, Mapping[str, Any]] = {}
    for row in progress["tasks"]:
        if row.get("status") == "completed":
            if row.get("result_path") is None:
                row["status"] = "pending"
                row["elapsed_seconds"] = None
                row["result_sha256"] = None
                continue
            recorded_path = Path(str(row.get("result_path")))
            result_path = (
                recorded_path if recorded_path.is_absolute() else output_dir / recorded_path
            )
            if not result_path.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"task result escapes output directory: {result_path}")
            if not result_path.is_file():
                raise FileNotFoundError(f"completed task result is missing: {result_path}")
            _require_equal(
                f"{row['task_id']} result SHA", _sha256(result_path), row.get("result_sha256")
            )
            completed[row["task_id"]] = _read_strict_json_object(
                result_path, label=f"{row['task_id']} result"
            )
        elif row.get("status") in {"pending", "running", "failed"}:
            row["status"] = "pending"
            row["elapsed_seconds"] = None
            row["result_path"] = None
            row["result_sha256"] = None
        else:
            raise ValueError(f"invalid task status for {row.get('task_id')}: {row.get('status')}")
    progress["run_status"] = "running"
    progress["accepted_final_manifest"] = False
    _atomic_write_json(progress_path, progress)
    return execution_path, progress_path, progress, completed


def _run_repeatability_packet_locked(
    paths: RepeatabilityPaths,
    *,
    block_manifest: Path,
    transfer_thresholds: Path | None = None,
    spatial_summary: Path | None = None,
    input_manifest: Path | None = None,
    workers: int = 1,
    timing_pilot: bool = False,
    resume: bool = False,
    expected_timing_pilot_sha256: str | None = None,
    expected_resource_admission_sha256: str | None = None,
    resource_admission_path: Path | None = None,
) -> Path:
    """Execute a repeatability mode while its output-identity lock is held."""
    root = _repo_root(paths)
    if not timing_pilot:
        assert expected_timing_pilot_sha256 is not None
        assert expected_resource_admission_sha256 is not None
        assert resource_admission_path is not None
        timing_path = paths.output_dir / "timing_pilot.json"
        execution_path = paths.output_dir / "execution_manifest.json"
        _require_equal(
            "timing-pilot SHA-256 before scientific input reads",
            secure_sha256_file(timing_path),
            expected_timing_pilot_sha256,
        )
        validate_resource_admission(
            resource_admission_path=resource_admission_path,
            expected_sha256=expected_resource_admission_sha256,
            expected_timing_pilot_sha256=expected_timing_pilot_sha256,
            expected_execution_manifest_sha256=secure_sha256_file(execution_path),
            rule_path=root / RESOURCE_RULE_RELATIVE_PATH,
            source_manifest_path=root / RESOURCE_SOURCE_MANIFEST_RELATIVE_PATH,
            verifier_script_path=root / VERIFIER_SCRIPT_RELATIVE_PATH,
            verifier_module_path=root / VERIFIER_MODULE_RELATIVE_PATH,
        )
    input_manifest_path = input_manifest or root / INPUT_MANIFEST_RELATIVE_PATH
    source_inventory = _execution_source_inventory(paths, root, input_manifest_path)
    threshold_path = transfer_thresholds or block_manifest.with_name("transfer_thresholds.csv")
    handoffs, thresholds, handoff_provenance = _load_repeatability_handoff(
        paths, block_manifest, threshold_path, spatial_summary
    )
    tasks = _task_order(thresholds)
    if not tasks:
        raise ValueError("repeatability handoff declares no pair/layer tasks")
    execution_manifest = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
        "permitted_actions": ["timing_pilot", "full"],
        "accepted_scientific_result": False,
        "resume_validates_completed_results": True,
        "protocol_artifacts": {
            key: handoff_provenance[key]
            for key in (
                "block_manifest",
                "block_manifest_sha256",
                "transfer_thresholds",
                "transfer_thresholds_sha256",
                "spatial_summary",
                "spatial_summary_sha256",
                "spatial_prereg_sha256",
            )
        },
        "block_handoffs": {
            site_id: {
                "raster_path": str(handoff.raster_path.resolve()),
                "raster_sha256": handoff.raster_sha256,
                "shape": list(handoff.shape),
                "crs": handoff.crs,
                "transform": list(handoff.transform)[:6],
            }
            for site_id, handoff in sorted(handoffs.items())
        },
        "source_inventory": source_inventory,
        "member_order": {
            "scenes": {site_id: list(site_scene_order(site_id)) for site_id in sorted(SITES)},
            "complete_block_ids": {
                site_id: list(handoffs[site_id].complete_block_ids) for site_id in sorted(handoffs)
            },
            "tasks": tasks,
        },
        "resampling": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "null_replicates_maximum": NULL_REPLICATES,
            "seed": SEED,
            "bootstrap_order": "one default_rng.integers call per replicate",
            "null_order": "all permutations if factorial <= 9999 else sorted seeded unique draws",
            "finite_replicate_fraction": FINITE_REPLICATE_FRACTION,
            "full_block_shapes_and_moving_missingness": True,
            "metric_components": list(_RESAMPLED_METRIC_COMPONENTS),
            "boundary_null_tail": "lower_5",
            "all_other_applicable_null_tails": "upper_95",
            "prevalence_ratio_null": "not_applicable",
        },
        "compute_controls": {
            "workers": workers,
            "task_result_order": "frozen pair order then sorted manifest-declared layer",
            "spearman_method": "unchanged scipy.stats.spearmanr per draw",
            "thread_chunk_results_restored_to_seeded_order": True,
            "binary_method": "exact integer block and cross-block sufficient statistics",
            "boundary_method": "within-block joint-finite 8-neighbour coordinates",
            "rockwell_null": "fixed reference with permuted repeat score blocks and missingness",
            "statistics_batch_bytes": STATISTICS_BATCH_BYTES,
        },
    }
    pilot_task, pilot_pair = _timing_pilot_task(tasks, thresholds)
    if not timing_pilot:
        assert expected_timing_pilot_sha256 is not None
        assert expected_resource_admission_sha256 is not None
        assert resource_admission_path is not None
        _validate_timing_pilot_admission(
            paths.output_dir,
            expected_sha256=expected_timing_pilot_sha256,
            execution_manifest=execution_manifest,
            pilot_task=pilot_task,
        )
    execution_path, progress_path, progress, completed = _load_resumable_results(
        paths.output_dir, execution_manifest, tasks, resume=resume
    )
    progress_rows = {row["task_id"]: row for row in progress["tasks"]}
    active_task_id: str | None = None

    def record_progress(
        pair: PairSpec,
        layer: str,
        status: str,
        elapsed_seconds: float | None,
        result: Mapping[str, Any] | None,
    ) -> None:
        nonlocal active_task_id
        task_id = _pair_task_id(pair, layer)
        row = progress_rows[task_id]
        if status == "running":
            if row["status"] != "pending":
                raise ValueError(f"task {task_id} cannot start from status {row['status']}")
            row["status"] = "running"
            row["attempts"] = int(row["attempts"]) + 1
            active_task_id = task_id
        elif status == "completed":
            if row["status"] != "running" or result is None or elapsed_seconds is None:
                raise ValueError(f"task {task_id} completion record is incomplete")
            row["status"] = "completed"
            row["elapsed_seconds"] = elapsed_seconds
            if timing_pilot:
                row["result_path"] = None
                row["result_sha256"] = _stable_json_sha256(result)
            else:
                result_path = _task_result_path(paths.output_dir, row, int(row["attempts"]))
                _atomic_write_json(result_path, result)
                row["result_path"] = str(result_path.relative_to(paths.output_dir))
                row["result_sha256"] = _sha256(result_path)
            active_task_id = None
        else:
            raise ValueError(f"unknown progress status {status!r}")
        _atomic_write_json(progress_path, progress)
        logger.info("repeatability task %s: %s", task_id, status)

    try:
        if timing_pilot:
            required_pairs = (pilot_pair,)
            selected_layers = {pilot_task["task_id"]: (str(pilot_task["layer"]),)}
        else:
            pilot_task = tasks[0]
            required_pairs = (*PRIMARY_PAIRS, *SECONDARY_PAIRS)
            selected_layers = {}
        needed_scenes = {
            (pair.site_id, scene_id)
            for pair in required_pairs
            for scene_id in (pair.anchor_scene_id, pair.repeat_scene_id, _ANCHORS[pair.site_id])
        }
        raw_hashes = {
            (row["site_id"], row["scene_id"]): row["sha256"]
            for row in source_inventory["raw_scenes"]
        }
        implementation_sha = source_inventory["code_bytes"][0]["sha256"]
        preregistration_sha = handoff_provenance["spatial_prereg_sha256"]
        speclib_sha = source_inventory["spectral_library"]["tree_sha256"]
        products: dict[tuple[str, str], SceneProducts] = {}
        for site_id in sorted({site for site, _scene in needed_scenes}):
            frozen_features: tuple[FeatureDef, ...] | None = None
            frozen_endmember_samples: dict[str, str] | None = None
            for scene_id in site_scene_order(site_id):
                if (site_id, scene_id) not in needed_scenes:
                    continue
                product, features, samples = _load_scene_products(
                    site_id,
                    scene_id,
                    paths,
                    frozen_features=frozen_features,
                    frozen_endmember_samples=frozen_endmember_samples,
                    source_sha256=raw_hashes[(site_id, scene_id)],
                    implementation_sha256=implementation_sha,
                    preregistration_sha256=preregistration_sha,
                    speclib_sha256=speclib_sha,
                )
                if frozen_features is None:
                    frozen_features = features
                    frozen_endmember_samples = samples
                products[(site_id, scene_id)] = product
        references = {
            site_id: _load_anchor_reference(paths, site_id)
            for site_id in sorted({pair.site_id for pair in required_pairs})
        }
        pair_results: list[dict[str, Any]] = []
        for pair in required_pairs:
            pair_precomputed = {
                layer: completed[_pair_task_id(pair, layer)]
                for layer in sorted(thresholds[pair.site_id])
                if _pair_task_id(pair, layer) in completed
            }
            pair_results.append(
                _pair_result(
                    pair,
                    products[(pair.site_id, pair.anchor_scene_id)],
                    products[(pair.site_id, pair.repeat_scene_id)],
                    products[(pair.site_id, _ANCHORS[pair.site_id])],
                    thresholds[pair.site_id],
                    handoffs[pair.site_id],
                    *references[pair.site_id],
                    workers=workers,
                    selected_layer_keys=selected_layers.get(pilot_task["task_id"]),
                    precomputed_layers=pair_precomputed,
                    progress_callback=record_progress,
                )
            )
        if timing_pilot:
            row = progress_rows[pilot_task["task_id"]]
            branch_schedule = _timing_pilot_branch_schedule(
                pair_results[0], str(pilot_task["layer"])
            )
            timing_path = paths.output_dir / "timing_pilot.json"
            _atomic_write_json(
                timing_path,
                {
                    "schema_version": TIMING_PILOT_SCHEMA_VERSION,
                    "mode": "timing",
                    "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
                    "accepted_scientific_result": False,
                    "contains_endpoint_values": False,
                    "execution_manifest": str(execution_path.resolve()),
                    "execution_manifest_sha256": _sha256(execution_path),
                    "task_id": row["task_id"],
                    "workers": workers,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "null_replicates_maximum": NULL_REPLICATES,
                    "resampling_branch_schedule": branch_schedule,
                    "elapsed_seconds": row["elapsed_seconds"],
                    "result_sha256": row["result_sha256"],
                },
            )
            progress["run_status"] = "timing_pilot_complete"
            _atomic_write_json(progress_path, progress)
            return timing_path

        incomplete_tasks = [
            row["task_id"] for row in progress["tasks"] if row["status"] != "completed"
        ]
        if incomplete_tasks:
            raise RuntimeError(
                "refusing final manifest with incomplete repeatability tasks: "
                + ", ".join(incomplete_tasks)
            )
        goldfield_gates = [
            result["layers"]["feature:al_oh_doublet"]["goldfield_al_oh_doublet_pair_gate"]
            for result in pair_results
            if result["site_id"] == "goldfield" and result["comparison_role"] == "primary"
        ]
        goldfield_classification = classify_goldfield_repeatability(goldfield_gates)
        manifest = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
            "protocol": "docs/m2_spatial_validation_preregistration.md",
            "execution_manifest": str(execution_path),
            "execution_manifest_sha256": _sha256(execution_path),
            "parameters": {
                "mtmf_ridge": MTMF_RIDGE,
                "max_infeasibility": MAX_INFEASIBILITY,
                "rank_relative_quantile": UPPER_DECILE_QUANTILE,
                "feature_and_endmember_freezing": (
                    "site-anchor specific; samples resampled per scene"
                ),
                "registration_shifts": [[dy, dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
                "registration_selection": "full range reported; no best shift selected",
                "workers_compute_only": workers,
            },
            "frozen_anchors": _ANCHORS,
            "spatial_validation_handoff": handoff_provenance,
            "scene_products": [
                {
                    "site_id": product.site_id,
                    "scene_id": product.scene_id,
                    "directory": str(
                        _scene_output_dir(paths.output_dir, product.site_id, product.scene_id)
                    ),
                    "feature_definitions": [
                        asdict(feature) for feature in product.feature_definitions
                    ],
                    "endmember_samples": product.endmember_samples,
                    "score_layers": sorted(product.scores),
                }
                for product in products.values()
            ],
            "strict_inductive_covariance": {
                "status": "separate_required_stage",
                "reason": (
                    "run scripts/run_strict_inductive.py against the same verified "
                    "spatial-validation handoff; this repeatability command does not "
                    "duplicate the cube-level covariance folds"
                ),
            },
            "goldfield_al_oh_doublet_repeatability": {
                "classification": goldfield_classification,
                "primary_pair_gates": goldfield_gates,
            },
            "combined_public_gate": combined_public_gate(
                handoff_provenance["external_reference_gate"], goldfield_classification
            ),
            "comparisons": pair_results,
        }
        manifest_path = paths.output_dir / "manifest.json"
        _atomic_write_json(manifest_path, manifest)
        progress["run_status"] = "completed"
        progress["accepted_final_manifest"] = True
        progress["final_manifest_sha256"] = _sha256(manifest_path)
        _atomic_write_json(progress_path, progress)
        logger.info("wrote %s", manifest_path)
        return manifest_path
    except BaseException as error:
        if active_task_id is not None:
            progress_rows[active_task_id]["status"] = "failed"
        progress["run_status"] = "failed"
        progress["accepted_final_manifest"] = False
        progress["failure_type"] = type(error).__name__
        try:
            _atomic_write_json(progress_path, progress)
        except Exception:
            logger.exception("failed to record repeatability failure")
        raise


def run_repeatability_packet(
    paths: RepeatabilityPaths,
    *,
    block_manifest: Path,
    transfer_thresholds: Path | None = None,
    spatial_summary: Path | None = None,
    input_manifest: Path | None = None,
    workers: int = 1,
    timing_pilot: bool = False,
    resume: bool = False,
    expected_timing_pilot_sha256: str | None = None,
    expected_resource_admission_sha256: str | None = None,
    resource_admission_path: Path | None = None,
) -> Path:
    """Process frozen scenes with deterministic compute controls and fail-closed output."""
    _validate_workers(workers)
    if timing_pilot:
        if resume:
            raise ValueError("timing mode cannot resume existing repeatability state")
        if expected_timing_pilot_sha256 is not None:
            raise ValueError("timing mode does not admit a timing-pilot digest")
        if expected_resource_admission_sha256 is not None or resource_admission_path is not None:
            raise ValueError("timing mode does not admit resource-admission evidence")
        mode = "timing"
    else:
        if not resume:
            raise ValueError("full mode requires --resume from an admitted timing pilot")
        if expected_timing_pilot_sha256 is None:
            raise ValueError("full mode requires an expected timing-pilot SHA-256")
        if expected_resource_admission_sha256 is None or resource_admission_path is None:
            raise ValueError("full mode requires reviewed resource-admission path and SHA-256")
        _require_sha256(expected_timing_pilot_sha256, label="expected timing-pilot SHA-256")
        _require_sha256(
            expected_resource_admission_sha256,
            label="expected resource-admission SHA-256",
        )
        mode = "full"
    with _exclusive_output_lock(paths.output_dir, mode=mode):
        return _run_repeatability_packet_locked(
            paths,
            block_manifest=block_manifest,
            transfer_thresholds=transfer_thresholds,
            spatial_summary=spatial_summary,
            input_manifest=input_manifest,
            workers=workers,
            timing_pilot=timing_pilot,
            resume=resume,
            expected_timing_pilot_sha256=expected_timing_pilot_sha256,
            expected_resource_admission_sha256=expected_resource_admission_sha256,
            resource_admission_path=resource_admission_path,
        )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BlockHandoff",
    "FINITE_REPLICATE_FRACTION",
    "MAX_INFEASIBILITY",
    "MTMF_RIDGE",
    "NULL_REPLICATES",
    "PRIMARY_PAIRS",
    "SECONDARY_PAIRS",
    "PairSpec",
    "RepeatabilityPaths",
    "TransferThreshold",
    "binary_overlap_metrics",
    "classify_goldfield_repeatability",
    "combined_public_gate",
    "fixed_threshold_reference_metrics",
    "goldfield_pair_gate",
    "paired_block_bootstrap",
    "paired_block_null",
    "registration_sensitivity",
    "resample_frozen_endmembers",
    "run_repeatability_packet",
    "site_scene_order",
    "symmetric_boundary_distance_m",
]
