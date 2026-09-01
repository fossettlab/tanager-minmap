"""Spatially blocked validation primitives for Tanager alteration scores.

This module implements the frozen design in
``docs/m2_spatial_validation_preregistration.md`` without changing the legacy
pixelwise routines in :mod:`tanager_minmap.validate`.  Functions are written as
pure transformations over arrays and immutable records so synthetic tests can
exercise the statistical contract without repository data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from math import ceil, log

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import rankdata

LAG_PIXELS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
MAX_PAIRS_PER_LAG = 200_000
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 9_999
FINITE_REPLICATE_FRACTION = 0.95
SEED = 42


@dataclass(frozen=True)
class VariogramPoint:
    """One empirical semivariogram estimate at a declared lag."""

    lag_pixels: int
    distance: float
    semivariance: float
    available_pairs: int
    used_pairs: int


@dataclass(frozen=True)
class VariogramFit:
    """Exponential variogram fit or its preregistered empirical fallback."""

    nugget: float
    sill: float
    scale: float
    practical_range: float
    method: str
    fallback_reason: str | None


@dataclass(frozen=True)
class Block:
    """A complete square spatial block expressed in zero-based raster cells."""

    block_id: str
    block_row: int
    block_col: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int


@dataclass(frozen=True)
class BlockSample:
    """A score and binary reference field on one complete spatial block."""

    block: Block
    score: np.ndarray
    reference: np.ndarray

    def paired_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Return finite score/reference pairs as flat float and integer arrays."""
        if self.score.shape != self.reference.shape:
            raise ValueError(f"block {self.block.block_id} score/reference shapes differ")
        valid = np.isfinite(self.score) & np.isfinite(self.reference)
        score = self.score[valid].astype(float)
        reference = self.reference[valid]
        if not np.all(np.isin(reference, (0.0, 1.0))):
            raise ValueError(f"block {self.block.block_id} reference is not binary")
        return score, reference.astype(np.int8)


@dataclass(frozen=True)
class FoldThreshold:
    """Threshold and sample counts for one held-out complete block."""

    block_id: str
    threshold: float
    n_test: int
    n_pos: int
    n_neg: int
    n_training_blocks: int


@dataclass(frozen=True)
class OutOfFoldResult:
    """Pooled predictions from leave-one-block-out spatial cross-fitting."""

    scores: np.ndarray
    references: np.ndarray
    predictions: np.ndarray
    block_ids: np.ndarray
    folds: tuple[FoldThreshold, ...]
    skipped_blocks: tuple[str, ...]
    auc_scores: np.ndarray | None = None
    auc_references: np.ndarray | None = None
    auc_block_ids: np.ndarray | None = None


@dataclass(frozen=True)
class MetricSet:
    """Pooled rank and thresholded classification metrics.

    ``rank_n_*`` describes the AUC and prevalence support, while
    ``threshold_n_*`` describes balanced accuracy, F1, TPR, and FPR support.
    ``n_pos`` and ``n_neg`` remain backward-compatible aliases for the
    rank-support counts.
    """

    auc: float
    balanced_accuracy: float
    positive_f1: float
    negative_f1: float
    macro_f1: float
    tpr: float
    fpr: float
    prevalence: float
    n_pos: int
    n_neg: int
    rank_n_pos: int
    rank_n_neg: int
    threshold_n_pos: int
    threshold_n_neg: int


@dataclass(frozen=True)
class ConfidenceInterval:
    """Percentile interval from paired complete-block bootstrap replicates."""

    metric: str
    lower: float
    upper: float
    scheduled_replicates: int
    valid_replicates: int
    finite_fraction: float
    gate_eligible: bool
    unavailable_reason: str | None


@dataclass(frozen=True)
class PermutationResult:
    """One-sided whole-block permutation results for the two primary metrics."""

    auc_p_value: float | None
    balanced_accuracy_p_value: float | None
    valid_auc_permutations: int
    valid_balanced_accuracy_permutations: int


def _direction_differences(field: np.ndarray, lag: int) -> list[np.ndarray]:
    """Collect finite squared differences in four declared directions."""
    directions = ((0, lag), (lag, 0), (lag, lag), (lag, -lag))
    differences: list[np.ndarray] = []
    n_rows, n_cols = field.shape
    for row_offset, col_offset in directions:
        row_a = slice(0, n_rows - row_offset) if row_offset >= 0 else slice(-row_offset, n_rows)
        row_b = slice(row_offset, n_rows) if row_offset >= 0 else slice(0, n_rows + row_offset)
        if col_offset >= 0:
            col_a = slice(0, n_cols - col_offset)
            col_b = slice(col_offset, n_cols)
        else:
            col_a = slice(-col_offset, n_cols)
            col_b = slice(0, n_cols + col_offset)
        first = field[row_a, col_a]
        second = field[row_b, col_b]
        valid = np.isfinite(first) & np.isfinite(second)
        if np.any(valid):
            differences.append(np.square(first[valid] - second[valid]))
    return differences


def empirical_semivariogram(
    field: np.ndarray,
    *,
    pixel_size: float,
    lags: Sequence[int] = LAG_PIXELS,
    max_pairs: int = MAX_PAIRS_PER_LAG,
) -> tuple[VariogramPoint, ...]:
    """Estimate the preregistered omnidirectional empirical semivariogram.

    Horizontal, vertical, and two diagonal directions are pooled at each
    declared pixel lag.  When more than ``max_pairs`` finite pairs are
    available, evenly spaced indices over the deterministic pooled order are
    retained.  This avoids a stochastic or result-dependent thinning step.
    """
    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("field must be a two-dimensional array")
    if pixel_size <= 0 or not np.isfinite(pixel_size):
        raise ValueError("pixel_size must be finite and positive")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")

    max_lag = min(values.shape) // 4
    points: list[VariogramPoint] = []
    for lag in lags:
        if lag <= 0 or lag > max_lag:
            continue
        parts = _direction_differences(values, int(lag))
        if not parts:
            continue
        squared = np.concatenate(parts)
        available = int(squared.size)
        if available > max_pairs:
            index = np.linspace(0, available - 1, max_pairs, dtype=np.int64)
            squared = squared[index]
        points.append(
            VariogramPoint(
                lag_pixels=int(lag),
                distance=float(lag * pixel_size),
                semivariance=float(0.5 * np.mean(squared)),
                available_pairs=available,
                used_pairs=int(squared.size),
            )
        )
    return tuple(points)


def _exponential_variogram(distance: np.ndarray, nugget: float, sill: float, scale: float):
    return nugget + sill * (1.0 - np.exp(-distance / scale))


def _variogram_fallback(
    points: Sequence[VariogramPoint], field_variance: float, reason: str
) -> VariogramFit:
    target = 0.95 * field_variance
    reached = next((point for point in points if point.semivariance >= target), None)
    selected = reached if reached is not None else points[-1]
    suffix = "first_lag_at_95pct_variance" if reached is not None else "largest_evaluated_lag"
    return VariogramFit(
        nugget=float("nan"),
        sill=float("nan"),
        scale=float("nan"),
        practical_range=float(selected.distance),
        method=f"empirical_fallback_{suffix}",
        fallback_reason=reason,
    )


def fit_exponential_variogram(
    points: Sequence[VariogramPoint], *, field_variance: float
) -> VariogramFit:
    """Fit the frozen exponential model or apply its deterministic fallback."""
    if not points:
        raise ValueError("at least one empirical variogram point is required")
    if field_variance < 0 or not np.isfinite(field_variance):
        raise ValueError("field_variance must be finite and non-negative")
    if len(points) < 3:
        return _variogram_fallback(points, field_variance, "fewer_than_three_lags")

    distance = np.asarray([point.distance for point in points], dtype=float)
    gamma = np.asarray([point.semivariance for point in points], dtype=float)
    finite = np.isfinite(distance) & np.isfinite(gamma)
    if np.count_nonzero(finite) < 3:
        return _variogram_fallback(points, field_variance, "fewer_than_three_finite_lags")
    distance = distance[finite]
    gamma = gamma[finite]
    epsilon = np.finfo(float).eps
    nugget0 = max(0.0, float(np.min(gamma)))
    sill0 = max(float(np.max(gamma) - nugget0), field_variance, epsilon)
    scale0 = max(float(np.median(distance)), epsilon)
    try:
        parameters, _ = curve_fit(
            _exponential_variogram,
            distance,
            gamma,
            p0=(nugget0, sill0, scale0),
            bounds=((0.0, 0.0, epsilon), (np.inf, np.inf, np.inf)),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return _variogram_fallback(points, field_variance, f"fit_failed:{type(error).__name__}")

    nugget, sill, scale = (float(value) for value in parameters)
    practical_range = -scale * log(0.05)
    fitted = np.asarray((nugget, sill, scale, practical_range))
    if not np.all(np.isfinite(fitted)) or sill <= 0:
        return _variogram_fallback(points, field_variance, "nonfinite_or_nonpositive_sill")
    largest_evaluated_distance = float(np.max(distance))
    if practical_range > largest_evaluated_distance:
        finite_points = tuple(
            point
            for point in points
            if np.isfinite(point.distance) and np.isfinite(point.semivariance)
        )
        return _variogram_fallback(
            finite_points,
            field_variance,
            "fitted_practical_range_beyond_largest_evaluated_lag",
        )
    return VariogramFit(
        nugget=nugget,
        sill=sill,
        scale=scale,
        practical_range=practical_range,
        method="exponential_bounded_least_squares",
        fallback_reason=None,
    )


def block_dimensions(practical_ranges: Iterable[float], pixel_size: float) -> tuple[int, int]:
    """Return primary block side ``L`` and halo width in whole pixels."""
    ranges = np.asarray(tuple(practical_ranges), dtype=float)
    if ranges.size == 0 or not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
        raise ValueError("practical_ranges must contain finite positive values")
    if pixel_size <= 0 or not np.isfinite(pixel_size):
        raise ValueError("pixel_size must be finite and positive")
    site_range = float(np.max(ranges))
    halo_pixels = max(1, int(ceil(site_range / pixel_size)))
    block_side_pixels = max(1, int(ceil(2.0 * site_range / pixel_size)))
    return block_side_pixels, halo_pixels


def complete_blocks(shape: tuple[int, int], block_side_pixels: int) -> tuple[Block, ...]:
    """Tile a raster from its upper-left origin and retain only complete blocks."""
    n_rows, n_cols = shape
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("shape must be positive")
    if block_side_pixels <= 0:
        raise ValueError("block_side_pixels must be positive")
    blocks: list[Block] = []
    for block_row in range(n_rows // block_side_pixels):
        for block_col in range(n_cols // block_side_pixels):
            row_start = block_row * block_side_pixels
            col_start = block_col * block_side_pixels
            blocks.append(
                Block(
                    block_id=f"r{block_row:04d}_c{block_col:04d}",
                    block_row=block_row,
                    block_col=block_col,
                    row_start=row_start,
                    row_stop=row_start + block_side_pixels,
                    col_start=col_start,
                    col_stop=col_start + block_side_pixels,
                )
            )
    return tuple(blocks)


def categorical_block_raster(
    shape: tuple[int, int], blocks: Sequence[Block]
) -> tuple[np.ndarray, dict[int, str]]:
    """Encode complete blocks as deterministic positive integer raster IDs.

    Zero is reserved for nodata outside complete blocks.  Positive IDs follow
    the supplied block order, which is row-major for :func:`complete_blocks`.
    The returned mapping preserves the human-readable block IDs used by the
    CSV manifest.
    """
    n_rows, n_cols = shape
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("shape must be positive")
    if len(blocks) >= np.iinfo(np.uint32).max:
        raise ValueError("too many blocks for uint32 categorical IDs")
    raster = np.zeros(shape, dtype=np.uint32)
    mapping: dict[int, str] = {}
    seen_string_ids: set[str] = set()
    for numeric_id, block in enumerate(blocks, start=1):
        if block.block_id in seen_string_ids:
            raise ValueError(f"duplicate string block ID: {block.block_id}")
        if (
            block.row_start < 0
            or block.col_start < 0
            or block.row_stop > n_rows
            or block.col_stop > n_cols
            or block.row_stop <= block.row_start
            or block.col_stop <= block.col_start
        ):
            raise ValueError(f"block {block.block_id} lies outside the raster")
        rows = slice(block.row_start, block.row_stop)
        cols = slice(block.col_start, block.col_stop)
        if np.any(raster[rows, cols] != 0):
            raise ValueError(f"block {block.block_id} overlaps an earlier block")
        raster[rows, cols] = numeric_id
        mapping[numeric_id] = block.block_id
        seen_string_ids.add(block.block_id)
    return raster, mapping


def sample_blocks(
    score: np.ndarray, reference: np.ndarray, blocks: Sequence[Block]
) -> tuple[BlockSample, ...]:
    """Slice aligned score and binary-reference arrays onto complete blocks."""
    score_array = np.asarray(score, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if score_array.shape != reference_array.shape or score_array.ndim != 2:
        raise ValueError("score and reference must be aligned two-dimensional arrays")
    samples = []
    for block in blocks:
        if (
            block.row_start < 0
            or block.col_start < 0
            or block.row_stop > score_array.shape[0]
            or block.col_stop > score_array.shape[1]
            or block.row_stop <= block.row_start
            or block.col_stop <= block.col_start
        ):
            raise ValueError(f"block {block.block_id} lies outside the aligned arrays")
        rows = slice(block.row_start, block.row_stop)
        cols = slice(block.col_start, block.col_stop)
        samples.append(
            BlockSample(
                block=block,
                score=score_array[rows, cols].copy(),
                reference=reference_array[rows, cols].copy(),
            )
        )
    return tuple(samples)


def _intersects_test_halo(training: Block, test: Block, halo_pixels: int) -> bool:
    return (
        training.row_start < test.row_stop + halo_pixels
        and training.row_stop > test.row_start - halo_pixels
        and training.col_start < test.col_stop + halo_pixels
        and training.col_stop > test.col_start - halo_pixels
    )


def block_balanced_youden(samples: Sequence[BlockSample]) -> float | None:
    """Select the exact block-balanced Youden threshold from training blocks.

    Each positive observation receives total weight ``1 / n_positive_blocks``
    within its block; negatives are weighted analogously.  This is algebraically
    identical to averaging block TPR and FPR while permitting an exact scan of
    every unique training score.  The descending scan returns the highest
    threshold when multiple candidates attain the same maximum J.
    """
    positive: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    for sample in samples:
        score, reference = sample.paired_values()
        pos = score[reference == 1]
        neg = score[reference == 0]
        if pos.size:
            positive.append(pos)
        if neg.size:
            negative.append(neg)
    if not positive or not negative:
        return None

    score_parts: list[np.ndarray] = []
    positive_weight_parts: list[np.ndarray] = []
    negative_weight_parts: list[np.ndarray] = []
    n_positive_blocks = len(positive)
    n_negative_blocks = len(negative)
    for values in positive:
        score_parts.append(values)
        positive_weight_parts.append(np.full(values.size, 1.0 / (n_positive_blocks * values.size)))
        negative_weight_parts.append(np.zeros(values.size))
    for values in negative:
        score_parts.append(values)
        positive_weight_parts.append(np.zeros(values.size))
        negative_weight_parts.append(np.full(values.size, 1.0 / (n_negative_blocks * values.size)))

    scores = np.concatenate(score_parts)
    positive_weights = np.concatenate(positive_weight_parts)
    negative_weights = np.concatenate(negative_weight_parts)
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    positive_weights = positive_weights[order]
    negative_weights = negative_weights[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    group_stops = np.r_[group_starts[1:], scores.size]
    cumulative_positive = np.cumsum(positive_weights)
    cumulative_negative = np.cumsum(negative_weights)
    youden = cumulative_positive[group_stops - 1] - cumulative_negative[group_stops - 1]
    best = int(np.argmax(youden))
    return float(scores[group_starts[best]])


def spatial_cross_fit(samples: Sequence[BlockSample], *, halo_pixels: int) -> OutOfFoldResult:
    """Run leave-one-complete-block-out threshold calibration and prediction."""
    if halo_pixels < 0:
        raise ValueError("halo_pixels cannot be negative")
    pooled_scores: list[np.ndarray] = []
    pooled_reference: list[np.ndarray] = []
    pooled_predictions: list[np.ndarray] = []
    pooled_block_ids: list[np.ndarray] = []
    auc_scores: list[np.ndarray] = []
    auc_references: list[np.ndarray] = []
    auc_block_ids: list[np.ndarray] = []
    folds: list[FoldThreshold] = []
    skipped: list[str] = []

    for test_index, test in enumerate(samples):
        test_score, test_reference = test.paired_values()
        if test_score.size:
            auc_scores.append(test_score)
            auc_references.append(test_reference)
            auc_block_ids.append(np.full(test_score.size, test.block.block_id, dtype=object))
        training = [
            sample
            for index, sample in enumerate(samples)
            if index != test_index
            and not _intersects_test_halo(sample.block, test.block, halo_pixels)
        ]
        threshold = block_balanced_youden(training)
        if threshold is None or test_score.size == 0:
            skipped.append(test.block.block_id)
            continue
        predictions = test_score >= threshold
        pooled_scores.append(test_score)
        pooled_reference.append(test_reference)
        pooled_predictions.append(predictions)
        pooled_block_ids.append(np.full(test_score.size, test.block.block_id, dtype=object))
        folds.append(
            FoldThreshold(
                block_id=test.block.block_id,
                threshold=threshold,
                n_test=int(test_score.size),
                n_pos=int(np.count_nonzero(test_reference == 1)),
                n_neg=int(np.count_nonzero(test_reference == 0)),
                n_training_blocks=len(training),
            )
        )

    empty_float = np.asarray([], dtype=float)
    empty_int = np.asarray([], dtype=np.int8)
    empty_object = np.asarray([], dtype=object)
    return OutOfFoldResult(
        scores=np.concatenate(pooled_scores) if pooled_scores else empty_float,
        references=np.concatenate(pooled_reference) if pooled_reference else empty_int,
        predictions=(
            np.concatenate(pooled_predictions) if pooled_predictions else np.asarray([], dtype=bool)
        ),
        block_ids=np.concatenate(pooled_block_ids) if pooled_block_ids else empty_object,
        folds=tuple(folds),
        skipped_blocks=tuple(skipped),
        auc_scores=np.concatenate(auc_scores) if auc_scores else empty_float,
        auc_references=np.concatenate(auc_references) if auc_references else empty_int,
        auc_block_ids=np.concatenate(auc_block_ids) if auc_block_ids else empty_object,
    )


def rank_auc(scores: np.ndarray, references: np.ndarray) -> float:
    """Return tie-aware rank AUC, or NaN when either reference class is absent."""
    score = np.asarray(scores, dtype=float)
    reference_values = np.asarray(references)
    if score.ndim != 1 or reference_values.ndim != 1 or score.size != reference_values.size:
        raise ValueError("scores and references must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(score)):
        raise ValueError("scores must be finite")
    if not np.all(np.isin(reference_values, (0, 1))):
        raise ValueError("references must be binary")
    reference = reference_values.astype(np.int8)
    positive = reference == 1
    n_pos = int(np.count_nonzero(positive))
    n_neg = int(np.count_nonzero(~positive))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    u_statistic = float(np.sum(ranks[positive]) - n_pos * (n_pos + 1) / 2)
    return u_statistic / (n_pos * n_neg)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _auc_arrays(result: OutOfFoldResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rank-evaluation arrays, falling back for legacy constructed results."""
    supplied = (
        result.auc_scores is not None,
        result.auc_references is not None,
        result.auc_block_ids is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError("AUC score, reference, and block arrays must be supplied together")
    if all(supplied):
        return result.auc_scores, result.auc_references, result.auc_block_ids
    return result.scores, result.references, result.block_ids


def pooled_metrics(result: OutOfFoldResult) -> MetricSet:
    """Compute pooled rank metrics and successful-threshold OOF metrics.

    F1 is set to zero when its denominator is zero, matching the usual
    zero-division convention for a class that receives no positive prediction.
    Rank AUC uses every pairwise-finite block observation, including held-out
    blocks whose training data could not calibrate a threshold.
    """
    auc_scores, auc_references, _ = _auc_arrays(result)
    if auc_references.size == 0:
        raise ValueError("out-of-fold result has no rank-evaluated observations")
    reference = result.references.astype(bool)
    prediction = result.predictions.astype(bool)
    if reference.size != prediction.size:
        raise ValueError("thresholded references and predictions must be aligned")
    auc = rank_auc(auc_scores, auc_references)
    rank_reference = np.asarray(auc_references, dtype=np.int8)
    rank_n_pos = int(np.count_nonzero(rank_reference == 1))
    rank_n_neg = int(np.count_nonzero(rank_reference == 0))
    threshold_n_pos = int(np.count_nonzero(reference))
    threshold_n_neg = int(reference.size - threshold_n_pos)
    if reference.size == 0 or len(np.unique(reference)) < 2:
        return MetricSet(
            auc=auc,
            balanced_accuracy=float("nan"),
            positive_f1=float("nan"),
            negative_f1=float("nan"),
            macro_f1=float("nan"),
            tpr=float("nan"),
            fpr=float("nan"),
            prevalence=_safe_ratio(rank_n_pos, rank_n_pos + rank_n_neg),
            n_pos=rank_n_pos,
            n_neg=rank_n_neg,
            rank_n_pos=rank_n_pos,
            rank_n_neg=rank_n_neg,
            threshold_n_pos=threshold_n_pos,
            threshold_n_neg=threshold_n_neg,
        )
    tp = int(np.count_nonzero(prediction & reference))
    fp = int(np.count_nonzero(prediction & ~reference))
    tn = int(np.count_nonzero(~prediction & ~reference))
    fn = int(np.count_nonzero(~prediction & reference))
    tpr = _safe_ratio(tp, tp + fn)
    fpr = _safe_ratio(fp, fp + tn)
    positive_f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _safe_ratio(2 * tn, 2 * tn + fp + fn)
    return MetricSet(
        auc=auc,
        balanced_accuracy=0.5 * (tpr + (1.0 - fpr)),
        positive_f1=positive_f1,
        negative_f1=negative_f1,
        macro_f1=0.5 * (positive_f1 + negative_f1),
        tpr=tpr,
        fpr=fpr,
        prevalence=_safe_ratio(rank_n_pos, rank_n_pos + rank_n_neg),
        n_pos=rank_n_pos,
        n_neg=rank_n_neg,
        rank_n_pos=rank_n_pos,
        rank_n_neg=rank_n_neg,
        threshold_n_pos=threshold_n_pos,
        threshold_n_neg=threshold_n_neg,
    )


def _pairwise_auc_numerator(positive: np.ndarray, negative: np.ndarray) -> float:
    sorted_negative = np.sort(negative)
    lower = np.searchsorted(sorted_negative, positive, side="left")
    upper = np.searchsorted(sorted_negative, positive, side="right")
    return float(np.sum(lower + 0.5 * (upper - lower)))


def _governed_confidence_interval(
    metric: str,
    values: np.ndarray,
    *,
    scheduled_replicates: int,
) -> ConfidenceInterval:
    """Summarize bootstrap values under the preregistered finite-replicate gate."""
    if scheduled_replicates <= 0:
        raise ValueError("scheduled_replicates must be positive")
    values = np.asarray(values, dtype=float).ravel()
    if values.size != scheduled_replicates:
        raise ValueError("bootstrap values must match scheduled_replicates")
    finite = values[np.isfinite(values)]
    valid_replicates = int(finite.size)
    finite_fraction = valid_replicates / scheduled_replicates
    gate_eligible = valid_replicates >= ceil(FINITE_REPLICATE_FRACTION * scheduled_replicates)
    if gate_eligible:
        lower, upper = (float(value) for value in np.percentile(finite, (2.5, 97.5)))
        unavailable_reason = None
    else:
        lower = upper = float("nan")
        unavailable_reason = "fewer_than_95_percent_finite_replicates"
    return ConfidenceInterval(
        metric=metric,
        lower=lower,
        upper=upper,
        scheduled_replicates=scheduled_replicates,
        valid_replicates=valid_replicates,
        finite_fraction=finite_fraction,
        gate_eligible=gate_eligible,
        unavailable_reason=unavailable_reason,
    )


def block_bootstrap_intervals(
    result: OutOfFoldResult,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> tuple[ConfidenceInterval, ...]:
    """Bootstrap complete blocks for rank and successful-threshold metrics."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    auc_scores, auc_references, auc_result_block_ids = _auc_arrays(result)
    block_ids = np.asarray(sorted(set(auc_result_block_ids.tolist())), dtype=object)
    if block_ids.size == 0:
        raise ValueError("out-of-fold result has no evaluated blocks")
    n_blocks = block_ids.size
    n_pos = np.zeros(n_blocks)
    n_neg = np.zeros(n_blocks)
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for index, block_id in enumerate(block_ids):
        auc_use = auc_result_block_ids == block_id
        auc_reference = auc_references[auc_use].astype(bool)
        block_scores = auc_scores[auc_use]
        positives.append(block_scores[auc_reference])
        negatives.append(block_scores[~auc_reference])
        n_pos[index] = positives[-1].size
        n_neg[index] = negatives[-1].size

    auc_numerator = np.zeros((n_blocks, n_blocks), dtype=float)
    for pos_index, pos_values in enumerate(positives):
        if pos_values.size == 0:
            continue
        for neg_index, neg_values in enumerate(negatives):
            if neg_values.size:
                auc_numerator[pos_index, neg_index] = _pairwise_auc_numerator(
                    pos_values, neg_values
                )

    threshold_block_ids = np.asarray(sorted(set(result.block_ids.tolist())), dtype=object)
    n_threshold_blocks = threshold_block_ids.size
    tp = np.zeros(n_threshold_blocks)
    fp = np.zeros(n_threshold_blocks)
    tn = np.zeros(n_threshold_blocks)
    fn = np.zeros(n_threshold_blocks)
    for index, block_id in enumerate(threshold_block_ids):
        use = result.block_ids == block_id
        reference = result.references[use].astype(bool)
        prediction = result.predictions[use].astype(bool)
        tp[index] = np.count_nonzero(prediction & reference)
        fp[index] = np.count_nonzero(prediction & ~reference)
        tn[index] = np.count_nonzero(~prediction & ~reference)
        fn[index] = np.count_nonzero(~prediction & reference)

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=replicates)
    if np.array_equal(threshold_block_ids, block_ids):
        threshold_weights = weights
    elif n_threshold_blocks:
        threshold_weights = rng.multinomial(
            n_threshold_blocks,
            np.full(n_threshold_blocks, 1.0 / n_threshold_blocks),
            size=replicates,
        )
    else:
        threshold_weights = np.zeros((replicates, 0), dtype=int)
    boot_tp = threshold_weights @ tp
    boot_fp = threshold_weights @ fp
    boot_tn = threshold_weights @ tn
    boot_fn = threshold_weights @ fn
    boot_pos = weights @ n_pos
    boot_neg = weights @ n_neg
    auc_denominator = boot_pos * boot_neg
    auc_value = np.full(replicates, np.nan)
    valid_auc = auc_denominator > 0
    auc_value[valid_auc] = (
        np.einsum("bi,ij,bj->b", weights, auc_numerator, weights)[valid_auc]
        / auc_denominator[valid_auc]
    )
    tpr = np.divide(
        boot_tp,
        boot_tp + boot_fn,
        out=np.full(replicates, np.nan),
        where=(boot_tp + boot_fn) > 0,
    )
    fpr = np.divide(
        boot_fp,
        boot_fp + boot_tn,
        out=np.full(replicates, np.nan),
        where=(boot_fp + boot_tn) > 0,
    )
    positive_f1 = np.divide(
        2 * boot_tp,
        2 * boot_tp + boot_fp + boot_fn,
        out=np.full(replicates, np.nan),
        where=(2 * boot_tp + boot_fp + boot_fn) > 0,
    )
    negative_f1 = np.divide(
        2 * boot_tn,
        2 * boot_tn + boot_fp + boot_fn,
        out=np.full(replicates, np.nan),
        where=(2 * boot_tn + boot_fp + boot_fn) > 0,
    )
    distributions = {
        "auc": auc_value,
        "balanced_accuracy": 0.5 * (tpr + 1.0 - fpr),
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "macro_f1": 0.5 * (positive_f1 + negative_f1),
        "tpr": tpr,
        "fpr": fpr,
        "prevalence": np.divide(
            boot_pos,
            boot_pos + boot_neg,
            out=np.full(replicates, np.nan),
            where=(boot_pos + boot_neg) > 0,
        ),
    }
    return tuple(
        _governed_confidence_interval(
            metric,
            values,
            scheduled_replicates=replicates,
        )
        for metric, values in distributions.items()
    )


def permute_score_blocks(
    samples: Sequence[BlockSample], permutation: Sequence[int]
) -> tuple[BlockSample, ...]:
    """Pair each target reference block with one permuted score block."""
    if len(samples) != len(permutation) or sorted(permutation) != list(range(len(samples))):
        raise ValueError("permutation must contain every sample index exactly once")
    return tuple(
        BlockSample(block=target.block, score=samples[source].score, reference=target.reference)
        for target, source in zip(samples, permutation, strict=True)
    )


def _permutation_metric_values(
    samples: tuple[BlockSample, ...], halo_pixels: int, permutation: np.ndarray
) -> tuple[float | None, float | None]:
    """Evaluate one pre-generated permutation without changing replicate filtering."""
    permuted = permute_score_blocks(samples, permutation)
    oof = spatial_cross_fit(permuted, halo_pixels=halo_pixels)
    _, auc_references, _ = _auc_arrays(oof)
    if auc_references.size == 0 or len(np.unique(auc_references)) < 2:
        return None, None
    metrics = pooled_metrics(oof)
    auc = metrics.auc if np.isfinite(metrics.auc) else None
    balanced_accuracy = (
        metrics.balanced_accuracy
        if oof.references.size
        and len(np.unique(oof.references)) == 2
        and np.isfinite(metrics.balanced_accuracy)
        else None
    )
    return auc, balanced_accuracy


def _permutation_chunk_metrics(
    samples: tuple[BlockSample, ...], halo_pixels: int, permutations: np.ndarray
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Evaluate one ordered chunk of parent-generated permutation index arrays."""
    auc_values: list[float] = []
    balanced_accuracy_values: list[float] = []
    for permutation in permutations:
        auc, balanced_accuracy = _permutation_metric_values(samples, halo_pixels, permutation)
        if auc is not None:
            auc_values.append(auc)
        if balanced_accuracy is not None:
            balanced_accuracy_values.append(balanced_accuracy)
    return tuple(auc_values), tuple(balanced_accuracy_values)


def whole_block_permutation_test(
    samples: Sequence[BlockSample],
    *,
    halo_pixels: int,
    permutations: int = PERMUTATION_REPLICATES,
    seed: int = SEED,
    workers: int | None = None,
) -> PermutationResult:
    """Permute score blocks and rerun cross-fitted calibration.

    The default ``workers=None`` retains serial execution.  When more than one
    worker is requested, the parent creates the complete seeded permutation
    sequence before dispatching contiguous chunks for computation.  Results are
    collected in chunk order, preserving the serial calculation exactly.
    """
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if workers is not None:
        if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)):
            raise TypeError("workers must be a positive integer or None")
        if workers <= 0:
            raise ValueError("workers must be a positive integer or None")
    observed = pooled_metrics(spatial_cross_fit(samples, halo_pixels=halo_pixels))
    rng = np.random.default_rng(seed)
    sample_tuple = tuple(samples)
    permutation_indices = np.empty((permutations, len(sample_tuple)), dtype=np.intp)
    for index in range(permutations):
        permutation_indices[index] = rng.permutation(len(sample_tuple))

    auc_null: list[float] = []
    balanced_null: list[float] = []
    if workers is None or workers == 1 or permutations == 1:
        chunk_results = (
            _permutation_chunk_metrics(sample_tuple, halo_pixels, permutation_indices),
        )
    else:
        worker_count = min(int(workers), permutations)
        chunks = tuple(np.array_split(permutation_indices, worker_count))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            chunk_results = tuple(
                executor.map(
                    _permutation_chunk_metrics,
                    repeat(sample_tuple),
                    repeat(halo_pixels),
                    chunks,
                )
            )
    for auc_values, balanced_accuracy_values in chunk_results:
        auc_null.extend(auc_values)
        balanced_null.extend(balanced_accuracy_values)
    auc_exceed = np.count_nonzero(np.asarray(auc_null) >= observed.auc)
    balanced_exceed = np.count_nonzero(np.asarray(balanced_null) >= observed.balanced_accuracy)
    auc_p_value = (
        float((auc_exceed + 1) / (len(auc_null) + 1))
        if auc_null and np.isfinite(observed.auc)
        else None
    )
    balanced_p_value = (
        float((balanced_exceed + 1) / (len(balanced_null) + 1))
        if balanced_null and np.isfinite(observed.balanced_accuracy)
        else None
    )
    return PermutationResult(
        auc_p_value=auc_p_value,
        balanced_accuracy_p_value=balanced_p_value,
        valid_auc_permutations=len(auc_null),
        valid_balanced_accuracy_permutations=len(balanced_null),
    )


def bearing_block_counts(samples: Sequence[BlockSample]) -> tuple[int, int]:
    """Count complete blocks containing at least one usable positive or negative."""
    positive = 0
    negative = 0
    for sample in samples:
        _, reference = sample.paired_values()
        positive += int(np.any(reference == 1))
        negative += int(np.any(reference == 0))
    return positive, negative


def governance_status(positive_blocks: int, negative_blocks: int) -> str:
    """Apply the frozen independent-block governance thresholds symmetrically."""
    if positive_blocks < 0 or negative_blocks < 0:
        raise ValueError("block counts cannot be negative")
    if positive_blocks < 5 or negative_blocks < 5:
        return "counts_and_maps_only"
    if positive_blocks < 10 or negative_blocks < 10:
        return "exploratory_only"
    return "confirmatory_eligible"


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite_index = np.flatnonzero(np.isfinite(values))
    if finite_index.size == 0:
        return adjusted
    finite_values = values[finite_index]
    order = np.argsort(finite_values, kind="mergesort")
    ranked = finite_values[order]
    raw = ranked * ranked.size / np.arange(1, ranked.size + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    restored = np.empty_like(monotone)
    restored[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite_index] = restored
    return adjusted
