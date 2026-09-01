"""Preregistered scene-level native-versus-Sentinel-2 sensor ablation.

The analytical functions in this module are side-effect free.  The companion
``scripts/run_scene_ablation.py`` owns input/output and provenance recording.
Only spectral response convolution is applied to the degraded branches; the
native spatial grid and pixel values are otherwise unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from tanager_spec.srf import SpectralResponse

from .degrade import degrade_cube, degrade_endmembers, srf_band_stats
from .reference import ROCKWELL_EXCLUDED
from .spatial_validation import (
    PERMUTATION_REPLICATES,
    Block,
    BlockSample,
    OutOfFoldResult,
    benjamini_hochberg,
    block_balanced_youden,
    pooled_metrics,
)
from .spatial_validation import (
    rank_auc as binary_rank_auc,
)
from .speclib import Endmember
from .unmix import mtmf

RIDGE = 1e-2
MIN_COVERAGE = 0.5
BOOTSTRAP_REPLICATES = 10_000
SEED = 42
POSITIVE_CLASS = 3
NEGATIVE_CLASS = 4
MIN_CONFIRMATORY_BLOCKS = 10
FDR_ALPHA = 0.05
CONFIRMATORY_STATUS = "confirmatory"
EXPLORATORY_STATUS = "exploratory"
COUNTS_MAPS_ONLY_STATUS = "counts_maps_only"

METRIC_NAMES = (
    "auc",
    "balanced_accuracy",
    "macro_f1",
    "tpr",
    "fpr",
    "prevalence",
)


@dataclass(frozen=True)
class BlockFold:
    """One complete held-out block and its halo-excluded training support."""

    block_id: str
    test_mask: np.ndarray
    train_mask: np.ndarray


@dataclass(frozen=True)
class BlockDesign:
    """Frozen block geometry converted to masks on the scene grid."""

    name: str
    folds: tuple[BlockFold, ...]
    block_ids: np.ndarray


@dataclass(frozen=True)
class RobustMarginScale:
    """Training-only robust location and scale for the two mineral scores."""

    available: bool
    alunite_median: float = float("nan")
    alunite_iqr: float = float("nan")
    kaolinite_median: float = float("nan")
    kaolinite_iqr: float = float("nan")
    reason: str | None = None


@dataclass(frozen=True)
class SensorEvaluation:
    """Cross-fitted scores and predictions for one sensor branch."""

    scores: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    block_ids: np.ndarray
    thresholds: tuple[float, ...]
    fold_parameters: tuple[dict[str, Any], ...]
    metrics: dict[str, float]
    support_pixels: int
    positive_blocks: int
    negative_blocks: int


@dataclass(frozen=True)
class PairedEvaluation:
    """Native and degraded evaluations retained on identical held-out pixels."""

    native: SensorEvaluation
    degraded: SensorEvaluation
    unavailable_folds: tuple[dict[str, str], ...]
    support_pixels: int
    positive_blocks: int
    negative_blocks: int


@dataclass(frozen=True)
class PairedRandomizationResult:
    """Paired within-block sensor-label randomization result for an AUC delta."""

    observed_delta: float
    permuted_deltas: np.ndarray
    p_value: float
    exceedances: int
    randomizations: int
    support_pixels: int
    complete_blocks: int


@dataclass(frozen=True)
class SupportGovernance:
    """Inferential outputs permitted by the frozen M2 block-count tiers."""

    status: str
    effect_estimates: bool
    bootstrap_cis: bool
    permutation_inference: bool
    bh_adjustment: bool


def _canonical_geometry(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace("-", "")
    if text in {"l", "primary", "1l"}:
        return "L"
    if text in {"2l", "double", "secondary"}:
        return "2L"
    return str(value).strip()


def _boolean_series(values: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values
    normalized = values.astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not bool(valid.all()):
        raise ValueError(f"{column} must contain only boolean values")
    return normalized.isin({"true", "1"})


def _rename_manifest_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "scale": "geometry",
        "block_geometry": "geometry",
        "is_complete": "complete",
        "complete_block": "complete",
        "halo_width_pixels": "halo_pixels",
        "r_site_pixels": "halo_pixels",
        "row_end": "row_stop",
        "col_end": "col_stop",
        "column_start": "col_start",
        "column_stop": "col_stop",
    }
    renamed = frame.copy()
    for source, target in aliases.items():
        if target not in renamed.columns and source in renamed.columns:
            renamed = renamed.rename(columns={source: target})
    return renamed


def block_designs_from_frame(
    manifest: pd.DataFrame,
    shape: tuple[int, int],
    *,
    halo_pixels: int | None = None,
    manifest_contains_only_complete_blocks: bool = False,
) -> dict[str, BlockDesign]:
    """Convert an M2 block-manifest table to exact test/training masks.

    The accepted table is one row per block and must record ``geometry``,
    ``block_id``, half-open pixel bounds (``row_start``, ``row_stop``,
    ``col_start``, ``col_stop``), ``complete``, and ``halo_pixels``.  A small
    set of explicit aliases is accepted for compatibility with M2 outputs.
    No geometry, completeness decision, or halo is inferred.
    """
    frame = _rename_manifest_columns(manifest)
    if "complete" not in frame and manifest_contains_only_complete_blocks:
        frame["complete"] = True
    if "halo_pixels" not in frame and halo_pixels is not None:
        frame["halo_pixels"] = halo_pixels
    required = {
        "geometry",
        "block_id",
        "row_start",
        "row_stop",
        "col_start",
        "col_stop",
        "complete",
        "halo_pixels",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"block manifest missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("block manifest contains no blocks")

    ny, nx = shape
    frame = frame.copy()
    frame["geometry"] = frame["geometry"].map(_canonical_geometry)
    frame["complete"] = _boolean_series(frame["complete"], "complete")
    for column in ("row_start", "row_stop", "col_start", "col_stop", "halo_pixels"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain integer pixel indices")
        frame[column] = numeric.astype(int)

    designs: dict[str, BlockDesign] = {}
    for geometry, records in frame.groupby("geometry", sort=False):
        complete = records.loc[records["complete"]].copy()
        if complete.empty:
            raise ValueError(f"geometry {geometry!r} has no complete blocks")
        if complete["block_id"].astype(str).duplicated().any():
            raise ValueError(f"geometry {geometry!r} has duplicate complete block IDs")

        block_ids = np.full(shape, "", dtype=object)
        bounds_by_id: dict[str, tuple[int, int, int, int, int]] = {}
        for row in complete.itertuples(index=False):
            r0, r1 = int(row.row_start), int(row.row_stop)
            c0, c1 = int(row.col_start), int(row.col_stop)
            halo = int(row.halo_pixels)
            if not (0 <= r0 < r1 <= ny and 0 <= c0 < c1 <= nx):
                raise ValueError(f"block {row.block_id!r} bounds fall outside scene shape {shape}")
            if halo < 0:
                raise ValueError("halo_pixels must be non-negative")
            occupied = block_ids[r0:r1, c0:c1] != ""
            if occupied.any():
                raise ValueError(f"complete blocks overlap in geometry {geometry!r}")
            block_id = str(row.block_id)
            block_ids[r0:r1, c0:c1] = block_id
            bounds_by_id[block_id] = (r0, r1, c0, c1, halo)

        folds: list[BlockFold] = []
        for block_id, (r0, r1, c0, c1, halo) in bounds_by_id.items():
            test = block_ids == block_id
            train = np.zeros(shape, dtype=bool)
            for candidate_id, (tr0, tr1, tc0, tc1, _) in bounds_by_id.items():
                intersects_halo = (
                    tr0 < r1 + halo and tr1 > r0 - halo and tc0 < c1 + halo and tc1 > c0 - halo
                )
                if candidate_id != block_id and not intersects_halo:
                    train |= block_ids == candidate_id
            folds.append(BlockFold(block_id=block_id, test_mask=test, train_mask=train))
        designs[str(geometry)] = BlockDesign(str(geometry), tuple(folds), block_ids)
    return designs


def fit_robust_margin(
    alunite: np.ndarray,
    kaolinite: np.ndarray,
    train_mask: np.ndarray,
) -> RobustMarginScale:
    """Fit median/IQR parameters on finite training pixels only."""
    a = np.asarray(alunite, dtype=float)
    k = np.asarray(kaolinite, dtype=float)
    train = np.asarray(train_mask, dtype=bool)
    if a.shape != k.shape or a.shape != train.shape:
        raise ValueError("alunite, kaolinite, and training mask must share one shape")
    use = train & np.isfinite(a) & np.isfinite(k)
    if not use.any():
        return RobustMarginScale(False, reason="no_finite_training_pixels")
    a_train = a[use]
    k_train = k[use]
    a_q1, a_q3 = np.quantile(a_train, [0.25, 0.75])
    k_q1, k_q3 = np.quantile(k_train, [0.25, 0.75])
    a_iqr = float(a_q3 - a_q1)
    k_iqr = float(k_q3 - k_q1)
    if a_iqr == 0.0:
        return RobustMarginScale(False, reason="zero_alunite_iqr")
    if k_iqr == 0.0:
        return RobustMarginScale(False, reason="zero_kaolinite_iqr")
    return RobustMarginScale(
        True,
        alunite_median=float(np.median(a_train)),
        alunite_iqr=a_iqr,
        kaolinite_median=float(np.median(k_train)),
        kaolinite_iqr=k_iqr,
    )


def apply_robust_margin(
    alunite: np.ndarray,
    kaolinite: np.ndarray,
    scale: RobustMarginScale,
) -> np.ndarray:
    """Apply a previously fitted robust alunite-minus-kaolinite margin."""
    if not scale.available:
        raise ValueError(f"robust margin scale is unavailable: {scale.reason}")
    return (np.asarray(alunite, float) - scale.alunite_median) / scale.alunite_iqr - (
        np.asarray(kaolinite, float) - scale.kaolinite_median
    ) / scale.kaolinite_iqr


def block_balanced_youden_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    block_ids: np.ndarray,
) -> float | None:
    """Training-only block-balanced Youden threshold with highest-threshold ties."""
    score = np.asarray(scores, float).ravel()
    label = np.asarray(labels).ravel()
    blocks = np.asarray(block_ids).astype(str).ravel()
    use = np.isfinite(score) & np.isin(label, [POSITIVE_CLASS, NEGATIVE_CLASS])
    score, label, blocks = score[use], label[use], blocks[use]
    if (
        score.size == 0
        or not (label == POSITIVE_CLASS).any()
        or not (label == NEGATIVE_CLASS).any()
    ):
        return None

    samples: list[BlockSample] = []
    for index, block_id in enumerate(np.unique(blocks)):
        block_use = blocks == block_id
        block = Block(str(block_id), 0, index, 0, 1, index, index + 1)
        samples.append(
            BlockSample(
                block=block,
                score=score[block_use][None, :],
                reference=(label[block_use] == POSITIVE_CLASS).astype(float)[None, :],
            )
        )
    return block_balanced_youden(samples)


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC with average ranks for ties."""
    score = np.asarray(scores, float)
    label = np.asarray(labels)
    use = np.isfinite(score) & np.isin(label, [POSITIVE_CLASS, NEGATIVE_CLASS])
    score, label = score[use], label[use]
    positive = label == POSITIVE_CLASS
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    binary = positive.astype(np.int8)
    return float(binary_rank_auc(score, binary))


def classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Compute the preregistered pooled held-out metrics."""
    score = np.asarray(scores, float)
    label = np.asarray(labels)
    binary = (label == POSITIVE_CLASS).astype(np.int8)
    result = OutOfFoldResult(
        scores=score,
        references=binary,
        predictions=np.asarray(predictions, bool),
        block_ids=np.full(score.size, "pooled", dtype=object),
        folds=(),
        skipped_blocks=(),
    )
    metrics = pooled_metrics(result)
    return {
        "auc": metrics.auc,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_f1": metrics.macro_f1,
        "tpr": metrics.tpr,
        "fpr": metrics.fpr,
        "prevalence": metrics.prevalence,
    }


def _empty_evaluation() -> SensorEvaluation:
    metrics = {metric: float("nan") for metric in METRIC_NAMES}
    empty_float = np.asarray([], dtype=float)
    empty_int = np.asarray([], dtype=int)
    empty_str = np.asarray([], dtype=str)
    return SensorEvaluation(
        empty_float,
        empty_int,
        np.asarray([], dtype=bool),
        empty_str,
        (),
        (),
        metrics,
        0,
        0,
        0,
    )


def _assemble_evaluation(
    scores: list[np.ndarray],
    labels: list[np.ndarray],
    predictions: list[np.ndarray],
    blocks: list[np.ndarray],
    thresholds: list[float],
    fold_parameters: list[dict[str, Any]],
) -> SensorEvaluation:
    if not scores:
        return _empty_evaluation()
    score = np.concatenate(scores)
    label = np.concatenate(labels)
    prediction = np.concatenate(predictions)
    block = np.concatenate(blocks).astype(str)
    unique_blocks = np.unique(block)
    positive_blocks = sum(
        ((block == item) & (label == POSITIVE_CLASS)).any() for item in unique_blocks
    )
    negative_blocks = sum(
        ((block == item) & (label == NEGATIVE_CLASS)).any() for item in unique_blocks
    )
    return SensorEvaluation(
        scores=score,
        labels=label,
        predictions=prediction,
        block_ids=block,
        thresholds=tuple(thresholds),
        fold_parameters=tuple(fold_parameters),
        metrics=classification_metrics(score, label, prediction),
        support_pixels=int(score.size),
        positive_blocks=int(positive_blocks),
        negative_blocks=int(negative_blocks),
    )


def _class_support(reference: np.ndarray, *scores: np.ndarray) -> np.ndarray:
    support = np.isin(reference, [POSITIVE_CLASS, NEGATIVE_CLASS])
    for score in scores:
        support &= np.isfinite(score)
    return support


def _support_counts(
    support: np.ndarray,
    labels: np.ndarray,
    design: BlockDesign,
) -> tuple[int, int, int]:
    complete_support = support & (design.block_ids != "")
    positive_blocks = 0
    negative_blocks = 0
    for fold in design.folds:
        use = fold.test_mask & support
        positive_blocks += int(np.any(labels[use] == POSITIVE_CLASS))
        negative_blocks += int(np.any(labels[use] == NEGATIVE_CLASS))
    return int(np.count_nonzero(complete_support)), positive_blocks, negative_blocks


def evaluate_sensor_pair(
    native_alunite: np.ndarray,
    native_kaolinite: np.ndarray,
    degraded_alunite: np.ndarray,
    degraded_kaolinite: np.ndarray,
    reference: np.ndarray,
    design: BlockDesign,
) -> PairedEvaluation:
    """Cross-fit the standardized primary margin on paired sensor support."""
    arrays = [
        np.asarray(value, float)
        for value in (native_alunite, native_kaolinite, degraded_alunite, degraded_kaolinite)
    ]
    ref = np.asarray(reference)
    if any(value.shape != ref.shape for value in arrays) or design.block_ids.shape != ref.shape:
        raise ValueError("scores, reference, and block design must share one spatial shape")
    support = _class_support(ref, *arrays)
    support_counts = _support_counts(support, ref, design)

    native_parts: tuple[list, ...] = ([], [], [], [], [], [])
    degraded_parts: tuple[list, ...] = ([], [], [], [], [], [])
    unavailable: list[dict[str, str]] = []
    for fold in design.folds:
        train = fold.train_mask & support
        test = fold.test_mask & support
        if not test.any():
            unavailable.append({"block_id": fold.block_id, "reason": "no_paired_test_support"})
            continue
        native_scale = fit_robust_margin(arrays[0], arrays[1], train)
        degraded_scale = fit_robust_margin(arrays[2], arrays[3], train)
        if not native_scale.available or not degraded_scale.available:
            reasons = [
                scale.reason for scale in (native_scale, degraded_scale) if not scale.available
            ]
            reason = ";".join(dict.fromkeys(map(str, reasons)))
            unavailable.append({"block_id": fold.block_id, "reason": reason})
            continue
        native_margin = apply_robust_margin(arrays[0], arrays[1], native_scale)
        degraded_margin = apply_robust_margin(arrays[2], arrays[3], degraded_scale)
        train_blocks = design.block_ids[train]
        native_threshold = block_balanced_youden_threshold(
            native_margin[train], ref[train], train_blocks
        )
        degraded_threshold = block_balanced_youden_threshold(
            degraded_margin[train], ref[train], train_blocks
        )
        if native_threshold is None or degraded_threshold is None:
            unavailable.append(
                {"block_id": fold.block_id, "reason": "threshold_training_unavailable"}
            )
            continue
        labels = ref[test]
        block_values = np.full(labels.size, fold.block_id, dtype=object)
        native_score = native_margin[test]
        degraded_score = degraded_margin[test]
        native_params = {
            "block_id": fold.block_id,
            "alunite_median": native_scale.alunite_median,
            "alunite_iqr": native_scale.alunite_iqr,
            "kaolinite_median": native_scale.kaolinite_median,
            "kaolinite_iqr": native_scale.kaolinite_iqr,
        }
        degraded_params = {
            "block_id": fold.block_id,
            "alunite_median": degraded_scale.alunite_median,
            "alunite_iqr": degraded_scale.alunite_iqr,
            "kaolinite_median": degraded_scale.kaolinite_median,
            "kaolinite_iqr": degraded_scale.kaolinite_iqr,
        }
        for parts, score, threshold, params in (
            (native_parts, native_score, native_threshold, native_params),
            (degraded_parts, degraded_score, degraded_threshold, degraded_params),
        ):
            parts[0].append(score)
            parts[1].append(labels)
            parts[2].append(score >= threshold)
            parts[3].append(block_values)
            parts[4].append(threshold)
            parts[5].append(params)

    native = _assemble_evaluation(*native_parts)
    degraded = _assemble_evaluation(*degraded_parts)
    return PairedEvaluation(native, degraded, tuple(unavailable), *support_counts)


def evaluate_score_pair(
    native_score: np.ndarray,
    degraded_score: np.ndarray,
    reference: np.ndarray,
    design: BlockDesign,
    *,
    positive_classes: frozenset[int],
    excluded_classes: frozenset[int] = ROCKWELL_EXCLUDED,
) -> PairedEvaluation:
    """Cross-fit a secondary raw MTMF layer against mapped Rockwell zones."""
    native = np.asarray(native_score, float)
    degraded = np.asarray(degraded_score, float)
    ref_raw = np.asarray(reference)
    if native.shape != ref_raw.shape or degraded.shape != ref_raw.shape:
        raise ValueError("scores and reference must share one spatial shape")
    labels = np.where(np.isin(ref_raw, list(positive_classes)), POSITIVE_CLASS, NEGATIVE_CLASS)
    usable_reference = np.isfinite(ref_raw) & ~np.isin(ref_raw, list(excluded_classes))
    support = usable_reference & np.isfinite(native) & np.isfinite(degraded)
    support_counts = _support_counts(support, labels, design)

    native_parts: tuple[list, ...] = ([], [], [], [], [], [])
    degraded_parts: tuple[list, ...] = ([], [], [], [], [], [])
    unavailable: list[dict[str, str]] = []
    for fold in design.folds:
        train = fold.train_mask & support
        test = fold.test_mask & support
        if not test.any():
            unavailable.append({"block_id": fold.block_id, "reason": "no_paired_test_support"})
            continue
        train_blocks = design.block_ids[train]
        native_threshold = block_balanced_youden_threshold(
            native[train], labels[train], train_blocks
        )
        degraded_threshold = block_balanced_youden_threshold(
            degraded[train], labels[train], train_blocks
        )
        if native_threshold is None or degraded_threshold is None:
            unavailable.append(
                {"block_id": fold.block_id, "reason": "threshold_training_unavailable"}
            )
            continue
        test_labels = labels[test]
        block_values = np.full(test_labels.size, fold.block_id, dtype=object)
        for parts, score, threshold in (
            (native_parts, native[test], native_threshold),
            (degraded_parts, degraded[test], degraded_threshold),
        ):
            parts[0].append(score)
            parts[1].append(test_labels)
            parts[2].append(score >= threshold)
            parts[3].append(block_values)
            parts[4].append(threshold)
            parts[5].append({"block_id": fold.block_id})
    return PairedEvaluation(
        _assemble_evaluation(*native_parts),
        _assemble_evaluation(*degraded_parts),
        tuple(unavailable),
        *support_counts,
    )


def _block_auc_components(
    evaluation: SensorEvaluation, blocks: np.ndarray
) -> tuple[np.ndarray, ...]:
    n = blocks.size
    pos_counts = np.zeros(n, dtype=float)
    neg_counts = np.zeros(n, dtype=float)
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for index, block in enumerate(blocks):
        use = evaluation.block_ids == block
        positives.append(evaluation.scores[use & (evaluation.labels == POSITIVE_CLASS)])
        negatives.append(np.sort(evaluation.scores[use & (evaluation.labels == NEGATIVE_CLASS)]))
        pos_counts[index] = positives[-1].size
        neg_counts[index] = negatives[-1].size
    u_matrix = np.zeros((n, n), dtype=float)
    for i, pos in enumerate(positives):
        for j, neg in enumerate(negatives):
            if pos.size and neg.size:
                u_matrix[i, j] = _pairwise_auc_numerator(pos, neg)
    return pos_counts, neg_counts, u_matrix


def _pairwise_auc_numerator(positive: np.ndarray, negative_sorted: np.ndarray) -> float:
    lower = np.searchsorted(negative_sorted, positive, side="left").sum()
    upper = np.searchsorted(negative_sorted, positive, side="right").sum()
    return float(lower + 0.5 * (upper - lower))


def _block_confusion(evaluation: SensorEvaluation, blocks: np.ndarray) -> np.ndarray:
    result = np.zeros((blocks.size, 4), dtype=float)
    for index, block in enumerate(blocks):
        use = evaluation.block_ids == block
        label = evaluation.labels[use]
        pred = evaluation.predictions[use]
        positive = label == POSITIVE_CLASS
        negative = label == NEGATIVE_CLASS
        result[index] = [
            np.sum(pred & positive),
            np.sum(~pred & positive),
            np.sum(pred & negative),
            np.sum(~pred & negative),
        ]
    return result


def _bootstrap_metrics(
    counts: np.ndarray,
    evaluation: SensorEvaluation,
    blocks: np.ndarray,
) -> dict[str, np.ndarray]:
    pos_counts, neg_counts, u_matrix = _block_auc_components(evaluation, blocks)
    denominator = (counts @ pos_counts) * (counts @ neg_counts)
    auc = np.full(counts.shape[0], np.nan)
    batch_size = 256
    for start in range(0, counts.shape[0], batch_size):
        stop = min(start + batch_size, counts.shape[0])
        batch = counts[start:stop]
        numerator = np.sum((batch @ u_matrix) * batch, axis=1)
        valid = denominator[start:stop] > 0
        batch_auc = np.full(stop - start, np.nan)
        batch_auc[valid] = numerator[valid] / denominator[start:stop][valid]
        auc[start:stop] = batch_auc

    confusion = counts @ _block_confusion(evaluation, blocks)
    tp, fn, fp, tn = confusion.T

    def divide(numerator: np.ndarray, divisor: np.ndarray) -> np.ndarray:
        out = np.full(numerator.shape, np.nan, dtype=float)
        return np.divide(numerator, divisor, out=out, where=divisor != 0)

    tpr = divide(tp, tp + fn)
    fpr = divide(fp, fp + tn)
    f1_pos = divide(2 * tp, 2 * tp + fp + fn)
    f1_neg = divide(2 * tn, 2 * tn + fp + fn)
    return {
        "auc": auc,
        "balanced_accuracy": (tpr + (1.0 - fpr)) / 2.0,
        "macro_f1": (f1_pos + f1_neg) / 2.0,
        "tpr": tpr,
        "fpr": fpr,
        "prevalence": divide(tp + fn, tp + fn + fp + tn),
    }


def paired_block_bootstrap(
    evaluation: PairedEvaluation,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Resample complete blocks once per replicate for both sensor branches."""
    native_blocks = np.unique(evaluation.native.block_ids)
    degraded_blocks = np.unique(evaluation.degraded.block_ids)
    if not np.array_equal(native_blocks, degraded_blocks):
        raise ValueError("native and degraded evaluations do not share identical blocks")
    if native_blocks.size == 0:
        raise ValueError("no paired held-out blocks are available for bootstrap")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        native_blocks.size,
        np.full(native_blocks.size, 1.0 / native_blocks.size),
        size=n_boot,
    )
    native = _bootstrap_metrics(counts, evaluation.native, native_blocks)
    degraded = _bootstrap_metrics(counts, evaluation.degraded, native_blocks)
    output: dict[str, np.ndarray] = {"replicate": np.arange(n_boot, dtype=int)}
    for metric in METRIC_NAMES:
        output[f"native_{metric}"] = native[metric]
        output[f"degraded_{metric}"] = degraded[metric]
        output[f"delta_{metric}"] = native[metric] - degraded[metric]
    return pd.DataFrame(output)


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Finite-only percentile interval for a bootstrap distribution."""
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(finite, [tail, 1.0 - tail])
    return float(lower), float(upper)


def governed_metric_summary(
    value: float,
    bootstrap_values: np.ndarray | None,
    governance: SupportGovernance,
) -> tuple[float, float, float]:
    """Mask an effect and interval according to frozen support governance."""
    if not governance.effect_estimates:
        return float("nan"), float("nan"), float("nan")
    if bootstrap_values is None or not governance.bootstrap_cis:
        return float(value), float("nan"), float("nan")
    lower, upper = percentile_interval(bootstrap_values)
    return float(value), lower, upper


def _randomized_auc_numerator(
    choices: np.ndarray,
    pairwise_numerators: np.ndarray,
) -> np.ndarray:
    result = np.zeros(choices.shape[0], dtype=float)
    indicators = (choices == 0, choices == 1)
    for positive_source in (0, 1):
        for negative_source in (0, 1):
            left = indicators[positive_source].astype(float)
            right = indicators[negative_source].astype(float)
            result += np.sum(
                (left @ pairwise_numerators[positive_source, negative_source]) * right,
                axis=1,
            )
    return result


def paired_sensor_auc_randomization(
    native_score: np.ndarray,
    degraded_score: np.ndarray,
    reference: np.ndarray,
    design: BlockDesign,
    *,
    positive_classes: frozenset[int],
    excluded_classes: frozenset[int] = ROCKWELL_EXCLUDED,
    randomizations: int = PERMUTATION_REPLICATES,
    seed: int = SEED,
) -> PairedRandomizationResult:
    """Test a paired native-minus-degraded AUC delta by within-block swaps.

    Native/degraded sensor labels are independently swapped once per complete
    block. Labels and paired finite support stay fixed in every randomization.
    """
    native = np.asarray(native_score, dtype=float)
    degraded = np.asarray(degraded_score, dtype=float)
    ref = np.asarray(reference)
    if native.shape != degraded.shape or native.shape != ref.shape:
        raise ValueError("native, degraded, and reference arrays must share one shape")
    if design.block_ids.shape != ref.shape:
        raise ValueError("block design and score arrays must share one spatial shape")
    if not positive_classes:
        raise ValueError("positive_classes cannot be empty")
    if randomizations <= 0:
        raise ValueError("randomizations must be positive")

    usable_reference = np.isfinite(ref) & ~np.isin(ref, list(excluded_classes))
    support = (
        (design.block_ids != "") & usable_reference & np.isfinite(native) & np.isfinite(degraded)
    )
    labels = np.where(np.isin(ref, list(positive_classes)), POSITIVE_CLASS, NEGATIVE_CLASS)
    if not np.any(support & (labels == POSITIVE_CLASS)) or not np.any(
        support & (labels == NEGATIVE_CLASS)
    ):
        raise ValueError("paired randomization support must contain both reference classes")

    blocks = np.asarray([fold.block_id for fold in design.folds], dtype=object)
    pairwise = np.zeros((2, 2, blocks.size, blocks.size), dtype=float)
    score_sources = (native, degraded)
    for positive_source, positive_score in enumerate(score_sources):
        positive_by_block = [
            positive_score[support & (design.block_ids == block) & (labels == POSITIVE_CLASS)]
            for block in blocks
        ]
        for negative_source, negative_score in enumerate(score_sources):
            negative_by_block = [
                np.sort(
                    negative_score[
                        support & (design.block_ids == block) & (labels == NEGATIVE_CLASS)
                    ]
                )
                for block in blocks
            ]
            for i, positive in enumerate(positive_by_block):
                for j, negative in enumerate(negative_by_block):
                    if positive.size and negative.size:
                        pairwise[positive_source, negative_source, i, j] = _pairwise_auc_numerator(
                            positive, negative
                        )

    n_positive = int(np.count_nonzero(support & (labels == POSITIVE_CLASS)))
    n_negative = int(np.count_nonzero(support & (labels == NEGATIVE_CLASS)))
    denominator = float(n_positive * n_negative)
    observed_delta = float((pairwise[0, 0].sum() - pairwise[1, 1].sum()) / denominator)

    rng = np.random.default_rng(seed)
    choices = rng.integers(0, 2, size=(randomizations, blocks.size), dtype=np.int8)
    permuted = np.empty(randomizations, dtype=float)
    batch_size = 256
    for start in range(0, randomizations, batch_size):
        stop = min(start + batch_size, randomizations)
        native_choices = choices[start:stop]
        degraded_choices = 1 - native_choices
        native_numerator = _randomized_auc_numerator(native_choices, pairwise)
        degraded_numerator = _randomized_auc_numerator(degraded_choices, pairwise)
        permuted[start:stop] = (native_numerator - degraded_numerator) / denominator

    exceedances = int(np.count_nonzero(np.abs(permuted) >= abs(observed_delta)))
    p_value = float((1 + exceedances) / (randomizations + 1))
    return PairedRandomizationResult(
        observed_delta=observed_delta,
        permuted_deltas=permuted,
        p_value=p_value,
        exceedances=exceedances,
        randomizations=randomizations,
        support_pixels=int(np.count_nonzero(support)),
        complete_blocks=int(blocks.size),
    )


def benjamini_hochberg_by_family(
    p_values: np.ndarray,
    families: np.ndarray,
) -> np.ndarray:
    """Apply BH independently within each named comparison family."""
    values = np.asarray(p_values, dtype=float)
    labels = np.asarray(families)
    if values.shape != labels.shape:
        raise ValueError("p_values and families must share one shape")
    adjusted = np.full(values.shape, np.nan)
    for family in dict.fromkeys(labels.ravel().tolist()):
        use = labels == family
        adjusted[use] = benjamini_hochberg(values[use])
    return adjusted


def confirmatory_bh_by_family(
    p_values: np.ndarray,
    families: np.ndarray,
    statuses: np.ndarray,
) -> np.ndarray:
    """Apply separate BH corrections to confirmatory endpoints only."""
    values = np.asarray(p_values, dtype=float)
    labels = np.asarray(families)
    support_status = np.asarray(statuses)
    if values.shape != labels.shape or values.shape != support_status.shape:
        raise ValueError("p_values, families, and statuses must share one shape")
    adjusted = np.full(values.shape, np.nan)
    eligible = (support_status == CONFIRMATORY_STATUS) & np.isfinite(values)
    if eligible.any():
        adjusted[eligible] = benjamini_hochberg_by_family(values[eligible], labels[eligible])
    return adjusted


def support_governance(positive_blocks: int, negative_blocks: int) -> SupportGovernance:
    """Return the frozen M2 permissions for the limiting complete-block class."""
    limiting_blocks = min(positive_blocks, negative_blocks)
    if limiting_blocks >= MIN_CONFIRMATORY_BLOCKS:
        return SupportGovernance(
            status=CONFIRMATORY_STATUS,
            effect_estimates=True,
            bootstrap_cis=True,
            permutation_inference=True,
            bh_adjustment=True,
        )
    if limiting_blocks >= 5:
        return SupportGovernance(
            status=EXPLORATORY_STATUS,
            effect_estimates=True,
            bootstrap_cis=True,
            permutation_inference=False,
            bh_adjustment=False,
        )
    return SupportGovernance(
        status=COUNTS_MAPS_ONLY_STATUS,
        effect_estimates=False,
        bootstrap_cis=False,
        permutation_inference=False,
        bh_adjustment=False,
    )


def inference_status(positive_blocks: int, negative_blocks: int) -> str:
    """M2 support classification inherited by the M3 evaluation."""
    return support_governance(positive_blocks, negative_blocks).status


def _degraded_endmembers_for_mtmf(
    endmembers: dict[str, Endmember],
    wavelengths: np.ndarray,
    srf: SpectralResponse,
    min_coverage: float,
) -> dict[str, Endmember]:
    degraded = degrade_endmembers(endmembers, wavelengths, srf, min_coverage=min_coverage)
    centers, _ = srf_band_stats(srf)
    return {
        mineral: Endmember(
            mineral=mineral,
            sample=endmembers[mineral].sample,
            spectrometer=endmembers[mineral].spectrometer,
            wavelengths_nm=centers,
            reflectance=reflectance,
        )
        for mineral, reflectance in degraded.items()
    }


def compute_sensor_mtmf(
    cube: xr.DataArray,
    wavelengths: np.ndarray,
    endmembers: dict[str, Endmember],
    degraded_srfs: dict[str, SpectralResponse],
    *,
    ridge: float = RIDGE,
    min_coverage: float = MIN_COVERAGE,
) -> dict[str, xr.Dataset]:
    """Run native MTMF and SRF-only degraded MTMF in each sensor band space."""
    outputs = {"native": mtmf(cube, endmembers, ridge=ridge)}
    for sensor, srf in degraded_srfs.items():
        degraded_cube = degrade_cube(
            cube,
            wavelengths,
            srf,
            min_coverage=min_coverage,
        )
        degraded_endmembers = _degraded_endmembers_for_mtmf(
            endmembers,
            wavelengths,
            srf,
            min_coverage,
        )
        outputs[sensor] = mtmf(degraded_cube, degraded_endmembers, ridge=ridge)
    return outputs


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CONFIRMATORY_STATUS",
    "COUNTS_MAPS_ONLY_STATUS",
    "EXPLORATORY_STATUS",
    "FDR_ALPHA",
    "METRIC_NAMES",
    "MIN_CONFIRMATORY_BLOCKS",
    "MIN_COVERAGE",
    "NEGATIVE_CLASS",
    "PERMUTATION_REPLICATES",
    "POSITIVE_CLASS",
    "RIDGE",
    "SEED",
    "BlockDesign",
    "BlockFold",
    "PairedEvaluation",
    "PairedRandomizationResult",
    "RobustMarginScale",
    "SensorEvaluation",
    "SupportGovernance",
    "apply_robust_margin",
    "benjamini_hochberg",
    "benjamini_hochberg_by_family",
    "block_balanced_youden_threshold",
    "block_designs_from_frame",
    "classification_metrics",
    "confirmatory_bh_by_family",
    "compute_sensor_mtmf",
    "evaluate_score_pair",
    "evaluate_sensor_pair",
    "fit_robust_margin",
    "governed_metric_summary",
    "inference_status",
    "paired_block_bootstrap",
    "paired_sensor_auc_randomization",
    "percentile_interval",
    "rank_auc",
    "support_governance",
]
