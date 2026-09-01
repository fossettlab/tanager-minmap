"""Compare legacy fill-only outputs with corrected mask-policy outputs.

Raster validity follows each GeoTIFF's declared nodata mask plus non-finite
values. Continuous metrics are computed only where both rasters are valid;
categorical rasters use exact agreement on the same overlap. Validation deltas
are corrected minus legacy and retain the sample counts reported in the source
CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEGACY_ROOT = ROOT / "data" / "processed" / "mask_sensitivity" / "legacy_fill_only"
DEFAULT_CORRECTED_ROOT = ROOT / "data" / "intermediate"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "mask_sensitivity"
DEFAULT_REPORT_PATH = ROOT / "docs" / "mask_sensitivity_report.md"

LOW_POSITIVE_COUNT = 20
CATEGORICAL_SUFFIXES = ("_amd_agp.tif", "_sam_class.tif")
VALIDATION_REQUIRED_COLUMNS = {
    "kind",
    "layer",
    "positive_classes",
    "n_pos",
    "n_neg",
    "auc",
    "threshold",
}

RASTER_COLUMNS = [
    "raster",
    "comparison_type",
    "height",
    "width",
    "legacy_dtype",
    "corrected_dtype",
    "legacy_nodata",
    "corrected_nodata",
    "legacy_valid_count",
    "corrected_valid_count",
    "valid_overlap_count",
    "newly_excluded_count",
    "newly_included_count",
    "legacy_valid_retained_fraction",
    "valid_mask_jaccard",
    "pearson_r",
    "mae",
    "rmse",
    "categorical_agreement_count",
    "categorical_agreement_fraction",
]

VALIDATION_COLUMNS = [
    "validation_file",
    "kind",
    "layer",
    "legacy_positive_classes",
    "corrected_positive_classes",
    "positive_classes_match",
    "legacy_auc",
    "corrected_auc",
    "auc_delta",
    "legacy_threshold",
    "corrected_threshold",
    "threshold_delta",
    "legacy_n_pos",
    "corrected_n_pos",
    "n_pos_delta",
    "legacy_n_neg",
    "corrected_n_neg",
    "n_neg_delta",
    "legacy_sample_count",
    "corrected_sample_count",
    "sample_count_delta",
    "n_pos_note",
]


@dataclass(frozen=True)
class RasterBand:
    """A single raster band and the validity implied by its stored metadata."""

    values: np.ndarray
    valid: np.ndarray
    dtype: np.dtype
    nodata: float | int | None
    crs: Any
    transform: Affine


def _read_raster_band(path: Path) -> RasterBand:
    """Read one band without replacing nodata or NaN values."""
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"expected one band in {path}, found {src.count}")
        band = src.read(1, masked=True)
        values = np.asarray(band.data)
        valid = ~np.ma.getmaskarray(band) & np.isfinite(values)
        return RasterBand(
            values=values,
            valid=valid,
            dtype=np.dtype(src.dtypes[0]),
            nodata=src.nodata,
            crs=src.crs,
            transform=src.transform,
        )


def _check_same_grid(name: str, legacy: RasterBand, corrected: RasterBand) -> None:
    """Reject unaligned pairs instead of silently resampling either policy."""
    mismatches = []
    if legacy.values.shape != corrected.values.shape:
        mismatches.append(f"shape {legacy.values.shape} != {corrected.values.shape}")
    if legacy.crs != corrected.crs:
        mismatches.append(f"CRS {legacy.crs} != {corrected.crs}")
    if legacy.transform != corrected.transform:
        mismatches.append(f"transform {legacy.transform} != {corrected.transform}")
    if mismatches:
        raise ValueError(f"unaligned raster pair {name}: {'; '.join(mismatches)}")


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Pearson correlation, or ``None`` when it is undefined."""
    if x.size < 2:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    if denominator == 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def _is_categorical(name: str, legacy_dtype: np.dtype, corrected_dtype: np.dtype) -> bool:
    """Identify the current class products and other integer class rasters."""
    integer_pair = np.issubdtype(legacy_dtype, np.integer) and np.issubdtype(
        corrected_dtype, np.integer
    )
    return name.endswith(CATEGORICAL_SUFFIXES) or integer_pair


def compare_raster_pair(legacy_path: Path, corrected_path: Path) -> dict[str, Any]:
    """Compare one same-name, same-grid raster pair."""
    if legacy_path.name != corrected_path.name:
        raise ValueError(f"raster names differ: {legacy_path.name} != {corrected_path.name}")

    legacy = _read_raster_band(legacy_path)
    corrected = _read_raster_band(corrected_path)
    _check_same_grid(legacy_path.name, legacy, corrected)

    overlap = legacy.valid & corrected.valid
    union = legacy.valid | corrected.valid
    newly_excluded = legacy.valid & ~corrected.valid
    newly_included = ~legacy.valid & corrected.valid
    legacy_valid_count = int(legacy.valid.sum())
    corrected_valid_count = int(corrected.valid.sum())
    overlap_count = int(overlap.sum())
    union_count = int(union.sum())
    categorical = _is_categorical(legacy_path.name, legacy.dtype, corrected.dtype)

    row: dict[str, Any] = {
        "raster": legacy_path.name,
        "comparison_type": "categorical" if categorical else "continuous",
        "height": int(legacy.values.shape[0]),
        "width": int(legacy.values.shape[1]),
        "legacy_dtype": legacy.dtype.name,
        "corrected_dtype": corrected.dtype.name,
        "legacy_nodata": legacy.nodata,
        "corrected_nodata": corrected.nodata,
        "legacy_valid_count": legacy_valid_count,
        "corrected_valid_count": corrected_valid_count,
        "valid_overlap_count": overlap_count,
        "newly_excluded_count": int(newly_excluded.sum()),
        "newly_included_count": int(newly_included.sum()),
        "legacy_valid_retained_fraction": (
            overlap_count / legacy_valid_count if legacy_valid_count else None
        ),
        "valid_mask_jaccard": overlap_count / union_count if union_count else None,
        "pearson_r": None,
        "mae": None,
        "rmse": None,
        "categorical_agreement_count": None,
        "categorical_agreement_fraction": None,
    }

    if not overlap_count:
        return row

    legacy_values = legacy.values[overlap]
    corrected_values = corrected.values[overlap]
    if categorical:
        agreement_count = int(np.equal(legacy_values, corrected_values).sum())
        row["categorical_agreement_count"] = agreement_count
        row["categorical_agreement_fraction"] = agreement_count / overlap_count
    else:
        legacy_float = legacy_values.astype(np.float64, copy=False)
        corrected_float = corrected_values.astype(np.float64, copy=False)
        differences = corrected_float - legacy_float
        row["pearson_r"] = _pearson_r(legacy_float, corrected_float)
        row["mae"] = float(np.mean(np.abs(differences)))
        row["rmse"] = float(np.sqrt(np.mean(np.square(differences))))
    return row


def _named_files(directory: Path, pattern: str) -> dict[str, Path]:
    """Return direct child files keyed by name."""
    if not directory.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {directory}")
    return {path.name: path for path in sorted(directory.glob(pattern)) if path.is_file()}


def compare_raster_directories(
    legacy_dir: Path, corrected_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare every same-name GeoTIFF in two map directories."""
    legacy_files = _named_files(legacy_dir, "*.tif")
    corrected_files = _named_files(corrected_dir, "*.tif")
    shared = sorted(legacy_files.keys() & corrected_files.keys())
    if not shared:
        raise ValueError(f"no same-name GeoTIFFs in {legacy_dir} and {corrected_dir}")
    rows = [compare_raster_pair(legacy_files[name], corrected_files[name]) for name in shared]
    inventory = {
        "legacy_file_count": len(legacy_files),
        "corrected_file_count": len(corrected_files),
        "shared_file_count": len(shared),
        "legacy_only_files": sorted(legacy_files.keys() - corrected_files.keys()),
        "corrected_only_files": sorted(corrected_files.keys() - legacy_files.keys()),
    }
    return rows, inventory


def _read_validation(path: Path) -> pd.DataFrame:
    """Read and validate one source validation CSV without dropping rows."""
    frame = pd.read_csv(path, dtype={"kind": str, "layer": str, "positive_classes": str})
    missing = sorted(VALIDATION_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if frame[["kind", "layer", "positive_classes"]].isna().any().any():
        raise ValueError(f"{path} contains a missing comparison key or positive_classes value")
    duplicates = frame.duplicated(["kind", "layer"], keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, ["kind", "layer"]].to_dict("records")
        raise ValueError(f"{path} contains duplicate comparison keys: {keys}")

    for column in ("n_pos", "n_neg", "auc", "threshold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path} contains non-finite {column} values")
    for column in ("n_pos", "n_neg"):
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all() or (values < 0).any():
            raise ValueError(f"{path} contains invalid {column} counts")
        frame[column] = frame[column].astype("int64")
    return frame.set_index(["kind", "layer"])


def _low_positive_note(legacy_n_pos: int, corrected_n_pos: int) -> str:
    """Describe source rows below the user-specified positive-count flag."""
    legacy_low = legacy_n_pos < LOW_POSITIVE_COUNT
    corrected_low = corrected_n_pos < LOW_POSITIVE_COUNT
    if legacy_low and corrected_low:
        return (
            f"legacy n_pos={legacy_n_pos} and corrected n_pos={corrected_n_pos} "
            f"are both <{LOW_POSITIVE_COUNT}"
        )
    if legacy_low:
        return f"legacy n_pos={legacy_n_pos} is <{LOW_POSITIVE_COUNT}"
    if corrected_low:
        return f"corrected n_pos={corrected_n_pos} is <{LOW_POSITIVE_COUNT}"
    return ""


def compare_validation_pair(
    legacy_path: Path, corrected_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare same-key validation rows from one same-name CSV pair."""
    if legacy_path.name != corrected_path.name:
        raise ValueError(f"validation names differ: {legacy_path.name} != {corrected_path.name}")
    legacy = _read_validation(legacy_path)
    corrected = _read_validation(corrected_path)
    legacy_keys = set(legacy.index)
    corrected_keys = set(corrected.index)
    shared_keys = sorted(legacy_keys & corrected_keys)
    if not shared_keys:
        raise ValueError(f"no same-key validation rows in {legacy_path} and {corrected_path}")

    rows = []
    for kind, layer in shared_keys:
        old = legacy.loc[(kind, layer)]
        new = corrected.loc[(kind, layer)]
        legacy_n_pos = int(old["n_pos"])
        corrected_n_pos = int(new["n_pos"])
        legacy_n_neg = int(old["n_neg"])
        corrected_n_neg = int(new["n_neg"])
        legacy_samples = legacy_n_pos + legacy_n_neg
        corrected_samples = corrected_n_pos + corrected_n_neg
        old_classes = str(old["positive_classes"])
        new_classes = str(new["positive_classes"])
        rows.append(
            {
                "validation_file": legacy_path.name,
                "kind": str(kind),
                "layer": str(layer),
                "legacy_positive_classes": old_classes,
                "corrected_positive_classes": new_classes,
                "positive_classes_match": old_classes == new_classes,
                "legacy_auc": float(old["auc"]),
                "corrected_auc": float(new["auc"]),
                "auc_delta": float(new["auc"] - old["auc"]),
                "legacy_threshold": float(old["threshold"]),
                "corrected_threshold": float(new["threshold"]),
                "threshold_delta": float(new["threshold"] - old["threshold"]),
                "legacy_n_pos": legacy_n_pos,
                "corrected_n_pos": corrected_n_pos,
                "n_pos_delta": corrected_n_pos - legacy_n_pos,
                "legacy_n_neg": legacy_n_neg,
                "corrected_n_neg": corrected_n_neg,
                "n_neg_delta": corrected_n_neg - legacy_n_neg,
                "legacy_sample_count": legacy_samples,
                "corrected_sample_count": corrected_samples,
                "sample_count_delta": corrected_samples - legacy_samples,
                "n_pos_note": _low_positive_note(legacy_n_pos, corrected_n_pos),
            }
        )

    inventory = {
        "legacy_row_count": len(legacy),
        "corrected_row_count": len(corrected),
        "shared_row_count": len(shared_keys),
        "legacy_only_rows": [list(key) for key in sorted(legacy_keys - corrected_keys)],
        "corrected_only_rows": [list(key) for key in sorted(corrected_keys - legacy_keys)],
    }
    return rows, inventory


def compare_validation_directories(
    legacy_dir: Path, corrected_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare all same-name validation CSVs in two directories."""
    legacy_files = _named_files(legacy_dir, "*.csv")
    corrected_files = _named_files(corrected_dir, "*.csv")
    shared = sorted(legacy_files.keys() & corrected_files.keys())
    if not shared:
        raise ValueError(f"no same-name validation CSVs in {legacy_dir} and {corrected_dir}")

    rows: list[dict[str, Any]] = []
    row_inventories: dict[str, Any] = {}
    for name in shared:
        pair_rows, pair_inventory = compare_validation_pair(
            legacy_files[name], corrected_files[name]
        )
        rows.extend(pair_rows)
        row_inventories[name] = pair_inventory
    inventory = {
        "legacy_file_count": len(legacy_files),
        "corrected_file_count": len(corrected_files),
        "shared_file_count": len(shared),
        "legacy_only_files": sorted(legacy_files.keys() - corrected_files.keys()),
        "corrected_only_files": sorted(corrected_files.keys() - legacy_files.keys()),
        "files": row_inventories,
    }
    return rows, inventory


def _csv_value(value: Any) -> Any:
    """Format values deterministically while leaving undefined metrics blank."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else format(float(value), ".12g")
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write a stable-column, stable-float CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _extreme(
    rows: list[dict[str, Any]], field: str, *, largest_absolute: bool = False
) -> dict[str, Any] | None:
    """Return one deterministic observed extreme for a populated metric."""
    populated = [row for row in rows if row.get(field) is not None]
    if not populated:
        return None
    if largest_absolute:
        ordered = sorted(
            populated,
            key=lambda row: (-abs(float(row[field])), str(row.get("raster", "")), str(row)),
        )
    else:
        ordered = sorted(
            populated,
            key=lambda row: (-float(row[field]), str(row.get("raster", "")), str(row)),
        )
    return ordered[0]


def _portable_path(path: Path) -> str:
    """Use repository-relative paths when possible so outputs are portable."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_summary(
    raster_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    raster_inventory: dict[str, Any],
    validation_inventory: dict[str, Any],
    legacy_root: Path,
    corrected_root: Path,
) -> dict[str, Any]:
    """Summarize observed comparisons without assigning materiality."""
    continuous = [row for row in raster_rows if row["comparison_type"] == "continuous"]
    categorical = [row for row in raster_rows if row["comparison_type"] == "categorical"]
    low_positive = [row for row in validation_rows if row["n_pos_note"]]
    positive_class_changes = [row for row in validation_rows if not row["positive_classes_match"]]

    return {
        "comparison": {
            "legacy_root": _portable_path(legacy_root),
            "corrected_root": _portable_path(corrected_root),
            "delta_direction": "corrected_minus_legacy",
            "validity": "declared raster nodata mask plus finite values",
            "materiality_threshold": None,
            "n_pos_flag": f"n_pos < {LOW_POSITIVE_COUNT}",
        },
        "rasters": {
            "inventory": raster_inventory,
            "continuous_count": len(continuous),
            "categorical_count": len(categorical),
            "largest_newly_excluded": _extreme(raster_rows, "newly_excluded_count"),
            "lowest_continuous_pearson": (
                sorted(
                    (row for row in continuous if row["pearson_r"] is not None),
                    key=lambda row: (float(row["pearson_r"]), row["raster"]),
                )[0]
                if any(row["pearson_r"] is not None for row in continuous)
                else None
            ),
            "largest_continuous_rmse": _extreme(continuous, "rmse"),
            "lowest_categorical_agreement": (
                sorted(
                    (
                        row
                        for row in categorical
                        if row["categorical_agreement_fraction"] is not None
                    ),
                    key=lambda row: (float(row["categorical_agreement_fraction"]), row["raster"]),
                )[0]
                if any(row["categorical_agreement_fraction"] is not None for row in categorical)
                else None
            ),
        },
        "validation": {
            "inventory": validation_inventory,
            "comparison_row_count": len(validation_rows),
            "largest_absolute_auc_delta": _extreme(
                validation_rows, "auc_delta", largest_absolute=True
            ),
            "largest_absolute_threshold_delta": _extreme(
                validation_rows, "threshold_delta", largest_absolute=True
            ),
            "largest_absolute_sample_count_delta": _extreme(
                validation_rows, "sample_count_delta", largest_absolute=True
            ),
            "n_pos_below_20": low_positive,
            "positive_class_changes": positive_class_changes,
        },
    }


def _json_ready(value: Any) -> Any:
    """Convert NumPy and non-finite values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple deterministic Markdown table."""
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return "\n".join(lines)


def _format_metric(value: Any, digits: int = 6, *, signed: bool = False) -> str:
    """Format a numeric result for the report."""
    if value is None or not np.isfinite(value):
        return "undefined"
    return f"{float(value):+.{digits}g}" if signed else f"{float(value):.{digits}g}"


def _mask_pattern_table(raster_rows: list[dict[str, Any]]) -> str:
    """Group repeated per-layer validity patterns without collapsing metrics."""
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in raster_rows:
        site = row["raster"].split("_", maxsplit=1)[0]
        key = (
            site,
            row["comparison_type"],
            row["legacy_valid_count"],
            row["corrected_valid_count"],
            row["newly_excluded_count"],
            row["newly_included_count"],
            row["valid_overlap_count"],
        )
        groups[key].append(row["raster"])
    table_rows = []
    for key, names in sorted(groups.items()):
        site, comparison_type, old_count, new_count, excluded, included, overlap = key
        table_rows.append(
            [
                str(site),
                str(comparison_type),
                str(len(names)),
                f"{old_count:,}",
                f"{new_count:,}",
                f"{excluded:,}",
                f"{included:,}",
                f"{overlap:,}",
            ]
        )
    return _markdown_table(
        [
            "Site",
            "Type",
            "Rasters",
            "Legacy valid",
            "Corrected valid",
            "Newly excluded",
            "Newly included",
            "Valid overlap",
        ],
        table_rows,
    )


def render_report(
    raster_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    """Render the comparison report directly from result rows and summary."""
    raster_inventory = summary["rasters"]["inventory"]
    validation_inventory = summary["validation"]["inventory"]
    continuous = [row for row in raster_rows if row["comparison_type"] == "continuous"]
    categorical = [row for row in raster_rows if row["comparison_type"] == "categorical"]

    lowest_correlations = sorted(
        (row for row in continuous if row["pearson_r"] is not None),
        key=lambda row: (float(row["pearson_r"]), row["raster"]),
    )[:5]
    largest_rmse = sorted(
        (row for row in continuous if row["rmse"] is not None),
        key=lambda row: (-float(row["rmse"]), row["raster"]),
    )[:5]
    low_positive = [row for row in validation_rows if row["n_pos_note"]]
    identical_continuous = [row for row in continuous if row["mae"] == 0.0 and row["rmse"] == 0.0]
    site_patterns: dict[str, set[tuple[int, int, int, int]]] = defaultdict(set)
    for row in continuous:
        site = row["raster"].split("_", maxsplit=1)[0]
        site_patterns[site].add(
            (
                row["legacy_valid_count"],
                row["corrected_valid_count"],
                row["newly_excluded_count"],
                row["newly_included_count"],
            )
        )
    highlight_lines = []
    for site, patterns in sorted(site_patterns.items()):
        if len(patterns) != 1:
            continue
        old_count, new_count, excluded, included = next(iter(patterns))
        excluded_fraction = excluded / old_count if old_count else 0.0
        highlight_lines.append(
            f"- {site.capitalize()}: continuous-map validity decreased from {old_count:,} to "
            f"{new_count:,} cells; {excluded:,} previously valid cells were excluded "
            f"({excluded_fraction:.2%}) and {included:,} were newly included."
        )
    highlight_lines.append(
        f"- {len(identical_continuous)} of {len(continuous)} continuous rasters were exactly "
        "unchanged on their valid overlap (Pearson r=1, MAE=0, RMSE=0)."
    )
    auc_extreme = summary["validation"]["largest_absolute_auc_delta"]
    threshold_extreme = summary["validation"]["largest_absolute_threshold_delta"]
    if auc_extreme is not None:
        formatted_auc = _format_metric(auc_extreme["auc_delta"], digits=5, signed=True)
        auc_note = (
            f" The row is flagged: {auc_extreme['n_pos_note']}."
            if auc_extreme["n_pos_note"]
            else ""
        )
        highlight_lines.append(
            f"- The largest absolute AUC delta was {formatted_auc} "
            f"for {auc_extreme['kind']} `{auc_extreme['layer']}` in "
            f"`{auc_extreme['validation_file']}`.{auc_note}"
        )
    if threshold_extreme is not None:
        highlight_lines.append(
            "- The largest absolute threshold delta was "
            f"{_format_metric(threshold_extreme['threshold_delta'], digits=6, signed=True)} for "
            f"{threshold_extreme['kind']} `{threshold_extreme['layer']}` in "
            f"`{threshold_extreme['validation_file']}`."
        )

    lines = [
        "# Mask-sensitivity comparison",
        "",
        "This report compares the legacy fill-only snapshot with the corrected mask-policy "
        "outputs. It assigns no materiality threshold; rankings below identify only the largest "
        "observed changes.",
        "",
        "## Comparison rules",
        "",
        "A pixel is valid when it is not masked by the GeoTIFF's declared nodata semantics and "
        "its stored value is finite. Continuous Pearson correlation, mean absolute error (MAE), "
        "and root mean squared error (RMSE) use only pixels valid under both policies. "
        "Categorical agreement is exact equality on that same valid overlap. Validation deltas "
        "are corrected minus legacy.",
        "",
        "The SAM class rasters declare no nodata value. Their `-1` values therefore remain a "
        "stored category in this comparison; the rasters alone cannot distinguish unclassified "
        "pixels from pixels whose source spectra were invalid.",
        "",
        "Validation rows with `n_pos < 20` are flagged descriptively, as requested; the flag is "
        "not a significance or materiality rule.",
        "",
        "## Input inventory",
        "",
        _markdown_table(
            ["Artifact", "Legacy", "Corrected", "Same-name compared"],
            [
                [
                    "GeoTIFFs",
                    str(raster_inventory["legacy_file_count"]),
                    str(raster_inventory["corrected_file_count"]),
                    str(raster_inventory["shared_file_count"]),
                ],
                [
                    "Validation CSVs",
                    str(validation_inventory["legacy_file_count"]),
                    str(validation_inventory["corrected_file_count"]),
                    str(validation_inventory["shared_file_count"]),
                ],
            ],
        ),
        "",
        f"Legacy-only rasters: {raster_inventory['legacy_only_files'] or 'none'}. Corrected-only "
        f"rasters: {raster_inventory['corrected_only_files'] or 'none'}.",
        "",
        "## Observed highlights",
        "",
        *highlight_lines,
        "",
        "## Raster validity changes",
        "",
        _mask_pattern_table(raster_rows),
        "",
        "Complete per-raster values are in "
        "[`raster_comparison.csv`](../data/processed/mask_sensitivity/raster_comparison.csv).",
        "",
        "## Continuous-map agreement",
        "",
        "Lowest observed Pearson correlations:",
        "",
        _markdown_table(
            ["Raster", "Pearson r", "MAE", "RMSE", "Overlap"],
            [
                [
                    f"`{row['raster']}`",
                    _format_metric(row["pearson_r"]),
                    _format_metric(row["mae"]),
                    _format_metric(row["rmse"]),
                    f"{row['valid_overlap_count']:,}",
                ]
                for row in lowest_correlations
            ],
        ),
        "",
        "Largest observed RMSE values (products have different native scales):",
        "",
        _markdown_table(
            ["Raster", "RMSE", "MAE", "Pearson r", "Overlap"],
            [
                [
                    f"`{row['raster']}`",
                    _format_metric(row["rmse"]),
                    _format_metric(row["mae"]),
                    _format_metric(row["pearson_r"]),
                    f"{row['valid_overlap_count']:,}",
                ]
                for row in largest_rmse
            ],
        ),
        "",
        "## Categorical-map agreement",
        "",
        _markdown_table(
            ["Raster", "Agreement", "Matching cells", "Overlap", "Newly excluded"],
            [
                [
                    f"`{row['raster']}`",
                    _format_metric(row["categorical_agreement_fraction"]),
                    f"{row['categorical_agreement_count']:,}",
                    f"{row['valid_overlap_count']:,}",
                    f"{row['newly_excluded_count']:,}",
                ]
                for row in sorted(categorical, key=lambda item: item["raster"])
            ],
        ),
        "",
        "## Validation deltas",
        "",
        "All same-key validation rows are shown. Sample count is `n_pos + n_neg` from each source "
        "CSV.",
        "",
        _markdown_table(
            [
                "File",
                "Kind",
                "Layer",
                "AUC delta",
                "Threshold delta",
                "n_pos delta",
                "n_neg delta",
                "Sample delta",
            ],
            [
                [
                    f"`{row['validation_file']}`",
                    str(row["kind"]),
                    str(row["layer"]),
                    _format_metric(row["auc_delta"], digits=5, signed=True),
                    _format_metric(row["threshold_delta"], digits=6, signed=True),
                    f"{row['n_pos_delta']:+d}",
                    f"{row['n_neg_delta']:+d}",
                    f"{row['sample_count_delta']:+d}",
                ]
                for row in validation_rows
            ],
        ),
        "",
        "Complete source and corrected values are in "
        "[`validation_comparison.csv`](../data/processed/mask_sensitivity/validation_comparison.csv).",
        "",
        "### Positive groups below 20 samples",
        "",
    ]

    if low_positive:
        lines.append(
            _markdown_table(
                ["File", "Kind", "Layer", "Flag"],
                [
                    [
                        f"`{row['validation_file']}`",
                        str(row["kind"]),
                        str(row["layer"]),
                        str(row["n_pos_note"]),
                    ]
                    for row in low_positive
                ],
            )
        )
    else:
        lines.append("No compared validation row has `n_pos < 20` under either policy.")
    lines.extend(
        [
            "",
            "The machine-readable summary is "
            "[`summary.json`](../data/processed/mask_sensitivity/summary.json).",
            "",
        ]
    )
    return "\n".join(lines)


def run_comparison(
    legacy_root: Path = DEFAULT_LEGACY_ROOT,
    corrected_root: Path = DEFAULT_CORRECTED_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Run all comparisons and write CSV, JSON, and Markdown outputs."""
    raster_rows, raster_inventory = compare_raster_directories(
        legacy_root / "maps", corrected_root / "maps"
    )
    validation_rows, validation_inventory = compare_validation_directories(
        legacy_root / "validation", corrected_root / "validation"
    )
    summary = build_summary(
        raster_rows,
        validation_rows,
        raster_inventory,
        validation_inventory,
        legacy_root,
        corrected_root,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "raster_comparison.csv", raster_rows, RASTER_COLUMNS)
    _write_csv(output_dir / "validation_comparison.csv", validation_rows, VALIDATION_COLUMNS)
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(raster_rows, validation_rows, summary), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Parse paths and run the deterministic comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)
    summary = run_comparison(
        legacy_root=args.legacy_root,
        corrected_root=args.corrected_root,
        output_dir=args.output_dir,
        report_path=args.report_path,
    )
    print(
        f"Compared {summary['rasters']['inventory']['shared_file_count']} rasters and "
        f"{summary['validation']['comparison_row_count']} validation rows."
    )


if __name__ == "__main__":
    main()
