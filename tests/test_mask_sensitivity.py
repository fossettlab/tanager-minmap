"""Focused tests for deterministic mask-policy comparisons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_mask_policies import (  # noqa: E402
    compare_raster_pair,
    compare_validation_pair,
    run_comparison,
)

TRANSFORM = from_origin(100.0, 200.0, 30.0, 30.0)
CRS = "EPSG:32612"


def _write_raster(path: Path, values: np.ndarray, *, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=CRS,
        transform=TRANSFORM,
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)


def _validation_row(
    *,
    kind: str = "feature",
    layer: str = "al_oh_doublet",
    n_pos: int = 25,
    n_neg: int = 75,
    auc: float = 0.6,
    threshold: float = 0.1,
) -> dict[str, object]:
    return {
        "kind": kind,
        "layer": layer,
        "positive_classes": "3 4",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc": auc,
        "threshold": threshold,
    }


def _write_validation(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def test_continuous_comparison_honors_nodata_and_nan(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy" / "score.tif"
    corrected_path = tmp_path / "corrected" / "score.tif"
    legacy = np.array([[1.0, 2.0, np.nan], [4.0, -9999.0, 6.0]], dtype="float32")
    corrected = np.array([[1.5, np.nan, np.nan], [3.5, 5.0, 7.0]], dtype="float32")
    _write_raster(legacy_path, legacy, nodata=-9999.0)
    _write_raster(corrected_path, corrected, nodata=-9999.0)

    result = compare_raster_pair(legacy_path, corrected_path)

    assert result["comparison_type"] == "continuous"
    assert result["legacy_valid_count"] == 4
    assert result["corrected_valid_count"] == 4
    assert result["valid_overlap_count"] == 3
    assert result["newly_excluded_count"] == 1
    assert result["newly_included_count"] == 1
    np.testing.assert_allclose(result["mae"], 2.0 / 3.0)
    np.testing.assert_allclose(result["rmse"], np.sqrt(0.5))
    np.testing.assert_allclose(
        result["pearson_r"], np.corrcoef([1.0, 4.0, 6.0], [1.5, 3.5, 7.0])[0, 1]
    )


def test_categorical_comparison_uses_declared_nodata(tmp_path: Path) -> None:
    name = "site_scene_amd_agp.tif"
    legacy_path = tmp_path / "legacy" / name
    corrected_path = tmp_path / "corrected" / name
    legacy = np.array([[0, 1, -1], [2, 2, 3]], dtype="int16")
    corrected = np.array([[0, 2, -1], [2, -1, 3]], dtype="int16")
    _write_raster(legacy_path, legacy, nodata=-1)
    _write_raster(corrected_path, corrected, nodata=-1)

    result = compare_raster_pair(legacy_path, corrected_path)

    assert result["comparison_type"] == "categorical"
    assert result["legacy_valid_count"] == 5
    assert result["corrected_valid_count"] == 4
    assert result["valid_overlap_count"] == 4
    assert result["newly_excluded_count"] == 1
    assert result["categorical_agreement_count"] == 3
    assert result["categorical_agreement_fraction"] == pytest.approx(0.75)
    assert result["pearson_r"] is None


def test_sam_minus_one_remains_a_category_without_nodata(tmp_path: Path) -> None:
    name = "site_scene_sam_class.tif"
    legacy_path = tmp_path / "legacy" / name
    corrected_path = tmp_path / "corrected" / name
    legacy = np.array([[-1, 0], [1, -1]], dtype="int16")
    corrected = np.array([[-1, 1], [1, -1]], dtype="int16")
    _write_raster(legacy_path, legacy)
    _write_raster(corrected_path, corrected)

    result = compare_raster_pair(legacy_path, corrected_path)

    assert result["legacy_valid_count"] == 4
    assert result["corrected_valid_count"] == 4
    assert result["newly_excluded_count"] == 0
    assert result["categorical_agreement_fraction"] == pytest.approx(0.75)


def test_validation_deltas_and_low_positive_flag(tmp_path: Path) -> None:
    name = "validation_site_scene.csv"
    legacy_path = tmp_path / "legacy" / name
    corrected_path = tmp_path / "corrected" / name
    _write_validation(
        legacy_path,
        [_validation_row(n_pos=12, n_neg=88, auc=0.61, threshold=0.02)],
    )
    _write_validation(
        corrected_path,
        [_validation_row(n_pos=8, n_neg=72, auc=0.56, threshold=0.03)],
    )

    rows, inventory = compare_validation_pair(legacy_path, corrected_path)

    assert inventory["shared_row_count"] == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["auc_delta"] == pytest.approx(-0.05)
    assert row["threshold_delta"] == pytest.approx(0.01)
    assert row["n_pos_delta"] == -4
    assert row["n_neg_delta"] == -16
    assert row["sample_count_delta"] == -20
    assert row["n_pos_note"] == "legacy n_pos=12 and corrected n_pos=8 are both <20"


def test_run_comparison_is_deterministic(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    corrected_root = tmp_path / "corrected"
    output_dir = tmp_path / "outputs"
    report_path = tmp_path / "docs" / "report.md"

    score_name = "site_scene_score.tif"
    class_name = "site_scene_sam_class.tif"
    _write_raster(
        legacy_root / "maps" / score_name,
        np.array([[1.0, 2.0], [3.0, np.nan]], dtype="float32"),
    )
    _write_raster(
        corrected_root / "maps" / score_name,
        np.array([[1.0, np.nan], [4.0, np.nan]], dtype="float32"),
    )
    _write_raster(
        legacy_root / "maps" / class_name,
        np.array([[-1, 0], [1, 1]], dtype="int16"),
    )
    _write_raster(
        corrected_root / "maps" / class_name,
        np.array([[-1, 1], [1, 1]], dtype="int16"),
    )
    validation_name = "validation_site_scene.csv"
    _write_validation(
        legacy_root / "validation" / validation_name,
        [_validation_row(n_pos=10, n_neg=30, auc=0.7, threshold=0.2)],
    )
    _write_validation(
        corrected_root / "validation" / validation_name,
        [_validation_row(n_pos=9, n_neg=21, auc=0.65, threshold=0.25)],
    )

    run_comparison(legacy_root, corrected_root, output_dir, report_path)
    paths = [
        output_dir / "raster_comparison.csv",
        output_dir / "validation_comparison.csv",
        output_dir / "summary.json",
        report_path,
    ]
    first_run = {path: path.read_bytes() for path in paths}
    run_comparison(legacy_root, corrected_root, output_dir, report_path)

    assert {path: path.read_bytes() for path in paths} == first_run
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["rasters"]["inventory"]["shared_file_count"] == 2
    assert summary["validation"]["comparison_row_count"] == 1
    assert summary["comparison"]["materiality_threshold"] is None
    report = report_path.read_text(encoding="utf-8")
    assert "assigns no materiality threshold" in report
    assert "legacy n_pos=10 and corrected n_pos=9 are both <20" in report


def test_unaligned_rasters_raise(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy" / "score.tif"
    corrected_path = tmp_path / "corrected" / "score.tif"
    _write_raster(legacy_path, np.ones((2, 2), dtype="float32"))
    _write_raster(corrected_path, np.ones((3, 2), dtype="float32"))

    with pytest.raises(ValueError, match="unaligned raster pair"):
        compare_raster_pair(legacy_path, corrected_path)
