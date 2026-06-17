"""Tests for zone-agreement validation on synthetic reference/score grids."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tanager_rocks.validate import analysis_domain, discriminate, validate_scores


def _da(values: np.ndarray) -> xr.DataArray:
    ny, nx = values.shape
    return xr.DataArray(values, dims=("y", "x"), coords={"y": np.arange(ny), "x": np.arange(nx)})


def test_analysis_domain_drops_excluded_classes():
    ref = _da(np.array([[0, 3], [48, 5]], dtype="float64"))  # 0, 48 excluded
    dom = analysis_domain(ref)
    assert dom.tolist() == [[False, True], [False, True]]


def test_discriminate_separates_zone():
    rng = np.random.default_rng(0)
    ref = np.full((20, 20), 5, dtype="float64")  # background = class 5
    ref[:, :10] = 3  # left half = positive class 3
    score = rng.normal(0.1, 0.02, size=(20, 20))
    score[:, :10] += 0.5  # class-3 pixels score higher
    res = discriminate(_da(score), _da(ref), frozenset({3}), layer="alunite")
    assert res is not None
    assert res.n_pos == 200 and res.n_neg == 200
    assert res.auc > 0.95  # near-perfect separation
    assert res.median_pos > res.median_neg
    assert res.median_neg < res.threshold < res.median_pos
    assert res.tpr > 0.9 and res.fpr < 0.1


def test_discriminate_none_when_positive_class_absent():
    ref = _da(np.full((5, 5), 5, dtype="float64"))
    score = _da(np.ones((5, 5)))
    assert discriminate(score, ref, frozenset({3}), layer="alunite") is None


def test_validate_scores_skips_unmapped_and_absent():
    ref = np.full((10, 10), 5, dtype="float64")
    ref[:, :5] = 3
    scores = xr.Dataset(
        {
            "alunite": _da(np.where(np.arange(100).reshape(10, 10) % 10 < 5, 1.0, 0.0)),
            "gypsum": _da(np.ones((10, 10))),  # not in mapping -> skipped
        }
    )
    mapping = {"alunite": frozenset({3}), "jarosite": frozenset({8})}  # jarosite absent
    out = validate_scores(scores, _da(ref), mapping)
    assert set(out) == {"alunite"}  # gypsum unmapped, jarosite-class absent
