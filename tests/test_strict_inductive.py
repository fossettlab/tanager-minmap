"""Synthetic tests for strict-inductive MTMF covariance sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine
from rasterio.transform import from_origin

import tanager_rocks.strict_inductive as strict
from tanager_rocks.speclib import Endmember
from tanager_rocks.unmix import fit_mtmf_background


def _cube() -> tuple[xr.DataArray, dict[str, Endmember]]:
    rng = np.random.default_rng(12)
    wavelengths = np.array([1000.0, 1100.0, 1200.0, 1300.0])
    values = rng.normal(size=(4, 6, 6))
    cube = xr.DataArray(
        values,
        dims=("band", "y", "x"),
        coords={"band": wavelengths, "y": np.arange(6), "x": np.arange(6)},
    )
    endmembers = {
        "first": Endmember(
            "first", "first.txt", "ASD", wavelengths, np.array([0.2, 0.4, 0.1, 0.5])
        ),
        "second": Endmember(
            "second", "second.txt", "ASD", wavelengths, np.array([0.5, 0.1, 0.3, 0.2])
        ),
    }
    return cube, endmembers


def _block_ids() -> np.ndarray:
    values = np.zeros((6, 6), dtype=np.uint32)
    values[2:4, 2:4] = 1
    return values


def test_integer_nodata_raster_is_cast_before_nan_fill(tmp_path: Path):
    path = tmp_path / "reference_uint8.tif"
    transform = from_origin(100.0, 220.0, 30.0, 30.0)
    values = np.array([[1, 255], [2, 3]], dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint8",
        crs="EPSG:32611",
        transform=transform,
        nodata=255,
    ) as dataset:
        dataset.write(values, 1)

    observed = strict._read_raster_values(
        path,
        strict.GridSpec(shape=(2, 2), crs="EPSG:32611", transform=transform),
    )

    assert observed.dtype == np.float64
    np.testing.assert_array_equal(observed[[0, 1, 1], [0, 0, 1]], [1.0, 2.0, 3.0])
    assert np.isnan(observed[0, 1])


def test_held_block_and_halo_cannot_influence_fitted_background():
    cube, endmembers = _cube()
    blocks = _block_ids()
    exclusion = strict.held_block_halo_mask(blocks, 1, halo_pixels=1)
    first = fit_mtmf_background(cube, endmembers, fit_mask=~exclusion)

    perturbed = cube.copy(deep=True)
    perturbed.values[:, exclusion] += 10_000.0
    second = fit_mtmf_background(perturbed, endmembers, fit_mask=~exclusion)

    np.testing.assert_array_equal(first.valid_bands, second.valid_bands)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.covariance_inverse, second.covariance_inverse)
    assert first.sample_count == second.sample_count


def test_halo_perturbation_does_not_change_held_scores():
    cube, endmembers = _cube()
    blocks = _block_ids()
    first, _ = strict.strict_fold_scores(cube, endmembers, blocks, 1, halo_pixels=1)
    exclusion = strict.held_block_halo_mask(blocks, 1, halo_pixels=1)
    halo_only = exclusion & (blocks != 1)
    perturbed = cube.copy(deep=True)
    perturbed.values[:, halo_only] -= 50_000.0
    second, _ = strict.strict_fold_scores(perturbed, endmembers, blocks, 1, halo_pixels=1)

    for variable in first.data_vars:
        np.testing.assert_array_equal(first[variable].values, second[variable].values)


def test_strict_fold_scores_only_held_block_and_fits_once(monkeypatch):
    cube, endmembers = _cube()
    blocks = _block_ids()
    calls = 0
    original = strict.fit_mtmf_background

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(strict, "fit_mtmf_background", counted)
    scores, support = strict.strict_fold_scores(cube, endmembers, blocks, 1, halo_pixels=1)

    assert calls == 1
    assert set(scores.data_vars) == {
        "first_mf",
        "first_infeas",
        "second_mf",
        "second_infeas",
    }
    held = blocks == 1
    for variable in scores.data_vars:
        assert np.isfinite(scores[variable].values[held]).all()
        assert np.isnan(scores[variable].values[~held]).all()
    assert support["held_geometric_pixels"] == 4
    assert support["excluded_geometric_pixels"] == 16


def _write_block_raster(
    path: Path,
    values: np.ndarray,
    transform: Affine,
) -> str:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="uint32",
        crs="EPSG:32611",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(values, 1)
    return strict.sha256_file(path)


def _manifest_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "root"
    protocol = root / strict.M2_PROTOCOL_RELATIVE_PATH
    protocol.parent.mkdir(parents=True)
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    artifact = root / "data" / "processed" / "spatial_validation"
    artifact.mkdir(parents=True)
    transform = from_origin(100.0, 220.0, 30.0, 30.0)
    sites: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for site_id, scene_id in strict.ANCHOR_SCENES.items():
        l_values = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ],
            dtype=np.uint32,
        )
        two_l_values = np.ones((4, 4), dtype=np.uint32)
        l_name = f"block_ids_{site_id}_L.tif"
        two_l_name = f"block_ids_{site_id}_2L.tif"
        l_hash = _write_block_raster(artifact / l_name, l_values, transform)
        two_l_hash = _write_block_raster(artifact / two_l_name, two_l_values, transform)
        l_names = {
            str(value): f"r{(value - 1) // 2:04d}_c{(value - 1) % 2:04d}" for value in range(1, 5)
        }
        scales = {
            "L": {
                "scale": "L",
                "site_id": site_id,
                "anchor_scene_id": scene_id,
                "block_raster": l_name,
                "block_raster_sha256": l_hash,
                "complete_block_ids": [1, 2, 3, 4],
                "complete_blocks": 4,
                "numeric_to_string_block_ids": l_names,
                "block_side_pixels": 2,
                "halo_pixels": 1,
            },
            "2L": {
                "scale": "2L",
                "site_id": site_id,
                "anchor_scene_id": scene_id,
                "block_raster": two_l_name,
                "block_raster_sha256": two_l_hash,
                "complete_block_ids": [1],
                "complete_blocks": 1,
                "numeric_to_string_block_ids": {"1": "r0000_c0000"},
                "block_side_pixels": 4,
                "halo_pixels": 1,
            },
        }
        sites[site_id] = {
            "scene_id": scene_id,
            "primary_scale": "L",
            "block_raster": l_name,
            "complete_block_ids": [1, 2, 3, 4],
            "grid": {
                "shape": [4, 4],
                "crs": "EPSG:32611",
                "transform": list(transform)[:6],
            },
            "scales": scales,
        }
        for scale, scale_values in (("L", l_values), ("2L", two_l_values)):
            ids = [1, 2, 3, 4] if scale == "L" else [1]
            side = 2 if scale == "L" else 4
            names = l_names if scale == "L" else {"1": "r0000_c0000"}
            for numeric_id in ids:
                found_rows, found_cols = np.nonzero(scale_values == numeric_id)
                rows.append(
                    {
                        "site": site_id,
                        "scene_id": scene_id,
                        "scale": scale,
                        "block_id": names[str(numeric_id)],
                        "numeric_block_id": numeric_id,
                        "complete": True,
                        "halo_pixels": 1,
                        "row_start": int(found_rows.min()),
                        "row_stop": int(found_rows.min()) + side,
                        "col_start": int(found_cols.min()),
                        "col_stop": int(found_cols.min()) + side,
                        "crs": "EPSG:32611",
                    }
                )
    manifest = {
        "schema_version": "1.0",
        "manifest_type": "spatial_validation_complete_blocks",
        "protocol": {
            "path": strict.M2_PROTOCOL_RELATIVE_PATH,
            "sha256": strict.sha256_file(protocol),
            "parameters": strict.PROTOCOL_PARAMETERS,
            "protocol_compliant": True,
        },
        "sites": sites,
        "blocks": rows,
    }
    manifest_path = artifact / "block_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = strict.sha256_file(manifest_path)
    summary_path = artifact / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "path": strict.M2_PROTOCOL_RELATIVE_PATH,
                    "sha256": strict.sha256_file(protocol),
                    "protocol_compliant": True,
                    "parameters": strict.PROTOCOL_PARAMETERS,
                },
                "block_manifest": {
                    "path": manifest_path.name,
                    "sha256": manifest_hash,
                },
                "block_manifest_sha256": manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    return root, manifest_path, summary_path


def test_block_handoff_requires_exact_protocol_hash_grids_and_both_scales(tmp_path):
    root, manifest_path, summary_path = _manifest_fixture(tmp_path)
    handoff = strict.validate_block_handoff(
        manifest_path,
        root=root,
        summary_path=summary_path,
    )
    assert set(handoff.sites) == set(strict.ANCHOR_SCENES)
    assert all(set(site.scales) == {"L", "2L"} for site in handoff.sites.values())
    assert all(len(site.scales["L"].blocks) == 4 for site in handoff.sites.values())

    first_site = next(iter(strict.ANCHOR_SCENES))
    raster_path = manifest_path.parent / f"block_ids_{first_site}_L.tif"
    with raster_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(strict.StrictInductiveError, match="SHA-256 mismatch"):
        strict.validate_block_handoff(
            manifest_path,
            root=root,
            summary_path=summary_path,
        )


def test_swapped_summary_and_manifest_are_rejected(tmp_path):
    root, manifest_path, summary_path = _manifest_fixture(tmp_path)
    swapped_manifest_path = manifest_path.with_name("swapped_block_manifest.json")
    swapped_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    swapped_payload["swapped_fixture"] = True
    swapped_manifest_path.write_text(json.dumps(swapped_payload), encoding="utf-8")
    swapped_hash = strict.sha256_file(swapped_manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["block_manifest"] = {
        "path": swapped_manifest_path.name,
        "sha256": swapped_hash,
    }
    summary["block_manifest_sha256"] = swapped_hash
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        strict.StrictInductiveError,
        match="summary top-level block-manifest SHA-256 mismatch",
    ):
        strict.validate_block_handoff(
            manifest_path,
            root=root,
            summary_path=summary_path,
        )


def test_nondefault_manifest_protocol_parameters_are_rejected(tmp_path):
    root, manifest_path, summary_path = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"]["parameters"]["bootstrap_replicates"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        strict.StrictInductiveError,
        match="manifest protocol parameters mismatch",
    ):
        strict.validate_block_handoff(
            manifest_path,
            root=root,
            summary_path=summary_path,
        )


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("manifest", "manifest protocol compliance mismatch"),
        ("summary", "summary protocol compliance mismatch"),
    ],
)
def test_false_protocol_compliance_is_rejected(tmp_path, artifact, message):
    root, manifest_path, summary_path = _manifest_fixture(tmp_path)
    target = manifest_path if artifact == "manifest" else summary_path
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["protocol"]["protocol_compliant"] = False
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(strict.StrictInductiveError, match=message):
        strict.validate_block_handoff(
            manifest_path,
            root=root,
            summary_path=summary_path,
        )


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("top_hash", "summary top-level block-manifest SHA-256 mismatch"),
        ("nested_hash", "summary nested block-manifest SHA-256 mismatch"),
        ("path", "summary block-manifest path mismatch"),
    ],
)
def test_summary_manifest_path_and_hash_mismatches_are_rejected(
    tmp_path,
    mismatch,
    message,
):
    root, manifest_path, summary_path = _manifest_fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mismatch == "top_hash":
        summary["block_manifest_sha256"] = "0" * 64
    elif mismatch == "nested_hash":
        summary["block_manifest"]["sha256"] = "0" * 64
    else:
        summary["block_manifest"]["path"] = "other_block_manifest.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(strict.StrictInductiveError, match=message):
        strict.validate_block_handoff(
            manifest_path,
            root=root,
            summary_path=summary_path,
        )


def test_stale_protocol_is_rejected_before_analysis(tmp_path):
    root, manifest_path, summary_path = _manifest_fixture(tmp_path)
    protocol = root / strict.M2_PROTOCOL_RELATIVE_PATH
    protocol.write_text("amended protocol\n", encoding="utf-8")

    with pytest.raises(strict.StrictInductiveError, match="protocol SHA-256 mismatch"):
        strict.validate_block_handoff(
            manifest_path,
            root=root,
            summary_path=summary_path,
        )


def test_insufficient_external_support_is_unavailable_and_pairwise_finite(tmp_path):
    transform = from_origin(100.0, 220.0, 30.0, 30.0)
    blocks = np.array([[1, 2], [3, 4]], dtype=np.uint32)
    block_objects = tuple(
        strict.Block(
            block_id=f"b{numeric_id}",
            block_row=(numeric_id - 1) // 2,
            block_col=(numeric_id - 1) % 2,
            row_start=(numeric_id - 1) // 2,
            row_stop=(numeric_id - 1) // 2 + 1,
            col_start=(numeric_id - 1) % 2,
            col_stop=(numeric_id - 1) % 2 + 1,
        )
        for numeric_id in range(1, 5)
    )
    scale = strict.ScaleHandoff(
        "L",
        tmp_path / "blocks.tif",
        "0" * 64,
        (1, 2, 3, 4),
        {value: f"b{value}" for value in range(1, 5)},
        1,
        0,
        blocks,
        block_objects,
    )
    site = strict.SiteHandoff(
        "goldfield",
        strict.ANCHOR_SCENES["goldfield"],
        strict.GridSpec((2, 2), "EPSG:32611", transform),
        {"L": scale},
    )
    score = np.array([[0.9, np.nan], [0.2, 0.1]])
    reference = np.array([[1.0, 1.0], [0.0, np.nan]])
    result = strict._evaluate_layer(
        site=site,
        mineral="alunite",
        strict_score=score,
        reference=reference,
        full_score=None,
        full_path=tmp_path / "full.tif",
        workers=None,
    )

    metric = result["metrics"][0]
    assert metric["metric_status"] == "unavailable"
    assert metric["governance_status"] == "counts_and_maps_only"
    assert metric["auc"] is None
    assert sum(row["pairwise_finite"] for row in result["support"]) == 2
    assert result["comparisons"][0]["comparison_status"] == "unavailable"


def test_rank_auc_survives_when_every_threshold_fold_fails(tmp_path):
    transform = from_origin(100.0, 220.0, 30.0, 30.0)
    block_values = np.arange(1, 11, dtype=np.uint32).reshape(1, 10)
    block_objects = tuple(
        strict.Block(
            block_id=f"b{numeric_id}",
            block_row=0,
            block_col=numeric_id - 1,
            row_start=0,
            row_stop=1,
            col_start=numeric_id - 1,
            col_stop=numeric_id,
        )
        for numeric_id in range(1, 11)
    )
    scale = strict.ScaleHandoff(
        "L",
        tmp_path / "blocks.tif",
        "0" * 64,
        tuple(range(1, 11)),
        {value: f"b{value}" for value in range(1, 11)},
        1,
        20,
        block_values,
        block_objects,
    )
    site = strict.SiteHandoff(
        "goldfield",
        strict.ANCHOR_SCENES["goldfield"],
        strict.GridSpec((1, 10), "EPSG:32611", transform),
        {"L": scale},
    )
    score = np.array([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    reference = np.array([[1.0] * 5 + [0.0] * 5])

    result = strict._evaluate_layer(
        site=site,
        mineral="alunite",
        strict_score=score,
        reference=reference,
        full_score=None,
        full_path=tmp_path / "full.tif",
        workers=None,
    )

    metric = result["metrics"][0]
    assert metric["governance_status"] == "exploratory_only"
    assert metric["metric_status"] == "rank_available_threshold_unavailable"
    assert metric["rank_status"] == "available"
    assert metric["auc"] == 1.0
    assert metric["rank_evaluated_blocks"] == 10
    assert metric["rank_observations"] == 10
    assert metric["rank_n_pos"] == 5
    assert metric["rank_n_neg"] == 5
    assert metric["threshold_status"] == "unavailable"
    assert metric["threshold_unavailable_reason"] == "no_successful_threshold_folds"
    assert metric["threshold_evaluated_blocks"] == 0
    assert metric["threshold_observations"] == 0
    assert metric["threshold_n_pos"] == 0
    assert metric["threshold_n_neg"] == 0
    assert metric["balanced_accuracy"] is None
    assert metric["threshold_min"] is None
    assert metric["threshold_median"] is None
    assert metric["threshold_max"] is None
    assert result["thresholds"] == []
    intervals = {row["metric"]: row for row in result["intervals"]}
    assert intervals["auc"]["interval_status"] == "available"
    assert intervals["balanced_accuracy"]["interval_status"] == "unavailable"
    assert result["failures"][0]["stage"] == "threshold_calibration"


def test_strict_json_replaces_nonfinite_values():
    encoded = strict.strict_json_dumps({"nan": float("nan"), "inf": float("inf")})
    assert json.loads(encoded) == {"inf": None, "nan": None}
    assert "NaN" not in encoded
