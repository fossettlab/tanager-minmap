"""Synthetic-only tests for the sealed E6 post-run artifact verifier."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tanager_rocks.config import TARGET_MINERALS
from tanager_rocks.ensemble_sensitivity import (
    BASELINE_ENDMEMBERS,
    EXPECTED_CANDIDATE_COUNTS,
    MtmfFit,
    _fit_checksum,
    build_design,
)

ROOT = Path(__file__).resolve().parents[1]
V2_IMPORTED_TANAGER_ROCKS_MODULES = frozenset(
    {
        "src/tanager_rocks/__init__.py",
        "src/tanager_rocks/compare.py",
        "src/tanager_rocks/config.py",
        "src/tanager_rocks/degrade.py",
        "src/tanager_rocks/emit.py",
        "src/tanager_rocks/ensemble_sensitivity.py",
        "src/tanager_rocks/features.py",
        "src/tanager_rocks/hazard.py",
        "src/tanager_rocks/pipeline.py",
        "src/tanager_rocks/quality.py",
        "src/tanager_rocks/reference.py",
        "src/tanager_rocks/spatial_validation.py",
        "src/tanager_rocks/speclib.py",
        "src/tanager_rocks/strict_inductive.py",
        "src/tanager_rocks/unmix.py",
        "src/tanager_rocks/validate.py",
        "src/tanager_rocks/viz.py",
    }
)


def _load_verifier() -> ModuleType:
    path = ROOT / "scripts" / "verify_ensemble_artifacts.py"
    spec = importlib.util.spec_from_file_location("_verify_ensemble_artifacts_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load verifier module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()


@dataclass(frozen=True)
class SyntheticCapsule:
    run_dir: Path
    source_manifest: Path
    block_manifest: Path
    source_sha256: str
    timing_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_hash(label: str) -> str:
    return _sha256_bytes(label.encode("utf-8"))


def _logical_fit_checksum(payload: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for mineral in VERIFIER.TARGET_MINERALS:
        digest.update(np.ascontiguousarray(payload[f"mf_{mineral}"]).tobytes())
        digest.update(np.ascontiguousarray(payload[f"infeas_{mineral}"]).tobytes())
    return digest.hexdigest()


def _synthetic_fit_payload(fit_index: int) -> dict[str, np.ndarray]:
    grid = np.arange(4, dtype=np.float64).reshape(2, 2)
    payload = {
        "valid_support": np.ones(grid.shape, dtype=np.bool_),
        "contributing_pixels": np.asarray(grid.size),
        "retained_bands": np.asarray(VERIFIER.RETAINED_BANDS),
    }
    for mineral_index, mineral in enumerate(VERIFIER.TARGET_MINERALS):
        offset = float(fit_index * len(VERIFIER.TARGET_MINERALS) + mineral_index)
        payload[f"mf_{mineral}"] = grid + offset
        payload[f"infeas_{mineral}"] = grid + offset + 0.25
    return payload


def _cache_path(run_dir: Path, fit_id: str) -> Path:
    site = fit_id.split(":", maxsplit=1)[0]
    return run_dir / ".score_cache" / site / f"{_sha256_bytes(fit_id.encode('utf-8'))}.npz"


def _read_fit_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _rewrite_fit_cache(path: Path, mutate) -> None:
    payload = _read_fit_payload(path)
    mutate(payload)
    np.savez_compressed(path, **payload)


def _candidate_population() -> dict[str, tuple[str, ...]]:
    population: dict[str, tuple[str, ...]] = {}
    for mineral in TARGET_MINERALS:
        medoid = BASELINE_ENDMEMBERS[mineral]
        extras = tuple(
            f"splib07a_{mineral.title()}_candidate_{index:02d}_ASDFRa_AREF.txt"
            for index in range(EXPECTED_CANDIDATE_COUNTS[mineral] - 1)
        )
        population[mineral] = tuple(sorted((medoid, *extras)))
    return population


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _block_manifest_bytes() -> bytes:
    sites = {}
    for site in VERIFIER.FROZEN_SITES:
        sites[site] = {
            "scene_id": VERIFIER.ANCHOR_SCENES[site],
            "grid": {
                "shape": [2, 2],
                "crs": "EPSG:32611",
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 2.0],
            },
            "scales": {
                scale: {
                    "anchor_scene_id": VERIFIER.ANCHOR_SCENES[site],
                    "block_raster": f"{site}_{scale}.tif",
                    "block_raster_sha256": _synthetic_hash(f"{site} {scale} raster"),
                    "complete_block_ids": [1, 2],
                    "complete_blocks": 2,
                }
                for scale in ("L", "2L")
            },
        }
    payload = {
        "manifest_type": "spatial_validation_complete_blocks",
        "protocol": {
            "path": "docs/m2_spatial_validation_preregistration.md",
            "sha256": _synthetic_hash("M2 preregistration"),
        },
        "sites": sites,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _design_and_members() -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    design, members = build_design(
        candidates=_candidate_population(),
        complete_blocks={"goldfield": (1, 2), "bingham": (1, 2)},
    )
    source_hashes = {
        relative: (
            VERIFIER.FROZEN_PREREGISTRATION_SHA256
            if relative == "docs/m2_ensemble_sensitivity_preregistration.md"
            else _synthetic_hash(relative)
        )
        for relative in VERIFIER.GOVERNING_FILES
    }
    input_manifest_hash = _synthetic_hash("docs/input_manifest.json")
    block_manifest_bytes = _block_manifest_bytes()
    block_manifest_hash = _sha256_bytes(block_manifest_bytes)
    rockwell_hash = _synthetic_hash("synthetic Rockwell reference")
    design.update(
        {
            "protocol": {
                "path": "docs/m2_ensemble_sensitivity_preregistration.md",
                "sha256": VERIFIER.FROZEN_PREREGISTRATION_SHA256,
                "expected_sha256": VERIFIER.FROZEN_PREREGISTRATION_SHA256,
                "protocol_compliant": True,
                "amendment": None,
            },
            "protocol_deviations": {},
            "protocol_amendment": None,
            "code_commit": "a" * 40,
            "governing_files": [
                {
                    "path": relative,
                    "sha256": source_hashes[relative],
                    "git_status": "  ",
                    "tracked": True,
                    "dirty": False,
                }
                for relative in VERIFIER.GOVERNING_FILES
            ],
            "lockfile_sha256": _synthetic_hash("uv.lock"),
            "input_manifest": {
                "path": "docs/input_manifest.json",
                "sha256": input_manifest_hash,
                "inputs": [
                    {
                        "id": "synthetic_input",
                        "logical_path": "data/raw/synthetic.bin",
                        "size_bytes": 1,
                        "sha256": _synthetic_hash("synthetic input"),
                    }
                ],
            },
            "rockwell_reference": {
                "path": "data/reference/rockwell_goldfield_20240925_185504_87_4001.tif",
                "sha256": rockwell_hash,
                "shape": [2, 2],
                "crs": "EPSG:32611",
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 2.0],
                "anchor_scene": "20240925_185504_87_4001",
                "m2_block_manifest": {
                    "path": "data/processed/spatial_validation/block_manifest.json",
                    "sha256": block_manifest_hash,
                },
                "m2_block_rasters": {
                    "L": {
                        "sha256": _synthetic_hash("goldfield L raster"),
                        "complete_blocks": 2,
                    },
                    "2L": {
                        "sha256": _synthetic_hash("goldfield 2L raster"),
                        "complete_blocks": 2,
                    },
                },
            },
            "quality_policy": {
                "path": "docs/tanager_quality_mask_policy.md",
                "sha256": source_hashes["docs/tanager_quality_mask_policy.md"],
                "retained_bands": 363,
            },
            "block_manifest": {
                "path": "data/processed/spatial_validation/block_manifest.json",
                "sha256": block_manifest_hash,
            },
            "software": {"numpy": "synthetic"},
            "compute_controls": {
                "device": "cpu",
                "batch_size": 1,
                "storage_layout": "disk",
                "numpy_reference": True,
                "accelerator_backend": None,
            },
        }
    )
    identity = {
        key: value
        for key, value in design.items()
        if key not in {"compute_controls", "scientific_design_sha256"}
    }
    design["scientific_design_sha256"] = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return design, members, block_manifest_bytes


def _write_members(
    run_dir: Path, design_sha: str, members: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    completed: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}
    for member in members:
        fit_id = str(member["fit_id"])
        if fit_id not in checksums:
            payload = _synthetic_fit_payload(len(checksums))
            checksums[fit_id] = _logical_fit_checksum(payload)
            cache_path = _cache_path(run_dir, fit_id)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **payload)
        checksum = checksums[fit_id]
        completed.append(
            {
                **member,
                "contributing_pixels": 4,
                "retained_bands": 363,
                "status": "complete",
                "failure_reason": None,
                "output_checksum": checksum,
                "wall_time_seconds": None,
                "peak_memory_bytes": None,
                "design_sha256": design_sha,
            }
        )
    _write_csv(run_dir / "members.csv", VERIFIER.MEMBER_FIELDS, completed)
    return completed, checksums


def _write_metrics(run_dir: Path, members: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for member in members:
        for mineral in TARGET_MINERALS:
            rows.append(
                {
                    "site": member["site"],
                    "scene": member["scene"],
                    "mineral": mineral,
                    "member_id": member["member_id"],
                    "member_class": member["member_class"],
                    "stochastic_replicate": member["stochastic_replicate"],
                    "ridge": member["ridge"],
                    "detection_quantile": member["detection_quantile"],
                    "infeasibility_gate": member["infeasibility_gate"],
                    "aggregation": "scene",
                    "block_scale": "L",
                    "block_id": 0,
                    "common_support_pixels": 4,
                    "external_status": "not_applicable",
                    "covariance_scope": "not_applicable",
                    "strict_covariance_exclusion_status": "not_applicable",
                }
            )
    _write_csv(run_dir / "member_metrics.csv", VERIFIER.METRIC_FIELDS, rows)


def _write_factor_and_calibration(run_dir: Path) -> None:
    factor_levels = {
        "axis": ("endmember_only", "covariance_only", "calibration_only"),
        "ridge": ("0.001", "0.1"),
        "quantile": ("0.85", "0.95"),
        "gate": ("none",),
    }
    reference_levels = {
        "axis": "baseline",
        "ridge": "0.01",
        "quantile": "0.9",
        "gate": "1",
    }
    endpoints = (
        "detection_prevalence",
        "common_support_loss_fraction",
        "rank_correlation",
        "dominant_class_switch_frequency",
        "auc",
        "balanced_accuracy",
        "positive_f1",
        "negative_f1",
        "macro_f1",
        "tpr",
        "fpr",
        "prevalence",
    )
    factor_rows = []
    for site in VERIFIER.FROZEN_SITES:
        for mineral in TARGET_MINERALS:
            for block_scale in ("L", "2L"):
                for factor, levels in factor_levels.items():
                    for level in levels:
                        for endpoint in endpoints:
                            factor_rows.append(
                                {
                                    "site": site,
                                    "mineral": mineral,
                                    "block_scale": block_scale,
                                    "factor": factor,
                                    "level": level,
                                    "reference_level": reference_levels[factor],
                                    "endpoint": endpoint,
                                    "scheduled_replicates": 10_000,
                                    "valid_replicates": 0,
                                    "finite_fraction": 0.0,
                                    "interval_available": False,
                                    "unavailable_reason": "synthetic_unavailable",
                                    "n_pairs": 1,
                                    "complete_blocks": 2,
                                    "paired_support_pixels": 0,
                                    "contrast_status": "descriptive_paired_complete_block",
                                }
                            )
    _write_csv(
        run_dir / "factor_effects.csv",
        VERIFIER.FACTOR_FIELDS,
        factor_rows,
    )
    calibration_rows = []
    for mineral in (item for item in TARGET_MINERALS if item != "gypsum"):
        for index in range(10):
            calibration_rows.append(
                {
                    "site": "goldfield",
                    "mineral": mineral,
                    "confidence_bin": f"[{index / 10:.1f},{(index + 1) / 10:.1f}"
                    + ("]" if index == 9 else ")"),
                    "support_blocks": 0,
                    "support_pixels": 0,
                    "scheduled_replicates": 10_000,
                    "valid_replicates": 0,
                    "finite_fraction": 0.0,
                    "interval_available": False,
                    "unavailable_reason": "synthetic_empty_bin",
                    "brier_interval_available": False,
                    "brier_valid_replicates": 0,
                    "brier_finite_fraction": 0.0,
                    "ece_interval_available": False,
                    "ece_valid_replicates": 0,
                    "ece_finite_fraction": 0.0,
                    "status": "empty_fixed_bin",
                }
            )
    _write_csv(run_dir / "calibration.csv", VERIFIER.CALIBRATION_FIELDS, calibration_rows)


def _write_maps(run_dir: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    transform = rasterio.transform.from_origin(0, 2, 1, 1)
    metadata = {
        "n_valid": ("uint16", 0, np.zeros((2, 2), dtype=np.uint16)),
        "detection_frequency": (
            "float32",
            np.nan,
            np.full((2, 2), np.nan, dtype=np.float32),
        ),
        "confidence_class": ("int8", -1, np.full((2, 2), -1, dtype=np.int8)),
        "modal_class": ("int16", -2, np.full((2, 2), -2, dtype=np.int16)),
        "modal_frequency": (
            "float32",
            np.nan,
            np.full((2, 2), np.nan, dtype=np.float32),
        ),
        "class_entropy": (
            "float32",
            np.nan,
            np.full((2, 2), np.nan, dtype=np.float32),
        ),
        "switch_frequency": (
            "float32",
            np.nan,
            np.full((2, 2), np.nan, dtype=np.float32),
        ),
    }
    maps_dir = run_dir / "maps"
    maps_dir.mkdir()
    for relative in sorted(VERIFIER._map_paths()):
        suffix = next(name for name in metadata if relative.endswith(f"_{name}.tif"))
        dtype, nodata, values = metadata[suffix]
        with rasterio.open(
            run_dir / relative,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype=dtype,
            crs="EPSG:32611",
            transform=transform,
            nodata=nodata,
        ) as dataset:
            dataset.write(values, 1)


def _cell_counts() -> dict[str, int]:
    return {
        json.dumps(
            {"ridge": float(ridge), "quantile": float(quantile), "gate": gate},
            sort_keys=True,
            separators=(",", ":"),
        ): 16
        for ridge in VERIFIER.RIDGES
        for quantile in VERIFIER.QUANTILES
        for gate in VERIFIER.GATES
    }


def _external_rows() -> list[dict[str, Any]]:
    rows = []
    for scope in ("full_scene_covariance", "strict_covariance_exclusion"):
        for scale, metric in (("L", "auc"), ("L", "balanced_accuracy"), ("2L", "auc")):
            rows.append(
                {
                    "covariance_scope": scope,
                    "mineral": "alunite",
                    "scale": scale,
                    "metric": metric,
                    "interval_available": True,
                    "confirmatory_support": True,
                    "lower_95": 0.75,
                    "point_estimate": 0.80,
                }
            )
    return rows


def _site_summary(site: str) -> dict[str, Any]:
    nested = {
        "shared_draw_per_replicate": True,
        "member_summary_within_replicate": "median",
        "replicates": 10_000,
        "finite_replicate_fraction_required": 0.95,
        "dominant_class_switch_lower_95": 0.01,
        "dominant_class_switch_upper_95": 0.10,
        "endpoint_intervals": (
            [
                {
                    "scale": "L",
                    "mineral": "alunite",
                    "metric": "dominant_class_switch_frequency",
                    "upper_95": 0.10,
                    "interval_available": True,
                }
            ]
            if site == "goldfield"
            else []
        ),
        "external_intervals": _external_rows() if site == "goldfield" else [],
    }
    summary: dict[str, Any] = {
        "site": site,
        "recorded_variants": 355,
        "unique_mtmf_fits": 83,
        "joint_valid_members": 288,
        "failed_members": [],
        "analytical_cell_valid_member_counts": _cell_counts(),
        "external_covariance_estimand": "full_scene_covariance_operational_transductive",
        "goldfield_alunite_gate_components": {
            "stable_core_retention": 0.91,
            "median_rank_correlation": 0.90,
            "rank_correlation_5th_percentile": 0.70,
            "dominant_class_switch_nested_bootstrap_upper_95": 0.10,
            "external_interval_gate": True if site == "goldfield" else None,
        },
        "nested_block_bootstrap": nested,
        "strict_covariance_exclusion": {
            "status": "complete" if site == "goldfield" else "not_applicable",
            "pooled_with_operational": False,
            "fold_failures": {},
        },
    }
    if site == "goldfield":
        summary.update(
            {
                "confirmatory_gate_available": True,
                "confirmatory_gate_pass": True,
                "analytical_cells_complete": True,
                "stability_available": True,
                "stability_pass": True,
                "external_interval_available": True,
                "external_pass": True,
                "strict_covariance_interval_available": True,
                "strict_covariance_pass": True,
                "permitted_claim_classification": (
                    "validated_analytically_robust_alteration_zone_discrimination"
                ),
            }
        )
    else:
        summary.update(
            {
                "confirmatory_gate_available": False,
                "confirmatory_gate_pass": None,
                "permitted_claim_classification": ("map_stability_only_no_external_reference"),
            }
        )
    return summary


def _write_timing(run_dir: Path, checksums: dict[str, str]) -> str:
    records = []
    for site in VERIFIER.FROZEN_SITES:
        for member_class, fit_id, replicate in (
            ("baseline", f"{site}:fit:baseline:r0.01", None),
            ("joint", f"{site}:fit:joint:r00:ridge0.01", 0),
        ):
            records.append(
                {
                    "site": site,
                    "scene": VERIFIER.ANCHOR_SCENES[site],
                    "fit_id": fit_id,
                    "member_class": member_class,
                    "stochastic_replicate": replicate,
                    "wall_time_seconds": 1.0,
                    "peak_memory_bytes": 1,
                    "output_sha256": checksums[fit_id],
                    "device": "cpu",
                    "scientific_outputs_retained": False,
                }
            )
    path = run_dir / "timing_pilot.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "mode": "timing_pilot_only",
                "fit_count": 4,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return _sha256_file(path)


def _write_source_manifest(base: Path, design: dict[str, Any]) -> tuple[Path, str]:
    entries: dict[str, str] = {
        record["path"]: record["sha256"] for record in design["governing_files"]
    }
    for path in V2_IMPORTED_TANAGER_ROCKS_MODULES:
        entries.setdefault(path, _synthetic_hash(path))
    entries.update(
        {
            "uv.lock": design["lockfile_sha256"],
            "docs/input_manifest.json": design["input_manifest"]["sha256"],
            "data/processed/spatial_validation/block_manifest.json": (
                design["block_manifest"]["sha256"]
            ),
            design["rockwell_reference"]["path"]: design["rockwell_reference"]["sha256"],
            "data/processed/spatial_validation/goldfield_L.tif": (
                design["rockwell_reference"]["m2_block_rasters"]["L"]["sha256"]
            ),
            "data/processed/spatial_validation/goldfield_2L.tif": (
                design["rockwell_reference"]["m2_block_rasters"]["2L"]["sha256"]
            ),
        }
    )
    filler_index = 0
    while len(entries) < VERIFIER.EXPECTED_SOURCE_MANIFEST_ENTRIES:
        path = f"../tanager-spec/synthetic_{filler_index:02d}.py"
        entries[path] = _synthetic_hash(path)
        filler_index += 1
    manifest = base / "source_manifest.sha256"
    manifest.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(entries.items())),
        encoding="utf-8",
    )
    return manifest, _sha256_file(manifest)


def _build_capsule(base: Path) -> SyntheticCapsule:
    run_dir = base / "completed_run"
    run_dir.mkdir()
    design, member_design, block_manifest_bytes = _design_and_members()
    block_manifest = base / "block_manifest.json"
    block_manifest.write_bytes(block_manifest_bytes)
    design_path = run_dir / "design.json"
    design_path.write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed, checksums = _write_members(run_dir, _sha256_file(design_path), member_design)
    _write_metrics(run_dir, completed)
    _write_factor_and_calibration(run_dir)
    _write_maps(run_dir)
    timing_sha = _write_timing(run_dir, checksums)

    artifact_paths = {
        "design.json",
        "members.csv",
        "member_metrics.csv",
        "factor_effects.csv",
        "calibration.csv",
        *VERIFIER._map_paths(),
    }
    goldfield = _site_summary("goldfield")
    bingham = _site_summary("bingham")
    summary = {
        "schema_version": "1.0",
        "frequency_estimand": "finite_design_empirical_frequency",
        "sites": [goldfield, bingham],
        "counts": {
            "recorded_variants": 710,
            "unique_mtmf_fits": 166,
            "failed_members": 0,
        },
        "artifact_sha256": {
            relative: _sha256_file(run_dir / relative) for relative in sorted(artifact_paths)
        },
        "permitted_claim_classification": (
            "validated_analytically_robust_alteration_zone_discrimination"
        ),
        "axis_contrasts": "descriptive_paired_only",
        "compute_controls": {
            "device": "cpu",
            "batch_size": 1,
            "storage_layout": "disk",
            "scientifically_inert": True,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_bytes(VERIFIER._expected_report(summary))
    source_manifest, source_sha = _write_source_manifest(base, design)
    return SyntheticCapsule(run_dir, source_manifest, block_manifest, source_sha, timing_sha)


@pytest.fixture(scope="module")
def synthetic_capsule(tmp_path_factory) -> SyntheticCapsule:
    return _build_capsule(tmp_path_factory.mktemp("ensemble_verifier"))


def _copy_capsule(capsule: SyntheticCapsule, tmp_path: Path) -> SyntheticCapsule:
    copied_run = tmp_path / "completed_run"
    shutil.copytree(capsule.run_dir, copied_run)
    copied_manifest = tmp_path / "source_manifest.sha256"
    shutil.copy2(capsule.source_manifest, copied_manifest)
    copied_block_manifest = tmp_path / "block_manifest.json"
    shutil.copy2(capsule.block_manifest, copied_block_manifest)
    return SyntheticCapsule(
        copied_run,
        copied_manifest,
        copied_block_manifest,
        capsule.source_sha256,
        capsule.timing_sha256,
    )


def _replace_source_manifest(capsule: SyntheticCapsule, payload: bytes) -> SyntheticCapsule:
    capsule.source_manifest.write_bytes(payload)
    return SyntheticCapsule(
        capsule.run_dir,
        capsule.source_manifest,
        capsule.block_manifest,
        _sha256_file(capsule.source_manifest),
        capsule.timing_sha256,
    )


def _verify(capsule: SyntheticCapsule):
    return VERIFIER.verify_run(
        run_dir=capsule.run_dir,
        source_manifest=capsule.source_manifest,
        block_manifest=capsule.block_manifest,
        expected_source_manifest_sha256=capsule.source_sha256,
        expected_timing_pilot_sha256=capsule.timing_sha256,
    )


def _rewrite_summary(run_dir: Path, mutate) -> None:
    path = run_dir / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _member_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "members.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_fit_cache(run_dir: Path) -> Path:
    return _cache_path(run_dir, _member_rows(run_dir)[0]["fit_id"])


def test_complete_synthetic_capsule_passes_without_endpoint_disclosure(synthetic_capsule):
    manifest_bytes = synthetic_capsule.source_manifest.read_bytes()
    manifest_lines = manifest_bytes.decode("utf-8").splitlines()
    manifest_paths = [line.split("  ", maxsplit=1)[1] for line in manifest_lines]

    assert manifest_bytes.endswith(b"\n")
    assert b"\r" not in manifest_bytes
    assert len(manifest_lines) == 49
    assert manifest_paths == sorted(manifest_paths)
    assert any(path.startswith("../tanager-spec/") for path in manifest_paths)
    assert all(VERIFIER.SOURCE_LINE.fullmatch(line) for line in manifest_lines)

    report = _verify(synthetic_capsule)
    rendered = report.render()

    assert report.passed
    assert "PASS check=overall count=0" in rendered
    assert "count=710" in rendered
    assert "count=166" in rendered
    assert "PASS check=fit_cache_semantics count=166" in rendered
    assert "PASS check=source_manifest_binding count=49" in rendered
    for forbidden in (
        "validated_analytically",
        "stable_core_retention",
        "rank_correlation",
        "auc",
        "0.91",
    ):
        assert forbidden not in rendered


def test_synthetic_source_manifest_contains_complete_v2_imported_module_set(
    synthetic_capsule,
):
    manifest_paths = {
        line.split("  ", maxsplit=1)[1]
        for line in synthetic_capsule.source_manifest.read_text(encoding="utf-8").splitlines()
    }
    imported_modules = {path for path in manifest_paths if path.startswith("src/tanager_rocks/")}

    assert imported_modules == V2_IMPORTED_TANAGER_ROCKS_MODULES


def test_synthetic_checksum_matches_producer_contract():
    payload = _synthetic_fit_payload(7)
    fit = MtmfFit(
        matched_filter={mineral: payload[f"mf_{mineral}"] for mineral in VERIFIER.TARGET_MINERALS},
        infeasibility={
            mineral: payload[f"infeas_{mineral}"] for mineral in VERIFIER.TARGET_MINERALS
        },
        valid_support=payload["valid_support"],
        contributing_pixels=int(payload["contributing_pixels"].item()),
        retained_bands=int(payload["retained_bands"].item()),
    )

    assert _logical_fit_checksum(payload) == _fit_checksum(fit)


def test_fit_cache_array_mutation_without_ledger_change_fails_closed(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        changed = payload["mf_alunite"].copy()
        changed.flat[0] += 1.0
        payload["mf_alunite"] = changed

    _rewrite_fit_cache(_first_fit_cache(capsule.run_dir), mutate)

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert "discrepancy=logical_checksum_mismatch" in rendered
    assert ".npz" not in rendered


@pytest.mark.parametrize("mutation", ["inject", "remove"])
def test_fit_cache_member_key_closure_fails_closed(synthetic_capsule, tmp_path, mutation):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        if mutation == "inject":
            payload["unexpected_array"] = np.zeros((2, 2), dtype=np.float64)
        else:
            payload.pop("mf_alunite")

    _rewrite_fit_cache(_first_fit_cache(capsule.run_dir), mutate)

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert "discrepancy=member_key_closure_mismatch" in rendered
    assert "unexpected_array" not in rendered
    assert ".npz" not in rendered


def test_fit_cache_object_dtype_cannot_enable_pickle_loading(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        payload["mf_alunite"] = np.full(
            payload["mf_alunite"].shape, "ENDPOINT_SENTINEL", dtype=object
        )

    _rewrite_fit_cache(_first_fit_cache(capsule.run_dir), mutate)

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert "discrepancy=array_load_rejected" in rendered
    assert "ENDPOINT_SENTINEL" not in rendered
    assert ".npz" not in rendered


@pytest.mark.parametrize("mutation", ["non_2d", "shape_mismatch"])
def test_fit_cache_shape_mismatch_fails_before_checksum(synthetic_capsule, tmp_path, mutation):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        shape = (4,) if mutation == "non_2d" else (1, 4)
        payload["infeas_alunite"] = payload["infeas_alunite"].reshape(shape)

    _rewrite_fit_cache(_first_fit_cache(capsule.run_dir), mutate)

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert "discrepancy=score_shape_mismatch" in rendered
    assert ".npz" not in rendered


@pytest.mark.parametrize("mutation", ["dtype", "shape"])
def test_fit_cache_valid_support_schema_fails_closed(synthetic_capsule, tmp_path, mutation):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        if mutation == "dtype":
            payload["valid_support"] = payload["valid_support"].astype(np.uint8)
        else:
            payload["valid_support"] = payload["valid_support"].reshape(4)

    _rewrite_fit_cache(_first_fit_cache(capsule.run_dir), mutate)

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert "discrepancy=valid_support_invalid" in rendered
    assert ".npz" not in rendered


def test_fit_cache_valid_support_must_match_score_finiteness(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        changed = payload["valid_support"].copy()
        changed.flat[0] = False
        payload["valid_support"] = changed

    _rewrite_fit_cache(_first_fit_cache(capsule.run_dir), mutate)

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert "discrepancy=score_support_mismatch" in rendered
    assert ".npz" not in rendered


def test_fit_cache_score_dtype_must_match_producer(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        payload["mf_alunite"] = payload["mf_alunite"].astype(np.int64)

    _rewrite_fit_cache(_first_fit_cache(capsule.run_dir), mutate)

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert "discrepancy=score_dtype_mismatch" in rendered
    assert ".npz" not in rendered


def test_fit_cache_infeasibility_cannot_be_negative(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        changed = payload["infeas_alunite"].copy()
        changed.flat[0] = -1.0
        payload["infeas_alunite"] = changed

    _rewrite_fit_cache(_first_fit_cache(capsule.run_dir), mutate)

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert "discrepancy=infeasibility_value_invalid" in rendered
    assert ".npz" not in rendered


@pytest.mark.parametrize(
    ("mutation", "discrepancy"),
    [
        ("contributing_value", "ledger_metadata_mismatch"),
        ("retained_value", "ledger_metadata_mismatch"),
        ("scalar_shape", "metadata_scalar_invalid"),
    ],
)
def test_fit_cache_metadata_mismatch_fails_closed(
    synthetic_capsule, tmp_path, mutation, discrepancy
):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        if mutation == "contributing_value":
            payload["contributing_pixels"] = np.asarray(
                int(payload["contributing_pixels"].item()) + 1
            )
        elif mutation == "retained_value":
            payload["retained_bands"] = np.asarray(int(payload["retained_bands"].item()) + 1)
        else:
            payload["contributing_pixels"] = np.asarray(
                [int(payload["contributing_pixels"].item())]
            )

    _rewrite_fit_cache(_first_fit_cache(capsule.run_dir), mutate)

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert f"discrepancy={discrepancy}" in rendered
    assert ".npz" not in rendered


def test_fit_cache_checksum_is_bound_to_every_reused_fit_row(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    members_path = capsule.run_dir / "members.csv"
    rows = _member_rows(capsule.run_dir)
    target_fit = next(
        row["fit_id"]
        for row in rows
        if row["member_class"] == "analytical_grid" and row["ridge"] == "0.001"
    )
    replacement_checksum = next(
        row["output_checksum"] for row in rows if row["fit_id"] != target_fit
    )
    reused_rows = [row for row in rows if row["fit_id"] == target_fit]
    assert len(reused_rows) > 1
    for row in reused_rows:
        row["output_checksum"] = replacement_checksum
    _write_csv(members_path, VERIFIER.MEMBER_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "members.csv", _sha256_file(members_path)
        ),
    )

    report = _verify(capsule)
    rendered = report.render()

    assert not report.passed
    assert "FAIL check=fit_cache_semantics" in rendered
    assert "discrepancy=logical_checksum_mismatch" in rendered
    assert ".npz" not in rendered


def test_unexpected_file_fails_closed(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    (capsule.run_dir / "unexpected.txt").write_text("sealed synthetic value", encoding="utf-8")

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=unexpected_file" in report.render()
    assert "sealed synthetic value" not in report.render()


def test_unexpected_filename_cannot_disclose_endpoint_sentinel(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    sentinel = "ENDPOINT_SENTINEL_0.91"
    (capsule.run_dir / f"{sentinel}.txt").write_text("synthetic", encoding="utf-8")

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=unexpected_file" in report.render()
    assert sentinel not in report.render()


def test_external_filename_is_not_rendered_on_success(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    sentinel = "ENDPOINT_SENTINEL_0.91"
    renamed_manifest = tmp_path / f"{sentinel}.sha256"
    capsule.source_manifest.rename(renamed_manifest)
    capsule = SyntheticCapsule(
        capsule.run_dir,
        renamed_manifest,
        capsule.block_manifest,
        capsule.source_sha256,
        capsule.timing_sha256,
    )

    report = _verify(capsule)

    assert report.passed
    assert sentinel not in report.render()


def test_symlink_is_rejected_without_following_target(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("do not disclose this sentinel", encoding="utf-8")
    (capsule.run_dir / "escape-link").symlink_to(outside)

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=symlink_rejected" in report.render()
    assert "do not disclose this sentinel" not in report.render()


def test_nested_symlink_is_rejected_without_following_target(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("nested target sentinel", encoding="utf-8")
    (capsule.run_dir / "maps" / "nested-link").symlink_to(outside)

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=symlink_rejected" in report.render()
    assert "nested target sentinel" not in report.render()


def test_hardlink_is_rejected(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    os.link(capsule.run_dir / "design.json", capsule.run_dir / "design-hardlink.json")

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=hardlink_rejected" in report.render()


def test_parent_component_in_run_directory_is_rejected(synthetic_capsule, tmp_path):
    path_with_parent = tmp_path / "unused" / ".." / synthetic_capsule.run_dir.name
    capsule = SyntheticCapsule(
        path_with_parent,
        synthetic_capsule.source_manifest,
        synthetic_capsule.block_manifest,
        synthetic_capsule.source_sha256,
        synthetic_capsule.timing_sha256,
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=parent_component_rejected" in report.render()


def test_open_file_detects_mutation_while_descriptor_is_in_use(tmp_path):
    run_dir = tmp_path / "sealed"
    run_dir.mkdir()
    artifact = run_dir / "artifact.txt"
    artifact.write_text("before", encoding="utf-8")

    with VERIFIER.SealedDirectory(run_dir) as sealed:
        sealed.inventory(allowed_directories=set())
        with pytest.raises(VERIFIER.SealViolation, match="artifact_changed_during_verification"):
            with sealed.open_file("artifact.txt"):
                artifact.write_text("after!", encoding="utf-8")


def test_external_file_detects_mutation_while_descriptor_is_in_use(tmp_path):
    artifact = tmp_path / "external.txt"
    artifact.write_text("before", encoding="utf-8")

    with pytest.raises(VERIFIER.SealViolation, match="changed_during_verification"):
        with VERIFIER._open_external_file(artifact, check="external"):
            artifact.write_text("after!", encoding="utf-8")


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_late_unexpected_entry_after_inventory_fails_closed(
    synthetic_capsule, tmp_path, monkeypatch, entry_kind
):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    original_inventory = VERIFIER.SealedDirectory.inventory
    inventory_calls = 0

    def inventory_with_late_file(self, *, allowed_directories):
        nonlocal inventory_calls
        observed = original_inventory(self, allowed_directories=allowed_directories)
        inventory_calls += 1
        if inventory_calls == 1:
            late_entry = self.path / f"late-unexpected-{entry_kind}"
            if entry_kind == "file":
                late_entry.write_text("synthetic", encoding="utf-8")
            else:
                late_entry.mkdir()
        return observed

    monkeypatch.setattr(VERIFIER.SealedDirectory, "inventory", inventory_with_late_file)

    report = _verify(capsule)

    assert not report.passed
    assert "check=artifact_closure" in report.render()


def test_transient_directory_entry_changes_root_identity(synthetic_capsule, tmp_path, monkeypatch):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    original_closure_digest = VERIFIER._closure_digest

    def closure_with_transient_entry(sealed, files):
        digest = original_closure_digest(sealed, files)
        transient = sealed.path / "transient-entry"
        transient.write_text("synthetic", encoding="utf-8")
        transient.unlink()
        return digest

    monkeypatch.setattr(VERIFIER, "_closure_digest", closure_with_transient_entry)

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=directory_or_artifact_changed_during_verification" in report.render()


def test_declared_artifact_hash_mismatch_fails_closed(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    with (capsule.run_dir / "factor_effects.csv").open("a", encoding="utf-8") as handle:
        handle.write("synthetic mutation\n")

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=declared_hash_mismatch" in report.render()
    assert "synthetic mutation" not in report.render()


def test_nonterminal_member_status_fails_even_with_updated_declared_hash(
    synthetic_capsule, tmp_path
):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    members_path = capsule.run_dir / "members.csv"
    with members_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["status"] = "running"
    _write_csv(members_path, VERIFIER.MEMBER_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "members.csv", _sha256_file(members_path)
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=nonconforming_status_rows" in report.render()


def test_duplicate_member_row_fails_exact_row_count(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    members_path = capsule.run_dir / "members.csv"
    with members_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.append(dict(rows[0]))
    _write_csv(members_path, VERIFIER.MEMBER_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "members.csv", _sha256_file(members_path)
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=row_count_mismatch" in report.render()


def test_reused_fit_requires_one_contributing_pixel_identity(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    members_path = capsule.run_dir / "members.csv"
    with members_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    shared_fit = rows[0]["fit_id"]
    duplicate_index = next(
        index for index, row in enumerate(rows[1:], start=1) if row["fit_id"] == shared_fit
    )
    rows[duplicate_index]["contributing_pixels"] = "5"
    _write_csv(members_path, VERIFIER.MEMBER_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "members.csv", _sha256_file(members_path)
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=fit_provenance_identity_conflict" in report.render()


def test_nonfinite_metric_csv_value_fails_closed(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    metrics_path = capsule.run_dir / "member_metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["rank_correlation"] = "NaN"
    _write_csv(metrics_path, VERIFIER.METRIC_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "member_metrics.csv", _sha256_file(metrics_path)
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=numeric_field_invalid" in report.render()


def test_metric_status_must_match_producer_schema(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    metrics_path = capsule.run_dir / "member_metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["external_status"] = "synthetic_wrong_status"
    _write_csv(metrics_path, VERIFIER.METRIC_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "member_metrics.csv", _sha256_file(metrics_path)
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=status_identity_invalid" in report.render()


def test_csv_boolean_field_does_not_accept_numeric_text(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    factor_path = capsule.run_dir / "factor_effects.csv"
    with factor_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["interval_available"] = "1"
    _write_csv(factor_path, VERIFIER.FACTOR_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "factor_effects.csv", _sha256_file(factor_path)
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=boolean_field_invalid" in report.render()


def test_factor_effect_identity_set_must_be_complete(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    factor_path = capsule.run_dir / "factor_effects.csv"
    with factor_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.pop()
    _write_csv(factor_path, VERIFIER.FACTOR_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "factor_effects.csv", _sha256_file(factor_path)
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=factor_identity_closure_mismatch" in report.render()


def test_nonfinite_calibration_csv_value_fails_closed(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    calibration_path = capsule.run_dir / "calibration.csv"
    with calibration_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["brier_score"] = "Inf"
    _write_csv(calibration_path, VERIFIER.CALIBRATION_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "calibration.csv", _sha256_file(calibration_path)
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=numeric_field_invalid" in report.render()


def test_calibration_mineral_bin_set_must_be_complete(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    calibration_path = capsule.run_dir / "calibration.csv"
    with calibration_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["mineral"] != "muscovite"]
    _write_csv(calibration_path, VERIFIER.CALIBRATION_FIELDS, rows)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["artifact_sha256"].__setitem__(
            "calibration.csv", _sha256_file(calibration_path)
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=calibration_identity_closure_mismatch" in report.render()


def test_summary_boolean_cannot_coerce_to_batch_size(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["compute_controls"].__setitem__("batch_size", True),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=compute_control_mismatch" in report.render()


def test_gate_boolean_cannot_coerce_from_integer(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["sites"][0].__setitem__("stability_available", 1),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=gate_or_classification_mismatch" in report.render()


def test_site_summary_rejects_unexpected_producer_field(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    _rewrite_summary(
        capsule.run_dir,
        lambda payload: payload["sites"][0].__setitem__(
            "unexpected_endpoint_field", "ENDPOINT_SENTINEL_0.91"
        ),
    )

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=site_field_closure_mismatch" in report.render()
    assert "ENDPOINT_SENTINEL" not in report.render()


def test_gate_classification_inconsistency_fails_without_printing_it(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        payload["sites"][0]["permitted_claim_classification"] = "synthetic_wrong_label"
        payload["permitted_claim_classification"] = "synthetic_wrong_label"

    _rewrite_summary(capsule.run_dir, mutate)

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=gate_or_classification_mismatch" in report.render()
    assert "synthetic_wrong_label" not in report.render()


def test_structurally_unavailable_gate_is_conformant_and_not_disclosed(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)

    def mutate(payload):
        goldfield = payload["sites"][0]
        external_rows = goldfield["nested_block_bootstrap"]["external_intervals"]
        required_row = next(
            row
            for row in external_rows
            if row["covariance_scope"] == "full_scene_covariance"
            and row["scale"] == "L"
            and row["metric"] == "auc"
        )
        required_row["interval_available"] = False
        required_row["confirmatory_support"] = False
        required_row["lower_95"] = None
        goldfield["external_interval_available"] = False
        goldfield["external_pass"] = None
        goldfield["confirmatory_gate_available"] = False
        goldfield["confirmatory_gate_pass"] = None
        goldfield["permitted_claim_classification"] = "unavailable_required_evidence"
        goldfield["goldfield_alunite_gate_components"]["external_interval_gate"] = False
        payload["permitted_claim_classification"] = "unavailable_required_evidence"

    _rewrite_summary(capsule.run_dir, mutate)
    summary = json.loads((capsule.run_dir / "summary.json").read_text(encoding="utf-8"))
    (capsule.run_dir / "report.md").write_bytes(VERIFIER._expected_report(summary))

    report = _verify(capsule)

    assert report.passed
    assert "unavailable_required_evidence" not in report.render()


def test_detached_source_manifest_hash_is_mandatory(synthetic_capsule):
    report = VERIFIER.verify_run(
        run_dir=synthetic_capsule.run_dir,
        source_manifest=synthetic_capsule.source_manifest,
        block_manifest=synthetic_capsule.block_manifest,
        expected_source_manifest_sha256="0" * 64,
        expected_timing_pilot_sha256=synthetic_capsule.timing_sha256,
    )

    assert not report.passed
    assert "discrepancy=detached_hash_mismatch" in report.render()


@pytest.mark.parametrize("entry_count", [48, 50])
def test_source_manifest_requires_exact_v2_entry_count(synthetic_capsule, tmp_path, entry_count):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    lines = capsule.source_manifest.read_text(encoding="utf-8").splitlines()
    if entry_count < VERIFIER.EXPECTED_SOURCE_MANIFEST_ENTRIES:
        lines = lines[:entry_count]
    else:
        while len(lines) < entry_count:
            path = f"../tanager-spec/extra_{len(lines):02d}.py"
            lines.append(f"{_synthetic_hash(path)}  {path}")
    lines.sort(key=lambda line: line.split("  ", maxsplit=1)[1])
    count_capsule = _replace_source_manifest(capsule, ("\n".join(lines) + "\n").encode("utf-8"))

    report = _verify(count_capsule)

    assert not report.passed
    assert "FAIL check=source_manifest discrepancy=entry_count_mismatch" in report.render()


def test_detached_timing_hash_is_mandatory(synthetic_capsule):
    report = VERIFIER.verify_run(
        run_dir=synthetic_capsule.run_dir,
        source_manifest=synthetic_capsule.source_manifest,
        block_manifest=synthetic_capsule.block_manifest,
        expected_source_manifest_sha256=synthetic_capsule.source_sha256,
        expected_timing_pilot_sha256="0" * 64,
    )

    assert not report.passed
    assert "FAIL check=timing_pilot" in report.render()
    assert "discrepancy=detached_hash_mismatch" in report.render()


def test_frozen_preregistration_hash_is_mandatory(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    design_path = capsule.run_dir / "design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    design["protocol"]["sha256"] = "0" * 64
    design_path.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=frozen_hash_or_identity_mismatch" in report.render()


@pytest.mark.parametrize(
    "invalid_path",
    [
        "../other/member.py",
        "../tanager-spec/../escape.py",
        "/absolute/member.py",
        "src\\tanager_rocks\\config.py",
        "src/tanager_rocks/./config.py",
        "src/tanager_rocks/../escape.py",
        "tanager-rocks/src/tanager_rocks/config.py",
        "tanager-spec/src/tanager_spec/config.py",
    ],
)
def test_source_manifest_noncanonical_path_is_rejected(synthetic_capsule, tmp_path, invalid_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    manifest_text = capsule.source_manifest.read_text(encoding="utf-8")
    mutated = manifest_text.replace("../tanager-spec/synthetic_00.py", invalid_path, 1)
    assert mutated != manifest_text
    invalid_capsule = _replace_source_manifest(capsule, mutated.encode("utf-8"))

    report = _verify(invalid_capsule)

    assert not report.passed
    assert "discrepancy=unsafe_relative_path" in report.render()
    assert invalid_path not in report.render()


def test_source_manifest_unsorted_paths_are_rejected(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    lines = capsule.source_manifest.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    unsorted_capsule = _replace_source_manifest(capsule, ("\n".join(lines) + "\n").encode("utf-8"))

    report = _verify(unsorted_capsule)

    assert not report.passed
    assert "discrepancy=noncanonical_order" in report.render()


def test_source_manifest_duplicate_path_is_rejected(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    lines = capsule.source_manifest.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]
    duplicate_capsule = _replace_source_manifest(capsule, ("\n".join(lines) + "\n").encode("utf-8"))

    report = _verify(duplicate_capsule)

    assert not report.passed
    assert "discrepancy=duplicate_manifest_path" in report.render()


@pytest.mark.parametrize("newline_form", ["crlf", "missing_final_lf"])
def test_source_manifest_noncanonical_newline_is_rejected(
    synthetic_capsule, tmp_path, newline_form
):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    payload = capsule.source_manifest.read_bytes()
    if newline_form == "crlf":
        payload = payload.replace(b"\n", b"\r\n")
    else:
        payload = payload.removesuffix(b"\n")
    newline_capsule = _replace_source_manifest(capsule, payload)

    report = _verify(newline_capsule)

    assert not report.passed
    assert "discrepancy=noncanonical_newline" in report.render()


@pytest.mark.parametrize("line_form", ["one_space", "binary_marker", "uppercase_digest"])
def test_source_manifest_noncanonical_record_is_rejected(synthetic_capsule, tmp_path, line_form):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    lines = capsule.source_manifest.read_text(encoding="utf-8").splitlines()
    digest, path = lines[0].split("  ", maxsplit=1)
    if line_form == "one_space":
        lines[0] = f"{digest} {path}"
    elif line_form == "binary_marker":
        lines[0] = f"{digest} *{path}"
    else:
        lines[0] = f"{'A' * 64}  {path}"
    malformed_capsule = _replace_source_manifest(capsule, ("\n".join(lines) + "\n").encode("utf-8"))

    report = _verify(malformed_capsule)

    assert not report.passed
    assert "discrepancy=invalid_sha256sum_record" in report.render()


def test_json_numeric_overflow_is_rejected(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    timing_path = capsule.run_dir / "timing_pilot.json"
    payload = timing_path.read_text(encoding="utf-8")
    mutated = payload.replace('"wall_time_seconds": 1.0', '"wall_time_seconds": 1e999', 1)
    assert mutated != payload
    timing_path.write_text(mutated, encoding="utf-8")
    overflow_capsule = SyntheticCapsule(
        capsule.run_dir,
        capsule.source_manifest,
        capsule.block_manifest,
        capsule.source_sha256,
        _sha256_file(timing_path),
    )

    report = _verify(overflow_capsule)

    assert not report.passed
    assert "discrepancy=nonfinite_json_number" in report.render()


def test_duplicate_json_key_is_rejected(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    summary_path = capsule.run_dir / "summary.json"
    payload = summary_path.read_text(encoding="utf-8")
    mutated = payload.replace(
        '"schema_version": "1.0",',
        '"schema_version": "1.0",\n  "schema_version": "1.0",',
        1,
    )
    assert mutated != payload
    summary_path.write_text(mutated, encoding="utf-8")

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=duplicate_json_key" in report.render()


def test_explicit_block_manifest_must_match_the_design_hash(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    with capsule.block_manifest.open("ab") as handle:
        handle.write(b"\n")

    report = _verify(capsule)

    assert not report.passed
    assert "FAIL check=block_manifest" in report.render()
    assert "discrepancy=design_hash_binding_mismatch" in report.render()


def test_required_map_cannot_be_missing(synthetic_capsule, tmp_path):
    capsule = _copy_capsule(synthetic_capsule, tmp_path)
    missing = next(iter(sorted(VERIFIER._map_paths())))
    (capsule.run_dir / missing).unlink()

    report = _verify(capsule)

    assert not report.passed
    assert "discrepancy=required_file_missing" in report.render()


def test_run_directory_symlink_is_rejected(synthetic_capsule, tmp_path):
    linked_run = tmp_path / "linked_run"
    linked_run.symlink_to(synthetic_capsule.run_dir, target_is_directory=True)
    capsule = SyntheticCapsule(
        linked_run,
        synthetic_capsule.source_manifest,
        synthetic_capsule.block_manifest,
        synthetic_capsule.source_sha256,
        synthetic_capsule.timing_sha256,
    )

    report = _verify(capsule)

    assert not report.passed
    assert "FAIL check=run_directory" in report.render()


def test_cli_requires_an_explicit_run_directory():
    parser = VERIFIER.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--source-manifest",
                "source.sha256",
                "--block-manifest",
                "block_manifest.json",
                "--expected-source-manifest-sha256",
                "0" * 64,
                "--expected-timing-pilot-sha256",
                "0" * 64,
            ]
        )


def test_cli_help_describes_exact_v2_source_manifest():
    help_text = VERIFIER.build_parser().format_help()

    assert "49-entry v2 source" in help_text
    assert "40-entry" not in help_text
