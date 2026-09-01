#!/usr/bin/env python3
"""Run the frozen M3 Goldfield native-versus-Sentinel-2 sensor ablation.

This script is intentionally not part of the routine submission build.  It is
the confirmatory driver to run only after review and after the M2 block
manifest has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rioxarray
from tanager_spec.io import load_tanager_sr_hdf5
from tanager_spec.srf import load_s2_srf

from tanager_minmap.config import TANAGER_SR_ASSET, TARGET_MINERALS
from tanager_minmap.quality import mask_tanager_scene
from tanager_minmap.reference import MINERAL_TO_ROCKWELL
from tanager_minmap.sensor_ablation import (
    BOOTSTRAP_REPLICATES,
    CONFIRMATORY_STATUS,
    FDR_ALPHA,
    METRIC_NAMES,
    MIN_COVERAGE,
    PERMUTATION_REPLICATES,
    RIDGE,
    SEED,
    PairedEvaluation,
    block_designs_from_frame,
    compute_sensor_mtmf,
    confirmatory_bh_by_family,
    evaluate_score_pair,
    evaluate_sensor_pair,
    governed_metric_summary,
    paired_block_bootstrap,
    paired_sensor_auc_randomization,
    support_governance,
)
from tanager_minmap.speclib import load_library, select_endmembers

ROOT = Path(__file__).resolve().parents[1]
SCENE_ID = "20240925_185504_87_4001"
DEFAULT_SCENE = ROOT / "data" / "raw" / f"{SCENE_ID}_{TANAGER_SR_ASSET}.h5"
DEFAULT_REFERENCE = ROOT / "data" / "reference" / f"rockwell_goldfield_{SCENE_ID}.tif"
DEFAULT_SPECLIB = ROOT / "data" / "speclib" / "ASCIIdata_splib07a"
DEFAULT_INPUT_MANIFEST = ROOT / "docs" / "input_manifest.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "sensor_ablation"
PREREGISTRATION = ROOT / "docs" / "m3_sensor_ablation_preregistration.md"
M2_PREREGISTRATION = ROOT / "docs" / "m2_spatial_validation_preregistration.md"
SENSITIVITY_SENSORS = ("S2A", "S2B")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_block_rows(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if not isinstance(payload, dict):
        raise ValueError("JSON block manifest must be an object or list")
    for key in ("blocks", "records"):
        if isinstance(payload.get(key), list):
            return [dict(row) for row in payload[key]]
    geometries = payload.get("geometries")
    if isinstance(geometries, list):
        rows: list[dict[str, Any]] = []
        for geometry in geometries:
            if not isinstance(geometry, dict) or not isinstance(geometry.get("blocks"), list):
                raise ValueError("each JSON geometry must contain a blocks list")
            name = geometry.get("geometry", geometry.get("name"))
            halo = geometry.get("halo_pixels", geometry.get("r_site_pixels"))
            for block in geometry["blocks"]:
                row = dict(block)
                row.setdefault("geometry", name)
                row.setdefault("halo_pixels", halo)
                rows.append(row)
        return rows
    if isinstance(geometries, dict):
        rows = []
        for name, geometry in geometries.items():
            blocks = geometry.get("blocks") if isinstance(geometry, dict) else geometry
            if not isinstance(blocks, list):
                raise ValueError("each JSON geometry must resolve to a blocks list")
            halo = geometry.get("halo_pixels") if isinstance(geometry, dict) else None
            for block in blocks:
                row = dict(block)
                row.setdefault("geometry", name)
                row.setdefault("halo_pixels", halo)
                rows.append(row)
        return rows
    raise ValueError("JSON block manifest contains no blocks/records/geometries collection")


def load_block_manifest(path: Path) -> pd.DataFrame:
    """Read the frozen M2 block table."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.DataFrame(_json_block_rows(json.loads(path.read_text(encoding="utf-8"))))
    raise ValueError("block manifest must be CSV or JSON")


def _m2_manifest_inputs(path: Path, site: str, scene_id: str) -> tuple[pd.DataFrame, int, Path]:
    """Select the Goldfield M2 rows and recover the frozen halo from its summary."""
    frame = load_block_manifest(path)
    if "site" in frame:
        frame = frame.loc[frame["site"].astype(str) == site]
    if "scene_id" in frame:
        frame = frame.loc[frame["scene_id"].astype(str) == scene_id]
    if frame.empty:
        raise ValueError(f"M2 block manifest has no rows for {site}/{scene_id}")
    summary_path = path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"M2 summary with the frozen halo is required beside the block manifest: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_protocol = summary.get("protocol")
    if not isinstance(summary_protocol, dict) or summary_protocol.get("sha256") != sha256_file(
        M2_PREREGISTRATION
    ):
        raise ValueError("M2 summary protocol hash is stale or missing")
    if summary_protocol.get("protocol_compliant") is not True:
        raise ValueError("M2 summary was not generated with the frozen protocol parameters")
    matches = [
        record
        for record in summary.get("sites", [])
        if record.get("site") == site and record.get("scene_id") == scene_id
    ]
    if len(matches) != 1 or "halo_pixels" not in matches[0]:
        raise ValueError("M2 summary must contain exactly one Goldfield halo record")
    return frame, int(matches[0]["halo_pixels"]), summary_path


def resolve_block_manifest(explicit: Path | None) -> Path:
    """Require the authoritative JSON M2 block handoff explicitly."""
    if explicit is None:
        raise ValueError("--block-manifest is required for the frozen M3 analysis")
    if not explicit.is_file():
        raise FileNotFoundError(f"M2 block manifest not found: {explicit}")
    if explicit.suffix.lower() != ".json":
        raise ValueError("--block-manifest must name the authoritative JSON handoff")
    return explicit


def _validate_m2_manifest(
    path: Path,
    *,
    site: str,
    scene_id: str,
    shape: tuple[int, int],
    crs: str,
    transform: tuple[float, ...],
) -> None:
    """Reject stale or geometrically mismatched M2 block handoffs."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("sha256") != sha256_file(M2_PREREGISTRATION):
        raise ValueError("M2 block manifest protocol hash is stale or missing")
    site_entry = payload.get("sites", {}).get(site)
    if not isinstance(site_entry, dict) or site_entry.get("scene_id") != scene_id:
        raise ValueError("M2 block manifest does not name the frozen site/anchor scene")
    grid = site_entry.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("M2 block manifest site entry has no grid record")
    if tuple(grid.get("shape", ())) != tuple(shape):
        raise ValueError("M2 block manifest shape differs from the analysis grid")
    if grid.get("crs") != crs:
        raise ValueError("M2 block manifest CRS differs from the analysis grid")
    if tuple(grid.get("transform", ())) != tuple(transform):
        raise ValueError("M2 block manifest transform differs from the analysis grid")
    scales = site_entry.get("scales")
    if not isinstance(scales, dict) or not {"L", "2L"}.issubset(scales):
        raise ValueError("M2 block manifest lacks the frozen L and 2L scale records")
    for scale in ("L", "2L"):
        record = scales[scale]
        if not isinstance(record, dict):
            raise ValueError(f"M2 {scale} block record is invalid")
        raster_name = record.get("block_raster")
        expected_hash = record.get("block_raster_sha256")
        if not isinstance(raster_name, str) or not isinstance(expected_hash, str):
            raise ValueError(f"M2 {scale} block record lacks raster provenance")
        raster_path = path.parent / raster_name
        if not raster_path.is_file() or sha256_file(raster_path) != expected_hash:
            raise ValueError(f"M2 {scale} block raster is missing or has a stale hash")


def _input_by_id(manifest: dict[str, Any], input_id: str) -> dict[str, Any]:
    matches = [record for record in manifest.get("inputs", []) if record.get("id") == input_id]
    if len(matches) != 1:
        raise ValueError(f"input manifest must contain exactly one {input_id!r} record")
    return matches[0]


def _metric_rows(
    evaluation: PairedEvaluation,
    bootstrap: pd.DataFrame | None,
    *,
    endpoint: str,
    layer: str,
    geometry: str,
    comparator: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    governance = support_governance(
        evaluation.positive_blocks,
        evaluation.negative_blocks,
    )
    for sensor, result in (("native", evaluation.native), (comparator, evaluation.degraded)):
        bootstrap_prefix = "native" if sensor == "native" else "degraded"
        for metric, value in result.metrics.items():
            bootstrap_values = (
                None if bootstrap is None else bootstrap[f"{bootstrap_prefix}_{metric}"].to_numpy()
            )
            value, lower, upper = governed_metric_summary(
                value,
                bootstrap_values,
                governance,
            )
            rows.append(
                {
                    "endpoint": endpoint,
                    "layer": layer,
                    "geometry": geometry,
                    "comparator": comparator,
                    "sensor": sensor,
                    "metric": metric,
                    "value": value,
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "support_pixels": evaluation.support_pixels,
                    "evaluated_pixels": result.support_pixels,
                    "positive_blocks": evaluation.positive_blocks,
                    "negative_blocks": evaluation.negative_blocks,
                    "inference_status": governance.status,
                }
            )
    for metric in evaluation.native.metrics:
        delta = evaluation.native.metrics[metric] - evaluation.degraded.metrics[metric]
        bootstrap_values = None if bootstrap is None else bootstrap[f"delta_{metric}"].to_numpy()
        delta, lower, upper = governed_metric_summary(delta, bootstrap_values, governance)
        rows.append(
            {
                "endpoint": endpoint,
                "layer": layer,
                "geometry": geometry,
                "comparator": comparator,
                "sensor": "native_minus_degraded",
                "metric": metric,
                "value": delta,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
                "support_pixels": evaluation.support_pixels,
                "evaluated_pixels": evaluation.native.support_pixels,
                "positive_blocks": evaluation.positive_blocks,
                "negative_blocks": evaluation.negative_blocks,
                "inference_status": governance.status,
            }
        )
    return rows


def _bootstrap_rows(
    frame: pd.DataFrame,
    *,
    endpoint: str,
    layer: str,
    geometry: str,
    comparator: str,
) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "comparator", comparator)
    output.insert(0, "geometry", geometry)
    output.insert(0, "layer", layer)
    output.insert(0, "endpoint", endpoint)
    return output


def _permutation_rows(
    deltas: np.ndarray,
    *,
    layer: str,
    geometry: str,
    comparator: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "endpoint": "secondary_mtmf",
            "layer": layer,
            "geometry": geometry,
            "comparator": comparator,
            "permutation_replicate": np.arange(deltas.size, dtype=int),
            "permutation_delta_auc": deltas,
        }
    )


def _bootstrap_if_permitted(evaluation: PairedEvaluation) -> pd.DataFrame | None:
    """Return bootstrap draws only for tiers permitted to report intervals."""
    governance = support_governance(evaluation.positive_blocks, evaluation.negative_blocks)
    if not governance.bootstrap_cis:
        return None
    if evaluation.native.block_ids.size:
        return paired_block_bootstrap(evaluation)
    output: dict[str, np.ndarray] = {"replicate": np.arange(BOOTSTRAP_REPLICATES, dtype=int)}
    for metric in evaluation.native.metrics:
        output[f"native_{metric}"] = np.full(BOOTSTRAP_REPLICATES, np.nan)
        output[f"degraded_{metric}"] = np.full(BOOTSTRAP_REPLICATES, np.nan)
        output[f"delta_{metric}"] = np.full(BOOTSTRAP_REPLICATES, np.nan)
    return pd.DataFrame(output)


def _fold_rows(
    evaluation: PairedEvaluation,
    *,
    endpoint: str,
    layer: str,
    geometry: str,
    comparator: str,
) -> list[dict[str, Any]]:
    governance = support_governance(evaluation.positive_blocks, evaluation.negative_blocks)
    if not governance.effect_estimates:
        return []
    rows: list[dict[str, Any]] = []
    for sensor, result in (("native", evaluation.native), (comparator, evaluation.degraded)):
        for threshold, parameters in zip(result.thresholds, result.fold_parameters, strict=True):
            rows.append(
                {
                    "endpoint": endpoint,
                    "layer": layer,
                    "geometry": geometry,
                    "comparator": comparator,
                    "sensor": sensor,
                    "threshold": threshold,
                    **parameters,
                    "status": "available",
                }
            )
    for unavailable in evaluation.unavailable_folds:
        rows.append(
            {
                "endpoint": endpoint,
                "layer": layer,
                "geometry": geometry,
                "comparator": comparator,
                "sensor": "paired",
                "threshold": np.nan,
                **unavailable,
                "status": "unavailable",
            }
        )
    return rows


def _decision_summary(primary_rows: pd.DataFrame) -> dict[str, Any]:
    def row(comparator: str, geometry: str, metric: str) -> pd.Series | None:
        selected = primary_rows.loc[
            (primary_rows["comparator"] == comparator)
            & (primary_rows["geometry"] == geometry)
            & (primary_rows["sensor"] == "native_minus_degraded")
            & (primary_rows["metric"] == metric)
        ]
        return None if selected.empty else selected.iloc[0]

    def confirmatory(selected: pd.Series | None) -> bool:
        return bool(selected is not None and selected["inference_status"] == CONFIRMATORY_STATUS)

    s2a_l_auc = row("S2A", "L", "auc")
    s2a_l_bal = row("S2A", "L", "balanced_accuracy")
    s2a_2l_auc = row("S2A", "2L", "auc")
    s2b_l_auc = row("S2B", "L", "auc")
    checks = {
        "independent_block_support": bool(confirmatory(s2a_l_auc)),
        "s2a_auc_interval_above_zero": bool(
            confirmatory(s2a_l_auc) and s2a_l_auc["bootstrap_ci_lower"] > 0
        ),
        "s2a_balanced_accuracy_interval_above_zero": bool(
            confirmatory(s2a_l_bal) and s2a_l_bal["bootstrap_ci_lower"] > 0
        ),
        "s2a_auc_direction_positive_at_2l": bool(
            confirmatory(s2a_2l_auc) and s2a_2l_auc["value"] > 0
        ),
        "s2b_auc_direction_positive": bool(confirmatory(s2b_l_auc) and s2b_l_auc["value"] > 0),
    }
    return {"gate_passed": all(checks.values()), "checks": checks}


def run(args: argparse.Namespace) -> dict[str, Path]:
    """Execute the frozen analysis and write all tabular/provenance outputs."""
    block_manifest_path = resolve_block_manifest(args.block_manifest)
    input_manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    scene_record = _input_by_id(input_manifest, "tanager-goldfield-1")
    if Path(str(scene_record["logical_path"])).name != args.scene.name:
        raise ValueError("scene path does not match tanager-goldfield-1 in the input manifest")

    cube_raw, wavelengths = load_tanager_sr_hdf5(args.scene)
    cube, quality = mask_tanager_scene(cube_raw, wavelengths, args.scene)
    if quality.retained_bands != 363:
        raise RuntimeError(
            f"quality policy retained {quality.retained_bands} bands; preregistration requires 363"
        )
    endmembers = select_endmembers(load_library(args.speclib, wavelengths))
    missing_minerals = sorted(set(TARGET_MINERALS) - set(endmembers))
    if missing_minerals:
        raise RuntimeError(f"missing frozen endmembers: {', '.join(missing_minerals)}")

    srfs = {sensor: load_s2_srf(sensor) for sensor in SENSITIVITY_SENSORS}
    srf_provenance: dict[str, dict[str, Any]] = {}
    for sensor in SENSITIVITY_SENSORS:
        manifest_record = _input_by_id(input_manifest, f"tanager-spec-{sensor.lower()}-srf")
        resource = Path(str(files("tanager_spec").joinpath("data", f"{sensor}_SRF.csv")))
        observed_hash = sha256_file(resource)
        if observed_hash != manifest_record["sha256"]:
            raise RuntimeError(f"{sensor} SRF hash differs from the frozen input manifest")
        srf_provenance[sensor] = {
            "input_manifest_record": manifest_record,
            "resolved_path": str(resource),
            "observed_sha256": observed_hash,
        }

    sensor_scores = compute_sensor_mtmf(
        cube,
        wavelengths,
        endmembers,
        srfs,
        ridge=RIDGE,
        min_coverage=MIN_COVERAGE,
    )
    reference_raw = rioxarray.open_rasterio(args.reference, masked=False).squeeze("band", drop=True)
    if (
        reference_raw.shape != cube.shape[1:]
        or reference_raw.rio.crs != cube.rio.crs
        or reference_raw.rio.transform() != cube.rio.transform()
    ):
        raise ValueError(
            "corrected Rockwell reference is not exactly aligned to the Goldfield cube"
        )
    reference = reference_raw.values
    if cube.rio.crs is None:
        raise ValueError("Goldfield cube has no CRS")
    cube_transform = cube.rio.transform()
    _validate_m2_manifest(
        block_manifest_path,
        site="goldfield",
        scene_id=SCENE_ID,
        shape=reference.shape,
        crs=cube.rio.crs.to_string(),
        transform=(
            float(cube_transform.a),
            float(cube_transform.b),
            float(cube_transform.c),
            float(cube_transform.d),
            float(cube_transform.e),
            float(cube_transform.f),
        ),
    )
    block_frame, halo_pixels, m2_summary_path = _m2_manifest_inputs(
        block_manifest_path, "goldfield", SCENE_ID
    )
    designs = block_designs_from_frame(
        block_frame,
        reference.shape,
        halo_pixels=halo_pixels,
        manifest_contains_only_complete_blocks=True,
    )
    missing_geometries = {"L", "2L"} - set(designs)
    if missing_geometries:
        raise ValueError(
            "M2 block manifest lacks preregistered geometries: "
            + ", ".join(sorted(missing_geometries))
        )

    primary_metric_rows: list[dict[str, Any]] = []
    secondary_metric_rows: list[dict[str, Any]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    permutation_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    secondary_family_members: list[tuple[int, float, str, str]] = []

    for comparator in SENSITIVITY_SENSORS:
        for geometry in ("L", "2L"):
            paired = evaluate_sensor_pair(
                sensor_scores["native"]["alunite_mf"].values,
                sensor_scores["native"]["kaolinite_mf"].values,
                sensor_scores[comparator]["alunite_mf"].values,
                sensor_scores[comparator]["kaolinite_mf"].values,
                reference,
                designs[geometry],
            )
            boot = _bootstrap_if_permitted(paired)
            primary_metric_rows.extend(
                _metric_rows(
                    paired,
                    boot,
                    endpoint="primary_margin",
                    layer="alunite_minus_kaolinite",
                    geometry=geometry,
                    comparator=comparator,
                )
            )
            if boot is not None:
                bootstrap_frames.append(
                    _bootstrap_rows(
                        boot,
                        endpoint="primary_margin",
                        layer="alunite_minus_kaolinite",
                        geometry=geometry,
                        comparator=comparator,
                    )
                )
            fold_rows.extend(
                _fold_rows(
                    paired,
                    endpoint="primary_margin",
                    layer="alunite_minus_kaolinite",
                    geometry=geometry,
                    comparator=comparator,
                )
            )

            for mineral in TARGET_MINERALS:
                if mineral not in MINERAL_TO_ROCKWELL:
                    secondary_metric_rows.append(
                        {
                            "endpoint": "secondary_mtmf",
                            "layer": mineral,
                            "geometry": geometry,
                            "comparator": comparator,
                            "sensor": "not_evaluated",
                            "metric": "auc",
                            "value": np.nan,
                            "bootstrap_ci_lower": np.nan,
                            "bootstrap_ci_upper": np.nan,
                            "support_pixels": 0,
                            "evaluated_pixels": 0,
                            "positive_blocks": 0,
                            "negative_blocks": 0,
                            "inference_status": "unavailable_no_rockwell_mapping",
                            "permutation_p": np.nan,
                            "permutation_bh_q": np.nan,
                            "permutation_bh_reject_q_0_05": pd.NA,
                            "permutation_randomizations": np.nan,
                            "permutation_exceedances": np.nan,
                        }
                    )
                    continue
                secondary = evaluate_score_pair(
                    sensor_scores["native"][f"{mineral}_mf"].values,
                    sensor_scores[comparator][f"{mineral}_mf"].values,
                    reference,
                    designs[geometry],
                    positive_classes=MINERAL_TO_ROCKWELL[mineral],
                )
                secondary_governance = support_governance(
                    secondary.positive_blocks,
                    secondary.negative_blocks,
                )
                secondary_boot = _bootstrap_if_permitted(secondary)
                start = len(secondary_metric_rows)
                rows = _metric_rows(
                    secondary,
                    secondary_boot,
                    endpoint="secondary_mtmf",
                    layer=mineral,
                    geometry=geometry,
                    comparator=comparator,
                )
                randomization = None
                if secondary_governance.permutation_inference:
                    randomization = paired_sensor_auc_randomization(
                        sensor_scores["native"][f"{mineral}_mf"].values,
                        sensor_scores[comparator][f"{mineral}_mf"].values,
                        reference,
                        designs[geometry],
                        positive_classes=MINERAL_TO_ROCKWELL[mineral],
                        randomizations=PERMUTATION_REPLICATES,
                        seed=SEED,
                    )
                delta_auc = next(
                    row["value"]
                    for row in rows
                    if row["sensor"] == "native_minus_degraded" and row["metric"] == "auc"
                )
                if randomization is not None and not np.isclose(
                    randomization.observed_delta,
                    delta_auc,
                    equal_nan=True,
                ):
                    raise RuntimeError(
                        "secondary AUC metric and randomization do not share fixed support"
                    )
                p_value = np.nan if randomization is None else randomization.p_value
                for row in rows:
                    is_delta_auc = (
                        row["sensor"] == "native_minus_degraded" and row["metric"] == "auc"
                    )
                    row["permutation_p"] = p_value if is_delta_auc else np.nan
                    row["permutation_bh_q"] = np.nan
                    row["permutation_bh_reject_q_0_05"] = pd.NA
                    row["permutation_randomizations"] = (
                        randomization.randomizations
                        if is_delta_auc and randomization is not None
                        else np.nan
                    )
                    row["permutation_exceedances"] = (
                        randomization.exceedances
                        if is_delta_auc and randomization is not None
                        else np.nan
                    )
                secondary_metric_rows.extend(rows)
                fold_rows.extend(
                    _fold_rows(
                        secondary,
                        endpoint="secondary_mtmf",
                        layer=mineral,
                        geometry=geometry,
                        comparator=comparator,
                    )
                )
                delta_auc_index = next(
                    index
                    for index in range(start, len(secondary_metric_rows))
                    if secondary_metric_rows[index]["sensor"] == "native_minus_degraded"
                    and secondary_metric_rows[index]["metric"] == "auc"
                )
                secondary_family_members.append(
                    (
                        delta_auc_index,
                        p_value,
                        comparator,
                        secondary_governance.status,
                    )
                )
                if randomization is not None:
                    permutation_frames.append(
                        _permutation_rows(
                            randomization.permuted_deltas,
                            layer=mineral,
                            geometry=geometry,
                            comparator=comparator,
                        )
                    )
                if secondary_boot is not None:
                    bootstrap_frames.append(
                        _bootstrap_rows(
                            secondary_boot,
                            endpoint="secondary_mtmf",
                            layer=mineral,
                            geometry=geometry,
                            comparator=comparator,
                        )
                    )

    adjusted = confirmatory_bh_by_family(
        np.asarray([p_value for _, p_value, _, _ in secondary_family_members]),
        np.asarray([comparator for _, _, comparator, _ in secondary_family_members]),
        np.asarray([status for _, _, _, status in secondary_family_members]),
    )
    for (index, _, _, _), q_value in zip(secondary_family_members, adjusted, strict=True):
        secondary_metric_rows[index]["permutation_bh_q"] = q_value
        secondary_metric_rows[index]["permutation_bh_reject_q_0_05"] = (
            bool(q_value <= FDR_ALPHA) if np.isfinite(q_value) else pd.NA
        )

    args.output.mkdir(parents=True, exist_ok=True)
    primary_path = args.output / "primary_metrics.csv"
    secondary_path = args.output / "secondary_metrics.csv"
    bootstrap_path = args.output / "bootstrap_ci_distributions.csv"
    permutation_path = args.output / "secondary_permutation_deltas.csv"
    folds_path = args.output / "folds.csv"
    summary_path = args.output / "summary.json"
    provenance_path = args.output / "provenance.json"
    primary_frame = pd.DataFrame(primary_metric_rows)
    primary_frame.to_csv(primary_path, index=False, na_rep="NA")
    pd.DataFrame(secondary_metric_rows).to_csv(secondary_path, index=False, na_rep="NA")
    bootstrap_columns = ["endpoint", "layer", "geometry", "comparator", "replicate"]
    for metric in METRIC_NAMES:
        bootstrap_columns.extend([f"native_{metric}", f"degraded_{metric}", f"delta_{metric}"])
    bootstrap_frame = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames
        else pd.DataFrame(columns=bootstrap_columns)
    )
    bootstrap_frame.to_csv(bootstrap_path, index=False, na_rep="NA")
    permutation_columns = [
        "endpoint",
        "layer",
        "geometry",
        "comparator",
        "permutation_replicate",
        "permutation_delta_auc",
    ]
    permutation_frame = (
        pd.concat(permutation_frames, ignore_index=True)
        if permutation_frames
        else pd.DataFrame(columns=permutation_columns)
    )
    permutation_frame.to_csv(permutation_path, index=False, na_rep="NA")
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False, na_rep="NA")
    summary = {
        "scene_id": SCENE_ID,
        "primary_endpoint": "class 3 advanced argillic vs class 4 argillic",
        "decision": _decision_summary(primary_frame),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "secondary_randomizations": PERMUTATION_REPLICATES,
        "seed": SEED,
        "negative_control": {
            "pair": "jarosite_goethite",
            "scope": "existing library-level result; no scene-level class claim",
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "preregistration": {
            "path": str(PREREGISTRATION.relative_to(ROOT)),
            "sha256": sha256_file(PREREGISTRATION),
        },
        "input_manifest": {
            "path": str(args.input_manifest),
            "sha256": sha256_file(args.input_manifest),
            "scene_record": scene_record,
        },
        "inputs": {
            "scene": str(args.scene),
            "reference": str(args.reference),
            "reference_sha256": sha256_file(args.reference),
            "speclib": str(args.speclib),
            "block_manifest": str(block_manifest_path),
            "block_manifest_sha256": sha256_file(block_manifest_path),
            "m2_summary": str(m2_summary_path),
            "m2_summary_sha256": sha256_file(m2_summary_path),
            "srf": srf_provenance,
        },
        "fixed_parameters": {
            "ridge": RIDGE,
            "min_coverage": MIN_COVERAGE,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "secondary_randomizations": PERMUTATION_REPLICATES,
            "seed": SEED,
            "positive_class": 3,
            "negative_class": 4,
            "quality_policy": "tanager_minmap.quality.mask_tanager_scene",
            "retained_bands": quality.retained_bands,
            "mtmf_covariance_scope": "full_scene_label_free_per_sensor",
            "secondary_fdr": {
                "families": ["S2A", "S2B"],
                "family_scope": (
                    "confirmatory mapped MTMF endpoints at L and 2L within comparator"
                ),
                "p_value": (
                    "paired within-complete-block sensor-label randomization of AUC delta; "
                    "two-sided add-one"
                ),
                "alpha": FDR_ALPHA,
            },
            "support_governance": {
                "confirmatory": (
                    ">=10 positive-bearing and >=10 negative-bearing complete blocks; "
                    "effects, bootstrap CIs, permutation p-values, and BH q-values"
                ),
                "exploratory": ("5-9 blocks in the limiting class; effects and bootstrap CIs only"),
                "counts_maps_only": (
                    "<5 blocks in the limiting class; counts and support only in tabular outputs"
                ),
            },
        },
        "endmember_samples": {
            mineral: endmember.sample for mineral, endmember in endmembers.items()
        },
        "degradation_factors": {
            "spectral_response_convolution": True,
            "spatial_point_spread": False,
            "spatial_resampling": False,
            "radiometric_noise": False,
            "quantization": False,
            "atmospheric_difference": False,
        },
        "software_versions": {
            package: metadata.version(package)
            for package in ("tanager-minmap", "tanager-spec", "numpy", "pandas", "xarray")
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "primary_metrics": primary_path,
        "secondary_metrics": secondary_path,
        "bootstrap_ci_distributions": bootstrap_path,
        "secondary_permutation_deltas": permutation_path,
        "folds": folds_path,
        "summary": summary_path,
        "provenance": provenance_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--speclib", type=Path, default=DEFAULT_SPECLIB)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument(
        "--block-manifest",
        type=Path,
        required=True,
        help="authoritative data/processed/spatial_validation/block_manifest.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    outputs = run(parse_args())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
