"""Preregistered E6 MTMF ensemble design, execution, and aggregation.

The module keeps scientific controls separate from compute controls.  The
default design is the finite, paired sensitivity ensemble frozen in
``docs/m2_ensemble_sensitivity_preregistration.md``.  It deliberately exposes
small NumPy functions so the design, failure handling, bootstrap nesting, and
checkpoint behavior can be verified without loading either anchor scene.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.resources
import json
import math
import os
import subprocess
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .config import TARGET_MINERALS

FROZEN_PREREGISTRATION_SHA256 = "4c228fac93828d039c36b535331dce36411717e38c68ef2fc1a355b73fdacb22"
FROZEN_SITES = ("goldfield", "bingham")
ANCHOR_SCENES = {
    "goldfield": "20240925_185504_87_4001",
    "bingham": "20250911_191523_58_4001",
}
FROZEN_RIDGES = (0.001, 0.01, 0.1)
FROZEN_QUANTILES = (0.85, 0.90, 0.95)
FROZEN_GATES: tuple[float | None, ...] = (None, 1.0)
FROZEN_STOCHASTIC_REPLICATES = 16
FROZEN_BOOTSTRAP_REPLICATES = 10_000
FROZEN_SEED = 42
FROZEN_RETAINED_BANDS = 363
FROZEN_ANALYTICAL_CELLS = 18
FROZEN_VARIANTS_PER_SCENE = 355
FROZEN_UNIQUE_FITS_PER_SCENE = 83
FINITE_REPLICATE_FRACTION = 0.95
MINIMUM_INTERVAL_BLOCKS = 2
CONFIRMATORY_POSITIVE_BLOCKS = 10
CONFIRMATORY_NEGATIVE_BLOCKS = 10
EXPLORATORY_BLOCKS = 5

AMENDMENT_SCHEMA_VERSION = "1.0"
AMENDMENT_TYPE = "e6_pre_result_protocol_amendment"
GOVERNING_FILES = (
    "src/tanager_minmap/ensemble_sensitivity.py",
    "scripts/run_ensemble_sensitivity.py",
    "tests/test_ensemble_sensitivity.py",
    "docs/m2_ensemble_sensitivity_preregistration.md",
    "docs/m2_spatial_validation_preregistration.md",
    "docs/tanager_quality_mask_policy.md",
    "src/tanager_minmap/spatial_validation.py",
    "src/tanager_minmap/strict_inductive.py",
    "src/tanager_minmap/unmix.py",
    "src/tanager_minmap/quality.py",
    "src/tanager_minmap/speclib.py",
    "src/tanager_minmap/reference.py",
    "src/tanager_minmap/config.py",
    "src/tanager_minmap/pipeline.py",
    "src/tanager_minmap/viz.py",
)

EXPECTED_CANDIDATE_COUNTS = {
    "alunite": 11,
    "kaolinite": 8,
    "dickite": 2,
    "jarosite": 9,
    "hematite": 12,
    "goethite": 8,
    "gypsum": 6,
    "muscovite": 16,
}
BASELINE_ENDMEMBERS = {
    "alunite": "splib07a_Alunite_SUSTDA-20_BECKb_AREF.txt",
    "kaolinite": "splib07a_Kaolinite_CM5_BECKb_AREF.txt",
    "dickite": "splib07a_Dickite_NMNH106242_BECKb_AREF.txt",
    "jarosite": "splib07a_Jarosite_GDS101_Na_200C_Syn_BECKa_AREF.txt",
    "hematite": "splib07a_Hematite_GDS69.e_20-30um_BECKb_AREF.txt",
    "goethite": "splib07a_Goethite_MPCMA2-C_M-Crsgrad2_BECKb_AREF.txt",
    "gypsum": "splib07a_Gypsum_HS333.2B_(Selenite)_ASDFRa_AREF.txt",
    "muscovite": "splib07a_Muscovite_GDS118_Capitan_BECKa_AREF.txt",
}

CONFIDENCE_NODATA = -1
CONFIDENCE_STABLE_NEGATIVE = 0
CONFIDENCE_CHOICE_SENSITIVE = 1
CONFIDENCE_STABLE_POSITIVE = 2
DOMINANT_NODATA = -2
DOMINANT_NONE = -1

_MUTABLE_MEMBER_FIELDS = frozenset(
    {
        "status",
        "failure_reason",
        "contributing_pixels",
        "retained_bands",
        "output_checksum",
        "wall_time_seconds",
        "peak_memory_bytes",
    }
)


class ProtocolError(ValueError):
    """Raised when execution would depart from the frozen protocol."""


class FitFailure(RuntimeError):
    """A preregistered MTMF member failure that must not be rescued scientifically."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a bounded-memory SHA-256 digest for one regular file."""
    if not path.is_file():
        raise FileNotFoundError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def strict_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write standards-compliant JSON, mapping non-finite values to null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _compact_json(value: Any) -> str:
    return json.dumps(_json_safe(value), separators=(",", ":"), sort_keys=True, allow_nan=False)


def validate_protocol_amendment(
    path: Path,
    *,
    expected_changes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an explicitly authorized, dated, pre-result E6 amendment.

    The amendment is JSON so authorization and the exact scientific changes
    are machine-checkable. Free-form or merely non-empty files fail closed.
    """
    if not path.is_file():
        raise ProtocolError("protocol amendment must be an existing JSON file")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ProtocolError(f"protocol amendment contains duplicate field {key!r}")
            parsed[key] = value
        return parsed

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("protocol amendment must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ProtocolError("protocol amendment must contain one JSON object")
    required_values = {
        "schema_version": AMENDMENT_SCHEMA_VERSION,
        "amendment_type": AMENDMENT_TYPE,
        "authorized": True,
        "pre_result": True,
        "results_seen": False,
    }
    for field, expected in required_values.items():
        if _compact_json(payload.get(field)) != _compact_json(expected):
            raise ProtocolError(f"protocol amendment field {field!r} must equal {expected!r}")
    amendment_date = payload.get("amendment_date")
    if not isinstance(amendment_date, str):
        raise ProtocolError("protocol amendment requires amendment_date in YYYY-MM-DD form")
    try:
        parsed_date = date.fromisoformat(amendment_date)
    except ValueError as error:
        raise ProtocolError(
            "protocol amendment requires amendment_date in YYYY-MM-DD form"
        ) from error
    if parsed_date > date.today():
        raise ProtocolError("protocol amendment date cannot be in the future")
    for field in ("authorized_by", "authorization_basis", "rationale"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ProtocolError(f"protocol amendment requires non-empty {field}")
    changes = payload.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise ProtocolError("protocol amendment requires a non-empty changes object")
    if expected_changes is None:
        raise ProtocolError("protocol amendment supplied when no governed changes are expected")
    for key, expected in expected_changes.items():
        if key not in changes:
            raise ProtocolError(f"protocol amendment does not authorize change {key!r}")
        if _compact_json(changes[key]) != _compact_json(expected):
            raise ProtocolError(f"protocol amendment change {key!r} does not match execution")
    extra_changes = set(changes).difference(expected_changes)
    if extra_changes:
        raise ProtocolError(
            "protocol amendment contains unexecuted changes: "
            + ", ".join(repr(key) for key in sorted(extra_changes))
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "amendment_date": amendment_date,
        "authorized_by": payload["authorized_by"],
        "authorization_basis": payload["authorization_basis"],
        "changes": changes,
    }


def validate_protocol_file(
    preregistration: Path,
    *,
    expected_sha256: str = FROZEN_PREREGISTRATION_SHA256,
    protocol_amendment: Path | None = None,
) -> dict[str, Any]:
    """Validate the preregistration bytes or require and record an amendment."""
    observed = sha256_file(preregistration)
    compliant = observed == expected_sha256
    if not compliant and protocol_amendment is None:
        raise ProtocolError(
            "preregistration hash differs from the frozen implementation hash; "
            "supply --protocol-amendment to record the deviation"
        )
    amendment_record = None
    if protocol_amendment is not None:
        expected_changes = None
        if not compliant:
            expected_changes = {
                "preregistration_sha256": {
                    "expected": expected_sha256,
                    "observed": observed,
                }
            }
        amendment_record = validate_protocol_amendment(
            protocol_amendment,
            expected_changes=expected_changes,
        )
    return {
        "path": str(preregistration),
        "sha256": observed,
        "expected_sha256": expected_sha256,
        "protocol_compliant": compliant,
        "amendment": amendment_record,
    }


def _gate_label(value: float | None) -> str:
    return "none" if value is None else format(float(value), ".12g")


def _as_gate(value: Any) -> float | None:
    if value is None or str(value).lower() == "none":
        return None
    return float(value)


def validate_protocol_arguments(args: Any) -> dict[str, dict[str, Any]]:
    """Refuse scientific CLI deviations unless an amendment path is supplied."""
    observed = {
        "sites": tuple(args.sites),
        "ridge": tuple(float(value) for value in args.ridge),
        "detection_quantiles": tuple(float(value) for value in args.detection_quantiles),
        "infeasibility_gates": tuple(_as_gate(value) for value in args.infeasibility_gates),
        "stochastic_replicates": int(args.stochastic_replicates),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
    }
    expected = {
        "sites": FROZEN_SITES,
        "ridge": FROZEN_RIDGES,
        "detection_quantiles": FROZEN_QUANTILES,
        "infeasibility_gates": FROZEN_GATES,
        "stochastic_replicates": FROZEN_STOCHASTIC_REPLICATES,
        "bootstrap_replicates": FROZEN_BOOTSTRAP_REPLICATES,
        "seed": FROZEN_SEED,
    }
    deviations = {
        key: {"expected": expected[key], "observed": value}
        for key, value in observed.items()
        if value != expected[key]
    }
    amendment = getattr(args, "protocol_amendment", None)
    if deviations and amendment is None:
        flags = ", ".join(f"--{key.replace('_', '-')}" for key in deviations)
        raise ProtocolError(
            f"frozen protocol deviation in {flags}; supply --protocol-amendment to proceed"
        )
    if amendment is not None:
        expected_changes = {"scientific_cli": deviations} if deviations else None
        validate_protocol_amendment(amendment, expected_changes=expected_changes)
    return deviations


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ProtocolError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def _resolve_manifest_input(root: Path, logical_path: str) -> Path:
    if logical_path.startswith("package:tanager_spec/"):
        relative = logical_path.removeprefix("package:tanager_spec/")
        return Path(str(importlib.resources.files("tanager_spec").joinpath(relative)))
    path = (root / logical_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ProtocolError(f"input manifest path escapes repository: {logical_path}") from error
    return path


def validate_input_manifest(path: Path, *, root: Path) -> dict[str, Any]:
    """Verify every declared scientific input byte hash and size."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_equal("input manifest hash algorithm", payload.get("hash_algorithm"), "sha256")
    records = payload.get("inputs")
    if not isinstance(records, list) or not records:
        raise ProtocolError("input manifest has no inputs")
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ProtocolError("input manifest contains a non-object record")
        input_id = str(record.get("id", ""))
        logical_path = str(record.get("logical_path", ""))
        if not input_id or input_id in seen or not logical_path:
            raise ProtocolError("input manifest IDs must be non-empty and unique")
        seen.add(input_id)
        resolved = _resolve_manifest_input(root, logical_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"declared input is missing: {logical_path}")
        _require_equal(f"{input_id} size", resolved.stat().st_size, int(record["size_bytes"]))
        observed_hash = sha256_file(resolved)
        _require_equal(f"{input_id} SHA-256", observed_hash, record.get("sha256"))
        verified.append(
            {
                "id": input_id,
                "logical_path": logical_path,
                "size_bytes": resolved.stat().st_size,
                "sha256": observed_hash,
            }
        )
    return {"path": str(path), "sha256": sha256_file(path), "inputs": verified}


@dataclass(frozen=True)
class BlockScaleHandoff:
    """One validated complete-block raster at a frozen M2 scale."""

    scale: str
    block_ids: tuple[int, ...]
    raster_path: Path
    raster_sha256: str
    halo_pixels: int


@dataclass(frozen=True)
class BlockManifestSite:
    """Validated L and 2L block handoff for one frozen anchor."""

    site: str
    scene: str
    block_ids: tuple[int, ...]
    raster_path: Path
    raster_sha256: str
    shape: tuple[int, int]
    crs: str
    transform: tuple[float, ...]
    halo_pixels: int
    scales: Mapping[str, BlockScaleHandoff]


def validate_m2_manifest(
    path: Path,
    *,
    m2_preregistration: Path,
    anchors: Mapping[str, str] = ANCHOR_SCENES,
    expected_grids: Mapping[str, Mapping[str, Any]] | None = None,
    require_stochastic_blocks: bool = True,
) -> dict[str, BlockManifestSite]:
    """Validate current M2 protocol, every raster hash, anchor, and grid."""
    import rasterio
    from affine import Affine

    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_equal(
        "M2 manifest type", payload.get("manifest_type"), "spatial_validation_complete_blocks"
    )
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ProtocolError("M2 block manifest has no protocol record")
    _require_equal(
        "M2 manifest protocol path",
        protocol.get("path"),
        "docs/m2_spatial_validation_preregistration.md",
    )
    _require_equal(
        "M2 manifest protocol hash", protocol.get("sha256"), sha256_file(m2_preregistration)
    )
    site_payload = payload.get("sites")
    if not isinstance(site_payload, dict):
        raise ProtocolError("M2 block manifest has no sites object")
    validated: dict[str, BlockManifestSite] = {}
    for site, scene in anchors.items():
        entry = site_payload.get(site)
        if not isinstance(entry, dict):
            raise ProtocolError(f"M2 block manifest has no {site} site entry")
        _require_equal(f"{site} anchor scene", entry.get("scene_id"), scene)
        grid = entry.get("grid")
        if not isinstance(grid, dict):
            raise ProtocolError(f"M2 block manifest has no {site} grid")
        shape_values = grid.get("shape")
        transform_values = grid.get("transform")
        if not isinstance(shape_values, list) or len(shape_values) != 2:
            raise ProtocolError(f"{site} grid shape is invalid")
        if not isinstance(transform_values, list) or len(transform_values) != 6:
            raise ProtocolError(f"{site} grid transform is invalid")
        shape = (int(shape_values[0]), int(shape_values[1]))
        transform = tuple(float(value) for value in transform_values)
        crs = str(grid.get("crs"))
        scales = entry.get("scales")
        if not isinstance(scales, dict) or set(scales) != {"L", "2L"}:
            raise ProtocolError(f"{site} must contain exactly the frozen L and 2L scales")
        scale_handoffs: dict[str, BlockScaleHandoff] = {}
        for scale in ("L", "2L"):
            scale_entry = scales[scale]
            if not isinstance(scale_entry, dict):
                raise ProtocolError(f"{site}/{scale} scale record is invalid")
            _require_equal(f"{site}/{scale} anchor", scale_entry.get("anchor_scene_id"), scene)
            raster_name = scale_entry.get("block_raster")
            raster_hash = scale_entry.get("block_raster_sha256")
            if not isinstance(raster_name, str) or not isinstance(raster_hash, str):
                raise ProtocolError(f"{site}/{scale} lacks block-raster provenance")
            raster_path = path.parent / raster_name
            _require_equal(
                f"{site}/{scale} block raster hash", sha256_file(raster_path), raster_hash
            )
            complete_ids = tuple(int(value) for value in scale_entry.get("complete_block_ids", ()))
            if len(set(complete_ids)) != len(complete_ids) or any(
                value <= 0 for value in complete_ids
            ):
                raise ProtocolError(
                    f"{site}/{scale} complete block IDs are not unique positive integers"
                )
            _require_equal(
                f"{site}/{scale} complete block count",
                scale_entry.get("complete_blocks"),
                len(complete_ids),
            )
            with rasterio.open(raster_path) as dataset:
                _require_equal(f"{site}/{scale} raster shape", dataset.shape, shape)
                if dataset.crs is None:
                    raise ProtocolError(f"{site}/{scale} block raster has no CRS")
                _require_equal(f"{site}/{scale} raster CRS", dataset.crs.to_string(), crs)
                _require_equal(
                    f"{site}/{scale} raster transform", dataset.transform, Affine(*transform)
                )
                _require_equal(f"{site}/{scale} raster nodata", dataset.nodata, 0.0)
                _require_equal(f"{site}/{scale} raster dtype", dataset.dtypes, ("uint32",))
                values = dataset.read(1, masked=False)
            raster_ids = tuple(sorted(int(value) for value in np.unique(values) if value > 0))
            _require_equal(f"{site}/{scale} raster IDs", raster_ids, tuple(sorted(complete_ids)))
            scale_handoffs[scale] = BlockScaleHandoff(
                scale=scale,
                block_ids=complete_ids,
                raster_path=raster_path,
                raster_sha256=raster_hash,
                halo_pixels=int(scale_entry["halo_pixels"]),
            )
            if scale == "L":
                _require_equal(f"{site} primary raster", entry.get("block_raster"), raster_name)
                _require_equal(
                    f"{site} primary complete IDs",
                    tuple(entry.get("complete_block_ids", ())),
                    complete_ids,
                )
                if require_stochastic_blocks and len(complete_ids) < 2:
                    raise ProtocolError(
                        f"{site} stochastic axes require at least two unique complete L blocks"
                    )
        primary_scale = scale_handoffs.get("L")
        if primary_scale is None:
            raise AssertionError("primary L validation did not produce a handoff")
        primary = BlockManifestSite(
            site=site,
            scene=scene,
            block_ids=primary_scale.block_ids,
            raster_path=primary_scale.raster_path,
            raster_sha256=primary_scale.raster_sha256,
            shape=shape,
            crs=crs,
            transform=transform,
            halo_pixels=primary_scale.halo_pixels,
            scales=scale_handoffs,
        )
        if expected_grids is not None:
            expected = expected_grids[site]
            _require_equal(f"{site} anchor grid shape", primary.shape, tuple(expected["shape"]))
            _require_equal(f"{site} anchor grid CRS", primary.crs, str(expected["crs"]))
            _require_equal(
                f"{site} anchor grid transform",
                primary.transform,
                tuple(float(value) for value in expected["transform"]),
            )
        validated[site] = primary
    return validated


def balanced_endmember_schedules(
    candidates: Mapping[str, Sequence[str]],
    *,
    replicates: int = FROZEN_STOCHASTIC_REPLICATES,
    seed: int = FROZEN_SEED,
) -> tuple[dict[str, str], ...]:
    """Return seeded schedules with exact floor/ceiling candidate frequencies.

    Each mineral receives an equal base count plus a seeded random allocation
    of the remainder, followed by a seeded shuffle.  This implements the
    preregistered floor/ceiling balance without filename-order bias.
    """
    if replicates <= 0:
        raise ProtocolError("stochastic replicates must be positive")
    by_mineral: dict[str, list[str]] = {}
    for mineral_index, mineral in enumerate(TARGET_MINERALS):
        population = sorted(str(item) for item in candidates.get(mineral, ()))
        if not population:
            raise ProtocolError(f"no eligible measured spectrum for {mineral}")
        if len(population) != len(set(population)):
            raise ProtocolError(f"duplicate candidate filename for {mineral}")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, mineral_index])))
        quotient, remainder = divmod(replicates, len(population))
        repeated_items = [item for item in population for _ in range(quotient)]
        if remainder:
            extra_indices = rng.permutation(len(population))[:remainder]
            repeated_items.extend(population[int(index)] for index in extra_indices)
        repeated = np.asarray(repeated_items, dtype=object)
        rng.shuffle(repeated)
        by_mineral[mineral] = [str(item) for item in repeated.tolist()]
        frequencies = Counter(by_mineral[mineral])
        if max(frequencies.values()) - min(frequencies.values()) > 1:
            raise AssertionError("balanced schedule construction failed")
    return tuple(
        {mineral: by_mineral[mineral][replicate] for mineral in TARGET_MINERALS}
        for replicate in range(replicates)
    )


def _seed_entropy(seed: int, axis: int, replicate: int) -> str:
    return _compact_json([seed, axis, replicate])


def _block_draw(
    block_ids: Sequence[int], *, seed: int, axis: int, replicate: int
) -> tuple[int, ...]:
    unique = tuple(int(value) for value in block_ids)
    if len(unique) < 2 or len(set(unique)) != len(unique) or min(unique, default=0) <= 0:
        raise ProtocolError("stochastic axes require at least two unique positive complete blocks")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, axis, replicate])))
    return tuple(int(value) for value in rng.choice(unique, size=len(unique), replace=True))


def _draw_record(draw: Sequence[int]) -> str:
    counts = Counter(int(value) for value in draw)
    return _compact_json(
        [{"block_id": block_id, "multiplicity": counts[block_id]} for block_id in sorted(counts)]
    )


def _member_base(
    *,
    site: str,
    member_id: str,
    member_class: str,
    fit_id: str,
    endmembers: Mapping[str, str],
    replicate: int | None,
    covariance_mode: str,
    calibration_mode: str,
    covariance_draw: Sequence[int] = (),
    calibration_draw: Sequence[int] = (),
    covariance_seed_entropy: str | None = None,
    calibration_seed_entropy: str | None = None,
    ridge: float = 0.01,
    quantile: float = 0.90,
    gate: float | None = 1.0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene": ANCHOR_SCENES[site],
        "site": site,
        "member_id": member_id,
        "member_class": member_class,
        "fit_id": fit_id,
        "stochastic_replicate": replicate,
        "covariance_mode": covariance_mode,
        "calibration_mode": calibration_mode,
        "covariance_draw": _draw_record(covariance_draw),
        "calibration_draw": _draw_record(calibration_draw),
        "covariance_seed_entropy": covariance_seed_entropy,
        "calibration_seed_entropy": calibration_seed_entropy,
        "ridge": float(ridge),
        "detection_quantile": float(quantile),
        "infeasibility_gate": _gate_label(gate),
        "contributing_pixels": None,
        "retained_bands": None,
        "status": "pending",
        "failure_reason": None,
    }
    row.update({f"endmember_{mineral}": endmembers[mineral] for mineral in TARGET_MINERALS})
    return row


def build_design(
    *,
    candidates: Mapping[str, Sequence[str]],
    complete_blocks: Mapping[str, Sequence[int]],
    sites: Sequence[str] = FROZEN_SITES,
    ridges: Sequence[float] = FROZEN_RIDGES,
    quantiles: Sequence[float] = FROZEN_QUANTILES,
    gates: Sequence[float | None] = FROZEN_GATES,
    stochastic_replicates: int = FROZEN_STOCHASTIC_REPLICATES,
    bootstrap_replicates: int = FROZEN_BOOTSTRAP_REPLICATES,
    seed: int = FROZEN_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Materialize the complete deterministic finite design in member order."""
    sites_tuple = tuple(str(site) for site in sites)
    if any(site not in ANCHOR_SCENES for site in sites_tuple):
        raise ProtocolError("design contains an unknown site")
    populations = {mineral: tuple(sorted(candidates[mineral])) for mineral in TARGET_MINERALS}
    for mineral, expected_count in EXPECTED_CANDIDATE_COUNTS.items():
        if len(populations[mineral]) != expected_count:
            raise ProtocolError(
                f"{mineral} eligible population is {len(populations[mineral])}; "
                f"frozen count is {expected_count}"
            )
        if BASELINE_ENDMEMBERS[mineral] not in populations[mineral]:
            raise ProtocolError(f"frozen {mineral} medoid is absent from the eligible population")

    schedules = balanced_endmember_schedules(
        populations, replicates=stochastic_replicates, seed=seed
    )
    members: list[dict[str, Any]] = []
    for site in sites_tuple:
        site_index = FROZEN_SITES.index(site)
        site_blocks = tuple(int(value) for value in complete_blocks[site])
        covariance_draws = {
            replicate: _block_draw(
                site_blocks, seed=seed, axis=1000 + site_index, replicate=replicate
            )
            for replicate in range(stochastic_replicates)
        }
        calibration_draws = {
            replicate: _block_draw(
                site_blocks, seed=seed, axis=2000 + site_index, replicate=replicate
            )
            for replicate in range(stochastic_replicates)
        }
        baseline_fit = f"{site}:fit:baseline:r0.01"
        members.append(
            _member_base(
                site=site,
                member_id=f"{site}:baseline",
                member_class="baseline",
                fit_id=baseline_fit,
                endmembers=BASELINE_ENDMEMBERS,
                replicate=None,
                covariance_mode="full_scene",
                calibration_mode="full_scene",
            )
        )
        for replicate, endmembers in enumerate(schedules):
            members.append(
                _member_base(
                    site=site,
                    member_id=f"{site}:endmember_only:r{replicate:02d}",
                    member_class="endmember_only",
                    fit_id=f"{site}:fit:endmember:r{replicate:02d}:ridge0.01",
                    endmembers=endmembers,
                    replicate=replicate,
                    covariance_mode="full_scene",
                    calibration_mode="full_scene",
                )
            )
        for replicate in range(stochastic_replicates):
            members.append(
                _member_base(
                    site=site,
                    member_id=f"{site}:covariance_only:r{replicate:02d}",
                    member_class="covariance_only",
                    fit_id=f"{site}:fit:covariance:r{replicate:02d}:ridge0.01",
                    endmembers=BASELINE_ENDMEMBERS,
                    replicate=replicate,
                    covariance_mode="bootstrap_blocks",
                    calibration_mode="full_scene",
                    covariance_draw=covariance_draws[replicate],
                    covariance_seed_entropy=_seed_entropy(seed, 1000 + site_index, replicate),
                )
            )
        for replicate in range(stochastic_replicates):
            members.append(
                _member_base(
                    site=site,
                    member_id=f"{site}:calibration_only:r{replicate:02d}",
                    member_class="calibration_only",
                    fit_id=baseline_fit,
                    endmembers=BASELINE_ENDMEMBERS,
                    replicate=replicate,
                    covariance_mode="full_scene",
                    calibration_mode="bootstrap_blocks",
                    calibration_draw=calibration_draws[replicate],
                    calibration_seed_entropy=_seed_entropy(seed, 2000 + site_index, replicate),
                )
            )
        for ridge in ridges:
            ridge_value = float(ridge)
            fit_id = (
                baseline_fit
                if ridge_value == 0.01
                else f"{site}:fit:analytical:ridge{format(ridge_value, '.12g')}"
            )
            for quantile in quantiles:
                for gate in gates:
                    members.append(
                        _member_base(
                            site=site,
                            member_id=(
                                f"{site}:analytical:ridge{format(ridge_value, '.12g')}:"
                                f"q{format(float(quantile), '.12g')}:gate{_gate_label(gate)}"
                            ),
                            member_class="analytical_grid",
                            fit_id=fit_id,
                            endmembers=BASELINE_ENDMEMBERS,
                            replicate=None,
                            covariance_mode="full_scene",
                            calibration_mode="full_scene",
                            ridge=ridge_value,
                            quantile=float(quantile),
                            gate=gate,
                        )
                    )
        for replicate, endmembers in enumerate(schedules):
            for ridge in ridges:
                ridge_value = float(ridge)
                fit_id = f"{site}:fit:joint:r{replicate:02d}:ridge{format(ridge_value, '.12g')}"
                for quantile in quantiles:
                    for gate in gates:
                        members.append(
                            _member_base(
                                site=site,
                                member_id=(
                                    f"{site}:joint:r{replicate:02d}:"
                                    f"ridge{format(ridge_value, '.12g')}:"
                                    f"q{format(float(quantile), '.12g')}:gate{_gate_label(gate)}"
                                ),
                                member_class="joint",
                                fit_id=fit_id,
                                endmembers=endmembers,
                                replicate=replicate,
                                covariance_mode="bootstrap_blocks",
                                calibration_mode="bootstrap_blocks",
                                covariance_draw=covariance_draws[replicate],
                                calibration_draw=calibration_draws[replicate],
                                covariance_seed_entropy=_seed_entropy(
                                    seed, 1000 + site_index, replicate
                                ),
                                calibration_seed_entropy=_seed_entropy(
                                    seed, 2000 + site_index, replicate
                                ),
                                ridge=ridge_value,
                                quantile=float(quantile),
                                gate=gate,
                            )
                        )

    per_scene_counts = Counter(row["site"] for row in members)
    fit_counts = {
        site: len({row["fit_id"] for row in members if row["site"] == site}) for site in sites_tuple
    }
    analytical_cells = len(tuple(ridges)) * len(tuple(quantiles)) * len(tuple(gates))
    design = {
        "schema_version": "1.0",
        "frequency_estimand": "finite_design_empirical_frequency",
        "sites": list(sites_tuple),
        "anchor_scenes": {site: ANCHOR_SCENES[site] for site in sites_tuple},
        "target_minerals": list(TARGET_MINERALS),
        "candidate_populations": {
            mineral: list(populations[mineral]) for mineral in TARGET_MINERALS
        },
        "baseline_endmembers": dict(BASELINE_ENDMEMBERS),
        "endmember_schedules": list(schedules),
        "schedule_algorithm": "balanced_resize_then_pcg64_shuffle",
        "seed_derivations": {
            "endmember": [[seed, index] for index in range(len(TARGET_MINERALS))],
            "covariance": {
                site: [
                    [seed, 1000 + index, replicate] for replicate in range(stochastic_replicates)
                ]
                for index, site in enumerate(sites_tuple)
            },
            "calibration": {
                site: [
                    [seed, 2000 + index, replicate] for replicate in range(stochastic_replicates)
                ]
                for index, site in enumerate(sites_tuple)
            },
        },
        "ridges": [float(value) for value in ridges],
        "detection_quantiles": [float(value) for value in quantiles],
        "infeasibility_gates": [_gate_label(value) for value in gates],
        "stochastic_replicates": stochastic_replicates,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "analytical_cells": analytical_cells,
        "recorded_variants_per_scene": next(iter(per_scene_counts.values()), 0),
        "recorded_variants_total": len(members),
        "unique_mtmf_fits_per_scene": next(iter(fit_counts.values()), 0),
        "unique_mtmf_fits_total": sum(fit_counts.values()),
        "axis_contrasts": "descriptive_paired_only",
        "covariance_terminology": {
            "operational": "spatially_cross_fitted_thresholds_under_full_scene_covariance",
            "strict_sensitivity": "held_out_block_and_halo_excluded_from_covariance",
        },
    }
    if (
        sites_tuple == FROZEN_SITES
        and tuple(float(value) for value in ridges) == FROZEN_RIDGES
        and tuple(float(value) for value in quantiles) == FROZEN_QUANTILES
        and tuple(gates) == FROZEN_GATES
        and stochastic_replicates == FROZEN_STOCHASTIC_REPLICATES
    ):
        if any(value != FROZEN_VARIANTS_PER_SCENE for value in per_scene_counts.values()):
            raise AssertionError("frozen member accounting is not 355 variants per scene")
        if any(value != FROZEN_UNIQUE_FITS_PER_SCENE for value in fit_counts.values()):
            raise AssertionError("frozen score reuse is not 83 fits per scene")
    return design, members


@dataclass(frozen=True)
class DetectionResult:
    """One member/mineral operational threshold result."""

    status: str
    reason: str | None
    threshold: float | None
    detections: np.ndarray | None
    valid_support: np.ndarray | None
    calibration_pixels: int


def _weighted_order_statistic(values: np.ndarray, weights: np.ndarray, index: int) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.int64)
    position = int(np.searchsorted(cumulative, index + 1, side="left"))
    return float(sorted_values[position])


def _repeated_linear_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    total = int(np.sum(weights))
    if total <= 0:
        raise ValueError("weighted quantile requires positive total weight")
    position = float(quantile) * (total - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    lower_value = _weighted_order_statistic(values, weights, lower)
    if upper == lower:
        return lower_value
    upper_value = _weighted_order_statistic(values, weights, upper)
    return lower_value + (position - lower) * (upper_value - lower_value)


def operational_detection(
    scores: np.ndarray,
    infeasibility: np.ndarray,
    block_ids: np.ndarray,
    *,
    calibration_draw: Sequence[int],
    quantile: float,
    max_infeasibility: float | None,
) -> DetectionResult:
    """Apply gated empirical calibration and the raw threshold to a scene."""
    score = np.asarray(scores, dtype=float)
    infeas = np.asarray(infeasibility, dtype=float)
    blocks = np.asarray(block_ids)
    if score.shape != infeas.shape or score.shape != blocks.shape or score.ndim != 2:
        raise ValueError("score, infeasibility, and block arrays must be aligned 2-D arrays")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    draw = tuple(int(value) for value in calibration_draw)
    if not draw:
        return DetectionResult(
            status="unavailable",
            reason="zero_eligible_calibration_blocks",
            threshold=None,
            detections=None,
            valid_support=None,
            calibration_pixels=0,
        )
    draw_counts = Counter(draw)
    support = np.isfinite(score) & np.isfinite(infeas)
    gate_support = support.copy()
    if max_infeasibility is not None:
        gate_support &= infeas < float(max_infeasibility)
    calibration = gate_support & np.isin(blocks, tuple(draw_counts)) & (score > 0)
    if not np.any(calibration):
        return DetectionResult(
            status="complete",
            reason="no_positive_calibration_scores",
            threshold=None,
            detections=np.zeros(score.shape, dtype=bool),
            valid_support=support,
            calibration_pixels=0,
        )
    calibration_scores = score[calibration]
    calibration_weights = np.asarray(
        [draw_counts[int(block)] for block in blocks[calibration]], dtype=np.int64
    )
    threshold = _repeated_linear_quantile(calibration_scores, calibration_weights, quantile)
    detections = gate_support & (score >= threshold)
    return DetectionResult(
        status="complete",
        reason=None,
        threshold=threshold,
        detections=detections,
        valid_support=support,
        calibration_pixels=int(np.sum(calibration_weights)),
    )


@dataclass(frozen=True)
class MtmfFit:
    """NumPy-reference MTMF scores and provenance for one unique fit."""

    matched_filter: dict[str, np.ndarray]
    infeasibility: dict[str, np.ndarray]
    valid_support: np.ndarray
    contributing_pixels: int
    retained_bands: int


def fit_mtmf_numpy(
    cube: Any,
    endmembers: Mapping[str, Any],
    *,
    ridge: float,
    block_ids: np.ndarray | None = None,
    covariance_draw: Sequence[int] = (),
) -> MtmfFit:
    """Compute one MTMF fit with full-scene or multiplicity-weighted covariance.

    This follows :mod:`tanager_minmap.unmix`: shared finite bands, a shared
    scene background, ``ridge * trace(C) / n_band`` loading, and a direct
    inverse.  Failed inverses and degenerate covariance are surfaced as failed
    members; no pseudoinverse or redraw is attempted.
    """
    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    if hasattr(cube, "transpose") and hasattr(cube, "values"):
        data = np.asarray(cube.transpose("band", "y", "x").values, dtype=float)
    else:
        data = np.asarray(cube, dtype=float)
    if data.ndim != 3:
        raise ValueError("cube must have band, y, x dimensions")
    if tuple(endmembers) != tuple(TARGET_MINERALS):
        raise FitFailure("endmember order/population differs from TARGET_MINERALS")
    reflectance = {
        mineral: np.asarray(endmembers[mineral].reflectance, dtype=float)
        for mineral in TARGET_MINERALS
    }
    n_band, ny, nx = data.shape
    if any(values.shape != (n_band,) for values in reflectance.values()):
        raise FitFailure("endmember wavelength axes do not match the scene cube")
    flat = data.reshape(n_band, ny * nx)
    valid_bands = np.isfinite(flat).any(axis=1) & np.all(
        np.stack([np.isfinite(reflectance[mineral]) for mineral in TARGET_MINERALS]), axis=0
    )
    retained_bands = int(np.count_nonzero(valid_bands))
    if retained_bands < 2:
        raise FitFailure("fewer_than_two_finite_retained_channels")
    map_support = np.isfinite(flat[valid_bands]).all(axis=0)
    samples = flat[valid_bands][:, map_support].T
    if block_ids is None or not covariance_draw:
        weights = np.ones(samples.shape[0], dtype=np.int64)
    else:
        blocks = np.asarray(block_ids)
        if blocks.shape != (ny, nx):
            raise ValueError("covariance block raster does not match the scene grid")
        counts = Counter(int(value) for value in covariance_draw)
        support_blocks = blocks.reshape(-1)[map_support]
        weights = np.asarray(
            [counts.get(int(value), 0) for value in support_blocks], dtype=np.int64
        )
        use = weights > 0
        samples = samples[use]
        weights = weights[use]
    contributing_pixels = int(np.sum(weights))
    if contributing_pixels < 2:
        raise FitFailure("fewer_than_two_contributing_pixels")
    mu = np.average(samples, axis=0, weights=weights)
    centered_covariance = samples - mu
    covariance = (centered_covariance.T @ (centered_covariance * weights[:, np.newaxis])) / (
        contributing_pixels - 1
    )
    trace = float(np.trace(covariance))
    if not math.isfinite(trace) or trace <= 0:
        raise FitFailure("nonfinite_or_zero_covariance_trace")
    loaded = covariance + ridge * trace / retained_bands * np.eye(retained_bands)
    try:
        covariance_inverse = np.linalg.inv(loaded)
    except np.linalg.LinAlgError as error:
        raise FitFailure("covariance_inverse_failed") from error
    if not np.all(np.isfinite(covariance_inverse)):
        raise FitFailure("nonfinite_covariance_inverse")

    scene_samples = flat[valid_bands][:, map_support].T
    scene_centered = scene_samples - mu
    whitened = scene_centered @ covariance_inverse
    rx = np.einsum("ij,ij->i", whitened, scene_centered)
    norm = math.sqrt(max(retained_bands - 1, 1))
    matched_filter: dict[str, np.ndarray] = {}
    infeasibility: dict[str, np.ndarray] = {}
    for mineral in TARGET_MINERALS:
        difference = reflectance[mineral][valid_bands] - mu
        weight = covariance_inverse @ difference
        eta = float(difference @ weight)
        if not math.isfinite(eta) or eta == 0:
            raise FitFailure(f"{mineral}_nonfinite_or_zero_filter_denominator")
        abundance = scene_centered @ weight / eta
        infeas = np.sqrt(np.clip(rx - abundance**2 * eta, 0.0, None)) / norm
        if not np.all(np.isfinite(abundance)) or not np.all(np.isfinite(infeas)):
            raise FitFailure(f"{mineral}_nonfinite_mtmf_output")
        abundance_map = np.full(ny * nx, np.nan, dtype=float)
        infeas_map = np.full(ny * nx, np.nan, dtype=float)
        abundance_map[map_support] = abundance
        infeas_map[map_support] = infeas
        matched_filter[mineral] = abundance_map.reshape(ny, nx)
        infeasibility[mineral] = infeas_map.reshape(ny, nx)
    return MtmfFit(
        matched_filter=matched_filter,
        infeasibility=infeasibility,
        valid_support=map_support.reshape(ny, nx),
        contributing_pixels=contributing_pixels,
        retained_bands=retained_bands,
    )


def fit_strict_covariance_exclusion_numpy(
    cube: Any,
    endmembers: Mapping[str, Any],
    *,
    ridge: float,
    held_out_block: Mapping[str, Any],
    halo_pixels: int,
) -> MtmfFit:
    """Re-estimate covariance after excluding one held-out block and its halo."""
    if halo_pixels < 0:
        raise ValueError("halo_pixels cannot be negative")
    if hasattr(cube, "sizes"):
        shape = (int(cube.sizes["y"]), int(cube.sizes["x"]))
    else:
        values = np.asarray(cube)
        if values.ndim != 3:
            raise ValueError("cube must have band, y, x dimensions")
        shape = (values.shape[1], values.shape[2])
    allowed = np.ones(shape, dtype=np.uint8)
    row_start = max(0, int(held_out_block["row_start"]) - halo_pixels)
    row_stop = min(shape[0], int(held_out_block["row_stop"]) + halo_pixels)
    col_start = max(0, int(held_out_block["col_start"]) - halo_pixels)
    col_stop = min(shape[1], int(held_out_block["col_stop"]) + halo_pixels)
    allowed[row_start:row_stop, col_start:col_stop] = 0
    return fit_mtmf_numpy(
        cube,
        endmembers,
        ridge=ridge,
        block_ids=allowed,
        covariance_draw=(1,),
    )


def full_scene_detection(
    scores: np.ndarray,
    infeasibility: np.ndarray,
    *,
    quantile: float,
    max_infeasibility: float | None,
) -> DetectionResult:
    """Apply the deterministic current full-scene calibration convention."""
    blocks = np.ones(np.asarray(scores).shape, dtype=np.uint8)
    return operational_detection(
        scores,
        infeasibility,
        blocks,
        calibration_draw=(1,),
        quantile=quantile,
        max_infeasibility=max_infeasibility,
    )


def dominant_class(
    scores: Mapping[str, np.ndarray], detections: Mapping[str, DetectionResult]
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic dominant class and normalized strength."""
    shape = np.asarray(scores[TARGET_MINERALS[0]]).shape
    strength = np.full((len(TARGET_MINERALS), *shape), np.nan, dtype=float)
    for index, mineral in enumerate(TARGET_MINERALS):
        result = detections[mineral]
        if result.detections is None or result.threshold is None or result.threshold <= 0:
            continue
        use = result.detections
        strength[index, use] = np.asarray(scores[mineral])[use] / result.threshold
    filled = np.where(np.isfinite(strength), strength, -np.inf)
    classes = np.argmax(filled, axis=0).astype(np.int16)
    peak = np.max(filled, axis=0)
    classes[~np.isfinite(peak)] = DOMINANT_NONE
    valid_support = np.logical_and.reduce(
        [np.isfinite(np.asarray(scores[mineral])) for mineral in TARGET_MINERALS]
    )
    classes[~valid_support] = DOMINANT_NODATA
    peak[~np.isfinite(peak)] = np.nan
    return classes, peak


@dataclass(frozen=True)
class DominantSummary:
    """Finite-design dominant-class uncertainty maps."""

    modal_class: np.ndarray
    modal_frequency: np.ndarray
    normalized_entropy: np.ndarray
    switch_frequency: np.ndarray
    valid_count: np.ndarray


def summarize_dominant_classes(
    member_classes: Sequence[np.ndarray], baseline_class: np.ndarray
) -> DominantSummary:
    """Summarize modal class, entropy, and switches over valid members."""
    baseline = np.asarray(baseline_class, dtype=int)
    if not member_classes:
        raise ValueError("at least one member class map is required")
    stack = np.stack([np.asarray(item, dtype=int) for item in member_classes], axis=0)
    if stack.shape[1:] != baseline.shape:
        raise ValueError("dominant class maps are not aligned")
    category_count = len(TARGET_MINERALS) + 1
    member_valid = stack >= DOMINANT_NONE
    encoded = stack + 1
    counts = np.stack(
        [np.sum(encoded == category, axis=0) for category in range(category_count)], axis=0
    )
    valid = np.sum(member_valid, axis=0)
    modal_encoded = np.argmax(counts, axis=0)
    modal_count = np.max(counts, axis=0)
    modal = (modal_encoded - 1).astype(np.int16)
    modal[valid == 0] = DOMINANT_NODATA
    modal_frequency = np.divide(
        modal_count,
        valid,
        out=np.full(baseline.shape, np.nan, dtype=float),
        where=valid > 0,
    )
    probabilities = np.divide(
        counts,
        valid[np.newaxis, ...],
        out=np.zeros_like(counts, dtype=float),
        where=valid[np.newaxis, ...] > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)
    entropy = -np.sum(entropy_terms, axis=0) / math.log(category_count)
    entropy[valid == 0] = np.nan
    common = member_valid & (baseline[np.newaxis, ...] >= DOMINANT_NONE)
    switches = np.sum((stack != baseline[np.newaxis, ...]) & common, axis=0)
    switch_denominator = np.sum(common, axis=0)
    switch_frequency = np.divide(
        switches,
        switch_denominator,
        out=np.full(baseline.shape, np.nan, dtype=float),
        where=switch_denominator > 0,
    )
    return DominantSummary(modal, modal_frequency, entropy, switch_frequency, valid)


@dataclass(frozen=True)
class CrossFittedMetrics:
    """Spatial threshold evaluation with explicit covariance terminology."""

    status: str
    reason: str | None
    covariance_scope: str
    auc: float | None
    balanced_accuracy: float | None
    evaluated_blocks: int
    unavailable_blocks: int
    n_pixels: int
    block_results: tuple[Mapping[str, Any], ...] = ()


def _block_intersects_halo(
    candidate: Mapping[str, Any], held_out: Mapping[str, Any], halo_pixels: int
) -> bool:
    return (
        int(candidate["row_start"]) < int(held_out["row_stop"]) + halo_pixels
        and int(candidate["row_stop"]) > int(held_out["row_start"]) - halo_pixels
        and int(candidate["col_start"]) < int(held_out["col_stop"]) + halo_pixels
        and int(candidate["col_stop"]) > int(held_out["col_start"]) - halo_pixels
    )


def spatially_cross_fitted_threshold_evaluation(
    scores: np.ndarray,
    binary_reference: np.ndarray,
    block_ids: np.ndarray,
    block_records: Sequence[Mapping[str, Any]],
    *,
    halo_pixels: int,
    site_index: int,
    stochastic_replicate: int,
    seed: int = FROZEN_SEED,
    covariance_scope: str = "full_scene_covariance",
) -> CrossFittedMetrics:
    """Bootstrap training blocks and evaluate each held-out complete block.

    ``scores`` must already have been fit under the named covariance scope.
    The default is the preregistered operational/transductive estimand.  Strict
    covariance-exclusion scores are passed separately with
    ``covariance_scope='strict_covariance_exclusion'``; the two are never
    pooled or relabeled.
    """
    from .spatial_validation import Block, BlockSample, block_balanced_youden, rank_auc

    if covariance_scope not in {"full_scene_covariance", "strict_covariance_exclusion"}:
        raise ValueError("unknown covariance scope")
    score = np.asarray(scores, dtype=float)
    reference = np.asarray(binary_reference, dtype=float)
    blocks = np.asarray(block_ids)
    if score.shape != reference.shape or score.shape != blocks.shape:
        raise ValueError("cross-fitted score, reference, and block arrays must align")
    records = [dict(record) for record in block_records]
    record_by_id = {int(record["numeric_block_id"]): record for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("complete block records contain duplicate numeric IDs")
    pooled_scores: list[np.ndarray] = []
    pooled_reference: list[np.ndarray] = []
    pooled_predictions: list[np.ndarray] = []
    block_results: list[dict[str, Any]] = []
    unavailable = 0
    for held_out_id in sorted(record_by_id):
        held_out = record_by_id[held_out_id]
        block_domain = (blocks == held_out_id) & np.isfinite(score) & np.isfinite(reference)
        block_result: dict[str, Any] = {
            "block_id": held_out_id,
            "scores": score[block_domain],
            "references": reference[block_domain].astype(np.int8),
            "predictions": None,
        }
        eligible = [
            block_id
            for block_id, record in record_by_id.items()
            if block_id != held_out_id and not _block_intersects_halo(record, held_out, halo_pixels)
        ]
        if not eligible:
            unavailable += 1
            block_results.append(block_result)
            continue
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, 2000 + site_index, stochastic_replicate]))
        )
        draw = tuple(int(value) for value in rng.choice(eligible, size=len(eligible), replace=True))
        training: list[BlockSample] = []
        for draw_index, block_id in enumerate(draw):
            record = record_by_id[block_id]
            rows = slice(int(record["row_start"]), int(record["row_stop"]))
            cols = slice(int(record["col_start"]), int(record["col_stop"]))
            block = Block(
                block_id=f"{record['block_id']}@draw{draw_index}",
                block_row=int(record["block_row"]),
                block_col=int(record["block_col"]),
                row_start=int(record["row_start"]),
                row_stop=int(record["row_stop"]),
                col_start=int(record["col_start"]),
                col_stop=int(record["col_stop"]),
            )
            training.append(BlockSample(block, score[rows, cols], reference[rows, cols]))
        threshold = block_balanced_youden(training)
        test = (blocks == held_out_id) & np.isfinite(score) & np.isfinite(reference)
        if threshold is None or not np.any(test):
            unavailable += 1
            block_results.append(block_result)
            continue
        test_score = score[test]
        test_reference = reference[test].astype(np.int8)
        pooled_scores.append(test_score)
        pooled_reference.append(test_reference)
        prediction = test_score >= threshold
        pooled_predictions.append(prediction)
        block_result["predictions"] = prediction
        block_results.append(block_result)
    all_domain = (blocks > 0) & np.isfinite(score) & np.isfinite(reference)
    all_scores = score[all_domain]
    all_reference = reference[all_domain].astype(np.int8)
    auc: float | None = None
    if all_scores.size and len(np.unique(all_reference)) == 2:
        value = rank_auc(all_scores, all_reference)
        auc = float(value) if math.isfinite(value) else None
    if not pooled_scores:
        return CrossFittedMetrics(
            status="unavailable",
            reason="zero_evaluable_cross_fitted_blocks",
            covariance_scope=covariance_scope,
            auc=auc,
            balanced_accuracy=None,
            evaluated_blocks=0,
            unavailable_blocks=unavailable,
            n_pixels=0,
            block_results=tuple(block_results),
        )
    observed = np.concatenate(pooled_reference).astype(bool)
    predicted = np.concatenate(pooled_predictions).astype(bool)
    positives = int(np.count_nonzero(observed))
    negatives = int(observed.size - positives)
    if positives == 0 or negatives == 0:
        balanced = None
        reason = "cross_fitted_reference_lacks_both_classes"
        status = "unavailable"
    else:
        tpr = np.count_nonzero(predicted & observed) / positives
        tnr = np.count_nonzero(~predicted & ~observed) / negatives
        balanced = float(0.5 * (tpr + tnr))
        reason = None
        status = "complete"
    return CrossFittedMetrics(
        status=status,
        reason=reason,
        covariance_scope=covariance_scope,
        auc=auc,
        balanced_accuracy=balanced,
        evaluated_blocks=len(pooled_scores),
        unavailable_blocks=unavailable,
        n_pixels=int(observed.size),
        block_results=tuple(block_results),
    )


def strict_covariance_cross_fitted_threshold_evaluation(
    fold_scores: Mapping[int, np.ndarray],
    binary_reference: np.ndarray,
    block_ids: np.ndarray,
    block_records: Sequence[Mapping[str, Any]],
    *,
    halo_pixels: int,
    site_index: int,
    stochastic_replicate: int,
    seed: int = FROZEN_SEED,
) -> CrossFittedMetrics:
    """Evaluate strict folds using each held-out fold's own covariance score map."""
    from .spatial_validation import Block, BlockSample, block_balanced_youden

    reference = np.asarray(binary_reference, dtype=float)
    blocks = np.asarray(block_ids)
    if reference.shape != blocks.shape:
        raise ValueError("strict reference and block arrays must align")
    records = [dict(record) for record in block_records]
    record_by_id = {int(record["numeric_block_id"]): record for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("complete block records contain duplicate numeric IDs")
    unknown_folds = set(int(value) for value in fold_scores) - set(record_by_id)
    if unknown_folds:
        raise ValueError("strict covariance scores contain unknown held-out blocks")
    block_results: list[dict[str, Any]] = []
    unavailable = 0
    for held_out_id in sorted(record_by_id):
        score_value = fold_scores.get(held_out_id)
        if score_value is None:
            unavailable += 1
            block_results.append(
                {
                    "block_id": held_out_id,
                    "scores": np.asarray([], dtype=float),
                    "references": np.asarray([], dtype=np.int8),
                    "predictions": None,
                }
            )
            continue
        score = np.asarray(score_value, dtype=float)
        if score.shape != reference.shape:
            raise ValueError("strict covariance fold score changed anchor shape")
        held_out = record_by_id[held_out_id]
        test = (blocks == held_out_id) & np.isfinite(score) & np.isfinite(reference)
        block_result: dict[str, Any] = {
            "block_id": held_out_id,
            "scores": score[test],
            "references": reference[test].astype(np.int8),
            "predictions": None,
        }
        eligible = [
            block_id
            for block_id, record in record_by_id.items()
            if block_id != held_out_id and not _block_intersects_halo(record, held_out, halo_pixels)
        ]
        if not eligible:
            unavailable += 1
            block_results.append(block_result)
            continue
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, 2000 + site_index, stochastic_replicate]))
        )
        draw = tuple(int(value) for value in rng.choice(eligible, size=len(eligible), replace=True))
        training: list[BlockSample] = []
        for draw_index, block_id in enumerate(draw):
            record = record_by_id[block_id]
            rows = slice(int(record["row_start"]), int(record["row_stop"]))
            columns = slice(int(record["col_start"]), int(record["col_stop"]))
            block = Block(
                block_id=f"{record['block_id']}@draw{draw_index}",
                block_row=int(record["block_row"]),
                block_col=int(record["block_col"]),
                row_start=int(record["row_start"]),
                row_stop=int(record["row_stop"]),
                col_start=int(record["col_start"]),
                col_stop=int(record["col_stop"]),
            )
            training.append(BlockSample(block, score[rows, columns], reference[rows, columns]))
        threshold = block_balanced_youden(training)
        if threshold is None or not np.any(test):
            unavailable += 1
            block_results.append(block_result)
            continue
        block_result["predictions"] = score[test] >= threshold
        block_results.append(block_result)

    metric_values = _external_metric_values(block_results)
    auc_value = metric_values["auc"]
    balanced_value = metric_values["balanced_accuracy"]
    evaluated = sum(block.get("predictions") is not None for block in block_results)
    n_pixels = sum(
        np.asarray(block.get("predictions", ())).size
        for block in block_results
        if block.get("predictions") is not None
    )
    auc = float(auc_value) if math.isfinite(auc_value) else None
    balanced = float(balanced_value) if math.isfinite(balanced_value) else None
    status = "complete" if balanced is not None else "unavailable"
    reason = None if status == "complete" else "zero_evaluable_cross_fitted_blocks"
    return CrossFittedMetrics(
        status=status,
        reason=reason,
        covariance_scope="strict_covariance_exclusion",
        auc=auc,
        balanced_accuracy=balanced,
        evaluated_blocks=evaluated,
        unavailable_blocks=unavailable,
        n_pixels=int(n_pixels),
        block_results=tuple(block_results),
    )


def confidence_classes(frequency: np.ndarray) -> np.ndarray:
    """Classify empirical finite-design frequency using the frozen 0.20/0.80 rule."""
    values = np.asarray(frequency, dtype=float)
    out = np.full(values.shape, CONFIDENCE_NODATA, dtype=np.int8)
    finite = np.isfinite(values)
    out[finite] = CONFIDENCE_CHOICE_SENSITIVE
    out[finite & (values <= 0.20)] = CONFIDENCE_STABLE_NEGATIVE
    out[finite & (values >= 0.80)] = CONFIDENCE_STABLE_POSITIVE
    return out


class MapAccumulator:
    """Accumulate valid-member detections without treating failures as negatives."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.valid_count = np.zeros(shape, dtype=np.uint16)
        self.detection_count = np.zeros(shape, dtype=np.uint16)
        self.failures: list[dict[str, str]] = []

    def add(self, detections: np.ndarray, valid_support: np.ndarray) -> None:
        detected = np.asarray(detections, dtype=bool)
        valid = np.asarray(valid_support, dtype=bool)
        if detected.shape != self.valid_count.shape or valid.shape != self.valid_count.shape:
            raise ValueError("map accumulator inputs do not match its shape")
        if np.any(detected & ~valid):
            raise ValueError("a detection cannot lie outside valid member support")
        self.valid_count += valid.astype(np.uint16)
        self.detection_count += detected.astype(np.uint16)

    def record_failure(self, member_id: str, reason: str) -> None:
        self.failures.append({"member_id": member_id, "reason": reason})

    def frequency(self) -> np.ndarray:
        return np.divide(
            self.detection_count,
            self.valid_count,
            out=np.full(self.valid_count.shape, np.nan, dtype=float),
            where=self.valid_count > 0,
        )


@dataclass(frozen=True)
class NestedBootstrapResult:
    """Nested block-bootstrap draws and replicate-level member summaries."""

    block_ids: tuple[int, ...]
    member_ids: tuple[str, ...]
    draws: np.ndarray
    member_values: np.ndarray
    replicate_summaries: np.ndarray
    lower_95: float | None
    upper_95: float | None
    scheduled_replicates: int
    valid_replicates: int
    finite_fraction: float
    interval_available: bool
    unavailable_reason: str | None


def nested_block_bootstrap(
    member_block_values: Mapping[str, Mapping[int, float]],
    *,
    replicates: int = FROZEN_BOOTSTRAP_REPLICATES,
    seed: int = FROZEN_SEED,
    member_summary: Callable[[np.ndarray], float] = np.nanmedian,
    minimum_blocks: int = MINIMUM_INTERVAL_BLOCKS,
) -> NestedBootstrapResult:
    """Draw blocks once per replicate, then summarize members within replicate."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if minimum_blocks < 1:
        raise ValueError("minimum_blocks must be positive")
    member_ids = tuple(member_block_values)
    if not member_ids:
        return NestedBootstrapResult(
            block_ids=(),
            member_ids=(),
            draws=np.empty((replicates, 0), dtype=int),
            member_values=np.empty((replicates, 0), dtype=float),
            replicate_summaries=np.full(replicates, np.nan),
            lower_95=None,
            upper_95=None,
            scheduled_replicates=replicates,
            valid_replicates=0,
            finite_fraction=0.0,
            interval_available=False,
            unavailable_reason="zero_analytical_members",
        )
    block_sets = [set(int(key) for key in member_block_values[item]) for item in member_ids]
    if any(block_set != block_sets[0] for block_set in block_sets[1:]):
        raise ValueError("all members must provide one common block set")
    block_ids = tuple(sorted(block_sets[0]))
    if not block_ids:
        return NestedBootstrapResult(
            block_ids=(),
            member_ids=member_ids,
            draws=np.empty((replicates, 0), dtype=int),
            member_values=np.full((replicates, len(member_ids)), np.nan),
            replicate_summaries=np.full(replicates, np.nan),
            lower_95=None,
            upper_95=None,
            scheduled_replicates=replicates,
            valid_replicates=0,
            finite_fraction=0.0,
            interval_available=False,
            unavailable_reason="zero_complete_blocks",
        )
    matrix = np.asarray(
        [[member_block_values[member][block] for block in block_ids] for member in member_ids],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(block_ids), size=(replicates, len(block_ids)), replace=True)
    lookup = {block: index for index, block in enumerate(block_ids)}
    member_values = np.full((replicates, len(member_ids)), np.nan, dtype=float)
    summaries = np.full(replicates, np.nan, dtype=float)
    for replicate, draw in enumerate(draws):
        indices = np.asarray([lookup[int(block)] for block in draw], dtype=int)
        selected = matrix[:, indices]
        finite_counts = np.sum(np.isfinite(selected), axis=1)
        member_values[replicate] = np.divide(
            np.nansum(selected, axis=1),
            finite_counts,
            out=np.full(len(member_ids), np.nan, dtype=float),
            where=finite_counts > 0,
        )
        finite = member_values[replicate][np.isfinite(member_values[replicate])]
        if finite.size:
            summaries[replicate] = float(member_summary(finite))
    from .spatial_validation import _governed_confidence_interval

    governed = _governed_confidence_interval(
        "nested_member_summary",
        summaries,
        scheduled_replicates=replicates,
    )
    interval_available = governed.gate_eligible and len(block_ids) >= minimum_blocks
    if interval_available:
        lower = governed.lower
        upper = governed.upper
        unavailable_reason = None
    else:
        lower = upper = None
        unavailable_reason = (
            f"fewer_than_{minimum_blocks}_complete_blocks"
            if len(block_ids) < minimum_blocks
            else governed.unavailable_reason
        )
    return NestedBootstrapResult(
        block_ids=block_ids,
        member_ids=member_ids,
        draws=draws,
        member_values=member_values,
        replicate_summaries=summaries,
        lower_95=lower,
        upper_95=upper,
        scheduled_replicates=replicates,
        valid_replicates=governed.valid_replicates,
        finite_fraction=governed.finite_fraction,
        interval_available=interval_available,
        unavailable_reason=unavailable_reason,
    )


def _nested_result_from_values(
    *,
    block_ids: tuple[int, ...],
    member_ids: tuple[str, ...],
    draws: np.ndarray,
    member_values: np.ndarray,
    replicates: int,
    minimum_blocks: int,
) -> NestedBootstrapResult:
    summaries = np.full(replicates, np.nan, dtype=float)
    for replicate in range(replicates):
        finite = member_values[replicate][np.isfinite(member_values[replicate])]
        if finite.size:
            summaries[replicate] = float(np.median(finite))
    from .spatial_validation import _governed_confidence_interval

    governed = _governed_confidence_interval(
        "nested_member_summary",
        summaries,
        scheduled_replicates=replicates,
    )
    enough_blocks = len(block_ids) >= minimum_blocks
    available = governed.gate_eligible and enough_blocks
    return NestedBootstrapResult(
        block_ids=block_ids,
        member_ids=member_ids,
        draws=draws,
        member_values=member_values,
        replicate_summaries=summaries,
        lower_95=governed.lower if available else None,
        upper_95=governed.upper if available else None,
        scheduled_replicates=replicates,
        valid_replicates=governed.valid_replicates,
        finite_fraction=governed.finite_fraction,
        interval_available=available,
        unavailable_reason=(
            None
            if available
            else (
                f"fewer_than_{minimum_blocks}_complete_blocks"
                if not enough_blocks
                else governed.unavailable_reason
            )
        ),
    )


def nested_ratio_bootstrap(
    member_block_counts: Mapping[str, Mapping[int, tuple[float, float]]],
    *,
    replicates: int = FROZEN_BOOTSTRAP_REPLICATES,
    seed: int = FROZEN_SEED,
    minimum_blocks: int = MINIMUM_INTERVAL_BLOCKS,
) -> NestedBootstrapResult:
    """Recompute pooled count ratios under one shared complete-block draw."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    member_ids = tuple(member_block_counts)
    if not member_ids:
        return nested_block_bootstrap({}, replicates=replicates, seed=seed)
    block_sets = [set(values) for values in member_block_counts.values()]
    if any(block_set != block_sets[0] for block_set in block_sets[1:]):
        raise ValueError("all members must provide one common block set")
    block_ids = tuple(sorted(int(value) for value in block_sets[0]))
    if not block_ids:
        return nested_block_bootstrap(
            {member_id: {} for member_id in member_ids},
            replicates=replicates,
            seed=seed,
        )
    numerators = np.asarray(
        [[member_block_counts[member][block][0] for block in block_ids] for member in member_ids],
        dtype=float,
    )
    denominators = np.asarray(
        [[member_block_counts[member][block][1] for block in block_ids] for member in member_ids],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(block_ids), size=(replicates, len(block_ids)), replace=True)
    lookup = {block_id: index for index, block_id in enumerate(block_ids)}
    values = np.full((replicates, len(member_ids)), np.nan, dtype=float)
    for replicate, draw in enumerate(draws):
        indices = np.asarray([lookup[int(block)] for block in draw], dtype=int)
        numerator = np.sum(numerators[:, indices], axis=1)
        denominator = np.sum(denominators[:, indices], axis=1)
        values[replicate] = np.divide(
            numerator,
            denominator,
            out=np.full(len(member_ids), np.nan, dtype=float),
            where=denominator > 0,
        )
    return _nested_result_from_values(
        block_ids=block_ids,
        member_ids=member_ids,
        draws=draws,
        member_values=values,
        replicates=replicates,
        minimum_blocks=minimum_blocks,
    )


def nested_spearman_bootstrap(
    member_block_pairs: Mapping[str, Mapping[int, tuple[np.ndarray, np.ndarray]]],
    *,
    replicates: int = FROZEN_BOOTSTRAP_REPLICATES,
    seed: int = FROZEN_SEED,
    minimum_blocks: int = MINIMUM_INTERVAL_BLOCKS,
) -> NestedBootstrapResult:
    """Recompute member Spearman correlations after each shared block draw."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    member_ids = tuple(member_block_pairs)
    if not member_ids:
        return nested_block_bootstrap({}, replicates=replicates, seed=seed)
    block_sets = [set(values) for values in member_block_pairs.values()]
    if any(block_set != block_sets[0] for block_set in block_sets[1:]):
        raise ValueError("all members must provide one common block set")
    block_ids = tuple(sorted(int(value) for value in block_sets[0]))
    if not block_ids:
        return nested_block_bootstrap(
            {member_id: {} for member_id in member_ids},
            replicates=replicates,
            seed=seed,
        )
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(block_ids), size=(replicates, len(block_ids)), replace=True)
    values = np.full((replicates, len(member_ids)), np.nan, dtype=float)
    for replicate, draw in enumerate(draws):
        for member_index, member_id in enumerate(member_ids):
            left_parts = [member_block_pairs[member_id][int(block)][0] for block in draw]
            right_parts = [member_block_pairs[member_id][int(block)][1] for block in draw]
            left = np.concatenate(left_parts) if left_parts else np.asarray([], dtype=float)
            right = np.concatenate(right_parts) if right_parts else np.asarray([], dtype=float)
            value = _spearman(left, right)
            if value is not None:
                values[replicate, member_index] = value
    return _nested_result_from_values(
        block_ids=block_ids,
        member_ids=member_ids,
        draws=draws,
        member_values=values,
        replicates=replicates,
        minimum_blocks=minimum_blocks,
    )


def external_support_tier(block_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify M2 positive/negative-bearing block support without adaptation."""
    positive = 0
    negative = 0
    for block in block_results:
        reference = np.asarray(block.get("references", ()), dtype=float)
        finite = reference[np.isfinite(reference)]
        positive += int(np.any(finite == 1))
        negative += int(np.any(finite == 0))
    limiting = min(positive, negative)
    if limiting >= CONFIRMATORY_POSITIVE_BLOCKS:
        tier = "confirmatory"
    elif limiting >= EXPLORATORY_BLOCKS:
        tier = "exploratory"
    else:
        tier = "counts_maps_only"
    return {
        "positive_bearing_blocks": positive,
        "negative_bearing_blocks": negative,
        "support_tier": tier,
        "confirmatory_support": tier == "confirmatory",
    }


def _external_metric_values(blocks: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Recompute declared external metrics after one complete-block draw."""
    from .spatial_validation import rank_auc

    score_parts = [np.asarray(block.get("scores", ()), dtype=float) for block in blocks]
    reference_parts = [np.asarray(block.get("references", ()), dtype=float) for block in blocks]
    scores = np.concatenate(score_parts) if score_parts else np.asarray([], dtype=float)
    references = np.concatenate(reference_parts) if reference_parts else np.asarray([], dtype=float)
    rank_domain = np.isfinite(scores) & np.isfinite(references)
    scores = scores[rank_domain]
    references = references[rank_domain].astype(bool)
    auc = float("nan")
    if scores.size and np.any(references) and np.any(~references):
        candidate = float(rank_auc(scores, references.astype(np.int8)))
        if math.isfinite(candidate):
            auc = candidate

    threshold_references: list[np.ndarray] = []
    threshold_predictions: list[np.ndarray] = []
    for block in blocks:
        prediction = block.get("predictions")
        if prediction is None:
            continue
        block_reference = np.asarray(block.get("references", ()), dtype=float)
        block_prediction = np.asarray(prediction, dtype=bool)
        if block_reference.shape != block_prediction.shape:
            raise ValueError("cross-fitted block predictions and references do not align")
        finite = np.isfinite(block_reference)
        threshold_references.append(block_reference[finite].astype(bool))
        threshold_predictions.append(block_prediction[finite])
    if threshold_references:
        observed = np.concatenate(threshold_references)
        predicted = np.concatenate(threshold_predictions)
    else:
        observed = np.asarray([], dtype=bool)
        predicted = np.asarray([], dtype=bool)
    tp = int(np.count_nonzero(predicted & observed))
    fp = int(np.count_nonzero(predicted & ~observed))
    tn = int(np.count_nonzero(~predicted & ~observed))
    fn = int(np.count_nonzero(~predicted & observed))

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else float("nan")

    tpr = ratio(tp, tp + fn)
    fpr = ratio(fp, fp + tn)
    positive_f1 = ratio(2 * tp, 2 * tp + fp + fn)
    negative_f1 = ratio(2 * tn, 2 * tn + fp + fn)
    balanced = 0.5 * (tpr + 1.0 - fpr) if math.isfinite(tpr + fpr) else float("nan")
    macro_f1 = (
        0.5 * (positive_f1 + negative_f1)
        if math.isfinite(positive_f1 + negative_f1)
        else float("nan")
    )
    return {
        "auc": auc,
        "balanced_accuracy": balanced,
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "macro_f1": macro_f1,
        "tpr": tpr,
        "fpr": fpr,
        "prevalence": ratio(int(np.count_nonzero(references)), references.size),
    }


def nested_external_metric_intervals(
    member_results: Mapping[str, CrossFittedMetrics],
    *,
    scale: str,
    replicates: int = FROZEN_BOOTSTRAP_REPLICATES,
    seed: int = FROZEN_SEED,
) -> list[dict[str, Any]]:
    """Compute nested complete-block intervals for every M2 external endpoint."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    metric_names = (
        "auc",
        "balanced_accuracy",
        "positive_f1",
        "negative_f1",
        "macro_f1",
        "tpr",
        "fpr",
        "prevalence",
    )
    member_ids = tuple(member_results)
    if not member_ids:
        return [
            {
                "scale": scale,
                "metric": metric,
                "point_estimate": None,
                "lower_95": None,
                "upper_95": None,
                "scheduled_replicates": replicates,
                "valid_replicates": 0,
                "finite_fraction": 0.0,
                "interval_available": False,
                "unavailable_reason": "zero_analytical_members",
                "support_tier": "counts_maps_only",
                "confirmatory_support": False,
                "complete_blocks": 0,
            }
            for metric in metric_names
        ]
    by_member: dict[str, dict[int, Mapping[str, Any]]] = {}
    block_sets: list[set[int]] = []
    for member_id, result in member_results.items():
        blocks = {int(block["block_id"]): block for block in result.block_results}
        by_member[member_id] = blocks
        block_sets.append(set(blocks))
    if any(block_set != block_sets[0] for block_set in block_sets[1:]):
        raise ValueError("external members must share the same complete block IDs")
    block_ids = tuple(sorted(block_sets[0]))
    member_support = [
        external_support_tier(tuple(by_member[member_id][block_id] for block_id in block_ids))
        for member_id in member_ids
    ]
    positive_blocks = min(
        (int(item["positive_bearing_blocks"]) for item in member_support),
        default=0,
    )
    negative_blocks = min(
        (int(item["negative_bearing_blocks"]) for item in member_support),
        default=0,
    )
    limiting = min(positive_blocks, negative_blocks)
    support_tier = (
        "confirmatory"
        if limiting >= CONFIRMATORY_POSITIVE_BLOCKS
        else "exploratory"
        if limiting >= EXPLORATORY_BLOCKS
        else "counts_maps_only"
    )
    support = {
        "positive_bearing_blocks": positive_blocks,
        "negative_bearing_blocks": negative_blocks,
        "support_tier": support_tier,
        "confirmatory_support": support_tier == "confirmatory",
    }
    observed_by_metric = {metric: [] for metric in metric_names}
    for member_id in member_ids:
        observed = _external_metric_values(
            [by_member[member_id][block_id] for block_id in block_ids]
        )
        for metric, value in observed.items():
            if math.isfinite(value):
                observed_by_metric[metric].append(value)
    if not block_ids:
        return [
            {
                "scale": scale,
                "metric": metric,
                "point_estimate": None,
                "lower_95": None,
                "upper_95": None,
                "scheduled_replicates": replicates,
                "valid_replicates": 0,
                "finite_fraction": 0.0,
                "interval_available": False,
                "unavailable_reason": "zero_complete_blocks",
                **support,
                "complete_blocks": 0,
            }
            for metric in metric_names
        ]
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(block_ids), size=(replicates, len(block_ids)), replace=True)
    distributions = {metric: np.full(replicates, np.nan, dtype=float) for metric in metric_names}
    for replicate, draw in enumerate(draws):
        member_metrics = {metric: [] for metric in metric_names}
        for member_id in member_ids:
            drawn_blocks = [by_member[member_id][int(block_id)] for block_id in draw]
            values = _external_metric_values(drawn_blocks)
            for metric, value in values.items():
                if math.isfinite(value):
                    member_metrics[metric].append(value)
        for metric in metric_names:
            if member_metrics[metric]:
                distributions[metric][replicate] = float(np.median(member_metrics[metric]))
    from .spatial_validation import _governed_confidence_interval

    rows: list[dict[str, Any]] = []
    for metric, values in distributions.items():
        governed = _governed_confidence_interval(
            metric,
            values,
            scheduled_replicates=replicates,
        )
        enough_blocks = len(block_ids) >= MINIMUM_INTERVAL_BLOCKS
        interval_available = governed.gate_eligible and enough_blocks
        rows.append(
            {
                "scale": scale,
                "metric": metric,
                "point_estimate": (
                    float(np.median(observed_by_metric[metric]))
                    if observed_by_metric[metric]
                    else None
                ),
                "lower_95": governed.lower if interval_available else None,
                "upper_95": governed.upper if interval_available else None,
                "scheduled_replicates": replicates,
                "valid_replicates": governed.valid_replicates,
                "finite_fraction": governed.finite_fraction,
                "interval_available": interval_available,
                "unavailable_reason": (
                    None
                    if interval_available
                    else (
                        "fewer_than_two_complete_blocks"
                        if not enough_blocks
                        else governed.unavailable_reason
                    )
                ),
                **support,
                "complete_blocks": len(block_ids),
            }
        )
    return rows


def classify_permitted_claim(
    *,
    stability_pass: bool | None,
    external_pass: bool | None,
    strict_covariance_pass: bool | None,
) -> str:
    """Return the deterministic preregistered public-claim category."""
    if None in {stability_pass, external_pass, strict_covariance_pass}:
        return "unavailable_required_evidence"
    if stability_pass and external_pass and strict_covariance_pass:
        return "validated_analytically_robust_alteration_zone_discrimination"
    if stability_pass and not external_pass:
        return "analytically_stable_spatial_pattern_only"
    if external_pass and not strict_covariance_pass:
        return "operational_discrimination_not_strictly_held_out"
    if external_pass and not stability_pass:
        return "discriminative_but_analytically_sensitive"
    return "negative_or_unstable_result"


def evaluate_goldfield_claim_gate(
    *,
    analytical_cells_complete: bool,
    stable_core_retention: float | None,
    median_rank_correlation: float | None,
    rank_correlation_5th_percentile: float | None,
    switch_interval: NestedBootstrapResult,
    operational_intervals: Sequence[Mapping[str, Any]],
    strict_intervals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the frozen gate without converting missing evidence into success."""

    def indexed(
        rows: Sequence[Mapping[str, Any]], scale: str, metric: str
    ) -> Mapping[str, Any] | None:
        matches = [row for row in rows if row.get("scale") == scale and row.get("metric") == metric]
        return matches[0] if len(matches) == 1 else None

    stability_available = (
        stable_core_retention is not None
        and median_rank_correlation is not None
        and rank_correlation_5th_percentile is not None
        and switch_interval.interval_available
        and switch_interval.upper_95 is not None
    )
    stability_pass = None
    if stability_available:
        stability_pass = (
            stable_core_retention >= 0.80
            and median_rank_correlation >= 0.80
            and rank_correlation_5th_percentile > 0.50
            and float(switch_interval.upper_95) <= 0.20
        )

    def external_component(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, bool | None]:
        auc_l = indexed(rows, "L", "auc")
        balanced_l = indexed(rows, "L", "balanced_accuracy")
        auc_2l = indexed(rows, "2L", "auc")
        required = (auc_l, balanced_l, auc_2l)
        available = all(
            row is not None
            and bool(row.get("interval_available"))
            and bool(row.get("confirmatory_support"))
            for row in required
        )
        if not available:
            return False, None
        assert auc_l is not None and balanced_l is not None and auc_2l is not None
        passed = (
            float(auc_l["lower_95"]) > 0.5
            and float(balanced_l["lower_95"]) > 0.5
            and float(auc_2l["point_estimate"]) > 0.5
        )
        return True, passed

    external_available, external_pass = external_component(operational_intervals)
    strict_available, strict_pass = external_component(strict_intervals)
    confirmatory_available = (
        analytical_cells_complete
        and stability_available
        and external_available
        and strict_available
    )
    classification = (
        classify_permitted_claim(
            stability_pass=stability_pass,
            external_pass=external_pass,
            strict_covariance_pass=strict_pass,
        )
        if analytical_cells_complete
        else "unavailable_required_evidence"
    )
    return {
        "confirmatory_gate_available": confirmatory_available,
        "confirmatory_gate_pass": (
            bool(stability_pass and external_pass and strict_pass)
            if confirmatory_available
            else None
        ),
        "analytical_cells_complete": analytical_cells_complete,
        "stability_available": stability_available,
        "stability_pass": stability_pass,
        "external_interval_available": external_available,
        "external_pass": external_pass,
        "strict_covariance_interval_available": strict_available,
        "strict_covariance_pass": strict_pass,
        "permitted_claim_classification": classification,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _compact_json(value)
    return value


class MemberLedger:
    """Atomic ordered ``members.csv`` ledger with strict resume validation."""

    def __init__(self, path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
        self.path = path
        self.rows = rows
        self.fieldnames = fieldnames

    @classmethod
    def initialize(
        cls,
        path: Path,
        rows: Sequence[Mapping[str, Any]],
        *,
        design_sha256: str,
        resume: bool = False,
    ) -> MemberLedger:
        expected = [dict(row, design_sha256=design_sha256) for row in rows]
        if resume:
            if not path.is_file():
                raise ProtocolError("resume requested but members.csv does not exist")
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                observed = list(reader)
                fieldnames = list(reader.fieldnames or ())
            if [row.get("member_id") for row in observed] != [
                row.get("member_id") for row in expected
            ]:
                raise ProtocolError("resume member order differs from the materialized design")
            if any(row.get("design_sha256") != design_sha256 for row in observed):
                raise ProtocolError("resume design hash differs from members.csv")
            for observed_row, expected_row in zip(observed, expected, strict=True):
                for key, expected_value in expected_row.items():
                    if key in _MUTABLE_MEMBER_FIELDS:
                        continue
                    if observed_row.get(key, "") != str(_csv_value(expected_value)):
                        raise ProtocolError(
                            f"resume member definition differs for {observed_row.get('member_id')}"
                        )
            return cls(path, observed, fieldnames)

        seen: set[str] = set()
        for row in expected:
            member_id = str(row.get("member_id", ""))
            if not member_id or member_id in seen:
                raise ProtocolError("member IDs must be non-empty and unique")
            seen.add(member_id)
        canonical_tail = [
            "contributing_pixels",
            "retained_bands",
            "status",
            "failure_reason",
            "output_checksum",
            "wall_time_seconds",
            "peak_memory_bytes",
            "design_sha256",
        ]
        keys = [key for row in expected for key in row]
        fieldnames = list(dict.fromkeys([key for key in keys if key not in canonical_tail]))
        fieldnames.extend(key for key in canonical_tail if key not in fieldnames)
        ledger = cls(path, expected, fieldnames)
        ledger._write()
        return ledger

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in self.fieldnames})
        temporary.replace(self.path)

    def update(self, member_id: str, *, status: str, **fields: Any) -> None:
        if status not in {"pending", "running", "complete", "failed", "unavailable"}:
            raise ValueError(f"invalid member status: {status}")
        matches = [row for row in self.rows if row.get("member_id") == member_id]
        if len(matches) != 1:
            raise KeyError(f"member ledger has no unique member {member_id!r}")
        unknown = set(fields) - set(self.fieldnames)
        if unknown:
            insertion = self.fieldnames.index("design_sha256")
            for field in sorted(unknown):
                self.fieldnames.insert(insertion, field)
                insertion += 1
        matches[0]["status"] = status
        matches[0].update(fields)
        self._write()

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(str(row.get("status")) for row in self.rows))


def timing_pilot_fit_ids(members: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, str]]:
    """Select exactly baseline and joint replicate-zero fits for each site."""
    selected: dict[str, tuple[str, str]] = {}
    for site in dict.fromkeys(str(row["site"]) for row in members):
        rows = [row for row in members if row["site"] == site]
        baselines = [row for row in rows if row["member_class"] == "baseline"]
        replicate_zero = [
            row
            for row in rows
            if row["member_class"] == "joint"
            and int(row["stochastic_replicate"]) == 0
            and float(row["ridge"]) == 0.01
        ]
        if len(baselines) != 1 or not replicate_zero:
            raise ProtocolError(f"timing pilot members are incomplete for {site}")
        selected[site] = (str(baselines[0]["fit_id"]), str(replicate_zero[0]["fit_id"]))
    return selected


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def governing_file_provenance(root: Path) -> list[dict[str, Any]]:
    """Hash governing executable/protocol bytes and record their Git state."""
    records: list[dict[str, Any]] = []
    for relative in GOVERNING_FILES:
        path = root / relative
        digest = sha256_file(path)
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", relative],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in result.stdout.splitlines() if line]
        if len(lines) > 1:
            raise ProtocolError(f"ambiguous Git status for governing file: {relative}")
        porcelain = lines[0][:2] if lines else "  "
        records.append(
            {
                "path": relative,
                "sha256": digest,
                "git_status": porcelain,
                "tracked": porcelain != "??",
                "dirty": bool(lines),
            }
        )
    return records


def validate_rockwell_reference(
    root: Path,
    handoff: BlockManifestSite,
    *,
    block_manifest: Path,
) -> dict[str, Any]:
    """Bind the derived Goldfield Rockwell raster to bytes, grid, and M2 handoff."""
    import rasterio
    from affine import Affine

    path = root / "data" / "reference" / (f"rockwell_goldfield_{ANCHOR_SCENES['goldfield']}.tif")
    if not path.is_file():
        raise FileNotFoundError(f"Goldfield external-reference raster is missing: {path}")
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ProtocolError("Goldfield reference raster has no CRS")
        _require_equal("Goldfield reference shape", dataset.shape, handoff.shape)
        _require_equal("Goldfield reference CRS", dataset.crs.to_string(), handoff.crs)
        _require_equal(
            "Goldfield reference transform",
            dataset.transform,
            Affine(*handoff.transform),
        )
        shape = tuple(int(value) for value in dataset.shape)
        crs = dataset.crs.to_string()
        affine = dataset.transform
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "shape": shape,
        "crs": crs,
        "transform": [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f],
        "anchor_scene": handoff.scene,
        "m2_block_manifest": {
            "path": str(block_manifest),
            "sha256": sha256_file(block_manifest),
        },
        "m2_block_rasters": {
            scale: {
                "sha256": scale_handoff.raster_sha256,
                "complete_blocks": len(scale_handoff.block_ids),
            }
            for scale, scale_handoff in sorted(handoff.scales.items())
        },
    }


def _software_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "rasterio", "xarray", "tanager-spec", "tanager-minmap")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "editable-or-unavailable"
    return versions


@dataclass(frozen=True)
class SceneInspection:
    """Preflight-only scene facts needed to freeze the design."""

    site: str
    scene: str
    candidates: dict[str, tuple[str, ...]]
    shape: tuple[int, int]
    crs: str
    transform: tuple[float, ...]
    retained_bands: int


def _scene_path(root: Path, scene: str) -> Path:
    from .config import TANAGER_SR_ASSET

    return root / "data" / "raw" / f"{scene}_{TANAGER_SR_ASSET}.h5"


def _eligible_library_population(
    library: Sequence[Any], retained: np.ndarray
) -> dict[str, tuple[str, ...]]:
    population: dict[str, list[str]] = {mineral: [] for mineral in TARGET_MINERALS}
    for endmember in library:
        reflectance = np.asarray(endmember.reflectance, dtype=float)
        finite = retained & np.isfinite(reflectance)
        if np.count_nonzero(finite) < 2:
            continue
        norm = float(np.linalg.norm(reflectance[finite]))
        if math.isfinite(norm) and norm > 0:
            population[endmember.mineral].append(str(endmember.sample))
    return {mineral: tuple(sorted(population[mineral])) for mineral in TARGET_MINERALS}


def inspect_anchor_scene(root: Path, site: str) -> SceneInspection:
    """Load one anchor only for quality, grid, and library-population preflight."""
    from tanager_spec.io import load_tanager_sr_hdf5

    from .pipeline import PipelinePaths
    from .quality import mask_tanager_scene
    from .speclib import load_library, select_endmembers

    scene = ANCHOR_SCENES[site]
    raw_path = _scene_path(root, scene)
    cube, wavelengths = load_tanager_sr_hdf5(raw_path)
    masked, quality = mask_tanager_scene(cube, wavelengths, raw_path)
    if quality.retained_bands != FROZEN_RETAINED_BANDS:
        raise ProtocolError(
            f"{site} quality policy retained {quality.retained_bands} channels; "
            f"frozen count is {FROZEN_RETAINED_BANDS}"
        )
    retained = np.isfinite(masked.values).any(axis=(1, 2))
    library_dir = PipelinePaths.repo_default(root).speclib_dir
    library = load_library(library_dir, wavelengths)
    population = _eligible_library_population(library, retained)
    for mineral, expected in EXPECTED_CANDIDATE_COUNTS.items():
        _require_equal(f"{site} {mineral} candidate count", len(population[mineral]), expected)
    eligible_names = {name for values in population.values() for name in values}
    eligible = [item for item in library if item.sample in eligible_names]
    selected = select_endmembers(eligible)
    medoids = {mineral: selected[mineral].sample for mineral in TARGET_MINERALS}
    _require_equal(f"{site} baseline medoids", medoids, BASELINE_ENDMEMBERS)
    if masked.rio.crs is None:
        raise ProtocolError(f"{site} anchor cube has no CRS")
    affine = masked.rio.transform()
    return SceneInspection(
        site=site,
        scene=scene,
        candidates=population,
        shape=(masked.sizes["y"], masked.sizes["x"]),
        crs=masked.rio.crs.to_string(),
        transform=(affine.a, affine.b, affine.c, affine.d, affine.e, affine.f),
        retained_bands=quality.retained_bands,
    )


def _decode_draw(value: str) -> tuple[int, ...]:
    records = json.loads(value)
    draw: list[int] = []
    for record in records:
        draw.extend([int(record["block_id"])] * int(record["multiplicity"]))
    return tuple(draw)


def _member_endmember_names(row: Mapping[str, Any]) -> dict[str, str]:
    return {mineral: str(row[f"endmember_{mineral}"]) for mineral in TARGET_MINERALS}


def _design_payload(
    design: Mapping[str, Any],
    *,
    root: Path,
    args: Any,
    protocol: Mapping[str, Any],
    block_manifest: Path,
    input_manifest: Mapping[str, Any],
    rockwell_reference: Mapping[str, Any] | None,
    deviations: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **design,
        "protocol": dict(protocol),
        "protocol_deviations": dict(deviations),
        "protocol_amendment": protocol.get("amendment"),
        "code_commit": _git_revision(root),
        "governing_files": governing_file_provenance(root),
        "lockfile_sha256": sha256_file(root / "uv.lock"),
        "input_manifest": input_manifest,
        "rockwell_reference": rockwell_reference,
        "quality_policy": {
            "path": "docs/tanager_quality_mask_policy.md",
            "sha256": sha256_file(root / "docs" / "tanager_quality_mask_policy.md"),
            "retained_bands": FROZEN_RETAINED_BANDS,
        },
        "block_manifest": {
            "path": str(block_manifest),
            "sha256": sha256_file(block_manifest),
        },
        "software": _software_versions(),
        "compute_controls": {
            "device": args.device,
            "batch_size": args.batch_size,
            "storage_layout": args.storage_layout,
            "numpy_reference": True,
            "accelerator_backend": None,
        },
    }


def _write_members_table(
    path: Path, rows: Sequence[Mapping[str, Any]], design_sha: str
) -> MemberLedger:
    return MemberLedger.initialize(path, rows, design_sha256=design_sha)


def scientific_design_sha256(design: Mapping[str, Any]) -> str:
    """Hash scientific identity while excluding preregistered inert compute controls."""
    identity = {
        key: value
        for key, value in design.items()
        if key not in {"compute_controls", "scientific_design_sha256"}
    }
    return hashlib.sha256(_compact_json(identity).encode("utf-8")).hexdigest()


def _materialize_design(
    output_dir: Path,
    design: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    *,
    resume: bool,
) -> tuple[Path, MemberLedger]:
    design_path = output_dir / "design.json"
    members_path = output_dir / "members.csv"
    expected_identity = scientific_design_sha256(design)
    if design.get("scientific_design_sha256") != expected_identity:
        raise ProtocolError("materialized design carries an invalid scientific identity hash")
    if resume:
        if not design_path.is_file():
            raise ProtocolError("resume requested but design.json does not exist")
        observed_design = json.loads(design_path.read_text(encoding="utf-8"))
        observed_identity = scientific_design_sha256(observed_design)
        if observed_design.get("scientific_design_sha256") != observed_identity:
            raise ProtocolError("resume design.json has an invalid scientific identity hash")
        if observed_identity != expected_identity:
            raise ProtocolError("resume scientific design identity differs from current preflight")
        design_sha = sha256_file(design_path)
        ledger = MemberLedger.initialize(
            members_path, members, design_sha256=design_sha, resume=True
        )
        return design_path, ledger
    if design_path.exists() or members_path.exists():
        raise ProtocolError(
            "design artifacts already exist; use --resume or a new output directory"
        )
    strict_json_dump(design_path, design)
    design_sha = sha256_file(design_path)
    ledger = _write_members_table(members_path, members, design_sha)
    return design_path, ledger


def _load_scene_for_execution(root: Path, site: str) -> tuple[Any, dict[str, Any], np.ndarray, Any]:
    import rasterio
    from tanager_spec.io import load_tanager_sr_hdf5

    from .pipeline import PipelinePaths
    from .quality import mask_tanager_scene
    from .speclib import load_library

    scene = ANCHOR_SCENES[site]
    raw_path = _scene_path(root, scene)
    cube, wavelengths = load_tanager_sr_hdf5(raw_path)
    masked, quality = mask_tanager_scene(cube, wavelengths, raw_path)
    if quality.retained_bands != FROZEN_RETAINED_BANDS:
        raise ProtocolError(f"{site} retained-band count changed after preflight")
    library = load_library(PipelinePaths.repo_default(root).speclib_dir, wavelengths)
    by_name = {item.sample: item for item in library}
    return masked, by_name, wavelengths, rasterio


def _fit_checksum(fit: MtmfFit) -> str:
    digest = hashlib.sha256()
    for mineral in TARGET_MINERALS:
        digest.update(np.ascontiguousarray(fit.matched_filter[mineral]).tobytes())
        digest.update(np.ascontiguousarray(fit.infeasibility[mineral]).tobytes())
    return digest.hexdigest()


def _fit_from_row(
    cube: Any,
    row: Mapping[str, Any],
    library: Mapping[str, Any],
    block_values: np.ndarray,
) -> MtmfFit:
    names = _member_endmember_names(row)
    try:
        endmembers = {mineral: library[names[mineral]] for mineral in TARGET_MINERALS}
    except KeyError as error:
        raise FitFailure(f"materialized endmember disappeared: {error.args[0]}") from error
    draw = _decode_draw(str(row["covariance_draw"]))
    return fit_mtmf_numpy(
        cube,
        endmembers,
        ridge=float(row["ridge"]),
        block_ids=block_values if draw else None,
        covariance_draw=draw,
    )


def _strict_covariance_score_maps(
    cube: Any,
    row: Mapping[str, Any],
    library: Mapping[str, Any],
    block_records: Sequence[Mapping[str, Any]],
    *,
    shape: tuple[int, int],
    halo_pixels: int,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, str]]:
    """Build full-scene score maps under each held-out covariance exclusion."""
    names = _member_endmember_names(row)
    try:
        endmembers = {mineral: library[names[mineral]] for mineral in TARGET_MINERALS}
    except KeyError as error:
        raise FitFailure(f"materialized endmember disappeared: {error.args[0]}") from error
    scores: dict[str, dict[int, np.ndarray]] = {mineral: {} for mineral in TARGET_MINERALS}
    failures: dict[int, str] = {}
    for record in block_records:
        block_id = int(record["numeric_block_id"])
        try:
            fit = fit_strict_covariance_exclusion_numpy(
                cube,
                endmembers,
                ridge=float(row["ridge"]),
                held_out_block=record,
                halo_pixels=halo_pixels,
            )
        except FitFailure as error:
            failures[block_id] = str(error)
            continue
        for mineral in TARGET_MINERALS:
            score = np.asarray(fit.matched_filter[mineral], dtype=float)
            if score.shape != shape:
                raise FitFailure("strict covariance score map changed anchor shape")
            scores[mineral][block_id] = score
    return scores, failures


def _save_fit(path: Path, fit: MtmfFit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    payload: dict[str, Any] = {
        "valid_support": fit.valid_support,
        "contributing_pixels": np.asarray(fit.contributing_pixels),
        "retained_bands": np.asarray(fit.retained_bands),
    }
    for mineral in TARGET_MINERALS:
        payload[f"mf_{mineral}"] = fit.matched_filter[mineral]
        payload[f"infeas_{mineral}"] = fit.infeasibility[mineral]
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _load_fit(path: Path) -> MtmfFit:
    with np.load(path, allow_pickle=False) as values:
        return MtmfFit(
            matched_filter={mineral: values[f"mf_{mineral}"] for mineral in TARGET_MINERALS},
            infeasibility={mineral: values[f"infeas_{mineral}"] for mineral in TARGET_MINERALS},
            valid_support=values["valid_support"],
            contributing_pixels=int(values["contributing_pixels"]),
            retained_bands=int(values["retained_bands"]),
        )


def _cache_name(fit_id: str) -> str:
    return hashlib.sha256(fit_id.encode("utf-8")).hexdigest() + ".npz"


def _run_timing_pilot(
    *,
    root: Path,
    output_dir: Path,
    members: Sequence[Mapping[str, Any]],
    handoffs: Mapping[str, BlockManifestSite],
) -> Path:
    records: list[dict[str, Any]] = []
    selected = timing_pilot_fit_ids(members)
    for site, fit_ids in selected.items():
        cube, library, _, rasterio = _load_scene_for_execution(root, site)
        with rasterio.open(handoffs[site].raster_path) as dataset:
            blocks = dataset.read(1, masked=False)
        site_rows = [row for row in members if row["site"] == site]
        for fit_id in fit_ids:
            row = next(row for row in site_rows if row["fit_id"] == fit_id)
            tracemalloc.start()
            start = time.perf_counter()
            fit = _fit_from_row(cube, row, library, blocks)
            wall_time = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            records.append(
                {
                    "site": site,
                    "scene": ANCHOR_SCENES[site],
                    "fit_id": fit_id,
                    "member_class": row["member_class"],
                    "stochastic_replicate": row["stochastic_replicate"],
                    "wall_time_seconds": wall_time,
                    "peak_memory_bytes": peak,
                    "output_sha256": _fit_checksum(fit),
                    "device": "cpu",
                    "scientific_outputs_retained": False,
                }
            )
    path = output_dir / "timing_pilot.json"
    strict_json_dump(
        path,
        {
            "schema_version": "1.0",
            "mode": "timing_pilot_only",
            "fit_count": len(records),
            "records": records,
        },
    )
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    temporary.replace(path)


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    common = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(common) < 2:
        return None
    value = float(spearmanr(left[common], right[common]).statistic)
    return value if math.isfinite(value) else None


def _write_geotiff(
    path: Path,
    values: np.ndarray,
    *,
    crs: str,
    transform: Sequence[float],
    dtype: str,
    nodata: float | int,
) -> None:
    import rasterio
    from affine import Affine

    array = np.asarray(values).astype(dtype, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=Affine(*transform),
        nodata=nodata,
        compress="lzw",
    ) as dataset:
        dataset.write(array, 1)
    temporary.replace(path)


def _block_records(payload: Mapping[str, Any], site: str, scale: str = "L") -> list[dict[str, Any]]:
    records = [
        dict(record)
        for record in payload.get("blocks", ())
        if record.get("site") == site and record.get("scale") == scale and record.get("complete")
    ]
    return sorted(records, key=lambda record: int(record["numeric_block_id"]))


def _binary_rockwell_reference(reference: np.ndarray, mineral: str) -> np.ndarray | None:
    from .reference import MINERAL_TO_ROCKWELL, ROCKWELL_EXCLUDED

    positive = MINERAL_TO_ROCKWELL.get(mineral)
    if positive is None:
        return None
    values = np.asarray(reference)
    binary = np.full(values.shape, np.nan, dtype=float)
    domain = np.isfinite(values) & ~np.isin(values, tuple(ROCKWELL_EXCLUDED))
    binary[domain] = np.isin(values[domain], tuple(positive)).astype(float)
    return binary


def calibration_diagnostic(
    frequency: np.ndarray,
    binary_reference: np.ndarray,
    block_ids: np.ndarray,
    *,
    site: str,
    mineral: str,
    bootstrap_replicates: int = FROZEN_BOOTSTRAP_REPLICATES,
    seed: int = FROZEN_SEED,
) -> list[dict[str, Any]]:
    """Compute fixed-bin alteration compatibility and paired block intervals."""
    values = np.asarray(frequency, dtype=float)
    reference = np.asarray(binary_reference, dtype=float)
    blocks = np.asarray(block_ids)
    if values.shape != reference.shape or values.shape != blocks.shape:
        raise ValueError("calibration frequency, reference, and blocks must align")
    domain = np.isfinite(values) & np.isfinite(reference) & (blocks > 0)
    unique_blocks = tuple(sorted(int(value) for value in np.unique(blocks[domain])))
    if not unique_blocks:
        return [
            {
                "site": site,
                "mineral": mineral,
                "confidence_bin": f"[{index / 10:.1f},{(index + 1) / 10:.1f}"
                + ("]" if index == 9 else ")"),
                "support_blocks": 0,
                "support_pixels": 0,
                "compatible_positive_rate": None,
                "interval_lower": None,
                "interval_upper": None,
                "scheduled_replicates": bootstrap_replicates,
                "valid_replicates": 0,
                "finite_fraction": 0.0,
                "interval_available": False,
                "unavailable_reason": "zero_complete_blocks",
                "brier_score": None,
                "brier_interval_lower": None,
                "brier_interval_upper": None,
                "brier_interval_available": False,
                "brier_valid_replicates": 0,
                "brier_finite_fraction": 0.0,
                "expected_calibration_error": None,
                "ece_interval_lower": None,
                "ece_interval_upper": None,
                "ece_interval_available": False,
                "ece_valid_replicates": 0,
                "ece_finite_fraction": 0.0,
                "status": "unavailable_zero_complete_blocks",
            }
            for index in range(10)
        ]
    brier = float(np.mean((values[domain] - reference[domain]) ** 2))
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        np.asarray(unique_blocks),
        size=(bootstrap_replicates, len(unique_blocks)),
        replace=True,
    )
    brier_replicates = np.full(bootstrap_replicates, np.nan, dtype=float)
    ece_replicates = np.full(bootstrap_replicates, np.nan, dtype=float)
    for replicate, draw in enumerate(draws):
        draw_counts = Counter(int(value) for value in draw)
        squared_error_sum = 0.0
        total_pixels = 0
        ece_weighted_sum = 0.0
        for block_id, multiplicity in draw_counts.items():
            use = domain & (blocks == block_id)
            count = int(np.count_nonzero(use))
            if count:
                squared_error_sum += multiplicity * float(
                    np.sum((values[use] - reference[use]) ** 2)
                )
                total_pixels += multiplicity * count
        if total_pixels:
            brier_replicates[replicate] = squared_error_sum / total_pixels
            for bin_index in range(10):
                bin_lower = bin_index / 10
                bin_upper = (bin_index + 1) / 10
                observed_sum = 0.0
                prediction_sum = 0.0
                bin_pixels = 0
                for block_id, multiplicity in draw_counts.items():
                    use = (
                        domain
                        & (blocks == block_id)
                        & (values >= bin_lower)
                        & ((values <= bin_upper) if bin_index == 9 else (values < bin_upper))
                    )
                    count = int(np.count_nonzero(use))
                    if count:
                        observed_sum += multiplicity * float(np.sum(reference[use]))
                        prediction_sum += multiplicity * float(np.sum(values[use]))
                        bin_pixels += multiplicity * count
                if bin_pixels:
                    ece_weighted_sum += bin_pixels * abs(
                        observed_sum / bin_pixels - prediction_sum / bin_pixels
                    )
            ece_replicates[replicate] = ece_weighted_sum / total_pixels
    from .spatial_validation import _governed_confidence_interval

    brier_interval = _governed_confidence_interval(
        "brier_score",
        brier_replicates,
        scheduled_replicates=bootstrap_replicates,
    )
    ece_interval = _governed_confidence_interval(
        "expected_calibration_error",
        ece_replicates,
        scheduled_replicates=bootstrap_replicates,
    )
    enough_blocks = len(unique_blocks) >= MINIMUM_INTERVAL_BLOCKS
    brier_available = brier_interval.gate_eligible and enough_blocks
    ece_available = ece_interval.gate_eligible and enough_blocks
    rows: list[dict[str, Any]] = []
    bin_rates: list[tuple[int, float, float]] = []
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        in_bin = (
            domain & (values >= lower) & ((values <= upper) if index == 9 else (values < upper))
        )
        support_pixels = int(np.count_nonzero(in_bin))
        support_blocks = int(len(np.unique(blocks[in_bin]))) if support_pixels else 0
        observed_rate = float(np.mean(reference[in_bin])) if support_pixels else None
        mean_frequency = float(np.mean(values[in_bin])) if support_pixels else None
        if observed_rate is not None and mean_frequency is not None:
            bin_rates.append((support_pixels, observed_rate, mean_frequency))
        replicate_rates = np.full(bootstrap_replicates, np.nan, dtype=float)
        for replicate, draw in enumerate(draws):
            draw_counts = Counter(int(value) for value in draw)
            numerator = 0.0
            denominator = 0
            for block_id, multiplicity in draw_counts.items():
                use = in_bin & (blocks == block_id)
                count = int(np.count_nonzero(use))
                if count:
                    numerator += multiplicity * float(np.sum(reference[use]))
                    denominator += multiplicity * count
            if denominator:
                replicate_rates[replicate] = numerator / denominator
        governed = _governed_confidence_interval(
            "compatible_positive_rate",
            replicate_rates,
            scheduled_replicates=bootstrap_replicates,
        )
        interval_available = governed.gate_eligible and len(unique_blocks) >= 2
        interval_lower = governed.lower if interval_available else None
        interval_upper = governed.upper if interval_available else None
        rows.append(
            {
                "site": site,
                "mineral": mineral,
                "confidence_bin": f"[{lower:.1f},{upper:.1f}" + ("]" if index == 9 else ")"),
                "support_blocks": support_blocks,
                "support_pixels": support_pixels,
                "compatible_positive_rate": observed_rate,
                "interval_lower": interval_lower,
                "interval_upper": interval_upper,
                "scheduled_replicates": bootstrap_replicates,
                "valid_replicates": governed.valid_replicates,
                "finite_fraction": governed.finite_fraction,
                "interval_available": interval_available,
                "unavailable_reason": (
                    None
                    if interval_available
                    else (
                        "fewer_than_two_complete_blocks"
                        if len(unique_blocks) < 2
                        else governed.unavailable_reason
                    )
                ),
                "brier_score": brier,
                "brier_interval_lower": (brier_interval.lower if brier_available else None),
                "brier_interval_upper": (brier_interval.upper if brier_available else None),
                "brier_interval_available": brier_available,
                "brier_valid_replicates": brier_interval.valid_replicates,
                "brier_finite_fraction": brier_interval.finite_fraction,
                "expected_calibration_error": None,
                "ece_interval_lower": ece_interval.lower if ece_available else None,
                "ece_interval_upper": ece_interval.upper if ece_available else None,
                "ece_interval_available": ece_available,
                "ece_valid_replicates": ece_interval.valid_replicates,
                "ece_finite_fraction": ece_interval.finite_fraction,
                "status": "complete" if support_pixels else "empty_fixed_bin",
            }
        )
    total = sum(count for count, _, _ in bin_rates)
    ece = (
        sum(count * abs(rate - prediction) for count, rate, prediction in bin_rates) / total
        if total
        else None
    )
    for row in rows:
        row["expected_calibration_error"] = ece
    return rows


def _load_goldfield_reference(root: Path, shape: tuple[int, int]) -> np.ndarray:
    import rasterio

    path = root / "data" / "reference" / (f"rockwell_goldfield_{ANCHOR_SCENES['goldfield']}.tif")
    if not path.is_file():
        raise FileNotFoundError(f"Goldfield external-reference raster is missing: {path}")
    with rasterio.open(path) as dataset:
        _require_equal("Goldfield reference shape", dataset.shape, shape)
        # The categorical Rockwell reference is stored as an integer raster.
        # Convert the masked array before filling so NumPy can represent the
        # missing category as NaN without attempting an invalid integer cast.
        return dataset.read(1, masked=True).astype(np.float64).filled(np.nan)


def paired_factor_effect_rows(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = FROZEN_BOOTSTRAP_REPLICATES,
) -> list[dict[str, Any]]:
    """Return frozen, member-paired factor deltas on common block support."""
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
    rows_by_context: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in metric_rows:
        if row.get("aggregation") != "block":
            continue
        context = (
            str(row["site"]),
            str(row["mineral"]),
            str(row.get("block_scale", "L")),
            int(row.get("block_id", 0)),
        )
        rows_by_context.setdefault(context, []).append(row)

    paired: dict[tuple[str, str, str, str, str, str], dict[str, dict[int, float]]] = {}
    support_pixels: dict[tuple[str, str, str, str, str, str], int] = {}
    for (site, mineral, scale, block_id), context_rows in rows_by_context.items():
        baselines = [row for row in context_rows if row["member_class"] == "baseline"]
        if len(baselines) != 1:
            continue
        baseline = baselines[0]
        candidate_pairs: list[tuple[str, str, str, Mapping[str, Any], Mapping[str, Any]]] = []
        for row in context_rows:
            member_class = str(row["member_class"])
            replicate = row.get("stochastic_replicate")
            if member_class in {"endmember_only", "covariance_only", "calibration_only"}:
                candidate_pairs.append(
                    ("axis", member_class, f"replicate:{replicate}", row, baseline)
                )
            elif member_class == "joint":
                controls = (
                    ("ridge", "0.01", "ridge", "0.01"),
                    ("quantile", "0.9", "detection_quantile", "0.9"),
                    ("gate", "1", "infeasibility_gate", "1"),
                )
                for factor, reference_level, field, reference_value in controls:
                    level = str(row[field])
                    if level == reference_level:
                        continue
                    matches = [
                        other
                        for other in context_rows
                        if other["member_class"] == "joint"
                        and other.get("stochastic_replicate") == replicate
                        and str(other[field]) == reference_value
                        and all(
                            str(other[other_field]) == str(row[other_field])
                            for other_field in (
                                "ridge",
                                "detection_quantile",
                                "infeasibility_gate",
                            )
                            if other_field != field
                        )
                    ]
                    if len(matches) == 1:
                        pair_id = _compact_json(
                            {
                                "replicate": replicate,
                                "ridge": row["ridge"] if field != "ridge" else reference_value,
                                "quantile": (
                                    row["detection_quantile"]
                                    if field != "detection_quantile"
                                    else reference_value
                                ),
                                "gate": (
                                    row["infeasibility_gate"]
                                    if field != "infeasibility_gate"
                                    else reference_value
                                ),
                            }
                        )
                        candidate_pairs.append((factor, level, pair_id, row, matches[0]))
        for factor, level, pair_id, treatment, reference_row in candidate_pairs:
            for endpoint in endpoints:
                treatment_value = treatment.get(endpoint)
                reference_value = reference_row.get(endpoint)
                delta = float("nan")
                if treatment_value is not None and reference_value is not None:
                    candidate = float(treatment_value) - float(reference_value)
                    if math.isfinite(candidate):
                        delta = candidate
                key = (site, mineral, scale, factor, level, endpoint)
                paired.setdefault(key, {}).setdefault(pair_id, {})[block_id] = delta
                support_pixels[key] = support_pixels.get(key, 0) + min(
                    int(treatment.get("common_support_pixels") or 0),
                    int(reference_row.get("common_support_pixels") or 0),
                )

    output: list[dict[str, Any]] = []
    for key, member_values in sorted(paired.items()):
        site, mineral, scale, factor, level, endpoint = key
        interval = nested_block_bootstrap(
            member_values,
            replicates=bootstrap_replicates,
            seed=FROZEN_SEED,
        )
        observed = np.asarray(
            [value for values in member_values.values() for value in values.values()],
            dtype=float,
        )
        finite = observed[np.isfinite(observed)]
        output.append(
            {
                "site": site,
                "mineral": mineral,
                "block_scale": scale,
                "factor": factor,
                "level": level,
                "reference_level": "baseline"
                if factor == "axis"
                else {
                    "ridge": "0.01",
                    "quantile": "0.9",
                    "gate": "1",
                }[factor],
                "endpoint": endpoint,
                "paired_delta_median": float(np.median(finite)) if finite.size else None,
                "interval_lower": interval.lower_95,
                "interval_upper": interval.upper_95,
                "scheduled_replicates": interval.scheduled_replicates,
                "valid_replicates": interval.valid_replicates,
                "finite_fraction": interval.finite_fraction,
                "interval_available": interval.interval_available,
                "unavailable_reason": interval.unavailable_reason,
                "n_pairs": len(member_values),
                "complete_blocks": len(interval.block_ids),
                "paired_support_pixels": support_pixels[key],
                "contrast_status": "descriptive_paired_complete_block",
            }
        )
    return output


def _execute_site(
    *,
    root: Path,
    output_dir: Path,
    site: str,
    site_index: int,
    members: Sequence[Mapping[str, Any]],
    ledger: MemberLedger,
    handoff: BlockManifestSite,
    manifest_payload: Mapping[str, Any],
    storage_layout: str,
    bootstrap_replicates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[Path]]:
    cube, library, _, rasterio = _load_scene_for_execution(root, site)
    block_values_by_scale: dict[str, np.ndarray] = {}
    for scale, scale_handoff in handoff.scales.items():
        with rasterio.open(scale_handoff.raster_path) as dataset:
            block_values_by_scale[scale] = dataset.read(1, masked=False)
    block_values = block_values_by_scale["L"]
    site_rows = [row for row in members if row["site"] == site]
    fit_rows: dict[str, Mapping[str, Any]] = {}
    for row in site_rows:
        fit_rows.setdefault(str(row["fit_id"]), row)
    if len(fit_rows) != FROZEN_UNIQUE_FITS_PER_SCENE:
        raise ProtocolError(f"{site} score plan does not contain 83 unique fits")
    memory_cache: dict[str, MtmfFit] = {}
    fit_failures: dict[str, str] = {}
    fit_checksums: dict[str, str] = {}
    external_cache: dict[tuple[str, str, int, str, str], CrossFittedMetrics] = {}
    strict_score_cache: dict[
        tuple[str, str], tuple[dict[str, dict[int, np.ndarray]], dict[int, str]]
    ] = {}
    nested_external_members: dict[tuple[str, str, str], dict[str, CrossFittedMetrics]] = {}
    cache_dir = output_dir / ".score_cache" / site
    for fit_id, row in fit_rows.items():
        cache_path = cache_dir / _cache_name(fit_id)
        if cache_path.is_file():
            fit = _load_fit(cache_path)
        else:
            try:
                fit = _fit_from_row(cube, row, library, block_values)
            except FitFailure as error:
                fit_failures[fit_id] = str(error)
                continue
            if storage_layout == "disk":
                _save_fit(cache_path, fit)
        if storage_layout == "memory":
            memory_cache[fit_id] = fit
        fit_checksums[fit_id] = _fit_checksum(fit)

    shape = handoff.shape
    accumulators = {mineral: MapAccumulator(shape) for mineral in TARGET_MINERALS}
    metric_rows: list[dict[str, Any]] = []
    baseline_scores: dict[str, np.ndarray] | None = None
    baseline_detections: dict[str, DetectionResult] | None = None
    baseline_dominant: np.ndarray | None = None
    baseline_fit_support: np.ndarray | None = None
    joint_dominant: list[np.ndarray] = []
    endpoint_block_counts: dict[
        tuple[str, str, str], dict[str, dict[int, tuple[float, float]]]
    ] = {}
    rank_block_pairs: dict[
        tuple[str, str, str],
        dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    ] = {}
    failed_rows: list[dict[str, str]] = []
    reference = _load_goldfield_reference(root, shape) if site == "goldfield" else None
    records_by_scale = {
        scale: _block_records(manifest_payload, site, scale) for scale in ("L", "2L")
    }

    for row in site_rows:
        member_id = str(row["member_id"])
        fit_id = str(row["fit_id"])
        if fit_id in fit_failures:
            reason = fit_failures[fit_id]
            ledger.update(member_id, status="failed", failure_reason=reason)
            failed_rows.append({"member_id": member_id, "fit_id": fit_id, "reason": reason})
            if row["member_class"] == "joint":
                for accumulator in accumulators.values():
                    accumulator.record_failure(member_id, reason)
            continue
        fit = memory_cache.get(fit_id)
        if fit is None:
            fit = _load_fit(cache_dir / _cache_name(fit_id))
        calibration_draw = _decode_draw(str(row["calibration_draw"]))
        gate = _as_gate(row["infeasibility_gate"])
        detections: dict[str, DetectionResult] = {}
        for mineral in TARGET_MINERALS:
            if row["calibration_mode"] == "full_scene":
                result = full_scene_detection(
                    fit.matched_filter[mineral],
                    fit.infeasibility[mineral],
                    quantile=float(row["detection_quantile"]),
                    max_infeasibility=gate,
                )
            else:
                result = operational_detection(
                    fit.matched_filter[mineral],
                    fit.infeasibility[mineral],
                    block_values,
                    calibration_draw=calibration_draw,
                    quantile=float(row["detection_quantile"]),
                    max_infeasibility=gate,
                )
            detections[mineral] = result
        unavailable = [
            result.reason for result in detections.values() if result.status == "unavailable"
        ]
        if unavailable:
            reason = ";".join(sorted(set(str(item) for item in unavailable)))
            ledger.update(
                member_id,
                status="unavailable",
                failure_reason=reason,
                contributing_pixels=fit.contributing_pixels,
                retained_bands=fit.retained_bands,
                output_checksum=fit_checksums[fit_id],
            )
            continue
        classes, _ = dominant_class(fit.matched_filter, detections)
        if row["member_class"] == "baseline":
            baseline_scores = fit.matched_filter
            baseline_detections = detections
            baseline_dominant = classes
            baseline_fit_support = fit.valid_support
        if (
            baseline_scores is None
            or baseline_detections is None
            or baseline_dominant is None
            or baseline_fit_support is None
        ):
            raise AssertionError("baseline member must be first in frozen member order")
        common_class = baseline_fit_support & fit.valid_support
        class_switch = (
            float(np.mean(classes[common_class] != baseline_dominant[common_class]))
            if np.any(common_class)
            else None
        )
        for mineral in TARGET_MINERALS:
            result = detections[mineral]
            assert result.detections is not None and result.valid_support is not None
            if row["member_class"] == "joint":
                accumulators[mineral].add(result.detections, result.valid_support)
            baseline_support = np.isfinite(baseline_scores[mineral])
            member_support = np.isfinite(fit.matched_filter[mineral])
            common = baseline_support & member_support
            rank = _spearman(baseline_scores[mineral], fit.matched_filter[mineral])
            external_by_scale: dict[str, CrossFittedMetrics] = {}
            strict_by_scale: dict[str, CrossFittedMetrics] = {}
            if reference is not None:
                binary = _binary_rockwell_reference(reference, mineral)
                if binary is not None:
                    replicate = int(row["stochastic_replicate"] or 0)
                    for scale in ("L", "2L"):
                        scale_handoff = handoff.scales[scale]
                        scale_blocks = block_values_by_scale[scale]
                        records = records_by_scale[scale]
                        cache_key = (
                            fit_id,
                            mineral,
                            replicate,
                            scale,
                            "full_scene_covariance",
                        )
                        external = external_cache.get(cache_key)
                        if external is None:
                            external = spatially_cross_fitted_threshold_evaluation(
                                fit.matched_filter[mineral],
                                binary,
                                scale_blocks,
                                records,
                                halo_pixels=scale_handoff.halo_pixels,
                                site_index=site_index,
                                stochastic_replicate=replicate,
                                covariance_scope="full_scene_covariance",
                            )
                            external_cache[cache_key] = external
                        external_by_scale[scale] = external

                        strict_cache_key = (fit_id, scale)
                        strict_cached = strict_score_cache.get(strict_cache_key)
                        if strict_cached is None:
                            strict_cached = _strict_covariance_score_maps(
                                cube,
                                row,
                                library,
                                records,
                                shape=shape,
                                halo_pixels=scale_handoff.halo_pixels,
                            )
                            strict_score_cache[strict_cache_key] = strict_cached
                        strict_scores, _ = strict_cached
                        strict_key = (
                            fit_id,
                            mineral,
                            replicate,
                            scale,
                            "strict_covariance_exclusion",
                        )
                        strict = external_cache.get(strict_key)
                        if strict is None:
                            strict = strict_covariance_cross_fitted_threshold_evaluation(
                                strict_scores[mineral],
                                binary,
                                scale_blocks,
                                records,
                                halo_pixels=scale_handoff.halo_pixels,
                                site_index=site_index,
                                stochastic_replicate=replicate,
                            )
                            external_cache[strict_key] = strict
                        strict_by_scale[scale] = strict
                        if row["member_class"] == "joint":
                            nested_external_members.setdefault(
                                ("full_scene_covariance", scale, mineral), {}
                            )[member_id] = external
                            nested_external_members.setdefault(
                                ("strict_covariance_exclusion", scale, mineral), {}
                            )[member_id] = strict
            valid_pixels = int(np.count_nonzero(result.valid_support))
            external_l = external_by_scale.get("L")
            pooled_external = (
                _external_metric_values(external_l.block_results) if external_l is not None else {}
            )
            base_fields = {
                "site": site,
                "scene": ANCHOR_SCENES[site],
                "mineral": mineral,
                "member_id": member_id,
                "member_class": row["member_class"],
                "stochastic_replicate": row["stochastic_replicate"],
                "ridge": row["ridge"],
                "detection_quantile": row["detection_quantile"],
                "infeasibility_gate": row["infeasibility_gate"],
            }
            metric_rows.append(
                {
                    **base_fields,
                    "aggregation": "scene",
                    "block_scale": "L",
                    "block_id": 0,
                    "common_support_pixels": int(np.count_nonzero(common)),
                    "common_support_loss_fraction": (
                        1.0 - np.count_nonzero(common) / np.count_nonzero(baseline_support)
                        if np.count_nonzero(baseline_support)
                        else None
                    ),
                    "detection_prevalence": (
                        float(np.mean(result.detections[result.valid_support]))
                        if valid_pixels
                        else None
                    ),
                    "rank_correlation": rank,
                    "dominant_class_switch_frequency": class_switch,
                    **{
                        name: pooled_external.get(name)
                        for name in (
                            "auc",
                            "balanced_accuracy",
                            "positive_f1",
                            "negative_f1",
                            "macro_f1",
                            "tpr",
                            "fpr",
                            "prevalence",
                        )
                    },
                    "external_status": (
                        "not_applicable" if external_l is None else external_l.status
                    ),
                    "covariance_scope": (
                        "not_applicable" if external_l is None else external_l.covariance_scope
                    ),
                    "strict_covariance_exclusion_status": (
                        "not_applicable"
                        if "L" not in strict_by_scale
                        else strict_by_scale["L"].status
                    ),
                }
            )
            for scale in ("L", "2L"):
                scale_blocks = block_values_by_scale[scale]
                external_blocks = {
                    int(item["block_id"]): item
                    for item in external_by_scale.get(
                        scale,
                        CrossFittedMetrics(
                            "unavailable",
                            "not_applicable",
                            "full_scene_covariance",
                            None,
                            None,
                            0,
                            0,
                            0,
                        ),
                    ).block_results
                }
                for block_record in records_by_scale[scale]:
                    block_id = int(block_record["numeric_block_id"])
                    in_block = scale_blocks == block_id
                    block_common = common & in_block
                    block_valid = result.valid_support & in_block
                    block_baseline = baseline_support & in_block
                    block_external = external_blocks.get(block_id)
                    block_external_values = (
                        _external_metric_values((block_external,))
                        if block_external is not None
                        else {}
                    )
                    block_rank = _spearman(
                        np.where(in_block, baseline_scores[mineral], np.nan),
                        np.where(in_block, fit.matched_filter[mineral], np.nan),
                    )
                    switch_support = common_class & in_block
                    block_switch = (
                        float(np.mean(classes[switch_support] != baseline_dominant[switch_support]))
                        if np.any(switch_support)
                        else None
                    )
                    block_row = {
                        **base_fields,
                        "aggregation": "block",
                        "block_scale": scale,
                        "block_id": block_id,
                        "common_support_pixels": int(np.count_nonzero(block_common)),
                        "common_support_loss_fraction": (
                            1.0 - np.count_nonzero(block_common) / np.count_nonzero(block_baseline)
                            if np.count_nonzero(block_baseline)
                            else None
                        ),
                        "detection_prevalence": (
                            float(np.mean(result.detections[block_valid]))
                            if np.any(block_valid)
                            else None
                        ),
                        "rank_correlation": block_rank,
                        "dominant_class_switch_frequency": block_switch,
                        **{
                            name: block_external_values.get(name)
                            for name in (
                                "auc",
                                "balanced_accuracy",
                                "positive_f1",
                                "negative_f1",
                                "macro_f1",
                                "tpr",
                                "fpr",
                                "prevalence",
                            )
                        },
                        "external_status": (
                            "not_applicable" if block_external is None else "block_recorded"
                        ),
                        "covariance_scope": (
                            "not_applicable" if block_external is None else "full_scene_covariance"
                        ),
                        "strict_covariance_exclusion_status": (
                            "not_applicable"
                            if scale not in strict_by_scale
                            else strict_by_scale[scale].status
                        ),
                    }
                    metric_rows.append(block_row)
                    if row["member_class"] == "joint":
                        count_metrics = {
                            "detection_prevalence": (
                                float(np.count_nonzero(result.detections & block_valid)),
                                float(np.count_nonzero(block_valid)),
                            ),
                            "common_support_loss_fraction": (
                                float(
                                    np.count_nonzero(block_baseline)
                                    - np.count_nonzero(block_common)
                                ),
                                float(np.count_nonzero(block_baseline)),
                            ),
                            "dominant_class_switch_frequency": (
                                float(
                                    np.count_nonzero(
                                        (classes != baseline_dominant) & switch_support
                                    )
                                ),
                                float(np.count_nonzero(switch_support)),
                            ),
                        }
                        for endpoint, counts in count_metrics.items():
                            endpoint_block_counts.setdefault(
                                (scale, mineral, endpoint), {}
                            ).setdefault(member_id, {})[block_id] = counts
                        rank_block_pairs.setdefault(
                            (scale, mineral, "rank_correlation"), {}
                        ).setdefault(member_id, {})[block_id] = (
                            baseline_scores[mineral][block_common],
                            fit.matched_filter[mineral][block_common],
                        )
        if row["member_class"] == "joint":
            joint_dominant.append(classes)
        ledger.update(
            member_id,
            status="complete",
            failure_reason=None,
            contributing_pixels=fit.contributing_pixels,
            retained_bands=fit.retained_bands,
            output_checksum=fit_checksums[fit_id],
        )

    if baseline_dominant is None or baseline_detections is None:
        raise ProtocolError(f"{site} baseline member failed; aggregate maps are unavailable")
    if not joint_dominant:
        raise ProtocolError(f"{site} has no valid joint members; aggregate maps are unavailable")
    endpoint_intervals: list[dict[str, Any]] = []
    endpoint_results: dict[tuple[str, str, str], NestedBootstrapResult] = {}
    for key, member_counts in sorted(endpoint_block_counts.items()):
        scale, mineral, endpoint = key
        interval = nested_ratio_bootstrap(
            member_counts,
            replicates=bootstrap_replicates,
            seed=FROZEN_SEED,
        )
        endpoint_results[key] = interval
        endpoint_intervals.append(
            {
                "scale": scale,
                "mineral": mineral,
                "metric": endpoint,
                "lower_95": interval.lower_95,
                "upper_95": interval.upper_95,
                "scheduled_replicates": interval.scheduled_replicates,
                "valid_replicates": interval.valid_replicates,
                "finite_fraction": interval.finite_fraction,
                "interval_available": interval.interval_available,
                "unavailable_reason": interval.unavailable_reason,
                "complete_blocks": len(interval.block_ids),
            }
        )
    for key, member_pairs in sorted(rank_block_pairs.items()):
        scale, mineral, endpoint = key
        interval = nested_spearman_bootstrap(
            member_pairs,
            replicates=bootstrap_replicates,
            seed=FROZEN_SEED,
        )
        endpoint_results[key] = interval
        endpoint_intervals.append(
            {
                "scale": scale,
                "mineral": mineral,
                "metric": endpoint,
                "lower_95": interval.lower_95,
                "upper_95": interval.upper_95,
                "scheduled_replicates": interval.scheduled_replicates,
                "valid_replicates": interval.valid_replicates,
                "finite_fraction": interval.finite_fraction,
                "interval_available": interval.interval_available,
                "unavailable_reason": interval.unavailable_reason,
                "complete_blocks": len(interval.block_ids),
            }
        )
    external_intervals: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for (scope, scale, mineral), member_results in sorted(nested_external_members.items()):
        external_intervals[(scope, scale, mineral)] = nested_external_metric_intervals(
            member_results,
            scale=scale,
            replicates=bootstrap_replicates,
            seed=FROZEN_SEED,
        )
    maps_dir = output_dir / "maps"
    written: list[Path] = []
    calibration_rows: list[dict[str, Any]] = []
    for mineral, accumulator in accumulators.items():
        frequency = accumulator.frequency()
        classes = confidence_classes(frequency)
        baseline_detection = baseline_detections[mineral].detections
        for scale in ("L", "2L"):
            scale_blocks = block_values_by_scale[scale]
            area_counts: dict[str, dict[int, tuple[float, float]]] = {
                "stable_positive_area_fraction": {},
                "stable_negative_area_fraction": {},
                "stable_core_retention": {},
            }
            for block_id in handoff.scales[scale].block_ids:
                finite = np.isfinite(frequency) & (scale_blocks == block_id)
                area_counts["stable_positive_area_fraction"][block_id] = (
                    float(np.count_nonzero(finite & (classes == CONFIDENCE_STABLE_POSITIVE))),
                    float(np.count_nonzero(finite)),
                )
                area_counts["stable_negative_area_fraction"][block_id] = (
                    float(np.count_nonzero(finite & (classes == CONFIDENCE_STABLE_NEGATIVE))),
                    float(np.count_nonzero(finite)),
                )
                baseline_block = (
                    np.asarray(baseline_detection, dtype=bool) & (scale_blocks == block_id)
                    if baseline_detection is not None
                    else np.zeros(shape, dtype=bool)
                )
                area_counts["stable_core_retention"][block_id] = (
                    float(
                        np.count_nonzero(baseline_block & (classes == CONFIDENCE_STABLE_POSITIVE))
                    ),
                    float(np.count_nonzero(baseline_block)),
                )
            for endpoint, block_counts in area_counts.items():
                interval = nested_ratio_bootstrap(
                    {"ensemble": block_counts},
                    replicates=bootstrap_replicates,
                    seed=FROZEN_SEED,
                )
                endpoint_results[(scale, mineral, endpoint)] = interval
                endpoint_intervals.append(
                    {
                        "scale": scale,
                        "mineral": mineral,
                        "metric": endpoint,
                        "lower_95": interval.lower_95,
                        "upper_95": interval.upper_95,
                        "scheduled_replicates": interval.scheduled_replicates,
                        "valid_replicates": interval.valid_replicates,
                        "finite_fraction": interval.finite_fraction,
                        "interval_available": interval.interval_available,
                        "unavailable_reason": interval.unavailable_reason,
                        "complete_blocks": len(interval.block_ids),
                    }
                )
        if reference is not None:
            binary = _binary_rockwell_reference(reference, mineral)
            if binary is not None:
                calibration_rows.extend(
                    calibration_diagnostic(
                        frequency,
                        binary,
                        block_values,
                        site=site,
                        mineral=mineral,
                        bootstrap_replicates=bootstrap_replicates,
                    )
                )
        products = {
            "n_valid": (accumulator.valid_count, "uint16", 0),
            "detection_frequency": (frequency, "float32", np.nan),
            "confidence_class": (classes, "int8", CONFIDENCE_NODATA),
        }
        for suffix, (values, dtype, nodata) in products.items():
            path = maps_dir / f"{site}_{mineral}_{suffix}.tif"
            _write_geotiff(
                path,
                values,
                crs=handoff.crs,
                transform=handoff.transform,
                dtype=dtype,
                nodata=nodata,
            )
            written.append(path)
    dominant = summarize_dominant_classes(joint_dominant, baseline_dominant)
    dominant_products = {
        "modal_class": (dominant.modal_class, "int16", DOMINANT_NODATA),
        "modal_frequency": (dominant.modal_frequency, "float32", np.nan),
        "class_entropy": (dominant.normalized_entropy, "float32", np.nan),
        "switch_frequency": (dominant.switch_frequency, "float32", np.nan),
    }
    for suffix, (values, dtype, nodata) in dominant_products.items():
        path = maps_dir / f"{site}_{suffix}.tif"
        _write_geotiff(
            path,
            values,
            crs=handoff.crs,
            transform=handoff.transform,
            dtype=dtype,
            nodata=nodata,
        )
        written.append(path)
    cell_counts: dict[str, int] = {}
    for row in ledger.rows:
        if row.get("site") != site or row.get("member_class") != "joint":
            continue
        key = _compact_json(
            {
                "ridge": row.get("ridge"),
                "quantile": row.get("detection_quantile"),
                "gate": row.get("infeasibility_gate"),
            }
        )
        cell_counts[key] = cell_counts.get(key, 0) + int(row.get("status") == "complete")
    alunite_frequency = accumulators["alunite"].frequency()
    alunite_baseline = baseline_detections["alunite"].detections
    if alunite_baseline is None or not np.any(alunite_baseline):
        stable_core_retention = None
    else:
        stable_core_retention = float(np.mean(alunite_frequency[alunite_baseline] >= 0.80))
    alunite_rank = np.asarray(
        [
            row["rank_correlation"]
            for row in metric_rows
            if row["mineral"] == "alunite"
            and row["member_class"] == "joint"
            and row["aggregation"] == "scene"
            and row["rank_correlation"] is not None
        ],
        dtype=float,
    )
    switch_bootstrap = endpoint_results.get(
        ("L", "alunite", "dominant_class_switch_frequency"),
        nested_block_bootstrap({}, replicates=bootstrap_replicates),
    )
    analytical_cells_complete = (
        all(count == FROZEN_STOCHASTIC_REPLICATES for count in cell_counts.values())
        and len(cell_counts) == FROZEN_ANALYTICAL_CELLS
    )
    operational_alunite = [
        {**record, "covariance_scope": "full_scene_covariance"}
        for scale in ("L", "2L")
        for record in external_intervals.get(("full_scene_covariance", scale, "alunite"), ())
    ]
    strict_alunite = [
        {**record, "covariance_scope": "strict_covariance_exclusion"}
        for scale in ("L", "2L")
        for record in external_intervals.get(("strict_covariance_exclusion", scale, "alunite"), ())
    ]
    median_rank = float(np.median(alunite_rank)) if alunite_rank.size else None
    rank_fifth = float(np.percentile(alunite_rank, 5)) if alunite_rank.size else None
    if site == "goldfield":
        claim_gate = evaluate_goldfield_claim_gate(
            analytical_cells_complete=analytical_cells_complete,
            stable_core_retention=stable_core_retention,
            median_rank_correlation=median_rank,
            rank_correlation_5th_percentile=rank_fifth,
            switch_interval=switch_bootstrap,
            operational_intervals=operational_alunite,
            strict_intervals=strict_alunite,
        )
    else:
        claim_gate = {
            "confirmatory_gate_available": False,
            "confirmatory_gate_pass": None,
            "permitted_claim_classification": "map_stability_only_no_external_reference",
        }
    serialized_external_intervals = [
        {"covariance_scope": scope, "mineral": mineral, **record}
        for (scope, _scale, mineral), records_for_key in sorted(external_intervals.items())
        for record in records_for_key
    ]
    summary = {
        "site": site,
        "recorded_variants": len(site_rows),
        "unique_mtmf_fits": len(fit_rows),
        "joint_valid_members": len(joint_dominant),
        "failed_members": failed_rows,
        "analytical_cell_valid_member_counts": cell_counts,
        **claim_gate,
        "external_covariance_estimand": "full_scene_covariance_operational_transductive",
        "goldfield_alunite_gate_components": {
            "stable_core_retention": stable_core_retention,
            "median_rank_correlation": median_rank,
            "rank_correlation_5th_percentile": rank_fifth,
            "dominant_class_switch_nested_bootstrap_upper_95": switch_bootstrap.upper_95,
            "external_interval_gate": claim_gate.get("external_interval_available"),
        },
        "nested_block_bootstrap": {
            "shared_draw_per_replicate": True,
            "member_summary_within_replicate": "median",
            "replicates": bootstrap_replicates,
            "finite_replicate_fraction_required": FINITE_REPLICATE_FRACTION,
            "dominant_class_switch_lower_95": switch_bootstrap.lower_95,
            "dominant_class_switch_upper_95": switch_bootstrap.upper_95,
            "endpoint_intervals": endpoint_intervals,
            "external_intervals": serialized_external_intervals,
        },
        "strict_covariance_exclusion": {
            "status": (
                "complete"
                if strict_alunite
                else "not_applicable"
                if site != "goldfield"
                else "unavailable"
            ),
            "pooled_with_operational": False,
            "fold_failures": {
                f"{fit_id}:{scale}": failures
                for (fit_id, scale), (_, failures) in strict_score_cache.items()
                if failures
            },
        },
    }
    return metric_rows, calibration_rows, summary, written


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# E6 MTMF ensemble sensitivity",
        "",
        "This report is generated from the frozen finite sensitivity design.",
        "Detection frequencies are empirical design frequencies, not probabilities.",
        "Axis contrasts are descriptive paired contrasts and do not identify variance components.",
        "",
        "## Site status",
        "",
    ]
    for site in summary["sites"]:
        lines.append(
            f"- {site['site']}: {site['joint_valid_members']} valid joint members; "
            f"confirmatory gate available = {site['confirmatory_gate_available']}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ensemble_sensitivity(
    args: Any,
    *,
    root: Path,
    deviations: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Preflight, materialize, and optionally execute the frozen E6 pipeline."""
    root = root.resolve()
    preregistration = args.preregistration
    if not preregistration.is_absolute():
        preregistration = root / preregistration
    block_manifest = args.block_manifest
    if not block_manifest.is_absolute():
        block_manifest = root / block_manifest
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    protocol = validate_protocol_file(
        preregistration,
        protocol_amendment=args.protocol_amendment,
    )
    m2_preregistration = root / "docs" / "m2_spatial_validation_preregistration.md"
    handoffs = validate_m2_manifest(
        block_manifest,
        m2_preregistration=m2_preregistration,
        anchors={site: ANCHOR_SCENES[site] for site in args.sites},
    )
    input_manifest_path = root / "docs" / "input_manifest.json"
    input_manifest = validate_input_manifest(input_manifest_path, root=root)
    inspections = {site: inspect_anchor_scene(root, site) for site in args.sites}
    populations = [inspection.candidates for inspection in inspections.values()]
    if not populations:
        raise ProtocolError("no anchor scenes selected")
    for population in populations[1:]:
        _require_equal("anchor-site library populations", population, populations[0])
    expected_grids = {
        site: {
            "shape": inspection.shape,
            "crs": inspection.crs,
            "transform": inspection.transform,
        }
        for site, inspection in inspections.items()
    }
    validate_m2_manifest(
        block_manifest,
        m2_preregistration=m2_preregistration,
        anchors={site: ANCHOR_SCENES[site] for site in args.sites},
        expected_grids=expected_grids,
    )
    rockwell_reference = (
        validate_rockwell_reference(
            root,
            handoffs["goldfield"],
            block_manifest=block_manifest,
        )
        if "goldfield" in handoffs
        else None
    )
    gates = tuple(_as_gate(value) for value in args.infeasibility_gates)
    design, members = build_design(
        candidates=populations[0],
        complete_blocks={site: handoffs[site].block_ids for site in args.sites},
        sites=args.sites,
        ridges=args.ridge,
        quantiles=args.detection_quantiles,
        gates=gates,
        stochastic_replicates=args.stochastic_replicates,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    payload = _design_payload(
        design,
        root=root,
        args=args,
        protocol=protocol,
        block_manifest=block_manifest,
        input_manifest=input_manifest,
        rockwell_reference=rockwell_reference,
        deviations=deviations or {},
    )
    payload["scientific_design_sha256"] = scientific_design_sha256(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    design_path, ledger = _materialize_design(output_dir, payload, members, resume=args.resume)
    outputs = {"design": design_path, "members": ledger.path}
    if args.design_only:
        return outputs
    if args.timing_pilot:
        outputs["timing_pilot"] = _run_timing_pilot(
            root=root,
            output_dir=output_dir,
            members=members,
            handoffs=handoffs,
        )
        return outputs

    manifest_payload = json.loads(block_manifest.read_text(encoding="utf-8"))
    all_metrics: list[dict[str, Any]] = []
    all_calibration: list[dict[str, Any]] = []
    site_summaries: list[dict[str, Any]] = []
    map_paths: list[Path] = []
    for site in args.sites:
        site_index = FROZEN_SITES.index(site)
        metrics, calibration, site_summary, written = _execute_site(
            root=root,
            output_dir=output_dir,
            site=site,
            site_index=site_index,
            members=members,
            ledger=ledger,
            handoff=handoffs[site],
            manifest_payload=manifest_payload,
            storage_layout=args.storage_layout,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        all_metrics.extend(metrics)
        all_calibration.extend(calibration)
        site_summaries.append(site_summary)
        map_paths.extend(written)
    metric_fields = [
        "site",
        "scene",
        "mineral",
        "member_id",
        "member_class",
        "stochastic_replicate",
        "ridge",
        "detection_quantile",
        "infeasibility_gate",
        "aggregation",
        "block_scale",
        "block_id",
        "common_support_pixels",
        "common_support_loss_fraction",
        "detection_prevalence",
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
        "external_status",
        "covariance_scope",
        "strict_covariance_exclusion_status",
    ]
    metrics_path = output_dir / "member_metrics.csv"
    _write_csv(metrics_path, all_metrics, metric_fields)
    outputs["member_metrics"] = metrics_path
    factor_rows = paired_factor_effect_rows(
        all_metrics,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    factor_path = output_dir / "factor_effects.csv"
    _write_csv(
        factor_path,
        factor_rows,
        (
            "site",
            "mineral",
            "block_scale",
            "factor",
            "level",
            "reference_level",
            "endpoint",
            "paired_delta_median",
            "interval_lower",
            "interval_upper",
            "scheduled_replicates",
            "valid_replicates",
            "finite_fraction",
            "interval_available",
            "unavailable_reason",
            "n_pairs",
            "complete_blocks",
            "paired_support_pixels",
            "contrast_status",
        ),
    )
    outputs["factor_effects"] = factor_path
    calibration_path = output_dir / "calibration.csv"
    _write_csv(
        calibration_path,
        all_calibration,
        (
            "site",
            "mineral",
            "confidence_bin",
            "support_blocks",
            "support_pixels",
            "compatible_positive_rate",
            "interval_lower",
            "interval_upper",
            "scheduled_replicates",
            "valid_replicates",
            "finite_fraction",
            "interval_available",
            "unavailable_reason",
            "brier_score",
            "brier_interval_lower",
            "brier_interval_upper",
            "brier_interval_available",
            "brier_valid_replicates",
            "brier_finite_fraction",
            "expected_calibration_error",
            "ece_interval_lower",
            "ece_interval_upper",
            "ece_interval_available",
            "ece_valid_replicates",
            "ece_finite_fraction",
            "status",
        ),
    )
    outputs["calibration"] = calibration_path
    artifact_paths = [
        design_path,
        ledger.path,
        metrics_path,
        factor_path,
        calibration_path,
        *map_paths,
    ]
    summary = {
        "schema_version": "1.0",
        "frequency_estimand": "finite_design_empirical_frequency",
        "sites": site_summaries,
        "counts": {
            "recorded_variants": len(members),
            "unique_mtmf_fits": sum(summary["unique_mtmf_fits"] for summary in site_summaries),
            "failed_members": sum(len(summary["failed_members"]) for summary in site_summaries),
        },
        "artifact_sha256": {
            str(path.relative_to(output_dir)): sha256_file(path) for path in artifact_paths
        },
        "permitted_claim_classification": next(
            (
                site_summary["permitted_claim_classification"]
                for site_summary in site_summaries
                if site_summary["site"] == "goldfield"
            ),
            "map_stability_only_no_external_reference",
        ),
        "axis_contrasts": "descriptive_paired_only",
        "compute_controls": {
            "device": args.device,
            "batch_size": args.batch_size,
            "storage_layout": args.storage_layout,
            "scientifically_inert": True,
        },
    }
    summary_path = output_dir / "summary.json"
    strict_json_dump(summary_path, summary)
    outputs["summary"] = summary_path
    report_path = output_dir / "report.md"
    _write_report(report_path, summary)
    outputs["report"] = report_path
    return outputs


__all__ = [
    "ANCHOR_SCENES",
    "BASELINE_ENDMEMBERS",
    "BlockManifestSite",
    "BlockScaleHandoff",
    "CONFIDENCE_CHOICE_SENSITIVE",
    "CONFIDENCE_NODATA",
    "CONFIDENCE_STABLE_NEGATIVE",
    "CONFIDENCE_STABLE_POSITIVE",
    "DOMINANT_NODATA",
    "DOMINANT_NONE",
    "DetectionResult",
    "EXPECTED_CANDIDATE_COUNTS",
    "FROZEN_BOOTSTRAP_REPLICATES",
    "FROZEN_GATES",
    "FROZEN_PREREGISTRATION_SHA256",
    "FROZEN_QUANTILES",
    "FROZEN_RETAINED_BANDS",
    "FROZEN_RIDGES",
    "FROZEN_SEED",
    "FROZEN_SITES",
    "FROZEN_STOCHASTIC_REPLICATES",
    "MapAccumulator",
    "MemberLedger",
    "NestedBootstrapResult",
    "ProtocolError",
    "balanced_endmember_schedules",
    "build_design",
    "confidence_classes",
    "classify_permitted_claim",
    "evaluate_goldfield_claim_gate",
    "external_support_tier",
    "governing_file_provenance",
    "nested_block_bootstrap",
    "nested_external_metric_intervals",
    "nested_ratio_bootstrap",
    "nested_spearman_bootstrap",
    "operational_detection",
    "paired_factor_effect_rows",
    "run_ensemble_sensitivity",
    "sha256_file",
    "scientific_design_sha256",
    "strict_json_dump",
    "strict_covariance_cross_fitted_threshold_evaluation",
    "summarize_dominant_classes",
    "timing_pilot_fit_ids",
    "validate_m2_manifest",
    "validate_protocol_amendment",
    "validate_protocol_arguments",
    "validate_protocol_file",
    "validate_rockwell_reference",
]
