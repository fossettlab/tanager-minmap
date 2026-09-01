#!/usr/bin/env python3
"""Independently verify one endpoint-sealed E4 non-result bundle.

This verifier reads only bundle control artifacts.  It never opens EMIT L2B
arrays, Tanager scenes, score fields, or scientific result products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tanager_rocks.emit_l2b_nonresult import (
    BUNDLE_SCHEMAS,
    NonResultError,
    read_regular_bytes,
    strict_json_load_bytes,
    validate_decision_record,
    validate_legacy_synthetic_resource_policy,
    validate_legacy_synthetic_resource_telemetry,
    validate_resource_admission_evidence_files,
    verify_embedded_resource_admission_provenance,
    verify_nonresult_bundle,
)

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _json(path: Path) -> dict[str, Any]:
    payload = strict_json_load_bytes(read_regular_bytes(path))
    if not isinstance(payload, dict):
        raise NonResultError("invalid_control_payload")
    return payload


def _csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        text = read_regular_bytes(path).decode("utf-8")
    except UnicodeDecodeError as error:
        raise NonResultError("invalid_control_csv") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != columns:
        raise NonResultError("invalid_control_csv_schema")
    rows = list(reader)
    if not rows or any(set(row) != set(columns) or None in row for row in rows):
        raise NonResultError("invalid_control_csv_rows")
    return rows


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _verify_code_manifest(bundle: Path) -> None:
    payload = _json(bundle / "code_manifest.json")
    files = payload.get("files")
    if payload.get("schema_version") != "e4-nonresult-code-manifest/v1":
        raise NonResultError("invalid_code_manifest_schema")
    if not isinstance(files, dict) or set(files) != {
        "emit_l2b.py",
        "emit_l2b_nonresult.py",
        "run_emit_l2b_validation.py",
    }:
        raise NonResultError("invalid_code_manifest_files")
    if not all(_valid_sha(value) for value in files.values()):
        raise NonResultError("invalid_code_manifest_hash")


def _verify_mapping(bundle: Path) -> None:
    manifest = _json(bundle / "mapping_manifest.json")
    if (
        manifest.get("operation") != "mapping_only"
        or manifest.get("endpoint_execution") != "forbidden"
    ):
        raise NonResultError("invalid_mapping_manifest")
    source = _json(bundle / "source_pair_identity.json")
    if set(source) != {"identity", "min", "minuncert"}:
        raise NonResultError("invalid_mapping_source_identity")
    identity = source["identity"]
    if not isinstance(identity, dict) or set(identity) != {
        "kind",
        "version",
        "acquisition",
        "orbit",
        "scene",
    }:
        raise NonResultError("invalid_mapping_product_identity")
    if identity["kind"] != "MIN" or not all(
        isinstance(value, str) and value for value in identity.values()
    ):
        raise NonResultError("invalid_mapping_product_identity")
    source_hashes = manifest.get("source_pair_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != {"min", "minuncert"}:
        raise NonResultError("invalid_mapping_source_hashes")
    for member in ("min", "minuncert"):
        values = source[member]
        if not isinstance(values, dict) or set(values) != {"sha256", "size_bytes", "filename"}:
            raise NonResultError("invalid_mapping_source_member")
        if (
            not _valid_sha(values["sha256"])
            or values["sha256"] != source_hashes[member]
            or isinstance(values["size_bytes"], bool)
            or not isinstance(values["size_bytes"], int)
            or values["size_bytes"] <= 0
            or not isinstance(values["filename"], str)
            or Path(values["filename"]).name != values["filename"]
        ):
            raise NonResultError("invalid_mapping_source_member")
    geometry = _json(bundle / "geometry_contract.json")
    if set(geometry) != {"shape", "transform", "crs"}:
        raise NonResultError("invalid_mapping_geometry")
    shape = geometry["shape"]
    transform = geometry["transform"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
        )
        or not isinstance(transform, list)
        or len(transform) != 6
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in transform
        )
        or not isinstance(geometry["crs"], str)
        or not geometry["crs"]
    ):
        raise NonResultError("invalid_mapping_geometry")
    glt = _json(bundle / "glt_validation.json")
    if set(glt) != {
        "min_shape",
        "min_fill_locations_agree",
        "minuncert_shape",
        "minuncert_fill_locations_agree",
    }:
        raise NonResultError("invalid_mapping_glt_validation")
    if (
        glt["min_fill_locations_agree"] is not True
        or glt["minuncert_fill_locations_agree"] is not True
        or glt["min_shape"] != shape
        or glt["minuncert_shape"] != shape
    ):
        raise NonResultError("invalid_mapping_glt_validation")
    rows = _csv(
        bundle / "source_mineral_inventory.csv",
        ("index", "name", "group", "library"),
    )
    identities: set[tuple[int, int]] = set()
    for row in rows:
        if not row["index"].isdigit() or not row["group"].isdigit():
            raise NonResultError("invalid_mapping_mineral_inventory")
        identity_key = (int(row["group"]), int(row["index"]))
        if identity_key[0] not in {1, 2} or identity_key[1] <= 0 or identity_key in identities:
            raise NonResultError("invalid_mapping_mineral_inventory")
        if not row["name"].strip() or not row["library"].strip():
            raise NonResultError("invalid_mapping_mineral_inventory")
        identities.add(identity_key)
    if {group for group, _ in identities} != {1, 2}:
        raise NonResultError("invalid_mapping_mineral_inventory")
    contract = _json(bundle / "m2_mapping_contract.json")
    if contract.get("schema_version") != "e4-m2-mapping-contract/v1" or not _valid_sha(
        contract.get("block_manifest_sha256")
    ):
        raise NonResultError("invalid_m2_mapping_contract")
    _verify_code_manifest(bundle)


def _verify_resource_pilot(bundle: Path) -> None:
    manifest = _json(bundle / "pilot_manifest.json")
    summary = _json(bundle / "resource_summary.json")
    audit = _json(bundle / "forbidden_output_audit.json")
    if (
        manifest.get("operation") != "resource_pilot"
        or manifest.get("endpoint_execution") != "forbidden"
        or manifest.get("synthetic_fixture_only") is not True
    ):
        raise NonResultError("invalid_resource_manifest")
    if summary.get("admission_status") != "not_admissible" or not summary.get(
        "synthetic_fixture_only"
    ):
        raise NonResultError("invalid_synthetic_resource_bundle")
    if audit != {
        "scientific_endpoint_called": False,
        "scientific_output_count": 0,
        "admission_status": "not_admissible",
    }:
        raise NonResultError("invalid_forbidden_output_audit")
    policy = validate_legacy_synthetic_resource_policy(summary.get("policy"))
    rows = _csv(
        bundle / "stage_telemetry.csv",
        (
            "stage",
            "wall_seconds",
            "cpu_seconds",
            "peak_rss_bytes",
            "input_bytes",
            "scratch_bytes",
            "exit_status",
        ),
    )
    if len(rows) != 1 or rows[0]["stage"] != "synthetic_fixture_guard":
        raise NonResultError("invalid_synthetic_resource_telemetry")
    typed_row: dict[str, Any] = {"stage": rows[0]["stage"]}
    for key, value in rows[0].items():
        if key == "stage":
            continue
        if not re.fullmatch(r"0|[1-9][0-9]*", value):
            raise NonResultError("invalid_synthetic_resource_telemetry")
        typed_row[key] = int(value)
    validate_legacy_synthetic_resource_telemetry([typed_row], policy)
    if any(
        typed_row[key] != 0
        for key in ("wall_seconds", "cpu_seconds", "peak_rss_bytes", "scratch_bytes", "exit_status")
    ):
        raise NonResultError("invalid_synthetic_resource_telemetry")
    _verify_code_manifest(bundle)


def _verify_preflight(bundle: Path) -> None:
    decision = validate_decision_record(_json(bundle / "decision_record.json"))
    summary = _json(bundle / "preflight_summary.json")
    manifest = _json(bundle / "preflight_manifest.json")
    if (
        summary.get("endpoint_execution") != "forbidden"
        or summary.get("scientific_run_command") != "absent"
    ):
        raise NonResultError("preflight_endpoint_boundary_failed")
    registry_bytes = read_regular_bytes(bundle / "expected_scientific_output_registry.json")
    crosswalk_bytes = read_regular_bytes(bundle / "ontology_crosswalk.csv")
    if decision["output_registry"]["sha256"] != hashlib.sha256(registry_bytes).hexdigest():
        raise NonResultError("preflight_registry_binding_failed")
    if decision["ontology"]["crosswalk_sha256"] != hashlib.sha256(crosswalk_bytes).hexdigest():
        raise NonResultError("preflight_ontology_binding_failed")
    if manifest.get("output_registry_sha256") != decision["output_registry"]["sha256"]:
        raise NonResultError("preflight_registry_binding_failed")
    if not _valid_sha(manifest.get("resource_policy_sha256")):
        raise NonResultError("invalid_resource_policy_binding")
    if summary.get("resource_policy_sha256") != manifest.get("resource_policy_sha256"):
        raise NonResultError("preflight_resource_policy_binding_failed")
    validate_resource_admission_evidence_files(
        bundle,
        expected_policy_sha256=manifest["resource_policy_sha256"],
    )
    admission_closure = manifest.get("resource_admission_closure_sha256")
    if (
        not _valid_sha(admission_closure)
        or summary.get("resource_admission_closure_sha256") != admission_closure
    ):
        raise NonResultError("preflight_resource_admission_binding_failed")
    verify_embedded_resource_admission_provenance(
        bundle,
        expected_closure_sha256=admission_closure,
    )
    if summary.get("mapping_closure_sha256") != manifest.get("mapping_closure_sha256"):
        raise NonResultError("preflight_mapping_binding_failed")
    mapping = _json(bundle / "mapping_admission.json")
    if (
        set(mapping) != {"mapping_closure_sha256", "bundle_id"}
        or mapping.get("mapping_closure_sha256") != manifest.get("mapping_closure_sha256")
        or not _valid_sha(mapping.get("mapping_closure_sha256"))
        or not isinstance(mapping.get("bundle_id"), str)
        or not mapping["bundle_id"]
    ):
        raise NonResultError("invalid_mapping_admission")
    _verify_code_manifest(bundle)


def verify(bundle: Path, expected_type: str | None = None) -> str:
    """Verify closure plus type-specific non-result semantics and return its digest."""
    receipt = verify_nonresult_bundle(bundle, expected_type=expected_type)
    if receipt.bundle_type == "mapping":
        _verify_mapping(receipt.bundle_path)
    elif receipt.bundle_type == "resource_pilot":
        _verify_resource_pilot(receipt.bundle_path)
    elif receipt.bundle_type == "resource_admission":
        policy_sha256 = hashlib.sha256(
            read_regular_bytes(receipt.bundle_path / "resource_policy.json")
        ).hexdigest()
        validate_resource_admission_evidence_files(
            receipt.bundle_path,
            expected_policy_sha256=policy_sha256,
        )
    elif receipt.bundle_type == "preflight":
        _verify_preflight(receipt.bundle_path)
    else:  # pragma: no cover - guarded by shared verifier.
        raise NonResultError("unknown_bundle_type")
    final_receipt = verify_nonresult_bundle(bundle, expected_type=receipt.bundle_type)
    if final_receipt != receipt:
        raise NonResultError("bundle_changed_during_semantic_verification")
    return final_receipt.closure_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--type", choices=sorted(BUNDLE_SCHEMAS))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        digest = verify(args.bundle, args.type)
    except (OSError, ValueError, NonResultError) as error:
        raise SystemExit(f"E4_NONRESULT_VERIFY_FAILED:{type(error).__name__}") from error
    print(f"E4_NONRESULT_VERIFY_OK:{args.type or 'auto'}:{digest}")


if __name__ == "__main__":
    main()
