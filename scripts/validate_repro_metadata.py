"""Validate the experiment registry and public-claim ledger.

The metadata documents use the JSON-compatible subset of YAML so this script
can parse them with the Python standard library. Paths are repository-relative;
present files are checked for existence and recorded SHA-256 values for form.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_RELATIVE_PATH = Path("experiments/registry.yaml")
CLAIM_LEDGER_RELATIVE_PATH = Path("docs/claim_ledger.yaml")

REGISTRY_REQUIRED_FIELDS = (
    "id",
    "title",
    "status",
    "classification",
    "hypothesis",
    "code",
    "inputs",
    "split",
    "parameters",
    "checkpoint",
    "command",
    "outputs",
    "primary_endpoint",
    "uncertainty_method",
    "gate",
    "data_quality",
)
CLAIM_REQUIRED_FIELDS = (
    "id",
    "claim_text",
    "metric",
    "source_artifact",
    "generating_command",
    "public_destinations",
    "status",
    "notes",
)
ARTIFACT_REQUIRED_FIELDS = ("id", "path", "kind", "sha256", "availability", "notes")
SPLIT_REQUIRED_FIELDS = ("unit", "manifest_path", "memberships", "notes")
PARAMETER_REQUIRED_FIELDS = ("values", "grid", "thresholds", "seed", "stopping_rule")
GATE_REQUIRED_FIELDS = ("rule", "decision", "evidence")
DATA_QUALITY_REQUIRED_FIELDS = (
    "dropped_samples",
    "nan_policy",
    "coverage_masks",
    "known_deviations",
)
SOURCE_REQUIRED_FIELDS = ("path", "selector", "sha256")
METRIC_REQUIRED_FIELDS = ("name", "source_value", "reported_values", "unit")
DESTINATION_REQUIRED_FIELDS = ("path", "locator", "state")

REGISTRY_STATUSES = frozenset({"current", "pending"})
CLAIM_STATUSES = frozenset({"current", "needs_reconfirmation", "needs_correction", "source_gap"})
ARTIFACT_AVAILABILITY = frozenset({"present", "pending", "unrecorded", "not_applicable"})
ARTIFACT_KINDS = frozenset({"file", "directory", "not_applicable"})
CLASSIFICATIONS = frozenset({"confirmatory", "exploratory", "mechanical"})
GATE_DECISIONS = frozenset({"pass", "fail", "pending", "not_applicable"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _location(label: str, index: int, item: Mapping[str, Any]) -> str:
    item_id = item.get("id", "<missing id>")
    return f"{label}[{index}] ({item_id})"


def _require_mapping(value: Any, location: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{location}: expected an object")
        return None
    return value


def _require_fields(
    value: Mapping[str, Any], required: tuple[str, ...], location: str, errors: list[str]
) -> None:
    for field in required:
        if field not in value:
            errors.append(f"{location}: missing required field '{field}'")


def _require_nonempty_string(value: Any, location: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{location}: expected a non-empty string")


def _resolve_repo_path(
    value: Any,
    root: Path,
    location: str,
    errors: list[str],
    *,
    must_exist: bool,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{location}: expected a non-empty repository-relative path")
        return None
    relative = Path(value)
    if relative.is_absolute():
        errors.append(f"{location}: absolute paths are not allowed: {value}")
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{location}: path escapes repository root: {value}")
        return None
    if must_exist and not candidate.exists():
        errors.append(f"{location}: path does not exist: {value}")
    return candidate


def _validate_digest(
    value: Any,
    location: str,
    errors: list[str],
    *,
    required: bool,
) -> None:
    if value is None:
        if required:
            errors.append(f"{location}: SHA-256 is required for a present file")
        return
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        errors.append(f"{location}: expected 64 lowercase hexadecimal characters")


def _validate_artifact(value: Any, root: Path, location: str, errors: list[str]) -> None:
    artifact = _require_mapping(value, location, errors)
    if artifact is None:
        return
    _require_fields(artifact, ARTIFACT_REQUIRED_FIELDS, location, errors)
    if any(field not in artifact for field in ARTIFACT_REQUIRED_FIELDS):
        return

    _require_nonempty_string(artifact["id"], f"{location}.id", errors)
    _require_nonempty_string(artifact["notes"], f"{location}.notes", errors)
    availability = artifact["availability"]
    kind = artifact["kind"]
    if availability not in ARTIFACT_AVAILABILITY:
        errors.append(f"{location}.availability: expected one of {sorted(ARTIFACT_AVAILABILITY)}")
    if kind not in ARTIFACT_KINDS:
        errors.append(f"{location}.kind: expected one of {sorted(ARTIFACT_KINDS)}")

    path_value = artifact["path"]
    if path_value is None:
        if availability == "present":
            errors.append(f"{location}.path: present artifacts require a path")
        candidate = None
    else:
        candidate = _resolve_repo_path(
            path_value,
            root,
            f"{location}.path",
            errors,
            must_exist=availability == "present",
        )
    if candidate is not None and availability == "present":
        if kind == "file" and candidate.exists() and not candidate.is_file():
            errors.append(f"{location}.path: expected a file: {path_value}")
        if kind == "directory" and candidate.exists() and not candidate.is_dir():
            errors.append(f"{location}.path: expected a directory: {path_value}")
    _validate_digest(
        artifact["sha256"],
        f"{location}.sha256",
        errors,
        required=availability == "present" and kind == "file",
    )


def _validate_registry_schema(schema: Any, location: str, errors: list[str]) -> None:
    mapping = _require_mapping(schema, location, errors)
    if mapping is None:
        return
    documented = mapping.get("required_experiment_fields")
    if documented != list(REGISTRY_REQUIRED_FIELDS):
        errors.append(f"{location}.required_experiment_fields: must match the validator contract")
    definitions = mapping.get("status_definitions")
    if not isinstance(definitions, Mapping) or set(definitions) != set(REGISTRY_STATUSES):
        errors.append(f"{location}.status_definitions: must document every registry status")
    _require_nonempty_string(mapping.get("description"), f"{location}.description", errors)
    _require_nonempty_string(mapping.get("document_format"), f"{location}.document_format", errors)


def _validate_claim_schema(schema: Any, location: str, errors: list[str]) -> None:
    mapping = _require_mapping(schema, location, errors)
    if mapping is None:
        return
    documented = mapping.get("required_claim_fields")
    if documented != list(CLAIM_REQUIRED_FIELDS):
        errors.append(f"{location}.required_claim_fields: must match the validator contract")
    definitions = mapping.get("status_definitions")
    if not isinstance(definitions, Mapping) or set(definitions) != set(CLAIM_STATUSES):
        errors.append(f"{location}.status_definitions: must document every claim status")
    _require_nonempty_string(mapping.get("description"), f"{location}.description", errors)
    _require_nonempty_string(mapping.get("document_format"), f"{location}.document_format", errors)


def validate_registry(data: Any, root: Path = ROOT) -> list[str]:
    """Return deterministic validation errors for registry data."""
    errors: list[str] = []
    document = _require_mapping(data, str(REGISTRY_RELATIVE_PATH), errors)
    if document is None:
        return sorted(errors)
    for field in ("schema_version", "snapshot_date", "_schema", "experiments"):
        if field not in document:
            errors.append(f"{REGISTRY_RELATIVE_PATH}: missing required field '{field}'")
    if "_schema" in document:
        _validate_registry_schema(document["_schema"], f"{REGISTRY_RELATIVE_PATH}._schema", errors)
    experiments = document.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        errors.append(f"{REGISTRY_RELATIVE_PATH}.experiments: expected a non-empty list")
        return sorted(errors)

    seen: set[str] = set()
    for index, raw in enumerate(experiments):
        experiment = _require_mapping(raw, f"{REGISTRY_RELATIVE_PATH}.experiments[{index}]", errors)
        if experiment is None:
            continue
        location = _location(f"{REGISTRY_RELATIVE_PATH}.experiments", index, experiment)
        _require_fields(experiment, REGISTRY_REQUIRED_FIELDS, location, errors)
        if any(field not in experiment for field in REGISTRY_REQUIRED_FIELDS):
            continue
        experiment_id = experiment["id"]
        _require_nonempty_string(experiment_id, f"{location}.id", errors)
        if isinstance(experiment_id, str):
            if experiment_id in seen:
                errors.append(f"{location}.id: duplicate experiment id '{experiment_id}'")
            seen.add(experiment_id)
        for field in ("title", "hypothesis", "command", "primary_endpoint"):
            _require_nonempty_string(experiment[field], f"{location}.{field}", errors)
        if experiment["status"] not in REGISTRY_STATUSES:
            errors.append(f"{location}.status: expected one of {sorted(REGISTRY_STATUSES)}")
        if experiment["classification"] not in CLASSIFICATIONS:
            errors.append(f"{location}.classification: expected one of {sorted(CLASSIFICATIONS)}")
        uncertainty = experiment["uncertainty_method"]
        if uncertainty is not None:
            _require_nonempty_string(uncertainty, f"{location}.uncertainty_method", errors)

        code = _require_mapping(experiment["code"], f"{location}.code", errors)
        if code is not None:
            _require_fields(
                code,
                ("commit", "commit_source", "dependency_lock"),
                f"{location}.code",
                errors,
            )
            if "commit" in code and code["commit"] is not None:
                _require_nonempty_string(code["commit"], f"{location}.code.commit", errors)
            if "commit_source" in code:
                _require_nonempty_string(
                    code["commit_source"], f"{location}.code.commit_source", errors
                )
            if "dependency_lock" in code:
                _validate_artifact(
                    code["dependency_lock"], root, f"{location}.code.dependency_lock", errors
                )

        for field in ("inputs", "outputs"):
            artifacts = experiment[field]
            if not isinstance(artifacts, list) or not artifacts:
                errors.append(f"{location}.{field}: expected a non-empty list")
                continue
            for artifact_index, artifact in enumerate(artifacts):
                _validate_artifact(artifact, root, f"{location}.{field}[{artifact_index}]", errors)

        split = _require_mapping(experiment["split"], f"{location}.split", errors)
        if split is not None:
            _require_fields(split, SPLIT_REQUIRED_FIELDS, f"{location}.split", errors)
            if "unit" in split:
                _require_nonempty_string(split["unit"], f"{location}.split.unit", errors)
            if "notes" in split:
                _require_nonempty_string(split["notes"], f"{location}.split.notes", errors)
            if split.get("manifest_path") is not None:
                _resolve_repo_path(
                    split["manifest_path"],
                    root,
                    f"{location}.split.manifest_path",
                    errors,
                    must_exist=True,
                )

        parameters = _require_mapping(experiment["parameters"], f"{location}.parameters", errors)
        if parameters is not None:
            _require_fields(parameters, PARAMETER_REQUIRED_FIELDS, f"{location}.parameters", errors)
        _validate_artifact(experiment["checkpoint"], root, f"{location}.checkpoint", errors)

        gate = _require_mapping(experiment["gate"], f"{location}.gate", errors)
        if gate is not None:
            _require_fields(gate, GATE_REQUIRED_FIELDS, f"{location}.gate", errors)
            if gate.get("decision") not in GATE_DECISIONS:
                errors.append(f"{location}.gate.decision: expected one of {sorted(GATE_DECISIONS)}")
            for field in ("rule", "evidence"):
                if field in gate:
                    _require_nonempty_string(gate[field], f"{location}.gate.{field}", errors)

        quality = _require_mapping(experiment["data_quality"], f"{location}.data_quality", errors)
        if quality is not None:
            _require_fields(
                quality, DATA_QUALITY_REQUIRED_FIELDS, f"{location}.data_quality", errors
            )
            for field in DATA_QUALITY_REQUIRED_FIELDS:
                if field in quality and quality[field] is None:
                    errors.append(f"{location}.data_quality.{field}: explicit value required")
    return sorted(errors)


def _validate_source_artifact(
    value: Any, status: Any, root: Path, location: str, errors: list[str]
) -> None:
    source = _require_mapping(value, location, errors)
    if source is None:
        return
    _require_fields(source, SOURCE_REQUIRED_FIELDS, location, errors)
    if any(field not in source for field in SOURCE_REQUIRED_FIELDS):
        return
    _require_nonempty_string(source["selector"], f"{location}.selector", errors)
    path_value = source["path"]
    if path_value is None:
        if status != "source_gap":
            errors.append(f"{location}.path: only source_gap claims may use null")
    else:
        _resolve_repo_path(path_value, root, f"{location}.path", errors, must_exist=True)
    _validate_digest(
        source["sha256"],
        f"{location}.sha256",
        errors,
        required=False,
    )


def validate_claim_ledger(data: Any, root: Path = ROOT) -> list[str]:
    """Return deterministic validation errors for claim-ledger data."""
    errors: list[str] = []
    document = _require_mapping(data, str(CLAIM_LEDGER_RELATIVE_PATH), errors)
    if document is None:
        return sorted(errors)
    for field in ("schema_version", "snapshot_date", "scope", "_schema", "claims"):
        if field not in document:
            errors.append(f"{CLAIM_LEDGER_RELATIVE_PATH}: missing required field '{field}'")
    if "_schema" in document:
        _validate_claim_schema(document["_schema"], f"{CLAIM_LEDGER_RELATIVE_PATH}._schema", errors)
    claims = document.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append(f"{CLAIM_LEDGER_RELATIVE_PATH}.claims: expected a non-empty list")
        return sorted(errors)

    seen: set[str] = set()
    for index, raw in enumerate(claims):
        claim = _require_mapping(raw, f"{CLAIM_LEDGER_RELATIVE_PATH}.claims[{index}]", errors)
        if claim is None:
            continue
        location = _location(f"{CLAIM_LEDGER_RELATIVE_PATH}.claims", index, claim)
        _require_fields(claim, CLAIM_REQUIRED_FIELDS, location, errors)
        if any(field not in claim for field in CLAIM_REQUIRED_FIELDS):
            continue
        claim_id = claim["id"]
        _require_nonempty_string(claim_id, f"{location}.id", errors)
        if isinstance(claim_id, str):
            if claim_id in seen:
                errors.append(f"{location}.id: duplicate claim id '{claim_id}'")
            seen.add(claim_id)
        for field in ("claim_text", "notes"):
            _require_nonempty_string(claim[field], f"{location}.{field}", errors)
        status = claim["status"]
        if status not in CLAIM_STATUSES:
            errors.append(f"{location}.status: expected one of {sorted(CLAIM_STATUSES)}")

        metric = _require_mapping(claim["metric"], f"{location}.metric", errors)
        if metric is not None:
            _require_fields(metric, METRIC_REQUIRED_FIELDS, f"{location}.metric", errors)
            for field in ("name", "unit"):
                if field in metric:
                    _require_nonempty_string(metric[field], f"{location}.metric.{field}", errors)
            reported = metric.get("reported_values")
            if not isinstance(reported, list) or not reported:
                errors.append(f"{location}.metric.reported_values: expected a non-empty list")
            if metric.get("source_value") is None:
                errors.append(f"{location}.metric.source_value: explicit value required")

        _validate_source_artifact(
            claim["source_artifact"], status, root, f"{location}.source_artifact", errors
        )
        command = claim["generating_command"]
        if command is not None:
            _require_nonempty_string(command, f"{location}.generating_command", errors)
        destinations = claim["public_destinations"]
        if not isinstance(destinations, list) or not destinations:
            errors.append(f"{location}.public_destinations: expected a non-empty list")
            continue
        for destination_index, raw_destination in enumerate(destinations):
            destination_location = f"{location}.public_destinations[{destination_index}]"
            destination = _require_mapping(raw_destination, destination_location, errors)
            if destination is None:
                continue
            _require_fields(destination, DESTINATION_REQUIRED_FIELDS, destination_location, errors)
            if all(field in destination for field in DESTINATION_REQUIRED_FIELDS):
                _resolve_repo_path(
                    destination["path"],
                    root,
                    f"{destination_location}.path",
                    errors,
                    must_exist=True,
                )
                for field in ("locator", "state"):
                    _require_nonempty_string(
                        destination[field], f"{destination_location}.{field}", errors
                    )
    return sorted(errors)


def _load_document(path: Path, label: str) -> tuple[Any | None, list[str]]:
    if not path.is_file():
        return None, [f"{label}: metadata file does not exist"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except json.JSONDecodeError as exc:
        return None, [
            f"{label}: invalid JSON-compatible YAML at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ]


def validate_repository(root: Path = ROOT) -> list[str]:
    """Validate both metadata documents below ``root``."""
    root = root.resolve()
    registry, errors = _load_document(root / REGISTRY_RELATIVE_PATH, str(REGISTRY_RELATIVE_PATH))
    ledger, ledger_errors = _load_document(
        root / CLAIM_LEDGER_RELATIVE_PATH, str(CLAIM_LEDGER_RELATIVE_PATH)
    )
    errors.extend(ledger_errors)
    if registry is not None:
        errors.extend(validate_registry(registry, root))
    if ledger is not None:
        errors.extend(validate_claim_ledger(ledger, root))
    return sorted(errors)


def main(argv: list[str] | None = None) -> int:
    """Run metadata validation and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    args = parser.parse_args(argv)
    errors = validate_repository(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"reproducibility metadata invalid: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("reproducibility metadata valid: registry and claim ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
