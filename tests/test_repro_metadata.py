"""Tests for the reproducibility metadata contract."""

from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "experiments" / "registry.yaml"
CLAIM_LEDGER_PATH = ROOT / "docs" / "claim_ledger.yaml"
VALIDATOR = run_path(str(ROOT / "scripts" / "validate_repro_metadata.py"))
validate_claim_ledger = VALIDATOR["validate_claim_ledger"]
validate_registry = VALIDATOR["validate_registry"]
validate_repository = VALIDATOR["validate_repository"]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_repository_reproducibility_metadata_is_valid() -> None:
    assert validate_repository(ROOT) == []


def test_missing_required_claim_field_has_clear_error() -> None:
    ledger = _load(CLAIM_LEDGER_PATH)
    del ledger["claims"][0]["status"]

    errors = validate_claim_ledger(ledger, ROOT)

    assert errors == sorted(errors)
    assert any("missing required field 'status'" in error for error in errors)


def test_missing_claim_source_path_has_clear_error() -> None:
    ledger = _load(CLAIM_LEDGER_PATH)
    source = ledger["claims"][0]["source_artifact"]
    source["path"] = "data/processed/does-not-exist.csv"
    source["sha256"] = None

    errors = validate_claim_ledger(ledger, ROOT)

    assert any(
        "source_artifact.path: path does not exist: data/processed/does-not-exist.csv" in error
        for error in errors
    )


def test_missing_nested_registry_field_has_clear_error() -> None:
    registry = _load(REGISTRY_PATH)
    del registry["experiments"][0]["parameters"]["seed"]

    errors = validate_registry(registry, ROOT)

    assert any("parameters: missing required field 'seed'" in error for error in errors)


def test_invalid_sha256_format_is_reported() -> None:
    ledger = _load(CLAIM_LEDGER_PATH)
    ledger["claims"][0]["source_artifact"]["sha256"] = "not-a-digest"

    errors = validate_claim_ledger(ledger, ROOT)

    assert any(
        "source_artifact.sha256: expected 64 lowercase hexadecimal characters" in error
        for error in errors
    )
