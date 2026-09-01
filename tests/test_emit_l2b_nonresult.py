"""Synthetic-only tests for endpoint-sealed E4 non-result primitives."""

from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

import tanager_rocks.emit_l2b as emit_l2b
from tanager_rocks.emit_l2b import load_emit_l2b_metadata
from tanager_rocks.emit_l2b_nonresult import (
    REQUIRED_BUNDLE_FILES,
    RESOURCE_ADMISSION_EVIDENCE_FILES_V2,
    RESOURCE_ATTESTATION_FILENAMES_V2,
    RESOURCE_MEASUREMENT_CONTRACT_V2,
    RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2,
    RESOURCE_STAGES_V2,
    RESOURCE_TELEMETRY_COLUMNS_V2,
    NonResultError,
    atomic_write_bundle,
    canonical_json_bytes,
    capture_resource_policy_binding_evidence,
    csv_bytes,
    strict_json_load_bytes,
    validate_decision_record,
    validate_legacy_synthetic_resource_policy,
    validate_legacy_synthetic_resource_telemetry,
    validate_resource_admission_evidence_files,
    validate_resource_admission_receipt,
    validate_resource_policy,
    validate_resource_telemetry,
    verify_embedded_resource_admission_provenance,
    verify_nonresult_bundle,
    verify_resource_admission_bundle,
    verify_resource_policy_bindings,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_product(directory: Path, kind: str) -> Path:
    path = directory / f"EMIT_L2B_{kind}_001_20230804T191650_2321613_007.nc"
    with h5py.File(path, "w") as handle:
        handle.attrs["time_coverage_start"] = "2023-08-04T19:16:50Z"
        handle.attrs["time_coverage_end"] = "2023-08-04T19:16:50Z"
        handle.attrs["flight_line"] = "emit20230804t191650_o21613_s000"
        handle.attrs["product_version"] = "V001"
        handle.attrs["geotransform"] = np.asarray([100.0, 1.0, 0.0, 200.0, 0.0, -1.0])
        handle.attrs["spatial_ref"] = "EPSG:4326"
        location = handle.create_group("location")
        location.create_dataset("glt_x", data=np.ones((2, 2), dtype=np.int32))
        location.create_dataset("glt_y", data=np.ones((2, 2), dtype=np.int32))
        if kind == "MIN":
            for group in (1, 2):
                handle.create_dataset(f"group_{group}_mineral_id", data=np.ones((1, 1)))
                handle.create_dataset(f"group_{group}_band_depth", data=np.ones((1, 1)))
            metadata = handle.create_group("mineral_metadata")
            metadata.create_dataset("index", data=np.asarray([1, 2], dtype=np.int16))
            metadata.create_dataset("name", data=np.asarray([b"alpha", b"beta"]))
            metadata.create_dataset("group", data=np.asarray([1, 2], dtype=np.int16))
            metadata.create_dataset("library", data=np.asarray([b"splib", b"splib"]))
        else:
            for group in (1, 2):
                handle.create_dataset(f"group_{group}_band_depth_unc", data=np.ones((1, 1)))
                handle.create_dataset(f"group_{group}_fit", data=np.ones((1, 1)))
    return path


def _mapping_payloads() -> dict[str, bytes]:
    return {
        "source_pair_identity.json": canonical_json_bytes(
            {"identity": {}, "min": {}, "minuncert": {}}
        ),
        "source_mineral_inventory.csv": csv_bytes(
            ("index", "name", "group", "library"),
            [{"index": 1, "name": "alpha", "group": 1, "library": "splib"}],
        ),
        "geometry_contract.json": canonical_json_bytes(
            {"shape": [1, 1], "transform": [], "crs": "EPSG:4326"}
        ),
        "glt_validation.json": canonical_json_bytes(
            {"min_fill_locations_agree": True, "minuncert_fill_locations_agree": True}
        ),
        "m2_mapping_contract.json": canonical_json_bytes(
            {"schema_version": "e4-m2-mapping-contract/v1"}
        ),
        "code_manifest.json": canonical_json_bytes({"schema_version": "test"}),
    }


def _decision(registry_sha: str, **changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "e4-decision-record/v1",
        "ontology": {
            "crosswalk_sha256": "b" * 64,
            "entries": [
                {
                    "group": 1,
                    "index": 1,
                    "name": "Alunite",
                    "library": "splib",
                    "mapping": "exact",
                    "target": "alunite",
                }
            ],
        },
        "support": {"primary_geometry": "L", "sensitivity_geometry": "2L"},
        "covariance_estimand": "operational_transductive_primary",
        "negative_control": {"kind": "none", "specificity_claims": False},
        "claim_class": "operational_association_only",
        "output_registry": {"sha256": registry_sha},
    }
    payload.update(changes)
    return payload


def _policy() -> dict[str, object]:
    return {
        "schema_version": "e4-resource-policy/v1",
        "allocation_memory_bytes": 100,
        "memory_reserve_bytes": 10,
        "allocation_wall_seconds": 100,
        "wall_reserve_seconds": 10,
        "scratch_capacity_bytes": 100,
        "scratch_reserve_bytes": 10,
    }


def _policy_v2() -> dict[str, object]:
    return {
        "schema_version": "e4-resource-policy/v2",
        "policy_class": "real_nonresult_resource_pilot",
        "account": "synthetic-account",
        "qos": "synthetic-qos",
        "partition": "synthetic-partition",
        "allocation_cpus": 4,
        "numerical_threads": 2,
        "allocation_memory_bytes": 1_000,
        "memory_reserve_bytes": 100,
        "allocation_wall_seconds": 100,
        "wall_reserve_seconds": 10,
        "scratch_budget_bytes": 1_000,
        "scratch_reserve_bytes": 100,
        "allowed_stages": list(RESOURCE_STAGES_V2),
        "stage_bindings": [
            {
                "stage": stage,
                "input_manifest_sha256": f"{index + 1:064x}",
                "expected_input_bytes": 12 + index,
                "stage_contract_sha256": f"{index + 17:064x}",
                "wall_limit_seconds": 10,
                "cpu_limit_seconds": 20,
                "peak_rss_limit_bytes": 500,
                "scratch_limit_bytes": 500,
            }
            for index, stage in enumerate(RESOURCE_STAGES_V2)
        ],
        "bootstrap_count": 12,
        "permutation_count": 12,
        "seed": 7,
        "scheduler_snapshot_sha256": "a" * 64,
        "source_capsule_sha256": "b" * 64,
        "synthetic_workload_registry_sha256": "c" * 64,
        "measurement_contract": dict(RESOURCE_MEASUREMENT_CONTRACT_V2),
    }


def _telemetry_v2() -> list[dict[str, object]]:
    attestations = _attestation_bytes_v2()
    return [
        {
            "stage": stage,
            "wall_seconds": 5,
            "cpu_seconds": 10,
            "peak_rss_bytes": 100,
            "input_bytes": 12 + index,
            "scratch_bytes": 100,
            "exit_status": 0,
            "input_manifest_sha256": f"{index + 1:064x}",
            "stage_contract_sha256": f"{index + 17:064x}",
            "measurement_attestation_sha256": hashlib.sha256(attestations[index]).hexdigest(),
        }
        for index, stage in enumerate(RESOURCE_STAGES_V2)
    ]


def _attestation_payload_v2(index: int, stage: str) -> dict[str, object]:
    measurement_digest = hashlib.sha256(
        canonical_json_bytes(RESOURCE_MEASUREMENT_CONTRACT_V2)
    ).hexdigest()
    start = 10_000_000_000 + index * 20_000_000_000
    end = start + 5_000_000_000
    return {
        "schema_version": "e4-resource-measurement-attestation/v2",
        "stage": stage,
        "stage_contract_sha256": f"{index + 17:064x}",
        "input_manifest_sha256": f"{index + 1:064x}",
        "measurement_contract_sha256": measurement_digest,
        "wall_start_monotonic_ns": start,
        "wall_end_monotonic_ns": end,
        "wall_seconds": 5,
        "cpu_self_user_before_ns": 0,
        "cpu_self_user_after_ns": 4_000_000_000,
        "cpu_self_system_before_ns": 0,
        "cpu_self_system_after_ns": 1_000_000_000,
        "cpu_children_user_before_ns": 0,
        "cpu_children_user_after_ns": 4_000_000_000,
        "cpu_children_system_before_ns": 0,
        "cpu_children_system_after_ns": 1_000_000_000,
        "cpu_seconds": 10,
        "cgroup_identity_sha256": f"{index + 33:064x}",
        "cgroup_v2_memory_peak_available": True,
        "peak_rss_bytes": 100,
        "scratch_root_identity_sha256": f"{index + 49:064x}",
        "scratch_sampler_cadence_ns": 1_000_000_000,
        "scratch_first_sample_monotonic_ns": start,
        "scratch_last_sample_monotonic_ns": end,
        "scratch_max_sample_gap_ns": 1_000_000_000,
        "scratch_missed_samples": 0,
        "scratch_final_scan_completed": True,
        "scratch_sampler_failed": False,
        "scratch_escape_detected": False,
        "scratch_peak_allocated_bytes": 100,
        "exit_status": 0,
    }


def _attestation_bytes_v2() -> list[bytes]:
    return [
        canonical_json_bytes(_attestation_payload_v2(index, stage))
        for index, stage in enumerate(RESOURCE_STAGES_V2)
    ]


def _validate_v2(
    rows: list[dict[str, object]] | None = None,
    policy: dict[str, object] | None = None,
    attestations: list[bytes] | None = None,
) -> None:
    validate_resource_telemetry(
        _telemetry_v2() if rows is None else rows,
        validate_resource_policy(_policy_v2() if policy is None else policy),
        measurement_attestation_bytes=(
            _attestation_bytes_v2() if attestations is None else attestations
        ),
    )


def _resource_admission_v2(
    policy: dict[str, object], *, policy_sha256: str = "f" * 64
) -> dict[str, object]:
    return {
        "schema_version": "e4-resource-admission/v2",
        "admission_status": "PASS",
        "policy_class": "real_nonresult_resource_pilot",
        "resource_policy_sha256": policy_sha256,
        "scheduler_snapshot_sha256": policy["scheduler_snapshot_sha256"],
        "source_capsule_sha256": policy["source_capsule_sha256"],
        "synthetic_workload_registry_sha256": policy["synthetic_workload_registry_sha256"],
        "telemetry_sha256": "d" * 64,
        "measurement_attestation_closure_sha256": "e" * 64,
        "binding_evidence_closure_sha256": "a" * 64,
        "synthetic_fixture_only": False,
    }


def _attestation_closure_sha256(attestations: list[bytes]) -> str:
    rows = b"".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode()
        for name, data in zip(RESOURCE_ATTESTATION_FILENAMES_V2, attestations, strict=True)
    )
    return hashlib.sha256(rows).hexdigest()


def _write_resource_admission_bundle(
    tmp_path: Path,
    *,
    mutate_admission: dict[str, object] | None = None,
    mutate_binding: str | None = None,
) -> tuple[Path, dict[str, object]]:
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)
    binding_payloads = capture_resource_policy_binding_evidence(
        validate_resource_policy(policy),
        verified_at_utc="2026-08-11T00:00:01Z",
        **artifacts,
    )
    if mutate_binding == "source_member":
        name = "policy_bindings/source_member_attestation.json"
        payload = strict_json_load_bytes(binding_payloads[name])
        payload["members"][0]["sha256"] = "0" * 64
        binding_payloads[name] = canonical_json_bytes(payload)
    elif mutate_binding == "workload_registry":
        name = "policy_bindings/synthetic_workload_registry.json"
        payload = strict_json_load_bytes(binding_payloads[name])
        payload["workloads"][0]["block_count"] += 1
        binding_payloads[name] = canonical_json_bytes(payload)
    elif mutate_binding == "runner":
        name = "policy_bindings/00_source_snapshot_and_verify_runner.bin"
        binding_payloads[name] += b"# drift\n"
    elif mutate_binding == "arguments":
        name = "policy_bindings/00_source_snapshot_and_verify_arguments.json"
        payload = strict_json_load_bytes(binding_payloads[name])
        payload["arguments"]["fixture"] = "drifted"
        binding_payloads[name] = canonical_json_bytes(payload)
    elif mutate_binding is not None:
        raise AssertionError(f"unknown binding mutation: {mutate_binding}")
    policy_bytes = canonical_json_bytes(policy)
    attestations: list[bytes] = []
    telemetry: list[dict[str, object]] = []
    for index, stage in enumerate(RESOURCE_STAGES_V2):
        binding = policy["stage_bindings"][index]
        attestation_payload = _attestation_payload_v2(index, stage)
        attestation_payload["stage_contract_sha256"] = binding["stage_contract_sha256"]
        attestation_payload["input_manifest_sha256"] = binding["input_manifest_sha256"]
        attestation = canonical_json_bytes(attestation_payload)
        attestations.append(attestation)
        telemetry.append(
            {
                "stage": stage,
                "wall_seconds": 5,
                "cpu_seconds": 10,
                "peak_rss_bytes": 100,
                "input_bytes": binding["expected_input_bytes"],
                "scratch_bytes": 100,
                "exit_status": 0,
                "input_manifest_sha256": binding["input_manifest_sha256"],
                "stage_contract_sha256": binding["stage_contract_sha256"],
                "measurement_attestation_sha256": hashlib.sha256(attestation).hexdigest(),
            }
        )
    telemetry_bytes = canonical_json_bytes(telemetry)
    admission = _resource_admission_v2(
        policy,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
    )
    admission["telemetry_sha256"] = hashlib.sha256(telemetry_bytes).hexdigest()
    admission["measurement_attestation_closure_sha256"] = _attestation_closure_sha256(attestations)
    binding_rows = b"".join(
        f"{hashlib.sha256(binding_payloads[name]).hexdigest()}  {name}\n".encode()
        for name in sorted(RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2)
    )
    admission["binding_evidence_closure_sha256"] = hashlib.sha256(binding_rows).hexdigest()
    if mutate_admission:
        admission.update(mutate_admission)
    payloads = {
        "resource_policy.json": policy_bytes,
        "resource_admission.json": canonical_json_bytes(admission),
        "stage_telemetry.json": telemetry_bytes,
        "forbidden_output_audit.json": canonical_json_bytes(
            {
                "schema_version": "e4-resource-forbidden-output-audit/v2",
                "scientific_endpoint_called": False,
                "scientific_output_count": 0,
                "audit_completed": True,
            }
        ),
        **{
            name: data
            for name, data in zip(
                RESOURCE_ATTESTATION_FILENAMES_V2,
                attestations,
                strict=True,
            )
        },
        **binding_payloads,
    }
    assert set(payloads) == RESOURCE_ADMISSION_EVIDENCE_FILES_V2
    receipt = atomic_write_bundle(
        tmp_path,
        bundle_type="resource_admission",
        bundle_id="resource-admission-synthetic",
        manifest={
            "operation": "resource_admission",
            "endpoint_execution": "forbidden",
        },
        payloads=payloads,
    )
    return receipt.bundle_path, policy


def _write_policy_binding_artifacts(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    policy = _policy_v2()
    scheduler = tmp_path / "scheduler.json"
    scheduler.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "e4-scheduler-snapshot/v2",
                "captured_at_utc": "2026-08-11T00:00:00Z",
                "account": policy["account"],
                "qos": policy["qos"],
                "partition": policy["partition"],
                "allocation_cpus": policy["allocation_cpus"],
                "allocation_memory_bytes": policy["allocation_memory_bytes"],
                "allocation_wall_seconds": policy["allocation_wall_seconds"],
            }
        )
    )
    source_root = tmp_path / "source-root"
    source_root.mkdir()
    source_member = source_root / "source.py"
    source_member.write_text("# frozen synthetic source\n", encoding="utf-8")
    source = tmp_path / "source-capsule.json"
    source.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "e4-resource-source-capsule/v2",
                "members": [
                    {
                        "path": "source.py",
                        "sha256": _sha(source_member),
                        "size_bytes": source_member.stat().st_size,
                    }
                ],
            }
        )
    )
    stage_contracts: dict[str, Path] = {}
    stage_runners: dict[str, Path] = {}
    stage_arguments: dict[str, Path] = {}
    input_manifests: dict[str, Path] = {}
    input_roots: dict[str, Path] = {}
    measurement_digest = hashlib.sha256(
        canonical_json_bytes(RESOURCE_MEASUREMENT_CONTRACT_V2)
    ).hexdigest()
    bindings = policy["stage_bindings"]
    assert isinstance(bindings, list)
    for index, stage in enumerate(RESOURCE_STAGES_V2):
        runner = tmp_path / f"stage-runner-{index}.py"
        runner.write_text(f"# synthetic runner for {stage}\n", encoding="utf-8")
        arguments = tmp_path / f"stage-arguments-{index}.json"
        arguments.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "e4-resource-stage-arguments/v2",
                    "stage": stage,
                    "arguments": {"fixture": "closed", "stage_index": index},
                }
            )
        )
        input_root = tmp_path / f"inputs-{index}"
        input_root.mkdir()
        input_file = input_root / "input.bin"
        input_file.write_bytes(f"input-{stage}".encode())
        manifest = tmp_path / f"input-manifest-{index}.json"
        manifest.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "e4-resource-stage-input-manifest/v2",
                    "stage": stage,
                    "files": [
                        {
                            "path": "input.bin",
                            "sha256": _sha(input_file),
                            "size_bytes": input_file.stat().st_size,
                        }
                    ],
                }
            )
        )
        contract = tmp_path / f"stage-contract-{index}.json"
        contract.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "e4-resource-stage-contract/v2",
                    "stage": stage,
                    "runner_sha256": _sha(runner),
                    "arguments_sha256": _sha(arguments),
                    "measurement_contract_sha256": measurement_digest,
                }
            )
        )
        binding = bindings[index]
        binding["input_manifest_sha256"] = _sha(manifest)
        binding["expected_input_bytes"] = input_file.stat().st_size
        binding["stage_contract_sha256"] = _sha(contract)
        stage_contracts[stage] = contract
        stage_runners[stage] = runner
        stage_arguments[stage] = arguments
        input_manifests[stage] = manifest
        input_roots[stage] = input_root

    workload = tmp_path / "workload.json"
    workload.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "e4-resource-synthetic-workload-registry/v2",
                "workloads": [
                    {
                        "stage": stage,
                        "workload_id": f"{stage}-worst-shape",
                        "arrays": [
                            {"name": "block_ids", "shape": [8, 8], "dtype": "int64"},
                            {"name": "score", "shape": [8, 8], "dtype": "float64"},
                        ],
                        "block_count": 16,
                        "covariance_branches": ["operational", "strict_inductive"],
                        "scheduled_iterations": (
                            policy["bootstrap_count"]
                            if stage == "synthetic_bootstrap_kernel"
                            else policy["permutation_count"]
                        ),
                        "seed": policy["seed"],
                        "numerical_threads": policy["numerical_threads"],
                        "process_count": 1,
                        "generator_sha256": _sha(stage_runners[stage]),
                        "arguments_sha256": _sha(stage_arguments[stage]),
                    }
                    for stage in (
                        "synthetic_bootstrap_kernel",
                        "synthetic_spatial_null_kernel",
                    )
                ],
            }
        )
    )
    policy["scheduler_snapshot_sha256"] = _sha(scheduler)
    policy["source_capsule_sha256"] = _sha(source)
    policy["synthetic_workload_registry_sha256"] = _sha(workload)
    return policy, {
        "scheduler_snapshot": scheduler,
        "source_capsule": source,
        "source_root": source_root,
        "synthetic_workload_registry": workload,
        "stage_contracts": stage_contracts,
        "stage_runners": stage_runners,
        "stage_arguments": stage_arguments,
        "input_manifests": input_manifests,
        "input_roots": input_roots,
    }


def test_metadata_loader_never_calls_result_field_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    minimum = _write_product(tmp_path, "MIN")
    uncertainty = _write_product(tmp_path, "MINUNCERT")

    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("result field reader must not be used")

    monkeypatch.setattr(emit_l2b, "_read_field", forbidden)
    packet = load_emit_l2b_metadata(minimum, uncertainty)

    assert [record.name for record in packet.mineral_metadata] == ["alpha", "beta"]
    assert packet.min_glt_x.shape == (2, 2)


def test_atomic_mapping_bundle_exact_closure_and_verification(tmp_path: Path):
    receipt = atomic_write_bundle(
        tmp_path,
        bundle_type="mapping",
        bundle_id="mapping-test",
        manifest={"operation": "test"},
        payloads=_mapping_payloads(),
    )

    assert receipt.bundle_path.is_dir()
    assert (
        set(path.name for path in receipt.bundle_path.iterdir()) == REQUIRED_BUNDLE_FILES["mapping"]
    )
    assert verify_nonresult_bundle(receipt.bundle_path).closure_sha256 == receipt.closure_sha256


@pytest.mark.parametrize("fault_after", range(1, 8))
def test_atomic_bundle_fault_injection_leaves_no_bundle(tmp_path: Path, fault_after: int):
    with pytest.raises(RuntimeError, match="injected_staging_failure"):
        atomic_write_bundle(
            tmp_path,
            bundle_type="mapping",
            bundle_id=f"mapping-fault-{fault_after}",
            manifest={},
            payloads=_mapping_payloads(),
            fault_after=fault_after,
        )
    assert not (tmp_path / f"mapping-fault-{fault_after}").exists()


def test_atomic_bundle_rejects_existing_target(tmp_path: Path):
    atomic_write_bundle(
        tmp_path,
        bundle_type="mapping",
        bundle_id="mapping-existing",
        manifest={},
        payloads=_mapping_payloads(),
    )
    with pytest.raises(NonResultError, match="bundle_target_exists"):
        atomic_write_bundle(
            tmp_path,
            bundle_type="mapping",
            bundle_id="mapping-existing",
            manifest={},
            payloads=_mapping_payloads(),
        )


@pytest.mark.parametrize("attack", ["extra", "hash", "symlink", "hardlink"])
def test_verifier_rejects_bundle_file_attacks(tmp_path: Path, attack: str):
    receipt = atomic_write_bundle(
        tmp_path,
        bundle_type="mapping",
        bundle_id=f"mapping-{attack}",
        manifest={},
        payloads=_mapping_payloads(),
    )
    root = receipt.bundle_path
    if attack == "extra":
        (root / "attacker-controlled-name.txt").write_text("x", encoding="utf-8")
    elif attack == "hash":
        (root / "geometry_contract.json").write_text("{}\n", encoding="utf-8")
    elif attack == "symlink":
        target = root / "geometry_contract.json"
        replacement = root / "replacement.json"
        target.rename(replacement)
        target.symlink_to(replacement.name)
    else:
        target = root / "geometry_contract.json"
        duplicate = root / "duplicate.json"
        os.link(target, duplicate)
    with pytest.raises(NonResultError):
        verify_nonresult_bundle(root)


def test_decision_record_rejects_unresolved_and_incompatible_choices():
    registry = "a" * 64
    assert (
        validate_decision_record(_decision(registry))["claim_class"]
        == "operational_association_only"
    )
    with pytest.raises(NonResultError, match="unresolved_estimand"):
        validate_decision_record(_decision(registry, covariance_estimand="TBD"))
    with pytest.raises(NonResultError, match="estimand_claim_incompatibility"):
        validate_decision_record(_decision(registry, claim_class="held_block_association_only"))
    with pytest.raises(NonResultError, match="negative_control_claim_incompatibility"):
        validate_decision_record(
            _decision(registry, negative_control={"kind": "none", "specificity_claims": True})
        )


def test_decision_record_rejects_ontology_without_authority_or_exact_name():
    registry = "a" * 64
    broader = _decision(registry)
    broader["ontology"] = {
        "crosswalk_sha256": "b" * 64,
        "entries": [
            {
                "group": 1,
                "index": 1,
                "name": "alunite",
                "library": "splib",
                "mapping": "broader",
                "target": "alteration",
            }
        ],
    }
    with pytest.raises(NonResultError, match="invalid_broader_ontology"):
        validate_decision_record(broader)
    mismatch = _decision(registry)
    mismatch["ontology"] = {
        "crosswalk_sha256": "b" * 64,
        "entries": [
            {
                "group": 1,
                "index": 1,
                "name": "alunite",
                "library": "splib",
                "mapping": "exact",
                "target": "jarosite",
            }
        ],
    }
    with pytest.raises(NonResultError, match="exact_ontology_name_mismatch"):
        validate_decision_record(mismatch)


def test_resource_policy_and_telemetry_fail_at_capacity_boundary():
    policy = validate_legacy_synthetic_resource_policy(_policy())
    valid = {
        "stage": "synthetic",
        "wall_seconds": 90,
        "cpu_seconds": 1,
        "peak_rss_bytes": 90,
        "input_bytes": 1,
        "scratch_bytes": 90,
        "exit_status": 0,
    }
    validate_legacy_synthetic_resource_telemetry([valid], policy)
    invalid = dict(valid, peak_rss_bytes=91)
    with pytest.raises(NonResultError, match="resource_memory_limit_exceeded"):
        validate_legacy_synthetic_resource_telemetry([invalid], policy)


def test_legacy_resource_policy_cannot_enter_real_pilot_validator():
    with pytest.raises(NonResultError, match="invalid_resource_policy_schema"):
        validate_resource_policy(_policy())


def test_resource_admission_v2_binds_policy_and_governing_hashes():
    policy = _policy_v2()
    admission = _resource_admission_v2(policy)
    assert (
        validate_resource_admission_receipt(
            admission,
            policy=validate_resource_policy(policy),
            resource_policy_sha256="f" * 64,
        )["admission_status"]
        == "PASS"
    )

    admission["source_capsule_sha256"] = "0" * 64
    with pytest.raises(NonResultError, match="resource_admission_governing_hash_mismatch"):
        validate_resource_admission_receipt(
            admission,
            policy=validate_resource_policy(policy),
            resource_policy_sha256="f" * 64,
        )


def test_legacy_or_synthetic_resource_admission_cannot_enter_preflight():
    policy = validate_resource_policy(_policy_v2())
    admission = _resource_admission_v2(_policy_v2())
    admission["schema_version"] = "e4-resource-admission/v1"
    with pytest.raises(NonResultError, match="invalid_resource_admission"):
        validate_resource_admission_receipt(
            admission,
            policy=policy,
            resource_policy_sha256="f" * 64,
        )


def test_real_resource_admission_bundle_recomputes_raw_evidence(tmp_path: Path):
    bundle, policy = _write_resource_admission_bundle(tmp_path)
    expected_policy_sha256 = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()

    evidence = verify_resource_admission_bundle(
        bundle,
        expected_policy_sha256=expected_policy_sha256,
    )
    direct = validate_resource_admission_evidence_files(
        bundle,
        expected_policy_sha256=expected_policy_sha256,
    )

    assert evidence.bundle_receipt is not None
    assert evidence.admission["admission_status"] == "PASS"
    assert direct.admission == evidence.admission


@pytest.mark.parametrize(
    "field",
    ["telemetry_sha256", "measurement_attestation_closure_sha256"],
)
def test_resource_admission_bundle_rejects_unproven_receipt_hashes(
    tmp_path: Path,
    field: str,
):
    bundle, policy = _write_resource_admission_bundle(
        tmp_path,
        mutate_admission={field: "f" * 64},
    )
    with pytest.raises(
        NonResultError,
        match="resource_telemetry_hash_mismatch|measurement_attestation_closure_mismatch",
    ):
        verify_resource_admission_bundle(
            bundle,
            expected_policy_sha256=hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
        )

    admission = _resource_admission_v2(_policy_v2())
    admission["synthetic_fixture_only"] = True
    with pytest.raises(NonResultError, match="invalid_resource_admission"):
        validate_resource_admission_receipt(
            admission,
            policy=policy,
            resource_policy_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("source_member", "resource_member_attestation_manifest_mismatch"),
        ("workload_registry", "synthetic_workload_registry_hash_mismatch"),
        ("runner", "stage_runner_hash_mismatch"),
        ("arguments", "stage_arguments_hash_mismatch"),
    ],
)
def test_resource_admission_bundle_rejects_policy_binding_drift(
    tmp_path: Path,
    mutation: str,
    error: str,
):
    bundle, policy = _write_resource_admission_bundle(
        tmp_path,
        mutate_binding=mutation,
    )

    with pytest.raises(NonResultError, match=error):
        verify_resource_admission_bundle(
            bundle,
            expected_policy_sha256=hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
        )


def test_embedded_resource_admission_provenance_rejects_manifest_drift(tmp_path: Path):
    bundle, _ = _write_resource_admission_bundle(tmp_path)
    original_manifest = bundle / "resource_admission_manifest.json"
    original_checksums = bundle / "output_checksums.sha256"
    embedded_manifest = bundle / "resource_admission_bundle_manifest.json"
    embedded_checksums = bundle / "resource_admission_bundle_checksums.sha256"
    embedded_manifest.write_bytes(original_manifest.read_bytes())
    embedded_checksums.write_bytes(original_checksums.read_bytes())
    expected_closure = hashlib.sha256(original_checksums.read_bytes()).hexdigest()

    verify_embedded_resource_admission_provenance(
        bundle,
        expected_closure_sha256=expected_closure,
    )
    payload = strict_json_load_bytes(embedded_manifest.read_bytes())
    payload["operation"] = "drifted"
    embedded_manifest.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(NonResultError, match="embedded_resource_admission_checksum_mismatch"):
        verify_embedded_resource_admission_provenance(
            bundle,
            expected_closure_sha256=expected_closure,
        )


def test_resource_policy_v2_and_telemetry_accept_closed_synthetic_contract():
    policy = validate_resource_policy(_policy_v2())

    validate_resource_telemetry(
        _telemetry_v2(),
        policy,
        measurement_attestation_bytes=_attestation_bytes_v2(),
    )

    assert RESOURCE_TELEMETRY_COLUMNS_V2 == (
        "stage",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "input_bytes",
        "scratch_bytes",
        "exit_status",
        "input_manifest_sha256",
        "stage_contract_sha256",
        "measurement_attestation_sha256",
    )
    assert RESOURCE_MEASUREMENT_CONTRACT_V2 == {
        "wall_seconds_mechanism": "monotonic_ns_elapsed",
        "cpu_seconds_mechanism": "getrusage_self_plus_children",
        "peak_rss_bytes_mechanism": "linux_cgroup_v2_memory.peak",
        "input_bytes_mechanism": "manifest_bound_regular_file_size_bytes",
        "scratch_bytes_mechanism": (
            "recursive_st_blocks_512_allocated_bytes_whole_run_directory_including_hidden_files"
        ),
        "scratch_sampler_cadence_ns": 1_000_000_000,
        "scratch_sampler_availability": "required_fail_closed",
        "scheduler_cross_check": "slurm_independent_cross_check_only",
    }


def test_resource_policy_v2_no_follow_binding_verifier_accepts_exact_artifacts(
    tmp_path: Path,
):
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)

    verify_resource_policy_bindings(validate_resource_policy(policy), **artifacts)


def test_resource_policy_v2_binding_verifier_rejects_semantic_scheduler_drift(
    tmp_path: Path,
):
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)
    scheduler = artifacts["scheduler_snapshot"]
    assert isinstance(scheduler, Path)
    payload = strict_json_load_bytes(scheduler.read_bytes())
    payload["allocation_cpus"] = 3
    scheduler.write_bytes(canonical_json_bytes(payload))
    policy["scheduler_snapshot_sha256"] = _sha(scheduler)

    with pytest.raises(NonResultError, match="scheduler_snapshot_policy_mismatch"):
        verify_resource_policy_bindings(validate_resource_policy(policy), **artifacts)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_resource_policy_v2_binding_verifier_rejects_linked_stage_input(
    tmp_path: Path, link_kind: str
):
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)
    first_root = artifacts["input_roots"][RESOURCE_STAGES_V2[0]]
    input_file = first_root / "input.bin"
    original = first_root / "original.bin"
    input_file.rename(original)
    if link_kind == "symlink":
        input_file.symlink_to(original.name)
    else:
        os.link(original, input_file)

    with pytest.raises(
        NonResultError,
        match="unsafe_or_missing_stage_input|nonregular_or_linked_stage_input",
    ):
        verify_resource_policy_bindings(validate_resource_policy(policy), **artifacts)


def test_resource_policy_v2_binding_verifier_rejects_manifest_byte_sum_drift(
    tmp_path: Path,
):
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)
    policy["stage_bindings"][0]["expected_input_bytes"] += 1

    with pytest.raises(NonResultError, match="stage_input_bytes_mismatch"):
        verify_resource_policy_bindings(validate_resource_policy(policy), **artifacts)


def test_resource_policy_v2_rejects_empty_source_capsule(tmp_path: Path):
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)
    source = artifacts["source_capsule"]
    assert isinstance(source, Path)
    source.write_bytes(
        canonical_json_bytes({"schema_version": "e4-resource-source-capsule/v2", "members": []})
    )
    policy["source_capsule_sha256"] = _sha(source)

    with pytest.raises(NonResultError, match="resource_source_capsule_schema_mismatch"):
        verify_resource_policy_bindings(validate_resource_policy(policy), **artifacts)


def test_resource_policy_v2_rejects_empty_workload_registry(tmp_path: Path):
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)
    workload = artifacts["synthetic_workload_registry"]
    assert isinstance(workload, Path)
    workload.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "e4-resource-synthetic-workload-registry/v2",
                "workloads": [],
            }
        )
    )
    policy["synthetic_workload_registry_sha256"] = _sha(workload)

    with pytest.raises(NonResultError, match="synthetic_workload_registry_stage_mismatch"):
        verify_resource_policy_bindings(validate_resource_policy(policy), **artifacts)


def test_resource_policy_v2_rejects_runner_artifact_drift(tmp_path: Path):
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)
    runners = artifacts["stage_runners"]
    runner = runners[RESOURCE_STAGES_V2[0]]
    runner.write_text("# drifted runner\n", encoding="utf-8")

    with pytest.raises(NonResultError, match="stage_runner_hash_mismatch"):
        verify_resource_policy_bindings(validate_resource_policy(policy), **artifacts)


def test_resource_policy_v2_rejects_empty_canonical_arguments(tmp_path: Path):
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)
    stage = RESOURCE_STAGES_V2[0]
    arguments = artifacts["stage_arguments"][stage]
    arguments.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "e4-resource-stage-arguments/v2",
                "stage": stage,
                "arguments": {},
            }
        )
    )
    contract = artifacts["stage_contracts"][stage]
    contract_payload = strict_json_load_bytes(contract.read_bytes())
    contract_payload["arguments_sha256"] = _sha(arguments)
    contract.write_bytes(canonical_json_bytes(contract_payload))
    policy["stage_bindings"][0]["stage_contract_sha256"] = _sha(contract)

    with pytest.raises(NonResultError, match="resource_stage_arguments_schema_mismatch"):
        verify_resource_policy_bindings(validate_resource_policy(policy), **artifacts)


def test_resource_policy_v2_requires_strict_utc_scheduler_timestamp(tmp_path: Path):
    policy, artifacts = _write_policy_binding_artifacts(tmp_path)
    scheduler = artifacts["scheduler_snapshot"]
    assert isinstance(scheduler, Path)
    payload = strict_json_load_bytes(scheduler.read_bytes())
    payload["captured_at_utc"] = "not-a-time"
    scheduler.write_bytes(canonical_json_bytes(payload))
    policy["scheduler_snapshot_sha256"] = _sha(scheduler)

    with pytest.raises(NonResultError, match="invalid_scheduler_snapshot_timestamp"):
        verify_resource_policy_bindings(validate_resource_policy(policy), **artifacts)


@pytest.mark.parametrize("fault", ["missing", "unknown"])
def test_resource_policy_v2_rejects_nonclosed_top_level(fault: str):
    policy = _policy_v2()
    if fault == "missing":
        del policy["allocation_cpus"]
    else:
        policy["scratch_authority_path"] = "/synthetic/not-an-authority"

    with pytest.raises(NonResultError, match="resource_policy_v2_schema_mismatch"):
        validate_resource_policy(policy)


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "reordered", "extra", "missing", "disguised_scientific"],
)
def test_resource_policy_v2_rejects_noncanonical_allowed_stages(fault: str):
    policy = _policy_v2()
    stages = policy["allowed_stages"]
    assert isinstance(stages, list)
    if fault == "duplicate":
        stages[1] = stages[0]
    elif fault == "reordered":
        stages[0], stages[1] = stages[1], stages[0]
    elif fault == "extra":
        stages.append("synthetic_extra_kernel")
    elif fault == "missing":
        stages.pop()
    else:
        stages[2] = "tanager_scientific_result_kernel"

    with pytest.raises(NonResultError, match="stage_sequence_mismatch"):
        validate_resource_policy(policy)


@pytest.mark.parametrize("fault", ["missing", "unknown", "reordered", "hash", "bytes"])
def test_resource_policy_v2_rejects_invalid_stage_bindings(fault: str):
    policy = _policy_v2()
    bindings = policy["stage_bindings"]
    assert isinstance(bindings, list)
    if fault == "missing":
        bindings.pop()
    elif fault == "unknown":
        bindings[0]["artifact_path"] = "dynamic.json"
    elif fault == "reordered":
        bindings[0], bindings[1] = bindings[1], bindings[0]
    elif fault == "hash":
        bindings[0]["stage_contract_sha256"] = "A" * 64
    else:
        bindings[0]["expected_input_bytes"] = True

    with pytest.raises(
        NonResultError,
        match="resource_stage_bindings_mismatch|invalid_resource_stage_binding",
    ):
        validate_resource_policy(policy)


def test_resource_policy_v2_rejects_stage_envelopes_above_global_budget():
    policy = _policy_v2()
    policy["stage_bindings"][0]["peak_rss_limit_bytes"] = 901
    with pytest.raises(NonResultError, match="resource_stage_limit_exceeds_effective_allocation"):
        validate_resource_policy(policy)

    policy = _policy_v2()
    for binding in policy["stage_bindings"]:
        binding["wall_limit_seconds"] = 16
    with pytest.raises(NonResultError, match="resource_stage_limits_exceed_aggregate_allocation"):
        validate_resource_policy(policy)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("wall_seconds_mechanism", "wall_clock_seconds"),
        ("cpu_seconds_mechanism", "slurm_cpu_authoritative"),
        ("peak_rss_bytes_mechanism", "process_rss_polling"),
        ("scratch_bytes_mechanism", "visible_file_apparent_bytes"),
        ("scratch_sampler_cadence_ns", 2_000_000_000),
        ("scratch_sampler_availability", "best_effort"),
        ("scheduler_cross_check", "slurm_authoritative"),
    ],
)
def test_resource_policy_v2_rejects_changed_measurement_contract(key: str, value: object):
    policy = _policy_v2()
    contract = policy["measurement_contract"]
    assert isinstance(contract, dict)
    contract[key] = value

    with pytest.raises(NonResultError, match="invalid_measurement_contract"):
        validate_resource_policy(policy)


@pytest.mark.parametrize("fault", ["missing", "unknown"])
def test_resource_policy_v2_rejects_measurement_contract_key_drift(fault: str):
    policy = _policy_v2()
    contract = policy["measurement_contract"]
    assert isinstance(contract, dict)
    if fault == "missing":
        del contract["input_bytes_mechanism"]
    else:
        contract["scheduler_authority"] = "none"

    with pytest.raises(NonResultError, match="invalid_measurement_contract"):
        validate_resource_policy(policy)


@pytest.mark.parametrize(
    "key",
    [
        "scheduler_snapshot_sha256",
        "source_capsule_sha256",
        "synthetic_workload_registry_sha256",
    ],
)
def test_resource_policy_v2_requires_lowercase_sha256(key: str):
    policy = _policy_v2()
    policy[key] = "A" * 64

    with pytest.raises(NonResultError, match="invalid_resource_policy_v2_hash"):
        validate_resource_policy(policy)


@pytest.mark.parametrize("fault", ["missing", "unknown"])
def test_resource_telemetry_v2_rejects_nonclosed_rows(fault: str):
    rows = _telemetry_v2()
    if fault == "missing":
        del rows[0]["input_bytes"]
    else:
        rows[0]["scheduler_job_id"] = "synthetic"

    with pytest.raises(NonResultError, match="resource_telemetry_v2_schema_mismatch"):
        _validate_v2(rows=rows)


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "reordered", "extra", "missing", "disguised_scientific"],
)
def test_resource_telemetry_v2_rejects_noncanonical_stage_rows(fault: str):
    rows = _telemetry_v2()
    if fault == "duplicate":
        rows[1]["stage"] = rows[0]["stage"]
    elif fault == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "extra":
        rows.append(deepcopy(rows[-1]))
    elif fault == "missing":
        rows.pop()
    else:
        rows[2]["stage"] = "tanager_scientific_result_kernel"

    with pytest.raises(NonResultError, match="stage_sequence_mismatch"):
        _validate_v2(rows=rows)


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("wall_seconds", -1),
        ("cpu_seconds", 1.5),
        ("peak_rss_bytes", True),
        ("input_bytes", -1),
        ("scratch_bytes", -1),
    ],
)
def test_resource_telemetry_v2_requires_nonnegative_integer_metrics(metric: str, value: object):
    rows = _telemetry_v2()
    rows[0][metric] = value

    with pytest.raises(NonResultError, match="invalid_resource_telemetry_v2_value"):
        _validate_v2(rows=rows)


@pytest.mark.parametrize("value", [False, 0.0, 1, -1])
def test_resource_telemetry_v2_requires_zero_integer_exit_status(value: object):
    rows = _telemetry_v2()
    rows[0]["exit_status"] = value

    with pytest.raises(NonResultError, match="resource_pilot_nonzero_exit"):
        _validate_v2(rows=rows)


@pytest.mark.parametrize(
    "key",
    [
        "input_manifest_sha256",
        "stage_contract_sha256",
        "measurement_attestation_sha256",
    ],
)
def test_resource_telemetry_v2_requires_lowercase_sha256(key: str):
    rows = _telemetry_v2()
    rows[0][key] = "F" * 64

    with pytest.raises(NonResultError, match="invalid_resource_telemetry_v2_hash"):
        _validate_v2(rows=rows)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cgroup_v2_memory_peak_available", False),
        ("scratch_sampler_cadence_ns", 2_000_000_000),
        ("scratch_max_sample_gap_ns", 1_000_000_001),
        ("scratch_missed_samples", 1),
        ("scratch_final_scan_completed", False),
        ("scratch_sampler_failed", True),
        ("scratch_escape_detected", True),
    ],
)
def test_resource_telemetry_v2_requires_positive_measurement_attestation(field: str, value: object):
    payload = _attestation_payload_v2(0, RESOURCE_STAGES_V2[0])
    payload[field] = value
    attestations = _attestation_bytes_v2()
    attestations[0] = canonical_json_bytes(payload)
    rows = _telemetry_v2()
    rows[0]["measurement_attestation_sha256"] = hashlib.sha256(attestations[0]).hexdigest()
    with pytest.raises(NonResultError, match="resource_measurement_attestation_failed"):
        _validate_v2(rows=rows, attestations=attestations)


def test_resource_telemetry_v2_binds_raw_attestation_bytes():
    attestations = _attestation_bytes_v2()
    attestations[0] += b" "
    with pytest.raises(NonResultError, match="measurement_attestation_hash_mismatch"):
        _validate_v2(attestations=attestations)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload.update(wall_end_monotonic_ns=16_000_000_001),
            "measurement_attestation_wall_mismatch",
        ),
        (
            lambda payload: payload.update(cpu_self_user_after_ns=-1),
            "invalid_measurement_attestation_value",
        ),
        (
            lambda payload: payload.update(peak_rss_bytes=101),
            "measurement_attestation_telemetry_mismatch",
        ),
    ],
)
def test_resource_telemetry_v2_reconstructs_attested_measurements(mutation, error: str):
    payload = _attestation_payload_v2(0, RESOURCE_STAGES_V2[0])
    mutation(payload)
    attestations = _attestation_bytes_v2()
    attestations[0] = canonical_json_bytes(payload)
    rows = _telemetry_v2()
    rows[0]["measurement_attestation_sha256"] = hashlib.sha256(attestations[0]).hexdigest()
    with pytest.raises(NonResultError, match=error):
        _validate_v2(rows=rows, attestations=attestations)


def test_resource_telemetry_v2_requires_one_attestation_per_stage():
    with pytest.raises(NonResultError, match="measurement_attestation_count_mismatch"):
        _validate_v2(attestations=_attestation_bytes_v2()[:-1])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_manifest_sha256", "f" * 64),
        ("stage_contract_sha256", "f" * 64),
        ("input_bytes", 13),
    ],
)
def test_resource_telemetry_v2_must_equal_frozen_stage_bindings(field: str, value: object):
    rows = _telemetry_v2()
    rows[0][field] = value
    with pytest.raises(NonResultError, match="resource_stage_binding_mismatch"):
        _validate_v2(rows=rows)


def test_resource_telemetry_v2_enforces_per_stage_cpu_plausibility():
    rows = _telemetry_v2()
    rows[0]["wall_seconds"] = 1
    rows[0]["cpu_seconds"] = 5
    with pytest.raises(NonResultError, match="resource_stage_cpu_plausibility_failed"):
        _validate_v2(rows=rows)


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("wall_seconds", 11),
        ("cpu_seconds", 21),
        ("peak_rss_bytes", 501),
        ("scratch_bytes", 501),
    ],
)
def test_resource_telemetry_v2_enforces_stage_limits(metric: str, value: int):
    rows = _telemetry_v2()
    rows[0][metric] = value
    if metric == "cpu_seconds":
        rows[0]["wall_seconds"] = 6

    with pytest.raises(NonResultError, match="resource_stage_limit_exceeded"):
        _validate_v2(rows=rows)


def test_json_duplicate_keys_and_nonfinite_values_fail_closed():
    from tanager_rocks.emit_l2b_nonresult import strict_json_load_bytes

    with pytest.raises(NonResultError, match="duplicate_json_key"):
        strict_json_load_bytes(b'{"a": 1, "a": 2}')
    with pytest.raises(NonResultError, match="nonfinite_json_value"):
        canonical_json_bytes({"bad": float("nan")})
