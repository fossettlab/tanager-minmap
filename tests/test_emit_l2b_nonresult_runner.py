"""Synthetic command-boundary tests for the E4 non-result driver."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest

import tanager_rocks.emit_l2b as emit_l2b
from tanager_rocks.emit_l2b_nonresult import (
    RESOURCE_ADMISSION_EVIDENCE_FILES_V2,
    RESOURCE_ATTESTATION_FILENAMES_V2,
    RESOURCE_MEASUREMENT_CONTRACT_V2,
    RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2,
    RESOURCE_STAGES_V2,
    NonResultError,
    atomic_write_bundle,
    canonical_json_bytes,
    capture_resource_policy_binding_evidence,
    verify_nonresult_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_emit_l2b_validation.py"
VERIFIER = ROOT / "scripts" / "verify_emit_l2b_nonresult_bundle.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _real_preflight_policy_v2() -> dict[str, object]:
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


def _write_policy_binding_artifacts(
    tmp_path: Path,
    policy: dict[str, object],
) -> dict[str, object]:
    root = tmp_path / "runner-policy-bindings"
    root.mkdir()
    scheduler = root / "scheduler.json"
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
    source_root = root / "source-root"
    source_root.mkdir()
    source_member = source_root / "source.py"
    source_member.write_text("# frozen runner source\n", encoding="utf-8")
    source_capsule = root / "source-capsule.json"
    source_capsule.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "e4-resource-source-capsule/v2",
                "members": [
                    {
                        "path": source_member.name,
                        "sha256": _sha(source_member),
                        "size_bytes": source_member.stat().st_size,
                    }
                ],
            }
        )
    )
    measurement_sha256 = hashlib.sha256(
        canonical_json_bytes(RESOURCE_MEASUREMENT_CONTRACT_V2)
    ).hexdigest()
    stage_contracts: dict[str, Path] = {}
    stage_runners: dict[str, Path] = {}
    stage_arguments: dict[str, Path] = {}
    input_manifests: dict[str, Path] = {}
    input_roots: dict[str, Path] = {}
    bindings = policy["stage_bindings"]
    assert isinstance(bindings, list)
    for index, stage in enumerate(RESOURCE_STAGES_V2):
        stage_runner = root / f"runner-{index}.py"
        stage_runner.write_text(f"# runner for {stage}\n", encoding="utf-8")
        arguments = root / f"arguments-{index}.json"
        arguments.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "e4-resource-stage-arguments/v2",
                    "stage": stage,
                    "arguments": {"fixture": "closed", "stage_index": index},
                }
            )
        )
        input_root = root / f"inputs-{index}"
        input_root.mkdir()
        input_file = input_root / "input.bin"
        input_file.write_bytes(f"input-{stage}".encode())
        input_manifest = root / f"input-manifest-{index}.json"
        input_manifest.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "e4-resource-stage-input-manifest/v2",
                    "stage": stage,
                    "files": [
                        {
                            "path": input_file.name,
                            "sha256": _sha(input_file),
                            "size_bytes": input_file.stat().st_size,
                        }
                    ],
                }
            )
        )
        contract = root / f"contract-{index}.json"
        contract.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "e4-resource-stage-contract/v2",
                    "stage": stage,
                    "runner_sha256": _sha(stage_runner),
                    "arguments_sha256": _sha(arguments),
                    "measurement_contract_sha256": measurement_sha256,
                }
            )
        )
        binding = bindings[index]
        binding["input_manifest_sha256"] = _sha(input_manifest)
        binding["expected_input_bytes"] = input_file.stat().st_size
        binding["stage_contract_sha256"] = _sha(contract)
        stage_contracts[stage] = contract
        stage_runners[stage] = stage_runner
        stage_arguments[stage] = arguments
        input_manifests[stage] = input_manifest
        input_roots[stage] = input_root
    workload = root / "workload.json"
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
    policy["source_capsule_sha256"] = _sha(source_capsule)
    policy["synthetic_workload_registry_sha256"] = _sha(workload)
    return {
        "scheduler_snapshot": scheduler,
        "source_capsule": source_capsule,
        "source_root": source_root,
        "synthetic_workload_registry": workload,
        "stage_contracts": stage_contracts,
        "stage_runners": stage_runners,
        "stage_arguments": stage_arguments,
        "input_manifests": input_manifests,
        "input_roots": input_roots,
    }


def _write_real_resource_admission_bundle(
    tmp_path: Path,
    policy: dict[str, object],
) -> Path:
    artifacts = _write_policy_binding_artifacts(tmp_path, policy)
    binding_payloads = capture_resource_policy_binding_evidence(
        policy,
        verified_at_utc="2026-08-11T00:00:01Z",
        **artifacts,
    )
    measurement_sha256 = hashlib.sha256(
        canonical_json_bytes(RESOURCE_MEASUREMENT_CONTRACT_V2)
    ).hexdigest()
    attestations: list[bytes] = []
    telemetry: list[dict[str, object]] = []
    for index, stage in enumerate(RESOURCE_STAGES_V2):
        binding = policy["stage_bindings"][index]
        start = 10_000_000_000 + index * 20_000_000_000
        end = start + 5_000_000_000
        attestation = canonical_json_bytes(
            {
                "schema_version": "e4-resource-measurement-attestation/v2",
                "stage": stage,
                "stage_contract_sha256": binding["stage_contract_sha256"],
                "input_manifest_sha256": binding["input_manifest_sha256"],
                "measurement_contract_sha256": measurement_sha256,
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
        )
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
    policy_bytes = canonical_json_bytes(policy)
    telemetry_bytes = canonical_json_bytes(telemetry)
    attestation_closure = hashlib.sha256(
        b"".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode()
            for name, data in zip(RESOURCE_ATTESTATION_FILENAMES_V2, attestations, strict=True)
        )
    ).hexdigest()
    admission = {
        "schema_version": "e4-resource-admission/v2",
        "admission_status": "PASS",
        "policy_class": "real_nonresult_resource_pilot",
        "resource_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "scheduler_snapshot_sha256": policy["scheduler_snapshot_sha256"],
        "source_capsule_sha256": policy["source_capsule_sha256"],
        "synthetic_workload_registry_sha256": policy["synthetic_workload_registry_sha256"],
        "telemetry_sha256": hashlib.sha256(telemetry_bytes).hexdigest(),
        "measurement_attestation_closure_sha256": attestation_closure,
        "binding_evidence_closure_sha256": hashlib.sha256(
            b"".join(
                f"{hashlib.sha256(binding_payloads[name]).hexdigest()}  {name}\n".encode()
                for name in sorted(RESOURCE_POLICY_BINDING_EVIDENCE_FILES_V2)
            )
        ).hexdigest(),
        "synthetic_fixture_only": False,
    }
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
            for name, data in zip(RESOURCE_ATTESTATION_FILENAMES_V2, attestations, strict=True)
        },
        **binding_payloads,
    }
    assert set(payloads) == RESOURCE_ADMISSION_EVIDENCE_FILES_V2
    receipt = atomic_write_bundle(
        tmp_path,
        bundle_type="resource_admission",
        bundle_id="resource-admission-runner-test",
        manifest={"operation": "resource_admission", "endpoint_execution": "forbidden"},
        payloads=payloads,
    )
    return receipt.bundle_path


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


def _runner() -> dict[str, object]:
    return runpy.run_path(str(RUNNER))


def test_cli_exposes_only_nonresult_modes():
    parser = _runner()["_parser"]()
    help_text = parser.format_help()
    assert "mapping-only" in help_text
    assert "resource-pilot" in help_text
    assert "preflight" in help_text
    assert "scientific-run" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_mapping_only_reads_no_result_fields_and_writes_exact_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    minimum = _write_product(tmp_path, "MIN")
    uncertainty = _write_product(tmp_path, "MINUNCERT")
    contract = tmp_path / "m2_contract.json"
    _write_json(
        contract,
        {"schema_version": "e4-m2-mapping-contract/v1", "block_manifest_sha256": "b" * 64},
    )
    runner = _runner()

    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("mapping-only attempted a result-field read")

    monkeypatch.setattr(emit_l2b, "_read_field", forbidden)
    args = [
        "mapping-only",
        "--emit-min",
        str(minimum),
        "--emit-minuncert",
        str(uncertainty),
        "--expected-emit-min-sha256",
        _sha(minimum),
        "--expected-emit-minuncert-sha256",
        _sha(uncertainty),
        "--m2-mapping-contract",
        str(contract),
        "--output",
        str(tmp_path / "mapping"),
    ]
    parsed = runner["_parser"]().parse_args(args)
    output = runner["run_mapping_only"](parsed)

    receipt = verify_nonresult_bundle(output, expected_type="mapping")
    assert receipt.bundle_path == output
    assert (output / "source_mineral_inventory.csv").is_file()
    verifier = runpy.run_path(str(VERIFIER))
    assert verifier["verify"](output, "mapping") == receipt.closure_sha256


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_mapping_only_rejects_linked_source(tmp_path: Path, link_kind: str):
    minimum = _write_product(tmp_path, "MIN")
    uncertainty = _write_product(tmp_path, "MINUNCERT")
    linked = tmp_path / "linked.nc"
    if link_kind == "symlink":
        linked.symlink_to(minimum.name)
    else:
        os.link(minimum, linked)
    contract = tmp_path / "m2_contract.json"
    _write_json(
        contract,
        {"schema_version": "e4-m2-mapping-contract/v1", "block_manifest_sha256": "b" * 64},
    )
    runner = _runner()
    arguments = runner["_parser"]().parse_args(
        [
            "mapping-only",
            "--emit-min",
            str(linked if link_kind == "symlink" else minimum),
            "--emit-minuncert",
            str(uncertainty),
            "--expected-emit-min-sha256",
            _sha(minimum),
            "--expected-emit-minuncert-sha256",
            _sha(uncertainty),
            "--m2-mapping-contract",
            str(contract),
            "--output",
            str(tmp_path / "mapping"),
        ]
    )
    with pytest.raises(NonResultError, match="unsafe_or_missing_file|nonregular_or_linked_file"):
        runner["run_mapping_only"](arguments)


def test_resource_mode_requires_explicit_synthetic_guard(tmp_path: Path):
    runner = _runner()
    policy = tmp_path / "policy.json"
    _write_json(
        policy,
        {
            "schema_version": "e4-resource-policy/v1",
            "allocation_memory_bytes": 100,
            "memory_reserve_bytes": 10,
            "allocation_wall_seconds": 100,
            "wall_reserve_seconds": 10,
            "scratch_capacity_bytes": 100,
            "scratch_reserve_bytes": 10,
        },
    )
    with pytest.raises(SystemExit):
        runner["_parser"]().parse_args(["resource-pilot"])
    assert "run_resource_pilot" in runner


def test_preflight_validates_controls_without_calling_metadata_loader(tmp_path: Path):
    runner = _runner()
    minimum = _write_product(tmp_path, "MIN")
    uncertainty = _write_product(tmp_path, "MINUNCERT")
    contract = tmp_path / "m2_contract.json"
    _write_json(
        contract,
        {"schema_version": "e4-m2-mapping-contract/v1", "block_manifest_sha256": "b" * 64},
    )
    mapping_args = runner["_parser"]().parse_args(
        [
            "mapping-only",
            "--emit-min",
            str(minimum),
            "--emit-minuncert",
            str(uncertainty),
            "--expected-emit-min-sha256",
            _sha(minimum),
            "--expected-emit-minuncert-sha256",
            _sha(uncertainty),
            "--m2-mapping-contract",
            str(contract),
            "--output",
            str(tmp_path / "mapping"),
        ]
    )
    mapping = runner["run_mapping_only"](mapping_args)
    policy = tmp_path / "policy.json"
    policy_payload = _real_preflight_policy_v2()
    resource_admission_bundle = _write_real_resource_admission_bundle(
        tmp_path,
        policy_payload,
    )
    policy.write_bytes(canonical_json_bytes(policy_payload))
    input_manifest = tmp_path / "input_manifest.json"
    _write_json(input_manifest, {"schema_version": "input-manifest/v1"})
    crosswalk = tmp_path / "ontology.csv"
    crosswalk.write_text("ontology_version,index,name,group,library\nv1,1,alunite,1,splib\n")
    evidence = tmp_path / "evidence.json"
    _write_json(evidence, {"schema_version": "e4-ontology-evidence-manifest/v1"})
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {"schema_version": "e4-scientific-output-registry/v1", "expected_files": ["metrics.csv"]},
    )
    decision = tmp_path / "decision.json"
    _write_json(
        decision,
        {
            "schema_version": "e4-decision-record/v1",
            "ontology": {
                "crosswalk_sha256": _sha(crosswalk),
                "entries": [
                    {
                        "group": 1,
                        "index": 1,
                        "name": "alunite",
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
            "output_registry": {"sha256": _sha(registry)},
        },
    )
    parser = runner["_parser"]()
    arguments = parser.parse_args(
        [
            "preflight",
            "--decision-record",
            str(decision),
            "--resource-policy",
            str(policy),
            "--mapping-admission",
            str(mapping),
            "--resource-admission-bundle",
            str(resource_admission_bundle),
            "--input-manifest",
            str(input_manifest),
            "--ontology-crosswalk",
            str(crosswalk),
            "--ontology-evidence-manifest",
            str(evidence),
            "--output-registry",
            str(registry),
            "--output",
            str(tmp_path / "preflight"),
        ]
    )
    assert not hasattr(arguments, "emit_min")
    assert not hasattr(arguments, "tanager_scene")
    runner["load_emit_l2b_metadata"] = lambda *args: (_ for _ in ()).throw(AssertionError())
    output = runner["run_preflight"](arguments)
    verifier = runpy.run_path(str(VERIFIER))
    assert verifier["verify"](output, "preflight")

    crosswalk.write_text(
        "ontology_version,index,name,group,library\nv1,1,jarosite,1,splib\n",
        encoding="utf-8",
    )
    with pytest.raises(NonResultError, match="decision_ontology_crosswalk_mismatch"):
        runner["run_preflight"](arguments)

    crosswalk.write_text(
        "ontology_version,index,name,group,library\nv1,1,alunite,1,splib\n",
        encoding="utf-8",
    )
    embedded_runner = (
        resource_admission_bundle / "policy_bindings/00_source_snapshot_and_verify_runner.bin"
    )
    embedded_runner.write_bytes(embedded_runner.read_bytes() + b"# drift\n")
    with pytest.raises(NonResultError, match="checksum_mismatch"):
        runner["run_preflight"](arguments)


def test_independent_verifier_rechecks_closure_after_semantic_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    minimum = _write_product(tmp_path, "MIN")
    uncertainty = _write_product(tmp_path, "MINUNCERT")
    contract = tmp_path / "m2_contract.json"
    _write_json(
        contract,
        {"schema_version": "e4-m2-mapping-contract/v1", "block_manifest_sha256": "b" * 64},
    )
    runner = _runner()
    arguments = runner["_parser"]().parse_args(
        [
            "mapping-only",
            "--emit-min",
            str(minimum),
            "--emit-minuncert",
            str(uncertainty),
            "--expected-emit-min-sha256",
            _sha(minimum),
            "--expected-emit-minuncert-sha256",
            _sha(uncertainty),
            "--m2-mapping-contract",
            str(contract),
            "--output",
            str(tmp_path / "mapping"),
        ]
    )
    output = runner["run_mapping_only"](arguments)
    verifier = runpy.run_path(str(VERIFIER))
    original = verifier["_verify_mapping"]

    def mutate_after_semantic_read(bundle: Path) -> None:
        original(bundle)
        (bundle / "geometry_contract.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setitem(
        verifier["verify"].__globals__,
        "_verify_mapping",
        mutate_after_semantic_read,
    )
    with pytest.raises(NonResultError, match="checksum_mismatch"):
        verifier["verify"](output, "mapping")


def test_sanitized_cli_failure_does_not_echo_attacker_filename(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    runner = _runner()
    attacker = tmp_path / "LEAK_ME_987654.json"
    attacker.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        runner["main"](
            [
                "mapping-only",
                "--emit-min",
                str(attacker),
                "--emit-minuncert",
                str(attacker),
                "--expected-emit-min-sha256",
                "0" * 64,
                "--expected-emit-minuncert-sha256",
                "0" * 64,
                "--m2-mapping-contract",
                str(attacker),
                "--output",
                str(tmp_path / "output"),
            ]
        )
    captured = capsys.readouterr()
    assert "LEAK_ME_987654" not in str(error.value)
    assert "LEAK_ME_987654" not in captured.out + captured.err
    assert str(error.value).startswith("E4_NONRESULT_FAILED:")


def test_verifier_help_smoke():
    verifier = runpy.run_path(str(VERIFIER))
    parser = verifier["build_parser"]()
    assert "--bundle" in parser.format_help()
