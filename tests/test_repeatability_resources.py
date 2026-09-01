"""Synthetic tests for the endpoint-blind repeatability resource admission gate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import tanager_rocks.repeatability as repeatability_module
import tanager_rocks.repeatability_resources as resource_module
from tanager_rocks.repeatability import RepeatabilityPaths, run_repeatability_packet
from tanager_rocks.repeatability_resources import (
    BOOTSTRAP_REPLICATES,
    NULL_REPLICATES,
    RESAMPLED_METRIC_COMPONENTS,
    RESOURCE_ADMISSION_IDENTITY,
    RESOURCE_ADMISSION_SCHEMA_VERSION,
    ResourceAdmissionError,
    _expected_tasks,
    atomic_write_json,
    build_footprint_manifest,
    ensure_output_outside_roots,
    produce_resource_admission,
    read_strict_json_object,
    sha256_file,
    validate_resource_admission,
)

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "docs" / "m2_repeatability_resource_admission_rule_v1.json"
SCRIPT = ROOT / "scripts" / "admit_repeatability_resources.py"
MODULE = ROOT / "src" / "tanager_rocks" / "repeatability_resources.py"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _schedule(n_blocks: int = 4) -> dict[str, object]:
    components = {}
    for metric in RESAMPLED_METRIC_COMPONENTS:
        components[metric] = {
            "bootstrap": {"scheduled_replicates": BOOTSTRAP_REPLICATES},
            "spatial_null": (
                {"status": "not_applicable"}
                if metric.endswith("prevalence_ratio")
                else {"status": "scheduled", "scheduled_replicates": 24}
            ),
        }
    return {
        "contains_endpoint_values": False,
        "n_complete_paired_overlap_blocks": n_blocks,
        "components": components,
    }


def _execution_manifest(tasks: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "scientific_execution_identity": "paired-complete-block-metric-contract-v2",
        "permitted_actions": ["timing_pilot", "full"],
        "accepted_scientific_result": False,
        "resume_validates_completed_results": True,
        "protocol_artifacts": {},
        "block_handoffs": {},
        "source_inventory": {},
        "member_order": {
            "scenes": {
                "bingham": ["20250911_191523_58_4001", "20250911_191547_88_4001"],
                "goldfield": [
                    "20240925_185504_87_4001",
                    "20240925_185509_74_4001",
                    "20250222_190233_00_4001",
                    "20250222_190237_16_4001",
                    "20250222_190241_32_4001",
                ],
            },
            "complete_block_ids": {"bingham": [1, 2], "goldfield": [1, 2, 3]},
            "tasks": tasks,
        },
        "resampling": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "null_replicates_maximum": NULL_REPLICATES,
            "metric_components": list(RESAMPLED_METRIC_COMPONENTS),
        },
        "compute_controls": {
            "workers": 4,
            "task_result_order": "frozen pair order then sorted manifest-declared layer",
        },
    }


def _stable_json_sha256(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path | int]:
    timing_root = tmp_path / "timing"
    runtime_root = tmp_path / "runtime"
    timing_root.mkdir(parents=True)
    runtime_root.mkdir(parents=True)
    execution = timing_root / "execution_manifest.json"
    tasks = _expected_tasks()
    execution_value = _execution_manifest(tasks)
    _write_json(execution, execution_value)
    timing = timing_root / "timing_pilot.json"
    _write_json(
        timing,
        {
            "schema_version": "2.0",
            "mode": "timing",
            "scientific_execution_identity": "paired-complete-block-metric-contract-v2",
            "accepted_scientific_result": False,
            "contains_endpoint_values": False,
            "execution_manifest": str(execution),
            "execution_manifest_sha256": sha256_file(execution),
            "task_id": tasks[0]["task_id"],
            "workers": 4,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "null_replicates_maximum": NULL_REPLICATES,
            "resampling_branch_schedule": _schedule(),
            "elapsed_seconds": 10.0,
            "result_sha256": "a" * 64,
        },
    )
    _write_json(
        timing_root / "progress.json",
        {
            "schema_version": "2.0",
            "execution_manifest_sha256": _stable_json_sha256(execution_value),
            "run_status": "timing_pilot_complete",
            "accepted_final_manifest": False,
            "tasks": [
                {
                    **task,
                    "status": "completed" if task["index"] == 0 else "pending",
                    "attempts": 1 if task["index"] == 0 else 0,
                    "elapsed_seconds": 10.0 if task["index"] == 0 else None,
                    "result_path": None,
                    "result_sha256": "a" * 64 if task["index"] == 0 else None,
                }
                for task in tasks
            ],
        },
    )
    (runtime_root / "timing_python_version.txt").write_text("Python 3.11", encoding="utf-8")
    scheduler = tmp_path / "scheduler_record.json"
    _write_json(
        scheduler,
        {
            "schema_version": "1.0",
            "job_id": "synthetic-private-job-id",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "oom_killed": False,
            "elapsed_raw_seconds": 20,
            "timelimit_raw_seconds": 86400,
            "alloc_cpus": 4,
            "req_mem_bytes": 64 * 1024**3,
            "max_rss_bytes": 8 * 1024**3,
            "max_vm_bytes": 9 * 1024**3,
            "max_disk_read_bytes": 1024,
            "max_disk_write_bytes": 1024,
            "node": "synthetic-private-node",
        },
    )
    source_root = tmp_path / "source_repo"
    source_manifest = source_root / "docs" / "source_manifest.sha256"
    source_manifest.parent.mkdir(parents=True)
    source_lines = []
    for index in range(47):
        item = source_root / "capsule" / f"source_{index:02d}.txt"
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text(f"source {index}\n", encoding="utf-8")
        source_lines.append(f"{sha256_file(item)}  capsule/{item.name}")
    source_manifest.write_text("\n".join(sorted(source_lines)) + "\n", encoding="utf-8")
    footprint = tmp_path / "footprint.json"
    atomic_write_json(footprint, build_footprint_manifest(timing_root, runtime_root))
    return {
        "timing_root": timing_root,
        "runtime_root": runtime_root,
        "timing": timing,
        "scheduler": scheduler,
        "source_manifest": source_manifest,
        "footprint": footprint,
        "free_bytes": 32 * 1024**3,
    }


def _produce(fixture: dict[str, Path | int]) -> dict[str, object]:
    return produce_resource_admission(
        timing_pilot_path=fixture["timing"],  # type: ignore[arg-type]
        scheduler_record_path=fixture["scheduler"],  # type: ignore[arg-type]
        footprint_manifest_path=fixture["footprint"],  # type: ignore[arg-type]
        timing_output_root=fixture["timing_root"],  # type: ignore[arg-type]
        runtime_root=fixture["runtime_root"],  # type: ignore[arg-type]
        free_bytes=fixture["free_bytes"],
        source_manifest_path=fixture["source_manifest"],  # type: ignore[arg-type]
        rule_path=RULE,
        verifier_script_path=SCRIPT,
        verifier_module_path=MODULE,
    )


def _write_admission(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "admission" / "resource_admission.json"
    atomic_write_json(path, value)
    return path


def test_producer_passes_conservative_64_gib_rule_and_sanitizes_inputs(tmp_path):
    fixture = _fixture(tmp_path)
    admission = _produce(fixture)

    assert admission["schema_version"] == RESOURCE_ADMISSION_SCHEMA_VERSION
    assert admission["resource_admission_identity"] == RESOURCE_ADMISSION_IDENTITY
    assert admission["status"] == "PASS-64"
    assert admission["full_mode_admitted"] is True
    assert admission["failure_codes"] == []
    assert admission["accepted_scientific_result"] is False
    assert admission["contains_endpoint_values"] is False
    rendered = json.dumps(admission, sort_keys=True)
    assert "synthetic-private-job-id" not in rendered
    assert "synthetic-private-node" not in rendered
    assert admission["computed_bounds"] == {
        "memory_bound_bytes": 32 * 1024**3,
        "wall_bound_seconds": 2460.0,
        "disk_bound_bytes": 2 * 1024**3 + 4 * admission["telemetry"]["footprint_bytes"],
        "required_free_bytes": 16 * 1024**3,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_status", "failure"),
    [
        (lambda record: record.update({"max_rss_bytes": 13 * 1024**3}), "HOLD", "memory_bound"),
        (lambda record: record.update({"elapsed_raw_seconds": 40000}), "HOLD", "wall_bound"),
        (
            lambda record: record.update({"state": "OUT_OF_MEMORY", "oom_killed": True}),
            "FAIL",
            "scheduler_state",
        ),
        (lambda record: record.update({"exit_code": "1:0"}), "FAIL", "scheduler_exit_code"),
        (lambda record: record.update({"alloc_cpus": 8}), "FAIL", "scheduler_cpus"),
        (lambda record: record.update({"max_rss_bytes": True}), "FAIL", "scheduler_max_rss"),
    ],
)
def test_producer_classifies_bounds_and_scheduler_failures(
    tmp_path, mutation, expected_status, failure
):
    fixture = _fixture(tmp_path)
    record_path = fixture["scheduler"]
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    mutation(record)
    _write_json(Path(record_path), record)

    admission = _produce(fixture)

    assert admission["status"] == expected_status
    assert admission["full_mode_admitted"] is (expected_status == "PASS-64")
    assert failure in admission["failure_codes"]


def test_producer_fails_closed_for_boolean_free_space_nonfinite_timing_and_missing_telemetry(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    fixture["free_bytes"] = True
    assert _produce(fixture)["status"] == "FAIL"

    fixture = _fixture(tmp_path / "nonfinite")
    timing_path = fixture["timing"]
    timing = json.loads(Path(timing_path).read_text(encoding="utf-8"))
    timing["elapsed_seconds"] = float("nan")
    Path(timing_path).write_text(json.dumps(timing, allow_nan=True), encoding="utf-8")
    assert _produce(fixture)["failure_codes"] == ["malformed_timing_capsule"]

    fixture = _fixture(tmp_path / "missing")
    scheduler_path = fixture["scheduler"]
    scheduler = json.loads(Path(scheduler_path).read_text(encoding="utf-8"))
    scheduler.pop("max_rss_bytes")
    _write_json(Path(scheduler_path), scheduler)
    assert _produce(fixture)["failure_codes"] == ["scheduler_field_closure"]


def test_structural_failure_preserves_frozen_resource_field_closure(tmp_path):
    fixture = _fixture(tmp_path)
    Path(fixture["timing"]).write_text("not-json\n", encoding="utf-8")

    admission = _produce(fixture)
    rule = json.loads(RULE.read_text(encoding="utf-8"))
    required = set(rule["schemas"]["resource_admission"]["required_fields"])

    assert set(admission) == required
    assert admission["status"] == "FAIL"
    assert admission["full_mode_admitted"] is False
    assert admission["failure_codes"] == ["malformed_timing_capsule"]
    assert admission["telemetry"] == {}
    assert admission["computed_bounds"] == {}


def test_producer_rejects_tampered_or_reordered_footprint(tmp_path):
    fixture = _fixture(tmp_path)
    footprint_path = fixture["footprint"]
    footprint = json.loads(Path(footprint_path).read_text(encoding="utf-8"))
    footprint["files"].reverse()
    _write_json(Path(footprint_path), footprint)

    assert _produce(fixture)["failure_codes"] == ["footprint_mismatch"]


def test_producer_rejects_wrong_timing_task_count_or_order(tmp_path):
    fixture = _fixture(tmp_path)
    progress_path = Path(fixture["timing_root"]) / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["tasks"].reverse()
    _write_json(progress_path, progress)
    assert _produce(fixture)["failure_codes"] == ["timing_progress_task_order"]

    fixture = _fixture(tmp_path / "short")
    execution_path = Path(fixture["timing_root"]) / "execution_manifest.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["member_order"]["tasks"].pop()
    _write_json(execution_path, execution)
    timing_path = Path(fixture["timing"])
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["execution_manifest_sha256"] = sha256_file(execution_path)
    _write_json(timing_path, timing)
    assert _produce(fixture)["failure_codes"] == ["timing_task_order"]

    fixture = _fixture(tmp_path / "changed")
    Path(fixture["runtime_root"] / "timing_python_version.txt").write_text(
        "changed", encoding="utf-8"
    )
    assert _produce(fixture)["failure_codes"] == ["footprint_mismatch"]


def test_full_validator_requires_exact_reviewed_pass_record_and_binds_all_identities(tmp_path):
    fixture = _fixture(tmp_path)
    admission = _produce(fixture)
    path = _write_admission(tmp_path, admission)
    timing = json.loads(Path(fixture["timing"]).read_text(encoding="utf-8"))

    admitted = validate_resource_admission(
        resource_admission_path=path,
        expected_sha256=sha256_file(path),
        expected_timing_pilot_sha256=sha256_file(Path(fixture["timing"])),
        expected_execution_manifest_sha256=timing["execution_manifest_sha256"],
        rule_path=RULE,
        source_manifest_path=Path(fixture["source_manifest"]),
        verifier_script_path=SCRIPT,
        verifier_module_path=MODULE,
    )
    assert admitted == path

    altered = read_strict_json_object(path, label="resource_admission")
    altered["telemetry"]["max_rss_bytes"] = 0
    path.unlink()
    atomic_write_json(path, altered)
    with pytest.raises(ResourceAdmissionError, match="resource admission digest"):
        validate_resource_admission(
            resource_admission_path=path,
            expected_sha256=sha256_file(path)[:-1] + "0",
            expected_timing_pilot_sha256=sha256_file(Path(fixture["timing"])),
            expected_execution_manifest_sha256=timing["execution_manifest_sha256"],
            rule_path=RULE,
            source_manifest_path=Path(fixture["source_manifest"]),
            verifier_script_path=SCRIPT,
            verifier_module_path=MODULE,
        )


def test_full_validator_hashes_and_parses_one_descriptor_snapshot(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    admission = _produce(fixture)
    path = _write_admission(tmp_path, admission)
    reviewed_sha256 = sha256_file(path)
    timing = json.loads(Path(fixture["timing"]).read_text(encoding="utf-8"))
    replacement = tmp_path / "replacement.json"
    atomic_write_json(replacement, {"invalid": "replacement"})
    original_read = resource_module._read_bound_regular
    admission_reads = 0

    def replace_after_bound_read(candidate):
        nonlocal admission_reads
        result = original_read(candidate)
        if Path(candidate) == path:
            admission_reads += 1
            if admission_reads > 1:
                raise AssertionError("resource admission was reopened")
            os.replace(replacement, path)
        return result

    monkeypatch.setattr(resource_module, "_read_bound_regular", replace_after_bound_read)

    assert (
        validate_resource_admission(
            resource_admission_path=path,
            expected_sha256=reviewed_sha256,
            expected_timing_pilot_sha256=sha256_file(Path(fixture["timing"])),
            expected_execution_manifest_sha256=timing["execution_manifest_sha256"],
            rule_path=RULE,
            source_manifest_path=Path(fixture["source_manifest"]),
            verifier_script_path=SCRIPT,
            verifier_module_path=MODULE,
        )
        == path
    )
    assert admission_reads == 1
    assert original_read(path)[1] != reviewed_sha256


def test_full_validator_rejects_hold_fail_and_timing_hash_tampering(tmp_path):
    fixture = _fixture(tmp_path)
    scheduler = json.loads(Path(fixture["scheduler"]).read_text(encoding="utf-8"))
    scheduler["max_rss_bytes"] = 13 * 1024**3
    _write_json(Path(fixture["scheduler"]), scheduler)
    admission = _produce(fixture)
    path = _write_admission(tmp_path, admission)
    timing = json.loads(Path(fixture["timing"]).read_text(encoding="utf-8"))

    with pytest.raises(ResourceAdmissionError, match="resource status"):
        validate_resource_admission(
            resource_admission_path=path,
            expected_sha256=sha256_file(path),
            expected_timing_pilot_sha256=sha256_file(Path(fixture["timing"])),
            expected_execution_manifest_sha256=timing["execution_manifest_sha256"],
            rule_path=RULE,
            source_manifest_path=Path(fixture["source_manifest"]),
            verifier_script_path=SCRIPT,
            verifier_module_path=MODULE,
        )


def test_full_mode_requires_a_reviewed_resource_admission_before_any_run_lock(tmp_path):
    paths = RepeatabilityPaths(
        raw_dir=tmp_path / "raw",
        speclib_dir=tmp_path / "speclib",
        validation_dir=tmp_path / "validation",
        output_dir=tmp_path / "output",
        reference_dir=tmp_path / "reference",
    )

    with pytest.raises(ValueError, match="reviewed resource-admission path"):
        run_repeatability_packet(
            paths,
            block_manifest=tmp_path / "block_manifest.json",
            resume=True,
            expected_timing_pilot_sha256="a" * 64,
        )
    assert not paths.output_dir.exists()

    fixture = _fixture(tmp_path / "timing-tamper")
    admission = _produce(fixture)
    path = _write_admission(tmp_path / "timing-tamper", admission)
    timing = json.loads(Path(fixture["timing"]).read_text(encoding="utf-8"))
    timing["elapsed_seconds"] = 11.0
    _write_json(Path(fixture["timing"]), timing)
    with pytest.raises(ResourceAdmissionError, match="resource timing sha256"):
        validate_resource_admission(
            resource_admission_path=path,
            expected_sha256=sha256_file(path),
            expected_timing_pilot_sha256=sha256_file(Path(fixture["timing"])),
            expected_execution_manifest_sha256=timing["execution_manifest_sha256"],
            rule_path=RULE,
            source_manifest_path=Path(fixture["source_manifest"]),
            verifier_script_path=SCRIPT,
            verifier_module_path=MODULE,
        )


def test_rule_is_exact_pretty_canonical_bytes():
    value = json.loads(RULE.read_text(encoding="utf-8"))
    expected = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert RULE.read_text(encoding="utf-8") == expected


def test_repository_source_manifest_is_canonical_and_verifies():
    manifest = ROOT / "docs" / "m2_repeatability_bigmem_source_manifest.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()

    assert lines == sorted(lines)
    digest, entries = resource_module.verify_source_manifest(manifest, expected_count=47)
    assert digest == sha256_file(manifest)
    assert len(entries) == 47


def test_atomic_output_refuses_overwrite_symlink_and_symlinked_parent(tmp_path):
    output = tmp_path / "fresh" / "admission.json"
    atomic_write_json(output, {"status": "first"})
    with pytest.raises(ResourceAdmissionError, match="resource output exists"):
        atomic_write_json(output, {"status": "second"})
    assert read_strict_json_object(output, label="output") == {"status": "first"}

    target = tmp_path / "target.json"
    target.write_text("target\n", encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(ResourceAdmissionError, match="resource output exists"):
        atomic_write_json(symlink, {"status": "blocked"})

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ResourceAdmissionError):
        atomic_write_json(linked_parent / "blocked.json", {"status": "blocked"})
    assert not (real_parent / "blocked.json").exists()


def test_hardlinks_and_overlapping_or_self_including_roots_fail_closed(tmp_path):
    timing = tmp_path / "timing"
    runtime = tmp_path / "runtime"
    timing.mkdir()
    runtime.mkdir()
    original = timing / "one.txt"
    original.write_text("one\n", encoding="utf-8")
    os.link(original, timing / "two.txt")
    with pytest.raises(ResourceAdmissionError, match="footprint nonregular"):
        build_footprint_manifest(timing, runtime)

    nested = timing / "nested"
    nested.mkdir()
    with pytest.raises(ResourceAdmissionError, match="footprint roots overlap"):
        build_footprint_manifest(timing, nested)
    with pytest.raises(ResourceAdmissionError, match="output inside footprint root"):
        ensure_output_outside_roots(timing / "footprint.json", timing, runtime)


def test_exact_task_and_branch_contract_rejects_coherent_rewrite(tmp_path):
    fixture = _fixture(tmp_path)
    execution_path = Path(fixture["timing_root"]) / "execution_manifest.json"
    progress_path = Path(fixture["timing_root"]) / "progress.json"
    timing_path = Path(fixture["timing"])
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    execution["member_order"]["tasks"][0]["layer"] = "feature:attacker"
    execution["member_order"]["tasks"][0]["task_id"] = "attacker-task"
    progress["tasks"][0].update(execution["member_order"]["tasks"][0])
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["task_id"] = "attacker-task"
    _write_json(execution_path, execution)
    _write_json(progress_path, progress)
    timing["execution_manifest_sha256"] = sha256_file(execution_path)
    _write_json(timing_path, timing)
    assert _produce(fixture)["failure_codes"] == ["timing_task_order"]

    fixture = _fixture(tmp_path / "schedule")
    timing_path = Path(fixture["timing"])
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["resampling_branch_schedule"]["components"].pop("rockwell_macro_f1")
    _write_json(timing_path, timing)
    assert _produce(fixture)["failure_codes"] == ["timing_schedule_component_closure"]


@pytest.mark.parametrize("bad_flag", [0, 1, "false", None])
def test_scheduler_oom_flag_requires_a_json_boolean(tmp_path, bad_flag):
    fixture = _fixture(tmp_path)
    scheduler_path = Path(fixture["scheduler"])
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    scheduler["oom_killed"] = bad_flag
    _write_json(scheduler_path, scheduler)
    assert _produce(fixture)["failure_codes"] == ["scheduler_oom_flag"]


def test_footprint_mutation_between_validation_and_final_closure_fails(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    original = resource_module._snapshot_footprint
    calls = 0

    def changing_snapshot(timing_root, runtime_root):
        nonlocal calls
        calls += 1
        manifest, identities = original(timing_root, runtime_root)
        if calls >= 2:
            identities = (*identities, ("runtime", "late", 1, 2, 3, 4))
        return manifest, identities

    monkeypatch.setattr(resource_module, "_snapshot_footprint", changing_snapshot)
    assert _produce(fixture)["failure_codes"] == ["footprint_changed"]


def test_full_mode_checks_resource_gate_before_source_inventory(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    admission = _produce(fixture)
    admission_path = _write_admission(tmp_path, admission)
    paths = RepeatabilityPaths(
        raw_dir=tmp_path / "raw",
        speclib_dir=tmp_path / "speclib",
        validation_dir=tmp_path / "validation",
        output_dir=Path(fixture["timing_root"]),
        reference_dir=tmp_path / "reference",
    )
    monkeypatch.setattr(repeatability_module, "_repo_root", lambda _paths: ROOT)

    def forbidden_source_read(*_args, **_kwargs):
        raise AssertionError("scientific source inventory was reached")

    monkeypatch.setattr(repeatability_module, "_execution_source_inventory", forbidden_source_read)
    with pytest.raises(ResourceAdmissionError):
        run_repeatability_packet(
            paths,
            block_manifest=tmp_path / "block_manifest.json",
            resume=True,
            expected_timing_pilot_sha256=sha256_file(Path(fixture["timing"])),
            expected_resource_admission_sha256=sha256_file(admission_path),
            resource_admission_path=admission_path,
        )
