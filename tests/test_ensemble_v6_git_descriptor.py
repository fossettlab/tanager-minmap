"""Regression tests for the E6-v6 Git-descriptor lifetime repair."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5_WRAPPER = ROOT / "scripts" / "run_ensemble_bigmem_v5.sbatch"
V6_WRAPPER = ROOT / "scripts" / "run_ensemble_bigmem_v6.sbatch"

EXPECTED_DESIGN_SHA256 = "80247968c5d919f175cc833289353f85bf0acc85b9b1d8b47510fcbe895e3c62"
EXPECTED_MEMBERS_SHA256 = "3960b512e39fc81c4bfaac0297c5ac008c20be297f658d3d97ecf97a05d6b3dc"
EXPECTED_BUNDLE_SHA256 = "9caed40ffaa805632bddb282d324fb9fc0f856e4b6c6b437618ddd59c9c8cc6e"
EXPECTED_STATUS_SHA256 = "8d076a9b9f22351b1f788e672275d4b064a88d46f5d3be88e3c0dba60f344ef4"


def _payload(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _analysis_invocation(payload: str) -> str:
    start = payload.index("COMMON_ARGS=(")
    set_plus_e = payload.index("\nset +e\n", start)
    end = payload.index("\nset -e\n", set_plus_e) + len("\nset -e\n")
    return payload[start:end]


def test_v6_uses_wrapper_owned_procfs_path_for_python_grandchildren():
    payload = _payload(V6_WRAPPER)

    assert 'GIT_DESCRIPTOR_OWNER_PID="${BASHPID}"' in payload
    assert (
        'GIT_OBJECT_REPO_FOR_CHILD="/proc/${GIT_DESCRIPTOR_OWNER_PID}/fd/8/'
        'jobs/${SLURM_JOB_ID:-manual}/git-object-repo"'
    ) in payload
    assert (
        'export GIT_ALTERNATE_OBJECT_DIRECTORIES="${GIT_OBJECT_REPO_FOR_CHILD}/objects"' in payload
    )
    assert (
        'expected_alternate = f"/proc/{owner_pid}/fd/8/jobs/{job_id}/git-object-repo/objects"'
    ) in payload
    assert 'owner_rebound = os.stat(f"/proc/{owner_pid}/fd/8")' in payload


def test_v6_exercises_default_python_subprocess_closing_before_and_after_analysis():
    payload = _payload(V6_WRAPPER)
    function_start = payload.index("verify_python_git_provenance() {")
    function_end = payload.index("\n}\n\nrequire_exact_output_members", function_start)
    function = payload[function_start:function_end]

    assert function.count("subprocess.run(") == 3
    assert "pass_fds" not in function
    assert payload.count("\nverify_python_git_provenance\n") == 2
    first_check = payload.index("\nverify_python_git_provenance\n", function_end)
    analysis = payload.index("\nset +e\n", first_check)
    second_check = payload.index("\nverify_python_git_provenance\n", analysis)
    assert first_check < analysis < second_check


def test_v6_preserves_frozen_scientific_identity_and_uses_fresh_paths():
    payload = _payload(V6_WRAPPER)

    assert f'EXPECTED_DESIGN_SHA256="{EXPECTED_DESIGN_SHA256}"' in payload
    assert f'EXPECTED_MEMBERS_SHA256="{EXPECTED_MEMBERS_SHA256}"' in payload
    assert f'EXPECTED_GIT_BUNDLE_SHA256="{EXPECTED_BUNDLE_SHA256}"' in payload
    assert f'EXPECTED_GIT_STATUS_SHA256="{EXPECTED_STATUS_SHA256}"' in payload
    assert 'SOURCE_MANIFEST="docs/m2_ensemble_bigmem_source_manifest_v6.sha256"' in payload
    assert 'OUTPUT_DIR="data/processed/ensemble_sensitivity_bigmem_20260812_v6"' in payload
    assert 'RUNTIME_PATH="${RUN_ROOT}/runtime/v6"' in payload
    assert 'LOCK_PATH="${RUNTIME_PATH}/ensemble_v6.lock"' in payload
    assert "SOURCE_MANIFEST_ENTRIES=51" in payload


def test_v6_scientific_invocation_is_byte_identical_to_v5():
    assert _analysis_invocation(_payload(V6_WRAPPER)) == _analysis_invocation(_payload(V5_WRAPPER))
