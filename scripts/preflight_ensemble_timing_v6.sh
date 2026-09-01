#!/usr/bin/env bash
# Endpoint-blind preflight for one separately authorized E6-v6 timing pilot.
#
# This control never submits a job and never parses scientific or timing
# payloads. Its sole argument is the expected SHA-256 of the admission-bundle
# manifest staged beside this script.

set -euo pipefail

CHECK_NAME=ensemble_timing_v6_preflight
RUN_ROOT=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810
REPO_ROOT="${RUN_ROOT}/Tanager/tanager-rocks"
OUTPUT_DIR="${REPO_ROOT}/data/processed/ensemble_sensitivity_bigmem_20260812_v6"
RUNTIME_DIR="${RUN_ROOT}/runtime/v6"
LOCK_PATH="${RUNTIME_DIR}/ensemble_v6.lock"
CONTROL_ROOT=/scratch2/fs1/alexander.s.bradley/tanager-rocks-e6-v6-timing-admission-controls-20260813T1520Z
BUNDLE_MANIFEST=docs/m2_ensemble_bigmem_v6_timing_admission_bundle_2026-08-13.sha256
PYTHON_311=/home/alexander.s.bradley/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11
SOURCE_SHA=3b7fc29124998d03c04fa7d194a89ff915ffa0c62260f380bef1b6da5f87a5e8
WRAPPER_SHA=d07b157e4c566054413348e22494e1e0b225943e0259264e799a8902d5ce806c
DESIGN_SHA=80247968c5d919f175cc833289353f85bf0acc85b9b1d8b47510fcbe895e3c62
MEMBERS_SHA=3960b512e39fc81c4bfaac0297c5ac008c20be297f658d3d97ecf97a05d6b3dc
PROPOSAL_SHA=590f8ff6575731bfd4e8a7f629e1773c1983f9b68b0b705635ad6ce72058500b
APPROVAL_SHA=8670eea477b02ea24405c3373adda51efc02bba4084f416ba09f34216cccb8a9
CONTROL_MANIFEST_SHA=80aaf3d1246ba2b1a1bf6668ad933bd707df26fe294975acff09747f5428d4b8
STAGING_MANIFEST_SHA=280ec55d498e87033919632b75cc6b957c06712f905cc911d2a92afdbbd14082

fail() {
  printf 'FAIL check=%s reason=%s\n' "${CHECK_NAME}" "$1" >&2
  exit 1
}

canonical_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

if [[ "$#" -ne 1 ]] || ! canonical_sha256 "$1"; then
  fail cli_arguments
fi
EXPECTED_BUNDLE_MANIFEST_SHA="$1"

printf 'UTC=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'HOST=%s\n' "$(hostname)"

sacct -X -n -P -j 2721612,2769429,2770281,2770462,2770489 \
  --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem

if ! queued_v6_jobs="$(
  squeue -h -u "${USER}" -n tanager-e6-v6 -o '%i|%j|%T'
)"; then
  fail scheduler_query
fi
if [[ -n "${queued_v6_jobs}" ]]; then
  fail active_or_queued_v6_job
fi
printf 'PASS active_or_queued_v6_jobs=0\n'

active_processes="$(
  ps -u "${USER}" -o pid=,ppid=,stat=,etime=,args= |
    awk '/run_ensemble_(bigmem_v6|sensitivity)/ && $0 !~ /awk/ {print}'
)"
if [[ -n "${active_processes}" ]]; then
  fail active_e6_process
fi
printf 'PASS active_e6_processes_on_login=0\n'

if [[ ! -d "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  fail output_directory
fi
mapfile -t output_members < <(
  find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -printf '%f|%y|%n\n' |
    LC_ALL=C sort
)
if [[ "${#output_members[@]}" -ne 2 ]]; then
  fail output_member_count
fi
if [[ "${output_members[0]}" != 'design.json|f|1' || \
      "${output_members[1]}" != 'members.csv|f|1' ]]; then
  fail output_member_identity
fi
if [[ -e "${OUTPUT_DIR}/timing_pilot.json" || \
      -L "${OUTPUT_DIR}/timing_pilot.json" || \
      -e "${OUTPUT_DIR}/.score_cache" || \
      -L "${OUTPUT_DIR}/.score_cache" ]]; then
  fail timing_destination_or_cache_present
fi
observed_design="$(sha256sum -- "${OUTPUT_DIR}/design.json")"
observed_design="${observed_design%% *}"
observed_members="$(sha256sum -- "${OUTPUT_DIR}/members.csv")"
observed_members="${observed_members%% *}"
if [[ "${observed_design}" != "${DESIGN_SHA}" || \
      "${observed_members}" != "${MEMBERS_SHA}" ]]; then
  fail design_or_members_sha256
fi
printf 'PASS output_members=2 timing_destination=absent score_cache=absent\n'
printf 'PASS design_sha256=%s members_sha256=%s\n' \
  "${observed_design}" "${observed_members}"

if [[ ! -d "${RUNTIME_DIR}" || -L "${RUNTIME_DIR}" ]]; then
  fail runtime_directory
fi
lock_stat="$(stat -Lc '%F|%h|%s|%i' -- "${LOCK_PATH}")"
if [[ "${lock_stat}" != 'regular empty file|1|0|15102171065245460978' ]]; then
  fail lock_identity
fi
if ! flock -n "${LOCK_PATH}" /bin/true; then
  fail lock_held
fi
mapfile -t runtime_jobs < <(
  find "${RUNTIME_DIR}/jobs" -mindepth 1 -maxdepth 1 -printf '%f|%y\n' |
    LC_ALL=C sort
)
if [[ "${#runtime_jobs[@]}" -ne 1 || \
      "${runtime_jobs[0]}" != '2770489|d' ]]; then
  fail runtime_jobs
fi
printf 'PASS lock=%s status=unheld runtime_job=2770489\n' "${lock_stat}"

if [[ ! -d "${CONTROL_ROOT}" || -L "${CONTROL_ROOT}" ]]; then
  fail control_root
fi
expected_controls=(
  'docs/m2_ensemble_bigmem_v6_timing_admission_bundle_2026-08-13.sha256|f|1'
  'docs/m2_ensemble_bigmem_v6_timing_control_review_2026-08-12.md|f|1'
  'docs/m2_ensemble_bigmem_v6_timing_control_staging_2026-08-13.sha256|f|1'
  'docs/m2_ensemble_bigmem_v6_timing_controls_2026-08-12.sha256|f|1'
  'docs/m2_ensemble_bigmem_v6_timing_option_a_approval_2026-08-13.md|f|1'
  'docs/m2_ensemble_bigmem_v6_timing_replacement_admission_packet_2026-08-13.md|f|1'
  'docs/m2_ensemble_bigmem_v7_timing_telemetry_amendment_proposal_2026-08-12.md|f|1'
  'docs|d|2'
  'scripts/capture_ensemble_timing_scheduler_receipt.py|f|1'
  'scripts/preflight_ensemble_timing_v6.sh|f|1'
  'scripts/verify_ensemble_timing_artifact.py|f|1'
  'scripts/verify_ensemble_timing_scheduler_receipt.py|f|1'
  'scripts|d|2'
  'tests/test_capture_ensemble_timing_scheduler_receipt.py|f|1'
  'tests/test_verify_ensemble_timing_artifact.py|f|1'
  'tests/test_verify_ensemble_timing_scheduler_receipt.py|f|1'
  'tests|d|2'
)
mapfile -t observed_controls < <(
  find "${CONTROL_ROOT}" -mindepth 1 -maxdepth 2 -printf '%P|%y|%n\n' |
    LC_ALL=C sort
)
if [[ "${#observed_controls[@]}" -ne "${#expected_controls[@]}" ]]; then
  fail control_inventory_count
fi
for index in "${!expected_controls[@]}"; do
  if [[ "${observed_controls[index]}" != "${expected_controls[index]}" ]]; then
    fail control_inventory_identity
  fi
done

cd "${CONTROL_ROOT}"
observed_bundle_manifest="$(shasum -a 256 "${BUNDLE_MANIFEST}")"
observed_bundle_manifest="${observed_bundle_manifest%% *}"
if [[ "${observed_bundle_manifest}" != "${EXPECTED_BUNDLE_MANIFEST_SHA}" ]]; then
  fail admission_bundle_manifest_sha256
fi
shasum -a 256 -c "${BUNDLE_MANIFEST}" >/dev/null || fail admission_bundle_members
shasum -a 256 -c \
  docs/m2_ensemble_bigmem_v6_timing_control_staging_2026-08-13.sha256 \
  >/dev/null || fail staging_manifest_members
shasum -a 256 -c \
  docs/m2_ensemble_bigmem_v6_timing_controls_2026-08-12.sha256 \
  >/dev/null || fail control_manifest_members

observed_staging_manifest="$(
  shasum -a 256 \
    docs/m2_ensemble_bigmem_v6_timing_control_staging_2026-08-13.sha256
)"
observed_staging_manifest="${observed_staging_manifest%% *}"
observed_control_manifest="$(
  shasum -a 256 docs/m2_ensemble_bigmem_v6_timing_controls_2026-08-12.sha256
)"
observed_control_manifest="${observed_control_manifest%% *}"
observed_proposal="$(
  shasum -a 256 \
    docs/m2_ensemble_bigmem_v7_timing_telemetry_amendment_proposal_2026-08-12.md
)"
observed_proposal="${observed_proposal%% *}"
observed_approval="$(
  shasum -a 256 \
    docs/m2_ensemble_bigmem_v6_timing_option_a_approval_2026-08-13.md
)"
observed_approval="${observed_approval%% *}"
if [[ "${observed_staging_manifest}" != "${STAGING_MANIFEST_SHA}" || \
      "${observed_control_manifest}" != "${CONTROL_MANIFEST_SHA}" || \
      "${observed_proposal}" != "${PROPOSAL_SHA}" || \
      "${observed_approval}" != "${APPROVAL_SHA}" ]]; then
  fail control_digest
fi
printf 'PASS admission_control_files=14 admission_control_directories=3\n'
printf 'PASS admission_bundle_manifest_sha256=%s\n' \
  "${observed_bundle_manifest}"
printf 'PASS proposal_sha256=%s approval_sha256=%s\n' \
  "${observed_proposal}" "${observed_approval}"

cd "${REPO_ROOT}"
"${PYTHON_311}" scripts/verify_source_capsule.py \
  --manifest docs/m2_ensemble_bigmem_source_manifest_v6.sha256 \
  --expected-manifest-sha256 "${SOURCE_SHA}" \
  --expected-entry-count 51 \
  --project-root "${REPO_ROOT}"

observed_wrapper="$(sha256sum scripts/run_ensemble_bigmem_v6.sbatch)"
observed_wrapper="${observed_wrapper%% *}"
observed_source_manifest="$(
  sha256sum docs/m2_ensemble_bigmem_source_manifest_v6.sha256
)"
observed_source_manifest="${observed_source_manifest%% *}"
if [[ "${observed_wrapper}" != "${WRAPPER_SHA}" || \
      "${observed_source_manifest}" != "${SOURCE_SHA}" ]]; then
  fail source_or_wrapper_digest
fi
printf 'PASS wrapper_sha256=%s source_manifest_sha256=%s\n' \
  "${observed_wrapper}" "${observed_source_manifest}"
printf 'PASS check=%s\n' "${CHECK_NAME}"
