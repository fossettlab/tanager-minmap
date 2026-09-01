#!/usr/bin/env bash
#SBATCH --job-name=tanager-repeat-v2
#SBATCH --account=compute2-alexander.s.bradley
#SBATCH --partition=general-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810/Tanager/tanager-rocks
#SBATCH --output=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810/slurm_logs/%x-%j.out
#SBATCH --error=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810/slurm_logs/%x-%j.err

set -euo pipefail

RUN_ROOT="${TANAGER_BIGMEM_ROOT:-/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810}"
REPO_ROOT="${RUN_ROOT}/Tanager/tanager-rocks"
MODE="${TANAGER_MODE:-}"
EXPECTED_SOURCE_MANIFEST_SHA256="${TANAGER_SOURCE_MANIFEST_SHA256:-}"
EXPECTED_TIMING_PILOT_SHA256="${TANAGER_TIMING_PILOT_SHA256:-}"
EXPECTED_RESOURCE_ADMISSION_SHA256="${TANAGER_RESOURCE_ADMISSION_SHA256:-}"
PYTHON_311="/home/alexander.s.bradley/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
OUTPUT_DIR="data/processed/repeatability_metric_contract_v2_20260811"
RESOURCE_ADMISSION_PATH="${TANAGER_RESOURCE_ADMISSION_PATH:-${OUTPUT_DIR}.resource_admission.json}"
RUNTIME_DIR="${RUN_ROOT}/runtime/repeatability_metric_contract_v2_20260811"
UV_CACHE_DIR="${RUN_ROOT}/uv-cache"

case "${MODE}" in
  timing | full) ;;
  *)
    echo "TANAGER_MODE must be one of: timing, full" >&2
    exit 2
    ;;
esac

if [[ "${PWD}" != "${REPO_ROOT}" ]]; then
  echo "unexpected working directory: ${PWD}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_311}" ]]; then
  echo "required CPython 3.11 interpreter is missing: ${PYTHON_311}" >&2
  exit 2
fi
if [[ -z "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "TANAGER_SOURCE_MANIFEST_SHA256 is required" >&2
  exit 2
fi

mkdir -p "${RUNTIME_DIR}" "${UV_CACHE_DIR}"
if ! command -v flock >/dev/null 2>&1; then
  echo "required flock executable is unavailable" >&2
  exit 2
fi
exec 9>"${RUNTIME_DIR}/wrapper.lock"
if ! flock -n 9; then
  echo "another timing/full wrapper already holds this output identity" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR

SOURCE_MANIFEST_LINE="$(sha256sum docs/m2_repeatability_bigmem_source_manifest.sha256)"
OBSERVED_SOURCE_MANIFEST_SHA256="${SOURCE_MANIFEST_LINE%% *}"
if [[ "${OBSERVED_SOURCE_MANIFEST_SHA256}" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "source-manifest SHA-256 differs from the submitted execution identity" >&2
  exit 2
fi
sha256sum -c docs/m2_repeatability_bigmem_source_manifest.sha256

uv sync \
  --frozen \
  --extra dev \
  --python "${PYTHON_311}"

uv --version > "${RUNTIME_DIR}/${MODE}_uv_version.txt"
"${PYTHON_311}" --version > "${RUNTIME_DIR}/${MODE}_python_version.txt" 2>&1
git rev-parse HEAD > "${RUNTIME_DIR}/${MODE}_git_revision.txt"
sha256sum docs/m2_repeatability_bigmem_source_manifest.sha256 \
  > "${RUNTIME_DIR}/${MODE}_source_manifest.sha256"

COMMON_ARGS=(
  --input-manifest docs/input_manifest.json
  --block-manifest data/processed/spatial_validation/block_manifest.json
  --transfer-thresholds data/processed/spatial_validation/transfer_thresholds.csv
  --spatial-summary data/processed/spatial_validation/summary.json
  --output-dir "${OUTPUT_DIR}"
  --workers 4
)

case "${MODE}" in
  timing)
    if [[ -n "${EXPECTED_TIMING_PILOT_SHA256}" ]]; then
      echo "timing mode must not receive TANAGER_TIMING_PILOT_SHA256" >&2
      exit 2
    fi
    if [[ -e "${OUTPUT_DIR}" ]]; then
      echo "timing mode requires a fresh output directory: ${OUTPUT_DIR}" >&2
      exit 2
    fi
    .venv/bin/python scripts/run_repeatability.py \
      "${COMMON_ARGS[@]}" \
      --timing-pilot
    ;;
  full)
    if [[ -z "${EXPECTED_TIMING_PILOT_SHA256}" ]]; then
      echo "full mode requires TANAGER_TIMING_PILOT_SHA256 from the verifier" >&2
      exit 2
    fi
    if [[ -z "${EXPECTED_RESOURCE_ADMISSION_SHA256}" ]]; then
      echo "full mode requires TANAGER_RESOURCE_ADMISSION_SHA256 from the verifier" >&2
      exit 2
    fi
    if [[ ! -f "${RESOURCE_ADMISSION_PATH}" || -L "${RESOURCE_ADMISSION_PATH}" ]]; then
      echo "full mode requires a regular reviewed resource-admission artifact" >&2
      exit 2
    fi
    if [[ ! -f "${OUTPUT_DIR}/timing_pilot.json" ]]; then
      echo "full mode requires the admitted timing-pilot artifact" >&2
      exit 2
    fi
    if [[ -e "${OUTPUT_DIR}/manifest.json" ]]; then
      echo "accepted repeatability manifest already exists; refusing overwrite" >&2
      exit 2
    fi
    TIMING_PILOT_LINE="$(sha256sum "${OUTPUT_DIR}/timing_pilot.json")"
    OBSERVED_TIMING_PILOT_SHA256="${TIMING_PILOT_LINE%% *}"
    if [[ "${OBSERVED_TIMING_PILOT_SHA256}" != "${EXPECTED_TIMING_PILOT_SHA256}" ]]; then
      echo "timing-pilot SHA-256 differs from the verifier-supplied digest" >&2
      exit 2
    fi
    RESOURCE_ADMISSION_LINE="$(sha256sum "${RESOURCE_ADMISSION_PATH}")"
    OBSERVED_RESOURCE_ADMISSION_SHA256="${RESOURCE_ADMISSION_LINE%% *}"
    if [[ "${OBSERVED_RESOURCE_ADMISSION_SHA256}" != "${EXPECTED_RESOURCE_ADMISSION_SHA256}" ]]; then
      echo "resource-admission SHA-256 differs from the verifier-supplied digest" >&2
      exit 2
    fi
    .venv/bin/python scripts/run_repeatability.py \
      "${COMMON_ARGS[@]}" \
      --resume \
      --expected-timing-pilot-sha256 "${EXPECTED_TIMING_PILOT_SHA256}" \
      --resource-admission "${RESOURCE_ADMISSION_PATH}" \
      --expected-resource-admission-sha256 "${EXPECTED_RESOURCE_ADMISSION_SHA256}"
    ;;
esac
