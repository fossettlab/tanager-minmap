#!/usr/bin/env bash
#SBATCH --job-name=tanager-e6
#SBATCH --account=compute2-alexander.s.bradley
#SBATCH --partition=general-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810/Tanager/tanager-rocks
#SBATCH --output=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810/slurm_logs/%x-%j.out
#SBATCH --error=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810/slurm_logs/%x-%j.err

set -euo pipefail

RUN_ROOT="${TANAGER_BIGMEM_ROOT:-/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810}"
REPO_ROOT="${RUN_ROOT}/Tanager/tanager-rocks"
MODE="${TANAGER_MODE:-}"
EXPECTED_SOURCE_MANIFEST_SHA256="${TANAGER_SOURCE_MANIFEST_SHA256:-}"
PYTHON_311="/home/alexander.s.bradley/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
OUTPUT_DIR="data/processed/ensemble_sensitivity_bigmem_20260810"
RUNTIME_DIR="${RUN_ROOT}/runtime"
UV_CACHE_DIR="${RUN_ROOT}/uv-cache"

case "${MODE}" in
  design | timing | full) ;;
  *)
    echo "TANAGER_MODE must be one of: design, timing, full" >&2
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

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR

SOURCE_MANIFEST_LINE="$(sha256sum docs/m2_ensemble_bigmem_source_manifest.sha256)"
OBSERVED_SOURCE_MANIFEST_SHA256="${SOURCE_MANIFEST_LINE%% *}"
if [[ "${OBSERVED_SOURCE_MANIFEST_SHA256}" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "source-manifest SHA-256 differs from the submitted execution identity" >&2
  exit 2
fi
sha256sum -c docs/m2_ensemble_bigmem_source_manifest.sha256

uv sync \
  --frozen \
  --extra dev \
  --python "${PYTHON_311}"

uv --version > "${RUNTIME_DIR}/${MODE}_uv_version.txt"
"${PYTHON_311}" --version > "${RUNTIME_DIR}/${MODE}_python_version.txt" 2>&1
git rev-parse HEAD > "${RUNTIME_DIR}/${MODE}_git_revision.txt"
sha256sum docs/m2_ensemble_bigmem_source_manifest.sha256 \
  > "${RUNTIME_DIR}/${MODE}_source_manifest.sha256"

COMMON_ARGS=(
  --preregistration docs/m2_ensemble_sensitivity_preregistration.md
  --block-manifest data/processed/spatial_validation/block_manifest.json
  --output-dir "${OUTPUT_DIR}"
  --device cpu
  --batch-size 1
  --storage-layout disk
)

case "${MODE}" in
  design)
    .venv/bin/python scripts/run_ensemble_sensitivity.py \
      "${COMMON_ARGS[@]}" \
      --design-only
    ;;
  timing)
    .venv/bin/python scripts/run_ensemble_sensitivity.py \
      "${COMMON_ARGS[@]}" \
      --resume \
      --timing-pilot
    ;;
  full)
    .venv/bin/python scripts/run_ensemble_sensitivity.py \
      "${COMMON_ARGS[@]}" \
      --resume
    ;;
esac
