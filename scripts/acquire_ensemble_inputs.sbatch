#!/usr/bin/env bash
#SBATCH --job-name=tanager-e6-inputs
#SBATCH --account=compute2-alexander.s.bradley
#SBATCH --partition=general-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --chdir=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810/Tanager/tanager-rocks
#SBATCH --output=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810/slurm_logs/%x-%j.out
#SBATCH --error=/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810/slurm_logs/%x-%j.err

set -euo pipefail

RUN_ROOT="${TANAGER_BIGMEM_ROOT:-/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810}"
REPO_ROOT="${RUN_ROOT}/Tanager/tanager-rocks"
PYTHON_311="/home/alexander.s.bradley/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
UV_CACHE_DIR="${RUN_ROOT}/uv-cache"

if [[ "${PWD}" != "${REPO_ROOT}" ]]; then
  echo "unexpected working directory: ${PWD}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_311}" ]]; then
  echo "required CPython 3.11 interpreter is missing: ${PYTHON_311}" >&2
  exit 2
fi

mkdir -p "${UV_CACHE_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR

sha256sum -c docs/m2_ensemble_bigmem_source_manifest.sha256

uv sync \
  --frozen \
  --extra dev \
  --python "${PYTHON_311}"

.venv/bin/python scripts/download_speclib.py
.venv/bin/python scripts/download_scenes.py --site all
.venv/bin/python scripts/download_reference.py --site goldfield
