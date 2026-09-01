#!/usr/bin/env bash
#SBATCH --job-name=tanager-e6-v3
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
umask 077

RUN_ROOT="${TANAGER_BIGMEM_ROOT:-/scratch2/fs1/alexander.s.bradley/tanager-rocks-bigmem-20260810}"
REPO_ROOT="${RUN_ROOT}/Tanager/tanager-rocks"
MODE="${TANAGER_MODE:-}"
EXPECTED_SOURCE_MANIFEST_SHA256="${TANAGER_SOURCE_MANIFEST_SHA256:-}"
EXPECTED_TIMING_PILOT_SHA256="${TANAGER_TIMING_PILOT_SHA256:-}"
PYTHON_311="/home/alexander.s.bradley/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
UV_BIN="/home/alexander.s.bradley/.local/bin/uv"
SOURCE_MANIFEST="docs/m2_ensemble_bigmem_source_manifest_v3.sha256"
SOURCE_VERIFIER="scripts/verify_source_capsule.py"
SOURCE_MANIFEST_ENTRIES=49
OUTPUT_DIR="data/processed/ensemble_sensitivity_bigmem_20260812_v3"
RUNTIME_DIR="${RUN_ROOT}/runtime/v3"
UV_CACHE_DIR="${RUN_ROOT}/uv-cache"
JOB_RUNTIME_DIR="${RUNTIME_DIR}/jobs/${SLURM_JOB_ID:-manual}"
LOCK_PATH="${RUNTIME_DIR}/ensemble_v3.lock"
EXPECTED_DESIGN_SHA256="80247968c5d919f175cc833289353f85bf0acc85b9b1d8b47510fcbe895e3c62"
EXPECTED_MEMBERS_SHA256="3960b512e39fc81c4bfaac0297c5ac008c20be297f658d3d97ecf97a05d6b3dc"

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
if [[ ! -x "${PYTHON_311}" || ! -x "${UV_BIN}" ]]; then
  echo "required pinned CPython or uv executable is missing" >&2
  exit 2
fi
if [[ -z "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "TANAGER_SOURCE_MANIFEST_SHA256 is required" >&2
  exit 2
fi
if [[ "${MODE}" == "full" && -z "${EXPECTED_TIMING_PILOT_SHA256}" ]]; then
  echo "TANAGER_TIMING_PILOT_SHA256 is required in full mode" >&2
  exit 2
fi

if [[ "${TANAGER_LOCK_HELD:-}" != "1" ]]; then
  exec /usr/bin/env -u PYTHONHOME -u PYTHONPATH \
    "${PYTHON_311}" -I -S - "${BASH_SOURCE[0]}" "${LOCK_PATH}" <<'PY'
import errno
import fcntl
import os
import stat
import sys


def fail(reason: str) -> None:
    raise SystemExit(f"FAIL check=execution_lock reason={reason}")


def directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
    )


script = os.path.abspath(sys.argv[1])
lock_path = os.path.abspath(sys.argv[2])
parent_path, name = os.path.split(lock_path)
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
descriptors = [os.open(os.sep, flags)]
try:
    parent_parts = [part for part in parent_path.split(os.sep) if part]
    for index, part in enumerate(parent_parts):
        parent_fd = descriptors[-1]
        try:
            before = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if index != len(parent_parts) - 1:
                fail("missing_parent")
            try:
                os.mkdir(part, 0o700, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileExistsError:
                pass
            before = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
            fail("invalid_parent")
        descriptor = os.open(part, flags, dir_fd=parent_fd)
        if directory_identity(os.fstat(descriptor)) != directory_identity(before):
            os.close(descriptor)
            fail("parent_changed")
        descriptors.append(descriptor)
    parent_fd = descriptors[-1]
    open_flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            descriptor = os.open(
                name,
                open_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as error:
            if error.errno in {errno.EEXIST, errno.ELOOP}:
                fail("lock_race")
            raise
        opened = os.fstat(descriptor)
    else:
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != 0
        ):
            fail("invalid_lock_file")
        try:
            descriptor = os.open(name, open_flags, dir_fd=parent_fd)
        except OSError as error:
            if error.errno == errno.ELOOP:
                fail("lock_race")
            raise
        opened = os.fstat(descriptor)
        if file_identity(opened) != file_identity(before):
            os.close(descriptor)
            fail("lock_changed")
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_size != 0:
        os.close(descriptor)
        fail("invalid_open_lock")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        fail("lock_contended")
    rebound = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    after = os.fstat(descriptor)
    if file_identity(after) != file_identity(opened) or file_identity(rebound) != file_identity(
        opened
    ):
        os.close(descriptor)
        fail("lock_replaced")
    lock_transfer = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 20)
    parent_transfer = fcntl.fcntl(parent_fd, fcntl.F_DUPFD_CLOEXEC, 20)
    os.dup2(parent_transfer, 8, inheritable=True)
    os.dup2(lock_transfer, 9, inheritable=True)
    os.close(parent_transfer)
    os.close(lock_transfer)
    os.close(descriptor)
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["TANAGER_LOCK_HELD"] = "1"
    environment["TANAGER_LOCK_DEVICE"] = str(opened.st_dev)
    environment["TANAGER_LOCK_INODE"] = str(opened.st_ino)
    environment["TANAGER_LOCK_PARENT_DEVICE"] = str(os.fstat(8).st_dev)
    environment["TANAGER_LOCK_PARENT_INODE"] = str(os.fstat(8).st_ino)
    os.execve("/bin/bash", ["/bin/bash", script], environment)
finally:
    for descriptor in reversed(descriptors):
        os.close(descriptor)
PY
fi

if ! /usr/bin/env -u PYTHONHOME -u PYTHONPATH \
  "${PYTHON_311}" -I -S - \
  "${LOCK_PATH##*/}" \
  "${TANAGER_LOCK_DEVICE:-}" "${TANAGER_LOCK_INODE:-}" \
  "${TANAGER_LOCK_PARENT_DEVICE:-}" "${TANAGER_LOCK_PARENT_INODE:-}" <<'PY'
import fcntl
import os
import stat
import sys

name = sys.argv[1]
expected_device = int(sys.argv[2])
expected_inode = int(sys.argv[3])
expected_parent_device = int(sys.argv[4])
expected_parent_inode = int(sys.argv[5])
parent = os.fstat(8)
opened = os.fstat(9)
rebound = os.stat(name, dir_fd=8, follow_symlinks=False)
valid = (
    "/" not in name
    and stat.S_ISDIR(parent.st_mode)
    and (parent.st_dev, parent.st_ino) == (expected_parent_device, expected_parent_inode)
    and stat.S_ISREG(opened.st_mode)
    and opened.st_nlink == 1
    and opened.st_size == 0
    and (opened.st_dev, opened.st_ino) == (expected_device, expected_inode)
    and (
        rebound.st_dev,
        rebound.st_ino,
        rebound.st_mode,
        rebound.st_nlink,
        rebound.st_size,
    )
    == (
        opened.st_dev,
        opened.st_ino,
        opened.st_mode,
        opened.st_nlink,
        opened.st_size,
    )
)
if not valid:
    raise SystemExit("FAIL check=execution_lock reason=post_exec_rebind")
try:
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("FAIL check=execution_lock reason=post_exec_lock") from None
print(f"PASS check=execution_lock inode={expected_inode}")
PY
then
  exit 2
fi

if [[ -e "${JOB_RUNTIME_DIR}" || -L "${JOB_RUNTIME_DIR}" ]]; then
  echo "job runtime path already exists" >&2
  exit 2
fi
mkdir -p "${RUNTIME_DIR}" "${UV_CACHE_DIR}" "${JOB_RUNTIME_DIR}"
mkdir -p \
  "${JOB_RUNTIME_DIR}/home" \
  "${JOB_RUNTIME_DIR}/mplconfig" \
  "${JOB_RUNTIME_DIR}/tmp" \
  "${JOB_RUNTIME_DIR}/xdg-cache" \
  "${JOB_RUNTIME_DIR}/xdg-config"

export HOME="${JOB_RUNTIME_DIR}/home"
export MPLCONFIGDIR="${JOB_RUNTIME_DIR}/mplconfig"
export MPLBACKEND=Agg
export TMPDIR="${JOB_RUNTIME_DIR}/tmp"
export XDG_CACHE_HOME="${JOB_RUNTIME_DIR}/xdg-cache"
export XDG_CONFIG_HOME="${JOB_RUNTIME_DIR}/xdg-config"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export UV_CACHE_DIR
export LC_ALL=C
export LANG=C
unset PYTHONHOME PYTHONPATH

verify_source() {
  "${PYTHON_311}" "${SOURCE_VERIFIER}" \
    --manifest "${SOURCE_MANIFEST}" \
    --expected-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
    --expected-entry-count "${SOURCE_MANIFEST_ENTRIES}" \
    --project-root "${REPO_ROOT}"
}

require_regular_file() {
  local path="$1"
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "required regular, non-link file is missing: ${path}" >&2
    exit 2
  fi
}

require_digest() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(/usr/bin/sha256sum -- "${path}")"
  observed="${observed%% *}"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "SHA-256 mismatch: ${path}" >&2
    exit 2
  fi
}

require_exact_output_members() {
  local -a expected=("$@")
  local -a observed=()
  local path
  local name
  local wanted
  local allowed

  shopt -s dotglob nullglob
  for path in "${OUTPUT_DIR}"/*; do
    observed+=("${path##*/}")
  done
  shopt -u dotglob nullglob

  if [[ "${#observed[@]}" -ne "${#expected[@]}" ]]; then
    echo "unexpected output member count for mode ${MODE}" >&2
    exit 2
  fi
  for name in "${observed[@]}"; do
    allowed=0
    for wanted in "${expected[@]}"; do
      if [[ "${name}" == "${wanted}" ]]; then
        allowed=1
        break
      fi
    done
    if [[ "${allowed}" -ne 1 ]]; then
      echo "unexpected output member for mode ${MODE}" >&2
      exit 2
    fi
  done
}

require_timing_nonresult_contract() {
  "${PYTHON_311}" -c '
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
records = payload.get("records")
valid = (
    payload.get("schema_version") == "1.0"
    and payload.get("mode") == "timing_pilot_only"
    and payload.get("fit_count") == 4
    and isinstance(records, list)
    and len(records) == 4
    and all(row.get("scientific_outputs_retained") is False for row in records)
)
if not valid:
    raise SystemExit("invalid timing-pilot non-result contract")
' "${OUTPUT_DIR}/timing_pilot.json"
}

verify_source

"${UV_BIN}" sync \
  --frozen \
  --extra dev \
  --python "${PYTHON_311}"

verify_source

"${UV_BIN}" --version > "${JOB_RUNTIME_DIR}/${MODE}_uv_version.txt"
"${PYTHON_311}" --version > "${JOB_RUNTIME_DIR}/${MODE}_python_version.txt" 2>&1
/usr/bin/git rev-parse HEAD > "${JOB_RUNTIME_DIR}/${MODE}_git_revision.txt"
/usr/bin/sha256sum "${SOURCE_MANIFEST}" \
  > "${JOB_RUNTIME_DIR}/${MODE}_source_manifest.sha256"

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
    if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
      echo "clean design requires an absent output path" >&2
      exit 2
    fi
    ;;
  timing)
    if [[ ! -d "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
      echo "timing mode requires the admitted design directory" >&2
      exit 2
    fi
    require_regular_file "${OUTPUT_DIR}/design.json"
    require_regular_file "${OUTPUT_DIR}/members.csv"
    require_digest "${OUTPUT_DIR}/design.json" "${EXPECTED_DESIGN_SHA256}"
    require_digest "${OUTPUT_DIR}/members.csv" "${EXPECTED_MEMBERS_SHA256}"
    require_exact_output_members "design.json" "members.csv"
    if [[ -e "${OUTPUT_DIR}/timing_pilot.json" || -L "${OUTPUT_DIR}/timing_pilot.json" ]]; then
      echo "timing pilot destination already exists" >&2
      exit 2
    fi
    if [[ -e "${OUTPUT_DIR}/.score_cache" || -L "${OUTPUT_DIR}/.score_cache" ]]; then
      echo "timing mode refuses a pre-existing score cache" >&2
      exit 2
    fi
    ;;
  full)
    if [[ ! -d "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
      echo "full mode requires the admitted timing directory" >&2
      exit 2
    fi
    require_regular_file "${OUTPUT_DIR}/design.json"
    require_regular_file "${OUTPUT_DIR}/members.csv"
    require_regular_file "${OUTPUT_DIR}/timing_pilot.json"
    require_digest "${OUTPUT_DIR}/design.json" "${EXPECTED_DESIGN_SHA256}"
    require_digest "${OUTPUT_DIR}/members.csv" "${EXPECTED_MEMBERS_SHA256}"
    require_digest "${OUTPUT_DIR}/timing_pilot.json" "${EXPECTED_TIMING_PILOT_SHA256}"
    require_exact_output_members "design.json" "members.csv" "timing_pilot.json"
    if [[ -e "${OUTPUT_DIR}/.score_cache" || -L "${OUTPUT_DIR}/.score_cache" ]]; then
      echo "full mode refuses a pre-existing score cache" >&2
      exit 2
    fi
    ;;
esac

set +e
case "${MODE}" in
  design)
    .venv/bin/python scripts/run_ensemble_sensitivity.py \
      "${COMMON_ARGS[@]}" \
      --design-only
    analysis_status=$?
    ;;
  timing)
    .venv/bin/python scripts/run_ensemble_sensitivity.py \
      "${COMMON_ARGS[@]}" \
      --resume \
      --timing-pilot
    analysis_status=$?
    ;;
  full)
    .venv/bin/python scripts/run_ensemble_sensitivity.py \
      "${COMMON_ARGS[@]}" \
      --resume
    analysis_status=$?
    ;;
esac
set -e

post_source_status=0
verify_source || post_source_status=$?
if [[ "${analysis_status}" -ne 0 || "${post_source_status}" -ne 0 ]]; then
  echo "analysis or post-execution source verification failed" >&2
  exit 1
fi

case "${MODE}" in
  design)
    require_regular_file "${OUTPUT_DIR}/design.json"
    require_regular_file "${OUTPUT_DIR}/members.csv"
    require_digest "${OUTPUT_DIR}/design.json" "${EXPECTED_DESIGN_SHA256}"
    require_digest "${OUTPUT_DIR}/members.csv" "${EXPECTED_MEMBERS_SHA256}"
    require_exact_output_members "design.json" "members.csv"
    ;;
  timing)
    require_regular_file "${OUTPUT_DIR}/timing_pilot.json"
    require_exact_output_members "design.json" "members.csv" "timing_pilot.json"
    require_timing_nonresult_contract
    ;;
  full) ;;
esac
