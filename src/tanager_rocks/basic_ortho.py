"""Pre-result native/basic-to-ortho mapping and provenance controls.

This module implements only the schema, mapping, and exact-copy audit layer
frozen in ``docs/m1b_basic_ortho_sensitivity_preregistration.md``.  It never
computes a mineral, feature, MTMF, classification, or external-reference
endpoint.  Scientific fields produced elsewhere may be projected only through
the explicitly 2-D scalar interface in :func:`project_scalar_nearest`.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import resource
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import h5py
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coordinates
from scipy.spatial import cKDTree

from .quality import QA_ALLOWED_VALUES, QA_FIELDS

SCHEMA_VERSION = "1.0"
MANIFEST_TYPE = "m1b_basic_ortho_mapping"
SCIENTIFIC_EXECUTION_IDENTITY = "native-basic-ortho-mapping-contract-v1"
FROZEN_PREREGISTRATION_SHA256 = "d546eee8971fb0467b99775dc1f8d8c73ec18b1781116af13f192658fe95d1ba"
FROZEN_ACQUISITION_MANIFEST_SHA256 = (
    "6257ebb9a0c520df75be5cc2dd43cb0540c28b8df11f9ebd387246347b29a930"
)
FROZEN_ORTHO_MANIFEST_SHA256 = "50c2473ab34cd47440fb6cd583736463df17abced98da8002bb0f5c4e418e3f3"
FROZEN_SCENES = {
    "goldfield": "20240925_185504_87_4001",
    "bingham": "20250911_191523_58_4001",
}
MAPPING_ARTIFACT_NAMES = (
    "source_index.tif",
    "mapping_distance_m.tif",
    "source_multiplicity.tif",
    "mapping_status.tif",
)
TANAGER_SPEC_EDITABLE_LOGICAL_ROOT = "../tanager-spec"
TANAGER_SPEC_MODULE_FILES = {
    "tanager_spec": "../tanager-spec/src/tanager_spec/__init__.py",
    "tanager_spec.bands": "../tanager-spec/src/tanager_spec/bands.py",
    "tanager_spec.config": "../tanager-spec/src/tanager_spec/config.py",
    "tanager_spec.io": "../tanager-spec/src/tanager_spec/io.py",
    "tanager_spec.mask": "../tanager-spec/src/tanager_spec/mask.py",
    "tanager_spec.sample": "../tanager-spec/src/tanager_spec/sample.py",
    "tanager_spec.srf": "../tanager-spec/src/tanager_spec/srf.py",
    "tanager_spec.stac": "../tanager-spec/src/tanager_spec/stac.py",
}
TANAGER_SPEC_PACKAGE_DATA_FILES = frozenset(
    {
        "../tanager-spec/src/tanager_spec/data/S2A_SRF.csv",
        "../tanager-spec/src/tanager_spec/data/S2B_SRF.csv",
        "../tanager-spec/src/tanager_spec/data/SOURCE.md",
    }
)
GOVERNING_FILE_KEYS = frozenset(
    {
        "scripts/run_basic_ortho_sensitivity.py",
        "src/tanager_rocks/__init__.py",
        "src/tanager_rocks/basic_ortho.py",
        "src/tanager_rocks/config.py",
        "src/tanager_rocks/features.py",
        "src/tanager_rocks/quality.py",
        "src/tanager_rocks/speclib.py",
        "src/tanager_rocks/unmix.py",
        "src/tanager_rocks/viz.py",
        *TANAGER_SPEC_MODULE_FILES.values(),
        *TANAGER_SPEC_PACKAGE_DATA_FILES,
    }
)
RESOURCE_PILOT_EXECUTION_IDENTITY = "resource-pilot-non-promotable-v1"
RESOURCE_PILOT_BRANCH_ASSETS = {
    "B": "basic_sr_hdf5",
    "O": "ortho_sr_hdf5",
}
RESOURCE_PILOT_DEFAULT_SITE = next(iter(FROZEN_SCENES))
RESOURCE_PILOT_DEFAULT_BRANCH = next(iter(RESOURCE_PILOT_BRANCH_ASSETS))

# Platform ABI values from Darwin sys/fcntl.h + sys/stdio.h and Linux fcntl.h.
_DARWIN_AT_FDCWD = -2
_DARWIN_RENAME_EXCL = 0x00000004
_DARWIN_RENAME_NOFOLLOW_ANY = 0x00000010
_LINUX_AT_FDCWD = -100
_LINUX_RENAME_NOREPLACE = 1

_BASIC_DATA_GROUPS = (
    "HDFEOS/SWATHS/HYP/Data Fields",
    "HDFEOS/SWATHS/HYP/Data_Fields",
)
_BASIC_GEO_GROUPS = (
    "HDFEOS/SWATHS/HYP/Geolocation Fields",
    "HDFEOS/SWATHS/HYP/Geolocation_Fields",
)
_ORTHO_DATA_GROUPS = (
    "HDFEOS/GRIDS/HYP/Data Fields",
    "HDFEOS/GRIDS/HYP/Data_Fields",
)
_STRUCT_METADATA_PATH = "HDFEOS INFORMATION/StructMetadata.0"
_REFLECTANCE_FIELD = "surface_reflectance"

SOURCE_INVALID_GEOLOCATION = np.uint8(1)
SOURCE_UNUSED = np.uint8(2)
SOURCE_USED = np.uint8(3)
TARGET_ORTHO_QA_INVALID = np.uint8(1)
TARGET_NO_GEOLOCATED_SOURCE = np.uint8(2)
TARGET_BASIC_QA_INVALID = np.uint8(3)
TARGET_MAPPED = np.uint8(4)


class ProtocolError(ValueError):
    """Raised when an input or action departs from the frozen M1b protocol."""


class CleanupResidueError(ProtocolError):
    """Raised when an owned bundle cannot be removed without risking a replacement."""


@dataclass(frozen=True)
class DatasetSchema:
    """Metadata-only description of one HDF5 dataset."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    dimension_labels: tuple[str, ...]
    attributes: dict[str, dict[str, object]]


@dataclass(frozen=True)
class ProductSchema:
    """Validated schema of one native/basic or ortho SR product."""

    product_geometry: str
    path: str
    reflectance: DatasetSchema
    spectral_axis: int
    spatial_axes: tuple[int, int]
    spatial_shape: tuple[int, int]
    wavelength_count: int
    wavelengths_sha256: str
    good_wavelengths_sha256: str
    retained_product_bands: int
    geolocation: dict[str, DatasetSchema]
    qa: dict[str, DatasetSchema]
    axis_evidence: str


@dataclass(frozen=True)
class FrozenSceneInput:
    """Hash-verified basic/ortho pair and its frozen STAC geometry metadata."""

    site: str
    scene_id: str
    basic_path: Path
    basic_size_bytes: int
    basic_sha256: str
    ortho_path: Path
    ortho_size_bytes: int
    ortho_sha256: str
    basic_stac_shape: tuple[int, int]
    basic_stac_crs: str
    basic_stac_resolution_m: float
    ortho_stac_shape: tuple[int, int]
    ortho_stac_crs: str
    ortho_stac_resolution_m: float


@dataclass(frozen=True)
class ValidatedInputs:
    """Verified manifest identities and the two frozen scene pairs."""

    acquisition_manifest_path: Path
    acquisition_manifest_sha256: str
    ortho_manifest_path: Path
    ortho_manifest_sha256: str
    scenes: tuple[FrozenSceneInput, ...]


@dataclass(frozen=True)
class OrthoGrid:
    """Projected target grid used only for nearest-cell scalar assignment."""

    shape: tuple[int, int]
    transform: Affine
    crs: CRS

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ProtocolError(f"invalid ortho grid shape: {self.shape}")
        if self.transform.b != 0.0 or self.transform.d != 0.0:
            raise ProtocolError("rotated/sheared target grids are not supported")
        if self.transform.a == 0.0 or self.transform.e == 0.0:
            raise ProtocolError("target grid has a zero pixel dimension")
        if not self.crs.is_projected:
            raise ProtocolError("mapping distance requires a projected ortho CRS")
        unit_name, unit_factor = self.crs.linear_units_factor
        if unit_name not in {"metre", "meter"} or unit_factor != 1.0:
            raise ProtocolError(f"mapping distance must be measured in metres, got {unit_name!r}")


@dataclass(frozen=True)
class MappingCounts:
    """Source-use, repeated-source, and target no-call accounting."""

    total_source_samples: int
    invalid_qa_source_samples: int
    invalid_geolocation_source_samples: int
    used_source_samples: int
    unused_source_samples: int
    sources_with_multiple_target_cells: int
    duplicate_target_assignments: int
    total_target_cells: int
    invalid_qa_target_cells: int
    basic_qa_no_call_target_cells: int
    no_geolocated_source_target_cells: int
    mapped_target_cells: int
    unmapped_target_cells: int


@dataclass(frozen=True)
class NativeToOrthoMapping:
    """Traceable nearest-source mapping for every eligible ortho target cell."""

    source_row: np.ndarray
    source_col: np.ndarray
    source_flat_index: np.ndarray
    source_multiplicity: np.ndarray
    mapping_distance_m: np.ndarray
    target_status: np.ndarray
    source_status: np.ndarray
    target_count_per_source: np.ndarray
    counts: MappingCounts

    def __post_init__(self) -> None:
        target_shape = self.source_row.shape
        for name in (
            "source_col",
            "source_flat_index",
            "source_multiplicity",
            "mapping_distance_m",
            "target_status",
        ):
            if getattr(self, name).shape != target_shape:
                raise ProtocolError(f"{name} is not aligned to the target grid")
        if self.target_count_per_source.shape != self.source_status.shape:
            raise ProtocolError("target_count_per_source is not aligned to the native grid")


@dataclass(frozen=True)
class SpectrumCopyAudit:
    """Exact finite-vector matching counts; no tolerance or interpolation."""

    retained_bands: int
    valid_basic_spectra: int
    valid_ortho_spectra: int
    ortho_exact_match_to_any_basic: int
    ortho_exact_match_to_any_basic_fraction: float | None
    mapped_valid_ortho_spectra: int
    mapped_exact_match_to_selected_basic: int
    mapped_exact_match_to_selected_basic_fraction: float | None


@dataclass(frozen=True)
class ResourcePilotTelemetry:
    """Resource measurements from loading one declared reflectance branch."""

    wall_seconds: float
    user_cpu_seconds: float
    system_cpu_seconds: float
    process_max_rss_before_bytes: int
    process_max_rss_after_bytes: int
    loaded_array_bytes: int


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a bounded-memory SHA-256 digest for a regular file."""
    if not path.is_file():
        raise FileNotFoundError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the source metadata required to detect mutation or replacement."""
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _read_bound_validation_bytes(path: Path, *, label: str) -> tuple[bytes, tuple[int, ...]]:
    """Read one regular file once and bind its bytes to full pre/post metadata."""
    file_fd, initial_info = _open_regular_nofollow(path, label=label)
    try:
        with os.fdopen(file_fd, "rb") as handle:
            payload = handle.read()
            final_info = os.fstat(handle.fileno())
    except OSError as error:
        raise ProtocolError(f"cannot read {label}: {path}") from error
    initial = _stat_fingerprint(initial_info)
    final = _stat_fingerprint(final_info)
    if initial != final or len(payload) != initial_info.st_size:
        raise ProtocolError(f"{label} changed while its bytes were sealed: {path}")
    return payload, final


def _require_validation_source_unchanged(
    path: Path,
    fingerprint: tuple[int, ...] | None,
    *,
    label: str,
) -> None:
    """Reject a path replacement or in-place mutation after sealed bytes were parsed."""
    if fingerprint is None:
        return
    file_fd, info = _open_regular_nofollow(path, label=f"{label} post-parse check")
    os.close(file_fd)
    if _stat_fingerprint(info) != fingerprint:
        raise ProtocolError(f"{label} changed between sealing and parsing: {path}")


def _validation_bytes(
    path: Path,
    *,
    label: str,
    snapshot_bytes: bytes | None,
) -> tuple[bytes, tuple[int, ...] | None]:
    if snapshot_bytes is not None:
        return bytes(snapshot_bytes), None
    return _read_bound_validation_bytes(path, label=label)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Affine):
        return list(value)[:6]
    if isinstance(value, CRS):
        return value.to_string()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def strict_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write standards-compliant JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_directory_contents_fd(directory_fd: int, *, label: str) -> None:
    """Delete one already-open directory tree without following symlinks."""
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise CleanupResidueError(
                    f"{label} entry changed during descriptor-safe cleanup: {entry.name}"
                ) from error
            if stat.S_ISDIR(info.st_mode):
                try:
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError as error:
                    raise CleanupResidueError(
                        f"{label} directory entry cannot be opened safely: {entry.name}"
                    ) from error
                try:
                    child_info = os.fstat(child_fd)
                    if (child_info.st_dev, child_info.st_ino) != (info.st_dev, info.st_ino):
                        raise CleanupResidueError(
                            f"{label} directory entry was replaced during cleanup: {entry.name}"
                        )
                    _remove_directory_contents_fd(child_fd, label=label)
                finally:
                    os.close(child_fd)
                try:
                    os.rmdir(entry.name, dir_fd=directory_fd)
                except OSError as error:
                    raise CleanupResidueError(
                        f"{label} directory residue could not be removed: {entry.name}"
                    ) from error
            else:
                try:
                    os.unlink(entry.name, dir_fd=directory_fd)
                except OSError as error:
                    raise CleanupResidueError(
                        f"{label} file residue could not be removed: {entry.name}"
                    ) from error


def _restore_mismatched_quarantine(
    quarantine_target: Path,
    original_path: Path,
    quarantine_root: Path,
    *,
    label: str,
) -> None:
    """Restore a detached replacement when possible and always report owned residue."""
    try:
        _rename_directory_noreplace(quarantine_target, original_path)
        os.rmdir(quarantine_root)
    except BaseException as restore_error:
        raise CleanupResidueError(
            f"{label} pathname held a replacement; replacement remains quarantined at "
            f"{quarantine_target} and owned residue requires inspection"
        ) from restore_error
    raise CleanupResidueError(
        f"{label} pathname held a replacement; replacement was restored and owned residue "
        "requires inspection"
    )


def _quarantine_and_remove_directory(
    path: Path,
    identity: tuple[int, int],
    *,
    label: str,
) -> None:
    """Atomically detach an owned directory, verify its inode, then remove by descriptor."""
    quarantine_root = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.cleanup.", suffix=".quarantine", dir=path.parent)
    )
    quarantine_identity = _directory_identity(quarantine_root)
    quarantine_target = quarantine_root / "detached"
    try:
        _rename_directory_noreplace(path, quarantine_target)
    except BaseException as error:
        try:
            os.rmdir(quarantine_root)
        except OSError:
            pass
        raise CleanupResidueError(
            f"{label} is missing or could not be atomically quarantined; "
            f"owned residue requires inspection: {path}"
        ) from error

    try:
        moved_identity = _directory_identity(quarantine_target)
    except (FileNotFoundError, ProtocolError) as error:
        raise CleanupResidueError(
            f"{label} disappeared after quarantine; owned residue requires inspection: "
            f"{quarantine_target}"
        ) from error
    if moved_identity != identity:
        _restore_mismatched_quarantine(
            quarantine_target,
            path,
            quarantine_root,
            label=label,
        )

    quarantine_fd = os.open(
        quarantine_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    detached_fd: int | None = None
    try:
        if (os.fstat(quarantine_fd).st_dev, os.fstat(quarantine_fd).st_ino) != quarantine_identity:
            raise CleanupResidueError(f"{label} quarantine root identity changed")
        detached_fd = os.open(
            quarantine_target.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=quarantine_fd,
        )
        detached_info = os.fstat(detached_fd)
        if (detached_info.st_dev, detached_info.st_ino) != identity:
            raise CleanupResidueError(f"{label} detached inode identity changed")
        _remove_directory_contents_fd(detached_fd, label=label)
        final_info = os.fstat(detached_fd)
        if (final_info.st_dev, final_info.st_ino) != identity:
            raise CleanupResidueError(f"{label} detached inode changed during cleanup")
    except OSError as error:
        raise CleanupResidueError(
            f"{label} could not be removed safely; residue remains in {quarantine_root}"
        ) from error
    finally:
        if detached_fd is not None:
            os.close(detached_fd)
        os.close(quarantine_fd)

    try:
        os.rmdir(quarantine_target)
        _require_exact(
            f"{label} quarantine identity",
            _directory_identity(quarantine_root),
            quarantine_identity,
        )
        os.rmdir(quarantine_root)
    except (OSError, ProtocolError) as error:
        raise CleanupResidueError(
            f"{label} cleanup left explicit quarantine residue: {quarantine_root}"
        ) from error


def _cleanup_staging_directory(path: Path, identity: tuple[int, int]) -> None:
    """Atomically detach and safely remove the exact staging directory we created."""
    _quarantine_and_remove_directory(path, identity, label="owned staging directory")


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ProtocolError(f"bundle root is not a regular directory: {path}")
    return info.st_dev, info.st_ino


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing any existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            _DARWIN_AT_FDCWD,
            source_bytes,
            _DARWIN_AT_FDCWD,
            destination_bytes,
            _DARWIN_RENAME_EXCL | _DARWIN_RENAME_NOFOLLOW_ANY,
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise ProtocolError(
                "atomic no-replace rename is unavailable on this platform"
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            _LINUX_AT_FDCWD,
            source_bytes,
            _LINUX_AT_FDCWD,
            destination_bytes,
            _LINUX_RENAME_NOREPLACE,
        )
    else:
        raise ProtocolError("atomic no-replace directory promotion is unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ProtocolError(f"refusing to replace existing accepted bundle: {destination}")
    raise OSError(error_number, os.strerror(error_number), destination)


def _remove_owned_published_bundle(path: Path, identity: tuple[int, int]) -> None:
    """Atomically detach and remove the exact failed publication we promoted."""
    _quarantine_and_remove_directory(path, identity, label="failed published bundle")


def atomic_write_run_bundle(
    output_root: Path,
    run_id: str,
    *,
    writer: Callable[[Path], Path],
    verifier: Callable[[Path], None],
) -> Path:
    """Build and verify a complete run before publishing its identity directory.

    The writer receives an attempt-specific hidden sibling directory and returns
    the final manifest path inside it.  The verifier must validate the complete
    staged bundle.  Only then is the staging directory atomically renamed to the
    run identity.  Any failure removes the staging directory and leaves no final
    run path.
    """
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ProtocolError(f"unsafe run identity: {run_id!r}")
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_root = output_root.resolve()
    if not resolved_root.is_dir():
        raise ProtocolError(f"bundle output root is not a directory: {output_root}")
    final_root = resolved_root / run_id
    if final_root.exists() or final_root.is_symlink():
        raise ProtocolError(f"refusing to replace existing accepted bundle: {final_root}")

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".staging", dir=resolved_root)
    )
    staging_identity = _directory_identity(staging_root)
    published = False
    try:
        staged_manifest = writer(staging_root)
        try:
            manifest_relative = staged_manifest.relative_to(staging_root)
        except ValueError as error:
            raise ProtocolError("bundle writer returned a manifest outside staging") from error
        verifier(staging_root)
        if final_root.exists() or final_root.is_symlink():
            raise ProtocolError(f"refusing to replace existing accepted bundle: {final_root}")
        _rename_directory_noreplace(staging_root, final_root)
        published = True
        try:
            _require_exact(
                "promoted bundle directory identity",
                _directory_identity(final_root),
                staging_identity,
            )
            verifier(final_root)
        except BaseException:
            try:
                _remove_owned_published_bundle(final_root, staging_identity)
            except BaseException as cleanup_error:
                raise CleanupResidueError(
                    f"post-promotion verification and fail-closed removal both failed: {final_root}"
                ) from cleanup_error
            raise
        return final_root / manifest_relative
    finally:
        if not published:
            _cleanup_staging_directory(staging_root, staging_identity)


def _bundle_file_set(bundle_root: Path) -> set[str]:
    files: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError(f"bundle entries must not be symlinks: {path}")
        if path.is_file():
            files.add(path.relative_to(bundle_root).as_posix())
        elif not path.is_dir():
            raise ProtocolError(f"bundle entry is not a regular file or directory: {path}")
    return files


def _require_bundle_file(bundle_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ProtocolError(f"bundle path is not safely relative: {relative_path!r}")
    path = bundle_root / candidate
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"expected regular bundle file is missing: {relative_path}")
    try:
        path.resolve().relative_to(bundle_root.resolve())
    except ValueError as error:
        raise ProtocolError(f"bundle file escapes staging: {relative_path!r}") from error
    return path


def validate_protocol_file(
    path: Path,
    *,
    expected_sha256: str = FROZEN_PREREGISTRATION_SHA256,
    snapshot_bytes: bytes | None = None,
) -> str:
    """Require the exact preregistered protocol bytes."""
    payload, fingerprint = _validation_bytes(
        path,
        label="M1b preregistration",
        snapshot_bytes=snapshot_bytes,
    )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ProtocolError(
            f"M1b preregistration hash mismatch: expected {expected_sha256}, observed {observed}"
        )
    _require_validation_source_unchanged(path, fingerprint, label="M1b preregistration")
    return observed


def _resolve_repo_path(root: Path, logical_path: str) -> Path:
    if not logical_path or Path(logical_path).is_absolute():
        raise ProtocolError(f"input path must be repository-relative: {logical_path!r}")
    root = root.resolve()
    resolved = (root / logical_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ProtocolError(f"input path escapes the repository: {logical_path!r}") from error
    if resolved.name.endswith(".part"):
        raise ProtocolError(f"partial download cannot be used: {logical_path}")
    return resolved


def _require_exact(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ProtocolError(f"{label}: expected {expected!r}, observed {observed!r}")


def _require_object_fields(value: Any, *, label: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    _require_exact(f"{label} fields", set(value), fields)
    return value


def validate_governing_files(
    value: Any,
    *,
    label: str = "governing files",
) -> dict[str, str]:
    """Require the exact local runtime file set and canonical SHA-256 digests."""
    records = _require_object_fields(value, label=label, fields=set(GOVERNING_FILE_KEYS))
    for logical_path, digest in records.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ProtocolError(f"{label} {logical_path} must be a lowercase SHA-256 digest")
    return {key: records[key] for key in sorted(records)}


def tanager_spec_dependency_trust(
    governing_files: Mapping[str, str],
) -> dict[str, Any]:
    """Build the deterministic hash-bound record for the editable sibling closure."""
    governing = validate_governing_files(governing_files)
    python_source_files = {
        logical_path: governing[logical_path]
        for logical_path in sorted(TANAGER_SPEC_MODULE_FILES.values())
    }
    package_data_files = {
        logical_path: governing[logical_path]
        for logical_path in sorted(TANAGER_SPEC_PACKAGE_DATA_FILES)
    }
    inventory_payload = {
        "python_source_files": python_source_files,
        "package_data_files": package_data_files,
        "module_origins": dict(sorted(TANAGER_SPEC_MODULE_FILES.items())),
    }
    trust = {
        "tanager_spec": {
            "classification": "captured_hash_bound_editable_dependency",
            "hash_bound": True,
            "editable_root_logical_path": TANAGER_SPEC_EDITABLE_LOGICAL_ROOT,
            **inventory_payload,
            "inventory_sha256": execution_identity(inventory_payload),
        }
    }
    return _validate_residual_dependency_trust(trust, governing_files=governing)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"manifest root must be an object: {path}")
    return value


def _load_json_bytes(payload: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"manifest root must be an object: {path}")
    return value


def _verify_file_record(path: Path, *, size_bytes: Any, sha256: Any) -> tuple[int, str]:
    file_fd, initial_info = _open_regular_nofollow(path, label="frozen HDF5 input")
    digest = hashlib.sha256()
    copied_bytes = 0
    with os.fdopen(file_fd, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            copied_bytes += len(chunk)
        final_info = os.fstat(handle.fileno())
    if _stat_fingerprint(initial_info) != _stat_fingerprint(final_info):
        raise ProtocolError(f"frozen HDF5 input changed while hashing: {path}")
    _require_exact(f"{path} byte size", copied_bytes, int(size_bytes))
    observed_hash = digest.hexdigest()
    _require_exact(f"{path} SHA-256", observed_hash, str(sha256))
    check_fd, current_info = _open_regular_nofollow(path, label="frozen HDF5 post-hash check")
    os.close(check_fd)
    if _stat_fingerprint(current_info) != _stat_fingerprint(final_info):
        raise ProtocolError(f"frozen HDF5 input changed after hashing: {path}")
    return copied_bytes, observed_hash


def validate_frozen_inputs(
    acquisition_manifest: Path,
    ortho_manifest: Path,
    *,
    root: Path,
    expected_acquisition_sha256: str = FROZEN_ACQUISITION_MANIFEST_SHA256,
    expected_ortho_sha256: str = FROZEN_ORTHO_MANIFEST_SHA256,
    verify_files: bool = True,
    acquisition_snapshot_bytes: bytes | None = None,
    ortho_snapshot_bytes: bytes | None = None,
) -> ValidatedInputs:
    """Validate both manifests and the exact two basic/ortho input pairs.

    ``verify_files=False`` is reserved for callers that separately bind every
    referenced HDF5 input to a descriptor snapshot and verify its size and hash.
    """
    acquisition_bytes, acquisition_fingerprint = _validation_bytes(
        acquisition_manifest,
        label="acquisition manifest",
        snapshot_bytes=acquisition_snapshot_bytes,
    )
    acquisition_hash = hashlib.sha256(acquisition_bytes).hexdigest()
    _require_exact("acquisition manifest SHA-256", acquisition_hash, expected_acquisition_sha256)
    acquisition = _load_json_bytes(acquisition_bytes, path=acquisition_manifest)
    _require_exact("acquisition manifest schema", acquisition.get("schema_version"), "1.0")
    protocol = acquisition.get("preregistration")
    if not isinstance(protocol, dict):
        raise ProtocolError("acquisition manifest lacks preregistration identity")
    _require_exact(
        "acquisition preregistration SHA-256",
        protocol.get("sha256"),
        FROZEN_PREREGISTRATION_SHA256,
    )
    if acquisition.get("scientific_endpoint_values_inspected") is not False:
        raise ProtocolError("acquisition manifest does not preserve the pre-result boundary")

    assets = acquisition.get("assets")
    if not isinstance(assets, list) or len(assets) != len(FROZEN_SCENES):
        raise ProtocolError("acquisition manifest must contain exactly the two frozen scenes")
    by_site: dict[str, dict[str, Any]] = {}
    for item in assets:
        if not isinstance(item, dict):
            raise ProtocolError("acquisition asset record is not an object")
        site = str(item.get("site", ""))
        if site in by_site:
            raise ProtocolError(f"duplicate acquisition record for {site!r}")
        by_site[site] = item
    _require_exact("acquisition sites", set(by_site), set(FROZEN_SCENES))

    ortho_bytes, ortho_fingerprint = _validation_bytes(
        ortho_manifest,
        label="ortho manifest",
        snapshot_bytes=ortho_snapshot_bytes,
    )
    ortho_hash = hashlib.sha256(ortho_bytes).hexdigest()
    _require_exact("ortho manifest SHA-256", ortho_hash, expected_ortho_sha256)
    ortho = _load_json_bytes(ortho_bytes, path=ortho_manifest)
    _require_exact("ortho manifest hash algorithm", ortho.get("hash_algorithm"), "sha256")
    ortho_records = ortho.get("inputs")
    if not isinstance(ortho_records, list):
        raise ProtocolError("ortho manifest has no input records")
    ortho_by_path: dict[str, dict[str, Any]] = {}
    for item in ortho_records:
        if not isinstance(item, dict):
            raise ProtocolError("ortho input record is not an object")
        logical_path = str(item.get("logical_path", ""))
        if not logical_path or logical_path in ortho_by_path:
            raise ProtocolError(f"ortho manifest path is empty or duplicated: {logical_path!r}")
        ortho_by_path[logical_path] = item

    frozen: list[FrozenSceneInput] = []
    for site, scene_id in FROZEN_SCENES.items():
        item = by_site[site]
        _require_exact(f"{site} scene", item.get("scene_id"), scene_id)
        _require_exact(f"{site} asset key", item.get("asset_key"), "basic_sr_hdf5")
        _require_exact(f"{site} basic geolocation CRS", item.get("stac_proj_code"), "EPSG:4326")
        basic_path = _resolve_repo_path(root, str(item.get("local_path", "")))
        ortho_logical = f"data/raw/{scene_id}_ortho_sr_hdf5.h5"
        if ortho_logical not in ortho_by_path:
            raise ProtocolError(f"ortho manifest lacks {ortho_logical}")
        ortho_item = ortho_by_path[ortho_logical]
        ortho_path = _resolve_repo_path(root, ortho_logical)

        if verify_files:
            basic_size, basic_hash = _verify_file_record(
                basic_path,
                size_bytes=item.get("content_length"),
                sha256=item.get("sha256"),
            )
            ortho_size, ortho_file_hash = _verify_file_record(
                ortho_path,
                size_bytes=ortho_item.get("size_bytes"),
                sha256=ortho_item.get("sha256"),
            )
        else:
            basic_size = int(item.get("content_length"))
            basic_hash = str(item.get("sha256"))
            ortho_size = int(ortho_item.get("size_bytes"))
            ortho_file_hash = str(ortho_item.get("sha256"))

        basic_shape = tuple(int(value) for value in item.get("stac_proj_shape", []))
        ortho_shape = tuple(int(value) for value in item.get("paired_ortho_proj_shape", []))
        if len(basic_shape) != 2 or len(ortho_shape) != 2:
            raise ProtocolError(f"{site} manifest geometry must contain two-dimensional shapes")
        basic_resolution = float(item.get("stac_raster_spatial_resolution_m"))
        ortho_resolution = float(item.get("paired_ortho_raster_spatial_resolution_m"))
        if basic_resolution <= 0.0 or ortho_resolution <= 0.0:
            raise ProtocolError(f"{site} manifest resolutions must be positive")
        frozen.append(
            FrozenSceneInput(
                site=site,
                scene_id=scene_id,
                basic_path=basic_path,
                basic_size_bytes=basic_size,
                basic_sha256=basic_hash,
                ortho_path=ortho_path,
                ortho_size_bytes=ortho_size,
                ortho_sha256=ortho_file_hash,
                basic_stac_shape=basic_shape,
                basic_stac_crs=str(item.get("stac_proj_code")),
                basic_stac_resolution_m=basic_resolution,
                ortho_stac_shape=ortho_shape,
                ortho_stac_crs=str(item.get("paired_ortho_proj_code")),
                ortho_stac_resolution_m=ortho_resolution,
            )
        )
    validated = ValidatedInputs(
        acquisition_manifest_path=acquisition_manifest,
        acquisition_manifest_sha256=acquisition_hash,
        ortho_manifest_path=ortho_manifest,
        ortho_manifest_sha256=ortho_hash,
        scenes=tuple(frozen),
    )
    _require_validation_source_unchanged(
        acquisition_manifest,
        acquisition_fingerprint,
        label="acquisition manifest",
    )
    _require_validation_source_unchanged(
        ortho_manifest,
        ortho_fingerprint,
        label="ortho manifest",
    )
    return validated


def _attribute_schema(dataset: h5py.Dataset) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name in sorted(dataset.attrs):
        value = np.asarray(dataset.attrs[name])
        result[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return result


def _dataset_schema(dataset: h5py.Dataset) -> DatasetSchema:
    return DatasetSchema(
        path=dataset.name.lstrip("/"),
        shape=tuple(int(value) for value in dataset.shape),
        dtype=str(dataset.dtype),
        dimension_labels=tuple(dataset.dims[index].label or "" for index in range(dataset.ndim)),
        attributes=_attribute_schema(dataset),
    )


def _find_unique_dataset(
    handle: h5py.File,
    candidates: Sequence[str],
    *,
    label: str,
) -> h5py.Dataset:
    matches = [path for path in candidates if path in handle]
    if len(matches) != 1:
        raise ProtocolError(f"{label}: expected one dataset, found {matches}")
    value = handle[matches[0]]
    if not isinstance(value, h5py.Dataset):
        raise ProtocolError(f"{label} is not a dataset: {matches[0]}")
    return value


def _field_candidates(groups: Sequence[str], field: str) -> tuple[str, ...]:
    return tuple(f"{group}/{field}" for group in groups)


def _required_attribute(dataset: h5py.Dataset, name: str) -> np.ndarray:
    if name not in dataset.attrs:
        raise ProtocolError(f"{dataset.name} lacks required attribute {name!r}")
    return np.asarray(dataset.attrs[name])


def _reflectance_axes(
    dataset: h5py.Dataset,
    spatial_shape: tuple[int, int],
) -> tuple[int, tuple[int, int], np.ndarray, np.ndarray]:
    if dataset.ndim != 3 or not np.issubdtype(dataset.dtype, np.floating):
        raise ProtocolError(
            f"reflectance must be a floating 3-D array, got {dataset.shape} {dataset.dtype}"
        )
    wavelengths = _required_attribute(dataset, "wavelengths")
    good = _required_attribute(dataset, "good_wavelengths")
    fill = _required_attribute(dataset, "_FillValue")
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ProtocolError("wavelengths must be a one-dimensional vector")
    if not np.issubdtype(wavelengths.dtype, np.floating):
        raise ProtocolError("wavelengths must use a floating dtype")
    if not np.isfinite(wavelengths).all() or not np.all(np.diff(wavelengths) > 0):
        raise ProtocolError("wavelengths must be finite and strictly increasing")
    if good.shape != wavelengths.shape or not (
        np.issubdtype(good.dtype, np.bool_) or np.issubdtype(good.dtype, np.integer)
    ):
        raise ProtocolError("good_wavelengths must be a boolean/integer vector per band")
    if fill.ndim != 0:
        raise ProtocolError("reflectance _FillValue must be scalar")

    candidates = [axis for axis, size in enumerate(dataset.shape) if size == wavelengths.size]
    if len(candidates) != 1:
        raise ProtocolError(
            "spectral axis is ambiguous from the wavelength count: "
            f"shape={dataset.shape}, wavelengths={wavelengths.size}"
        )
    spectral_axis = candidates[0]
    spatial_axes = tuple(axis for axis in range(3) if axis != spectral_axis)
    observed_spatial = tuple(int(dataset.shape[axis]) for axis in spatial_axes)
    if observed_spatial != spatial_shape:
        raise ProtocolError(
            "reflectance spatial-axis order does not match geolocation/QA: "
            f"{observed_spatial} != {spatial_shape}"
        )
    if spatial_shape[0] == spatial_shape[1]:
        labels = tuple(dataset.dims[axis].label or "" for axis in spatial_axes)
        if not all(labels):
            raise ProtocolError(
                "square native spatial geometry lacks dimension labels; row/column "
                "orientation cannot be proved"
            )
    return spectral_axis, (spatial_axes[0], spatial_axes[1]), wavelengths, good


def _inspect_basic_schema_handle(
    handle: h5py.File,
    *,
    display_path: Path,
    expected_shape: tuple[int, int],
) -> ProductSchema:
    reflectance = _find_unique_dataset(
        handle,
        _field_candidates(_BASIC_DATA_GROUPS, _REFLECTANCE_FIELD),
        label="basic reflectance",
    )
    latitude = _find_unique_dataset(
        handle,
        tuple(
            f"{group}/{name}" for group in _BASIC_GEO_GROUPS for name in ("Latitude", "latitude")
        ),
        label="basic latitude",
    )
    longitude = _find_unique_dataset(
        handle,
        tuple(
            f"{group}/{name}" for group in _BASIC_GEO_GROUPS for name in ("Longitude", "longitude")
        ),
        label="basic longitude",
    )
    if latitude.ndim != 2 or longitude.shape != latitude.shape:
        raise ProtocolError("basic latitude/longitude arrays are not aligned 2-D fields")
    spatial_shape = tuple(int(value) for value in latitude.shape)
    _require_exact("basic HDF shape versus frozen STAC shape", spatial_shape, expected_shape)
    if not np.issubdtype(latitude.dtype, np.floating) or not np.issubdtype(
        longitude.dtype, np.floating
    ):
        raise ProtocolError("basic latitude/longitude fields must be floating point")

    qa_datasets: dict[str, h5py.Dataset] = {}
    for field in QA_FIELDS:
        dataset = _find_unique_dataset(
            handle,
            _field_candidates(_BASIC_DATA_GROUPS, field),
            label=f"basic QA {field}",
        )
        if dataset.shape != spatial_shape or not np.issubdtype(dataset.dtype, np.integer):
            raise ProtocolError(f"basic QA {field} is not integer or aligned: {dataset.shape}")
        qa_datasets[field] = dataset

    spectral_axis, spatial_axes, wavelengths, good = _reflectance_axes(reflectance, spatial_shape)
    return ProductSchema(
        product_geometry="basic_native_swath",
        path=str(display_path),
        reflectance=_dataset_schema(reflectance),
        spectral_axis=spectral_axis,
        spatial_axes=spatial_axes,
        spatial_shape=spatial_shape,
        wavelength_count=int(wavelengths.size),
        wavelengths_sha256=_array_sha256(wavelengths),
        good_wavelengths_sha256=_array_sha256(good),
        retained_product_bands=int(np.asarray(good, dtype=bool).sum()),
        geolocation={
            "latitude": _dataset_schema(latitude),
            "longitude": _dataset_schema(longitude),
        },
        qa={name: _dataset_schema(dataset) for name, dataset in qa_datasets.items()},
        axis_evidence=(
            "unique wavelength-length spectral axis; ordered remaining axes exactly "
            "match non-square latitude, longitude, and all QA shapes"
        ),
    )


def inspect_basic_schema(path: Path, *, expected_shape: tuple[int, int]) -> ProductSchema:
    """Inspect native HDF metadata without reading reflectance samples."""
    if path.name.endswith(".part"):
        raise ProtocolError(f"partial download cannot be inspected: {path}")
    with h5py.File(path, "r") as handle:
        return _inspect_basic_schema_handle(
            handle,
            display_path=path,
            expected_shape=expected_shape,
        )


def _inspect_ortho_schema_handle(
    handle: h5py.File,
    *,
    display_path: Path,
    expected_shape: tuple[int, int],
) -> ProductSchema:
    reflectance = _find_unique_dataset(
        handle,
        _field_candidates(_ORTHO_DATA_GROUPS, _REFLECTANCE_FIELD),
        label="ortho reflectance",
    )
    qa_datasets: dict[str, h5py.Dataset] = {}
    for field in QA_FIELDS:
        dataset = _find_unique_dataset(
            handle,
            _field_candidates(_ORTHO_DATA_GROUPS, field),
            label=f"ortho QA {field}",
        )
        if dataset.shape != expected_shape or not np.issubdtype(dataset.dtype, np.integer):
            raise ProtocolError(f"ortho QA {field} is not integer or aligned: {dataset.shape}")
        qa_datasets[field] = dataset
    spectral_axis, spatial_axes, wavelengths, good = _reflectance_axes(reflectance, expected_shape)
    return ProductSchema(
        product_geometry="ortho_projected_grid",
        path=str(display_path),
        reflectance=_dataset_schema(reflectance),
        spectral_axis=spectral_axis,
        spatial_axes=spatial_axes,
        spatial_shape=expected_shape,
        wavelength_count=int(wavelengths.size),
        wavelengths_sha256=_array_sha256(wavelengths),
        good_wavelengths_sha256=_array_sha256(good),
        retained_product_bands=int(np.asarray(good, dtype=bool).sum()),
        geolocation={},
        qa={name: _dataset_schema(dataset) for name, dataset in qa_datasets.items()},
        axis_evidence=(
            "unique wavelength-length spectral axis; ordered remaining axes exactly "
            "match the frozen projected shape and all QA shapes"
        ),
    )


def inspect_ortho_schema(path: Path, *, expected_shape: tuple[int, int]) -> ProductSchema:
    """Inspect ortho HDF metadata without reading reflectance samples."""
    if path.name.endswith(".part"):
        raise ProtocolError(f"partial download cannot be inspected: {path}")
    with h5py.File(path, "r") as handle:
        return _inspect_ortho_schema_handle(
            handle,
            display_path=path,
            expected_shape=expected_shape,
        )


def validate_schema_pair(basic: ProductSchema, ortho: ProductSchema) -> None:
    """Require exact band metadata parity within a basic/ortho scene pair."""
    _require_exact("paired wavelength count", basic.wavelength_count, ortho.wavelength_count)
    _require_exact(
        "paired wavelength metadata SHA-256",
        basic.wavelengths_sha256,
        ortho.wavelengths_sha256,
    )
    _require_exact(
        "paired good_wavelengths SHA-256",
        basic.good_wavelengths_sha256,
        ortho.good_wavelengths_sha256,
    )


def load_basic_geolocation_and_qa(
    path: Path,
    schema: ProductSchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Read native geolocation/QA only and apply the existing all-QA-zero policy.

    Reflectance is deliberately not opened through this function.  Unknown QA
    values fail rather than being reinterpreted.
    """
    with h5py.File(path, "r") as handle:
        return _load_basic_geolocation_and_qa_handle(handle, schema)


def _load_basic_geolocation_and_qa_handle(
    handle: h5py.File,
    schema: ProductSchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    latitude_ds = handle[schema.geolocation["latitude"].path]
    longitude_ds = handle[schema.geolocation["longitude"].path]
    latitude = np.asarray(latitude_ds[...], dtype=np.float64)
    longitude = np.asarray(longitude_ds[...], dtype=np.float64)
    qa_arrays = {
        name: np.asarray(handle[descriptor.path][...]) for name, descriptor in schema.qa.items()
    }
    for coordinate, dataset in ((latitude, latitude_ds), (longitude, longitude_ds)):
        if "_FillValue" in dataset.attrs:
            coordinate[coordinate == dataset.attrs["_FillValue"]] = np.nan

    qa_valid = np.ones(schema.spatial_shape, dtype=bool)
    invalid_counts: dict[str, int] = {}
    for name in QA_FIELDS:
        values = qa_arrays[name]
        unknown = set(int(value) for value in np.unique(values)) - QA_ALLOWED_VALUES
        if unknown:
            raise ProtocolError(f"{name} contains undocumented QA values: {sorted(unknown)}")
        invalid = values != 0
        invalid_counts[name] = int(invalid.sum())
        qa_valid &= ~invalid
    return longitude, latitude, qa_valid, invalid_counts


def load_ortho_qa(
    path: Path,
    schema: ProductSchema,
) -> tuple[np.ndarray, dict[str, int]]:
    """Read and validate the target-grid QA fields without reflectance access."""
    with h5py.File(path, "r") as handle:
        return _load_ortho_qa_handle(handle, schema)


def _load_ortho_qa_handle(
    handle: h5py.File,
    schema: ProductSchema,
) -> tuple[np.ndarray, dict[str, int]]:
    qa_arrays = {
        name: np.asarray(handle[descriptor.path][...]) for name, descriptor in schema.qa.items()
    }
    qa_valid = np.ones(schema.spatial_shape, dtype=bool)
    invalid_counts: dict[str, int] = {}
    for name in QA_FIELDS:
        values = qa_arrays[name]
        unknown = set(int(value) for value in np.unique(values)) - QA_ALLOWED_VALUES
        if unknown:
            raise ProtocolError(f"{name} contains undocumented QA values: {sorted(unknown)}")
        invalid = values != 0
        invalid_counts[name] = int(invalid.sum())
        qa_valid &= ~invalid
    return qa_valid, invalid_counts


def _parse_hdfeos_utm_grid(struct_metadata: str, grid_name: str = "HYP") -> OrthoGrid:
    block = None
    for chunk in re.split(r"GROUP=GRID_\d+", struct_metadata):
        match = re.search(r'GridName="([^"]+)"', chunk)
        if match is not None and match.group(1) == grid_name:
            block = chunk
            break
    if block is None:
        raise ProtocolError(f"grid {grid_name!r} not found in StructMetadata")

    def find(key: str) -> str:
        match = re.search(rf"\b{key}=([^\n]+)", block)
        if match is None:
            raise ProtocolError(f"{key} absent from StructMetadata grid {grid_name!r}")
        return match.group(1).strip()

    projection = find("Projection")
    if "UTM" not in projection.upper():
        raise ProtocolError(f"only the delivered UTM ortho grid is supported: {projection}")
    nx = int(find("XDim"))
    ny = int(find("YDim"))
    ulx, uly = (float(value) for value in re.findall(r"[-\d.]+", find("UpperLeftPointMtrs")))
    lrx, lry = (float(value) for value in re.findall(r"[-\d.]+", find("LowerRightMtrs")))
    zone = int(find("ZoneCode"))
    epsg = (32600 if zone > 0 else 32700) + abs(zone)
    transform = Affine((lrx - ulx) / nx, 0.0, ulx, 0.0, -(uly - lry) / ny, uly)
    return OrthoGrid(shape=(ny, nx), transform=transform, crs=CRS.from_epsg(epsg))


def load_ortho_grid(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    expected_crs: str,
    expected_resolution_m: float,
) -> OrthoGrid:
    """Read the delivered projected-grid metadata, never reflectance values."""
    with h5py.File(path, "r") as handle:
        return _load_ortho_grid_handle(
            handle,
            expected_shape=expected_shape,
            expected_crs=expected_crs,
            expected_resolution_m=expected_resolution_m,
        )


def _load_ortho_grid_handle(
    handle: h5py.File,
    *,
    expected_shape: tuple[int, int],
    expected_crs: str,
    expected_resolution_m: float,
) -> OrthoGrid:
    if _STRUCT_METADATA_PATH not in handle:
        raise ProtocolError(f"ortho file lacks {_STRUCT_METADATA_PATH}")
    raw = handle[_STRUCT_METADATA_PATH][()]
    if isinstance(raw, bytes):
        text = raw.decode(errors="strict")
    elif isinstance(raw, np.ndarray) and raw.ndim == 0:
        text = raw.item().decode(errors="strict")
    else:
        text = str(raw)
    grid = _parse_hdfeos_utm_grid(text)
    _require_exact("ortho metadata shape", grid.shape, expected_shape)
    _require_exact("ortho metadata CRS", grid.crs.to_string(), expected_crs)
    _require_exact("ortho metadata x resolution", grid.transform.a, expected_resolution_m)
    _require_exact("ortho metadata y resolution", abs(grid.transform.e), expected_resolution_m)
    return grid


def map_native_to_ortho(
    longitude: np.ndarray,
    latitude: np.ndarray,
    source_qa_valid: np.ndarray,
    target_qa_valid: np.ndarray,
    grid: OrthoGrid,
) -> NativeToOrthoMapping:
    """Assign each eligible ortho cell to its nearest geolocated native sample.

    The delivered ortho QA defines target support, preventing extrapolation
    into the rectangular grid's no-data margin without inventing a distance
    threshold.  The geometrically nearest source is selected before source QA
    is applied; an invalid nearest source therefore creates an explicit no-call
    rather than silently filling the location from a more distant valid sample.
    Reuse of one native sample by multiple ortho cells is retained as source
    multiplicity. Exact-distance ties choose the lower row-major source index.
    """
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    source_qa_valid = np.asarray(source_qa_valid, dtype=bool)
    target_qa_valid = np.asarray(target_qa_valid, dtype=bool)
    if (
        longitude.ndim != 2
        or latitude.shape != longitude.shape
        or source_qa_valid.shape != longitude.shape
    ):
        raise ProtocolError("longitude, latitude, and source QA must be aligned 2-D arrays")
    if target_qa_valid.shape != grid.shape:
        raise ProtocolError("target QA is not aligned to the ortho grid")

    source_shape = longitude.shape
    geo_valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180.0)
        & (longitude <= 180.0)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
    )
    source_status = np.full(source_shape, SOURCE_INVALID_GEOLOCATION, dtype=np.uint8)
    source_status[geo_valid] = SOURCE_UNUSED
    geolocated_source = np.flatnonzero(geo_valid)

    source_row = np.full(grid.shape, -1, dtype=np.int32)
    source_col = np.full(grid.shape, -1, dtype=np.int32)
    source_flat_index = np.full(grid.shape, -1, dtype=np.int64)
    source_multiplicity = np.zeros(grid.shape, dtype=np.uint32)
    mapping_distance = np.full(grid.shape, np.nan, dtype=np.float64)
    target_status = np.full(grid.shape, TARGET_ORTHO_QA_INVALID, dtype=np.uint8)
    target_count_per_source = np.zeros(source_shape, dtype=np.uint32)
    candidate_target = np.flatnonzero(target_qa_valid)

    source_points = np.empty((0, 2), dtype=np.float64)
    if geolocated_source.size:
        source_x_raw, source_y_raw = transform_coordinates(
            "EPSG:4326",
            grid.crs,
            longitude.ravel()[geolocated_source],
            latitude.ravel()[geolocated_source],
        )
        source_points = np.column_stack(
            (
                np.asarray(source_x_raw, dtype=np.float64),
                np.asarray(source_y_raw, dtype=np.float64),
            )
        )
        projected_valid = np.isfinite(source_points).all(axis=1)
        if not projected_valid.all():
            invalid_projected = geolocated_source[~projected_valid]
            geo_valid.ravel()[invalid_projected] = False
            source_status.ravel()[invalid_projected] = SOURCE_INVALID_GEOLOCATION
            geolocated_source = geolocated_source[projected_valid]
            source_points = source_points[projected_valid]

    if geolocated_source.size and candidate_target.size:
        target_rows, target_cols = np.unravel_index(candidate_target, grid.shape)
        target_x, target_y = grid.transform * (target_cols + 0.5, target_rows + 0.5)
        target_points = np.column_stack((target_x, target_y))
        tree = cKDTree(source_points)
        if geolocated_source.size == 1:
            nearest_distance, nearest_position = tree.query(target_points, k=1, workers=1)
            nearest_distance = np.asarray(nearest_distance, dtype=np.float64)
            nearest_position = np.asarray(nearest_position, dtype=np.int64)
        else:
            neighbor_distance, neighbor_position = tree.query(target_points, k=2, workers=1)
            nearest_distance = np.asarray(neighbor_distance[:, 0], dtype=np.float64)
            nearest_position = np.asarray(neighbor_position[:, 0], dtype=np.int64)
            tied = neighbor_distance[:, 0] == neighbor_distance[:, 1]
            for target_position in np.flatnonzero(tied):
                radius = np.nextafter(nearest_distance[target_position], np.inf)
                positions = tree.query_ball_point(target_points[target_position], radius)
                exact_distances = np.linalg.norm(
                    source_points[positions] - target_points[target_position], axis=1
                )
                minimum = exact_distances.min()
                exact_positions = np.asarray(positions, dtype=np.int64)[exact_distances == minimum]
                nearest_position[target_position] = exact_positions[
                    np.argmin(geolocated_source[exact_positions])
                ]
                nearest_distance[target_position] = minimum

        chosen_source = geolocated_source[nearest_position]
        chosen_rows, chosen_cols = np.unravel_index(chosen_source, source_shape)
        source_row.ravel()[candidate_target] = chosen_rows
        source_col.ravel()[candidate_target] = chosen_cols
        source_flat_index.ravel()[candidate_target] = chosen_source
        mapping_distance.ravel()[candidate_target] = nearest_distance
        counts_by_source = np.bincount(chosen_source, minlength=longitude.size).astype(np.uint32)
        target_count_per_source.ravel()[:] = counts_by_source
        source_multiplicity.ravel()[candidate_target] = counts_by_source[chosen_source]
        used_source = np.flatnonzero(counts_by_source)
        source_status.ravel()[used_source] = SOURCE_USED
        chosen_qa_valid = source_qa_valid.ravel()[chosen_source]
        target_status.ravel()[candidate_target[chosen_qa_valid]] = TARGET_MAPPED
        target_status.ravel()[candidate_target[~chosen_qa_valid]] = TARGET_BASIC_QA_INVALID
    elif candidate_target.size:
        target_status.ravel()[candidate_target] = TARGET_NO_GEOLOCATED_SOURCE

    target_counts = target_count_per_source.ravel()
    used_source_count = int((target_counts > 0).sum())
    mapped_count = int((target_status == TARGET_MAPPED).sum())
    counts = MappingCounts(
        total_source_samples=int(longitude.size),
        invalid_qa_source_samples=int((~source_qa_valid).sum()),
        invalid_geolocation_source_samples=int((~geo_valid).sum()),
        used_source_samples=used_source_count,
        unused_source_samples=int(longitude.size - used_source_count),
        sources_with_multiple_target_cells=int((target_counts > 1).sum()),
        duplicate_target_assignments=int((target_counts[target_counts > 0] - 1).sum()),
        total_target_cells=int(np.prod(grid.shape)),
        invalid_qa_target_cells=int((~target_qa_valid).sum()),
        basic_qa_no_call_target_cells=int((target_status == TARGET_BASIC_QA_INVALID).sum()),
        no_geolocated_source_target_cells=int((target_status == TARGET_NO_GEOLOCATED_SOURCE).sum()),
        mapped_target_cells=mapped_count,
        unmapped_target_cells=int(np.prod(grid.shape) - mapped_count),
    )
    if (
        counts.invalid_qa_target_cells
        + counts.basic_qa_no_call_target_cells
        + counts.no_geolocated_source_target_cells
        + counts.mapped_target_cells
        != counts.total_target_cells
    ):
        raise ProtocolError("target-accounting categories do not sum to the target population")
    return NativeToOrthoMapping(
        source_row=source_row,
        source_col=source_col,
        source_flat_index=source_flat_index,
        source_multiplicity=source_multiplicity,
        mapping_distance_m=mapping_distance,
        target_status=target_status,
        source_status=source_status,
        target_count_per_source=target_count_per_source,
        counts=counts,
    )


def project_scalar_nearest(
    source_scalar: np.ndarray,
    mapping: NativeToOrthoMapping,
    *,
    fill_value: int | float = np.nan,
) -> np.ndarray:
    """Project one 2-D scalar field through the frozen source-index ledger."""
    values = np.asarray(source_scalar)
    if values.ndim != 2 or values.shape != mapping.source_status.shape:
        raise ProtocolError(
            "only an aligned 2-D scalar field may be projected; spectra and cubes are forbidden"
        )
    dtype = np.result_type(values.dtype, np.asarray(fill_value).dtype)
    projected = np.full(mapping.source_row.shape, fill_value, dtype=dtype)
    mapped = mapping.target_status == TARGET_MAPPED
    projected[mapped] = values.ravel()[mapping.source_flat_index[mapped]]
    return projected


def canonical_band_y_x(array: np.ndarray, *, spectral_axis: int) -> np.ndarray:
    """Return a band-first view after validating a three-dimensional cube."""
    values = np.asarray(array)
    if values.ndim != 3 or spectral_axis not in range(3):
        raise ProtocolError("spectral data must be three-dimensional with a valid spectral axis")
    return np.moveaxis(values, spectral_axis, 0)


def _spectrum_key(vector: np.ndarray) -> str:
    return _array_sha256(vector)


def exact_spectrum_copy_audit(
    basic_cube: np.ndarray,
    ortho_cube: np.ndarray,
    mapping: NativeToOrthoMapping,
    *,
    retained_bands: np.ndarray,
    basic_valid: np.ndarray,
    ortho_valid: np.ndarray,
) -> SpectrumCopyAudit:
    """Count exact finite-vector copies without tolerance or resampling."""
    basic = np.asarray(basic_cube)
    ortho = np.asarray(ortho_cube)
    if basic.ndim != 3 or ortho.ndim != 3:
        raise ProtocolError("copy audit requires band-first three-dimensional cubes")
    if basic.shape[0] != ortho.shape[0] or basic.dtype != ortho.dtype:
        raise ProtocolError("copy audit requires matching band count and delivered dtype")
    if basic.shape[1:] != mapping.source_status.shape:
        raise ProtocolError("basic cube is not aligned to the source-index ledger")
    if ortho.shape[1:] != mapping.source_row.shape:
        raise ProtocolError("ortho cube is not aligned to the target grid")

    bands = np.asarray(retained_bands, dtype=bool)
    if bands.shape != (basic.shape[0],) or not bands.any():
        raise ProtocolError("retained_bands must select at least one aligned band")
    basic_valid_mask = np.asarray(basic_valid, dtype=bool)
    ortho_valid_mask = np.asarray(ortho_valid, dtype=bool)
    if basic_valid_mask.shape != basic.shape[1:] or ortho_valid_mask.shape != ortho.shape[1:]:
        raise ProtocolError("copy-audit validity masks are not spatially aligned")

    basic_vectors = np.moveaxis(basic[bands], 0, -1).reshape(-1, int(bands.sum()))
    ortho_vectors = np.moveaxis(ortho[bands], 0, -1).reshape(-1, int(bands.sum()))
    basic_ok = basic_valid_mask.ravel() & np.isfinite(basic_vectors).all(axis=1)
    ortho_ok = ortho_valid_mask.ravel() & np.isfinite(ortho_vectors).all(axis=1)

    index: dict[str, list[int]] = {}
    for basic_index in np.flatnonzero(basic_ok):
        index.setdefault(_spectrum_key(basic_vectors[basic_index]), []).append(int(basic_index))

    any_matches = 0
    for ortho_index in np.flatnonzero(ortho_ok):
        vector = ortho_vectors[ortho_index]
        candidates = index.get(_spectrum_key(vector), ())
        if any(np.array_equal(vector, basic_vectors[candidate]) for candidate in candidates):
            any_matches += 1

    mapped = (mapping.target_status.ravel() == TARGET_MAPPED) & ortho_ok
    mapped_exact = 0
    for ortho_index in np.flatnonzero(mapped):
        source_index = int(mapping.source_flat_index.ravel()[ortho_index])
        if basic_ok[source_index] and np.array_equal(
            ortho_vectors[ortho_index], basic_vectors[source_index]
        ):
            mapped_exact += 1

    valid_ortho_count = int(ortho_ok.sum())
    mapped_count = int(mapped.sum())
    return SpectrumCopyAudit(
        retained_bands=int(bands.sum()),
        valid_basic_spectra=int(basic_ok.sum()),
        valid_ortho_spectra=valid_ortho_count,
        ortho_exact_match_to_any_basic=any_matches,
        ortho_exact_match_to_any_basic_fraction=(
            any_matches / valid_ortho_count if valid_ortho_count else None
        ),
        mapped_valid_ortho_spectra=mapped_count,
        mapped_exact_match_to_selected_basic=mapped_exact,
        mapped_exact_match_to_selected_basic_fraction=(
            mapped_exact / mapped_count if mapped_count else None
        ),
    )


def _validate_spectrum_copy_audit(value: Any, *, label: str) -> dict[str, Any]:
    record = _require_object_fields(
        value,
        label=label,
        fields=set(SpectrumCopyAudit.__dataclass_fields__),
    )
    count_fields = (
        "retained_bands",
        "valid_basic_spectra",
        "valid_ortho_spectra",
        "ortho_exact_match_to_any_basic",
        "mapped_valid_ortho_spectra",
        "mapped_exact_match_to_selected_basic",
    )
    for name in count_fields:
        item = record[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ProtocolError(f"{label} {name} must be a nonnegative integer")
    if record["retained_bands"] <= 0:
        raise ProtocolError(f"{label} must retain at least one spectral band")
    if record["ortho_exact_match_to_any_basic"] > record["valid_ortho_spectra"]:
        raise ProtocolError(f"{label} any-basic matches exceed valid ortho support")
    if record["mapped_valid_ortho_spectra"] > record["valid_ortho_spectra"]:
        raise ProtocolError(f"{label} mapped support exceeds valid ortho support")
    if record["mapped_exact_match_to_selected_basic"] > record["mapped_valid_ortho_spectra"]:
        raise ProtocolError(f"{label} selected-basic matches exceed mapped support")
    fraction_pairs = (
        (
            "ortho_exact_match_to_any_basic_fraction",
            "ortho_exact_match_to_any_basic",
            "valid_ortho_spectra",
        ),
        (
            "mapped_exact_match_to_selected_basic_fraction",
            "mapped_exact_match_to_selected_basic",
            "mapped_valid_ortho_spectra",
        ),
    )
    for fraction_name, numerator_name, denominator_name in fraction_pairs:
        denominator = record[denominator_name]
        expected = record[numerator_name] / denominator if denominator else None
        _require_exact(f"{label} {fraction_name}", record[fraction_name], expected)
    return record


def execution_identity(payload: Mapping[str, Any]) -> str:
    """Return a deterministic execution identity for canonical JSON content."""
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def schema_document(
    inputs: ValidatedInputs,
    schemas: Mapping[str, tuple[ProductSchema, ProductSchema]],
    grids: Mapping[str, OrthoGrid],
    *,
    protocol_sha256: str,
    governing_files: Mapping[str, str],
) -> dict[str, Any]:
    """Build the non-result-bearing schema manifest and execution identity."""
    governing = validate_governing_files(governing_files)
    dependency_trust = tanager_spec_dependency_trust(governing)
    identity_inputs = {
        "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
        "mode": "schema-only",
        "protocol_sha256": protocol_sha256,
        "acquisition_manifest_sha256": inputs.acquisition_manifest_sha256,
        "ortho_manifest_sha256": inputs.ortho_manifest_sha256,
        "governing_files": governing,
        "residual_dependency_trust": dependency_trust,
        "scene_inputs": [
            {
                "site": item.site,
                "scene_id": item.scene_id,
                "basic_sha256": item.basic_sha256,
                "ortho_sha256": item.ortho_sha256,
            }
            for item in inputs.scenes
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "mode": "schema-only",
        "scientific_endpoint_values_inspected": False,
        "scientific_outputs_produced": False,
        "execution_id": execution_identity(identity_inputs),
        "execution_identity_inputs": identity_inputs,
        "residual_dependency_trust": dependency_trust,
        "frozen_input_geometry": {
            item.site: {
                "scene_id": item.scene_id,
                "basic_path": item.basic_path,
                "basic_size_bytes": item.basic_size_bytes,
                "basic_sha256": item.basic_sha256,
                "basic_stac_shape": item.basic_stac_shape,
                "basic_stac_crs": item.basic_stac_crs,
                "basic_stac_resolution_m": item.basic_stac_resolution_m,
                "ortho_path": item.ortho_path,
                "ortho_size_bytes": item.ortho_size_bytes,
                "ortho_sha256": item.ortho_sha256,
                "ortho_stac_shape": item.ortho_stac_shape,
                "ortho_stac_crs": item.ortho_stac_crs,
                "ortho_stac_resolution_m": item.ortho_stac_resolution_m,
            }
            for item in inputs.scenes
        },
        "scenes": {
            site: {
                "basic": asdict(pair[0]),
                "ortho": asdict(pair[1]),
                "ortho_grid": {
                    "shape": grids[site].shape,
                    "transform": grids[site].transform,
                    "crs": grids[site].crs,
                    "resolution_m": (
                        grids[site].transform.a,
                        abs(grids[site].transform.e),
                    ),
                },
            }
            for site, pair in sorted(schemas.items())
        },
    }


def design_document(
    *,
    protocol_sha256: str,
    acquisition_manifest_sha256: str,
    ortho_manifest_sha256: str,
    governing_files: Mapping[str, str],
) -> dict[str, Any]:
    """Build the pre-result design-only artifact."""
    governing = validate_governing_files(governing_files)
    dependency_trust = tanager_spec_dependency_trust(governing)
    identity_inputs = {
        "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
        "mode": "design-only",
        "protocol_sha256": protocol_sha256,
        "acquisition_manifest_sha256": acquisition_manifest_sha256,
        "ortho_manifest_sha256": ortho_manifest_sha256,
        "governing_files": governing,
        "residual_dependency_trust": dependency_trust,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "mode": "design-only",
        "scientific_endpoint_values_inspected": False,
        "scientific_outputs_produced": False,
        "execution_id": execution_identity(identity_inputs),
        "execution_identity_inputs": identity_inputs,
        "residual_dependency_trust": dependency_trust,
        "frozen_scenes": dict(FROZEN_SCENES),
        "mapping_contract": {
            "source_geometry": "native HDF-EOS5 swath geolocation arrays",
            "target_geometry": "paired delivered ortho HDF-EOS5 grid",
            "coordinate_operation": "WGS84 geolocation to delivered projected CRS",
            "assignment": (
                "each eligible ortho target cell selects the nearest geolocated basic source sample"
            ),
            "support": (
                "target all-QA-zero mask; source QA is applied after geometric selection "
                "and creates an explicit no-call"
            ),
            "reuse_accounting": "ortho target-cell count per basic source sample",
            "tie_rule": "exact-distance ties choose the lower row-major source index",
            "mapping_distance_threshold": None,
            "interpolation": None,
            "projectable_rank": 2,
            "unmapped_target": "explicit no-call",
        },
        "planned_artifacts": [
            "mapping_manifest.json",
            "source_index.tif",
            "mapping_distance_m.tif",
            "source_multiplicity.tif",
            "mapping_status.tif",
        ],
    }


def select_resource_pilot_scene(
    inputs: ValidatedInputs,
    *,
    site: str = RESOURCE_PILOT_DEFAULT_SITE,
    branch: str = RESOURCE_PILOT_DEFAULT_BRANCH,
) -> FrozenSceneInput:
    """Select one preregistered scene and branch for non-promotable resource telemetry."""
    if site not in FROZEN_SCENES:
        raise ProtocolError(f"resource-pilot site is not frozen: {site!r}")
    if branch not in RESOURCE_PILOT_BRANCH_ASSETS:
        raise ProtocolError(f"resource-pilot branch is not declared: {branch!r}")
    matches = [scene for scene in inputs.scenes if scene.site == site]
    if len(matches) != 1:
        raise ProtocolError(f"resource-pilot site must resolve to one frozen scene: {site!r}")
    scene = matches[0]
    _require_exact("resource-pilot scene identity", scene.scene_id, FROZEN_SCENES[site])
    return scene


def resource_pilot_execution_identity(identity_inputs: Mapping[str, Any]) -> str:
    """Return a deterministic identity that cannot be confused with a scientific run."""
    return f"resource-pilot-non-promotable-{execution_identity(identity_inputs)}"


def _open_regular_nofollow(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    """Open a regular file without following its final or ancestor symlinks."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ProtocolError(f"{label} requires O_NOFOLLOW and O_DIRECTORY support")
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise ProtocolError(f"{label} path is not normalized and absolute: {path}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fds: list[int] = []
    try:
        directory_fd = os.open(absolute.anchor, directory_flags)
        directory_fds.append(directory_fd)
        for part in absolute.parts[1:-1]:
            directory_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            directory_fds.append(directory_fd)
        file_fd = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(file_fd)
            raise ProtocolError(f"{label} is not a regular file: {path}")
        return file_fd, info
    except OSError as error:
        raise ProtocolError(
            f"{label} cannot be opened without following symlinks: {path}"
        ) from error
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


@contextmanager
def _verified_input_snapshot(
    source_path: Path,
    snapshot_path: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> Iterator[BinaryIO]:
    """Copy one descriptor-bound input into an unlinked verified snapshot."""
    source_fd, source_info = _open_regular_nofollow(source_path, label="resource-pilot input")
    if source_info.st_size != expected_size_bytes:
        os.close(source_fd)
        raise ProtocolError(
            f"resource-pilot input byte size does not match manifest: {source_path}"
        )
    snapshot_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        snapshot_fd = os.open(snapshot_path, snapshot_flags, 0o600)
    except OSError as error:
        os.close(source_fd)
        raise ProtocolError(
            f"cannot create private resource-pilot snapshot: {snapshot_path}"
        ) from error

    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        with os.fdopen(source_fd, "rb") as source, os.fdopen(snapshot_fd, "w+b") as snapshot:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                snapshot.write(chunk)
                copied_bytes += len(chunk)
            snapshot.flush()
            os.fsync(snapshot.fileno())
            if copied_bytes != expected_size_bytes or digest.hexdigest() != expected_sha256:
                raise ProtocolError(
                    f"resource-pilot input snapshot does not match manifest: {source_path}"
                )
            snapshot.seek(0)
            snapshot_path.unlink()
            yield snapshot
    finally:
        if snapshot_path.exists() or snapshot_path.is_symlink():
            snapshot_path.unlink()


def _process_max_rss_bytes() -> int:
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1024


def measure_resource_pilot_load(
    source_path: Path,
    *,
    snapshot_path: Path | None = None,
    input_snapshot: BinaryIO | None = None,
    branch: str,
    expected_shape: tuple[int, int],
    expected_size_bytes: int,
    expected_sha256: str,
) -> ResourcePilotTelemetry:
    """Load one verified input snapshot once and retain resource telemetry only."""
    if branch not in RESOURCE_PILOT_BRANCH_ASSETS:
        raise ProtocolError(f"resource-pilot branch is not declared: {branch!r}")

    @contextmanager
    def verified_snapshot() -> Iterator[BinaryIO]:
        if input_snapshot is not None:
            info = os.fstat(input_snapshot.fileno())
            if info.st_size != expected_size_bytes:
                raise ProtocolError("resource-pilot sealed snapshot has an unexpected byte size")
            input_snapshot.seek(0)
            yield input_snapshot
            return
        if snapshot_path is None:
            raise ProtocolError("resource-pilot requires a sealed input snapshot")
        with _verified_input_snapshot(
            source_path,
            snapshot_path,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        ) as created_snapshot:
            yield created_snapshot

    with verified_snapshot() as snapshot:
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        rss_before = _process_max_rss_bytes()
        wall_start = time.perf_counter()
        with h5py.File(snapshot, "r") as handle:
            if branch == "B":
                schema = _inspect_basic_schema_handle(
                    handle,
                    display_path=source_path,
                    expected_shape=expected_shape,
                )
            else:
                schema = _inspect_ortho_schema_handle(
                    handle,
                    display_path=source_path,
                    expected_shape=expected_shape,
                )
            values = np.asarray(handle[schema.reflectance.path][...])
        wall_seconds = time.perf_counter() - wall_start
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        rss_after = _process_max_rss_bytes()
        loaded_array_bytes = int(values.nbytes)
        del values
    return ResourcePilotTelemetry(
        wall_seconds=wall_seconds,
        user_cpu_seconds=float(usage_after.ru_utime - usage_before.ru_utime),
        system_cpu_seconds=float(usage_after.ru_stime - usage_before.ru_stime),
        process_max_rss_before_bytes=rss_before,
        process_max_rss_after_bytes=rss_after,
        loaded_array_bytes=loaded_array_bytes,
    )


def resource_pilot_identity_inputs(
    scene: FrozenSceneInput,
    *,
    branch: str,
    protocol_sha256: str,
    acquisition_manifest_sha256: str,
    ortho_manifest_sha256: str,
    governing_files: Mapping[str, str],
) -> dict[str, Any]:
    """Build the exact deterministic identity inputs for one resource pilot."""
    if branch not in RESOURCE_PILOT_BRANCH_ASSETS:
        raise ProtocolError(f"resource-pilot branch is not declared: {branch!r}")
    governing = validate_governing_files(governing_files)
    dependency_trust = tanager_spec_dependency_trust(governing)
    if branch == "B":
        input_path = scene.basic_path
        input_size_bytes = scene.basic_size_bytes
        input_sha256 = scene.basic_sha256
    else:
        input_path = scene.ortho_path
        input_size_bytes = scene.ortho_size_bytes
        input_sha256 = scene.ortho_sha256
    return {
        "execution_class": RESOURCE_PILOT_EXECUTION_IDENTITY,
        "mode": "resource-pilot",
        "protocol_sha256": protocol_sha256,
        "acquisition_manifest_sha256": acquisition_manifest_sha256,
        "ortho_manifest_sha256": ortho_manifest_sha256,
        "governing_files": governing,
        "residual_dependency_trust": dependency_trust,
        "selector": {
            "site": scene.site,
            "scene_id": scene.scene_id,
            "branch": branch,
            "asset_key": RESOURCE_PILOT_BRANCH_ASSETS[branch],
        },
        "input": {
            "path": str(input_path),
            "size_bytes": input_size_bytes,
            "sha256": input_sha256,
        },
    }


def resource_pilot_document(
    scene: FrozenSceneInput,
    *,
    branch: str,
    protocol_sha256: str,
    acquisition_manifest_sha256: str,
    ortho_manifest_sha256: str,
    governing_files: Mapping[str, str],
    telemetry: ResourcePilotTelemetry,
) -> dict[str, Any]:
    """Build a telemetry-only manifest with an explicitly non-promotable identity."""
    identity_inputs = resource_pilot_identity_inputs(
        scene,
        branch=branch,
        protocol_sha256=protocol_sha256,
        acquisition_manifest_sha256=acquisition_manifest_sha256,
        ortho_manifest_sha256=ortho_manifest_sha256,
        governing_files=governing_files,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "m1b_basic_ortho_resource_pilot",
        "mode": "resource-pilot",
        "scientific_endpoint_values_inspected": False,
        "scientific_outputs_produced": False,
        "scientific_execution_promotable": False,
        "execution_id": resource_pilot_execution_identity(identity_inputs),
        "execution_identity_inputs": identity_inputs,
        "residual_dependency_trust": identity_inputs["residual_dependency_trust"],
        "selector": identity_inputs["selector"],
        "input": identity_inputs["input"],
        "telemetry": asdict(telemetry),
        "output_contract": {
            "allowed_files": ["resource_pilot_manifest.json"],
            "scientific_metrics": [],
            "scientific_maps": [],
            "reflectance_values": [],
        },
    }


def verify_resource_pilot_bundle(
    bundle_root: Path,
    *,
    expected_run_id: str,
    expected_protocol_sha256: str,
) -> None:
    """Verify a telemetry-only resource-pilot bundle before publication."""
    manifest, _digest, _size, _manifest_path = _read_bundle_json(
        bundle_root,
        "resource_pilot_manifest.json",
        label="resource-pilot manifest",
    )
    _require_exact(
        "resource-pilot manifest fields",
        set(manifest),
        {
            "schema_version",
            "manifest_type",
            "mode",
            "scientific_endpoint_values_inspected",
            "scientific_outputs_produced",
            "scientific_execution_promotable",
            "execution_id",
            "execution_identity_inputs",
            "residual_dependency_trust",
            "selector",
            "input",
            "telemetry",
            "output_contract",
        },
    )
    _require_exact(
        "resource-pilot manifest type",
        manifest.get("manifest_type"),
        "m1b_basic_ortho_resource_pilot",
    )
    _require_exact("resource-pilot schema version", manifest.get("schema_version"), SCHEMA_VERSION)
    _require_exact("resource-pilot mode", manifest.get("mode"), "resource-pilot")
    _require_exact("resource-pilot execution ID", manifest.get("execution_id"), expected_run_id)
    identity_inputs = _require_object_fields(
        manifest.get("execution_identity_inputs"),
        label="resource-pilot execution identity inputs",
        fields={
            "execution_class",
            "mode",
            "protocol_sha256",
            "acquisition_manifest_sha256",
            "ortho_manifest_sha256",
            "governing_files",
            "residual_dependency_trust",
            "selector",
            "input",
        },
    )
    _require_exact(
        "resource-pilot execution class",
        identity_inputs.get("execution_class"),
        RESOURCE_PILOT_EXECUTION_IDENTITY,
    )
    _require_exact("resource-pilot identity mode", identity_inputs.get("mode"), "resource-pilot")
    _require_exact(
        "resource-pilot externally sealed preregistration SHA-256",
        identity_inputs.get("protocol_sha256"),
        expected_protocol_sha256,
    )
    governing = validate_governing_files(
        identity_inputs.get("governing_files"),
        label="resource-pilot identity governing files",
    )
    identity_dependency_trust = _validate_residual_dependency_trust(
        identity_inputs.get("residual_dependency_trust"),
        governing_files=governing,
    )
    manifest_dependency_trust = _validate_residual_dependency_trust(
        manifest.get("residual_dependency_trust"),
        governing_files=governing,
    )
    _require_exact(
        "resource-pilot dependency trust identity linkage",
        manifest_dependency_trust,
        identity_dependency_trust,
    )
    identity_selector = _require_object_fields(
        identity_inputs.get("selector"),
        label="resource-pilot identity selector",
        fields={"site", "scene_id", "branch", "asset_key"},
    )
    selector = _require_object_fields(
        manifest.get("selector"),
        label="resource-pilot selector",
        fields={"site", "scene_id", "branch", "asset_key"},
    )
    _require_exact("resource-pilot selector identity", selector, identity_inputs.get("selector"))
    if selector.get("site") not in FROZEN_SCENES:
        raise ProtocolError("resource-pilot selector contains an unfrozen site")
    _require_exact(
        "resource-pilot selector scene",
        selector.get("scene_id"),
        FROZEN_SCENES[str(selector["site"])],
    )
    if selector.get("branch") not in RESOURCE_PILOT_BRANCH_ASSETS:
        raise ProtocolError("resource-pilot selector contains an undeclared branch")
    _require_exact(
        "resource-pilot selector asset",
        selector.get("asset_key"),
        RESOURCE_PILOT_BRANCH_ASSETS[str(selector["branch"])],
    )
    _require_exact("resource-pilot identity selector linkage", identity_selector, selector)
    identity_input = _require_object_fields(
        identity_inputs.get("input"),
        label="resource-pilot identity input",
        fields={"path", "size_bytes", "sha256"},
    )
    pilot_input = _require_object_fields(
        manifest.get("input"),
        label="resource-pilot input",
        fields={"path", "size_bytes", "sha256"},
    )
    _require_exact("resource-pilot input identity", pilot_input, identity_input)
    if not isinstance(pilot_input["path"], str) or not pilot_input["path"]:
        raise ProtocolError("resource-pilot input path must be a non-empty string")
    if (
        isinstance(pilot_input["size_bytes"], bool)
        or not isinstance(pilot_input["size_bytes"], int)
        or pilot_input["size_bytes"] <= 0
    ):
        raise ProtocolError("resource-pilot input size must be a positive integer")
    if not isinstance(pilot_input["sha256"], str) or not pilot_input["sha256"]:
        raise ProtocolError("resource-pilot input SHA-256 must be a non-empty string")
    telemetry = _require_object_fields(
        manifest.get("telemetry"),
        label="resource-pilot telemetry",
        fields={
            "wall_seconds",
            "user_cpu_seconds",
            "system_cpu_seconds",
            "process_max_rss_before_bytes",
            "process_max_rss_after_bytes",
            "loaded_array_bytes",
        },
    )
    for name, value in telemetry.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ProtocolError(f"resource-pilot telemetry is invalid: {name}={value!r}")
    for name in (
        "process_max_rss_before_bytes",
        "process_max_rss_after_bytes",
        "loaded_array_bytes",
    ):
        if not isinstance(telemetry[name], int):
            raise ProtocolError(f"resource-pilot telemetry must be integer bytes: {name}")
    _require_exact(
        "resource-pilot endpoint inspection",
        manifest.get("scientific_endpoint_values_inspected"),
        False,
    )
    _require_exact(
        "resource-pilot scientific outputs",
        manifest.get("scientific_outputs_produced"),
        False,
    )
    _require_exact(
        "resource-pilot promotability",
        manifest.get("scientific_execution_promotable"),
        False,
    )
    output_contract = _require_object_fields(
        manifest.get("output_contract"),
        label="resource-pilot output contract",
        fields={"allowed_files", "scientific_metrics", "scientific_maps", "reflectance_values"},
    )
    if output_contract != {
        "allowed_files": ["resource_pilot_manifest.json"],
        "scientific_metrics": [],
        "scientific_maps": [],
        "reflectance_values": [],
    }:
        raise ProtocolError("resource-pilot output contract permits scientific artifacts")
    _require_exact(
        "resource-pilot recomputed execution ID",
        resource_pilot_execution_identity(identity_inputs),
        expected_run_id,
    )
    _require_exact(
        "resource-pilot file set",
        _bundle_file_set(bundle_root),
        {"resource_pilot_manifest.json"},
    )


def _atomic_write_raster(
    path: Path,
    arrays: Sequence[np.ndarray],
    *,
    grid: OrthoGrid,
    dtype: str,
    nodata: int | float,
    descriptions: Sequence[str],
    tags: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    profile = {
        "driver": "GTiff",
        "height": grid.shape[0],
        "width": grid.shape[1],
        "count": len(arrays),
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    try:
        with rasterio.open(temporary, "w", **profile) as destination:
            for band, (array, description) in enumerate(zip(arrays, descriptions, strict=True), 1):
                if array.shape != grid.shape:
                    raise ProtocolError(f"raster band {description!r} is not target-grid aligned")
                destination.write(np.asarray(array, dtype=dtype), band)
                destination.set_band_description(band, description)
            destination.update_tags(**dict(tags))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_mapping_bundle(
    output_root: Path,
    *,
    scene: FrozenSceneInput,
    grid: OrthoGrid,
    mapping: NativeToOrthoMapping,
    protocol_sha256: str,
    acquisition_manifest_sha256: str,
    ortho_manifest_sha256: str,
    governing_files: Mapping[str, str],
    qa_invalid_counts: Mapping[str, int],
    spectral_copy_audit: SpectrumCopyAudit,
) -> Path:
    """Write one identity-scoped scene bundle inside a run staging directory."""
    governing = validate_governing_files(governing_files)
    dependency_trust = tanager_spec_dependency_trust(governing)
    identity_inputs = {
        "scientific_execution_identity": SCIENTIFIC_EXECUTION_IDENTITY,
        "mode": "mapping-only",
        "site": scene.site,
        "scene_id": scene.scene_id,
        "protocol_sha256": protocol_sha256,
        "acquisition_manifest_sha256": acquisition_manifest_sha256,
        "ortho_manifest_sha256": ortho_manifest_sha256,
        "basic_sha256": scene.basic_sha256,
        "ortho_sha256": scene.ortho_sha256,
        "governing_files": governing,
        "residual_dependency_trust": dependency_trust,
    }
    run_id = execution_identity(identity_inputs)
    destination = output_root / run_id / scene.site
    if destination.exists() or destination.is_symlink():
        raise ProtocolError(f"refusing to replace existing scene mapping bundle: {destination}")
    tags = {
        "execution_id": run_id,
        "scene_id": scene.scene_id,
        "source_product": "basic_sr_hdf5",
        "target_product": "ortho_sr_hdf5",
        "resampling": "target_cell_to_nearest_basic_source",
        "spectral_interpolation": "none",
    }
    source_index_path = destination / "source_index.tif"
    distance_path = destination / "mapping_distance_m.tif"
    multiplicity_path = destination / "source_multiplicity.tif"
    status_path = destination / "mapping_status.tif"
    _atomic_write_raster(
        source_index_path,
        (mapping.source_row, mapping.source_col),
        grid=grid,
        dtype="int32",
        nodata=-1,
        descriptions=("basic_source_row", "basic_source_col"),
        tags=tags,
    )
    _atomic_write_raster(
        distance_path,
        (mapping.mapping_distance_m,),
        grid=grid,
        dtype="float64",
        nodata=np.nan,
        descriptions=("target_centre_to_basic_source_distance_m",),
        tags=tags,
    )
    _atomic_write_raster(
        multiplicity_path,
        (mapping.source_multiplicity,),
        grid=grid,
        dtype="uint32",
        nodata=0,
        descriptions=("ortho_target_cells_assigned_to_basic_source",),
        tags=tags,
    )
    _atomic_write_raster(
        status_path,
        (mapping.target_status,),
        grid=grid,
        dtype="uint8",
        nodata=0,
        descriptions=("mapping_status_code",),
        tags=tags,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "mode": "mapping-only",
        "scientific_endpoint_values_inspected": False,
        "scientific_outputs_produced": False,
        "execution_id": run_id,
        "execution_identity_inputs": identity_inputs,
        "frozen_input": asdict(scene),
        "grid": {
            "shape": grid.shape,
            "transform": grid.transform,
            "crs": grid.crs,
        },
        "source_accounting": asdict(mapping.counts),
        "qa_invalid_counts_nonexclusive": dict(qa_invalid_counts),
        "spectral_copy_audit": asdict(spectral_copy_audit),
        "mapping_distance_threshold_m": None,
        "multiplicity_basis": (
            "target-QA-valid geometric assignments before basic-source QA no-calls"
        ),
        "mapping_status_codes": {
            "0": "reserved raster nodata",
            str(int(TARGET_ORTHO_QA_INVALID)): "ortho QA invalid",
            str(int(TARGET_NO_GEOLOCATED_SOURCE)): "no geolocated basic source available",
            str(int(TARGET_BASIC_QA_INVALID)): "nearest basic source QA invalid",
            str(int(TARGET_MAPPED)): "mapped to QA-valid basic source",
        },
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (source_index_path, distance_path, multiplicity_path, status_path)
        },
    }
    manifest_path = destination / "mapping_manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        _require_exact("existing mapping execution ID", existing.get("execution_id"), run_id)
    strict_json_dump(manifest_path, manifest)
    return manifest_path


def _read_bundle_json(
    bundle_root: Path,
    relative_path: str,
    *,
    label: str,
) -> tuple[dict[str, Any], str, int, Path]:
    path = _require_bundle_file(bundle_root, relative_path)
    file_fd, info = _open_regular_nofollow(path, label=label)
    digest = hashlib.sha256()
    try:
        with os.fdopen(file_fd, "rb") as handle:
            payload_bytes = handle.read()
            digest.update(payload_bytes)
    except OSError as error:
        raise ProtocolError(f"cannot read {label}: {path}") from error
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot parse {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label} root must be an object: {path}")
    return payload, digest.hexdigest(), info.st_size, path


@contextmanager
def _verified_bundle_raster(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> Iterator[rasterio.io.DatasetReader]:
    file_fd, info = _open_regular_nofollow(path, label=label)
    digest = hashlib.sha256()
    with os.fdopen(file_fd, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
        _require_exact(f"{label} byte size", info.st_size, expected_size_bytes)
        _require_exact(f"{label} SHA-256", digest.hexdigest(), expected_sha256)
        handle.seek(0)
        try:
            with rasterio.open(handle) as dataset:
                yield dataset
        except (OSError, rasterio.errors.RasterioError) as error:
            raise ProtocolError(f"cannot open verified raster snapshot: {label}") from error


_MAPPING_RASTER_SPECS: dict[str, dict[str, Any]] = {
    "source_index.tif": {
        "count": 2,
        "dtype": "int32",
        "nodata": -1,
        "descriptions": ("basic_source_row", "basic_source_col"),
    },
    "mapping_distance_m.tif": {
        "count": 1,
        "dtype": "float64",
        "nodata": np.nan,
        "descriptions": ("target_centre_to_basic_source_distance_m",),
    },
    "source_multiplicity.tif": {
        "count": 1,
        "dtype": "uint32",
        "nodata": 0,
        "descriptions": ("ortho_target_cells_assigned_to_basic_source",),
    },
    "mapping_status.tif": {
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "descriptions": ("mapping_status_code",),
    },
}


def _require_raster_nodata(
    observed: tuple[float | None, ...],
    expected: int | float,
    *,
    label: str,
) -> None:
    for value in observed:
        if isinstance(expected, float) and math.isnan(expected):
            if value is None or not math.isnan(value):
                raise ProtocolError(f"{label} nodata must be NaN, observed {observed!r}")
        elif value != expected:
            raise ProtocolError(f"{label} nodata: expected {expected!r}, observed {observed!r}")


def _read_semantic_mapping_raster(
    path: Path,
    *,
    artifact_name: str,
    artifact_record: Mapping[str, Any],
    grid: OrthoGrid,
    scene_run_id: str,
    scene_id: str,
) -> np.ndarray:
    spec = _MAPPING_RASTER_SPECS[artifact_name]
    expected_tags = {
        "execution_id": scene_run_id,
        "scene_id": scene_id,
        "source_product": "basic_sr_hdf5",
        "target_product": "ortho_sr_hdf5",
        "resampling": "target_cell_to_nearest_basic_source",
        "spectral_interpolation": "none",
    }
    with _verified_bundle_raster(
        path,
        expected_sha256=str(artifact_record["sha256"]),
        expected_size_bytes=int(artifact_record["size_bytes"]),
        label=f"{scene_id} {artifact_name}",
    ) as dataset:
        _require_exact(f"{artifact_name} shape", dataset.shape, grid.shape)
        _require_exact(f"{artifact_name} transform", dataset.transform, grid.transform)
        _require_exact(f"{artifact_name} CRS", dataset.crs, grid.crs)
        _require_exact(f"{artifact_name} band count", dataset.count, spec["count"])
        _require_exact(
            f"{artifact_name} dtypes",
            dataset.dtypes,
            tuple(spec["dtype"] for _ in range(spec["count"])),
        )
        _require_raster_nodata(dataset.nodatavals, spec["nodata"], label=artifact_name)
        _require_exact(
            f"{artifact_name} descriptions",
            dataset.descriptions,
            spec["descriptions"],
        )
        observed_tags = dict(dataset.tags())
        if observed_tags.get("AREA_OR_POINT") == "Area":
            observed_tags.pop("AREA_OR_POINT")
        _require_exact(f"{artifact_name} tags", observed_tags, expected_tags)
        return dataset.read()


def _validated_scene_grid(
    scene_manifest: Mapping[str, Any],
    frozen_input: Mapping[str, Any],
) -> OrthoGrid:
    grid_record = _require_object_fields(
        scene_manifest.get("grid"),
        label="mapping grid",
        fields={"shape", "transform", "crs"},
    )
    shape_value = grid_record["shape"]
    transform_value = grid_record["transform"]
    if not isinstance(shape_value, list) or len(shape_value) != 2:
        raise ProtocolError("mapping grid shape must contain exactly two dimensions")
    if not isinstance(transform_value, list) or len(transform_value) != 9:
        raise ProtocolError("mapping grid transform must contain exactly nine matrix coefficients")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in shape_value):
        raise ProtocolError("mapping grid shape must contain integers")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in transform_value
    ):
        raise ProtocolError("mapping grid transform must contain finite numbers")
    if not isinstance(grid_record["crs"], str):
        raise ProtocolError("mapping grid CRS must be a string")
    _require_exact("mapping grid homogeneous transform row", transform_value[6:], [0.0, 0.0, 1.0])
    grid = OrthoGrid(
        shape=(int(shape_value[0]), int(shape_value[1])),
        transform=Affine(*transform_value[:6]),
        crs=CRS.from_user_input(grid_record["crs"]),
    )
    _require_exact(
        "mapping grid versus frozen ortho shape",
        list(grid.shape),
        frozen_input.get("ortho_stac_shape"),
    )
    _require_exact(
        "mapping grid versus frozen ortho CRS",
        grid.crs,
        CRS.from_user_input(frozen_input.get("ortho_stac_crs")),
    )
    expected_resolution = float(frozen_input.get("ortho_stac_resolution_m"))
    _require_exact("mapping grid x resolution", grid.transform.a, expected_resolution)
    _require_exact("mapping grid y resolution", abs(grid.transform.e), expected_resolution)
    return grid


def _validate_mapping_array_semantics(
    rasters: Mapping[str, np.ndarray],
    *,
    frozen_input: Mapping[str, Any],
    source_accounting: Mapping[str, int],
) -> None:
    source_row, source_col = rasters["source_index.tif"]
    distance = rasters["mapping_distance_m.tif"][0]
    multiplicity = rasters["source_multiplicity.tif"][0]
    status = rasters["mapping_status.tif"][0]
    allowed_status = {
        int(TARGET_ORTHO_QA_INVALID),
        int(TARGET_NO_GEOLOCATED_SOURCE),
        int(TARGET_BASIC_QA_INVALID),
        int(TARGET_MAPPED),
    }
    observed_status = {int(value) for value in np.unique(status)}
    if not observed_status <= allowed_status:
        raise ProtocolError(f"mapping status raster contains unknown codes: {observed_status}")

    ortho_invalid = status == TARGET_ORTHO_QA_INVALID
    no_source = status == TARGET_NO_GEOLOCATED_SOURCE
    basic_invalid = status == TARGET_BASIC_QA_INVALID
    mapped = status == TARGET_MAPPED
    selected = basic_invalid | mapped
    if not np.array_equal(source_row == -1, source_col == -1):
        raise ProtocolError("source row and column no-call sentinels are inconsistent")
    if not np.all((source_row[selected] >= 0) & (source_col[selected] >= 0)):
        raise ProtocolError("selected targets lack valid basic source indices")
    if not np.all((source_row[~selected] == -1) & (source_col[~selected] == -1)):
        raise ProtocolError("unselected targets carry basic source indices")
    basic_shape = frozen_input.get("basic_stac_shape")
    if not isinstance(basic_shape, list) or len(basic_shape) != 2:
        raise ProtocolError("frozen basic shape is malformed")
    basic_rows, basic_cols = (int(basic_shape[0]), int(basic_shape[1]))
    if not np.all((source_row[selected] < basic_rows) & (source_col[selected] < basic_cols)):
        raise ProtocolError("selected basic source indices exceed the frozen native grid")
    if not np.isfinite(distance[selected]).all() or np.any(distance[selected] < 0):
        raise ProtocolError("selected targets require finite nonnegative mapping distances")
    if not np.isnan(distance[~selected]).all():
        raise ProtocolError("unselected targets must have NaN mapping distance")
    if np.any(multiplicity[~selected] != 0) or np.any(multiplicity[selected] == 0):
        raise ProtocolError("source multiplicity is inconsistent with mapping status")

    flat_source = source_row[selected].astype(np.int64) * basic_cols + source_col[selected]
    unique_source, target_counts = np.unique(flat_source, return_counts=True)
    count_by_source = dict(zip(unique_source.tolist(), target_counts.tolist(), strict=True))
    expected_multiplicity = np.array(
        [count_by_source[int(value)] for value in flat_source],
        dtype=multiplicity.dtype,
    )
    if not np.array_equal(multiplicity[selected], expected_multiplicity):
        raise ProtocolError("source multiplicity does not equal repeated target assignments")

    total_targets = int(status.size)
    used_sources = int(unique_source.size)
    repeated_sources = int((target_counts > 1).sum())
    duplicate_assignments = int((target_counts - 1).sum())
    total_sources = basic_rows * basic_cols
    derived_counts = {
        "total_source_samples": total_sources,
        "used_source_samples": used_sources,
        "unused_source_samples": total_sources - used_sources,
        "sources_with_multiple_target_cells": repeated_sources,
        "duplicate_target_assignments": duplicate_assignments,
        "total_target_cells": total_targets,
        "invalid_qa_target_cells": int(ortho_invalid.sum()),
        "basic_qa_no_call_target_cells": int(basic_invalid.sum()),
        "no_geolocated_source_target_cells": int(no_source.sum()),
        "mapped_target_cells": int(mapped.sum()),
        "unmapped_target_cells": int((~mapped).sum()),
    }
    for name, expected in derived_counts.items():
        _require_exact(f"source accounting {name}", source_accounting.get(name), expected)
    for name in ("invalid_qa_source_samples", "invalid_geolocation_source_samples"):
        value = source_accounting.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= total_sources:
            raise ProtocolError(f"source accounting {name} is outside the source population")


def _validate_snapshot_record(value: Any, *, label: str) -> dict[str, Any]:
    record = _require_object_fields(
        value,
        label=label,
        fields={
            "source_path",
            "source_pre",
            "source_post",
            "copied_byte_count",
            "sha256",
        },
    )
    if not isinstance(record["source_path"], str) or not record["source_path"]:
        raise ProtocolError(f"{label} source path must be a non-empty string")
    stat_fields = {"device", "inode", "size_bytes", "mtime_ns", "ctime_ns"}
    pre = _require_object_fields(
        record["source_pre"], label=f"{label} pre-stat", fields=stat_fields
    )
    post = _require_object_fields(
        record["source_post"],
        label=f"{label} post-stat",
        fields=stat_fields,
    )
    for phase, observed in (("pre", pre), ("post", post)):
        for name, item in observed.items():
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ProtocolError(f"{label} {phase}-{name} must be a nonnegative integer")
    _require_exact(f"{label} stable source metadata", post, pre)
    copied = record["copied_byte_count"]
    if isinstance(copied, bool) or not isinstance(copied, int) or copied < 0:
        raise ProtocolError(f"{label} copied byte count must be a nonnegative integer")
    _require_exact(f"{label} copied byte count", copied, pre["size_bytes"])
    digest = record["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ProtocolError(f"{label} SHA-256 must be a lowercase digest")
    return record


def _validate_input_snapshot_evidence(
    value: Any,
    *,
    expected_scenes: Mapping[str, str],
) -> dict[str, Any]:
    evidence = _require_object_fields(
        value,
        label="mapping-run input snapshot evidence",
        fields={"preregistration", "acquisition_manifest", "ortho_manifest", "scenes"},
    )
    for name in ("preregistration", "acquisition_manifest", "ortho_manifest"):
        _validate_snapshot_record(evidence[name], label=f"mapping-run {name} snapshot")
    scenes = _require_object_fields(
        evidence["scenes"],
        label="mapping-run scene snapshot evidence",
        fields=set(expected_scenes),
    )
    for site in expected_scenes:
        products = _require_object_fields(
            scenes[site],
            label=f"{site} scene snapshot evidence",
            fields={"basic", "ortho"},
        )
        for product in ("basic", "ortho"):
            _validate_snapshot_record(
                products[product],
                label=f"{site} {product} snapshot",
            )
    return evidence


def _validate_sha256_inventory(
    value: Any,
    *,
    label: str,
    expected_paths: set[str],
) -> dict[str, str]:
    records = _require_object_fields(value, label=label, fields=expected_paths)
    for logical_path, digest in records.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ProtocolError(f"{label} {logical_path} must be a lowercase SHA-256 digest")
    return {key: records[key] for key in sorted(records)}


def _validate_residual_dependency_trust(
    value: Any,
    *,
    governing_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    trust = _require_object_fields(
        value,
        label="residual dependency trust",
        fields={"tanager_spec"},
    )
    record = _require_object_fields(
        trust["tanager_spec"],
        label="tanager_spec hash-bound dependency trust",
        fields={
            "classification",
            "hash_bound",
            "editable_root_logical_path",
            "python_source_files",
            "package_data_files",
            "module_origins",
            "inventory_sha256",
        },
    )
    _require_exact(
        "tanager_spec dependency trust classification",
        record["classification"],
        "captured_hash_bound_editable_dependency",
    )
    _require_exact("tanager_spec dependency hash binding", record["hash_bound"], True)
    _require_exact(
        "tanager_spec editable logical root",
        record["editable_root_logical_path"],
        TANAGER_SPEC_EDITABLE_LOGICAL_ROOT,
    )
    python_source_files = _validate_sha256_inventory(
        record["python_source_files"],
        label="tanager_spec Python source inventory",
        expected_paths=set(TANAGER_SPEC_MODULE_FILES.values()),
    )
    package_data_files = _validate_sha256_inventory(
        record["package_data_files"],
        label="tanager_spec package-data inventory",
        expected_paths=set(TANAGER_SPEC_PACKAGE_DATA_FILES),
    )
    origins = _require_object_fields(
        record["module_origins"],
        label="tanager_spec module origins",
        fields=set(TANAGER_SPEC_MODULE_FILES),
    )
    _require_exact(
        "tanager_spec exact logical module origins",
        origins,
        TANAGER_SPEC_MODULE_FILES,
    )
    inventory_payload = {
        "python_source_files": python_source_files,
        "package_data_files": package_data_files,
        "module_origins": {key: origins[key] for key in sorted(origins)},
    }
    digest = record["inventory_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ProtocolError("tanager_spec inventory SHA-256 must be a lowercase digest")
    _require_exact(
        "tanager_spec recomputed inventory SHA-256",
        digest,
        execution_identity(inventory_payload),
    )
    if governing_files is not None:
        governing = validate_governing_files(governing_files)
        expected_source = {
            path: governing[path] for path in sorted(TANAGER_SPEC_MODULE_FILES.values())
        }
        expected_data = {path: governing[path] for path in sorted(TANAGER_SPEC_PACKAGE_DATA_FILES)}
        _require_exact(
            "tanager_spec source hashes linked to governing identity",
            python_source_files,
            expected_source,
        )
        _require_exact(
            "tanager_spec package-data hashes linked to governing identity",
            package_data_files,
            expected_data,
        )
    return {
        "tanager_spec": {
            "classification": record["classification"],
            "hash_bound": record["hash_bound"],
            "editable_root_logical_path": record["editable_root_logical_path"],
            **inventory_payload,
            "inventory_sha256": digest,
        }
    }


def verify_mapping_run_bundle(
    bundle_root: Path,
    *,
    expected_run_id: str,
    expected_scenes: Mapping[str, str],
    expected_semantics: Mapping[str, Mapping[str, Any]],
    expected_protocol_sha256: str,
    expected_snapshot_evidence: Mapping[str, Any],
    expected_residual_dependency_trust: Mapping[str, Any],
) -> None:
    """Verify mapping files against bundle structure and an external semantic anchor."""
    run_manifest, _run_digest, _run_size, _run_manifest_path = _read_bundle_json(
        bundle_root,
        "mapping_run_manifest.json",
        label="mapping-run manifest",
    )
    _require_exact(
        "mapping-run manifest fields",
        set(run_manifest),
        {
            "schema_version",
            "manifest_type",
            "mode",
            "scientific_endpoint_values_inspected",
            "scientific_outputs_produced",
            "execution_id",
            "execution_identity_inputs",
            "scene_manifests",
            "input_snapshot_evidence",
            "residual_dependency_trust",
        },
    )
    _require_exact("mapping-run schema version", run_manifest.get("schema_version"), SCHEMA_VERSION)
    _require_exact(
        "mapping-run manifest type",
        run_manifest.get("manifest_type"),
        "m1b_basic_ortho_mapping_run",
    )
    _require_exact("mapping-run mode", run_manifest.get("mode"), "mapping-only")
    _require_exact("mapping-run execution ID", run_manifest.get("execution_id"), expected_run_id)
    identity_inputs = _require_object_fields(
        run_manifest.get("execution_identity_inputs"),
        label="mapping-run execution identity inputs",
        fields={
            "mode",
            "protocol_sha256",
            "acquisition_manifest_sha256",
            "ortho_manifest_sha256",
            "governing_files",
            "residual_dependency_trust",
            "input_sha256",
        },
    )
    _require_exact("mapping-run identity mode", identity_inputs.get("mode"), "mapping-only")
    _require_exact(
        "mapping-run externally sealed preregistration SHA-256",
        identity_inputs.get("protocol_sha256"),
        expected_protocol_sha256,
    )
    run_governing = validate_governing_files(
        identity_inputs.get("governing_files"),
        label="mapping-run identity governing files",
    )
    identity_dependency_trust = _validate_residual_dependency_trust(
        identity_inputs.get("residual_dependency_trust"),
        governing_files=run_governing,
    )
    _require_exact(
        "mapping-run recomputed execution ID",
        execution_identity(identity_inputs),
        expected_run_id,
    )
    _require_exact(
        "mapping-run endpoint inspection",
        run_manifest.get("scientific_endpoint_values_inspected"),
        False,
    )
    _require_exact(
        "mapping-run scientific outputs",
        run_manifest.get("scientific_outputs_produced"),
        False,
    )
    snapshot_evidence = _validate_input_snapshot_evidence(
        run_manifest.get("input_snapshot_evidence"),
        expected_scenes=expected_scenes,
    )
    _require_exact(
        "mapping-run external input snapshot evidence",
        snapshot_evidence,
        dict(expected_snapshot_evidence),
    )
    residual_dependency_trust = _validate_residual_dependency_trust(
        run_manifest.get("residual_dependency_trust"),
        governing_files=run_governing,
    )
    _require_exact(
        "mapping-run dependency trust identity linkage",
        residual_dependency_trust,
        identity_dependency_trust,
    )
    _require_exact(
        "mapping-run external residual dependency trust",
        residual_dependency_trust,
        _validate_residual_dependency_trust(
            dict(expected_residual_dependency_trust),
            governing_files=run_governing,
        ),
    )
    _require_exact(
        "mapping-run preregistration snapshot SHA-256",
        snapshot_evidence["preregistration"]["sha256"],
        expected_protocol_sha256,
    )
    _require_exact(
        "mapping-run acquisition snapshot SHA-256",
        snapshot_evidence["acquisition_manifest"]["sha256"],
        identity_inputs.get("acquisition_manifest_sha256"),
    )
    _require_exact(
        "mapping-run ortho-manifest snapshot SHA-256",
        snapshot_evidence["ortho_manifest"]["sha256"],
        identity_inputs.get("ortho_manifest_sha256"),
    )

    input_sha256 = _require_object_fields(
        identity_inputs.get("input_sha256"),
        label="mapping-run input SHA-256 records",
        fields=set(expected_scenes),
    )
    for site in expected_scenes:
        site_sha256 = _require_object_fields(
            input_sha256.get(site),
            label=f"{site} run-level input identity",
            fields={"basic", "ortho"},
        )
        for product in ("basic", "ortho"):
            _require_exact(
                f"{site} {product} snapshot SHA-256",
                snapshot_evidence["scenes"][site][product]["sha256"],
                site_sha256[product],
            )

    scene_records = _require_object_fields(
        run_manifest.get("scene_manifests"),
        label="mapping-run scene manifests",
        fields=set(expected_scenes),
    )
    _require_exact("mapping-run scene set", set(scene_records), set(expected_scenes))
    semantic_records = _require_object_fields(
        expected_semantics,
        label="external mapping semantic attestations",
        fields=set(expected_scenes),
    )
    expected_files = {"mapping_run_manifest.json"}
    for site, expected_scene_id in expected_scenes.items():
        record = _require_object_fields(
            scene_records[site],
            label=f"{site} mapping-run scene record",
            fields={"path", "sha256"},
        )
        relative_manifest = str(record.get("path", ""))
        scene_manifest, scene_digest, _scene_size, scene_manifest_path = _read_bundle_json(
            bundle_root,
            relative_manifest,
            label=f"{site} mapping manifest",
        )
        _require_exact(
            f"{site} mapping manifest SHA-256",
            scene_digest,
            record.get("sha256"),
        )
        _require_exact(
            f"{site} mapping manifest fields",
            set(scene_manifest),
            {
                "schema_version",
                "manifest_type",
                "mode",
                "scientific_endpoint_values_inspected",
                "scientific_outputs_produced",
                "execution_id",
                "execution_identity_inputs",
                "frozen_input",
                "grid",
                "source_accounting",
                "qa_invalid_counts_nonexclusive",
                "spectral_copy_audit",
                "mapping_distance_threshold_m",
                "multiplicity_basis",
                "mapping_status_codes",
                "artifacts",
            },
        )
        _require_exact(
            f"{site} mapping schema version",
            scene_manifest.get("schema_version"),
            SCHEMA_VERSION,
        )
        _require_exact(
            f"{site} mapping manifest type",
            scene_manifest.get("manifest_type"),
            MANIFEST_TYPE,
        )
        _require_exact(f"{site} mapping mode", scene_manifest.get("mode"), "mapping-only")
        _require_exact(
            f"{site} endpoint inspection",
            scene_manifest.get("scientific_endpoint_values_inspected"),
            False,
        )
        _require_exact(
            f"{site} scientific outputs",
            scene_manifest.get("scientific_outputs_produced"),
            False,
        )
        scene_identity_inputs = _require_object_fields(
            scene_manifest.get("execution_identity_inputs"),
            label=f"{site} mapping identity inputs",
            fields={
                "scientific_execution_identity",
                "mode",
                "site",
                "scene_id",
                "protocol_sha256",
                "acquisition_manifest_sha256",
                "ortho_manifest_sha256",
                "basic_sha256",
                "ortho_sha256",
                "governing_files",
                "residual_dependency_trust",
            },
        )
        scene_run_id = str(scene_manifest.get("execution_id", ""))
        _require_exact(
            f"{site} mapping manifest path",
            relative_manifest,
            f"{scene_run_id}/{site}/mapping_manifest.json",
        )
        _require_exact(
            f"{site} recomputed mapping execution ID",
            execution_identity(scene_identity_inputs),
            scene_run_id,
        )
        _require_exact(
            f"{site} scientific execution identity",
            scene_identity_inputs.get("scientific_execution_identity"),
            SCIENTIFIC_EXECUTION_IDENTITY,
        )
        _require_exact(f"{site} identity mode", scene_identity_inputs.get("mode"), "mapping-only")
        _require_exact(f"{site} identity site", scene_identity_inputs.get("site"), site)
        _require_exact(
            f"{site} identity scene",
            scene_identity_inputs.get("scene_id"),
            expected_scene_id,
        )
        for key in (
            "protocol_sha256",
            "acquisition_manifest_sha256",
            "ortho_manifest_sha256",
            "governing_files",
            "residual_dependency_trust",
        ):
            _require_exact(
                f"{site} identity {key}",
                scene_identity_inputs.get(key),
                identity_inputs.get(key),
            )
        validate_governing_files(
            scene_identity_inputs.get("governing_files"),
            label=f"{site} mapping identity governing files",
        )
        _validate_residual_dependency_trust(
            scene_identity_inputs.get("residual_dependency_trust"),
            governing_files=run_governing,
        )
        _require_exact(
            f"{site} mapping governing files",
            scene_identity_inputs.get("governing_files"),
            run_governing,
        )
        site_sha256 = _require_object_fields(
            input_sha256.get(site),
            label=f"{site} run-level input identity",
            fields={"basic", "ortho"},
        )
        _require_exact(
            f"{site} identity basic SHA-256",
            scene_identity_inputs.get("basic_sha256"),
            site_sha256.get("basic"),
        )
        _require_exact(
            f"{site} identity ortho SHA-256",
            scene_identity_inputs.get("ortho_sha256"),
            site_sha256.get("ortho"),
        )
        frozen_input = _require_object_fields(
            scene_manifest.get("frozen_input"),
            label=f"{site} frozen input record",
            fields=set(FrozenSceneInput.__dataclass_fields__),
        )
        _require_exact(f"{site} frozen site", frozen_input.get("site"), site)
        _require_exact(f"{site} frozen scene", frozen_input.get("scene_id"), expected_scene_id)
        _require_exact(
            f"{site} frozen basic SHA-256",
            frozen_input.get("basic_sha256"),
            scene_identity_inputs.get("basic_sha256"),
        )
        _require_exact(
            f"{site} frozen ortho SHA-256",
            frozen_input.get("ortho_sha256"),
            scene_identity_inputs.get("ortho_sha256"),
        )
        grid = _validated_scene_grid(scene_manifest, frozen_input)
        source_accounting = _require_object_fields(
            scene_manifest.get("source_accounting"),
            label=f"{site} source accounting",
            fields=set(MappingCounts.__dataclass_fields__),
        )
        for name, value in source_accounting.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(
                    f"{site} source accounting must be nonnegative integers: {name}"
                )
        qa_counts = _require_object_fields(
            scene_manifest.get("qa_invalid_counts_nonexclusive"),
            label=f"{site} QA invalid counts",
            fields={f"{geometry}:{name}" for geometry in ("basic", "ortho") for name in QA_FIELDS},
        )
        for name, value in qa_counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"{site} QA invalid count must be nonnegative: {name}")
            population_count = (
                source_accounting["invalid_qa_source_samples"]
                if name.startswith("basic:")
                else source_accounting["invalid_qa_target_cells"]
            )
            if value > population_count:
                raise ProtocolError(f"{site} QA field count exceeds its combined invalid support")
        spectral_copy = _validate_spectrum_copy_audit(
            scene_manifest.get("spectral_copy_audit"),
            label=f"{site} spectral-copy audit",
        )
        _require_exact(
            f"{site} mapping distance threshold",
            scene_manifest.get("mapping_distance_threshold_m"),
            None,
        )
        _require_exact(
            f"{site} multiplicity basis",
            scene_manifest.get("multiplicity_basis"),
            "target-QA-valid geometric assignments before basic-source QA no-calls",
        )
        _require_exact(
            f"{site} mapping status codes",
            scene_manifest.get("mapping_status_codes"),
            {
                "0": "reserved raster nodata",
                str(int(TARGET_ORTHO_QA_INVALID)): "ortho QA invalid",
                str(int(TARGET_NO_GEOLOCATED_SOURCE)): "no geolocated basic source available",
                str(int(TARGET_BASIC_QA_INVALID)): "nearest basic source QA invalid",
                str(int(TARGET_MAPPED)): "mapped to QA-valid basic source",
            },
        )
        artifacts = _require_object_fields(
            scene_manifest.get("artifacts"),
            label=f"{site} artifact records",
            fields=set(MAPPING_ARTIFACT_NAMES),
        )
        expected_files.add(relative_manifest)
        raster_arrays: dict[str, np.ndarray] = {}
        for artifact_name in MAPPING_ARTIFACT_NAMES:
            artifact_record = _require_object_fields(
                artifacts[artifact_name],
                label=f"{site} {artifact_name} artifact record",
                fields={"sha256", "size_bytes"},
            )
            if (
                not isinstance(artifact_record["sha256"], str)
                or isinstance(artifact_record["size_bytes"], bool)
                or not isinstance(artifact_record["size_bytes"], int)
                or artifact_record["size_bytes"] <= 0
            ):
                raise ProtocolError(f"{site} {artifact_name} artifact identity is malformed")
            artifact_path = _require_bundle_file(
                bundle_root,
                (Path(relative_manifest).parent / artifact_name).as_posix(),
            )
            raster_arrays[artifact_name] = _read_semantic_mapping_raster(
                artifact_path,
                artifact_name=artifact_name,
                artifact_record=artifact_record,
                grid=grid,
                scene_run_id=scene_run_id,
                scene_id=expected_scene_id,
            )
            expected_files.add(artifact_path.relative_to(bundle_root).as_posix())
        _validate_mapping_array_semantics(
            raster_arrays,
            frozen_input=frozen_input,
            source_accounting=source_accounting,
        )
        expected_attestation = _require_object_fields(
            semantic_records[site],
            label=f"{site} external mapping semantic attestation",
            fields={
                "arrays",
                "source_accounting",
                "qa_invalid_counts_nonexclusive",
                "spectral_copy_audit",
            },
        )
        expected_arrays = _require_object_fields(
            expected_attestation["arrays"],
            label=f"{site} external mapping semantic arrays",
            fields=set(MAPPING_ARTIFACT_NAMES),
        )
        for artifact_name in MAPPING_ARTIFACT_NAMES:
            expected_array = expected_arrays[artifact_name]
            if not isinstance(expected_array, np.ndarray):
                raise ProtocolError(f"{site} external semantic array is malformed: {artifact_name}")
            observed_array = raster_arrays[artifact_name]
            if (
                expected_array.shape != observed_array.shape
                or expected_array.dtype != observed_array.dtype
            ):
                raise ProtocolError(
                    f"{site} externally attested mapping schema differs for {artifact_name}"
                )
            equal = np.array_equal(
                observed_array,
                expected_array,
                equal_nan=np.issubdtype(observed_array.dtype, np.floating),
            )
            if not equal:
                raise ProtocolError(
                    f"{site} externally attested mapping semantics differ for {artifact_name}"
                )
        _require_exact(
            f"{site} externally attested source accounting",
            source_accounting,
            expected_attestation["source_accounting"],
        )
        _require_exact(
            f"{site} externally attested QA invalid counts",
            qa_counts,
            expected_attestation["qa_invalid_counts_nonexclusive"],
        )
        expected_spectral_copy = _validate_spectrum_copy_audit(
            expected_attestation["spectral_copy_audit"],
            label=f"{site} external spectral-copy attestation",
        )
        _require_exact(
            f"{site} externally attested spectral-copy audit",
            spectral_copy,
            expected_spectral_copy,
        )
    _require_exact("mapping-run complete file set", _bundle_file_set(bundle_root), expected_files)


__all__ = [
    "FROZEN_ACQUISITION_MANIFEST_SHA256",
    "FROZEN_ORTHO_MANIFEST_SHA256",
    "FROZEN_PREREGISTRATION_SHA256",
    "FROZEN_SCENES",
    "GOVERNING_FILE_KEYS",
    "MAPPING_ARTIFACT_NAMES",
    "CleanupResidueError",
    "MappingCounts",
    "NativeToOrthoMapping",
    "OrthoGrid",
    "ProductSchema",
    "ProtocolError",
    "RESOURCE_PILOT_BRANCH_ASSETS",
    "RESOURCE_PILOT_DEFAULT_BRANCH",
    "RESOURCE_PILOT_DEFAULT_SITE",
    "RESOURCE_PILOT_EXECUTION_IDENTITY",
    "ResourcePilotTelemetry",
    "SCIENTIFIC_EXECUTION_IDENTITY",
    "TARGET_BASIC_QA_INVALID",
    "TARGET_MAPPED",
    "TARGET_NO_GEOLOCATED_SOURCE",
    "TARGET_ORTHO_QA_INVALID",
    "TANAGER_SPEC_EDITABLE_LOGICAL_ROOT",
    "TANAGER_SPEC_MODULE_FILES",
    "TANAGER_SPEC_PACKAGE_DATA_FILES",
    "SpectrumCopyAudit",
    "ValidatedInputs",
    "atomic_write_run_bundle",
    "canonical_band_y_x",
    "design_document",
    "exact_spectrum_copy_audit",
    "execution_identity",
    "inspect_basic_schema",
    "inspect_ortho_schema",
    "load_basic_geolocation_and_qa",
    "load_ortho_qa",
    "load_ortho_grid",
    "map_native_to_ortho",
    "measure_resource_pilot_load",
    "project_scalar_nearest",
    "resource_pilot_document",
    "resource_pilot_execution_identity",
    "resource_pilot_identity_inputs",
    "schema_document",
    "select_resource_pilot_scene",
    "sha256_file",
    "strict_json_dump",
    "tanager_spec_dependency_trust",
    "validate_frozen_inputs",
    "validate_governing_files",
    "validate_protocol_file",
    "validate_schema_pair",
    "verify_mapping_run_bundle",
    "verify_resource_pilot_bundle",
    "write_mapping_bundle",
]
