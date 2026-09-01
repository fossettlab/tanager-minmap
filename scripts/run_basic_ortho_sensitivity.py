"""Materialize preregistered M1b pre-result design, schema, mapping, or pilot artifacts.

No mode in this script computes or exposes a scientific sensitivity endpoint.
Project modules are loaded only after their exact source bytes have been bound
into a private import capsule.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, BinaryIO


def _descriptor_bound_runner_bootstrap() -> tuple[
    Path, bytes, str, tuple[int, int, int, int, int], Path
]:
    """Accept only bytes supplied to ``compile`` by the residual-trust launcher."""
    namespace = globals()
    required = {
        "_DESCRIPTOR_BOUND_RUNNER_PATH",
        "_DESCRIPTOR_BOUND_RUNNER_SOURCE",
        "_DESCRIPTOR_BOUND_RUNNER_SHA256",
        "_DESCRIPTOR_BOUND_RUNNER_STAT",
        "_DESCRIPTOR_BOUND_LAUNCHER_PATH",
    }
    missing = sorted(required - namespace.keys())
    if missing:
        raise RuntimeError(
            "runner requires descriptor-bound execution through "
            "scripts/run_basic_ortho_sensitivity_launcher.py; "
            f"missing bootstrap fields: {missing}"
        )

    raw_path = namespace["_DESCRIPTOR_BOUND_RUNNER_PATH"]
    payload = namespace["_DESCRIPTOR_BOUND_RUNNER_SOURCE"]
    payload_sha256 = namespace["_DESCRIPTOR_BOUND_RUNNER_SHA256"]
    payload_stat = namespace["_DESCRIPTOR_BOUND_RUNNER_STAT"]
    raw_launcher_path = namespace["_DESCRIPTOR_BOUND_LAUNCHER_PATH"]
    if not isinstance(raw_path, str) or not isinstance(payload, bytes):
        raise RuntimeError("descriptor-bound runner path or payload has an invalid type")
    if not isinstance(payload_sha256, str) or hashlib.sha256(payload).hexdigest() != payload_sha256:
        raise RuntimeError("descriptor-bound runner payload hash is inconsistent")
    if (
        not isinstance(payload_stat, tuple)
        or len(payload_stat) != 5
        or not all(isinstance(value, int) for value in payload_stat)
    ):
        raise RuntimeError("descriptor-bound runner stat identity is invalid")
    if not isinstance(raw_launcher_path, str):
        raise RuntimeError("descriptor-bound launcher path has an invalid type")

    path = Path(os.path.abspath(raw_path))
    expected_path = Path(os.path.abspath(__file__))
    if path != expected_path:
        raise RuntimeError(
            f"descriptor-bound runner path differs from __file__: {path} != {expected_path}"
        )
    launcher_path = Path(os.path.abspath(raw_launcher_path))
    expected_launcher = path.with_name("run_basic_ortho_sensitivity_launcher.py")
    if launcher_path != expected_launcher:
        raise RuntimeError(
            "descriptor-bound runner was not loaded by its dedicated residual-trust launcher"
        )
    return path, payload, payload_sha256, payload_stat, launcher_path


(
    _BOOTSTRAP_RUNNER_PATH,
    _BOOTSTRAP_RUNNER_SOURCE,
    _BOOTSTRAP_RUNNER_SHA256,
    _BOOTSTRAP_RUNNER_STAT,
    _BOOTSTRAP_LAUNCHER_PATH,
) = _descriptor_bound_runner_bootstrap()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = ROOT / "docs" / "m1b_basic_ortho_sensitivity_preregistration.md"
DEFAULT_ACQUISITION_MANIFEST = ROOT / "docs" / "basic_ortho_acquisition_manifest.json"
DEFAULT_ORTHO_MANIFEST = ROOT / "docs" / "input_manifest.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "basic_ortho_sensitivity"

LAUNCHER_RESIDUAL_TRUST = {
    "classification": "residual_execution_bootstrap_trust",
    "hash_bound": False,
    "path": str(_BOOTSTRAP_LAUNCHER_PATH),
    "risk": "Python loads the minimal launcher before descriptor-bound runner handoff",
}

_PROJECT_MODULE_FILES = {
    "": "src/tanager_minmap/__init__.py",
    ".basic_ortho": "src/tanager_minmap/basic_ortho.py",
    ".config": "src/tanager_minmap/config.py",
    ".features": "src/tanager_minmap/features.py",
    ".quality": "src/tanager_minmap/quality.py",
    ".speclib": "src/tanager_minmap/speclib.py",
    ".unmix": "src/tanager_minmap/unmix.py",
    ".viz": "src/tanager_minmap/viz.py",
}
_TANAGER_SPEC_EDITABLE_LOGICAL_ROOT = "../tanager-spec"
_TANAGER_SPEC_PACKAGE_LOGICAL_ROOT = "../tanager-spec/src/tanager_spec"
_TANAGER_SPEC_MODULE_FILES = {
    "tanager_spec": "../tanager-spec/src/tanager_spec/__init__.py",
    "tanager_spec.bands": "../tanager-spec/src/tanager_spec/bands.py",
    "tanager_spec.config": "../tanager-spec/src/tanager_spec/config.py",
    "tanager_spec.io": "../tanager-spec/src/tanager_spec/io.py",
    "tanager_spec.mask": "../tanager-spec/src/tanager_spec/mask.py",
    "tanager_spec.sample": "../tanager-spec/src/tanager_spec/sample.py",
    "tanager_spec.srf": "../tanager-spec/src/tanager_spec/srf.py",
    "tanager_spec.stac": "../tanager-spec/src/tanager_spec/stac.py",
}
_TANAGER_SPEC_PACKAGE_DATA_FILES = (
    "../tanager-spec/src/tanager_spec/data/S2A_SRF.csv",
    "../tanager-spec/src/tanager_spec/data/S2B_SRF.csv",
    "../tanager-spec/src/tanager_spec/data/SOURCE.md",
)
_TANAGER_SPEC_PACKAGE_DIRECTORIES = ("data",)
_TANAGER_SPEC_RESOURCE_PACKAGE = "tanager_spec.data"
_RUNNER_LOGICAL_PATH = "scripts/run_basic_ortho_sensitivity.py"
GOVERNING_FILES = tuple(
    sorted(
        {
            _RUNNER_LOGICAL_PATH,
            *_PROJECT_MODULE_FILES.values(),
            *_TANAGER_SPEC_MODULE_FILES.values(),
            *_TANAGER_SPEC_PACKAGE_DATA_FILES,
        }
    )
)
_FROZEN_SCENES = {
    "goldfield": "20240925_185504_87_4001",
    "bingham": "20250911_191523_58_4001",
}
_QA_FIELDS = ("beta_cloud_mask", "beta_cirrus_mask", "nodata_pixels")
_QA_ALLOWED_VALUES = frozenset({0, 1, 255})
_TARGET_ORTHO_QA_INVALID = 1
_TARGET_NO_GEOLOCATED_SOURCE = 2
_TARGET_BASIC_QA_INVALID = 3
_TARGET_MAPPED = 4
_STRUCT_METADATA_PATH = "HDFEOS INFORMATION/StructMetadata.0"
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


class BootstrapError(RuntimeError):
    """Raised when local source binding or capsule inventory is not exact."""


class IndependentVerificationError(ValueError):
    """Raised when frozen inputs cannot support an independent mapping attestation."""


@dataclass(frozen=True)
class _CapturedRuntime:
    root: Path
    source_bytes: dict[str, bytes]
    governing_hashes: dict[str, str]
    dependency_inventory: tuple[str, ...]
    dependency_directories: tuple[str, ...]


@dataclass(frozen=True)
class _RuntimeBinding:
    basic_ortho: ModuleType
    capture: _CapturedRuntime
    finder: _SealedModuleFinder
    prefix: str
    modules_before: frozenset[str]
    canonical_modules: dict[str, ModuleType | None]
    canonical_dependency_modules: dict[str, ModuleType | None]
    sibling_modules_before: dict[str, ModuleType]
    loaded_local_files: tuple[str, ...]
    loaded_dependency_files: tuple[str, ...]
    new_local_module_origins: tuple[tuple[str, str], ...]
    residual_dependency_trust: dict[str, Any]

    @property
    def governing_hashes(self) -> dict[str, str]:
        return dict(self.capture.governing_hashes)


def _open_regular_nofollow(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise BootstrapError(f"{label} requires O_NOFOLLOW and O_DIRECTORY support")
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise BootstrapError(f"{label} path is not normalized and absolute: {path}")
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
            raise BootstrapError(f"{label} is not a regular file: {path}")
        return file_fd, info
    except OSError as error:
        raise BootstrapError(
            f"{label} cannot be opened without following symlinks: {path}"
        ) from error
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _file_stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _read_bound_source_identity(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    file_fd, initial_info = _open_regular_nofollow(path, label=label)
    with os.fdopen(file_fd, "rb") as handle:
        payload = handle.read()
        final_info = os.fstat(handle.fileno())
    initial = _file_stat_identity(initial_info)
    final = _file_stat_identity(final_info)
    if initial != final or len(payload) != initial_info.st_size:
        raise BootstrapError(f"{label} changed while its source bytes were bound: {path}")
    return payload, final


def _read_bound_source(path: Path, *, label: str) -> bytes:
    return _read_bound_source_identity(path, label=label)[0]


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    """Open an absolute directory path without following any symlink component."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise BootstrapError(f"{label} requires O_NOFOLLOW and O_DIRECTORY support")
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise BootstrapError(f"{label} path is not normalized and absolute: {path}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fds: list[int] = []
    try:
        directory_fd = os.open(absolute.anchor, flags)
        directory_fds.append(directory_fd)
        for part in absolute.parts[1:]:
            directory_fd = os.open(part, flags, dir_fd=directory_fd)
            directory_fds.append(directory_fd)
        result_fd = directory_fds.pop()
        return result_fd
    except OSError as error:
        raise BootstrapError(
            f"{label} cannot be opened without following symlinks: {path}"
        ) from error
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _scan_dependency_directory(
    directory_fd: int,
    *,
    relative_parts: tuple[str, ...] = (),
) -> tuple[set[str], set[str]]:
    """Return one stable, no-follow sibling package inventory from an open directory."""
    initial = _file_stat_identity(os.fstat(directory_fd))
    files: set[str] = set()
    directories: set[str] = set()
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as error:
        raise BootstrapError("tanager_spec package inventory cannot be listed") from error
    for name in names:
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise BootstrapError(
                f"tanager_spec package inventory entry cannot be inspected: {name}"
            ) from error
        relative = PurePosixPath(*relative_parts, name).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise BootstrapError(f"tanager_spec package inventory contains a symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            if name == "__pycache__":
                continue
            try:
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise BootstrapError(
                    f"tanager_spec package directory cannot be opened safely: {relative}"
                ) from error
            try:
                child_info = os.fstat(child_fd)
                if (child_info.st_dev, child_info.st_ino) != (info.st_dev, info.st_ino):
                    raise BootstrapError(
                        f"tanager_spec package directory changed during inventory: {relative}"
                    )
                child_files, child_directories = _scan_dependency_directory(
                    child_fd,
                    relative_parts=(*relative_parts, name),
                )
            finally:
                os.close(child_fd)
            directories.add(relative)
            directories.update(child_directories)
            files.update(child_files)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise BootstrapError(
                f"tanager_spec package inventory entry is not a regular file: {relative}"
            )
        files.add(relative)
    final = _file_stat_identity(os.fstat(directory_fd))
    if initial != final:
        display = PurePosixPath(*relative_parts).as_posix() if relative_parts else "."
        raise BootstrapError(f"tanager_spec package directory changed during inventory: {display}")
    return files, directories


def _dependency_package_root(root: Path) -> Path:
    return Path(os.path.abspath(root / _TANAGER_SPEC_PACKAGE_LOGICAL_ROOT))


def _dependency_editable_root(root: Path) -> Path:
    return Path(os.path.abspath(root / _TANAGER_SPEC_EDITABLE_LOGICAL_ROOT))


def _observed_dependency_inventory(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    package_root = _dependency_package_root(root)
    directory_fd = _open_directory_nofollow(
        package_root,
        label="tanager_spec package root",
    )
    try:
        relative_files, relative_directories = _scan_dependency_directory(directory_fd)
    finally:
        os.close(directory_fd)
    logical_files = tuple(
        sorted(f"{_TANAGER_SPEC_PACKAGE_LOGICAL_ROOT}/{path}" for path in relative_files)
    )
    return logical_files, tuple(sorted(relative_directories))


def _require_exact_dependency_inventory(
    observed_files: tuple[str, ...],
    observed_directories: tuple[str, ...],
) -> None:
    expected_files = tuple(
        sorted({*_TANAGER_SPEC_MODULE_FILES.values(), *_TANAGER_SPEC_PACKAGE_DATA_FILES})
    )
    if (
        observed_files != expected_files
        or observed_directories != _TANAGER_SPEC_PACKAGE_DIRECTORIES
    ):
        raise BootstrapError(
            "tanager_spec package inventory differs from the frozen source/data closure: "
            f"expected files {expected_files!r} and directories "
            f"{_TANAGER_SPEC_PACKAGE_DIRECTORIES!r}, observed files {observed_files!r} "
            f"and directories {observed_directories!r}"
        )


def _capture_runtime_sources(root: Path = ROOT) -> _CapturedRuntime:
    source_bytes: dict[str, bytes] = {}
    governing_hashes: dict[str, str] = {}
    resolved_root = root.resolve()
    dependency_inventory, dependency_directories = _observed_dependency_inventory(resolved_root)
    _require_exact_dependency_inventory(dependency_inventory, dependency_directories)
    for logical_path in GOVERNING_FILES:
        if logical_path == _RUNNER_LOGICAL_PATH and resolved_root == ROOT.resolve():
            expected_path = resolved_root / logical_path
            if _BOOTSTRAP_RUNNER_PATH != expected_path:
                raise BootstrapError(
                    "runner bootstrap source path differs from the governing runner path"
                )
            observed_payload, observed_stat = _read_bound_source_identity(
                expected_path,
                label="descriptor-bound runner source recheck",
            )
            if (
                observed_stat != _BOOTSTRAP_RUNNER_STAT
                or observed_payload != _BOOTSTRAP_RUNNER_SOURCE
                or hashlib.sha256(observed_payload).hexdigest() != _BOOTSTRAP_RUNNER_SHA256
            ):
                raise BootstrapError("runner source pathname changed after descriptor-bound launch")
            payload = _BOOTSTRAP_RUNNER_SOURCE
        else:
            payload = _read_bound_source(
                root / logical_path,
                label=f"governing source {logical_path}",
            )
        source_bytes[logical_path] = payload
        governing_hashes[logical_path] = hashlib.sha256(payload).hexdigest()
    final_inventory, final_directories = _observed_dependency_inventory(resolved_root)
    if final_inventory != dependency_inventory or final_directories != dependency_directories:
        raise BootstrapError("tanager_spec package inventory changed while bytes were captured")
    return _CapturedRuntime(
        root=resolved_root,
        source_bytes=source_bytes,
        governing_hashes=governing_hashes,
        dependency_inventory=dependency_inventory,
        dependency_directories=dependency_directories,
    )


class _SealedTraversable(Traversable):
    """Read-only in-memory package tree backed only by captured dependency bytes."""

    def __init__(
        self,
        finder: _SealedModuleFinder,
        parts: tuple[str, ...],
        *,
        root_name: str,
    ) -> None:
        self.finder = finder
        self.parts = parts
        self.root_name = root_name

    @property
    def name(self) -> str:
        return self.parts[-1] if self.parts else self.root_name

    def _key(self) -> str:
        return PurePosixPath(*self.parts).as_posix() if self.parts else ""

    def iterdir(self) -> Iterator[Traversable]:
        if not self.is_dir():
            raise FileNotFoundError(self._key())
        prefix = f"{self._key()}/" if self.parts else ""
        child_names: set[str] = set()
        for resource_path in self.finder.dependency_resource_bytes:
            if not resource_path.startswith(prefix):
                continue
            remainder = resource_path[len(prefix) :]
            if remainder:
                child_names.add(remainder.split("/", 1)[0])
        for child_name in sorted(child_names):
            yield self.joinpath(child_name)

    def is_dir(self) -> bool:
        if not self.finder.active:
            return False
        key = self._key()
        if not key:
            return True
        prefix = f"{key}/"
        return any(path.startswith(prefix) for path in self.finder.dependency_resource_bytes)

    def is_file(self) -> bool:
        return self.finder.active and self._key() in self.finder.dependency_resource_bytes

    def joinpath(self, *descendants: str) -> _SealedTraversable:
        parts = list(self.parts)
        for descendant in descendants:
            relative = PurePosixPath(descendant)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe sealed resource path: {descendant!r}")
            parts.extend(part for part in relative.parts if part != ".")
        return type(self)(self.finder, tuple(parts), root_name=self.root_name)

    def open(self, mode: str = "r", *args: Any, **kwargs: Any) -> BinaryIO:
        if not self.finder.active:
            raise BootstrapError("sealed dependency resource used after capsule teardown")
        if mode not in {"r", "rb"}:
            raise ValueError(f"sealed dependency resources are read-only: {mode!r}")
        key = self._key()
        try:
            payload = self.finder.dependency_resource_bytes[key]
            logical_path = self.finder.dependency_resource_logical_paths[key]
        except KeyError as error:
            raise FileNotFoundError(key) from error
        self.finder.accessed_dependency_files.add(logical_path)
        stream = io.BytesIO(payload)
        if mode == "rb":
            return stream
        return io.TextIOWrapper(stream, *args, **kwargs)  # type: ignore[return-value]


class _SealedResourceReader(importlib.abc.ResourceReader):
    """Expose captured package data without falling back to the editable tree."""

    def __init__(self, finder: _SealedModuleFinder, package: str) -> None:
        self.finder = finder
        self.package = package

    def files(self) -> Traversable:
        parts = self.finder.resource_package_roots[self.package]
        return _SealedTraversable(
            self.finder,
            parts,
            root_name=self.package.rsplit(".", 1)[-1],
        )

    def open_resource(self, resource: str) -> BinaryIO:
        return self.files().joinpath(resource).open("rb")

    def resource_path(self, resource: str) -> str:
        raise FileNotFoundError(
            f"sealed dependency resource has no filesystem path: {self.package}:{resource}"
        )

    def is_resource(self, path: str) -> bool:
        return self.files().joinpath(path).is_file()

    def contents(self) -> Iterator[str]:
        return (child.name for child in self.files().iterdir())


class _SealedSourceLoader(importlib.abc.Loader):
    def __init__(
        self,
        finder: _SealedModuleFinder,
        fullname: str,
        logical_path: str | None,
        payload: bytes | None,
        origin: Path,
        *,
        is_package: bool,
    ) -> None:
        self.finder = finder
        self.fullname = fullname
        self.logical_path = logical_path
        self.payload = payload
        self.origin = origin
        self.is_package = is_package

    def create_module(self, _spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def get_resource_reader(self, fullname: str) -> _SealedResourceReader | None:
        if fullname not in self.finder.resource_package_roots:
            return None
        return _SealedResourceReader(self.finder, fullname)

    def exec_module(self, module: ModuleType) -> None:
        if not self.finder.active:
            raise BootstrapError("sealed module loader used after capsule teardown")
        if self.finder.before_exec is not None:
            self.finder.before_exec(self.fullname)
        module.__file__ = str(self.origin)
        if self.is_package:
            package_root = self.origin if self.payload is None else self.origin.parent
            module.__path__ = [str(package_root)]
        capsule_builtins = dict(vars(builtins))
        capsule_builtins["__import__"] = self.finder.guarded_import
        module.__dict__["__builtins__"] = capsule_builtins
        try:
            if self.logical_path is not None and self.payload is not None:
                expected_digest = self.finder.capture.governing_hashes[self.logical_path]
                if hashlib.sha256(self.payload).hexdigest() != expected_digest:
                    raise BootstrapError(
                        "captured module payload hash changed before execution: "
                        f"{self.logical_path}"
                    )
                code = compile(self.payload, str(self.origin), "exec", dont_inherit=True)
                exec(code, module.__dict__)
                self.finder.executed_files.add(self.logical_path)
            self.finder.executed_modules.add(self.fullname)
        finally:
            if self.finder.after_exec is not None:
                self.finder.after_exec(self.fullname)


class _SealedModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        capture: _CapturedRuntime,
        prefix: str,
        *,
        before_exec: Callable[[str], None] | None = None,
        after_exec: Callable[[str], None] | None = None,
    ) -> None:
        self.capture = capture
        self.prefix = prefix
        self.before_exec = before_exec
        self.after_exec = after_exec
        self.executed_files: set[str] = set()
        self.executed_modules: set[str] = set()
        self.accessed_dependency_files: set[str] = set()
        self.blocked_imports: set[str] = set()
        self.blocked_canonical_imports: set[str] = set()
        self.blocked_dependency_imports: set[str] = set()
        self.blocked_out_of_capsule_dependency_modules: set[str] = set()
        self.active = True
        self.project_module_records = {
            f"{prefix}{suffix}": (
                logical_path,
                capture.source_bytes[logical_path],
                Path(os.path.abspath(capture.root / logical_path)),
                suffix == "",
            )
            for suffix, logical_path in _PROJECT_MODULE_FILES.items()
        }
        self.dependency_module_records = {
            module_name: (
                logical_path,
                capture.source_bytes[logical_path],
                Path(os.path.abspath(capture.root / logical_path)),
                module_name == "tanager_spec",
            )
            for module_name, logical_path in _TANAGER_SPEC_MODULE_FILES.items()
        }
        self.dependency_module_records[_TANAGER_SPEC_RESOURCE_PACKAGE] = (
            None,
            None,
            _dependency_package_root(capture.root) / "data",
            True,
        )
        self.module_records = {
            **self.project_module_records,
            **self.dependency_module_records,
        }
        package_prefix = f"{_TANAGER_SPEC_PACKAGE_LOGICAL_ROOT}/"
        self.dependency_resource_bytes = {
            logical_path.removeprefix(package_prefix): capture.source_bytes[logical_path]
            for logical_path in (
                *_TANAGER_SPEC_MODULE_FILES.values(),
                *_TANAGER_SPEC_PACKAGE_DATA_FILES,
            )
        }
        self.dependency_resource_logical_paths = {
            logical_path.removeprefix(package_prefix): logical_path
            for logical_path in (
                *_TANAGER_SPEC_MODULE_FILES.values(),
                *_TANAGER_SPEC_PACKAGE_DATA_FILES,
            )
        }
        self.resource_package_roots = {
            "tanager_spec": (),
            _TANAGER_SPEC_RESOURCE_PACKAGE: ("data",),
        }

    def _is_managed_dependency_module(self, name: str, module: object) -> bool:
        if name not in self.dependency_module_records or not isinstance(module, ModuleType):
            return False
        spec = getattr(module, "__spec__", None)
        loader = getattr(spec, "loader", None)
        return (
            isinstance(loader, _SealedSourceLoader)
            and loader.finder is self
            and loader.fullname == name
        )

    def _reject_out_of_capsule_dependency_modules(self) -> None:
        violations = {
            name
            for name, module in tuple(sys.modules.items())
            if _is_canonical_dependency_module(name)
            and not self._is_managed_dependency_module(name, module)
        }
        if violations:
            self.blocked_out_of_capsule_dependency_modules.update(violations)
            raise ModuleNotFoundError(
                f"out-of-capsule tanager_spec module blocked: {sorted(violations)}"
            )

    def guarded_import(
        self,
        name: str,
        globals_: Mapping[str, Any] | None = None,
        locals_: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        """Reject canonical project and out-of-capsule dependency imports."""
        if level == 0 and (name == "tanager_minmap" or name.startswith("tanager_minmap.")):
            self.blocked_canonical_imports.add(name)
            raise ModuleNotFoundError(f"canonical tanager_minmap import blocked in capsule: {name}")
        if level == 0 and _is_canonical_dependency_module(name):
            if name not in self.dependency_module_records:
                self.blocked_dependency_imports.add(name)
                raise ModuleNotFoundError(f"ungoverned tanager_spec import blocked: {name}")
            self._reject_out_of_capsule_dependency_modules()
        return builtins.__import__(name, globals_, locals_, fromlist, level)

    def find_spec(
        self,
        fullname: str,
        _path: Sequence[str] | None,
        _target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not self.active:
            return None
        if self.active and (fullname == "tanager_minmap" or fullname.startswith("tanager_minmap.")):
            self.blocked_canonical_imports.add(fullname)
            raise ModuleNotFoundError(
                f"canonical tanager_minmap import blocked in capsule: {fullname}"
            )
        if _is_canonical_dependency_module(fullname):
            self._reject_out_of_capsule_dependency_modules()
            record = self.dependency_module_records.get(fullname)
            if record is None:
                self.blocked_dependency_imports.add(fullname)
                raise ModuleNotFoundError(f"ungoverned tanager_spec import blocked: {fullname}")
        elif fullname == self.prefix or fullname.startswith(f"{self.prefix}."):
            record = self.project_module_records.get(fullname)
            if record is None:
                self.blocked_imports.add(fullname)
                raise ModuleNotFoundError(f"ungoverned capsule-local import blocked: {fullname}")
        else:
            return None
        logical_path, payload, origin, is_package = record
        loader = _SealedSourceLoader(
            self,
            fullname,
            logical_path,
            payload,
            origin,
            is_package=is_package,
        )
        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=str(origin),
            is_package=is_package,
        )


_CAPSULE_SERIAL = 0


def _next_capsule_prefix(capture: _CapturedRuntime) -> str:
    global _CAPSULE_SERIAL
    _CAPSULE_SERIAL += 1
    root_id = hashlib.sha256(str(capture.root).encode("utf-8")).hexdigest()[:12]
    return f"_tanager_minmap_m1b_{root_id}_{_CAPSULE_SERIAL}"


def _module_origin(module: ModuleType) -> Path | None:
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str) or not raw_origin:
        return None
    try:
        return Path(raw_origin).resolve()
    except OSError:
        return None


def _is_residual_launcher_module(name: str, origin: Path) -> bool:
    """Recognize only standard main-module aliases at the declared launcher origin."""
    return name in {"__main__", "__mp_main__"} and origin == _BOOTSTRAP_LAUNCHER_PATH.resolve()


def _new_repo_module_origins(
    modules_before: set[str] | frozenset[str],
    *,
    root: Path,
) -> tuple[tuple[str, str], ...]:
    records: list[tuple[str, str]] = []
    for name in sorted(set(sys.modules) - modules_before):
        module = sys.modules.get(name)
        if not isinstance(module, ModuleType):
            continue
        origin = _module_origin(module)
        if origin is None:
            continue
        if _is_residual_launcher_module(name, origin):
            continue
        try:
            relative = origin.relative_to(root)
        except ValueError:
            continue
        # The project-local virtual environment is dependency installation state,
        # not repository source. Repository source origins remain closed below.
        if relative.parts and relative.parts[0] == ".venv":
            continue
        records.append((name, relative.as_posix()))
    return tuple(records)


def _is_canonical_project_module(name: str) -> bool:
    return name == "tanager_minmap" or name.startswith("tanager_minmap.")


def _is_canonical_dependency_module(name: str) -> bool:
    return name == "tanager_spec" or name.startswith("tanager_spec.")


def _canonical_project_modules() -> dict[str, ModuleType | None]:
    return {
        name: module
        for name, module in tuple(sys.modules.items())
        if _is_canonical_project_module(name)
    }


def _canonical_dependency_modules() -> dict[str, ModuleType | None]:
    return {
        name: module
        for name, module in tuple(sys.modules.items())
        if _is_canonical_dependency_module(name)
    }


def _sibling_local_modules(root: Path) -> dict[str, ModuleType]:
    records: dict[str, ModuleType] = {}
    for name, module in tuple(sys.modules.items()):
        if not isinstance(module, ModuleType):
            continue
        origin = _module_origin(module)
        if origin is None:
            continue
        try:
            origin.relative_to(root)
        except ValueError:
            continue
        records[name] = module
    return records


def _sibling_origin_violations(binding: _RuntimeBinding) -> tuple[tuple[str, str], ...]:
    editable_root = _dependency_editable_root(binding.capture.root)
    violations: list[tuple[str, str]] = []
    for name, module in sorted(_sibling_local_modules(editable_root).items()):
        if binding.finder._is_managed_dependency_module(name, module):
            continue
        if binding.sibling_modules_before.get(name) is module:
            continue
        origin = _module_origin(module)
        if origin is None:
            violations.append((name, "<missing>"))
            continue
        violations.append((name, origin.relative_to(editable_root).as_posix()))
    return tuple(violations)


def _release_runtime_capsule_state(
    *,
    finder: _SealedModuleFinder,
    prefix: str,
    capture: _CapturedRuntime,
    modules_before: frozenset[str],
    canonical_modules: dict[str, ModuleType | None],
    canonical_dependency_modules: dict[str, ModuleType | None],
    sibling_modules_before: dict[str, ModuleType],
) -> None:
    """Remove one capsule and restore exact pre-capsule project/dependency modules."""
    finder.active = False
    while finder in sys.meta_path:
        sys.meta_path.remove(finder)
    editable_root = _dependency_editable_root(capture.root)
    for name in tuple(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]
            continue
        if _is_canonical_project_module(name):
            del sys.modules[name]
            continue
        if _is_canonical_dependency_module(name):
            del sys.modules[name]
            continue
        module = sys.modules.get(name)
        if isinstance(module, ModuleType):
            origin = _module_origin(module)
            if origin is not None:
                try:
                    origin.relative_to(editable_root)
                except ValueError:
                    pass
                else:
                    if sibling_modules_before.get(name) is not module:
                        del sys.modules[name]
                    continue
        if name in modules_before:
            continue
        if not isinstance(module, ModuleType):
            continue
        origin = _module_origin(module)
        if origin is None:
            continue
        if _is_residual_launcher_module(name, origin):
            continue
        try:
            relative = origin.relative_to(capture.root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] == ".venv":
            continue
        del sys.modules[name]
    sys.modules.update(canonical_modules)
    sys.modules.update(canonical_dependency_modules)
    sys.modules.update(sibling_modules_before)


def _load_runtime_capsule(
    capture: _CapturedRuntime,
    *,
    before_exec: Callable[[str], None] | None = None,
    after_exec: Callable[[str], None] | None = None,
) -> _RuntimeBinding:
    prefix = _next_capsule_prefix(capture)
    finder = _SealedModuleFinder(
        capture,
        prefix,
        before_exec=before_exec,
        after_exec=after_exec,
    )
    modules_before = frozenset(sys.modules)
    canonical_modules = _canonical_project_modules()
    canonical_dependency_modules = _canonical_dependency_modules()
    sibling_modules_before = _sibling_local_modules(_dependency_editable_root(capture.root))
    for name in {*canonical_modules, *canonical_dependency_modules}:
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    try:
        importlib.import_module(f"{prefix}.basic_ortho")
        importlib.import_module(_TANAGER_SPEC_RESOURCE_PACKAGE)
        basic_ortho = sys.modules[f"{prefix}.basic_ortho"]
        loaded_files = tuple(
            sorted(set(finder.executed_files) & set(_PROJECT_MODULE_FILES.values()))
        )
        loaded_dependency_files = tuple(
            sorted(set(finder.executed_files) & set(_TANAGER_SPEC_MODULE_FILES.values()))
        )
        new_local_origins = _new_repo_module_origins(modules_before, root=capture.root)
        dependency_trust = basic_ortho.tanager_spec_dependency_trust(capture.governing_hashes)
        binding = _RuntimeBinding(
            basic_ortho=basic_ortho,
            capture=capture,
            finder=finder,
            prefix=prefix,
            modules_before=modules_before,
            canonical_modules=canonical_modules,
            canonical_dependency_modules=canonical_dependency_modules,
            sibling_modules_before=sibling_modules_before,
            loaded_local_files=loaded_files,
            loaded_dependency_files=loaded_dependency_files,
            new_local_module_origins=new_local_origins,
            residual_dependency_trust=dependency_trust,
        )
        _validate_runtime_binding(binding)
        return binding
    except BaseException:
        _release_runtime_capsule_state(
            finder=finder,
            prefix=prefix,
            capture=capture,
            modules_before=modules_before,
            canonical_modules=canonical_modules,
            canonical_dependency_modules=canonical_dependency_modules,
            sibling_modules_before=sibling_modules_before,
        )
        raise


def _observed_governing_hashes(binding: _RuntimeBinding) -> dict[str, str]:
    return {
        logical_path: hashlib.sha256(
            _read_bound_source(
                binding.capture.root / logical_path,
                label=f"governing recheck {logical_path}",
            )
        ).hexdigest()
        for logical_path in GOVERNING_FILES
    }


def _validate_runtime_binding(binding: _RuntimeBinding) -> dict[str, str]:
    if not binding.finder.active or binding.finder not in sys.meta_path:
        raise BootstrapError("sealed runtime binding is not active")
    observed_canonical_modules = sorted(_canonical_project_modules())
    if observed_canonical_modules:
        raise BootstrapError(
            "canonical tanager_minmap module inventory was populated during the "
            f"sealed capsule lifetime: {observed_canonical_modules}"
        )
    expected_project_files = set(_PROJECT_MODULE_FILES.values())
    if set(binding.loaded_local_files) != expected_project_files:
        raise BootstrapError(
            "sealed runtime loaded-file set differs from the closed governing set: "
            f"{binding.loaded_local_files!r}"
        )
    expected_dependency_files = set(_TANAGER_SPEC_MODULE_FILES.values())
    if set(binding.loaded_dependency_files) != expected_dependency_files:
        raise BootstrapError(
            "sealed tanager_spec loaded-file set differs from the captured source closure: "
            f"{binding.loaded_dependency_files!r}"
        )
    if binding.finder.blocked_imports:
        raise BootstrapError(
            "ungoverned capsule-local imports were attempted: "
            f"{sorted(binding.finder.blocked_imports)}"
        )
    if binding.finder.blocked_canonical_imports:
        raise BootstrapError(
            "canonical tanager_minmap imports were attempted from the sealed capsule: "
            f"{sorted(binding.finder.blocked_canonical_imports)}"
        )
    if binding.finder.blocked_dependency_imports:
        raise BootstrapError(
            "ungoverned tanager_spec imports were attempted: "
            f"{sorted(binding.finder.blocked_dependency_imports)}"
        )
    if binding.finder.blocked_out_of_capsule_dependency_modules:
        raise BootstrapError(
            "out-of-capsule tanager_spec modules were observed: "
            f"{sorted(binding.finder.blocked_out_of_capsule_dependency_modules)}"
        )
    expected_modules = set(binding.finder.project_module_records)
    observed_modules = {
        name
        for name in sys.modules
        if name == binding.prefix or name.startswith(f"{binding.prefix}.")
    }
    if observed_modules != expected_modules:
        raise BootstrapError(
            "sealed runtime module inventory differs from the closed set: "
            f"expected {sorted(expected_modules)}, observed {sorted(observed_modules)}"
        )
    observed_files = {
        str(Path(sys.modules[name].__file__).resolve().relative_to(binding.capture.root))
        for name in observed_modules
    }
    if observed_files != expected_project_files:
        raise BootstrapError(
            "sealed runtime module files differ from the closed set: "
            f"expected {sorted(expected_project_files)}, observed {sorted(observed_files)}"
        )
    expected_new_local_origins = tuple(
        sorted(
            (module_name, record[0])
            for module_name, record in binding.finder.project_module_records.items()
        )
    )
    observed_new_local_origins = _new_repo_module_origins(
        binding.modules_before,
        root=binding.capture.root,
    )
    if (
        binding.new_local_module_origins != expected_new_local_origins
        or observed_new_local_origins != expected_new_local_origins
    ):
        raise BootstrapError(
            "newly loaded module origins under the repository root differ from the "
            "closed capsule set: "
            f"expected {expected_new_local_origins!r}, "
            f"initial {binding.new_local_module_origins!r}, "
            f"current {observed_new_local_origins!r}"
        )
    expected_dependency_modules = set(binding.finder.dependency_module_records)
    observed_dependency_modules = _canonical_dependency_modules()
    if set(observed_dependency_modules) != expected_dependency_modules:
        raise BootstrapError(
            "sealed tanager_spec module inventory differs from the closed set: "
            f"expected {sorted(expected_dependency_modules)}, "
            f"observed {sorted(observed_dependency_modules)}"
        )
    unmanaged_dependency_modules = sorted(
        name
        for name, module in observed_dependency_modules.items()
        if not binding.finder._is_managed_dependency_module(name, module)
    )
    if unmanaged_dependency_modules:
        raise BootstrapError(
            "canonical/out-of-capsule tanager_spec modules were populated during the "
            f"sealed capsule lifetime: {unmanaged_dependency_modules}"
        )
    if set(binding.finder.executed_modules) != (expected_modules | expected_dependency_modules):
        raise BootstrapError(
            "sealed executed-module inventory differs from the closed source set: "
            f"expected {sorted(expected_modules | expected_dependency_modules)}, "
            f"observed {sorted(binding.finder.executed_modules)}"
        )
    sibling_violations = _sibling_origin_violations(binding)
    if sibling_violations:
        raise BootstrapError(
            "newly loaded module origins under the tanager-spec sibling differ from the "
            f"closed capsule set: {sibling_violations!r}"
        )
    if set(binding.basic_ortho.GOVERNING_FILE_KEYS) != set(GOVERNING_FILES):
        raise BootstrapError("runtime and bootstrap governing-file key sets differ")
    if dict(binding.basic_ortho.TANAGER_SPEC_MODULE_FILES) != _TANAGER_SPEC_MODULE_FILES:
        raise BootstrapError("runtime and bootstrap tanager_spec module closures differ")
    if (
        binding.basic_ortho.TANAGER_SPEC_EDITABLE_LOGICAL_ROOT
        != _TANAGER_SPEC_EDITABLE_LOGICAL_ROOT
    ):
        raise BootstrapError("runtime and bootstrap tanager_spec logical roots differ")
    if set(binding.basic_ortho.TANAGER_SPEC_PACKAGE_DATA_FILES) != set(
        _TANAGER_SPEC_PACKAGE_DATA_FILES
    ):
        raise BootstrapError("runtime and bootstrap tanager_spec package-data closures differ")
    binding.basic_ortho.validate_governing_files(binding.capture.governing_hashes)
    observed_residual_trust = binding.basic_ortho.tanager_spec_dependency_trust(
        binding.capture.governing_hashes
    )
    if observed_residual_trust != binding.residual_dependency_trust:
        raise BootstrapError(
            "hash-bound tanager_spec dependency record changed after capsule binding"
        )
    observed_inventory, observed_directories = _observed_dependency_inventory(binding.capture.root)
    _require_exact_dependency_inventory(observed_inventory, observed_directories)
    if (
        observed_inventory != binding.capture.dependency_inventory
        or observed_directories != binding.capture.dependency_directories
    ):
        raise BootstrapError(
            "captured tanager_spec dependency inventory changed after capsule binding"
        )
    observed_hashes = _observed_governing_hashes(binding)
    if observed_hashes != binding.capture.governing_hashes:
        raise BootstrapError(
            "governing source changed after capsule binding: "
            f"expected {binding.capture.governing_hashes!r}, observed {observed_hashes!r}"
        )
    return observed_hashes


_RUNTIME_BINDING: _RuntimeBinding | None = None


def _bootstrap_runtime() -> _RuntimeBinding:
    global _RUNTIME_BINDING
    if _RUNTIME_BINDING is None:
        _RUNTIME_BINDING = _load_runtime_capsule(_capture_runtime_sources())
    return _RUNTIME_BINDING


def _validate_runtime_teardown(binding: _RuntimeBinding) -> None:
    if binding.finder.active or binding.finder in sys.meta_path:
        raise BootstrapError("sealed runtime finder remained active after teardown")
    remaining_capsule_modules = sorted(
        name
        for name in sys.modules
        if name == binding.prefix or name.startswith(f"{binding.prefix}.")
    )
    if remaining_capsule_modules:
        raise BootstrapError(
            f"sealed runtime modules remained after teardown: {remaining_capsule_modules}"
        )
    observed_canonical = _canonical_project_modules()
    if set(observed_canonical) != set(binding.canonical_modules) or any(
        observed_canonical[name] is not module for name, module in binding.canonical_modules.items()
    ):
        raise BootstrapError(
            "canonical tanager_minmap module inventory was not restored exactly at teardown"
        )
    observed_dependency = _canonical_dependency_modules()
    if set(observed_dependency) != set(binding.canonical_dependency_modules) or any(
        observed_dependency[name] is not module
        for name, module in binding.canonical_dependency_modules.items()
    ):
        raise BootstrapError(
            "preexisting tanager_spec module inventory was not restored exactly at teardown"
        )
    observed_sibling_modules = _sibling_local_modules(
        _dependency_editable_root(binding.capture.root)
    )
    if set(observed_sibling_modules) != set(binding.sibling_modules_before) or any(
        observed_sibling_modules[name] is not module
        for name, module in binding.sibling_modules_before.items()
    ):
        raise BootstrapError(
            "preexisting tanager-spec sibling module objects were not restored exactly at teardown"
        )


def _teardown_runtime(binding: _RuntimeBinding) -> None:
    """Validate the final live inventory, then restore pre-capsule module state."""
    global _RUNTIME_BINDING
    if not binding.finder.active:
        raise BootstrapError("sealed runtime binding was already torn down")
    try:
        _validate_runtime_binding(binding)
    finally:
        _release_runtime_capsule_state(
            finder=binding.finder,
            prefix=binding.prefix,
            capture=binding.capture,
            modules_before=binding.modules_before,
            canonical_modules=binding.canonical_modules,
            canonical_dependency_modules=binding.canonical_dependency_modules,
            sibling_modules_before=binding.sibling_modules_before,
        )
        if _RUNTIME_BINDING is binding:
            _RUNTIME_BINDING = None
        _validate_runtime_teardown(binding)


def _require_governing_unchanged(
    expected: Mapping[str, str],
    binding: _RuntimeBinding,
) -> None:
    try:
        observed = _validate_runtime_binding(binding)
    except BootstrapError as error:
        raise binding.basic_ortho.ProtocolError(str(error)) from error
    if observed != dict(expected):
        raise binding.basic_ortho.ProtocolError(
            "governing files changed during execution: "
            f"expected {dict(expected)!r}, observed {observed!r}"
        )


def _source_stat_record(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


@dataclass
class _StableSnapshot:
    source_path: Path
    handle: BinaryIO
    source_pre: dict[str, int]
    source_post: dict[str, int]
    copied_byte_count: int
    sha256: str

    def evidence(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_pre": dict(self.source_pre),
            "source_post": dict(self.source_post),
            "copied_byte_count": self.copied_byte_count,
            "sha256": self.sha256,
        }

    def read_bytes(self) -> bytes:
        self.handle.seek(0)
        payload = self.handle.read()
        self.handle.seek(0)
        if len(payload) != self.copied_byte_count:
            raise IndependentVerificationError(
                f"sealed snapshot byte count changed: {self.source_path}"
            )
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise IndependentVerificationError(
                f"sealed snapshot SHA-256 changed: {self.source_path}"
            )
        return payload


@contextmanager
def _stable_descriptor_snapshot(
    source_path: Path,
    *,
    label: str,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> Iterator[_StableSnapshot]:
    """Copy one no-follow source descriptor into one unnamed stable snapshot."""
    source_fd, initial_info = _open_regular_nofollow(source_path, label=label)
    digest = hashlib.sha256()
    copied_bytes = 0
    snapshot = tempfile.TemporaryFile(mode="w+b")
    try:
        with os.fdopen(source_fd, "rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                snapshot.write(chunk)
                copied_bytes += len(chunk)
            final_info = os.fstat(source.fileno())
        source_pre = _source_stat_record(initial_info)
        source_post = _source_stat_record(final_info)
        observed_sha256 = digest.hexdigest()
        if source_pre != source_post or copied_bytes != source_pre["size_bytes"]:
            raise IndependentVerificationError(
                f"{label} changed while its descriptor-bound snapshot was copied"
            )
        if expected_size_bytes is not None and copied_bytes != expected_size_bytes:
            raise IndependentVerificationError(
                f"{label} snapshot byte count does not match the frozen input"
            )
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise IndependentVerificationError(
                f"{label} snapshot SHA-256 does not match the frozen input"
            )
        snapshot.flush()
        os.fsync(snapshot.fileno())
        snapshot_info = os.fstat(snapshot.fileno())
        if snapshot_info.st_size != copied_bytes:
            raise IndependentVerificationError(f"{label} snapshot was not copied completely")
        snapshot.seek(0)
        yield _StableSnapshot(
            source_path=source_path,
            handle=snapshot,
            source_pre=source_pre,
            source_post=source_post,
            copied_byte_count=copied_bytes,
            sha256=observed_sha256,
        )
    finally:
        snapshot.close()


@contextmanager
def _snapshot_reader(snapshot: _StableSnapshot) -> Iterator[BinaryIO]:
    """Yield an independently closable descriptor view of one unnamed snapshot."""
    duplicate_fd = os.dup(snapshot.handle.fileno())
    os.lseek(duplicate_fd, 0, os.SEEK_SET)
    with os.fdopen(duplicate_fd, "rb") as reader:
        yield reader


def _revalidate_snapshot_source(
    snapshot: _StableSnapshot,
    *,
    label: str,
    verify_sha256: bool = False,
) -> None:
    """Require the source path to retain the sealed identity and optionally rehash it."""
    file_fd, initial_info = _open_regular_nofollow(snapshot.source_path, label=label)
    digest = hashlib.sha256()
    byte_count = 0
    with os.fdopen(file_fd, "rb") as source:
        if verify_sha256:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
        final_info = os.fstat(source.fileno())
    initial = _source_stat_record(initial_info)
    final = _source_stat_record(final_info)
    if initial != final or final != snapshot.source_post:
        raise IndependentVerificationError(
            f"{label} source metadata changed after snapshot sealing"
        )
    if verify_sha256 and (
        byte_count != snapshot.copied_byte_count or digest.hexdigest() != snapshot.sha256
    ):
        raise IndependentVerificationError(f"{label} source bytes changed after snapshot sealing")


def _oracle_unique_dataset(handle: Any, candidates: Sequence[str], *, label: str) -> Any:
    matches = [handle[path] for path in candidates if path in handle]
    if len(matches) != 1:
        raise IndependentVerificationError(
            f"independent oracle requires exactly one {label}, observed {len(matches)}"
        )
    return matches[0]


def _oracle_field_candidates(groups: Sequence[str], field: str) -> tuple[str, ...]:
    return tuple(f"{group}/{field}" for group in groups)


def _oracle_qa(
    handle: Any,
    groups: Sequence[str],
    shape: tuple[int, int],
) -> tuple[Any, dict[str, int]]:
    import numpy as np

    valid = np.ones(shape, dtype=bool)
    invalid_counts: dict[str, int] = {}
    for field in _QA_FIELDS:
        dataset = _oracle_unique_dataset(
            handle,
            _oracle_field_candidates(groups, field),
            label=f"QA field {field}",
        )
        values = np.asarray(dataset[...])
        if values.shape != shape or not np.issubdtype(values.dtype, np.integer):
            raise IndependentVerificationError(f"independent oracle QA is misaligned: {field}")
        unknown = {int(value) for value in np.unique(values)} - _QA_ALLOWED_VALUES
        if unknown:
            raise IndependentVerificationError(
                f"independent oracle found undocumented QA values in {field}: {sorted(unknown)}"
            )
        invalid = values != 0
        invalid_counts[field] = int(invalid.sum())
        valid &= ~invalid
    return valid, invalid_counts


def _oracle_parse_grid(raw: Any) -> tuple[tuple[int, int], Any, Any]:
    import numpy as np
    from affine import Affine
    from rasterio.crs import CRS

    if isinstance(raw, bytes):
        text = raw.decode(errors="strict")
    elif isinstance(raw, np.ndarray) and raw.ndim == 0:
        item = raw.item()
        text = item.decode(errors="strict") if isinstance(item, bytes) else str(item)
    else:
        text = str(raw)
    block = None
    for chunk in re.split(r"GROUP=GRID_\d+", text):
        match = re.search(r'GridName="([^"]+)"', chunk)
        if match is not None and match.group(1) == "HYP":
            block = chunk
            break
    if block is None:
        raise IndependentVerificationError("independent oracle cannot find HYP grid metadata")

    def value(key: str) -> str:
        match = re.search(rf"\b{key}=([^\n]+)", block)
        if match is None:
            raise IndependentVerificationError(f"independent oracle grid lacks {key}")
        return match.group(1).strip()

    if "UTM" not in value("Projection").upper():
        raise IndependentVerificationError("independent oracle supports only delivered UTM grids")
    nx = int(value("XDim"))
    ny = int(value("YDim"))
    ulx, uly = (float(item) for item in re.findall(r"[-\d.]+", value("UpperLeftPointMtrs")))
    lrx, lry = (float(item) for item in re.findall(r"[-\d.]+", value("LowerRightMtrs")))
    zone = int(value("ZoneCode"))
    epsg = (32600 if zone > 0 else 32700) + abs(zone)
    transform = Affine((lrx - ulx) / nx, 0.0, ulx, 0.0, -(uly - lry) / ny, uly)
    return (ny, nx), transform, CRS.from_epsg(epsg)


def _oracle_reflectance_cube(
    handle: Any,
    groups: Sequence[str],
    spatial_shape: tuple[int, int],
    *,
    label: str,
) -> tuple[Any, Any, Any]:
    import numpy as np

    dataset = _oracle_unique_dataset(
        handle,
        _oracle_field_candidates(groups, "surface_reflectance"),
        label=f"{label} reflectance",
    )
    if dataset.ndim != 3 or not np.issubdtype(dataset.dtype, np.floating):
        raise IndependentVerificationError(
            f"independent {label} reflectance must be a floating 3-D cube"
        )
    try:
        wavelengths = np.asarray(dataset.attrs["wavelengths"])
        good = np.asarray(dataset.attrs["good_wavelengths"])
    except KeyError as error:
        raise IndependentVerificationError(
            f"independent {label} reflectance lacks required band metadata"
        ) from error
    if (
        wavelengths.ndim != 1
        or wavelengths.size < 2
        or not np.issubdtype(wavelengths.dtype, np.floating)
        or not np.isfinite(wavelengths).all()
        or not np.all(np.diff(wavelengths) > 0)
    ):
        raise IndependentVerificationError(f"independent {label} wavelength metadata is invalid")
    if good.shape != wavelengths.shape or not (
        np.issubdtype(good.dtype, np.bool_) or np.issubdtype(good.dtype, np.integer)
    ):
        raise IndependentVerificationError(
            f"independent {label} good-wavelength metadata is invalid"
        )
    spectral_axes = [
        axis for axis, size in enumerate(dataset.shape) if int(size) == int(wavelengths.size)
    ]
    if len(spectral_axes) != 1:
        raise IndependentVerificationError(f"independent {label} spectral axis is not unique")
    spectral_axis = spectral_axes[0]
    spatial_axes = tuple(axis for axis in range(3) if axis != spectral_axis)
    if tuple(int(dataset.shape[axis]) for axis in spatial_axes) != spatial_shape:
        raise IndependentVerificationError(
            f"independent {label} reflectance is not spatially aligned"
        )
    values = np.asarray(dataset[...])
    cube = np.transpose(values, (spectral_axis, *spatial_axes))
    return cube, wavelengths, np.asarray(good, dtype=bool)


def _oracle_exact_spectral_copy(
    basic_cube: Any,
    ortho_cube: Any,
    *,
    retained_bands: Any,
    basic_valid: Any,
    ortho_valid: Any,
    source_row: Any,
    source_col: Any,
    status: Any,
) -> dict[str, Any]:
    """Independently count exact copies using raw vector bytes and direct equality."""
    import numpy as np

    basic = np.asarray(basic_cube)
    ortho = np.asarray(ortho_cube)
    bands = np.asarray(retained_bands, dtype=bool)
    if (
        basic.ndim != 3
        or ortho.ndim != 3
        or basic.dtype != ortho.dtype
        or basic.shape[0] != ortho.shape[0]
        or bands.shape != (basic.shape[0],)
        or not bands.any()
    ):
        raise IndependentVerificationError("independent spectral-copy cube contract failed")
    if basic.shape[1:] != np.asarray(basic_valid).shape:
        raise IndependentVerificationError("independent basic spectral support is misaligned")
    if ortho.shape[1:] != np.asarray(ortho_valid).shape:
        raise IndependentVerificationError("independent ortho spectral support is misaligned")

    retained_count = int(bands.sum())
    basic_vectors = np.transpose(basic[bands], (1, 2, 0)).reshape(-1, retained_count)
    ortho_vectors = np.transpose(ortho[bands], (1, 2, 0)).reshape(-1, retained_count)
    basic_ok = np.asarray(basic_valid, dtype=bool).ravel() & np.isfinite(basic_vectors).all(axis=1)
    ortho_ok = np.asarray(ortho_valid, dtype=bool).ravel() & np.isfinite(ortho_vectors).all(axis=1)

    exact_index: dict[bytes, list[int]] = {}
    for flat_index in np.flatnonzero(basic_ok):
        key = basic_vectors[flat_index].tobytes(order="C")
        exact_index.setdefault(key, []).append(int(flat_index))
    any_match_count = 0
    for flat_index in np.flatnonzero(ortho_ok):
        vector = ortho_vectors[flat_index]
        candidates = exact_index.get(vector.tobytes(order="C"), ())
        if any(np.array_equal(vector, basic_vectors[candidate]) for candidate in candidates):
            any_match_count += 1

    source_rows = np.asarray(source_row).ravel()
    source_cols = np.asarray(source_col).ravel()
    status_values = np.asarray(status).ravel()
    mapped = (status_values == _TARGET_MAPPED) & ortho_ok
    selected_match_count = 0
    basic_column_count = basic.shape[2]
    for target_index in np.flatnonzero(mapped):
        source_index = int(
            source_rows[target_index] * basic_column_count + source_cols[target_index]
        )
        if basic_ok[source_index] and np.array_equal(
            ortho_vectors[target_index], basic_vectors[source_index]
        ):
            selected_match_count += 1

    valid_ortho_count = int(ortho_ok.sum())
    mapped_count = int(mapped.sum())
    return {
        "retained_bands": retained_count,
        "valid_basic_spectra": int(basic_ok.sum()),
        "valid_ortho_spectra": valid_ortho_count,
        "ortho_exact_match_to_any_basic": any_match_count,
        "ortho_exact_match_to_any_basic_fraction": (
            any_match_count / valid_ortho_count if valid_ortho_count else None
        ),
        "mapped_valid_ortho_spectra": mapped_count,
        "mapped_exact_match_to_selected_basic": selected_match_count,
        "mapped_exact_match_to_selected_basic_fraction": (
            selected_match_count / mapped_count if mapped_count else None
        ),
    }


def _oracle_scene_inputs(
    scene: Any,
    snapshots: Mapping[tuple[str, str], _StableSnapshot],
) -> tuple[Any, Any, Any, Any, Any, dict[str, int], Any, Any, Any]:
    import h5py
    import numpy as np

    with _snapshot_reader(snapshots[(scene.site, "basic")]) as basic_reader:
        with h5py.File(basic_reader, "r") as basic_handle:
            latitude_ds = _oracle_unique_dataset(
                basic_handle,
                tuple(
                    f"{group}/{name}"
                    for group in _BASIC_GEO_GROUPS
                    for name in ("Latitude", "latitude")
                ),
                label="basic latitude",
            )
            longitude_ds = _oracle_unique_dataset(
                basic_handle,
                tuple(
                    f"{group}/{name}"
                    for group in _BASIC_GEO_GROUPS
                    for name in ("Longitude", "longitude")
                ),
                label="basic longitude",
            )
            latitude = np.asarray(latitude_ds[...], dtype=np.float64)
            longitude = np.asarray(longitude_ds[...], dtype=np.float64)
            if latitude.shape != scene.basic_stac_shape or longitude.shape != latitude.shape:
                raise IndependentVerificationError("independent basic geolocation is misaligned")
            for coordinate, dataset in ((latitude, latitude_ds), (longitude, longitude_ds)):
                if "_FillValue" in dataset.attrs:
                    coordinate[coordinate == dataset.attrs["_FillValue"]] = np.nan
            source_qa_valid, basic_counts = _oracle_qa(
                basic_handle,
                _BASIC_DATA_GROUPS,
                scene.basic_stac_shape,
            )
            basic_cube, basic_wavelengths, retained_bands = _oracle_reflectance_cube(
                basic_handle,
                _BASIC_DATA_GROUPS,
                scene.basic_stac_shape,
                label="basic",
            )

    with _snapshot_reader(snapshots[(scene.site, "ortho")]) as ortho_reader:
        with h5py.File(ortho_reader, "r") as ortho_handle:
            target_qa_valid, ortho_counts = _oracle_qa(
                ortho_handle,
                _ORTHO_DATA_GROUPS,
                scene.ortho_stac_shape,
            )
            if _STRUCT_METADATA_PATH not in ortho_handle:
                raise IndependentVerificationError("independent oracle lacks ortho grid metadata")
            grid_shape, grid_transform, grid_crs = _oracle_parse_grid(
                ortho_handle[_STRUCT_METADATA_PATH][()]
            )
            ortho_cube, ortho_wavelengths, ortho_retained_bands = _oracle_reflectance_cube(
                ortho_handle,
                _ORTHO_DATA_GROUPS,
                scene.ortho_stac_shape,
                label="ortho",
            )
    if (
        basic_wavelengths.dtype != ortho_wavelengths.dtype
        or not np.array_equal(basic_wavelengths, ortho_wavelengths)
        or not np.array_equal(retained_bands, ortho_retained_bands)
    ):
        raise IndependentVerificationError(
            "independent oracle found basic/ortho spectral metadata drift"
        )
    if grid_shape != scene.ortho_stac_shape:
        raise IndependentVerificationError("independent grid shape differs from frozen geometry")
    if grid_crs.to_string() != scene.ortho_stac_crs:
        raise IndependentVerificationError("independent grid CRS differs from frozen geometry")
    if (
        grid_transform.a != scene.ortho_stac_resolution_m
        or abs(grid_transform.e) != scene.ortho_stac_resolution_m
    ):
        raise IndependentVerificationError(
            "independent grid resolution differs from frozen geometry"
        )
    qa_counts = {
        **{f"basic:{name}": count for name, count in basic_counts.items()},
        **{f"ortho:{name}": count for name, count in ortho_counts.items()},
    }
    return (
        longitude,
        latitude,
        source_qa_valid,
        target_qa_valid,
        (grid_shape, grid_transform, grid_crs),
        qa_counts,
        basic_cube,
        ortho_cube,
        retained_bands,
    )


def _independent_scene_semantics(
    scene: Any,
    snapshots: Mapping[tuple[str, str], _StableSnapshot],
) -> dict[str, Any]:
    import numpy as np
    from rasterio.warp import transform as transform_coordinates
    from scipy.spatial import KDTree

    (
        longitude,
        latitude,
        source_qa_valid,
        target_qa_valid,
        grid,
        qa_counts,
        basic_cube,
        ortho_cube,
        retained_bands,
    ) = _oracle_scene_inputs(scene, snapshots)
    grid_shape, grid_transform, grid_crs = grid
    source_shape = longitude.shape
    geo_valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180.0)
        & (longitude <= 180.0)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
    )
    geolocated_source = np.flatnonzero(geo_valid)
    source_points = np.empty((0, 2), dtype=np.float64)
    if geolocated_source.size:
        source_x, source_y = transform_coordinates(
            "EPSG:4326",
            grid_crs,
            longitude.ravel()[geolocated_source],
            latitude.ravel()[geolocated_source],
        )
        source_points = np.column_stack(
            (np.asarray(source_x, dtype=np.float64), np.asarray(source_y, dtype=np.float64))
        )
        projected_valid = np.isfinite(source_points).all(axis=1)
        geo_valid.ravel()[geolocated_source[~projected_valid]] = False
        geolocated_source = geolocated_source[projected_valid]
        source_points = source_points[projected_valid]

    source_row = np.full(grid_shape, -1, dtype=np.int32)
    source_col = np.full(grid_shape, -1, dtype=np.int32)
    distance = np.full(grid_shape, np.nan, dtype=np.float64)
    multiplicity = np.zeros(grid_shape, dtype=np.uint32)
    status = np.full(grid_shape, _TARGET_ORTHO_QA_INVALID, dtype=np.uint8)
    target_count_per_source = np.zeros(source_shape, dtype=np.uint32)
    candidate_target = np.flatnonzero(target_qa_valid)

    if geolocated_source.size and candidate_target.size:
        target_rows, target_cols = np.unravel_index(candidate_target, grid_shape)
        target_x, target_y = grid_transform * (target_cols + 0.5, target_rows + 0.5)
        target_points = np.column_stack((target_x, target_y))
        tree = KDTree(source_points)
        nearest_distance, _nearest_position = tree.query(target_points, k=1, workers=1)
        chosen_source = np.empty(candidate_target.size, dtype=np.int64)
        exact_distance = np.empty(candidate_target.size, dtype=np.float64)
        for target_position, target_point in enumerate(target_points):
            radius = np.nextafter(float(nearest_distance[target_position]), np.inf)
            positions = np.asarray(tree.query_ball_point(target_point, radius), dtype=np.int64)
            candidate_distances = np.linalg.norm(source_points[positions] - target_point, axis=1)
            minimum = float(candidate_distances.min())
            tied_positions = positions[candidate_distances == minimum]
            tied_flat_sources = geolocated_source[tied_positions]
            chosen_source[target_position] = int(tied_flat_sources.min())
            exact_distance[target_position] = (
                minimum if tied_positions.size > 1 else float(nearest_distance[target_position])
            )

        chosen_rows, chosen_cols = np.unravel_index(chosen_source, source_shape)
        source_row.ravel()[candidate_target] = chosen_rows
        source_col.ravel()[candidate_target] = chosen_cols
        distance.ravel()[candidate_target] = exact_distance
        counts_by_source = np.bincount(chosen_source, minlength=longitude.size).astype(np.uint32)
        target_count_per_source.ravel()[:] = counts_by_source
        multiplicity.ravel()[candidate_target] = counts_by_source[chosen_source]
        chosen_qa_valid = source_qa_valid.ravel()[chosen_source]
        status.ravel()[candidate_target[chosen_qa_valid]] = _TARGET_MAPPED
        status.ravel()[candidate_target[~chosen_qa_valid]] = _TARGET_BASIC_QA_INVALID
    elif candidate_target.size:
        status.ravel()[candidate_target] = _TARGET_NO_GEOLOCATED_SOURCE

    target_counts = target_count_per_source.ravel()
    mapped_count = int((status == _TARGET_MAPPED).sum())
    source_accounting = {
        "total_source_samples": int(longitude.size),
        "invalid_qa_source_samples": int((~source_qa_valid).sum()),
        "invalid_geolocation_source_samples": int((~geo_valid).sum()),
        "used_source_samples": int((target_counts > 0).sum()),
        "unused_source_samples": int((target_counts == 0).sum()),
        "sources_with_multiple_target_cells": int((target_counts > 1).sum()),
        "duplicate_target_assignments": int((target_counts[target_counts > 0] - 1).sum()),
        "total_target_cells": int(np.prod(grid_shape)),
        "invalid_qa_target_cells": int((~target_qa_valid).sum()),
        "basic_qa_no_call_target_cells": int((status == _TARGET_BASIC_QA_INVALID).sum()),
        "no_geolocated_source_target_cells": int((status == _TARGET_NO_GEOLOCATED_SOURCE).sum()),
        "mapped_target_cells": mapped_count,
        "unmapped_target_cells": int(np.prod(grid_shape) - mapped_count),
    }
    spectral_copy_audit = _oracle_exact_spectral_copy(
        basic_cube,
        ortho_cube,
        retained_bands=retained_bands,
        basic_valid=source_qa_valid,
        ortho_valid=target_qa_valid,
        source_row=source_row,
        source_col=source_col,
        status=status,
    )
    return {
        "arrays": {
            "source_index.tif": np.stack((source_row, source_col)),
            "mapping_distance_m.tif": distance[np.newaxis, ...],
            "source_multiplicity.tif": multiplicity[np.newaxis, ...],
            "mapping_status.tif": status[np.newaxis, ...],
        },
        "source_accounting": source_accounting,
        "qa_invalid_counts_nonexclusive": qa_counts,
        "spectral_copy_audit": spectral_copy_audit,
    }


def _independent_mapping_attestations(
    inputs: Any,
    snapshots: Mapping[tuple[str, str], _StableSnapshot],
) -> dict[str, dict[str, Any]]:
    observed_scenes = {scene.site: scene.scene_id for scene in inputs.scenes}
    if observed_scenes != _FROZEN_SCENES:
        raise IndependentVerificationError(
            f"independent oracle scene set differs from frozen scenes: {observed_scenes!r}"
        )
    return {scene.site: _independent_scene_semantics(scene, snapshots) for scene in inputs.scenes}


def _inspect_all_schemas(
    basic_ortho: ModuleType,
    inputs: Any,
    snapshots: Mapping[tuple[str, str], _StableSnapshot],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import h5py

    schemas: dict[str, Any] = {}
    grids: dict[str, Any] = {}
    for scene in inputs.scenes:
        with _snapshot_reader(snapshots[(scene.site, "basic")]) as basic_reader:
            with h5py.File(basic_reader, "r") as basic_handle:
                basic = basic_ortho._inspect_basic_schema_handle(
                    basic_handle,
                    display_path=scene.basic_path,
                    expected_shape=scene.basic_stac_shape,
                )
        with _snapshot_reader(snapshots[(scene.site, "ortho")]) as ortho_reader:
            with h5py.File(ortho_reader, "r") as ortho_handle:
                ortho = basic_ortho._inspect_ortho_schema_handle(
                    ortho_handle,
                    display_path=scene.ortho_path,
                    expected_shape=scene.ortho_stac_shape,
                )
                grid = basic_ortho._load_ortho_grid_handle(
                    ortho_handle,
                    expected_shape=scene.ortho_stac_shape,
                    expected_crs=scene.ortho_stac_crs,
                    expected_resolution_m=scene.ortho_stac_resolution_m,
                )
        basic_ortho.validate_schema_pair(basic, ortho)
        schemas[scene.site] = (basic, ortho)
        grids[scene.site] = grid
    return schemas, grids


def _production_scene_arrays(
    basic_ortho: ModuleType,
    scene: Any,
    schemas: tuple[Any, Any],
    snapshots: Mapping[tuple[str, str], _StableSnapshot],
) -> tuple[Any, Any, Any, dict[str, int], Any, dict[str, int], Any, Any, Any]:
    """Read production mapping and copy-audit inputs only from the sealed snapshots."""
    import h5py
    import numpy as np

    basic_schema, ortho_schema = schemas
    with _snapshot_reader(snapshots[(scene.site, "basic")]) as basic_reader:
        with h5py.File(basic_reader, "r") as basic_handle:
            longitude, latitude, basic_qa_valid, basic_qa_counts = (
                basic_ortho._load_basic_geolocation_and_qa_handle(
                    basic_handle,
                    basic_schema,
                )
            )
            basic_dataset = basic_handle[basic_schema.reflectance.path]
            basic_cube = basic_ortho.canonical_band_y_x(
                np.asarray(basic_dataset[...]),
                spectral_axis=basic_schema.spectral_axis,
            )
            retained_bands = np.asarray(
                basic_dataset.attrs["good_wavelengths"],
                dtype=bool,
            )
    with _snapshot_reader(snapshots[(scene.site, "ortho")]) as ortho_reader:
        with h5py.File(ortho_reader, "r") as ortho_handle:
            ortho_qa_valid, ortho_qa_counts = basic_ortho._load_ortho_qa_handle(
                ortho_handle,
                ortho_schema,
            )
            ortho_dataset = ortho_handle[ortho_schema.reflectance.path]
            ortho_cube = basic_ortho.canonical_band_y_x(
                np.asarray(ortho_dataset[...]),
                spectral_axis=ortho_schema.spectral_axis,
            )
    return (
        longitude,
        latitude,
        basic_qa_valid,
        basic_qa_counts,
        ortho_qa_valid,
        ortho_qa_counts,
        basic_cube,
        ortho_cube,
        retained_bands,
    )


@contextmanager
def _snapshot_hdf_inputs(
    inputs: Any,
    *,
    selected: set[tuple[str, str]] | None = None,
) -> Iterator[dict[tuple[str, str], _StableSnapshot]]:
    with ExitStack() as stack:
        snapshots: dict[tuple[str, str], _StableSnapshot] = {}
        for scene in inputs.scenes:
            products = {
                "basic": (
                    scene.basic_path,
                    scene.basic_size_bytes,
                    scene.basic_sha256,
                ),
                "ortho": (
                    scene.ortho_path,
                    scene.ortho_size_bytes,
                    scene.ortho_sha256,
                ),
            }
            for product, (path, size_bytes, digest) in products.items():
                key = (scene.site, product)
                if selected is not None and key not in selected:
                    continue
                snapshots[key] = stack.enter_context(
                    _stable_descriptor_snapshot(
                        path,
                        label=f"{scene.site} {product} HDF5 input",
                        expected_size_bytes=size_bytes,
                        expected_sha256=digest,
                    )
                )
        yield snapshots


@dataclass
class _SealedRunInputs:
    protocol_hash: str
    inputs: Any
    preregistration: _StableSnapshot
    acquisition_manifest: _StableSnapshot
    ortho_manifest: _StableSnapshot
    hdf: dict[tuple[str, str], _StableSnapshot]

    def snapshot_evidence(self) -> dict[str, Any]:
        scene_evidence: dict[str, dict[str, Any]] = {}
        for (site, product), snapshot in sorted(self.hdf.items()):
            scene_evidence.setdefault(site, {})[product] = snapshot.evidence()
        return {
            "preregistration": self.preregistration.evidence(),
            "acquisition_manifest": self.acquisition_manifest.evidence(),
            "ortho_manifest": self.ortho_manifest.evidence(),
            "scenes": scene_evidence,
        }

    def all_snapshots(self) -> tuple[_StableSnapshot, ...]:
        return (
            self.preregistration,
            self.acquisition_manifest,
            self.ortho_manifest,
            *tuple(self.hdf[key] for key in sorted(self.hdf)),
        )


@contextmanager
def _sealed_run_inputs(
    basic_ortho: ModuleType,
    *,
    preregistration: Path,
    acquisition_manifest: Path,
    ortho_manifest: Path,
    hdf_mode: str,
    pilot_site: str | None = None,
    pilot_branch: str | None = None,
) -> Iterator[_SealedRunInputs]:
    """Seal every parsed/read input once and retain the descriptors through publication."""
    with ExitStack() as stack:
        protocol_snapshot = stack.enter_context(
            _stable_descriptor_snapshot(preregistration, label="M1b preregistration")
        )
        acquisition_snapshot = stack.enter_context(
            _stable_descriptor_snapshot(acquisition_manifest, label="acquisition manifest")
        )
        ortho_snapshot = stack.enter_context(
            _stable_descriptor_snapshot(ortho_manifest, label="ortho manifest")
        )
        protocol_hash = basic_ortho.validate_protocol_file(
            preregistration,
            expected_sha256=basic_ortho.FROZEN_PREREGISTRATION_SHA256,
            snapshot_bytes=protocol_snapshot.read_bytes(),
        )
        if protocol_snapshot.sha256 != protocol_hash:
            raise IndependentVerificationError(
                "preregistration validation did not bind the sealed snapshot"
            )
        inputs = basic_ortho.validate_frozen_inputs(
            acquisition_manifest,
            ortho_manifest,
            root=ROOT,
            expected_acquisition_sha256=basic_ortho.FROZEN_ACQUISITION_MANIFEST_SHA256,
            expected_ortho_sha256=basic_ortho.FROZEN_ORTHO_MANIFEST_SHA256,
            verify_files=False,
            acquisition_snapshot_bytes=acquisition_snapshot.read_bytes(),
            ortho_snapshot_bytes=ortho_snapshot.read_bytes(),
        )
        if acquisition_snapshot.sha256 != inputs.acquisition_manifest_sha256:
            raise IndependentVerificationError(
                "acquisition manifest validation did not bind the sealed snapshot"
            )
        if ortho_snapshot.sha256 != inputs.ortho_manifest_sha256:
            raise IndependentVerificationError(
                "ortho manifest validation did not bind the sealed snapshot"
            )

        selected: set[tuple[str, str]] | None
        if hdf_mode == "none":
            selected = set()
        elif hdf_mode == "all":
            selected = None
        elif hdf_mode == "pilot":
            if pilot_site is None or pilot_branch not in {"B", "O"}:
                raise IndependentVerificationError("resource-pilot snapshot selector is invalid")
            selected = {(pilot_site, "basic" if pilot_branch == "B" else "ortho")}
        else:
            raise IndependentVerificationError(f"unknown HDF5 snapshot mode: {hdf_mode}")

        with _snapshot_hdf_inputs(inputs, selected=selected) as hdf_snapshots:
            yield _SealedRunInputs(
                protocol_hash=protocol_hash,
                inputs=inputs,
                preregistration=protocol_snapshot,
                acquisition_manifest=acquisition_snapshot,
                ortho_manifest=ortho_snapshot,
                hdf=hdf_snapshots,
            )


def _revalidate_sealed_sources(sealed: _SealedRunInputs) -> None:
    """Check all source identities and rehash the sealed preregistration."""
    for snapshot in sealed.all_snapshots():
        _revalidate_snapshot_source(
            snapshot,
            label=f"sealed input {snapshot.source_path}",
            verify_sha256=snapshot is sealed.preregistration,
        )


def _write_once(basic_ortho: ModuleType, path: Path, payload: dict[str, object]) -> Path:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("execution_id") != payload.get("execution_id"):
            raise ValueError(f"refusing to replace a different execution identity: {path}")
    basic_ortho.strict_json_dump(path, payload)
    return path


def _start_run() -> tuple[_RuntimeBinding, ModuleType, dict[str, str]]:
    binding = _bootstrap_runtime()
    governing = binding.governing_hashes
    _require_governing_unchanged(governing, binding)
    return binding, binding.basic_ortho, governing


def run_design_only(
    *,
    preregistration: Path,
    acquisition_manifest: Path,
    ortho_manifest: Path,
    output_dir: Path,
) -> Path:
    """Write only the protocol-bound design; do not hash source imagery."""
    binding, basic_ortho, governing = _start_run()
    with _sealed_run_inputs(
        basic_ortho,
        preregistration=preregistration,
        acquisition_manifest=acquisition_manifest,
        ortho_manifest=ortho_manifest,
        hdf_mode="none",
    ) as sealed:
        payload = basic_ortho.design_document(
            protocol_sha256=sealed.protocol_hash,
            acquisition_manifest_sha256=sealed.inputs.acquisition_manifest_sha256,
            ortho_manifest_sha256=sealed.inputs.ortho_manifest_sha256,
            governing_files=governing,
        )
        _revalidate_sealed_sources(sealed)
        _require_governing_unchanged(governing, binding)
        return _write_once(basic_ortho, output_dir / "design.json", payload)


def run_schema_only(
    *,
    preregistration: Path,
    acquisition_manifest: Path,
    ortho_manifest: Path,
    output_dir: Path,
) -> Path:
    """Hash inputs and validate HDF metadata without reading reflectance samples."""
    binding, basic_ortho, governing = _start_run()
    with _sealed_run_inputs(
        basic_ortho,
        preregistration=preregistration,
        acquisition_manifest=acquisition_manifest,
        ortho_manifest=ortho_manifest,
        hdf_mode="all",
    ) as sealed:
        schemas, grids = _inspect_all_schemas(basic_ortho, sealed.inputs, sealed.hdf)
        payload = basic_ortho.schema_document(
            sealed.inputs,
            schemas,
            grids,
            protocol_sha256=sealed.protocol_hash,
            governing_files=governing,
        )
        run_dir = output_dir / str(payload["execution_id"])
        _revalidate_sealed_sources(sealed)
        _require_governing_unchanged(governing, binding)
        return _write_once(basic_ortho, run_dir / "schema_manifest.json", payload)


def run_mapping_only(
    *,
    preregistration: Path,
    acquisition_manifest: Path,
    ortho_manifest: Path,
    output_dir: Path,
) -> Path:
    """Build source-index mappings and the preregistered exact-copy audit only."""
    binding, basic_ortho, governing = _start_run()
    with _sealed_run_inputs(
        basic_ortho,
        preregistration=preregistration,
        acquisition_manifest=acquisition_manifest,
        ortho_manifest=ortho_manifest,
        hdf_mode="all",
    ) as sealed:
        inputs = sealed.inputs
        schemas, grids = _inspect_all_schemas(basic_ortho, inputs, sealed.hdf)
        residual_dependency_trust = dict(binding.residual_dependency_trust)
        run_identity_inputs = {
            "mode": "mapping-only",
            "protocol_sha256": sealed.protocol_hash,
            "acquisition_manifest_sha256": inputs.acquisition_manifest_sha256,
            "ortho_manifest_sha256": inputs.ortho_manifest_sha256,
            "governing_files": governing,
            "residual_dependency_trust": residual_dependency_trust,
            "input_sha256": {
                scene.site: {"basic": scene.basic_sha256, "ortho": scene.ortho_sha256}
                for scene in inputs.scenes
            },
        }
        run_id = basic_ortho.execution_identity(run_identity_inputs)
        expected_scenes = {scene.site: scene.scene_id for scene in inputs.scenes}
        expected_semantics = _independent_mapping_attestations(inputs, sealed.hdf)
        snapshot_evidence = sealed.snapshot_evidence()

        def write_staged_bundle(staging_root: Path) -> Path:
            scene_manifests: dict[str, dict[str, object]] = {}
            for scene in inputs.scenes:
                (
                    longitude,
                    latitude,
                    basic_qa_valid,
                    basic_qa_counts,
                    ortho_qa_valid,
                    ortho_qa_counts,
                    basic_cube,
                    ortho_cube,
                    retained_bands,
                ) = _production_scene_arrays(
                    basic_ortho,
                    scene,
                    schemas[scene.site],
                    sealed.hdf,
                )
                mapping = basic_ortho.map_native_to_ortho(
                    longitude,
                    latitude,
                    basic_qa_valid,
                    ortho_qa_valid,
                    grids[scene.site],
                )
                spectral_copy_audit = basic_ortho.exact_spectrum_copy_audit(
                    basic_cube,
                    ortho_cube,
                    mapping,
                    retained_bands=retained_bands,
                    basic_valid=basic_qa_valid,
                    ortho_valid=ortho_qa_valid,
                )
                qa_counts = {
                    **{f"basic:{name}": count for name, count in basic_qa_counts.items()},
                    **{f"ortho:{name}": count for name, count in ortho_qa_counts.items()},
                }
                manifest = basic_ortho.write_mapping_bundle(
                    staging_root,
                    scene=scene,
                    grid=grids[scene.site],
                    mapping=mapping,
                    protocol_sha256=sealed.protocol_hash,
                    acquisition_manifest_sha256=inputs.acquisition_manifest_sha256,
                    ortho_manifest_sha256=inputs.ortho_manifest_sha256,
                    governing_files=governing,
                    qa_invalid_counts=qa_counts,
                    spectral_copy_audit=spectral_copy_audit,
                )
                scene_manifests[scene.site] = {
                    "path": str(manifest.relative_to(staging_root)),
                    "sha256": basic_ortho.sha256_file(manifest),
                }

            payload: dict[str, object] = {
                "schema_version": "1.0",
                "manifest_type": "m1b_basic_ortho_mapping_run",
                "mode": "mapping-only",
                "scientific_endpoint_values_inspected": False,
                "scientific_outputs_produced": False,
                "execution_id": run_id,
                "execution_identity_inputs": run_identity_inputs,
                "scene_manifests": scene_manifests,
                "input_snapshot_evidence": snapshot_evidence,
                "residual_dependency_trust": residual_dependency_trust,
            }
            manifest_path = staging_root / "mapping_run_manifest.json"
            basic_ortho.strict_json_dump(manifest_path, payload)
            return manifest_path

        def verify_staged_bundle(bundle_root: Path) -> None:
            _revalidate_sealed_sources(sealed)
            _require_governing_unchanged(governing, binding)
            basic_ortho.verify_mapping_run_bundle(
                bundle_root,
                expected_run_id=run_id,
                expected_scenes=expected_scenes,
                expected_semantics=expected_semantics,
                expected_protocol_sha256=sealed.protocol_hash,
                expected_snapshot_evidence=snapshot_evidence,
                expected_residual_dependency_trust=residual_dependency_trust,
            )

        return basic_ortho.atomic_write_run_bundle(
            output_dir,
            run_id,
            writer=write_staged_bundle,
            verifier=verify_staged_bundle,
        )


def run_resource_pilot(
    *,
    preregistration: Path,
    acquisition_manifest: Path,
    ortho_manifest: Path,
    output_dir: Path,
    pilot_site: str | None = None,
    pilot_branch: str | None = None,
) -> Path:
    """Measure one declared reflectance load without computing or writing endpoints."""
    binding, basic_ortho, governing = _start_run()
    if pilot_site is None:
        pilot_site = basic_ortho.RESOURCE_PILOT_DEFAULT_SITE
    if pilot_branch is None:
        pilot_branch = basic_ortho.RESOURCE_PILOT_DEFAULT_BRANCH
    with _sealed_run_inputs(
        basic_ortho,
        preregistration=preregistration,
        acquisition_manifest=acquisition_manifest,
        ortho_manifest=ortho_manifest,
        hdf_mode="pilot",
        pilot_site=pilot_site,
        pilot_branch=pilot_branch,
    ) as sealed:
        inputs = sealed.inputs
        scene = basic_ortho.select_resource_pilot_scene(
            inputs,
            site=pilot_site,
            branch=pilot_branch,
        )
        if pilot_branch == "B":
            input_path = scene.basic_path
            input_key = (scene.site, "basic")
            expected_size = scene.basic_size_bytes
            expected_sha256 = scene.basic_sha256
            expected_shape = scene.basic_stac_shape
        else:
            input_path = scene.ortho_path
            input_key = (scene.site, "ortho")
            expected_size = scene.ortho_size_bytes
            expected_sha256 = scene.ortho_sha256
            expected_shape = scene.ortho_stac_shape
        identity_inputs = basic_ortho.resource_pilot_identity_inputs(
            scene,
            branch=pilot_branch,
            protocol_sha256=sealed.protocol_hash,
            acquisition_manifest_sha256=inputs.acquisition_manifest_sha256,
            ortho_manifest_sha256=inputs.ortho_manifest_sha256,
            governing_files=governing,
        )
        run_id = basic_ortho.resource_pilot_execution_identity(identity_inputs)

        def write_staged_bundle(staging_root: Path) -> Path:
            with _snapshot_reader(sealed.hdf[input_key]) as input_snapshot:
                telemetry = basic_ortho.measure_resource_pilot_load(
                    input_path,
                    input_snapshot=input_snapshot,
                    branch=pilot_branch,
                    expected_shape=expected_shape,
                    expected_size_bytes=expected_size,
                    expected_sha256=expected_sha256,
                )
            payload = basic_ortho.resource_pilot_document(
                scene,
                branch=pilot_branch,
                protocol_sha256=sealed.protocol_hash,
                acquisition_manifest_sha256=inputs.acquisition_manifest_sha256,
                ortho_manifest_sha256=inputs.ortho_manifest_sha256,
                governing_files=governing,
                telemetry=telemetry,
            )
            if payload["execution_id"] != run_id:
                raise basic_ortho.ProtocolError(
                    "resource-pilot execution identity changed during staging"
                )
            manifest_path = staging_root / "resource_pilot_manifest.json"
            basic_ortho.strict_json_dump(manifest_path, payload)
            return manifest_path

        def verify_staged_bundle(bundle_root: Path) -> None:
            _revalidate_sealed_sources(sealed)
            _require_governing_unchanged(governing, binding)
            basic_ortho.verify_resource_pilot_bundle(
                bundle_root,
                expected_run_id=run_id,
                expected_protocol_sha256=sealed.protocol_hash,
            )

        return basic_ortho.atomic_write_run_bundle(
            output_dir,
            run_id,
            writer=write_staged_bundle,
            verifier=verify_staged_bundle,
        )


def _main_with_binding(
    binding: _RuntimeBinding,
    argv: Sequence[str] | None = None,
) -> None:
    basic_ortho = binding.basic_ortho
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--design-only", action="store_true")
    mode.add_argument("--schema-only", action="store_true")
    mode.add_argument("--mapping-only", action="store_true")
    mode.add_argument("--resource-pilot", action="store_true")
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--acquisition-manifest", type=Path, default=DEFAULT_ACQUISITION_MANIFEST)
    parser.add_argument("--ortho-manifest", type=Path, default=DEFAULT_ORTHO_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pilot-site",
        choices=tuple(basic_ortho.FROZEN_SCENES),
        default=basic_ortho.RESOURCE_PILOT_DEFAULT_SITE,
    )
    parser.add_argument(
        "--pilot-branch",
        choices=tuple(basic_ortho.RESOURCE_PILOT_BRANCH_ASSETS),
        default=basic_ortho.RESOURCE_PILOT_DEFAULT_BRANCH,
    )
    args = parser.parse_args(argv)

    common = {
        "preregistration": args.preregistration,
        "acquisition_manifest": args.acquisition_manifest,
        "ortho_manifest": args.ortho_manifest,
        "output_dir": args.output_dir,
    }
    if args.design_only:
        artifact = run_design_only(**common)
    elif args.schema_only:
        artifact = run_schema_only(**common)
    elif args.mapping_only:
        artifact = run_mapping_only(**common)
    else:
        artifact = run_resource_pilot(
            **common,
            pilot_site=args.pilot_site,
            pilot_branch=args.pilot_branch,
        )
    print(f"wrote {artifact}")


def main(argv: Sequence[str] | None = None) -> None:
    binding = _bootstrap_runtime()
    try:
        _main_with_binding(binding, argv)
    finally:
        _teardown_runtime(binding)


if __name__ == "__main__":
    main()
