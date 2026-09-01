"""Descriptor-bound launcher for the M1b basic-vs-ortho runner.

This minimal file is residual execution-bootstrap trust: Python reads and
compiles it before it can bind its own bytes. Its only job is to open the
governing runner once, hash that payload, and compile and execute those exact
same bytes.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

RUNNER_PATH = Path(__file__).with_name("run_basic_ortho_sensitivity.py")
_RUNNER_MODULE_NAME = "_tanager_rocks_m1b_runner"


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _read_runner_descriptor(
    path: Path,
) -> tuple[Path, bytes, str, tuple[int, int, int, int, int]]:
    """Read one regular runner through no-follow ancestor and file descriptors."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("runner launcher requires O_NOFOLLOW and O_DIRECTORY support")
    absolute = Path(os.path.abspath(path))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fds: list[int] = []
    try:
        directory_fd = os.open(absolute.anchor, directory_flags)
        directory_fds.append(directory_fd)
        for part in absolute.parts[1:-1]:
            directory_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            directory_fds.append(directory_fd)
        file_fd = os.open(
            absolute.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        initial_info = os.fstat(file_fd)
        if not stat.S_ISREG(initial_info.st_mode):
            os.close(file_fd)
            raise RuntimeError(f"runner source is not a regular file: {absolute}")
    except OSError as error:
        raise RuntimeError(
            f"runner source cannot be opened without following symlinks: {absolute}"
        ) from error
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)

    with os.fdopen(file_fd, "rb") as handle:
        payload = handle.read()
        final_info = os.fstat(handle.fileno())
    initial = _stat_identity(initial_info)
    final = _stat_identity(final_info)
    if initial != final or len(payload) != initial_info.st_size:
        raise RuntimeError(f"runner source changed during descriptor read: {absolute}")
    return absolute, payload, hashlib.sha256(payload).hexdigest(), final


def load_runner_module(
    path: Path = RUNNER_PATH,
    *,
    module_name: str = _RUNNER_MODULE_NAME,
) -> ModuleType:
    """Compile and execute the exact descriptor-bound runner payload."""
    if module_name in sys.modules:
        raise RuntimeError(f"runner module name is already loaded: {module_name}")
    absolute, payload, payload_sha256, payload_stat = _read_runner_descriptor(path)
    module = ModuleType(module_name)
    module.__file__ = str(absolute)
    module.__package__ = ""
    module.__dict__.update(
        {
            "_DESCRIPTOR_BOUND_RUNNER_PATH": str(absolute),
            "_DESCRIPTOR_BOUND_RUNNER_SOURCE": payload,
            "_DESCRIPTOR_BOUND_RUNNER_SHA256": payload_sha256,
            "_DESCRIPTOR_BOUND_RUNNER_STAT": payload_stat,
            "_DESCRIPTOR_BOUND_LAUNCHER_PATH": str(Path(os.path.abspath(__file__))),
        }
    )
    sys.modules[module_name] = module
    try:
        code = compile(payload, str(absolute), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    return module


def main(argv: Sequence[str] | None = None) -> None:
    runner = load_runner_module()
    runner_main: Any = getattr(runner, "main", None)
    if not callable(runner_main):
        raise RuntimeError("descriptor-bound runner does not define main()")
    try:
        runner_main(argv)
    finally:
        if sys.modules.get(_RUNNER_MODULE_NAME) is runner:
            del sys.modules[_RUNNER_MODULE_NAME]


if __name__ == "__main__":
    main()
