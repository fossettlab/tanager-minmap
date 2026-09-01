#!/usr/bin/env python3
"""Fail-closed checks for the local Tanager Rocks release archives."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import re
import stat
import sys
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
PROJECT_NAME: Final = "tanager-rocks"
PACKAGE_NAME: Final = "tanager_rocks"
SOURCE_PACKAGE_PATH: Final = PurePosixPath("src") / PACKAGE_NAME

SDIST_CONFIGURED_MEMBERS: Final = frozenset(
    {
        "CITATION.cff",
        "LICENSE",
        "METHODS.md",
        "NOTICE.md",
        "README.md",
        "REPRODUCIBILITY.md",
    }
)
# Hatchling forces the project/readme/license/VCS files into an sdist and
# generates PKG-INFO. Keeping this set exact makes backend drift fail closed.
SDIST_HATCHLING_AUTOMATIC_MEMBERS: Final = frozenset(
    {".gitignore", "LICENSE", "NOTICE.md", "PKG-INFO", "README.md", "pyproject.toml"}
)
# These are the exact dist-info files Hatchling generates for this project.
WHEEL_HATCHLING_AUTOMATIC_DIST_INFO_MEMBERS: Final = frozenset(
    {
        "METADATA",
        "RECORD",
        "WHEEL",
        "entry_points.txt",
        "licenses/LICENSE",
        "licenses/NOTICE.md",
    }
)

FORBIDDEN_LITERALS: Final = (
    ("/Users/", b"/users/"),
    ("file://", b"file://"),
    ("../tanager-spec", b"../tanager-spec"),
)
EDITABLE_DECLARATION_PATTERNS: Final = (
    re.compile(rb"(?im)(?<![A-Za-z0-9_-])[\"']?editable[\"']?\s*(?:=|:)"),
    re.compile(rb"(?im)(?:^|[\r\n])\s*(?:-e|--editable)(?:\s|=)+"),
)


class ArtifactCheckError(RuntimeError):
    """Raised when a release archive violates the local release contract."""


@dataclass(frozen=True)
class ProjectIdentity:
    """Project identity and the corresponding normalized archive stem."""

    name: str
    version: str
    archive_stem: str


@dataclass(frozen=True)
class ArchiveContents:
    """Validated archive members read directly without extraction."""

    files: dict[str, bytes]
    directories: frozenset[str]
    members: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactReport:
    """Members accepted by the release-artifact checks."""

    wheel_members: tuple[str, ...]
    sdist_members: tuple[str, ...]
    package_members: tuple[str, ...]


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _load_project_identity(source_root: Path) -> ProjectIdentity:
    pyproject_path = source_root / "pyproject.toml"
    if pyproject_path.is_symlink() or not pyproject_path.is_file():
        raise ArtifactCheckError(
            f"source project file is missing or is a symlink: {pyproject_path}"
        )

    try:
        document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = document["project"]
        name = project["name"]
        version = project["version"]
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise ArtifactCheckError(
            f"cannot read project identity from {pyproject_path}: {exc}"
        ) from exc

    if name != PROJECT_NAME or not isinstance(version, str) or not version:
        raise ArtifactCheckError(
            f"expected project {PROJECT_NAME!r} with a non-empty string version, "
            f"found name={name!r}, version={version!r}"
        )
    if "/" in version or "\\" in version:
        raise ArtifactCheckError(f"project version is not archive-safe: {version!r}")

    stem = f"{_normalize_distribution_name(name)}-{version}"
    return ProjectIdentity(name=name, version=version, archive_stem=stem)


def _validate_member_name(raw_name: str, archive_label: str) -> str:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise ArtifactCheckError(f"{archive_label} has an invalid member name: {raw_name!r}")
    if raw_name.startswith("/") or re.match(r"^[A-Za-z]:", raw_name):
        raise ArtifactCheckError(f"{archive_label} has an absolute member name: {raw_name!r}")

    name = raw_name[:-1] if raw_name.endswith("/") else raw_name
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArtifactCheckError(f"{archive_label} has a traversal member name: {raw_name!r}")
    return name


def _forbidden_content_labels(data: bytes) -> tuple[str, ...]:
    lowered = data.lower()
    labels = [label for label, needle in FORBIDDEN_LITERALS if needle in lowered]
    if any(pattern.search(data) for pattern in EDITABLE_DECLARATION_PATTERNS):
        labels.append("editable declaration")
    return tuple(labels)


def _check_forbidden_content(data: bytes, archive_label: str, member_name: str) -> None:
    labels = _forbidden_content_labels(data)
    if labels:
        rendered = ", ".join(repr(label) for label in labels)
        raise ArtifactCheckError(
            f"{archive_label} member {member_name!r} contains forbidden content: {rendered}"
        )


def _register_member(name: str, seen: set[str], archive_label: str) -> None:
    if name in seen:
        raise ArtifactCheckError(f"{archive_label} has a duplicate member name: {name!r}")
    seen.add(name)


def _zip_member_kind(info: zipfile.ZipInfo) -> str:
    unix_mode = info.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    if file_type == stat.S_IFLNK:
        return "symlink"
    if info.is_dir() or file_type == stat.S_IFDIR:
        return "directory"
    if file_type not in {0, stat.S_IFREG}:
        return "special"
    return "file"


def _read_wheel(path: Path) -> ArchiveContents:
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    seen: set[str] = set()

    try:
        with zipfile.ZipFile(path) as archive:
            _check_forbidden_content(archive.comment, "wheel", "<archive comment>")
            entries: list[tuple[zipfile.ZipInfo, str, str]] = []
            for info in archive.infolist():
                name = _validate_member_name(info.filename, "wheel")
                _register_member(name, seen, "wheel")
                _check_forbidden_content(info.comment, "wheel", f"{name} comment")
                _check_forbidden_content(info.extra, "wheel", f"{name} extra metadata")
                if info.flag_bits & 0x1:
                    raise ArtifactCheckError(f"wheel member is encrypted: {name!r}")
                kind = _zip_member_kind(info)
                if kind == "symlink":
                    raise ArtifactCheckError(f"wheel contains a symlink: {name!r}")
                if kind == "special":
                    raise ArtifactCheckError(f"wheel contains a non-regular member: {name!r}")
                entries.append((info, name, kind))

            bad_member = archive.testzip()
            if bad_member is not None:
                raise ArtifactCheckError(f"wheel CRC check failed for member: {bad_member!r}")

            for info, name, kind in entries:
                if kind == "directory":
                    directories.add(name)
                    continue
                data = archive.read(info)
                if len(data) != info.file_size:
                    raise ArtifactCheckError(f"wheel member has an inconsistent size: {name!r}")
                _check_forbidden_content(data, "wheel", name)
                files[name] = data
    except ArtifactCheckError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ArtifactCheckError(f"cannot read wheel archive {path}: {exc}") from exc

    return ArchiveContents(files, frozenset(directories), tuple(sorted(seen)))


def _tar_metadata_bytes(member: tarfile.TarInfo) -> bytes:
    fields = [member.name, member.linkname, member.uname, member.gname]
    fields.extend(f"{key}={value}" for key, value in sorted(member.pax_headers.items()))
    return "\n".join(fields).encode("utf-8", errors="surrogateescape")


def _read_sdist(path: Path) -> ArchiveContents:
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    seen: set[str] = set()

    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
            validated: list[tuple[tarfile.TarInfo, str]] = []
            for member in members:
                name = _validate_member_name(member.name, "sdist")
                _register_member(name, seen, "sdist")
                _check_forbidden_content(_tar_metadata_bytes(member), "sdist", f"{name} metadata")
                if member.issym():
                    raise ArtifactCheckError(f"sdist contains a symlink: {name!r}")
                if member.islnk():
                    raise ArtifactCheckError(f"sdist contains a hardlink: {name!r}")
                if not (member.isfile() or member.isdir()):
                    raise ArtifactCheckError(f"sdist contains a non-regular member: {name!r}")
                validated.append((member, name))

            for member, name in validated:
                if member.isdir():
                    directories.add(name)
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ArtifactCheckError(f"cannot read sdist member: {name!r}")
                data = stream.read()
                if len(data) != member.size:
                    raise ArtifactCheckError(f"sdist member has an inconsistent size: {name!r}")
                _check_forbidden_content(data, "sdist", name)
                files[name] = data
    except ArtifactCheckError:
        raise
    except (EOFError, OSError, tarfile.TarError) as exc:
        raise ArtifactCheckError(f"cannot read sdist archive {path}: {exc}") from exc

    return ArchiveContents(files, frozenset(directories), tuple(sorted(seen)))


def _assert_directories_are_ancestors(contents: ArchiveContents, archive_label: str) -> None:
    expected: set[str] = set()
    for name in contents.files:
        expected.update(str(parent) for parent in PurePosixPath(name).parents if str(parent) != ".")
    unexpected = sorted(contents.directories - expected)
    if unexpected:
        raise ArtifactCheckError(f"{archive_label} has unexpected empty directories: {unexpected}")


def _assert_no_direct_url(contents: ArchiveContents, archive_label: str) -> None:
    matches = sorted(
        name
        for name in (*contents.files, *contents.directories)
        if PurePosixPath(name).name.casefold() == "direct_url.json"
    )
    if matches:
        raise ArtifactCheckError(f"{archive_label} contains direct_url.json: {matches}")


def _validate_metadata_identity(
    data: bytes, identity: ProjectIdentity, archive_label: str, member_name: str
) -> None:
    try:
        metadata = BytesParser(policy=policy.default).parsebytes(data)
    except (TypeError, ValueError) as exc:
        raise ArtifactCheckError(
            f"cannot parse {archive_label} metadata {member_name!r}: {exc}"
        ) from exc
    if metadata.get("Name") != identity.name or metadata.get("Version") != identity.version:
        raise ArtifactCheckError(
            f"{archive_label} metadata identity mismatch in {member_name!r}: "
            f"Name={metadata.get('Name')!r}, Version={metadata.get('Version')!r}"
        )


def _validate_wheel_record(files: dict[str, bytes], record_name: str) -> None:
    try:
        text = files[record_name].decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (KeyError, UnicodeError, csv.Error) as exc:
        raise ArtifactCheckError(f"cannot parse wheel RECORD: {exc}") from exc

    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise ArtifactCheckError(f"wheel RECORD row does not have three fields: {row!r}")
        raw_name, encoded_hash, raw_size = row
        name = _validate_member_name(raw_name, "wheel RECORD")
        _register_member(name, recorded, "wheel RECORD")
        if name not in files:
            raise ArtifactCheckError(f"wheel RECORD names a missing member: {name!r}")
        if name == record_name:
            if encoded_hash or raw_size:
                raise ArtifactCheckError("wheel RECORD must leave its own hash and size empty")
            continue
        if not encoded_hash.startswith("sha256=") or not raw_size:
            raise ArtifactCheckError(f"wheel RECORD lacks a sha256 hash or size for {name!r}")
        try:
            digest_text = encoded_hash.removeprefix("sha256=")
            padding = "=" * (-len(digest_text) % 4)
            recorded_digest = base64.b64decode(digest_text + padding, altchars=b"-_", validate=True)
            recorded_size = int(raw_size)
        except (ValueError, binascii.Error) as exc:
            raise ArtifactCheckError(
                f"wheel RECORD has invalid integrity data for {name!r}"
            ) from exc
        data = files[name]
        if recorded_size != len(data) or recorded_digest != hashlib.sha256(data).digest():
            raise ArtifactCheckError(f"wheel RECORD integrity mismatch for {name!r}")

    if recorded != set(files):
        missing = sorted(set(files) - recorded)
        raise ArtifactCheckError(f"wheel RECORD omits archive members: {missing}")


def _is_transient_package_path(relative_name: str) -> bool:
    path = PurePosixPath(relative_name)
    return "__pycache__" in path.parts or path.name == ".DS_Store" or path.suffix == ".pyc"


def _load_source_package(source_root: Path) -> dict[str, bytes]:
    package_root = source_root / SOURCE_PACKAGE_PATH
    if package_root.is_symlink() or not package_root.is_dir():
        raise ArtifactCheckError(f"source package is missing or is a symlink: {package_root}")

    files: dict[str, bytes] = {}
    for path in sorted(package_root.rglob("*")):
        relative_name = path.relative_to(package_root).as_posix()
        if _is_transient_package_path(relative_name):
            continue
        if path.is_symlink():
            raise ArtifactCheckError(f"source package contains a symlink: {relative_name!r}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArtifactCheckError(
                f"source package contains a non-regular entry: {relative_name!r}"
            )
        try:
            files[relative_name] = path.read_bytes()
        except OSError as exc:
            raise ArtifactCheckError(f"cannot read source package file {path}: {exc}") from exc
    if not files:
        raise ArtifactCheckError(f"source package has no release files: {package_root}")
    return files


def _validate_wheel(
    path: Path, contents: ArchiveContents, identity: ProjectIdentity
) -> dict[str, bytes]:
    expected_name = f"{identity.archive_stem}-py3-none-any.whl"
    if path.name != expected_name:
        raise ArtifactCheckError(
            f"unexpected wheel filename: {path.name!r}; expected {expected_name!r}"
        )

    _assert_directories_are_ancestors(contents, "wheel")
    _assert_no_direct_url(contents, "wheel")
    dist_info = f"{identity.archive_stem}.dist-info"
    package_prefix = f"{PACKAGE_NAME}/"
    dist_info_prefix = f"{dist_info}/"
    package_files: dict[str, bytes] = {}
    metadata_files: dict[str, bytes] = {}

    for name, data in contents.files.items():
        if name.startswith(package_prefix):
            relative_name = name.removeprefix(package_prefix)
            if not relative_name or _is_transient_package_path(relative_name):
                raise ArtifactCheckError(f"wheel contains a transient package member: {name!r}")
            package_files[relative_name] = data
        elif name.startswith(dist_info_prefix):
            metadata_files[name.removeprefix(dist_info_prefix)] = data
        else:
            raise ArtifactCheckError(f"wheel has an unexpected member: {name!r}")

    actual_metadata = frozenset(metadata_files)
    if actual_metadata != WHEEL_HATCHLING_AUTOMATIC_DIST_INFO_MEMBERS:
        missing = sorted(WHEEL_HATCHLING_AUTOMATIC_DIST_INFO_MEMBERS - actual_metadata)
        unexpected = sorted(actual_metadata - WHEEL_HATCHLING_AUTOMATIC_DIST_INFO_MEMBERS)
        raise ArtifactCheckError(
            f"wheel dist-info member mismatch; missing={missing}, unexpected={unexpected}"
        )

    metadata_name = f"{dist_info}/METADATA"
    _validate_metadata_identity(contents.files[metadata_name], identity, "wheel", metadata_name)
    _validate_wheel_record(contents.files, f"{dist_info}/RECORD")
    return package_files


def _validate_sdist(
    path: Path, contents: ArchiveContents, identity: ProjectIdentity
) -> dict[str, bytes]:
    expected_name = f"{identity.archive_stem}.tar.gz"
    if path.name != expected_name:
        raise ArtifactCheckError(
            f"unexpected sdist filename: {path.name!r}; expected {expected_name!r}"
        )

    _assert_directories_are_ancestors(contents, "sdist")
    _assert_no_direct_url(contents, "sdist")
    roots = {PurePosixPath(name).parts[0] for name in contents.members}
    if roots != {identity.archive_stem}:
        raise ArtifactCheckError(
            f"sdist must have exactly one root {identity.archive_stem!r}; found {sorted(roots)}"
        )

    root_prefix = f"{identity.archive_stem}/"
    package_prefix = f"{SOURCE_PACKAGE_PATH.as_posix()}/"
    exact_members = SDIST_CONFIGURED_MEMBERS | SDIST_HATCHLING_AUTOMATIC_MEMBERS
    relative_files: dict[str, bytes] = {}
    package_files: dict[str, bytes] = {}

    for name, data in contents.files.items():
        if not name.startswith(root_prefix):
            raise ArtifactCheckError(f"sdist member is outside its root: {name!r}")
        relative_name = name.removeprefix(root_prefix)
        relative_files[relative_name] = data
        if relative_name.startswith(package_prefix):
            package_name = relative_name.removeprefix(package_prefix)
            if not package_name or _is_transient_package_path(package_name):
                raise ArtifactCheckError(f"sdist contains a transient package member: {name!r}")
            package_files[package_name] = data
        elif relative_name not in exact_members:
            raise ArtifactCheckError(f"sdist has an unexpected member: {name!r}")

    missing = sorted(exact_members - relative_files.keys())
    if missing:
        raise ArtifactCheckError(f"sdist is missing required members: {missing}")

    pkg_info_name = f"{identity.archive_stem}/PKG-INFO"
    _validate_metadata_identity(contents.files[pkg_info_name], identity, "sdist", pkg_info_name)
    return package_files


def _compare_package_files(
    wheel_files: dict[str, bytes],
    sdist_files: dict[str, bytes],
    source_files: dict[str, bytes],
) -> tuple[str, ...]:
    wheel_names = set(wheel_files)
    sdist_names = set(sdist_files)
    source_names = set(source_files)
    if not (wheel_names == sdist_names == source_names):
        raise ArtifactCheckError(
            "package member mismatch; "
            f"wheel_extra={sorted(wheel_names - source_names)}, "
            f"sdist_extra={sorted(sdist_names - source_names)}, "
            f"wheel_missing={sorted(source_names - wheel_names)}, "
            f"sdist_missing={sorted(source_names - sdist_names)}"
        )

    for name in sorted(source_names):
        if wheel_files[name] != sdist_files[name] or wheel_files[name] != source_files[name]:
            raise ArtifactCheckError(
                f"package byte mismatch among wheel, sdist, and source: {name!r}"
            )
    return tuple(sorted(source_names))


def _validate_artifact_path(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ArtifactCheckError(f"{label} path is missing, non-regular, or a symlink: {path}")


def check_release_artifacts(
    wheel_path: Path, sdist_path: Path, source_root: Path = PROJECT_ROOT
) -> ArtifactReport:
    """Validate release archives against fixed boundaries and the source package.

    Parameters
    ----------
    wheel_path
        Path to the wheel archive.
    sdist_path
        Path to the gzipped source distribution.
    source_root
        Project root used for identity and package-byte comparison.

    Returns
    -------
    ArtifactReport
        Accepted archive and package member names.

    Raises
    ------
    ArtifactCheckError
        If either archive violates any release-artifact invariant.
    """

    wheel_path = Path(wheel_path)
    sdist_path = Path(sdist_path)
    source_root = Path(source_root)
    _validate_artifact_path(wheel_path, "wheel")
    _validate_artifact_path(sdist_path, "sdist")
    identity = _load_project_identity(source_root)
    source_files = _load_source_package(source_root)

    wheel = _read_wheel(wheel_path)
    sdist = _read_sdist(sdist_path)
    wheel_package = _validate_wheel(wheel_path, wheel, identity)
    sdist_package = _validate_sdist(sdist_path, sdist, identity)
    package_members = _compare_package_files(wheel_package, sdist_package, source_files)
    return ArtifactReport(wheel.members, sdist.members, package_members)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path, help="wheel archive to check")
    parser.add_argument("--sdist", required=True, type=Path, help="source archive to check")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = check_release_artifacts(args.wheel, args.sdist)
    except ArtifactCheckError as exc:
        print(f"release artifact check failed: {exc}", file=sys.stderr)
        return 1

    print("release artifact check passed")
    print("wheel members:")
    for name in report.wheel_members:
        print(f"  {name}")
    print("sdist members:")
    for name in report.sdist_members:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
