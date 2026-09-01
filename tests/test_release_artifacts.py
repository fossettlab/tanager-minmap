"""Focused synthetic tests for the release-artifact boundary."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_checker() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_release_artifacts.py"
    spec = importlib.util.spec_from_file_location("release_artifact_checker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load release checker from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()
ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_STEM = "tanager_rocks-0.1.0"
PACKAGE_BYTES = b'__version__ = "0.1.0"\n'
PROJECT_FILE = b'[project]\nname = "tanager-rocks"\nversion = "0.1.0"\n'
PKG_INFO = b"Metadata-Version: 2.4\nName: tanager-rocks\nVersion: 0.1.0\n\n"


def _record_bytes(files: dict[str, bytes], record_name: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", len(data)))
    writer.writerow((record_name, "", ""))
    return output.getvalue().encode()


def _base_sdist_files() -> dict[str, bytes]:
    top_level = {
        ".gitignore": b"dist/\n",
        "CITATION.cff": b"cff-version: 1.2.0\n",
        "LICENSE": b"synthetic license\n",
        "METHODS.md": b"# Methods\n",
        "NOTICE.md": b"# Notice\n",
        "PKG-INFO": PKG_INFO,
        "README.md": b"# Tanager Rocks\n",
        "REPRODUCIBILITY.md": b"# Reproducibility\n",
        "pyproject.toml": PROJECT_FILE,
    }
    files = {f"{ARCHIVE_STEM}/{name}": data for name, data in top_level.items()}
    files[f"{ARCHIVE_STEM}/src/tanager_rocks/__init__.py"] = PACKAGE_BYTES
    return files


def _base_wheel_files() -> tuple[dict[str, bytes], str]:
    dist_info = f"{ARCHIVE_STEM}.dist-info"
    files = {
        "tanager_rocks/__init__.py": PACKAGE_BYTES,
        f"{dist_info}/METADATA": PKG_INFO,
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{dist_info}/entry_points.txt": (
            b"[console_scripts]\ntanager-minmap = tanager_rocks.cli:main\n"
        ),
        f"{dist_info}/licenses/LICENSE": b"synthetic license\n",
        f"{dist_info}/licenses/NOTICE.md": b"# Notice\n",
    }
    return files, f"{dist_info}/RECORD"


def _write_source(source_root: Path) -> None:
    files = {
        ".gitignore": b"dist/\n",
        "CITATION.cff": b"cff-version: 1.2.0\n",
        "LICENSE": b"synthetic license\n",
        "METHODS.md": b"# Methods\n",
        "NOTICE.md": b"# Notice\n",
        "README.md": b"# Tanager Rocks\n",
        "REPRODUCIBILITY.md": b"# Reproducibility\n",
        "pyproject.toml": PROJECT_FILE,
        "src/tanager_rocks/__init__.py": PACKAGE_BYTES,
    }
    for relative_name, data in files.items():
        path = source_root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _write_wheel(path: Path, files: dict[str, bytes], record_name: str) -> None:
    payloads = dict(files)
    payloads[record_name] = _record_bytes(payloads, record_name)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            archive.writestr(name, data)


def _write_sdist(
    path: Path,
    files: dict[str, bytes],
    special_member: tuple[str, bytes, str] | None = None,
) -> None:
    with tarfile.open(path, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if special_member is not None:
            name, member_type, linkname = special_member
            member = tarfile.TarInfo(name)
            member.type = member_type
            member.linkname = linkname
            archive.addfile(member)


def _make_artifacts(
    tmp_path: Path,
    *,
    wheel_package: bytes = PACKAGE_BYTES,
    sdist_package: bytes = PACKAGE_BYTES,
    extra_wheel: dict[str, bytes] | None = None,
    extra_sdist: dict[str, bytes] | None = None,
    sdist_overrides: dict[str, bytes] | None = None,
    special_member: tuple[str, bytes, str] | None = None,
) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_source(source_root)

    wheel_files, record_name = _base_wheel_files()
    wheel_files["tanager_rocks/__init__.py"] = wheel_package
    wheel_files.update(extra_wheel or {})
    wheel_path = tmp_path / f"{ARCHIVE_STEM}-py3-none-any.whl"
    _write_wheel(wheel_path, wheel_files, record_name)

    sdist_files = _base_sdist_files()
    sdist_files[f"{ARCHIVE_STEM}/src/tanager_rocks/__init__.py"] = sdist_package
    sdist_files.update(extra_sdist or {})
    for relative_name, data in (sdist_overrides or {}).items():
        sdist_files[f"{ARCHIVE_STEM}/{relative_name}"] = data
    sdist_path = tmp_path / f"{ARCHIVE_STEM}.tar.gz"
    _write_sdist(sdist_path, sdist_files, special_member)
    return source_root, wheel_path, sdist_path


def test_expected_hatchling_automatic_members_are_explicit() -> None:
    assert checker.SDIST_HATCHLING_AUTOMATIC_MEMBERS == frozenset(
        {".gitignore", "LICENSE", "NOTICE.md", "PKG-INFO", "README.md", "pyproject.toml"}
    )
    assert checker.WHEEL_HATCHLING_AUTOMATIC_DIST_INFO_MEMBERS == frozenset(
        {
            "METADATA",
            "RECORD",
            "WHEEL",
            "entry_points.txt",
            "licenses/LICENSE",
            "licenses/NOTICE.md",
        }
    )


def test_sdist_target_is_a_positive_allowlist() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    target = document["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert target == {
        "include": [
            "/src/tanager_rocks",
            "/README.md",
            "/METHODS.md",
            "/REPRODUCIBILITY.md",
            "/LICENSE",
            "/NOTICE.md",
            "/CITATION.cff",
        ],
        "exclude": ["**/__pycache__", "**/*.pyc", "**/.DS_Store"],
    }


def test_valid_synthetic_archives_pass(tmp_path: Path) -> None:
    source_root, wheel_path, sdist_path = _make_artifacts(tmp_path)

    report = checker.check_release_artifacts(wheel_path, sdist_path, source_root)

    assert report.package_members == ("__init__.py",)
    assert f"{ARCHIVE_STEM}/PKG-INFO" in report.sdist_members
    assert f"{ARCHIVE_STEM}.dist-info/RECORD" in report.wheel_members


@pytest.mark.parametrize(
    "unsafe_name",
    [f"{ARCHIVE_STEM}/../escape.py", "/absolute.py"],
)
def test_unsafe_member_names_fail(tmp_path: Path, unsafe_name: str) -> None:
    source_root, wheel_path, sdist_path = _make_artifacts(
        tmp_path, extra_sdist={unsafe_name: b"escape\n"}
    )

    with pytest.raises(checker.ArtifactCheckError, match="absolute|traversal"):
        checker.check_release_artifacts(wheel_path, sdist_path, source_root)


@pytest.mark.parametrize(
    ("member_type", "expected"),
    [(tarfile.SYMTYPE, "symlink"), (tarfile.LNKTYPE, "hardlink")],
)
def test_tar_links_fail(tmp_path: Path, member_type: bytes, expected: str) -> None:
    link_name = f"{ARCHIVE_STEM}/src/tanager_rocks/link.py"
    target = f"{ARCHIVE_STEM}/src/tanager_rocks/__init__.py"
    source_root, wheel_path, sdist_path = _make_artifacts(
        tmp_path, special_member=(link_name, member_type, target)
    )

    with pytest.raises(checker.ArtifactCheckError, match=expected):
        checker.check_release_artifacts(wheel_path, sdist_path, source_root)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"/Users/example/private\n", "/Users/"),
        (b"file:///private/source\n", "file://"),
        (b'path = "../tanager-spec"\n', "../tanager-spec"),
        (b"editable = true\n", "editable declaration"),
    ],
)
def test_forbidden_content_fails(tmp_path: Path, payload: bytes, expected: str) -> None:
    source_root, wheel_path, sdist_path = _make_artifacts(
        tmp_path, sdist_overrides={"README.md": payload}
    )

    with pytest.raises(checker.ArtifactCheckError, match=re.escape(expected)):
        checker.check_release_artifacts(wheel_path, sdist_path, source_root)


def test_direct_url_member_fails(tmp_path: Path) -> None:
    member = f"{ARCHIVE_STEM}.dist-info/direct_url.json"
    source_root, wheel_path, sdist_path = _make_artifacts(tmp_path, extra_wheel={member: b"{}\n"})

    with pytest.raises(checker.ArtifactCheckError, match="direct_url.json"):
        checker.check_release_artifacts(wheel_path, sdist_path, source_root)


def test_wheel_record_integrity_failure_fails(tmp_path: Path) -> None:
    source_root, wheel_path, sdist_path = _make_artifacts(tmp_path)
    with zipfile.ZipFile(wheel_path) as archive:
        payloads = {info.filename: archive.read(info) for info in archive.infolist()}
    record_name = f"{ARCHIVE_STEM}.dist-info/RECORD"
    payloads[record_name] = payloads[record_name].replace(b"sha256=", b"sha256=A", 1)
    with zipfile.ZipFile(wheel_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            archive.writestr(name, data)

    with pytest.raises(checker.ArtifactCheckError, match="wheel RECORD"):
        checker.check_release_artifacts(wheel_path, sdist_path, source_root)


def test_package_member_mismatch_fails(tmp_path: Path) -> None:
    source_root, wheel_path, sdist_path = _make_artifacts(
        tmp_path, extra_wheel={"tanager_rocks/extra.py": b"pass\n"}
    )

    with pytest.raises(checker.ArtifactCheckError, match="package member mismatch"):
        checker.check_release_artifacts(wheel_path, sdist_path, source_root)


def test_package_byte_mismatch_fails(tmp_path: Path) -> None:
    source_root, wheel_path, sdist_path = _make_artifacts(
        tmp_path, wheel_package=b'__version__ = "different"\n'
    )

    with pytest.raises(checker.ArtifactCheckError, match="package byte mismatch"):
        checker.check_release_artifacts(wheel_path, sdist_path, source_root)
