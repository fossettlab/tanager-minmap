"""Fail-closed contract and provenance helpers for public video releases.

Draft rendering deliberately remains convenient and permissive. Public release
rendering is a separate mode: it consumes a frozen contract, verifies every
non-Git media/scientific input by SHA-256, requires a clean tagged source tree,
forces redistribution-safe Tanager-derived visuals for beats 05 and 07, and
emits a self-checking release bundle. No field is inferred from account state or
working notes; unavailable rights/provenance evidence remains a blocking null in
the contract templates.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import secrets
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import warnings
import zlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from common import (
    DISSOLVE_D,
    ENCODE_ARGS,
    FONT_BOLD,
    FONT_REGULAR,
    FPS,
    HEIGHT,
    ROOT,
    SEGMENT_FILES,
    WIDTH,
)

CONTRACT_KIND = "tanager-rocks-video-release-contract"
CONTRACT_SCHEMA_VERSION = 2
RENDER_KIND = "tanager-rocks-video-render-manifest"
RENDER_SCHEMA_VERSION = 2
CANONICAL_CONTRACT_LOCATOR = "video/manifests/release_contract.json"
CANONICAL_BUNDLE_PREFIX = "tanager-rocks-video-"
READY_SENTINEL = "READY.json"
CAPSULE_KIND = "tanager-rocks-video-execution-capsule"
CAPSULE_SEAL = "darwin-uf-immutable-v1"
RIGHTS_TRUST_ROOT = "operator-attestation"
RIGHTS_EVIDENCE_BASIS = "provider-account-and-generation-plan-records"
RIGHTS_PROVIDER_ACCOUNT_EVIDENCE = "generation_records.tts+music.account_plan"
LEGAL_RIGHTS_STATEMENT = "operator attestation; code does not establish legal rights"

BEAT_ORDER = ("00", "01", "02", "03", "04", "05", "06a", "06b", "07", "08")
DESIGNED_BEATS = {"00", "01", "02", "06a", "08"}
SWAPPABLE_TIERS = {
    "03": {"upgrade", "fallback"},
    "04": {"upgrade", "fallback"},
    "05": {"tanager-still"},
    # Strict mode always regenerates the corrected 0.58 correlation caption.
    # Draft mode may still use a locally rendered upgrade.
    "06b": {"fallback"},
    "07": {"tanager-still"},
}
SAFE_PUBLIC_BEAT_ASSETS = {
    "05": "video/build/v2/fallback_05.png",
    "07": "video/build/v2/fallback_07.png",
}
REQUIRED_FIGURES = {
    "submission/figures/bingham_20250911_191523_58_4001_amd_agp.png",
    "submission/figures/bingham_20250911_191523_58_4001_band_ablation.png",
    "submission/figures/bingham_rgb.png",
    "submission/figures/goldfield_20240925_185504_87_4001_emit_comparison.png",
    "submission/figures/goldfield_20240925_185504_87_4001_hero_mineral_map.png",
    "submission/figures/goldfield_rgb.png",
    "submission/figures/goldfield_spectra.png",
    "submission/figures/goldfield_validation_pair.png",
}
REQUIRED_NARRATION_TEXT = {
    "video/narration_script_v2.md",
    *(
        f"video/segments_v2/{name}.txt"
        for name in (
            "00_title",
            "01_hook",
            "02_stakes",
            "03_data",
            "04_ablation",
            "05_livemap",
            "06_validation",
            "07_amd",
            "08_close",
        )
    ),
}
REQUIRED_MUSIC_PLAN = "video/build/music_v2_composition_plan.json"
CURATED_PUBLIC_SOURCE_PATHS = frozenset(
    {
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "NOTICE.md",
        "docs/edit_plan.md",
        "docs/storyboard.md",
        "docs/video_reproduction.md",
        "scripts/video/anim_03_spectra.py",
        "scripts/video/anim_04_ablation.py",
        "scripts/video/anim_06b_emit.py",
        "scripts/video/assemble.py",
        "scripts/video/audio.py",
        "scripts/video/beats.py",
        "scripts/video/captions.py",
        "scripts/video/common.py",
        "scripts/video/overlays.py",
        "scripts/video/qc.py",
        "scripts/video/release.py",
        "scripts/video/render_v2.py",
        "scripts/video/stills_05_07.py",
        "scripts/video/test_release.py",
        "video/CREDITS.md",
        "video/README.md",
        "video/build/music_v2_composition_plan.json",
        "video/build/render_motif.py",
        "video/manifests/README.md",
        "video/manifests/doi_evidence.template.json",
        "video/manifests/music.template.json",
        "video/manifests/release_contract.schema.json",
        "video/manifests/release_contract.template.json",
        "video/manifests/render_manifest.schema.json",
        "video/manifests/tts.template.jsonl",
        "video/narration_script_v2.md",
        "uv.lock",
        *(
            f"video/segments_v2/{name}.txt"
            for name in (
                "00_title",
                "01_hook",
                "02_stakes",
                "03_data",
                "04_ablation",
                "05_livemap",
                "06_validation",
                "07_amd",
                "08_close",
            )
        ),
    }
)
REQUIRED_RIGHTS_ATTESTATIONS = (
    "claims_frozen",
    "planet_material_reviewed",
    "elevenlabs_narration_reviewed",
    "elevenlabs_music_reviewed",
    "third_party_visuals_reviewed",
)
MEDIA_MASTER_ROLES = {"narration_audio", "music", "beat_asset"}
INPUT_ROLES = {
    "figure",
    "narration_text",
    "narration_audio",
    "music",
    "music_plan",
    "beat_asset",
    "terms_snapshot",
}
CONTRACT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "release",
    "source",
    "rights",
    "distribution",
    "generation_records",
    "audio",
    "beats",
    "inputs",
}
RELEASE_KEYS = {
    "id",
    "title",
    "repository_url",
    "archive_doi",
    "doi_evidence",
    "output_basename",
    "bundle_name",
    "contract_locator",
}
SOURCE_KEYS = {"commit", "tag", "dirty"}
RIGHTS_KEYS = {
    *REQUIRED_RIGHTS_ATTESTATIONS,
    "reviewer",
    "reviewed_at_utc",
    "attestation",
    "trust_root",
    "operator",
    "evidence_basis",
    "provider_account_evidence",
    "generation_plan_evidence",
    "legal_rights_statement",
}
DISTRIBUTION_KEYS = {"include_media_masters", "master_asset_uri"}
GENERATION_RECORD_KEYS = {"tts", "music"}
AUDIO_KEYS = {"segments", "music_bed"}
ASSET_KEYS = {"role", "path", "sha256"}
BEAT_KEYS = {"tier", "asset_path"}
TTS_KEYS = {
    "schema_version",
    "segment",
    "selected",
    "provider",
    "text",
    "output",
    "generation",
    "voice",
    "model",
    "settings",
    "terms",
    "rights_review",
    "editorial",
}
MUSIC_KEYS = {
    "schema_version",
    "selected",
    "provider",
    "output",
    "generation",
    "model",
    "request",
    "composition_plan",
    "terms",
    "rights_review",
    "editorial",
}
PROVIDER_KEYS = {"name", "product"}
FILE_REF_KEYS = {"path", "sha256"}
TTS_GENERATION_KEYS = {"id", "generated_at_utc", "account_plan", "service_non_beta"}
TTS_VOICE_KEYS = {"name", "voice_id", "category", "library_status", "terms_url"}
TTS_MODEL_KEYS = {"model_id", "output_format"}
TTS_SETTINGS_KEYS = {
    "stability",
    "similarity_boost",
    "style",
    "speaker_boost",
    "speed",
    "seed",
    "unavailable_fields",
}
MUSIC_MODEL_KEYS = {"model_id"}
MUSIC_REQUEST_KEYS = {"output_format", "force_instrumental", "seed", "seed_unavailable_reason"}
TERMS_KEYS = {"url", "retrieved_at_utc", "snapshot_path", "snapshot_sha256"}
RIGHTS_REVIEW_KEYS = {
    "publication_rights_attested",
    "reviewer",
    "reviewed_at_utc",
}
EDITORIAL_KEYS = {"decision", "notes"}
DOI_EVIDENCE_KEYS = {
    "schema_version",
    "provider",
    "status",
    "record_id",
    "doi_url",
    "record_url",
    "provider_record",
    "minted_at_utc",
    "retrieved_at_utc",
    "reviewer",
    "reviewed_at_utc",
}
FORBIDDEN_PLACEHOLDER_RE = re.compile(
    r"(?:^|[._/ -])(pending|placeholder|example|test|todo|tbd|dummy|sample)(?:$|[._/ -])",
    re.IGNORECASE,
)
NON_PUBLIC_PLANS = {"free", "trial", "unknown", "pending", "placeholder", "test"}
REQUIRED_BUNDLE_FILES = {
    "CITATION.cff",
    "CREDITS.md",
    "LICENSE",
    "NOTICE.md",
    "release_contract.json",
    "render.json",
    "evidence/doi.json",
    "evidence/music.json",
    "evidence/tts.jsonl",
    "evidence/zenodo-record.json",
}
EXPECTED_AUTOMATED_QC_CHECKS = 6
EXPECTED_ACCEPTANCE_FRAMES = 21
EXPECTED_QC_CHECK_NAMES = (
    "WP-ASM total duration == VO length (+/-1 frame)",
    "WP-MUX stream params (1920x1080/30fps/yuv420p)",
    "WP-MUX loudness (-16 LUFS / -1.5 dBTP)",
    "WP-MUX A/V equal length",
    "WP-AUD vo_master.wav == VO length (+/-0.05)",
    "WP-SRT stakes cue + final cue text",
)
EXPECTED_QC_MESSAGES = (
    "replayed bundled picture and VO-master timing against frozen segment timing",
    "replayed bundled final MP4 stream structure and parameters",
    "replayed bundled final MP4 loudness and true peak",
    "replayed bundled final MP4 video/audio durations",
    "replayed bundled VO master against frozen segment timing",
    "replayed bundled strict SRT structure and required cue text",
)
QC_REPLAY_ARTIFACTS = (
    ("assembled_picture", "qc/replay/picture.mp4"),
    ("vo_master", "qc/replay/vo_master.wav"),
)
EXPECTED_ACCEPTANCE_FRAME_LABELS = (
    "WP-00-fade-up",
    "WP-00-title",
    "WP-01-lower-third",
    "WP-01-push-in",
    "WP-02-two-up",
    "WP-03-callout-2200",
    "WP-03-all-callouts",
    "WP-04-before",
    "WP-04-native-loss",
    "WP-05-legend",
    "WP-05-cluster",
    "WP-06a-validation",
    "WP-06a-highlights",
    "WP-06b-caption",
    "WP-07-legend",
    "WP-07-tailings",
    "WP-08-end-card",
    "WP-08-fade",
    "WP-ASM-00-01",
    "WP-ASM-06a-06b",
    "WP-ASM-07-08",
)
RENDER_KEYS = {
    "schema_version",
    "kind",
    "status",
    "generated_at_utc",
    "release",
    "render",
    "source_files",
    "inputs",
    "generation_evidence",
    "media_masters",
    "outputs",
    "environment",
    "qc",
    "rights",
}
RENDER_SECTION_KEYS = {
    "command",
    "contract_locator",
    "contract_sha256",
    "settings",
    "selected_tiers",
    "selected_sources",
    "workspace_isolation",
    "generated_artifacts",
}
RENDER_RELEASE_KEYS = {
    "id",
    "title",
    "source_commit",
    "source_tag",
    "source_dirty",
    "repository_url",
    "archive_doi",
    "output_basename",
    "bundle_name",
    "contract_locator",
}
RENDER_SETTINGS_KEYS = {
    "width",
    "height",
    "fps",
    "pixel_format",
    "video_encoder_args",
    "dissolve_seconds",
    "audio_codec",
    "audio_bitrate",
}
SOURCE_FILE_RECORD_KEYS = {"path", "sha256", "size_bytes"}
INPUT_RECORD_KEYS = {"role", "path", "sha256", "size_bytes"}
EVIDENCE_RECORD_KEYS = {"kind", "path", "sha256", "size_bytes"}
MASTER_RECORD_KEYS = {"source_path", "role", "path", "sha256", "size_bytes"}
OUTPUT_RECORD_KEYS = {"role", "path", "sha256", "size_bytes"}
FRAME_RECORD_KEYS = {"label", "path", "sha256", "size_bytes"}
QC_KEYS = {
    "automated_all_passed",
    "automated",
    "acceptance_frames",
    "human_playback",
    "replay",
}
QC_ROW_KEYS = {"name", "passed", "message"}
HUMAN_PLAYBACK_KEYS = {"status", "reviewed_at_utc", "reviewer", "notes"}
QC_REPLAY_KEYS = {"artifacts", "vo_segments", "measurements"}
QC_REPLAY_ARTIFACT_KEYS = {"role", "path", "sha256", "size_bytes"}
QC_VO_SEGMENT_KEYS = {"segment", "source_path", "source_sha256", "duration_seconds"}
QC_MEASUREMENT_KEYS = {
    "expected_vo_duration_seconds",
    "picture_duration_seconds",
    "vo_master_duration_seconds",
    "mux_duration_seconds",
    "video_duration_seconds",
    "audio_duration_seconds",
    "width",
    "height",
    "fps",
    "pixel_format",
    "integrated_lufs",
    "true_peak_dbtp",
    "srt_cue_count",
    "srt_stakes_cue_present",
    "srt_final_cue_present",
    "srt_jarosite_correction_present",
}
MEDIA_MASTER_KEYS = {"included", "external_uri", "files"}
ENVIRONMENT_KEYS = {
    "python",
    "executable",
    "platform",
    "ffmpeg",
    "ffprobe",
    "packages",
    "uv_lock_sha256",
    "fonts",
    "playwright",
    "chromium",
    "playwright_note",
    "worker_mode",
    "code_root",
    "capsule_manifest_sha256",
}
ENVIRONMENT_PACKAGE_KEYS = {"matplotlib", "numpy", "pillow", "scipy", "xarray"}
ZENODO_EXPORT_KEYS = {"id", "doi", "links", "files"}
ZENODO_LINK_KEYS = {"html", "self"}
ZENODO_FILE_KEYS = {"key", "size", "checksum"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DOI_URL_RE = re.compile(r"^https://doi\.org/10\.\d{4,9}/\S+$", re.IGNORECASE)
ZENODO_DOI_RE = re.compile(
    r"^https://doi\.org/10\.5281/zenodo\.(?P<record_id>[1-9]\d*)$", re.IGNORECASE
)
SRT_TIMING_RE = re.compile(
    r"^(?P<start>\d{2}:[0-5]\d:[0-5]\d,\d{3}) --> "
    r"(?P<end>\d{2}:[0-5]\d:[0-5]\d,\d{3})$"
)


class ReleaseContractError(RuntimeError):
    """The frozen release contract or bundle failed a release gate."""


class ReleaseCleanupWarning(RuntimeWarning):
    """A READY bundle was promoted, but its private staging cleanup was incomplete."""


def _report_cleanup_residue(final: Path, staging_root: Path, exc: Exception) -> None:
    """Report post-promotion residue without allowing reporting to fail finalization."""
    message = (
        f"release finalized at {final}; private staging cleanup is incomplete at "
        f"{staging_root}: {exc}"
    )
    try:
        warnings.warn(message, ReleaseCleanupWarning, stacklevel=3)
    except Exception:
        try:
            print(f"{ReleaseCleanupWarning.__name__}: {message}", file=sys.stderr)
        except Exception:
            pass


@dataclass(frozen=True)
class SrtCue:
    """One parsed SubRip cue with integer millisecond boundaries."""

    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class ProviderResponse:
    """Minimal dependency-injectable HTTPS response for final provider checks."""

    status: int
    url: str
    body: bytes


@dataclass(frozen=True)
class VerifiedAsset:
    """One repository-relative, regular, hash-verified input."""

    role: str
    relative_path: str
    path: Path
    sha256: str
    size_bytes: int

    def record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class BeatSource:
    """The only source tier and optional exact binary selected for a beat."""

    tier: str
    asset: VerifiedAsset | None


@dataclass(frozen=True)
class ReleaseStaging:
    """One isolated strict-render workspace and its promotable bundle."""

    root: Path
    work: Path
    snapshot: Path
    bundle: Path
    final: Path
    capsule_manifest_sha256: str = ""


@dataclass(frozen=True)
class ReleaseContract:
    """Parsed and fully verified release contract."""

    path: Path
    contract_sha256: str
    raw: dict[str, Any]
    release_id: str
    title: str
    source_commit: str
    source_tag: str
    repository_url: str
    archive_doi: str
    output_basename: str
    bundle_name: str
    contract_locator: str
    beats: dict[str, BeatSource]
    segment_paths: dict[str, Path]
    music_bed: Path
    assets: tuple[VerifiedAsset, ...]
    tts_record: VerifiedAsset
    music_record: VerifiedAsset
    doi_record: VerifiedAsset
    doi_provider_record: VerifiedAsset
    include_media_masters: bool
    master_asset_uri: str | None

    @property
    def strict_sources(self) -> dict[str, tuple[str, Path | None]]:
        return {
            name: (source.tier, None if source.asset is None else source.asset.path)
            for name, source in self.beats.items()
        }


def sha256_file(path: Path) -> str:
    """Stream one file into SHA-256 without loading a media master into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_relative(value: Any, field: str) -> str:
    rel = _require_string(value, field)
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or "\\" in rel or not pure.parts:
        raise ReleaseContractError(f"{field} must be a normalized repository-relative path")
    return pure.as_posix()


def _open_regular_nofollow(
    root: Path, relative_path: str, field: str
) -> tuple[int, os.stat_result]:
    """Open one root-relative regular file without following any symlink component."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise ReleaseContractError("strict release requires O_NOFOLLOW support")
    relative = _normalized_relative(relative_path, field)
    root_abs = root.absolute()
    if root_abs.is_symlink():
        raise ReleaseContractError(f"{field} root is a symlink: {root_abs}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    open_directories: list[int] = []
    try:
        directory_fd = os.open(root_abs, directory_flags)
        open_directories.append(directory_fd)
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            directory_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            open_directories.append(directory_fd)
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(file_fd)
            raise ReleaseContractError(f"{field} is not a regular file: {relative}")
        return file_fd, info
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ReleaseContractError(
            f"{field} is missing or has a non-directory ancestor: {relative}"
        ) from exc
    except OSError as exc:
        raise ReleaseContractError(
            f"{field} cannot be opened without following symlinks: {relative}"
        ) from exc
    finally:
        for directory_fd in reversed(open_directories):
            os.close(directory_fd)


def _hash_regular_nofollow(root: Path, relative_path: str, field: str) -> tuple[str, int]:
    file_fd, info = _open_regular_nofollow(root, relative_path, field)
    digest = hashlib.sha256()
    with os.fdopen(file_fd, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), info.st_size


def _read_regular_nofollow(root: Path, relative_path: str, field: str) -> bytes:
    file_fd, _info = _open_regular_nofollow(root, relative_path, field)
    with os.fdopen(file_fd, "rb") as handle:
        return handle.read()


def _relative_path_within(root: Path, path: Path, field: str) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ReleaseContractError(f"{field} escapes its required root") from exc
    return _normalized_relative(relative.as_posix(), field)


def _copy_regular_nofollow(
    source_root: Path,
    source_relative: str,
    destination_root: Path,
    destination_relative: str,
    *,
    expected_sha256: str,
    field: str,
) -> VerifiedAsset:
    """Copy one immutable source descriptor into a private snapshot and verify it."""
    source_relative = _normalized_relative(source_relative, f"{field}.source")
    destination_relative = _normalized_relative(destination_relative, f"{field}.destination")
    source_fd, source_info = _open_regular_nofollow(source_root, source_relative, field)
    destination = destination_root.joinpath(*PurePosixPath(destination_relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_output_path(destination_root, destination, f"{field} destination")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    digest = hashlib.sha256()
    try:
        destination_fd = os.open(destination, flags, 0o600)
    except OSError as exc:
        os.close(source_fd)
        raise ReleaseContractError(f"cannot create strict snapshot file: {destination}") from exc
    try:
        with os.fdopen(source_fd, "rb") as source, os.fdopen(destination_fd, "wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    actual = digest.hexdigest()
    if actual != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ReleaseContractError(
            f"snapshot hash mismatch for {source_relative}: "
            f"expected {expected_sha256}, found {actual}"
        )
    copied_digest, copied_size = _hash_regular_nofollow(
        destination_root, destination_relative, f"copied {field}"
    )
    if copied_digest != expected_sha256 or copied_size != source_info.st_size:
        destination.unlink(missing_ok=True)
        raise ReleaseContractError(f"copied snapshot verification failed: {destination_relative}")
    return VerifiedAsset("", destination_relative, destination, copied_digest, copied_size)


def _write_regular_bytes_nofollow(
    destination_root: Path,
    destination_relative: str,
    payload: bytes,
    *,
    field: str,
) -> VerifiedAsset:
    """Create one private snapshot file from already verified immutable bytes."""
    destination_relative = _normalized_relative(destination_relative, f"{field}.destination")
    if not payload:
        raise ReleaseContractError(f"{field} source blob is empty")
    destination = destination_root.joinpath(*PurePosixPath(destination_relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_output_path(destination_root, destination, f"{field} destination")
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        raise ReleaseContractError(f"cannot create strict snapshot file: {destination}") from exc
    try:
        with os.fdopen(destination_fd, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    digest, size = _hash_regular_nofollow(
        destination_root,
        destination_relative,
        f"written {field}",
    )
    expected = hashlib.sha256(payload).hexdigest()
    if digest != expected or size != len(payload):
        destination.unlink(missing_ok=True)
        raise ReleaseContractError(f"written snapshot verification failed: {destination_relative}")
    return VerifiedAsset("", destination_relative, destination, digest, size)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseContractError(f"{field} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ReleaseContractError(
            f"{field} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseContractError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_string(value, field).lower()
    if SHA256_RE.fullmatch(digest) is None or digest == "0" * 64:
        raise ReleaseContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _require_utc_timestamp(value: Any, field: str) -> str:
    text = _require_string(value, field)
    if not text.endswith("Z"):
        raise ReleaseContractError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ReleaseContractError(f"{field} must be a valid ISO-8601 UTC timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ReleaseContractError(f"{field} must be UTC")
    return text


def _require_https_url(value: Any, field: str) -> str:
    text = _require_string(value, field)
    if not text.startswith("https://") or any(char.isspace() for char in text):
        raise ReleaseContractError(f"{field} must be an HTTPS URL")
    if FORBIDDEN_PLACEHOLDER_RE.search(text):
        raise ReleaseContractError(f"{field} contains a placeholder value")
    return text


def _require_non_placeholder_identity(value: Any, field: str) -> str:
    text = _require_string(value, field)
    if FORBIDDEN_PLACEHOLDER_RE.search(text):
        raise ReleaseContractError(f"{field} contains a placeholder value")
    return text


def _require_publication_plan(value: Any, field: str) -> str:
    plan = _require_non_placeholder_identity(value, field)
    if plan.casefold() in NON_PUBLIC_PLANS:
        raise ReleaseContractError(f"{field} must identify a non-free publication-rights plan")
    return plan


def _validate_doi_url(value: Any, field: str = "release.archive_doi") -> str:
    doi_url = _require_https_url(value, field)
    if DOI_URL_RE.fullmatch(doi_url) is None:
        raise ReleaseContractError(f"{field} must be an https://doi.org/10... URL")
    lower = doi_url.casefold()
    registrant = lower.removeprefix("https://doi.org/").split("/", 1)[0]
    if registrant in {"10.0000", "10.1234", "10.9999"} or FORBIDDEN_PLACEHOLDER_RE.search(lower):
        raise ReleaseContractError(f"{field} contains a placeholder DOI")
    return doi_url


def _zenodo_record_id(doi_url: str, field: str = "release.archive_doi") -> int:
    match = ZENODO_DOI_RE.fullmatch(doi_url)
    if match is None:
        raise ReleaseContractError(f"{field} must be a Zenodo record DOI under registrant 10.5281")
    return int(match.group("record_id"))


def _resolve_repo_file(root: Path, relative_path: Any, field: str) -> tuple[str, Path]:
    rel = _normalized_relative(relative_path, field)
    pure = PurePosixPath(rel)
    candidate = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ReleaseContractError(f"{field} traverses a symlink: {rel}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ReleaseContractError(f"{field} is missing: {rel}") from exc
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ReleaseContractError(f"{field} escapes the repository: {rel}")
    if not resolved.is_file():
        raise ReleaseContractError(f"{field} is not a regular file: {rel}")
    return pure.as_posix(), resolved


def _assert_safe_output_path(root: Path, candidate: Path, field: str) -> None:
    """Reject lexical escapes and any existing symlink in an output path."""
    root_abs = root.absolute()
    candidate_abs = candidate.absolute()
    try:
        relative = candidate_abs.relative_to(root_abs)
    except ValueError as exc:
        raise ReleaseContractError(f"{field} escapes the repository") from exc
    cursor = root_abs
    if cursor.is_symlink():
        raise ReleaseContractError(f"{field} repository root is a symlink")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ReleaseContractError(f"{field} traverses a symlink: {cursor}")


def _reject_symlinks_in_tree(root: Path, field: str) -> None:
    """Reject file, directory, and dangling symlinks without following them."""
    if root.is_symlink():
        raise ReleaseContractError(f"{field} is a symlink: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReleaseContractError(f"{field} contains a symlink: {path}")


def _verified_asset(root: Path, value: Any, field: str) -> VerifiedAsset:
    item = _require_mapping(value, field)
    _require_exact_keys(item, ASSET_KEYS, field)
    role = _require_string(item.get("role"), f"{field}.role")
    relative, path = _resolve_repo_file(root, item.get("path"), f"{field}.path")
    expected = _require_sha256(item.get("sha256"), f"{field}.sha256")
    actual, size = _hash_regular_nofollow(root, relative, f"{field}.path")
    if actual != expected:
        raise ReleaseContractError(
            f"hash mismatch for {relative}: expected {expected}, found {actual}"
        )
    return VerifiedAsset(role, relative, path, actual, size)


def _validate_contract_structure(raw: Mapping[str, Any]) -> None:
    """Enforce the release schema's closed-world object keys at runtime."""
    _require_exact_keys(raw, CONTRACT_KEYS, "contract")
    release = _require_mapping(raw.get("release"), "release")
    _require_exact_keys(release, RELEASE_KEYS, "release")
    _require_exact_keys(
        _require_mapping(release.get("doi_evidence"), "release.doi_evidence"),
        ASSET_KEYS,
        "release.doi_evidence",
    )
    _require_exact_keys(_require_mapping(raw.get("source"), "source"), SOURCE_KEYS, "source")
    _require_exact_keys(_require_mapping(raw.get("rights"), "rights"), RIGHTS_KEYS, "rights")
    _require_exact_keys(
        _require_mapping(raw.get("distribution"), "distribution"),
        DISTRIBUTION_KEYS,
        "distribution",
    )
    generation = _require_mapping(raw.get("generation_records"), "generation_records")
    _require_exact_keys(generation, GENERATION_RECORD_KEYS, "generation_records")
    for name in GENERATION_RECORD_KEYS:
        _require_exact_keys(
            _require_mapping(generation.get(name), f"generation_records.{name}"),
            ASSET_KEYS,
            f"generation_records.{name}",
        )
    audio = _require_mapping(raw.get("audio"), "audio")
    _require_exact_keys(audio, AUDIO_KEYS, "audio")
    beats = _require_mapping(raw.get("beats"), "beats")
    if set(beats) != set(BEAT_ORDER):
        raise ReleaseContractError("beats must define all and only the ten picture clips")
    for beat in BEAT_ORDER:
        _require_exact_keys(
            _require_mapping(beats[beat], f"beats.{beat}"), BEAT_KEYS, f"beats.{beat}"
        )
    inputs = raw.get("inputs")
    if not isinstance(inputs, list):
        raise ReleaseContractError("inputs must be an array")
    for index, value in enumerate(inputs):
        _require_exact_keys(
            _require_mapping(value, f"inputs[{index}]"), ASSET_KEYS, f"inputs[{index}]"
        )


def _require_exact_consumed_inputs(
    inputs_by_path: Mapping[str, VerifiedAsset], consumed_paths: set[str]
) -> None:
    actual = set(inputs_by_path)
    if actual != consumed_paths:
        raise ReleaseContractError(
            f"inputs must contain all and only consumed assets: "
            f"missing={sorted(consumed_paths - actual)}, extra={sorted(actual - consumed_paths)}"
        )


def validate_release_tier(beat: str, tier: Any, asset_path: str | None) -> str:
    """Validate one strict beat selection without looking for alternatives."""
    tier_text = _require_string(tier, f"beats.{beat}.tier")
    if beat in SAFE_PUBLIC_BEAT_ASSETS:
        required = SAFE_PUBLIC_BEAT_ASSETS[beat]
        if tier_text != "tanager-still" or asset_path != required:
            raise ReleaseContractError(
                f"public beat {beat} must use {required}; live Esri captures are not release inputs"
            )
    if beat in DESIGNED_BEATS:
        if tier_text != "designed":
            raise ReleaseContractError(f"beat {beat} must use tier 'designed'")
        if beat != "00" and asset_path is not None:
            raise ReleaseContractError(f"designed beat {beat} must have asset_path=null")
    elif tier_text not in SWAPPABLE_TIERS[beat]:
        allowed = sorted(SWAPPABLE_TIERS[beat])
        raise ReleaseContractError(f"beat {beat} tier must be one of {allowed}")
    if tier_text == "upgrade" and asset_path != f"video/build/v2/upgrades/{beat}.mp4":
        raise ReleaseContractError(
            f"beat {beat} upgrade must bind video/build/v2/upgrades/{beat}.mp4"
        )
    return tier_text


def beat_source_records(beats: Mapping[str, BeatSource]) -> dict[str, dict[str, str | None]]:
    """Bind each beat to its closed tier and exact optional source bytes."""
    if tuple(beats) != BEAT_ORDER:
        raise ReleaseContractError("beat source mapping order is not canonical")
    return {
        beat: {
            "tier": source.tier,
            "asset_path": None if source.asset is None else source.asset.relative_path,
            "asset_sha256": None if source.asset is None else source.asset.sha256,
        }
        for beat, source in beats.items()
    }


def _git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ReleaseContractError(f"git {' '.join(args)} failed: {exc.output.strip()}") from exc


def _git_blob_bytes(root: Path, commit: str, relative_path: str) -> bytes:
    """Read exact regular-file bytes from one commit, never from the index/worktree."""
    relative = _normalized_relative(relative_path, "curated commit source")
    tree = subprocess.run(
        ["git", "ls-tree", "-z", "--full-tree", commit, "--", relative],
        cwd=root,
        capture_output=True,
    )
    if tree.returncode != 0:
        detail = tree.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseContractError(f"cannot inspect tagged source {relative}: {detail}")
    entries = [entry for entry in tree.stdout.split(b"\0") if entry]
    if len(entries) != 1:
        raise ReleaseContractError(f"tagged commit does not contain one source file: {relative}")
    try:
        metadata, raw_path = entries[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        committed_path = raw_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReleaseContractError(f"tagged source metadata is malformed: {relative}") from exc
    if committed_path != relative or object_type != "blob" or mode not in {"100644", "100755"}:
        raise ReleaseContractError(f"tagged source is not a regular file: {relative}")
    blob = subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=root,
        capture_output=True,
    )
    if blob.returncode != 0:
        detail = blob.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseContractError(f"cannot read tagged source blob {relative}: {detail}")
    if not blob.stdout:
        raise ReleaseContractError(f"tagged source blob is empty: {relative}")
    return blob.stdout


def _verify_source_state(root: Path, source: Mapping[str, Any]) -> tuple[str, str]:
    commit, tag = _validate_frozen_source(source)
    head = _git_output(root, "rev-parse", "HEAD")
    if head != commit:
        raise ReleaseContractError(f"source commit mismatch: contract={commit}, HEAD={head}")
    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ReleaseContractError("strict release requires a clean working tree")
    tag_ref = f"refs/tags/{tag}"
    _git_output(root, "check-ref-format", tag_ref)
    _git_output(root, "show-ref", "--verify", "--hash", tag_ref)
    tagged_commit = _git_output(root, "rev-parse", f"{tag_ref}^{{commit}}")
    if tagged_commit != commit:
        raise ReleaseContractError(
            f"source tag {tag!r} resolves to {tagged_commit}, not contract commit {commit}"
        )
    for relative in sorted(CURATED_PUBLIC_SOURCE_PATHS):
        _git_blob_bytes(root, commit, relative)
    return commit, tag


def _require_non_null_record_fields(
    record: Mapping[str, Any], fields: Iterable[str], name: str
) -> None:
    for field in fields:
        value: Any = record
        for part in field.split("."):
            value = _require_mapping(value, f"{name}.{field}").get(part)
        if value is None or value == "" or value == []:
            raise ReleaseContractError(f"{name}.{field} is required for public release")


def _verify_provider(value: Any, *, product: str, field: str) -> None:
    provider = _require_mapping(value, field)
    _require_exact_keys(provider, PROVIDER_KEYS, field)
    if provider.get("name") != "ElevenLabs" or provider.get("product") != product:
        raise ReleaseContractError(f"{field} must identify ElevenLabs / {product} exactly")


def _verify_terms(
    value: Any,
    *,
    field: str,
    inputs_by_path: Mapping[str, VerifiedAsset],
) -> str:
    terms = _require_mapping(value, field)
    _require_exact_keys(terms, TERMS_KEYS, field)
    _require_https_url(terms.get("url"), f"{field}.url")
    _require_utc_timestamp(terms.get("retrieved_at_utc"), f"{field}.retrieved_at_utc")
    snapshot_path = _require_string(terms.get("snapshot_path"), f"{field}.snapshot_path")
    snapshot_sha = _require_sha256(terms.get("snapshot_sha256"), f"{field}.snapshot_sha256")
    snapshot = inputs_by_path.get(snapshot_path)
    if snapshot is None or snapshot.role != "terms_snapshot":
        raise ReleaseContractError(f"{field}.snapshot_path is not a terms_snapshot input")
    if snapshot.sha256 != snapshot_sha:
        raise ReleaseContractError(f"{field} snapshot hash does not match the frozen input")
    return snapshot_path


def _verify_rights_review(value: Any, field: str) -> None:
    review = _require_mapping(value, field)
    _require_exact_keys(review, RIGHTS_REVIEW_KEYS, field)
    if review.get("publication_rights_attested") is not True:
        raise ReleaseContractError(f"{field}.publication_rights_attested must be true")
    _require_non_placeholder_identity(review.get("reviewer"), f"{field}.reviewer")
    _require_utc_timestamp(review.get("reviewed_at_utc"), f"{field}.reviewed_at_utc")


def _verify_tts_evidence(
    record_asset: VerifiedAsset,
    audio_by_segment: Mapping[str, VerifiedAsset],
    text_assets: Mapping[str, VerifiedAsset],
    inputs_by_path: Mapping[str, VerifiedAsset],
) -> set[str]:
    records: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(record_asset.path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(_require_mapping(json.loads(line), f"tts line {line_no}"))
        except json.JSONDecodeError as exc:
            raise ReleaseContractError(f"invalid JSON in TTS evidence line {line_no}") from exc
    selected_records = [record for record in records if record.get("selected") is True]
    selected_segments = [record.get("segment") for record in selected_records]
    if len(selected_segments) != len(set(selected_segments)):
        raise ReleaseContractError("TTS evidence contains duplicate selected segment records")
    if len(records) != len(audio_by_segment) or len(selected_records) != len(records):
        raise ReleaseContractError(
            "TTS evidence must contain only the nine selected narration records"
        )
    selected = {str(record.get("segment")): record for record in selected_records}
    if set(selected) != set(audio_by_segment):
        raise ReleaseContractError(
            "TTS evidence must contain exactly one selected record per narration segment"
        )
    snapshots: set[str] = set()
    for segment, record in selected.items():
        _require_exact_keys(record, TTS_KEYS, f"tts[{segment}]")
        if record.get("schema_version") != 1:
            raise ReleaseContractError(f"tts[{segment}].schema_version must be 1")
        _verify_provider(
            record.get("provider"),
            product="Text to Speech",
            field=f"tts[{segment}].provider",
        )
        generation = _require_mapping(record.get("generation"), f"tts[{segment}].generation")
        _require_exact_keys(generation, TTS_GENERATION_KEYS, f"tts[{segment}].generation")
        _require_non_placeholder_identity(generation.get("id"), f"tts[{segment}].generation.id")
        _require_utc_timestamp(
            generation.get("generated_at_utc"), f"tts[{segment}].generation.generated_at_utc"
        )
        _require_publication_plan(
            generation.get("account_plan"), f"tts[{segment}].generation.account_plan"
        )
        _require_non_null_record_fields(
            record,
            (
                "generation.id",
                "generation.generated_at_utc",
                "generation.account_plan",
                "generation.service_non_beta",
                "voice.name",
                "voice.voice_id",
                "voice.category",
                "voice.library_status",
                "voice.terms_url",
                "model.model_id",
                "model.output_format",
                "settings.stability",
                "settings.similarity_boost",
                "settings.style",
                "settings.speaker_boost",
                "editorial.decision",
            ),
            f"tts[{segment}]",
        )
        if generation["service_non_beta"] is not True:
            raise ReleaseContractError(f"tts[{segment}] must confirm a non-beta service")
        voice = _require_mapping(record.get("voice"), f"tts[{segment}].voice")
        _require_exact_keys(voice, TTS_VOICE_KEYS, f"tts[{segment}].voice")
        for key in ("name", "voice_id", "category", "library_status"):
            _require_non_placeholder_identity(voice.get(key), f"tts[{segment}].voice.{key}")
        _require_https_url(voice.get("terms_url"), f"tts[{segment}].voice.terms_url")
        model = _require_mapping(record.get("model"), f"tts[{segment}].model")
        _require_exact_keys(model, TTS_MODEL_KEYS, f"tts[{segment}].model")
        for key in TTS_MODEL_KEYS:
            _require_non_placeholder_identity(model.get(key), f"tts[{segment}].model.{key}")
        settings = _require_mapping(record.get("settings"), f"tts[{segment}].settings")
        _require_exact_keys(settings, TTS_SETTINGS_KEYS, f"tts[{segment}].settings")
        editorial = _require_mapping(record.get("editorial"), f"tts[{segment}].editorial")
        _require_exact_keys(editorial, EDITORIAL_KEYS, f"tts[{segment}].editorial")
        if record["editorial"]["decision"] != "selected":
            raise ReleaseContractError(f"tts[{segment}] editorial decision must be 'selected'")
        snapshots.add(
            _verify_terms(
                record.get("terms"),
                field=f"tts[{segment}].terms",
                inputs_by_path=inputs_by_path,
            )
        )
        _verify_rights_review(record.get("rights_review"), f"tts[{segment}].rights_review")
        audio = _require_mapping(record.get("output"), f"tts[{segment}].output")
        text = _require_mapping(record.get("text"), f"tts[{segment}].text")
        _require_exact_keys(audio, FILE_REF_KEYS, f"tts[{segment}].output")
        _require_exact_keys(text, FILE_REF_KEYS, f"tts[{segment}].text")
        expected_audio = audio_by_segment[str(segment)]
        expected_text = text_assets[f"video/segments_v2/{segment}.txt"]
        if (
            audio.get("path") != expected_audio.relative_path
            or audio.get("sha256") != expected_audio.sha256
        ):
            raise ReleaseContractError(
                f"tts[{segment}] output does not match the frozen audio asset"
            )
        if (
            text.get("path") != expected_text.relative_path
            or text.get("sha256") != expected_text.sha256
        ):
            raise ReleaseContractError(
                f"tts[{segment}] text does not match the frozen narration text"
            )
    return snapshots


def _verify_music_evidence(
    record_asset: VerifiedAsset,
    music: VerifiedAsset,
    music_plan: VerifiedAsset,
    inputs_by_path: Mapping[str, VerifiedAsset],
) -> str:
    try:
        record = _require_mapping(json.loads(record_asset.path.read_text()), "music evidence")
    except json.JSONDecodeError as exc:
        raise ReleaseContractError("music evidence is not valid JSON") from exc
    _require_exact_keys(record, MUSIC_KEYS, "music")
    if record.get("schema_version") != 1:
        raise ReleaseContractError("music.schema_version must be 1")
    _verify_provider(record.get("provider"), product="Eleven Music", field="music.provider")
    generation = _require_mapping(record.get("generation"), "music.generation")
    _require_exact_keys(generation, TTS_GENERATION_KEYS, "music.generation")
    _require_non_placeholder_identity(generation.get("id"), "music.generation.id")
    _require_utc_timestamp(generation.get("generated_at_utc"), "music.generation.generated_at_utc")
    _require_publication_plan(generation.get("account_plan"), "music.generation.account_plan")
    model = _require_mapping(record.get("model"), "music.model")
    _require_exact_keys(model, MUSIC_MODEL_KEYS, "music.model")
    request = _require_mapping(record.get("request"), "music.request")
    _require_exact_keys(request, MUSIC_REQUEST_KEYS, "music.request")
    _require_non_placeholder_identity(request.get("output_format"), "music.request.output_format")
    output = _require_mapping(record.get("output"), "music.output")
    _require_exact_keys(output, FILE_REF_KEYS, "music.output")
    plan = _require_mapping(record.get("composition_plan"), "music.composition_plan")
    _require_exact_keys(plan, FILE_REF_KEYS, "music.composition_plan")
    editorial = _require_mapping(record.get("editorial"), "music.editorial")
    _require_exact_keys(editorial, EDITORIAL_KEYS, "music.editorial")
    _require_non_null_record_fields(
        record,
        (
            "selected",
            "generation.id",
            "generation.generated_at_utc",
            "generation.account_plan",
            "generation.service_non_beta",
            "model.model_id",
            "request.output_format",
            "request.force_instrumental",
            "composition_plan.path",
            "composition_plan.sha256",
            "terms.url",
            "terms.retrieved_at_utc",
            "terms.snapshot_sha256",
            "editorial.decision",
        ),
        "music",
    )
    if record["generation"]["service_non_beta"] is not True:
        raise ReleaseContractError("music evidence must confirm a non-beta service")
    if record["selected"] is not True:
        raise ReleaseContractError("music evidence must identify the record as selected")
    if record["model"]["model_id"] != "music_v2":
        raise ReleaseContractError("music evidence must identify the selected model as music_v2")
    if record["request"]["force_instrumental"] is not True:
        raise ReleaseContractError("music evidence must confirm force_instrumental=true")
    if record["editorial"]["decision"] != "selected":
        raise ReleaseContractError("music editorial decision must be 'selected'")
    if output.get("path") != music.relative_path or output.get("sha256") != music.sha256:
        raise ReleaseContractError("music evidence output does not match the frozen music asset")
    if plan.get("path") != music_plan.relative_path or plan.get("sha256") != music_plan.sha256:
        raise ReleaseContractError("music evidence does not match the frozen composition plan")
    snapshot = _verify_terms(
        record.get("terms"), field="music.terms", inputs_by_path=inputs_by_path
    )
    _verify_rights_review(record.get("rights_review"), "music.rights_review")
    return snapshot


def _read_doi_evidence(record_asset: VerifiedAsset) -> Mapping[str, Any]:
    try:
        return _require_mapping(json.loads(record_asset.path.read_text()), "DOI evidence")
    except json.JSONDecodeError as exc:
        raise ReleaseContractError("DOI evidence is not valid JSON") from exc


def _doi_provider_reference(record_asset: VerifiedAsset) -> Mapping[str, Any]:
    record = _read_doi_evidence(record_asset)
    _require_exact_keys(record, DOI_EVIDENCE_KEYS, "DOI evidence")
    reference = _require_mapping(record.get("provider_record"), "DOI evidence.provider_record")
    _require_exact_keys(reference, ASSET_KEYS, "DOI evidence.provider_record")
    if reference.get("role") != "provider_record":
        raise ReleaseContractError("DOI evidence.provider_record must use role 'provider_record'")
    return reference


def _verify_zenodo_provider_record(
    provider_asset: VerifiedAsset,
    *,
    record_id: int,
    archive_doi: str,
    record_url: str,
) -> None:
    try:
        export = _require_mapping(
            json.loads(provider_asset.path.read_text()), "Zenodo provider record"
        )
    except json.JSONDecodeError as exc:
        raise ReleaseContractError("Zenodo provider record is not valid JSON") from exc
    missing = ZENODO_EXPORT_KEYS - set(export)
    if missing:
        raise ReleaseContractError(
            f"Zenodo provider record is missing provider fields: {sorted(missing)}"
        )
    if export.get("id") != record_id:
        raise ReleaseContractError("Zenodo provider record id does not match the DOI record id")
    expected_doi = archive_doi.removeprefix("https://doi.org/")
    if export.get("doi") != expected_doi:
        raise ReleaseContractError("Zenodo provider record DOI does not match the archive DOI")
    links = _require_mapping(export.get("links"), "Zenodo provider record.links")
    missing_links = ZENODO_LINK_KEYS - set(links)
    if missing_links:
        raise ReleaseContractError(
            f"Zenodo provider record links are incomplete: {sorted(missing_links)}"
        )
    if links.get("html") != record_url:
        raise ReleaseContractError("Zenodo provider record HTML URL does not match record_url")
    if links.get("self") != f"https://zenodo.org/api/records/{record_id}":
        raise ReleaseContractError("Zenodo provider record API URL does not match the record id")
    _zenodo_file_records(export.get("files"), "Zenodo provider record.files")


def _zenodo_file_records(value: Any, field: str) -> dict[str, tuple[int, str]]:
    """Return a closed, checksum-bearing Zenodo file inventory."""
    if not isinstance(value, list) or not value:
        raise ReleaseContractError(f"{field} must contain at least one provider file")
    records: dict[str, tuple[int, str]] = {}
    for index, raw in enumerate(value):
        record = _require_mapping(raw, f"{field}[{index}]")
        missing = ZENODO_FILE_KEYS - set(record)
        if missing:
            raise ReleaseContractError(
                f"{field}[{index}] is missing provider fields: {sorted(missing)}"
            )
        key = _normalized_relative(record.get("key"), f"{field}[{index}].key")
        size = record.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ReleaseContractError(f"{field}[{index}].size must be a positive integer")
        checksum = _require_string(record.get("checksum"), f"{field}[{index}].checksum").lower()
        if re.fullmatch(r"(?:md5:[0-9a-f]{32}|sha256:[0-9a-f]{64})", checksum) is None:
            raise ReleaseContractError(
                f"{field}[{index}].checksum must be an md5: or sha256: provider checksum"
            )
        if key in records:
            raise ReleaseContractError(f"duplicate {field} key: {key}")
        records[key] = (size, checksum)
    return records


def _verify_doi_evidence(
    record_asset: VerifiedAsset,
    archive_doi: str,
    provider_asset: VerifiedAsset,
) -> None:
    record = _read_doi_evidence(record_asset)
    _require_exact_keys(record, DOI_EVIDENCE_KEYS, "DOI evidence")
    if record.get("schema_version") != 1 or record.get("provider") != "Zenodo":
        raise ReleaseContractError("DOI evidence must identify schema 1 / Zenodo")
    if record.get("status") != "minted":
        raise ReleaseContractError("DOI evidence status must be 'minted'")
    if _validate_doi_url(record.get("doi_url"), "DOI evidence.doi_url") != archive_doi:
        raise ReleaseContractError("DOI evidence does not match release.archive_doi")
    record_id = record.get("record_id")
    if not isinstance(record_id, int) or isinstance(record_id, bool) or record_id <= 0:
        raise ReleaseContractError("DOI evidence.record_id must be a positive integer")
    if record_id != _zenodo_record_id(archive_doi):
        raise ReleaseContractError("DOI evidence.record_id does not match the archive DOI")
    record_url = _require_https_url(record.get("record_url"), "DOI evidence.record_url")
    if record_url != f"https://zenodo.org/records/{record_id}":
        raise ReleaseContractError("DOI evidence.record_url must be the exact Zenodo record URL")
    provider_reference = _doi_provider_reference(record_asset)
    if (
        provider_reference.get("path") != provider_asset.relative_path
        or provider_reference.get("sha256") != provider_asset.sha256
        or provider_asset.role != "provider_record"
    ):
        raise ReleaseContractError(
            "DOI evidence provider_record does not match the hashed Zenodo export"
        )
    _verify_zenodo_provider_record(
        provider_asset,
        record_id=record_id,
        archive_doi=archive_doi,
        record_url=record_url,
    )
    _require_utc_timestamp(record.get("minted_at_utc"), "DOI evidence.minted_at_utc")
    _require_utc_timestamp(record.get("retrieved_at_utc"), "DOI evidence.retrieved_at_utc")
    _require_non_placeholder_identity(record.get("reviewer"), "DOI evidence.reviewer")
    _require_utc_timestamp(record.get("reviewed_at_utc"), "DOI evidence.reviewed_at_utc")


def load_release_contract(
    path: Path,
    root: Path = ROOT,
    *,
    asset_root: Path | None = None,
    verify_source_checkout: bool = True,
) -> ReleaseContract:
    """Load and verify a frozen contract; return only after every gate passes."""
    path = path.absolute()
    _assert_safe_output_path(root, path, "release contract")
    contract_relative = _relative_path_within(root, path, "release contract")
    try:
        contract_bytes = _read_regular_nofollow(root, contract_relative, "release contract")
        raw = json.loads(contract_bytes.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(f"cannot read release contract {path}: {exc}") from exc
    inputs_root = root if asset_root is None else asset_root
    raw = dict(_require_mapping(raw, "contract"))
    template_locator = "video/manifests/release_contract.template.json"
    if contract_relative != CANONICAL_CONTRACT_LOCATOR and not (
        contract_relative == template_locator and raw.get("status") == "template"
    ):
        raise ReleaseContractError(
            f"release contract must use canonical locator {CANONICAL_CONTRACT_LOCATOR}"
        )
    _validate_contract_structure(raw)
    if raw.get("schema_version") != CONTRACT_SCHEMA_VERSION or raw.get("kind") != CONTRACT_KIND:
        raise ReleaseContractError("unsupported release contract kind/schema_version")
    if raw.get("status") != "frozen":
        raise ReleaseContractError(
            "release contract status must be 'frozen'; templates cannot render"
        )

    release = _require_mapping(raw.get("release"), "release")
    _require_exact_keys(release, RELEASE_KEYS, "release")
    release_id = _require_string(release.get("id"), "release.id")
    output_basename = _require_string(release.get("output_basename"), "release.output_basename")
    bundle_name = _require_string(release.get("bundle_name"), "release.bundle_name")
    contract_locator = _normalized_relative(
        release.get("contract_locator"), "release.contract_locator"
    )
    if (
        SAFE_NAME_RE.fullmatch(release_id) is None
        or SAFE_NAME_RE.fullmatch(output_basename) is None
    ):
        raise ReleaseContractError("release.id and release.output_basename must be safe filenames")
    if output_basename != "tanager-rocks-video":
        raise ReleaseContractError("release.output_basename must be 'tanager-rocks-video'")
    if bundle_name != f"{CANONICAL_BUNDLE_PREFIX}{release_id}":
        raise ReleaseContractError("release.bundle_name does not match the canonical release name")
    if contract_locator != CANONICAL_CONTRACT_LOCATOR:
        raise ReleaseContractError("release.contract_locator is not canonical")
    title = _require_string(release.get("title"), "release.title")
    repository_url = _require_https_url(release.get("repository_url"), "release.repository_url")
    archive_doi = _validate_doi_url(release.get("archive_doi"))
    _zenodo_record_id(archive_doi)
    doi_record = _verified_asset(inputs_root, release.get("doi_evidence"), "release.doi_evidence")
    if doi_record.role != "doi_evidence":
        raise ReleaseContractError("release.doi_evidence must use role 'doi_evidence'")
    doi_provider_record = _verified_asset(
        inputs_root,
        _doi_provider_reference(doi_record),
        "release.doi_evidence.provider_record",
    )
    _verify_doi_evidence(doi_record, archive_doi, doi_provider_record)
    source = _require_mapping(raw.get("source"), "source")
    commit, tag = (
        _verify_source_state(root, source)
        if verify_source_checkout
        else _validate_frozen_source(source)
    )
    if release_id != tag:
        raise ReleaseContractError("release.id must exactly equal source.tag")

    _validate_rights(raw.get("rights"))

    input_values = raw.get("inputs")
    if not isinstance(input_values, list):
        raise ReleaseContractError("inputs must be an array")
    assets = tuple(
        _verified_asset(inputs_root, value, f"inputs[{index}]")
        for index, value in enumerate(input_values)
    )
    by_path: dict[str, VerifiedAsset] = {}
    for asset in assets:
        if asset.role not in INPUT_ROLES:
            raise ReleaseContractError(f"unsupported input role: {asset.role}")
        if asset.relative_path in by_path:
            raise ReleaseContractError(f"duplicate input path: {asset.relative_path}")
        by_path[asset.relative_path] = asset

    figure_paths = {asset.relative_path for asset in assets if asset.role == "figure"}
    text_paths = {asset.relative_path for asset in assets if asset.role == "narration_text"}
    if figure_paths != REQUIRED_FIGURES:
        raise ReleaseContractError(
            f"figure contract mismatch: missing={sorted(REQUIRED_FIGURES - figure_paths)}, "
            f"extra={sorted(figure_paths - REQUIRED_FIGURES)}"
        )
    if text_paths != REQUIRED_NARRATION_TEXT:
        missing_text = sorted(REQUIRED_NARRATION_TEXT - text_paths)
        extra_text = sorted(text_paths - REQUIRED_NARRATION_TEXT)
        raise ReleaseContractError(
            f"narration text contract mismatch: missing={missing_text}, extra={extra_text}"
        )
    music_plans = {asset.relative_path for asset in assets if asset.role == "music_plan"}
    if music_plans != {REQUIRED_MUSIC_PLAN}:
        raise ReleaseContractError(f"music_plan input must be exactly {REQUIRED_MUSIC_PLAN}")

    audio = _require_mapping(raw.get("audio"), "audio")
    _require_exact_keys(audio, AUDIO_KEYS, "audio")
    segments = _require_mapping(audio.get("segments"), "audio.segments")
    expected_segment_keys = {
        path.rsplit("/", 1)[-1].removesuffix(".txt")
        for path in REQUIRED_NARRATION_TEXT
        if path.endswith(".txt")
    }
    if set(segments) != expected_segment_keys:
        raise ReleaseContractError(
            "audio.segments must map all and only the nine narration segments"
        )
    audio_by_segment: dict[str, VerifiedAsset] = {}
    for segment, rel in segments.items():
        rel_text = _require_string(rel, f"audio.segments.{segment}")
        asset = by_path.get(rel_text)
        if asset is None or asset.role != "narration_audio":
            raise ReleaseContractError(f"audio segment {segment} is not a narration_audio input")
        audio_by_segment[str(segment)] = asset
    if (
        len({asset.relative_path for asset in audio_by_segment.values()}) != 9
        or len({asset.sha256 for asset in audio_by_segment.values()}) != 9
    ):
        raise ReleaseContractError(
            "audio.segments must bind nine distinct paths and hashes for narration"
        )
    music_rel = _require_string(audio.get("music_bed"), "audio.music_bed")
    music_asset = by_path.get(music_rel)
    if music_asset is None or music_asset.role != "music":
        raise ReleaseContractError("audio.music_bed is not the frozen music input")

    beat_values = _require_mapping(raw.get("beats"), "beats")
    if set(beat_values) != set(BEAT_ORDER):
        raise ReleaseContractError("beats must define all and only the ten picture clips")
    beat_sources: dict[str, BeatSource] = {}
    for beat in BEAT_ORDER:
        item = _require_mapping(beat_values[beat], f"beats.{beat}")
        _require_exact_keys(item, BEAT_KEYS, f"beats.{beat}")
        asset_rel_raw = item.get("asset_path")
        asset_rel = (
            None
            if asset_rel_raw is None
            else _require_string(asset_rel_raw, f"beats.{beat}.asset_path")
        )
        tier = validate_release_tier(beat, item.get("tier"), asset_rel)
        asset = None
        if asset_rel is not None:
            asset = by_path.get(asset_rel)
            if asset is None or asset.role != "beat_asset":
                raise ReleaseContractError(f"beats.{beat}.asset_path is not a beat_asset input")
        if tier == "upgrade" and asset is None:
            raise ReleaseContractError(f"beat {beat} upgrade requires a frozen asset")
        if tier == "fallback" and asset is not None:
            raise ReleaseContractError(
                f"beat {beat} fallback must be generated from frozen figures"
            )
        if beat == "00" and asset_rel != "video/build/motif.mp4":
            raise ReleaseContractError("beat 00 must bind the frozen procedural motif asset")
        beat_sources[beat] = BeatSource(tier, asset)

    evidence = _require_mapping(raw.get("generation_records"), "generation_records")
    _require_exact_keys(evidence, GENERATION_RECORD_KEYS, "generation_records")
    tts_record = _verified_asset(inputs_root, evidence.get("tts"), "generation_records.tts")
    music_record = _verified_asset(inputs_root, evidence.get("music"), "generation_records.music")
    if tts_record.role != "generation_record" or music_record.role != "generation_record":
        raise ReleaseContractError("generation records must use role 'generation_record'")
    text_assets = {asset.relative_path: asset for asset in assets if asset.role == "narration_text"}
    tts_snapshots = _verify_tts_evidence(tts_record, audio_by_segment, text_assets, by_path)
    music_snapshot = _verify_music_evidence(
        music_record, music_asset, by_path[REQUIRED_MUSIC_PLAN], by_path
    )

    consumed_inputs = {
        *REQUIRED_FIGURES,
        *REQUIRED_NARRATION_TEXT,
        *(asset.relative_path for asset in audio_by_segment.values()),
        music_asset.relative_path,
        REQUIRED_MUSIC_PLAN,
        *(source.asset.relative_path for source in beat_sources.values() if source.asset),
        *tts_snapshots,
        music_snapshot,
    }
    _require_exact_consumed_inputs(by_path, consumed_inputs)

    distribution = _require_mapping(raw.get("distribution"), "distribution")
    _require_exact_keys(distribution, DISTRIBUTION_KEYS, "distribution")
    include_masters = distribution.get("include_media_masters")
    if not isinstance(include_masters, bool):
        raise ReleaseContractError("distribution.include_media_masters must be true or false")
    master_uri_raw = distribution.get("master_asset_uri")
    master_uri = (
        None
        if master_uri_raw is None
        else _require_string(master_uri_raw, "distribution.master_asset_uri")
    )
    if not include_masters:
        expected_uris = {
            archive_doi,
            f"https://zenodo.org/records/{_zenodo_record_id(archive_doi)}",
        }
        if master_uri not in expected_uris:
            raise ReleaseContractError(
                "a release that omits media masters must name the matching immutable "
                "Zenodo DOI or record URL"
            )
    elif master_uri is not None:
        raise ReleaseContractError(
            "a release that includes media masters must set master_asset_uri to null"
        )

    return ReleaseContract(
        path=path,
        contract_sha256=hashlib.sha256(contract_bytes).hexdigest(),
        raw=raw,
        release_id=release_id,
        title=title,
        source_commit=commit,
        source_tag=tag,
        repository_url=repository_url,
        archive_doi=archive_doi,
        output_basename=output_basename,
        bundle_name=bundle_name,
        contract_locator=contract_locator,
        beats=beat_sources,
        segment_paths={name: asset.path for name, asset in audio_by_segment.items()},
        music_bed=music_asset.path,
        assets=assets,
        tts_record=tts_record,
        music_record=music_record,
        doi_record=doi_record,
        doi_provider_record=doi_provider_record,
        include_media_masters=include_masters,
        master_asset_uri=master_uri,
    )


def _immutable_flag() -> int:
    flag = getattr(stat, "UF_IMMUTABLE", None)
    if platform.system() != "Darwin" or not hasattr(os, "chflags") or flag is None:
        raise ReleaseContractError(
            "strict release requires the implemented Darwin UF_IMMUTABLE capsule seal"
        )
    return int(flag)


def _seal_capsule(capsule: Path) -> None:
    """Apply and verify the Darwin user-immutable flag on every capsule inode."""
    flag = _immutable_flag()
    _reject_symlinks_in_tree(capsule, "execution capsule")
    paths = sorted(capsule.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in [*paths, capsule]:
        os.chflags(path, path.stat(follow_symlinks=False).st_flags | flag, follow_symlinks=False)
    _verify_capsule_seal(capsule)


def _verify_capsule_seal(capsule: Path) -> None:
    flag = _immutable_flag()
    _reject_symlinks_in_tree(capsule, "execution capsule")
    for path in (capsule, *capsule.rglob("*")):
        if not path.stat(follow_symlinks=False).st_flags & flag:
            raise ReleaseContractError(f"execution capsule inode is not immutable: {path}")


def _unseal_capsule(capsule: Path) -> None:
    """Clear this process's user-immutable flags only for final workspace cleanup."""
    if not capsule.exists():
        return
    flag = _immutable_flag()
    paths = [capsule, *sorted(capsule.rglob("*"), key=lambda path: len(path.parts))]
    for path in paths:
        os.chflags(path, path.stat(follow_symlinks=False).st_flags & ~flag, follow_symlinks=False)


def prepare_release_staging(
    contract: ReleaseContract,
    root: Path = ROOT,
    *,
    read_commit_blob: Callable[[Path, str, str], bytes] = _git_blob_bytes,
) -> ReleaseStaging:
    """Create and OS-seal one deterministic source-and-input execution capsule."""
    release_root = root / "output" / "releases"
    final = release_root / contract.bundle_name
    _assert_safe_output_path(root, release_root, "release output root")
    _assert_safe_output_path(root, final, "release final bundle")
    if final.exists() or final.is_symlink():
        raise ReleaseContractError(f"release bundle already exists: {final}")
    release_root.mkdir(parents=True, exist_ok=True)
    _assert_safe_output_path(root, release_root, "release output root")
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{contract.bundle_name}.staging-", dir=release_root)
    )
    work = staging_root / "work"
    snapshot = work / "capsule"
    bundle = staging_root / "bundle"
    (work / "build" / "v2").mkdir(parents=True)
    snapshot.mkdir()
    bundle.mkdir(mode=0o700)
    contract_relative = _relative_path_within(root, contract.path, "release contract")
    if contract_relative != contract.contract_locator:
        raise ReleaseContractError("release contract path does not match its canonical locator")
    selected_assets = (
        *contract.assets,
        contract.tts_record,
        contract.music_record,
        contract.doi_record,
        contract.doi_provider_record,
    )
    curated_blobs: dict[str, bytes] = {}
    capsule_inputs: dict[str, tuple[str, int, set[str]]] = {}
    for relative in sorted(CURATED_PUBLIC_SOURCE_PATHS):
        payload = read_commit_blob(root, contract.source_commit, relative)
        curated_blobs[relative] = payload
        capsule_inputs[relative] = (
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            {"tagged_source_blob"},
        )
    capsule_inputs[contract_relative] = (
        contract.contract_sha256,
        contract.path.stat().st_size,
        {"release_contract"},
    )
    for asset in selected_assets:
        previous = capsule_inputs.get(asset.relative_path)
        if previous is not None:
            if previous[0] != asset.sha256 or previous[1] != asset.size_bytes:
                raise ReleaseContractError(
                    f"capsule path has conflicting bytes: {asset.relative_path}"
                )
            previous[2].add(asset.role)
        else:
            capsule_inputs[asset.relative_path] = (
                asset.sha256,
                asset.size_bytes,
                {asset.role},
            )
    capsule_records: list[dict[str, Any]] = []
    for relative, (digest, size, purposes) in sorted(capsule_inputs.items()):
        if relative in curated_blobs:
            copied = _write_regular_bytes_nofollow(
                snapshot,
                relative,
                curated_blobs[relative],
                field=f"tagged capsule source {relative}",
            )
            if copied.sha256 != digest:
                raise ReleaseContractError(f"tagged source changed while copying: {relative}")
        else:
            copied = _copy_regular_nofollow(
                root,
                relative,
                snapshot,
                relative,
                expected_sha256=digest,
                field=f"capsule {relative}",
            )
        if copied.size_bytes != size:
            raise ReleaseContractError(f"capsule size changed while copying: {relative}")
        capsule_records.append(
            {
                "path": copied.relative_path,
                "purposes": sorted(purposes),
                "sha256": copied.sha256,
                "size_bytes": copied.size_bytes,
            }
        )
    capsule_manifest = snapshot / "capsule.json"
    capsule_manifest.write_text(
        json.dumps(
            {
                "kind": CAPSULE_KIND,
                "seal": CAPSULE_SEAL,
                "release_id": contract.release_id,
                "bundle_name": contract.bundle_name,
                "source_commit": contract.source_commit,
                "source_tag": contract.source_tag,
                "contract_locator": contract.contract_locator,
                "contract_sha256": contract.contract_sha256,
                "files": capsule_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    capsule_manifest_sha256 = sha256_file(capsule_manifest)
    _seal_capsule(snapshot)
    (staging_root / "staging.json").write_text(
        json.dumps(
            {
                "kind": "tanager-rocks-video-release-staging",
                "release_id": contract.release_id,
                "bundle_name": contract.bundle_name,
                "contract_locator": contract.contract_locator,
                "contract_sha256": contract.contract_sha256,
                "capsule_manifest_sha256": capsule_manifest_sha256,
                "capsule_seal": CAPSULE_SEAL,
            },
            sort_keys=True,
        )
        + "\n"
    )
    for field, path in (
        ("release staging root", staging_root),
        ("release work directory", work),
        ("sealed execution capsule", snapshot),
        ("release bundle staging directory", bundle),
    ):
        _assert_safe_output_path(root, path, field)
    staging = ReleaseStaging(
        staging_root,
        work,
        snapshot,
        bundle,
        final,
        capsule_manifest_sha256,
    )
    verify_release_snapshot(snapshot_release_contract(contract, staging), staging)
    return staging


def _snapshot_asset(asset: VerifiedAsset, snapshot: Path) -> VerifiedAsset:
    path = snapshot.joinpath(*PurePosixPath(asset.relative_path).parts)
    return replace(asset, path=path)


def snapshot_release_contract(
    contract: ReleaseContract,
    staging: ReleaseStaging,
) -> ReleaseContract:
    """Rebind a verified contract to the immutable staging snapshot paths."""
    assets = tuple(_snapshot_asset(asset, staging.snapshot) for asset in contract.assets)
    by_path = {asset.relative_path: asset for asset in assets}
    original_relative_by_path = {asset.path: asset.relative_path for asset in contract.assets}
    beats = {
        beat: BeatSource(
            source.tier,
            None if source.asset is None else by_path[source.asset.relative_path],
        )
        for beat, source in contract.beats.items()
    }
    return replace(
        contract,
        path=staging.snapshot.joinpath(*PurePosixPath(contract.contract_locator).parts),
        beats=beats,
        segment_paths={
            segment: staging.snapshot.joinpath(
                *PurePosixPath(original_relative_by_path[path]).parts
            )
            for segment, path in contract.segment_paths.items()
        },
        music_bed=staging.snapshot.joinpath(
            *PurePosixPath(original_relative_by_path[contract.music_bed]).parts
        ),
        assets=assets,
        tts_record=_snapshot_asset(contract.tts_record, staging.snapshot),
        music_record=_snapshot_asset(contract.music_record, staging.snapshot),
        doi_record=_snapshot_asset(contract.doi_record, staging.snapshot),
        doi_provider_record=_snapshot_asset(contract.doi_provider_record, staging.snapshot),
    )


def verify_release_snapshot(contract: ReleaseContract, staging: ReleaseStaging) -> None:
    """Recompare every sealed capsule byte to its deterministic manifest."""
    _verify_capsule_seal(staging.snapshot)
    try:
        marker = _require_mapping(
            json.loads((staging.root / "staging.json").read_text()), "staging marker"
        )
        capsule_manifest = _require_mapping(
            json.loads((staging.snapshot / "capsule.json").read_text()), "capsule manifest"
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ReleaseContractError("strict capsule metadata is missing or invalid") from exc
    _require_exact_keys(
        marker,
        {
            "kind",
            "release_id",
            "bundle_name",
            "contract_locator",
            "contract_sha256",
            "capsule_manifest_sha256",
            "capsule_seal",
        },
        "staging marker",
    )
    capsule_manifest_digest = sha256_file(staging.snapshot / "capsule.json")
    expected_marker = {
        "kind": "tanager-rocks-video-release-staging",
        "release_id": contract.release_id,
        "bundle_name": contract.bundle_name,
        "contract_locator": contract.contract_locator,
        "contract_sha256": contract.contract_sha256,
        "capsule_manifest_sha256": capsule_manifest_digest,
        "capsule_seal": CAPSULE_SEAL,
    }
    if dict(marker) != expected_marker:
        raise ReleaseContractError("staging marker does not match the sealed capsule")
    _require_exact_keys(
        capsule_manifest,
        {
            "kind",
            "seal",
            "release_id",
            "bundle_name",
            "source_commit",
            "source_tag",
            "contract_locator",
            "contract_sha256",
            "files",
        },
        "capsule manifest",
    )
    expected_identity = {
        "kind": CAPSULE_KIND,
        "seal": CAPSULE_SEAL,
        "release_id": contract.release_id,
        "bundle_name": contract.bundle_name,
        "source_commit": contract.source_commit,
        "source_tag": contract.source_tag,
        "contract_locator": contract.contract_locator,
        "contract_sha256": contract.contract_sha256,
    }
    if {key: capsule_manifest.get(key) for key in expected_identity} != expected_identity:
        raise ReleaseContractError("capsule manifest identity mismatch")
    records = capsule_manifest.get("files")
    if not isinstance(records, list):
        raise ReleaseContractError("capsule manifest files must be an array")
    expected_files = {"capsule.json"}
    previous_path = ""
    for index, raw in enumerate(records):
        record = _require_mapping(raw, f"capsule manifest.files[{index}]")
        _require_exact_keys(
            record,
            {"path", "purposes", "sha256", "size_bytes"},
            f"capsule manifest.files[{index}]",
        )
        relative = _normalized_relative(record.get("path"), f"capsule manifest.files[{index}].path")
        if relative <= previous_path:
            raise ReleaseContractError("capsule manifest paths are not unique and sorted")
        previous_path = relative
        purposes = record.get("purposes")
        if not isinstance(purposes, list) or not purposes or purposes != sorted(set(purposes)):
            raise ReleaseContractError("capsule file purposes must be a non-empty sorted set")
        digest = _require_sha256(record.get("sha256"), f"capsule manifest.files[{index}].sha256")
        size = record.get("size_bytes")
        actual_digest, actual_size = _hash_regular_nofollow(
            staging.snapshot, relative, f"capsule file {relative}"
        )
        if not isinstance(size, int) or size <= 0 or (actual_digest, actual_size) != (digest, size):
            raise ReleaseContractError(f"sealed capsule file mismatch: {relative}")
        expected_files.add(relative)
    actual_files = {
        path.relative_to(staging.snapshot).as_posix()
        for path in staging.snapshot.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ReleaseContractError("sealed capsule file set mismatch")
    contract_digest, _size = _hash_regular_nofollow(
        staging.snapshot, contract.contract_locator, "capsule release contract"
    )
    if contract_digest != contract.contract_sha256:
        raise ReleaseContractError("capsule release contract hash mismatch")
    _reverify_assets(
        (
            *contract.assets,
            contract.tts_record,
            contract.music_record,
            contract.doi_record,
            contract.doi_provider_record,
        ),
        boundary_root=staging.snapshot,
    )
    _verify_doi_evidence(
        contract.doi_record,
        contract.archive_doi,
        contract.doi_provider_record,
    )
    text_assets = {
        asset.relative_path: asset for asset in contract.assets if asset.role == "narration_text"
    }
    audio_by_segment = {
        segment: next(
            asset
            for asset in contract.assets
            if asset.path == path and asset.role == "narration_audio"
        )
        for segment, path in contract.segment_paths.items()
    }
    inputs_by_path = {asset.relative_path: asset for asset in contract.assets}
    _verify_tts_evidence(contract.tts_record, audio_by_segment, text_assets, inputs_by_path)
    music_asset = next(asset for asset in contract.assets if asset.path == contract.music_bed)
    _verify_music_evidence(
        contract.music_record,
        music_asset,
        inputs_by_path[REQUIRED_MUSIC_PLAN],
        inputs_by_path,
    )


def open_release_staging(
    contract: ReleaseContract, staging_root: Path, root: Path = ROOT
) -> ReleaseStaging:
    """Open only a parent-created, contract-bound strict staging directory."""
    staging_root = staging_root.absolute()
    _assert_safe_output_path(root, staging_root, "release staging root")
    marker_path = staging_root / "staging.json"
    if marker_path.is_symlink():
        raise ReleaseContractError("release staging marker is a symlink")
    try:
        marker = _require_mapping(json.loads(marker_path.read_text()), "release staging marker")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ReleaseContractError("release staging marker is missing or invalid") from exc
    marker_keys = {
        "kind",
        "release_id",
        "bundle_name",
        "contract_locator",
        "contract_sha256",
        "capsule_manifest_sha256",
        "capsule_seal",
    }
    _require_exact_keys(marker, marker_keys, "staging marker")
    expected_marker = {
        "kind": "tanager-rocks-video-release-staging",
        "release_id": contract.release_id,
        "bundle_name": contract.bundle_name,
        "contract_locator": contract.contract_locator,
        "contract_sha256": contract.contract_sha256,
        "capsule_manifest_sha256": sha256_file(staging_root / "work" / "capsule" / "capsule.json"),
        "capsule_seal": CAPSULE_SEAL,
    }
    if marker != expected_marker:
        raise ReleaseContractError("release staging marker does not match the contract")
    staging = ReleaseStaging(
        root=staging_root,
        work=staging_root / "work",
        snapshot=staging_root / "work" / "capsule",
        bundle=staging_root / "bundle",
        final=root / "output" / "releases" / contract.bundle_name,
        capsule_manifest_sha256=marker["capsule_manifest_sha256"],
    )
    for field, path in (
        ("release work directory", staging.work),
        ("sealed execution capsule", staging.snapshot),
        ("release bundle staging directory", staging.bundle),
    ):
        _assert_safe_output_path(root, path, field)
        if not path.is_dir() or path.is_symlink():
            raise ReleaseContractError(f"{field} is not a safe directory: {path}")
    verify_release_snapshot(contract, staging)
    return staging


def _default_provider_fetch(url: str) -> ProviderResponse:
    request = urllib.request.Request(url, headers={"User-Agent": "tanager-rocks-release/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                raise ReleaseContractError("provider response exceeds the bounded evidence size")
            return ProviderResponse(int(response.status), response.geturl(), body)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ReleaseContractError(f"live provider request failed for {url}") from exc


def _digest_for_provider_checksum(path: Path, checksum: str) -> str:
    algorithm, expected = checksum.split(":", 1)
    digest = hashlib.md5(usedforsecurity=False) if algorithm == "md5" else hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ReleaseContractError(f"Zenodo checksum does not match frozen master: {path.name}")
    return actual


def verify_live_provider(
    contract: ReleaseContract,
    *,
    fetch: Callable[[str], ProviderResponse] = _default_provider_fetch,
) -> dict[str, Any]:
    """Resolve the DOI and compare the live Zenodo record/files/checksums."""
    record_id = _zenodo_record_id(contract.archive_doi)
    record_url = f"https://zenodo.org/records/{record_id}"
    api_url = f"https://zenodo.org/api/records/{record_id}"
    resolved = fetch(contract.archive_doi)
    if resolved.status != 200 or resolved.url.rstrip("/") != record_url:
        raise ReleaseContractError("DOI did not resolve to the authoritative Zenodo record")
    provider = fetch(api_url)
    if provider.status != 200 or provider.url.rstrip("/") != api_url:
        raise ReleaseContractError("authoritative Zenodo API record could not be confirmed")
    try:
        live = _require_mapping(json.loads(provider.body), "live Zenodo provider record")
        frozen = _require_mapping(
            json.loads(contract.doi_provider_record.path.read_text()),
            "frozen Zenodo provider record",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseContractError("Zenodo provider response is not valid JSON") from exc
    if live.get("id") != record_id or live.get("doi") != contract.archive_doi.removeprefix(
        "https://doi.org/"
    ):
        raise ReleaseContractError("live Zenodo record identity does not match the release")
    live_files = _zenodo_file_records(live.get("files"), "live Zenodo files")
    frozen_files = _zenodo_file_records(frozen.get("files"), "frozen Zenodo files")
    if live_files != frozen_files:
        raise ReleaseContractError("live Zenodo file inventory differs from frozen evidence")
    if not contract.include_media_masters:
        masters = [asset for asset in contract.assets if asset.role in MEDIA_MASTER_ROLES]
        by_name: dict[str, VerifiedAsset] = {}
        for asset in masters:
            name = PurePosixPath(asset.relative_path).name
            if name in by_name:
                raise ReleaseContractError("media-master basenames are not unique for Zenodo")
            by_name[name] = asset
        missing = set(by_name) - set(live_files)
        if missing:
            raise ReleaseContractError(
                f"Zenodo record lacks frozen media masters: {sorted(missing)}"
            )
        for name, asset in by_name.items():
            size, checksum = live_files[name]
            if size != asset.size_bytes:
                raise ReleaseContractError(f"Zenodo size differs for frozen master: {name}")
            _digest_for_provider_checksum(asset.path, checksum)
    return {
        "doi_resolution": {
            "requested_url": contract.archive_doi,
            "resolved_url": record_url,
            "status": resolved.status,
        },
        "zenodo": {
            "record_id": record_id,
            "api_url": api_url,
            "response_sha256": hashlib.sha256(provider.body).hexdigest(),
            "files": [
                {"key": key, "size": size, "checksum": checksum}
                for key, (size, checksum) in sorted(live_files.items())
            ],
        },
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Use Darwin's atomic RENAME_EXCL; fail where no implemented primitive exists."""
    if platform.system() != "Darwin":
        raise ReleaseContractError("strict promotion requires Darwin renameatx_np(RENAME_EXCL)")
    libc = ctypes.CDLL(None, use_errno=True)
    renameatx_np = libc.renameatx_np
    renameatx_np.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameatx_np.restype = ctypes.c_int
    result = renameatx_np(
        -2,
        os.fsencode(source),
        -2,
        os.fsencode(destination),
        0x00000004,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ReleaseContractError(f"exclusive promotion destination exists: {destination}")
        raise ReleaseContractError(
            f"exclusive promotion failed: {source} -> {destination}: {os.strerror(error)}"
        )


def _quarantine_incomplete(path: Path, bundle_name: str) -> Path:
    ready = path / READY_SENTINEL
    if ready.exists() and not ready.is_symlink():
        ready.rename(path / "FAILED_FINAL_VERIFICATION.json")
    quarantine = path.parent / f".{bundle_name}.quarantine-{secrets.token_hex(8)}"
    _rename_noreplace(path, quarantine)
    return quarantine


def finalize_release_staging(
    staging: ReleaseStaging,
    contract: ReleaseContract,
    *,
    live_contract: ReleaseContract,
    root: Path = ROOT,
    fetch: Callable[[str], ProviderResponse] = _default_provider_fetch,
) -> Path:
    """Live-verify, finalize privately, then exclusively expose the canonical name."""
    for field, path in (
        ("release staging root", staging.root),
        ("release bundle staging directory", staging.bundle),
        ("release final bundle", staging.final),
    ):
        _assert_safe_output_path(root, path, field)
    expected_environment = environment_record(staging.snapshot, worker_mode=True)
    _verify_candidate_bundle(staging.bundle, expected_environment=expected_environment)
    candidate_manifest = _require_mapping(
        json.loads((staging.bundle / "render.json").read_text()), "render"
    )
    if candidate_manifest.get("source_files") != curated_source_records(staging.snapshot):
        raise ReleaseContractError("render source records were not recomputed from the capsule")
    reverify_release_contract(live_contract, root)
    verify_release_snapshot(contract, staging)
    live_evidence = verify_live_provider(contract, fetch=fetch)
    reverify_release_contract(live_contract, root)
    verify_release_snapshot(contract, staging)

    finalizing = staging.final.parent / (
        f".{contract.bundle_name}.finalizing-{secrets.token_hex(8)}"
    )
    _rename_noreplace(staging.bundle, finalizing)
    try:
        _verify_candidate_bundle(finalizing, expected_environment=expected_environment)
        manifest = _require_mapping(json.loads((finalizing / "render.json").read_text()), "render")
        ready = {
            "kind": "tanager-rocks-video-ready",
            "status": "finalized_after_live_provider_verification",
            "release_id": contract.release_id,
            "bundle_name": contract.bundle_name,
            "contract_locator": contract.contract_locator,
            "contract_sha256": contract.contract_sha256,
            "source_commit": contract.source_commit,
            "source_tag": contract.source_tag,
            "render_sha256": sha256_file(finalizing / "render.json"),
            "environment_sha256": _json_sha256(manifest.get("environment")),
            "verified_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            **live_evidence,
        }
        (finalizing / READY_SENTINEL).write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
        _write_sha256sums(finalizing)
        _verify_candidate_bundle(
            finalizing,
            require_ready=True,
            expected_environment=expected_environment,
        )
        _rename_noreplace(finalizing, staging.final)
    except Exception:
        if finalizing.exists() and not finalizing.is_symlink():
            quarantine = _quarantine_incomplete(finalizing, contract.bundle_name)
            raise ReleaseContractError(
                f"final verification failed; incomplete bundle quarantined at {quarantine}"
            )
        raise
    try:
        _unseal_capsule(staging.snapshot)
        shutil.rmtree(staging.root)
    except Exception as exc:
        _report_cleanup_residue(staging.final, staging.root, exc)
    return staging.final


def _command_first_line(command: list[str]) -> str | None:
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0].strip() if output else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_record(root: Path = ROOT, *, worker_mode: bool | None = None) -> dict[str, Any]:
    """Recompute the worker software identity needed to interpret a render hash."""
    fonts = []
    for font in (FONT_REGULAR, FONT_BOLD):
        if font.is_file():
            fonts.append(
                {
                    "path": f"matplotlib/mpl-data/fonts/ttf/{font.name}",
                    "sha256": sha256_file(font),
                    "size_bytes": font.stat().st_size,
                }
            )
    lock = root / "uv.lock"
    return {
        "python": sys.version.splitlines()[0],
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "ffmpeg": _command_first_line(["ffmpeg", "-version"]),
        "ffprobe": _command_first_line(["ffprobe", "-version"]),
        "packages": {
            name: _package_version(name)
            for name in ("matplotlib", "numpy", "pillow", "scipy", "xarray")
        },
        "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "fonts": fonts,
        "playwright": None,
        "chromium": None,
        "playwright_note": "not used by strict public beats 05/07",
        "worker_mode": (
            os.environ.get("TANAGER_VIDEO_STRICT_WORKER") == "1"
            if worker_mode is None
            else worker_mode
        ),
        "code_root": "sealed-execution-capsule",
        "capsule_manifest_sha256": sha256_file(root / "capsule.json"),
    }


def _reverify_assets(
    assets: Iterable[VerifiedAsset],
    *,
    boundary_root: Path | None = None,
) -> None:
    """Reject a file changed after its initial contract verification."""
    for asset in assets:
        root = asset.path.parent if boundary_root is None else boundary_root
        relative = (
            asset.path.name
            if boundary_root is None
            else _relative_path_within(root, asset.path, asset.relative_path)
        )
        actual, size = _hash_regular_nofollow(root, relative, asset.relative_path)
        if actual != asset.sha256 or size != asset.size_bytes:
            raise ReleaseContractError(
                f"input changed after preflight: {asset.relative_path}; "
                f"expected {asset.sha256}/{asset.size_bytes}, found {actual}/{size}"
            )


def reverify_release_contract(contract: ReleaseContract, root: Path = ROOT) -> None:
    """Repeat immutable-input and source-state gates immediately before packaging."""
    contract_relative = _relative_path_within(root, contract.path, "release contract")
    actual_contract, _size = _hash_regular_nofollow(root, contract_relative, "release contract")
    if actual_contract != contract.contract_sha256:
        raise ReleaseContractError(
            "release contract changed after preflight: "
            f"expected {contract.contract_sha256}, found {actual_contract}"
        )
    _verify_source_state(root, _require_mapping(contract.raw.get("source"), "source"))
    _reverify_assets(
        (
            *contract.assets,
            contract.tts_record,
            contract.music_record,
            contract.doi_record,
            contract.doi_provider_record,
        ),
        boundary_root=root,
    )


def curated_source_records(root: Path = ROOT) -> list[dict[str, Any]]:
    """Hash the lightweight public video source independently of Git objects."""
    records = []
    for relative_path in sorted(CURATED_PUBLIC_SOURCE_PATHS):
        relative, path = _resolve_repo_file(root, relative_path, "curated public source")
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _copy_bundle_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest, _source_size = _hash_regular_nofollow(
        source.parent, source.name, f"bundle source {source.name}"
    )
    copied = _copy_regular_nofollow(
        source.parent,
        source.name,
        destination.parent,
        destination.name,
        expected_sha256=source_digest,
        field=f"bundle copy {destination.name}",
    )
    return {
        "path": destination.as_posix(),
        "sha256": copied.sha256,
        "size_bytes": copied.size_bytes,
    }


def _copy_media_masters(contract: ReleaseContract, staging: Path) -> list[dict[str, Any]]:
    if not contract.include_media_masters:
        return []
    copied = []
    seen: set[str] = set()
    for asset in contract.assets:
        if asset.role not in MEDIA_MASTER_ROLES or asset.relative_path in seen:
            continue
        seen.add(asset.relative_path)
        destination = staging / "masters" / asset.relative_path
        record = _copy_bundle_file(asset.path, destination)
        record["source_path"] = asset.relative_path
        record["role"] = asset.role
        record["path"] = destination.relative_to(staging).as_posix()
        copied.append(record)
    return copied


def _generated_artifact_record(
    path: Path,
    *,
    workspace: Path,
    role: str,
    artifact_id: str,
    tier: str | None = None,
) -> dict[str, Any]:
    if path.is_symlink():
        raise ReleaseContractError(f"generated {role} is a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(workspace.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ReleaseContractError(
            f"generated {role} is outside the strict workspace: {path}"
        ) from exc
    if not resolved.is_file():
        raise ReleaseContractError(f"generated {role} is not a regular file: {path}")
    record: dict[str, Any] = {
        "role": role,
        "id": artifact_id,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if tier is not None:
        record["tier"] = tier
    return record


def _write_sha256sums(bundle: Path) -> Path:
    _reject_symlinks_in_tree(bundle, "release bundle")
    checksum_path = bundle / "SHA256SUMS"
    files = sorted(path for path in bundle.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}\n" for path in files)
    )
    return checksum_path


def _safe_bundle_rel(value: Any, field: str) -> str:
    rel = _require_string(value, field)
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or "\\" in rel or rel == "SHA256SUMS":
        raise ReleaseContractError(f"{field} is not a safe bundle-relative path")
    return pure.as_posix()


def _manifest_file_paths(records: Any, field: str) -> set[str]:
    if not isinstance(records, list):
        raise ReleaseContractError(f"{field} must be an array")
    paths: set[str] = set()
    for index, record_value in enumerate(records):
        record = _require_mapping(record_value, f"{field}[{index}]")
        rel = _safe_bundle_rel(record.get("path"), f"{field}[{index}].path")
        _require_sha256(record.get("sha256"), f"{field}[{index}].sha256")
        if rel in paths:
            raise ReleaseContractError(f"duplicate {field} path: {rel}")
        paths.add(rel)
    return paths


def bind_acceptance_frames(frames: Mapping[str, Path]) -> dict[str, Path]:
    """Replace timing-derived filenames with the exact semantic release labels."""
    if len(frames) != len(EXPECTED_ACCEPTANCE_FRAME_LABELS):
        raise ReleaseContractError(
            f"acceptance frame count mismatch: {len(frames)} != "
            f"{len(EXPECTED_ACCEPTANCE_FRAME_LABELS)}"
        )
    return dict(zip(EXPECTED_ACCEPTANCE_FRAME_LABELS, frames.values(), strict=True))


def _verify_manifest_file_records(
    bundle: Path,
    expected: Mapping[str, str],
    record_groups: Iterable[tuple[str, Any]],
) -> None:
    for field, records in record_groups:
        for index, value in enumerate(records):
            record = _require_mapping(value, f"{field}[{index}]")
            rel = _safe_bundle_rel(record.get("path"), f"{field}[{index}].path")
            digest = _require_sha256(record.get("sha256"), f"{field}[{index}].sha256")
            size = record.get("size_bytes")
            if digest != expected.get(rel):
                raise ReleaseContractError(f"{field}[{index}] hash is not bound to SHA256SUMS")
            path = bundle / rel
            if not isinstance(size, int) or size != path.stat().st_size:
                raise ReleaseContractError(f"{field}[{index}] size does not match the bundle file")


def _validate_rights(value: Any, field: str = "rights") -> Mapping[str, Any]:
    rights = _require_mapping(value, field)
    _require_exact_keys(rights, RIGHTS_KEYS, field)
    for attestation in REQUIRED_RIGHTS_ATTESTATIONS:
        if rights.get(attestation) is not True:
            raise ReleaseContractError(f"{field}.{attestation} must be true for public release")
    _require_non_placeholder_identity(rights.get("reviewer"), f"{field}.reviewer")
    operator = _require_non_placeholder_identity(rights.get("operator"), f"{field}.operator")
    if rights.get("reviewer") != operator:
        raise ReleaseContractError(f"{field}.reviewer must be the operator trust root")
    _require_utc_timestamp(rights.get("reviewed_at_utc"), f"{field}.reviewed_at_utc")
    if rights.get("attestation") != "approved_for_publication":
        raise ReleaseContractError(f"{field}.attestation must be 'approved_for_publication'")
    if rights.get("trust_root") != RIGHTS_TRUST_ROOT:
        raise ReleaseContractError(f"{field}.trust_root must identify operator attestation")
    if rights.get("evidence_basis") != RIGHTS_EVIDENCE_BASIS:
        raise ReleaseContractError(
            f"{field}.evidence_basis must bind provider-account and generation-plan records"
        )
    if rights.get("provider_account_evidence") != RIGHTS_PROVIDER_ACCOUNT_EVIDENCE:
        raise ReleaseContractError(
            f"{field}.provider_account_evidence must bind TTS/music account-plan records"
        )
    if rights.get("generation_plan_evidence") != REQUIRED_MUSIC_PLAN:
        raise ReleaseContractError(
            f"{field}.generation_plan_evidence must bind the frozen music plan"
        )
    if rights.get("legal_rights_statement") != LEGAL_RIGHTS_STATEMENT:
        raise ReleaseContractError(
            f"{field}.legal_rights_statement must state that code does not establish rights"
        )
    return rights


def _validate_frozen_source(value: Any) -> tuple[str, str]:
    source = _require_mapping(value, "source")
    _require_exact_keys(source, SOURCE_KEYS, "source")
    commit = _require_string(source.get("commit"), "source.commit").lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ReleaseContractError("source.commit must be a full 40-character Git SHA")
    tag = _require_string(source.get("tag"), "source.tag")
    if tag.startswith("refs/") or tag in {"HEAD", "FETCH_HEAD", "ORIG_HEAD"}:
        raise ReleaseContractError("source.tag must be an exact tag name")
    if source.get("dirty") is not False:
        raise ReleaseContractError("source.dirty must be false")
    return commit, tag


def _validate_environment(value: Any) -> None:
    environment = _require_mapping(value, "render.environment")
    _require_exact_keys(environment, ENVIRONMENT_KEYS, "render.environment")
    for field in ("python", "executable", "platform", "ffmpeg", "ffprobe", "code_root"):
        _require_non_placeholder_identity(environment.get(field), f"render.environment.{field}")
    if not Path(str(environment["executable"])).is_absolute():
        raise ReleaseContractError("render.environment.executable must be absolute")
    if environment.get("code_root") != "sealed-execution-capsule":
        raise ReleaseContractError("render.environment.code_root must identify the sealed capsule")
    _require_sha256(
        environment.get("capsule_manifest_sha256"),
        "render.environment.capsule_manifest_sha256",
    )
    if environment.get("worker_mode") is not True:
        raise ReleaseContractError("render.environment must be worker-recomputed in strict mode")
    packages = _require_mapping(environment.get("packages"), "render.environment.packages")
    _require_exact_keys(packages, ENVIRONMENT_PACKAGE_KEYS, "render.environment.packages")
    for name in sorted(ENVIRONMENT_PACKAGE_KEYS):
        _require_non_placeholder_identity(packages.get(name), f"render.environment.packages.{name}")
    _require_sha256(environment.get("uv_lock_sha256"), "render.environment.uv_lock_sha256")
    fonts = environment.get("fonts")
    if not isinstance(fonts, list) or len(fonts) != 2:
        raise ReleaseContractError("render.environment.fonts must contain the two render fonts")
    font_names: set[str] = set()
    for index, value in enumerate(fonts):
        record = _require_mapping(value, f"render.environment.fonts[{index}]")
        _require_exact_keys(record, SOURCE_FILE_RECORD_KEYS, f"render.environment.fonts[{index}]")
        path = _normalized_relative(record.get("path"), f"render.environment.fonts[{index}].path")
        font_names.add(PurePosixPath(path).name)
        _require_sha256(record.get("sha256"), f"render.environment.fonts[{index}].sha256")
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] <= 0:
            raise ReleaseContractError("render environment font sizes must be positive integers")
    if font_names != {"DejaVuSans.ttf", "DejaVuSans-Bold.ttf"}:
        raise ReleaseContractError("render.environment.fonts do not name the exact render fonts")
    if environment.get("playwright") is not None or environment.get("chromium") is not None:
        raise ReleaseContractError("strict public render must not record browser dependencies")
    if environment.get("playwright_note") != "not used by strict public beats 05/07":
        raise ReleaseContractError("render.environment browser exclusion note mismatch")


def _validate_source_records(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ReleaseContractError("render.source_files must be an array")
    expected_paths = tuple(sorted(CURATED_PUBLIC_SOURCE_PATHS))
    actual_paths: list[str] = []
    for index, raw in enumerate(value):
        record = _require_mapping(raw, f"render.source_files[{index}]")
        _require_exact_keys(record, SOURCE_FILE_RECORD_KEYS, f"render.source_files[{index}]")
        actual_paths.append(
            _normalized_relative(record.get("path"), f"render.source_files[{index}].path")
        )
        _require_sha256(record.get("sha256"), f"render.source_files[{index}].sha256")
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] <= 0:
            raise ReleaseContractError("render source file sizes must be positive integers")
    if tuple(actual_paths) != expected_paths:
        raise ReleaseContractError("render source records do not match the curated source set")
    return {
        str(record["path"]): _require_mapping(record, "render.source_files") for record in value
    }


def _virtual_input_assets(
    contract: Mapping[str, Any], manifest: Mapping[str, Any], bundle: Path
) -> tuple[tuple[VerifiedAsset, ...], dict[str, VerifiedAsset]]:
    contract_inputs = contract.get("inputs")
    manifest_inputs = manifest.get("inputs")
    if not isinstance(contract_inputs, list) or not isinstance(manifest_inputs, list):
        raise ReleaseContractError("contract and render inputs must be arrays")
    if len(contract_inputs) != len(manifest_inputs):
        raise ReleaseContractError("render inputs do not match the frozen contract")
    assets: list[VerifiedAsset] = []
    by_path: dict[str, VerifiedAsset] = {}
    for index, (contract_raw, manifest_raw) in enumerate(
        zip(contract_inputs, manifest_inputs, strict=True)
    ):
        contract_record = _require_mapping(contract_raw, f"contract.inputs[{index}]")
        manifest_record = _require_mapping(manifest_raw, f"render.inputs[{index}]")
        _require_exact_keys(contract_record, ASSET_KEYS, f"contract.inputs[{index}]")
        _require_exact_keys(manifest_record, INPUT_RECORD_KEYS, f"render.inputs[{index}]")
        projected = {key: manifest_record.get(key) for key in ASSET_KEYS}
        if projected != dict(contract_record):
            raise ReleaseContractError("render inputs do not exactly match the frozen contract")
        role = _require_string(contract_record.get("role"), f"contract.inputs[{index}].role")
        if role not in INPUT_ROLES:
            raise ReleaseContractError(f"unsupported input role: {role}")
        relative = _normalized_relative(
            contract_record.get("path"), f"contract.inputs[{index}].path"
        )
        digest = _require_sha256(contract_record.get("sha256"), f"contract.inputs[{index}].sha256")
        size = manifest_record.get("size_bytes")
        if not isinstance(size, int) or size <= 0:
            raise ReleaseContractError("render input sizes must be positive integers")
        if relative in by_path:
            raise ReleaseContractError(f"duplicate input path: {relative}")
        asset = VerifiedAsset(role, relative, bundle / ".unbundled" / relative, digest, size)
        assets.append(asset)
        by_path[relative] = asset
    return tuple(assets), by_path


def _bundled_evidence_asset(
    bundle: Path,
    record: Mapping[str, Any],
    contract_reference: Mapping[str, Any],
    *,
    kind: str,
) -> VerifiedAsset:
    _require_exact_keys(record, EVIDENCE_RECORD_KEYS, f"render.generation_evidence.{kind}")
    _require_exact_keys(contract_reference, ASSET_KEYS, f"contract evidence.{kind}")
    digest = _require_sha256(record.get("sha256"), f"render.generation_evidence.{kind}.sha256")
    contract_digest = _require_sha256(
        contract_reference.get("sha256"), f"contract evidence.{kind}.sha256"
    )
    if digest != contract_digest:
        raise ReleaseContractError(f"generation evidence hash mismatch for {kind}")
    path = bundle / _safe_bundle_rel(record.get("path"), f"render.generation_evidence.{kind}.path")
    if sha256_file(path) != digest:
        raise ReleaseContractError(f"generation evidence file hash mismatch for {kind}")
    size = record.get("size_bytes")
    if not isinstance(size, int) or size != path.stat().st_size or size <= 0:
        raise ReleaseContractError(f"generation evidence size mismatch for {kind}")
    return VerifiedAsset(
        _require_string(contract_reference.get("role"), f"contract evidence.{kind}.role"),
        _normalized_relative(contract_reference.get("path"), f"contract evidence.{kind}.path"),
        path,
        digest,
        size,
    )


def _verify_bundled_contract_semantics(
    bundle: Path,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_contract_structure(contract)
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("status") != "frozen"
    ):
        raise ReleaseContractError("bundled release contract is not a frozen contract")
    release = _require_mapping(contract.get("release"), "release")
    release_id = _require_string(release.get("id"), "release.id")
    output_basename = _require_string(release.get("output_basename"), "release.output_basename")
    bundle_name = _require_string(release.get("bundle_name"), "release.bundle_name")
    contract_locator = _normalized_relative(
        release.get("contract_locator"), "release.contract_locator"
    )
    if (
        SAFE_NAME_RE.fullmatch(release_id) is None
        or SAFE_NAME_RE.fullmatch(output_basename) is None
    ):
        raise ReleaseContractError("release id and output basename must be safe filenames")
    if output_basename != "tanager-rocks-video":
        raise ReleaseContractError("release output basename is not canonical")
    if bundle_name != f"{CANONICAL_BUNDLE_PREFIX}{release_id}":
        raise ReleaseContractError("release bundle name is not canonical")
    if contract_locator != CANONICAL_CONTRACT_LOCATOR:
        raise ReleaseContractError("release contract locator is not canonical")
    title = _require_string(release.get("title"), "release.title")
    repository_url = _require_https_url(release.get("repository_url"), "release.repository_url")
    archive_doi = _validate_doi_url(release.get("archive_doi"))
    record_id = _zenodo_record_id(archive_doi)
    source_commit, source_tag = _validate_frozen_source(contract.get("source"))
    if release_id != source_tag:
        raise ReleaseContractError("release id and source tag diverge")
    rights = _validate_rights(contract.get("rights"))
    assets, by_path = _virtual_input_assets(contract, manifest, bundle)

    figure_paths = {asset.relative_path for asset in assets if asset.role == "figure"}
    text_paths = {asset.relative_path for asset in assets if asset.role == "narration_text"}
    if figure_paths != REQUIRED_FIGURES:
        raise ReleaseContractError(
            f"figure contract mismatch: missing={sorted(REQUIRED_FIGURES - figure_paths)}, "
            f"extra={sorted(figure_paths - REQUIRED_FIGURES)}"
        )
    if text_paths != REQUIRED_NARRATION_TEXT:
        raise ReleaseContractError("narration text contract mismatch")
    music_plans = {asset.relative_path for asset in assets if asset.role == "music_plan"}
    if music_plans != {REQUIRED_MUSIC_PLAN}:
        raise ReleaseContractError(f"music_plan input must be exactly {REQUIRED_MUSIC_PLAN}")

    audio = _require_mapping(contract.get("audio"), "audio")
    _require_exact_keys(audio, AUDIO_KEYS, "audio")
    segments = _require_mapping(audio.get("segments"), "audio.segments")
    expected_segments = {
        PurePosixPath(path).stem for path in REQUIRED_NARRATION_TEXT if path.endswith(".txt")
    }
    if set(segments) != expected_segments:
        raise ReleaseContractError("audio.segments must map all nine narration segments")
    audio_by_segment: dict[str, VerifiedAsset] = {}
    for segment, value in segments.items():
        relative = _normalized_relative(value, f"audio.segments.{segment}")
        asset = by_path.get(relative)
        if asset is None or asset.role != "narration_audio":
            raise ReleaseContractError(f"audio segment {segment} is not a narration_audio input")
        audio_by_segment[str(segment)] = asset
    if (
        len({asset.relative_path for asset in audio_by_segment.values()}) != 9
        or len({asset.sha256 for asset in audio_by_segment.values()}) != 9
    ):
        raise ReleaseContractError("audio.segments must bind nine distinct paths and hashes")
    music_relative = _normalized_relative(audio.get("music_bed"), "audio.music_bed")
    music_asset = by_path.get(music_relative)
    if music_asset is None or music_asset.role != "music":
        raise ReleaseContractError("audio.music_bed is not the frozen music input")

    beats = _require_mapping(contract.get("beats"), "beats")
    if set(beats) != set(BEAT_ORDER):
        raise ReleaseContractError("beats must define all ten picture clips")
    beat_sources: dict[str, BeatSource] = {}
    for beat in BEAT_ORDER:
        item = _require_mapping(beats[beat], f"beats.{beat}")
        _require_exact_keys(item, BEAT_KEYS, f"beats.{beat}")
        raw_path = item.get("asset_path")
        relative = (
            None if raw_path is None else _normalized_relative(raw_path, f"beats.{beat}.asset_path")
        )
        tier = validate_release_tier(beat, item.get("tier"), relative)
        asset = None if relative is None else by_path.get(relative)
        if relative is not None and (asset is None or asset.role != "beat_asset"):
            raise ReleaseContractError(f"beats.{beat}.asset_path is not a beat_asset input")
        if tier == "upgrade" and asset is None:
            raise ReleaseContractError(f"beat {beat} upgrade requires a frozen asset")
        if tier == "fallback" and asset is not None:
            raise ReleaseContractError(f"beat {beat} fallback may not bind an asset")
        if beat == "00" and relative != "video/build/motif.mp4":
            raise ReleaseContractError("beat 00 must bind the frozen procedural motif asset")
        beat_sources[beat] = BeatSource(tier, asset)

    evidence_records = manifest.get("generation_evidence")
    if not isinstance(evidence_records, list):
        raise ReleaseContractError("render.generation_evidence must be an array")
    expected_kinds = ("tts", "music", "doi", "zenodo_record")
    kinds = tuple(
        _require_string(
            _require_mapping(record, "render.generation_evidence").get("kind"),
            "render.generation_evidence.kind",
        )
        for record in evidence_records
    )
    if kinds != expected_kinds:
        raise ReleaseContractError("generation evidence set or order is incomplete")
    evidence_by_kind = {
        kind: _require_mapping(record, f"render.generation_evidence.{kind}")
        for kind, record in zip(kinds, evidence_records, strict=True)
    }
    generation = _require_mapping(contract.get("generation_records"), "generation_records")
    tts_record = _bundled_evidence_asset(
        bundle, evidence_by_kind["tts"], _require_mapping(generation.get("tts"), "tts"), kind="tts"
    )
    music_record = _bundled_evidence_asset(
        bundle,
        evidence_by_kind["music"],
        _require_mapping(generation.get("music"), "music"),
        kind="music",
    )
    doi_record = _bundled_evidence_asset(
        bundle,
        evidence_by_kind["doi"],
        _require_mapping(release.get("doi_evidence"), "release.doi_evidence"),
        kind="doi",
    )
    provider_reference = _doi_provider_reference(doi_record)
    provider_record = _bundled_evidence_asset(
        bundle,
        evidence_by_kind["zenodo_record"],
        provider_reference,
        kind="zenodo_record",
    )
    if tts_record.role != "generation_record" or music_record.role != "generation_record":
        raise ReleaseContractError("generation records must use role 'generation_record'")
    if doi_record.role != "doi_evidence" or provider_record.role != "provider_record":
        raise ReleaseContractError("DOI evidence roles do not identify provider-origin evidence")
    _verify_doi_evidence(doi_record, archive_doi, provider_record)
    text_assets = {asset.relative_path: asset for asset in assets if asset.role == "narration_text"}
    tts_snapshots = _verify_tts_evidence(tts_record, audio_by_segment, text_assets, by_path)
    music_snapshot = _verify_music_evidence(
        music_record, music_asset, by_path[REQUIRED_MUSIC_PLAN], by_path
    )
    consumed = {
        *REQUIRED_FIGURES,
        *REQUIRED_NARRATION_TEXT,
        *(asset.relative_path for asset in audio_by_segment.values()),
        music_asset.relative_path,
        REQUIRED_MUSIC_PLAN,
        *(source.asset.relative_path for source in beat_sources.values() if source.asset),
        *tts_snapshots,
        music_snapshot,
    }
    _require_exact_consumed_inputs(by_path, consumed)

    distribution = _require_mapping(contract.get("distribution"), "distribution")
    _require_exact_keys(distribution, DISTRIBUTION_KEYS, "distribution")
    included = distribution.get("include_media_masters")
    if not isinstance(included, bool):
        raise ReleaseContractError("distribution.include_media_masters must be boolean")
    master_uri = distribution.get("master_asset_uri")
    if not included and master_uri not in {archive_doi, f"https://zenodo.org/records/{record_id}"}:
        raise ReleaseContractError("omitted media masters must bind the matching Zenodo record")
    if included and master_uri is not None:
        raise ReleaseContractError("included media masters require master_asset_uri=null")
    return {
        "release": {
            "id": release_id,
            "title": title,
            "source_commit": source_commit,
            "source_tag": source_tag,
            "source_dirty": False,
            "repository_url": repository_url,
            "archive_doi": archive_doi,
            "output_basename": output_basename,
            "bundle_name": bundle_name,
            "contract_locator": contract_locator,
        },
        "rights": dict(rights),
        "tiers": {beat: source.tier for beat, source in beat_sources.items()},
        "sources": beat_source_records(beat_sources),
        "assets": assets,
        "included_masters": included,
        "master_uri": master_uri,
    }


def _validate_png(path: Path) -> None:
    """Fully decompress and validate non-interlaced PNG scanlines."""
    idat_parts: list[bytes] = []
    color_type = bit_depth = 0
    saw_plte = False
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ReleaseContractError(f"acceptance frame is not a PNG: {path.name}")
        saw_ihdr = False
        saw_idat = False
        saw_iend = False
        while not saw_iend:
            header = handle.read(8)
            if len(header) != 8:
                raise ReleaseContractError(f"truncated PNG: {path.name}")
            length, chunk_type = struct.unpack(">I4s", header)
            payload = handle.read(length)
            crc_bytes = handle.read(4)
            if len(payload) != length or len(crc_bytes) != 4:
                raise ReleaseContractError(f"truncated PNG chunk: {path.name}")
            expected_crc = struct.unpack(">I", crc_bytes)[0]
            actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
            if expected_crc != actual_crc:
                raise ReleaseContractError(f"invalid PNG CRC: {path.name}")
            if not saw_ihdr:
                if chunk_type != b"IHDR" or length != 13:
                    raise ReleaseContractError(f"PNG lacks a leading IHDR: {path.name}")
                width, height = struct.unpack(">II", payload[:8])
                if (width, height) != (WIDTH, HEIGHT):
                    raise ReleaseContractError(
                        f"acceptance PNG dimensions must be {WIDTH}x{HEIGHT}: {path.name}"
                    )
                bit_depth, color_type, compression, filtering, interlace = payload[8:13]
                valid_depths = {
                    0: {1, 2, 4, 8, 16},
                    2: {8, 16},
                    3: {1, 2, 4, 8},
                    4: {8, 16},
                    6: {8, 16},
                }
                if (
                    bit_depth not in valid_depths.get(color_type, set())
                    or compression != 0
                    or filtering != 0
                    or interlace != 0
                ):
                    raise ReleaseContractError(
                        f"unsupported or invalid PNG pixel format: {path.name}"
                    )
                saw_ihdr = True
            elif chunk_type == b"PLTE":
                saw_plte = True
            elif chunk_type == b"IDAT":
                saw_idat = True
                idat_parts.append(payload)
            elif chunk_type == b"IEND":
                if length != 0:
                    raise ReleaseContractError(f"invalid PNG IEND: {path.name}")
                saw_iend = True
        if not saw_idat or handle.read(1):
            raise ReleaseContractError(f"PNG pixel data or terminator is invalid: {path.name}")
    if color_type == 3 and not saw_plte:
        raise ReleaseContractError(f"indexed PNG lacks PLTE: {path.name}")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    row_bytes = math.ceil(WIDTH * channels * bit_depth / 8)
    expected_size = HEIGHT * (row_bytes + 1)
    try:
        decompressor = zlib.decompressobj()
        pixels = decompressor.decompress(b"".join(idat_parts), expected_size + 1)
        pixels += decompressor.flush()
    except zlib.error as exc:
        raise ReleaseContractError(f"PNG IDAT zlib stream is invalid: {path.name}") from exc
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or len(pixels) != expected_size
    ):
        raise ReleaseContractError(f"PNG scanline byte count is invalid: {path.name}")
    stride = row_bytes + 1
    if any(pixels[offset] > 4 for offset in range(0, len(pixels), stride)):
        raise ReleaseContractError(f"PNG contains an invalid scanline filter: {path.name}")
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            if image.size != (WIDTH, HEIGHT):
                raise ReleaseContractError(f"decoded PNG dimensions changed: {path.name}")
    except (OSError, ValueError) as exc:
        raise ReleaseContractError(f"PNG pixels failed full decode: {path.name}") from exc


def _mp4_top_level_boxes(path: Path) -> set[bytes]:
    boxes: set[bytes] = set()
    total = path.stat().st_size
    with path.open("rb") as handle:
        offset = 0
        while offset < total:
            header = handle.read(8)
            if len(header) != 8:
                raise ReleaseContractError("release video has a truncated MP4 box")
            size, box_type = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                extended = handle.read(8)
                if len(extended) != 8:
                    raise ReleaseContractError("release video has a truncated MP4 box size")
                size = struct.unpack(">Q", extended)[0]
                header_size = 16
            elif size == 0:
                size = total - offset
            if size < header_size or offset + size > total:
                raise ReleaseContractError("release video has an invalid MP4 box size")
            boxes.add(box_type)
            handle.seek(size - header_size, os.SEEK_CUR)
            offset += size
    return boxes


def _validate_mp4(path: Path) -> dict[str, float | int | str]:
    """Probe, fully decode, and independently recompute objective MP4 gates."""
    boxes = _mp4_top_level_boxes(path)
    if not {b"ftyp", b"moov", b"mdat"}.issubset(boxes):
        raise ReleaseContractError("release video lacks required ftyp/moov/mdat MP4 boxes")
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,pix_fmt,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        probe = subprocess.run(probe_command, check=True, capture_output=True, text=True)
        payload = _require_mapping(json.loads(probe.stdout), "ffprobe output")
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ReleaseContractError("ffprobe rejected the release MP4") from exc
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ReleaseContractError("ffprobe did not return media streams")
    videos = [
        stream
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
    ]
    audios = [
        stream
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("codec_type") == "audio"
    ]
    if len(streams) != 2 or len(videos) != 1 or len(audios) != 1:
        raise ReleaseContractError(
            "release MP4 must contain exactly one video and one audio stream"
        )
    video = videos[0]
    audio = audios[0]
    format_record = _require_mapping(payload.get("format"), "ffprobe format")
    try:
        duration = float(format_record.get("duration"))
        video_duration = float(video.get("duration", duration))
        audio_duration = float(audio.get("duration", duration))
        frame_rate = Fraction(str(video.get("r_frame_rate")))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ReleaseContractError("release MP4 durations/frame rate are invalid") from exc
    if not all(
        math.isfinite(value) and value > 0 for value in (duration, video_duration, audio_duration)
    ):
        raise ReleaseContractError("release MP4 must have positive finite durations")
    if (
        video.get("width") != WIDTH
        or video.get("height") != HEIGHT
        or video.get("pix_fmt") != "yuv420p"
        or frame_rate != FPS
    ):
        raise ReleaseContractError("release MP4 stream parameters are not 1920x1080/30fps/yuv420p")
    if abs(video_duration - audio_duration) > (1 / FPS) + 0.05:
        raise ReleaseContractError("release MP4 audio/video durations differ beyond the QC limit")
    decode = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if decode.returncode != 0:
        raise ReleaseContractError("release MP4 failed full audio/video decode")
    loudness = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if loudness.returncode != 0:
        raise ReleaseContractError("release MP4 loudness measurement failed")
    log = loudness.stdout + loudness.stderr
    try:
        stats = json.loads(log[log.rindex("{") : log.rindex("}") + 1])
        integrated_lufs = float(stats["input_i"])
        true_peak_dbtp = float(stats["input_tp"])
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ReleaseContractError("release MP4 loudness output is invalid") from exc
    if abs(integrated_lufs + 16.0) > 1.0 or true_peak_dbtp > -1.5:
        raise ReleaseContractError("release MP4 fails the -16 LUFS / -1.5 dBTP gates")
    return {
        "duration_seconds": duration,
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": FPS,
        "pixel_format": str(video["pix_fmt"]),
        "integrated_lufs": integrated_lufs,
        "true_peak_dbtp": true_peak_dbtp,
    }


def _parse_srt_timestamp(value: str) -> int:
    hours = int(value[0:2])
    minutes = int(value[3:5])
    seconds = int(value[6:8])
    milliseconds = int(value[9:12])
    return (((hours * 60) + minutes) * 60 + seconds) * 1000 + milliseconds


def _parse_srt(text: str) -> list[SrtCue]:
    """Parse strict SubRip cues with canonical indices and non-overlapping timing."""
    if text.startswith("\ufeff"):
        raise ReleaseContractError("release captions must not contain a UTF-8 BOM")
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized or not normalized.strip():
        raise ReleaseContractError("release captions are not canonical populated SRT text")
    blocks = normalized.strip("\n").split("\n\n")
    if not blocks or any(not block for block in blocks):
        raise ReleaseContractError("release captions contain an empty cue block")
    cues: list[SrtCue] = []
    previous_end_ms = -1
    for expected_index, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 3 or lines[0] != str(expected_index):
            raise ReleaseContractError("release caption cue indices must be consecutive from 1")
        timing = SRT_TIMING_RE.fullmatch(lines[1])
        if timing is None:
            raise ReleaseContractError(f"release caption cue {expected_index} has invalid timing")
        text_lines = lines[2:]
        if any(not line.strip() for line in text_lines):
            raise ReleaseContractError(f"release caption cue {expected_index} requires text")
        start_ms = _parse_srt_timestamp(timing.group("start"))
        end_ms = _parse_srt_timestamp(timing.group("end"))
        if end_ms <= start_ms:
            raise ReleaseContractError(
                f"release caption cue {expected_index} must have positive duration"
            )
        if start_ms < previous_end_ms:
            raise ReleaseContractError(
                f"release caption cue {expected_index} overlaps or is out of order"
            )
        cues.append(SrtCue(expected_index, start_ms, end_ms, "\n".join(text_lines)))
        previous_end_ms = end_ms
    return cues


def _validate_srt(path: Path) -> dict[str, int | bool]:
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseContractError("release captions are not UTF-8 text") from exc
    cues = _parse_srt(text)
    lower_text = text.casefold()
    required_correction = "jarosite at zero point five eight"
    jarosite_present = required_correction in lower_text
    if not jarosite_present:
        raise ReleaseContractError("release captions lack the corrected jarosite r=0.58 wording")
    stakes_terms = ("usgs", "blm", "stakes", "matters")
    stakes_present = any(
        20_000 <= cue.start_ms < 30_000
        and any(term in cue.text.casefold() for term in stakes_terms)
        for cue in cues
    )
    if not stakes_present:
        raise ReleaseContractError("release captions lack the required stakes cue near 00:00:2x")
    final_present = "open data community" in cues[-1].text.casefold()
    if not final_present:
        raise ReleaseContractError("release captions lack the required final cue text")
    return {
        "cue_count": len(cues),
        "stakes_cue_present": stakes_present,
        "final_cue_present": final_present,
        "jarosite_correction_present": jarosite_present,
    }


def _single_stream_duration(path: Path, expected_type: str) -> float:
    """Probe and fully decode one replay artifact containing one accepted stream."""
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = _require_mapping(json.loads(probe.stdout), f"ffprobe {path.name}")
        streams = payload.get("streams")
        if (
            not isinstance(streams, list)
            or len(streams) != 1
            or not isinstance(streams[0], Mapping)
            or streams[0].get("codec_type") != expected_type
        ):
            raise ReleaseContractError(
                f"QC replay artifact {path.name} must contain one {expected_type} stream"
            )
        duration = float(_require_mapping(payload.get("format"), "ffprobe format")["duration"])
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(f"cannot probe QC replay artifact: {path.name}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ReleaseContractError(f"QC replay duration is invalid: {path.name}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ReleaseContractError(f"QC replay duration is not positive: {path.name}")
    try:
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ReleaseContractError("ffmpeg is required to decode QC replay artifacts") from exc
    if decoded.returncode != 0:
        raise ReleaseContractError(f"QC replay artifact failed full decode: {path.name}")
    return round(duration, 6)


def _segment_timing_records(contract: ReleaseContract) -> list[dict[str, Any]]:
    """Bind each frozen narration source hash to its independently probed duration."""
    assets_by_path = {asset.path: asset for asset in contract.assets}
    records = []
    for segment in SEGMENT_FILES:
        path = contract.segment_paths[segment]
        asset = assets_by_path.get(path)
        if asset is None or asset.role != "narration_audio":
            raise ReleaseContractError(f"missing frozen narration asset for QC timing: {segment}")
        records.append(
            {
                "segment": segment,
                "source_path": asset.relative_path,
                "source_sha256": asset.sha256,
                "duration_seconds": _single_stream_duration(path, "audio"),
            }
        )
    return records


def _replay_qc_measurements(
    video_path: Path,
    srt_path: Path,
    picture_path: Path,
    vo_master_path: Path,
    expected_vo_duration: float,
) -> dict[str, Any]:
    """Recompute every automated gate from bundle-carried replay inputs."""
    expected_vo_duration = round(expected_vo_duration, 6)
    if not math.isfinite(expected_vo_duration) or expected_vo_duration <= 0:
        raise ReleaseContractError("frozen narration duration evidence is invalid")
    mp4 = _validate_mp4(video_path)
    picture_duration = _single_stream_duration(picture_path, "video")
    vo_master_duration = _single_stream_duration(vo_master_path, "audio")
    srt = _validate_srt(srt_path)
    picture_tolerance = (1 / FPS) + 0.02
    if (
        abs(picture_duration - expected_vo_duration) > picture_tolerance
        or abs(float(mp4["duration_seconds"]) - expected_vo_duration) > picture_tolerance
    ):
        raise ReleaseContractError("picture or mux duration differs from frozen VO timing")
    if abs(vo_master_duration - expected_vo_duration) > 0.05:
        raise ReleaseContractError("vo_master.wav duration differs from frozen VO timing")
    return {
        "expected_vo_duration_seconds": expected_vo_duration,
        "picture_duration_seconds": picture_duration,
        "vo_master_duration_seconds": vo_master_duration,
        "mux_duration_seconds": round(float(mp4["duration_seconds"]), 6),
        "video_duration_seconds": round(float(mp4["video_duration_seconds"]), 6),
        "audio_duration_seconds": round(float(mp4["audio_duration_seconds"]), 6),
        "width": mp4["width"],
        "height": mp4["height"],
        "fps": mp4["fps"],
        "pixel_format": mp4["pixel_format"],
        "integrated_lufs": round(float(mp4["integrated_lufs"]), 6),
        "true_peak_dbtp": round(float(mp4["true_peak_dbtp"]), 6),
        "srt_cue_count": srt["cue_count"],
        "srt_stakes_cue_present": srt["stakes_cue_present"],
        "srt_final_cue_present": srt["final_cue_present"],
        "srt_jarosite_correction_present": srt["jarosite_correction_present"],
    }


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_ready_sentinel(
    bundle: Path,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    try:
        ready = _require_mapping(
            json.loads((bundle / READY_SENTINEL).read_text()), "READY sentinel"
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ReleaseContractError("release bundle lacks a valid READY sentinel") from exc
    keys = {
        "kind",
        "status",
        "release_id",
        "bundle_name",
        "contract_locator",
        "contract_sha256",
        "source_commit",
        "source_tag",
        "render_sha256",
        "environment_sha256",
        "verified_at_utc",
        "doi_resolution",
        "zenodo",
    }
    _require_exact_keys(ready, keys, "READY sentinel")
    release = _require_mapping(contract.get("release"), "release")
    source = _require_mapping(contract.get("source"), "source")
    expected_identity = {
        "kind": "tanager-rocks-video-ready",
        "status": "finalized_after_live_provider_verification",
        "release_id": release.get("id"),
        "bundle_name": release.get("bundle_name"),
        "contract_locator": release.get("contract_locator"),
        "contract_sha256": sha256_file(bundle / "release_contract.json"),
        "source_commit": source.get("commit"),
        "source_tag": source.get("tag"),
        "render_sha256": sha256_file(bundle / "render.json"),
        "environment_sha256": _json_sha256(manifest.get("environment")),
    }
    if {key: ready.get(key) for key in expected_identity} != expected_identity:
        raise ReleaseContractError("READY sentinel identity does not match the bundle")
    _require_utc_timestamp(ready.get("verified_at_utc"), "READY.verified_at_utc")
    resolution = _require_mapping(ready.get("doi_resolution"), "READY.doi_resolution")
    _require_exact_keys(
        resolution,
        {"requested_url", "resolved_url", "status"},
        "READY.doi_resolution",
    )
    if (
        resolution.get("requested_url") != release.get("archive_doi")
        or resolution.get("resolved_url")
        != f"https://zenodo.org/records/{_zenodo_record_id(str(release.get('archive_doi')))}"
        or resolution.get("status") != 200
    ):
        raise ReleaseContractError("READY DOI-resolution evidence is not authoritative")
    zenodo = _require_mapping(ready.get("zenodo"), "READY.zenodo")
    _require_exact_keys(
        zenodo,
        {"record_id", "api_url", "response_sha256", "files"},
        "READY.zenodo",
    )
    record_id = _zenodo_record_id(str(release.get("archive_doi")))
    if (
        zenodo.get("record_id") != record_id
        or zenodo.get("api_url") != f"https://zenodo.org/api/records/{record_id}"
    ):
        raise ReleaseContractError("READY Zenodo identity mismatch")
    _require_sha256(zenodo.get("response_sha256"), "READY.zenodo.response_sha256")
    ready_files = _zenodo_file_records(zenodo.get("files"), "READY.zenodo.files")
    try:
        frozen_provider = _require_mapping(
            json.loads((bundle / "evidence/zenodo-record.json").read_text()),
            "bundled Zenodo provider record",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseContractError("bundled Zenodo provider record is invalid") from exc
    frozen_files = _zenodo_file_records(
        frozen_provider.get("files"), "bundled Zenodo provider files"
    )
    if ready_files != frozen_files:
        raise ReleaseContractError("READY live file evidence differs from frozen Zenodo evidence")


def _verify_candidate_bundle(
    bundle: Path,
    *,
    require_ready: bool = False,
    enforce_canonical_name: bool = False,
    expected_environment: Mapping[str, Any] | None = None,
) -> list[str]:
    """Semantically replay the frozen contract and verify candidate/final bytes."""
    if bundle.is_symlink():
        raise ReleaseContractError(f"release bundle is a symlink: {bundle}")
    try:
        bundle = bundle.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ReleaseContractError(f"release bundle is missing: {bundle}") from exc
    if not bundle.is_dir():
        raise ReleaseContractError(f"release bundle is not a directory: {bundle}")
    _reject_symlinks_in_tree(bundle, "release bundle")
    checksum_path = bundle / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ReleaseContractError(f"missing checksum manifest: {checksum_path}")
    expected: dict[str, str] = {}
    for line_no, line in enumerate(checksum_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, raw_path = line.split("  ", 1)
        except ValueError as exc:
            raise ReleaseContractError(f"malformed SHA256SUMS line {line_no}") from exc
        digest = _require_sha256(digest, f"SHA256SUMS line {line_no}")
        relative = _safe_bundle_rel(raw_path, f"SHA256SUMS line {line_no}")
        if relative in expected:
            raise ReleaseContractError(f"duplicate SHA256SUMS path: {relative}")
        expected[relative] = digest

    try:
        manifest = _require_mapping(json.loads((bundle / "render.json").read_text()), "render")
        contract = _require_mapping(
            json.loads((bundle / "release_contract.json").read_text()), "release contract"
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ReleaseContractError("bundle lacks valid render/contract JSON") from exc
    _require_exact_keys(manifest, RENDER_KEYS, "render")
    if (
        manifest.get("schema_version") != RENDER_SCHEMA_VERSION
        or manifest.get("kind") != RENDER_KIND
        or manifest.get("status") != "automated_qc_passed_pending_human_review"
    ):
        raise ReleaseContractError("render manifest kind/schema/status mismatch")
    _require_utc_timestamp(manifest.get("generated_at_utc"), "render.generated_at_utc")
    semantic = _verify_bundled_contract_semantics(bundle, contract, manifest)
    if enforce_canonical_name and bundle.name != semantic["release"]["bundle_name"]:
        raise ReleaseContractError("release bundle directory name is not canonical")

    release_section = _require_mapping(manifest.get("release"), "render.release")
    _require_exact_keys(release_section, RENDER_RELEASE_KEYS, "render.release")
    if dict(release_section) != semantic["release"]:
        raise ReleaseContractError("render release identity does not match the frozen contract")
    rights = _require_mapping(manifest.get("rights"), "render.rights")
    if dict(rights) != semantic["rights"]:
        raise ReleaseContractError("render rights do not match the frozen contract")

    render_section = _require_mapping(manifest.get("render"), "render.render")
    _require_exact_keys(render_section, RENDER_SECTION_KEYS, "render.render")
    if render_section.get("workspace_isolation") != CAPSULE_SEAL:
        raise ReleaseContractError("render workspace isolation evidence is missing")
    if render_section.get("contract_locator") != semantic["release"]["contract_locator"]:
        raise ReleaseContractError("render contract locator is not canonical")
    if render_section.get("contract_sha256") != sha256_file(bundle / "release_contract.json"):
        raise ReleaseContractError("render contract hash does not match release_contract.json")
    settings = _require_mapping(render_section.get("settings"), "render.render.settings")
    _require_exact_keys(settings, RENDER_SETTINGS_KEYS, "render.render.settings")
    if dict(settings) != {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "pixel_format": "yuv420p",
        "video_encoder_args": ENCODE_ARGS,
        "dissolve_seconds": DISSOLVE_D,
        "audio_codec": "aac",
        "audio_bitrate": "192k",
    }:
        raise ReleaseContractError("render settings do not match the strict renderer")
    selected_tiers = _require_mapping(
        render_section.get("selected_tiers"), "render.render.selected_tiers"
    )
    if dict(selected_tiers) != semantic["tiers"]:
        raise ReleaseContractError("render selected tiers do not match the frozen contract")
    selected_sources = _require_mapping(
        render_section.get("selected_sources"), "render.render.selected_sources"
    )
    if dict(selected_sources) != semantic["sources"]:
        raise ReleaseContractError("render beat-to-source mapping does not match the contract")
    command = render_section.get("command")
    if (
        not isinstance(command, list)
        or len(command) != 8
        or command[:6]
        != ["uv", "run", "python", "scripts/video/render_v2.py", "--release", "--contract"]
        or not all(isinstance(value, str) and value for value in command)
        or command[-1] != semantic["release"]["id"]
    ):
        raise ReleaseContractError("render command does not identify the strict release invocation")
    if command[6] != semantic["release"]["contract_locator"]:
        raise ReleaseContractError("render command uses a non-canonical contract locator")

    generated = render_section.get("generated_artifacts")
    if not isinstance(generated, list) or len(generated) != len(BEAT_ORDER) + 3:
        raise ReleaseContractError("generated-artifact evidence is incomplete")
    for index, value in enumerate(generated):
        record = _require_mapping(value, f"render.generated_artifacts[{index}]")
        keys = {"role", "id", "sha256", "size_bytes"}
        if index < len(BEAT_ORDER):
            keys.add("tier")
        _require_exact_keys(record, keys, f"render.generated_artifacts[{index}]")
        role = _require_string(record.get("role"), f"render.generated_artifacts[{index}].role")
        artifact_id = _require_string(record.get("id"), f"render.generated_artifacts[{index}].id")
        _require_sha256(record.get("sha256"), f"render.generated_artifacts[{index}].sha256")
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] <= 0:
            raise ReleaseContractError("generated artifact size must be positive")
        if index < len(BEAT_ORDER):
            beat = BEAT_ORDER[index]
            if (role, artifact_id, record.get("tier")) != (
                "clip",
                beat,
                semantic["tiers"][beat],
            ):
                raise ReleaseContractError("generated clip evidence does not match selected tiers")
        else:
            expected_identity = (
                ("assembled_picture", "picture"),
                ("final_audio", "audio"),
                ("vo_master", "vo_master"),
            )[index - len(BEAT_ORDER)]
            if (role, artifact_id) != expected_identity:
                raise ReleaseContractError("generated-artifact identity/order is incomplete")

    required_paths = set(REQUIRED_BUNDLE_FILES)
    if require_ready:
        required_paths.add(READY_SENTINEL)
    output_records = manifest.get("outputs")
    output_paths = _manifest_file_paths(output_records, "render.outputs")
    if not isinstance(output_records, list) or len(output_records) != 2:
        raise ReleaseContractError("render.outputs must contain video and captions")
    expected_outputs = (
        ("video", f"{semantic['release']['output_basename']}.mp4"),
        ("captions", f"{semantic['release']['output_basename']}.srt"),
    )
    for index, (raw, identity) in enumerate(zip(output_records, expected_outputs, strict=True)):
        record = _require_mapping(raw, f"render.outputs[{index}]")
        _require_exact_keys(record, OUTPUT_RECORD_KEYS, f"render.outputs[{index}]")
        if (record.get("role"), record.get("path")) != identity:
            raise ReleaseContractError("render output filenames do not match output_basename")
    required_paths.update(output_paths)

    evidence_records = manifest.get("generation_evidence")
    evidence_paths = _manifest_file_paths(evidence_records, "render.generation_evidence")
    if evidence_paths != {
        "evidence/tts.jsonl",
        "evidence/music.json",
        "evidence/doi.json",
        "evidence/zenodo-record.json",
    }:
        raise ReleaseContractError("generation evidence set is incomplete")

    qc = _require_mapping(manifest.get("qc"), "render.qc")
    _require_exact_keys(qc, QC_KEYS, "render.qc")
    automated = qc.get("automated")
    if (
        qc.get("automated_all_passed") is not True
        or not isinstance(automated, list)
        or len(automated) != EXPECTED_AUTOMATED_QC_CHECKS
    ):
        raise ReleaseContractError("automated QC evidence is incomplete or contains failures")
    qc_names: list[str] = []
    qc_messages: list[str] = []
    for index, raw in enumerate(automated):
        row = _require_mapping(raw, f"render.qc.automated[{index}]")
        _require_exact_keys(row, QC_ROW_KEYS, f"render.qc.automated[{index}]")
        if row.get("passed") is not True:
            raise ReleaseContractError("automated QC evidence contains a failure")
        qc_names.append(_require_string(row.get("name"), f"render.qc.automated[{index}].name"))
        qc_messages.append(
            _require_string(row.get("message"), f"render.qc.automated[{index}].message")
        )
    if tuple(qc_names) != EXPECTED_QC_CHECK_NAMES:
        raise ReleaseContractError("automated QC names do not match the six expected checks")
    if tuple(qc_messages) != EXPECTED_QC_MESSAGES:
        raise ReleaseContractError("automated QC messages do not match replayable checks")
    replay = _require_mapping(qc.get("replay"), "render.qc.replay")
    _require_exact_keys(replay, QC_REPLAY_KEYS, "render.qc.replay")
    replay_records = replay.get("artifacts")
    replay_paths = _manifest_file_paths(replay_records, "render.qc.replay.artifacts")
    if not isinstance(replay_records, list) or len(replay_records) != len(QC_REPLAY_ARTIFACTS):
        raise ReleaseContractError("QC replay artifact evidence is incomplete")
    for index, (raw, identity) in enumerate(zip(replay_records, QC_REPLAY_ARTIFACTS, strict=True)):
        record = _require_mapping(raw, f"render.qc.replay.artifacts[{index}]")
        _require_exact_keys(
            record,
            QC_REPLAY_ARTIFACT_KEYS,
            f"render.qc.replay.artifacts[{index}]",
        )
        if (record.get("role"), record.get("path")) != identity:
            raise ReleaseContractError("QC replay artifact identity or path is not canonical")
    if (
        replay_records[0].get("sha256") != generated[len(BEAT_ORDER)].get("sha256")
        or replay_records[0].get("size_bytes") != generated[len(BEAT_ORDER)].get("size_bytes")
        or replay_records[1].get("sha256") != generated[len(BEAT_ORDER) + 2].get("sha256")
        or replay_records[1].get("size_bytes") != generated[len(BEAT_ORDER) + 2].get("size_bytes")
    ):
        raise ReleaseContractError("QC replay artifacts do not match generated artifact evidence")
    vo_segments = replay.get("vo_segments")
    if not isinstance(vo_segments, list) or len(vo_segments) != len(SEGMENT_FILES):
        raise ReleaseContractError("QC narration timing evidence is incomplete")
    narration_assets = {
        asset.relative_path: asset
        for asset in semantic["assets"]
        if asset.role == "narration_audio"
    }
    vo_duration_values: list[float] = []
    for index, (raw, segment) in enumerate(zip(vo_segments, SEGMENT_FILES, strict=True)):
        record = _require_mapping(raw, f"render.qc.replay.vo_segments[{index}]")
        _require_exact_keys(
            record,
            QC_VO_SEGMENT_KEYS,
            f"render.qc.replay.vo_segments[{index}]",
        )
        source_path = _normalized_relative(
            record.get("source_path"),
            f"render.qc.replay.vo_segments[{index}].source_path",
        )
        asset = narration_assets.get(source_path)
        if (
            record.get("segment") != segment
            or asset is None
            or record.get("source_sha256") != asset.sha256
        ):
            raise ReleaseContractError("QC narration timing does not match frozen audio inputs")
        duration_value = record.get("duration_seconds")
        if (
            not isinstance(duration_value, (int, float))
            or isinstance(duration_value, bool)
            or not math.isfinite(float(duration_value))
            or float(duration_value) <= 0
        ):
            raise ReleaseContractError("QC narration timing contains an invalid duration")
        vo_duration_values.append(float(duration_value))
    expected_vo_duration = round(sum(vo_duration_values), 6)
    measurements = _require_mapping(replay.get("measurements"), "render.qc.replay.measurements")
    _require_exact_keys(measurements, QC_MEASUREMENT_KEYS, "render.qc.replay.measurements")
    if measurements.get("expected_vo_duration_seconds") != expected_vo_duration:
        raise ReleaseContractError("QC measurements do not match frozen narration timing")
    required_paths.update(replay_paths)
    human = _require_mapping(qc.get("human_playback"), "render.qc.human_playback")
    _require_exact_keys(human, HUMAN_PLAYBACK_KEYS, "render.qc.human_playback")
    if dict(human) != {
        "status": "pending",
        "reviewed_at_utc": None,
        "reviewer": None,
        "notes": None,
    }:
        raise ReleaseContractError("strict automated render must leave human playback pending")
    frame_records = qc.get("acceptance_frames")
    frame_paths = _manifest_file_paths(frame_records, "render.qc.acceptance_frames")
    if not isinstance(frame_records, list) or len(frame_records) != EXPECTED_ACCEPTANCE_FRAMES:
        raise ReleaseContractError("acceptance-frame evidence set is incomplete")
    labels: list[str] = []
    for index, raw in enumerate(frame_records):
        record = _require_mapping(raw, f"render.qc.acceptance_frames[{index}]")
        _require_exact_keys(record, FRAME_RECORD_KEYS, f"render.qc.acceptance_frames[{index}]")
        label = _require_string(record.get("label"), f"render.qc.acceptance_frames[{index}].label")
        labels.append(label)
        if record.get("path") != f"qc/{label}.png":
            raise ReleaseContractError("acceptance-frame path does not match its label")
    if tuple(labels) != EXPECTED_ACCEPTANCE_FRAME_LABELS:
        raise ReleaseContractError("acceptance-frame labels do not match the 21 expected checks")
    required_paths.update(frame_paths)

    media = _require_mapping(manifest.get("media_masters"), "render.media_masters")
    _require_exact_keys(media, MEDIA_MASTER_KEYS, "render.media_masters")
    master_records = media.get("files")
    master_paths = _manifest_file_paths(master_records, "render.media_masters.files")
    if not isinstance(master_records, list):
        raise ReleaseContractError("render.media_masters.files must be an array")
    master_assets = [asset for asset in semantic["assets"] if asset.role in MEDIA_MASTER_ROLES]
    if semantic["included_masters"]:
        if media.get("included") is not True or media.get("external_uri") is not None:
            raise ReleaseContractError("included media masters mismatch the frozen contract")
        if len(master_records) != len(master_assets):
            raise ReleaseContractError("media-master set does not match the frozen contract")
        for index, (raw, asset) in enumerate(zip(master_records, master_assets, strict=True)):
            record = _require_mapping(raw, f"render.media_masters.files[{index}]")
            _require_exact_keys(record, MASTER_RECORD_KEYS, f"render.media_masters.files[{index}]")
            if dict(record) != {
                "source_path": asset.relative_path,
                "role": asset.role,
                "path": f"masters/{asset.relative_path}",
                "sha256": asset.sha256,
                "size_bytes": asset.size_bytes,
            }:
                raise ReleaseContractError("media-master record does not match a frozen input")
    elif (
        media.get("included") is not False
        or master_records
        or media.get("external_uri") != semantic["master_uri"]
    ):
        raise ReleaseContractError("omitted media masters do not match the frozen Zenodo URI")
    required_paths.update(master_paths)

    source_records = _validate_source_records(manifest.get("source_files"))
    for source_path, bundle_path in {
        "CITATION.cff": "CITATION.cff",
        "LICENSE": "LICENSE",
        "NOTICE.md": "NOTICE.md",
        "video/CREDITS.md": "CREDITS.md",
    }.items():
        record = source_records[source_path]
        bundled_path = bundle / bundle_path
        if (
            record.get("sha256") != sha256_file(bundled_path)
            or record.get("size_bytes") != bundled_path.stat().st_size
        ):
            raise ReleaseContractError(
                f"bundled {bundle_path} does not match curated source record {source_path}"
            )
    _validate_environment(manifest.get("environment"))
    if expected_environment is not None and dict(manifest["environment"]) != dict(
        expected_environment
    ):
        raise ReleaseContractError("render environment was not recomputed by the strict worker")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if actual != required_paths:
        raise ReleaseContractError(
            f"bundle file-set mismatch: missing={sorted(required_paths - actual)}, "
            f"extra={sorted(actual - required_paths)}"
        )
    if set(expected) != required_paths:
        raise ReleaseContractError(
            f"checksum file-set mismatch: missing={sorted(required_paths - set(expected))}, "
            f"extra={sorted(set(expected) - required_paths)}"
        )
    _verify_manifest_file_records(
        bundle,
        expected,
        (
            ("render.outputs", output_records),
            ("render.generation_evidence", evidence_records),
            ("render.qc.replay.artifacts", replay_records),
            ("render.qc.acceptance_frames", frame_records),
            ("render.media_masters.files", master_records),
        ),
    )
    allowed_dirs = {
        parent.as_posix()
        for relative in required_paths
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    actual_dirs = {
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_dir()
    }
    if actual_dirs != allowed_dirs:
        raise ReleaseContractError(
            f"bundle directory-set mismatch: missing={sorted(allowed_dirs - actual_dirs)}, "
            f"extra={sorted(actual_dirs - allowed_dirs)}"
        )
    for relative, digest in expected.items():
        _, path = _resolve_repo_file(bundle, relative, f"SHA256SUMS[{relative}]")
        if sha256_file(path) != digest:
            raise ReleaseContractError(f"bundle checksum mismatch: {relative}")
    replayed_measurements = _replay_qc_measurements(
        bundle / expected_outputs[0][1],
        bundle / expected_outputs[1][1],
        bundle / QC_REPLAY_ARTIFACTS[0][1],
        bundle / QC_REPLAY_ARTIFACTS[1][1],
        expected_vo_duration,
    )
    if dict(measurements) != replayed_measurements:
        raise ReleaseContractError("recorded QC measurements do not match independent replay")
    for record in frame_records:
        _validate_png(bundle / record["path"])
    if require_ready:
        _validate_ready_sentinel(bundle, contract, manifest)
    return sorted(expected)


def verify_release_bundle(bundle: Path) -> list[str]:
    """Verify only a canonical, live-finalized bundle carrying READY.json."""
    return _verify_candidate_bundle(
        bundle,
        require_ready=True,
        enforce_canonical_name=True,
    )


def write_release_bundle(
    contract: ReleaseContract,
    staging: Path,
    *,
    video_path: Path,
    srt_path: Path,
    picture_path: Path,
    audio_path: Path,
    strict_workspace: Path,
    clips: Mapping[str, tuple[Path, str]],
    qc_results: list[tuple[str, bool, str]],
    acceptance_frames: Mapping[str, Path],
    command: list[str],
    release_staging: ReleaseStaging,
    root: Path = ROOT,
) -> Path:
    """Write provenance, evidence, checksums, and optional exact media masters."""
    verify_release_snapshot(contract, release_staging)
    if any(not ok for _, ok, _ in qc_results):
        raise ReleaseContractError("refusing to package a render with failed automated QC")
    if tuple(name for name, _ok, _message in qc_results) != EXPECTED_QC_CHECK_NAMES:
        raise ReleaseContractError("automated QC names or order do not match the release contract")
    if tuple(acceptance_frames) != EXPECTED_ACCEPTANCE_FRAME_LABELS:
        raise ReleaseContractError(
            "acceptance frame labels or order do not match the release contract"
        )
    selected_tiers = {name: tier for name, (_path, tier) in clips.items()}
    expected_tiers = {name: source.tier for name, source in contract.beats.items()}
    if selected_tiers != expected_tiers:
        raise ReleaseContractError(
            f"rendered beat tiers differ from contract: {selected_tiers} != {expected_tiers}"
        )
    _assert_safe_output_path(release_staging.root, staging, "release bundle staging directory")
    _assert_safe_output_path(release_staging.root, strict_workspace, "strict render workspace")
    _reject_symlinks_in_tree(strict_workspace, "strict render workspace")
    expected_video = staging / f"{contract.output_basename}.mp4"
    expected_srt = staging / f"{contract.output_basename}.srt"
    if video_path != expected_video or srt_path != expected_srt:
        raise ReleaseContractError("release outputs do not match the frozen output basename")
    generated_artifacts = [
        _generated_artifact_record(
            path,
            workspace=strict_workspace,
            role="clip",
            artifact_id=name,
            tier=tier,
        )
        for name, (path, tier) in sorted(clips.items())
    ]
    generated_artifacts.extend(
        (
            _generated_artifact_record(
                picture_path,
                workspace=strict_workspace,
                role="assembled_picture",
                artifact_id="picture",
            ),
            _generated_artifact_record(
                audio_path,
                workspace=strict_workspace,
                role="final_audio",
                artifact_id="audio",
            ),
            _generated_artifact_record(
                strict_workspace / "build" / "v2" / "vo_master.wav",
                workspace=strict_workspace,
                role="vo_master",
                artifact_id="vo_master",
            ),
        )
    )

    evidence_dir = staging / "evidence"
    copied_evidence = []
    for label, asset, filename in (
        ("tts", contract.tts_record, "tts.jsonl"),
        ("music", contract.music_record, "music.json"),
        ("doi", contract.doi_record, "doi.json"),
        ("zenodo_record", contract.doi_provider_record, "zenodo-record.json"),
    ):
        record = _copy_bundle_file(asset.path, evidence_dir / filename)
        record["kind"] = label
        record["path"] = (evidence_dir / filename).relative_to(staging).as_posix()
        copied_evidence.append(record)
    for rel in ("LICENSE", "NOTICE.md", "CITATION.cff", "video/CREDITS.md"):
        source = root / rel
        destination = staging / Path(rel).name
        _copy_bundle_file(source, destination)
    _copy_bundle_file(contract.path, staging / "release_contract.json")

    qc_dir = staging / "qc"
    replay_sources = (
        ("assembled_picture", picture_path, staging / "qc/replay/picture.mp4"),
        (
            "vo_master",
            strict_workspace / "build" / "v2" / "vo_master.wav",
            staging / "qc/replay/vo_master.wav",
        ),
    )
    replay_records = []
    for role, source, destination in replay_sources:
        record = _copy_bundle_file(source, destination)
        record["role"] = role
        record["path"] = destination.relative_to(staging).as_posix()
        replay_records.append(record)
    vo_timing = _segment_timing_records(contract)
    expected_vo_duration = round(
        sum(float(record["duration_seconds"]) for record in vo_timing),
        6,
    )
    measurements = _replay_qc_measurements(
        video_path,
        srt_path,
        staging / "qc/replay/picture.mp4",
        staging / "qc/replay/vo_master.wav",
        expected_vo_duration,
    )
    frame_records = []
    for label, source in acceptance_frames.items():
        destination = qc_dir / f"{label}.png"
        record = _copy_bundle_file(source, destination)
        record["label"] = label
        record["path"] = destination.relative_to(staging).as_posix()
        frame_records.append(record)
    master_records = _copy_media_masters(contract, staging)

    manifest = {
        "schema_version": RENDER_SCHEMA_VERSION,
        "kind": RENDER_KIND,
        "status": "automated_qc_passed_pending_human_review",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "release": {
            "id": contract.release_id,
            "title": contract.title,
            "source_commit": contract.source_commit,
            "source_tag": contract.source_tag,
            "source_dirty": False,
            "repository_url": contract.repository_url,
            "archive_doi": contract.archive_doi,
            "output_basename": contract.output_basename,
            "bundle_name": contract.bundle_name,
            "contract_locator": contract.contract_locator,
        },
        "render": {
            "command": command,
            "contract_locator": contract.contract_locator,
            "contract_sha256": contract.contract_sha256,
            "settings": {
                "width": WIDTH,
                "height": HEIGHT,
                "fps": FPS,
                "pixel_format": "yuv420p",
                "video_encoder_args": ENCODE_ARGS,
                "dissolve_seconds": DISSOLVE_D,
                "audio_codec": "aac",
                "audio_bitrate": "192k",
            },
            "selected_tiers": selected_tiers,
            "selected_sources": beat_source_records(contract.beats),
            "workspace_isolation": CAPSULE_SEAL,
            "generated_artifacts": generated_artifacts,
        },
        "source_files": curated_source_records(root),
        "inputs": [asset.record() for asset in contract.assets],
        "generation_evidence": copied_evidence,
        "media_masters": {
            "included": contract.include_media_masters,
            "external_uri": contract.master_asset_uri,
            "files": master_records,
        },
        "outputs": [
            {
                "role": "video",
                "path": video_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(video_path),
                "size_bytes": video_path.stat().st_size,
            },
            {
                "role": "captions",
                "path": srt_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(srt_path),
                "size_bytes": srt_path.stat().st_size,
            },
        ],
        "environment": environment_record(root),
        "qc": {
            "automated_all_passed": True,
            "automated": [
                {"name": name, "passed": True, "message": message}
                for name, message in zip(
                    EXPECTED_QC_CHECK_NAMES,
                    EXPECTED_QC_MESSAGES,
                    strict=True,
                )
            ],
            "replay": {
                "artifacts": replay_records,
                "vo_segments": vo_timing,
                "measurements": measurements,
            },
            "acceptance_frames": frame_records,
            "human_playback": {
                "status": "pending",
                "reviewed_at_utc": None,
                "reviewer": None,
                "notes": None,
            },
        },
        "rights": dict(contract.raw["rights"]),
    }
    manifest_path = staging / "render.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_sha256sums(staging)
    _verify_candidate_bundle(staging, expected_environment=manifest["environment"])
    verify_release_snapshot(contract, release_staging)
    return manifest_path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    contract_parser = sub.add_parser("verify-contract", help="run strict release preflight")
    contract_parser.add_argument("contract", type=Path)
    bundle_parser = sub.add_parser("verify-bundle", help="verify a generated release bundle")
    bundle_parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "verify-contract":
        contract = load_release_contract(args.contract)
        print(
            f"PASS: {contract.release_id} at {contract.source_tag} "
            f"({len(contract.assets)} verified inputs)"
        )
    else:
        files = verify_release_bundle(args.bundle)
        print(f"PASS: {len(files)} checksummed release files")


if __name__ == "__main__":
    _main()
