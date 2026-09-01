"""Adversarial tests for the fail-closed video release boundary.

The complete fixture is synthetic but provider-bound and semantically complete.
Tests never invoke ffmpeg, render media, inspect scientific results, or write
repository outputs.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
import subprocess
import tempfile
import unittest
import warnings
import zlib
from pathlib import Path
from unittest.mock import patch

import audio
import captions
import common
import release
from jsonschema import Draft202012Validator

ZENODO_RECORD_ID = 12345678
ZENODO_DOI = f"https://doi.org/10.5281/zenodo.{ZENODO_RECORD_ID}"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
UTC_TIME = "2026-08-10T00:00:00Z"


class ReleaseBoundaryTests(unittest.TestCase):
    def _valid_srt(self) -> str:
        return (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Opening cue\n\n"
            "2\n"
            "00:00:20,000 --> 00:00:21,000\n"
            "That distinction matters to the USGS and BLM.\n\n"
            "3\n"
            "00:00:21,000 --> 00:00:22,000\n"
            "Jarosite at zero point five eight.\n\n"
            "4\n"
            "00:00:22,000 --> 00:00:23,000\n"
            "Built for the open data community.\n"
        )

    def _synthetic_vo_timing(self, contract: release.ReleaseContract) -> list[dict[str, object]]:
        by_path = {asset.path: asset for asset in contract.assets}
        return [
            {
                "segment": segment,
                "source_path": by_path[contract.segment_paths[segment]].relative_path,
                "source_sha256": by_path[contract.segment_paths[segment]].sha256,
                "duration_seconds": 1.0,
            }
            for segment in common.SEGMENT_FILES
        ]

    def _synthetic_qc_measurements(self) -> dict[str, object]:
        return {
            "expected_vo_duration_seconds": 9.0,
            "picture_duration_seconds": 9.0,
            "vo_master_duration_seconds": 9.0,
            "mux_duration_seconds": 9.0,
            "video_duration_seconds": 9.0,
            "audio_duration_seconds": 9.0,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "pixel_format": "yuv420p",
            "integrated_lufs": -16.0,
            "true_peak_dbtp": -2.0,
            "srt_cue_count": 4,
            "srt_stakes_cue_present": True,
            "srt_final_cue_present": True,
            "srt_jarosite_correction_present": True,
        }

    def _write_bytes(self, root: Path, relative: str, payload: bytes) -> release.VerifiedAsset:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return release.VerifiedAsset(
            role="",
            relative_path=relative,
            path=path,
            sha256=release.sha256_file(path),
            size_bytes=path.stat().st_size,
        )

    def _input_asset(
        self, root: Path, role: str, relative: str, payload: bytes
    ) -> release.VerifiedAsset:
        asset = self._write_bytes(root, relative, payload)
        return release.VerifiedAsset(role, relative, asset.path, asset.sha256, asset.size_bytes)

    def _complete_contract(self, root: Path) -> release.ReleaseContract:
        for relative in release.CURATED_PUBLIC_SOURCE_PATHS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic source:{relative}\n".encode())
        for relative in ("LICENSE", "NOTICE.md", "CITATION.cff", "video/CREDITS.md"):
            self._write_bytes(root, relative, f"synthetic {relative}\n".encode())

        assets: list[release.VerifiedAsset] = []
        for relative in sorted(release.REQUIRED_FIGURES):
            assets.append(self._input_asset(root, "figure", relative, relative.encode()))
        for relative in sorted(release.REQUIRED_NARRATION_TEXT):
            assets.append(
                self._input_asset(root, "narration_text", relative, f"text:{relative}\n".encode())
            )

        segment_paths: dict[str, Path] = {}
        audio_assets: dict[str, release.VerifiedAsset] = {}
        for segment, filename in common.SEGMENT_FILES.items():
            relative = f"video/audio_v2/{filename}"
            asset = self._input_asset(
                root, "narration_audio", relative, f"audio:{segment}\n".encode()
            )
            assets.append(asset)
            segment_paths[segment] = asset.path
            audio_assets[segment] = asset

        music = self._input_asset(
            root, "music", "video/audio_v2/music_bed_v2a.mp3", b"synthetic music\n"
        )
        music_plan = self._input_asset(
            root,
            "music_plan",
            release.REQUIRED_MUSIC_PLAN,
            b'{"plan":"synthetic deterministic fixture"}\n',
        )
        tts_terms = self._input_asset(
            root,
            "terms_snapshot",
            "video/manifests/evidence/elevenlabs-tts-terms.html",
            b"synthetic TTS terms snapshot\n",
        )
        music_terms = self._input_asset(
            root,
            "terms_snapshot",
            "video/manifests/evidence/eleven-music-terms.html",
            b"synthetic music terms snapshot\n",
        )
        motif = self._input_asset(root, "beat_asset", "video/build/motif.mp4", b"synthetic motif\n")
        fallback_05 = self._input_asset(
            root, "beat_asset", release.SAFE_PUBLIC_BEAT_ASSETS["05"], b"synthetic 05\n"
        )
        fallback_07 = self._input_asset(
            root, "beat_asset", release.SAFE_PUBLIC_BEAT_ASSETS["07"], b"synthetic 07\n"
        )
        assets.extend((music, music_plan, tts_terms, music_terms, motif, fallback_05, fallback_07))
        by_path = {asset.relative_path: asset for asset in assets}

        tts_records = []
        for segment, output in audio_assets.items():
            text = by_path[f"video/segments_v2/{segment}.txt"]
            tts_records.append(
                {
                    "schema_version": 1,
                    "segment": segment,
                    "selected": True,
                    "provider": {"name": "ElevenLabs", "product": "Text to Speech"},
                    "text": {"path": text.relative_path, "sha256": text.sha256},
                    "output": {"path": output.relative_path, "sha256": output.sha256},
                    "generation": {
                        "id": f"generation-{segment}",
                        "generated_at_utc": UTC_TIME,
                        "account_plan": "Creator",
                        "service_non_beta": True,
                    },
                    "voice": {
                        "name": "Narrator",
                        "voice_id": "voice-001",
                        "category": "premade",
                        "library_status": "available",
                        "terms_url": "https://elevenlabs.io/terms-of-use",
                    },
                    "model": {"model_id": "eleven_multilingual_v2", "output_format": "mp3"},
                    "settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "speaker_boost": True,
                        "speed": 1.0,
                        "seed": None,
                        "unavailable_fields": ["seed"],
                    },
                    "terms": {
                        "url": "https://elevenlabs.io/terms-of-use",
                        "retrieved_at_utc": UTC_TIME,
                        "snapshot_path": tts_terms.relative_path,
                        "snapshot_sha256": tts_terms.sha256,
                    },
                    "rights_review": {
                        "publication_rights_attested": True,
                        "reviewer": "QA Reviewer",
                        "reviewed_at_utc": UTC_TIME,
                    },
                    "editorial": {"decision": "selected", "notes": "Synthetic unit fixture"},
                }
            )
        tts_path = root / "video/manifests/evidence/tts.jsonl"
        tts_path.parent.mkdir(parents=True, exist_ok=True)
        tts_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in tts_records))
        tts_record = release.VerifiedAsset(
            "generation_record",
            tts_path.relative_to(root).as_posix(),
            tts_path,
            release.sha256_file(tts_path),
            tts_path.stat().st_size,
        )

        music_payload = {
            "schema_version": 1,
            "selected": True,
            "provider": {"name": "ElevenLabs", "product": "Eleven Music"},
            "output": {"path": music.relative_path, "sha256": music.sha256},
            "generation": {
                "id": "generation-music-001",
                "generated_at_utc": UTC_TIME,
                "account_plan": "Creator",
                "service_non_beta": True,
            },
            "model": {"model_id": "music_v2"},
            "request": {
                "output_format": "mp3",
                "force_instrumental": True,
                "seed": None,
                "seed_unavailable_reason": "Provider did not expose a seed",
            },
            "composition_plan": {
                "path": music_plan.relative_path,
                "sha256": music_plan.sha256,
            },
            "terms": {
                "url": "https://elevenlabs.io/terms-of-use",
                "retrieved_at_utc": UTC_TIME,
                "snapshot_path": music_terms.relative_path,
                "snapshot_sha256": music_terms.sha256,
            },
            "rights_review": {
                "publication_rights_attested": True,
                "reviewer": "QA Reviewer",
                "reviewed_at_utc": UTC_TIME,
            },
            "editorial": {"decision": "selected", "notes": "Synthetic unit fixture"},
        }
        music_path = root / "video/manifests/evidence/music.json"
        music_path.write_text(json.dumps(music_payload, sort_keys=True) + "\n")
        music_record = release.VerifiedAsset(
            "generation_record",
            music_path.relative_to(root).as_posix(),
            music_path,
            release.sha256_file(music_path),
            music_path.stat().st_size,
        )

        media_assets = [asset for asset in assets if asset.role in release.MEDIA_MASTER_ROLES]
        zenodo_files = [
            {
                "key": Path(asset.relative_path).name,
                "size": asset.size_bytes,
                "checksum": "md5:"
                + hashlib.md5(asset.path.read_bytes(), usedforsecurity=False).hexdigest(),
            }
            for asset in media_assets
        ]
        zenodo_export = {
            "id": ZENODO_RECORD_ID,
            "doi": ZENODO_DOI.removeprefix("https://doi.org/"),
            "links": {
                "html": ZENODO_RECORD_URL,
                "self": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}",
            },
            "files": zenodo_files,
        }
        zenodo_path = root / "video/manifests/evidence/zenodo-record.json"
        zenodo_path.write_text(json.dumps(zenodo_export, sort_keys=True) + "\n")
        zenodo_record = release.VerifiedAsset(
            "provider_record",
            zenodo_path.relative_to(root).as_posix(),
            zenodo_path,
            release.sha256_file(zenodo_path),
            zenodo_path.stat().st_size,
        )
        doi_payload = {
            "schema_version": 1,
            "provider": "Zenodo",
            "status": "minted",
            "record_id": ZENODO_RECORD_ID,
            "doi_url": ZENODO_DOI,
            "record_url": ZENODO_RECORD_URL,
            "provider_record": {
                "role": zenodo_record.role,
                "path": zenodo_record.relative_path,
                "sha256": zenodo_record.sha256,
            },
            "minted_at_utc": UTC_TIME,
            "retrieved_at_utc": UTC_TIME,
            "reviewer": "QA Reviewer",
            "reviewed_at_utc": UTC_TIME,
        }
        doi_path = root / "video/manifests/evidence/doi.json"
        doi_path.write_text(json.dumps(doi_payload, sort_keys=True) + "\n")
        doi_record = release.VerifiedAsset(
            "doi_evidence",
            doi_path.relative_to(root).as_posix(),
            doi_path,
            release.sha256_file(doi_path),
            doi_path.stat().st_size,
        )

        tiers = {
            "00": "designed",
            "01": "designed",
            "02": "designed",
            "03": "fallback",
            "04": "fallback",
            "05": "tanager-still",
            "06a": "designed",
            "06b": "fallback",
            "07": "tanager-still",
            "08": "designed",
        }
        beat_paths = {
            "00": motif.relative_path,
            "05": fallback_05.relative_path,
            "07": fallback_07.relative_path,
        }
        raw_contract = {
            "schema_version": release.CONTRACT_SCHEMA_VERSION,
            "kind": release.CONTRACT_KIND,
            "status": "frozen",
            "release": {
                "id": "v1.0.0",
                "title": "The color you cannot see",
                "repository_url": "https://github.com/bradleylab/tanager-rocks",
                "archive_doi": ZENODO_DOI,
                "doi_evidence": {
                    "role": doi_record.role,
                    "path": doi_record.relative_path,
                    "sha256": doi_record.sha256,
                },
                "output_basename": "tanager-rocks-video",
                "bundle_name": "tanager-rocks-video-v1.0.0",
                "contract_locator": release.CANONICAL_CONTRACT_LOCATOR,
            },
            "source": {"commit": "a" * 40, "tag": "v1.0.0", "dirty": False},
            "rights": {
                "claims_frozen": True,
                "planet_material_reviewed": True,
                "elevenlabs_narration_reviewed": True,
                "elevenlabs_music_reviewed": True,
                "third_party_visuals_reviewed": True,
                "reviewer": "QA Reviewer",
                "reviewed_at_utc": UTC_TIME,
                "attestation": "approved_for_publication",
                "trust_root": release.RIGHTS_TRUST_ROOT,
                "operator": "QA Reviewer",
                "evidence_basis": release.RIGHTS_EVIDENCE_BASIS,
                "provider_account_evidence": release.RIGHTS_PROVIDER_ACCOUNT_EVIDENCE,
                "generation_plan_evidence": release.REQUIRED_MUSIC_PLAN,
                "legal_rights_statement": release.LEGAL_RIGHTS_STATEMENT,
            },
            "distribution": {
                "include_media_masters": False,
                "master_asset_uri": ZENODO_RECORD_URL,
            },
            "generation_records": {
                "tts": {
                    "role": tts_record.role,
                    "path": tts_record.relative_path,
                    "sha256": tts_record.sha256,
                },
                "music": {
                    "role": music_record.role,
                    "path": music_record.relative_path,
                    "sha256": music_record.sha256,
                },
            },
            "audio": {
                "segments": {
                    segment: asset.relative_path for segment, asset in audio_assets.items()
                },
                "music_bed": music.relative_path,
            },
            "beats": {
                beat: {"tier": tier, "asset_path": beat_paths.get(beat)}
                for beat, tier in tiers.items()
            },
            "inputs": [
                {"role": asset.role, "path": asset.relative_path, "sha256": asset.sha256}
                for asset in assets
            ],
        }
        contract_path = root / "video/manifests/release_contract.json"
        contract_path.write_text(json.dumps(raw_contract, indent=2, sort_keys=True) + "\n")
        beat_sources = {
            beat: release.BeatSource(
                tier,
                None if beat not in beat_paths else by_path[beat_paths[beat]],
            )
            for beat, tier in tiers.items()
        }
        return release.ReleaseContract(
            path=contract_path,
            contract_sha256=release.sha256_file(contract_path),
            raw=raw_contract,
            release_id="v1.0.0",
            title="The color you cannot see",
            source_commit="a" * 40,
            source_tag="v1.0.0",
            repository_url="https://github.com/bradleylab/tanager-rocks",
            archive_doi=ZENODO_DOI,
            output_basename="tanager-rocks-video",
            bundle_name="tanager-rocks-video-v1.0.0",
            contract_locator=release.CANONICAL_CONTRACT_LOCATOR,
            beats=beat_sources,
            segment_paths=segment_paths,
            music_bed=music.path,
            assets=tuple(assets),
            tts_record=tts_record,
            music_record=music_record,
            doi_record=doi_record,
            doi_provider_record=zenodo_record,
            include_media_masters=False,
            master_asset_uri=ZENODO_RECORD_URL,
        )

    def _valid_png(self) -> bytes:
        width, height = 1920, 1080
        signature = b"\x89PNG\r\n\x1a\n"

        def chunk(name: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + name
                + payload
                + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
        rows = b"".join(b"\x00" + (b"\x00" * width) for _ in range(height))
        return (
            signature
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b"")
        )

    def _source_records(self, root: Path) -> list[dict[str, object]]:
        records = []
        for relative in sorted(release.CURATED_PUBLIC_SOURCE_PATHS):
            path = root / relative
            payload = path.read_bytes() if path.is_file() else relative.encode()
            records.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        return records

    def _environment_record(self, code_root: Path) -> dict[str, object]:
        return {
            "python": "3.11.9",
            "executable": "/synthetic/python3",
            "platform": "Darwin-arm64",
            "ffmpeg": "ffmpeg version 7.1",
            "ffprobe": "ffprobe version 7.1",
            "packages": {
                "matplotlib": "3.10.0",
                "numpy": "2.2.0",
                "pillow": "11.0.0",
                "scipy": "1.15.0",
                "xarray": "2026.1.0",
            },
            "uv_lock_sha256": hashlib.sha256(b"synthetic uv lock").hexdigest(),
            "fonts": [
                {
                    "path": ".venv/mpl-data/fonts/ttf/DejaVuSans.ttf",
                    "sha256": hashlib.sha256(b"regular font").hexdigest(),
                    "size_bytes": 12,
                },
                {
                    "path": ".venv/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
                    "sha256": hashlib.sha256(b"bold font").hexdigest(),
                    "size_bytes": 9,
                },
            ],
            "playwright": None,
            "chromium": None,
            "playwright_note": "not used by strict public beats 05/07",
            "worker_mode": True,
            "code_root": "sealed-execution-capsule",
            "capsule_manifest_sha256": hashlib.sha256(code_root.as_posix().encode()).hexdigest(),
        }

    def _write_complete_bundle_fixture(self, root: Path) -> tuple[Path, release.ReleaseContract]:
        contract = self._complete_contract(root)
        staging_root = root / "staging"
        bundle = staging_root / "bundle"
        workspace = staging_root / "work"
        capsule = workspace / "capsule"
        bundle.mkdir(parents=True)
        workspace.mkdir(exist_ok=True)
        capsule.mkdir()
        staging = release.ReleaseStaging(
            root=staging_root,
            work=workspace,
            snapshot=capsule,
            bundle=bundle,
            final=root / contract.bundle_name,
        )

        clips: dict[str, tuple[Path, str]] = {}
        for beat, source in contract.beats.items():
            path = workspace / f"clip-{beat}.mp4"
            path.write_bytes(f"generated clip {beat}".encode())
            clips[beat] = (path, source.tier)
        picture = workspace / "picture.mp4"
        picture.write_bytes(b"generated picture")
        audio_path = workspace / "audio.wav"
        audio_path.write_bytes(b"generated audio")
        vo_master = workspace / "build" / "v2" / "vo_master.wav"
        vo_master.parent.mkdir(parents=True)
        vo_master.write_bytes(b"generated VO master")
        frames = {}
        png = self._valid_png()
        for label in release.EXPECTED_ACCEPTANCE_FRAME_LABELS:
            path = workspace / f"{label}.png"
            path.write_bytes(png)
            frames[label] = path
        video = bundle / f"{contract.output_basename}.mp4"
        video.write_bytes(struct.pack(">I4s4sI8s", 24, b"ftyp", b"isom", 0, b"isomiso2"))
        captions = bundle / f"{contract.output_basename}.srt"
        captions.write_text(self._valid_srt())
        qc_results = [(name, True, "pass") for name in release.EXPECTED_QC_CHECK_NAMES]

        with (
            patch.object(release, "reverify_release_contract"),
            patch.object(release, "verify_release_snapshot"),
            patch.object(
                release,
                "_segment_timing_records",
                return_value=self._synthetic_vo_timing(contract),
            ),
            patch.object(
                release,
                "_replay_qc_measurements",
                return_value=self._synthetic_qc_measurements(),
            ),
            patch.object(
                release, "curated_source_records", return_value=self._source_records(root)
            ),
            patch.object(
                release,
                "environment_record",
                return_value=self._environment_record(capsule),
            ),
        ):
            release.write_release_bundle(
                contract,
                bundle,
                video_path=video,
                srt_path=captions,
                picture_path=picture,
                audio_path=audio_path,
                strict_workspace=workspace,
                clips=clips,
                qc_results=qc_results,
                acceptance_frames=frames,
                command=[
                    "uv",
                    "run",
                    "python",
                    "scripts/video/render_v2.py",
                    "--release",
                    "--contract",
                    "video/manifests/release_contract.json",
                    contract.release_id,
                ],
                release_staging=staging,
                root=root,
            )
        return bundle, contract

    def _verify_candidate(self, bundle: Path) -> list[str]:
        with patch.object(
            release,
            "_replay_qc_measurements",
            return_value=self._synthetic_qc_measurements(),
        ):
            return release._verify_candidate_bundle(bundle)

    def _fixture_staging(
        self, bundle: Path, contract: release.ReleaseContract
    ) -> release.ReleaseStaging:
        staging_root = bundle.parent
        return release.ReleaseStaging(
            root=staging_root,
            work=staging_root / "work",
            snapshot=staging_root / "work" / "capsule",
            bundle=bundle,
            final=staging_root.parent / contract.bundle_name,
        )

    def _provider_fetch(self, contract: release.ReleaseContract):
        api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

        def fetch(url: str) -> release.ProviderResponse:
            if url == ZENODO_DOI:
                return release.ProviderResponse(200, ZENODO_RECORD_URL, b"")
            if url == api_url:
                return release.ProviderResponse(
                    200,
                    api_url,
                    contract.doi_provider_record.path.read_bytes(),
                )
            raise AssertionError(f"unexpected network URL: {url}")

        return fetch

    def _rewrite_checksums(self, bundle: Path) -> None:
        release._write_sha256sums(bundle)

    def _rewrite_manifest(self, bundle: Path, manifest: dict[str, object]) -> None:
        (bundle / "render.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self._rewrite_checksums(bundle)

    def _rewrite_contract_and_manifest(
        self, bundle: Path, contract: dict[str, object], manifest: dict[str, object]
    ) -> None:
        contract_path = bundle / "release_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        manifest["render"]["contract_sha256"] = release.sha256_file(contract_path)
        self._rewrite_manifest(bundle, manifest)

    def test_complete_provider_bound_candidate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, contract = self._write_complete_bundle_fixture(Path(tmp))
            verified = self._verify_candidate(bundle)
            self.assertIn(f"{contract.output_basename}.mp4", verified)
            self.assertIn("evidence/zenodo-record.json", verified)

    def test_candidate_without_live_finalization_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, contract = self._write_complete_bundle_fixture(Path(tmp))
            canonical = Path(tmp) / contract.bundle_name
            bundle.rename(canonical)
            with self.assertRaisesRegex(release.ReleaseContractError, "READY|file-set"):
                release.verify_release_bundle(canonical)

    def test_pseudo_mp4_is_rejected_before_ffprobe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pseudo = Path(tmp) / "pseudo.mp4"
            pseudo.write_bytes(struct.pack(">I4s4sI8s", 24, b"ftyp", b"isom", 0, b"isomiso2"))
            with self.assertRaisesRegex(release.ReleaseContractError, "moov/mdat"):
                release._validate_mp4(pseudo)

    def test_caption_timestamp_rounding_carries_into_the_next_second(self) -> None:
        self.assertEqual(captions._fmt_ts(1.9996), "00:00:02,000")

    def test_strict_srt_parser_enforces_structure_and_timing(self) -> None:
        release._parse_srt(self._valid_srt())
        invalid = {
            "nonsequential index": self._valid_srt().replace("\n2\n", "\n7\n", 1),
            "millisecond overflow": self._valid_srt().replace(",000 -->", ",1000 -->", 1),
            "invalid seconds": self._valid_srt().replace("00:00:01,000", "00:00:60,000", 1),
            "zero duration": self._valid_srt().replace(
                "00:00:00,000 --> 00:00:01,000",
                "00:00:00,000 --> 00:00:00,000",
                1,
            ),
            "overlap": self._valid_srt().replace(
                "00:00:20,000 --> 00:00:21,000",
                "00:00:00,500 --> 00:00:21,000",
                1,
            ),
            "missing text": self._valid_srt().replace("Opening cue\n\n", "\n", 1),
        }
        for label, text in invalid.items():
            with self.subTest(label=label), self.assertRaises(release.ReleaseContractError):
                release._parse_srt(text)

    def test_mp4_requires_exact_stream_set_and_successful_full_decode(self) -> None:
        def box(kind: bytes, payload: bytes = b"") -> bytes:
            return struct.pack(">I4s", len(payload) + 8, kind) + payload

        mp4_bytes = box(b"ftyp", b"isom\x00\x00\x00\x00isom") + box(b"moov") + box(b"mdat", b"x")
        video_stream = {
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "pix_fmt": "yuv420p",
            "r_frame_rate": "30/1",
            "duration": "1.0",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "structural.mp4"
            path.write_bytes(mp4_bytes)
            no_audio = subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps({"streams": [video_stream], "format": {"duration": "1.0"}}),
                stderr="",
            )
            with (
                patch.object(release.subprocess, "run", return_value=no_audio),
                self.assertRaisesRegex(release.ReleaseContractError, "one video and one audio"),
            ):
                release._validate_mp4(path)

            extra_audio = subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "streams": [
                            video_stream,
                            {"codec_type": "audio", "duration": "1.0"},
                            {"codec_type": "audio", "duration": "1.0"},
                        ],
                        "format": {"duration": "1.0"},
                    }
                ),
                stderr="",
            )
            with (
                patch.object(release.subprocess, "run", return_value=extra_audio),
                self.assertRaisesRegex(release.ReleaseContractError, "one video and one audio"),
            ):
                release._validate_mp4(path)

            probe = subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "streams": [
                            video_stream,
                            {"codec_type": "audio", "duration": "1.0"},
                        ],
                        "format": {"duration": "1.0"},
                    }
                ),
                stderr="",
            )
            decode_failure = subprocess.CompletedProcess([], 1, stdout="", stderr="decode error")
            with (
                patch.object(
                    release.subprocess,
                    "run",
                    side_effect=[probe, decode_failure],
                ),
                self.assertRaisesRegex(release.ReleaseContractError, "full audio/video decode"),
            ):
                release._validate_mp4(path)

            decoded = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            loudness = subprocess.CompletedProcess(
                [],
                0,
                stdout="",
                stderr='{"input_i":"-16.00","input_tp":"-2.00"}',
            )
            with patch.object(
                release.subprocess,
                "run",
                side_effect=[probe, decoded, loudness],
            ) as run:
                release._validate_mp4(path)
            decode_command = run.call_args_list[1].args[0]
            self.assertIn(
                ["-map", "0"],
                [decode_command[index : index + 2] for index in range(len(decode_command) - 1)],
            )

    def test_png_with_valid_shell_and_invalid_zlib_is_rejected(self) -> None:
        png = bytearray(self._valid_png())
        idat = png.index(b"IDAT")
        payload_start = idat + 4
        png[payload_start] ^= 0xFF
        payload_length = struct.unpack(">I", png[idat - 4 : idat])[0]
        crc_start = payload_start + payload_length
        png[crc_start : crc_start + 4] = struct.pack(
            ">I", zlib.crc32(b"IDAT" + png[payload_start:crc_start]) & 0xFFFFFFFF
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid-zlib.png"
            path.write_bytes(png)
            with self.assertRaisesRegex(release.ReleaseContractError, "zlib"):
                release._validate_png(path)

    def test_fabricated_environment_is_rejected_against_worker_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            manifest = json.loads((bundle / "render.json").read_text())
            expected = copy.deepcopy(manifest["environment"])
            manifest["environment"]["python"] = "fabricated 99.0"
            self._rewrite_manifest(bundle, manifest)
            with (
                patch.object(
                    release,
                    "_replay_qc_measurements",
                    return_value=self._synthetic_qc_measurements(),
                ),
                self.assertRaisesRegex(release.ReleaseContractError, "worker"),
            ):
                release._verify_candidate_bundle(bundle, expected_environment=expected)

    def test_contract_locator_release_tag_and_bundle_name_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = self._complete_contract(root)
            arbitrary = root / "video/manifests/arbitrary.json"
            arbitrary.write_bytes(contract.path.read_bytes())
            with self.assertRaisesRegex(release.ReleaseContractError, "canonical locator"):
                release.load_release_contract(arbitrary, root=root)

            raw = copy.deepcopy(contract.raw)
            raw["release"]["id"] = "v2.0.0"
            raw["release"]["bundle_name"] = "tanager-rocks-video-v2.0.0"
            contract.path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
            with (
                patch.object(release, "_verify_source_state", return_value=("a" * 40, "v1.0.0")),
                self.assertRaisesRegex(release.ReleaseContractError, "source.tag"),
            ):
                release.load_release_contract(contract.path, root=root)

        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            with (
                patch.object(
                    release,
                    "_replay_qc_measurements",
                    return_value=self._synthetic_qc_measurements(),
                ),
                self.assertRaisesRegex(release.ReleaseContractError, "directory name"),
            ):
                release._verify_candidate_bundle(bundle, enforce_canonical_name=True)

    def test_beat_03_cannot_rebind_to_beat_05_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            contract = json.loads((bundle / "release_contract.json").read_text())
            manifest = json.loads((bundle / "render.json").read_text())
            contract["beats"]["03"] = {
                "tier": "upgrade",
                "asset_path": release.SAFE_PUBLIC_BEAT_ASSETS["05"],
            }
            manifest["render"]["selected_tiers"]["03"] = "upgrade"
            self._rewrite_contract_and_manifest(bundle, contract, manifest)
            with (
                patch.object(
                    release,
                    "_replay_qc_measurements",
                    return_value=self._synthetic_qc_measurements(),
                ),
                self.assertRaisesRegex(release.ReleaseContractError, "beat 03 upgrade"),
            ):
                release._verify_candidate_bundle(bundle)

    def test_live_provider_confirmation_is_required_and_checks_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, contract = self._write_complete_bundle_fixture(root)
            staging = self._fixture_staging(bundle, contract)
            manifest = json.loads((bundle / "render.json").read_text())
            environment = manifest["environment"]

            def offline(_url: str) -> release.ProviderResponse:
                raise release.ReleaseContractError("offline")

            with (
                patch.object(
                    release,
                    "_replay_qc_measurements",
                    return_value=self._synthetic_qc_measurements(),
                ),
                patch.object(release, "reverify_release_contract"),
                patch.object(release, "verify_release_snapshot"),
                patch.object(release, "environment_record", return_value=environment),
                patch.object(
                    release,
                    "curated_source_records",
                    return_value=manifest["source_files"],
                ),
                self.assertRaisesRegex(release.ReleaseContractError, "offline"),
            ):
                release.finalize_release_staging(
                    staging,
                    contract,
                    live_contract=contract,
                    root=root,
                    fetch=offline,
                )
            self.assertFalse(staging.final.exists())
            self.assertFalse((bundle / release.READY_SENTINEL).exists())

            bad_provider = json.loads(contract.doi_provider_record.path.read_text())
            bad_provider["files"][0]["checksum"] = "md5:" + "0" * 32
            api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

            def bad_fetch(url: str) -> release.ProviderResponse:
                if url == ZENODO_DOI:
                    return release.ProviderResponse(200, ZENODO_RECORD_URL, b"")
                return release.ProviderResponse(
                    200,
                    api_url,
                    json.dumps(bad_provider).encode(),
                )

            with self.assertRaisesRegex(release.ReleaseContractError, "inventory differs"):
                release.verify_live_provider(contract, fetch=bad_fetch)

    def test_forced_final_verifier_failure_is_quarantined_without_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, contract = self._write_complete_bundle_fixture(root)
            staging = self._fixture_staging(bundle, contract)
            manifest = json.loads((bundle / "render.json").read_text())
            environment = manifest["environment"]
            with (
                patch.object(
                    release,
                    "_verify_candidate_bundle",
                    side_effect=[[], release.ReleaseContractError("forced final verifier")],
                ),
                patch.object(release, "reverify_release_contract"),
                patch.object(release, "verify_release_snapshot"),
                patch.object(release, "environment_record", return_value=environment),
                patch.object(
                    release,
                    "curated_source_records",
                    return_value=manifest["source_files"],
                ),
                self.assertRaisesRegex(release.ReleaseContractError, "quarantined"),
            ):
                release.finalize_release_staging(
                    staging,
                    contract,
                    live_contract=contract,
                    root=root,
                    fetch=self._provider_fetch(contract),
                )
            quarantines = list(root.glob(f".{contract.bundle_name}.quarantine-*"))
            self.assertEqual(len(quarantines), 1)
            self.assertFalse((quarantines[0] / release.READY_SENTINEL).exists())
            self.assertFalse(staging.final.exists())

    def test_live_verified_bundle_is_ready_only_after_private_final_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, contract = self._write_complete_bundle_fixture(root)
            staging = self._fixture_staging(bundle, contract)
            manifest = json.loads((bundle / "render.json").read_text())
            with (
                patch.object(
                    release,
                    "_replay_qc_measurements",
                    return_value=self._synthetic_qc_measurements(),
                ),
                patch.object(release, "reverify_release_contract"),
                patch.object(release, "verify_release_snapshot"),
                patch.object(
                    release,
                    "environment_record",
                    return_value=manifest["environment"],
                ),
                patch.object(
                    release,
                    "curated_source_records",
                    return_value=manifest["source_files"],
                ),
            ):
                final = release.finalize_release_staging(
                    staging,
                    contract,
                    live_contract=contract,
                    root=root,
                    fetch=self._provider_fetch(contract),
                )
                verified = release.verify_release_bundle(final)
            self.assertEqual(final.name, contract.bundle_name)
            self.assertIn(release.READY_SENTINEL, verified)
            self.assertFalse(staging.root.exists())

    def test_cleanup_failure_after_ready_promotion_is_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, contract = self._write_complete_bundle_fixture(root)
            staging = self._fixture_staging(bundle, contract)
            manifest = json.loads((bundle / "render.json").read_text())
            with (
                patch.object(
                    release,
                    "_replay_qc_measurements",
                    return_value=self._synthetic_qc_measurements(),
                ),
                patch.object(release, "reverify_release_contract"),
                patch.object(release, "verify_release_snapshot"),
                patch.object(
                    release,
                    "environment_record",
                    return_value=manifest["environment"],
                ),
                patch.object(
                    release,
                    "curated_source_records",
                    return_value=manifest["source_files"],
                ),
                patch.object(
                    release,
                    "_unseal_capsule",
                    side_effect=OSError("forced cleanup failure"),
                ),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                final = release.finalize_release_staging(
                    staging,
                    contract,
                    live_contract=contract,
                    root=root,
                    fetch=self._provider_fetch(contract),
                )
            self.assertEqual(final, staging.final)
            self.assertTrue((final / release.READY_SENTINEL).is_file())
            self.assertTrue(staging.root.exists())
            self.assertTrue(
                any(issubclass(item.category, release.ReleaseCleanupWarning) for item in caught)
            )

    def test_cleanup_warning_as_error_cannot_retract_ready_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, contract = self._write_complete_bundle_fixture(root)
            staging = self._fixture_staging(bundle, contract)
            manifest = json.loads((bundle / "render.json").read_text())
            report = io.StringIO()
            with (
                patch.object(
                    release,
                    "_replay_qc_measurements",
                    return_value=self._synthetic_qc_measurements(),
                ),
                patch.object(release, "reverify_release_contract"),
                patch.object(release, "verify_release_snapshot"),
                patch.object(
                    release,
                    "environment_record",
                    return_value=manifest["environment"],
                ),
                patch.object(
                    release,
                    "curated_source_records",
                    return_value=manifest["source_files"],
                ),
                patch.object(
                    release,
                    "_unseal_capsule",
                    side_effect=OSError("forced cleanup failure"),
                ),
                patch.object(release.sys, "stderr", report),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("error")
                final = release.finalize_release_staging(
                    staging,
                    contract,
                    live_contract=contract,
                    root=root,
                    fetch=self._provider_fetch(contract),
                )
                verified = release.verify_release_bundle(final)
            self.assertEqual(final, staging.final)
            self.assertIn(release.READY_SENTINEL, verified)
            self.assertTrue(staging.root.exists())
            self.assertIn(release.ReleaseCleanupWarning.__name__, report.getvalue())
            self.assertIn(str(staging.root), report.getvalue())

    def test_destination_race_never_overwrites_canonical_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, contract = self._write_complete_bundle_fixture(root)
            staging = self._fixture_staging(bundle, contract)
            staging.final.mkdir()
            marker = staging.final / "racer.txt"
            marker.write_text("racer wins\n")
            manifest = json.loads((bundle / "render.json").read_text())
            environment = manifest["environment"]
            with (
                patch.object(
                    release,
                    "_replay_qc_measurements",
                    return_value=self._synthetic_qc_measurements(),
                ),
                patch.object(release, "reverify_release_contract"),
                patch.object(release, "verify_release_snapshot"),
                patch.object(release, "environment_record", return_value=environment),
                patch.object(
                    release,
                    "curated_source_records",
                    return_value=manifest["source_files"],
                ),
                self.assertRaisesRegex(release.ReleaseContractError, "quarantined"),
            ):
                release.finalize_release_staging(
                    staging,
                    contract,
                    live_contract=contract,
                    root=root,
                    fetch=self._provider_fetch(contract),
                )
            self.assertEqual(marker.read_text(), "racer wins\n")
            quarantines = list(root.glob(f".{contract.bundle_name}.quarantine-*"))
            self.assertEqual(len(quarantines), 1)
            self.assertFalse((quarantines[0] / release.READY_SENTINEL).exists())

    def test_pseudo_bundle_is_rejected_semantically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            contract = json.loads((bundle / "release_contract.json").read_text())
            manifest = json.loads((bundle / "render.json").read_text())
            contract["inputs"] = [{"role": "figure", "path": "made-up.png", "sha256": "a" * 64}]
            manifest["inputs"] = [
                {
                    "role": "figure",
                    "path": "made-up.png",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                }
            ]
            self._rewrite_contract_and_manifest(bundle, contract, manifest)
            with self.assertRaisesRegex(release.ReleaseContractError, "figure contract mismatch"):
                self._verify_candidate(bundle)

    def test_empty_release_and_rights_are_rejected(self) -> None:
        for field in ("release", "rights"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
                contract = json.loads((bundle / "release_contract.json").read_text())
                manifest = json.loads((bundle / "render.json").read_text())
                if field == "release":
                    contract["release"]["title"] = ""
                    manifest["release"]["title"] = ""
                else:
                    contract["rights"]["claims_frozen"] = False
                    manifest["rights"]["claims_frozen"] = False
                self._rewrite_contract_and_manifest(bundle, contract, manifest)
                with self.assertRaises(release.ReleaseContractError):
                    self._verify_candidate(bundle)

    def test_rights_packet_names_operator_and_supporting_evidence_not_code_proof(self) -> None:
        mutations = {
            "trust_root": "code",
            "provider_account_evidence": "self-asserted",
            "generation_plan_evidence": "video/build/another-plan.json",
            "legal_rights_statement": "code proves legal rights",
        }
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
                contract = json.loads((bundle / "release_contract.json").read_text())
                manifest = json.loads((bundle / "render.json").read_text())
                contract["rights"][field] = value
                manifest["rights"][field] = value
                self._rewrite_contract_and_manifest(bundle, contract, manifest)
                with self.assertRaises(release.ReleaseContractError):
                    self._verify_candidate(bundle)

    def test_arbitrary_qc_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            manifest = json.loads((bundle / "render.json").read_text())
            manifest["qc"]["automated"][0]["name"] = "arbitrary-pass"
            self._rewrite_manifest(bundle, manifest)
            with self.assertRaisesRegex(release.ReleaseContractError, "QC names"):
                self._verify_candidate(bundle)

    def test_arbitrary_qc_messages_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            manifest = json.loads((bundle / "render.json").read_text())
            manifest["qc"]["automated"][0]["message"] = "trust me"
            self._rewrite_manifest(bundle, manifest)
            with self.assertRaisesRegex(release.ReleaseContractError, "QC messages"):
                self._verify_candidate(bundle)

    def test_qc_replay_artifacts_cover_picture_and_vo_master(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            manifest = json.loads((bundle / "render.json").read_text())
            replay = manifest["qc"]["replay"]
            self.assertEqual(
                [record["role"] for record in replay["artifacts"]],
                ["assembled_picture", "vo_master"],
            )
            for record in replay["artifacts"]:
                self.assertTrue((bundle / record["path"]).is_file())
            self.assertEqual(
                [record["segment"] for record in replay["vo_segments"]],
                list(common.SEGMENT_FILES),
            )

            replay_path = bundle / replay["artifacts"][0]["path"]
            replay_path.write_bytes(b"tampered picture")
            with self.assertRaisesRegex(release.ReleaseContractError, "size|checksum"):
                self._verify_candidate(bundle)

    def test_fake_png_bytes_are_rejected_even_when_rechecksummed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            manifest = json.loads((bundle / "render.json").read_text())
            frame = manifest["qc"]["acceptance_frames"][0]
            path = bundle / frame["path"]
            path.write_bytes(b"not a PNG")
            frame["sha256"] = release.sha256_file(path)
            frame["size_bytes"] = path.stat().st_size
            self._rewrite_manifest(bundle, manifest)
            with self.assertRaisesRegex(release.ReleaseContractError, "PNG"):
                self._verify_candidate(bundle)

    def test_manual_only_fake_doi_packet_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            contract = json.loads((bundle / "release_contract.json").read_text())
            manifest = json.loads((bundle / "render.json").read_text())
            doi_path = bundle / "evidence/doi.json"
            doi = json.loads(doi_path.read_text())
            doi.pop("provider_record")
            doi_path.write_text(json.dumps(doi, sort_keys=True) + "\n")
            digest = release.sha256_file(doi_path)
            contract["release"]["doi_evidence"]["sha256"] = digest
            record = next(row for row in manifest["generation_evidence"] if row["kind"] == "doi")
            record["sha256"] = digest
            record["size_bytes"] = doi_path.stat().st_size
            self._rewrite_contract_and_manifest(bundle, contract, manifest)
            with self.assertRaisesRegex(release.ReleaseContractError, "provider_record"):
                self._verify_candidate(bundle)

    def test_zenodo_record_id_must_match_archive_doi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            contract = json.loads((bundle / "release_contract.json").read_text())
            manifest = json.loads((bundle / "render.json").read_text())
            doi_path = bundle / "evidence/doi.json"
            doi = json.loads(doi_path.read_text())
            doi["record_id"] = ZENODO_RECORD_ID + 1
            doi["record_url"] = f"https://zenodo.org/records/{ZENODO_RECORD_ID + 1}"
            doi_path.write_text(json.dumps(doi, sort_keys=True) + "\n")
            digest = release.sha256_file(doi_path)
            contract["release"]["doi_evidence"]["sha256"] = digest
            record = next(row for row in manifest["generation_evidence"] if row["kind"] == "doi")
            record["sha256"] = digest
            record["size_bytes"] = doi_path.stat().st_size
            self._rewrite_contract_and_manifest(bundle, contract, manifest)
            with self.assertRaisesRegex(release.ReleaseContractError, "record_id"):
                self._verify_candidate(bundle)

    def test_mismatched_generation_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            manifest = json.loads((bundle / "render.json").read_text())
            music_path = bundle / "evidence/music.json"
            music = json.loads(music_path.read_text())
            music["output"]["sha256"] = "b" * 64
            music_path.write_text(json.dumps(music, sort_keys=True) + "\n")
            record = next(row for row in manifest["generation_evidence"] if row["kind"] == "music")
            record["sha256"] = release.sha256_file(music_path)
            record["size_bytes"] = music_path.stat().st_size
            self._rewrite_manifest(bundle, manifest)
            with self.assertRaisesRegex(release.ReleaseContractError, "generation evidence"):
                self._verify_candidate(bundle)

    def test_duplicate_narration_output_is_rejected(self) -> None:
        for duplicate in ("path", "hash"):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                contract = self._complete_contract(root)
                raw = copy.deepcopy(contract.raw)
                first_relative = raw["audio"]["segments"]["00_title"]
                second_relative = raw["audio"]["segments"]["01_hook"]
                if duplicate == "path":
                    raw["audio"]["segments"]["01_hook"] = first_relative
                else:
                    first_bytes = (root / first_relative).read_bytes()
                    (root / second_relative).write_bytes(first_bytes)
                    duplicate_hash = release.sha256_file(root / second_relative)
                    next(item for item in raw["inputs"] if item["path"] == second_relative)[
                        "sha256"
                    ] = duplicate_hash
                contract.path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
                with (
                    patch.object(
                        release, "_verify_source_state", return_value=("a" * 40, "v1.0.0")
                    ),
                    self.assertRaisesRegex(
                        release.ReleaseContractError, "distinct paths and hashes"
                    ),
                ):
                    release.load_release_contract(contract.path, root=root)

    def test_transient_source_substitution_cannot_enter_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = self._complete_contract(root)
            tagged_blobs = {
                relative: (root / relative).read_bytes()
                for relative in release.CURATED_PUBLIC_SOURCE_PATHS
            }

            def read_tagged_blob(blob_root: Path, commit: str, relative: str) -> bytes:
                self.assertEqual(blob_root, root)
                self.assertEqual(commit, contract.source_commit)
                return tagged_blobs[relative]

            source = contract.segment_paths["00_title"]
            original = source.read_bytes()
            source.write_bytes(b"transient substitution")
            try:
                with self.assertRaisesRegex(release.ReleaseContractError, "snapshot hash mismatch"):
                    release.prepare_release_staging(
                        contract,
                        root=root,
                        read_commit_blob=read_tagged_blob,
                    )
            finally:
                source.write_bytes(original)

            repository_code = root / "scripts/video/release.py"
            repository_code.write_bytes(b"skip-worktree-style worktree mutation\n")
            staging = release.prepare_release_staging(
                contract,
                root=root,
                read_commit_blob=read_tagged_blob,
            )
            try:
                snapshotted = release.load_release_contract(
                    staging.snapshot / release.CANONICAL_CONTRACT_LOCATOR,
                    root=staging.snapshot,
                    asset_root=staging.snapshot,
                    verify_source_checkout=False,
                )
                opened = release.open_release_staging(snapshotted, staging.root, root=root)
                self.assertEqual(opened.snapshot, staging.snapshot)
                source.write_bytes(b"second transient substitution")
                source.write_bytes(original)
                self.assertEqual(
                    snapshotted.segment_paths["00_title"].read_bytes(),
                    original,
                )
                sealed_code = staging.snapshot / "scripts/video/release.py"
                self.assertEqual(
                    sealed_code.read_bytes(),
                    tagged_blobs["scripts/video/release.py"],
                )
                self.assertNotEqual(sealed_code.read_bytes(), repository_code.read_bytes())
                release.verify_release_snapshot(snapshotted, staging)
                with self.assertRaises(PermissionError):
                    snapshotted.segment_paths["00_title"].write_bytes(b"mutate then restore")
            finally:
                release._unseal_capsule(staging.snapshot)

    def test_bundle_rejects_extra_files_directories_and_symlinks(self) -> None:
        mutations = {
            "file": lambda bundle: (bundle / "unexpected.txt").write_text("extra"),
            "directory": lambda bundle: (bundle / "unexpected-dir").mkdir(),
            "symlink": lambda bundle: (bundle / "unexpected-link").symlink_to("missing-target"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
                mutate(bundle)
                with self.assertRaises(release.ReleaseContractError):
                    self._verify_candidate(bundle)

    def test_bundle_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, contract = self._write_complete_bundle_fixture(Path(tmp))
            payload = bundle / f"{contract.output_basename}.mp4"
            payload.write_bytes(b"changed-byte")
            with self.assertRaises(release.ReleaseContractError):
                self._verify_candidate(bundle)

    def test_bundle_rejects_missing_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            manifest = json.loads((bundle / "render.json").read_text())
            manifest["qc"]["automated"].pop()
            self._rewrite_manifest(bundle, manifest)
            with self.assertRaisesRegex(release.ReleaseContractError, "QC"):
                self._verify_candidate(bundle)

    def test_bundle_rejects_missing_manifest_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            checksum_path = bundle / "SHA256SUMS"
            lines = [
                line
                for line in checksum_path.read_text().splitlines()
                if not line.endswith("  render.json")
            ]
            checksum_path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(release.ReleaseContractError, "checksum file-set mismatch"):
                self._verify_candidate(bundle)

    def test_release_schema_closes_frozen_release_and_source_gates(self) -> None:
        schema_path = release.ROOT / "video/manifests/release_contract.schema.json"
        schema = json.loads(schema_path.read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        template = json.loads(
            (release.ROOT / "video/manifests/release_contract.template.json").read_text()
        )
        self.assertFalse(list(validator.iter_errors(template)))

        with tempfile.TemporaryDirectory() as tmp:
            contract = self._complete_contract(Path(tmp)).raw
            self.assertFalse(list(validator.iter_errors(contract)))
            mutations = (
                ("release.id", lambda value: value["release"].__setitem__("id", None)),
                (
                    "release.doi_evidence.sha256",
                    lambda value: value["release"]["doi_evidence"].__setitem__("sha256", None),
                ),
                ("source.commit", lambda value: value["source"].__setitem__("commit", None)),
                ("source.tag", lambda value: value["source"].__setitem__("tag", None)),
                ("source.dirty", lambda value: value["source"].__setitem__("dirty", None)),
            )
            for label, mutate in mutations:
                candidate = copy.deepcopy(contract)
                mutate(candidate)
                with self.subTest(label=label):
                    self.assertTrue(list(validator.iter_errors(candidate)))

    def test_render_schema_closes_qc_and_clip_source_mappings(self) -> None:
        schema_path = release.ROOT / "video/manifests/render_manifest.schema.json"
        schema = json.loads(schema_path.read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self._write_complete_bundle_fixture(Path(tmp))
            manifest = json.loads((bundle / "render.json").read_text())
            self.assertFalse(list(validator.iter_errors(manifest)))

            mutations = (
                (
                    "QC name",
                    lambda value: value["qc"]["automated"][0].__setitem__("name", "arbitrary"),
                ),
                (
                    "QC message",
                    lambda value: value["qc"]["automated"][0].__setitem__("message", "arbitrary"),
                ),
                (
                    "clip tier",
                    lambda value: value["render"]["generated_artifacts"][0].pop("tier"),
                ),
                (
                    "beat source",
                    lambda value: value["render"]["selected_sources"]["05"].__setitem__(
                        "tier", "fallback"
                    ),
                ),
                (
                    "cross-field beat mapping",
                    lambda value: value["render"]["selected_tiers"].__setitem__("03", "upgrade"),
                ),
            )
            for label, mutate in mutations:
                candidate = copy.deepcopy(manifest)
                mutate(candidate)
                with self.subTest(label=label):
                    self.assertTrue(list(validator.iter_errors(candidate)))

    def test_schemas_match_runtime_for_exact_tags_and_beat_sources(self) -> None:
        release_schema = json.loads(
            (release.ROOT / "video/manifests/release_contract.schema.json").read_text()
        )
        render_schema = json.loads(
            (release.ROOT / "video/manifests/render_manifest.schema.json").read_text()
        )
        release_validator = Draft202012Validator(release_schema)
        render_validator = Draft202012Validator(render_schema)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, contract = self._write_complete_bundle_fixture(root)
            manifest = json.loads((bundle / "render.json").read_text())

            for exact_tag in ("v1.0.0", "v2.0.0-rc.1"):
                with self.subTest(exact_tag=exact_tag):
                    source = {"commit": "a" * 40, "tag": exact_tag, "dirty": False}
                    self.assertEqual(
                        release._validate_frozen_source(source),
                        ("a" * 40, exact_tag),
                    )
                    candidate = copy.deepcopy(contract.raw)
                    candidate["source"]["tag"] = exact_tag
                    candidate["release"]["id"] = exact_tag
                    candidate["release"]["bundle_name"] = (
                        f"{release.CANONICAL_BUNDLE_PREFIX}{exact_tag}"
                    )
                    self.assertFalse(list(release_validator.iter_errors(candidate)))

                    render_candidate = copy.deepcopy(manifest)
                    render_candidate["release"]["source_tag"] = exact_tag
                    render_candidate["release"]["id"] = exact_tag
                    render_candidate["release"]["bundle_name"] = (
                        f"{release.CANONICAL_BUNDLE_PREFIX}{exact_tag}"
                    )
                    self.assertFalse(list(render_validator.iter_errors(render_candidate)))

            for pseudo_ref in (
                "HEAD",
                "FETCH_HEAD",
                "ORIG_HEAD",
                "refs/tags/v1.0.0",
                "refs/heads/main",
            ):
                with self.subTest(pseudo_ref=pseudo_ref):
                    source = {"commit": "a" * 40, "tag": pseudo_ref, "dirty": False}
                    with self.assertRaisesRegex(release.ReleaseContractError, "exact tag name"):
                        release._validate_frozen_source(source)

                    candidate = copy.deepcopy(contract.raw)
                    candidate["source"]["tag"] = pseudo_ref
                    self.assertTrue(list(release_validator.iter_errors(candidate)))

                    render_candidate = copy.deepcopy(manifest)
                    render_candidate["release"]["source_tag"] = pseudo_ref
                    self.assertTrue(list(render_validator.iter_errors(render_candidate)))

            valid_beats = (
                ("00", "designed", "video/build/motif.mp4"),
                ("01", "designed", None),
                ("02", "designed", None),
                ("03", "fallback", None),
                ("03", "upgrade", "video/build/v2/upgrades/03.mp4"),
                ("04", "fallback", None),
                ("04", "upgrade", "video/build/v2/upgrades/04.mp4"),
                ("05", "tanager-still", "video/build/v2/fallback_05.png"),
                ("06a", "designed", None),
                ("06b", "fallback", None),
                ("07", "tanager-still", "video/build/v2/fallback_07.png"),
                ("08", "designed", None),
            )
            for beat, tier, asset_path in valid_beats:
                with self.subTest(valid_beat=beat, tier=tier):
                    candidate = copy.deepcopy(contract.raw)
                    candidate["beats"][beat] = {"tier": tier, "asset_path": asset_path}
                    self.assertFalse(list(release_validator.iter_errors(candidate)))
                    self.assertEqual(
                        release.validate_release_tier(beat, tier, asset_path),
                        tier,
                    )

            invalid_beats = (
                ("00", "designed", None),
                ("01", "designed", "video/build/motif.mp4"),
                ("02", "fallback", None),
                ("03", "upgrade", "video/build/v2/upgrades/04.mp4"),
                ("04", "fallback", "video/build/v2/upgrades/04.mp4"),
                ("05", "upgrade", "video/build/v2/upgrades/05.mp4"),
                ("06a", "designed", "video/build/motif.mp4"),
                ("06b", "upgrade", "video/build/v2/upgrades/06b.mp4"),
                ("07", "upgrade", "video/build/v2/upgrades/07.mp4"),
                ("08", "fallback", None),
            )
            for beat, tier, asset_path in invalid_beats:
                with self.subTest(invalid_beat=beat, tier=tier):
                    candidate = copy.deepcopy(contract.raw)
                    candidate["beats"][beat] = {"tier": tier, "asset_path": asset_path}
                    self.assertTrue(list(release_validator.iter_errors(candidate)))
                    contract.path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
                    with (
                        patch.object(
                            release,
                            "_verify_source_state",
                            return_value=("a" * 40, "v1.0.0"),
                        ),
                        self.assertRaises(release.ReleaseContractError),
                    ):
                        release.load_release_contract(contract.path, root=root)

    def test_strict_audio_rejects_vo_only_fallback(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "contract-selected music bed"):
            audio.mix_with_music(Path("missing-vo.wav"), [], music_bed=None, strict=True)

    def test_narration_mapping_is_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "narration segment mapping mismatch"):
            common.probe_segment_durations({})

    def test_public_map_beats_reject_live_capture_tier(self) -> None:
        with self.assertRaisesRegex(release.ReleaseContractError, "live Esri captures"):
            release.validate_release_tier("05", "upgrade", "video/build/v2/upgrades/05.mp4")
        self.assertEqual(
            release.validate_release_tier("05", "tanager-still", "video/build/v2/fallback_05.png"),
            "tanager-still",
        )

    def test_repo_path_rejects_traversal_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "safe.txt").write_text("safe")
            with self.assertRaisesRegex(release.ReleaseContractError, "normalized"):
                release._resolve_repo_file(root, "../safe.txt", "asset")
            (root / "alias.txt").symlink_to(root / "safe.txt")
            with self.assertRaisesRegex(release.ReleaseContractError, "symlink"):
                release._resolve_repo_file(root, "alias.txt", "asset")

    def test_curated_sources_exclude_local_capture_utilities(self) -> None:
        self.assertIn("scripts/video/release.py", release.CURATED_PUBLIC_SOURCE_PATHS)
        self.assertNotIn("scripts/video/capture_05_07.py", release.CURATED_PUBLIC_SOURCE_PATHS)
        self.assertNotIn("scripts/video/_pick_target.py", release.CURATED_PUBLIC_SOURCE_PATHS)

        exposed_scripts = {
            line.removeprefix("!")
            for line in (release.ROOT / ".gitignore").read_text().splitlines()
            if line.startswith("!scripts/video/") and line.endswith(".py")
        }
        manifested_scripts = {
            path
            for path in release.CURATED_PUBLIC_SOURCE_PATHS
            if path.startswith("scripts/video/")
        }
        self.assertEqual(exposed_scripts, manifested_scripts)


if __name__ == "__main__":
    unittest.main()
