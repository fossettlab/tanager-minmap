"""Build draft videos or a fail-closed, provenance-bound public release.

Draft mode preserves the existing convenience behavior::

    uv run python scripts/video/render_v2.py v5

Release mode requires a frozen, hash-complete contract and never discovers or
substitutes media inputs::

    uv run python scripts/video/render_v2.py \
      --release \
      --contract video/manifests/release_contract.json \
      v1.0.0

Use ``--preflight-only`` with release mode to verify the contract without
rendering. Release outputs are staged, verified, then exclusively promoted as
``output/releases/tanager-rocks-video-<release-id>/``; prior drafts and bundles
are never overwritten.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ``-I`` removes the script directory from sys.path. Strict workers restore
# only their sealed sibling-module directory, never the live repository.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import release  # noqa: E402
from common import (  # noqa: E402
    BUILD_V2,
    LOGS_V2,
    OUTPUT,
    QC_V2,
    ROOT,
    STRICT_BUILD_ENV,
    STRICT_SNAPSHOT_ENV,
    STRICT_STAGING_ENV,
    STRICT_WORKER_ENV,
    build_edl,
    probe_segment_durations,
    run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        nargs="?",
        help="draft suffix, or release id (defaults to v2 for drafts and contract id for releases)",
    )
    parser.add_argument(
        "--release", action="store_true", help="enable the fail-closed public release path"
    )
    parser.add_argument(
        "--contract", type=Path, help="frozen release contract (required with --release)"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify a release contract and exit without rendering",
    )
    parser.add_argument("--_release-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_staging-root", type=Path, help=argparse.SUPPRESS)
    return parser


def _mux(
    picture: Path,
    audio_final: Path,
    output: Path,
    *,
    contract: release.ReleaseContract | None,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(picture),
        "-i",
        str(audio_final),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
    ]
    if contract is not None:
        command += [
            "-metadata",
            f"title={contract.title}",
            "-metadata",
            "artist=Alex Bradley",
            "-metadata",
            f"comment=Provenance and media terms: {contract.archive_doi}",
            "-metadata",
            "copyright=Code MIT; media rights and credits are documented separately",
        ]
    command.append(str(output))
    run(command, LOGS_V2 / "final_mux.log")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    contract: release.ReleaseContract | None = None
    staging: release.ReleaseStaging | None = None

    if args.release:
        if args.contract is None:
            raise SystemExit("--contract is required with --release")
        if args._release_worker:
            if args._staging_root is None:
                raise release.ReleaseContractError("strict worker requires a staging root")
            snapshot_root = args._staging_root / "work" / "capsule"
            expected_contract = snapshot_root / release.CANONICAL_CONTRACT_LOCATOR
            if args.contract.absolute() != expected_contract.absolute():
                raise release.ReleaseContractError(
                    "strict worker must load the snapshotted release contract"
                )
            contract = release.load_release_contract(
                args.contract,
                root=snapshot_root,
                asset_root=snapshot_root,
                verify_source_checkout=False,
            )
        else:
            contract = release.load_release_contract(args.contract)
        version = args.version or contract.release_id
        if version != contract.release_id:
            raise release.ReleaseContractError(
                f"release id mismatch: command={version!r}, contract={contract.release_id!r}"
            )
        print(
            f"release preflight PASS: {contract.release_id} / {contract.source_tag} / "
            f"{len(contract.assets)} hashed inputs"
        )
        if args.preflight_only:
            return
        if args._release_worker:
            assert args._staging_root is not None
            host_root = args._staging_root.parents[2]
            staging = release.open_release_staging(contract, args._staging_root, root=host_root)
            expected_build = staging.work / "build" / "v2"
            if BUILD_V2.absolute() != expected_build.absolute():
                raise release.ReleaseContractError(
                    f"strict worker build mismatch: {BUILD_V2} != {expected_build}"
                )
        else:
            if args._staging_root is not None:
                raise release.ReleaseContractError("staging root is internal to strict workers")
            staging = release.prepare_release_staging(contract)
            child_env = os.environ.copy()
            child_env[STRICT_BUILD_ENV] = str(staging.work / "build" / "v2")
            child_env[STRICT_SNAPSHOT_ENV] = str(staging.snapshot)
            child_env[STRICT_STAGING_ENV] = str(staging.root)
            child_env[STRICT_WORKER_ENV] = "1"
            child_env["PYTHONNOUSERSITE"] = "1"
            child_env["PYTHONPATH"] = str(staging.snapshot / "scripts" / "video")
            for name in ("PYTHONHOME", "PYTHONSTARTUP", "PYTHONBREAKPOINT"):
                child_env.pop(name, None)
            command = [
                sys.executable,
                "-I",
                str(staging.snapshot / "scripts" / "video" / "render_v2.py"),
                "--release",
                "--contract",
                str(staging.snapshot / release.CANONICAL_CONTRACT_LOCATOR),
                "--_release-worker",
                "--_staging-root",
                str(staging.root),
                contract.release_id,
            ]
            try:
                subprocess.run(command, cwd=staging.snapshot, env=child_env, check=True)
            except subprocess.CalledProcessError as exc:
                raise release.ReleaseContractError(
                    f"strict render failed; isolated staging retained at {staging.root}"
                ) from exc
            sealed_contract = release.snapshot_release_contract(contract, staging)
            release.verify_release_snapshot(sealed_contract, staging)
            release.reverify_release_contract(contract)
            final = release.finalize_release_staging(
                staging,
                sealed_contract,
                live_contract=contract,
            )
            print(f"strict finalized candidate PASS -> {final.relative_to(ROOT)}")
            return
    else:
        if (
            args.contract is not None
            or args.preflight_only
            or args._release_worker
            or args._staging_root
        ):
            raise SystemExit("--contract and --preflight-only require --release")
        version = args.version or "v2"

    # Keep render-only dependencies (notably Matplotlib) behind the strict
    # preflight gate. A contract check must not initialize graphics or media
    # tooling, and a blocked template should fail in well under render time.
    import assemble
    import audio
    import beats
    import captions
    import qc

    BUILD_V2.mkdir(parents=True, exist_ok=True)
    LOGS_V2.mkdir(parents=True, exist_ok=True)
    QC_V2.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("== probing segment durations ==")
    selected_segments = None if contract is None else contract.segment_paths
    durations = probe_segment_durations(selected_segments)
    edl = build_edl(durations)
    total_vo = edl[-1].abs_end
    print(f"total VO duration: {total_vo:.3f}s")

    print("== building clips ==")
    clips = beats.build_all_clips(
        edl,
        None if contract is None else contract.strict_sources,
        release_archive_doi=None if contract is None else contract.archive_doi,
        release_repository=None if contract is None else contract.repository_url,
    )

    print("== assembling picture (xfade seams) ==")
    picture = assemble.assemble_picture(edl, clips)

    print("== audio backbone ==")
    audio_final = audio.build_audio(
        edl,
        segment_paths=selected_segments,
        music_bed=None if contract is None else contract.music_bed,
        strict=contract is not None,
    )

    print("== final mux ==")
    if contract is None:
        video_path = OUTPUT / f"draft_{version}.mp4"
        srt_path = OUTPUT / f"draft_{version}.srt"
    else:
        assert staging is not None
        video_path = staging.bundle / f"{contract.output_basename}.mp4"
        srt_path = staging.bundle / f"{contract.output_basename}.srt"
    _mux(picture, audio_final, video_path, contract=contract)
    print(f"wrote {video_path.relative_to(ROOT)}")

    print("== captions ==")
    srt = captions.build_srt(edl, srt_path)
    print(f"wrote {srt.relative_to(ROOT)}")

    print("== QC ==")
    qc_results = qc.run_qc(video_path, srt, total_vo)
    for name, ok, message in qc_results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {message}")
    frames = qc.extract_acceptance_frames(video_path, edl)
    if contract is not None:
        frames = release.bind_acceptance_frames(frames)
    print(f"  extracted {len(frames)} acceptance frames to {QC_V2.relative_to(ROOT)}/")

    if contract is not None:
        assert staging is not None
        if any(not ok for _, ok, _ in qc_results):
            raise release.ReleaseContractError(
                f"release QC failed; partial staging retained for diagnosis: {staging.root}"
            )
        release.write_release_bundle(
            contract,
            staging.bundle,
            video_path=video_path,
            srt_path=srt,
            picture_path=picture,
            audio_path=audio_final,
            strict_workspace=staging.work,
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
                contract.contract_locator,
                contract.release_id,
            ],
            release_staging=staging,
            root=staging.snapshot,
        )
        print(f"strict candidate bundle verified at {staging.bundle}")


if __name__ == "__main__":
    main()
