# Video release manifests

This directory contains public, lightweight schemas and deliberately incomplete
templates. Generated media and completed release records are release assets, not
Git inputs.

Release-contract and render-manifest schema version 2 is the sealed-capsule,
live-provider, exclusive-promotion boundary. Version 1 packets are not accepted.

The templates are not evidence. Every `null` must remain `null` until it can be
filled from a verifiable source such as an ElevenLabs generation/history record,
a checked account agreement, a minted DOI, or a local SHA-256 calculation. The
contract schema permits those nulls only while `status` is `template`; a frozen
contract must provide every release/source gate, including a full commit, tag,
and `dirty: false`. Both schemas reject runtime-prohibited pseudo-tags (`HEAD`,
`FETCH_HEAD`, `ORIG_HEAD`, and `refs/*`), while frozen contract beats admit only
the runtime-approved tier/path/null combinations. Runtime also refuses a dirty
or untagged source tree, missing files, substituted beat tiers, and hash
mismatches.

After preflight, strict mode reads every curated source path as a regular-file
blob from the exact tagged commit. It writes those bytes, the contract, every
selected input, and all generation/DOI evidence into a unique no-follow
execution capsule. Worktree or skip-worktree mutations cannot become curated
capsule source. Every capsule inode is sealed with Darwin `UF_IMMUTABLE`;
strict rendering uses `python -I` and reads capsule paths only. Strict mode
fails where that sealing primitive is unavailable.

Files:

- `release_contract.template.json`: copy outside Git as
  `video/manifests/release_contract.json`, then fill and freeze it.
- `release_contract.schema.json`: structural contract schema. Runtime checks in
  `scripts/video/release.py` add file, commit-blob, hash, Git,
  rights-evidence, and cross-record constraints that JSON Schema cannot
  express.
- `tts.template.jsonl`: one template record; create one selected record per
  narration segment in `video/manifests/tts.jsonl`.
- `music.template.json`: selected music generation, terms, and editorial record.
- `doi_evidence.template.json`: incomplete review packet that must bind a hashed,
  unchanged Zenodo API record export containing provider files, sizes, and
  checksums. A DOI typed only into this packet is not sufficient evidence.
- `render_manifest.schema.json`: schema for `render.json`, emitted into the
  external candidate bundle after automated QC passes. It closes all six QC
  names/messages, requires the ten ordered clip IDs/tiers, constrains each
  beat-to-source mapping, and requires the picture/VO replay packet.

The completed `release_contract.json`, `tts.jsonl`, `music.json`, DOI packet,
hashed Zenodo record export, rendered media, media masters, QC frames, QC replay
artifacts, and `SHA256SUMS` remain ignored by Git. The bundle always carries
`evidence/doi.json`, `evidence/zenodo-record.json`,
`qc/replay/picture.mp4`, and `qc/replay/vo_master.wav`. Exact selected media
masters are additionally copied under `masters/` only when the frozen
distribution policy authorizes them.

Automated strict rendering produces a candidate, not a publishable release.
The bundle verifier parses SRT cues strictly, accepts exactly one video and one
audio stream in the final MP4, maps and decodes the complete accepted stream
set, and replays picture, VO-master, stream, timing, loudness, and caption gates
from checksummed bundle artifacts. Free-form replacement QC names or messages
are rejected.

Finalization must resolve the DOI and retrieve the authoritative Zenodo record
through the injected live-provider gate. `evidence/doi.json` binds the reviewed
minted-DOI packet, `evidence/zenodo-record.json` preserves the provider export,
and `READY.json` records the live DOI resolution plus Zenodo response hash and
file inventory. READY is written and verified under a hidden name before
exclusive/no-replace promotion. Once the canonical READY directory appears,
finalization has succeeded; a later private-staging cleanup failure emits
`ReleaseCleanupWarning` and leaves staging for cleanup without retracting the
READY result. A warnings-as-errors policy is contained after promotion and falls
back to a best-effort stderr advisory naming the surviving staging path; cleanup
reporting cannot turn the truthful READY result into failure. Human real-time
playback of the complete candidate must still pass before publication.

The render environment records the host Python version and resolved executable
path and the first `ffmpeg`/`ffprobe` version lines. Those host executables are
not hashed or signed, so host Python/FFmpeg binary identity and process
integrity remain residual trust.

Validate a completed contract without rendering:

```bash
uv run python scripts/video/render_v2.py \
  --release \
  --preflight-only \
  --contract video/manifests/release_contract.json
```

Validate an emitted bundle:

```bash
uv run python scripts/video/release.py \
  verify-bundle output/releases/tanager-rocks-video-<release-id>
```
