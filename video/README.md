# Reproducible competition video

This directory exposes the lightweight source for the Tanager Rocks competition
video: narration text, editorial plans, rendering code, provenance schemas, and
the procedural opening-motif source. Generated speech, music, intermediate
clips, browser captures, QC images, and final exports remain outside Git and are
distributed only as versioned release/archive assets when their rights permit.

The current historical draft is not a public release candidate. Scientific
claims, generation evidence, a clean source tag, a minted archive DOI, and a
rights review must be frozen before strict release mode will run. A successful
strict automated render creates a candidate only. The candidate must then pass
human real-time playback before publication.

## Two intentionally different modes

Draft mode is convenient for editorial iteration. It can discover an available
animation, use a documented still fallback, or produce a VO-only mix when music
is absent:

```bash
uv run python scripts/video/render_v2.py v5
```

Release mode is fail closed. It accepts only the beat tier and exact file hash
named in a frozen contract, requires all narration/music/figure hashes, requires
generation and rights evidence, checks a clean tagged Git commit, fails on any
automated QC error, and writes a checksummed candidate inside a sealed execution
workspace. It does not mark the candidate approved for publication:

```bash
uv run python scripts/video/render_v2.py \
  --release \
  --contract video/manifests/release_contract.json \
  <release-id>
```

The public release policy forces beats 05 and 07 to the 1920×1080 composites
made by `scripts/video/stills_05_07.py`. Those frames use repository-generated
Tanager products and legends, not Esri World Imagery captures. Local editorial
work may retain a Playwright capture utility, but it is outside the curated
public source set and strict release mode rejects its outputs.

## Source layout

```text
scripts/video/                 assembly, graphics, QC, and release enforcement
video/segments_v2/             one narration text file per beat
video/narration_script_v2.md   timed narration and visual cues
video/build/render_motif.py    procedural 426-band opening graphic
video/build/music_v2_composition_plan.json
video/manifests/               public schemas and incomplete evidence templates
video/CREDITS.md               media provenance, attribution, and license boundary
docs/video_reproduction.md     clone-to-release procedure and acceptance gates
```

`docs/storyboard.md` and `docs/edit_plan.md` are preserved as historical
editorial records because the rendering code cites their cue decisions. They
are not release authority and include superseded draft directions (including
the former 425-band wording and Esri capture route). This README, the
reproduction guide, strict code, and frozen contract control a public release.

Generated paths such as `video/audio_v2/`, `video/build/v2/`, and `output/`
remain ignored. A completed release bundle is written to
`output/releases/tanager-rocks-video-<release-id>/` and contains the MP4, SRT,
`render.json`, `READY.json`, exact
evidence records, credits/notices, QC frames, and `SHA256SUMS`. A contract may
also copy the exact binary media masters into that external bundle after the
rights review; otherwise it must name their immutable HTTPS release-asset URI.

## Preparing the external inputs

1. Regenerate and editorially approve narration only after the claims are
   frozen. Preserve one TTS evidence row per selected segment using
   `video/manifests/tts.template.jsonl`; do not infer missing provider fields.
2. Preserve the selected music-generation and terms evidence using
   `video/manifests/music.template.json`.
3. Export the minted Zenodo record JSON from the provider API, including its
   non-empty file/checksum inventory, preserve it unchanged, hash it, and bind
   that export in a completed copy of
   `video/manifests/doi_evidence.template.json`. A manually typed DOI and record
   URL are not provider-origin evidence.
4. Render the procedural motif with:

   ```bash
   uv run python video/build/render_motif.py
   ```

5. Build the redistribution-safe map composites with:

   ```bash
   uv run python scripts/video/stills_05_07.py
   ```

6. Freeze all selected paths and SHA-256 values in the one canonical locator,
   `video/manifests/release_contract.json`. `release.id` must equal the exact
   source tag and `release.bundle_name` must be
   `tanager-rocks-video-<release.id>`. Change `status` to
   `frozen` only after the source tag, DOI, claim review, and media-rights
   attestations are real.
7. Run the non-rendering preflight:

   ```bash
   uv run python scripts/video/render_v2.py \
     --release \
     --preflight-only \
     --contract video/manifests/release_contract.json
   ```

8. Finalization resolves the DOI over HTTPS, retrieves the authoritative Zenodo
   API record, and compares its file/checksum inventory to the frozen export and
   omitted media masters. Tests inject this client and never use the network.
   Only after private final verification is `READY.json` written and the bundle
   exposed with an atomic no-replace rename.
9. After the strict automated candidate is generated, play the complete MP4 in
   real time and record a passing human review before any publication step.

See [the reproduction guide](../docs/video_reproduction.md) for the full
contract, environment, bundle, and human-review procedure.

## Rights boundary

The repository's MIT license covers repository-authored code and prose. It does
not relicense Planet imagery or adapted material, NASA/USGS inputs, ElevenLabs
speech/music, third-party basemaps, or other generated media. See
[`CREDITS.md`](CREDITS.md) and the repository-level [`NOTICE.md`](../NOTICE.md).
Strict validation checks the operator's evidence packet for completeness and
congruence; it does not prove legal rights. The named operator is the trust root,
and provider account-plan plus generation-plan records are supporting evidence.

## Execution trust boundary

Strict workers run `python -I` from a deterministic capsule containing the
curated render source, canonical contract, lockfile, and every selected input.
The capsule is opened with no-follow descriptor copies, then every inode is
sealed with Darwin `UF_IMMUTABLE`; hashes and seal flags are checked before and
after worker use. Strict mode fails on hosts without this implemented sealing
primitive. This prevents accidental and ordinary concurrent mutation, path
replacement, and mutate-then-restore attacks against the live repository. It is
not protection from a malicious process running as the same macOS user that
deliberately clears `UF_IMMUTABLE`; that process is inside the operator trust
boundary. chmod permissions alone are never accepted as sealing.
