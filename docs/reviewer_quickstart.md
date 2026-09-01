# tanager-minmap reviewer quick start

Status: release-candidate guide, 2026-08-10. This path uses precomputed
repository artifacts and does not claim independent reproduction. Final public
URLs, release checksums, and the archive DOI remain publication gates.

## Five-minute path

From the repository root, serve the static submission locally:

```bash
python3 -m http.server 8000 --directory submission
```

Open `http://localhost:8000`. The story page gives the scientific question,
two-site context, methods summary, limitations, and links to the precomputed
figures. The embedded interactive maps request Esri basemap tiles and therefore
need network access; every core scientific panel is also available as a local
PNG without those tiles.

Read these panels in order:

1. `submission/figures/goldfield_spectra.png` — measured USGS library spectra
   and mapped-scene spectra in the diagnostic VSWIR region;
2. `submission/figures/bingham_20250911_191523_58_4001_band_ablation.png` —
   the source-preserving Tanager-to-Sentinel-2 spectral-response comparison;
3. `submission/figures/hard_pairs_story.png` — mechanically selected
   RGB-ambiguous patches with model-derived weak labels and separable SWIR
   spectra;
4. `submission/figures/goldfield_validation_pair.png` and
   `submission/figures/goldfield_20240925_185504_87_4001_emit_comparison.png`
   — independent-product alteration-group agreement and shared-method
   cross-sensor consistency; and
5. `submission/figures/bingham_20250911_191523_58_4001_amd_agp.png` — a
   scene-relative acid-generating-potential screening layer, not measured pH.

## Claim and provenance check

Validate the machine-readable experiment registry and public-claim ledger:

```bash
uv run python scripts/validate_repro_metadata.py
```

The validator checks schema, source paths, generating commands, values, status,
and public destinations. It does not rerun the scientific pipeline. Each public
number remains governed by `docs/claim_ledger.yaml`; unavailable or exploratory
evidence is retained rather than converted into a positive result.

Local verification on 2026-08-10: the command completed successfully against
the current registry and claim ledger.

At release time, the exact reviewer bundle will also contain `SHA256SUMS`.
Verify it from inside the extracted bundle with:

```bash
shasum -a 256 -c SHA256SUMS
```

That checksum step is pending until the final release artifacts are frozen.

## Smallest code-level check

The analytical unit tests use synthetic arrays and do not require the raw
Planet scenes:

```bash
uv sync --extra dev
uv run pytest tests/test_features.py tests/test_unmix.py tests/test_hazard.py -q
```

This checks the continuum-removal arithmetic, planted-target MTMF behavior,
infeasibility gating, and ordinal screening-tier logic. It is a software check,
not validation of the two case-study maps.

Local verification on 2026-08-10: the exact `uv run pytest` command above
completed with 15 passing tests. The preceding `uv sync` setup step was not
rerun while a separate long-lived governed analysis process was active; clean-
clone environment setup remains a release-candidate gate.

## Full reproduction

Use `REPRODUCIBILITY.md` for the complete data acquisition and seven-command
workflow. That route requires the public `tanager-spec==0.1.0` dependency and
the large upstream datasets. Until the dependency is tagged and publicly
reachable, the quick-start above is a review of frozen precomputed artifacts,
not a clean-clone install claim.

## Interpretation boundary

- Goldfield and Bingham are two case studies, not a geographic-generalization
  sample.
- The Rockwell comparison is independent remote-sensing agreement at the
  alteration-group level, not field ground truth.
- The EMIT comparison shares code and endmembers and is therefore a
  cross-sensor consistency check.
- The hard-pairs labels are MTMF-derived weak labels.
- The Bingham screening tiers are relative within the scene and require field
  spectroscopy or geochemistry before operational use.
