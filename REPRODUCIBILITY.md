# Reproducibility

The chain is **clone → `uv sync` → download the public inputs → run the
`tanager-minmap` CLI**. Both repositories are public; the Setup below clones
them side by side, and `[tool.uv.sources]` resolves the `tanager-spec`
dependency from the sibling checkout. The current source checkout reproduces
the corrected products below.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/).
- The shared data layer
  [`tanager-spec`](https://github.com/fossettlab/tanager-spec) checked out
  alongside this repo at `../tanager-spec`. The wheel metadata requires the
  exact `tanager-spec==0.1.0` release, while `[tool.uv.sources]` substitutes the
  editable sibling during development. The two repos therefore must sit side
  by side until that immutable release is publicly reachable.
- For the `emit` stage only: NASA Earthdata credentials in the environment
  (`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`); a free Earthdata Login account.

## Setup

```bash
git clone https://github.com/fossettlab/tanager-spec.git
git clone https://github.com/fossettlab/tanager-minmap.git
cd tanager-minmap
uv sync --extra dev
uv run pytest          # full suite; release-verification tests that need
                       # internal staging artifacts are excluded from this repo
```

## Acquire the public inputs

```bash
uv run python scripts/download_speclib.py                    # USGS splib07a (21.8 MB here)
uv run python scripts/download_scenes.py --site all          # seven Planet Tanager SR scenes (7.79 GB here)
uv run python scripts/download_reference.py --site goldfield # Rockwell ASTER map clip
uv run python scripts/download_reference.py --site bingham
```

The Goldfield EMIT L2A input is pinned as
`EMIT_L2A_RFL_001_20230804T191650_2321613_007`. The `emit` stage uses that
cached file without network access; Earthdata credentials are needed only when
the pinned input is absent and must be downloaded.

## Run the pipeline

```bash
uv run tanager-minmap map      --site bingham   --output out/   # diagnostic band depths
uv run tanager-minmap unmix    --site bingham   --output out/   # SAM + MTMF
uv run tanager-minmap ablate   --site bingham   --output out/   # Sentinel-2 band-ablation
uv run tanager-minmap amd      --site bingham   --output out/   # acid-generating-potential
uv run tanager-minmap hero     --site goldfield --output out/   # dominant-mineral hero map
uv run tanager-minmap emit     --site goldfield --output out/   # EMIT comparison (Earthdata)
uv run tanager-minmap validate --site goldfield --output out/   # vs Rockwell ASTER map
```

The `scripts/*_site.py` drivers run the identical pipeline with the in-repo
layout (`data/` in; `figures/` + `data/intermediate/` out).

## Current corrected development run

The authoritative Planet QA policy is documented in
`docs/tanager_quality_mask_policy.md`. Input hashes are frozen in
`docs/input_manifest.json`, and source/output provenance is recorded in
`experiments/registry.yaml` and `docs/claim_ledger.yaml`.

| Stage | Corrected result | Artifact SHA-256 |
|---|---|---|
| `amd --site bingham` | high/moderate/low/background = 22,536 / 30,678 / 12,949 / 448,099 (514,262 in-scene pixels) | counts CSV `c0ed7bd1…d832` |
| `ablate --site bingham` | alunite–kaolinite 5.138° → 2.578° (49.8% loss) | CSV `dc75f47c…83b5` |
| `validate --site goldfield` | descriptive pixelwise Al-OH AUC 0.7840; alunite MTMF AUC 0.7013 | CSV `1d4a0f19…5a72` |
| `emit --site goldfield` | scene-mean Pearson r = 0.9622 over 240 bands (3.7202°); jarosite map r = 0.5838 | CSV `77dd7fc9…9790` |

The validation values above are descriptive pixelwise estimates, not
independent-replicate inference. Spatially blocked results are generated and
reported separately under the frozen M2 protocol.

## Historical verified reproduction

A clean side-by-side clone of `tanager-minmap` + `tanager-spec`, `uv sync
--extra dev`, then the CLI run against the public inputs reproduced every
headline number exactly at commit `d36046b`. This table is retained as a
historical pre-QA record and must not be substituted for the corrected table
above:

| Stage | Headline result | Reproduced |
|---|---|---|
| `amd --site bingham` | AGP tiers high/moderate/low/background = 25618 / 33555 / 16942 / 518349 | ✓ |
| `ablate --site bingham` | alunite–kaolinite separability 5.14° → 2.58° on Sentinel-2 (50% loss) | ✓ |
| `validate --site goldfield` | Al-OH band-depth AUC 0.780; alunite MTMF AUC 0.706 | ✓ |
| `emit --site goldfield` | scene-mean spectral Pearson r = 0.914 (5.66°); jarosite detection r = +0.594 | ✓ |

The test suite (59 tests at that commit) passed in the fresh environment and
all seven CLI subcommands were registered. Current test counts and the release
commit will be recorded after the final full-suite and clean-wheel checks; no
stale count is presented as current here.

**Scope of this pass (honest note).** `download_speclib.py` was re-run
end-to-end (it fetched and extracted splib07a into the fresh clone). The
multi-gigabyte Planet Tanager scene download (7.79 GB for these seven files)
and the Rockwell reference download (3.07 GB for the IMG/IGE pair) were **not**
re-pulled in this pass; their command-line
interfaces were confirmed to parse, and they are the documented source of the
already-present inputs the run read from. A reviewer reproducing from nothing
runs all of the download commands above first.
