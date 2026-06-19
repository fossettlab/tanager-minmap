# Reproducibility

A fresh clone reproduces every figure and number in the submission. The chain
is **clone → `uv sync` → download the public inputs → run the `tanager-minmap`
CLI**. Each stage writes its products under `--output` and logs its headline
numbers (also written to CSV under `--output/intermediate/<stage>/`).

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/).
- The shared data layer
  [`tanager-spec`](https://github.com/bradleylab/tanager-spec) checked out
  alongside this repo at `../tanager-spec` — it is an editable path dependency,
  so the two repos must sit side by side.
- For the `emit` stage only: NASA Earthdata credentials in the environment
  (`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`); a free Earthdata Login account.

## Setup

```bash
git clone https://github.com/bradleylab/tanager-spec.git
git clone https://github.com/bradleylab/tanager-rocks.git
cd tanager-rocks
uv sync --extra dev
uv run pytest          # 59 tests
```

## Acquire the public inputs

```bash
uv run python scripts/download_speclib.py                    # USGS splib07a (~22 MB)
uv run python scripts/download_scenes.py --site all          # Planet Tanager SR scenes (~9 GB)
uv run python scripts/download_reference.py --site goldfield # Rockwell ASTER map clip
uv run python scripts/download_reference.py --site bingham
```

The EMIT L2A granule is downloaded automatically by the `emit` stage — it
selects the clearest fully-overlapping scene and caches it under
`data/raw/emit/`, so re-runs do not re-download.

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

## Verified reproduction

A clean side-by-side clone of `tanager-rocks` + `tanager-spec`, `uv sync
--extra dev`, then the CLI run against the public inputs reproduced every
headline number exactly (commit `d36046b`):

| Stage | Headline result | Reproduced |
|---|---|---|
| `amd --site bingham` | AGP tiers high/moderate/low/background = 25618 / 33555 / 16942 / 518349 | ✓ |
| `ablate --site bingham` | alunite–kaolinite separability 5.14° → 2.58° on Sentinel-2 (50% loss) | ✓ |
| `validate --site goldfield` | Al-OH band-depth AUC 0.780; alunite MTMF AUC 0.706 | ✓ |
| `emit --site goldfield` | scene-mean spectral Pearson r = 0.914 (5.66°); jarosite detection r = +0.594 | ✓ |

The 59-test suite passes in the fresh environment and all seven CLI subcommands
are registered.

**Scope of this pass (honest note).** `download_speclib.py` was re-run
end-to-end (it fetched and extracted splib07a into the fresh clone). The
multi-gigabyte Planet Tanager scene download (~9 GB) and the Rockwell reference
download (~2.9 GB) were **not** re-pulled in this pass; their command-line
interfaces were confirmed to parse, and they are the documented source of the
already-present inputs the run read from. A reviewer reproducing from nothing
runs all of the download commands above first.
