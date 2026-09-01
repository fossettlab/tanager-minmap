# tanager-rocks

Mineral and hydrothermal-alteration mapping from Planet Tanager 426-band VSWIR
surface reflectance at two named US mine sites — **Bingham Canyon / Kennecott
(UT)** and the **Goldfield district (NV)** — with a Sentinel-2 band-ablation
comparison and an EMIT cross-sensor benchmark.

Prepared for the Planet Tanager Open Data Competition. See
[`METHODS.md`](METHODS.md) for the full pipeline and
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the clone-to-run recipe.

## The question

How much diagnostic alteration- and mine-waste mineral structure does
Tanager's contiguous 426-band VSWIR preserve relative to Sentinel-2's broad
bands, and what can that support as a screening product for acid-generating
potential?

## Architecture

This repo holds only the analysis specific to the flagship. The data layer
(STAC ingest, georeferenced cube IO, masking, Sentinel-2 SRF simulation,
sampling) is the shared [`tanager-spec`](https://github.com/fossettlab/tanager-spec)
package. Release artifacts require `tanager-spec==0.1.0`; this development
workspace overrides that requirement with the editable sibling checkout.

```
src/tanager_rocks/
  config.py     # sites, target assemblage, diagnostic absorptions
  speclib.py    # USGS / ECOSTRESS reference-library loader
  features.py   # continuum removal + diagnostic-feature maps
  unmix.py      # SAM + MTMF against the reference library
  degrade.py    # Sentinel-2 SRF band-ablation
  hazard.py     # AMD acid-generating-potential proxy
  emit.py       # EMIT L2A query / download / orthorectification
  compare.py    # cross-sensor agreement metrics
  reference.py  # Rockwell ASTER reference + class maps
  validate.py   # rank-AUC discrimination vs the reference map
  viz.py        # hero map, band-ablation panel, EMIT comparison
  figures.py    # submission-story composite figures
  interactive.py # Folium mineral and AGP maps
  quality.py    # authoritative QA/valid-range masking contract
  spatial_validation.py # blocked validation and uncertainty
  strict_inductive.py # held-out covariance sensitivity
  repeatability.py # repeat-acquisition/site transfer analysis
  ensemble_sensitivity.py # endmember/covariance/ridge ensemble
  sensor_ablation.py # full-pipeline native vs Sentinel-2 comparison
  basic_ortho.py # native/basic vs ortho geometry sensitivity
  emit_l2b.py   # EMIT L2B source/schema and agreement validation
  pairs.py      # weak-label RGB-ambiguous/SWIR-separable probe
  pipeline.py   # stage orchestration (shared by the scripts and the CLI)
  cli.py        # tanager-minmap entry point
```

## Install

Until `tanager-spec` 0.1.0 is tagged and made public, its source checkout must
sit alongside this repo (`../tanager-spec`) for the development override.

```bash
uv sync --extra dev
```

## Use

One installed command runs each pipeline stage; inputs come from `--data-root`
(default `data/`), all products land under `--output`.

```bash
# offline stages (a local Tanager scene + the splib07 library):
uv run tanager-minmap map      --site bingham   --output out/   # diagnostic band depths
uv run tanager-minmap unmix    --site bingham   --output out/   # SAM + MTMF
uv run tanager-minmap ablate   --site bingham   --output out/   # Sentinel-2 band-ablation
uv run tanager-minmap amd      --site bingham   --output out/   # acid-generating-potential
uv run tanager-minmap hero     --site goldfield --output out/   # dominant-mineral hero map

# EMIT cross-sensor comparison — needs NASA Earthdata credentials in the
# environment (EARTHDATA_USERNAME / EARTHDATA_PASSWORD):
uv run tanager-minmap emit     --site goldfield --output out/

# validation vs the Rockwell ASTER reference — fetch the reference clip first:
uv run python scripts/download_reference.py --site goldfield
uv run tanager-minmap validate --site goldfield --output out/
```

The `scripts/*_site.py` drivers wrap the same pipeline functions with the repo
layout (`data/` in, `figures/` + `data/intermediate/` out) for development.

## Reproducibility

`uv` with a committed lockfile; STAC queries live in code; methods are tracked
in [`METHODS.md`](METHODS.md). The full clone → sync → download → run recipe and
the verified headline numbers are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
The frozen release will archive its declared derivatives on Zenodo with a DOI;
the release candidate records that deposit as pending until the artifact set
is final.

Reviewers who want the precomputed scientific story without downloading the
large source scenes can use the
[`reviewer quick start`](docs/reviewer_quickstart.md). It explicitly separates
artifact inspection and synthetic software checks from full reproduction.

## Citation

Alex Bradley, Department of Earth, Environmental, and Planetary Sciences,
Washington University in St. Louis — `abradley@wustl.edu`. The machine-readable
software citation is in [`CITATION.cff`](CITATION.cff).

## License

Repository-authored source code and documentation are licensed under the MIT
License. Planet imagery and adapted figures, NASA/EMIT products, USGS inputs,
basemap tiles, and generated media retain their own terms and attribution
requirements; see [`NOTICE.md`](NOTICE.md). The MIT license does not relicense
those third-party materials.
