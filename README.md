# tanager-rocks

Mineral and hydrothermal-alteration mapping from Planet Tanager 425-band VSWIR
surface reflectance at two named US mine sites — **Bingham Canyon / Kennecott
(UT)** and the **Goldfield district (NV)** — with a Sentinel-2 band-ablation
comparison and an EMIT cross-sensor benchmark.

Track I of the [Planet Tanager Open Data Competition](../Planet-TermsConditions-TanagerCompetition.pdf)
(deadline: 31 August 2026). See [`spec.md`](spec.md) for the full pipeline,
rubric mapping, and timeline.

## The question

Can Tanager's contiguous 425-band VSWIR resolve the alteration- and
mine-waste mineralogy — alunite, kaolinite, jarosite, Fe-oxides, gypsum,
muscovite — at a specificity that Sentinel-2 provably cannot, and what does
that reveal about acid-mine-drainage hazard?

## Architecture

This repo holds only the analysis specific to the flagship. The data layer
(STAC ingest, georeferenced cube IO, masking, Sentinel-2 SRF simulation,
sampling) is the shared [`tanager-spec`](https://github.com/bradleylab/tanager-spec)
package, consumed as an editable path dependency.

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
  pipeline.py   # stage orchestration (shared by the scripts and the CLI)
  cli.py        # tanager-minmap entry point
```

## Install

`tanager-spec` must sit alongside this repo (`../tanager-spec`).

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
Derivatives are deposited on Zenodo with a DOI.

## Citation

Alex Bradley, Department of Earth, Environmental, and Planetary Sciences,
Washington University in St. Louis — `abradley@wustl.edu`.

## License

MIT.
