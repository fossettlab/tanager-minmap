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
  viz.py        # hero map, band-ablation panel, EMIT comparison
  cli.py        # tanager-minmap entry point
```

## Install

`tanager-spec` must sit alongside this repo (`../tanager-spec`).

```bash
uv sync --extra dev
```

## Use

```bash
uv run tanager-minmap map --site bingham --output out/
```

(Pipeline implementation is in progress — see `spec.md` timeline.)

## Reproducibility

`uv` with a committed lockfile; STAC queries live in code; methods are tracked
in [`METHODS.md`](METHODS.md). Derivatives are deposited on Zenodo with a DOI.

## Citation

Alex Bradley, Department of Earth, Environmental, and Planetary Sciences,
Washington University in St. Louis — `abradley@wustl.edu`.

## License

MIT.
