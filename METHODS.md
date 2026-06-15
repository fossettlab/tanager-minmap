# Methods

This document is the prose source of truth for the tanager-rocks pipeline. It
tracks the code in `src/tanager_rocks/` and is updated in the same commit as any
change to methodology (per the lab reproducibility rule). Sections marked
*(pending)* describe steps whose implementation and parameter choices are not
yet fixed; they will be completed as the corresponding modules are built.

## Data sources

- **Tanager surface reflectance.** The `ortho_sr_hdf5` product from Planet's
  Open STAC catalog, selected via `tanager_spec.config.TANAGER_SR_ASSET` and
  loaded with `tanager_spec.io.load_tanager_sr_hdf5`. These scenes are already
  L2A surface reflectance; no radiance product is used and no atmospheric
  correction is run in this pipeline. `tanager-isofit` is cited as the bundled
  open-source correction tool but is not executed here.
- **Reference spectra.** USGS and ECOSTRESS spectral libraries, restricted to
  the target alteration assemblage (alunite, kaolinite, dickite, jarosite,
  hematite, goethite, gypsum, muscovite).
- **EMIT L2A.** Queried from the LP DAAC STAC at whichever study site EMIT
  overlaps (confirmed in Week 1).
- **Validation reference.** Published USGS mineral/alteration maps; Cuprite is
  the canonical neighbour to Goldfield, and Bingham/Kennecott is
  well-characterised in the literature.

## Sites

Bingham Canyon / Kennecott (UT, hero) and the Goldfield district (NV,
alteration showcase). Site footprints recorded in `config.SITES` are
approximate scene-centroid coordinates and are flagged `confirmed=False` until
verified against USGS USMIN/MRDS and a basemap (data-integrity rule).

## Pipeline overview

1. **Site + product confirmation** — confirm footprints, the `ortho_sr_hdf5`
   asset, and EMIT overlap (`tanager_spec.stac`).
2. **Ingest + masking** — load the SR cube; mask O2/H2O absorption bands
   (`tanager_spec.io`, `tanager_spec.mask`).
3. **Diagnostic-feature mapping** *(pending)* — continuum removal, then band
   depth at the 2200 nm Al-OH doublet, 2265 nm jarosite, 2340 nm
   gypsum/carbonate, and the VNIR Fe-oxide features (`features.py`).
4. **Unmixing** *(pending)* — SAM and MTMF against the reference library, with
   MTMF as the primary method (`unmix.py`).
5. **Band ablation** *(pending)* — SRF-degrade Tanager to Sentinel-2 bands
   (`tanager_spec.srf.simulate`), repeat steps 3–4, and quantify the loss.
6. **EMIT comparison** *(pending)* — the same mapping at the overlapping site;
   report spectral correlation, detection agreement, and spatial detail.
7. **AMD-hazard proxy** *(pending)* — jarosite + Fe-oxide + gypsum assemblage
   as a qualitative acid-generating-potential layer.

## Key parameters

- **Atmospheric masks.** O2 and H2O absorption windows, owned by
  `tanager_spec.mask`.
- **Diagnostic absorptions.** `config.DIAGNOSTIC_NM`, from spec.md step 3. Fe-oxide
  VNIR centres are resolved from the reference library, not hard-coded.
- **Primary method.** MTMF (covariance-aware matched filter), chosen over a
  band-independent classifier because the methodology suite found Tanager's
  information lives in covariance-aware statistics.
- **Detection gating, library provenance, MNF component count** *(pending)* —
  to be recorded here when fixed.

## Software versions

Python ≥ 3.11. Dependencies and exact versions are pinned in `uv.lock`;
direct dependencies are declared in `pyproject.toml`. The shared data layer is
`tanager-spec`, consumed as an editable path dependency.

## Known caveats

- 30 m GSD resolves features larger than roughly 1 ha.
- Surface mineralogy only — not bulk chemistry and not depth.
- Spectral-library mismatch is possible at exotic phases; scope is held to the
  well-characterised alteration assemblage.
- No field validation; validation is against published USGS maps.
- The AMD layer is a spectral indicator, not a measured pH or flux.
