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
- **Reference spectra.** USGS Spectral Library Version 7 base spectra
  (splib07a; Kokaly et al. 2017, USGS Data Series 1035), acquired by
  `scripts/download_speclib.py` and loaded by `tanager_rocks.speclib`,
  restricted to the target alteration assemblage (alunite, kaolinite, dickite,
  jarosite, hematite, goethite, gypsum, muscovite). The minerals are measured
  on two lab spectrometers — ASD (2151 ch, 0.35–2.5 µm) and Beckman (0.2–3.0 µm)
  — and each spectrum is resampled from its own grid onto the Tanager
  wavelength axis. The ECOSTRESS library remains a possible cross-check.
- **EMIT L2A.** Queried from the LP DAAC STAC at whichever study site EMIT
  overlaps (confirmed in Week 1).
- **Validation reference.** Published USGS mineral/alteration maps; Cuprite is
  the canonical neighbour to Goldfield, and Bingham/Kennecott is
  well-characterised in the literature.

## Sites

Bingham Canyon / Kennecott (UT, hero) and the Goldfield district (NV,
alteration showcase). `scripts/confirm_sites.py` walked the open STAC catalog
(2026-06-15) and matched the scenes whose footprints intersect each site:
Bingham returns 2 scenes (2025-09-11, cloud 3–13 %) and Goldfield 5 scenes
(2024-09-25 and 2025-02-22, cloud 0–2 %), matching the spec's scene counts.
All carry the `ortho_sr_hdf5` asset, and EMIT L2A overlaps both sites (88 and
80 granules). Site identity was then verified against the USGS Mineral
Resources Data System (`scripts/confirm_site_identity.py`, MRDS WFS): the
search box around each site contains a developed deposit of the expected name
and commodity — Bingham Open Pit Mine (Producer, Cu-Mo) and Goldfield District
Gold Deposits (Producer, Au) — so `config.SITES` records `confirmed=True`.
Scene IDs are recorded in `config.SITES`.

## Pipeline overview

1. **Site + product confirmation** — confirm footprints, the `ortho_sr_hdf5`
   asset, and EMIT overlap (`tanager_spec.stac`).
2. **Ingest + masking** — load the SR cube (`tanager_spec.io.load_tanager_sr_hdf5`,
   downloaded by `scripts/download_scenes.py`); mask O2/H2O absorption bands
   (`tanager_spec.mask`). Verified on the Bingham 2025-09-11 scene: 426 bands
   over 376–2499 nm, EPSG:32612 (UTM 12N) at 30 m, reflectance in fraction
   units (scene median 0.185); the absorption windows drop 53 of 426 bands.
3. **Diagnostic-feature mapping** — continuum-removed band depth (Clark & Roush)
   at the 2200 nm Al-OH doublet, 2265 nm jarosite, and 2340 nm gypsum/carbonate
   (`features.py`, run by `scripts/map_site.py`). Each feature's two continuum
   shoulders are derived data-driven from the median splib07 endmember of the
   diagnostic mineral (kaolinite for Al-OH, jarosite, gypsum), not hand-picked.
   First Bingham map produced; the maps are spatially coherent but not yet
   validated, and the Al-OH low shoulder currently pins to the search-window
   edge (~2102 nm) because kaolinite reflectance rises shortward — a shoulder
   refinement to revisit. VNIR Fe-oxide features are still to be added.
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
- The L2A reflectance carries physically out-of-range values (the Bingham
  scene spans −1.9 to 14.6 about a 0.185 median) from cloud/shadow and
  atmospheric-correction overshoot, and ~33 % of pixels are off-nadir
  nodata fill. A valid-range clamp and the invalid-pixel mask
  (`tanager_spec.mask.invalid_pixel_mask`) are applied before analysis.
