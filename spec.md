# tanager-rocks — Flagship Spec

Competition: Planet Tanager Open Data Competition
Deadline: August 31, 2026, 11:59 PM PST
Submission type: Track I flagship — **solo build, tight scope**
GitHub: `github.com/bradleylab/tanager-rocks`

> **Re-scoped 2026-06-10.** This supersedes the three-part ML-geochemistry +
> national-atlas plan (preserved at `archive_pre_competition/spec_3part_geochem_superseded.md`).
> A rubric bake-off across all Tanager candidates selected mineral/alteration
> mapping at two iconic named US mine sites as the flagship; the ML angle moved
> to the `TanagerFM` individual entry, and the methodology suite
> (`tanager-spectralsep` / `-infotheory` / `-anomaly` / `-featureimp`) is the
> rigor backbone, not a standalone submission.

## One sharp question

Can Tanager's contiguous 425-band VSWIR resolve the alteration- and mine-waste
mineralogy — alunite, kaolinite, jarosite, Fe-oxides, gypsum, muscovite — at
Bingham Canyon and Goldfield at a specificity that Sentinel-2 provably cannot,
and what does that reveal about acid-mine-drainage hazard?

## Goals

A focused, visual demonstration that Tanager's SWIR resolves mine-waste and
hydrothermal-alteration minerals multispectral sensors cannot, at two iconic
named US sites, delivered with a hero map, a quantitative EMIT cross-sensor
comparison, and a reusable open-source mineral-mapping tool. The entry targets
all four rubric categories and all three tie-breakers, and it makes the implicit
archive argument the prize rewards: Planet should open more scenes over mining
districts.

## Sites

| Site | Scenes | Role | Why |
|---|---|---|---|
| **Bingham Canyon / Kennecott, UT** | 2 (centroids 40.56/-112.08, 40.78/-112.01) | Hero | World's largest open-pit Cu porphyry + a massive tailings impoundment; named, iconic, US, with a real acid-mine-drainage and residual-value story |
| **Goldfield district, NV** | 5 (centroids ~37.4-37.7/-117.1-117.2) | Alteration showcase | Classic hydrothermal alteration (alunite/kaolinite/jarosite/Fe-oxides) — the textbook VSWIR mineral-mapping case, adjacent to the USGS Cuprite benchmark |

Site identities are from scene-centroid coordinates and must be confirmed against
a basemap and USGS USMIN/MRDS footprints in Week 1 before being asserted in the
submission (data-integrity rule).

## Data

- **Tanager:** the **`ortho_sr_hdf5` surface-reflectance product** from Planet's
  Open STAC catalog — selected explicitly via `tanager_spec.config.TANAGER_SR_ASSET`
  / `load_tanager_sr_hdf5`. No radiance product, no atmospheric correction run;
  the scenes are already L2A surface reflectance.
- **`tanager-isofit`** is cited as a bundled open-source artifact (the correction
  tool that *would* produce SR from radiance), not executed in this pipeline.
- **Reference spectra:** USGS and ECOSTRESS spectral libraries (the target
  alteration assemblage).
- **EMIT L2A** via LP DAAC STAC at whichever of the two sites it overlaps.
- **Validation reference:** published USGS mineral/alteration maps (Cuprite is the
  canonical neighbour; Bingham/Kennecott is well-characterized in the literature).

## Pipeline (reuse-first)

1. **Site + product confirmation.** Confirm Bingham/Goldfield footprints; confirm
   the `ortho_sr_hdf5` asset; confirm EMIT overlap. *(reuse `stac.py`,
   `tanager_spec.stac`)*
2. **Ingest + masking.** Load the SR cube; mask O2/H2O absorption bands.
   *(reuse `tanager_spec.io`, `tanager_spec.mask`)*
3. **Diagnostic-feature mapping.** Continuum removal; map the 2200 nm Al-OH
   doublet (alunite vs kaolinite/dickite), 2265 nm jarosite, 2340 nm
   gypsum/carbonate, and VNIR Fe-oxide features (hematite/goethite).
4. **Unmixing.** SAM + **MTMF** against the USGS/ECOSTRESS library →
   per-mineral abundance maps with per-pixel scores.
5. **Band-ablation (novelty lever).** SRF-degrade Tanager → Sentinel-2 bands,
   repeat steps 3-4, and quantify what S2 loses — it cannot split the Al-OH
   doublet, so it cannot separate alunite from kaolinite. *(reuse `degrade.py`)*
6. **EMIT comparison.** Same mapping at the overlapping site; report spectral
   correlation, mineral-detection agreement, and spatial detail.
7. **AMD-hazard proxy.** Jarosite + Fe-oxide + gypsum assemblage → a qualitative
   acid-generating-potential layer, hedged (a spectral indicator, not a
   measured pH or flux).

## Reuse vs build

- **Reuse:** `tanager-spec` (ingest/mask/SRF/sample); existing `stac.py`,
  `spectra.py`, `degrade.py`, `viz.py`.
- **Build:** continuum-removal + diagnostic-feature module; SAM/MTMF unmixing
  module; USGS-library loader; `tanager-minmap` CLI; the hero figure; `pyproject.toml`.
- **Archive:** `geochem.py`, `cv.py` and the geochem-prediction notebooks →
  `archive_pre_competition/` (the ML-geochem scope moved to TanagerFM).

## Methods & integrity

- Mineral identification is library-anchored (USGS/ECOSTRESS), spectroscopically
  defensible (diagnostic absorptions), and validated against published USGS maps.
- Mineral choice for the unmixing model follows the methodology suite's result:
  covariance-aware methods are where Tanager's information lives, which is why
  MTMF (a covariance-aware matched filter) is the primary method over a
  band-independent classifier.
- **Honest limits:** 30 m GSD (features >~1 ha), surface mineralogy only (not
  bulk chemistry, not depth), spectral-library mismatch at exotic phases, no
  field validation.
- **Reproducibility:** `uv` with committed lockfile; STAC query in code (no
  hand-copied URLs); one-command run; derivatives deposited on Zenodo with a DOI.

## Deliverables -> rubric

| Rubric | Deliverable |
|---|---|
| Scientific Integrity & Innovation (30) | Library-anchored mineral ID; band-ablation proving VSWIR necessity; validation vs USGS maps; honest limits |
| Application / Use Case (30) | Named sites + end-user (state surveys / EPA / mining operators); AMD-hazard + residual-value framing; "open more mining-district scenes" |
| Workflow & Tool (20) | `tanager-minmap` CLI; STAC-driven; `uv.lock`; one-command run; Zenodo DOI |
| Visualization & Storytelling (20) | Hero mineral map (Bingham) + band-ablation panel + stand-alone impact statement |
| +5 strategic vertical | Material identification |
| +5 EMIT comparison | Quantitative cross-sensor comparison at one site |
| +5 open source | `tanager-minmap` + `tanager-isofit` + the methodology suite |

## Timeline (solo, ~11 weeks -> Aug 31)

| Wk | Focus | Gate |
|---|---|---|
| 1 | Confirm Bingham/Goldfield footprints vs USMIN/basemap; confirm `ortho_sr_hdf5` + EMIT overlap; scaffold repo (`pyproject`, uv, CI) | Sites + product + EMIT confirmed |
| 2-3 | Continuum removal + diagnostic-feature maps on Bingham; first real mineral map | One real map |
| 4-5 | SAM + MTMF vs USGS library; Goldfield; validate vs published USGS maps | Validated maps |
| 6 | Band-ablation (S2 degrade) + quantify loss | Novelty panel |
| 7 | EMIT comparison | Tie-breaker landed |
| 8 | AMD/value layers; `tanager-minmap` CLI + tests | Tool shippable |
| 9 | Hero figure + memo draft | Hero readable in 10 s |
| 10 | Video + Zenodo deposit + fresh-clone reproducibility pass | Clean reproduce |
| 11 | Buffer + internal review | Submit |

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ~~EMIT does not overlap either site~~ (retired 2026-06-15) | — | — | Resolved: EMIT L2A overlaps **both** sites (88 granules Bingham, 80 Goldfield) per `scripts/confirm_sites.py` |
| 30 m too coarse for small piles | Medium | Medium | Bingham tailings + Goldfield alteration are large; report detection limit |
| Spectral-library mismatch at exotic phases | Medium | Low | Restrict to the well-characterized alteration assemblage |
| Validation map unavailable for a site | Low | Medium | Goldfield/Cuprite are USGS-mapped; lead validation there |
| Scope creep back toward the atlas | Medium | High | Two sites, qualitative AMD only; no commodity-scoring system |

## Submission package

```
submission/
  README.md          # 1-page summary + impact statement + reproduce command
  memo.pdf           # 2-3 pages
  hero_figure.png    # Bingham mineral map + band-ablation panel
  figures/           # Goldfield maps, EMIT comparison, AMD layer
  video.mp4          # <3 min walkthrough
  links.md           # GitHub, Zenodo DOI, STAC
```

## Open items

- ✅ **Product + scenes confirmed (2026-06-15).** `scripts/confirm_sites.py`
  walked the open STAC catalog: Bingham = 2 scenes, Goldfield = 5 scenes
  (counts match this spec), all carrying the `ortho_sr_hdf5` asset. Scene IDs
  recorded in `tanager_rocks.config.SITES`.
- ✅ **EMIT overlap confirmed (2026-06-15)** at both sites (88 / 80 granules).
- ⬜ Confirm exact site *identity*/footprints against USGS USMIN/MRDS on a
  basemap before asserting site names in the submission (`confirmed` stays
  `False` in `config.SITES` until then).
