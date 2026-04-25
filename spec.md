# Track I Spec — Tanager Rocks + Mine-Waste Atlas

Competition: Planet Tanager Open Data Competition
Deadline: August 31, 2026, 11:59 PM PST
Submission type: Team entry (Track I of two)
Target rubric score: ~105 / 115

## Scope

A unified submission that (1) trains ML models to predict rock geochemistry from Tanager 425-band surface reflectance, (2) benchmarks Tanager against multispectral sensors and EMIT, and (3) applies the resulting pipeline to produce a public atlas characterizing US mine tailings and waste-rock piles for residual critical-mineral value and environmental hazard.

## Parts

### Part 1 — Surface Reflectance (tanager-isofit)

Source code location: `~/Desktop/EDC/tanager-isofit/`.

Scope within Track I:
- Consumed as dependency; included in submission as a bundled open-source artifact.
- Preserve existing API.
- Add a minimal test harness that confirms end-to-end L1B → L2A reflectance on one Tanager scene.
- Package for PyPI (optional — GitHub install is acceptable).
- Deposit versioned release on Zenodo with DOI.

Trigger logic: only engage Part 1 if Planet's Open STAC does not serve L2A reflectance. Default path reads L2A directly from STAC.

### Part 2 — Tanager Rocks (spectra → oxide geochemistry)

Primary research question: Does Tanager's 425-band VSWIR coverage improve prediction of major-element oxide geochemistry over multispectral alternatives (ASTER, Sentinel-2, Landsat) and EMIT?

Cascading sub-questions:
1. Can Tanager spectra predict major-element oxides (SiO2, Al2O3, Fe2O3(T), MgO, CaO, Na2O, K2O, TiO2)?
2. Which spectral regions drive predictions, and do they correspond to known mineral absorptions?
3. How does Tanager compare against EMIT at shared AOIs (spectral correlation, detection skill, spatial detail, revisit)?
4. How do models trained on one scene transfer to others (LOSO performance)?

Notebooks (00–06):
- `00_scene_selection` — STAC query; spatial join with GEOROC/PetDB; rank candidate scenes; produce selection map.
- `01_geochem_prep` — Load, clean, harmonize GEOROC + PetDB; coordinate-precision filter; Fe harmonization; anhydrous normalization; report attrition.
- `02_spectral_extraction` — Retrieve Tanager scenes; apply correction if needed; extract spectra at sample locations; mask atmospheric absorption bands; QA.
- `03_modeling` — LOSO XGBoost per oxide, PLS and ElasticNet baselines, SHAP feature importance, bootstrap CIs.
- `04_spectral_degradation` — Convolve 425-band spectra with published SRFs for ASTER/S2/Landsat; retrain on degraded spectra; compare.
- `05_emit_comparison` — Co-located EMIT L2A retrieval at shared AOIs; retrain models on EMIT; produce cross-sensor comparison figure.
- `06_interpretation` — SHAP-on-wavelength figure with mineral-absorption annotations; publication figures.

Library: `src/tanager_geochem/` — pip-installable. CLI: `tanager-geochem predict --scene <stac-id> --output oxides.tif`.

Outputs:
- Trained models (8 oxides × 2 sensors: Tanager, EMIT) with bootstrap CIs.
- Predicted oxide maps for all 52 Tanager scenes (Cloud Optimized GeoTIFFs).
- Cross-sensor comparison tables and figures.
- SHAP-on-wavelength interpretation figure.

### Part 3 — Mine-Waste Mineralogy Atlas

Primary research question: Can Tanager characterize individual mine tailings and waste-rock piles for residual critical-mineral value and environmental hazard at actionable spatial resolution?

Pipeline:
```
Part 2 oxide maps  ─┐
                    ▼
MRDS/AML polygons ─► A0 Pile discovery (prior-informed anomaly detection)
                    │
                    ▼
Tanager reflectance ─► A1 Mineral unmixing (MTMF + deep unmixing)
                    │
                    ▼
                    A2 Score layers:
                       • REE_score, Li_score, Ni_Co_score, Cu_score
                       • AMD_score, Asbestos_score
                    │
                    ▼
                    A3 Per-site report cards + STAC atlas
```

Notebooks (atlas/A0–A3) — see `CLAUDE.md` for file locations.

Library: `src/tanager_minewaste/` — pip-installable. CLI: `tanager-minewaste atlas --site <site-id> --output atlas/sites/<site-id>/`.

Tier-1 AOIs (finalized in Week 1):
- Cuprite, NV — USGS mineralogical benchmark
- Butte, MT — Cu/Mo + Superfund hazard
- Climax/Henderson, CO — Mo/W + pegmatite Li nearby
- Silver Valley, ID — Pb/Zn + AMD
- Iron Mountain, CA — pyrite/AMD end-member
- San Bernardino ultramafics — Ni/Co + asbestos dual signal

Outputs per site (Cloud Optimized GeoTIFFs):
- RGB composite
- Pile mask + confidence
- 8 mineral-abundance layers
- 4 residual-value scores + uncertainty
- 2 hazard scores + uncertainty
- 1-page report card PDF

Atlas v1.0 on Zenodo with DOI; STAC catalog served under `tanager_rocks/atlas/stac/`.

## Data Sources

Tanager imagery: Planet Open STAC.

Geochemistry:
- GEOROC parquet (`data/georoc/`)
- PetDB parquet (`data/petdb/`)

Spectral response functions: `data/srf/` (ASTER, S2, Landsat-8/9; published SRFs).

Cross-sensor: EMIT L2A via LP DAAC STAC at shared AOIs.

Mining ground truth:
- USGS MRDS (Mineral Resources Data System)
- USGS Abandoned Mine Lands inventory
- EPA Superfund / NPL
- USGS NURE, REE deposit studies
- USGS Asbestos Hazard Program
- State geological survey AML inventories (CO, NV, AZ, MT, CA)

Reference spectra: USGS and ECOSTRESS spectral libraries.

## Methods Summary

Atmospheric correction: L2A from STAC if available; tanager-isofit fallback.

Modeling (Part 2):
- Leave-One-Scene-Out CV as primary split
- Baselines: PLS, ElasticNet
- Primary: XGBoost with early stopping, per-oxide
- Dimensionality reduction: optional PCA if N < 100
- Feature importance: SHAP TreeExplainer
- Bootstrap 95% CIs (n=1000)

Multispectral degradation: SRF convolution, not simple averaging. Retrain on degraded spectra with identical CV splits.

EMIT comparison: match AOIs, resample to common grid where appropriate, compare R²/RMSE per oxide + spatial-detail metrics + revisit analysis.

Pile segmentation (Part 3): anomaly detection on continuum-removed spectra; MRDS/AML polygons as priors (not labels).

Mineral unmixing (Part 3):
- Baseline: Tetracorder-style reference matching (USGS library)
- Primary: Mixture Tuned Matched Filtering (MTMF)
- Upgrade: Unsupervised physics-constrained autoencoder unmixing
- Output: per-pixel mineral abundance + uncertainty

Scoring (Part 3): weighted composites of diagnostic minerals → commodity / hazard scores. Document weighting rationale in memo.

Validation:
- Part 2: GEOROC/PetDB held-out samples, LOSO, spatial-block holdout.
- Part 3: published USGS maps at Tier-1 AOIs; ROC/PR per commodity; leave-one-district-out.

## Rubric-Targeted Deliverables

| Rubric item | Deliverable |
|---|---|
| Scientific Integrity | LOSO CV, bootstrap CIs, attrition logging, honest limits section |
| Innovation | First Tanager-based public mine-waste atlas + cross-sensor benchmark |
| Relevance/Impact | Named decision-maker: USGS/BLM/EPA exploration geologists; critical-minerals + environmental-justice framing |
| Feasibility | Working CLI, <1-hour first-pass prospectivity map on new scene |
| Efficiency | STAC-driven, pinned env, `make reproduce` target |
| Accessibility | Three pip-installable libraries; Zenodo atlas; STAC catalog of derivatives |
| Clarity | 5-panel hero figure |
| Narrative | Impact statement + scene-selection recommendation |
| +5 Vertical | Material ID + environmental (both named in tie-breaker) |
| +5 EMIT comparison | Explicit quantitative benchmark |
| +5 Open source | Three libraries + atlas + trained models |

## Timeline

Aug 31, 2026 deadline. Current date: April 18, 2026. ~19 weeks runway.

| Phase | Weeks | Focus |
|---|---|---|
| 1 — Foundation | 1–3 | AOI finalization, STAC + MRDS/AML ingest, end-to-end pipeline on 2 scenes |
| 2 — Part 2 core | 4–7 | LOSO XGBoost, SHAP, multispectral degradation, full 52-scene retrieval |
| 3 — EMIT | 8–9 | Co-located EMIT retrieval, cross-sensor comparison |
| 4 — Part 3 unmixing | 10–12 | MTMF baseline, deep unmixing, Tier-1 site processing |
| 5 — Atlas publication | 13–14 | Score layers, STAC catalog, Zenodo deposit |
| 6 — Narrative + viz | 15–17 | Memo, hero figure, video walkthrough |
| 7 — Polish | 18–19 | Internal review, final submission by Aug 31 |

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Tanager L2A unavailable from STAC | Medium | Medium | tanager-isofit fallback (already built) |
| Too few matched samples per scene | Medium | High | Pool across scenes; PLS/ElasticNet for low-N |
| EMIT coverage does not overlap priority AOIs | Medium | Medium | Select AOIs by EMIT coverage first |
| 30 m pixel is too coarse for small piles | High | Medium | Focus on >1 ha impoundments; report detection limit honestly |
| Mineral library mismatches at exotic sites | Medium | Medium | Extend library from USGS + published spectra; report per-site |
| Scoring weights are heuristic | High | Low | Ship mineral abundances as primary output; scores as secondary |
| Atmospheric correction artifacts | Low | High | Compare L2A vs isofit outputs at 3 sites |
| Spatial autocorrelation inflates metrics | High | Medium | LOSO primary; report k-fold inflation for context |
| Scope bloat | High | High | Part 3 Tier-2 sites are stretch only; core = Tier-1 six sites |

## Submission Package

```
submission/
  README.md                       # 1-page summary + impact statement
  memo.pdf                        # 3 pages
  hero_figure.png                 # 5-panel story figure
  figures/                        # supporting figures
  video.mp4                       # <3 min walkthrough
  links.md                        # GitHub, Zenodo, HF, STAC catalog URLs
```

Repo artifacts referenced:
- `github.com/<org>/tanager-rocks` (code)
- Zenodo DOIs: tanager-isofit, tanager-geochem, tanager-minewaste, AML Atlas v1.0
- STAC catalog URL for atlas derivatives

## Open Questions

- Is Planet Open STAC serving Tanager L2A, or L1B only? Determines Part 1 load.
- Which EMIT granules overlap Tier-1 AOIs? Run STAC cross-query Week 1.
- GitHub org: `washu-eeps` or personal? Default to `washu-eeps` for visibility.
- Hugging Face presence needed for Part 2 trained-model release?
- Team composition and role assignments (TBD).
