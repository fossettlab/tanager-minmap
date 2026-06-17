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
- **Validation reference.** The USGS *Digital map of hydrothermal alteration
  type, key mineral groups, and green vegetation of the western United States
  derived from automated analysis of ASTER satellite data* (Rockwell & Bonham
  2017, USGS data release, doi:10.5066/F7CR5RK7, public domain). A single
  categorical raster (~30 m, EPSG:4326, 24 alteration/mineral-group classes)
  whose footprint covers Bingham, Goldfield, and the Cuprite benchmark.
  `scripts/download_reference.py` resolves the download from the ScienceBase
  item API, pulls the ERDAS Imagine pair, and clips/reprojects it onto a site's
  lead-scene grid (nearest-neighbour, to preserve class codes). The class table
  is transcribed verbatim from the data-release FGDC metadata into
  `reference.ROCKWELL_CLASSES`. Cuprite (the canonical USGS imaging-spectroscopy
  mineral-mapping site; Swayze et al. 2014) falls inside the Goldfield lead
  scene, so Goldfield validation is anchored at Cuprite.

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
   The VNIR Fe-oxide band is also mapped: its center is not fixed by the spec,
   so it is located within a 700–1000 nm window from the hematite endmember
   (841 nm on the Bingham scene). First Bingham map produced (4 features);
   the maps are spatially coherent but not yet validated, and the Al-OH low
   shoulder currently pins to the search-window edge (~2102 nm) because
   kaolinite reflectance rises shortward — a shoulder refinement to revisit.
4. **Unmixing** — against one medoid endmember per mineral (the real splib07
   sample with the smallest spectral angle to the mineral's median across
   samples; `speclib.select_endmembers`). The SAM baseline is implemented
   (`unmix.spectral_angle` / `sam_classify`, run by `scripts/unmix_site.py`).
   On Bingham, full-spectrum SAM against pure endmembers is weak: best-match
   angles center near 0.21 rad (p5 ≈ 0.14), so the angles are dominated by
   overall spectral shape and mixing rather than diagnostic absorptions — at a
   0.15 rad acceptance threshold only ~6 % of pixels classify (mostly muscovite,
   gypsum, alunite, kaolinite, in coherent clusters over the pit/tailings). This
   is an honest baseline and motivates the covariance-aware matched filter.
   **Matched filter** (`unmix.matched_filter_maps`) is the covariance-aware
   abundance half of MTMF: per endmember,
   `(t-mu)^T C^-1 (x-mu) / (t-mu)^T C^-1 (t-mu)` against the scene mean and band
   covariance (1 at target, 0 at background). Adjacent VSWIR bands are nearly
   collinear so the full-band covariance is singular; it is stabilised by
   diagonal loading (`ridge` fraction, default 1e-2). On Bingham the per-mineral
   MF maps are continuous and spatially coherent — a clear improvement over SAM;
   absolute scores are small (≤~0.02) because 30 m pixels are mixtures of the
   pure endmember. The mixture-tuned **infeasibility** (`unmix.mtmf`) completes
   MTMF: in the background-whitened metric the matched filter explains the
   target-direction component (abundance `alpha`) and the infeasibility is the
   magnitude of the residual orthogonal to it,
   `infeas^2 = (x-mu)^T C^-1 (x-mu) - alpha^2 (t-mu)^T C^-1 (t-mu)` (RX anomaly
   minus the explained part). The exact ENVI infeasibility normalisation is
   proprietary/unpublished (Boardman 1998 is only conceptual), so this is the
   operational feasibility check from the whitened residual, not ENVI-identical;
   its absolute scale is not unit-variance (diagonal loading), so detections are
   gated by the infeasibility distribution. On Bingham the MF-vs-infeasibility
   scatter shows the expected feasibility "nose" — a low-infeasibility tongue to
   higher abundance (true sub-pixel detections) plus a high-infeasibility tail
   (false positives the gate removes); gating at infeasibility < 1.0 yields
   coherent per-mineral abundance maps. Thresholds are coarse, not
   ground-truth-calibrated (that comes with USGS-map validation).
4b. **Validation** — zone agreement against the Rockwell ASTER reference
   (`reference.py`, `validate.py`, run by `scripts/validate_site.py`). The
   reference is categorical (alteration *type* / mineral-*group* classes), so
   the comparison is not a continuous regression but a discrimination test:
   each continuous score map (a diagnostic band depth or an MTMF abundance) is
   tested for how well it separates the published class(es) that contain its
   mineral group from the other classified ground. The positive-class sets are
   derived from the published class *definitions* — e.g. alunite ↔ advanced
   argillic (class 3); kaolinite/dickite ↔ advanced argillic + argillic (3, 4);
   jarosite ↔ class 8; muscovite ↔ sericite classes (5, 10, 12, 16); Fe-oxides
   ↔ ferric-iron classes (1, 2) — and recorded in `reference.MINERAL_TO_ROCKWELL`
   / `FEATURE_TO_ROCKWELL` with per-class justification. Unclassified/nodata
   (0, 48), vegetation (14, 45–47) and the two semi-corrupted-SWIR flags (49,
   50) are excluded, so discrimination is tested only among classified, reliable
   pixels (including bare ground as negatives would inflate separability). The
   rank ROC AUC (= Mann-Whitney U / pair count) gives both separability and a
   significance value in one test, and the Youden-J-optimal score cutoff per
   layer is reported as a *calibrated* detection threshold — the value that best
   matches the external map — which is how the otherwise distribution-informed
   SAM/MTMF thresholds are tied to ground reference. Two caveats are intrinsic:
   gypsum has no ASTER class and cannot be validated this way (the 2340 nm
   feature is checked against the carbonate classes instead), and the ferric
   class does not speciate hematite vs goethite. Goldfield leads (its acid-
   sulfate alteration is the case the Rockwell method was validated on, and its
   lead scene contains Cuprite); Bingham follows.
5. **Band ablation** — the novelty lever (`degrade.py`, run by
   `scripts/ablate_site.py`). The splib07 alteration endmembers, resampled to a
   scene's Tanager wavelength grid, are convolved to Sentinel-2's 13 bands with
   ESA's published spectral response functions (`tanager_spec.srf.load_s2_srf` /
   `simulate`), and pairwise separability is measured as the spectral angle
   between minerals in each sensor's band space. Sentinel-2's entire SWIR is two
   bands — B11 (~1610 nm) and B12 (~2200 nm) — so one broad band spans the whole
   2100–2280 nm Al-OH region. The result (Bingham scene, splib07 medoids):
   the alunite–kaolinite angle falls from 5.1° (Tanager) to 2.6° (S2), a 50 %
   loss of separability; alunite–muscovite −36 %, kaolinite–muscovite −22 %.
   The loss is specific to the SWIR Al-OH region, not universal — the
   VNIR-driven jarosite–goethite contrast (8.6° → 12.5°) is not lost, which is
   the honest control showing the effect is the doublet collapse, not a generic
   degradation. So Sentinel-2 cannot separate advanced argillic (alunite) from
   argillic (kaolinite) alteration, which Tanager resolves; this is quantified
   per pair and shown in the band-ablation figure (`viz.band_ablation_panel`).
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
  information lives in covariance-aware statistics. SAM is the band-independent
  baseline; its weakness on mixed pixels (above) is consistent with this.
- **Endmember selection.** One medoid per mineral (real splib07 sample nearest
  the mineral's median; recorded per run by `select_endmembers`).
- **SAM acceptance threshold.** 0.15 rad default — a distribution-informed
  coarse cutoff, not ground-truth-calibrated.
- **Matched-filter diagonal loading.** `ridge` = 1e-2 default — regularises the
  singular full-band covariance; a numerical parameter, not physical.
- **MTMF infeasibility gate.** `max_infeas` = 1.0 default — distribution-informed
  (background ~0.2, anomalous tail >2). Calibrated against the Rockwell zones in
  step 4b (Youden-J-optimal per-layer thresholds).
- **Validation positive-class sets.** `reference.MINERAL_TO_ROCKWELL` /
  `FEATURE_TO_ROCKWELL`, derived from the Rockwell FGDC class definitions (a
  class is positive for a layer only when its definition names that mineral
  group). Excluded reference classes: `reference.ROCKWELL_EXCLUDED` =
  {0, 14, 45, 46, 47, 48, 49, 50} (nodata, vegetation, semi-corrupted SWIR).
- **Validation statistic.** Rank ROC AUC (= Mann-Whitney U / n+·n−, one-sided);
  detection threshold = Youden-J optimum per layer.

## Software versions

Python ≥ 3.11. Dependencies and exact versions are pinned in `uv.lock`;
direct dependencies are declared in `pyproject.toml`. The shared data layer is
`tanager-spec`, consumed as an editable path dependency.

## Known caveats

- 30 m GSD resolves features larger than roughly 1 ha.
- Surface mineralogy only — not bulk chemistry and not depth.
- Spectral-library mismatch is possible at exotic phases; scope is held to the
  well-characterised alteration assemblage.
- No field validation; validation is against a published USGS map (Rockwell &
  Bonham 2017), itself an automated ASTER product — an independent remote-sensing
  reference, not ground truth. It is ~30 m categorical alteration-*type* zones,
  so it bounds agreement at the alteration-group level, not per-mineral
  abundance. Validation numbers are produced by `scripts/validate_site.py` once
  the reference clip exists; the reference download (doi:10.5066/F7CR5RK7) was
  unavailable from ScienceBase at build time, so the run is pending acquisition.
- The AMD layer is a spectral indicator, not a measured pH or flux.
- The L2A reflectance carries physically out-of-range values (the Bingham
  scene spans −1.9 to 14.6 about a 0.185 median) from cloud/shadow and
  atmospheric-correction overshoot, and ~33 % of pixels are off-nadir
  nodata fill. A valid-range clamp and the invalid-pixel mask
  (`tanager_spec.mask.invalid_pixel_mask`) are applied before analysis.
