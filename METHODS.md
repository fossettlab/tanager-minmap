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

### Validation results (Goldfield lead scene, contains Cuprite)

Run on the Goldfield lead scene against the aligned Rockwell reference, after
excluding nodata, vegetation, and ~175k semi-corrupted-SWIR pixels (Rockwell
classes 49/50, an unusually large fraction over this area); the scene's
classified ground is dominated by sericite (class 5). Rank AUC of each score
discriminating its published zone:

- **Agree with the published map (AUC 0.69–0.78):** Al-OH band depth 0.78,
  gypsum/carbonate band depth 0.78, alunite MTMF 0.71, muscovite (sericite)
  MTMF 0.69. Alunite abundance peaks in the advanced-argillic class (3) and
  muscovite in the dominant sericite class (5) — Tanager's alteration mapping
  matches the independent USGS product where the assemblage is well represented.
- **Do not discriminate (AUC 0.46–0.56):** kaolinite/dickite MTMF (0.47),
  hematite MTMF (0.46), goethite (0.56), Fe-oxide band depth (0.37). These are
  reported, not suppressed. A score-by-class cross-tab grounds the cause: the
  Fe-oxide signal concentrates in the ferric-iron-*bearing alteration* classes
  (3, 4, 11, 12) rather than Rockwell's standalone, clay-free "ferric iron"
  classes (1, 2) used as the positives — a class-taxonomy mismatch, not a
  detection failure. Kaolinite/dickite abundances are near zero everywhere and
  not concentrated in the argillic class, consistent with kaolinite's spectral
  entanglement with alunite (the very Al-OH proximity the band-ablation result
  exploits). The a-priori positive-class mappings were not adjusted to raise
  these AUCs.
- **Not interpretable:** jarosite (n+=6; the jarosite class is essentially
  absent at Goldfield).

The per-layer Youden-J thresholds (e.g. alunite MTMF 0.0010, Al-OH band depth
0.0289) calibrate detection for the layers that discriminate; full numbers in
`data/intermediate/validation/validation_goldfield_*.csv`.

### Validation results (Bingham lead scene)

Bingham is the weaker validation site, by design and by geology. It is a
porphyry-Cu(-Mo) system, not an acid-sulfate epithermal one: the Rockwell map
labels its near-surface alteration as pervasive sericite (classes 5/10/12/16,
~45 % of the classified domain) and ferrous/coarse-ferric iron (class 21,
~39 %), with only small argillic and advanced-argillic patches (classes 3/4,
~5 % combined). The acid-sulfate mineral suite (alunite, kaolinite, jarosite)
that drove the Goldfield agreement is sparse here, so most of the discriminating
power that validated at Cuprite has little signal to act on.

- **Strongest interpretable agreement:** gypsum/carbonate band depth AUC 0.66,
  consistent with detectable propylitic carbonate. Al-OH band depth is only
  weakly discriminating (0.54) because the Al-OH feature is shared across the
  whole pervasive clay–mica halo rather than concentrated in one zone.
- **MTMF scores do not separate Rockwell's sericite-vs-argillic-vs-ferrous
  splits** (alunite 0.52, kaolinite/dickite 0.56, muscovite 0.44). A
  score-by-class cross-tab grounds the muscovite result, which falls below 0.5:
  the muscovite matched filter actually peaks in the *small argillic and
  advanced-argillic* classes (class 4 median +0.00089, class 3 +0.00042) rather
  than the large sericite classes it is mapped to (class 5 −0.00011, class 12
  −0.00041), and the dominant ferrous class 21 carries a higher muscovite score
  (+0.00022) than the sericite positives. The 2200 nm Al-OH absorption is shared
  across the muscovite/sericite/illite/kaolinite family, so the filter responds
  to the entire pervasive Al-OH halo, not the specific sericite zones — a
  class-taxonomy-granularity mismatch, the same family of explanation as the
  Goldfield Fe-oxide and kaolinite results. The a-priori mappings were not
  adjusted.
- **Same Fe-oxide anti-pattern as Goldfield:** Fe-oxide band depth 0.37,
  goethite/hematite MTMF 0.37/0.48 — the Fe signal sits in the ferric-iron-
  *bearing alteration* classes, not Rockwell's standalone clay-free ferric
  classes (1, 2) used as positives.
- **Not interpretable:** jarosite (n+=5; the class is essentially absent).

Full numbers in `data/intermediate/validation/validation_bingham_*.csv`. The
contrast between sites is itself informative: Tanager's continuous alteration
scores validate cleanly against an independent map where a distinct mineral
assemblage is well represented (Goldfield/Cuprite acid-sulfate), and degrade
predictably where the published categorical splits subdivide a single pervasive
spectral family (Bingham porphyry sericite/argillic).
6. **EMIT comparison** — the identical alteration-mapping pipeline (diagnostic
   band depths + MTMF) is run on an overlapping NASA EMIT L2A reflectance scene,
   and the two sensors' products are compared on the Goldfield lead scene.
7. **AMD-hazard proxy** — the secondary AMD-indicator assemblage (jarosite,
   Fe-oxyhydroxides, gypsum) is reduced to a qualitative ordinal
   acid-generating-potential layer (`hazard.acid_generating_potential`).

### EMIT cross-sensor comparison (Goldfield lead scene, contains Cuprite)

EMIT is the only other spaceborne imaging spectrometer with comparable VSWIR
coverage (285 bands, 381-2493 nm, ~60 m), so re-running the *same* pipeline on
an EMIT scene over the shared site is an external check on Tanager's maps with
no shared code, calibration, or acquisition. The clearest fully-overlapping
EMIT L2A granule was selected programmatically (`EMIT_L2A_RFL_001_20230804T1916`,
2023-08-04, 4 % cloud, 100 % footprint coverage; queried via the NASA Earthdata
STAC, downloaded with `earthaccess`), orthorectified from its raw
`(downtrack, crosstrack)` array with the granule's geometry lookup table
(`emit.load_emit_reflectance`), and masked identically. The endmember library
and diagnostic features are resampled to EMIT's wavelength axis, so the
mineralogy is computed the same way on both sensors. Acquisition dates differ
(EMIT 2023-08, Tanager 2024-09), which is acceptable because the target is
static surface mineralogy; this is stated as a caveat, not hidden.

- **Spectral agreement** (scene-mean reflectance, resampled to EMIT's 240
  shared finite bands): Pearson r = 0.91, spectral angle 5.7° — the two
  spectrometers see the same reflectance shape over the shared ground.
- **Mineral-detection agreement** (per-mineral MTMF map, Tanager reprojected
  onto the EMIT grid, Pearson r over ~198k common pixels): **all six minerals
  are positively correlated** — jarosite +0.59, goethite +0.55, alunite +0.55,
  kaolinite +0.47, hematite +0.43, muscovite +0.34. Two independent sensors
  light up the same ground for each mineral. Correlations are moderate rather
  than near-unity, as expected from the date offset, the 2× resolution
  difference (resampling), and the fact that MTMF abundance is not absolutely
  calibrated across differing band sets — the figure therefore uses a per-map
  color stretch and lets the correlation carry the quantitative claim.
- **Spatial detail.** Tanager's 30 m GSD is 2× finer than EMIT's ~60 m (4× the
  pixel density), so Tanager resolves a smaller minimum mappable feature; the
  comparison figure shows the same alunite distribution at both resolutions.

Numbers in `data/intermediate/emit/emit_comparison_goldfield_*.csv`; figure
`figures/goldfield_*_emit_comparison.png`.

### Hero figure (dominant-mineral map)

The submission hero is the **Goldfield/Cuprite** dominant-alteration-mineral map
(`scripts/hero_map.py` → `viz.mineral_map`). Goldfield is used rather than
Bingham because it is the site whose maps validate cleanly against the Rockwell
ASTER reference (step 4b): its acid-sulfate system gives a distinct, mappable
assemblage, whereas Bingham's pervasive porphyry sericite does not separate into
the published categorical zones. The map composites the infeasibility-gated MTMF
abundance layers into a single dominant-mineral image: each mineral is gated to
its own upper-decile abundance (so the pervasive low-level soil signal does not
wash the map) and normalised by that threshold so the layers are comparable
despite differing matched-filter scales; the per-pixel dominant mineral is the
one most strongly expressed relative to its own detection floor, with opacity
scaled by that strength. The result recovers the expected zoning — a
sericite/phyllic core, an alunite advanced-argillic centre and NE lineament at
Cuprite, with kaolinite, jarosite, and Fe-oxides distributed around them.
Figure `figures/goldfield_*_hero_mineral_map.png`.

### AMD-hazard proxy (acid-generating-potential)

The acid-mine-drainage layer (`scripts/amd_site.py` → `hazard.acid_generating_potential`)
is a qualitative, ordinal **acid-generating-potential (AGP)** map built from the
secondary AMD-indicator assemblage — jarosite, the Fe-oxyhydroxides
(hematite/goethite), and gypsum. The tiers follow the iron-mineral pH zonation
of supergene weathering over sulfide-bearing ground (Swayze et al. 2000, USGS
OFR 2000-0205; Williams & Hauff 2007): jarosite `KFe₃(SO₄)₂(OH)₆` is stable only
in acidic (pH ≈ 2–4), oxidising, sulfate-rich conditions and is the diagnostic
active-acid indicator; the Fe-oxyhydroxides are the higher-pH, partly-neutralised
oxidation products; gypsum, absent the acidic iron phases, points to a buffered
setting. Matched-filter abundances are **not** summed across minerals (their
scores are not on a common scale — see the EMIT per-map stretch); instead each
mineral is reduced to a per-pixel presence call using the *same* per-mineral
upper-tail detection floor as the hero map, and each pixel is assigned the tier
of the most acidic indicator present: jarosite → high, else Fe-oxide → moderate,
else gypsum → low, else background. Off-scene/nodata pixels are left unclassified.

The proxy is **relative within a scene, not an absolute acidity**: because
presence is the per-mineral upper decile, "high" means *among the most
jarosite-like pixels in this scene*, not a measured pH. It is run on both sites;
the headline figure is **Bingham/Kennecott**, the mine-waste narrative site
(`figures/bingham_*_amd_agp.png`), where the high-AGP pixels cluster around the
pit/tailings ground rather than spreading uniformly. Tier rasters are written to
`data/intermediate/maps/<site>_*_amd_agp.tif`. Honest caveat: at Bingham
jarosite was absent from the Rockwell *regional alteration* reference (step 4b),
so the AGP layer there is an unvalidated spectral indicator over the waste, not a
map checked against an independent acidity product; at Goldfield jarosite has the
strongest cross-sensor support (EMIT detection r +0.59).

## Key parameters

- **Atmospheric masks.** O2 and H2O absorption windows, owned by
  `tanager_spec.mask`.
- **Diagnostic absorptions.** `config.DIAGNOSTIC_NM`. Fe-oxide
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
- **AMD presence floor.** `quantile` = 0.90 default — a mineral is "present"
  where its infeasibility-gated abundance is in its own top decile (the hero
  map's detection floor reused, so detection means one thing project-wide). AGP
  tiers are assigned by the most acidic indicator present, never by summing
  matched-filter scores across minerals.

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
  abundance, and only where the published assemblage is well represented (it
  validates the alunite/sericite/Al-OH/carbonate signal at Goldfield but not the
  Fe-oxide or kaolinite/dickite layers; see Validation results above). Numbers
  come from `scripts/validate_site.py`; both sites have been validated.
- The AMD layer is a spectral indicator, not a measured pH or flux, and its
  tiers are relative within a scene (per-mineral upper-tail presence), not an
  absolute acidity scale.
- The L2A reflectance carries physically out-of-range values (the Bingham
  scene spans −1.9 to 14.6 about a 0.185 median) from cloud/shadow and
  atmospheric-correction overshoot, and ~33 % of pixels are off-nadir
  nodata fill. A valid-range clamp and the invalid-pixel mask
  (`tanager_spec.mask.invalid_pixel_mask`) are applied before analysis.
