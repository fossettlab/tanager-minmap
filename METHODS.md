# Methods

This document is the prose source of truth for the tanager-minmap pipeline. It
tracks the code in `src/tanager_minmap/` and is updated in the same commit as any
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
  `scripts/download_speclib.py` and loaded by `tanager_minmap.speclib`,
  restricted to the target alteration assemblage (alunite, kaolinite, dickite,
  jarosite, hematite, goethite, gypsum, muscovite). The minerals are measured
  on two lab spectrometers — ASD (2151 ch, 0.35–2.5 µm) and Beckman (0.2–3.0 µm)
  — and each spectrum is resampled from its own grid onto the Tanager
  wavelength axis. The ECOSTRESS library remains a possible cross-check.
- **EMIT L2A.** The pinned Goldfield acquisition
  `EMIT_L2A_RFL_001_20230804T191650_2321613_007`, queried from the LP DAAC
  STAC.
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
   downloaded by `scripts/download_scenes.py`) and apply the shared policy in
   `tanager_minmap.quality.mask_tanager_scene`. A pixel is excluded through the
   full cube if Planet's embedded `beta_cloud_mask`, `beta_cirrus_mask`, or
   `nodata_pixels` field is nonzero, or if reflectance is non-finite. A channel
   is excluded if the product's `good_wavelengths` flag is 0 or it falls in the
   project's fixed O2/H2O windows. Verified on all seven local scenes by
   `scripts/audit_tanager_quality.py`: 426 bands span 376–2499 nm; the product
   flags 58 bands, the configured windows cover 53, and their union removes 63
   (363 retained). No numeric reflectance clamp is used: Planet specifies that
   surface reflectance is *typically* 0–1 rather than defining a strict range,
   and a zero lower bound would remove a scene-dependent 0.03–33.04% of
   otherwise QA-clear pixels. See `docs/tanager_quality_mask_policy.md`.
3. **Diagnostic-feature mapping** — continuum-removed band depth (Clark & Roush)
   at the 2200 nm Al-OH doublet, 2265 nm jarosite, and 2340 nm gypsum/carbonate
   (`features.py`, run by `scripts/map_site.py`). Each feature's two continuum
   shoulders are derived data-driven from the median splib07 endmember of the
   diagnostic mineral (kaolinite for Al-OH, jarosite, gypsum), not hand-picked.
   The VNIR Fe-oxide band is also mapped: its center is not fixed by the spec,
   so it is located within a 700–1000 nm window from the hematite endmember
   (841 nm on the Bingham scene). The Bingham maps (4 features)
   are spatially coherent; their comparison against the independent reference
   is reported in the validation sections below, and the Al-OH low
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
   coherent per-mineral abundance maps. Empirically that fixed gate passes
   ~99.8% (Bingham) / 99.9% (Goldfield) of on-domain pixels (p99 ~ 0.6-0.7), so
   it is a light feasibility filter that removes only the extreme-misfit tail;
   the operative detection selector is the per-mineral upper-decile abundance
   floor (step 7 / hero map). Thresholds are coarse, not ground-truth-calibrated
   (that comes with USGS-map validation).
4b. **Validation** — zone agreement against the Rockwell ASTER reference
   (`reference.py`, `validate.py`, run by `scripts/validate_site.py`). The
   reference is categorical (alteration *type* / mineral-*group* classes), so
   the comparison is not a continuous regression but a discrimination test:
   each continuous score map (a diagnostic band depth or an MTMF abundance
score) is
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
4c. **Spatially blocked validation and repeatability** — the confirmatory M2
   analysis (`spatial_validation.py`, `repeatability.py`,
   `strict_inductive.py`; run by `scripts/run_spatial_validation.py`,
   `scripts/run_repeatability.py`, and `scripts/run_strict_inductive.py`). A
   site-specific empirical semivariogram is fit without consulting validation
   performance. The largest documented practical range sets the primary
   square-block side `L`; `2L` is the fixed scale sensitivity, and the same
   range defines the exclusion halo. Only complete geometric blocks are used;
   a QA-failed cell remains `NaN` inside its block rather than invalidating the
   whole block.

   Rank AUC uses every pairwise-finite held-block observation and does not
   depend on whether a threshold can be fit. Thresholded endpoints use
   leave-one-block-out spatial cross-fitting: the held block and its halo are
   excluded, mean block TPR minus mean block FPR is maximized over the unique
   training scores, the highest threshold breaks ties, and that threshold is
   applied once to the held block. Rank and thresholded positive/negative
   denominators are recorded separately. Uncertainty uses 10,000 paired
   complete-block bootstrap replicates (`SEED=42`). Confirmatory null tests use
   9,999 whole-block score/reference permutations and repeat the threshold fit;
   feature and MTMF secondary families receive separate Benjamini-Hochberg
   correction. An interval or null component is gate-eligible only when at
   least 95% of its scheduled replicates are finite.

   The operational MTMF estimand retains one label-free covariance estimate
   from the full QA-valid scene, matching map deployment. Its mandatory
   strict-inductive sensitivity fits a separate MTMF mean and covariance for
   each held block after excluding that block and its halo, then scores only
   the held block. The two estimands are reported separately; strict-inductive
   failure is a result, not permission to shrink blocks or retune the model.
   For repeat acquisitions, the anchor's primary-`L` block grid is reused
   byte-for-byte. One transfer threshold per site and layer is fit on usable
   anchor blocks and applied unchanged to repeat scenes. Continuous fields are
   bilinearly reprojected, categorical masks use nearest-neighbor resampling,
   and pairwise-finite filtering occurs only after block resampling or
   permutation so each acquisition carries its own missingness pattern.
   Empty-versus-empty binary maps have undefined IoU and Dice. Exact block
   permutations are enumerated when their factorial count is at most 9,999;
   otherwise 9,999 seeded unique permutations are used. The full frozen
   protocol and public decision gates are in
   `docs/m2_spatial_validation_preregistration.md`.
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
   degradation. So Sentinel-2 sampling reduces the alunite-kaolinite separation from 5.1°
   to 2.6°, a distinction Tanager retains; this is quantified
   per pair and shown in the band-ablation figure (`viz.band_ablation_panel`).

### Validation results (Goldfield lead scene, contains Cuprite)

Run on the Goldfield lead scene against the aligned Rockwell reference, after
excluding nodata, vegetation, and ~175k semi-corrupted-SWIR pixels (Rockwell
classes 49/50, an unusually large fraction over this area); the scene's
classified ground is dominated by sericite (class 5). Rank AUC of each score
discriminating its published zone:

- **Agree with the published map (AUC 0.70–0.78):** Al-OH band depth 0.784,
  gypsum/carbonate band depth 0.779, alunite MTMF 0.701, muscovite (sericite)
  MTMF 0.695. Alunite abundance peaks in the advanced-argillic class (3) and
  muscovite in the dominant sericite class (5) — Tanager's alteration mapping
  matches the independent USGS product where the assemblage is well represented.
- **Do not discriminate (AUC 0.37–0.58):** kaolinite/dickite MTMF
  (0.472/0.479), hematite MTMF (0.457), goethite (0.576), Fe-oxide band depth
  (0.365). These are
  reported, not suppressed. A score-by-class cross-tab grounds the cause: the
  Fe-oxide signal concentrates in the ferric-iron-*bearing alteration* classes
  (3, 4, 11, 12) rather than Rockwell's standalone, clay-free "ferric iron"
  classes (1, 2) used as the positives — a class-taxonomy mismatch, not a
  detection failure. Kaolinite/dickite abundances are near zero everywhere and
  not concentrated in the argillic class, consistent with kaolinite's spectral
  entanglement with alunite (the very Al-OH proximity the band-ablation result
  exploits). The a-priori positive-class mappings were not adjusted to raise
  these AUCs.
- **Not interpretable:** jarosite (n+=4; the jarosite class is essentially
  absent at Goldfield).

The current per-layer Youden-J thresholds (e.g. alunite MTMF 0.01085, Al-OH
band depth 0.03036) are in-sample descriptive values, not held-out calibration.
M2 replaces them with spatially cross-fitted thresholds; full current numbers
are in `data/intermediate/validation/validation_goldfield_*.csv`.

### Validation results (Bingham lead scene)

Bingham is the weaker validation site, by design and by geology. It is a
porphyry-Cu(-Mo) system, not an acid-sulfate epithermal one: the Rockwell map
labels its near-surface alteration as pervasive sericite (classes 5/10/12/16,
~45 % of the classified domain) and ferrous/coarse-ferric iron (class 21,
~39 %), with only small argillic and advanced-argillic patches (classes 3/4,
~5 % combined). The acid-sulfate mineral suite (alunite, kaolinite, jarosite)
that drove the Goldfield agreement is sparse here, so most of the discriminating
power that validated at Cuprite has little signal to act on.

- **Strongest interpretable agreement:** gypsum/carbonate band depth AUC 0.615,
  consistent with detectable propylitic carbonate. Al-OH band depth is only
  weakly discriminating (0.550) because the Al-OH feature is shared across the
  whole pervasive clay–mica halo rather than concentrated in one zone.
- **MTMF scores do not separate Rockwell's sericite-vs-argillic-vs-ferrous
  splits** (alunite 0.521, kaolinite/dickite 0.567/0.563, muscovite 0.448).
  The 2200 nm Al-OH absorption is shared
  across the muscovite/sericite/illite/kaolinite family, so the filter responds
  to the entire pervasive Al-OH halo, not the specific sericite zones — a
  class-taxonomy-granularity mismatch, the same family of explanation as the
  Goldfield Fe-oxide and kaolinite results. The a-priori mappings were not
  adjusted.
- **Same Fe-oxide anti-pattern as Goldfield:** Fe-oxide band depth 0.397,
  goethite/hematite MTMF 0.338/0.479 — the Fe signal sits in the ferric-iron-
  *bearing alteration* classes, not Rockwell's standalone clay-free ferric
  classes (1, 2) used as positives.
- **Not interpretable:** jarosite (n+=2; the class is essentially absent).

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

EMIT is a spaceborne imaging spectrometer with comparable VSWIR
coverage (285 bands, 381-2493 nm, ~60 m), so re-running the *same* pipeline on
an EMIT scene over the shared site is a cross-sensor consistency check. It is
independent in instrument and acquisition, but deliberately shares this
project's code and spectral library and is therefore not independent mineral
ground truth. The clearest fully-overlapping EMIT L2A granule was selected
programmatically and is now pinned for reproducibility
(`EMIT_L2A_RFL_001_20230804T191650_2321613_007`,
2023-08-04, 4 % cloud, 100 % footprint coverage; queried via the NASA Earthdata
STAC, downloaded with `earthaccess`), orthorectified from its raw
`(downtrack, crosstrack)` array with the granule's geometry lookup table
(`emit.load_emit_reflectance`), and masked identically. The endmember library
and diagnostic features are resampled to EMIT's wavelength axis, so the
mineralogy is computed the same way on both sensors. Acquisition dates differ
(EMIT 2023-08, Tanager 2024-09), which is acceptable because the target is
static surface mineralogy; this is stated as a caveat, not hidden.

- **Spectral agreement** (scene-mean reflectance, resampled to EMIT's 240
  shared finite bands): Pearson r = 0.962, spectral angle 3.72° — the two
  spectrometers see the same reflectance shape over the shared ground.
- **Mineral-detection agreement** (per-mineral MTMF map, Tanager reprojected
  onto the EMIT grid, Pearson r over 192,427 common pixels): **all six minerals
  are positively correlated** — jarosite +0.584, goethite +0.542, alunite
  +0.550, kaolinite +0.456, hematite +0.453, muscovite +0.335. Two sensors
  light up the same ground for each mineral. Correlations are moderate rather
  than near-unity, as expected from the date offset, the different delivered
  grid-cell sizes and reprojection, and the fact that MTMF abundance is not
  absolutely calibrated across differing band sets — the figure therefore
  uses a per-map color stretch and lets the correlation carry the quantitative
  claim.
- **Delivered product grids.** The Tanager ortho product uses a 30 m grid and
  the EMIT comparison product is on an approximately 60 m grid, so a Tanager
  grid cell covers about one-quarter the area of an EMIT grid cell. This is a
  product-grid comparison, not a claim about native sensor footprint,
  resolving power, or minimum mappable feature.

Numbers in `data/intermediate/emit/emit_comparison_goldfield_*.csv`; figure
`figures/goldfield_*_emit_comparison.png`.

### Hero figure (dominant-mineral map)

The submission hero is the **Goldfield/Cuprite** dominant-alteration-mineral map
(`scripts/hero_map.py` → `viz.mineral_map`). Goldfield is used rather than
Bingham because it gives the stronger alteration-group agreement with the
Rockwell ASTER reference (step 4b), whereas Bingham's pervasive porphyry
sericite does not separate into the published categorical zones. The map
composites the infeasibility-gated MTMF
abundance layers into a single dominant-mineral image: each mineral is gated to
its own upper-decile abundance (so the pervasive low-level soil signal does not
wash the map) and normalised by that threshold so the layers are comparable
despite differing matched-filter scales; the per-pixel dominant mineral is the
one most strongly expressed relative to its own detection floor, with opacity
scaled by that strength. The map shows scene-relative library matches that are
consistent with parts of the published alteration context: a muscovite-rich
core and an alunite-rich centre and lineament at Cuprite. Kaolinite and
iron-oxide agreement is weaker and is evaluated explicitly against the
independent map rather than treated as confirmed zoning. Figure
`figures/goldfield_*_hero_mineral_map.png`.

The repository-authored submission composites are generated by
`tanager_minmap.figures` from these same analytical rasters, spectra, masks,
and palettes. The interactive products are generated by
`tanager_minmap.interactive`, which converts the categorical or ordinal arrays
to georeferenced RGBA overlays for Folium without recomputing mineral scores.
`scripts/build_submission.py` assembles the story-page figures and maps from
those modules. These are presentation layers over the governed outputs, not
additional analytical estimators.

### AMD-hazard proxy (acid-generating-potential)

The acid-mine-drainage layer (`scripts/amd_site.py` → `hazard.acid_generating_potential`)
is a qualitative, ordinal **acid-generating-potential (AGP)** map built from the
secondary AMD-indicator assemblage — jarosite, the Fe-oxyhydroxides
(hematite/goethite), and gypsum. The tiers follow the iron-mineral pH zonation
of supergene weathering over sulfide-bearing ground (Swayze et al. 2000, Environ. Sci.
Technol. 34, 47-54): jarosite `KFe₃(SO₄)₂(OH)₆` is stable only
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
strongest cross-sensor support (corrected EMIT detection r +0.584).

### Hard-pair probe (RGB-ambiguous, SWIR-separable patch pairs)

A supplementary probe (`scripts/find_hard_pairs.py`, `scripts/plot_hard_pairs.py`)
adapts the "Similar-but-Different" Sentinel-2 land-cover benchmark
(Robinson, C. & Corley, I., 2026, *Similar but Different: A Benchmark for
Measuring Whether Models Actually Use Multispectral Bands*,
https://geospatialml.com/posts/similar-but-different/), which mines patch pairs
that are near-identical in true-color statistics but pull apart cleanly in the
non-visible bands, to argue that accuracy above an RGB-only ceiling has to come
from those bands. This project applies the same recipe to Tanager's dominant-
alteration-mineral labels (the hero map's own MTMF product,
`viz.dominant_mineral_class`) in place of WorldCover land cover, and to a SWIR
spectral-angle separability check in place of a NIR-mean-difference threshold.

Both sites' lead scenes are tiled into non-overlapping 11×11 px patches (330 m
footprint at Tanager's 30 m GSD — the closer integer to the blog's 320 m / 10 m
Sentinel-2 patch: 11 px is +3.1%, 10 px is −6.25%). A patch is discarded if any
pixel is invalid under the shared Tanager cloud/cirrus/no-data policy or the
RGB display's valid-range rule (`figures.RGB_VALID_RANGE`; zero-tolerance, mirroring the blog's
"discard windows with any cloud/shadow pixel" rule), if its modal dominant-
mineral class (the mode of the hero map's per-pixel class code, gated
identically — infeasibility < 1.0, per-mineral 90th-percentile abundance floor)
is "no detection," or if that modal class's purity is below 70% (the blog's
WorldCover purity rule, used unmodified: this project's population of
confidently-labeled patches at that floor — 268 across both scenes — was ample
and did not require relaxing it). Discard accounting: of 7,290 candidate
patches at Bingham, 4,781 were dropped because at least one pixel failed the
shared validity policy, 2,420 for no dominant detection, and 76 for
sub-70%-purity, leaving 13 labeled; of 7,480 at Goldfield, 2,555 / 4,201 /
469 were dropped in the same order, leaving 255 labeled (268 total).

Each labeled patch's true-color statistics are computed in a shared
post-stretch uint8 space: the 2nd–98th percentile per-channel reflectance
bounds are pooled across BOTH scenes' valid pixels, rather than the per-scene
bounds `figures.rgb_context` uses for the standalone true-color figures, so
cross-scene DN distances sit on one absolute scale. This is the one deliberate
divergence from the repo's existing true-color convention, needed because this
probe compares patches across sites. Candidate cross-label pairs are those
whose RGB-mean-vector AND RGB-std-vector Euclidean distances both fall in the
bottom decile of the pooled cross-label distance distribution (from 30,414
cross-label patch pairs: mean-distance threshold 26.58 DN, std-distance
threshold 3.30 DN), yielding 415 RGB-ambiguous candidates.

SWIR separability is then checked with the project's own pairwise
spectral-angle metric (`speclib.pairwise_spectral_angle`) on each patch's mean
raw reflectance restricted to 2000–2450 nm — a window chosen to bracket the
three fixed SWIR diagnostic centers this project already maps
(`config.DIAGNOSTIC_NM`: Al-OH 2200 nm, jarosite 2265 nm, gypsum/carbonate
2340 nm) with margin, and consistent with the band-ablation finding (above)
that Sentinel-2's entire SWIR collapses to one broad band spanning roughly this
same range. Rather than an invented degree cutoff, the separability bar is
calibrated from the dataset's own same-mineral patch pairs (5,364 pairs,
spectral-angle range 0.08–8.27°): a candidate is called "separable" only if its
cross-label angle exceeds the 95th percentile of that same-label null
distribution (5.45°) — i.e., it differs in the SWIR more than 95% of pairs that
legitimately share the same dominant mineral differ from each other. 18 of the
415 RGB-ambiguous candidates clear that bar; `data/processed/hard_pairs/pairs.csv`
lists all 18, ranked by spectral angle, and
`data/processed/hard_pairs/summary.json` records every threshold and discard
count above.

`figures/hard_pairs.png` (`scripts/plot_hard_pairs.py`) renders the top five
pairs by spectral angle (8.17°–6.43°): the two patches' true-color chips (same
pooled stretch) beside their overlaid SWIR spectra, continuum-removed for
display only (a linear two-point continuum anchored at the 2000/2450 nm window
endpoints, `pairs.continuum_removed` — the same Clark & Roush convention as
`features.band_depth`, generalised to the whole display window rather than one
absorption's local shoulders; the SWIR-separability decision above uses raw
reflectance, not this transform). All five top pairs set hematite against an
Al-OH- or sulfate-bearing mineral (jarosite ×3, goethite ×2) — an honest,
unforced pattern from ranking by spectral angle, not a curated selection. The
highest-ranked pair is the only cross-site hard pair (Bingham jarosite versus
Goldfield hematite). Hematite's own diagnostic absorption is the VNIR Fe³⁺
band, not a SWIR one, so its SWIR reflectance is close to featureless, making
it the
partner most likely to LOOK like any other dark, iron-stained ground in RGB
while showing the least SWIR structure. The figure's spectra bear this out —
hematite's continuum-removed curve stays comparatively flat through the
Al-OH/jarosite/gypsum region in all five panels, while its partner shows a
visible absorption feature roughly where the corresponding diagnostic marker
falls (the exact minimum can sit tens of nm off the marker because the display
continuum spans the whole 2000–2450 nm window rather than each feature's own
local shoulders).

As with the blog's WorldCover labels, the mineral labels here are model
output, not ground truth: they are the hero map's own MTMF classification, so
a "hard pair" documents where this pipeline's SWIR-based mineral call
disagrees with what true color alone would suggest, not an independently
verified mineral identity. The Rockwell ASTER comparison (step 4b) is an
independent remote-sensing agreement check at the alteration-group level; it
is not field ground truth. This probe is downstream of that comparison, not a
substitute for it.

The 268 labeled patches (not just the 18 hard pairs) are also exported as a
standalone, local, evaluation-only dataset
(`scripts/build_hard_pairs_dataset.py` → `data/processed/hard_pairs_dataset/`:
a full-band GeoTIFF chip per patch under `chips/<scene_id>/`, self-contained
`patches.csv` / `pairs.csv` manifests, a `clusters.csv` of RGB-ambiguity-graph
connected components spanning ≥2 labels (the `TANAGER_HARD_PAIRS`
cluster-accuracy hook TanagerFM's band-ablation eval doc expects), and a
`DATASET_CARD.md` with its own construction, cluster-accuracy metric
definition, and limitations writeup) for band-reliance probing of pretrained
hyperspectral models. The build re-derives the RGB-ambiguity graph from
`patches.csv` alone (no cube reload, no re-running MTMF) and round-trip
verifies one seeded-random chip's pixels against its source scene exactly on
every run. Publishing it anywhere is a separate decision from building it.
The live Planet Open STAC `energy-mining` collection identifies the source
imagery as CC BY 4.0 and its catalog root supplies the required
adapted-material attribution; `DATASET_CARD.md` records both sources and the
exact attribution. The current release still excludes the chips pending a
separate operator publication decision.

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
- **MTMF infeasibility gate.** `max_infeas` = 1.0 default — a distribution-informed
  feasibility filter (background ~0.2, anomalous tail >2). Empirically it passes
  ~99.9% of on-domain pixels, so it removes only the extreme-misfit tail rather
  than selecting detections; the operative selector is the upper-decile abundance
  floor below. The Youden-J-optimal per-layer thresholds (step 4b) calibrate the
  detection *scores* against the Rockwell zones, a separate step.
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
- **Hard-pair patch size.** `pairs.PATCH_SIZE_PX` = 11 px (330 m) — the integer
  nearest the Similar-but-Different blog's 320 m Sentinel-2 patch at Tanager's
  30 m GSD.
- **Hard-pair label purity floor.** `pairs.PURITY_FLOOR` = 0.70 — the blog's
  WorldCover rule, reused unmodified (268 patches cleared it across both
  sites, so no relaxation was needed).
- **Hard-pair SWIR window.** `pairs.SWIR_WINDOW_NM` = (2000, 2450) nm —
  brackets `config.DIAGNOSTIC_NM`'s three fixed SWIR centers with margin;
  matches the band-ablation region where Sentinel-2's SWIR collapses to one
  broad band.
- **Hard-pair RGB-ambiguity threshold.** Bottom decile (`quantile` = 0.10) of
  the pooled cross-label RGB mean/std distance distributions — a
  distribution-informed cutoff computed fresh on every run, not a fixed DN
  value.
- **Hard-pair SWIR-separability threshold.** 95th percentile
  (`SWIR_NULL_QUANTILE` = 0.95) of the same-dominant-mineral patch pairs'
  spectral-angle distribution — a cross-label pair must differ more than 95%
  of genuinely-same-mineral pairs differ from each other to count as
  separable.

## Software versions

Python ≥ 3.11. Dependencies and exact versions are pinned in `uv.lock`;
direct dependencies are declared in `pyproject.toml`. The shared data layer is
the exact `tanager-spec==0.1.0` release, with an editable sibling override only
in the development workspace.

## Known caveats

- The delivered Tanager ortho maps use 30 m grid cells. Grid spacing alone does
  not establish native resolving power or a minimum mappable feature, and
  isolated subpixel features cannot be resolved.
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
  come from `scripts/validate_site.py`. Goldfield supplies the compatible
  alteration-zone comparison; Bingham reference overlap is retained as a
  support diagnostic and does not validate the AMD layer.
- The AMD layer is a spectral indicator, not a measured pH or flux, and its
  tiers are relative within a scene (per-mineral upper-tail presence), not an
  absolute acidity scale.
- The L2A reflectance contains negative retrieval estimates and rare values
  above 1 even after product QA. The primary method applies Planet's embedded
  cloud/cirrus/no-data masks, non-finite exclusion, `good_wavelengths`, and the
  fixed atmospheric windows, but no unsourced numeric clamp. Across the seven
  scenes, QA excludes 29.68–42.17% of spatial pixels; upper-bound exclusions at
  1.0 and 1.5 are reserved for the declared sensitivity analysis. See
  `docs/tanager_quality_mask_policy.md` and the generated JSON audit.
- The hard-pair probe's mineral labels are this pipeline's own MTMF output,
  not ground truth (the same caveat the Similar-but-Different blog states for
  its WorldCover labels); it documents where the pipeline's SWIR call
  disagrees with true color, not an independently verified identity. After the
  authoritative QA correction, 17 of 18 mined pairs are Goldfield–Goldfield
  and one is cross-site; Bingham contributes 13 labeled patches. The
  Goldfield-heavy result is an outcome of the fixed ranking and thresholds,
  not a filter applied to exclude Bingham.
