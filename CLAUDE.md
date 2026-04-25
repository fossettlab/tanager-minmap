# Track I: Tanager Rocks + Mine-Waste Atlas (Competition Submission)

This folder coordinates **Track I** of the Planet Tanager Open Data Competition strategy. Track I is a unified submission spanning three technical components that share data, infrastructure, and a narrative.

## Three Parts, One Submission

### Part 1 — Surface Reflectance Correction

Code: `~/Desktop/EDC/tanager-isofit/` (pre-existing, external to this folder).

- Wraps JPL's ISOFIT to convert Tanager L1B radiance → L2A surface reflectance.
- Role in Track I: atmospheric-correction layer. Used only if Planet's Open STAC does not serve L2A.
- Accessibility deliverable: packaged, versioned, Zenodo-deposited, pip-installable.
- Owner in Track I: keep API surface stable; add Tanager-specific test harness.

### Part 2 — Tanager Rocks (spectra → geochemistry)

Code: this folder's `src/` and `notebooks/`.

- Train ML models (XGBoost/PLS) on Tanager 426-band surface-reflectance spectra at GEOROC/PetDB sample coordinates to predict major-element oxides.
- Multispectral degradation comparison (ASTER, Sentinel-2, Landsat).
- EMIT cross-sensor comparison at 2–3 shared AOIs.
- Outputs: trained oxide-prediction models + per-scene predicted oxide maps.

### Part 3 — Mine-Waste Mineralogy Atlas

Code: `./atlas/` (subdirectory of this folder, shares `src/`).

- Ingests Part 2's oxide maps plus mineral-abundance unmixing to produce per-site characterization of tailings and waste-rock piles.
- Residual-value scores (REE, Li, Ni/Co, Cu) and hazard scores (AMD, asbestos).
- Outputs: public atlas on Zenodo + STAC catalog of derivatives.

## Why Unified

- Part 2 provides the oxide maps that Part 3 consumes.
- Part 1 provides the reflectance inputs that both depend on.
- The three together produce a complete, coherent submission that scores across all rubric categories.
- Splitting them into three separate submissions would dilute the narrative and waste rubric points on duplicated overhead.

## Rubric Alignment Summary

| Rubric item | Supplied by |
|---|---|
| Scientific Integrity | Parts 2 & 3 (LOSO CV, honest limits) |
| Innovation | Part 3 (first Tanager mine-waste atlas) + Part 2 (EMIT benchmark) |
| Relevance/Impact | Part 3 (named decision-maker: USGS/BLM/EPA exploration geologists) |
| Feasibility | All three (runnable CLI, Zenodo atlas) |
| Efficiency | All three |
| Accessibility | All three (Zenodo + STAC + pip installable) |
| Clarity | Part 3 hero figure |
| Narrative | Part 3 (critical minerals + environmental justice) |
| +5 Vertical | Part 3 (material ID) |
| +5 EMIT comparison | Part 2 |
| +5 Open source | Parts 1, 2, 3 — three libraries + atlas |

Target score: ~105/115.

## Stack (shared across all three parts)

- Python 3.11+
- Core: `numpy`, `pandas`, `geopandas`, `xarray`, `rioxarray`
- Geospatial: `pystac-client`, `stackstac`, `shapely`, `rasterio`
- ML: `scikit-learn`, `xgboost`, `shap`
- Unmixing (Part 3): `spectral`, custom MTMF / autoencoder modules
- Atmos correction (Part 1): `isofit` (via `tanager-isofit`)
- Viz: `matplotlib`, `seaborn`, `folium`, `lonboard`
- Package management: `pip` (requirements.txt per-part, unified at repo root)
- Package publication: `build` + `twine`
- Data publication: Zenodo API client

## Directory Layout

```
tanager_rocks/
  CLAUDE.md                  # this file — cross-part coordination
  spec.md                    # unified spec for the three-part submission
  archive_pre_competition/   # prior planning documents
  data/
    raw/                     # Tanager scenes (gitignored)
    georoc/                  # GEOROC parquet (gitignored)
    petdb/                   # PetDB parquet (gitignored)
    srf/                     # spectral response functions (committed)
    mrds_aml/                # USGS MRDS + AML databases (gitignored, cached)
    intermediate/            # processed outputs (gitignored)
  notebooks/
    00_scene_selection.ipynb
    01_geochem_prep.ipynb
    02_spectral_extraction.ipynb
    03_modeling.ipynb
    04_spectral_degradation.ipynb
    05_emit_comparison.ipynb
    06_interpretation.ipynb
    atlas/
      A0_pile_discovery.ipynb
      A1_unmixing.ipynb
      A2_value_and_hazard_scores.ipynb
      A3_atlas_publication.ipynb
  src/
    tanager_geochem/         # pip-installable Part 2 library
      __init__.py
      stac.py                # Tanager + EMIT STAC queries
      geochem.py              # GEOROC/PetDB loading, harmonization
      spectra.py             # Spectral extraction and QA
      degrade.py              # Multispectral simulation from SRFs
      cv.py                  # LOSO and spatial-block CV
      predict.py             # Inference CLI for new scenes
      viz.py
    tanager_minewaste/       # pip-installable Part 3 library
      __init__.py
      piles.py               # Pile segmentation
      unmix.py               # Mineral unmixing (MTMF + deep unmixing)
      scoring.py             # Value + hazard scores
      atlas.py               # STAC item generation for derivatives
  models/                    # trained model artifacts (gitignored)
  figures/                   # publication figures (committed)
  atlas/                     # Part 3 site-specific outputs
    sites/
      <site_id>/
        rgb.tif
        pile_mask.tif
        minerals/<mineral>.tif
        scores/<score>.tif
        report_card.pdf
    stac/                    # STAC catalog of atlas derivatives
  submission/                # final submission package
    README.md
    memo.pdf
    video.mp4
    hero_figure.png
```

## Parallel Work

- **Part 1 (tanager-isofit):** runs in its existing folder. Track I imports it via pip from a local editable install.
- **Part 2 (tanager_rocks core):** primary codebase in this folder.
- **Part 3 (mine-waste atlas):** lives in `./atlas/` subdir, shares the `src/` library.

## Commands

- Install: `pip install -e ~/Desktop/EDC/tanager-isofit && pip install -e .`
- Test: `pytest`
- Lint / format: `ruff check src/ && ruff format src/`
- Notebooks: `jupyter lab notebooks/`
- CLI (Part 2): `tanager-geochem predict --scene <stac-id> --output oxides.tif`
- CLI (Part 3): `tanager-minewaste atlas --site <site-id> --output atlas/sites/<site-id>/`

## Conventions (unified)

- Target oxides (Part 2): SiO2, Al2O3, Fe2O3(T), MgO, CaO, Na2O, K2O, TiO2
- Fe harmonization: convert FeO → Fe2O3(T) using factor 1.1113
- Anhydrous normalization: sum oxides to 100% before modeling
- Coordinate filtering: require ≥ 3 decimal places precision
- Band masking: remove O2 (755–770 nm), H2O (1350–1450 nm, 1800–1950 nm)
- Cross-validation: Leave-One-Scene-Out (LOSO) as primary strategy
- All data processing functions return DataFrames or xarray objects
- Use logging, not print statements
- Docstrings: NumPy style
- Reproducibility: pinned `environment.yml` at repo root

## Scientific Constraints (unified)

- Report sample attrition at each filtering step
- Bootstrap 95% CI on all R² and RMSE values (n=1000)
- Spatial autocorrelation must be addressed (no random k-fold as primary CV)
- Spectral degradation must use published spectral response functions, not simple averaging
- All mineral/score layers (Part 3) include per-pixel uncertainty
- Report pile-size detection limits honestly (30 m pixel grain)
- Honestly state when Tanager underperforms EMIT or ASTER; do not suppress negative results

## Related

Inside this Tanager workspace (relative to this repo):

- `../tanager-isofit/` — Part 1 codebase
- `../entry_optimization.md` — overall competition guidance
- `../tanager_rocks_upgrade_plan.md` — upgrade rationale
- `../idea_A_mine_waste_atlas.md` — Part 3 rationale

Assets that remain in the EDC tree (will not follow after the Tanager workspace is moved; update or copy as needed):

- `tanager_footprints.geojson` — 52 Tanager scene footprints (originally `~/Desktop/EDC/`)
- `petdb/column_mappings.json` — oxide name standardization (originally `~/Desktop/EDC/`)

## Submission Deadline

August 31, 2026, 11:59 PM PST. See timeline in `spec.md`.
