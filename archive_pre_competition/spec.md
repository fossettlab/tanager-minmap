# Tanager Rocks: Hyperspectral Geochemistry Prediction

## 1. Scientific Objective

**Primary question:** Does hyperspectral resolution (426 bands, 380–2500 nm) meaningfully improve predictions of major-element oxide geochemistry compared to multispectral sensors (ASTER, Sentinel-2, Landsat)?

### Cascading Questions

1. **Feasibility:** Can Tanager-1 spectra predict major-element oxides (SiO2, Al2O3, Fe2O3(T), MgO, CaO, Na2O, K2O, TiO2) at geochemistry sample locations?
2. **Resolution advantage:** Do 426 bands outperform 9 (ASTER), 13 (Sentinel-2), or 7 (Landsat) for any or all oxides?
3. **Band importance:** Which spectral regions drive predictions? Do they correspond to known mineral absorption features?
4. **Generalization:** Do models trained on one Tanager scene transfer to others (Leave-One-Scene-Out performance)?

---

## 2. Data Sources

### 2.1 Tanager-1 Hyperspectral Imagery
- **Sensor:** 426 contiguous bands, 380–2500 nm, ~5 nm sampling interval
- **Spatial resolution:** 30 m ground sample distance
- **Access:** STAC catalog (endpoint TBD; Planetary Computer or direct Planet API)
- **Footprints:** 49 scenes available in `~/Desktop/EDC/tanager_footprints.geojson`
- **Product levels:** L1B (radiance), L2A (surface reflectance, if available)

### 2.2 GEOROC Geochemistry Database
- **Format:** Parquet files in `data/georoc/`
- **Content:** Major/minor/trace element whole-rock analyses with coordinates
- **Source:** GEOROC (https://georoc.eu/)

### 2.3 PetDB Geochemistry Database
- **Format:** Parquet files in `data/petdb/`
- **Content:** Complementary whole-rock analyses
- **Column mappings:** `~/Desktop/EDC/petdb/column_mappings.json`

### 2.4 Spectral Response Functions
- **ASTER:** 9 VNIR/SWIR bands
- **Sentinel-2:** 13 bands (10–60 m native)
- **Landsat-8/9:** 7 OLI bands
- **Source:** Published SRFs from instrument teams, stored in `data/srf/`

---

## 3. Tier 1 Geographic Targets

Priority scenes selected for high geochemistry sample density within Tanager footprints:

| Region | Approx. Center | Rationale |
|--------|---------------|-----------|
| Cuprite, NV | 37.5°N, 117.2°W | USGS mineral mapping benchmark, well-characterized |
| Hiller Mtns, Antarctica | TBD | Exposed lithology, low vegetation |
| Oman ophiolite | 23.0°N, 57.5°E | Ultramafic/mafic, GEOROC-dense |
| Iceland | 64.5°N, 19.0°W | Basaltic, high sample density |
| Hawaii | 19.5°N, 155.5°W | Active volcanism, fresh surfaces |

*Exact target selection is a deliverable of notebook 00.*

---

## 4. Notebook Pipeline

### 00 — Scene Selection
**Goal:** Query STAC catalog, overlay with geochemistry databases, prioritize scenes.

- **Inputs:** Tanager STAC catalog, footprints GeoJSON, GEOROC/PetDB parquet
- **Outputs:** Ranked scene list with sample counts; candidate scene map
- **Key steps:**
  1. Load footprints from GeoJSON
  2. Load geochemistry coordinates
  3. Spatial join: count samples per scene
  4. Rank scenes by sample density and lithologic diversity
  5. Produce selection map figure

### 01 — Geochemistry Prep
**Goal:** Load, clean, harmonize, and normalize geochemistry data.

- **Inputs:** GEOROC/PetDB parquet files
- **Outputs:** Cleaned DataFrame with standardized oxides, coordinate-filtered
- **Key steps:**
  1. Load parquet files, apply column mappings
  2. Harmonize iron (FeO → Fe2O3(T))
  3. Filter coordinate precision (≥3 decimal places)
  4. Anhydrous normalization (sum to 100%)
  5. Report attrition at each step
  6. Save intermediate parquet

### 02 — Spectral Extraction
**Goal:** Download Tanager scenes and extract 426-band spectra at sample locations.

- **Inputs:** Selected scenes from 00, cleaned geochem from 01
- **Outputs:** Matched spectra–geochem DataFrame
- **Key steps:**
  1. Download scenes via STAC (surface reflectance preferred)
  2. Apply atmospheric correction if needed (three options below)
  3. Extract pixel spectra at sample coordinates
  4. Mask atmospheric absorption bands (O2, H2O)
  5. QC: remove saturated/cloudy pixels
  6. Save matched dataset

**Atmospheric correction options:**
1. **STAC L2A** — use surface reflectance directly if available
2. **tanager-isofit** — run `~/Desktop/EDC/tanager-isofit/` pipeline on L1B data
3. **TOA radiance fallback** — use L1B radiance with empirical line calibration

### 03 — Modeling
**Goal:** Train per-oxide regression models, evaluate with spatial CV.

- **Inputs:** Matched spectra–geochem DataFrame
- **Outputs:** Per-oxide R², RMSE (with CI); SHAP values; trained models
- **Key steps:**
  1. Split with Leave-One-Scene-Out (LOSO)
  2. Baseline models: PLS regression, ElasticNet
  3. Primary model: XGBoost per oxide
  4. Optional: PCA dimensionality reduction
  5. SHAP feature importance per oxide
  6. Bootstrap confidence intervals
  7. Comparison: LOSO vs ordinary k-fold (to quantify spatial autocorrelation bias)

### 04 — Spectral Degradation
**Goal:** Simulate multispectral sensors, retrain, and compare performance.

- **Inputs:** Tanager spectra, SRF files, trained models
- **Outputs:** Per-sensor, per-oxide R²/RMSE comparison table and figure
- **Key steps:**
  1. Load published SRFs (ASTER, S2, Landsat-8/9)
  2. Convolve 426-band spectra to each sensor's bandpasses
  3. Retrain XGBoost on degraded spectra (same CV splits)
  4. Compare R²/RMSE: Tanager vs ASTER vs S2 vs Landsat
  5. Statistical tests for significance of improvement

### 05 — Interpretation
**Goal:** Relate SHAP importances to known mineral physics; produce publication figures.

- **Inputs:** SHAP values, wavelength axis, mineral absorption tables
- **Outputs:** Publication-quality figures and interpretive summary
- **Key steps:**
  1. SHAP values overlaid on wavelength axis
  2. Annotate with mineral absorption features (e.g., Fe²⁺ ~1000 nm, Al-OH ~2200 nm)
  3. Identify which bands drive hyperspectral advantage
  4. Generate all publication figures

---

## 5. Data Flow Diagram

```
tanager_footprints.geojson ──┐
GEOROC/PetDB parquet ────────┤
                             ▼
                   [00 Scene Selection]
                             │
                    scene_list.csv
                             │
                             ▼
GEOROC/PetDB parquet ──► [01 Geochem Prep] ──► geochem_clean.parquet
                             │
                             ▼
Tanager STAC ───────────► [02 Spectral Extraction]
(+ optional isofit)          │
                    matched_spectra_geochem.parquet
                             │
                    ┌────────┼────────────┐
                    ▼        ▼            ▼
            [03 Modeling]  [04 Degrade]  [05 Interp]
                    │        │            │
              models/    comparison    figures/
              SHAP       tables
```

---

## 6. Implementation Details

### 6.1 Coordinate Filtering
- Count decimal places in lat/lon string representation
- Require both lat and lon ≥ 3 decimal places (~111 m precision)
- Log attrition: `N_before → N_after (N_removed, X%)`

### 6.2 Atmospheric Correction
Three-tier approach (choose per scene):
1. L2A surface reflectance from STAC (preferred, zero effort)
2. tanager-isofit (accurate but computationally expensive)
3. TOA radiance with empirical line calibration (least preferred)

### 6.3 Band Masking
Remove bands in atmospheric absorption regions before modeling:
- O2: 755–770 nm
- H2O: 1350–1450 nm
- H2O: 1800–1950 nm

Approximately 30–40 of 426 bands removed. Log exact count.

### 6.4 Cross-Validation Strategy
- **Primary:** Leave-One-Scene-Out (LOSO) — each scene is one fold
- **Comparison:** Ordinary k-fold (k=5) to quantify spatial autocorrelation bias
- **Spatial block CV** as additional validation where multiple scenes overlap geographically
- All metrics reported with bootstrap 95% CI (n=1000)

### 6.5 Modeling Approach
- **Small N strategy:** Expect 50–500 matched samples depending on scene selection
- **Baselines:** PLS regression (handles collinear bands), ElasticNet
- **Primary:** XGBoost with early stopping, per-oxide models
- **Dimensionality reduction:** Optional PCA to 10–50 components if N < 100
- **Feature importance:** SHAP TreeExplainer for XGBoost

### 6.6 Spectral Degradation
- Use published spectral response functions (not simple band averaging)
- Convolve: `simulated_band = Σ(reflectance × SRF) / Σ(SRF)` per band
- Retrain models with identical CV splits for fair comparison
- Compare: Tanager (426 bands) vs ASTER (9) vs S2 (13) vs Landsat (7)

---

## 7. Expected Findings

1. Tanager-1 spectra can predict SiO2 and Fe2O3(T) with R² > 0.5 at well-sampled sites
2. Hyperspectral advantage is greatest for oxides with narrow absorption features (Fe, Al-OH, Mg-OH)
3. SiO2 (broad spectral influence) shows modest improvement over multispectral
4. SHAP importances cluster at known mineral absorption wavelengths
5. LOSO R² is significantly lower than k-fold R², demonstrating spatial autocorrelation

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Too few matched samples (< 50) | Medium | High | Expand to Tier 2 scenes; use PLS/ElasticNet |
| Tanager L2A unavailable from STAC | Medium | Medium | Fall back to isofit or TOA radiance |
| Atmospheric correction artifacts | Low | High | Compare results across correction methods |
| Spatial autocorrelation inflates metrics | High | Medium | LOSO as primary CV; report both LOSO and k-fold |
| SRF files unavailable for sensors | Low | Low | Use published filter curves from literature |
| Vegetation/soil mixing at 30 m pixels | High | Medium | Target exposed rock sites (arid, alpine) |
| GEOROC/PetDB coordinate errors | Medium | Medium | Filter precision + visual QC on maps |
