# Project: Tanager Rocks

## Overview
Research pilot pairing Tanager-1 hyperspectral satellite data (426 bands, 380–2500 nm, 30 m pixels) with GEOROC/PetDB geochemistry databases to train ML models predicting major-element oxides from spectral data. Core question: does hyperspectral resolution meaningfully improve geochemistry predictions over multispectral sensors (ASTER, Sentinel-2)?

## Stack
- Python 3.11+
- Core: numpy, pandas, geopandas, xarray, rioxarray
- ML: scikit-learn, xgboost, shap
- Geospatial: pystac-client, stackstac, shapely, rasterio
- Viz: matplotlib, seaborn
- Package management: pip (requirements.txt)

## Commands
- Install: `pip install -r requirements.txt`
- Test: `pytest`
- Lint: `ruff check src/`
- Format: `ruff format src/`
- Notebooks: `jupyter lab notebooks/`

## Directory Layout
```
notebooks/   — 6 ordered research notebooks (00–05)
src/         — reusable modules (stac, geochem, spectra, degrade, cv, viz)
data/raw/    — downloaded Tanager scenes (gitignored)
data/georoc/ — GEOROC parquet files (gitignored)
data/petdb/  — PetDB parquet files (gitignored)
data/intermediate/ — processed outputs (gitignored)
data/srf/    — spectral response functions (committed)
models/      — trained model artifacts (gitignored)
figures/     — publication figures (committed)
```

## Related EDC Infrastructure
- `~/Desktop/EDC/tanager_footprints.geojson` — 49 scene footprints with polygons/dates
- `~/Desktop/EDC/tanager-isofit/` — atmospheric correction pipeline (optional; use only if STAC surface reflectance unavailable)
- `~/Desktop/EDC/petdb/column_mappings.json` — oxide name standardization mappings

## Conventions
- Target oxides: SiO2, Al2O3, Fe2O3(T), MgO, CaO, Na2O, K2O, TiO2
- Fe harmonization: convert FeO → Fe2O3(T) using factor 1.1113
- Anhydrous normalization: sum oxides to 100% before modeling
- Coordinate filtering: require >= 3 decimal places precision
- Band masking: remove atmospheric water vapor (~1350–1450 nm, ~1800–1950 nm) and O2 (~760 nm) bands
- Cross-validation: Leave-One-Scene-Out (LOSO) as primary strategy
- All data processing functions return DataFrames or xarray objects
- Use logging, not print statements
- Docstrings: NumPy style

## Scientific Constraints
- Report sample attrition at each filtering step
- Bootstrap confidence intervals on all R² and RMSE values
- Spatial autocorrelation must be addressed (no random k-fold as primary CV)
- Spectral degradation must use published spectral response functions, not simple averaging
