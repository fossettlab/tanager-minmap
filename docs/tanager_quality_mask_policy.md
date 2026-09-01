# Tanager quality-mask policy

Status: implemented 2026-08-09. This document records the authority,
diagnostic evidence, and fixed decision used by the principal Tanager analysis
path. It is deliberately separate from the later result-sensitivity report.

## Authority

Planet's current [Tanager Product Specification](https://docs.planet.com/data/imagery/tanager/techspec/)
defines the three embedded beta usable-data-mask fields as follows:

- `beta_cloud_mask == 1`: cloud;
- `beta_cirrus_mask == 1`: cirrus; and
- `nodata_pixels == 1`: no data.

The same specification defines `good_wavelengths == 1` as bands that accurately
represent surface reflectance and `0` as bands affected by water absorption or
other atmospheric features. It describes surface reflectance as unitless and
*typically* between 0 and 1; it does not define 0--1 as a strict validity
interval.

The public STAC item metadata independently assigns the HDF5 assets the
`cloud` and `data-mask` roles, names these same three fields, and declares 255
as their storage no-data value.

## Reproducible diagnostic

Run:

```bash
uv run python scripts/audit_tanager_quality.py
```

The machine-readable output is
`data/processed/quality_audit/scene_quality.json`. The audit reads all seven
local scenes, checks QA value domains, verifies the reflectance fill footprint,
and measures range behavior only on the 363 channels retained by the final
spectral policy.

| Scene | QA-union excluded | QA-clear pixels with any negative band | QA-clear pixels with any value >1.5 |
|---|---:|---:|---:|
| 20250911_191523_58_4001 | 42.17% | 2.04% | 10 pixels (0.0019%) |
| 20250911_191547_88_4001 | 31.99% | 2.12% | 0 |
| 20240925_185504_87_4001 | 30.82% | 28.34% | 0 |
| 20240925_185509_74_4001 | 29.68% | 33.04% | 0 |
| 20250222_190233_00_4001 | 29.75% | 1.77% | 0 |
| 20250222_190237_16_4001 | 31.37% | 1.62% | 0 |
| 20250222_190241_32_4001 | 30.35% | 0.03% | 0 |

For all seven scenes, `nodata_pixels == 1` exactly matches the `-9999`
reflectance fill footprint. Cloud and cirrus add independent exclusions that
the former fill-only pipeline did not apply. A strict all-band lower clamp at
zero would remove between 0.03% and 33.04% of otherwise QA-clear pixels in a
strongly scene-dependent way, despite Planet describing 0--1 as typical rather
than mandatory.

The product marks 58 of 426 wavelengths as bad. The project's existing fixed
O2/H2O windows remove 53 channels. Their union removes 63 channels and retains
363; ten product-bad edge channels were previously left in the analysis.

## Fixed primary policy

`tanager_rocks.quality.mask_tanager_scene` now owns the policy:

1. Exclude a spatial pixel through the complete cube if any beta QA field is
   nonzero or any reflectance value is non-finite.
2. Exclude a spectral channel if Planet sets `good_wavelengths == 0` **or** it
   falls in the pre-existing configured O2/H2O windows.
3. Do not apply a numeric reflectance clamp in the primary analysis. Negative
   retrieval estimates remain visible rather than being removed by an
   unsourced, scene-dependent rule.
4. Evaluate upper-bound exclusions at 1.0 and 1.5 only as predeclared
   sensitivity analyses. They cannot be selected after viewing headline
   results.

This decision separates product-authoritative invalidity from a later robust-
estimation question. The ten >1.5 pixels in one Bingham scene are retained in
the primary QA-only analysis and explicitly tested in sensitivity runs rather
than silently clipped.

## Preservation and comparison

The pre-correction generated products are preserved under
`data/processed/mask_sensitivity/legacy_fill_only/`. Corrected products must be
regenerated from the same raw scenes, after which a scripted before/after
report will quantify map, score, and public-claim changes.
