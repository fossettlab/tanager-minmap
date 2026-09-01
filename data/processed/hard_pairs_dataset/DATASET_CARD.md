---
license: cc-by-4.0
task_categories:
  - image-classification
  - other
tags:
  - hyperspectral
  - remote-sensing
  - mineral-mapping
  - band-reliance-probe
  - tanager
size_categories:
  - n<1K
---

# Tanager Hard-Pairs Probe: RGB-Ambiguous, SWIR-Separable Mineral Patches

**Status: local build only.** This dataset lives at
`data/processed/hard_pairs_dataset/` in the `tanager-rocks` repository, built by
`scripts/build_hard_pairs_dataset.py`. It has not been published anywhere;
publishing it (Hugging Face Hub or otherwise) is a separate, explicit
decision for the project team. The source license and required attribution are
resolved below; the present project release still excludes the chip files.

## Dataset summary

This is the mineralogical analog of the Sentinel-2 land-cover benchmark
**Similar-but-Different** (Robinson, C. & Corley, I., 2026, *Similar but
Different: A Benchmark for Measuring Whether Models Actually Use
Multispectral Bands*, https://geospatialml.com/posts/similar-but-different/).
It is built the same way: mine patches whose *visible-band* statistics carry
little information about the label, so that any model accuracy above an
RGB-only ceiling has to come from the non-visible bands. Where the blog
restricts *land-cover class* (ESA WorldCover), this dataset restricts
*dominant alteration mineral*, derived from the `tanager-rocks` pipeline's own
mixture-tuned matched filter (MTMF) product — the same one behind the
project's hero mineral map.

It is intended as a small **evaluation-only probe**: a frozen-embedding or
linear-probe eval to check whether a pretrained hyperspectral model (e.g.
TanagerFM, a separate workspace project) actually reads bands beyond
RGB when discriminating hydrothermal-alteration minerals, not a training set.
See "Intended uses" below for why. TanagerFM's own band-ablation eval design
doc (`TanagerFM/docs/band_ablation_eval_and_masking.md`, "A.7 Hard-pairs
cluster-accuracy hook") names this dataset `TANAGER_HARD_PAIRS` and expects
exactly the `clusters.csv` file described below to compute a cluster-accuracy
column.

## Dataset structure

```
hard_pairs_dataset/
├── DATASET_CARD.md          this file
├── patches.csv               one row per labeled patch (268 rows)
├── pairs.csv                  RGB-ambiguous, SWIR-separable patch pairs (18 rows)
├── clusters.csv                RGB-ambiguity-graph connected components (14 clusters, 175 member rows)
├── wavelengths.csv             per-scene band-center wavelengths (nm)
├── chips.sha256                sorted SHA-256 manifest, one entry per patch row
└── chips/
    └── <scene_id>/
        └── <patch_id>.tif     one 426-band, float32, LZW-compressed GeoTIFF per patch
```

**Format choice: CSV, not Parquet.** Every other manifest/table this project
writes (`data/intermediate/*/*.csv`, `data/processed/hard_pairs/*.csv`) is
CSV; `pyarrow` is not currently a project dependency, and adding it just for
~300-row tables would be a disproportionate new dependency for no real
benefit (Hugging Face `datasets` loads CSV natively, so this doesn't block
eventual publishing). CSV was chosen to match existing project convention
over introducing a new format.

### `patches.csv`

268 patches (11×11 px, 330 m footprint at Tanager's 30 m GSD), each carrying
a confident dominant-mineral label from the project's hero-map pipeline.
Self-contained: every field a downstream consumer needs (chip path, label,
geolocation, RGB stats) lives here, at full float precision (not the
2-decimal display rounding earlier drafts of this file used — see
"Reproducibility note" below).

| Field | Description |
|---|---|
| `patch_id` | `<site_id>_r<row>_c<col>` — stable, deterministic |
| `chip_path` | Relative path to the patch's GeoTIFF under `chips/<scene_id>/` |
| `site_id`, `scene_id` | Source site (`bingham` / `goldfield`) and Tanager scene ID |
| `row`, `col` | Patch-grid indices (non-overlapping 11×11 px tiling) |
| `y0`, `x0` | Pixel offset of the patch's top-left corner in the source scene |
| `patch_size_px`, `footprint_m` | 11 px, 330 m |
| `label` | Dominant mineral (mode of the hero map's per-pixel class over the patch; ≥70% purity) |
| `purity` | Fraction of patch pixels carrying that dominant label |
| `rgb_mean_r/g/b`, `rgb_std_r/g/b` | Patch true-color statistics in the pooled, cross-scene post-stretch uint8 space (see `METHODS.md` "Hard-pair probe"), full precision |
| `centroid_lon`, `centroid_lat` | Patch center, WGS84 |
| `crs` | Patch's native projected CRS (`EPSG:32611` Goldfield / `EPSG:32612` Bingham) |
| `n_bands` | 426 |

### `pairs.csv`

The 18 patch pairs that cleared both the RGB-ambiguity gate (bottom-decile
cross-label RGB mean/std distance) and the SWIR-separability gate (spectral
angle exceeding the 95th percentile of same-mineral patch pairs). `rank`
orders by `swir_angle_deg` descending (strongest separation first);
`figures/hard_pairs.png` in the parent repo renders the top 5.

| Field | Description |
|---|---|
| `rank` | 1 (most separable) .. 18 |
| `patch_id_a`, `label_a`, `patch_id_b`, `label_b` | The pair and their labels |
| `rgb_mean_l2`, `rgb_std_l2` | True-color distance (DN, pooled stretch) |
| `swir_angle_deg` | Spectral angle between the two patches' mean 2000–2450 nm reflectance |

### `clusters.csv` — the `TANAGER_HARD_PAIRS` cluster-accuracy hook

The blog's `test_hard_clusters.parquet` analog: "each cluster is a connected
component of the RGB mean/std similarity graph and spans at least two
[...] labels." Here, nodes are the 268 labeled patches and edges are the 415
RGB-ambiguous candidate pairs (the same bottom-decile-distance graph used to
select `pairs.csv`'s candidates, *before* the SWIR-separability filter is
applied) — re-derived deterministically from `patches.csv`'s own RGB
statistics by `tanager_rocks.pairs.rgb_ambiguity_clusters`, with no cube
reload and no re-running MTMF. Long format, one row per (cluster, patch)
membership:

| Field | Description |
|---|---|
| `cluster_id` | 0..13 |
| `patch_id` | Member patch, joins to `patches.csv` |
| `label` | That member's dominant-mineral label |
| `cluster_size` | Total members in this cluster |
| `n_labels_in_cluster` | Distinct labels spanned (always ≥2 by construction — edges are cross-label only) |

**Cluster-accuracy metric (definition only — no eval is run here; this
dataset has no model attached).** For a set of per-patch predictions, a
cluster counts as *correct* only if every one of its members is classified
correctly:

```
cluster_correct(cluster) = all(predicted_label[p] == true_label[p] for p in cluster.members)
cluster_accuracy = mean(cluster_correct(c) for c in clusters)
```

This is the blog's own metric, unchanged, applied to mineral clusters instead
of WorldCover clusters. It is strictly harder than per-patch accuracy and is
the sharpest single check of whether a model is actually using non-RGB bands
where RGB alone is ambiguous — see `band_ablation_eval_and_masking.md` A.7 for
how TanagerFM intends to consume it.

**Honest finding on cluster-size distribution.** Unlike the blog's clusters
(sizes 2–4, 53 clusters from a 30,927-patch pool), ours are bimodal: 13
clusters of size 2–7 (37 patches total), and **one giant cluster of 138
patches spanning all 8 minerals** — more than half of all labeled patches in
the dataset. This is a real, unforced consequence of transitive chaining at
this dataset's much smaller N (268 vs. 30,927): with a bottom-decile distance
threshold, near-identical RGB statistics chain patch-to-patch (A close to B,
B close to C, ...) into one dominant connected component rather than many
small isolated ones. It was not filtered or reshaped to look more like the
blog's distribution. A cluster-accuracy evaluation on cluster 1 alone is
closer in spirit to *macro-average patch accuracy on the RGB-ambiguous
subset* than to the blog's tight local-neighborhood clusters; the 13 small
clusters are the closer analog to what the blog reports. Both are provided;
a consumer should decide which is the right comparison for their claim.

### `wavelengths.csv`

Band-center wavelengths (nm) are recorded **per site**, not shared: the two
source scenes' wavelength axes differ by up to 0.22 nm (a per-scene
calibration artifact, not a processing choice), so a consumer aligning bands
across sites should use the axis matching each chip's `site_id`, not assume
band index N means an identical wavelength everywhere.

### Chips (`chips/<scene_id>/<patch_id>.tif`)

Each chip is the **raw** (uncorrected beyond Planet's L2A surface-reflectance
product; no absorption-band masking applied) 426-band Tanager reflectance
cube for that patch, float32, LZW-compressed, georeferenced (CRS + affine
transform) to its source scene. No band selection or preprocessing has been
applied — this mirrors the blog's own choice to ship raw L2A bands rather
than a preprocessed derivative, and lets a consumer choose their own masking.
The project's own O2/H2O absorption-window mask
(`tanager_spec.mask.mask_absorption_bands`,
`tanager_spec.config.ABSORPTION_MASKS_NM`) is available if wanted but not
pre-applied here. Chips are grouped in a per-`scene_id` subdirectory (two
scenes total) rather than one flat directory.

### Exact chip set and checksums

`patches.csv` is the authority for chip membership. `chips.sha256` contains
one standard, lowercase SHA-256 line for every referenced `chip_path`, sorted
lexicographically by that relative POSIX path. The checker fails closed when
it encounters a missing chip, an unreferenced file anywhere below `chips/`, a
duplicate or non-canonical path, a symlink, an unsorted or malformed checksum
manifest, or a digest mismatch. This prevents files left by an older build
from silently becoming part of a release.

The builder writes all chips and tables into a new sibling staging directory.
Before promotion it verifies the frozen release contract (268 patches, 18
pairs, 14 clusters, and 175 cluster-membership rows), confirms pair and
cluster references, writes `chips.sha256`, validates every digest and the
exact chip set, and round-trips one seeded-random patch against its source
scene for band count, CRS, and bit-exact pixel values (NaN-aware, with no
resampling tolerance). Only a fully validated staging directory is promoted
to the canonical path; same-filesystem renames preserve the prior directory
until validation finishes and restore it if promotion fails. A pre-existing
recovery backup causes the build to stop rather than guess which directory is
authoritative.

Build and check commands are:

```bash
uv run python scripts/build_hard_pairs_dataset.py
uv run python scripts/build_hard_pairs_dataset.py --check
```

The second command is read-only. A legacy pre-checksum build is not
release-ready until it has been rebuilt once and this check passes.

**Band count: 426, verified from the source HDF5's own metadata, not a
nominal figure.** The `surface_reflectance` field's `wavelengths` attribute
is a 426-element array (376.44–2499.00 nm, uniform ~5 nm spacing throughout,
including the last band -- no anomaly, gap, or duplicate at the boundary),
and Planet's own per-band `good_wavelengths` QA flag (also 426 elements,
read directly from the HDF5) marks the 426th band GOOD (value 1). Across the
full axis that QA field marks 58 bands bad; the project combines it with the
configured atmospheric windows rather than assuming the two masks coincide.
Band 426 is a real, good-quality spectral band, not a QA/mask layer folded
into the cube -- the file's ancillary QA layers (`beta_cloud_mask`,
`nodata_pixels`, `aerosol_optical_depth`, sensor/sun geometry, etc.) are
separate 2-D fields entirely outside the band-indexed `surface_reflectance`
array. **Interoperability note for TanagerFM:** TanagerFM's tokenizer is
fixed to exactly 425 input bands by design (`tanager_fm.constants.
N_BANDS_USED = 425`, chosen so 425 = 17 groups × 25 bands factors cleanly
for its spectral-grouping scheme -- see `TanagerFM/src/tanager_fm/
data/patches.py`) and deliberately drops Tanager's native last band to get
there. That is a TanagerFM-specific architectural choice, not evidence
against the instrument's true 426-band count. A consumer feeding these
426-band chips into TanagerFM's encoder must drop band index 425 (0-based,
i.e. the last band, 2499.0 nm) itself to match that fixed-425 input
contract; this dataset does not pre-drop it, since doing so would silently
bake one specific downstream model's convention into a general-purpose
probe dataset.

## Dataset creation

**Source data.** Two Planet Tanager `ortho_sr_hdf5` scenes: Goldfield
district, NV (2024-09-25, 255 patches) and Bingham Canyon / Kennecott, UT
(2025-09-11, 13 patches) — the same two lead scenes documented in the parent
repo's `METHODS.md`. Both sites are MRDS-confirmed developed mineral deposits
(Goldfield District Gold Deposits; Bingham Open Pit Mine).

**Labels.** Each patch's label is the mode of the hero map's per-pixel
dominant-mineral class (`tanager_rocks.viz.dominant_mineral_class`): an
infeasibility-gated (`max_infeas < 1.0`), per-mineral-90th-percentile-floor
mixture-tuned matched filter (MTMF) against USGS splib07a reference spectra
for eight target alteration minerals (alunite, kaolinite, dickite, jarosite,
hematite, goethite, gypsum, muscovite). **Labels are this pipeline's own
model output, not ground truth** — see "Limitations" below.

**Filtering.** Both sites' lead scenes were tiled into 7,290 (Bingham) and
7,480 (Goldfield) candidate 11×11 px patches. A patch was kept only if every
pixel was valid (no cloud, cirrus, nodata / off-nadir fill, or RGB overshoot)
and its modal
dominant-mineral class covered ≥70% of the patch (the blog's WorldCover
purity rule, used unmodified). Full discard accounting is in
`../hard_pairs/summary.json` and `METHODS.md`.

**Reproducibility note.** `patches.csv`'s RGB columns are written at full
float precision, not the 2-decimal display rounding an earlier internal
draft of the upstream `data/processed/hard_pairs/patches.csv` used. The
`clusters.csv` build re-derives the RGB-ambiguity graph from that CSV alone
and cross-checks its threshold against the cached value in
`data/processed/hard_pairs/summary.json`; 2-decimal rounding was enough to
shift that threshold by ~0.0007 (a real, measured, and now-fixed precision
loss — not a rounding tolerance that was loosened to make a check pass).

## Class distribution

| Label | Bingham | Goldfield | Total |
|---|---:|---:|---:|
| goethite | 3 | 55 | 58 |
| gypsum | 4 | 44 | 48 |
| alunite | 0 | 42 | 42 |
| muscovite | 0 | 37 | 37 |
| hematite | 3 | 33 | 36 |
| jarosite | 2 | 23 | 25 |
| dickite | 0 | 15 | 15 |
| kaolinite | 1 | 6 | 7 |
| **Total** | **13** | **255** | **268** |

The class distribution is unbalanced and reflects each site's actual
alteration mineralogy, not a designed sampling scheme — Bingham (a porphyry
system) yields far fewer confidently-labeled patches than Goldfield (an
acid-sulfate system with the assemblage this pipeline validates most
cleanly; see `METHODS.md` "Validation results").

## Intended uses

Frozen-embedding or linear/kNN probes that compare RGB-only vs. full-spectrum
(or SWIR-inclusive) accuracy at predicting the dominant-mineral label —
exactly the ablation protocol the blog runs for land cover, adapted here for
mineralogy. `pairs.csv` is the pairwise "does the model use non-visible
bands where it has to" check; `clusters.csv` is the sharper cluster-accuracy
metric (see above) — the metric the TanagerFM eval doc names as the single
strongest figure for demonstrating SWIR reliance on this project's own
mineralogy.

## Out-of-scope uses

- **Not a training set.** 268 patches from 2 source scenes is far too small
  and too scene-correlated for supervised training (the blog's dataset has
  30,927 patches from 2,709 scenes with a scene-disjoint train/val/test
  split; this dataset has no split at all, by design, and mixing patches
  from the same scene into both a "train" and "eval" role would leak
  spatially correlated context).
- **Not an operational land-cover or mineral-abundance product.** It probes
  band reliance, not absolute detection accuracy — see "Limitations."

## Limitations

- **Labels are model output, not ground truth.** They are this project's own
  MTMF classification; a "hard pair" or "hard cluster" documents where this
  pipeline's SWIR call disagrees with what true color alone suggests, not an
  independently verified mineral identity. The project's Rockwell-ASTER
  validation (`METHODS.md` "Validation results") is the ground-truth check on
  the underlying MTMF layers themselves — it agrees well for the alunite /
  sericite / Al-OH / carbonate signal at Goldfield, and much less well for
  Fe-oxide and kaolinite/dickite layers and at Bingham generally. Treat this
  dataset's accuracy as a measurement of *band reliance*, exactly the caveat
  the blog states for its own WorldCover labels — not of mineralogical
  accuracy.
- **Two source scenes only.** All 268 patches come from exactly two Tanager
  acquisitions. There is no guarantee that patterns learned or probed here
  generalize beyond these two sites' geology, sun angle, and acquisition
  conditions.
- **Site support is highly uneven.** Seventeen of 18 hard pairs are within
  Goldfield; one pairs a Bingham jarosite patch with a Goldfield hematite
  patch. Bingham contributes only 13 of 268 labeled patches, so the probe is
  primarily a Goldfield diagnostic rather than a balanced cross-site test.
- **Cluster-size distribution is bimodal, not blog-like** (one 138-member
  cluster spanning all 8 labels, plus 13 small 2-7-member clusters) — see the
  "Honest finding" note under `clusters.csv` above.
- **Unbalanced classes**, reflecting real site mineralogy (see table above),
  not a deliberate sampling design.
- **Raw, unmasked bands.** Chips include the product-bad and configured O2/H2O
  channels (their union removes 63 of 426 bands in the project analysis); a
  probe that naively averages or feeds all bands to a model should account for
  this. The project policy is implemented by
  `tanager_rocks.quality.mask_tanager_scene`.

## Licensing

**License: CC BY 4.0. Current release decision: do not publish the chip files.**
The license question is resolved, but publication remains a separate operator
decision because this is a two-scene, model-labeled evaluation probe rather
than an independently validated benchmark.

The Tanager Open Data Competition Terms & Conditions
(`Planet-TermsConditions-TanagerCompetition.pdf`, workspace root) address
Tanager-derived data in two places, neither of which names a specific license
variant:

- **"Representations and Warranties," clause (1):** Participant represents
  that "Use of Planet's Open STAC imagery will comply with the Creative
  Commons license applicable to such imagery."
- **"Licensing and Intellectual Property Rights":** "Planet retains all title
  and rights to the Tanager Open STAC data, and Participant does not receive
  any other license or ownership rights other than the license provided with
  the applicable Tanager Open STAC data."

The live Planet Open STAC catalog licenses the Tanager Core Imagery Licensed
Material under `CC-BY-4.0`. The exact Goldfield item is in the `natural-lands`
collection and its `properties.license` is `CC-BY-4.0`; the exact Bingham item
is in the `energy-mining` collection and carries the same item-level license.
The catalog root requires the following form for adapted material. Because the
two source scenes were captured in different years, both year-specific
attributions apply:

- Adapted from Tanager STAC Data, available at www.planet.com/data/stac
  © 2024 Planet Labs PBC. All Rights Reserved.
- Adapted from Tanager STAC Data, available at www.planet.com/data/stac
  © 2025 Planet Labs PBC. All Rights Reserved.

Exact source records: Planet's public
[Goldfield item](https://www.planet.com/data/stac/tanager-core-imagery/natural-lands/20240925_185504_87_4001/20240925_185504_87_4001.json),
[Bingham item](https://www.planet.com/data/stac/tanager-core-imagery/energy-mining/20250911_191523_58_4001/20250911_191523_58_4001.json), and
[Tanager catalog root](https://www.planet.com/data/stac/tanager-core-imagery/catalog.json),
verified 2026-08-11. Changes from the licensed source are explicit: the builder
crops non-overlapping 11×11-pixel windows from each `ortho_sr_hdf5` scene,
preserves the source reflectance samples without spectral resampling, writes
per-chip georeferencing and compression, and adds project-authored weak labels,
pair/cluster tables, manifests, checksums, and documentation. The resulting
chip dataset is Adapted Material distributed under CC BY 4.0 if publication is
separately approved. The `tanager-rocks` source code remains separately
licensed under MIT.

## Citation

If you use this dataset, credit the `tanager-rocks` project (Bradley Lab,
Washington University in St. Louis; use the repository's `CITATION.cff` until
a DOI is minted) and formally cite the benchmark-design paper it adapts:

```
@online{robinson2026similar,
  author = {Robinson, Caleb and Corley, Isaac},
  title = {Similar but {Different:} {A} {Benchmark} for {Measuring}
    {Whether} {Models} {Actually} {Use} {Multispectral} {Bands}},
  date = {2026-07-07},
  url = {https://geospatialml.com/posts/similar-but-different/},
  langid = {en}
}
```

Reference spectral library: Kokaly, R.F. et al. (2017), USGS Spectral
Library Version 7, USGS Data Series 1035. Validation reference for the
underlying MTMF layers: Rockwell, B.W. & Bonham, J.M. (2017), USGS data
release, doi:10.5066/F7CR5RK7.

Contact: Alex Bradley, Department of Earth, Environmental, and Planetary
Sciences, Washington University in St. Louis (abradley@wustl.edu).
