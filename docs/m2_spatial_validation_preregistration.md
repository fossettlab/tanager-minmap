# M2 preregistration: spatial validation and repeatability

Status: frozen before blocked-validation or repeatability results, 2026-08-09.
The earlier pixelwise results were already known and motivated this protocol;
they are not treated as confirmatory evidence. This document fixes the new
estimands, scene roles, spatial units, thresholds, uncertainty, and decision
rules before those analyses are run.

## Scientific scope

The Rockwell ASTER product is external alteration-zone context, not
mineral-level truth. EMIT is an independent instrument/acquisition but shares
this project's algorithm and library. All M2 claims therefore use
"alteration-zone discrimination," "cross-sensor consistency," or
"repeatability" rather than field validation or absolute mineral accuracy.

## Frozen scenes

Calibration anchors are the already-declared lead scenes:

- Goldfield: `20240925_185504_87_4001`;
- Bingham: `20250911_191523_58_4001`.

Held-out repeats are:

- Goldfield: `20240925_185509_74_4001`, `20250222_190233_00_4001`,
  `20250222_190237_16_4001`, and `20250222_190241_32_4001`;
- Bingham: `20250911_191547_88_4001`.

No anchor may be replaced and no repeat may be omitted after its result is
viewed. Same-date adjacent strips are not described as independent temporal
replicates.

## Endpoints

The primary external-reference endpoint is Goldfield `al_oh_doublet` rank AUC
against Rockwell classes `{3, 4, 5, 10, 11, 12, 16}`. Its estimand is the
probability that a randomly drawn compatible alteration-zone pixel has a
higher continuous score than a randomly drawn other classified-ground pixel.

The primary thresholded endpoint is cross-fitted balanced accuracy, using a
threshold learned without the held-out spatial block.

Key secondary endpoints are Goldfield `gypsum_carbonate`, alunite MTMF, and
muscovite MTMF, plus Bingham `gypsum_carbonate`. All other mapped layers are
exploratory and remain reported regardless of direction. Jarosite is
descriptive only because the corrected reference overlap has only four
Goldfield and two Bingham positive pixels.

For every eligible layer report rank AUC, balanced accuracy, positive- and
negative-class F1, macro-F1, TPR, FPR, threshold distribution, prevalence,
and positive/negative block counts. Existing asymptotic pixelwise p-values are
not inferential evidence.

## Spatial-correlation diagnostic and block geometry

Block size is selected without using AUC, accuracy, p-values, or visual map
quality.

1. For every prespecified continuous score and eligible binary Rockwell
   indicator, estimate an omnidirectional empirical semivariogram on the
   anchor grid. At each declared lag, pool horizontal, vertical, and the two
   diagonal directions; use only finite pairs and deterministically thin to at
   most 200,000 pairs per field/lag.
2. Lags in pixels are
   `{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}`, limited to one
   quarter of the shorter raster dimension. Convert using the grid's actual
   projected pixel size.
3. Fit the fixed exponential nugget-plus-sill model
   `gamma(h) = nugget + sill * (1 - exp(-h / a))` by bounded least squares.
   The practical range is `-a * log(0.05)`, where the fitted variogram reaches
   95% of its partial sill.
4. If the fit is non-finite, its sill is non-positive, or its fitted practical
   range lies beyond the largest evaluated lag, treat the range as
   unidentifiable on the observed scene support. Use the first empirical lag
   reaching 95% of the field variance. If no lag reaches it, use the largest
   evaluated lag and flag the fallback. The beyond-support condition was added
   on 2026-08-09 after the first geometry-only run produced a finite but
   unobserved 2,139-km extrapolation and zero Goldfield blocks; no endpoint
   metric had been computed or inspected.
5. Let `r_site` be the maximum practical range across eligible fields. The
   primary square-block side is `L = 2 * r_site`, rounded upward to whole
   pixels. The grid is anchored at the calibration raster's upper-left
   projected coordinate and reused across acquisitions.
6. Exclude a halo of width `r_site` around each held-out block from threshold
   training. Use complete **geometric** blocks for confirmatory inference and
   report edge exclusions. A block is complete when its full fixed rectangular
   footprint lies on the anchor grid; partial QA or finite-data coverage does
   not make that geometric block incomplete. Metric-specific calculations use
   the available pixels within each complete block.
7. Run the prespecified conservative sensitivity at side `2L`. Never choose
   between `L` and `2L` from their performance.

An endpoint requires at least ten positive-bearing and ten negative-bearing
blocks for confirmatory inference. Five to nine positive-bearing blocks permit
exploratory estimates only; fewer than five permit counts/maps only. Block
sizes are never reduced to rescue an endpoint.

## Cross-fitted thresholds

Use leave-one-block-out spatial cross-fitting. For each test block:

1. exclude that block and its spatial halo;
2. calculate mean block TPR and mean block FPR across the remaining training
   blocks for every unique training score;
3. maximize block-balanced Youden J (`mean TPR - mean FPR`);
4. break threshold ties by selecting the highest threshold; and
5. apply that threshold once to the held-out block.

Concatenate held-out predictions for the point estimate while retaining block
identity for uncertainty. Fixed endmembers, quality policy, feature
definitions, ridge, and mappings are not retuned. The operational primary
analysis may estimate label-free MTMF covariance from the full scene because
that matches deployment; a mandatory strict-inductive sensitivity re-estimates
covariance without each held-out block and halo. These are labeled as distinct
estimands.

## Spatial uncertainty

- Use 10,000 paired spatial-block bootstrap replicates with `SEED=42` and
  percentile 95% intervals.
- Use 9,999 whole-block permutations for null tests, permuting score-block
  identities relative to reference blocks and repeating threshold calibration.
- The singular Goldfield Al-OH primary endpoint is unadjusted.
- Apply Benjamini-Hochberg FDR at 0.05 separately to feature and MTMF
  secondary families.
- Always report effects and intervals, including failed or reversed effects.

## All-seven-scene repeatability

Run every scene independently with the frozen quality policy, endmembers,
ridge, feature definitions, and mappings. For each repeat, reproject continuous
scores to the anchor grid bilinearly and categorical masks by nearest
neighbor. Use only overlapping pixels passing QA in both acquisitions; do not
optimize registration against score agreement.

Primary comparisons are the four Goldfield anchor-to-repeat pairs and one
Bingham anchor-to-repeat pair. The other six Goldfield same-site pairs are
secondary. For each layer and pair report:

- Spearman rank correlation of continuous scores;
- IoU and Dice after transferring the anchor's externally calibrated raw
  threshold without retuning;
- IoU and Dice for each scene's upper-decile rank mask, labeled as
  rank-relative rather than calibrated;
- detection-prevalence ratio;
- symmetric 95th-percentile boundary distance in metres; and
- repeat-scene Rockwell AUC, balanced accuracy, and macro-F1 where reference
  coverage is usable, using the anchor threshold unchanged.

The transferred threshold is fit once per site and layer on **all usable
primary-`L` complete anchor blocks** with the same block-balanced Youden rule
and highest-threshold tie break used inside cross-fitting. It is not the legacy
pixelwise threshold and not the median of fold thresholds. A transfer
threshold is available only when the anchor endpoint has at least five
positive-bearing and five negative-bearing blocks; otherwise thresholded
repeatability for that layer is unavailable. The threshold artifact records
the spatial-protocol hash, block-manifest hash, source-raster hashes, support
counts, and fitted value.

The block handoff must match the current protocol hash, anchor scene, block
raster hash, shape, CRS, and transform exactly. Block IDs are never reprojected
or rebuilt in the repeatability stage. Uncertainty resamples complete paired
geometric blocks, retaining block-shaped arrays and explicit missing cells.
Metric functions apply pairwise-finite filtering only after each bootstrap or
block permutation. A complete block with at least one observed joint pair is
retained even when other cells fail QA. Spatial nulls permute one date's
complete block identities relative to the anchor while preserving within-block
coordinates and missingness. Enumerate all unique block permutations when the
factorial block count is at most 9,999; otherwise use 9,999 seeded random
permutations.

For every resampled metric, report total and finite replicate counts. Empty
versus empty thresholded maps have undefined IoU and Dice rather than perfect
agreement. A bootstrap interval or null quantile is gate-eligible only when at
least 95% of its scheduled replicates are finite; otherwise that component is
unavailable rather than conditioned silently on successful replicates. A
fixed registration sensitivity evaluates the unshifted grid and all eight
±1-pixel neighboring shifts and reports the full range, never the best shift;
only the unshifted result enters a decision gate.

## Decision rules

The external-reference gate passes only if Goldfield Al-OH has enough
independent blocks, its 95% block-bootstrap AUC interval lies wholly above
0.5, its cross-fitted balanced-accuracy interval lies wholly above 0.5, and
the direction remains positive at `2L`. If AUC passes but balanced accuracy
does not, the permitted claim is ranking discrimination, not a calibrated
detector.

For the Goldfield primary `al_oh_doublet` layer, one anchor-to-repeat
comparison passes only when all three conditions hold on the unshifted grid:

1. the paired-block bootstrap lower 95% bound for Spearman correlation exceeds
   zero;
2. the repeat-scene Rockwell balanced-accuracy block-bootstrap lower 95% bound
   at the unchanged transferred threshold exceeds `0.5`, with at least ten
   positive-bearing and ten negative-bearing blocks; and
3. the observed transferred-threshold IoU exceeds the 95th percentile of its
   whole-block spatial null.

Goldfield repeatability is then:

- **strong** if all four comparisons pass the full rule above;
- **date-dependent** if one to three comparisons pass;
- **unsupported** if all four comparisons are evaluable and none pass; or
- **unavailable** if none pass and one or more required comparison components
  is unavailable.

Bingham supplies one pair-level result only. A public "validated and
repeatable" statement requires both external-reference and repeatability
gates. Stable-but-inaccurate is called stable; accurate-in-one-acquisition is
called acquisition-specific.

## Stop and rescue rules

Large correlation ranges, too few positive blocks, ontology mismatch,
scene-dependent MTMF scale, coverage loss, registration sensitivity, or a
strict-inductive failure are reportable outcomes. None authorizes block
shrinking, remapping classes, dropping scenes, selecting a favorable shift, or
retuning thresholds after results.

## Pre-result implementation amendment (2026-08-09)

The geometric-completeness, missing-value, transfer-threshold, finite-replicate,
and explicit `unavailable` rules above were added after fresh-context code
review but before any repeatability endpoint was run or inspected. They repair
an implementation draft that would have discarded an entire block whenever a
single pixel failed QA and would have reused a legacy pixelwise threshold.
They do not change the frozen scenes, endpoint directions, block geometry,
number of resamples, or public decision thresholds.
