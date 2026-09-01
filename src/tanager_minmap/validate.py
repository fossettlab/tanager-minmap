"""Zone-agreement validation of mineral maps against the Rockwell ASTER reference.

spec.md pipeline step 4-5 ("validated maps"). Each continuous score map (a
diagnostic band depth or an MTMF abundance) is tested for how well it
discriminates the published alteration class(es) that contain its mineral group
(:mod:`tanager_minmap.reference` mappings) from the other classified, reliable
ground. The rank-based ROC AUC equals the Mann-Whitney U statistic divided by
the pair count, so one rank test yields both a significance value and the
separability (AUC). A Youden-J-optimal score threshold is reported per layer —
the cutoff that best separates the published zones — which is how the otherwise
distribution-informed SAM/MTMF thresholds are calibrated to an external map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.stats import mannwhitneyu

from .reference import ROCKWELL_EXCLUDED


@dataclass(frozen=True)
class Discrimination:
    """How well one score layer separates its reference alteration zone."""

    layer: str
    positive_classes: tuple[int, ...]
    n_pos: int
    n_neg: int
    auc: float  # rank AUC = P(score in zone > score outside) over the pair set
    u_statistic: float
    p_value: float
    median_pos: float
    median_neg: float
    threshold: float  # Youden-J-optimal score cutoff (calibrated detection level)
    tpr: float  # true-positive rate at ``threshold``
    fpr: float  # false-positive rate at ``threshold``
    youden_j: float  # tpr - fpr at ``threshold``


def analysis_domain(reference: xr.DataArray) -> np.ndarray:
    """Boolean mask of reference pixels usable for validation.

    Drops the excluded classes (nodata, vegetation, semi-corrupted SWIR; see
    :data:`tanager_minmap.reference.ROCKWELL_EXCLUDED`) and non-finite cells, so
    discrimination is tested only among classified, reliable ground.
    """
    vals = reference.values
    domain = np.isfinite(vals)
    for cls in ROCKWELL_EXCLUDED:
        domain &= vals != cls
    return domain


def _roc_youden(
    pos: np.ndarray, neg: np.ndarray, n_thresh: int = 512
) -> tuple[float, float, float, float]:
    """Youden-J-optimal threshold and its (TPR, FPR, J) for a one-sided detector.

    The detector flags ``score >= threshold``. Thresholds are scanned on a linear
    grid spanning both groups; TPR/FPR are read off the sorted groups by binary
    search (O(n log n)), so this scales to millions of pixels.
    """
    lo = float(min(pos.min(), neg.min()))
    hi = float(max(pos.max(), neg.max()))
    if hi <= lo:  # degenerate: all scores identical
        return lo, 0.0, 0.0, 0.0
    thr = np.linspace(lo, hi, n_thresh)
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    tpr = 1.0 - np.searchsorted(pos_sorted, thr, side="left") / pos.size
    fpr = 1.0 - np.searchsorted(neg_sorted, thr, side="left") / neg.size
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(tpr[k]), float(fpr[k]), float(j[k])


def discriminate(
    score: xr.DataArray,
    reference: xr.DataArray,
    positive_classes: frozenset[int],
    layer: str = "",
) -> Discrimination | None:
    """Test whether ``score`` separates ``positive_classes`` from other ground.

    Parameters
    ----------
    score : xr.DataArray
        Continuous score map (band depth or MTMF abundance), dims ``(y, x)``,
        aligned to ``reference``.
    reference : xr.DataArray
        Aligned categorical Rockwell class raster, dims ``(y, x)``.
    positive_classes : frozenset of int
        Rockwell class values that should score high for this layer.
    layer : str
        Layer name (recorded in the result).

    Returns
    -------
    Discrimination or None
        ``None`` if either the positive or negative group is empty after masking
        (e.g. a positive class absent from the scene).
    """
    domain = analysis_domain(reference)
    finite = np.isfinite(score.values)
    use = domain & finite
    ref_vals = reference.values[use]
    sc = score.values[use]

    is_pos = np.isin(ref_vals, list(positive_classes))
    pos = sc[is_pos]
    neg = sc[~is_pos]
    if pos.size == 0 or neg.size == 0:
        return None

    # One-sided: a correct detector scores the zone higher than the background.
    res = mannwhitneyu(pos, neg, alternative="greater", method="asymptotic")
    auc = float(res.statistic) / (pos.size * neg.size)
    thr, tpr, fpr, youden = _roc_youden(pos, neg)
    return Discrimination(
        layer=layer,
        positive_classes=tuple(sorted(positive_classes)),
        n_pos=int(pos.size),
        n_neg=int(neg.size),
        auc=auc,
        u_statistic=float(res.statistic),
        p_value=float(res.pvalue),
        median_pos=float(np.median(pos)),
        median_neg=float(np.median(neg)),
        threshold=thr,
        tpr=tpr,
        fpr=fpr,
        youden_j=youden,
    )


def validate_scores(
    scores: xr.Dataset,
    reference: xr.DataArray,
    mapping: dict[str, frozenset[int]],
) -> dict[str, Discrimination]:
    """Run :func:`discriminate` for every score layer that has a class mapping.

    Layers absent from ``mapping`` (e.g. gypsum, which has no Rockwell class) are
    skipped; layers whose positive class is absent from the scene return no
    result. The returned dict holds only layers that produced a discrimination.
    """
    out: dict[str, Discrimination] = {}
    for layer, classes in mapping.items():
        if layer not in scores.data_vars:
            continue
        result = discriminate(scores[layer], reference, classes, layer=layer)
        if result is not None:
            out[layer] = result
    return out
