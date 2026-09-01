"""Hard-pair mining: RGB-ambiguous, SWIR-separable patch pairs.

The mineralogical analog of the "Similar-but-Different" Sentinel-2 benchmark
(Robinson & Corley 2026): patches that are near-identical in true-color
statistics but carry different dominant-mineral labels and pull apart in the
SWIR. Where the blog restricts *land-cover class* using WorldCover, this
project restricts *dominant alteration mineral* using the same infeasibility-
gated MTMF product that drives the hero map (:func:`tanager_minmap.viz.
dominant_mineral_class`) -- so a patch pair is "hard" exactly where our own
published mineral map disagrees with what a look at the true-color chips alone
would suggest.

Every threshold below is derived from this dataset's own empirical
distributions (never borrowed from the blog's Sentinel-2 DN units), per the
project's data-integrity convention; see METHODS.md "Hard-pair probe" for the
full derivation writeup.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

import numpy as np

from .speclib import pairwise_spectral_angle

# Blog step 1 tiles 32x32 px at Sentinel-2's 10 m GSD (320 m footprint). At
# Tanager's 30 m GSD, 320 m / 30 m = 10.67 px; 11 px (330 m, +3.1%) is the
# closer integer than 10 px (300 m, -6.25%).
PATCH_SIZE_PX = 11

# Blog step 3's WorldCover rule: a patch counts as one class only if that
# class covers >=70% of the patch footprint. Used as the starting point here;
# scripts/find_hard_pairs.py logs the achieved purity distribution so a lower,
# justified floor can be substituted if 0.70 is infeasible for this project's
# covariance-matched-filter labels (sparser and noisier than a per-pixel
# land-cover product).
PURITY_FLOOR = 0.70

# SWIR alteration-diagnostic window (nm). Brackets the three fixed absorption
# centers this project already maps (config.DIAGNOSTIC_NM: Al-OH 2200 nm,
# jarosite 2265 nm, gypsum/carbonate 2340 nm) with margin on both sides, and
# matches the band-ablation finding (METHODS.md "Band ablation") that
# Sentinel-2's single broad SWIR band B12 spans ~2100-2280 nm -- exactly the
# doublet-collapse this window is built to probe with Tanager's full
# resolution instead.
SWIR_WINDOW_NM = (2000.0, 2450.0)

# Blog step 2's recipe: keep pairs in the bottom decile of cross-class RGB
# distance. Blog step 4's cluster construction is a nearest-neighbor
# threshold; here the SWIR bar is instead calibrated against THIS project's
# own same-mineral spectral-angle spread (see swir_separable_pairs).
RGB_CANDIDATE_QUANTILE = 0.10
SWIR_NULL_QUANTILE = 0.95


@dataclass(frozen=True)
class Patch:
    """One labeled, non-overlapping tile from a site's lead scene.

    ``rgb_mean`` / ``rgb_std`` are in the shared post-stretch uint8 space
    (see :func:`stretch_to_uint8`); ``swir_mean`` is the patch-mean raw
    reflectance restricted to :data:`SWIR_WINDOW_NM`.
    """

    site_id: str
    scene_id: str
    row: int  # patch-grid row index
    col: int  # patch-grid col index
    y0: int  # pixel row offset of the patch's top-left corner
    x0: int  # pixel col offset of the patch's top-left corner
    label: str  # dominant mineral name
    purity: float  # fraction of patch pixels carrying that dominant label
    rgb_mean: np.ndarray  # (3,)
    rgb_std: np.ndarray  # (3,)
    swir_mean: np.ndarray  # (n_window_bands,)


@dataclass(frozen=True)
class HardPair:
    """A patch pair that passed both the RGB-ambiguity and SWIR-separability gates."""

    a: Patch
    b: Patch
    rgb_mean_l2: float
    rgb_std_l2: float
    swir_angle_deg: float


@dataclass(frozen=True)
class RgbAmbiguityResult:
    """Cross-label candidate pairs plus the empirical thresholds that selected them."""

    candidates: list[tuple[int, int, float, float]]  # (i, j, mean_l2, std_l2)
    mean_threshold: float
    std_threshold: float
    cross_mean_distances: np.ndarray
    cross_std_distances: np.ndarray


@dataclass(frozen=True)
class SwirSeparabilityResult:
    """Final hard pairs plus the same-label null distribution that calibrated the bar."""

    pairs: list[HardPair]
    threshold_deg: float
    same_label_angles_deg: np.ndarray


def pooled_rgb_percentiles(
    channel_stacks: list[tuple[np.ndarray, np.ndarray]],
    pct: tuple[float, float] = (2.0, 98.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel low/high percentile bounds pooled across multiple scenes.

    Extends :func:`tanager_minmap.figures.rgb_context`'s single-scene 2nd-98th
    percentile stretch to a stretch SHARED across scenes, so post-stretch
    uint8 RGB values sit on one absolute scale and cross-scene distances are
    meaningful (a deliberate divergence from ``rgb_context``'s per-scene
    stretch; see METHODS.md).

    Parameters
    ----------
    channel_stacks : list of (rgb_raw, invalid)
        Per scene: ``rgb_raw`` is ``(3, ny, nx)`` raw reflectance at the
        nearest bands to :data:`tanager_minmap.figures.RGB_NM`; ``invalid`` is
        the matching ``(ny, nx)`` boolean invalid-pixel mask.
    pct : tuple of float
        Lower/upper percentiles.

    Returns
    -------
    lo, hi : np.ndarray
        Per-channel bounds, shape ``(3,)``.
    """
    lo_p, hi_p = pct
    lo = np.zeros(3)
    hi = np.zeros(3)
    for c in range(3):
        pooled = np.concatenate([rgb[c][~invalid] for rgb, invalid in channel_stacks])
        lo[c], hi[c] = np.percentile(pooled, [lo_p, hi_p])
    return lo, hi


def stretch_to_uint8(
    rgb_raw: np.ndarray, invalid: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Apply a fixed per-channel stretch: ``(3, ny, nx)`` reflectance -> ``(ny, nx, 3)`` uint8.

    Same clip-and-scale convention as :func:`tanager_minmap.figures.rgb_context`
    (there, per-scene percentiles; here, a shared ``lo``/``hi`` computed by
    :func:`pooled_rgb_percentiles`). Invalid pixels render white (255),
    matching ``rgb_context``'s white background for off-scene/invalid pixels.
    """
    ny, nx = rgb_raw.shape[1:]
    out = np.zeros((ny, nx, 3), dtype=np.uint8)
    for c in range(3):
        stretched = np.clip((rgb_raw[c] - lo[c]) / (hi[c] - lo[c] + 1e-9), 0.0, 1.0)
        # NaN (off-nadir fill) survives the clip; zero it before the uint8 cast
        # to avoid an "invalid value in cast" warning. These pixels are
        # overwritten to white below regardless (they are always `invalid`).
        stretched = np.nan_to_num(stretched, nan=0.0)
        out[..., c] = np.round(stretched * 255).astype(np.uint8)
    out[invalid] = 255
    return out


def continuum_removed(wl: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Linear 2-point continuum-removed reflectance, for DISPLAY only.

    Same convention as :func:`tanager_minmap.features.band_depth` (Clark &
    Roush 1984), generalised from a single absorption's shoulders to a
    display window's own endpoints as the continuum anchors:
    ``reflectance / continuum``, which dips below 1 wherever an absorption
    pulls reflectance below the straight endpoint-to-endpoint line. The
    SWIR-separability DECISION in :func:`swir_separable_pairs` uses the raw
    reflectance spectral angle, not this transform -- this is purely for
    making the absorptions visible in a figure.

    Parameters
    ----------
    wl : np.ndarray
        Band-center wavelengths (nm), strictly increasing.
    spectrum : np.ndarray
        Reflectance aligned to ``wl``.

    Returns
    -------
    np.ndarray
        Continuum-removed reflectance, same shape as ``spectrum``.
    """
    wl = np.asarray(wl, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    continuum = spectrum[0] + (spectrum[-1] - spectrum[0]) * (wl - wl[0]) / (wl[-1] - wl[0])
    return spectrum / continuum


def tile_and_label(
    dominant_code: np.ndarray,
    minerals: list[str],
    invalid_mask: np.ndarray,
    rgb_uint8: np.ndarray,
    swir_cube: np.ndarray,
    *,
    site_id: str,
    scene_id: str,
    patch_size: int = PATCH_SIZE_PX,
    purity_floor: float = PURITY_FLOOR,
) -> tuple[list[Patch], dict[str, int]]:
    """Tile a scene into non-overlapping patches, label, and filter.

    A patch is discarded (in order) if: any pixel is invalid (zero-tolerance,
    mirroring the blog's "discard windows with any cloud/shadow pixel" rule);
    its modal dominant-mineral class is "no detection" (-1); or its modal
    class's purity is below ``purity_floor``. Ties in the pixel-count mode are
    broken toward the lower class code (so a -1/mineral tie discards).

    Parameters
    ----------
    dominant_code : np.ndarray
        ``(ny, nx)`` int array from :func:`tanager_minmap.viz.
        dominant_mineral_class`; ``-1`` = no mineral clears its detection
        floor.
    minerals : list of str
        Class-code -> mineral name, in the order ``dominant_code`` uses.
    invalid_mask : np.ndarray
        ``(ny, nx)`` bool; ``True`` = invalid (QA, off-scene, or RGB overshoot).
    rgb_uint8 : np.ndarray
        ``(ny, nx, 3)`` post-stretch true-color image (see
        :func:`stretch_to_uint8`).
    swir_cube : np.ndarray
        ``(n_window_bands, ny, nx)`` raw reflectance restricted to
        :data:`SWIR_WINDOW_NM`.
    site_id, scene_id : str
        Recorded on every surviving :class:`Patch`.
    patch_size : int
        Patch side length in pixels.
    purity_floor : float
        Minimum modal-class fraction to accept a label.

    Returns
    -------
    patches : list of Patch
        Surviving labeled patches.
    counts : dict
        ``{"total", "invalid", "no_detection", "low_purity", "labeled"}``.
    """
    ny, nx = dominant_code.shape
    n_rows, n_cols = ny // patch_size, nx // patch_size
    counts = {"total": 0, "invalid": 0, "no_detection": 0, "low_purity": 0, "labeled": 0}
    patches: list[Patch] = []

    for row in range(n_rows):
        for col in range(n_cols):
            y0, x0 = row * patch_size, col * patch_size
            ys, xs = slice(y0, y0 + patch_size), slice(x0, x0 + patch_size)
            counts["total"] += 1

            if invalid_mask[ys, xs].any():
                counts["invalid"] += 1
                continue

            block = dominant_code[ys, xs].ravel()
            vals, freq = np.unique(block, return_counts=True)
            mode_val = int(vals[np.argmax(freq)])
            purity = float(freq.max() / block.size)

            if mode_val < 0:
                counts["no_detection"] += 1
                continue
            if purity < purity_floor:
                counts["low_purity"] += 1
                continue

            counts["labeled"] += 1
            rgb_block = rgb_uint8[ys, xs].reshape(-1, 3).astype(float)
            swir_block = swir_cube[:, ys, xs].reshape(swir_cube.shape[0], -1)
            patches.append(
                Patch(
                    site_id=site_id,
                    scene_id=scene_id,
                    row=row,
                    col=col,
                    y0=y0,
                    x0=x0,
                    label=minerals[mode_val],
                    purity=purity,
                    rgb_mean=rgb_block.mean(axis=0),
                    rgb_std=rgb_block.std(axis=0),
                    swir_mean=np.nanmean(swir_block, axis=1),
                )
            )
    return patches, counts


def write_chip_geotiff(cube, y0: int, x0: int, size: int, out_path: str | Path) -> None:
    """Write one patch as a full-band GeoTIFF chip.

    ``cube`` (an ``xr.DataArray`` with ``("band", "y", "x")`` dims, real
    projected ``y``/``x`` cell-center coordinates, and a written CRS -- e.g.
    from :func:`tanager_spec.io.load_tanager_sr_hdf5`) is sliced with
    ``.isel``. rioxarray derives the sub-window's GeoTransform directly from
    the sliced coordinate arrays (coordinates, not a manually-cached
    transform, are its source of truth for ``.rio.to_raster``), so the
    written chip is correctly georeferenced as long as the input cube's
    coordinates are self-consistent with its pixel grid -- true for every
    cube this project loads via ``tanager_spec.io``. See
    :func:`tests.test_pairs.test_write_chip_geotiff_has_correct_bounds` for
    the georeferencing regression check (an earlier version of this function
    tried to manually recompute the sub-window transform via
    ``rasterio.windows.transform`` and re-attach it with ``write_transform``;
    that call turned out to have no effect on ``to_raster``'s output, and was
    both redundant and misleading given rioxarray's actual coordinate-based
    behavior -- removed in favor of relying on the coordinates alone).

    Parameters
    ----------
    cube : xr.DataArray
        Full-scene cube, dims ``("band", "y", "x")``, with ``.rio.crs`` set
        and self-consistent ``y``/``x`` coordinates.
    y0, x0 : int
        Pixel offset of the patch's top-left corner.
    size : int
        Patch side length in pixels.
    out_path : str or Path
        Output GeoTIFF path.
    """
    patch = cube.isel(y=slice(y0, y0 + size), x=slice(x0, x0 + size))
    patch.rio.write_crs(cube.rio.crs, inplace=True)
    patch.rio.to_raster(out_path, compress="LZW")


@dataclass(frozen=True)
class ChipSetValidation:
    """Summary of a manifest-bound chip-set validation."""

    n_chips: int
    total_bytes: int
    checksum_manifest_sha256: str


_SHA256_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a regular, non-symlink file."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checksum target is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_chip_paths(
    patch_rows: Iterable[Mapping[str, object]],
) -> list[PurePosixPath]:
    paths: list[PurePosixPath] = []
    seen: set[PurePosixPath] = set()
    for row_number, row in enumerate(patch_rows, start=2):
        raw = str(row.get("chip_path", ""))
        rel = PurePosixPath(raw)
        if (
            not raw
            or "\\" in raw
            or rel.is_absolute()
            or raw != rel.as_posix()
            or ".." in rel.parts
            or len(rel.parts) < 2
            or rel.parts[0] != "chips"
            or rel.suffix.lower() != ".tif"
        ):
            raise ValueError(f"invalid chip_path on patches.csv row {row_number}: {raw!r}")
        if rel in seen:
            raise ValueError(f"duplicate chip_path in patches.csv: {raw}")
        seen.add(rel)
        paths.append(rel)
    if not paths:
        raise ValueError("patches.csv contains no chip rows")
    return sorted(paths, key=lambda path: path.as_posix())


def _chip_file(dataset_dir: Path, rel: PurePosixPath) -> Path:
    current = dataset_dir
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"chip path contains a symlink: {rel.as_posix()}")
    if not current.is_file():
        raise ValueError(f"manifest-referenced chip is missing: {rel.as_posix()}")
    return current


def _validate_exact_chip_set(
    dataset_dir: Path, patch_rows: Iterable[Mapping[str, object]]
) -> list[PurePosixPath]:
    dataset_dir = Path(dataset_dir)
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise ValueError(f"dataset directory is not a regular directory: {dataset_dir}")
    expected = _manifest_chip_paths(patch_rows)
    chips_dir = dataset_dir / "chips"
    if chips_dir.is_symlink() or not chips_dir.is_dir():
        raise ValueError(f"chips directory is not a regular directory: {chips_dir}")

    actual: list[PurePosixPath] = []
    for path in sorted(chips_dir.rglob("*")):
        rel = PurePosixPath(path.relative_to(dataset_dir).as_posix())
        if path.is_symlink():
            raise ValueError(f"chip tree contains a symlink: {rel.as_posix()}")
        if path.is_file():
            actual.append(rel)
        elif not path.is_dir():
            raise ValueError(f"chip tree contains a non-regular entry: {rel.as_posix()}")

    expected_set = set(expected)
    actual_set = set(actual)
    if expected_set != actual_set:
        missing = sorted(path.as_posix() for path in expected_set - actual_set)
        extra = sorted(path.as_posix() for path in actual_set - expected_set)
        raise ValueError(f"chip set does not match patches.csv; missing={missing}, extra={extra}")
    for rel in expected:
        _chip_file(dataset_dir, rel)
    return expected


def write_chip_checksum_manifest(
    dataset_dir: str | Path,
    patch_rows: Iterable[Mapping[str, object]],
    *,
    filename: str = "chips.sha256",
) -> Path:
    """Atomically write a sorted SHA-256 manifest for the exact chip set."""
    dataset_dir = Path(dataset_dir)
    rows = list(patch_rows)
    expected = _validate_exact_chip_set(dataset_dir, rows)
    manifest_path = dataset_dir / filename
    if manifest_path.parent != dataset_dir or manifest_path.name != filename:
        raise ValueError(f"checksum filename must be a dataset-root basename: {filename!r}")

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=dataset_dir,
            prefix=".chips.sha256.",
            delete=False,
        ) as fh:
            temporary_path = Path(fh.name)
            for rel in expected:
                fh.write(f"{sha256_file(_chip_file(dataset_dir, rel))}  {rel.as_posix()}\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return manifest_path


def validate_chip_dataset(
    dataset_dir: str | Path,
    patch_rows: Iterable[Mapping[str, object]],
    *,
    checksum_filename: str = "chips.sha256",
) -> ChipSetValidation:
    """Fail closed unless chips, patch rows, and recorded SHA-256 values agree."""
    dataset_dir = Path(dataset_dir)
    rows = list(patch_rows)
    expected = _validate_exact_chip_set(dataset_dir, rows)
    checksum_path = dataset_dir / checksum_filename
    if checksum_path.parent != dataset_dir or checksum_path.name != checksum_filename:
        raise ValueError(
            f"checksum filename must be a dataset-root basename: {checksum_filename!r}"
        )
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ValueError(f"checksum manifest is missing or is not a regular file: {checksum_path}")

    recorded_paths: list[PurePosixPath] = []
    recorded_hashes: dict[PurePosixPath, str] = {}
    with checksum_path.open(encoding="utf-8", newline="") as fh:
        for line_number, line in enumerate(fh, start=1):
            match = _SHA256_LINE.fullmatch(line.rstrip("\n"))
            if match is None:
                raise ValueError(f"malformed checksum line {line_number}: {line.rstrip()!r}")
            digest, raw = match.groups()
            rel = PurePosixPath(raw)
            if raw != rel.as_posix() or rel in recorded_hashes:
                raise ValueError(
                    f"non-canonical or duplicate checksum path on line {line_number}: {raw!r}"
                )
            recorded_paths.append(rel)
            recorded_hashes[rel] = digest

    if recorded_paths != expected:
        raise ValueError("chips.sha256 paths are not the sorted, exact patches.csv chip set")

    total_bytes = 0
    for rel in expected:
        path = _chip_file(dataset_dir, rel)
        observed = sha256_file(path)
        if observed != recorded_hashes[rel]:
            raise ValueError(
                f"chip checksum mismatch for {rel.as_posix()}: "
                f"recorded={recorded_hashes[rel]}, observed={observed}"
            )
        total_bytes += path.stat().st_size

    return ChipSetValidation(
        n_chips=len(expected),
        total_bytes=total_bytes,
        checksum_manifest_sha256=sha256_file(checksum_path),
    )


def promote_staged_dataset(staged_dir: str | Path, target_dir: str | Path) -> None:
    """Promote a validated sibling directory with rollback on rename failure."""
    staged_dir = Path(staged_dir)
    target_dir = Path(target_dir)
    if staged_dir.parent.resolve() != target_dir.parent.resolve():
        raise ValueError("staging and target directories must share a parent filesystem directory")
    if staged_dir.resolve() == target_dir.resolve():
        raise ValueError("staging and target directories must be different paths")
    if staged_dir.is_symlink() or not staged_dir.is_dir():
        raise ValueError(f"staging path is not a regular directory: {staged_dir}")
    if target_dir.is_symlink() or (target_dir.exists() and not target_dir.is_dir()):
        raise ValueError(f"target path is not a regular directory: {target_dir}")

    backup_dir = target_dir.with_name(f".{target_dir.name}.previous")
    if backup_dir.exists() or backup_dir.is_symlink():
        raise RuntimeError(
            f"refusing promotion because a previous-build backup exists: {backup_dir}"
        )

    moved_target = False
    if target_dir.exists():
        os.replace(target_dir, backup_dir)
        moved_target = True
    try:
        os.replace(staged_dir, target_dir)
    except BaseException:
        if moved_target and backup_dir.exists() and not target_dir.exists():
            os.replace(backup_dir, target_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def rgb_ambiguous_pairs(
    patches: list[Patch], *, quantile: float = RGB_CANDIDATE_QUANTILE
) -> RgbAmbiguityResult:
    """Cross-label patch pairs whose RGB mean AND std vectors are both close.

    "Close" is the ``quantile`` (default: bottom decile) of the pooled
    cross-label mean-vector and std-vector L2-distance distributions, so the
    bar is set from this dataset's own patches rather than the blog's
    Sentinel-2 DN thresholds.
    """
    n = len(patches)
    means = np.stack([p.rgb_mean for p in patches])
    stds = np.stack([p.rgb_std for p in patches])
    labels = np.array([p.label for p in patches])

    iu = np.triu_indices(n, k=1)
    mean_d = np.linalg.norm(means[iu[0]] - means[iu[1]], axis=1)
    std_d = np.linalg.norm(stds[iu[0]] - stds[iu[1]], axis=1)
    cross = labels[iu[0]] != labels[iu[1]]
    if not cross.any():
        raise ValueError("no cross-label patch pairs found; cannot derive RGB thresholds")

    mean_thr = float(np.quantile(mean_d[cross], quantile))
    std_thr = float(np.quantile(std_d[cross], quantile))
    keep = cross & (mean_d <= mean_thr) & (std_d <= std_thr)
    candidates = [
        (int(i), int(j), float(mean_d[k]), float(std_d[k]))
        for k, (i, j) in enumerate(zip(iu[0], iu[1], strict=True))
        if keep[k]
    ]
    return RgbAmbiguityResult(candidates, mean_thr, std_thr, mean_d[cross], std_d[cross])


@dataclass(frozen=True)
class HardCluster:
    """A connected component of the RGB-ambiguity graph spanning >=2 labels.

    The mineralogical analog of the blog's ``test_hard_clusters.parquet``:
    "each cluster is a connected component of the RGB mean/std similarity
    graph and spans at least two WorldCover labels." Here the graph's edges
    are the RGB-ambiguous candidate pairs from :func:`rgb_ambiguous_pairs`
    (already cross-label by construction), so every component of size >=2
    spans >=2 labels automatically.
    """

    cluster_id: int
    patches: list[Patch]

    @property
    def size(self) -> int:
        return len(self.patches)

    @property
    def labels(self) -> set[str]:
        return {p.label for p in self.patches}


def rgb_ambiguity_clusters(
    patches: list[Patch], candidates: list[tuple[int, int, float, float]]
) -> list[HardCluster]:
    """Connected components (size >= 2) of the RGB-ambiguity candidate graph.

    Nodes are patch indices into ``patches``; edges are the ``(i, j, ...)``
    pairs in ``candidates`` (from :func:`rgb_ambiguous_pairs`). Isolated
    patches (no RGB-ambiguous partner) are not clusters and are dropped,
    matching the blog's cluster sizes of 2-4, not 1. Every returned cluster's
    multi-label span is asserted, not just assumed, since it is the whole
    point of the metric downstream (a cluster-accuracy check is meaningless
    on a single-label group).

    Parameters
    ----------
    patches : list of Patch
        All labeled patches (indices match ``candidates``).
    candidates : list of (i, j, mean_l2, std_l2)
        RGB-ambiguous candidate pairs from :func:`rgb_ambiguous_pairs`.

    Returns
    -------
    list of HardCluster
        One per connected component of size >= 2, in discovery order.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)
    for i, j, *_ in candidates:
        adjacency[i].add(j)
        adjacency[j].add(i)

    seen: set[int] = set()
    clusters: list[HardCluster] = []
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members: list[int] = []
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(members) < 2:
            continue
        member_patches = [patches[idx] for idx in sorted(members)]
        n_labels = len({p.label for p in member_patches})
        if n_labels < 2:
            raise AssertionError(
                f"cluster {sorted(members)} is single-label ({member_patches[0].label}); "
                "the RGB-ambiguity graph should only ever connect different labels"
            )
        clusters.append(HardCluster(cluster_id=len(clusters), patches=member_patches))
    return clusters


def swir_separable_pairs(
    patches: list[Patch],
    candidates: list[tuple[int, int, float, float]],
    *,
    null_quantile: float = SWIR_NULL_QUANTILE,
) -> SwirSeparabilityResult:
    """Keep RGB-ambiguous candidates whose SWIR spectral angle beats the same-label null.

    The separability bar is calibrated from this project's own same-mineral
    patch pairs: the ``null_quantile`` percentile of :func:`tanager_minmap.
    speclib.pairwise_spectral_angle` computed on patch-mean SWIR-window
    spectra for every pair of patches sharing the SAME dominant label (the
    natural spread from sub-pixel mixing and noise). A cross-label candidate
    is "separable" only if it exceeds that bar -- it differs more, in the
    SWIR, than same-mineral patches typically differ from each other.
    """
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, p in enumerate(patches):
        by_label[p.label].append(idx)

    same_angles: list[float] = []
    for idxs in by_label.values():
        for i, j in combinations(idxs, 2):
            angle = pairwise_spectral_angle(
                patches[i].swir_mean, patches[j].swir_mean, degrees=True
            )
            if np.isfinite(angle):
                same_angles.append(angle)
    if not same_angles:
        raise ValueError("no same-label patch pairs found; cannot derive a SWIR null distribution")

    same_angles_arr = np.asarray(same_angles)
    threshold = float(np.quantile(same_angles_arr, null_quantile))

    pairs: list[HardPair] = []
    for i, j, mean_l2, std_l2 in candidates:
        angle = pairwise_spectral_angle(patches[i].swir_mean, patches[j].swir_mean, degrees=True)
        if np.isfinite(angle) and angle > threshold:
            pairs.append(HardPair(patches[i], patches[j], mean_l2, std_l2, angle))
    pairs.sort(key=lambda hp: hp.swir_angle_deg, reverse=True)
    return SwirSeparabilityResult(pairs, threshold, same_angles_arr)
