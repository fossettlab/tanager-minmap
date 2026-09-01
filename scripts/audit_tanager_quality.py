"""Audit the embedded Tanager quality layers and reflectance ranges.

This is a read-only diagnostic.  It reports the exact exclusion fractions from
the three beta usable-data-mask fields and measures reflectance-range behavior
on the spectral channels retained by the current analysis.  It does not choose
or apply an analysis threshold.

Run::

    uv run python scripts/audit_tanager_quality.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from tanager_spec.bands import indices_in_windows
from tanager_spec.config import ABSORPTION_MASKS_NM

from tanager_rocks.config import SITES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "quality_audit" / "scene_quality.json"
HDF_DATA_FIELDS = "HDFEOS/GRIDS/HYP/Data Fields"
QA_FIELDS = ("beta_cloud_mask", "beta_cirrus_mask", "nodata_pixels")


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _scene_ids() -> tuple[str, ...]:
    return tuple(scene_id for site in SITES.values() for scene_id in site.scene_ids)


def audit_scene(
    path: Path,
    *,
    row_chunk: int = 64,
    sample_stride: int = 20,
) -> dict[str, object]:
    """Return quality-mask and reflectance-range diagnostics for one scene."""
    with h5py.File(path, "r") as handle:
        fields = handle[HDF_DATA_FIELDS]
        reflectance = fields["surface_reflectance"]
        wavelengths = np.asarray(reflectance.attrs["wavelengths"], dtype=float)
        product_good = np.asarray(reflectance.attrs["good_wavelengths"], dtype=bool)
        fill_value = float(reflectance.attrs["_FillValue"])
        cube_shape = tuple(int(value) for value in reflectance.shape)

        qa = {name: np.asarray(fields[name][...]) for name in QA_FIELDS}
        qa_invalid = np.logical_or.reduce([values == 1 for values in qa.values()])
        qa_fill = np.logical_or.reduce([values == 255 for values in qa.values()])
        configured_bad = indices_in_windows(wavelengths, ABSORPTION_MASKS_NM)
        configured_retained = ~configured_bad
        analysis_bands = product_good & configured_retained
        analysis_indices = np.flatnonzero(analysis_bands)

        ny, nx = reflectance.shape[1:]
        negative_pixels = np.zeros((ny, nx), dtype=bool)
        above_one_pixels = np.zeros((ny, nx), dtype=bool)
        above_one_five_pixels = np.zeros((ny, nx), dtype=bool)
        fill_pixels = np.zeros((ny, nx), dtype=bool)
        finite_min = np.inf
        finite_max = -np.inf
        sampled_values: list[np.ndarray] = []

        for y0 in range(0, ny, row_chunk):
            y1 = min(y0 + row_chunk, ny)
            block = np.asarray(reflectance[analysis_indices, y0:y1, :], dtype=np.float32)
            block_fill = block == fill_value
            fill_pixels[y0:y1] |= block_fill.any(axis=0)

            clear = ~qa_invalid[y0:y1]
            usable = np.isfinite(block) & ~block_fill & clear[None, :, :]
            negative_pixels[y0:y1] |= ((block < 0) & usable).any(axis=0)
            above_one_pixels[y0:y1] |= ((block > 1) & usable).any(axis=0)
            above_one_five_pixels[y0:y1] |= ((block > 1.5) & usable).any(axis=0)

            usable_values = block[usable]
            if usable_values.size:
                finite_min = min(finite_min, float(usable_values.min()))
                finite_max = max(finite_max, float(usable_values.max()))

            sample_y = np.arange(y0, y1)
            take_y = (sample_y % sample_stride) == 0
            sampled = block[:, take_y, ::sample_stride]
            sampled_clear = clear[take_y, ::sample_stride]
            sampled_ok = np.isfinite(sampled) & (sampled != fill_value) & sampled_clear[None, :, :]
            sampled_values.append(sampled[sampled_ok])

    total_pixels = ny * nx
    clear_pixels = ~qa_invalid
    n_clear = int(clear_pixels.sum())
    sample = np.concatenate(sampled_values) if sampled_values else np.empty(0)
    quantiles = {}
    if sample.size:
        for quantile in (0.001, 0.01, 0.5, 0.99, 0.999):
            quantiles[str(quantile)] = float(np.quantile(sample, quantile))

    return {
        "scene_id": path.name.removesuffix("_ortho_sr_hdf5.h5"),
        "path": str(path.relative_to(ROOT)),
        "shape": list(cube_shape),
        "total_pixels": total_pixels,
        "fill_value": fill_value,
        "qa_value_counts": {name: _counts(values) for name, values in qa.items()},
        "qa_fill_value_pixels": int(qa_fill.sum()),
        "nodata_pixels": int((qa["nodata_pixels"] == 1).sum()),
        "cloud_pixels": int((qa["beta_cloud_mask"] == 1).sum()),
        "cirrus_pixels": int((qa["beta_cirrus_mask"] == 1).sum()),
        "qa_union_invalid_pixels": int(qa_invalid.sum()),
        "qa_union_invalid_fraction": float(qa_invalid.mean()),
        "qa_clear_pixels": n_clear,
        "qa_clear_fraction": float(clear_pixels.mean()),
        "fill_matches_nodata": bool(np.array_equal(fill_pixels, qa["nodata_pixels"] == 1)),
        "product_good_wavelength_bands": int(product_good.sum()),
        "product_bad_wavelength_bands": int((~product_good).sum()),
        "configured_window_retained_bands": int(configured_retained.sum()),
        "configured_retained_product_bad_bands": int((configured_retained & ~product_good).sum()),
        "final_retained_bands": int(analysis_bands.sum()),
        "qa_clear_pixels_with_any_negative": int((negative_pixels & clear_pixels).sum()),
        "qa_clear_pixels_with_any_negative_fraction": (
            float((negative_pixels & clear_pixels).sum() / n_clear) if n_clear else None
        ),
        "qa_clear_pixels_with_any_above_one": int((above_one_pixels & clear_pixels).sum()),
        "qa_clear_pixels_with_any_above_one_fraction": (
            float((above_one_pixels & clear_pixels).sum() / n_clear) if n_clear else None
        ),
        "qa_clear_pixels_with_any_above_one_five": int(
            (above_one_five_pixels & clear_pixels).sum()
        ),
        "qa_clear_pixels_with_any_above_one_five_fraction": (
            float((above_one_five_pixels & clear_pixels).sum() / n_clear) if n_clear else None
        ),
        "qa_clear_analysis_value_min": finite_min if np.isfinite(finite_min) else None,
        "qa_clear_analysis_value_max": finite_max if np.isfinite(finite_max) else None,
        "qa_clear_analysis_value_sample_stride": sample_stride,
        "qa_clear_analysis_value_sample_size": int(sample.size),
        "qa_clear_analysis_value_sample_quantiles": quantiles,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--sample-stride", type=int, default=20)
    args = parser.parse_args(argv)

    summaries = []
    for scene_id in _scene_ids():
        path = args.raw_dir / f"{scene_id}_ortho_sr_hdf5.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"auditing {scene_id}", flush=True)
        summaries.append(
            audit_scene(
                path,
                row_chunk=args.row_chunk,
                sample_stride=args.sample_stride,
            )
        )

    payload = {
        "policy_sources": {
            "planet_tanager_product_specification": (
                "https://docs.planet.com/data/imagery/tanager/techspec/"
            ),
            "qa_semantics": {
                "beta_cloud_mask": "1 indicates cloud",
                "beta_cirrus_mask": "1 indicates cirrus",
                "nodata_pixels": "1 indicates no data",
            },
            "reflectance_range_statement": "unitless and typically between 0 and 1",
        },
        "analysis_absorption_windows_nm": [list(window) for window in ABSORPTION_MASKS_NM],
        "scenes": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
