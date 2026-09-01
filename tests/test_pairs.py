"""Unit tests for hard-pair mining on synthetic, analytically-derivable arrays."""

from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine

import tanager_minmap.pairs as pairs_module
from tanager_minmap.pairs import (
    Patch,
    continuum_removed,
    pooled_rgb_percentiles,
    promote_staged_dataset,
    rgb_ambiguity_clusters,
    rgb_ambiguous_pairs,
    stretch_to_uint8,
    swir_separable_pairs,
    tile_and_label,
    validate_chip_dataset,
    write_chip_checksum_manifest,
    write_chip_geotiff,
)
from tanager_minmap.speclib import pairwise_spectral_angle


def _patch(label: str, rgb_mean, rgb_std, swir_mean) -> Patch:
    return Patch(
        site_id="s",
        scene_id="sc",
        row=0,
        col=0,
        y0=0,
        x0=0,
        label=label,
        purity=1.0,
        rgb_mean=np.asarray(rgb_mean, dtype=float),
        rgb_std=np.asarray(rgb_std, dtype=float),
        swir_mean=np.asarray(swir_mean, dtype=float),
    )


def test_tile_and_label_discard_reasons_and_survivor():
    # A 4x4 pixel grid tiled into four 2x2 patches, one of each fate:
    #   (0,0): uniform code 0 -> labeled "alunite", purity 1.0
    #   (0,1): uniform code -1 -> no_detection
    #   (1,0): two 0s / two 1s, tie broken to the lower code (0) -> purity 0.5 < floor -> low_purity
    #   (1,1): uniform code 0 but one invalid pixel -> invalid
    dominant_code = np.array(
        [
            [0, 0, -1, -1],
            [0, 0, -1, -1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    invalid_mask = np.zeros((4, 4), dtype=bool)
    invalid_mask[2, 2] = True  # inside patch (1,1)
    rgb_uint8 = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_uint8[0:2, 0:2] = (10, 20, 30)  # patch (0,0)
    swir_cube = np.zeros((2, 4, 4))
    swir_cube[:, 0:2, 0:2] = np.array([0.1, 0.2]).reshape(2, 1, 1)  # patch (0,0)

    patches, counts = tile_and_label(
        dominant_code,
        ["alunite", "muscovite"],
        invalid_mask,
        rgb_uint8,
        swir_cube,
        site_id="s",
        scene_id="sc",
        patch_size=2,
        purity_floor=0.70,
    )

    assert counts == {
        "total": 4,
        "invalid": 1,
        "no_detection": 1,
        "low_purity": 1,
        "labeled": 1,
    }
    assert len(patches) == 1
    p = patches[0]
    assert p.label == "alunite"
    assert p.purity == 1.0
    np.testing.assert_allclose(p.rgb_mean, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(p.rgb_std, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(p.swir_mean, [0.1, 0.2])


def test_rgb_ambiguous_pairs_selects_close_cross_label_pair():
    # Four patches, two labels, std held at zero everywhere so only the mean
    # distance drives selection. Cross-label L2 mean distances: (0,2)=1.0,
    # (0,3)=86.60, (1,2)=172.63, (1,3)=86.60. The bottom-decile (0.10) of that
    # set sits between the two smallest values (1.0 and 86.60), so only the
    # closest pair (0, 2) clears both the mean and the (degenerate, all-zero)
    # std threshold.
    patches = [
        _patch("alunite", [0, 0, 0], [0, 0, 0], [0.0]),
        _patch("alunite", [100, 100, 100], [0, 0, 0], [0.0]),
        _patch("muscovite", [1, 0, 0], [0, 0, 0], [0.0]),
        _patch("muscovite", [50, 50, 50], [0, 0, 0], [0.0]),
    ]
    result = rgb_ambiguous_pairs(patches, quantile=0.10)
    assert [(i, j) for i, j, *_ in result.candidates] == [(0, 2)]
    np.testing.assert_allclose(result.candidates[0][2], 1.0)
    assert result.std_threshold == 0.0


def test_rgb_ambiguous_pairs_requires_both_mean_and_std_close():
    # patch1 has the closest MEAN to patch0 (distance 0.5) of any cross pair,
    # but its std vector is far away (distance ~17.3); patch2 has a larger
    # mean distance (5.0) but a matching std (distance 0.0). Requiring BOTH
    # thresholds simultaneously excludes both -- the AND, not OR, semantics.
    patches = [
        _patch("alunite", [0, 0, 0], [0, 0, 0], [0.0]),
        _patch("muscovite", [0.5, 0, 0], [10, 10, 10], [0.0]),
        _patch("muscovite", [5.0, 0, 0], [0, 0, 0], [0.0]),
    ]
    result = rgb_ambiguous_pairs(patches, quantile=0.10)
    assert result.candidates == []


def test_swir_separable_pairs_keeps_only_the_orthogonal_candidate():
    # alunite and muscovite each contribute a near-parallel same-label pair
    # (angle ~0.57 deg) to the null distribution; gypsum has only one member
    # so contributes nothing. Of two candidates, (alunite, gypsum) is exactly
    # orthogonal (90 deg, from [1, 0] vs [0, 1]) and clears the null; (alunite,
    # muscovite) is built from identical vectors (angle 0 deg) and does not.
    patches = [
        _patch("alunite", [0, 0, 0], [0, 0, 0], [1.0, 0.0]),  # 0
        _patch("alunite", [0, 0, 0], [0, 0, 0], [1.0, 0.01]),  # 1
        _patch("muscovite", [0, 0, 0], [0, 0, 0], [1.0, 0.0]),  # 2
        _patch("muscovite", [0, 0, 0], [0, 0, 0], [1.0, 0.01]),  # 3
        _patch("gypsum", [0, 0, 0], [0, 0, 0], [0.0, 1.0]),  # 4
    ]
    candidates = [(0, 4, 1.0, 1.0), (0, 2, 1.0, 1.0)]
    result = swir_separable_pairs(patches, candidates, null_quantile=0.95)

    expected_threshold = pairwise_spectral_angle(
        patches[0].swir_mean, patches[1].swir_mean, degrees=True
    )
    np.testing.assert_allclose(result.threshold_deg, expected_threshold)
    assert len(result.pairs) == 1
    assert result.pairs[0].a.label == "alunite" and result.pairs[0].b.label == "gypsum"
    np.testing.assert_allclose(result.pairs[0].swir_angle_deg, 90.0)


def test_continuum_removed_flat_for_a_line_and_dips_for_an_absorption():
    wl = np.array([2000.0, 2100.0, 2200.0, 2300.0, 2400.0])
    # A perfectly linear spectrum continuum-removes to exactly 1.0 everywhere
    # (its own endpoint-to-endpoint line IS the spectrum).
    line = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
    np.testing.assert_allclose(continuum_removed(wl, line), np.ones(5), atol=1e-12)

    # The same line with a dip carved out of the center band drops below 1.0
    # only there; endpoints stay at exactly 1.0 by construction.
    dipped = line.copy()
    dipped[2] = 0.15
    out = continuum_removed(wl, dipped)
    assert out[2] < 1.0
    np.testing.assert_allclose(out[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(out[-1], 1.0, atol=1e-12)


def test_pooled_rgb_percentiles_and_stretch_to_uint8():
    # Two 1-pixel-tall "scenes"; channel 0 values pooled across both are
    # [0, 10, 100] with one flagged invalid (excluded), so the 0th/100th
    # percentile bounds are the min/max of the two VALID values (0 and 10).
    rgb_a = np.array([[[0.0, 10.0]], [[0.0, 0.0]], [[0.0, 0.0]]])  # (3, 1, 2)
    invalid_a = np.array([[False, False]])
    rgb_b = np.array([[[100.0]], [[0.0]], [[0.0]]])  # (3, 1, 1)
    invalid_b = np.array([[True]])  # excluded from the pooled stats

    lo, hi = pooled_rgb_percentiles([(rgb_a, invalid_a), (rgb_b, invalid_b)], pct=(0.0, 100.0))
    np.testing.assert_allclose(lo[0], 0.0)
    np.testing.assert_allclose(hi[0], 10.0)

    out = stretch_to_uint8(rgb_a, invalid_a, lo, hi)
    assert out[0, 0, 0] == 0
    assert out[0, 1, 0] == 255  # channel-0 value 10.0 maps to the top of [lo, hi]

    out_b = stretch_to_uint8(rgb_b, invalid_b, lo, hi)
    assert tuple(out_b[0, 0]) == (255, 255, 255)  # invalid pixel forced white


def test_write_chip_geotiff_has_correct_bounds(tmp_path):
    # A 6x6, 3-band, north-up scene: 30 m pixels, upper-left corner at
    # (500000, 4500180) -- i.e. the full scene spans x in [500000, 500180]
    # and y in [4500000, 4500180]. For window (x0=1, y0=2, size=3), the
    # expected sub-window bounds are computed by hand from that geometry
    # (independent of the function under test):
    #   left   = 500000 + 1*30          = 500030
    #   top    = 4500180 + 2*(-30)      = 4500120
    #   right  = left + 3*30            = 500120
    #   bottom = top + 3*(-30)          = 4500030
    # y/x are cell-CENTER coordinates built from the transform exactly as
    # tanager_spec.io.load_tanager_sr_hdf5 builds them -- self-consistency
    # between coordinates and the nominal transform is what rioxarray's
    # coordinate-derived georeferencing (see write_chip_geotiff's docstring)
    # actually depends on, so the test must construct it the same way.
    full_transform = Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4500180.0)
    nx = ny = 6
    xs = full_transform.c + (np.arange(nx) + 0.5) * full_transform.a
    ys = full_transform.f + (np.arange(ny) + 0.5) * full_transform.e
    cube = xr.DataArray(
        np.zeros((3, ny, nx)),
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": ys, "x": xs},
    )
    cube.rio.write_crs(CRS.from_epsg(32612), inplace=True)

    out_path = tmp_path / "chip.tif"
    write_chip_geotiff(cube, y0=2, x0=1, size=3, out_path=out_path)

    written = rioxarray.open_rasterio(out_path)
    assert written.shape == (3, 3, 3)
    np.testing.assert_allclose(written.rio.bounds(), (500030.0, 4500030.0, 500120.0, 4500120.0))
    assert written.rio.crs == CRS.from_epsg(32612)


def test_rgb_ambiguity_clusters_finds_components_and_drops_isolated_nodes():
    # Seven patches, seven distinct labels. Edges (0,1) and (1,2) chain into
    # one 3-node component spanning {A, B, C}; edges (3,4) form a separate
    # 2-node component spanning {D, E}; patches 5 and 6 have no edges at all
    # and must NOT appear as size-1 "clusters".
    patches = [
        _patch("A", [0, 0, 0], [0, 0, 0], [0.0]),  # 0
        _patch("B", [0, 0, 0], [0, 0, 0], [0.0]),  # 1
        _patch("C", [0, 0, 0], [0, 0, 0], [0.0]),  # 2
        _patch("D", [0, 0, 0], [0, 0, 0], [0.0]),  # 3
        _patch("E", [0, 0, 0], [0, 0, 0], [0.0]),  # 4
        _patch("F", [0, 0, 0], [0, 0, 0], [0.0]),  # 5, isolated
        _patch("G", [0, 0, 0], [0, 0, 0], [0.0]),  # 6, isolated
    ]
    candidates = [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0), (3, 4, 1.0, 1.0)]

    clusters = rgb_ambiguity_clusters(patches, candidates)

    assert len(clusters) == 2
    sizes = sorted(c.size for c in clusters)
    assert sizes == [2, 3]
    label_sets = sorted((frozenset(c.labels) for c in clusters), key=len)
    assert label_sets == [frozenset({"D", "E"}), frozenset({"A", "B", "C"})]
    # patches 5 and 6 (no edges) must not appear in any cluster.
    all_member_labels = {p.label for c in clusters for p in c.patches}
    assert "F" not in all_member_labels and "G" not in all_member_labels


def test_rgb_ambiguity_clusters_empty_candidates_yields_no_clusters():
    patches = [_patch("A", [0, 0, 0], [0, 0, 0], [0.0]), _patch("B", [0, 0, 0], [0, 0, 0], [0.0])]
    assert rgb_ambiguity_clusters(patches, []) == []


def _fake_chip_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    scene_dir = dataset_dir / "chips" / "scene"
    scene_dir.mkdir(parents=True)
    (scene_dir / "b.tif").write_bytes(b"second chip")
    (scene_dir / "a.tif").write_bytes(b"first chip")
    patch_rows = [
        {"patch_id": "b", "chip_path": "chips/scene/b.tif"},
        {"patch_id": "a", "chip_path": "chips/scene/a.tif"},
    ]
    return dataset_dir, patch_rows


def test_chip_checksum_manifest_is_sorted_deterministic_and_validated(tmp_path):
    dataset_dir, patch_rows = _fake_chip_dataset(tmp_path)

    manifest_path = write_chip_checksum_manifest(dataset_dir, patch_rows)
    first_bytes = manifest_path.read_bytes()
    report = validate_chip_dataset(dataset_dir, patch_rows)
    write_chip_checksum_manifest(dataset_dir, patch_rows)

    assert manifest_path.read_bytes() == first_bytes
    assert [line.split("  ", 1)[1] for line in manifest_path.read_text().splitlines()] == [
        "chips/scene/a.tif",
        "chips/scene/b.tif",
    ]
    assert report.n_chips == 2
    assert report.total_bytes == len(b"first chip") + len(b"second chip")
    assert len(report.checksum_manifest_sha256) == 64


def test_chip_validation_rejects_unreferenced_stale_file(tmp_path):
    dataset_dir, patch_rows = _fake_chip_dataset(tmp_path)
    write_chip_checksum_manifest(dataset_dir, patch_rows)
    (dataset_dir / "chips" / "scene" / "stale.tif").write_bytes(b"legacy")

    with pytest.raises(ValueError, match="chip set does not match patches.csv"):
        validate_chip_dataset(dataset_dir, patch_rows)


def test_chip_validation_rejects_post_manifest_modification(tmp_path):
    dataset_dir, patch_rows = _fake_chip_dataset(tmp_path)
    write_chip_checksum_manifest(dataset_dir, patch_rows)
    (dataset_dir / "chips" / "scene" / "a.tif").write_bytes(b"changed")

    with pytest.raises(ValueError, match="chip checksum mismatch"):
        validate_chip_dataset(dataset_dir, patch_rows)


def test_chip_validation_rejects_symlinked_chip(tmp_path):
    dataset_dir = tmp_path / "dataset"
    scene_dir = dataset_dir / "chips" / "scene"
    scene_dir.mkdir(parents=True)
    source = tmp_path / "outside.tif"
    source.write_bytes(b"outside")
    (scene_dir / "linked.tif").symlink_to(source)
    rows = [{"patch_id": "linked", "chip_path": "chips/scene/linked.tif"}]

    with pytest.raises(ValueError, match="symlink"):
        write_chip_checksum_manifest(dataset_dir, rows)


def test_promote_staged_dataset_replaces_target_and_removes_backup(tmp_path):
    target = tmp_path / "hard_pairs_dataset"
    target.mkdir()
    (target / "old.txt").write_text("old")
    staged = tmp_path / ".hard_pairs_dataset.staging-test"
    staged.mkdir()
    (staged / "new.txt").write_text("new")

    promote_staged_dataset(staged, target)

    assert not staged.exists()
    assert not (tmp_path / ".hard_pairs_dataset.previous").exists()
    assert not (target / "old.txt").exists()
    assert (target / "new.txt").read_text() == "new"


def test_promote_staged_dataset_restores_target_when_promotion_fails(tmp_path, monkeypatch):
    target = tmp_path / "hard_pairs_dataset"
    target.mkdir()
    (target / "old.txt").write_text("old")
    staged = tmp_path / ".hard_pairs_dataset.staging-test"
    staged.mkdir()
    (staged / "new.txt").write_text("new")
    real_replace = pairs_module.os.replace
    call_count = 0

    def fail_second_replace(source, destination):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("synthetic promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(pairs_module.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="synthetic promotion failure"):
        promote_staged_dataset(staged, target)

    assert staged.is_dir()
    assert (staged / "new.txt").read_text() == "new"
    assert (target / "old.txt").read_text() == "old"
    assert not (tmp_path / ".hard_pairs_dataset.previous").exists()
