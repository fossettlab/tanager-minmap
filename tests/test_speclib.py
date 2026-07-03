"""Unit tests for the splib07 loader on synthetic library files."""

from __future__ import annotations

import numpy as np

from tanager_rocks.speclib import (
    Endmember,
    by_mineral,
    load_library,
    pairwise_spectral_angle,
    select_endmembers,
    spectrometer_of,
)

# A 5-channel ASD-format grid (micrometres) and a spectrum with one deleted channel.
_WAVELENGTHS_UM = [0.50, 1.00, 1.50, 2.00, 2.50]
_REFLECTANCE = [0.1, 0.2, -1.23e34, 0.4, 0.5]  # sentinel at 1500 nm
_ASD_WL_FILE = "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"


def _write_splib(path, header, values):
    path.write_text(header + "\n" + "\n".join(f" {v:.7e}" for v in values) + "\n")


def _build_library(tmp_path):
    (tmp_path / "ChapterM_Minerals").mkdir()
    _write_splib(tmp_path / _ASD_WL_FILE, " splib07a Record=23: Wavelengths ASD", _WAVELENGTHS_UM)
    _write_splib(
        tmp_path / "ChapterM_Minerals" / "splib07a_Testmin_S1_ASDFRa_AREF.txt",
        " splib07a Record=1: Testmin S1 ASDFRa AREF",
        _REFLECTANCE,
    )
    return tmp_path


def test_spectrometer_detection():
    assert spectrometer_of("splib07a_Alunite_HS295_ASDFRa_AREF.txt") == "ASD"
    assert spectrometer_of("splib07a_Kaolinite_CM3_BECKa_AREF.txt") == "BECK"
    assert spectrometer_of("splib07a_Olivine_NIC4_AREF.txt") is None


def test_load_and_resample(tmp_path):
    lib = _build_library(tmp_path)
    target_nm = np.array([1000.0, 2000.0])
    ems = load_library(lib, target_nm, minerals=("testmin",))
    assert len(ems) == 1
    em = ems[0]
    assert em.mineral == "testmin"
    assert em.spectrometer == "ASD"
    # Interpolation skips the deleted 1500 nm channel; 1000->0.2, 2000->0.4.
    np.testing.assert_allclose(em.reflectance, [0.2, 0.4])


def test_by_mineral_groups(tmp_path):
    ems = load_library(_build_library(tmp_path), np.array([1000.0]), minerals=("testmin",))
    grouped = by_mineral(ems)
    assert set(grouped) == {"testmin"}
    assert len(grouped["testmin"]) == 1


def test_select_endmembers_picks_medoid():
    wl = np.array([1000.0, 1100.0, 1200.0])
    # Two near-identical samples and one outlier; the medoid must be a typical one.
    typical_a = Endmember("m", "a", "ASD", wl, np.array([0.5, 0.4, 0.5]))
    typical_b = Endmember("m", "b", "ASD", wl, np.array([0.5, 0.42, 0.5]))
    outlier = Endmember("m", "c", "ASD", wl, np.array([0.1, 0.9, 0.1]))
    chosen = select_endmembers([typical_a, typical_b, outlier], minerals=("m",))
    assert chosen["m"].sample in {"a", "b"}  # not the outlier


def test_pairwise_spectral_angle_parallel_orthogonal_and_guard():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert np.isclose(pairwise_spectral_angle(a, 2 * a), 0.0, atol=1e-6)
    assert np.isclose(pairwise_spectral_angle(a, b), np.pi / 2, atol=1e-6)
    assert np.isclose(pairwise_spectral_angle(a, b, degrees=True), 90.0, atol=1e-4)
    # Fewer than two shared finite bands returns NaN instead of dividing by zero.
    one = np.array([1.0, np.nan, np.nan])
    assert np.isnan(pairwise_spectral_angle(one, one.copy()))
