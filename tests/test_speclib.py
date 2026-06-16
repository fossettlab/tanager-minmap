"""Unit tests for the splib07 loader on synthetic library files."""

from __future__ import annotations

import numpy as np

from tanager_rocks.speclib import by_mineral, load_library, spectrometer_of

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
