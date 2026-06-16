"""USGS Spectral Library Version 7 (splib07a) loader.

Reads the base ASCII spectra (Kokaly et al. 2017, USGS Data Series 1035;
fetched by ``scripts/download_speclib.py``) for the target alteration
assemblage and resamples each onto the Tanager wavelength axis. Library
spectra are the only source of mineral identity in this project; nothing here
is synthesised.

The target minerals are measured on two lab spectrometers — ASD (2151 ch,
0.35-2.5 um; alunite, hematite, goethite, gypsum, muscovite) and Beckman
("BECK", 0.2-3.0 um; kaolinite, dickite, jarosite). Each spectrum's grid is
read from the matching per-spectrometer wavelength file, detected from the
spectrometer token in the filename. Each mineral has several samples; all are
loaded (selecting a single endmember per mineral is deferred to the unmixing
step).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TARGET_MINERALS

logger = logging.getLogger(__name__)

# splib07 flags deleted channels with a large negative sentinel.
_DELETED_SENTINEL = -1e30

# Per-spectrometer wavelength files (in the splib07a root). The token before
# "_AREF" in a spectrum filename selects the grid: "ASDFR*" -> ASD, "BECK*" -> BECK.
_WAVELENGTH_FILE = {
    "ASD": "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt",
    "BECK": "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt",
}


@dataclass(frozen=True)
class Endmember:
    """A reference spectrum resampled to the Tanager wavelength axis."""

    mineral: str
    sample: str  # splib07 filename (provenance)
    spectrometer: str  # "ASD" or "BECK"
    wavelengths_nm: np.ndarray
    reflectance: np.ndarray


def _read_splib_values(path: Path) -> np.ndarray:
    """Read a splib07 ASCII file (header line + one value per line) to an array.

    The deleted-channel sentinel is converted to ``NaN``.
    """
    with path.open() as fh:
        lines = fh.read().splitlines()
    values = [float(ln) for ln in lines[1:] if ln.strip()]
    arr = np.asarray(values, dtype=float)
    arr[arr <= _DELETED_SENTINEL] = np.nan
    return arr


def _read_wavelengths(path: Path) -> np.ndarray:
    """Read a splib07 wavelength file (micrometres) and return nanometres."""
    return _read_splib_values(path) * 1000.0


def spectrometer_of(filename: str) -> str | None:
    """Return ``"ASD"`` / ``"BECK"`` from a spectrum filename, or ``None``."""
    if "ASDFR" in filename:
        return "ASD"
    if "BECK" in filename:
        return "BECK"
    return None


def _resample(src_wl_nm: np.ndarray, src_refl: np.ndarray, target_nm: np.ndarray) -> np.ndarray:
    """Linearly resample a (possibly NaN-gapped) spectrum onto ``target_nm``.

    Interpolation uses only finite source channels; target wavelengths outside
    the source range become ``NaN`` rather than being extrapolated.
    """
    finite = np.isfinite(src_refl)
    return np.interp(target_nm, src_wl_nm[finite], src_refl[finite], left=np.nan, right=np.nan)


def load_library(
    library_dir: str | Path,
    wavelengths_nm: np.ndarray,
    minerals: tuple[str, ...] = TARGET_MINERALS,
) -> list[Endmember]:
    """Load and resample all target-mineral endmembers from splib07a.

    Parameters
    ----------
    library_dir : str or Path
        The extracted ``ASCIIdata_splib07a`` directory.
    wavelengths_nm : np.ndarray
        Tanager band centres to resample each endmember onto.
    minerals : tuple of str
        Minerals to load (matched against the ``ChapterM_Minerals`` filenames).
        Defaults to the full target assemblage.

    Returns
    -------
    list of Endmember
        Every matching sample, resampled to ``wavelengths_nm``.
    """
    library_dir = Path(library_dir)
    minerals_dir = library_dir / "ChapterM_Minerals"
    target_nm = np.asarray(wavelengths_nm, dtype=float)

    grids: dict[str, np.ndarray] = {}

    def grid(spectrometer: str) -> np.ndarray:
        if spectrometer not in grids:
            grids[spectrometer] = _read_wavelengths(library_dir / _WAVELENGTH_FILE[spectrometer])
        return grids[spectrometer]

    out: list[Endmember] = []
    for mineral in minerals:
        prefix = f"splib07a_{mineral}_".lower()
        for path in sorted(minerals_dir.glob("splib07a_*_AREF.txt")):
            if not path.name.lower().startswith(prefix):
                continue
            spec = spectrometer_of(path.name)
            if spec is None:
                continue
            refl = _read_splib_values(path)
            wl = grid(spec)
            if wl.size != refl.size:
                logger.warning(
                    "size mismatch for %s (%d vs %d); skipping", path.name, refl.size, wl.size
                )
                continue
            out.append(
                Endmember(
                    mineral=mineral,
                    sample=path.name,
                    spectrometer=spec,
                    wavelengths_nm=target_nm,
                    reflectance=_resample(wl, refl, target_nm),
                )
            )
    logger.info("loaded %d endmembers across %d minerals", len(out), len(minerals))
    return out


def by_mineral(endmembers: list[Endmember]) -> dict[str, list[Endmember]]:
    """Group endmembers by mineral name."""
    grouped: dict[str, list[Endmember]] = {}
    for e in endmembers:
        grouped.setdefault(e.mineral, []).append(e)
    return grouped


def _spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral angle (radians) between two spectra over their finite overlap."""
    m = np.isfinite(a) & np.isfinite(b)
    av, bv = a[m], b[m]
    cos = float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv)))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def select_endmembers(
    endmembers: list[Endmember],
    minerals: tuple[str, ...] = TARGET_MINERALS,
) -> dict[str, Endmember]:
    """Pick one representative (medoid) endmember per mineral.

    For each mineral the medoid is the real sample whose spectrum has the
    smallest spectral angle to the mineral's median across samples — a genuine
    measured spectrum (good for SAM/MTMF), reproducible, and outlier-resistant.

    Parameters
    ----------
    endmembers : list of Endmember
        All loaded samples (from :func:`load_library`).
    minerals : tuple of str
        Minerals to select for. Defaults to the full target assemblage.

    Returns
    -------
    dict
        Mineral name -> the chosen :class:`Endmember`. Minerals with no loaded
        sample are omitted.
    """
    grouped = by_mineral(endmembers)
    chosen: dict[str, Endmember] = {}
    for mineral in minerals:
        samples = grouped.get(mineral, [])
        if not samples:
            logger.warning("no samples for %s; skipping", mineral)
            continue
        median = np.nanmedian(np.vstack([e.reflectance for e in samples]), axis=0)
        angles = [_spectral_angle(e.reflectance, median) for e in samples]
        pick = samples[int(np.argmin(angles))]
        chosen[mineral] = pick
        logger.info("%s endmember: %s (medoid of %d)", mineral, pick.sample, len(samples))
    return chosen
