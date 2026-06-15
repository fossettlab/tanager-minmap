"""Reference spectral-library loader (USGS / ECOSTRESS).

Loads the target alteration assemblage (see :data:`tanager_rocks.config.TARGET_MINERALS`)
from a published spectral library and resamples each endmember onto the
Tanager wavelength axis so it can be used directly by the diagnostic-feature
(:mod:`tanager_rocks.features`) and unmixing (:mod:`tanager_rocks.unmix`)
modules.

Library spectra are the *only* source of mineral identity in this project;
no synthetic or hand-edited endmembers are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TARGET_MINERALS


@dataclass(frozen=True)
class Endmember:
    """A single reference spectrum resampled to the Tanager wavelength axis."""

    mineral: str
    source: str  # provenance string, e.g. "USGS splib07a: Alunite GDS84"
    wavelengths_nm: np.ndarray
    reflectance: np.ndarray


def load_library(
    library_dir: str | Path,
    wavelengths_nm: np.ndarray,
    minerals: tuple[str, ...] = TARGET_MINERALS,
) -> dict[str, Endmember]:
    """Load and resample reference endmembers for the target assemblage.

    Parameters
    ----------
    library_dir : str or Path
        Directory holding the USGS/ECOSTRESS library files.
    wavelengths_nm : np.ndarray
        Tanager band centres to resample each endmember onto.
    minerals : tuple of str
        Minerals to load. Defaults to the full target assemblage.

    Returns
    -------
    dict
        Mineral name -> :class:`Endmember`.
    """
    # TODO (spec pipeline step 4): read ENVI/ASCII library spectra (spectral
    # or numpy), record provenance per spectrum, resample to wavelengths_nm,
    # and validate band coverage over the SWIR diagnostic windows.
    raise NotImplementedError
