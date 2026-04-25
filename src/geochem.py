"""Geochemistry data loading, cleaning, and normalization.

Handles GEOROC and PetDB parquet files: loading, oxide harmonization
(FeO → Fe2O3(T)), coordinate filtering, and anhydrous normalization.

Typical usage::

    from src.geochem import load_georoc, load_petdb, normalize_anhydrous

    georoc = load_georoc("data/georoc/")
    petdb = load_petdb("data/petdb/")
    combined = normalize_anhydrous(pd.concat([georoc, petdb]))
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Target major-element oxides
TARGET_OXIDES = [
    "SiO2",
    "Al2O3",
    "Fe2O3T",  # total iron as Fe2O3
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "TiO2",
]

# Conversion factor: FeO → Fe2O3
FEO_TO_FE2O3_FACTOR = 1.1113

# Minimum coordinate decimal places for filtering
MIN_COORD_DECIMALS = 3


def load_georoc(
    data_dir: str | Path,
    column_mappings: dict | None = None,
) -> pd.DataFrame:
    """Load and standardize GEOROC parquet files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing GEOROC parquet files.
    column_mappings : dict, optional
        Column name mappings. If None, uses default oxide names.

    Returns
    -------
    pd.DataFrame
        Standardized geochemistry data with columns:
        latitude, longitude, sample_id, and oxide columns (wt%).
    """
    # TODO: implement
    # 1. Read all parquet files from data_dir
    # 2. Apply column_mappings (or load from EDC/petdb/column_mappings.json)
    # 3. Standardize oxide column names
    # 4. Return DataFrame with consistent schema
    raise NotImplementedError


def load_petdb(
    data_dir: str | Path,
    column_mappings: dict | None = None,
) -> pd.DataFrame:
    """Load and standardize PetDB parquet files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing PetDB parquet files.
    column_mappings : dict, optional
        Column name mappings.

    Returns
    -------
    pd.DataFrame
        Standardized geochemistry data.
    """
    # TODO: implement
    raise NotImplementedError


def harmonize_iron(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FeO to Fe2O3(T) and create unified total-iron column.

    If both FeO and Fe2O3 are present, computes:
        Fe2O3T = Fe2O3 + FeO * 1.1113

    Parameters
    ----------
    df : pd.DataFrame
        Geochemistry data with FeO and/or Fe2O3 columns.

    Returns
    -------
    pd.DataFrame
        Data with Fe2O3T column added, original Fe columns retained.
    """
    # TODO: implement
    # 1. Check which Fe columns exist
    # 2. Compute Fe2O3T = Fe2O3 + FeO * FEO_TO_FE2O3_FACTOR
    # 3. Handle cases where only one Fe column is present
    raise NotImplementedError


def filter_coordinate_precision(
    df: pd.DataFrame,
    min_decimals: int = MIN_COORD_DECIMALS,
) -> pd.DataFrame:
    """Filter samples by coordinate decimal-place precision.

    Removes samples with imprecise coordinates (fewer than ``min_decimals``
    decimal places), which likely represent regional averages rather than
    point samples.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'latitude' and 'longitude' columns.
    min_decimals : int
        Minimum required decimal places.

    Returns
    -------
    pd.DataFrame
        Filtered data. Logs attrition count.
    """
    # TODO: implement
    # 1. Count decimal places for lat/lon
    # 2. Filter rows where both >= min_decimals
    # 3. Log attrition: n_before - n_after
    raise NotImplementedError


def normalize_anhydrous(
    df: pd.DataFrame,
    oxide_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize oxides to anhydrous (volatile-free) 100% total.

    Parameters
    ----------
    df : pd.DataFrame
        Geochemistry data with oxide columns in wt%.
    oxide_cols : list of str, optional
        Columns to include in normalization. Defaults to TARGET_OXIDES.

    Returns
    -------
    pd.DataFrame
        Data with oxide columns normalized so they sum to 100%.
    """
    # TODO: implement
    # 1. Sum oxide_cols per row
    # 2. Divide each oxide by row sum, multiply by 100
    # 3. Handle rows where sum is zero or NaN
    raise NotImplementedError
