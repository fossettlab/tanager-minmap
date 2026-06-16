"""Download and extract the USGS Spectral Library Version 7 (base ASCII spectra).

Fetches ``ASCIIdata_splib07a.zip`` (the original-resolution base spectra,
Kokaly et al. 2017, USGS Data Series 1035) from its ScienceBase item and
extracts it under ``data/speclib/``. The download URL is resolved from the
ScienceBase item by file name, not hand-built. Existing files are reused.

Run::

    uv run python scripts/download_speclib.py
"""

from __future__ import annotations

import json
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("download_speclib")

USER_AGENT = "tanager-rocks/0.1 (research; abradley@wustl.edu)"
# ScienceBase item "Spectra of materials in ASCII format" (USGS DS 1035).
SCIENCEBASE_ITEM = "586e8c88e4b0f5ce109fccae"
ZIP_NAME = "ASCIIdata_splib07a.zip"
EXTRACTED_DIR = "ASCIIdata_splib07a"

SPECLIB_DIR = Path(__file__).resolve().parent.parent / "data" / "speclib"


def _get(url: str, timeout: int = 120):
    """Open a URL with the project User-Agent (ScienceBase 403s the default)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 (trusted USGS host)


def resolve_zip_url() -> str:
    """Resolve the splib07a zip download URL from the ScienceBase item by name."""
    api = f"https://www.sciencebase.gov/catalog/item/{SCIENCEBASE_ITEM}?format=json"
    with _get(api) as resp:
        item = json.load(resp)
    for f in item.get("files", []):
        if f.get("name") == ZIP_NAME:
            return f.get("downloadUri") or f["url"]
    raise KeyError(f"{ZIP_NAME} not found in ScienceBase item {SCIENCEBASE_ITEM}")


def download_zip(dest: Path) -> None:
    """Stream the zip to ``dest``, skipping a same-size existing file."""
    url = resolve_zip_url()
    with _get(url) as resp:
        remote_size = int(resp.headers.get("Content-Length", -1))
        if dest.exists() and dest.stat().st_size == remote_size:
            logger.info("skip %s (already present, %.1f MB)", dest.name, remote_size / 1e6)
            return
        logger.info("downloading %s (%.1f MB)", dest.name, remote_size / 1e6)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    tmp.replace(dest)


def main() -> None:
    SPECLIB_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SPECLIB_DIR / ZIP_NAME
    download_zip(zip_path)

    extracted = SPECLIB_DIR / EXTRACTED_DIR
    if extracted.exists():
        logger.info("already extracted at %s", extracted)
        return
    logger.info("extracting %s ...", zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(SPECLIB_DIR)
    logger.info("extracted to %s", extracted)


if __name__ == "__main__":
    main()
