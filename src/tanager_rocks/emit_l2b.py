"""Fail-closed EMIT L2B MIN/MINUNCERT ingestion and blocked concordance helpers.

The product fields and GLT semantics follow the EMIT L2BMIN User Guide V1.
This module deliberately contains no mineral-name heuristics and no quality
thresholds.  Ontology decisions and any future fit/uncertainty strata must be
supplied as versioned, source-supported inputs before outcomes are evaluated.
"""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import h5py
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from scipy.stats import spearmanr

from .spatial_validation import FINITE_REPLICATE_FRACTION, governance_status, rank_auc

EMIT_L2B_SHORT_NAME = "EMITL2BMIN"
GLT_FILL = 0
GROUPS = (1, 2)
ONTOLOGY_MAPPINGS = frozenset({"exact", "broader", "unmapped"})
EXPECTED_L2B_CRS = CRS.from_epsg(4326)
CRS_METADATA_ALIASES = (
    "spatial_ref",
    "crs",
    "crs_wkt",
    "coordinate_reference_system",
    "epsg",
    "epsg_code",
)
PRODUCT_PATTERN = re.compile(
    r"^EMIT_L2B_(?P<kind>MIN|MINUNCERT)_(?P<version>[A-Za-z0-9.-]+)_"
    r"(?P<acquisition>\d{8}T\d{6})_(?P<orbit>\d{7})_(?P<scene>\d{3})\.nc$"
)
L2A_PRODUCT_PATTERN = re.compile(
    r"^EMIT_L2A_RFL_(?P<version>[A-Za-z0-9.-]+)_"
    r"(?P<acquisition>\d{8}T\d{6})_(?P<orbit>\d{7})_(?P<scene>\d{3})\.nc$"
)
FLIGHT_LINE_PATTERN = re.compile(
    r"^emit(?P<acquisition>\d{8}t\d{6})_"
    r"o(?P<short_orbit>\d{5})_s(?P<s_token>\d{3})$"
)
PINNED_EMIT_L2A_INPUT_ID = "emit-goldfield-rfl"
REQUIRED_METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "product_version": ("product_version", "productVersion"),
    "time_coverage_start": ("time_coverage_start",),
    "time_coverage_end": ("time_coverage_end",),
    "flight_line": ("flight_line",),
}
ONTOLOGY_COLUMNS = (
    "ontology_version",
    "index",
    "name",
    "group",
    "library",
    "mapping",
    "target",
    "tanager_score",
    "source_path",
    "source_sha256",
    "evidence_id",
    "evidence_type",
    "evidence_locator",
    "unavailable_reason",
)
ONTOLOGY_EVIDENCE_SCHEMA = "emit-l2b-ontology-evidence/v3"
ONTOLOGY_AUTHORITY_CAPTURE_SCHEMA = "emit-l2b-authority-capture/v1"
ONTOLOGY_EVIDENCE_TYPES = frozenset(
    {
        "exact_name_equality",
        "explicit_broader_mapping",
        "unmapped_decision",
    }
)
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
PLACEHOLDER_AUTHORITY_HOSTS = frozenset(
    {
        "example.com",
        "example.net",
        "example.org",
        "localhost",
    }
)


class ProductMismatchError(ValueError):
    """Raised when the required L2B product pair cannot be proven identical."""


@dataclass(frozen=True)
class ProductIdentity:
    """Identity encoded in an EMIT L2B product filename."""

    kind: str
    version: str
    acquisition: str
    orbit: str
    scene: str


@dataclass(frozen=True)
class FilenameOrbitIdentity:
    """Mission components encoded by the seven-digit filename orbit."""

    year: str
    day_of_year: str
    daily_sequence: str


@dataclass(frozen=True)
class FlightLineIdentity:
    """Corroborating acquisition identity encoded by ``flight_line``."""

    value: str
    acquisition: str
    short_orbit: str
    s_token: str


@dataclass(frozen=True)
class PinnedEmitL2AInput:
    """Byte-verified local L2A input that fixes the E4 filename identity."""

    input_id: str
    logical_path: str
    path: Path
    size_bytes: int
    sha256: str
    identity: ProductIdentity


@dataclass(frozen=True)
class RasterGeometry:
    """North-up WGS84 output geometry carried by an EMIT GLT."""

    shape: tuple[int, int]
    transform: Affine
    crs: str = "EPSG:4326"


@dataclass(frozen=True)
class SourceMineral:
    """One exact record from ``mineral_metadata``."""

    index: int
    name: str
    group: int
    library: str


@dataclass(frozen=True)
class OntologyEntry:
    """One externally supplied, versioned ontology decision."""

    ontology_version: str
    index: int
    name: str
    group: int
    library: str
    mapping: str
    target: str
    tanager_score: str
    source_path: str
    source_sha256: str
    evidence_id: str
    evidence_type: str
    evidence_locator: str
    unavailable_reason: str | None


@dataclass(frozen=True)
class EmitL2BGroup:
    """Orthorectified identity, depth, uncertainty, and fit for one group."""

    group: int
    mineral_id: np.ndarray
    band_depth: np.ndarray
    uncertainty: np.ndarray
    fit: np.ndarray


@dataclass(frozen=True)
class EmitL2BSourcePair:
    """A byte-pinned, metadata/schema-validated MIN/MINUNCERT source packet."""

    identity: ProductIdentity
    geometry: RasterGeometry
    min_metadata: Mapping[str, Any]
    minuncert_metadata: Mapping[str, Any]
    min_path: Path
    minuncert_path: Path
    min_sha256: str
    minuncert_sha256: str


@dataclass(frozen=True)
class EmitL2BMetadataPacket:
    """Metadata-only EMIT L2B packet for endpoint-sealed non-result gates.

    This type deliberately excludes mineral identity, band-depth, fit, and
    uncertainty arrays.  It is safe for mapping-only work because its loader
    reads only declarations, GLTs, and the delivered mineral inventory.
    """

    source: EmitL2BSourcePair
    mineral_metadata: tuple[SourceMineral, ...]
    min_glt_x: np.ndarray
    min_glt_y: np.ndarray
    minuncert_glt_x: np.ndarray
    minuncert_glt_y: np.ndarray


@dataclass(frozen=True)
class EmitL2BPair:
    """A validated and orthorectified MIN/MINUNCERT product packet."""

    identity: ProductIdentity
    geometry: RasterGeometry
    groups: Mapping[int, EmitL2BGroup]
    mineral_metadata: tuple[SourceMineral, ...]
    min_metadata: Mapping[str, Any]
    minuncert_metadata: Mapping[str, Any]
    min_path: Path
    minuncert_path: Path
    min_sha256: str
    minuncert_sha256: str
    min_glt_x: np.ndarray
    min_glt_y: np.ndarray
    minuncert_glt_x: np.ndarray
    minuncert_glt_y: np.ndarray


@dataclass(frozen=True)
class EndpointMetrics:
    """Concordance metrics computed on one endpoint joint-support mask."""

    joint_support_n: int
    auc: float | None
    auc_n: int
    auc_positive: int
    auc_negative: int
    prevalence: float | None
    positive_blocks: int
    negative_blocks: int
    governance: str
    auc_unavailable_reason: str | None
    spearman: float | None
    spearman_n: int
    spearman_unavailable_reason: str | None


@dataclass(frozen=True)
class BootstrapDraw:
    """One complete-block bootstrap metric draw."""

    replicate: int
    metric: str
    value: float | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class SpatialNullDraw:
    """One whole-block spatial-null metric draw."""

    permutation: int
    metric: str
    value: float | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class BootstrapInterval:
    """Governed percentile interval for one complete-block bootstrap metric."""

    metric: str
    lower_95: float | None
    upper_95: float | None
    scheduled_replicates: int
    valid_replicates: int
    finite_fraction: float
    gate_eligible: bool
    unavailable_reason: str | None


@dataclass(frozen=True)
class SpatialNullSummary:
    """Governed one-sided whole-block null summary for greater concordance."""

    metric: str
    observed: float | None
    null_lower_95: float | None
    null_median: float | None
    null_upper_95: float | None
    p_value: float | None
    exceedances: int | None
    scheduled_permutations: int
    valid_permutations: int
    finite_fraction: float
    gate_eligible: bool
    unavailable_reason: str | None


@dataclass(frozen=True)
class BlockPacketDesign:
    """Prevalidated exchangeable complete-block packet layout."""

    block_ids: tuple[object, ...]
    packet_size: int


@dataclass(frozen=True)
class BlockFootprintSupport:
    """M2 assignment for complete L2B area-averaging footprints."""

    block_ids: np.ndarray
    crosses_block_boundary: np.ndarray
    incomplete_or_halo_support: np.ndarray


@dataclass(frozen=True)
class M2BlockScale:
    """One exact M2 complete-block raster and its provenance."""

    scale: str
    values: np.ndarray
    complete_block_ids: tuple[int, ...]
    transform: Affine
    crs: str
    source_path: Path
    source_sha256: str
    block_side_pixels: int
    halo_pixels: int


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Return a deterministic SHA-256 digest for a directory and its files."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"provenance directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"provenance directory contains no files: {root}")
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_acquisition_timestamp(value: str, *, field: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S")
    except ValueError as error:
        raise ProductMismatchError(
            f"{field} is not a valid acquisition timestamp: {value}"
        ) from error


def _filename_orbit_identity(identity: ProductIdentity, *, filename: str) -> FilenameOrbitIdentity:
    timestamp = _parse_acquisition_timestamp(identity.acquisition, field="filename acquisition")
    observed = FilenameOrbitIdentity(
        year=identity.orbit[:2],
        day_of_year=identity.orbit[2:5],
        daily_sequence=identity.orbit[5:7],
    )
    expected_year = f"{timestamp.year % 100:02d}"
    expected_day = f"{timestamp.timetuple().tm_yday:03d}"
    if observed.year != expected_year:
        raise ProductMismatchError(
            f"filename orbit year does not match acquisition for {filename}: "
            f"{observed.year!r} != {expected_year!r}"
        )
    if observed.day_of_year != expected_day:
        raise ProductMismatchError(
            f"filename orbit day of year does not match acquisition for {filename}: "
            f"{observed.day_of_year!r} != {expected_day!r}"
        )
    return observed


def parse_product_identity(
    path: str | Path, *, expected_kind: str | None = None
) -> ProductIdentity:
    """Parse a complete EMIT L2B filename without guessing omitted tokens."""
    match = PRODUCT_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ProductMismatchError(f"invalid EMIT L2B product filename: {Path(path).name}")
    identity = ProductIdentity(**match.groupdict())
    if expected_kind is not None and identity.kind != expected_kind:
        raise ProductMismatchError(
            f"expected {expected_kind} product, found {identity.kind}: {Path(path).name}"
        )
    _filename_orbit_identity(identity, filename=Path(path).name)
    return identity


def parse_l2a_product_identity(path: str | Path) -> ProductIdentity:
    """Parse the complete filename identity of the pinned EMIT L2A RFL input."""
    match = L2A_PRODUCT_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ProductMismatchError(f"invalid EMIT L2A RFL filename: {Path(path).name}")
    identity = ProductIdentity(kind="L2A_RFL", **match.groupdict())
    _filename_orbit_identity(identity, filename=Path(path).name)
    return identity


def parse_flight_line(value: Any) -> FlightLineIdentity:
    """Parse the exact delivered EMIT ``flight_line`` identity shape."""
    text = str(_decode(value)).strip()
    match = FLIGHT_LINE_PATTERN.fullmatch(text)
    if match is None:
        raise ProductMismatchError(f"flight_line metadata has an invalid shape: {text!r}")
    acquisition = match.group("acquisition").replace("t", "T")
    _parse_acquisition_timestamp(acquisition, field="flight_line acquisition")
    return FlightLineIdentity(
        value=text,
        acquisition=acquisition,
        short_orbit=match.group("short_orbit"),
        s_token=match.group("s_token"),
    )


def load_pinned_emit_l2a_input(manifest_path: str | Path) -> PinnedEmitL2AInput:
    """Verify and return the one local L2A record pinned for E4."""
    manifest = Path(manifest_path).resolve()
    try:
        payload = json.loads(
            manifest.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant in {manifest}: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"scientific input manifest is not valid JSON: {manifest}") from error
    if not isinstance(payload, dict) or payload.get("hash_algorithm") != "sha256":
        raise ValueError("scientific input manifest must declare SHA-256")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("scientific input manifest has no input records")
    matches = [
        record
        for record in inputs
        if isinstance(record, dict) and record.get("id") == PINNED_EMIT_L2A_INPUT_ID
    ]
    if len(matches) != 1:
        raise ValueError(
            f"scientific input manifest must contain exactly one {PINNED_EMIT_L2A_INPUT_ID!r}"
        )
    record = matches[0]
    logical_path = record.get("logical_path")
    if not isinstance(logical_path, str) or not logical_path or Path(logical_path).is_absolute():
        raise ValueError("pinned EMIT L2A logical path must be repository-relative")
    repository_root = manifest.parent.parent
    resolved = (repository_root / logical_path).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("pinned EMIT L2A logical path escapes the repository") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"pinned EMIT L2A input is missing: {resolved}")
    expected_size = record.get("size_bytes")
    expected_hash = record.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
    ):
        raise ValueError("pinned EMIT L2A record lacks a valid size or SHA-256")
    observed_size = resolved.stat().st_size
    if observed_size != expected_size or sha256_file(resolved) != expected_hash:
        raise ValueError(f"pinned EMIT L2A size or SHA-256 differs: {resolved}")
    return PinnedEmitL2AInput(
        input_id=PINNED_EMIT_L2A_INPUT_ID,
        logical_path=logical_path,
        path=resolved,
        size_bytes=observed_size,
        sha256=expected_hash,
        identity=parse_l2a_product_identity(resolved),
    )


def validate_l2b_identity_against_l2a(
    l2b_identity: ProductIdentity, l2a_identity: ProductIdentity
) -> None:
    """Require exact acquisition/orbit/scene filename identity across levels."""
    for field in ("acquisition", "orbit", "scene"):
        if getattr(l2b_identity, field) != getattr(l2a_identity, field):
            raise ProductMismatchError(
                f"L2B {field} filename identity differs from pinned L2A: "
                f"{getattr(l2b_identity, field)!r} != {getattr(l2a_identity, field)!r}"
            )


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode(value.item())
        return [_decode(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _decode(value.item())
    return value


def _global_metadata(handle: h5py.File) -> dict[str, Any]:
    return {str(key): _decode(value) for key, value in handle.attrs.items()}


def _metadata_value(metadata: Mapping[str, Any], field: str) -> Any:
    aliases = REQUIRED_METADATA_ALIASES[field]
    present = [(alias, metadata[alias]) for alias in aliases if alias in metadata]
    if not present:
        raise ProductMismatchError(f"required global metadata {field!r} is missing")
    normalized = {str(_decode(value)).strip() for _, value in present}
    if len(normalized) != 1:
        names = ", ".join(alias for alias, _ in present)
        raise ProductMismatchError(f"conflicting {field} metadata aliases: {names}")
    return present[0][1]


def _canonical_acquisition(value: Any) -> str:
    text = str(_decode(value)).strip()
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) < 14:
        raise ProductMismatchError(f"time coverage metadata is not a complete timestamp: {text}")
    canonical = f"{digits[:8]}T{digits[8:14]}"
    _parse_acquisition_timestamp(canonical, field="time coverage metadata")
    return canonical


def _canonical_version(value: Any) -> str:
    text = str(_decode(value)).strip()
    return text[1:] if text.lower().startswith("v") else text


def _validate_file_metadata(
    metadata: Mapping[str, Any], identity: ProductIdentity, path: Path
) -> FlightLineIdentity:
    expected: dict[str, Any] = {
        "product_version": _canonical_version(identity.version),
        "time_coverage_start": identity.acquisition,
    }
    observed: dict[str, Any] = {
        "product_version": _canonical_version(_metadata_value(metadata, "product_version")),
        "time_coverage_start": _canonical_acquisition(
            _metadata_value(metadata, "time_coverage_start")
        ),
    }
    for field in expected:
        if observed[field] != expected[field]:
            raise ProductMismatchError(
                f"{field} metadata does not match filename for {path.name}: "
                f"{observed[field]!r} != {expected[field]!r}"
            )
    _canonical_acquisition(_metadata_value(metadata, "time_coverage_end"))
    flight_line = parse_flight_line(_metadata_value(metadata, "flight_line"))
    if flight_line.acquisition != identity.acquisition:
        raise ProductMismatchError(
            f"flight_line acquisition does not match filename for {path.name}: "
            f"{flight_line.acquisition!r} != {identity.acquisition!r}"
        )
    if flight_line.short_orbit != identity.orbit[2:]:
        raise ProductMismatchError(
            f"flight_line short orbit does not match filename for {path.name}: "
            f"{flight_line.short_orbit!r} != {identity.orbit[2:]!r}"
        )
    granule_id = metadata.get("granule_id", metadata.get("GranuleUR"))
    if granule_id is not None and str(_decode(granule_id)).strip() != path.stem:
        raise ProductMismatchError(f"granule_id metadata does not match filename for {path.name}")
    return flight_line


def _validate_packet_metadata(
    min_metadata: Mapping[str, Any],
    min_identity: ProductIdentity,
    min_path: Path,
    minuncert_metadata: Mapping[str, Any],
    minuncert_identity: ProductIdentity,
    minuncert_path: Path,
) -> None:
    min_flight_line = _validate_file_metadata(min_metadata, min_identity, min_path)
    minuncert_flight_line = _validate_file_metadata(
        minuncert_metadata, minuncert_identity, minuncert_path
    )
    if min_flight_line != minuncert_flight_line:
        raise ProductMismatchError("MIN/MINUNCERT flight_line mismatch")
    for field in ("time_coverage_start", "time_coverage_end"):
        min_value = str(_decode(_metadata_value(min_metadata, field))).strip()
        minuncert_value = str(_decode(_metadata_value(minuncert_metadata, field))).strip()
        if min_value != minuncert_value:
            raise ProductMismatchError(f"MIN/MINUNCERT {field} mismatch")


def _verified_crs(metadata: Mapping[str, Any]) -> str:
    present = [(name, metadata[name]) for name in CRS_METADATA_ALIASES if name in metadata]
    if not present:
        raise ProductMismatchError(
            "L2B CRS cannot be verified from global metadata; expected one of "
            + ", ".join(CRS_METADATA_ALIASES)
        )
    parsed: list[tuple[str, CRS]] = []
    for name, raw in present:
        value = _decode(raw)
        if name in {"epsg", "epsg_code"}:
            text = str(value).strip()
            if not re.fullmatch(r"(?:EPSG:)?\d+", text, flags=re.IGNORECASE):
                raise ProductMismatchError(f"invalid {name} CRS metadata: {value!r}")
            value = f"EPSG:{re.sub(r'(?i)^EPSG:', '', text)}"
        try:
            parsed.append((name, CRS.from_user_input(value)))
        except Exception as error:
            raise ProductMismatchError(f"invalid {name} CRS metadata: {value!r}") from error
    if any(crs != parsed[0][1] for _, crs in parsed[1:]):
        raise ProductMismatchError("conflicting CRS metadata aliases")
    if parsed[0][1] != EXPECTED_L2B_CRS:
        raise ProductMismatchError(
            f"L2B GLT CRS must be EPSG:4326, found {parsed[0][1].to_string()}"
        )
    return EXPECTED_L2B_CRS.to_string()


def _geotransform(metadata: Mapping[str, Any], shape: tuple[int, int]) -> RasterGeometry:
    if "geotransform" not in metadata:
        raise ProductMismatchError("required global metadata 'geotransform' is missing")
    values = np.asarray(metadata["geotransform"], dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ProductMismatchError("geotransform must contain six finite values")
    x0, dx, x_rotation, y0, y_rotation, dy = values.tolist()
    if dx <= 0 or dy >= 0 or x_rotation != 0 or y_rotation != 0:
        raise ProductMismatchError("GLT geometry must be north-up with positive x and negative y")
    return RasterGeometry(
        shape=shape,
        transform=Affine.from_gdal(*values),
        crs=_verified_crs(metadata),
    )


def _declared_dataset_shape(handle: h5py.File, name: str) -> tuple[int, int]:
    if name not in handle or not isinstance(handle[name], h5py.Dataset):
        raise ProductMismatchError(f"required L2B field {name!r} is missing")
    dataset = handle[name]
    if dataset.ndim != 2 or any(size <= 0 for size in dataset.shape):
        raise ProductMismatchError(f"L2B field {name!r} must be a non-empty 2-D array")
    return int(dataset.shape[0]), int(dataset.shape[1])


def _declared_glt_shape(handle: h5py.File) -> tuple[int, int]:
    shapes = []
    for name in ("location/glt_x", "location/glt_y"):
        shape = _declared_dataset_shape(handle, name)
        dataset = handle[name]
        if not np.issubdtype(dataset.dtype, np.integer):
            raise ProductMismatchError(f"required GLT field {name!r} must contain integers")
        shapes.append(shape)
    if shapes[0] != shapes[1]:
        raise ProductMismatchError("GLT x/y field declarations must be aligned")
    return shapes[0]


def _validate_mineral_metadata_declarations(handle: h5py.File) -> None:
    shapes = []
    for name in ("index", "name", "group", "library"):
        path = f"mineral_metadata/{name}"
        if path not in handle or not isinstance(handle[path], h5py.Dataset):
            raise ProductMismatchError(f"required mineral metadata field {path!r} is missing")
        dataset = handle[path]
        if dataset.ndim != 1 or dataset.shape[0] <= 0:
            raise ProductMismatchError(
                f"mineral metadata field {path!r} must be a non-empty 1-D array"
            )
        shapes.append(dataset.shape)
    if len(set(shapes)) != 1:
        raise ProductMismatchError("mineral metadata field declarations have unequal lengths")


def _pair_filename_identities(
    min_path: Path, minuncert_path: Path
) -> tuple[ProductIdentity, ProductIdentity]:
    min_identity = parse_product_identity(min_path, expected_kind="MIN")
    minuncert_identity = parse_product_identity(minuncert_path, expected_kind="MINUNCERT")
    for field in ("version", "acquisition", "orbit", "scene"):
        if getattr(min_identity, field) != getattr(minuncert_identity, field):
            raise ProductMismatchError(f"MIN/MINUNCERT {field} mismatch")
    return min_identity, minuncert_identity


def validate_emit_l2b_source_pair(
    min_path: str | Path, minuncert_path: str | Path
) -> EmitL2BSourcePair:
    """Validate bytes, filename identity, metadata, and schema without array reads."""
    min_file = Path(min_path)
    minuncert_file = Path(minuncert_path)
    if not min_file.is_file() or not minuncert_file.is_file():
        missing = [str(path) for path in (min_file, minuncert_file) if not path.is_file()]
        raise FileNotFoundError(f"required EMIT L2B pair member missing: {', '.join(missing)}")
    min_identity, minuncert_identity = _pair_filename_identities(min_file, minuncert_file)

    with (
        h5py.File(min_file, "r") as min_handle,
        h5py.File(minuncert_file, "r") as minuncert_handle,
    ):
        min_metadata = _global_metadata(min_handle)
        minuncert_metadata = _global_metadata(minuncert_handle)
        _validate_packet_metadata(
            min_metadata,
            min_identity,
            min_file,
            minuncert_metadata,
            minuncert_identity,
            minuncert_file,
        )
        min_glt_shape = _declared_glt_shape(min_handle)
        minuncert_glt_shape = _declared_glt_shape(minuncert_handle)
        min_geometry = _geotransform(min_metadata, min_glt_shape)
        minuncert_geometry = _geotransform(minuncert_metadata, minuncert_glt_shape)
        if min_geometry != minuncert_geometry:
            raise ProductMismatchError("MIN/MINUNCERT GLT output geometry mismatch")

        min_shapes = {
            _declared_dataset_shape(min_handle, f"group_{group}_{field}")
            for group in GROUPS
            for field in ("mineral_id", "band_depth")
        }
        minuncert_shapes = {
            _declared_dataset_shape(minuncert_handle, f"group_{group}_{field}")
            for group in GROUPS
            for field in ("band_depth_unc", "fit")
        }
        if len(min_shapes) != 1 or len(minuncert_shapes) != 1:
            raise ProductMismatchError("L2B packet raw field declarations are not congruent")
        if min_shapes != minuncert_shapes:
            raise ProductMismatchError("MIN/MINUNCERT raw field declarations differ")
        _validate_mineral_metadata_declarations(min_handle)

    return EmitL2BSourcePair(
        identity=min_identity,
        geometry=min_geometry,
        min_metadata=min_metadata,
        minuncert_metadata=minuncert_metadata,
        min_path=min_file,
        minuncert_path=minuncert_file,
        min_sha256=sha256_file(min_file),
        minuncert_sha256=sha256_file(minuncert_file),
    )


def _read_glt(handle: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    for path in ("location/glt_x", "location/glt_y"):
        if path not in handle:
            raise ProductMismatchError(f"required GLT field {path!r} is missing")
    glt_x = np.asarray(handle["location/glt_x"][:])
    glt_y = np.asarray(handle["location/glt_y"][:])
    if glt_x.shape != glt_y.shape or glt_x.ndim != 2:
        raise ProductMismatchError("GLT x/y fields must be aligned two-dimensional arrays")
    if glt_x.size == 0:
        raise ProductMismatchError("GLT x/y fields cannot be empty")
    if not np.issubdtype(glt_x.dtype, np.integer) or not np.issubdtype(glt_y.dtype, np.integer):
        raise ProductMismatchError("GLT x/y fields must contain integer indices")
    x_index = glt_x.astype(np.int64)
    y_index = glt_y.astype(np.int64)
    x_fill = x_index == GLT_FILL
    y_fill = y_index == GLT_FILL
    if np.any(x_fill != y_fill):
        raise ProductMismatchError("GLT x/y fill locations must match exactly")
    if np.all(x_fill):
        raise ProductMismatchError("GLT contains no valid mapped cells")
    valid = ~x_fill
    if np.any(x_index[valid] < 1) or np.any(y_index[valid] < 1):
        raise ProductMismatchError("GLT contains negative or non-1-based valid indices")
    return x_index, y_index


def _read_field(handle: h5py.File, name: str) -> np.ndarray:
    if name not in handle:
        raise ProductMismatchError(f"required L2B field {name!r} is missing")
    dataset = handle[name]
    if dataset.ndim != 2:
        raise ProductMismatchError(f"L2B field {name!r} must be two-dimensional")
    values = np.asarray(dataset[:], dtype=float)
    for key in ("_FillValue", "missing_value"):
        if key in dataset.attrs:
            fill = float(np.asarray(dataset.attrs[key]).reshape(-1)[0])
            values[values == fill] = np.nan
    return values


def orthorectify_with_glt(raw: np.ndarray, glt_x: np.ndarray, glt_y: np.ndarray) -> np.ndarray:
    """Map a raw swath field through its 1-based GLT without interpolation."""
    source = np.asarray(raw, dtype=float)
    x_index = np.asarray(glt_x)
    y_index = np.asarray(glt_y)
    if source.ndim != 2 or x_index.shape != y_index.shape or x_index.ndim != 2:
        raise ValueError("raw field and aligned GLT x/y arrays must be two-dimensional")
    if not np.issubdtype(x_index.dtype, np.integer) or not np.issubdtype(y_index.dtype, np.integer):
        raise ValueError("GLT indices must be integers")
    x_fill = x_index == GLT_FILL
    y_fill = y_index == GLT_FILL
    if np.any(x_fill != y_fill):
        raise ProductMismatchError("GLT x/y fill locations must match exactly")
    valid = ~x_fill
    if not np.any(valid):
        raise ProductMismatchError("GLT contains no valid mapped cells")
    if np.any(x_index[valid] < 1) or np.any(y_index[valid] < 1):
        raise ProductMismatchError("GLT contains negative or non-1-based valid indices")
    if np.any(x_index[valid] > source.shape[1]) or np.any(y_index[valid] > source.shape[0]):
        raise ProductMismatchError("GLT references pixels outside the raw swath")
    output = np.full(x_index.shape, np.nan, dtype=float)
    output[valid] = source[y_index[valid] - 1, x_index[valid] - 1]
    return output


def _read_source_minerals(handle: h5py.File) -> tuple[SourceMineral, ...]:
    names = ("index", "name", "group", "library")
    paths = tuple(f"mineral_metadata/{name}" for name in names)
    missing = [path for path in paths if path not in handle]
    if missing:
        raise ProductMismatchError(f"missing mineral metadata fields: {', '.join(missing)}")
    columns = [np.asarray(handle[path][:]) for path in paths]
    lengths = {column.shape[0] for column in columns if column.ndim == 1}
    if len(lengths) != 1 or any(column.ndim != 1 for column in columns):
        raise ProductMismatchError("mineral metadata fields must be aligned one-dimensional arrays")
    if not np.issubdtype(columns[0].dtype, np.integer) or not np.issubdtype(
        columns[2].dtype, np.integer
    ):
        raise ProductMismatchError("mineral metadata index/group fields must contain integers")
    records = tuple(
        SourceMineral(
            index=int(index),
            name=str(_decode(name)).strip(),
            group=int(group),
            library=str(_decode(library)).strip(),
        )
        for index, name, group, library in zip(*columns, strict=True)
    )
    if any(record.group not in GROUPS for record in records):
        raise ProductMismatchError("mineral metadata group must be 1 or 2")
    if any(record.index <= 0 for record in records):
        raise ProductMismatchError("mineral metadata indices must be positive integers")
    if any(not record.name or not record.library for record in records):
        raise ProductMismatchError("mineral metadata names and libraries must be non-empty")
    keys = {(record.group, record.index) for record in records}
    if len(keys) != len(records):
        raise ProductMismatchError("mineral metadata contains duplicate (group, index) records")
    if {record.group for record in records} != set(GROUPS):
        raise ProductMismatchError("mineral metadata must contain records for groups 1 and 2")
    return records


def load_emit_l2b_metadata(
    min_path: str | Path,
    minuncert_path: str | Path,
) -> EmitL2BMetadataPacket:
    """Load only L2B declarations, GLTs, and mineral metadata.

    Unlike :func:`load_emit_l2b_pair`, this function never calls
    :func:`_read_field` and therefore never reads mineral-ID, band-depth, fit,
    or uncertainty result arrays.  It supplies the explicit metadata boundary
    required by E4 mapping-only validation.
    """
    source = validate_emit_l2b_source_pair(min_path, minuncert_path)
    with (
        h5py.File(source.min_path, "r") as min_handle,
        h5py.File(source.minuncert_path, "r") as uncertainty_handle,
    ):
        min_glt_x, min_glt_y = _read_glt(min_handle)
        uncertainty_glt_x, uncertainty_glt_y = _read_glt(uncertainty_handle)
        mineral_metadata = _read_source_minerals(min_handle)
    return EmitL2BMetadataPacket(
        source=source,
        mineral_metadata=mineral_metadata,
        min_glt_x=min_glt_x,
        min_glt_y=min_glt_y,
        minuncert_glt_x=uncertainty_glt_x,
        minuncert_glt_y=uncertainty_glt_y,
    )


def load_emit_l2b_pair(min_path: str | Path, minuncert_path: str | Path) -> EmitL2BPair:
    """Validate and orthorectify a matched L2B MIN/MINUNCERT pair.

    Both files are required.  Filename identity and required global metadata
    must agree exactly after only syntax normalization.  Each file's own GLT is
    then applied to its own fields; categorical IDs are never interpolated.
    """
    min_file = Path(min_path)
    uncertainty_file = Path(minuncert_path)
    if not min_file.is_file() or not uncertainty_file.is_file():
        missing = [str(path) for path in (min_file, uncertainty_file) if not path.is_file()]
        raise FileNotFoundError(f"required EMIT L2B pair member missing: {', '.join(missing)}")
    min_identity, uncertainty_identity = _pair_filename_identities(min_file, uncertainty_file)

    with (
        h5py.File(min_file, "r") as min_handle,
        h5py.File(uncertainty_file, "r") as uncertainty_handle,
    ):
        min_metadata = _global_metadata(min_handle)
        uncertainty_metadata = _global_metadata(uncertainty_handle)
        _validate_packet_metadata(
            min_metadata,
            min_identity,
            min_file,
            uncertainty_metadata,
            uncertainty_identity,
            uncertainty_file,
        )
        min_glt_x, min_glt_y = _read_glt(min_handle)
        uncertainty_glt_x, uncertainty_glt_y = _read_glt(uncertainty_handle)
        min_geometry = _geotransform(min_metadata, min_glt_x.shape)
        uncertainty_geometry = _geotransform(uncertainty_metadata, uncertainty_glt_x.shape)
        if min_geometry != uncertainty_geometry:
            raise ProductMismatchError("MIN/MINUNCERT GLT output geometry mismatch")
        source_minerals = _read_source_minerals(min_handle)
        groups: dict[int, EmitL2BGroup] = {}
        for group in GROUPS:
            mineral_id = _read_field(min_handle, f"group_{group}_mineral_id")
            band_depth = _read_field(min_handle, f"group_{group}_band_depth")
            uncertainty = _read_field(uncertainty_handle, f"group_{group}_band_depth_unc")
            fit = _read_field(uncertainty_handle, f"group_{group}_fit")
            if mineral_id.shape != band_depth.shape:
                raise ProductMismatchError(f"MIN group {group} raw field shapes differ")
            if uncertainty.shape != fit.shape:
                raise ProductMismatchError(f"MINUNCERT group {group} raw field shapes differ")
            finite_ids = mineral_id[np.isfinite(mineral_id)]
            if np.any(finite_ids != np.floor(finite_ids)):
                raise ProductMismatchError(f"MIN group {group} contains non-integer mineral IDs")
            if np.any(finite_ids < 0):
                raise ProductMismatchError(f"MIN group {group} contains negative mineral IDs")
            documented_ids = {record.index for record in source_minerals if record.group == group}
            unknown_ids = {
                int(value) for value in np.unique(finite_ids) if value > 0
            } - documented_ids
            if unknown_ids:
                raise ProductMismatchError(
                    f"MIN group {group} IDs are absent from mineral metadata: "
                    f"{sorted(unknown_ids)!r}"
                )
            groups[group] = EmitL2BGroup(
                group=group,
                mineral_id=orthorectify_with_glt(mineral_id, min_glt_x, min_glt_y),
                band_depth=orthorectify_with_glt(band_depth, min_glt_x, min_glt_y),
                uncertainty=orthorectify_with_glt(
                    uncertainty, uncertainty_glt_x, uncertainty_glt_y
                ),
                fit=orthorectify_with_glt(fit, uncertainty_glt_x, uncertainty_glt_y),
            )

    return EmitL2BPair(
        identity=min_identity,
        geometry=min_geometry,
        groups=groups,
        mineral_metadata=source_minerals,
        min_metadata=min_metadata,
        minuncert_metadata=uncertainty_metadata,
        min_path=min_file,
        minuncert_path=uncertainty_file,
        min_sha256=sha256_file(min_file),
        minuncert_sha256=sha256_file(uncertainty_file),
        min_glt_x=min_glt_x,
        min_glt_y=min_glt_y,
        minuncert_glt_x=uncertainty_glt_x,
        minuncert_glt_y=uncertainty_glt_y,
    )


def l2b_identity_evidence(pair: EmitL2BSourcePair | EmitL2BPair) -> dict[str, Any]:
    """Return the closed v4 filename/flight-line identity evidence object."""
    orbit = _filename_orbit_identity(pair.identity, filename=pair.min_path.name)
    min_flight_line = parse_flight_line(_metadata_value(pair.min_metadata, "flight_line"))
    minuncert_flight_line = parse_flight_line(
        _metadata_value(pair.minuncert_metadata, "flight_line")
    )
    if min_flight_line != minuncert_flight_line:
        raise ProductMismatchError("MIN/MINUNCERT flight_line mismatch")
    return {
        "filename_orbit_year": orbit.year,
        "filename_orbit_doy": orbit.day_of_year,
        "filename_orbit_daily_sequence": orbit.daily_sequence,
        "min_flight_line": min_flight_line.value,
        "minuncert_flight_line": minuncert_flight_line.value,
        "flight_line_acquisition": min_flight_line.acquisition,
        "flight_line_short_orbit": min_flight_line.short_orbit,
        "flight_line_s_token": min_flight_line.s_token,
        "delivered_scene_source": "filename_and_cmr",
        "flight_line_s_token_is_delivered_scene": False,
    }


def read_ontology_crosswalk(path: str | Path) -> tuple[OntologyEntry, ...]:
    """Read the strict E4 ontology CSV; no aliases or inferred fields are accepted."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ONTOLOGY_COLUMNS:
            raise ValueError(
                "ontology CSV columns must exactly equal: " + ", ".join(ONTOLOGY_COLUMNS)
            )
        entries = []
        for row_number, row in enumerate(reader, start=2):
            try:
                group = int(row["group"])
                index = int(row["index"])
            except ValueError as error:
                raise ValueError(
                    f"ontology row {row_number} index/group must be integers"
                ) from error
            entries.append(
                OntologyEntry(
                    ontology_version=row["ontology_version"].strip(),
                    index=index,
                    name=row["name"].strip(),
                    group=group,
                    library=row["library"].strip(),
                    mapping=row["mapping"].strip(),
                    target=row["target"].strip(),
                    tanager_score=row["tanager_score"].strip(),
                    source_path=row["source_path"].strip(),
                    source_sha256=row["source_sha256"].strip().lower(),
                    evidence_id=row["evidence_id"].strip(),
                    evidence_type=row["evidence_type"].strip(),
                    evidence_locator=row["evidence_locator"].strip(),
                    unavailable_reason=row["unavailable_reason"].strip() or None,
                )
            )
    return tuple(entries)


def _ontology_evidence_decisions(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant in ontology evidence: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"ontology evidence must be structured JSON: {path}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "decisions"}:
        raise ValueError("ontology evidence must contain only schema_version and decisions")
    if payload["schema_version"] != ONTOLOGY_EVIDENCE_SCHEMA:
        raise ValueError("ontology evidence has an unsupported schema_version")
    decisions = payload["decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("ontology evidence must contain row-specific decisions")
    required = {
        "authority_content_path",
        "authority_content_sha256",
        "authority_field_path",
        "evidence_id",
        "evidence_type",
        "evidence_locator",
        "evidence_assertion",
        "index",
        "name",
        "group",
        "library",
        "mapping",
        "target",
        "tanager_score",
        "unavailable_reason",
    }
    indexed: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != required:
            raise ValueError("every ontology evidence decision must use the exact frozen schema")
        evidence_id = decision["evidence_id"]
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in indexed:
            raise ValueError("ontology evidence IDs must be non-empty and unique")
        indexed[evidence_id] = decision
    return indexed


def _normalize_ontology_label(value: str) -> str:
    """Normalize only Unicode, case, and whitespace for literal-name equality."""
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def _validate_external_authority_locator(locator: str) -> None:
    """Accept only non-placeholder HTTPS citations or syntactically valid DOIs."""
    text = locator.strip()
    lowered = text.casefold()
    if "synthetic" in lowered or "placeholder" in lowered:
        raise ValueError("ontology authority locator cannot be synthetic or a placeholder")
    if lowered.startswith("doi:"):
        doi = text[4:].strip()
        if not DOI_PATTERN.fullmatch(doi):
            raise ValueError("ontology authority DOI is invalid")
        return

    parsed = urlsplit(text)
    host = (parsed.hostname or "").casefold().rstrip(".")
    local_hostname = (
        host == "localhost"
        or host.startswith("localhost.")
        or host.endswith(".localhost")
        or host.endswith(".localdomain")
    )
    if (
        parsed.scheme.casefold() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path
        or host in PLACEHOLDER_AUTHORITY_HOSTS
        or local_hostname
        or host.endswith((".invalid", ".local", ".test"))
        or host.endswith((".localhost", ".internal", ".home.arpa"))
    ):
        raise ValueError("nonliteral ontology mappings require a non-placeholder HTTPS URL or DOI")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        numeric_labels = host.split(".")
        if all(re.fullmatch(r"(?:[0-9]+|0x[0-9a-f]+)", label) for label in numeric_labels):
            raise ValueError("ontology authority cannot use a noncanonical numeric IP hostname")
        if "." not in host:
            raise ValueError("ontology authority must use a public fully-qualified host")
    else:
        if not address.is_global:
            raise ValueError("ontology authority cannot use a non-public IP address")
    if host in {"doi.org", "dx.doi.org"}:
        doi = unquote(parsed.path.lstrip("/"))
        if not DOI_PATTERN.fullmatch(doi):
            raise ValueError("ontology authority DOI URL is invalid")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while refusing duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"ontology authority capture has duplicate JSON key {key!r}")
        result[key] = value
    return result


def _authority_capture_decisions(
    evidence_path: Path,
    authority_content_path: Any,
    authority_sha256: str,
) -> tuple[str, dict[str, dict[str, str]]]:
    """Load one byte-pinned, strict structured capture of a cited public authority."""
    if not isinstance(authority_content_path, str) or not authority_content_path.strip():
        raise ValueError("nonliteral ontology evidence requires an authority content path")
    relative = Path(authority_content_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("ontology authority content path must stay beside the evidence packet")
    authority_path = (evidence_path.parent / relative).resolve()
    evidence_root = evidence_path.parent.resolve()
    if evidence_root not in authority_path.parents:
        raise ValueError("ontology authority content path escapes the evidence packet")
    if not authority_path.is_file():
        raise FileNotFoundError(f"ontology authority content is missing: {authority_path}")
    if sha256_file(authority_path) != authority_sha256:
        raise ValueError(f"ontology authority content SHA-256 mismatch: {authority_path}")
    try:
        text = authority_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("ontology authority capture must be UTF-8 JSON") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant in ontology authority capture: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"ontology authority capture must be structured JSON: {authority_path}"
        ) from error
    required_payload_fields = {"schema_version", "source_locator", "decisions"}
    if not isinstance(payload, dict) or set(payload) != required_payload_fields:
        raise ValueError("ontology authority capture must use the exact frozen schema")
    if payload["schema_version"] != ONTOLOGY_AUTHORITY_CAPTURE_SCHEMA:
        raise ValueError("ontology authority capture has an unsupported schema_version")
    source_locator = payload["source_locator"]
    if not isinstance(source_locator, str):
        raise ValueError("ontology authority capture source_locator must be a string")
    decisions = payload["decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("ontology authority capture must contain structured decisions")
    required_decision_fields = {"field_path", "source", "relation", "target"}
    indexed: dict[str, dict[str, str]] = {}
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != required_decision_fields:
            raise ValueError("every ontology authority decision must use the exact frozen schema")
        if any(not isinstance(decision[field], str) for field in required_decision_fields):
            raise ValueError("ontology authority decision fields must all be strings")
        field_path = decision["field_path"]
        if not field_path or field_path in indexed:
            raise ValueError(
                "ontology authority decision field_path values must be non-empty and unique"
            )
        indexed[field_path] = decision
    return source_locator, indexed


def _validate_ontology_evidence_decision(
    record: OntologyEntry,
    decision: Mapping[str, Any],
    *,
    evidence_path: Path,
) -> None:
    expected = {
        "evidence_id": record.evidence_id,
        "evidence_type": record.evidence_type,
        "evidence_locator": record.evidence_locator,
        "index": record.index,
        "name": record.name,
        "group": record.group,
        "library": record.library,
        "mapping": record.mapping,
        "target": record.target,
        "tanager_score": record.tanager_score,
        "unavailable_reason": record.unavailable_reason,
    }
    observed = {key: decision.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"ontology evidence {record.evidence_id!r} does not exactly bind its source row, "
            "target, score, and mapping decision"
        )
    assertion = decision.get("evidence_assertion")
    if not isinstance(assertion, str) or not assertion.strip():
        raise ValueError("ontology evidence decisions require a non-empty evidence_assertion")
    if record.evidence_type not in ONTOLOGY_EVIDENCE_TYPES:
        raise ValueError(f"unsupported ontology evidence type {record.evidence_type!r}")
    authority_sha256 = decision.get("authority_content_sha256")
    authority_content_path = decision.get("authority_content_path")
    authority_field_path = decision.get("authority_field_path")
    source_label = _normalize_ontology_label(record.name)
    target_label = _normalize_ontology_label(record.target)
    if record.mapping == "exact":
        if (
            record.evidence_type != "exact_name_equality"
            or source_label != target_label
            or record.evidence_locator != "mechanical:source_name_equals_target"
            or assertion != "normalized_source_name == normalized_target"
            or authority_content_path is not None
            or authority_sha256 is not None
            or authority_field_path is not None
        ):
            raise ValueError(
                "exact mappings require normalized literal source-name/target equality"
            )
    elif record.mapping == "broader":
        if record.evidence_type != "explicit_broader_mapping":
            raise ValueError("broader mappings require explicit row-specific evidence")
        if source_label == target_label:
            raise ValueError("literal-equal ontology labels must use an exact mapping")
        _validate_external_authority_locator(record.evidence_locator)
        if not isinstance(authority_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", authority_sha256
        ):
            raise ValueError(
                "nonliteral ontology evidence requires a pinned authority content SHA-256"
            )
        if not isinstance(authority_field_path, str) or not authority_field_path.strip():
            raise ValueError(
                "nonliteral ontology evidence requires a row-specific authority field path"
            )
        expected_assertion = f"{record.name} maps_to {record.target}"
        if assertion != expected_assertion:
            raise ValueError(
                "nonliteral ontology evidence assertion must use the positive canonical "
                "'<source> maps_to <target>' form"
            )
        source_locator, authority_decisions = _authority_capture_decisions(
            evidence_path,
            authority_content_path,
            authority_sha256,
        )
        if source_locator != record.evidence_locator:
            raise ValueError("ontology authority capture must name its exact external locator")
        authority_decision = authority_decisions.get(authority_field_path)
        if authority_decision is None:
            raise ValueError(
                "ontology authority capture does not contain the exact authority field path"
            )
        expected_authority_decision = {
            "field_path": authority_field_path,
            "source": record.name,
            "relation": "maps_to",
            "target": record.target,
        }
        if authority_decision != expected_authority_decision:
            raise ValueError(
                "ontology authority decision must exactly bind source, maps_to relation, and target"
            )
    else:
        if (
            record.evidence_type != "unmapped_decision"
            or not record.evidence_locator.startswith("schema-audit:unmapped:")
            or authority_content_path is not None
            or authority_sha256 is not None
            or authority_field_path is not None
        ):
            raise ValueError("unmapped inventory rows require a non-claiming schema-audit decision")
        expected_assertion = f"{record.name} is_unmapped"
        if assertion != expected_assertion:
            raise ValueError(
                "schema-audit unmapped evidence assertion must use the positive canonical "
                "'<source> is_unmapped' form"
            )


def validate_ontology_crosswalk(
    entries: Sequence[OntologyEntry],
    source_minerals: Sequence[SourceMineral],
    *,
    source_root: str | Path | None = None,
) -> tuple[OntologyEntry, ...]:
    """Require one exact, byte-identified source decision per metadata row."""
    records = tuple(entries)
    if not records:
        raise ValueError("ontology crosswalk is empty")
    versions = {record.ontology_version for record in records if record.ontology_version}
    if len(versions) != 1 or any(not record.ontology_version for record in records):
        raise ValueError("ontology crosswalk must contain one non-empty ontology_version")
    evidence_root = Path.cwd() if source_root is None else Path(source_root)
    verified_evidence: dict[Path, tuple[str, dict[str, dict[str, Any]]]] = {}
    used_evidence_ids: set[tuple[Path, str]] = set()
    for record in records:
        if record.group not in GROUPS or record.index <= 0:
            raise ValueError("ontology group/index values must be valid positive product IDs")
        if record.mapping not in ONTOLOGY_MAPPINGS:
            raise ValueError(f"unsupported ontology mapping {record.mapping!r}")
        if (
            not record.source_path
            or not re.fullmatch(r"[0-9a-f]{64}", record.source_sha256)
            or not record.evidence_id
            or not record.evidence_locator
        ):
            raise ValueError("every ontology row requires a source_path and SHA-256")
        evidence_path = Path(record.source_path)
        if not evidence_path.is_absolute():
            evidence_path = evidence_root / evidence_path
        evidence_path = evidence_path.resolve()
        if not evidence_path.is_file():
            raise FileNotFoundError(f"ontology source evidence is missing: {evidence_path}")
        if evidence_path not in verified_evidence:
            verified_evidence[evidence_path] = (
                sha256_file(evidence_path),
                _ontology_evidence_decisions(evidence_path),
            )
        observed_sha256, decisions = verified_evidence[evidence_path]
        if observed_sha256 != record.source_sha256:
            raise ValueError(f"ontology source SHA-256 mismatch: {evidence_path}")
        decision_key = (evidence_path, record.evidence_id)
        if decision_key in used_evidence_ids:
            raise ValueError("ontology evidence IDs may authorize only one crosswalk row")
        used_evidence_ids.add(decision_key)
        decision = decisions.get(record.evidence_id)
        if decision is None:
            raise ValueError(f"ontology evidence ID is absent: {record.evidence_id!r}")
        _validate_ontology_evidence_decision(record, decision, evidence_path=evidence_path)
        if record.mapping in {"exact", "broader"}:
            if not record.target or not record.tanager_score:
                raise ValueError("mapped ontology rows require target and tanager_score")
            if record.unavailable_reason is not None:
                raise ValueError("mapped ontology rows cannot carry an unavailable_reason")
        else:
            if record.target or record.tanager_score:
                raise ValueError("unmapped ontology rows cannot name a target or tanager_score")
            if not record.unavailable_reason:
                raise ValueError("unmapped ontology rows require an unavailable_reason")

    for evidence_path, (_, decisions) in verified_evidence.items():
        used = {evidence_id for path, evidence_id in used_evidence_ids if path == evidence_path}
        if used != set(decisions):
            raise ValueError(
                f"ontology evidence contains unused or missing decisions: {evidence_path}"
            )

    source_keys = {
        (record.index, record.name, record.group, record.library) for record in source_minerals
    }
    ontology_keys = {
        (record.index, record.name, record.group, record.library) for record in records
    }
    ontology_group_indices = {(record.group, record.index) for record in records}
    if len(ontology_group_indices) != len(records):
        raise ValueError("ontology crosswalk contains duplicate (group, index) records")
    if len(ontology_keys) != len(records):
        raise ValueError("ontology crosswalk contains duplicate exact source records")
    if ontology_keys != source_keys:
        missing = source_keys - ontology_keys
        extra = ontology_keys - source_keys
        raise ValueError(
            "ontology rows must exactly match downloaded mineral metadata; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    return records


def reproject_categorical_nearest(
    values: np.ndarray,
    *,
    source_transform: Affine,
    source_crs: str,
    destination_shape: tuple[int, int],
    destination_transform: Affine,
    destination_crs: str,
    nodata: int | float,
) -> np.ndarray:
    """Reproject a categorical raster with nearest-neighbour only."""
    source = np.asarray(values)
    if source.ndim != 2:
        raise ValueError("categorical source must be two-dimensional")
    destination = np.full(destination_shape, nodata, dtype=source.dtype)
    reproject(
        source=source,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=nodata,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        dst_nodata=nodata,
        resampling=Resampling.nearest,
    )
    return destination


def area_average_continuous(
    values: np.ndarray,
    *,
    source_transform: Affine,
    source_crs: str,
    destination: RasterGeometry,
) -> np.ndarray:
    """Aggregate a continuous Tanager field to L2B support by area averaging."""
    source = np.asarray(values, dtype=float)
    if source.ndim != 2:
        raise ValueError("continuous source must be two-dimensional")
    output = np.full(destination.shape, np.nan, dtype=np.float64)
    reproject(
        source=source,
        destination=output,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=np.nan,
        dst_transform=destination.transform,
        dst_crs=destination.crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return output


def _valid_block_mask(block_ids: np.ndarray) -> np.ndarray:
    blocks = np.asarray(block_ids, dtype=object)
    valid = np.ones(blocks.shape, dtype=bool)
    for index, value in np.ndenumerate(blocks):
        if value is None or value == "" or value == 0:
            valid[index] = False
        elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            valid[index] = False
    return valid


def joint_support_mask(
    *,
    score: np.ndarray,
    mineral_id: np.ndarray,
    band_depth: np.ndarray,
    qa_valid: np.ndarray,
    glt_valid: np.ndarray,
    block_ids: np.ndarray,
) -> np.ndarray:
    """Return the single governed endpoint/scale support intersection."""
    arrays = tuple(
        np.asarray(value)
        for value in (score, mineral_id, band_depth, qa_valid, glt_valid, block_ids)
    )
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError("joint-support fields must have identical shapes")
    score_values = np.asarray(score, dtype=float)
    identity_values = np.asarray(mineral_id, dtype=float)
    depth_values = np.asarray(band_depth, dtype=float)
    finite_identity = identity_values[np.isfinite(identity_values)]
    if np.any(finite_identity < 0) or np.any(finite_identity != np.floor(finite_identity)):
        raise ValueError("joint-support mineral IDs must be non-negative integers or missing")
    return (
        np.isfinite(score_values)
        & np.isfinite(identity_values)
        & np.isfinite(depth_values)
        & np.asarray(qa_valid, dtype=bool)
        & np.asarray(glt_valid, dtype=bool)
        & _valid_block_mask(np.asarray(block_ids, dtype=object))
    )


def compute_endpoint_metrics(
    *,
    score: np.ndarray,
    mineral_id: np.ndarray,
    band_depth: np.ndarray,
    target_ids: frozenset[int],
    block_ids: np.ndarray,
    joint_support: np.ndarray | None = None,
) -> EndpointMetrics:
    """Compute AUC and target-depth Spearman from one joint-support mask."""
    score_values = np.asarray(score, dtype=float).reshape(-1)
    identity_values = np.asarray(mineral_id, dtype=float).reshape(-1)
    depth_values = np.asarray(band_depth, dtype=float).reshape(-1)
    blocks = np.asarray(block_ids, dtype=object).reshape(-1)
    if not target_ids or any(value <= 0 for value in target_ids):
        raise ValueError("target_ids must contain positive mineral IDs")
    if not (score_values.size == identity_values.size == depth_values.size == blocks.size):
        raise ValueError("score, L2B fields, and block IDs must be aligned")
    intrinsic = joint_support_mask(
        score=score_values,
        mineral_id=identity_values,
        band_depth=depth_values,
        qa_valid=np.ones(score_values.shape, dtype=bool),
        glt_valid=np.ones(score_values.shape, dtype=bool),
        block_ids=blocks,
    )
    if joint_support is None:
        support = intrinsic
    else:
        support = np.asarray(joint_support, dtype=bool).reshape(-1)
        if support.size != score_values.size:
            raise ValueError("joint_support must align with endpoint fields")
        if np.any(support & ~intrinsic):
            raise ValueError("joint_support includes an invalid metric observation")
    auc_scores = score_values[support]
    identity = identity_values[support]
    auc_reference = np.isin(identity, tuple(target_ids)).astype(np.int8)
    auc_n = int(auc_scores.size)
    n_positive = int(np.count_nonzero(auc_reference == 1))
    n_negative = int(np.count_nonzero(auc_reference == 0))
    if n_positive and n_negative:
        auc = float(rank_auc(auc_scores, auc_reference))
        auc_reason = None
    else:
        auc = None
        auc_reason = "auc_requires_at_least_one_positive_and_one_negative_cell"
    prevalence = float(n_positive / auc_n) if auc_n else None
    auc_blocks = blocks[support]
    positive_blocks = len(set(auc_blocks[auc_reference == 1].tolist()))
    negative_blocks = len(set(auc_blocks[auc_reference == 0].tolist()))

    target_identity = np.isin(identity_values, tuple(target_ids))
    spearman_mask = support & target_identity
    spearman_scores = score_values[spearman_mask]
    spearman_depth = depth_values[spearman_mask]
    spearman_n = int(spearman_scores.size)
    if spearman_n < 2:
        correlation = None
        spearman_reason = "spearman_requires_at_least_two_joint_support_target_cells"
    elif np.unique(spearman_scores).size < 2 or np.unique(spearman_depth).size < 2:
        correlation = None
        spearman_reason = "spearman_undefined_for_constant_input"
    else:
        correlation = float(spearmanr(spearman_scores, spearman_depth).statistic)
        spearman_reason = None if math.isfinite(correlation) else "spearman_returned_nonfinite"
        if spearman_reason is not None:
            correlation = None

    return EndpointMetrics(
        joint_support_n=auc_n,
        auc=auc,
        auc_n=auc_n,
        auc_positive=n_positive,
        auc_negative=n_negative,
        prevalence=prevalence,
        positive_blocks=positive_blocks,
        negative_blocks=negative_blocks,
        governance=governance_status(positive_blocks, negative_blocks),
        auc_unavailable_reason=auc_reason,
        spearman=correlation,
        spearman_n=spearman_n,
        spearman_unavailable_reason=spearman_reason,
    )


def _ordered_block_ids(block_ids: np.ndarray) -> tuple[object, ...]:
    return tuple(dict.fromkeys(np.asarray(block_ids, dtype=object).reshape(-1).tolist()))


def validate_exchangeable_block_packets(
    fields: Mapping[str, np.ndarray],
    block_ids: np.ndarray,
    *,
    require_multiple: bool = True,
) -> BlockPacketDesign:
    """Require congruent, uniquely located complete packets in one fixed order."""
    blocks = np.asarray(block_ids, dtype=object).reshape(-1)
    if blocks.size == 0:
        raise ValueError("block packet design is empty")
    if not np.all(_valid_block_mask(blocks)):
        raise ValueError("block packet design contains empty, nodata, or nonfinite IDs")
    ordered = _ordered_block_ids(blocks)
    if require_multiple and len(ordered) < 2:
        raise ValueError("whole-block nulls require at least two complete block packets")
    positions = [np.flatnonzero(blocks == block_id) for block_id in ordered]
    packet_sizes = {int(position.size) for position in positions}
    if len(packet_sizes) != 1 or not packet_sizes or next(iter(packet_sizes)) == 0:
        raise ValueError("complete block packets must have one equal, nonzero observation count")
    missing_coordinates = sorted({"x", "y"} - set(fields))
    if missing_coordinates:
        raise ValueError(
            "complete block packet validation requires supplied x/y coordinates; missing "
            + ", ".join(missing_coordinates)
        )
    for name, values in fields.items():
        source = np.asarray(values)
        if source.ndim == 0 or source.shape[0] != blocks.size:
            raise ValueError(f"field {name!r} is not aligned on the packet dimension")
    x_coordinates = np.asarray(fields["x"], dtype=float)
    y_coordinates = np.asarray(fields["y"], dtype=float)
    if x_coordinates.ndim != 1 or y_coordinates.ndim != 1:
        raise ValueError("packet x/y coordinates must be one-dimensional")
    reference_footprint: np.ndarray | None = None
    for block_id, position in zip(ordered, positions, strict=True):
        coordinates = np.column_stack((x_coordinates[position], y_coordinates[position]))
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"complete block packet {block_id!r} has nonfinite coordinates")
        if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
            raise ValueError(f"complete block packet {block_id!r} has duplicate coordinates")
        relative_footprint = coordinates - coordinates[0]
        if reference_footprint is None:
            reference_footprint = relative_footprint
        elif not np.array_equal(relative_footprint, reference_footprint):
            raise ValueError(
                "complete block packets must have the same unique relative footprint "
                "geometry and consistent ordering"
            )
    return BlockPacketDesign(block_ids=ordered, packet_size=next(iter(packet_sizes)))


def permute_complete_block_packet(
    fields: Mapping[str, np.ndarray],
    block_ids: np.ndarray,
    *,
    permutation: Sequence[int],
) -> dict[str, np.ndarray]:
    """Permute complete L2B block packets while retaining all within-block fields."""
    blocks = np.asarray(block_ids, dtype=object).reshape(-1)
    design = validate_exchangeable_block_packets(fields, blocks, require_multiple=False)
    ordered = design.block_ids
    order = np.asarray(permutation)
    if (
        order.ndim != 1
        or order.size != len(ordered)
        or sorted(order.tolist()) != list(range(len(ordered)))
    ):
        raise ValueError("permutation must contain every complete block index exactly once")
    positions = [np.flatnonzero(blocks == block_id) for block_id in ordered]
    output: dict[str, np.ndarray] = {}
    for name, values in fields.items():
        source = np.asarray(values)
        permuted = np.empty_like(source)
        for target_index, source_index in enumerate(order):
            target_positions = positions[target_index]
            source_positions = positions[int(source_index)]
            permuted[target_positions] = source[source_positions]
        output[name] = permuted
    return output


def paired_block_bootstrap(
    *,
    score: np.ndarray,
    mineral_id: np.ndarray,
    band_depth: np.ndarray,
    target_ids: frozenset[int],
    block_ids: np.ndarray,
    replicates: int,
    seed: int,
    joint_support: np.ndarray | None = None,
) -> tuple[BootstrapDraw, ...]:
    """Resample complete blocks and rerun metrics on one joint support."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    score_values = np.asarray(score, dtype=float).reshape(-1)
    identity_values = np.asarray(mineral_id, dtype=float).reshape(-1)
    depth_values = np.asarray(band_depth, dtype=float).reshape(-1)
    all_blocks = np.asarray(block_ids, dtype=object).reshape(-1)
    if not (score_values.size == identity_values.size == depth_values.size == all_blocks.size):
        raise ValueError("bootstrap fields and block IDs must be aligned")
    intrinsic = joint_support_mask(
        score=score_values,
        mineral_id=identity_values,
        band_depth=depth_values,
        qa_valid=np.ones(score_values.shape, dtype=bool),
        glt_valid=np.ones(score_values.shape, dtype=bool),
        block_ids=all_blocks,
    )
    if joint_support is None:
        use = intrinsic
    else:
        use = np.asarray(joint_support, dtype=bool).reshape(-1)
        if use.size != score_values.size or np.any(use & ~intrinsic):
            raise ValueError("bootstrap joint_support is misaligned or includes invalid cells")
    score_values = score_values[use]
    identity_values = identity_values[use]
    depth_values = depth_values[use]
    blocks = all_blocks[use]
    ordered = _ordered_block_ids(blocks)
    if not ordered:
        raise ValueError("at least one complete block is required")
    positions = [np.flatnonzero(blocks == block_id) for block_id in ordered]
    rng = np.random.default_rng(seed)
    rows: list[BootstrapDraw] = []
    for replicate in range(replicates):
        selected = rng.integers(0, len(ordered), size=len(ordered))
        index = np.concatenate([positions[int(item)] for item in selected])
        draw_blocks = np.concatenate(
            [
                np.full(positions[int(item)].size, f"draw_{slot}", dtype=object)
                for slot, item in enumerate(selected)
            ]
        )
        metrics = compute_endpoint_metrics(
            score=score_values[index],
            mineral_id=identity_values[index],
            band_depth=depth_values[index],
            target_ids=target_ids,
            block_ids=draw_blocks,
        )
        rows.extend(
            (
                BootstrapDraw(replicate, "auc", metrics.auc, metrics.auc_unavailable_reason),
                BootstrapDraw(
                    replicate,
                    "spearman",
                    metrics.spearman,
                    metrics.spearman_unavailable_reason,
                ),
            )
        )
    return tuple(rows)


def whole_block_spatial_nulls(
    *,
    score: np.ndarray,
    l2b_fields: Mapping[str, np.ndarray],
    target_ids: frozenset[int],
    block_ids: np.ndarray,
    permutations: int,
    seed: int,
    tanager_valid: np.ndarray | None = None,
    observed_joint_support: np.ndarray | None = None,
) -> tuple[SpatialNullDraw, ...]:
    """Permute complete L2B packets relative to fixed Tanager score blocks."""
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    required = {"mineral_id", "band_depth", "l2b_valid", "x", "y"}
    missing = sorted(required - set(l2b_fields))
    if missing:
        raise ValueError(f"L2B packet lacks fields required for metrics: {', '.join(missing)}")
    score_values = np.asarray(score, dtype=float).reshape(-1)
    blocks = np.asarray(block_ids, dtype=object).reshape(-1)
    packet_fields = {name: np.asarray(values) for name, values in l2b_fields.items()}
    if score_values.size != blocks.size:
        raise ValueError("fixed Tanager scores and block packet IDs must be aligned")
    identity = np.asarray(packet_fields["mineral_id"], dtype=float).reshape(-1)
    depth = np.asarray(packet_fields["band_depth"], dtype=float).reshape(-1)
    if identity.size != blocks.size or depth.size != blocks.size:
        raise ValueError("L2B identity/depth fields and block packets must be aligned")
    packet_valid = np.asarray(packet_fields["l2b_valid"])
    if packet_valid.dtype != np.bool_ or packet_valid.ndim != 1 or packet_valid.size != blocks.size:
        raise ValueError(
            "l2b_valid must be a one-dimensional boolean field aligned with the packet"
        )
    fixed_valid = np.isfinite(score_values)
    if tanager_valid is not None:
        supplied_tanager_valid = np.asarray(tanager_valid, dtype=bool).reshape(-1)
        if supplied_tanager_valid.size != blocks.size:
            raise ValueError("tanager_valid must align with the null packet")
        fixed_valid &= supplied_tanager_valid
    observed_support = fixed_valid & packet_fields["l2b_valid"]
    if observed_joint_support is not None:
        supplied_observed = np.asarray(observed_joint_support, dtype=bool).reshape(-1)
        if supplied_observed.size != blocks.size or not np.array_equal(
            supplied_observed, observed_support
        ):
            raise ValueError("observed joint support differs from the null design intersection")
    design = validate_exchangeable_block_packets(packet_fields, blocks)
    ordered = design.block_ids
    rng = np.random.default_rng(seed)
    rows: list[SpatialNullDraw] = []
    for replicate in range(permutations):
        packet = permute_complete_block_packet(
            packet_fields, blocks, permutation=rng.permutation(len(ordered))
        )
        null_support = fixed_valid & np.asarray(packet["l2b_valid"], dtype=bool)
        metrics = compute_endpoint_metrics(
            score=score_values,
            mineral_id=packet["mineral_id"],
            band_depth=packet["band_depth"],
            target_ids=target_ids,
            block_ids=blocks,
            joint_support=null_support,
        )
        rows.extend(
            (
                SpatialNullDraw(replicate, "auc", metrics.auc, metrics.auc_unavailable_reason),
                SpatialNullDraw(
                    replicate,
                    "spearman",
                    metrics.spearman,
                    metrics.spearman_unavailable_reason,
                ),
            )
        )
    return tuple(rows)


def _draw_values(
    draws: Sequence[BootstrapDraw] | Sequence[SpatialNullDraw],
    *,
    metric: str,
    scheduled: int,
) -> np.ndarray:
    selected = [draw for draw in draws if draw.metric == metric]
    if len(selected) != scheduled:
        raise ValueError(f"{metric} draws do not match the scheduled replicate count")
    identifiers = [
        draw.replicate if isinstance(draw, BootstrapDraw) else draw.permutation for draw in selected
    ]
    if sorted(identifiers) != list(range(scheduled)):
        raise ValueError(f"{metric} draws do not contain each scheduled replicate exactly once")
    ordered = sorted(
        selected,
        key=lambda draw: draw.replicate if isinstance(draw, BootstrapDraw) else draw.permutation,
    )
    return np.asarray(
        [np.nan if draw.value is None else float(draw.value) for draw in ordered],
        dtype=float,
    )


def summarize_bootstrap_interval(
    draws: Sequence[BootstrapDraw], *, metric: str, scheduled_replicates: int
) -> BootstrapInterval:
    """Apply the frozen 95%-finite gate and return a percentile interval."""
    if scheduled_replicates <= 0:
        raise ValueError("scheduled_replicates must be positive")
    values = _draw_values(draws, metric=metric, scheduled=scheduled_replicates)
    finite = values[np.isfinite(values)]
    valid = int(finite.size)
    fraction = valid / scheduled_replicates
    eligible = valid >= math.ceil(FINITE_REPLICATE_FRACTION * scheduled_replicates)
    if eligible:
        lower, upper = (float(value) for value in np.percentile(finite, (2.5, 97.5)))
        reason = None
    else:
        lower = upper = None
        reason = "fewer_than_95_percent_finite_bootstrap_replicates"
    return BootstrapInterval(
        metric=metric,
        lower_95=lower,
        upper_95=upper,
        scheduled_replicates=scheduled_replicates,
        valid_replicates=valid,
        finite_fraction=fraction,
        gate_eligible=eligible,
        unavailable_reason=reason,
    )


def summarize_spatial_null(
    draws: Sequence[SpatialNullDraw],
    *,
    metric: str,
    observed: float | None,
    scheduled_permutations: int,
) -> SpatialNullSummary:
    """Summarize a greater-concordance null with a plus-one p-value."""
    if scheduled_permutations <= 0:
        raise ValueError("scheduled_permutations must be positive")
    values = _draw_values(draws, metric=metric, scheduled=scheduled_permutations)
    finite = values[np.isfinite(values)]
    valid = int(finite.size)
    fraction = valid / scheduled_permutations
    finite_gate = valid >= math.ceil(FINITE_REPLICATE_FRACTION * scheduled_permutations)
    observed_available = observed is not None and math.isfinite(float(observed))
    if finite_gate:
        lower, median, upper = (float(value) for value in np.percentile(finite, (2.5, 50.0, 97.5)))
    else:
        lower = median = upper = None
    if finite_gate and observed_available:
        exceedances = int(np.count_nonzero(finite >= float(observed)))
        p_value = float((exceedances + 1) / (valid + 1))
        reason = None
    else:
        exceedances = None
        p_value = None
        reason = (
            "observed_metric_unavailable"
            if not observed_available
            else "fewer_than_95_percent_finite_spatial_null_permutations"
        )
    return SpatialNullSummary(
        metric=metric,
        observed=observed,
        null_lower_95=lower,
        null_median=median,
        null_upper_95=upper,
        p_value=p_value,
        exceedances=exceedances,
        scheduled_permutations=scheduled_permutations,
        valid_permutations=valid,
        finite_fraction=fraction,
        gate_eligible=finite_gate and observed_available,
        unavailable_reason=reason,
    )


def load_m2_block_scales(path: str | Path, *, site: str, scene_id: str) -> dict[str, M2BlockScale]:
    """Load the exact L/2L categorical rasters from an M2 JSON handoff."""
    manifest_path = Path(path)
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant in M2 manifest: {value}")
        ),
    )
    protocol = payload.get("protocol")
    project_root = Path(__file__).resolve().parents[2]
    protocol_path = project_root / "docs" / "m2_spatial_validation_preregistration.md"
    if not isinstance(protocol, dict) or protocol.get("sha256") != sha256_file(protocol_path):
        raise ValueError("M2 block manifest protocol hash is stale or missing")
    if protocol.get("protocol_compliant") is not True:
        raise ValueError("M2 block manifest is not protocol compliant")
    site_entry = payload.get("sites", {}).get(site)
    if not isinstance(site_entry, dict) or site_entry.get("scene_id") != scene_id:
        raise ValueError("M2 block manifest does not name the requested site/anchor scene")
    grid = site_entry.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("M2 block manifest site entry lacks frozen grid provenance")
    scales = site_entry.get("scales")
    if not isinstance(scales, dict):
        raise ValueError("M2 block manifest site entry lacks scale records")
    if not {"L", "2L"}.issubset(scales):
        raise ValueError("M2 block manifest lacks frozen L and 2L records")

    output: dict[str, M2BlockScale] = {}
    for scale in ("L", "2L"):
        record = scales[scale]
        if not isinstance(record, dict):
            raise ValueError(f"M2 {scale} record is invalid")
        raster_name = record.get("block_raster")
        expected_hash = record.get("block_raster_sha256")
        complete_ids = record.get("complete_block_ids")
        block_side_pixels = record.get("block_side_pixels")
        halo_pixels = record.get("halo_pixels")
        if not isinstance(raster_name, str) or not isinstance(expected_hash, str):
            raise ValueError(f"M2 {scale} record lacks raster provenance")
        if not isinstance(complete_ids, list) or not complete_ids:
            raise ValueError(f"M2 {scale} record lacks complete block IDs")
        if (
            isinstance(block_side_pixels, bool)
            or not isinstance(block_side_pixels, int)
            or block_side_pixels <= 0
            or isinstance(halo_pixels, bool)
            or not isinstance(halo_pixels, int)
            or halo_pixels < 0
        ):
            raise ValueError(f"M2 {scale} record lacks valid block-side/halo geometry")
        raster_path = manifest_path.parent / raster_name
        if not raster_path.is_file() or sha256_file(raster_path) != expected_hash:
            raise ValueError(f"M2 {scale} block raster is missing or has a stale hash")
        with rasterio.open(raster_path) as dataset:
            if dataset.count != 1 or dataset.crs is None or dataset.nodata != 0:
                raise ValueError(f"M2 {scale} block raster geometry is incomplete")
            if tuple(grid.get("shape", ())) != dataset.shape:
                raise ValueError(f"M2 {scale} block raster shape differs from frozen grid")
            if grid.get("crs") != dataset.crs.to_string():
                raise ValueError(f"M2 {scale} block raster CRS differs from frozen grid")
            if tuple(grid.get("transform", ())) != tuple(dataset.transform)[:6]:
                raise ValueError(f"M2 {scale} block raster transform differs from frozen grid")
            values = dataset.read(1)
            raster_ids = set(int(value) for value in np.unique(values) if int(value) != 0)
            expected_ids = {int(value) for value in complete_ids}
            if len(expected_ids) != len(complete_ids) or any(value <= 0 for value in expected_ids):
                raise ValueError(f"M2 {scale} complete block IDs must be unique and positive")
            if raster_ids != expected_ids:
                raise ValueError(f"M2 {scale} raster IDs differ from its manifest record")
            expected_cells = block_side_pixels * block_side_pixels
            invalid_sizes = {
                block_id: int(np.count_nonzero(values == block_id))
                for block_id in expected_ids
                if np.count_nonzero(values == block_id) != expected_cells
            }
            if invalid_sizes:
                raise ValueError(
                    f"M2 {scale} raster contains incomplete block footprints: {invalid_sizes!r}"
                )
            output[scale] = M2BlockScale(
                scale=scale,
                values=values,
                complete_block_ids=tuple(int(value) for value in complete_ids),
                transform=dataset.transform,
                crs=dataset.crs.to_string(),
                source_path=raster_path,
                source_sha256=expected_hash,
                block_side_pixels=block_side_pixels,
                halo_pixels=halo_pixels,
            )
    return output


def _average_block_field(
    values: np.ndarray, scale: M2BlockScale, geometry: RasterGeometry
) -> np.ndarray:
    source = np.pad(np.asarray(values, dtype=np.float64), 1, constant_values=0.0)
    source_transform = scale.transform * Affine.translation(-1, -1)
    destination = np.full(geometry.shape, np.nan, dtype=np.float64)
    reproject(
        source=source,
        destination=destination,
        src_transform=source_transform,
        src_crs=scale.crs,
        src_nodata=None,
        dst_transform=geometry.transform,
        dst_crs=geometry.crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return destination


def block_footprint_support(scale: M2BlockScale, geometry: RasterGeometry) -> BlockFootprintSupport:
    """Assign only L2B cells fully contained in one complete M2 block."""
    complete = np.isin(scale.values, scale.complete_block_ids)
    coverage = _average_block_field(complete.astype(np.float64), scale, geometry)
    mean_id = _average_block_field(scale.values, scale, geometry)
    mean_square_id = _average_block_field(
        np.asarray(scale.values, dtype=np.float64) ** 2,
        scale,
        geometry,
    )
    full_complete = np.isfinite(coverage) & (coverage == 1.0)
    rounded = np.rint(mean_id)
    one_block = (
        full_complete
        & np.isfinite(mean_id)
        & (mean_id == rounded)
        & (mean_square_id == mean_id**2)
        & np.isin(rounded, scale.complete_block_ids)
    )
    block_ids = np.where(one_block, rounded, 0).astype(scale.values.dtype, copy=False)
    return BlockFootprintSupport(
        block_ids=block_ids,
        crosses_block_boundary=full_complete & ~one_block,
        incomplete_or_halo_support=~full_complete,
    )


def block_ids_on_l2b_grid(scale: M2BlockScale, geometry: RasterGeometry) -> np.ndarray:
    """Return M2 IDs only where a full L2B footprint lies in one complete block."""
    return block_footprint_support(scale, geometry).block_ids


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def strict_json_dumps(payload: Any) -> str:
    """Serialize to standards-compliant JSON with no NaN/Infinity tokens."""
    return json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_strict_json(path: str | Path, payload: Any) -> None:
    """Atomically write strict JSON after explicit non-finite normalization."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(strict_json_dumps(payload), encoding="utf-8")
    temporary.replace(destination)


__all__ = [
    "BlockFootprintSupport",
    "BlockPacketDesign",
    "BootstrapInterval",
    "BootstrapDraw",
    "EMIT_L2B_SHORT_NAME",
    "EmitL2BGroup",
    "EmitL2BMetadataPacket",
    "EmitL2BPair",
    "EndpointMetrics",
    "M2BlockScale",
    "OntologyEntry",
    "PinnedEmitL2AInput",
    "ProductIdentity",
    "ProductMismatchError",
    "RasterGeometry",
    "SourceMineral",
    "SpatialNullDraw",
    "SpatialNullSummary",
    "area_average_continuous",
    "block_footprint_support",
    "block_ids_on_l2b_grid",
    "compute_endpoint_metrics",
    "joint_support_mask",
    "load_emit_l2b_pair",
    "load_emit_l2b_metadata",
    "load_m2_block_scales",
    "load_pinned_emit_l2a_input",
    "orthorectify_with_glt",
    "paired_block_bootstrap",
    "parse_product_identity",
    "parse_l2a_product_identity",
    "permute_complete_block_packet",
    "read_ontology_crosswalk",
    "reproject_categorical_nearest",
    "sha256_file",
    "strict_json_dumps",
    "summarize_bootstrap_interval",
    "summarize_spatial_null",
    "validate_ontology_crosswalk",
    "validate_exchangeable_block_packets",
    "validate_l2b_identity_against_l2a",
    "whole_block_spatial_nulls",
    "write_strict_json",
]
