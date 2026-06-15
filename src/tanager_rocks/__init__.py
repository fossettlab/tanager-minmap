"""tanager-rocks: mineral and alteration mapping from Tanager VSWIR surface reflectance.

Track I of the Planet Tanager Open Data Competition. This package builds the
analysis specific to the flagship — continuum-removal diagnostic-feature
mapping, SAM/MTMF unmixing against a USGS/ECOSTRESS library, the Sentinel-2
band-ablation comparison, and the EMIT cross-sensor benchmark — on top of the
shared :mod:`tanager_spec` data layer (STAC ingest, cube IO, masking, SRF
simulation). See ``spec.md`` for the full pipeline and rubric mapping.
"""

from __future__ import annotations

from . import config, features, speclib, unmix, viz

__version__ = "0.1.0"

__all__ = ["config", "features", "speclib", "unmix", "viz", "__version__"]
