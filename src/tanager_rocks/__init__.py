"""tanager-rocks: mineral and alteration mapping from Tanager VSWIR surface reflectance.

Prepared for the Planet Tanager Open Data Competition. This package builds the
analysis-specific workflow — continuum-removal diagnostic-feature
mapping, SAM/MTMF unmixing against a USGS/ECOSTRESS library, the Sentinel-2
band-ablation comparison, and the EMIT cross-sensor benchmark — on top of the
shared :mod:`tanager_spec` data layer (STAC ingest, cube IO, masking, SRF
simulation). See ``METHODS.md`` for the public pipeline description.
"""

from __future__ import annotations

from . import config, features, speclib, unmix, viz

__version__ = "0.1.0"

__all__ = ["config", "features", "speclib", "unmix", "viz", "__version__"]
