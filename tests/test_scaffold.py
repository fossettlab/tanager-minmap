"""Scaffold smoke tests: the package imports and its structure is wired.

These assert the skeleton holds together (imports resolve, config is sane,
the CLI parses) without exercising any not-yet-implemented analysis.
"""

from __future__ import annotations

import tanager_rocks
from tanager_rocks import config
from tanager_rocks.cli import _build_parser


def test_package_imports():
    assert tanager_rocks.__version__ == "0.1.0"
    # All analysis submodules are importable from the top level.
    for name in ("config", "features", "speclib", "unmix", "viz"):
        assert hasattr(tanager_rocks, name)


def test_sites_present_and_confirmed():
    assert set(config.SITES) == {"bingham", "goldfield"}
    # Identity verified against USGS MRDS (scripts/confirm_site_identity.py).
    assert all(s.confirmed for s in config.SITES.values())


def test_scene_ids_match_declared_count():
    # The confirmed scene list and the spec's scene count must agree.
    for site in config.SITES.values():
        assert len(site.scene_ids) == site.n_scenes


def test_diagnostics_match_spec():
    # The three SWIR diagnostics explicitly named in spec.md step 3.
    assert config.DIAGNOSTIC_NM["al_oh_doublet"] == 2200.0
    assert config.DIAGNOSTIC_NM["jarosite"] == 2265.0
    assert config.DIAGNOSTIC_NM["gypsum_carbonate"] == 2340.0


def test_cli_parser_builds():
    parser = _build_parser()
    args = parser.parse_args(["map", "--site", "bingham", "--output", "out"])
    assert args.command == "map"
    assert args.site == "bingham"
