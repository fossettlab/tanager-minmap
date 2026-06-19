"""Tests for the ``tanager-minmap`` CLI dispatch and path wiring.

These are offline: the pipeline ``run_*`` functions are monkeypatched, so the
tests exercise argument parsing, path construction, and dispatch routing
without loading a scene.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tanager_rocks import cli
from tanager_rocks.pipeline import PipelinePaths


def _patch(monkeypatch, name: str) -> dict:
    """Replace a pipeline run_* in the cli namespace with a call recorder."""
    captured: dict = {}

    def recorder(site, paths, **kwargs):
        captured["site"] = site
        captured["paths"] = paths
        captured["kwargs"] = kwargs
        return Path("figure.png")

    monkeypatch.setattr(cli, name, recorder)
    return captured


def test_paths_from_cli_layout():
    paths = PipelinePaths.from_cli(Path("d"), Path("o"))
    assert paths.raw_dir == Path("d/raw")
    assert paths.speclib_dir == Path("d/speclib/ASCIIdata_splib07a")
    assert paths.maps_dir == Path("o/maps")
    assert paths.figures_dir == Path("o/figures")
    assert paths.tables_dir == Path("o/tables")


def test_paths_repo_default_layout():
    paths = PipelinePaths.repo_default(Path("/repo"))
    assert paths.maps_dir == Path("/repo/data/intermediate/maps")
    assert paths.figures_dir == Path("/repo/figures")
    assert paths.tables_dir == Path("/repo/data/intermediate/ablation")


def test_map_dispatches_with_site_and_paths(monkeypatch):
    captured = _patch(monkeypatch, "run_map")
    rc = cli.main(["map", "--site", "bingham", "--data-root", "d", "--output", "o"])
    assert rc == 0
    assert captured["site"].site_id == "bingham"
    assert captured["paths"].raw_dir == Path("d/raw")
    assert captured["paths"].maps_dir == Path("o/maps")
    assert captured["kwargs"] == {}


def test_unmix_passes_thresholds(monkeypatch):
    captured = _patch(monkeypatch, "run_unmix")
    cli.main(["unmix", "--site", "goldfield", "--max-angle", "0.2", "--max-infeas", "0.5"])
    assert captured["site"].site_id == "goldfield"
    assert captured["kwargs"] == {"max_angle": 0.2, "max_infeas": 0.5}


def test_amd_passes_quantile_and_infeas(monkeypatch):
    captured = _patch(monkeypatch, "run_amd")
    cli.main(["amd", "--site", "bingham", "--quantile", "0.8", "--max-infeas", "1.5"])
    assert captured["kwargs"] == {"max_infeas": 1.5, "quantile": 0.8}


def test_hero_defaults(monkeypatch):
    captured = _patch(monkeypatch, "run_hero")
    cli.main(["hero", "--site", "goldfield"])
    assert captured["kwargs"] == {"max_infeas": 1.0, "quantile": 0.90}


def test_ablate_takes_no_extra_kwargs(monkeypatch):
    captured = _patch(monkeypatch, "run_ablate")
    cli.main(["ablate", "--site", "bingham"])
    assert captured["kwargs"] == {}


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        cli.main(["bogus", "--site", "bingham"])


def test_missing_site_exits():
    with pytest.raises(SystemExit):
        cli.main(["map"])


def test_no_subcommand_exits():
    with pytest.raises(SystemExit):
        cli.main([])
