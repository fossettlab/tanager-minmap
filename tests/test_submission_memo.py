from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "render_submission_memo.py"
SPEC = importlib.util.spec_from_file_location("render_submission_memo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def test_insert_figures_places_each_figure_inside_its_section() -> None:
    source = """# Memo

## What Sentinel-2 spectral sampling blurs

First section.

## Validation against an independent map

Second section.

## Limits

Last section.
"""
    rendered = renderer.insert_figures(source)

    first_figure = rendered.index("Figure 1.")
    second_heading = rendered.index("## Validation against an independent map")
    second_figure = rendered.index("Figure 2.")
    limits_heading = rendered.index("## Limits")
    assert first_figure < second_heading < second_figure < limits_heading
    assert rendered.count('<figure class="memo-figure">') == 2


def test_memo_figure_references_follow_first_appearance() -> None:
    memo = (ROOT / "submission" / "memo.md").read_text(encoding="utf-8")
    assert memo.index("(Figure 1)") < memo.index("(Figure 2)") < memo.index("(Figure 3)")
    assert "(1) Tanager vs Sentinel-2 band-ablation" in memo
    assert "(2) Goldfield/Cuprite alteration-group validation" in memo
