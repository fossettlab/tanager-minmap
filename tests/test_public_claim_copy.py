"""Regression checks for release-bound scientific scope language."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_COPY = (
    ROOT / "README.md",
    ROOT / "METHODS.md",
    ROOT / "submission" / "memo.md",
    ROOT / "submission" / "index.html",
    ROOT / "docs" / "competition_form_draft.md",
)


def _combined_public_copy() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_COPY)


class _StoryStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.heading_levels: list[int] = []
        self.images: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if re.fullmatch(r"h[1-6]", tag):
            self.heading_levels.append(int(tag[1]))
        if tag == "img":
            self.images.append(dict(attrs))


def test_release_copy_excludes_superseded_absolute_claims() -> None:
    copy = _combined_public_copy().lower()

    forbidden = (
        "material identification in the strict sense",
        "gypsum is benign",
        "field-validated mineral",
        "absolute hazard map",
        "directly actionable without field",
        "replaces field sampling",
        "identical classifier was rerun",
        "collapse into one class",
        "uncertainty-aware candidate zones",
        "recovers the expected zoning",
        "resolves about four times",
        "smaller minimum mappable feature",
    )
    for phrase in forbidden:
        assert phrase not in copy


def test_release_copy_retains_screening_and_validation_limits() -> None:
    memo = (ROOT / "submission" / "memo.md").read_text(encoding="utf-8")
    story = (ROOT / "submission" / "index.html").read_text(encoding="utf-8")
    form = (ROOT / "docs" / "competition_form_draft.md").read_text(encoding="utf-8")
    form_normalized = " ".join(form.split())

    assert "each pixel receives a scene-relative score" in memo
    assert "screening layer for acid-mine-drainage risk" in memo
    assert "there is no field validation" in memo
    assert "candidate screening layer for field follow-up" in memo
    assert "does not replace sampling" in story
    assert "not independent mineral truth" in story
    assert "scene-relative library matches" in story
    assert "scene-relative candidate zones" in story
    assert "Kaolinite and iron-oxide agreement is weaker" in story
    assert "Tanager's 30&nbsp;m ortho pixels cover about one-quarter" in story
    assert "not a native-footprint claim" in story
    assert "rather than ground truth" in form_normalized


def test_story_headline_metrics_keep_their_scope_adjacent() -> None:
    story = (ROOT / "submission" / "index.html").read_text(encoding="utf-8")

    cards_match = re.search(
        r'<section class="evidence-grid".*?</section>',
        story,
        flags=re.DOTALL,
    )
    assert cards_match is not None
    cards = " ".join(cards_match.group(0).split())
    assert "49.8% loss" in cards
    assert "5.138° → 2.578°" in cards
    assert "0.78 AUC" in cards
    assert "not field validation" in cards
    assert "6 / 6 positive" in cards
    assert "r = +0.335 to +0.584" in cards
    assert "shared-pipeline consistency, not truth" in cards


def test_competition_form_draft_stays_within_declared_scope_and_limits() -> None:
    form = (ROOT / "docs" / "competition_form_draft.md").read_text(encoding="utf-8")
    description_match = re.search(
        r"## Project description.*?\n\n(.*?)(?=\n\n## Next steps)",
        form,
        flags=re.DOTALL,
    )
    next_steps_match = re.search(
        r"## Next steps.*?\n\n(.*?)(?=\n\n## Required)",
        form,
        flags=re.DOTALL,
    )
    assert description_match is not None
    assert next_steps_match is not None

    description = description_match.group(1)
    next_steps = next_steps_match.group(1)
    word_pattern = r"\b[\w’-]+\b"
    assert len(re.findall(word_pattern, description)) <= 300
    assert len(re.findall(word_pattern, next_steps)) <= 100

    submission_copy = f"{description}\n{next_steps}"
    for excluded in ("TanagerFM", "Track I", "Track II"):
        assert excluded not in submission_copy


def test_story_heading_order_and_image_alternatives_are_accessible() -> None:
    parser = _StoryStructureParser()
    parser.feed((ROOT / "submission" / "index.html").read_text(encoding="utf-8"))

    assert parser.heading_levels
    assert parser.heading_levels[0] == 1
    assert all(
        current <= previous + 1
        for previous, current in zip(parser.heading_levels, parser.heading_levels[1:], strict=False)
    )
    assert parser.images
    assert all(image.get("alt", "").strip() for image in parser.images)
