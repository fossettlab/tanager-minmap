"""Render the submission memo as reproducible HTML and PDF artifacts.

The renderer leaves ``submission/memo.md`` and all source figures untouched.
It inserts the four figures cited by the memo, converts Markdown with Pandoc,
renders PDF with WeasyPrint, and records hashes plus tool versions in a JSON
manifest. It reports page count but does not enforce an unpublished page limit.

Run::

    uv run python scripts/render_submission_memo.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "submission" / "memo.md"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "memo"

FIGURES = {
    "What Sentinel-2 spectral sampling blurs": (
        "submission/figures/bingham_20250911_191523_58_4001_band_ablation.png",
        "Figure 1. Tanager-to-Sentinel-2 spectral degradation at the 2200 nm Al-OH doublet.",
    ),
    "Validation against an independent map": (
        "submission/figures/goldfield_validation_pair.png",
        "Figure 2. Goldfield alteration-group comparison with the Rockwell ASTER map.",
    ),
    "Cross-sensor agreement with EMIT": (
        "submission/figures/goldfield_20240925_185504_87_4001_emit_comparison.png",
        "Figure 3. Tanager-EMIT cross-sensor consistency for the Goldfield scene.",
    ),
    "Acid-generating-potential screening": (
        "submission/figures/bingham_20250911_191523_58_4001_amd_agp.png",
        "Figure 4. Scene-relative acid-generating-potential screening at Bingham.",
    ),
}

CSS = """
@page {
  size: Letter;
  margin: 0.55in 0.58in 0.62in;
  @bottom-right {
    content: "tanager-minmap  ·  " counter(page) " / " counter(pages);
    color: #59657a;
    font-size: 8pt;
  }
}
:root { color-scheme: light; }
body {
  color: #111827;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9.2pt;
  line-height: 1.34;
  margin: 0;
}
h1 {
  color: #0a2745;
  font-size: 22pt;
  line-height: 1.04;
  margin: 0 0 0.16in;
  max-width: 7in;
}
h2 {
  border-top: 1px solid #bfd0df;
  color: #123e64;
  font-size: 12.3pt;
  margin: 0.16in 0 0.06in;
  padding-top: 0.07in;
  break-after: avoid;
}
p { margin: 0 0 0.08in; }
strong { color: #0a2745; }
a { color: #145b87; text-decoration: none; }
figure {
  break-inside: avoid;
  margin: 0.10in auto 0.15in;
  text-align: center;
}
figure img {
  display: block;
  margin: 0 auto;
  max-height: 4.55in;
  max-width: 100%;
  object-fit: contain;
}
figure.compact img { max-height: 3.35in; }
figcaption {
  color: #3f4d61;
  font-size: 8.2pt;
  line-height: 1.22;
  margin: 0.045in auto 0;
  max-width: 6.8in;
  text-align: left;
}
hr { border: 0; border-top: 1px solid #bfd0df; margin: 0.15in 0 0.10in; }
em { color: #354154; }
"""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(executable: str, *args: str) -> str:
    """Return the first output line from a version command."""

    result = subprocess.run(
        [executable, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0]


def insert_figures(markdown: str) -> str:
    """Insert cited figures at the end of their matching memo sections."""

    lines = markdown.splitlines()
    output: list[str] = []
    active_heading: str | None = None

    def append_active_figure() -> None:
        if active_heading not in FIGURES:
            return
        rel_path, caption = FIGURES[active_heading]
        figure_class = (
            "memo-figure compact"
            if active_heading == "Acid-generating-potential screening"
            else "memo-figure"
        )
        output.extend(
            [
                "",
                f'<figure class="{figure_class}">',
                f'  <img src="../../{rel_path}" alt="{caption}">',
                f"  <figcaption>{caption}</figcaption>",
                "</figure>",
                "",
            ]
        )

    for line in lines:
        if line.startswith("## "):
            append_active_figure()
            active_heading = line[3:].strip()
        output.append(line)
    append_active_figure()
    return "\n".join(output) + "\n"


def pdf_page_count(pdf: Path, pdfinfo: str) -> int:
    """Read the rendered page count from pdfinfo."""

    result = subprocess.run(
        [pdfinfo, str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("pdfinfo did not report a page count")


def render(source: Path, output_dir: Path) -> dict[str, object]:
    """Render the memo and return its provenance manifest."""

    pandoc = shutil.which("pandoc")
    weasyprint = shutil.which("weasyprint")
    pdfinfo = shutil.which("pdfinfo")
    executables = (
        ("pandoc", pandoc),
        ("weasyprint", weasyprint),
        ("pdfinfo", pdfinfo),
    )
    missing = [name for name, path in executables if path is None]
    if missing:
        raise RuntimeError(f"missing required executable(s): {', '.join(missing)}")

    source = source.resolve(strict=True)
    figure_paths = {
        heading: (ROOT / rel).resolve(strict=True) for heading, (rel, _) in FIGURES.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    html_out = output_dir / "tanager_minmap_memo.html"
    pdf_out = output_dir / "tanager_minmap_memo.pdf"
    manifest_out = output_dir / "tanager_minmap_memo_manifest.json"

    prepared = insert_figures(source.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="tanager-memo-") as tmp_name:
        tmp = Path(tmp_name)
        prepared_md = tmp / "memo_with_figures.md"
        fragment = tmp / "memo_fragment.html"
        prepared_md.write_text(prepared, encoding="utf-8")
        subprocess.run(
            [
                pandoc,
                "--from",
                "markdown",
                "--to",
                "html5",
                "--output",
                str(fragment),
                str(prepared_md),
            ],
            check=True,
        )
        document = (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<style>{CSS}</style></head><body>"
            + fragment.read_text(encoding="utf-8")
            + "</body></html>\n"
        )
        html_out.write_text(document, encoding="utf-8")

    subprocess.run([weasyprint, str(html_out), str(pdf_out)], check=True)
    page_count = pdf_page_count(pdf_out, pdfinfo)
    manifest: dict[str, object] = {
        "schema": "tanager-minmap.memo-render/1",
        "source": {"path": str(source.relative_to(ROOT)), "sha256": sha256(source)},
        "figures": [
            {
                "section": heading,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for heading, path in figure_paths.items()
        ],
        "tools": {
            "pandoc": command_version(pandoc, "--version"),
            "weasyprint": command_version(weasyprint, "--version"),
            "pdfinfo": command_version(pdfinfo, "-v"),
        },
        "outputs": {
            "html": {"path": str(html_out.relative_to(ROOT)), "sha256": sha256(html_out)},
            "pdf": {
                "path": str(pdf_out.relative_to(ROOT)),
                "sha256": sha256(pdf_out),
                "pages": page_count,
            },
        },
        "page_limit_assertion": None,
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render(args.source, args.output_dir)
    outputs = manifest["outputs"]
    assert isinstance(outputs, dict)
    pdf = outputs["pdf"]
    assert isinstance(pdf, dict)
    print(f"wrote {pdf['path']} ({pdf['pages']} pages)")
    print("page-limit compliance was not asserted because no authoritative limit is recorded")


if __name__ == "__main__":
    main()
