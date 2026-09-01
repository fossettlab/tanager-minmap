# Submission video — narration script v2

Produced cut, target **~2:50** narrated (hard cap 3:00). Builds on the proven v1
arc, tightened and re-paced, with one factual correction (EMIT: alunite 0.55 /
jarosite 0.58 after the authoritative QA rerun). Voice: measured, precise, no hype —
follows the memo. Every number traces to `METHODS.md` / committed CSVs.

Spoken text for each beat is the block under it — tomorrow, split these into
`segments_v2/00…08_*.txt` (spoken text only, no timecodes/cues) for
`generate_tts.sh`. On-screen and AI columns are production notes, **not spoken**.

| # | Time | On screen (all REAL unless marked) | Optional synthetic | Segment |
|---|------|-----------------------------------|--------------------|---------|
| 0 | 0:00–0:10 | Title card | *opt. procedural spectral-fan motif* | `00_title` |
| 1 | 0:10–0:28 | Goldfield RGB, slow zoom into a tailings/alteration zone | — | `01_hook` |
| 2 | 0:28–0:46 | Bingham + Goldfield RGB, two-site locator | — | `02_stakes` |
| 3 | 0:46–1:10 | spectra figure, absorptions drawn on | — | `03_data` |
| 4 | 1:10–1:40 | band-ablation figure, Tanager→Sentinel-2 degrade | — | `04_ablation` |
| 5 | 1:40–2:08 | Goldfield screening map; interactive web map linked | — | `05_livemap` |
| 6 | 2:08–2:32 | validation pair + EMIT comparison | — | `06_validation` |
| 7 | 2:32–2:48 | Bingham screening map; interactive web map linked | — | `07_amd` |
| 8 | 2:48–3:00 | story-page footer / links / DOI | *opt. disclosure end card* | `08_close` |

---

### 0 — Title · [procedural spectral motif → title card]

Four hundred and twenty-six colors of light. Most satellites see a handful. This one sees them all.

### 1 — Hook · [Goldfield RGB, slow zoom into a tailings/alteration zone]

Mine waste and altered bedrock can look alike in true color. But their surface mineral assemblages differ. Jarosite is associated with acidic sulfate conditions; gypsum without acidic iron phases can indicate more buffered conditions. Distinguishing them from orbit is the problem.

### 2 — Stakes · [two-site locator, Bingham + Goldfield]

That distinction matters to the USGS, BLM, and EPA geologists who screen abandoned mine lands for acid-drainage hazard and critical minerals. We ask whether Planet's Tanager imaging spectrometer can make it — at two sites: Bingham Canyon, Utah, and the Goldfield district, Nevada.

### 3 — The data · [spectra figure, absorptions drawn on]

Tanager's delivered ortho product contains 426 contiguous bands, from the visible to the shortwave infrared, on a thirty-meter grid. Minerals leave narrow absorptions there: aluminum-hydroxide near 2200 nanometers, jarosite at 2265, gypsum at 2340. Measured library spectra and spectra from mapped pixels show features in those regions.

### 4 — Central result · [band-ablation figure]

This is the central result. Convolve the measured alunite and kaolinite spectra with Sentinel-2's published spectral response, and their spectral-angle separation falls from five point one degrees to two point six — a fifty percent loss. The visible jarosite-goethite contrast survives, showing that the loss is feature-specific. Tanager records an Al-OH distinction that Sentinel-2 largely blurs.

### 5 — Map · [Goldfield screening map; interactive web map linked]

One deliverable is an interactive Goldfield screening map. Each color is the strongest scene-relative, library-matched candidate. Alunite and kaolinite trace candidate acid-sulfate alteration associated with the gold system. The web map helps select locations for field follow-up.

### 6 — Validation + EMIT · [validation pair + EMIT comparison]

Two complementary checks. The Tanager alunite map shows alteration-group agreement with a published USGS ASTER map — not field truth. Applying the same library-driven method to NASA's EMIT over the same ground also gives positive map consistency: alunite at zero point five five, jarosite at zero point five eight.

### 7 — AMD payoff · [Bingham screening map; interactive web map linked]

At Bingham Canyon, these minerals support a candidate screening layer. Jarosite, iron oxides, and gypsum define high, moderate, and low acid-generating-potential tiers, relative within the scene. The map prioritizes field follow-up; it does not measure pH or replace sampling.

### 8 — Close · [story-page footer / links / DOI]

Every analytical step is scripted, with public packaging, clean-clone verification, and a permanent archive fixed as release gates. Tanager's full spectrum preserves mineral structure that broadband sensors blur. That is what we would bring to the open data community.
