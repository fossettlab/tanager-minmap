# Video credits, attribution, and media-rights boundary

This file accompanies the rendered Tanager Rocks video and its release
manifest. It records attribution; it does not grant rights beyond the cited
source terms. The repository's MIT license applies to repository-authored code
and prose only, not to the data or generated media listed below.

## Creator

Alex Bradley  
Department of Earth, Environmental, and Planetary Sciences  
Washington University in St. Louis  
St. Louis, Missouri 63130, USA  
abradley@wustl.edu

## Planet Tanager source and adapted material

The video uses products derived from these Planet Tanager surface-reflectance
scenes:

- Goldfield district, Nevada: `20240925_185504_87_4001`
- Bingham Canyon/Kennecott, Utah: `20250911_191523_58_4001`

The exact Goldfield `natural-lands` item and Bingham `energy-mining` item each
declare CC BY 4.0. Preserve both catalog-required, year-specific notices:

> Adapted from Tanager STAC Data, available at www.planet.com/data/stac © 2024 Planet Labs PBC. All Rights Reserved.
>
> Adapted from Tanager STAC Data, available at www.planet.com/data/stac © 2025 Planet Labs PBC. All Rights Reserved.

Source items: [Goldfield](https://www.planet.com/data/stac/tanager-core-imagery/natural-lands/20240925_185504_87_4001/20240925_185504_87_4001.json) and [Bingham](https://www.planet.com/data/stac/tanager-core-imagery/energy-mining/20250911_191523_58_4001/20250911_191523_58_4001.json).  
License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Strict public release mode uses Tanager-derived figure composites for beats 05
and 07. It does not use the Esri World Imagery browser captures retained for
internal draft reproducibility. Any future use of those captures requires a new
rights review and the then-current, location-specific service attribution.

## NASA EMIT

The cross-sensor panel uses the NASA EMIT L2A reflectance product:

Green, R. (2022), *EMIT L2A Estimated Surface Reflectance and Uncertainty and
Masks 60 m V001*, NASA EOSDIS Land Processes DAAC,
[doi:10.5067/EMIT/EMITL2ARFL.001](https://doi.org/10.5067/EMIT/EMITL2ARFL.001).

The release manifest records the selected input and derived figure hashes.

## USGS reference products

- Kokaly, R. F., Clark, R. N., Swayze, G. A., Livo, K. E., Hoefen, T. M.,
  Pearson, N. C., Wise, R. A., Benzel, W. M., Lowers, H. A., Driscoll, R. L.,
  and Klein, A. J. (2017), *USGS Spectral Library Version 7*, USGS Data Series
  1035, [doi:10.3133/ds1035](https://doi.org/10.3133/ds1035).
- Rockwell, B. W., and Bonham, L. C. (2017), ASTER hydrothermal-alteration
  mapping data release, USGS,
  [doi:10.5066/F7CR5RK7](https://doi.org/10.5066/F7CR5RK7).

The video describes these products according to the scientific limitations in
the release claims and methods; attribution does not turn them into mineral-
level field truth.

## Generated narration and music

Narration was generated with ElevenLabs. The final voice name, voice ID,
category/library status, model, settings, generation ID/time, account plan,
non-beta confirmation, text hash, and audio hash must be taken from the selected
records in the release bundle's `evidence/tts.jsonl`. They are intentionally not
asserted here while the evidence templates remain incomplete.

Music was generated with Eleven Music v2, then edited and mixed by the scripted
gain envelope, sidechain ducking, and loudness-normalization pipeline in
`scripts/video/audio.py`. The final generation ID/time, plan, account status,
model terms snapshot, output hash, and editorial selection record must be taken
from `evidence/music.json` and `render.json` in the release bundle.

ElevenLabs media is not covered by the repository's MIT license. Public release
requires generation-time evidence that the applicable paid-plan and model terms
allow the intended use and that the selected services were non-beta.

## Procedural motif and editorial graphics

The opening 426-band spectral motif and its end-card bookend are procedural
graphics rendered by `video/build/render_motif.py` and `scripts/video/beats.py`.
They are not satellite imagery, measurements, or scientific data. Lower-thirds,
callouts, fades, and gain automation are repository-authored editorial elements.

## License separation

- Repository-authored code and prose: MIT; see `LICENSE`.
- Planet imagery and adapted material: source CC BY 4.0 plus the required Planet
  notices above.
- NASA and USGS products: retain their product-specific citations and metadata.
- ElevenLabs narration and music: governed by the applicable account agreement,
  voice/model terms, and the generation-time evidence in the release bundle.
- Final MP4/SRT and media masters: distribute only through the approved release
  route recorded in the frozen contract and `render.json`.

See the repository-level `NOTICE.md` for the broader asset inventory and links
to the primary terms checked during release preparation.
