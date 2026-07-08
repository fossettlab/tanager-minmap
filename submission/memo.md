# Tanager VSWIR resolves mine-waste and alteration mineralogy that multispectral sensors cannot

**Planet Tanager Open Data Competition — Track I.**
Alex Bradley, Department of Earth, Environmental, and Planetary Sciences,
Washington University in St. Louis (abradley@wustl.edu).

## The question

Hydrothermal alteration and mine waste are recorded in a small set of minerals —
alunite, kaolinite, jarosite, the iron oxides, gypsum, and the white micas —
whose diagnostic absorptions fall in the shortwave infrared. Can Tanager's
contiguous 426-band visible-to-shortwave-infrared (VSWIR) measurements resolve
that mineralogy at a specificity multispectral sensors cannot, and what does the
answer reveal about acid-mine-drainage hazard? This entry tests the question at
two named United States sites: the Bingham Canyon / Kennecott porphyry-copper
mine and tailings impoundment in Utah, and the Goldfield hydrothermal district
in Nevada, adjacent to the USGS Cuprite spectral benchmark.

## Approach

Mineral identity is anchored to the USGS spectral library (splib07a; Kokaly et
al. 2017): every endmember is a measured library spectrum resampled to the
Tanager wavelength axis, and no spectral shape is synthesized. Two methods map
the target assemblage on each scene's surface-reflectance product. Continuum-
removed band depth (Clark and Roush 1984) measures the diagnostic absorptions
directly — the 2200 nm Al-OH doublet that separates alunite from kaolinite, the
2265 nm jarosite feature, the 2340 nm gypsum-carbonate feature, and the
visible-to-near-infrared ferric-iron bands. A mixture-tuned matched filter
(MTMF; Boardman 1998) then estimates per-mineral abundance against the scene's
own background covariance, with an infeasibility score that suppresses
spectrally implausible detections. The whole pipeline runs from one command and
reproduces from a clean clone.

## What Tanager resolves that Sentinel-2 cannot

The case for contiguous VSWIR is sharpest at the 2200 nm Al-OH doublet. Alunite
and kaolinite (advanced-argillic versus argillic alteration, a distinction that
matters for both ore characterization and acid generation) differ in the precise
position and shape of this feature. Degrading the Tanager library spectra to
Sentinel-2 bands through the published spectral response functions collapses
their separability: the alunite-kaolinite spectral angle falls from 5.1° to
2.6°, a 50% loss, because a single broad Sentinel-2 band (B12) spans the entire
doublet (Figure 2). The loss is specific to the shortwave infrared, not
universal; the visible-to-near-infrared jarosite-goethite contrast survives
degradation, which is the control showing the effect is the doublet itself and
not coarse resampling.

## Validation against an independent map

At Goldfield the maps were tested against the USGS ASTER alteration map of the
district (Rockwell and Bonham 2017), an independent remote-sensing product with
no shared calibration or acquisition. The Tanager scores separate the published
alteration zones at the alteration-group level: the Al-OH band depth and the
gypsum-carbonate band depth each reach a rank-AUC of 0.78, the alunite MTMF
abundance 0.71, and the muscovite (sericite) MTMF 0.69, with alunite peaking in
the advanced-argillic zone at Cuprite and muscovite in the surrounding sericite
(Figure 1). The Fe-oxide and kaolinite layers do not separate as cleanly; the
published categorical classes subdivide these phases differently than the
spectral library groups them, and that disagreement is reported rather than
tuned away. Bingham validates more weakly, by design and by geology: a porphyry
system whose pervasive sericite does not partition into the published
acid-sulfate zones. That contrast is itself informative about where the method's
discrimination is strong.

## Cross-sensor agreement with EMIT

The identical pipeline was run on an overlapping NASA EMIT scene, the only other
spaceborne imaging spectrometer with comparable coverage. The two sensors agree
well: scene-mean reflectance correlates at Pearson r = 0.91 over 240 shared
bands, and all six target minerals correlate positively in MTMF detection, from
jarosite (r = +0.59) to muscovite (r = +0.34) (Figure 3). The agreement is
moderate rather than near-unity, as expected from the acquisition-date offset
and Tanager's finer grain — at 30 m Tanager resolves about four times the pixel
density of EMIT's ~60 m, and so a smaller minimum mappable feature.

## Acid-mine-drainage hazard

The secondary minerals that record acid generation are spectrally distinct,
which makes a hedged hazard proxy possible. Jarosite is stable only in acidic,
oxidizing, sulfate-rich conditions and is the diagnostic active-acid indicator;
the iron oxyhydroxides are the higher-pH oxidation products; gypsum, in the
absence of the acidic iron phases, points to a buffered setting (Swayze et al.
2000). Each pixel is assigned an ordinal acid-generating-potential tier from the
most acidic indicator present, rather than by summing abundances across
minerals. At Bingham the high-potential pixels cluster around the pit and
tailings ground (Figure 4). The layer is a spectral indicator of surface
mineralogy, not a measured pH or flux, and its tiers are relative within a
scene; it is a screening tool, not a substitute for sampling.

## Limits

The 30 m grain resolves features larger than about one hectare; the maps are of
surface mineralogy, not bulk chemistry or depth; the spectral library can
mismatch exotic phases, so the scope is held to the well-characterized
alteration assemblage; and there is no field validation — the Goldfield test is
against another remote-sensing product, which bounds agreement at the
alteration-group level rather than per-mineral abundance. The
acid-generating-potential layer is unvalidated at Bingham, where jarosite is
absent from the regional reference map.

## Impact and the case for more scenes

The end users are the state geological surveys and the federal agencies (USGS,
BLM, EPA) that characterize critical-mineral potential and mine-waste hazard,
for whom a reproducible, library-anchored mineral map over a named district is
directly usable. The result also makes an archive argument: Tanager's shortwave
infrared resolves mineralogy that the multispectral record cannot, and the
public benefit of that capability is largest over mining districts — the scenes
the open archive should grow. The pipeline ships as an open tool
(`tanager-minmap`), reproduces from a clean clone, and deposits its derivative
maps with a citable DOI.

---

*Figures.* (1) Goldfield/Cuprite dominant-alteration-mineral map. (2) Tanager
vs Sentinel-2 band-ablation at the 2200 nm Al-OH doublet. (3) Tanager–EMIT
cross-sensor comparison. (4) Bingham acid-generating-potential proxy.

*References.* Boardman (1998), 7th JPL Airborne Earth Science Workshop. Clark
and Roush (1984), J. Geophys. Res. 89, 6329–6340. Kokaly et al. (2017), USGS
Data Series 1035. Rockwell and Bonham (2017), ASTER-derived mineral and
alteration maps of the western US, USGS data release, doi:10.5066/F7CR5RK7.
Swayze et al. (2000), USGS OFR 2000-0205.
