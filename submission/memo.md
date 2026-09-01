# Tanager VSWIR reveals mine-waste and alteration minerals that broadband satellites blur

**Planet Tanager Open Data Competition entry.**
Alex Bradley, Department of Earth, Environmental, and Planetary Sciences,
Washington University in St. Louis (abradley@wustl.edu).

## The question

What can Tanager's 426-band visible-to-shortwave-infrared (VSWIR) spectra tell
us about mine waste that broadband satellite imagery cannot? Hydrothermal
alteration and mine waste are recorded in a small set of minerals — alunite,
kaolinite, jarosite, the iron oxides, gypsum, and the white micas — whose
diagnostic absorptions fall in the shortwave infrared. We test whether Tanager
preserves those signatures where Sentinel-2 blurs them, and whether they can
help identify mine waste showing evidence of acidic oxidative weathering, at
two well-studied districts: the Bingham Canyon / Kennecott porphyry-copper mine and tailings
impoundment in Utah, and the Goldfield hydrothermal district in Nevada,
adjacent to the USGS Cuprite spectral benchmark.

## Approach

Mineral identity is anchored to version 7 of the USGS spectral library
(splib07a; Kokaly et al. 2017): every endmember is a measured library spectrum resampled to the
Tanager wavelength axis, and no spectral shape is synthesized. We map the target minerals
from each scene's surface-reflectance product using two complementary methods. Continuum-
removed band depth (Clark and Roush 1984) measures the diagnostic absorptions
directly — the 2200 nm aluminum-hydroxyl (Al-OH) doublet that separates alunite from kaolinite, the
2265 nm jarosite feature, the 2340 nm gypsum-carbonate feature, and the
visible-to-near-infrared ferric-iron bands. A mixture-tuned matched filter
(MTMF; Boardman 1998) then produces a relative abundance score for each
mineral against the scene's own background covariance, with an infeasibility score that suppresses
spectrally implausible detections. The analytical stages run from one installed
command, and the code and its shared data layer are public.

## The 2200 nm test: Tanager vs Sentinel-2

The case for contiguous VSWIR is sharpest at the 2200 nm Al-OH doublet. Alunite
and kaolinite (advanced-argillic versus argillic alteration, an important
distinction in hydrothermal alteration mapping and ore characterization) differ in the precise
position and shape of this feature. Convolving the measured library spectra
with Sentinel-2's published spectral response functions reduces their
separability: the alunite-kaolinite spectral angle falls from 5.1° to
2.6°, a 50% loss, because a single broad Sentinel-2 band (B12) spans the entire
doublet (Figure 1). The loss is specific to the shortwave infrared, not
universal; the visible-to-near-infrared jarosite-goethite contrast survives
degradation, which is the control showing the effect is the doublet itself and
not coarse resampling.

## How the maps compare with USGS alteration mapping

At Goldfield the maps were compared with the USGS ASTER alteration map of the
district (Rockwell and Bonham 2017), an independent remote-sensing product with
no shared calibration or acquisition. The Tanager scores separate the published
alteration zones at the alteration-group level: the Al-OH band depth and the
gypsum-carbonate band depth each reach a descriptive pixelwise rank-AUC of
0.78, while alunite and muscovite MTMF each reach 0.70, with alunite peaking in
the advanced-argillic zone at Cuprite and muscovite in the surrounding sericite
(Figure 2). The Fe-oxide and kaolinite layers do not separate as cleanly; the
published categorical classes subdivide these phases differently than the
spectral library groups them, and that disagreement is reported rather than
tuned away. Bingham agrees more weakly with the regional alteration map: a porphyry
system whose pervasive sericite does not partition into the published
acid-sulfate zones, illustrating that agreement depends on how well the
spectral classes correspond to the reference map's classes.

## Cross-sensor agreement with EMIT

The same mineral-mapping pipeline was run on a pre-selected overlapping scene from
NASA's EMIT imaging spectrometer. Scene-mean reflectance correlates at Pearson r = 0.962
over 240 shared bands (spectral angle 3.72°), and all six target-mineral maps
correlate positively, ranging from muscovite (r = +0.335) to jarosite
(r = +0.584) (Figure 3). Because both analyses use the same code and mineral
endmembers, this tests cross-sensor consistency, not independent accuracy. The map agreement is
moderate, not near-unity, as expected from the acquisition-date offset
and the different delivered grids. Tanager's delivered 30 m pixels cover about
one-quarter the area of EMIT's ~60 m pixels; we compare the delivered
products, not native instrument footprints.

## Screening for acidic mine-waste conditions

Acid-generating mine waste leaves mineralogical clues at the surface, and those
minerals are spectrally distinct. Jarosite forms under acidic, oxidizing,
sulfate-rich conditions and provides a strong mineralogical indicator of
acidic oxidative weathering; iron oxyhydroxides are generally associated with
oxidation under less acidic conditions; gypsum without the acidic iron phases
is more consistent with a comparatively buffered setting (Swayze et al. 2000). Each pixel is assigned an ordinal screening tier from the mineral indicator
associated with the most acidic conditions detected there. The tiers rank
surface mineralogical evidence for acidic conditions within the scene; they
are not estimates of acid-generating capacity, pH, or acid flux. At Bingham,
many of the pixels with the strongest mineralogical indicators of acidic
conditions occur around the pit and tailings areas (Figure 4). The layer is a
screening proxy, not a substitute for sampling.

## Limits

The products have 30 m pixels, so isolated sub-pixel features and fine
mine-waste structures cannot be resolved; the maps are of surface mineralogy,
not bulk chemistry or depth; the spectral library can
mismatch exotic phases, so the scope is held to the well-characterized
alteration assemblage; and there is no field validation — the Goldfield test is
against another remote-sensing product, which bounds agreement at the
alteration-group level, not per-mineral abundance. The
mine-waste acidity screening proxy is unvalidated at Bingham, where jarosite
is absent from the regional reference map.

## Impact and the case for more scenes

This is material identification against measured reference spectra —
material-specific screening, not generic land-cover classification: each pixel receives a scene-relative score against a measured
reference library, and the strongest supported candidate is mapped. The end
users for this kind of product would be the state geological surveys and
federal agencies (USGS, BLM, EPA) that characterize critical-mineral potential
and mine-waste hazard; for them, a reproducible, library-anchored mineral map
over a named district could serve as a candidate screening layer for field
follow-up. The result also
makes an archive argument: the public benefit of shortwave-infrared mineral
mapping is largest over mining districts — the scenes the open archive should
grow. The pipeline ships as an open tool
(`tanager-minmap`, MIT) with its shared data layer at a tagged public
release, and the code and derivative mineral maps are archived at
doi:10.5281/zenodo.22218608.

---

*Figures.* (1) Tanager vs Sentinel-2 band-ablation at the 2200 nm Al-OH
doublet. (2) Goldfield/Cuprite alteration-group validation. (3) Tanager–EMIT
cross-sensor comparison. (4) Bingham surface-mineralogical acidity screening proxy.

*Data and references.* Tanager STAC data (www.planet.com/data/stac), © 2024–2025
Planet Labs PBC, CC BY 4.0; NASA EMIT reflectance via LP DAAC. Boardman (1998), 7th JPL Airborne Earth Science Workshop. Clark
and Roush (1984), J. Geophys. Res. 89, 6329–6340. Kokaly et al. (2017), USGS
Data Series 1035. Rockwell and Bonham (2017), ASTER-derived mineral and
alteration maps of the western US, USGS data release, doi:10.5066/F7CR5RK7.
Swayze et al. (2000), Environ. Sci. Technol. 34, 47-54, doi:10.1021/es990046w.
