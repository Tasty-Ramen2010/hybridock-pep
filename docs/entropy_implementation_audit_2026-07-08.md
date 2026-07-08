# Why our entropy term was misleading: WE replicated the physics wrong (implementation audit)

**Date:** 2026-07-08 · Ram's principle: *"physics can never say it's right and then fail — it is us who are wrong
in replicating physics."* Correct. The confinement method is sound, published thermodynamics. Our E354 result was
weak/unstable (corr(TΔS,residual) −0.06..−0.22, worse than a crude proline proxy). **So the bug is in our
implementation.** Four errors, each confirmed by the literature. All four push the same way: we computed the
*small* part of the entropy and missed the *dominant* part.

## Error 1 (THE core one): we captured the wrong entropy term
A flexible molecule's configurational entropy is **two parts**:
```
S_config  =  S_RRHO (vibration WITHIN one basin)  +  S_conformational (Gibbs–Shannon over occupancy of MANY basins)
```
([JCTC 2c00858, "Reliable Entropy on Flexible Molecules"](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00858): absolute
entropy = RRHO + Gibbs-Shannon of the Boltzmann occupation of conformational levels.) Our single-reference RMSD
confinement pins the peptide to **one** reference and measures its fluctuation around it — that is **only S_RRHO,
the small within-basin vibration.** For a floppy free peptide the **S_conformational (inter-basin) term dominates**
— and we computed **zero** of it. We measured the wrong, minor term. **This is why the crude proline/cyclic proxy
(+0.019 r) beat our confinement:** proline/disulfide directly encode inter-basin restriction — the dominant term we
missed — while confinement measured intra-basin jiggle.

## Error 2: Cartesian coordinates inflate/mis-estimate multi-well entropy
Quasi-harmonic/confinement in **Cartesian** coordinates *"markedly overestimates configurational entropy for systems
with multiple occupied wells… magnified by Cartesian instead of bond-angle-torsion coordinates"* ([JCTC ct0500904](https://pubs.acs.org/doi/10.1021/ct0500904)).
We used Cartesian RMSD — the worst case. The correct coordinate system is **internal / torsion (BAT)**.

## Error 3: implicit solvent gives the wrong ensemble
GB (gbn2) conformational ensembles are *"significantly different from explicit… overabundance of α-helix, salt
bridges overstabilized, do not identify the native state"* ([PMC4810457](https://pmc.ncbi.nlm.nih.gov/articles/PMC4810457/)).
Our free-peptide MD ran in gbn2 → an unphysical ensemble → wrong fluctuations → wrong entropy. Free-peptide entropy
needs **explicit solvent** (or an explicit-corrected estimator).

## Error 4: ~4 ps sampling cannot see the basins
Converging a free peptide's conformational ensemble needs sampling *"orders of magnitude greater than the folding
time"* ([Biophys Rev enhanced sampling](https://pmc.ncbi.nlm.nih.gov/articles/PMC3271212/)); inter-basin transitions
are the slow modes. Our 4 ps equilibration samples **one basin** — structurally guaranteeing we miss S_conformational
(Error 1). Even with perfect coordinates and solvent, 4 ps could not capture the dominant term.

## The verdict
The physics was right; our estimator violated **four** of its assumptions and, as a consequence, measured only the
minor within-basin vibrational entropy while the peptide-binding signal lives in the **between-basin conformational
entropy**. Our "entropy term" wasn't entropy — it was vibrational jiggle. Not a physics failure; a replication
failure, exactly as Ram argued.

## Corrected estimator — PRISM-S v2 (conformational entropy, the dominant term)
Compute **S_conformational** directly:
1. **Torsion-space (BAT) dihedral entropy** — histogram backbone φ/ψ + sidechain χ over the ensemble; Gibbs-Shannon
   over rotamer/basin occupancy, with **MIE 2nd/3rd-order** correlation corrections (backbone↔sidechain) — the term
   the config-entropy literature says dominates.
2. **Real ensembles, not one reference** — free-peptide ensemble in explicit solvent (or REST2/OPES enhanced
   sampling); bound ensemble from the docked-pose cluster. TΔS_bind = S_conf(free) − S_conf(bound).
3. **We already have partial machinery** — `src/hybridock_pep/scoring/free_entropy.py` and `per_residue_entropy.py`
   use dihedral/Ramachandran entropy. **AUDIT these for the same 4 errors** (single-basin? Cartesian? implicit? short?)
   — the fix may be repairing existing code rather than a new build.

## On --ultra being "QM + desolvation + derivatives" (not basic MD) — the component bugs
Ram is right that the intended --ultra is the PRISM physics stack, not just MM-GBSA. Their current defects:
- **QM cluster (E346)**: high variance, cluster-cutoff-sensitive, over-binds charged clusters. **Fix**: average over
  ≥3 cluster radii + consistent implicit solvent; report the ensemble mean, not a single cut.
- **RISM desolvation (E356)**: the 3-way exchem difference (complex−rec−pep) is a catastrophic cancellation of ~large
  size-dependent terms (+192..+499 kcal seen). **Fix**: use **per-voxel GIST** desolvation localized to the interface
  (displaced-water thermodynamics), not the whole-system 3-way difference — the same "derivative/local" trick that
  fixed the charged FEP.
- **ECC-FEP (E343)**: sound for charged ΔΔG. **Honest caveat**: QM/desolvation/ECC all target ΔΔG/charged terms,
  which we proved are **not** the absolute-Kd bottleneck (residual has no charge/desolv shape). Fixing them redeems
  --ultra for **selectivity/ΔΔG**, the regime where it belongs — not for absolute Kd.

## Where we still lack (updated)
- **The dominant entropy term (S_conformational) is still uncomputed correctly** — this is the real redemption and the
  one physics lever with the right size to matter. Build PRISM-S v2.
- **Absolute-Kd r-ceiling (~0.38 after fixes 1+3)** remains set by weak features + the general-model info limit; the
  fixes improve predictions honestly but do not smash it.

---
### Sources
[JCTC 2c00858 flexible-molecule entropy](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00858) · [ct0500904 QH Cartesian overestimation](https://pubs.acs.org/doi/10.1021/ct0500904) · [PMC4810457 GB ensemble bias](https://pmc.ncbi.nlm.nih.gov/articles/PMC4810457/) · [PMC3271212 enhanced sampling convergence](https://pmc.ncbi.nlm.nih.gov/articles/PMC3271212/) · [PMC2790395 quasiharmonic corrections](https://pmc.ncbi.nlm.nih.gov/articles/PMC2790395/) · [GIST displaced-water entropy](https://amberhub.chpc.utah.edu/gist/).
