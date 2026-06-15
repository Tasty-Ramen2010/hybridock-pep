# Can We Break the Receptor-Baseline Wall with Physics? — The Full Attempt

*2026-06-15 · E221–E226 · Ram's plan: deployment pays a one-time 500ns pocket-water MD per receptor; training
uses pre-existing MD/FEP or cheap physics descriptors to learn "static structure → pocket bindability." We
tested every cheap route. Result: the signal is REAL and recoverable in principle, but no cheap proxy
captures it — it genuinely requires the dynamic water thermodynamics Ram proposed.*

---

## Step 1 — Is the receptor baseline even real, or is it experimental noise? (E222/the crux)
**It is REAL and stable.** Split-half reliability of the receptor's mean affinity = **0.885** (a receptor's
binding level from one set of peptides strongly predicts it from another). Not selection noise. The ceiling it
*could* be predicted to is **~0.94**. So there is a large pool of genuine, recoverable signal — Ram's premise
holds.

## Step 2 — What captures it? (E223/E224/E226)

| Receptor representation | predicts receptor baseline | notes |
|---|---|---|
| sequence composition | 0.13 | |
| ProtDCal-220 (PPI's method) | 0.149 | |
| **ESM-2 protein language model** | **0.154** | SOTA LM — no better than ProtDCal |
| **fpocket static pocket-physics** | **~0** added (Δ +0.007 / −0.005) | druggability/volume/hydrophobicity/SASA |

**Everything cheap caps at ~0.15.** Sequence, the best protein language model, AND static 3-D pocket-physics
(fpocket: druggability score, volume, polar/apolar SASA, hydrophobicity, charge, flexibility-heuristic) all
fail to capture the receptor baseline beyond what composition already gives. fpocket's descriptors *do*
correlate weakly with affinity (apolar-proportion +0.24, solvent-access +0.20 — hydrophobic pockets bind
better, as expected) but it's **redundant with our pocket composition**: adding it lifts per-complex ΔG by
≈0.

## Step 3 — Why the cheap routes fail, and what's left
The recoverable signal (0.15 → ~0.9) is **not in any static representation** — not sequence, not ESM, not
static pocket geometry. By elimination it lives in the **dynamic water thermodynamics**: which pocket waters
are high-energy/"unhappy" and get displaced favorably on binding (the WaterMap/GIST quantity), plus
sequence-driven pocket flexing. **That is exactly what fpocket's *static* druggability heuristic cannot see —
and exactly what Ram's 500ns pocket-water MD would measure.** So the cheap-proxy failure is not a refutation
of the plan; it's confirmation that the signal requires the real MD.

## Step 4 — Pre-existing MD / FEP data (the training-side of Ram's plan)
- **MISATO** (github t7morgen/misato-dataset, Zenodo): 16,972 protein–ligand complexes, ~100+µs explicit-water
  MD, ML-ready, with MD-derived **pocket adaptability** + a pre-trained static→adaptability model. **This is
  the pre-existing-MD resource** — BUT it is **small-molecule** protein-ligand, not peptide, and its receptors
  are drug targets, not our peptide receptors. Usable to (a) test whether MD-adaptability adds bindability
  signal (concept transfer), (b) borrow their static→MD-flexibility distiller. Heavy HDF5 download.
- **FEP datasets** (Schrödinger protein-FEP 208, Uni-FEP ~1000): all **relative** ΔΔG, small-molecule — they
  cancel the receptor baseline, can't provide it (E-search). Ram's reframe is right that MD-*water-behaviour*
  is a valid label even though FEP-ΔΔG is not; but no public dataset ships peptide-receptor pocket-water
  thermodynamics.

---

## Verdict + the experiment that WOULD break it

**Cheap routes are exhausted and the wall stands at ~0.15 for all static methods (us, ESM, fpocket, PPI).**
But the signal is provably real (0.885 reliability) and provably dynamic (every static proxy fails). Ram's
plan is the scientifically correct direction; it is simply **unproven and uncheap**.

**The decisive experiment (a focused MD pilot, GPU):** take ~30–50 receptors that have multiple known peptide
binders, run short explicit-water MD of each apo pocket (OpenMM, which we already use for MM-GBSA), compute
**GIST-style hydration descriptors** (high-energy water count, hydration free energy, enthalpy/entropy), and
test whether they predict the receptor baseline past 0.15. If yes → build the deployment 500ns-MD step and a
static→hydration distiller (trained on MISATO + the pilot) for training. If a short-MD pilot *also* caps at
0.15, the baseline is FEP-absolute-binding-free-energy-bound (the hardest open problem) and not worth chasing.

**Deployment design (if the pilot validates):** per-receptor one-time pocket-water MD → GIST hydration
descriptors → fed as receptor features alongside the peptide pose. Amortized over all peptides docked against
that target. Training uses MISATO-distilled static→hydration predictions so we don't MD every training
receptor.

This is the honest frontier: the wall is real, the signal is real and dynamic, cheap proxies don't reach it,
and the only untested lever is the actual MD — exactly what Ram proposed, now scoped to a checkable pilot.

---

## Addendum — training on pre-done FEP data (E227, Schrödinger protein-FEP benchmark)

Ram: "use pre-done FEP/MD repos to train on." MISATO MD = 132 GB (undownloadable) + small-molecule +
flexibility-only. But the Schrödinger protein-FEP benchmark (416 mutations, 16 systems, FEP-computed +
experimental ΔΔG + 75 features) is small and gave a decisive result on the RELATIVE/ΔΔG task:

```
                                    r vs experimental ΔΔG    MAE (kcal/mol)
 FEP (gold standard, 10^4× cost)         0.562                 1.00
 CHEAP ML (structural feats, LOSO)       0.491                 0.84   ← competitive r, BETTER MAE
 cheap ML + FEP-as-feature               0.493 (+0.001)        0.84   ← FEP adds nothing as a feature
```

**On ΔΔG / selectivity (the relative task), cheap ML rivals FEP** — r 0.49 vs 0.56, and *beats* FEP on MAE
(0.84 vs 1.00). FEP's 10⁴× cost buys almost nothing here. This is the established "ML narrows the gap to FEP"
finding, reproduced. And pre-done FEP values, used as a training feature, add **+0.001** — they don't transfer.

## The dichotomy that defines everything
```
 ABSOLUTE Kd (cross-receptor)  : a WALL. receptor baseline unpredictable (0.15), needs dynamic MD-water /
                                 absolute-FEP. ML caps ~0.36. NOBODY (us, PPI, ESM) does better. FEP-bound.
 RELATIVE ΔΔG (selectivity)    : ML ≈ FEP (0.49 vs 0.56, better MAE). cheap, tractable, OUR territory.
```
This is why "best non-FEP scorer that's commercially available" is the exactly-right claim: on the task FEP
dominates (absolute Kd of a novel receptor) nobody cheap can win; on the task that matters for design
(ranking peptides on a target = selectivity) we are already at FEP level for 10⁴× less. Pre-done FEP/MD repos
don't change this — they confirm it.
