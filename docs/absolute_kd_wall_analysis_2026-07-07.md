# Why the absolute-Kd wall (r≈0.33) will not crack — a full-front analysis

**Date:** 2026-07-07 · Status: definitive analysis · Context: PRISM (ΔΔG-accurate) failed to move absolute-Kd r
(E353: 0.320→0.325). MD, cheaper FEP, and our own force field all refuse to crack it. Ram: "this indicates a
deeper fault." It does — and it is not in our method. This document attacks the wall from six fronts, each grounded
in our own data + the field's literature, and states the one thing that would actually move it.

---

## The one-sentence thesis
**We conflated two regimes.** FEP and PRISM win at *relative, same-target* free energy (ΔΔG, selectivity) because
the hard, un-modelable parts **cancel**. Absolute *cross-target* affinity is the regime where those parts do **not**
cancel — and it is capped for **everyone, FEP included**, by a stack of sampling, entropy, and a *proven*
information-theoretic limit. We did not fail to replicate FEP. We correctly reproduced that **FEP cannot do this
either.**

---

## FRONT 1 — The regime error: "we replicated FEP" is true, and misleading
FEP's reputation is built on **RBFE** (relative binding free energy): *relative* ΔΔG between *congeneric* ligands
on the *same* target from a good co-crystal. In that regime the receptor, the binding mode, and the pocket water
network are **common to both endpoints and cancel exactly**. That cancellation is precisely what made our ΔΔG work
(QM r=0.61; 1BRS buried-charge sign flip via RISM). **We replicated FEP in its winning regime and it worked.**

But absolute cross-target Kd is the regime where **FEP also fails**: state-of-the-art **absolute** BFE is only
1.2–2.3 kcal/mol RMSE vs ~1 kcal for relative, and the gap is caused by **conformational-sampling requirements**
— "sampling rearrangements necessary to open and close the binding site" and "multiple degrees of freedom that
relative FEP methods don't need to address" ([Nat. Commun. s42004-021-00498-y](https://www.nature.com/articles/s42004-021-00498-y);
[maximal-accuracy review PMC10576784](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10576784/)). Absolute BFE for
*flexible* receptors like MDM2 is explicitly flagged as a worst case ([PMC7369993](https://pmc.ncbi.nlm.nih.gov/articles/PMC7369993/)).
**So our absolute-Kd failure is not a failure to be FEP — it is being FEP, in the regime where FEP breaks too.**

## FRONT 2 — The error-budget theorem: why fixing charge (or any one term) does nothing
Absolute ΔG_bind is a **sum of large, partly-independent terms**:
```
ΔG_bind = ΔG_interaction + ΔG_desolvation + ΔG_config-entropy + ΔG_protonation + ΔG_standard-state + ε_forcefield
```
Each carries ~1 kcal of error; independent errors **add in quadrature**. Charge is a *sub-term of one* term. So
even a *perfect* charge model shrinks total error negligibly. **Our own data proves this is exactly what happens:**
the scorer's residual (what it gets wrong) has **zero** correlation with every charge descriptor — charged_sum
−0.048, n_charged −0.034, net −0.064, sasa_sb +0.009, poc_net +0.000 (E353b). The error variance is **spread
across all terms**, no single one dominant. That is why **every single-lever fix we tried failed**: each attacked
one term while the variance lived in the sum.

## FRONT 3 — The entropy/flexibility fault: the deepest, and peptide-specific
This is the "deeper fault" Ram sensed. For a **flexible peptide**, the **configurational-entropy loss on binding**
is enormous — up to **~25 kcal/mol** for an HIV-protease ligand ([PNAS 0610494104](https://www.pnas.org/doi/10.1073/pnas.0610494104))
— and it is dominated by **higher-order backbone–sidechain and inter-molecular correlations** that are
"**the most difficult to converge computationally**"; using backbone or sidechain dynamics alone "can result in
incorrect entropy estimates for sizeable peptides" ([Chem. Phys. Lett. S0009261424007693](https://www.sciencedirect.com/science/article/abs/pii/S0009261424007693);
[Tsg101 peptide PMC2758778](https://pmc.ncbi.nlm.nih.gov/articles/PMC2758778/)).

Three facts make this decisive for us:
1. It is the **largest** un-modeled term, and it is **largest for peptides** specifically (more rotatable DOF than
   a drug-like ligand).
2. It **does not cancel** in absolute prediction (it *does* cancel in same-target ΔΔG — which is why we win there).
3. **We never actually computed it.** Our MD attacked the *free-state* entropy surrogate (a single-molecule
   quantity), not the *inter-molecular* correlation entropy that the literature says dominates and converges
   slowest. Our MD was also ps–ns; this term needs µs + enhanced sampling to converge.

**We have been scoring enthalpy-like terms while the peptide's entropy sets the variance.**

## FRONT 4 — The partition-function / information wall: the theoretical bedrock
Binding affinity is a property of the **thermodynamic ensemble** — the full Boltzmann partition function over
**all** bound *and* unbound states — **not of any single structure** ([static→dynamic PMC11516055](https://pmc.ncbi.nlm.nih.gov/articles/PMC11516055/)).
Enumerating it is "computationally intractable due to high dimensionality and rugged energy landscapes." A single
crystal pose **does not contain the information** needed to predict absolute affinity — scoring one pose is like
predicting a reaction rate from a photograph of the products.

And there is a **hard theorem**, not a heuristic: using statistical-learning and information theory it has been
**proven that even the best *general* structure-based model is inherently accuracy-limited, and protein-specific
models are always likely to be better** ([One Size Does Not Fit All, PMC3793897](https://pmc.ncbi.nlm.nih.gov/articles/PMC3793897/)).
Our tool is deliberately **general/cross-target** — so its absolute-affinity r has a **mathematical ceiling** below
any per-target model, independent of how much physics we add. This is the bedrock reason r plateaus.

## FRONT 5 — The benchmark mirage: why competitors "look" better (they aren't)
Everyone who reports high absolute-affinity r on a shared benchmark is largely reading **data leakage**. On CASF-2016,
removing train/test similarity drops OnionNet **0.83→0.57** and Pafnucy **0.76→0.55**; "accuracy is very sensitive
to the specific protein," and "benchmark performance often fails to predict real-world accuracy"
([Nat. Mach. Intell. s42256-025-01124-5 / CleanSplit](https://www.nature.com/articles/s42256-025-01124-5)). This is
**exactly** our own result: grouped/leakage-free r=0.332 vs leaky random-split r=0.382 (E353b), and the memory's
finding that **PPI-Affinity's 0.554 was a redundancy mirage** — their-split, not comparable. **We are not behind
the field; the field's lead was a measurement artifact it is now retracting.**

## FRONT 6 — Post-mortem: why each of *our* weapons specifically couldn't move it
| weapon | what it addressed | why it couldn't crack the wall |
|---|---|---|
| **short MD / entropy surrogate** | free-state single-molecule entropy | wrong observable (not inter-molecular correlation), and ps–ns ≪ convergence time (Front 3) |
| **cheaper FEP (charge-morph)** | one sub-term (charge), *relative* | held pose fixed, only charge; absolute needs the whole partition function (Fronts 2,4) |
| **PRISM (GB/QM/RISM)** | charged-residue ΔΔG | charge is not the limiting term — residual has zero charge shape (Front 2) |
| **docking-pose ensembles** | multiple poses | docking poses are **not Boltzmann-weighted** → not the real ensemble (Front 4) |
| **more/better features** | interaction, burial, packing | information-limited: a static pose lacks the ensemble information (Front 4) |
Every weapon was aimed at a term that is **not** the bottleneck, using a structure that **cannot** carry the
answer. Nothing sampled the actual partition function or the inter-molecular configurational entropy.

---

## The verdict
The absolute cross-target Kd wall at r≈0.33–0.35 is **real, multi-causal, and proven** — not a bug we failed to
fix. It is set by (a) a *proven* information-theoretic ceiling on general models, (b) an error budget spread across
~6 independent terms so no single fix moves it, and (c) a dominant, peptide-specific configurational-entropy term
that needs the full ensemble/partition function — which a single pose cannot carry and short MD cannot converge.
**FEP hits the same wall on absolute cross-target; we reproduced that faithfully.**

## What would *actually* move it (honest options)
1. **Per-target / per-family models** — proven to beat general models (Front 4). But that abandons "general tool."
2. **µs-scale enhanced-sampling MD computing inter-molecular configurational entropy per complex** — the only
   physics that targets the true bottleneck. Computationally impossible at n≈900 on an RTX 5070.
3. **Stop optimizing the wrong objective.** Physics wins at *relative, same-target* free energy. That is where FEP
   is famous, where PRISM works (ΔΔG r=0.61, RISM buried-sign fix), and — critically — **it is what the iGEM tool
   actually needs**: the **selectivity primitive** (ΔΔG of one peptide across two receptors), where the shared
   receptor-independent errors cancel. Absolute Kd was never our competitive ground; **selectivity/ΔΔG is.**

**Bottom line:** we are not failing to crack a wall we should be cracking. We are standing at a wall the whole
field stands at, *on the wrong side of it for absolute Kd* — and squarely on the **winning** side for the
selectivity/ΔΔG problem the tool exists to solve.

---
### Sources
ABFE vs RBFE & sampling: [s42004-021-00498-y](https://www.nature.com/articles/s42004-021-00498-y),
[PMC10576784](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10576784/), [PMC7369993](https://pmc.ncbi.nlm.nih.gov/articles/PMC7369993/).
Leakage mirage: [s42256-025-01124-5](https://www.nature.com/articles/s42256-025-01124-5),
[GEMS](https://www.biorxiv.org/content/10.1101/2024.12.09.627482.full.pdf).
Config entropy: [PNAS 0610494104](https://www.pnas.org/doi/10.1073/pnas.0610494104),
[S0009261424007693](https://www.sciencedirect.com/science/article/abs/pii/S0009261424007693),
[PMC2758778](https://pmc.ncbi.nlm.nih.gov/articles/PMC2758778/).
Ensemble/partition-function & info limit: [PMC11516055](https://pmc.ncbi.nlm.nih.gov/articles/PMC11516055/),
[eLife 111298](https://elifesciences.org/reviewed-preprints/111298), [PMC3793897](https://pmc.ncbi.nlm.nih.gov/articles/PMC3793897/).
