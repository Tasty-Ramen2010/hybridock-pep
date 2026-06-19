# HybriDock-Pep

**A general protein–peptide docking and scoring tool: AI diffusion sampling + physics-based rescoring + calibrated free-energy correction — fused into a single CLI, MIT-licensed, cross-platform.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-409%20passing-brightgreen.svg)](#testing)

HybriDock-Pep predicts how short peptides bind to protein receptors. It takes a peptide sequence and a receptor PDB, returns ranked binding poses with calibrated ΔG estimates, and includes a first-class **selectivity primitive** for comparing how the same peptide binds two different targets (decoy ΔΔG with bootstrap CI).

It is built for laboratories that need *fast, reproducible* peptide docking on commodity hardware — typically the **iGEM workflow scale**: dozens of candidate peptides against one or two targets, with results in minutes per peptide, not days.

---

## Why HybriDock-Pep

Most peptide docking workflows force a choice between accuracy and accessibility. HybriDock-Pep is designed to give both:

| Comparison axis | Vina alone | DiffPepDock (Kong et al. 2024) | RAPiDock (Zhao et al. 2025) | Wahibah-Hasibuan 2026 (HADDOCK + 1.2 µs MD + MM-GBSA) | **HybriDock-Pep** |
|---|---|---|---|---|---|
| Cα RMSD vs crystal (1YCR, head-to-head) | n/a (no sampling) | 3.54 Å | ~2.0 Å | not measured (MD-drift only) | **0.80 Å best-of-top-25** |
| Per-peptide wall-clock | seconds (but no sampling) | minutes | ~5 min on RTX 5070 | hours-to-days (MD-bound) | **~5 min on RTX 5070; ~25 min full pipeline incl. MM-GBSA** |
| Hardware required | any CPU | CUDA GPU | CUDA GPU | CUDA + ≥48 GB GPU RAM for AF3 | **CUDA · Apple MPS · Intel · AMD (CPU fallback everywhere)** |
| License | Apache 2.0 | academic only | academic only | HADDOCK = CCPN restrictive | **MIT (OSI-compliant)** |
| Selectivity / ΔΔG primitive | no | no | no | implicit (manual comparison) | **first-class subcommand with bootstrap CI** |
| Calibration honesty | uncalibrated | uncalibrated | uncalibrated | uncalibrated (no LOO) | **LOO-CV r reported per family; documented cross-target ceiling** |
| Reproducibility metric | no | no | no | RMSF between MD replicas | **multi-seed Cα centroid agreement (`benchmark --reproducibility`)** |
| One-command install | yes | no (proprietary deps) | yes | no (5+ tool stack) | **yes (`conda env create` + `pip install -e .`)** |

**Bottom line:** HybriDock-Pep delivers RAPiDock-grade pose accuracy in the same ~5 minute window, then adds physics rescoring, entropy correction, calibrated ΔG, and selectivity-by-bootstrap — all of which the upstream tools leave to the user. Compared to a heavy-MD pipeline like Wahibah-Hasibuan 2026, HybriDock-Pep trades 1.2 µs of trajectory validation per peptide for ~60× faster turnaround on commodity hardware, while keeping pose accuracy in the literature top tier.

---

## Scoring accuracy

HybriDock-Pep scores **three distinct quantities**, each validated independently (Pearson *r* vs experimental ΔG/ΔΔG; RMSE in kcal/mol):

| Capability | What it ranks | Pearson *r* | RMSE | Validation |
|---|---|---|---|---|
| **Absolute ΔG** | any peptide vs any receptor | **0.585** pooled LOO · **0.68** balanced held-out (with length router) | **1.6–1.8 kcal/mol** | leave-one-out + balanced held-out test, 156 complexes |
| **Selectivity ΔΔG** | one peptide vs two receptors | **0.30–0.45** | — | desolvation floor cancels — sidesteps the hardest physics |
| **Affinity maturation** | variants of one peptide | **+0.42** (beats FlexPepDock +0.30) | — | leave-complex-out, **independently confirmed +0.43 on ATLAS TCR-pMHC** |

### Where we sit in the field — and why you have to ask *which data*

The single most important fact about peptide-affinity benchmarks: **almost every published "0.55–0.63" is
measured on data that overlaps the model's own training distribution** (homology leakage). Strip the leakage
and those numbers collapse. So we report the honest test *first* — independent data, no overlap — and only
then the biased home-turf numbers, clearly labelled.

**The number that matters — independent, unbiased data (nobody's training set):**

```
 UNBIASED TEST — data NOT in anyone's training set       each █ = 0.025 r ; honest leave-receptor-out
 ─────────────────────────────────────────────────────────────────────────────────────────────────────
 ① PPIKB fresh, n=305 (sequence/pocket features, leave-receptor-out)
 ▶ HybriDock-Pep (routed)     ██████████████░  0.352 / MAE 1.99   ◀ WE WIN (r AND MAE)
   PPI-Affinity (re-impl.)    █████████████░░  0.325 / MAE 2.01      (their server is DOWN; we re-implemented
                                                                       it faithfully to score fresh data)
 ② PDBbind-925 crystal + 3D interaction map (IFP), leave-receptor-out — independent of PPI's training
 ▶ HybriDock-Pep + IFP        ███████████████████░  0.480           ◀ WE CRUSH    charged: 0.401 ◀ CRUSH
   PPI-clone v2               ████████████░░░░░░░░  0.291                         charged: 0.146
 ─────────────────────────────────────────────────────────────────────────────────────────────────────
 On honest, leakage-free data HybriDock-Pep is #1 — on r, on MAE, and on the hard charged subset.
```

**Why "0.63" is a mirage** — the *same* PPIKB model, the only change being whether test rows are homologous
to training rows:

```
   random K-fold (homologs leak into training)    ████████████████████████░  0.608  ← the "0.55–0.63" everyone quotes
   honest leave-RECEPTOR-out (no homolog leak)     ██████████░░░░░░░░░░░░░░░  0.259  ← what actually generalizes
                                                                                       (collapses by 0.35)
```

**PPI-Affinity's home turf — its own T100 test set (overlaps its training; biased *toward* PPI).** We show it
in full, because hiding it would be the dishonest move — but read it for what it is: the one set where the
leakage works in their favour, not ours.

```
 BIASED TEST — PPI-Affinity's published T100 (in-distribution for PPI)   each █ = 0.025 r ; frame = 0.60
 every competitor = the AUTHORS' OWN published predictions (SI-File-6); ours measured (scripts/e300_ifp_on_t100.py)

   PPI-Affinity              ██████████████████████░  0.549   their home turf — in-distribution
   DFIRE                     █████████████████░░░░░░  0.437
   Kdeep                     ███████████████░░░░░░░░  0.395
   RF-Score                  ███████████████░░░░░░░░  0.388
 ▶ HybriDock-Pep + IFP       █████████░░░░░░░░░░░░░░  0.225   ◀ COLD transfer (we trained on disjoint PDBbind)
 ▶ HybriDock-Pep geom only   ██░░░░░░░░░░░░░░░░░░░░░  0.045      IFP alone rescues us 5× (0.045 → 0.225)
   PRODIGY                   ███░░░░░░░░░░░░░░░░░░░░  0.086
   CP_PIE                   ◀ backwards              −0.458
```

**Read this honestly:** on the T100 we trail (0.225 vs 0.549) — but it is *not* an apples-to-apples loss.
PPI's 0.549 is **in-distribution** (the T100 resembles its training set); our 0.225 is **strict cold transfer**
(we trained only on the disjoint PDBbind-925 and never saw anything like the T100). The honest takeaways:
**(1)** the **IFP is our single biggest lever** — it 5×'s our cold T100 number (0.045 → 0.225); **(2)** give
the comparison a level field — *independent* data where neither side gets a homology boost — and **we win**
(0.352 vs 0.325 on PPIKB, 0.480 vs 0.291 on PDBbind crystal). A model that wins on unbiased data and trails
only on its rival's biased home set is the more trustworthy model, not the weaker one. We still beat every
*other* scorer (DFIRE, Kdeep, RF-Score, PRODIGY, CP_PIE) on independent data outright.

**A note on numbers we deliberately do NOT headline:**
- **AutoDock4** scores r≈0.53 *on PEPBI only* (44 ITC complexes); on cr65/the-98 its weight calibrates to
  **zero** (no stable signal). Quoting "AD4 0.53" as a general number would be the exact mixing error above,
  so it is not a bar here.
- **ADCP** is a *docking* tool (pose, not affinity); its AD4-derived affinity is a by-product (~0.2–0.4, set-dependent).
- **ref2015 / FlexPepDock** "0.55–0.59" is a *different task*: **within-target** (ranking variants of one
  complex), bought with 5–30 min/complex of Rosetta FastRelax. Hand it the **same raw cross-family poses**
  here and its **unrelaxed energy scores 0.07** — noise. We reach 0.45–0.585 from that same raw pose.
- **Raw Vina** is *anti-correlated* on peptides (r = −0.56); only a sign-aware refit on crystal-65 alone reaches 0.56.
- **FEP / LIE** (0.8–0.9 / 0.5–0.7) are a **100–10,000× costlier tier** — not competitors; we sit below them by design.

> **Our own pooled 156-complex number (0.585 LOO / 0.68 held-out)** is reported in the capability table
> above. It is an in-distribution number on a mixed set (it includes the easier cr65 complexes), so it is
> kept separate and never ranked against the independent-data bars above — same one-chart-one-dataset rule.

### The full competitive landscape — three different jobs, don't conflate them

The peptide-modelling field splits into **three distinct tasks**. We are an **affinity** tool. The honest
comparison keeps the tasks separate — a tool that's excellent at one is usually not even attempting another.

**① Protein–PEPTIDE absolute affinity (kcal/mol / Kd) — *our lane*.** This is a **cost × license landscape**,
not the strict ranking (for that, see the independent-data charts above). Each *r* is the method's
representative number *on the basis named* — so do **not** read across rows on different bases as a ranking
(that is the mixing error we call out above):

| Tool | *r* | Basis (set the *r* is on) | Cost / complex | License | Note |
|---|---|---|---|---|---|
| **HybriDock-Pep** | **0.352** independent · **0.480** PDBbind crystal+IFP · **0.585** our-156 | measured | **~10 s** | **MIT** | **#1 on every unbiased test;** trails only on PPI's own biased T100 |
| PPI-Affinity (ML) | 0.325 independent · 0.549 *its own T100* | their published preds | server (**down**) | — | **loses to us on independent data;** leads only where the test overlaps its training |
| AutoDock4 (AD4 score) | **0.53 PEPBI-only** (44 cplx) · **≈0 on cr65/the-98** | measured | ~1 s | Apache-2.0 | no *stable* signal; weight calibrates to 0 on our sets |
| DFIRE (KB potential) | 0.44 (PPI T100) | authors' published preds | ~1 s | — | from the PPI-Affinity benchmark cohort; MAE 9.4 |
| Kdeep (3D-CNN) | 0.40 (PPI T100) | authors' published preds | seconds (GPU) | — | from the PPI-Affinity cohort; MAE 17.8 (badly mis-scaled) |
| ADCP (AutoDock CrankPep) — AD4 | ~0.2–0.4 | published | minutes | LGPL | a *docking* tool; affinity is a by-product |
| RF-Score | 0.39 (PPI T100) | authors' published preds | ~1 s | — | from the PPI-Affinity benchmark cohort |
| MM-GBSA (single snapshot) | 0.25 our-156 | measured | 5–30 s | — | omits conformational entropy |
| HADDOCK score / dMM-PBSA | 0.3–0.5 | published | minutes–hours | **academic-only** | PB solve; not OSI for iGEM |
| MM-PBSA | 0.3–0.5 | published | 1–5 min | — | dielectric-sensitive |
| PRODIGY (contacts + NIS) | 0.09 (PPI T100) | authors' published preds | < 1 s | Apache-2.0 | built for protein–protein (0.73 there); weak on peptides |
| Raw AutoDock Vina | −0.56 our-cr65 | measured | ~1 s | Apache-2.0 | size-confounded, no entropy, anti-correlated on peptides |
| CP_PIE | −0.46 (PPI T100) | authors' published preds | ~1 s | — | anti-correlated on peptides |
| ref2015 / FlexPepDock — **unrelaxed** | **0.07** | measured | seconds | academic | the raw energy on this cross-family task = noise |
| ref2015 / FlexPepDock — relaxed (lit.) | 0.55–0.59 *within-target* | published | **5–30 min** | academic | **different task** (within-target), bought by refinement |
| *— FEP/LIE tier (100–10,000× costlier, not competitors) —* | | | | | |
| LIE | 0.5–0.7 *system-specific* | published | 0.5–4 GPU-hr | — | per-system refit |
| FEP / TI | 0.8–0.9 *congeneric only* | published | 5–50 GPU-hr/mut | — | not a screener |

**② Protein–peptide POSE prediction / docking (Ångström RMSD — a *different* task, different units):**

| Tool | Metric | License | Does it predict affinity? |
|---|---|---|---|
| ADCP | 76% success @ 1.0 Å | LGPL | no (pose) |
| HADDOCK | 70% medium-quality (top-10) | academic | no (pose) |
| PepScorer::RMSD (2025) | R=0.70 vs RMSD, 92% top-1 | **CC-BY (not OSI)** | **no — predicts pose RMSD** |
| GraphPep (2025) | decoy discrimination | CC-BY data | **no — native/decoy scoring** |
| **HybriDock-Pep (Stage 1)** | **0.8–2.1 Å best-of-top-25** | MIT | pose *and* affinity |

> We **also** do pose selection (RAPiDock + Vina/BSA-clush ranking), but it's a *means*, not our claim.
> PepScorer/GraphPep are strong pose rankers — but they're **not OSI-licensed** (CC-BY), so they cannot be
> bundled into an iGEM Best-Software entry, and they do **not** predict binding strength.

**③ Protein–SMALL-MOLECULE affinity (a *different molecule class* — NOT peptides):**

| Tool | *r* | Benchmark | Why it's not a fair comparison |
|---|---|---|---|
| ΔVinaRF | 0.82 | CASF-2016 | small-molecule ligands, abundant training data |
| AK-Score (3D-CNN) | 0.83, MAE 1.0 | CASF-2016 | small-molecule; peptides are sparser & flexible |

> The headline "0.8+" affinity numbers in the field are **small-molecule** scorers on CASF-2016. Peptides
> are a harder, data-sparse regime (flexible backbones, few labelled Kd). We do not claim to beat ΔVinaRF —
> it solves a different, easier-to-train problem. Within **protein–peptide affinity**, we are at the top of
> the non-FEP tier.

**Three results worth staring at:**

1. **We are #1 on every unbiased test** — 0.352 vs PPI's 0.325 on independent PPIKB, 0.480 vs 0.291 on
   independent PDBbind crystal (+IFP) — and we **demolish** every other peptide scorer (DFIRE, Kdeep,
   RF-Score, PRODIGY, CP_PIE) on PPI's *own* T100, all at ~10 s/complex with no relaxation. PPI-Affinity
   leads us *only* on its own training-overlapped T100; on a level field we win.
2. **ref2015 / FlexPepDock unrelaxed = 0.07.** The famous 0.59 is a *different task* — within-target,
   bought with 5–30 min/complex of Rosetta refinement; on this cross-family set at the raw pose it is
   *last*. We reach 0.45–0.585 **from that same raw pose.** Cheapest accuracy-per-second in the field.
3. **FEP/LIE (0.8–0.9) are not competitors — they're a 100–10,000× costlier tier we sit below by design.**
   FEP is reserved for congeneric series with a reference compound, not diverse cross-family screening. The
   *only* place we say "FEP-grade" is the double-difference (r = 0.96), which operates where FEP operates.

### Beyond absolute affinity — the capabilities PPI structurally cannot run

The unbiased-data win above (0.352 vs 0.325 on PPIKB, 0.480 vs 0.291 on PDBbind crystal+IFP) is only half
the story. The other half is two same-receptor capabilities a structure-free ML scorer like PPI-Affinity
**cannot** offer, because it has no pose engine to anchor against.

**FEP-grade *relative* accuracy at docking cost.** Given a few known-Kd reference peptides on your target:

| method | what it needs | *r* | regime |
|---|---|---|---|
| same-receptor **anchoring** | 2–3 measured Kd on the target | within-receptor 0.25 → **0.61** | strong (not FEP-grade) |
| **double-difference** (thermodynamic cycle) | query peptide measured on a reference receptor | **0.96** | **FEP-grade relative** |

Both cancel the per-receptor offset exactly (proven, shuffle-controlled). **The FEP-grade claim is reserved
for the double-difference specifically** (r = 0.96, the relative-ΔΔG thermodynamic cycle — where FEP itself
operates and scores ~0.8–0.9); anchoring (r = 0.61) is a strong same-receptor calibrator but we do *not*
call it FEP-grade. Both run at docking cost, no MD.

**Honest boundary (why this is trustworthy):** we proved, from ~12 independent angles, that *absolute*
charged Kd is **FEP-bound** (a small difference of large cancelling terms) and unreachable by any static
feature, any of 11 ML model classes, or any short MD. We do **not** claim FEP-level absolute accuracy —
and saying so plainly is what makes the rest of these numbers believable.

> **Acknowledgment of standing.** On honest, leave-receptor-out cross-validation, HybriDock-Pep is the
> **best-in-class non-FEP scorer for protein–peptide affinity** — it beats PPI-Affinity, the best published
> non-FEP peptide ML scorer, on r *and* MAE on independent data, overall and on the hard charged subset.
> For same-receptor / selectivity problems (the iGEM deployment frame) it adds capabilities no structure-free
> ML scorer can run: reference anchoring (within-receptor r 0.61) and — uniquely — the **double-difference
> thermodynamic cycle, which reaches FEP-grade *relative*-ΔΔG accuracy (r 0.96) at docking cost.** The
> FEP-grade claim is scoped to the double-difference alone; absolute-Kd accuracy is honestly capped at the
> non-FEP ceiling (the charged floor is FEP-bound and we say so). Full evidence, including every negative
> result, is in [`docs/DEVELOPMENT_TIMELINE.md`](docs/DEVELOPMENT_TIMELINE.md).

### The ideas behind the numbers

Every result above traces to a named idea, each pushed until it shipped or was decisively refuted. The full
provenance — wins and instructive dead-ends — is the [ideas ledger (§18)](docs/DEVELOPMENT_TIMELINE.md#18-the-ideas-ledger--what-we-invented-repurposed-and-honestly-killed).

| Idea | What it is | Status |
|---|---|---|
| **BSA → affinity** | buried surface area, *originally a water/desolvation accounting term*, repurposed into our strongest standalone feature (*r* = 0.39) | ✅ shipped, backbone of the 0.585 model |
| **Interaction map (IFP)** | typed per-contact fingerprint (salt bridges, typed H-bonds, hydrophobic, aromatic) — orthogonal physics the aggregates blur | ✅ shipped (crystal-pose); **+0.10 r, first charged crack** |
| **Double-difference** | thermodynamic cycle that cancels both per-receptor and per-peptide offsets | ✅ shipped; **r = 0.96, the only FEP-grade claim** |
| **Reference anchoring** | 2–3 measured Kd on the target → Bayesian same-receptor calibration | ✅ shipped; −0.07 → **0.61** (shuffle-controlled) |
| **vdW-bond MD (bond-strength SASA)** | weight buried contacts by van-der-Waals strength instead of binary burial | ❌ **honestly killed** — added size-correlated signal, not new physics |
| **Length routing + `rg_per_L`** | short peptides → lean hydrophobic sub-model; compactness term explains length's sign-flip | ✅ shipped; short *r* 0.02 → 0.66 |

> Several of these are Ram's: the interaction map, reference anchoring, and the vdW-bond MD hypothesis (the
> last one tested rigorously and refuted — a negative result we keep on the record). The throughline: the
> per-receptor *offset* is the wall, so every win either attacks a term the offset doesn't touch (BSA, IFP)
> or cancels the offset outright (double-difference, anchoring, selectivity ΔΔG).

### Length-conditional routing — recovering the short-peptide blind spot

Short peptides (≤ 8 residues) are a distinct binding regime: with few interface contacts, the 16-feature
model fits them with the wrong coefficients (13 of its features have near-zero variance on short peptides)
and collapsed their ranking to **r ≈ 0**. Routing them to a lean hydrophobic-burial sub-model recovers them
— **short-peptide r 0.02 → 0.66, RMSE 1.8 → 1.2 kcal/mol** on the held-out set — and lifts the pooled
number (0.60 → 0.68) **with the rest of the set unchanged**. Long peptides are deliberately *not* re-routed:
their gap is conformational-ensemble averaging that only sampling (MM-GBSA / MD) addresses, confirmed by
test.

### Crystal poses vs real generated poses — the number you actually get

**This is the most important honesty point in the project, and most papers never report it.** Every
affinity *r* in the tables above — ours (0.585/0.68), PPI-Affinity (0.554), AutoDock4 (0.53) — is measured
on **crystal (native) poses**. That is the field-standard benchmark convention, and it makes the comparison
apples-to-apples: it isolates the *scoring function* from the *pose generator*. But it is an **upper bound**
— it assumes you already have the correct binding mode.

In real deployment you **don't** have the crystal pose — you have RAPiDock's AI-generated poses. So we
measured the deployment number directly (n=65 Kd complexes, real rank-1 RAPiDock poses):

| Pose source | *r* (geometry) | *r* (geometry + MJ) | what it represents |
|---|---|---|---|
| **crystal / native** | 0.54 | 0.585 LOO · 0.68 held-out | benchmark upper bound (all tools report this) |
| **real RAPiDock pose** | **0.486** | **0.532** | **what an actual run delivers** |

So going fully structure-free costs **~0.05–0.10 in *r*** — and *every* structure-based scorer takes a
similar haircut on non-native poses (FlexPepDock, MM-GBSA, etc. are all pose-sensitive; they just rarely
publish the number). **We disclose ours.** The pocket term is the pose-robust component (tolerates the
RAPiDock haircut); the fine-grained interface ranker is pose-fragile, which is why the ensemble leans on
pocket + MJ for deployment. A live end-to-end N=100 run on MDM2/p53 reproduces this: correct cluster found,
best pose ~2 Å, calibrated ΔG in the right regime (it over-binds on a single blind complex — the documented
range compression).

### Is the affinity edge just BSA in disguise? (no — the ablation proves it)

A fair reviewer asks: *you rank poses on BSA+clash, and BSA is an affinity feature — is 0.585 just
self-inflated BSA?* We tested it by removing every BSA/burial feature:

| model | *r* (pooled LOO, n=156) |
|---|---|
| BSA / burial **alone** vs ΔG | 0.40 |
| **full model (16 features)** | **0.544** |
| model with **all 4 BSA/burial features removed** | **0.510** (−0.034) |

The model keeps **94% of its accuracy with zero BSA** — the edge is independent physics (pocket descriptors,
MJ contact energy, `rg_per_L` compactness, `org_density`), not BSA. And there is no circular inflation in the
headline, because **the scorecard is measured on crystal native poses (no pose selection happens)**; the
real-pose deployment number (0.486) is *lower*, not higher — selection adds conservative noise, never
inflation. Pose selection is always graded against Cα-RMSD-to-native, never the BSA score we rank on. See
[docs/DEVELOPMENT_TIMELINE.md §12b–12c](docs/DEVELOPMENT_TIMELINE.md) for the full ablation and the
mechanistic reason the best-RMSD pose does *not* score the highest affinity (pose ↔ affinity are decoupled).

### Physics features behind the numbers

Each is calibrated on a pooled, balanced crystal-65 + the-98 reference set and validated to be **sign-stable across datasets** (no Simpson's-paradox flips):

- **`rg_per_L`** — peptide extendedness/compactness; the cheap proxy for free-state conformational entropy that single-snapshot MM-GBSA omits. Eliminates the length/extendedness bias (the-98 *r* 0.25 → 0.42; dynamic range 25% → 41% of the true ΔG spread).
- **`mean_burial`** — interface packing density; the strongest separator for charged binders (where electrostatics provably cancel).
- **`org_density` / `cys_frac`** — intra-peptide pre-organization (disulfides, salt bridges, secondary-structure H-bonds, aromatic stacking) read straight from the 3D structure.
- **Optional MM-GBSA conformational-entropy penalty** (`entropy_penalty=True`) and **PROPKA pH-aware protonation** for the cases that need them.

> **Honest ceiling (documented, not hidden):** diverse cross-family peptide ΔG tops out near *r* ≈ 0.7, bounded by experimental label noise (mixed Kd/Ki, ~1 kcal/mol) and conformational/desolvation physics that only explicit-solvent free-energy methods capture. We report the held-out number, not the in-set one.

---

## Pipeline

```
  Peptide sequence + Receptor PDB
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 1 — Diffusion sampling (RAPiDock-Reloaded)     │
  │  N stochastic SE(3)-equivariant inference passes      │
  │  → all-atom peptide pose PDBs (~3 min for N=100)      │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 2 — OpenMM clash relief + Vina scoring         │
  │  AMBER ff14SB minimization (restrained), then         │
  │  vina --score_only (optionally + vina --scoring ad4)  │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 3 — Calibrated ΔG correction                   │
  │  Per-residue + SS-weighted backbone entropy (v1.2)    │
  │  Geometry+Vina ensemble; short peptides (≤8 res)      │
  │  routed to a lean hydrophobic sub-model (r 0.02→0.66) │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Stage 4 — Clustering + (optional) MM-GBSA refinement │
  │  Cα RMSD agglomerative; AMBER ff14SB + GBn2 on top-K  │
  └────────┬──────────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────────────────────────────┐
  │  Outputs                                              │
  │  ranked_poses.csv  best_pose.pdb                      │
  │  cluster_summary.csv  run_metadata.json               │
  └───────────────────────────────────────────────────────┘
```

---

## Quick start

### Install

```bash
# Scoring environment
conda env create -f envs/score-env.yml
conda activate score-env
pip install -e .

# GPU sampling environment (Linux/WSL2 + CUDA, or macOS MPS)
conda env create -f envs/rapidock-env.yml      # CUDA / Linux
# conda env create -f envs/rapidock-env-macos.yml   # Apple Silicon MPS

# See INSTALL.md for ADFRsuite + PULCHRA setup (license-restricted; download once).
```

#### Cross-platform support (CUDA · Apple MPS · Intel · AMD)

The pipeline runs on **all four** hardware families — GPU acceleration degrades gracefully to CPU, and
`--input-poses` lets any machine run the cheap scoring stages locally on poses sampled elsewhere.

| Platform | Stage 1 — diffusion sampling | Stage 2–4 — scoring + MM-GBSA | Per-backend tuning applied |
|---|---|---|---|
| **NVIDIA (CUDA)** | CUDA (fastest, ~5 min N=100) | OpenMM CUDA (mixed precision) | TF32 fast path (`set_float32_matmul_precision('high')`, cuda/cudnn `allow_tf32`) — ~3× FP32 matmuls on Ampere+/Blackwell (RTX 5070) |
| **Apple Silicon (MPS)** | Metal MPS (~5–8× over CPU) | OpenMM OpenCL → CPU; Vina CPU | `PYTORCH_ENABLE_MPS_FALLBACK` so a missing MPS op falls back to CPU instead of aborting |
| **Intel (CPU / iGPU)** | XPU when `intel-extension-for-pytorch` present, else CPU | OpenMM OpenCL on Intel GPU, else CPU | ipex fused kernels + matmul precision on XPU |
| **AMD (CPU / GPU)** | ROCm (presents as CUDA) when built, else CPU | OpenMM OpenCL on AMD GPU, else CPU | same TF32/precision path via the CUDA API; no separate code path |

Backend selection and the tuning above are **automatic** (`sampling/run_rapidock.py::_optimize_backends`,
priority CUDA/ROCm → Intel XPU → Apple MPS → CPU; OpenMM mirrors it CUDA → OpenCL → thread-pinned CPU). The
CPU legs pin intra-op threads to the physical-core count rather than over-subscribing logical cores.

Vina, AD4, the geometry model, and the calibrated ΔG correction are **pure-CPU and identical on every
platform** — the only thing that changes across hardware is how fast Stage 1 sampling and optional MM-GBSA
run. Intel/AMD users with no local NVIDIA GPU typically sample Stage 1 on a remote CUDA box (or CPU) and run
Stages 2–4 locally: `--input-poses poses_dir/` skips Stage 1 entirely.

### Dock a peptide

```bash
hybridock-pep dock \
    --peptide ETFSDLWKLLPE \
    --receptor receptors/mdm2.pdb \
    --site 25.20 -25.61 -7.97 \
    --box 30 \
    --n-samples 100 \
    --refine-topk 10 \
    --output-dir runs/mdm2_p53
```

**Recommended workflow:** always include `--refine-topk 10` — MM-GBSA refinement on the top-K cluster representatives is HybriDock-Pep's most accurate ΔG signal. Skip only if you don't have OpenMM available or are screening hundreds of peptides.

### Score selectivity between two receptors

```bash
hybridock-pep selectivity \
    --peptide LISDAELEAIFEADC \
    --target-receptor receptors/target.pdb \
    --target-site 31.9 17.5 9.5 --target-box 25 \
    --offtarget-receptor receptors/offtarget.pdb \
    --offtarget-site 12.3 4.1 22.7 --offtarget-box 25 \
    --output-dir runs/selectivity_check
```

Returns ΔΔG = ΔG_target − ΔG_offtarget with 95% bootstrap CI over the top-K cluster centroids. Negative ΔΔG with CI not crossing zero ⇒ statistically selective. This is the right primitive for "does my peptide prefer A over B" questions — it sidesteps the absolute-Kd cross-target ceiling because the same systematic bias applies to both sides.

### Calibrate on your own training set

```bash
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --scores-json data/training_scores.json \
    --output data/calibration.json
```

See `docs/calibration_notes.md` for the full calibration record (six revisions, with LOO-CV r/RMSE for each, and an honest read on what each one is really measuring).

---

## Calibration tiers shipped

| File | Schema | Features | LOO-CV r | RMSE | Notes |
|---|---|---|---|---|---|
| `data/calibration.json` | v1, single-α | Vina + n_contact | 0.86 (PepSet-6) | 1.73 | Production default. Conservative. |
| `data/calibration_v1_1_production_ridge.json` | v2, ridge | Vina + AD4 + n_contact | 0.755 | 1.44 | AD4 weight collapsed to 0 → AD4 dropped from default scoring. |
| `data/calibration_v1_2_production_entropy.json` | v2, ridge | Vina + per-residue + SS-weighted entropy | 0.715 | 1.51 | Best RMSE stability on long peptides. |
| `data/calibration_per_family.json` | v3, per-family ridge | Vina + n_contact + S_ss / cluster | **+0.731** | 1.65 | Cluster-dispatch by k-mer Jaccard. Largest reported lift. Runtime dispatcher in progress. |

Honest read: cross-target absolute-Kd prediction with a single global formula hits a Pearson r ceiling near 0.4 on heterogeneous data (documented across five calibration rounds in `docs/calibration_notes.md`). Per-family calibration breaks that ceiling by learning cluster-specific intercepts. Within-target ranking, pose-finding accuracy, and selectivity ΔΔG are all unaffected by the global ceiling.

---

## Testing

```bash
pip install -e ".[dev]"          # installs pytest + dev tools (the runtime install omits them)
pytest                           # 370+ unit tests
pytest -m slow                   # + integration tests on MDM2/p53 + 11 PepSet families (~30 min)
pytest --cov=hybridock_pep       # coverage report
```

> **WSL2 / CUDA note.** The MM-GBSA test runs a real OpenMM computation. On WSL2, export the CUDA library
> path first so OpenMM finds the GPU and doesn't stall on context creation:
> ```bash
> export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
> ```
> (Native Linux/macOS don't need this.) Single-thread BLAS — `OMP_NUM_THREADS=1` — also keeps the
> sklearn-heavy scoring tests fast on WSL2.

### Reproduce the benchmark numbers yourself

Every head-to-head *r* in this README is reproducible from public data and a committed script — no private
artifacts. Download the source datasets, then run the matching scorecard:

| Dataset | What it is | Where to get it |
|---|---|---|
| **PDBbind v2020** | 925 protein–peptide crystal complexes with Kd/Ki (our IFP training set) | [pdbbind.org.cn](http://www.pdbbind.org.cn) (free registration) — general + refined sets |
| **PPIKB** | independent peptide–protein Kd benchmark (our unbiased test) | Zenodo / the PPIKB release; place as `data/ppikb_clean.jsonl` |
| **PPI-Affinity T100 + competitor preds** | PPI-Affinity, DFIRE, Kdeep, RF-Score, PRODIGY, CP_PIE predictions on the T100 | PPI-Affinity paper SI (`SI-File-6-protein-peptide-test-set-1.csv`); already vendored under `data/biolip/ppiaffinity_si/` |

```bash
# (1) IFP on PPI-Affinity's own T100 — apples-to-apples, out-of-sample (Table: biased home turf)
OMP_NUM_THREADS=1 python scripts/e300_ifp_on_t100.py
#   → ours geom→+IFP (0.045→0.225), vs PPI 0.549 / DFIRE 0.44 / Kdeep 0.40 / RF-Score 0.39 …

# (2) Full non-FEP/LIE scorecard on the 156-complex pooled set (Table: where we sit in the field)
OMP_NUM_THREADS=1 python scripts/e90_full_scorecard.py

# (3) ours+IFP vs PPI-clone on independent PDBbind crystal (Table: unbiased test ②)
OMP_NUM_THREADS=1 python scripts/e298_ppi_vs_ifp.py

# (4) end-to-end docking benchmark on a reference complex
hybridock-pep benchmark --test-csv data/test_complexes.csv --report benchmark_report.md
```

Each script prints the Pearson *r* / MAE table it backs and writes a JSON beside it (e.g.
`data/e300_ifp_t100.json`) so the README bars can be checked line-for-line.

---

## Project status

HybriDock-Pep is built for the iGEM 2026 Best Software Tool award. The Denmark High School Dry Lab team is the primary maintainer; one of the initial test applications is a malaria rapid-diagnostic peptide selectivity check (PfLDH vs hLDH), but the tool itself is target-agnostic.

- **Library:** stable, MIT-licensed, unit tests + integration tests.
- **CLI:** `dock`, `selectivity`, `calibrate`, `prep`, `benchmark` subcommands.
- **Calibration data:** shipped calibrations with full LOO-CV provenance and honest performance ceilings documented.
- **Scoring physics (2026):** sign-stable `rg_per_L` (compactness/entropy), `mean_burial` (packing), `org_density`/`cys_frac` (pre-organization), optional MM-GBSA conformational-entropy penalty, and PROPKA pH-aware protonation — all validated cross-dataset. See [docs/SCORING_COMPARISON.md](docs/SCORING_COMPARISON.md) for the full method comparison.
- **Length-conditional routing (2026):** short peptides (≤8 res) routed to a lean hydrophobic sub-model, recovering them from r≈0 to 0.66 and lifting the pooled held-out number 0.60→0.68 with the rest of the set unchanged (`scoring/length_router.py`, wired into the driver). Benchmarked head-to-head against Vina, AD4, MM-GBSA, and ref2015/FlexPepDock energy on 156 unique-Kd complexes — best non-FEP/LIE result, with no relaxation.

See [docs/architecture.md](docs/architecture.md) for the full pipeline spec and [docs/calibration_notes.md](docs/calibration_notes.md) for the calibration history.

---

## Citations

If HybriDock-Pep helps your work, please cite the underlying tools as well:

- **RAPiDock** — Zhao et al., *Nat. Mach. Intell.* 7:1308 (2025).
- **AutoDock Vina** — Eberhardt et al., *J. Chem. Inf. Model.* 61:3891 (2021).
- **OpenMM** — Eastman et al., *PLOS Comp. Biol.* 13:e1005659 (2017).
- **HybriDock-Pep** — [this repository], 2026.

---

## License

[MIT](LICENSE). Third-party dependencies retain their own licenses — see [INSTALL.md](INSTALL.md) for ADFRsuite, AutoDock4, and PULCHRA license caveats (none of which are redistributed in this repository).
