# Reference-Anchored Relative Scoring for Peptide–Protein Affinity

**Status:** design / pre-registration for an n=100 test
**Author:** Dry Lab (HybriDock-Pep), 2026-06-16
**One-line thesis:** We cannot *predict* the per-receptor error of an absolute ML scorer, but we can
*measure* it from a few known-Kd reference peptides on that receptor and subtract it. This converts the
FEP-bound absolute-Kd problem into the solvable within-receptor (relative) problem + a cheap calibration.

---

## 0. Why this is worth doing (grounding in what we already proved)

| Prior result | What it establishes | Role here |
|---|---|---|
| e246 forensics | ~55% of charged ΔΔG error variance is **between-receptor** (a per-receptor offset) | This is the term anchoring removes |
| e255 offset attack | That offset is **not predictable** from sequence, ESM, fpocket, Poisson–Boltzmann, or 0.6 ns GIST (all ≤ permutation null) | So we must *observe* it, not regress it |
| E254 two-stage ceiling | With the offset **known**, within-receptor charged reaches **r ≈ 0.755** | The anchored ceiling |
| Absolute-Kd ceiling (Jun 14) | Honest clustered-CV truth ≈ 0.35 for *everyone incl. PPI-Affinity*; the 0.55 numbers are redundancy/homology mirages | Why "just train a better absolute model" is a dead end |

The synthesis: the wall (b(R)) is real, large, and unpredictable — **but observable per-receptor.**
Anchoring is the only lever that turns an unpredictable nuisance into a measured constant.

---

## 1. Formalization

### 1.1 Error model

Let `S(p,R)` be our absolute ML score (estimated ΔG_bind) for peptide `p` on receptor `R`, and
`G(p,R)` the true binding free energy. Assume the error decomposes:

```
S(p,R) = G(p,R) + b(R) + c(p) + η(p,R)
```

- **b(R)** — per-receptor offset. Pocket dielectric / desolvation reference / scoring-function bias
  specific to that protein environment. Large, systematic, FEP-bound. **The target of anchoring.**
- **c(p)** — per-peptide systematic error. E.g. the scorer is off by ~X kcal/mol per unit net charge,
  or per salt bridge. Does **not** cancel unless test and reference are similar peptides.
- **η(p,R)** — irreducible interaction-specific residual (binding-mode dependent). Pure noise to us.

This additivity is **the central assumption** and is itself testable (§4, variance decomposition).

### 1.2 The thermodynamic cycle (why relative is computable when absolute isn't)

For test `p` and reference `r` on the **same** receptor `R`:

```
   p (unbound) + R   ──ΔG_bind(p)──►   p·R
        │                                  │
   alchemical p→r                     alchemical p→r
   in solvent  ΔG_free                in complex  ΔG_bound
        │                                  │
   r (unbound) + R   ──ΔG_bind(r)──►   r·R
```

Path independence of a state function closes the cycle:

```
ΔG_bind(p) − ΔG_bind(r) = ΔG_bound − ΔG_free ≡ ΔΔG(p→r)
```

The large absolute terms (full desolvation, the cancelling Coulomb vs Born terms that wreck charged
absolute scoring) are computed **inside ΔG_bind(p) and ΔG_bind(r) identically and subtracted away.**
We never form the small-difference-of-large-numbers. **That is the cancellation Ram is reaching for,
and it is exactly relative binding free energy (RBFE).**

### 1.3 Backing out an absolute Kd

If the reference Kd is known, `ΔG_exp(r) = −RT ln(1/K_d(r))`, then

```
ΔG_pred(p) = ΔG_exp(r) + ΔΔG_ML(p→r),   where ΔΔG_ML = S(p,R) − S(r,R)
```

Substituting the error model:

```
ΔG_pred(p) = G(p,R) + [c(p) − c(r)] + [η(p,R) − η(r,R)] + ε_exp(r)
                       └── b(R) CANCELLED ──┘
```

**The b(R) offset is gone exactly.** What remains:
- `c(p) − c(r)` — vanishes only if p and r are *similar* (⇒ **stratify by net charge**),
- `Δη` — irreducible scatter (⇒ **average over references**),
- `ε_exp(r)` — inherited reference assay noise (⇒ **high-quality Kd, average over references**).

### 1.4 Triangulation = empirical offset estimation (the clean result)

With K references `r_1…r_K` on receptor R, average the per-reference estimates:

```
ΔG_pred(p) = (1/K) Σ_k [ ΔG_exp(r_k) + S(p,R) − S(r_k,R) ]
           = S(p,R) − b̂(R),    b̂(R) ≡ (1/K) Σ_k [ S(r_k,R) − ΔG_exp(r_k) ]
```

**Triangulation over K references is literally an empirical per-receptor offset correction.** `b̂(R)` is
a direct estimate of `b(R)` from data, with variance

```
Var(b̂) = (1/K) · Var_r[ c(r) + η(r,R) + ε_exp(r) ]
```

Two consequences fall straight out — both confirming Ram's intuitions:
1. **More references → 1/K variance reduction** ("directional noise averages out").
2. **References with small, consistent `c(r)`** (same charge class) minimize the residual ⇒ stratify.

The optimal estimator is not a plain mean but a **precision/similarity-weighted mean** (§3, Bayesian).

### 1.5 The honest ceiling

Anchoring removes `b(R)` (≈55% of charged error variance) but **not** `c(p)` or `η`. So the realistic
ceiling is the **within-receptor** predictability, which E254 pegs at **r ≈ 0.75** for charged. Expect
anchoring to move charged from ~0 (cross-receptor) toward ~0.5–0.75 — **not** to 0.95. This is a
few-shot per-receptor calibrator, not a zero-data universal oracle. Say so plainly in the paper.

---

## 2. Failure modes (each tied to the algebra)

| # | Failure | Algebraic cause | Severity |
|---|---|---|---|
| F1 | Test and reference have **different net charge** | `c(p) − c(r)` large (charge-dependent scorer bias) | **High** (the charged case) |
| F2 | **Different binding mode / subsite** | `Δη` large; cycle assumes comparable complexes | High |
| F3 | **100 ps MD insufficient** | `ΔG_bound` not converged: slow side-chain/backbone/water/charge reorganization unsampled | Med–High |
| F4 | **Noisy reference Kd** (IC50≠Kd, mixed assays/conditions) | `ε_exp(r)` propagates 1:1 | Med |
| F5 | **ML error not additive** — real `p×R` cross term | `η(p,R)` not separable ⇒ anchoring can't reach it | **Fundamental cap** |
| F6 | **No reference for a novel/orphan receptor** | b̂(R) undefined | Med (scope limit) |
| F7 | **Reference too dissimilar** (extrapolation) | both `c(p)−c(r)` and `Δη` blow up | Med |
| F8 | **Evaluation leakage** (near-duplicate ref/test, or generic regularization mimicking the effect) | measures memorization, not cancellation | **Validity threat** |

---

## 3. Mitigations

- **F1 — charge stratification + classical Δ-correction.** Restrict references to the same net-charge
  class as p. *And* add a fast physics term to absorb the residual charge bias:
  `ΔG_pred = ΔG_exp(r) + [S(p)−S(r)] + λ·[E_elec(p) − E_elec(r)]`, where `E_elec` is a cheap
  Debye–Hückel-screened Coulomb or single-shot Poisson–Boltzmann (APBS, already wired in e248).
  λ fit on a held-out grid. This is delta-learning layered on the anchor.
- **F2 — geometric reference filtering.** Require references that share the pocket / epitope and have
  pose overlap (Cα-RMSD of the bound peptide core, shared anchor residues). Reject references binding a
  different subsite.
- **F3 — MD as relaxation, not alchemy; replicas; adaptive escalation.** Use 100 ps only to relax the
  pose + reorganize water, then score the endpoint. Run 3× short replicas, average. Flag high-variance
  pairs and escalate *only those* to real λ-FEP. Never claim 100 ps gives a converged alchemical ΔΔG for
  a large/charged perturbation.
- **F4 — Kd quality gate + averaging.** Kd/Ki only (no IC50 unless converted), single assay family per
  receptor where possible; variance ∝ 1/K from averaging.
- **F5 — measure it, then accept it.** Variance decomposition (§4) quantifies the between- vs
  within-receptor split; the within-receptor part is the hard ceiling. Report it; don't pretend past it.
- **F6 — homolog transfer / measure 2–3 anchors.** Anchor on a close homolog (accept residual b
  mismatch, downweight), or have the wet lab measure 2–3 reference Kds — which is *already* how the
  parent project picks receptors. For zero-data orphans the method degrades gracefully to absolute S.
- **F7 — similarity-weighted Bayesian references.** Weight reference k by a kernel
  `w_k ∝ exp(−d(p,r_k)²/2σ²) / σ_exp(r_k)²` (sequence/charge distance × inverse assay variance). Report
  a confidence that widens with min reference distance.
- **F8 — shuffle + permutation controls (§4.5).** Mandatory.

**Unifying upgrade — hierarchical Bayesian model.** Treat `b(R)` as a random intercept with a prior;
references update its posterior (strong shrinkage when few references, sharp when many). The K-reference
mean is the flat-prior special case. The fully rigorous form is the **DiffNet maximum-likelihood
estimator** (Xu 2019): reconcile *all* noisy relative edges `S(p)−S(r)` with the absolute anchors into
a single MLE of every node's absolute ΔG. Plain triangulation is the one-hop approximation.

---

## 4. Experimental protocol (n ≈ 100, implementable tomorrow)

### 4.1 Design

Within-receptor leave-one-peptide-out, on receptors that each have **many measured-Kd peptides**, so
every test peptide has same-receptor references. Target ~10–15 receptors × ~7–10 peptides ≈ 100 complexes.

### 4.2 Data sources

- **ATLAS** (TCR–pMHC) and **IEDB / MHC class-I binding** — many peptides per identical MHC receptor with
  measured affinity. Ideal absolute-Kd anchoring substrate. (ATLAS already cached.)
- **PDBbind peptide subset** (our 925, `data/pdbbind_peptides.jsonl`) grouped by receptor via UniProt
  mapping — proteases, bromodomains, SH3, MDM2 analogs give multi-peptide receptors.
- **SKEMPI v2** — same-receptor mutation ΔΔG (WT = built-in reference). Use for the **relative-engine /
  cancellation proof** (Phase 0), not absolute back-out (labels are ΔΔG). *We already have this staged in
  `scripts/e260_anchor_triangulation.py` on the 1,122-record charged set `data/e254_recs.json`.*
- **PROPEDIA / Propedia, AS-Bind** — supplementary multi-peptide-per-receptor complexes.

Selection rules: ≥5 Kd-quality peptides per receptor; cluster receptors (≤30% seq id between groups) to
prevent cross-receptor leakage; record net charge + length per peptide for stratification.

### 4.3 Per-complex pipeline (identical for test and references)

1. **Pose:** RAPiDock Stage 1 (or crystal pose where available).
2. **Relax:** 100 ps OpenMM, AMBER ff14SB, explicit TIP3P + 0.15 M NaCl (or GBn2 for speed), 3 replicas,
   weak restraints on receptor backbone. ~minutes/complex on the 5070.
3. **Score:** our absolute ML scorer `S` on each relaxed endpoint (mean over replicas).
4. **Auxiliary:** single-shot APBS `E_elec` for the F1 Δ-correction; record pose-overlap metrics for F2.

### 4.4 Estimators compared

| Arm | Formula |
|---|---|
| **Absolute (baseline)** | `S(p,R)` (also report PPI-Affinity absolute where runnable) |
| **Anchored k=1** | nearest same-charge reference |
| **Anchored k=3** | mean of 3 nearest |
| **Triangulated k=all** | `S(p) − b̂(R)` over all same-receptor refs |
| **Bayesian-weighted** | similarity × inverse-assay-variance weighted b̂ |
| **+ PB Δ-correction** | best anchor arm + `λ·ΔE_elec` |
| **Oracle offset (ceiling)** | subtract true per-receptor mean residual (E254-style) |

### 4.5 Controls (non-negotiable)

- **Shuffle control:** assign each test peptide its references from a **different random receptor** (wrong
  b(R)). Genuine cancellation ⇒ anchored performance **collapses to ≤ baseline**. If it stays high, the
  "gain" was generic regularization, not offset removal. **This is the make-or-break test.**
- **Permuted-Kd control:** shuffle reference Kd labels within the correct receptor — should also collapse.
- **Positive control:** oracle offset = the ceiling (expect ≈0.75 charged).

### 4.6 Metrics

Pearson r, Spearman ρ, RMSE (kcal/mol) — pooled **and stratified by net charge** (the headline is the
charged subset) and by ref/test charge-match. Plus **within-receptor enrichment**: can we rank the
tightest binder per receptor (top-1/top-3 accuracy, per-receptor AUC) — the selectivity use case.
Uncertainty via **cluster bootstrap over receptors** (95% CI). Report performance vs **#references** and
vs **ref–test distance** (the two knobs the theory predicts matter).

### 4.7 Decision rule (pre-registered)

**Success** = charged-subset anchored r ≥ 0.45 and RMSE cut ≥ 30% vs absolute baseline, **with the
shuffle control collapsing to baseline ± noise**. Anything less than a collapsing shuffle control ⇒ not
genuine cancellation, do not ship.

---

## 5. Literature anchors (for the writeup / to convince a referee this is principled)

- **RBFE / FEP+** — Wang et al., *JACS* 2015; Cournia et al., *JCIM* 2017 (review). Provides §1.2 cycle.
- **DiffNet optimal estimator** — Xu, *JCTC* 2019, "Optimal measurement network of pairwise differences":
  the rigorous MLE that reconciles relative edges + absolute anchors. **This is §1.4 triangulation done
  exactly.** Cite as the principled generalization.
- **FEP reference networks / maps with experimental anchors** — Open Free Energy / Cinnabar tooling;
  perturbation graphs anchored by ≥1 measured node.
- **Δ-machine-learning** — Ramakrishnan et al., *JCTC* 2015. Our PB Δ-correction layered on the anchor is
  delta-learning where the baseline is the reference-anchored estimate.
- **ML-potential / NNP-MM relative FEP** — Rufa et al. 2020; Sabanés Zariquiey et al. 2024 — sharpening
  the relative term with learned potentials (our upgrade path from 100 ps endpoint scoring).
- **Linear Interaction Energy (LIE)** — Åqvist et al. — per-system *calibrated* scoring; the classic
  precedent for anchoring absolute energetics to experimental references rather than computing them cold.
- **Per-target random-effects in QSAR** — hierarchical intercept models; the statistical form of §3's
  Bayesian b(R).

---

## 6. If n=100 works — implications

A collapsing shuffle control + charged r jumping from ~0 to ~0.5–0.75 would mean:

1. **We do not need a globally accurate absolute ML scorer.** We need a decent *relative* scorer (physics
   already gives this within-receptor) + a few experimental anchors. The hard, FEP-bound part (b(R)) is
   never predicted — it's measured once per receptor and amortized over every screened peptide.
2. **Low-cost & general for the realistic regime:** 2–3 reference Kds (mined or one cheap assay plate) +
   100 ps MD + our scorer ⇒ calibrated absolute Kd for *any* new peptide on that receptor — including
   receptors no global model (PPI-Affinity included) has ever seen.
3. **It IS our deployment frame.** Pick a receptor, measure a couple of known binders (already how the
   parent project chooses receptors), then screen variants. The unsolvable cross-target absolute-Kd
   problem becomes the solvable within-receptor problem + calibration. This is the legitimate,
   defensible "best non-FEP peptide scorer" story: not "we beat FEP at absolute energies," but "we get
   FEP-grade *relative* ranking at docking cost, anchored to cheap experiment."
4. **Honest scope:** few-shot, not zero-shot. Needs ≥1 anchor; degrades to absolute S for orphan
   receptors; capped by within-receptor `η`. State this — it's still novel and fundable.

---

## 7. Immediate next step (Phase 0, already staged)

Run `scripts/e260_anchor_triangulation.py` (charged SKEMPI, n=1122, score-env). It tests the
**cancellation engine** — does within-receptor anchoring recover charged signal that cross-receptor
prediction loses — using static features as a floor for the relative term. If even the static-feature
anchor beats cross-receptor (toward the E254 ≈0.75 ceiling) and the shuffle control collapses, the
physics is confirmed and the MD/ATLAS absolute-Kd benchmark (§4) is justified. If it doesn't, the idea
dies cheaply before any MD spend.
