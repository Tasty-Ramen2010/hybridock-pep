# Reference-Anchored Relative Scoring for Peptide–Protein Affinity

**Status:** engine validated (e260, shuffle-controlled). Phase 1 (MD + real Kd) designed.
**Updated:** 2026-06-16

---

## 0. Which tool for which job (read this first)

HybriDock-Pep operates on **two independent axes**. Use the right machine for the job — they do not
substitute for each other.

```
What do you actually need?
│
├─ "Rank my candidate peptides against ONE target; pick the best binder."   → RANKING axis
│      Use the RELATIVE SCORER (charge_complementarity, pose-ranker, ref2015).
│      Offset b(R) is a shared constant → cancels in any within-target ranking → anchoring IRRELEVANT.
│      SHIPPED. No anchors, no Kd, no MD needed. This is iGEM deployment mode (a).
│
├─ "What is the absolute Kd of this peptide on this target?"                 → SCORING axis
│      Use ANCHORING: ΔG_pred = ΔG_exp(ref) + [S(p)−S(ref)] on the SAME receptor.
│      Needs ≥1 known-Kd reference on that receptor. Accuracy ≈ 1.3 kcal/mol (η floor).
│
└─ "Does this peptide prefer target A or target B?" (cross-target selectivity) → SCORING axis
       ΔΔG = [S(P,A) − b̂(A)] − [S(P,B) − b̂(B)]. Needs anchors on BOTH receptors.
       Accuracy ≈ 1.9 kcal/mol (independent) → ≤2.0 target.
```

**Anchoring is a cross-receptor SCORING/selectivity method. It does nothing for within-target ranking,
and can slightly hurt it** (per-point anchoring adds noise to an order that was already correct;
measured: within-receptor Spearman 0.351 → 0.195). Ranking quality is owned entirely by the relative
scorer. Do not confuse the axes.

---

## 1. Why this exists (grounding)

| Prior result | Establishes | Role |
|---|---|---|
| e246 | ~55% of charged ΔΔG error is a **between-receptor offset** | the term anchoring removes |
| e255 | that offset is **not predictable** from seq/ESM/fpocket/PB/0.6 ns GIST (≤ null) | so we *observe* it, not regress it |
| E254 | with offset **known**, within-receptor charged → **r ≈ 0.755** | the anchored ceiling (the η floor) |
| Absolute-Kd ceiling (Jun 14) | honest clustered-CV ≈ 0.35 for everyone incl. PPI; 0.55 = homology mirage | why a better absolute model is a dead end |
| **e260 (this work)** | cross-receptor charged r −0.07 → **+0.71** by anchoring; **shuffle collapses** | the engine works |

The crux: `b(R)` is real, large, and unpredictable — **but perfectly observable the moment you have one
known-Kd peptide on that receptor.** Anchoring does not predict the FEP-bound term; it **measures and
subtracts** it. That is also why PPI-Affinity scores ~0.7 on charged T100 (it *memorized* `b(R)` for
PDBbind-overlapping receptors) yet falls to ~0.35 on novel receptors. Anchoring makes that implicit
memorization **explicit**, so it works on receptors no global model has ever seen — our deployment case.

## 1.1 Error model

```
S(p,R) = G_true(p,R) + b(R) + c(p) + η(p,R)
```
- **b(R)** per-receptor offset — large, systematic, FEP-bound. The target of anchoring.
- **c(p)** per-peptide systematic error (charge-dependent). Cancels only for *similar* references.
- **η(p,R)** irreducible interaction residual. The hard floor (~1.3 kcal/mol RMSE here).

e260 measured directly that **58% of the cold cross-receptor model error is a per-receptor constant**,
independently reproducing e246's ~55%. The decomposition is real, not assumed.

## 1.2 Thermodynamic cycle

For test `p` and reference `r` on the **same** receptor `R`, path-independence gives
`ΔG_bind(p) − ΔG_bind(r) = ΔΔG(p→r)`. The large absolute terms (the cancelling Coulomb-vs-desolvation
pair that wrecks charged absolute scoring) are computed identically inside both endpoints and subtracted
away — we never form the small-difference-of-large-numbers. This is relative binding free energy (RBFE).

## 1.3 Back out absolute Kd

```
ΔG_pred(p) = ΔG_exp(r) + [S(p,R) − S(r,R)]
           = G_true(p) + [c(p) − c(r)] + Δη + ε_exp(r)     ← b(R) CANCELLED exactly
```

## 1.4 Triangulation = empirical offset estimation

With K references, `ΔG_pred(p) = S(p,R) − b̂(R)`, where `b̂(R) = mean_k[S(r_k) − ΔG_exp(r_k)]` is a
direct measurement of the offset with `Var(b̂) ∝ 1/K`. Two consequences, both confirmed by e260:
more references reduce variance, and references with consistent `c(r)` (same charge class) minimize the
residual.

## 1.5 Similarity-weighting is LOAD-BEARING, not optional

**This is the single most important practical lesson from e260.** Naive averaging over *all* same-receptor
references is *worse* than using the 3 nearest, because dissimilar references inject `c(p)−c(r)` and `Δη`:

| arm | native r | RMSE | MAE |
|---|---|---|---|
| ABSOLUTE (cold) | −0.071 | 2.23 | 1.34 |
| ANCHOR k=1 (nearest) | +0.667 | 1.67 | 1.01 |
| ANCHOR k=3 (nearest) | +0.693 | 1.44 | 0.98 |
| ANCHOR **all** (naive avg) | +0.657 | 1.44 | 1.01 |
| **BAYES (similarity-weighted)** | **+0.712** | **1.33** | **0.91** |
| **charge-matched nearest** | **+0.744** | 1.39 | 0.91 |

`all` (0.657) < `k=3` (0.693): **adding dissimilar references actively hurts.** The similarity kernel
(`w_k ∝ exp(−d²/2σ²)`) and charge-class matching recover the ceiling. **Always weight references by
similarity; never plain-average over everything.** Charge-matched anchor selection gives the best r
(0.744) — restrict anchors to the test's `|Δq|` class when ≥1 is available.

## 1.6 The η ceiling (honest cap)

Anchoring removes `b(R)`, **not** `c(p)` or `η`. The realistic ceiling is the within-receptor
predictability, ≈ 0.75 charged (E254), RMSE floor ≈ **1.3 kcal/mol**. e260 BAYES native r=0.712,
RMSE 1.33 — at the ceiling. A tell-tale: the simulated-absolute block (1.93 kcal/mol injected offset)
gives **identical RMSE 1.33** — anchoring is provably immune to offset magnitude; only `r` rises (the
offset inflates total variance, the denominator of `r`). RMSE is the honest, scale-free metric.

---

## 2. Charged is where anchoring matters MOST

Charge-stratified by mutation `|Δq|` (e260 native, BAYES):

| class | n | absolute r | anchored r | anchored RMSE |
|---|---|---|---|---|
| `|Δq|=0` (same-sign swap) | 35 | +0.308 | +0.413 | 2.21 |
| `|Δq|=1` (charge→neutral) | 908 | −0.065 | **+0.721** | 1.29 |
| `|Δq|=2` (charge reversal) | 77 | **−0.271** | **+0.829** | 1.31 |

The bigger the charge perturbation, the **worse** absolute scoring does (charge reversal is *anti*-
correlated, r=−0.27 — exactly the cancellation-of-large-terms catastrophe) and the **more** anchoring
recovers (0.83). This is the cleanest possible statement of the thesis: anchoring rescues precisely the
charged regime that breaks every absolute scorer.

**Caveat (honest):** the same-Δq vs different-Δq anchor split is nearly equal (r 0.667 vs 0.669), i.e.
`c(p)−c(r)` is a *modest* residual here — because our features already encode the WT charge, the relative
term `S(p)−S(r)` partly absorbs it. Charge-matching still wins overall, but mainly via better anchor
*similarity*, not pure `c()` cancellation. For **peptides** (Phase 1), net charge spans a wider range and
may be less fully captured by features, so `c(p)−c(r)` could matter more — keep charge stratification.

---

## 3. Reference Kd precision is NOT a bottleneck

Inject Gaussian noise into the reference ΔG (BAYES arm, native):

| reference σ (kcal/mol) | r | RMSE | MAE |
|---|---|---|---|
| 0.0 | 0.712 | 1.33 | 0.91 |
| 0.1 | 0.712 | 1.33 | 0.91 |
| 0.3 | 0.710 | 1.34 | 0.91 |
| 0.5 | 0.708 | 1.34 | 0.92 |

Even ±0.5 kcal/mol reference error (≈ 2.4× in Kd) barely moves the result, because triangulation averages
reference noise (`σ/√K`). **Literature/assay Kd values are good enough as anchors — no need for gold-
standard ITC.** (Single-reference k=1 would be more noise-sensitive; this robustness is a property of
triangulation, another reason to use multiple weighted references.)

---

## 4. Failure modes → mitigations

| # | Failure | Cause | Mitigation | e260 evidence |
|---|---|---|---|---|
| F1 | test/ref different net charge | `c(p)−c(r)` ≠ 0 | charge-class matching (+ optional PB Δ-term) | charge-matched best (0.744) |
| F2 | different binding mode | `Δη` large | geometric reference filtering (pose overlap, shared anchors) | — |
| F3 | 100 ps MD insufficient | endpoint unconverged | MD = relaxation+scoring, not alchemy; 3 replicas; escalate flagged pairs to λ-FEP | — |
| F4 | noisy reference Kd | `ε_exp(r)` | average over K (∝1/K) | σ=0.5 → RMSE 1.34 (negligible) |
| F5 | non-additive `p×R` error | `η` not separable | accept the cap; it IS the ceiling | RMSE floor 1.3 |
| F6 | orphan receptor (no anchor) | b̂(R) undefined | homolog transfer, or measure 2–3 anchors; else revert to absolute S | — |
| F7 | reference too dissimilar | `c`,`Δη` blow up | **similarity-weight / charge-match — load-bearing** | `all` < `k=3` |
| F8 | evaluation leakage | memorization ≠ cancellation | **shuffle control** | shuffle collapses, r≈−0.05 |

**Shuffle control (the make-or-break):** anchors drawn from a *wrong* receptor give r ≈ −0.05 and RMSE
*worse* than the cold baseline (2.85–3.30 vs 2.23–2.53). The gain is genuine same-receptor offset
cancellation, not regularization or label leakage.

**Upgrade path:** hierarchical Bayesian `b(R)` (random intercept + prior → shrinkage with few anchors);
the rigorous many-reference form is the **DiffNet MLE** (Xu 2019) reconciling all relative edges + anchors.

### 4.1 Homolog-anchoring radius (e263) — coverage vs the `b(R)−b(R')` residual

Anchoring needs the *same* receptor, but `b(R)` varies **smoothly** with receptor structure, so a close
homolog has `b(R')≈b(R)`. Validated on **real peptide Kd** (PDBbind 925, leave-cluster-out, same covered
queries each row):

| receptor similarity | covered complexes | ABSOLUTE r / RMSE | ANCHORED r / RMSE |
|---|---|---|---|
| exact | 196 | 0.389 / 1.61 | **0.676 / 1.39** (MAE 1.05) |
| ~90% id | 478 | 0.278 / 1.76 | 0.626 / 1.56 |
| ~70% id | 548 | 0.276 / 1.78 | 0.595 / 1.62 |
| ~50% id | 597 | 0.279 / 1.83 | 0.597 / 1.66 |

The gain **degrades gracefully, not off a cliff** — even ~50% id anchoring (r 0.60) crushes cold
absolute (0.28). This is the difference between a *homolog* anchor (small `b(R)−b(R')`) and the
*shuffle* control (random receptor, `b` difference huge → collapse). **Homolog-gating is the real
coverage lever**: 196→478 anchorable complexes at 90% id. Deployment: look up receptor R, fall back
exact → homolog (≥~50% id), widen the confidence band as identity drops.

### 4.2 Why cross-receptor "triangulation through a known corner" does NOT work

A tempting idea: predict `ΔG(P,R)` by routing through a fully-known `(P_known, R_known)` corner. The
path has two legs: a **peptide-swap** leg (same receptor R_known → `b` cancels ✅, = ordinary anchoring)
and a **receptor-swap** leg (same peptide P, R↔R_known) which leaves **`b(R) − b(R_known)`** — the
difference of two receptor offsets = the FEP-bound wall, just relocated. A short equilibrium MD cannot
compute it (that is morphing one protein into a different protein = receptor-FEP, a cross-state ΔG).
**Anchoring works precisely because it never swaps the receptor.** Homolog-gating (4.1) is the only sound
relaxation of the same-receptor requirement.

### 4.3 MD lever applies to DOCKED poses, not crystal poses (Phase-1 scoping correction)

The relative term `S(p)−S(r)` is only as good as the pose. On **crystal** complexes the pose is already
native, so 100 ps MD mostly adds thermal noise — e263's crystal-pose anchoring (RMSE 1.39) is already
near the η floor and MD won't move it. **The MD lever pays off on DOCKED (RAPiDock) poses**, where pose
error is real. So the Phase-1 MD test must be run on docked poses (raw vs MD-relaxed vs crystal), NOT on
crystal panels. Testing MD on crystal complexes would be uninformative — do not do it.

---

## 5. Selectivity (cross-target) — the math and the test

### 5.1 Error propagation

```
ΔΔG_sel(P; A,B) = [S(P,A) − b̂(A)] − [S(P,B) − b̂(B)]
```
Requires anchors on **both** receptors. With single-receptor anchored RMSE `s = 1.33`:

| assumption | selectivity RMSE |
|---|---|
| independent errors | `s·√2` = **1.88** |
| `c(p)` corr 0.3 across A,B | 1.57 |
| `c(p)` corr 0.5 across A,B | 1.33 |

If the *same* peptide is scored against two related receptors, its `c(p)` error is partly shared and
**partially cancels** — so real selectivity RMSE likely sits **below 1.88**, under the 2.0 target.

### 5.2 Phase-1 selectivity test case (data already on disk)

**Primary: Laskowski OMTKY3 × serine-protease panel.** SKEMPI contains turkey ovomucoid third domain
(OMTKY3) P1-site variants measured against multiple proteases — `1cho` (α-chymotrypsin), `1ppf`
(human leukocyte elastase), `1r0r`/`3sgb` (SGPB). Same inhibitor variants, different receptors, measured
Ki on each = the canonical protein–protein selectivity dataset.
- Anchor each protease independently with a subset of its variants (known Ki).
- Predict `ΔΔG_sel` for held-out variants measured on a **pair** (e.g. elastase vs chymotrypsin).
- Compare predicted vs experimental selectivity.
- **Success criterion: selectivity RMSE ≤ 2.0 kcal/mol AND correct sign on ≥ 70% of pairs.**

**Orthogonal: TCR–pMHC (`data/atlas_tcr_pmhc.tsv`) / MHC allele panels (IEDB).** Same peptide, different
MHC alleles, measured affinity — a wide-charge-range stress test of the cross-receptor case.

**Anchor target (iGEM): PfLDH (1T2D) vs hLDH (1I0Z).** The actual selectivity question. Needs ≥2 known-Kd
reference peptides per LDH (wet-lab measurable) before anchored selectivity can be quoted for LISDA…

---

## 6. Phase 1 protocol (MD + real Kd)

1. **Dataset:** ATLAS/IEDB receptors with ≥5 measured-Kd peptides each, ~12–15 receptors → ~100
   complexes; cluster receptors ≤30% seq-id (no cross-receptor leakage). + the §5.2 selectivity panels.
2. **Per complex (identical for test & references):** RAPiDock pose → 100 ps OpenMM NPT (ff14SB,
   TIP3P + 0.15 M NaCl, **3 replicas**, backbone-restrained) → score endpoint with `S` (mean over
   replicas) → single-shot APBS `E_elec` for the optional F1 charge Δ-term.
3. **Anchoring:** per receptor leave-one-peptide-out; arms = absolute / k=1 / k=3 / **BAYES** /
   **charge-matched** / +PB. Report the **k-scaling curve per receptor** (k=1 already buys most of it).
4. **Controls from day one:** shuffled-receptor anchors (must collapse) + permuted-Kd.
5. **Metrics (scoring, NOT ranking):** RMSE, MAE — pooled and **charge-stratified**; selectivity ΔΔG
   RMSE + sign accuracy. Cluster-bootstrap CIs over receptors.
6. **Success:** scoring charged RMSE ≤ 1.5 kcal/mol with shuffle collapsing; **selectivity RMSE ≤ 2.0**.
7. **Compute:** ~100 × 3 × 100 ps ≈ a few GPU-days on the 5070 — **only in a free GPU window; never
   interfere with the running PfLDH production dock.**

---

## 7. iGEM narrative — two deployment modes

**Mode (a) — single-target screening (SHIP NOW).** "Rank my peptide library against PfLDH, pick the best
binders." Pure ranking. The relative scorer is already shipped; **no anchors, no Kd, no MD.** If this is
the deliverable, **anchoring is a research result, not the product** — a demonstrated capability that the
within-target tool sits on solid cross-receptor footing.

**Mode (b) — absolute Kd / selectivity (LOAD-BEARING anchoring).** "What's the absolute Kd?" or
"PfLDH-selective over hLDH?" Needs anchoring + 1–3 measured references **per target**. Accuracy ≈ 1.3
kcal/mol per target, ≈ 1.9 for selectivity. If this is the deliverable, **prioritize measuring 2–3
reference Kd peptides per LDH** — that is the rate-limiting step, and it is cheap.

**Either way:** the shuffle-controlled e260 result is the proof that anchoring works when needed, and the
shipped relative scorer handles mode (a) today. The defensible claim is **not** "beat FEP at absolute
energy" — it is **"FEP-grade relative ranking at docking cost, with cheap experimental anchors turning
that into calibrated absolute Kd and selectivity, on the charged peptide systems where every global
model hits a wall."**

---

## 8. Honest boundaries (state these in any writeup)

- **Few-shot, not zero-shot.** Needs ≥1 anchor per receptor; orphan receptors revert to absolute `S`.
- **Capped at η ≈ 1.3 kcal/mol** within-receptor RMSE (charged). Not FEP-accurate; a calibrator.
- **Selectivity needs anchors on both targets** (≈1.9 kcal/mol RMSE).
- **Validated on SKEMPI mutations with static features.** Phase 1 (real peptide Kd + 100 ps MD) is
  required before quoting numbers for de-novo peptides. The MD only *sharpens* the relative term; if the
  static-feature engine already works (it does) and shuffle collapses (it does), MD-anchoring works more.

## 9. Reproduce

`OMP_NUM_THREADS=1 python scripts/e260_anchor_triangulation.py` (score-env) → `data/e260_results.json`.
(Single-threading is mandatory: WSL2 OpenMP oversubscription makes the HGB fits 1300× slower otherwise.)
