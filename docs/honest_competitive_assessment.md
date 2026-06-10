# Honest Competitive Assessment — where we actually stand on peptide ΔG

**Date:** 2026-06-10. Written in "harshest critic" mode: no claiming wins the field
already banked. Backed by our own measurements + literature.

## 1. Brutal self-audit: our PRODUCTION absolute ΔG is worse than guessing

Measured on the clean 56-Kd holdout (`data/eval_kd_ki_clean.json`):

| our production model | Pearson r | RMSE (kcal/mol) | vs mean-baseline 2.47 |
|---|---|---|---|
| pred_entropy (v1.2, SHIPPED) | **−0.26** | 3.88 | WORSE + backwards sign |
| pred_ridge (v1.1) | −0.45 | 12.47 | catastrophic |
| pred_legacy (v1) | +0.07 | 63.4 | absurd units |

Our shipped v1.2 was calibrated to r=0.86 on 6 complexes (PepSet-6); it generalizes
to **negative correlation and RMSE worse than predicting the mean for everyone.**
That is the size/baseline confound living in our own code. **Action: stop reporting
absolute ΔG from the production scorer.** It is, at best, decorative; at worst
actively misleading (wrong sign).

## 2. Who succumbs to the size/baseline confound, and who resists

**Succumb (size- or baseline-confounded for cross-target ABSOLUTE affinity):**
- **AutoDock Vina** — documented systematic size bias (Vina/NNScore scores correlate
  with ligand size regardless of target; correction terms exist).
- **MM-GBSA/PBSA single-trajectory absolute** — ΔH scales with interface size; the
  canceling entropy term is under-captured (our e9 data: ⟨E_int⟩ backwards).
- **Knowledge-based potentials (DFIRE/ITScore)**, **PRODIGY** outside its
  protein–protein size regime — extensive contact sums.
- **Our v1.2 production** (§1).

**Resist — and HOW they do it (this is the lesson):**
- **FEP / TI / LIE** — compute ΔΔG *within one target*; the per-protein baseline
  cancels by construction. They don't beat the confound, they sidestep it by problem
  definition. (LIE's γ intercept = per-system baseline.)
- **NetMHCpan / MHC tools** — the field that genuinely "solved" it at scale, via
  **per-allele percentile-rank normalization**: each MHC's score is normalized
  against that MHC's own score distribution → size/system-independent. This is
  exactly per-protein-baseline removal, done in production for 20+ years.
- **ddG mutation predictors (FoldX, Flex-ddG, ThermoMPNN)** — predict *changes*
  within a complex → baseline cancels.
- **AlphaFold/Boltz-2 affinity** — use co-evolution/MSA, a NON-size information
  source; generalize cross-target (r=0.61 at <30% identity) precisely because the
  signal isn't interface size.
- **Glide** — reported to NOT overpredict affinity by size (unlike Vina).

**The throughline:** every method that works on peptide/protein affinity does so by
(a) staying RELATIVE / within-target (FEP, LIE, ddG), (b) normalizing per-system
(NetMHCpan percentile rank), or (c) using a non-size information source (co-evolution).
NOBODY does cheap, blind, ABSOLUTE, cross-target structure-based peptide ΔG well —
because it is the confound, not a solvable target.

## 3. So is our work novel? Honestly: NO as a discovery, YES as rigor.

- The size/baseline (Simpson) confound is **known and named** ("scoring bias", ligand
  efficiency size-bias, Kenny 2019) and **solved in practice** (NetMHCpan percentile
  rank; FEP/LIE relative). We did **not** discover or first-solve it.
- What we *did*: (a) proved it is the specific cause of cross-dataset non-replication
  in general protein–peptide docking, with a clean within-vs-between decomposition;
  (b) identified instant geometric features that are sign-stable across two
  independent datasets (interface H-bonds + aromatic contacts); (c) cross-dataset
  validated a within-target ΔΔG formula at r≈0.45. That is a **modest, honest,
  methods-level contribution**, not a breakthrough.

## 4. Where our number sits among within-target methods (the fair comparison)

| method | within-target r | cost | notes |
|---|---|---|---|
| FEP / ABFE | 0.7–0.9 | hours–days/peptide | gold standard, relative |
| LIE | ~0.79 | minutes (MD) | per-system α/β fit |
| Rosetta FlexPepDock | ~0.59 | minutes | reweighted REU, series |
| FoldX ddG | ~0.5 | seconds | within-complex mutations |
| **ours: H-bond + aromatic ΔΔG** | **~0.45** | **instant (geometry)** | cross-dataset validated |
| NetMHCpan (MHC only) | very high | instant | needs 10^4–10^5 labels |

We are at the **low end of within-target accuracy, but the cheapest tier** (instant,
no MD/Rosetta/training). Honest pitch: a fast first-pass within-target ranker; for
accuracy, cascade into MD-LIE on top candidates (the cascade design from earlier).
We should NOT claim to beat FEP/LIE/Rosetta — we don't.

## 5. Complex.zip — triaged, not worth mining

18,286 structures; only 285 carry any affinity label, 84 are Kd/Ki, and only **11 are
genuinely new** vs our existing sets. The dataset ceiling is LABELS, not structures.
Mining 18k cif files for 11 complexes is not worth it. Real expansion requires
literature ITC/SPR curation (the PEPBI-style effort) — a data project, not a parse.

## 6. What to actually do (recommendations)

1. **Deprecate production absolute ΔG.** It's backwards on held-out data. Report
   pose ranking + within-target ΔΔG + selectivity only; absolute only vs a reference.
2. **Adopt NetMHCpan's trick for single-target use:** percentile-rank candidate
   peptides within the target's own docked-score distribution. Size/baseline-free,
   proven, trivial to implement. Ideal for the PfLDH peptide-design use case.
3. **Use the universal ΔΔG (H-bond + aromatic, r≈0.45)** as the instant within-target
   ranker; cascade to MD-LIE (GPU, ~25 s/pose) for top-K accuracy.
4. **Frame for iGEM honestly:** "we diagnosed and correctly navigated the size/baseline
   confound that breaks naive peptide scoring; we deliver rigorous within-target
   relative ranking + selectivity, not an over-claimed absolute kcal/mol." That is
   defensible and true; a headline absolute-ΔG claim is not.

---

## 7. HARSHEST-CRITIC CORRECTION (e16) — the within-target r≈0.45 is a mirage

Before building a per-target ranker, leave-one-binding-group-out CV on PEPBI +
per-group breakdown:

| hb_count+aromatic (leave-group-out) | value |
|---|---|
| pooled held-out within-group r | +0.437 |
| **median per-group Spearman (n>=4 groups)** | **+0.05** |
| **fraction of groups ranked correct direction** | **50% (coin flip)** |
| per-group spread p10→p90 | −0.68 → +0.85 |

Per target: TtSlyD (n127) +0.55, SH3 (n35) +0.41, α-adaptin (n27) +0.40 work; PTPA
−0.26, SGT2 −1.00, SOCS −0.87 are backwards. **The pooled r=0.44 is n-weighted and
dominated by 2–3 large binding groups (TtSlyD alone = 40% of pairs).** Per arbitrary
target, the universal geometric ΔΔG ranker is ~chance on direction.

**Conclusion:** the universal within-target ΔΔG signal is NOT a reliable zero-shot
per-target tool. It works where binding is H-bond/aromatic-driven, fails otherwise,
and we cannot predict which a priori. (Partly confounded: failing groups are the
small ones, n=9–12, where narrow ΔG range + ITC noise inflate the coin-flip — so it
is partly measurement noise, not pure model failure. But per-target reliability is
UNPROVEN.) This matches the prior "oracle τ=0.97 but right-feature-is-target-specific"
finding: universal slopes cannot know which features matter for a given protein.

**Revised recommendation:** do NOT ship a universal per-target ΔΔG ranker with an
r≈0.45 claim. Honest deliverables remain: (1) pose ranking (τ≈0.18, validated),
(2) ΔΔG SELECTIVITY (target vs off-target — the size/baseline confound cancels in the
difference, and you compare the SAME peptide so per-target-feature-relevance is fixed),
(3) MD-LIE cascade for top-K accuracy. Per-target affinity ranking from instant
geometry is not reliable enough to claim. The science (mechanism, sign-stable features,
why it's walled) stands and is the honest contribution.
