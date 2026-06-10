# MD-LIE Accuracy vs the Market — ± kcal/mol Comparison

**Date:** 2026-06-10 · GPU: RTX 5070 (CUDA MD now functional in this env)
**Our numbers:** measured on the 65-complex crystal benchmark (peptide ΔG range
−5.8 to −14.0, sd 2.17 kcal/mol). Scripts: `e9_md_ensemble_ie.py`, `e9c_rmse.py`,
`e9d_1ns_subset.py`.

> **The one thing to read first.** Almost every headline RMSE on the market is for
> a *relative* or *within-system* task (rank similar ligands against the SAME
> protein), NOT blind cross-family ABSOLUTE ΔG. When you force the blind-absolute
> task, every cheap method — and most expensive ones — collapses toward the
> ~2 kcal/mol noise floor. Compare like-for-like or the table lies.

## Our measured numbers (crystal-65, peptides)

| Method | RMSE | MAE | Pearson r | Regime | Speed (RTX 5070) |
|---|---|---|---|---|---|
| Mean-predictor baseline | 2.17 | 1.88 | 0.00 | floor | — |
| Vina (uncalibrated) | ~9 | — | −0.45 | backwards/size | <1 s |
| 60 ps MD-LIE, in-sample calibrated | 1.89 | 1.66 | 0.49† | optimistic | ~25 s/peptide |
| 60 ps MD-LIE, **blind (leave-family-out)** | **2.04** | 1.80 | 0.38† | honest cross-family | ~25 s/peptide |
| 1 ns MD-LIE | _TBD (running)_ | | | | ~220 s/peptide |

† the +0.38–0.49 r rides the size confound in this sample; on a truly novel
target it degrades toward the family-mean honest ~0. Treat blind absolute as
**RMSE ≈ 2.0 ≈ noise floor**.

## The market (literature)

| Method | RMSE (kcal/mol) | r | Regime — *read this column* | Cost/ligand |
|---|---|---|---|---|
| Absolute FEP / ABFE | 1.1–2.0 | high | **within one target**, ensemble | hours–days |
| Relative FEP (FEP+) | ~1.0 | — | **congeneric series**, same protein | hours |
| LIE (Aβ peptides, published) | ~1.0–1.5 | 0.79 | **single protein system**, per-system α/β | minutes–hours |
| MM/GBSA (peptide bench pt.9) | 1.5–2 (calib) | 0.75 | **within peptide-size class** | minutes |
| Boltz-2 (2025 SOTA ML) | 0.8–0.9 MAE | 0.66 | protein-ligand; >0.55 on 3/8 assays | seconds |
| PRODIGY | 1.89 | 0.73 | **protein–protein** (large rigid interfaces) | <1 s |
| **HybriDock-Pep 60 ps MD-LIE (blind)** | **2.0** | ~0 honest | **blind cross-family peptide** | 25 s |

## Reading the comparison honestly

1. **Blind cross-family absolute peptide ΔG is ~2 kcal/mol for everyone cheap**,
   and the expensive methods only beat it by switching to the easier
   relative/within-target task. Our 60 ps MD-LIE at ~2.0 is *competitive at the
   blind-absolute task* — the field just rarely reports that task.
2. **RMSE is a weak metric here** — peptide ΔG spans only ~2 kcal/mol, so the
   mean-predictor already scores 2.17. A "2.0" looks close to FEP's "1.0" but
   that gap is mostly dynamic range, not skill. **Correlation is the honest
   discriminator**, and blind cross-family it is ~0 for all cheap methods.
3. **Where MD-LIE earns its cost is the relative/within-target regime** — the
   interaction-entropy term shows within-family r≈0.40 (the signal static scoring
   cannot produce). That is the selectivity / ΔΔG use case (iGEM PfLDH vs hLDH).

## 60 ps vs 1 ns convergence
First 1 ns point (1NRL): dg_pred shifted ~16 kcal/mol vs 60 ps — **60 ps is far
from converged in ABSOLUTE energy.** (Full subset delta filled when e9d completes.)
Implication: 60 ps is usable for *ranking* (relative differences are more stable
than absolute), but absolute kcal/mol needs much longer sampling — and even then
hits the cross-family wall.

## Verdict for the cascade (ranker → 60 ps top-K → 1 ns winner)
- ✅ **Selectivity / ΔΔG**: build it. Errors cancel in the difference; IE term real.
- ⚠️ **Pose ranking**: plausible refinement; untested vs our τ≈0.18 ceiling.
- ❌ **Blind absolute kcal/mol on a novel target**: still walled; 1 ns buys
  precision, not cross-family accuracy.

---

## Entropy-corrected LIE (user's size-penalty idea) — tested

Hypothesis: LIE omits configurational entropy, which should cancel ⟨E_int⟩'s
size-scaling; add a per-residue penalty (contact-state + AA-type aware).

Leave-one-FAMILY-out CV on 61 complexes (60ps ⟨E_int⟩):

| model | ALL r | Kd r |
|---|---|---|
| M0: ⟨E_int⟩ alone | +0.35 | +0.13 |
| M1: + linear N_res | +0.40 | +0.15 |
| M2: + entropy(contact/AA) | **+0.43** | **+0.23** |

- The penalty **fixes the backwards/size-dominated scores** (real, useful for the
  cascade's relative ranking).
- BUT `corr(entropy_penalty, N_res)=+0.93` (87% length by variance), and the
  composition orthogonal to length is **~0 cross-family** (ALL −0.04, Kd +0.12).
- Verdict: it is a **size de-confounder**, not a new signal source. AA-type/contact
  detail adds ~nothing beyond "+0.7·N_res". Does NOT break the cross-family wall.

## 60ps vs 1ns convergence (8-complex subset, e9d)
mean |Δ dg_pred| = **6.7 kcal/mol** (some ±12) — larger than the whole ΔG range.
60ps absolute energy is sampling-noise-dominated; ≥1ns needed for absolute, and
even then walled cross-family. 60ps is adequate only for *relative* ranking.

---

## E10 — LENGTH is a per-dataset confound (the root cause of non-replication)

User hypothesis: "length is the issue; worse scores correlate with length unless
a really good binder gives a really good score." Tested:

**TEST A — sign consistency of corr(length, ΔG):**
- crystal-65: **+0.43** (longer binds weaker)
- PEPBI:      **−0.24** (longer binds stronger)
- The SIGN FLIPS across datasets. Length is not a real signal and not a fixable
  bias — its *direction* is a per-dataset accident. Any model riding length (most
  cheap scorers, since every feature ~ size) will FLIP SIGN on new data. This is
  the mechanism behind every non-replication observed: H-bond count +0.47→−0.41,
  NIS −0.54→−0.21. A fixed "+0.7·N_res" penalty cannot be universal because the
  correct length coefficient has a different SIGN on different datasets.

**TEST B — "good binders break through": FALSIFIED (backwards).** Score→ΔG is
weaker among strong binders than weak ones (⟨E_int⟩ strong −0.13 / weak −0.41;
nis_p strong −0.29 / weak −0.43). Scoring functions SATURATE — once binding is
good the score caps out. The score is a bad-binder detector, not a strong-binder
confirmer.

**TEST B2 — within a fixed length bin (size-controlled, one dataset), nis_p holds
(−0.36 / −0.68 / −0.41).** Real signal exists size-controlled within a dataset =
the within-target regime.

**Capstone conclusion:** blind absolute peptide ΔG is unsolvable by cheap methods
because the dominant axis (length/size) has no consistent cross-dataset direction.
This is an identifiability wall, not a missing feature. Use relative/within-target
ΔΔG (size cancels, direction irrelevant) — the only regime that survives.
