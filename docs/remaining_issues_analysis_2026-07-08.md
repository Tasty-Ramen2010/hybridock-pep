# Beyond entropy: what else are we failing to account for? — systematic residual decomposition

**Date:** 2026-07-08 · Method: correlate the leakage-free scorer **residual** (y − prediction, the unexplained
variance, std 1.73 kcal at r=0.375) against descriptors for **every** candidate missing term, on 925 peptide-Kd
complexes. A term with residual shape = a real missing physics/data lever. Whichever are ~0 = not our problem.

## The result: NO single missing term is large

```
corr(residual, descriptor)         (|r| ranked)
  Pro_frac (proline content)     +0.096   ← largest, and it's an ENTROPY effect (below)
  n_basic (K+R)                  -0.064
  Ser+Thr fraction               +0.064
  net_charge                     -0.062
  n_Trp                          -0.060
  Gly_frac                       -0.046
  |net_charge|                   -0.032
  n_acidic (D+E)                 +0.032
  n_His (protonation ambiguity)  -0.025
  cooperativity/size (poc_n)     -0.024
  terminal-charge (1/length)     +0.019
  length, aromatic, aliphatic    < 0.02
  ─────────────────────────────────────
  confinement TΔS (E354, n=15)   -0.145   ← still the single strongest lever
```

**Every cheap sequence/structure descriptor for every candidate term correlates with the residual at |r|<0.1.**
Protonation (His), terminal charges, cooperativity, aromatic/cation-π, composition — all small. This is the
**error-budget theorem made empirical** (wall-analysis Front 2): the unexplained variance is **not** dominated by
any one missing term. It is spread thin **plus** substantial irreducible noise.

## The two real levers that survive

### Lever 1 — configurational entropy / dynamics (the only real *physics* lever)
- Confinement TΔS has the strongest residual shape of anything tested (−0.145), right sign, and is the first term
  to reduce absolute MAE (E355: 2.16→1.87 at n=15).
- **Proline is a concrete fingerprint of it.** Pro_frac is the top *cheap* descriptor (+0.096), and the literature
  says why: proline **cis/trans isomerization** changes binding by up to **~3 kcal** from a single residue — p53
  (17–29)→MDM2 is **−11.8 (trans) vs −8.9 (cis)**, *"primarily due to the loss of conformational entropy on
  binding"* ([PMC5444545](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444545/)); cis/trans conformational selection
  even controls 14-3-3 binding over 3 orders of magnitude ([JACS 4c13462](https://pubs.acs.org/doi/10.1021/jacs.4c13462)).
  A single crystal pose **cannot** know the cis/trans state or the free-state ensemble → this is entropy/dynamics,
  invisible to single-pose scoring. **This confirms entropy is the #1 lever and points the build (PRISM-S).**

### Lever 2 — label heterogeneity (irreducible, but *fixable*)
- The set is **863 Kd + 62 Ki** — two different thermodynamic quantities. The Ki subset is biased and noisier:
  mean residual **+0.13 (Ki) vs −0.02 (Kd)**, and **std 2.32 (Ki) vs 1.68 (Kd)**.
- Literature: *"Combining IC50 or Ki values from different sources is a source of significant noise"*; Ki is a
  different observable from Kd ([JCIM 4c00049](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049)). The fix is not
  physics — it's **data hygiene**: train/report on **Kd-only**, or apply the known offset correction and downweight.
- This raises the *achievable ceiling* without any new physics — the cheapest win on the board.

## Everything else — small or spread (honest catalog)
| candidate term | residual shape | verdict |
|---|---|---|
| charge / electrostatics | ~0 (E353b) | already captured by scorer's crude charge feats; not the wall |
| protonation / pKa (His) | −0.025 | minor; buried-pKa cases too rare in this set to dominate |
| terminal charges | +0.019 | negligible at these lengths |
| cooperativity / standard-state | −0.024 | minor (mostly 1:1 peptide complexes) |
| aromatic / cation-π | <0.02 | scorer already has poc_f_arom / arom_cc |
| desolvation / hydration | untested here | **RISM not cached for these pdbs (0 overlap)** — needs fresh compute to rule in/out |
| receptor flexibility / induced fit | not sequence-proxyable | the one structural lever we could NOT cheaply test — candidate for the spread |

## Synthesis: why we "fail to account for changes"
1. **The dominant recoverable lever is dynamics/entropy** — confinement TΔS + proline cis/trans — and it is *hard*
   precisely because it lives in the ensemble, not the pose (wall Front 3/4).
2. **A real chunk is irreducible label noise** (Ki/Kd mixing, cross-lab) — fixable by data hygiene, not physics.
3. **The rest is genuinely spread** across many ~0.05-level terms (Front 2) — no single build recovers it; this is
   the information-theoretic floor of a *general, single-structure* model (Front 4).

**We are not missing one big term. We are missing (a) the ensemble/entropy — worth building — and (b) clean labels
— worth cleaning — and everything else is noise-floor.**

## Brainstorm — ranked next actions
1. **Finish the entropy gate (n=30) and, if it holds, build PRISM-S** — the only physics lever with signal. Add a
   **proline cis/trans** term as a cheap high-value special case (sample both isomers, Boltzmann-weight).
2. **Clean the labels** — Kd-only benchmark + Ki-offset-corrected variant; report r/MAE/RMSE on the clean set.
   Expect a modest ceiling lift for free.
3. **Rule desolvation in or out honestly** — run RISM/GIST hydration on a ~30-complex subset (we have the E349
   machinery) and correlate with the residual, like we did for charge and entropy. It's the one untested physics
   term.
4. **Test the receptor-flexibility/induced-fit lever** — the only structural term not proxyable from sequence;
   B-factor / pocket-plasticity descriptors vs residual.
5. **Accept the floor for what it is** — after 1–4, whatever residual remains is the general-model information
   ceiling, and the honest product move is selectivity/ΔΔG (where it cancels), per the wall analysis.

---
### Sources
Ki/Kd/IC50 label noise: [JCIM 4c00049](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049), [OPIG affinity-dataset reliability](https://www.blopig.com/blog/2025/08/how-reliable-are-affinity-datasets-in-practice/).
Proline cis/trans entropy: [p53-MDM2 PMC5444545](https://pmc.ncbi.nlm.nih.gov/articles/PMC5444545/), [14-3-3 JACS 4c13462](https://pubs.acs.org/doi/10.1021/jacs.4c13462), [IDP proline FBL](https://www.imrpress.com/journal/FBL/28/6/10.31083/j.fbl2806127).
(Entropy/ensemble & info-limit fronts: see docs/absolute_kd_wall_analysis_2026-07-07.md.)
