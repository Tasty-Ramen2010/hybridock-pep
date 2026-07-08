# Are we at LIE/FEP kcal/mol, are we on the right track, and where do we converge?

**Date:** 2026-07-08 · 15+ literature searches + our data. Answering Ram: is our MAE/RMSE at LIE/FEP level, does
PRISM-S beat LIE, are we on the right track, what are we still missing (force field + method)?

## 1. The honest accuracy landscape (the numbers that matter)

| regime | method class | Pearson r | RMSE/MAE (kcal) | note |
|---|---|---|---|---|
| **same-target** (congeneric) | LIE, FEP+, ML | **0.80–0.88** | ~1.0–1.3 | the favorable regime; often leakage-inflated |
| LIE same-target | Åqvist LIE (fitted α,β,γ) | ~0.83 | 1.25–1.82 | HIV-1 protease 1.25/0.83; but γ **not transferable** cross-target |
| FEP+ protein-protein | alchemical | ~0.63 (R²0.4) | ~1.0 | even gold-standard ~0.6 r on absolute |
| **cross-target** (diverse) | best ML / QM-MM | **0.6–0.7** | 0.6–1.4 | diverse-protein ML 0.651; QM/MM multi-conformer 0.81 (curated) |
| **OURS** (cross-target peptides) | single-pose ML | **0.36–0.38** | MAE **1.37** | leakage-free |

## 2. Is our MAE at LIE/FEP level? Numerically yes — but it's a mirage
Our **MAE 1.37 kcal** sits right in the LIE (1.3–1.8) / FEP (~1.0) band. **But we beat "predict the mean" by only
0.11 kcal** (mean-MAE 1.48). The MAE looks good only because peptide Kd has a narrow range (std 1.85). LIE/FEP hit
similar MAE **with r ≈ 0.8** because they work same-target where signal is strong; we're cross-target where it's
weak. **MAE is the misleading metric; r is honest — and our r (0.38) is well below LIE's.** The literature says this
outright: *"low RMSE alone doesn't guarantee good correlation."*

## 3. CORRECTION to our earlier pessimism: the wall is NOT 0.35
We previously called ~0.35 the universal ceiling. **That was too pessimistic — it is OUR single-pose ceiling, not
the field's cross-target limit.** Cross-target SOTA reaches **r ≈ 0.6–0.7** (diverse-protein ML 0.651; FEP+ 0.63).
**The gap 0.38 → ~0.65 is real and closable** — but only with the methods those results use. We were measuring our
own limitation and mistaking it for physics.

## 4. What the SOTA does that we don't — the universal lever: MD ENSEMBLE, not single pose
Every cross-target method that reaches 0.6–0.7 shares one thing: **ensemble-averaged energies from MD, not a single
static structure.** The literature is explicit: *"binding affinity is fundamentally an ensemble property, not a
single-snapshot property… an MD trajectory provides more information than a single static structure."* **Our scorer
uses ONE pose — that is the universal cap at ~0.38.** This is not a missing energy *term*; it is the single-structure
→ ensemble *paradigm*. LIE is the cheapest embodiment of it (a few GPU-hours of MD → ⟨V_elec⟩,⟨V_vdw⟩).

## 5. Are we on the right track? YES — and the literature validates PRISM-S directly
- LIE alone **"misses entropic interactions"** (a documented LIE limitation).
- *"Changes in configurational entropy strongly oppose binding and MUST be included for accurate affinities."*
- **"LIE extended with conformational entropy substantially increased the correlation with experiment."**
This is *exactly* our plan: **LIE (ensemble enthalpy) + PRISM-S (conformational entropy)** — E359 runs precisely this
head-to-head at 500ps. We independently arrived at the literature's prescription for going beyond LIE.

## 6. Where our FORCE FIELD is still not perfect (ff14SB gaps)
- **Polarization** — ff14SB is fixed-charge with *inconsistent* polarization (pre-polarized charges + gas-phase
  dihedral fitting). It **over-stabilizes buried salt bridges / shifts buried His/Glu pKa** — exactly the charged-FEP
  errors we hit. **AMOEBA (polarizable)** reaches **MUE 0.85 kcal** and wins SAMPL7/8 — polarization is worth
  ~0.5–1 kcal on charged/buried cases. This is the force-field-completeness gap for the **charged** term (→ --ultra
  redemption territory).
- **Entropy** is not a force-field term at all — it's a sampling/estimator problem (PRISM-S).
- **Dihedral parameters** are empirical catch-alls, absorbing missing polarization — a known ff14SB compromise.

## 7. The convergent picture — what we actually need
Ranked by expected payoff toward the real cross-target ceiling (~0.6):
1. **Move from single-pose to MD-ENSEMBLE scoring** (LIE-style ⟨V_elec⟩,⟨V_vdw⟩ + ensemble features). This is THE
   universal lever — it's what separates our 0.38 from SOTA 0.6–0.7. E359 is the first test.
2. **Add conformational entropy** (PRISM-S) on top of the ensemble enthalpy — literature says "substantial" gain.
3. **Polarization for the charged term** (AMOEBA-class) — closes the charged/buried force-field gap; --ultra tier.
4. **More/cleaner data** (Kd-only, larger) — raises the ML ceiling.
5. **Per-family calibration** where a tool allows it — same-target regime is where r 0.8 lives.

## The bottom line for Ram
- **Not at LIE's r yet** (0.38 vs 0.8 same-target / 0.6–0.7 cross-target). MAE *looks* like LIE but it's a
  narrow-range mirage.
- **We were too pessimistic** — the real cross-target ceiling is ~0.6, not 0.35, and it's reachable.
- **We are on the right track**: LIE + conformational entropy is the literature-endorsed route past LIE, and that is
  exactly E359 / PRISM-S.
- **We must go ensemble.** The single-pose scorer is the cap. That is the universal thing we were missing — not one
  term, but the paradigm.

---
### Sources
LIE accuracy/transferability: [Falcipain LIE](https://www.sciencedirect.com/science/article/abs/pii/S1093326316301024), [LIE review PMC7311763](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7311763/), [LIE+entropy MC](https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2020.00114/full).
Cross-target ceiling: [ensembling SciRep](https://www.nature.com/articles/s41598-024-72784-3), [diverse-protein ML](https://arxiv.org/html/2410.00709v2), [FEP+ PP PMC11339910](https://pmc.ncbi.nlm.nih.gov/articles/PMC11339910/), [QM/MM mining-minima](https://www.nature.com/articles/s42004-024-01328-7).
Ensemble>single: [static→dynamic PMC11516055](https://pmc.ncbi.nlm.nih.gov/articles/PMC11516055/), [multiple binding modes PMC2877349](https://pmc.ncbi.nlm.nih.gov/articles/PMC2877349/).
ff14SB/polarization: [ff14SB PMC4821407](https://pmc.ncbi.nlm.nih.gov/articles/PMC4821407/), [ff14SB pKa limits PMC11398383](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11398383/), [AMOEBA host-guest PMC10878370](https://pmc.ncbi.nlm.nih.gov/articles/PMC10878370/).
