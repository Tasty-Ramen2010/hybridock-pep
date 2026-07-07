# Bridging the salt-bridge gap: why our FEP misses it, how real FEP doesn't, and what we can add (2026-07-07)

Ram's questions: can we add stuff from FEP to fix the salt-bridge under-estimate (+1.5 vs exp +5.9 on 2O3B
D75N)? How does real FEP get these and we can't? What about LIE/TI?

## The diagnosis is now solid — it's MISSING ELECTRONIC POLARIZATION, not sampling or estimator
Three independent facts converge:
1. **The salt bridge does NOT break.** Diagnostic: Asp75(OD)–Lys101(NZ) stays 2.6–3.0 Å across 200 ps unrestrained
   MD. So the small ΔΔG is not from losing the contact.
2. **More sampling doesn't help.** E334 short → +1.07; E335 NPT-equilibrated + 11 windows + 10× sampling → +1.49.
   A converged-but-wrong number ⇒ not a sampling/convergence problem.
3. **The literature names the cause.** Fixed-charge force fields *substantially under-estimate buried ion pairs*
   because they lack **electronic polarization**: for a buried Glu–Lys pair a nonpolarizable model made the
   ionized form ">40 kcal/mol less stable" than a polarizable model (JACS 2022). A buried salt bridge is exactly
   where polarization matters most, and amber14 has none.

So our +1.5 is the *correct answer for the wrong Hamiltonian*: amber14 under-stabilises the buried Asp⁻···Lys⁺
pair, so decharging it costs little, so the computed binding contribution is small.

## Why LIE / TI / FEP-MBAR all inherit this (the estimator is a red herring here)
LIE (β·⟨V_elec⟩), TI (∫⟨∂U/∂λ⟩), and FEP/MBAR are **estimators of the same free energy on the same potential
energy surface**. They differ in variance and bias of the *estimate*, not in the *physics of the force field*. If
the FF under-stabilises the ion pair, every estimator returns the same too-small ΔΔG. Switching TI↔FEP↔LIE cannot
add polarization. (LIE's β *could* be fit larger to absorb some of it — but that is fitting to data, receptor-
specific, and won't generalise; it is not a physics fix.) **Conclusion: the lever is the Hamiltonian, not the
estimator.** This is why all our earlier LIE-flavoured ideas (N2 ⟨V_elec⟩) also hit the wall.

## How production FEP that gets salt bridges right does it
- **Polarizable force fields** — AMOEBA (Ponder) or Drude oscillators (CHARMM) explicitly model the electronic
  response, so buried ion pairs are stabilised correctly. **OpenMM supports both natively.** Cost: 10–100× slower,
  harder parametrisation. This is the real fix.
- **Correct protonation states** — some "salt bridges" are actually neutral H-bonds (one partner protonated);
  constant-pH / alternate-state modelling matters and improves accuracy.
- **Full single/dual-topology mutation** (pmx, perses, FEP+) — morph atoms+vdW, not just charge (we do charge-only).
- **REST / long (100 ns+) sampling** — only for salt-bridge *networks* that rearrange (not our case; ours is rigid).
- **Charge-change finite-size corrections** (Rocklin/Hünenberger) — we already apply the leading term.

## What we can ADD to FEP — ranked cheapest → hardest (Ram's "add stuff from FEP only")
1. **Empirical polarization correction (HYBRID, most tractable, testable NOW).** The gap is *systematic* — it grows
   with burial and ion-pair strength. Calibrate `ΔΔG = ΔΔG_FEP + f(burial, |q|, n_saltbridge)` on the SKEMPI map
   (E337) and validate held-out. This mirrors published "polarization correction" schemes and turns a physics
   ceiling into a small learned bolt-on. Needs the E337 calc-vs-exp pattern to be clean (small≈ok, large under);
   if it is, a 1–2 parameter correction could recover most of the 4 kcal. **Risk:** it is fitting, so it needs a
   real held-out set and will not generalise beyond the training distribution — honest caveat mandatory.
2. **Alchemically-polarized charges (ACS Omega 2020).** Compute polarized partial charges for the *bound* state
   (quick QM or continuum-polarization) and use them in the bound leg only. Approximates polarization without a
   full polarizable FF. Moderate effort, no new MD engine.
3. **Redo the decharge with a POLARIZABLE FF (AMOEBA/Drude) in OpenMM.** The physically correct fix. Heavy: slow,
   parametrisation, per-complex GPU-days. This is the real T2-adjacent path.
4. **NNP potential (MACE / AIMNet2).** Learned near-QM electrostatics incl. polarization, differentiable → the
   modern FEP backend. The milestone's T2 tier; biggest build.

## The honest strategic read
- The charged-FEP tier is a real, working, precise engine — but its **accuracy ceiling on buried salt bridges is a
  force-field problem (polarization)**, confirmed from three angles. LIE/TI/estimator changes won't move it.
- The **cheapest real lever we own is idea #1** — an empirical polarization correction calibrated on the SKEMPI
  map. It is honest *if* held-out-validated and *if* we don't oversell it as first-principles.
- The **physically correct lever is a polarizable FF or NNP** (ideas #3/#4) — the T2 milestone, weeks not hours.
- If neither is in scope for the freeze, the honest deliverable is: the fast scorer + the N5 triage flag (which
  correctly flags these cases), with FEP documented as precise-but-polarization-limited on buried salt bridges.

*E337 (`scripts/e337_skempi_batch.py`, `data/e337_skempi_map.json`) tests whether the gap is the clean systematic
under-estimate idea #1 needs.* Sources: JACS 2022 buried ion pairs (10.1021/jacs.2c00312); JCTC 2023 charge-change
guidelines (10.1021/acs.jctc.3c00757); ACS Omega 2020 alchemically-polarized charges (10.1021/acsomega.0c01148);
charge-changing PPI FEP (PMC6453258).
