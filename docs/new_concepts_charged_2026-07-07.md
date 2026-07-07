# New theoretical concepts for the charged/desolvation wall (2026-07-07)

Ram's two questions: (1) can we run a **partial FEP that only learns the charged/desolvation term**, leaving the
shape to our fast scorer? (2) brainstorm **genuinely new** framings (not LIE/PB/FEP/Hess — those are ancient).

## Q1 — Partial FEP for charged only: YES, and it is the RIGHT design (not a compromise)

A full absolute-binding FEP (ABFE) decouples the peptide in two legs: **(a) electrostatics off** (charges →0)
and **(b) sterics off** (soft-core vdW/cavity). Leg (b) is the expensive, slowly-converging part (cavity
formation, soft-core singularities). **Our scorer already gets the shape/sterics right** (neutral r≈0.53). So
we only need leg (a) — the **electrostatic-decoupling leg** — which is much cheaper:
- charging free energies obey **linear response** → converge in ~3 λ-windows, not ~12;
- no soft-core sterics → no slow cavity sampling;
- run it as a *correction*: `ΔG = fast_scorer(charge-neutralized peptide) + ΔG_charge_leg`.

Estimated ~**10–50× cheaper** than full ABFE per candidate. This becomes the concrete first target of the
milestone (see `MILESTONE_physics_charged.md`, tier **T1-charged**), not full ABFE.

**But it MUST sample.** Decisive test (E317, n=40 charged complexes with structures): every *single-structure*
electrostatics descriptor — including the clever ones — is **r≈0 vs the charged residual** (what shape misses):

```
  descriptor vs charged residual (n=40)                 r
  distance-dependent-dielectric Coulomb (ε=r)        +0.00
  charge-scaling dE/dλ (linear-response slope)       +0.00
  Born desolvation (q²·burial)                       −0.02
  linear-response ½·⟨V⟩ (Marcus, single structure)   +0.00
  frustration |Coulomb|·desolvation                  −0.06
```

Why they all fail: the charged signal is the **reorganization energy** = ½·Var(V_elec) over an ensemble
(fluctuation–dissipation). A *single* structure has no variance, so ½⟨V⟩ is not a linear response — it is just
half of one number. **This is exactly why partial-FEP still needs a short MD** (to get the fluctuation), and
also why it is still cheap (only the charging leg, linear-response-converged). The n=40 test is the proof that
no static shortcut exists — it is the empirical floor under the whole milestone.

## Q2 — New framings (novel; each marked TESTABLE-NOW / NEEDS-BUILD)

### N1. Error-structure-defined alchemy (neutralization-delta) — NEEDS-BUILD, the flagship
Define the alchemical region by the **ML model's known blind spot**, not by chemistry. Our scorer is side-chain-
chemistry-blind (poly-ALA moved ΔG 0.07 kcal, E308) but shape-accurate. So: score the *charge-neutralized*
peptide with the fast scorer (its accurate regime), then let a charge-only FEP leg supply *only* the charged
Δ. Novel because the expensive method is aimed exactly at the cheap method's residual — a hybrid where FEP
computes the ML model's error, not the whole answer. This is Q1 made concrete.

### N2. Fluctuation-from-the-generative-cloud — TESTABLE-NOW (needs pose ensembles)
Reorganization energy = ½·Var(V_elec). We lack MD, but we have RAPiDock's **generative pose cloud**. Estimate
Var(V_elec) over the pose ensemble as a surrogate for the MD fluctuation → the desolvation penalty, *no MD*.
Caveat: RAPiDock poses are not Boltzmann-weighted (shown before) — but the *electrostatic variance* specifically
may carry signal even if the energies don't. Novel: fluctuation–dissipation on a diffusion-model ensemble.
Test: generate N poses per charged complex, compute Var(V_elec), correlate with the charged residual.

### N3. Learned local dielectric (delta-learn the cancellation) — TESTABLE-NOW at scale if PB energies exist
The catastrophe is Coulomb − desolvation ≈ small. Instead of assuming ε, **feed both large terms as inputs**
and train a model to output the residual: it learns ε(burial, packing, exposure) — the local dielectric as a
function of environment. Different from E311 (which used contact counts, not physical PB energies) and from the
linear tests above (which combined terms linearly). Needs real PB energies (APBS) at n≥100 — the n=40 structure
set is the current blocker, so this is a **data/curation task first**.

### N4. Cycle-closure as a training loss (Hess as a regularizer, not a predictor) — NEEDS-BUILD
FEP satisfies thermodynamic-cycle-closure by construction. Inject that as a **physics-informed auxiliary loss**:
penalize the scorer when its predicted ΔGs around a peptide×receptor grid don't close. Unlabeled grids become
training signal (semi-supervised). Novel as a *training objective* for a docking scorer — it imports FEP's
self-consistency into a cheap model without running FEP.

### N5. Charge-frustration triage flag — TESTABLE-NOW, promising lead
Can't fix charged, but can **flag which complexes need FEP**. E317: "frustration" (|Coulomb|·desolvation)
predicts the *magnitude* of the charged error at **Spearman −0.55** (n=40) — low-frustration charged complexes
have ~3× the error (1.48 vs 0.47 kcal). So a cheap frustration score could route only the high-error charged
cases to the expensive FEP leg, and let the fast scorer handle the rest. (Sign is counter-intuitive and n=40 —
follow up before shipping, but it is a real magnitude signal, unlike everything that tried to fix the value.)

### N6. Adiabatic-connection surrogate — BLOCKED (documented so it isn't re-proposed)
DFT gets exchange-correlation by integrating over a coupling-strength λ (non-interacting→interacting). Analogue:
integrate the *scorer's* response as electrostatic coupling scales 0→1, replacing the per-λ ensemble with the
scorer. Blocked by the same wall — our scorer has ~0 charge sensitivity (poly-ALA), so its λ-response is flat.
Needs a charge-sensitive base energy first (→ N1/T1-charged). Kept here to close the loop.

## Bottom line
- **Partial FEP for charged = yes, and it's the correct cheaper design** (charge-decoupling leg only, linear-
  response, as a correction to the neutralized-peptide scorer). New empirical floor (E317): no static electro-
  statics recovers the residual → the leg must sample, but only the cheap leg.
- **Most promising new, testable-now leads:** N2 (fluctuation from pose cloud) and N5 (frustration triage).
- **Flagship build:** N1 (error-structure-defined alchemy) = the milestone's T1-charged.
- Reproduce the tests: `scripts/e317_partial_fep_electrostatics.py`.
