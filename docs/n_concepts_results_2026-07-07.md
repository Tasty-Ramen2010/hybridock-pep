# N1–N5 tested + LIE/PB/Hess adaptation — results (2026-07-07)

Ram asked to (1) actually **run N1–N5** (not just theorize), (2) **spike-test T1-charged once** — noting it needs a
*special calibrated fast_scorer*, wireable under `--ultra` — and (3) mine **LIE / MM-PBSA(PB) / Hess-cycle**
literature for adaptable ideas, "address the ancient issue that way too." Scripts: `e318`–`e322`.

## Headline
- **N2 is the first crack in the single-structure charged wall.** The LIE electrostatic term is an *ensemble
  average* β·⟨V_elec⟩ (β=0.5 for charged, = linear response). We computed it over RAPiDock's **generative pose
  cloud** (no MD): on charged complexes ⟨V_elec⟩ tracks the charged residual at **r=−0.37** (single crystal
  structure = r≈0, E317), and adding the cloud's electrostatics lifts leave-one-out r **0.501→0.552**.
  *Underpowered* (n=24, boot95% [−0.71,−0.03], perm-p=0.074) — but it is the first thing that beat the
  single-structure floor, and it says exactly what to do next: build charged pose clouds at scale.
- **N5 is a shippable triage flag.** "Frustration" (|Coulomb|·desolvation) predicts the *magnitude* of the
  charged error at **Spearman −0.545**, boot95% [−0.72,−0.30], **perm-p=0.000**, held-out **3.4× error
  separation**. It can't fix a charged ΔG but reliably flags *which* charged complexes to route to the FEP leg.
- **T1-charged is architecturally sound and mechanically cheap.** The neutral-calibrated scorer's charged
  residual is sizeable (1.79 kcal) and mostly shape-orthogonal (mean shape-corr 0.09) → the decomposition
  `ΔG = scorer_neut + charge_leg` is clean; and the charging leg is **linear-response cheap** (3 λ-windows agree
  with 11 to 0.14 kcal/mol). The residual is only weakly charge-*indexed* (r=−0.16 vs |q|) → the correction is
  per-complex, **not** a |q| lookup — which is *why the leg must sample*.
- **N3, N4 = negative** (honest): no static/learned dielectric recovers the residual; cycle-closure isn't the
  bottleneck. Both confirm the wall is a *fluctuation* the single structure lacks.

## Per-concept results

| concept | test (script) | result | verdict |
|---|---|---|---|
| **N1** error-structure-defined alchemy / T1-charged | `e322` A+B | decomposition clean (resid 1.79 kcal, shape-corr 0.09); charging leg linear-response cheap (3-win≈11-win, 0.14 kcal) | **build** — the milestone flagship, now de-risked further |
| **N2** fluctuation from generative cloud (LIE β·⟨V_elec⟩) | `e318` | charged ⟨V_elec⟩~resid **r=−0.37**; LOO 0.501→0.552 | **real but underpowered** — the one positive; scale up clouds |
| **N3** learned local dielectric (MM-PBSA variable-ε) | `e319` | fixed-ε sweep flat (r≈+0.02 for all ε); no static recovery | **negative** — needs the ensemble, not a static ε |
| **N4** cycle-closure loss (Hess) | `e320` | ΔΔG itself r≈0 for both closure-respecting & closure-free models; closure not the bottleneck | **negative** — vacuous for a pointwise scorer |
| **N5** frustration triage | `e321` | Spearman −0.545, perm-p=0.000, held-out 3.4× | **ship** — a which-complex-needs-FEP router |

## LIE / PB / Hess — what the "ancient" methods actually teach, and how we adapted each

Ram's instinct was right: the classical end-point methods already fought this exact battle, and each one's
*mechanism* maps onto one of our concepts.

- **LIE (Åqvist 1994; Frontiers review 2020).** The binding electrostatics = **β·⟨V_elec⟩**, an **ensemble
  average** of the ligand–environment interaction energy over MD of the bound *and* free states, with **β=0.5
  for charged compounds** (lower — 0.43/0.37/0.33 — for neutral, by polar-group count). The key insight we
  adopted: the electrostatic term is *never computed absolutely* — it is a scaled ensemble mean, so the huge
  solvation terms cancel in the bound−free difference. **Adaptation → N2:** replace the MD trajectory with
  RAPiDock's generative pose cloud and compute β·⟨V_elec⟩ over it. This is *why* N2 worked where the single
  structure (E317) failed — LIE told us the quantity is an average, and the cloud supplies the average.
- **MM-PBSA variable dielectric / screening electrostatic energy (Genheden–Ryde; Wang 2021, JCIM 61).** The
  well-known MM-PBSA failure is over-stabilised charged interfaces because a *fixed low* dielectric under-screens;
  the field's fixes are a **higher/variable dielectric for high-charge sites** and an added **screening
  electrostatic term**. **Adaptation → N3:** don't assume ε — *learn* ε(environment) by feeding both large terms
  and regressing the residual. Result: no static ε (fixed or learned) recovers it at n=40 → consistent with LIE:
  the missing piece is the *fluctuation*, and a static dielectric is still a single-structure quantity. PB's
  lesson is real but points back to sampling.
- **Hess's law / thermodynamic-cycle closure (the basis of all relative FEP).** Free energy is a state function,
  so ΔΔG around any cycle must close. **Adaptation → N4:** impose closure as an auxiliary training loss on
  unlabeled grids. Result: a *pointwise* scorer closes every cycle by construction (adds nothing), and a
  relative pair-model's ΔΔG signal is itself ~0 on our features, so closure would only regularize noise. Hess
  helps FEP because FEP's per-edge ΔΔG is accurate; our per-edge signal isn't, so the cycle is empty.

**Net:** the ancient methods converge on one message — the charged term is an **ensemble/fluctuation** quantity
(LIE's ⟨⟩, PB's need for sampling to set ε, FEP's per-edge integral). Every static shortcut (E317, N3) fails for
the same reason; the two things that moved (**N2** with a real ensemble, **N5** as a magnitude flag) are exactly
the ensemble-aware ones. This is the strongest evidence yet that T1-charged (the charging leg *with* sampling)
is the right and only lever — and N2 shows even a cheap generative cloud already carries some of it.

## What ships vs what is milestone work
- **Now (documented, ready to wire):** N5 frustration triage as a per-complex "charged-confidence" flag
  (analogous to the existing `ranking_confidence`); N2's ⟨V_elec⟩-over-poses as an optional charged feature.
- **Milestone (T1-charged, post-freeze, GPU-hours):** the sampled charging leg on real complexes (gate G1-full),
  running the neutral-calibrated scorer under `--ultra` and adding the charge correction for the top-K.

Reproduce: `e318` (N2), `e319` (N3), `e320` (N4), `e321` (N5), `e322_*` (N1/T1-charged A+B).
