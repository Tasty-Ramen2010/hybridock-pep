# Milestone: physics-based refinement to break the charged/absolute wall

**Status:** scoped + feasibility-proven (E316). **NOT** started as a build. **Scope:** post-iGEM-freeze — this
is a multi-week engineering + GPU-compute project, not a freeze deliverable. Written honestly: nothing here is
implemented beyond the feasibility proof-of-concept.

## Why this milestone exists
Across E305–E315 (>15 ideas), the charged/absolute-ΔG wall is **feature/signal-limited and FEP-bound**: the
decisive quantity is the small *difference* between a charge's desolvation penalty and the compensating Coulomb,
hidden inside two large terms. Proven from every cheap angle:
- static feature engineering (E311, 10 ideas) — flat at r≈0.40;
- ML relative ΔΔG / feature-differencing (E312) — worse (r 0.10);
- single-point physics (Coulomb/screened/Born, GB/PB net) (E312) — net = noise;
- 3D-RISM pocket hydration (E315) — real but a receptor *offset*, doesn't generalize (PPIKB ≈0);
- stepwise mutation scoring (E313) — single mutations are 51% coin-flip (shape-dominance);
- `--ultra` / variance reduction (E314) — tightens ranking, cannot add signal.

The **only** path that creates the missing signal is real alchemical sampling: integrate ⟨dU/dλ⟩ so the large
solvation terms never appear absolutely (the same reason FEP works and cheap methods don't).

## Tiers (T1-charged is the cheaper, sharper first target — Ram's partial-FEP idea)
| tier | engine | new deps | accuracy | cost/ΔΔG | when |
|---|---|---|---|---|---|
| **T1-charged — electrostatic-leg-only** | openmm+openmmtools+pymbar (**installed**) | none | targets ONLY the charged term the scorer misses | ~10–50× cheaper than ABFE | **first** |
| **T1 — classical relative-FEP refine** | same | none | ~1–1.5 kcal peptide ΔΔG (lit: FEP+ ~1.1) | GPU-hours | after T1-charged |
| **T2 — NNP-FEP** | + MACE / TorchANI / AIMNet2 differentiable potential | torch NNP + weights | higher accuracy and/or 10–1000× faster | GPU-min–hr | follow-on |

**T1-charged (E317, Ram's "partial FEP for charged only"):** full ABFE decouples in two legs — electrostatics
and sterics; the sterics leg is the expensive slow one, and our scorer *already gets sterics/shape right*. So
run **only the electrostatic-decoupling leg** as a correction: `ΔG = fast_scorer(charge-neutralized peptide) +
ΔG_charging_leg`. Charging free energies obey linear response (≈3 λ-windows, no soft-core) → much cheaper.
**E317 proved it must still SAMPLE:** at n=40 no single-structure electrostatics (incl. charge-scaling
derivative, linear-response ½ factor, distance-dependent dielectric) correlates with the charged residual
(all r≈0) — the signal is the reorganization = ½·Var(V_elec), which needs an ensemble. So T1-charged = the
cheap charging leg *with* a short MD, aimed exactly at the scorer's blind spot (concept N1,
`docs/new_concepts_charged_2026-07-07.md`).

**Feasibility (E316, proven):** the T1 alchemical pipeline is mechanically buildable in `openmm-env` today — we
constructed a real `AbsoluteAlchemicalFactory`/`AlchemicalState` and swept `lambda_electrostatics` 1→0 with a
smooth potential decoupling (−21→+32 kcal/mol = the dU/dλ). **T1 needs no new dependency.** T2 needs an NNP
(none installed) and is the speed/accuracy optimization, not a prerequisite.

**T1-charged charging leg run + linear-response confirmed (E322 Part B, `scripts/e322_t1charged_partB_...`):**
the electrostatic-decoupling leg runs end-to-end with real Langevin sampling + MBAR in `openmm-env`, and the
**cheapness claim is now measured, not assumed**: a 3-λ-window schedule agrees with a dense 11-window schedule
to **0.14 kcal/mol** → linear response holds → ~3 windows suffice for charged-only, the basis of the
~10–50× saving over full ABFE. (The sampled number annihilates *all* partial charges; reproducing a real
peptide ΔΔG against explicit water is still G1-full.)

**Decomposition validated (E322 Part A):** a scorer calibrated on neutral complexes (Ram's "special calibrated
fast_scorer") has a charged residual that is sizeable (1.79 kcal) and mostly shape-orthogonal (mean shape-corr
0.09) → `ΔG = scorer_neut + charge_leg` is a clean split. The residual is only weakly charge-*indexed*
(r=−0.16 vs |q|), i.e. it is per-complex, **not** a |q| table — confirming the leg must sample. Architecture:
run scorer_neut under `--ultra` smoothing, add the sampled charge leg for the top-K only.

**N2 partial win — cheap ensemble already carries some of it (E318):** LIE's electrostatic term is β·⟨V_elec⟩
over an ensemble. Computed over RAPiDock's generative pose cloud (no MD), ⟨V_elec⟩ tracks the charged residual
at r=−0.37 (charged subset) and lifts LOO r 0.501→0.552. Underpowered (n=24) but the first signal to beat the
single-structure floor → **G2-precursor**: before full FEP, test whether a large charged pose-cloud set
(e93+e95 on disk = 151 clouds; PPIKB would need GPU generation) confirms ⟨V_elec⟩/Var(V_elec) as a no-MD
surrogate for the reorganization.

**N5 triage flag (E321):** frustration = |Coulomb|·desolvation predicts the *magnitude* of the charged error
(Spearman −0.545, perm-p=0.000, held-out 3.4× separation) → route only high-error charged complexes to the FEP
leg. Shippable as a per-complex "charged-confidence" flag independent of the FEP build.

## E333 relative charge-morph — where it sits vs literature and our own prior tests
Ram's "derivative it instead of cancelling two huge terms" → integrate the DIFFERENCE of ⟨∂U/∂morph⟩ curves for a
small charged→neutral morph, never forming two large absolute numbers.

**Vs literature (it is a known, sound idea):**
- **TI (Kirkwood 1935)** — e333 computes ⟨∂U/∂λ⟩ and integrates; textbook.
- **Relative FEP / RBFE (FEP+, pmx, perses)** — the whole reason RBFE beats two absolute (ABFE) runs is exactly
  Ram's point: compute the *difference* directly so common terms cancel and you never subtract two big numbers.
  e333's charge morph is a relative transformation; Ram independently re-derived the RBFE rationale.
- **Separated-topology / correlated RBFE (Baumann–Gapsys 2023; Rocklin 2013)** — formalise integrating a coupled
  difference for variance reduction. e333 is a lightweight cousin (fixed topology, charges only).
- **Charge/partial-charge alchemy** — morphing partial charges is the electrostatic leg of any mutation FEP.
- **Rocklin/Hünenberger net-charge correction** — still required (neutralisation changes net charge); applied.

**Honest gaps vs full RBFE:** (1) e333 morphs *charges only on a fixed topology* — it is the ELECTROSTATIC part
of a Lys→Gln mutation, not the vdW/atom change (a real pmx/perses single-topology mutation adds softcore-mapped
atoms; we lack those deps). That is fine here because the charged term is exactly what our scorer misses. (2) The
bound/free legs are still separate sims, so the difference-of-derivatives cancels common terms only insofar as
peptide↔solvent ∂U/∂m is similar across legs — a *partial* correlated-sampling gain, plus the real win of a
SMALL neutralisation perturbation vs annihilation. Fully correlated sampling would need shared noise/a coupled
box.

**Vs our own prior tests (why e333 is the right escalation):**
- E313 (mutations scored by OUR model) = coin-flip — scorer is charge-blind; e333 uses the MD free energy.
- E317/E327 (static Coulomb/Born, cheap neutralisation double-difference) = null — no sampling; e333 samples the
  reorganisation.
- E329 (annihilate) = −12.4 ± 39.2 — two ~+330 kcal legs; e333 removes the huge magnitude by morphing, not
  annihilating, and integrates the difference.
- E332 (decouple) = removes the intramolecular self-energy but still absolute; e333 goes further (relative,
  neutralise-not-remove, difference-of-derivatives).
- N2/N5 = the cheap ensemble/triage layer; e333 is the expensive lab-grade tier they route TO.

## T1 architecture (the buildable path)
```
  dock (fast, current) → top-K candidate poses on the target
        │  (only the few that matter — FEP is expensive)
        ▼
  --fep-refine K :
    1. solvate + parametrise complex (OpenFF/ff14SB via openmm-forcefields / tleap)
    2. single-topology alchemical map between candidate peptides (hard part: atom mapping;
       trivial for point mutations, needs a mapper for diverse panels)
    3. λ schedule (elec then sterics), replica-exchange (openmmtools ReplicaExchangeSampler)
    4. run BOTH legs — bound complex AND free peptide (Perses cycle) — MBAR estimate (pymbar)
    5. ΔΔG = ΔG_transform(bound) − ΔG_transform(free);  cycle-closure correction across the panel
        ▼
  fep_ddg column: FEP-grade RELATIVE affinities for the top candidates
```

## Go/no-go gates (each blocks the next)
- **G1 — validation:** reproduce a known ΔG_solv (single ion/side-chain analog) to <0.5 kcal, and a published
  peptide ΔΔG (e.g. an MDM2 or BH3 point mutant) to ~1 kcal. *If G1 fails, classical FEP can't do our peptides
  → escalate to T2 or stop.* **G1-partial DONE (E316):** the full build→sample→MBAR loop reproduces the
  *analytical* free energy of a harmonic-oscillator ladder to **0.01 kcal/mol** — the estimator machinery is
  present and correct.
  **G1-spike RAN on a real complex (E329, `scripts/e329_g1_charged_spike.py`):** full pipeline works end-to-end
  on a real charged complex (2jqk) — PDBFixer + amber14 parametrisation, explicit-TIP3P solvation (bound 20 373
  / free 5 331 atoms), decharge both legs, ReplicaExchange + MBAR, **on the Blackwell GPU** (OpenMM 8.5.1 CUDA
  works). BUT the naive result is **ΔΔG_elec = −12.4 ± 39.2 kcal/mol — dominated by noise, unusable.** Two hard
  lessons this surfaced concretely: (1) **catastrophic cancellation is numerical, not just conceptual** — the
  two legs are ~+325 and ~+337 kcal (full charge annihilation incl. intramolecular self-energy), and subtracting
  two ~+330 numbers with ±12–37 kcal noise destroys the ~−12 signal; (2) the **free leg is the convergence
  bottleneck** (±37 kcal — floppy solvated side chains). Reaching ±4/±2/±1 kcal needs ~96×/385×/1540× more
  sampling → GPU-days per complex *unless* the setup is fixed. **G1-full therefore requires** (a) a charge-
  balanced / co-alchemical or interaction-only scheme to shrink the ~+330 kcal absolute magnitudes, (b) the
  Rocklin/Hünenberger net-charge finite-size correction, (c) far more sampling (esp. the free leg). The spike
  confirms the path is *buildable and runs*, and quantifies exactly why a converged ΔΔG is expensive.
- **G2 — the charged proof:** on a charged case where static ranked BACKWARDS (importin/NLS: static −9.77,
  single-point MM-GBSA −92, both wrong), show FEP ranks it correctly vs a strong binder. *This is the
  north-star: sampling creates the signal static cannot.*
- **G3 — cost/benefit:** median ΔΔG error and wall-clock on a 5–10 pair charged benchmark; decide if the
  GPU-hours/ΔΔG are worth it for the iGEM use case (screening → wet lab). Likely: FEP only for the final 2–3
  candidates, not the panel.

## Honest risks
- **Sampling cost:** ns/window × ~12 windows × 2 legs × charged = GPU-hours per ΔΔG; only viable for a handful
  of final candidates, never a screen. This does not replace `rank_score`; it caps it.
- **Force-field limits:** fixed-charge FF may itself cap charged accuracy (polarization) — the exact reason T2
  (NNP/polarizable) exists.
- **Atom mapping** for diverse (non-point-mutant) panels is a real engineering problem.
- **Convergence/reproducibility** must be logged like everything else (seeds, λ schedule, overlap).

## What is DONE now (this session)
- E316 feasibility POC (`scripts/e316_fep_feasibility_poc.py`): (a) T1 alchemical pipeline mechanically
  buildable, no new deps — charges decouple smoothly with λ; (b) **G1-partial** — the full build→sample→MBAR
  estimation loop reproduces an analytical free energy to **0.01 kcal/mol**, so the estimator machinery is
  correct, not just the system construction.
- Tooling inventory: openmm/openmmtools/pymbar present; no NNP installed.
- This spec with go/no-go gates. **Nothing else is built.** Next concrete step = G1-full (solvate/parametrise a
  real complex, reproduce one published peptide ΔΔG) behind an experimental `--fep-refine` flag, clearly
  labelled non-production until G1/G2 pass. Honest status: the wall is FEP-bound, the FEP path is de-risked and
  buildable, but a converged peptide ΔΔG is GPU-hours and this is a post-freeze milestone, not a freeze feature.
