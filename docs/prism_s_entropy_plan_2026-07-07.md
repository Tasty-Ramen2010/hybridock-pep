# PRISM-S: attacking the entropy bottleneck by the confinement path (Ram's idea, formalized)

**Date:** 2026-07-07 · Status: method design + brainstorm · Trigger: Ram refuses the absolute-Kd wall and proposes
computing binding **entropy** the way FEP computes **energy** — along a derivative path (progressively "locking
pieces" and measuring what's lost), factoring inter-regional constraints and folding in both free and bound
states, so it is **not** the cancellation of two huge entropies.

**Headline: this is not a fantasy — it is a real, published method family (the confinement method / Mining-Minima
M2 / MIE), and Ram independently reconstructed its three hardest requirements.** Below: the mapping, the physics of
why it dodges the cancellation, the honest caveats, and the decisive experiment that gates the build.

---

## 1. Ram's idea ↔ the literature (near one-to-one)

| Ram's words | established method | source |
|---|---|---|
| "derive entropy along a derivative, not cancel two huge terms" | **Confinement method** — integrate the *work of restraining* the molecule along a harmonic-strength path λ; ΔS emerges from an integral of a smooth derivative | [Cecchini confinement PMC3710665](https://pmc.ncbi.nlm.nih.gov/articles/PMC3710665/), [Simplified Confinement Method, JPCB jp3080578](https://pubs.acs.org/doi/abs/10.1021/jp3080578) |
| "slowly lock pieces to see how much it loses out" | **M2 / Mining-Minima**: progressively *freeze* DOF; "removing DOF freezes them and reduces fluctuations of those that remain, because of coupling" | [M2 binding entropy PMC3064472](https://pmc.ncbi.nlm.nih.gov/articles/PMC3064472/), [ligand config entropy PNAS 0610494104](https://www.pnas.org/doi/10.1073/pnas.0610494104) |
| "factoring inter-regional constraints" | **Mutual Information Expansion (MIE)**: higher-order backbone↔sidechain correlations; single-DOF estimates are *wrong* | [MIE higher-order S0009261424007693](https://www.sciencedirect.com/science/article/abs/pii/S0009261424007693) |
| "folding issues in both free and bound environments" | **Coupled folding-and-binding**: peptides are mini-IDPs; entropy loss ∝ stability of the *unbound* ensemble | [coupled folding JACS 1c04214](https://pubs.acs.org/doi/10.1021/jacs.1c04214), [IDP conf. entropy JACS 0c03217](https://pubs.acs.org/doi/10.1021/jacs.0c03217) |

**Ram's instinct is correct and specific.** The confinement method *"estimates directly the configurational
entropy difference between two macrostates, without additional computation of free energy or enthalpy"* and *"is a
special case of thermodynamic integration, trivially parallel over the integration variable"* — i.e. the exact FEP
trick, applied to entropy.

## 2. Why the confinement path dodges the cancellation (the physics)

Direct entropy is `TΔS = T(S_bound − S_free)` — each S is tens of kcal, estimated from the *full* high-dimensional
fluctuation distribution → catastrophic cancellation **and** brutal convergence. That is the wall.

Confinement replaces it. Define a harmonic restraint toward a reference structure with strength `k(λ)`, λ: 0→∞.
- At λ=0 the molecule fluctuates freely (its real entropy).
- At λ=∞ it is pinned to the reference (a known analytic reference entropy — the Einstein-crystal / harmonic ideal).
- The **free energy of confinement** `ΔG_conf = ∫ ⟨∂U_restraint/∂λ⟩ dλ` is a smooth, convergent integral of a
  well-behaved derivative (mean restraint force) — **no cancellation of two huge numbers**.

Then `TΔS_config = ΔG_conf(free) − ΔG_conf(bound)` (each side an integral, referenced to the *same* analytic pinned
state, so the huge reference entropy cancels *analytically*, not numerically). **This is exactly Ram's "lock the
pieces and measure what's lost, along a derivative."** The Simplified Confinement Method needs no normal modes and
no force-field switch-off — minimal code over our existing OpenMM restraints.

## 3. The three subtleties Ram flagged — and why they're load-bearing
1. **Inter-regional constraints (coupling).** You cannot sum per-residue entropies: *"coupling among rotation/
   translation and internal DOF complicates decomposition into additive terms"* and freezing one region changes
   the others. The **path/order matters** — integrate while progressively locking, so coupling is captured as you
   go (MIE up to 3rd–4th order; the sign of ΔS_config literally flips with correlation order).
2. **Both free and bound.** Peptide entropy loss ∝ the *unbound* ensemble's breadth (coupled folding-binding). A
   floppy free peptide that rigidifies on binding pays a huge −TΔS; a pre-organized one pays little. **The free-
   state ensemble is half the signal** — and it is the half single-pose scoring is completely blind to.
3. **Solvent entropy compensates.** Protein/ligand/solvent entropies trade off ([galectin-3C JACS Au 0c00094](https://pubs.acs.org/doi/10.1021/jacsau.0c00094)).
   We already have the solvent piece cheaply: **GIST/3D-RISM** localizes water entropy on a grid ([GIST AmberHub](https://amberhub.chpc.utah.edu/gist/)),
   and our RISM Layer-3 infrastructure (E349) is the same machinery.

## 4. The honest caveats (this is hard, not a guaranteed win)
- **Entropy corrections have a mixed track record for affinity.** Normal-mode entropy *"is often omitted in ranking
  relative affinities as its inclusion often does not improve agreement"*; interaction-entropy exponential
  averaging is *"extremely poorly conditioned"* ([RSC c7cp07623a](https://pubs.rsc.org/en/content/articlelanding/2018/cp/c7cp07623a),
  [JCTC 1c00374](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00374)). **But those are the *bad* estimators.** The
  confinement path is the *well-conditioned* one, and peptides are the regime where entropy dominates most — the
  best-case for it to matter.
- **The cross-target info-theoretic ceiling (Front 4) still applies** to *learned* features. Confinement entropy is
  a *computed physics* term, not learned from the training distribution, so it is not bound by that ceiling the
  same way — *if* it is accurate.
- **Our residual test:** the scorer residual has **no crude-entropy shape** (corr with length −0.02, n_rot −0.04,
  flexibility −0.08; adding proxies Δr −0.001). Crude ≠ real — a residue count cannot fake MIE-order correlation
  entropy — so this does **not** rule entropy out. But it means we must test the **real** computed term.

## 5. The decisive experiment (the gate — do this before building the full tier)
**Compute real confinement ΔS_config on a subset (~30) of the peptide-Kd complexes, and test whether it correlates
with the scorer residual.**
- If `corr(TΔS_confinement, residual)` is meaningful (say |r|>0.25) → entropy *is* the missing physics; build the
  full PRISM-S tier. This would be the first thing in the entire campaign to have entropy shape.
- If it's ~0 → even correctly-computed entropy doesn't explain the residual → the wall is genuinely elsewhere
  (label heterogeneity / allostery / irreducible), and we stop with a proof, not a hunch.

This is the same discipline that killed the charged-Kd hope honestly (E353b) — point the forensic lens at the
*real* term, not a proxy.

## 6. PRISM-S design (if the gate passes)
```
TΔS_bind(peptide) = [ΔG_conf(free peptide) − ΔG_conf(bound peptide)]      (configurational, confinement TI)
                  + TΔS_solvent(GIST/RISM on displaced pocket water)       (we have the machinery, E349)
```
- **Free-state ensemble** = the RAPiDock N=100 generative cloud (we already generate it!) → no extra sampling for
  the unbound reference; confine each state toward its own mean.
- **Bound-state** = short restrained MD of the docked pose (receptor frozen) + confinement ramp.
- **Cheap path** = **nonequilibrium fast-switching** (Jarzynski/Crooks): *"200 sufficiently long switches, 40× cost
  reduction"* ([JCTC nonequilibrium](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01453)) instead of full equilibrium
  TI — the affordable version Ram wants.
- **Regional decomposition** = progressively confine backbone → then sidechains, integrating the coupling (MIE-2/3),
  so we get *where* the entropy is lost (per-residue) as a bonus diagnostic.
- Plugs into `--ultra` next to PRISM's charged term: ΔG = ΔG_scorer + ΔG_charged(PRISM) − TΔS(PRISM-S).

## 7. Why this could work where charge didn't
Charge failed on absolute Kd because charge is a *small, already-captured* slice (Front 2/E353b). **Entropy is the
opposite: it is the *largest* un-modeled term, it is *largest for peptides*, single-pose scoring is *completely*
blind to it (especially the free-state half), and it does not cancel in absolute prediction.** It is the one term
whose size could actually move an absolute number — *if* the gate experiment shows the residual has entropy shape.

## 8. Immediate next step
Prototype the confinement ΔG on 1–2 peptides (validate the OpenMM restraint-ramp pipeline), then run the ~30-complex
gate: `corr(TΔS_confinement, scorer_residual)`. Build PRISM-S only if it passes.

---
### Sources
Confinement/SCM: [PMC3710665](https://pmc.ncbi.nlm.nih.gov/articles/PMC3710665/), [jp3080578](https://pubs.acs.org/doi/abs/10.1021/jp3080578).
M2/Mining-Minima & decomposition coupling: [PMC3064472](https://pmc.ncbi.nlm.nih.gov/articles/PMC3064472/), [PNAS 0610494104](https://www.pnas.org/doi/10.1073/pnas.0610494104).
MIE correlations: [S0009261424007693](https://www.sciencedirect.com/science/article/abs/pii/S0009261424007693), [Tsg101 PMC2758778](https://pmc.ncbi.nlm.nih.gov/articles/PMC2758778/).
Coupled folding-binding: [JACS 1c04214](https://pubs.acs.org/doi/10.1021/jacs.1c04214), [JACS 0c03217](https://pubs.acs.org/doi/10.1021/jacs.0c03217).
GIST solvent entropy: [AmberHub GIST](https://amberhub.chpc.utah.edu/gist/), [PMC3416872](https://pmc.ncbi.nlm.nih.gov/articles/PMC3416872/).
Nonequilibrium fast-switching: [JCTC 4c01453](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01453), [escorted arXiv 0804.3055](https://arxiv.org/pdf/0804.3055).
Entropy-correction reality check: [RSC c7cp07623a](https://pubs.rsc.org/en/content/articlelanding/2018/cp/c7cp07623a), [JCTC 1c00374](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00374).
Entropy compensation: [JACS Au 0c00094](https://pubs.acs.org/doi/10.1021/jacsau.0c00094).
