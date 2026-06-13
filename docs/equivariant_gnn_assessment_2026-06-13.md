# Clebsch-Gordan equivariant peptide GNN — feasibility assessment (honest)

**Ram's ask:** an equivariant (CG / e3nn) model of the peptide as 3D structure, constraining motions to
physical ones and modeling bonds, to predict binding better — plus the electrostatic before→after model.

## Infrastructure: FEASIBLE
- `e3nn 0.6.0` + `torch 2.7.0+cu128` + CUDA are installed (rapidock env).
- RAPiDock itself ships `CGTensorProductEquivariantModel` (third_party/RAPiDock) — a working CG-equivariant
  peptide-receptor architecture we could fork for an affinity head instead of a pose-denoising head.
- So building it is an engineering project, not blocked by tooling.

## But the physics says a STATIC equivariant GNN hits the SAME floor — here's the evidence chain
Everything we tested this session converges on one conclusion: **the missing physics (conformational
entropy + electrostatic desolvation) is information that does not exist in a single static pose.**
- Atlas (e114): error in vlong grows with extendedness/disorder/charge = ensemble terms.
- Salt-bridge ML (e118): explicit salt-bridge + desolvation geometry + ML lifts high-charge only +0.017.
- Entropy proxies (e119): no static/sequence proxy surrogates the real MD s_free (|corr|≤0.16).
- Prior: GB desolvation flat on charged, εin screen didn't transfer, single-pose Coulomb washes.

An equivariant GNN is a *better function approximator over one static structure*. It cannot manufacture
information that the single frame does not contain. So a CG-GNN predicting ΔG from one bound pose will, on
our data, plateau where every static method plateaus (~0.4–0.5), and at ~10³ complexes with the
distribution shift we measured it will tend to OVERFIT (GBT+seq already overfit at n=156). It would be a
more expensive way to reach the same ceiling.

## Where an equivariant GNN IS the right tool (honest, high-value)
1. **Pose / bond geometry** — exactly what RAPiDock's CG model already does (pose generation), and what our
   shipped MIT pose-ranker approximates. Equivariance shines for *structure*, not for the missing
   thermodynamics.
2. **Learning a DYNAMICS-derived label** — if trained to predict the MD/FEP-derived ΔΔG or s_free (not the
   raw Kd from one pose), a GNN could distill the ensemble physics into a fast static surrogate. This is the
   only route where a GNN adds affinity signal: it learns the *shadow of dynamics*, with dynamics in the
   training labels. Requires the MD/FEP labels first (the e115 s_free run is step 1 of exactly this).
3. **Processing an ENSEMBLE** (multiple poses / MD frames) rather than one pose — then the network can see
   the conformational spread that encodes entropy. This is a real design, but needs the ensemble as input.

## Recommendation
- Do **not** build a static single-pose CG-GNN for ΔG now — predictable ceiling + overfit risk, and it's
  the "mere copy via NN" trap (a black-box that reaches the same floor).
- **Do** finish the physics-honest path already in motion: compute the ensemble terms (s_free via MD — e115,
  running), and IF they recover the failure regimes (e116 grade), THEN a small equivariant net trained to
  predict those dynamics-derived quantities from static structure becomes the worthwhile, novel build —
  a fast surrogate for the expensive physics, not a copy of PPI-Affinity.
- The electrostatic before→after idea is sound but the **static** realization (e118) doesn't crack the floor;
  its GNN version would face the same single-frame limit. The fix there is explicit-water/FEP labels, then
  distill — same pattern.

**One-line verdict:** equivariant GNNs model *structure* superbly and *thermodynamics* no better than any
other single-pose method — so point them at dynamics-derived labels, not raw Kd from one frame.
