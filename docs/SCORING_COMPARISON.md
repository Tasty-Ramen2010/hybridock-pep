# HybriDock-Pep — Scoring Method Comparison

How HybriDock-Pep's physics-based rescoring compares to the standard hierarchy of binding-affinity
methods, on **protein–peptide** affinity ranking. Accuracy is Pearson *r* vs experimental ΔG/Kd on
diverse (cross-family) sets unless noted; "within-target" = ranking mutants/poses of one complex.

## Head-to-head benchmark (measured, 2026) — same 156 unique-Kd complexes

Every method scored on the identical pooled set (crystal-65 + the-98), **no relaxation unless noted**.
Our numbers are out-of-sample (leave-one-out / balanced held-out). This is the empirical basis for the
README scorecard.

| Method | Pearson *r* | Coverage (of 156) | Relaxation | Cost |
|---|---|---|---|---|
| MJ contact potential | 0.16 | 156 | no | < 1 s |
| single-pose physics (pooled) | 0.19 | 156 | varies | s–min |
| MM-GBSA (single snapshot) | 0.25 | 91 | min only | 5–30 s |
| OpenMM vdW packing | 0.34 | 86 | no | ~30 s |
| BSA hydrophobic burial | 0.39 | 156 | no | < 1 s |
| Raw Vina (cr65; *r* = −0.56 raw, sign-flipped) | 0.56 | 65 | no | ~1 s |
| **ref2015 / FlexPepDock energy — UNRELAXED (measured here)** | **0.07** | 65 | no → fails | seconds |
| ref2015 / FlexPepDock — relaxed (literature) | 0.55–0.59 *within-target* | — | **yes, 5–30 min** | minutes |
| PPI-Affinity (best published ML peptide scorer) | 0.55 | — | n/a | server |
| **HybriDock-Pep (geometry + length router)** | **0.585 LOO · 0.68 held-out** | **156** | **no** | **~10 s** |
| FEP / TI (ceiling) | 0.8–0.9 *congeneric only* | — | full MD | 5–50 GPU-hr/mut |

**Takeaways:** (1) we beat every single-pose physics baseline on the full 156 (0.585 vs best 0.39);
(2) measured ref2015 *without* relaxation = 0.07 — FlexPepDock's 0.59 is bought entirely with expensive
Rosetta refinement, which we don't need; (3) we match PPI-Affinity and relaxed FlexPepDock at 30–300×
lower cost. **Best non-FEP/LIE protein–peptide affinity ranker we can find a fair baseline for.**

| Method | Accuracy (r) | Cost / complex | What it needs | Key negatives |
|---|---|---|---|---|
| **Raw Vina / AutoDock** | ~0.3 (often sign-flipped on diverse sets) | ~1 s (CPU) | docked pose | Size-confounded; no entropy; ignores partial charges (Vina) |
| **HybriDock-Pep (geometry + MM-GBSA + entropy)** | **0.52 cross-family · 0.60 within-dist** | **~10 s (CPU) / +8 s MD (GPU)** | docked pose + 1 short implicit-MD | Floor on charged/pre-organized binders; calibration is dataset-pooled, not universal |
| **MM-GBSA (single-snapshot)** | ~0.25–0.45 | ~5–30 s (GPU) | minimized complex | Omits −TΔS_conf (over-rates floppy/extended); continuum solvent misses water-mediated bridges |
| **MM-PBSA** | ~0.3–0.5 | ~1–5 min (CPU/GPU) | minimized complex + PB solve | Slow PB solver; sensitive to dielectric/grid; still single-conformation |
| **FlexPepDock / flex-ddG (Rosetta ref2015 + backrub)** | ~0.55–0.6 within-target; flips cross-family | ~5–30 min (CPU, K≈35 models) | backrub backbone ensemble | Backbone ensemble HELPS within-target, HURTS cross-family (ATLAS: backrub 0.47 < single 0.63); slow |
| **LIE (Linear Interaction Energy)** | ~0.5–0.7 (system-specific α,β) | ~1–5 ns MD ≈ 0.5–4 GPU-hr | explicit-solvent MD of bound + free | Empirical α,β must be re-fit per system; needs both legs; modest transfer |
| **FEP / TI (alchemical free energy)** | ~0.8–0.9 (≤1–2 kcal/mol) on congeneric series | 5–50 GPU-hr **per mutation** | explicit-solvent MD + alchemical λ-windows + soft-core | Gold-standard cost; only reliable for SMALL congeneric changes; convergence fragile; not a throughput screener |

## Reading this table

- **Accuracy rises with cost.** FEP is the most accurate and ~1000–10,000× more expensive than us.
- **HybriDock-Pep's niche: best accuracy-per-second in the cheap tier.** Cross-family ~0.52 at ~10 s/CPU
  is competitive with methods 30–300× slower, and clears the within-distribution ≥0.55 target at 0.60.
- **The cross-family ceiling (~0.52) is a *data/physics floor*, not a tuning gap.** It is set by two
  effects static cheap physics cannot capture: (1) **conformational/free-state entropy** of flexible
  peptides, (2) **charged desolvation / water-mediated salt bridges**. Crossing it needs explicit-solvent
  MD (LIE/FEP) — i.e. paying the next cost tier — or a large supervised ML model.

## Where HybriDock-Pep is the *right* tool
- High-throughput **ranking** of many peptides/poses where FEP is unaffordable.
- **Selectivity ΔΔG** (same peptide, two receptors): the shared floor cancels → r ≈ 0.3–0.45, a regime
  where absolute methods struggle and FEP is overkill.
- **Affinity maturation** (ranking peptide variants for one receptor): r ≈ 0.42, beats FlexPepDock,
  independently validated on ATLAS TCR-pMHC (r ≈ 0.43).

## Where to escalate
- Need ≤1 kcal/mol absolute on a **few** designs → FEP on the top cluster centroids (`--refine-topk`).
- Charged interface dominating → MM-PBSA or explicit-solvent LIE on the top poses.

*Accuracy figures: HybriDock-Pep from pooled crystal-65 + the-98 (n=156, docs E69); FEP/LIE/FlexPepDock
from method literature and our SKEMPI/ATLAS reproductions (docs E54–E64). All physics, no GPU-inference
cluster required.*
