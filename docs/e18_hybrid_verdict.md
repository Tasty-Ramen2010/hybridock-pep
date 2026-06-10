# E18 — Hybrid pipeline (SASA-dict + Ramachandran-W entropy + ESM cooperativity): VERDICT

**Date:** 2026-06-10. Built faithfully to Ram's full 3-stage spec + ESM cooperativity,
tested honestly (cross-dataset transfer + leave-group-out + ablation). Hypothesis:
the trained SYNTHESIS may beat the sum of parts. **Result: it does not.**

## Built (all 3 stages)
- Stage 1: per-residue ΔSASA × Eisenberg hydropathy (`e18_hybrid_features.py`)
- Stage 2: W_unbound = Π n_basin[aa] (Ramachandran wells); −TΔS = kT·lnW, W_bound≈1
- Stage 3: ESM-2 650M attention → per-residue coupling; discount entropy overcount
  lnW_eff = Σ ln(n_basin_i)(1−λ·coupling_i), λ=0.7 (`e18_esm_coupling.py`, rapidock env)
- Combine + linear-regression train vs experimental ΔG (`e18_train_eval.py`)

## Results
**Absolute cross-target (fit crystal-65 → predict PEPBI):** FAILED.
- all models negative transfer (−0.24 to −0.56); RMSE 3.2–3.4 vs mean-baseline 1.51.
- The synthesis does NOT predict absolute ΔG. (Catastrophic cancellation + per-protein
  baseline confound, as proven earlier.)

**Within-target leave-group-out (PEPBI):** NO reliable gain.
- combined vs baseline per-group: 7/14 groups up, 7/14 down = coin flip.
- "improvements" only on small narrow-range groups (PTPA/SGT2/SOCS, n=9-12) = noise;
  large reliable groups flat or worse (SH3 −0.11, CAPERα −0.87, Pyk2 −0.53).
- pooled LOGO combined 0.459 ≈ baseline 0.453 (no gain).

**Ablation:** combined ≈ sum of parts ≈ hb+aromatic baseline. ESM cooperativity term
did not systematically help. The "whole > sum" hypothesis is unsupported.

## Why (consistent with everything prior)
Each component is size-confounded; combining + cross-family training reproduces the
size/baseline confound (the v1.2 backwards-sign failure). The bottleneck is the data
ceiling (~20-30 independent Kd families) + the fundamental walls (cancellation, per-
protein baseline), not the feature recipe. NOT promoted to src/ (did not beat baseline).

## What was nonetheless validated
Rigorous, faithful test of a reasonable hypothesis with a clean negative answer, full
cross-dataset + per-group honesty. The ESM-attention-as-entropy-correction idea is
novel; here it adds no systematic signal (mean coupling 0.897 — attention is nearly
saturated/uniform on short peptides, so it provides little discriminating structure).
