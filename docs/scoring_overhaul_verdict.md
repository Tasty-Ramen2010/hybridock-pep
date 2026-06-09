# Scoring Overhaul — Final Verdict (2026-06-09)

Validated on the 65-complex crystal benchmark (`data/benchmark_crystal.json`),
held-out where relevant. This supersedes the optimistic intermediate numbers.

## The one finding that explains everything

On the 65 crystal complexes, the physics/geometry features are **one collinear
size–burial axis**:

| correlation with ΔG_exp | value |
|---|---|
| Vina            | **−0.559** (anti-correlated) |
| n_contact (size) | +0.464 |
| s_ss_weighted    | +0.418 |
| AD4             | −0.321 |
| **Vina ↔ n_contact** | **−0.877** (Vina ≈ −size) |

Every "win" we saw (single-ridge 0.56, per-family LOO 0.65, MM-GBSA raw 0.68)
is the **same artifact**: in this sampling bigger/more-buried peptides bind
weaker, and all features track interface size. A model reaches r≈0.55 only by
using Vina with a **negative** (backwards) slope. Forcing the physically-correct
Vina sign yields **negative** held-out correlation. So none of these generalize —
they will flip sign on prospective complexes.

## What was tested and rejected

| Approach | Result | Verdict |
|---|---|---|
| MM-GBSA single-traj (crystal, n=64) | per-residue r=−0.03; +length worse than length-alone | size meter, **no signal** |
| Per-family ridge (LOO) | r=0.65 | **LOO artifact** — near-duplicate co-crystals (fam 28: 5× same protein @ −13.1) leak the label via the family intercept |
| Per-family ridge (true held-out) | 0.541 vs single-ridge 0.558 | **does not generalize**; regularizing → converges to global |
| Single ridge [vina,nc,s_ss], held-out | 0.558 | **artifact** — relies on backwards Vina slope (size confound) |
| v1.2 production (entropy-only) | −0.42 on crystal | physically honest, hence anti-correlated like Vina |

## Honest conclusion

Absolute peptide ΔG is **not** predictable from Vina/AD4/MM-GBSA/contact/SS
features beyond the interface-size confound, which is a sampling accident, not
generalizable physics. This is a **feature/data ceiling**, not a method gap —
consistent with the 284-set, PEPBI, Wang, exhaustive-search, and burial-axis
findings.

**Therefore (no production calibration change shipped):**
- Do **not** swap in any "0.55" calibration — they encode the size artifact.
- The tool's honest, defensible value is **pose ranking** (2.49 Å, hit@5 91%)
  and **ΔΔG selectivity** (where the size confound cancels), **not** absolute ΔG.
- Absolute ΔG must be reported only as calibrated/relative to a known binder,
  with the ceiling stated — as `docs/scoring_accuracy_analysis.md` already does.

## Scripts (reproducible)
- `scripts/build_crystal_benchmark.py` → the 65-complex crystal set.
- `scripts/analyze_crystal_benchmark.py` → size-confound controls on MM-GBSA.
- `scripts/eval_per_family.py` → per-family LOO (shows the inflated number).
- `scripts/refit_per_family_clean.py` → true held-out (shows it doesn't hold).
