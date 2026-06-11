# E18 v2 — structure-based entropy + bond-strength SASA: VERDICT

**Date:** 2026-06-10. Built to Ram's spec (replace sequence-only entropy with real 3D
MD entropy; W_bound no longer ≡1; add favorability-weighted "bond-strength" SASA with a
penalty for buried-unsatisfied polar/charge). Prior session built all of it
(`e18v2_md.py`, `e18v2_features.py`) but died 4/65 complexes into the run; never evaluated.

This doc evaluates the **instant** half (`bond_strength_sasa`, GPU-free) through the same
honest harness as v1 (`e18v3_instant_eval.py`). The MD-entropy half was confirmed to run
on a single complex but NOT run at scale (see "Why we stopped").

## Result: NO (same wall as v1)

**Cross-dataset transfer (fit crystal-65 → predict PEPBI, and reverse), Pearson r:**

| model | cr→pb | pb→cr | RMSE |
|---|---|---|---|
| baseline hb+aromatic | **−0.538** | −0.510 | 3.27 |
| de_strength alone | +0.035 | +0.249 | 3.16 |
| de_strength + clash | −0.003 | −0.046 | 3.17 |
| baseline + de_strength | −0.497 | −0.454 | 3.46 |
| ALL (str+clash+sasa+base) | −0.426 | −0.419 | 3.40 |

**Within-target leave-binding-group-out on PEPBI (the regime that matters), pooled r:**

| model | pooled r | %correct | n_grp |
|---|---|---|---|
| baseline hb+aromatic | **+0.453** | 50% | 17 |
| de_strength alone | −0.300 | 40% | 17 |
| de_strength + clash | −0.152 | 44% | 17 |
| baseline + de_strength | +0.424 | 56% | 17 |
| ALL | +0.417 | 53% | 17 |

## Honest reading (two findings, one of them genuinely useful)

1. **Within-target, the bond-strength SASA HURTS.** Bare hb+aromatic = +0.453;
   adding de_strength drops it to +0.424; de_strength alone is *negative* (−0.300).
   The favorability multipliers (−0.3 H-bond, −0.6 salt bridge, +1.0 buried-unsat-polar)
   are **fixed hand weights that miscall borderline contacts** and inject more
   per-residue noise than the bare contact counts. "Change the SASA value by binding
   strength" is a reasonable idea, but a static rule-based favorability is noisier than
   just counting satisfied contacts.

2. **de_strength is the ONLY term that does not sign-flip across datasets** (+0.035 /
   +0.249 vs everything else at −0.4 to −0.5). It is *sign-stable but signal-less* —
   near-zero magnitude. It resists the confound without capturing the affinity.

## Where the error actually is (Ram's question: "where so much error to close the wall")

The data localizes the wall precisely. **The same feature ranks correctly WITHIN a
binding group (+0.45) and BACKWARDS across families (−0.54).** That sign flip IS the
wall — it is the per-protein-baseline / size (Simpson) confound, not a missing term:

  ΔG_abs ≈ (interface-size term, confounded) + (per-protein baseline constant) + (small specificity signal)

Our structure features estimate the size term well; the per-protein baseline constant is
**not recoverable from a single static structure**. Adding more static structure terms
(3D entropy, SS refinement, bond-strength SASA) adds more *size-correlated* signal — it
cannot add the missing per-system constant, so it cannot un-flip the sign. That is why
none of items 1–4 move the cross-family number, and why even a perfect entropy term won't.

**The three things that actually close it** (none are a static per-pose score, per
`honest_competitive_assessment.md`):
- per-system normalization (NetMHCpan percentile-rank) — needs many peptides per target;
- stay within-target / predict ΔΔG — where we already sit at r≈0.45 (shippable);
- a non-size info source (co-evolution / MSA, as AlphaFold/Boltz-2 affinity use).

## On the "2A" idea (RAPiDock pre-binding fold as the SS/free reference)

Legitimate *fidelity* improvement to the entropy term (a real pre-folded unbound
reference beats both "fully unfolded sequence" and "crystal-peptide-alone"), and it is
buildable. But it does **not** touch the wall: the wall is the missing per-protein
baseline, not a wrong free-state reference. It would make ΔS_conf more physical without
making cross-family absolute ΔG more correct. Worth doing only if we commit to the
within-target ΔΔG product, where a cleaner ΔS_conf could sharpen ranking.

## Why we stopped before the full MD run

The instant half (cheapest, same family as the features that DO replicate) is the most
likely to help and it does not. The MD-entropy half is (a) ~3 h of GPU for 65+ complexes,
(b) on the wrong side of `af5cf507f` (250 ps MD-LIE already loses to instant geometry
within-target), and (c) still size/baseline-confounded. Spending 3 h to reconfirm NO at
high cost is not warranted. NOT promoted to `src/`. Production scorer unchanged.
