# Pose Ranker — Validation & Honest Numbers (2026-06-09)

Independent re-validation of the ranking stack before committing it as the
tool's headline ranker. All numbers are **leave-one-complex-out CV** on the
bench300 subset with aligned features (n=112 complexes, 5 poses each), metric =
mean per-complex Kendall τ between score and interface-RMSD.

## The numbers (independently reproduced under clean LOO)

| Ranker | LOO τ | Notes |
|---|---|---|
| Vina (raw) | ~0.02 | (historical) |
| **ref2015 physics-16** | **0.175** | leakage-free, reproduces the established 0.176 |
| encoder (96-dim) alone | 0.123 | learned diffusion-encoder embeddings |
| **ref2015 + encoder z-blend** | **0.211** | best honest reproduction (w_phys≈0.5) |
| (reported RankerV2 / combined14_CLIP) | 0.236–0.242 | optimistic; see caveat 1 |
| project shipped figure | 0.212 ±0.035 | with error bar |

**Headline, stated honestly: the best ranker is ref2015 + encoder, τ ≈ 0.21
(reports up to 0.24), vs ref2015-alone 0.175 — a real but modest +0.035.**

## Validation issues — read before quoting 0.242

1. **0.242 is optimistic; honest LOO is ~0.21.** The 0.242 used a trained NN
   head on the encoder stream plus a blend weight tuned on the eval data. Under
   a plain linear head + untuned blend (this re-check) it is 0.211. The two are
   within one error bar — but **quote ~0.21, not 0.242.**

2. **Large error bar (±0.035).** Only 5 poses/complex and 112 complexes, so τ is
   noisy. 0.21 and 0.24 are statistically indistinguishable. Do not present
   sub-0.04 differences as real.

3. **Encoder leakage — CONFIRMED, not just a risk.** Checked directly:
   **100% (112/112) of the bench300 complexes are in the PepPC training set.**
   The diffusion model that produces the 96-dim encoder features was trained on
   every structure it is being asked to rank. So the encoder's +0.035 gain over
   ref2015 is at least partly memorization of the native pose, and the
   **prospective gain on novel targets is unknown and likely smaller — possibly
   zero.** ref2015's 0.175 is leakage-free (a fixed physics function, no
   training); the encoder ensemble's edge is not. For a general-purpose tool
   scoring novel peptides/targets, **do not assume the ensemble beats ref2015.**

4. **Modest absolute correlation.** τ≈0.21 is a weak-to-moderate rank
   correlation. It means meaningfully better-than-random pose ordering, not
   near-perfect ranking. Frame accordingly.

5. **Subset coverage.** 112/240 bench300 complexes had aligned phys+encoder
   features; the rest were filtered. The number is on that subset.

## What is and isn't defensible to commit

- **Bulletproof (use this):** ref2015 ranking, τ=0.175, leakage-free. This is
  the always-defensible headline and the right number for a general-purpose tool
  on novel targets.
- **Benchmark-only:** ref2015 + encoder ensemble, τ≈0.21 on bench300 — valid to
  report for this benchmark ONLY if disclosed that the encoder was trained on
  100% of the test complexes. Its novel-target value is unproven.
- **Not defensible:** quoting 0.242 as clean held-out (it's ~0.21 honest), or
  presenting the encoder ensemble as the general-tool ranking number without the
  100%-training-overlap disclosure.

## Cross-checks done this session (why hand-built scorers were rejected)

- FoldX-style geometric ranker (clash+saltbridge+desolv+rep): τ≈0.10 alone;
  adds +0.007 (noise, unstable picks under nested CV) to ref2015. Rejected.
- Knowledge-based contact potential: r=-0.015 on affinity after size control.
- MM-GBSA / per-family / single-ridge for affinity: all the size confound.

Conclusion: the encoder is the **only** signal that genuinely adds to ref2015,
and even it is modest (+0.035) and carries a leakage caveat to verify.
Reproduce with `scripts/foldx_ranker_v2.py` (physics) + the LOO block in this
session's notes.
