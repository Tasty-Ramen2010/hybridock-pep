# Finding b(R1) without same-receptor crossing — brainstorm, rebuttals, counterarguments

**For Ram, 2026-06-17.** Goal: estimate the receptor offset `b(R1)` for a query receptor — to do absolute
scoring — without the model-cross-corner substitution that fails (e289). Ranked, each rebutted + countered.

---

## The identifiability theorem (the spine of every answer)

`b(R1)` is **one unknown number per receptor**. For each peptide i on R1 the scorer gives:
```
S(Pi, R1) = G(Pi, R1) + b(R1) + c(Pi) + η
```
`b(R1)` appears **only in terms involving R1**. So:
- With **N≥1 measured Kd on R1**: N equations, 1 unknown → b(R1) is over-determined → solvable (anchoring).
- With **0 measured on R1**: b(R1) is **unidentifiable** — complexes on R2,R3,R4… contain b(R2),b(R3),b(R4),
  giving *zero* equations that involve b(R1).

This is not a modeling weakness; it's information theory. **Confirmed empirically:** matrix factorization
(the method built for exactly this) gives 0 receptor calibration on a cold-start receptor (e290). Every
idea below either supplies an R1-equation (works) or tries to dodge the theorem (fails).

---

## Ideas that SUPPLY an R1-equation (these work)

**B1. Collaborative filtering / matrix factorization — TESTED, small win.** Treat the peptide×receptor Kd
matrix as `Y[p,r]=mu+bias_p+bias_r+<u_p,v_r>`; `bias_r` *is* b(R). e290 warm-start: ANCHOR 0.653 →
**MF_bias 0.680** (+0.027) — joint peptide+receptor bias regularizes better than the mean-residual anchor.
Rank-2 latent adds nothing (peptides too sparse to embed). *Rebuttal:* only helps when R1 has ≥1 cell
(warm). *Counter:* that's the theorem — and within it, MF is the better estimator. **ADOPT MF-bias to
replace the simple anchor.**

**B2. Probe-peptide fingerprint (I2) / same-receptor anchoring.** Measure 2–3 peptides on R1; b̂(R1)=mean(y−S).
Validated r 0.52→0.61. *Rebuttal:* needs wet-lab on R1. *Counter:* it's 2–3 cheap measurements and it's the
iGEM mode-(b) plan. The theorem says you can't do better with less.

**B3. Bridge-peptide / DiffNet network.** A peptide measured on R1 *and* other receptors ties b(R1) into a
network of relative offsets solved by MLE (Xu 2019). *Rebuttal:* still needs the shared peptide *on R1*.
*Counter:* one shared "standard" peptide across a panel calibrates the whole network at once — efficient
for selectivity panels.

**B4. One FEP/TI calc per receptor (the non-experimental escape).** FEP gives the true `G(P0,R1)` for one
reference peptide → `b(R1) = S(P0,R1) − G(P0,R1)` directly. This injects the R1-equation by *computation*
instead of experiment. *Rebuttal:* one FEP is expensive (hours-GPU). *Counter:* **one** calc per receptor,
amortized over the whole library — cheap *per prediction*, and it's the only way to get b(R1) with zero
wet-lab. **This is the real "hybrid FEP" — FEP calibrates the offset, ML interpolates everything else.**

---

## Ideas that DODGE the theorem (these fail or are confounded)

**D1. Cross-receptor transfer by similarity (sequence/pocket/3D), any K, averaged.** *Rebuttal:* b(R) doesn't
transfer (best corr +0.084), and averaging gives the *population-mean* offset not b(R1) (bias, not noise).
Strict 90% gate makes it *worse* (e288: closest-5% absolute 0.616 → transfer −0.110). **DEAD (e266–e289).**

**D2. Learn b(R) from receptor/pocket features (GBT/Ridge/ESM).** *Rebuttal:* b(R)=E[G−S|R] is the model's
*own residual*, orthogonal to features it already used → unlearnable (e276 r≈0 < predict-mean). *Counter:*
a feature *truly orthogonal* to S could work — but the only orthogonal signal that correlates with b(R) is
explicit free-energy physics = B4 (FEP), not a cheap feature. **DEAD for static features.**

**D3. Unlabeled score-distribution calibration (novel).** Dock a fixed diverse library to R1; if the mean
*true* affinity of random peptides were receptor-independent, then `mean_P S(P,R1) − const ≈ b(R1)` — no
labels needed. *Rebuttal:* mean true affinity is NOT receptor-independent (deep hydrophobic pockets bind
random peptides better than shallow polar ones) → confounded with real signal S already captures. *Counter:*
use the *residual* mean after feature-prediction — but that's ~0 by construction (the model centered it).
**Likely confounded; only worth a quick test, low odds.**

**D4. Functional-family prior (EC class / fold).** Maybe b(R) clusters by *function* even if not by sequence.
*Rebuttal:* sequence/pocket families already failed (+0.084); function is coarser. *Counter:* untested with
real functional annotation — but the prior would be weak (family-mean offset, std still ~2 kcal/mol within
family). Low priority.

**D5. Active learning — which ONE peptide to measure on R1.** *Rebuttal:* still needs 1 measurement (it's
B2 optimized). *Counter:* picks the *most informative* probe (typical, high-leverage) so 1 measurement pins
b(R1) best — a real efficiency win on top of B2. **Worth building once B2/B1 are wired.**

---

## Verdict (the honest, complete answer)

You **cannot** find `b(R1)` from 3–5 complexes on *other* receptors — the theorem forbids it, and every
similarity/learning/averaging route has been exhausted (e266–e290). `b(R1)` is identifiable **only** with
≥1 equation involving R1. The cheapest such equations, ranked:

1. **Measure 1–3 Kd on R1** → anchor/MF (r≈0.68). *Wet-lab, cheapest.* ← iGEM mode-(b)
2. **Compute 1 FEP on R1** → b(R1) directly. *No wet-lab, GPU-hours, amortized.* ← the "hybrid FEP"
3. **A shared standard peptide across a panel** → DiffNet network calibration. *For selectivity panels.*

Everything else is the wall. The constructive next steps: **(a) replace the anchor with MF-bias (+0.027,
e290), (b) build the 1-FEP-per-receptor hybrid as the no-wet-lab path, (c) active-select the probe peptide.**
The dream of pure-compute absolute scoring on a novel receptor is information-theoretically closed — but
"1 cheap equation per receptor + ML for the rest" is a genuinely strong, deployable answer.
