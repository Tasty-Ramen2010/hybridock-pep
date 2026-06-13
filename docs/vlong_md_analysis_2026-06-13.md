# Very-long peptides (≥17): floating? Would MD fix it? — full analysis

**Date:** 2026-06-13 · **Script:** `e99_vlong_investigation.py` · **Cache:** `runs/e99_cache.json`
**Question (Ram):** vlong real-pose r=−0.515. Are the poses floating? Would 60 ps MD relaxation re-seat
them and let us score them? How do we beat PPI-Affinity (0.554)?

## Bottom line (rebuts the floating hypothesis AND my own prior "MD may help")
1. **The poses are NOT floating.** vlong interface coverage real=0.78 = crystal=0.78 (Δ=0.00), mean
   peptide→receptor separation 3.0 Å, rg real 9.6 < crystal 10.1 (real is *more* compact, not extended).
2. **MD relaxation cannot fix vlong, by construction.** The endpoint of relaxation is the bound crystal
   structure — and **crystal-pose vlong r is only +0.103** (essentially unpredictable). MD can push real
   poses *toward* crystal; it cannot exceed crystal. The ceiling is the model/features, not the poses.
3. **The vlong negative is real but a red herring for deployment.** Removing vlong *lowers* the pooled r
   (0.478→0.361) — they anchor the weak-affinity end and help cross-band ranking. Don't route them out.
4. **The real lever to beat PPI-Affinity is the med band + a conformational-entropy term**, not MD re-seating.

## A. DECISIVE — LOO r by length band: crystal (oracle) vs real
| band | n | crystal r | real top-5 r | real ML-5 r |
|---|---|---|---|---|
| med 9–12 | 40 | +0.352 | +0.159 | +0.289 |
| long 13–16 | 10 | +0.489 | +0.365 | **+0.487** |
| **vlong ≥17** | 15 | **+0.103** | −0.515 | −0.108 |

- **long 13–16 is FULLY RECOVERED** (real ML-5 0.487 = crystal 0.489) — no pose problem at all.
- **med 9–12 has a recoverable gap** (0.289 vs 0.352 ceiling = 0.063) — this is the genuine pose-denoising
  opportunity, and it's the *largest* band.
- **vlong ≥17 is capped at crystal 0.10** — even perfect poses can't predict it. The real-pose −0.515 is
  poses making a near-zero signal *worse*, but the ceiling is ~0.10 regardless.

> **This single table refutes the MD plan.** MD's best case = real→crystal. Crystal vlong = 0.10. So MD's
> entire upside on vlong is −0.515 → +0.10 (stop hurting), never → predictive. The limit is the model.

## B. Floating metrics — Ram's hypothesis, tested directly
| band | interface-cov crystal→real | mean sep | rg crystal→real |
|---|---|---|---|
| med 9–12 | 0.79 → 0.78 (Δ−0.01) | 3.2 Å | 5.4 → 6.0 (+0.6) |
| long 13–16 | 0.84 → 0.75 (Δ−0.08) | 3.3 Å | 7.1 → 7.2 (+0.1) |
| **vlong ≥17** | **0.78 → 0.78 (Δ0.00)** | **3.0 Å** | 10.1 → 9.6 (−0.5) |

vlong poses are **well-seated, not floating** — coverage matches crystal exactly, tight contact (3.0 Å),
slightly more compact. If any band floats it's **long 13–16** (Δcov −0.08), and that band scores *fine*.
Note: even crystal vlong coverage is only 0.78 — long peptides genuinely have dangling segments in the
*real* bound structure. Those segments are physical, not docking artifacts.

## C. Within-vlong correlations (n=15, real top-5)
`sasa_sb` r(feat,y)=**−0.64** (salt-bridge interface → stronger; the one real signal), then weak/mixed:
`strength_bur` +0.30, `bsa_hyd` +0.29, `org_density` −0.24. `net_charge` r=−0.28, `abs_charge_frac` +0.09.
→ vlong affinity is carried by **salt bridges / electrostatics** that the med-dominated pooled model
under-weights — a feature/regime problem, consistent with the charged floor, not a pose problem.

## D. Significance — is −0.515 real or n=15 noise?
Bootstrap 2000×: r=−0.515, 95% CI **[−0.89, −0.15]**, P(r<0)=1.00, over a compressed y-range of 3.6 kcal.
→ The negative is statistically real (not noise), but lives in a narrow affinity window.

## E. Lever — how vlong affects the pooled number (counterintuitive)
| | r (n=65) | excl. vlong (n=50) |
|---|---|---|
| real top-5 | +0.372 | +0.242 |
| real ML-5 | +0.478 | **+0.361** |

**Removing vlong HURTS the pooled r.** They anchor the weak-affinity end; the model gets the coarse
strong-vs-weak ranking right across bands even while being wrong *within* vlong (Simpson's paradox in
reverse). So vlong is not the blocker — the overall ceiling is.

## Hypotheses, rebuttals, counterarguments
- **H1 Floating (Ram).** *Rebutted by B* — coverage matches crystal, poses seated. *Counter:* long 13–16
  floats more yet scores fine → floating ≠ the failure.
- **H2 Over-extended diffusion poses.** *Rebutted* — real rg < crystal rg for vlong (more compact).
- **H3 n=15 noise.** *Partly rebutted by D* — CI excludes 0; the negative is real, but range-compressed.
- **H4 Charged floor / model ceiling.** *Supported by A + C* — crystal vlong only 0.10; signal is
  salt-bridge/electrostatic, under-weighted by the pooled model. This is the documented charged floor:
  static single-pose electrostatics wash; needs FEP or an entropy/solvation term.
- **My own prior claim "MD may help vlong":** *Rebutted by A.* MD's endpoint is crystal; crystal=0.10.
  Withdrawn.

## So would MD ever help — and how do we beat PPI-Affinity (0.554)?
- **Pose-relaxation MD (re-seat): NO.** Endpoint = crystal = the ceiling we already measured.
- **MD for CONFORMATIONAL ENTROPY (s_free): the one legitimate MD direction.** Long flexible peptides
  pay a large binding-entropy penalty a *single static pose* (crystal or real) cannot see. This is the
  documented free-state-entropy lever (real MD −TΔS, +0.08 pooled). It's a *feature addition*, not pose
  relaxation, and it targets exactly the long/flexible regime. This is the physically-correct path for
  vlong — but it lifts the *ceiling*, which static poses cap at 0.10.
- **Realistic path to 0.554:**
  1. Confirm **ML-best-5** (0.478) leak-clean (the98-trained ranker). Biggest single lever, mostly banked.
  2. Close the **med 9–12** gap (0.289→0.352): drop high-CV features (`poc_net` CV 9.6, `poc_eis` 3.8),
     denoise via ML-best-5. Largest band, genuine pose-recoverable signal.
  3. Add a **salt-bridge / charge-aware term** for the long regime (sasa_sb r=−0.64 within vlong is unused
     signal) and/or the **s_free entropy** term — the only physics that raises the vlong *ceiling*.
  4. **Leave vlong in** (routing it out hurts the pooled r).
- **MD is not the unlock.** The crystal ceilings (0.35 / 0.49 / 0.10 by band) say the limit is features
  and the charged floor, which MD-relaxation cannot move. Spend the compute on the entropy term, not 60 ps
  re-seating of poses that aren't floating.
