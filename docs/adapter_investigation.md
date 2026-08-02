# Frozen-base adapter — full investigation: weights, every training stat, the pattern, next steps

*2026-08-02. Deep dive requested after the adapter's first N=48 checks looked like a win and
raised the hypothesis: "maybe they just needed time to coordinate — they spiked, then calmed."
This adjudicates that with the actual weights and stats. Honest verdict up front, evidence below.*

---

## 0. TL;DR (read this first — it corrects some early excitement)

1. **A real, robust win exists: β-sheet.** Better than pretrained on **all 7 epoch-checks and
   both repeat probes**, by **~0.8 Å** (5.52 → ~4.7). Stable, reproducible → real.
2. **The "beating pretrained on vlong/hard classes" was mostly NOISE.** The *same* ep81
   checkpoint, probed twice at N=48, gave **vlong 7.66 vs 10.28 — a 2.6 Å swing.** The vlong
   "wins" (ep58, ep81) were lucky draws; averaged, vlong is ~tied/slightly worse.
3. **But the method is VALIDATED.** The frozen-base adapter did exactly what the forgetting
   diagnosis predicted: **base drift = 0.000000 throughout** (can't forget), short/med/long
   held at pretrained (no regression), and it **converged smoothly instead of collapsing** —
   the opposite of full fine-tuning. This is the first retrain approach that doesn't degrade.
4. **Your "coordination" intuition is right on mechanism, partly-noise on the docking read.**
   The adapter's weights *did* grow-then-settle (converge). The apparent "calm down" in docking
   is real convergence PLUS N=48 probe noise, not a dramatic self-correction.

**Net:** not the sweeping win the first reads suggested, but a genuine β-sheet improvement with
zero regression, and proof that the frozen-base adapter is the correct, non-collapsing method.

---

## 1. The discovery that reframes everything: N=48 is NOISY on hard classes

The single most important measurement. I probed the **identical ep81 checkpoint twice** at N=48:

| class | probe 1 | probe 2 | swing | pretrained |
|---|---|---|---|---|
| short | 1.26 | 1.99 | 0.73 | 1.33 |
| med | 2.38 | 2.67 | 0.29 | 2.82 |
| long | 4.44 | 4.45 | 0.01 | 4.30 |
| **vlong** | **7.66** | **10.28** | **2.62** | 8.38 |
| β-sheet | 4.61 | 4.78 | 0.17 | 5.52 |

Same weights. Same code. The only difference is the random seed of the 48 diffusion samples.
**vlong (28-mer) swings 2.6 Å between identical runs.** This means:
- Any *single* N=48 read of vlong is worth ±1.5 Å at least. A "−0.86 win" and a "+1.77 drift"
  can be the same model.
- **The stable classes (long, β-sheet) barely move (0.01, 0.17)** → their signal is
  trustworthy. β-sheet −0.8 is real *because* it's stable across probes AND epochs.
- **Every prior "vlong" excitement in this project** (including the full-FT collapse reads)
  carried this noise. It's why we mandated N≥24 — but even N=48 isn't enough for a 28-mer.

**Consequence for the verdict:** trust β-sheet (stable, repeated). Distrust single vlong reads.
The N=96 probe running now, and multi-complex eval (§6), are how we get a real vlong number.

---

## 2. Complete weight changes — what actually moved

Measured across saved checkpoints (ep3→131) vs pretrained base:

| epoch | adapter ‖W‖ | growth since prev | base drift (must be 0) |
|---|---|---|---|
| 3 | 6.4066 | — | 0.000000 |
| 14 | 6.5219 | +0.115 | 0.000000 |
| 36 | 6.5881 | +0.066 | 0.000000 |
| 58 | 6.6318 | +0.044 | 0.000000 |
| 81 | 6.6726 | +0.041 | 0.000000 |
| 103 | 6.7050 | +0.032 | 0.000000 |
| 125 | 6.7253 | +0.020 | 0.000000 |
| 131 | 6.7282 | +0.003 | 0.000000 |

Two facts:
- **The base NEVER moved (drift = 0 to six decimals).** The 7.55 M pretrained parameters are
  bit-identical. Only the 24,444 adapter params (0.32%) changed. This is why short/med/long
  can't catastrophically degrade — the machinery that produces them is untouched. Contrast the
  full-FT runs, where the *whole* model drifted ~1.5–2% and everything degraded.
- **The adapter grew, then decelerated to an asymptote (~6.73).** Growth per-epoch:
  0.115 → 0.066 → 0.044 → 0.041 → 0.032 → 0.020 → 0.003. It is *converging*, not exploding
  (contrast the collapse run's BN buffers, which grew 30×). From ep58 on, the model changes
  <2% total — ep81 and ep131 are nearly the same model. **This is the "coordination" you saw:
  a small correction settling into a stable value.**

The flip side: because ep81≈ep131 in weights, their *docking differences are almost entirely
probe noise* (§1), not real drift. The late "drift"-tagged checks were noise, not degradation.

---

## 3. Every training stat, in plain terms — tied to THIS run

| stat | this run | what it means | why it looks like this |
|---|---|---|---|
| **epoch** | 132 / 200 | one full pass over 3,432 training complexes | ~5–7 min each; ~13 h in |
| **learning rate** | 1e-5 → 2.9e-6 (cosine) | size of each weight nudge | decays on a cosine curve; deliberately small |
| **train loss** | swings 280 → **4 billion** | model's self-graded error on training batches | the billions are a **cosmetic artifact**: a few samples have huge score targets (score ∝ 1/σ) that spike the *mean*; a safety clamp (grad-clip) stops them harming the weights. Ignore the mean; it's not real dynamics. |
| **val loss (median)** | **flat ~3.2–3.5 the whole run** | score-matching error on held-out complexes | **flat because the base is frozen** — the frozen 99.7% of the model dominates this number, and it can't change. So here **val loss is a near-useless signal**; only docking RMSD tells us anything. (In full-FT it dropped to 0.77 while docking *worsened* — proof it's a bad proxy.) |
| **adapter ‖W‖** | 6.41 → 6.73, decelerating | how big a correction the adapter has learned | **this is the real learning signal** for a frozen-base run. Growing+settling = converging. |
| **base drift** | 0.000000 | how much the pretrained core moved | zero = the forgetting-proof guarantee is holding |
| **docking RMSD (N=48)** | β-sheet −0.8 (stable); vlong ±2.6 (noisy) | the actual goal: Å from the native pose | best-of-48 sampled poses; **noisy on long peptides** (§1) — needs repeats/higher-N |
| **per-term norms** (tr/rot/tor) | tr~4–9, rot~6–18, tor~150–600 | magnitude of each predicted score component | torsion targets are intrinsically large; the σ-normalized loss balances their *contribution* despite the raw-norm gap |
| **n_ok** | 3399 / 3432 | samples that trained OK per epoch | the 33 fails are pre-existing malformed data (same in every run), not a bug |

**The single most useful mental model:** in a *frozen-base adapter* run, ignore val loss (it's
pinned by the frozen base), watch **adapter ‖W‖** (is it converging?) and **docking RMSD on the
STABLE classes** (β-sheet/long), and treat single vlong reads as ±1.5–2.5 Å noise.

---

## 4. The emerging pattern

Putting three campaigns side by side:

| approach | base drift | what happened to docking | verdict |
|---|---|---|---|
| **full-FT (B1, BN-fix)** | ~2% | ALL classes degrade monotonically; vlong→12.5 | catastrophic forgetting |
| **full-FT + x0 (B2)** | ~1.5% | same, slightly milder | forgetting, not fixed by x0 |
| **frozen-base adapter** | **0.000000** | short/med/long held; **β-sheet −0.8 real**; vlong noisy-tie | **converges, no regression, one real gain** |

The pattern is now coherent and matches the literature (JCIM DiffDock-L; LoRA-vs-full-FT):
**freezing the base converts "fine-tuning destroys the model" into "fine-tuning safely adds a
small, real, targeted improvement."** The improvement we can *prove* is β-sheet (the class most
under-represented in RAPiDock's pretraining, so the class with the most room to gain). The
frozen base is why we get that gain without paying for it elsewhere.

---

## 5. Your "coordination" hypothesis — adjudicated

> "Maybe they just needed time to coordinate; they spiked, then calmed down."

- **TRUE (mechanism):** the adapter's weights grew and settled to an asymptote (§2) — it
  *did* self-organize into a stable correction, and unlike full-FT it did **not** collapse.
  The frozen base gave it a stable substrate to coordinate against. This is a real and
  important dynamic, and it's *why* this approach works where others failed.
- **PARTLY NOISE (the docking read):** the docking metric "calming down" is mostly (a) the
  weights converging + (b) N=48 probe noise (§1) — not a dramatic self-correction from bad to
  good. The model didn't "recover"; it converged to a state that is genuinely better on
  β-sheet and genuinely tied elsewhere, and the epoch-to-epoch wobble is the probe, not the model.
- **The train-loss "spikes"** you saw are cosmetic outlier samples (clipped), not the model
  destabilizing then healing.

So: right that they coordinated and stabilized (that's the win of the method); the "spiked then
calmed to a breakthrough" reading over-reads probe noise. The honest version is quieter but
still good: **a stable, converged adapter that safely improves β-sheet.**

---

## 6. Ways forward (ranked, evidence-driven)

1. **Get a TRUE, low-noise verdict** (in progress). Single N=48 is ±2.6 Å on vlong — useless
   for a go/no-go. Do: (a) N=96 on the converged checkpoint [running now]; (b) build a
   **multi-complex** probe set — ≥5 complexes *per* length class instead of 1 — and average.
   Only then can we say whether the adapter beats pretrained on vlong or just on β-sheet.
2. **Ship the β-sheet gain if it holds under (1).** A Stage-1 that beats pretrained by ~0.8 Å
   on β-sheets with **zero regression** is a legitimate, defensible improvement — β-sheets are
   RAPiDock's known weak spot (≈0% in its pretraining). Select the checkpoint on the *stable*
   β-sheet signal, not val loss.
3. **Push placement harder, still frozen-base: adapter + x0 term.** The adapter has exactly the
   capacity (translation) that the x0 COM term supervises, and the base stays protected. This
   is the natural "can we turn the β-sheet-only win into a long/vlong win too" experiment —
   low risk (frozen base), and we already have the x0 code.
4. **Stop wasting epochs.** The adapter converged by ~ep58 (weights change <2% after). Running
   to ep200 mostly adds probe-noise datapoints. Either stop and select, or switch the GPU to
   the adapter+x0 experiment (#3).
5. **Ensemble (Track-1B) remains the product-level play** — orthogonal to all of the above and
   still the highest-EV move for the *tool*. The adapter's β-sheet gain can feed *into* the
   ensemble.

**Bottom line:** the frozen-base adapter is the validated, non-collapsing method the diagnosis
called for. It yields one proven gain (β-sheet, ~0.8 Å, no regression) and — pending the
low-noise eval — possibly more. My earlier "architecture-limited, give up" stays retracted:
the model *is* safely improvable; it just improves *modestly and specifically*, not sweepingly.

*Evidence: /tmp/adapter_ep81_confirm (repeat-probe noise); adapter checkpoint weight-norm
series (base drift 0, adapter asymptote); logs/adapter_lowlr.log (flat val, cosmetic train
spikes); logs/adapter_2h_checks.log (7 epoch reads). See also
[[rapidock_finetune_failure_analysis]] (why full-FT failed) and [[project_bn_rootcause_aug1]].*
