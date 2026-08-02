# Why Arm A (full-param fine-tune) collapsed — a post-mortem

*2026-07-31. Grounded in our own logs, our prior experiments, and the diffusion/
docking literature. Written to support AND contradict my own earlier claims.*

## 1. The failure signature (what actually happened)

- Train loss fell cleanly **1589 → 168** (ep1→27), then ticked **up** at ep28 (173).
- Docking on the 6 held-out probes **collapsed by ep28**, across *every* class —
  including short 5-mers RAPiDock never struggled with:
  short 2.06→~8.0 (+6), med 4.36→8.6, long 5.12→8.2, vlong 9.42→13.4, sheet 5.23→10.1.
  Confirmed by an independent re-probe (short 7.57 and 8.02 on two runs).
- Weight drift was **tiny and smooth** the whole time: L2 40.7→46.6, ~0.06%/epoch,
  never accelerating. The guard never tripped.
- Validation translation norm spiked to **max 1654–2036** at low σ while the *mean*
  stayed 3–13.

Key tension: **a 2%-of-magnitude weight change produced a total behavioral collapse.**
That rules out "the weights wandered far away." Something amplified a tiny change.

## 2. Candidate mechanisms

### M1 — Objective mismatch: the denoising loss doesn't reward placement  ✅ primary
The training loss is a score/denoising objective, not "put the peptide in the pocket."
**Literature:** better likelihood/NELBO does **not** track sample quality in diffusion
models — a weighted loss can beat NELBO on FID while being a *worse* likelihood
([Kingma-style loss-weighting analyses], loss-function comparative study). So lowering
our loss is not evidence of better docking. **Our data:** loss ↓ 89% while docking
collapsed — the textbook decoupling. **This is the same "val loss lies" lesson we've
hit repeatedly** (we already select checkpoints on docking, not val loss, for exactly
this reason). *Verdict: strongly supported.*

### M2 — Score explosion near t≈0 amplifies placement-head perturbations  ✅ primary
Score models predict `s(x,t) = f(x)/σ_t`; as σ_t→0 the score **explodes**. This is a
documented training instability; the standard fixes are **capping the 1/σ factor** and
**large batches / grad accumulation** to denoise the low-σ loss. **RAPiDock does exactly
this division** — `tr_pred = tr_pred / tr_sigma` — so any error the fine-tune introduces
into the translation head gets **divided by a tiny σ** and blown up. **Our data is the
smoking gun:** val translation norm max hit **1654–2036** at low σ (mean stayed ~5).
Fine-tuning nudged the magnitude MLP; at low σ that nudge became a huge bodily
translation → the whole pose distribution slid off the pocket (the uniform +3–6 Å block
shift we measured). *Verdict: strongly supported, and it explains the "tiny weight
change → giant output change" paradox that plain forgetting can't.*

### M3 — The placement output is a single low-dim, high-leverage vector  ✅ supporting
Translation = one global 3-vector (+ scalar magnitude); conformation is spread over
hundreds of per-atom outputs. Low-dimensional outputs are fragile: one bad direction
moves the entire peptide. **Our data:** the pose distribution shifted as a rigid block
(placement), not scrambled (conformation). *Verdict: supported; it's the structural
reason M2's amplification lands on placement specifically.* This is largely our own
diagnosis, less external citation — I hold it a notch below M1/M2.

### M4 — Train/eval distribution shift (Propedia vs the probes)  ⚠️ under-weighted, real
I dismissed this earlier; the literature makes me put it back on the table. **DiffDock-L
"excels only on proteins represented in the training set" — deep docking models overfit
to families, success dictated by similarity to training structures.** We trained on
**Propedia with a 20 Å pocket truncation** and probe on **RecentSet + 2 Propedia**. If
Propedia's pocket/placement convention differs from RecentSet's, the model correctly
learns *Propedia* placement and looks worse on RecentSet probes. **Counter-evidence:**
short 5-mers collapsed too, and short binding modes are similar across DBs — pure
distribution shift shouldn't nuke short as hard as it did. So M4 is a **real contributor
but not sufficient alone.** *Verdict: partial; I was wrong to dismiss it, wrong to make
it the whole story. Falsifiable — see §4.*

### M5 — Small-data catastrophic forgetting / representation collapse  ⚠️ partial
3432 training complexes is small; small-data multi-epoch fine-tuning gives noisy
gradients, high variance, and representation collapse; **even LoRA can catastrophically
degrade without regularized replay (KL to the initial model).** **But** our drift was
only 2% and *smooth* — this isn't classic wholesale forgetting, it's a sharp
amplified-output failure (M2). *Verdict: contributory (noise + no KL-to-init anchor),
not the lead actor.*

### M6 — LR / schedule / resume confounders  ⚠️ can't fully exclude
LR peaked at 1e-6 (1000× below pretraining's 1e-3) and **still** diverged — which argues
the *direction* is wrong (M1), not just the step size. **But** we never tried 1e-7/1e-8,
and each resume reloads EMA weights (possible optimizer/EMA-state discontinuity at the
ep26 restart). *Verdict: unlikely to be primary, but not cleanly ruled out.*

## 3. Synthesis — the most defensible story

**M1 + M2 are the engine; M3 is why it hits placement; M4/M5/M6 are accelerants.**
The denoising loss doesn't reward placement (M1), so gradient descent freely perturbs the
translation head in whatever direction lowers the loss. Because that head's output is
divided by σ (M2) and is a single low-dim vector (M3), even a 2% weight change gets
amplified near t≈0 into a large rigid mis-translation of the whole peptide. Training on a
possibly-shifted distribution (M4) and small-data noise with no KL-to-init anchor (M5)
make it worse and faster. The result: loss down, placement destroyed — first mildly
(ep24), then catastrophically once the amplification compounds (ep28).

This is consistent with **all three of our prior fine-tune failures** (last-layer →
−0.14 Å ceiling; heads @5e-5 → norm→142 divergence; full-param @1e-6 → this) and with
the N-sweep finding that **the *pretrained* model is genuinely good** (7bh8 ~2 Å at N=4)
— i.e., we keep breaking a good model by perturbing a fragile, amplified head.

## 4. Where I might be wrong — falsification tests

- **If M4 dominates:** probe the ep28 checkpoint on **held-out Propedia** complexes. If
  it's *fine* on Propedia and only bad on RecentSet, it's distribution shift, not
  head-destruction — and the "collapse" is partly an eval artifact. *(Do this — it's cheap.)*
- **If M6 (LR) dominates:** a run at **lr 1e-7 with warmup 20** should degrade far slower.
  If it doesn't diverge, "full-param can't work" is false.
- **If M2 dominates:** **cap the `1/tr_sigma` scaling** (the literature's exact fix) and
  re-run. If the collapse vanishes, M2 is confirmed as the amplifier.
- **Selection artifact check:** the ep28 checkpoint may be the *raw* (non-EMA) weights;
  probe the **EMA** copy. If EMA is fine, we've been reading the wrong weights.

## 5. Does this analysis actually support the adapter (Arm B)? — honest answer

**Partly, and here's the catch I have to say out loud.** The adapter fixes **M3 and M5**:
frozen base can't be forgotten, zero-init can't regress, bounded correction can't explode.
But the adapter is trained on **the same denoising loss (M1)** and still writes into the
**same σ-scaled translation pathway (M2)**. So the adapter is **guaranteed not to *harm***
— but it is **not guaranteed to *learn placement***, because the objective that's supposed
to teach it placement is the very objective that doesn't reward placement. The adapter's
real value is the **safety asymmetry**, not a promise of a win. If M1/M2 are the true
engine, the adapter may simply **stay near zero** (no gain, no loss) rather than fix long
peptides. That's still strictly better than Arm A (which actively destroys the model), but
I should not oversell it.

**What would make the adapter actually win:** pair it with (a) a **placement-aware loss
term** (supervise the predicted COM against native, so the gradient *does* reward
placement), and/or (b) the **1/σ cap** so its corrections aren't amplified into noise, and
(c) **checkpoint selection on the docking meter** (never val loss). Without at least (a) or
(c), the adapter inherits M1.

## 6. Concrete next steps (ranked)

1. **Probe ep28 on held-out Propedia** (falsify M4) — 20 min, decides if the collapse is
   real or an eval-distribution artifact. *Highest information per minute.*
2. **Probe the EMA weights** of ep26/28 (falsify the selection artifact).
3. On the DGX adapter run: **select on the docking meter**, and if it flatlines at zero,
   add a **placement-supervised auxiliary loss** + **1/σ cap** — these attack M1/M2
   directly, which the vanilla adapter does not.
4. Keep the **pretrained model as the shipped Stage-1** until any retrain *beats it on the
   docking meter for ≥3 consecutive probes*. Right now pretrained is still the best we have.

## Sources
- Score explosion at small σ and the 1/σ-cap fix — score-based diffusion training-stability literature.
- Denoising loss ≠ sample quality (NELBO vs weighted loss / FID) — diffusion loss-function studies.
- Catastrophic forgetting on small-data fine-tuning; LoRA can degrade without KL-to-init replay.
- DiffDock/DiffDock-L overfitting to training families; translation/rotation-only fine-tuning strategies.
