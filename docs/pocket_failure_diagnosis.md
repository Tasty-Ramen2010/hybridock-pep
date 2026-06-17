# Deep diagnosis: why pocket-transfer fails, the double-difference, and non-FEP paths

**For Ram, 2026-06-17.** Answers: why/how the pocket idea fails, failure correlations, shortfalls, the MD
reframe, the double-difference math, the neutral failure, and concrete **non-FEP** ways forward.

---

## 1. WHY pocket-similarity transfer fails — the real mechanism (not "FEP")

`b(R)` is **our scorer's own systematic error on receptor R**: `b(R) = E[ G_true − S | R ]`. This is the
key. By construction it is the part of the physics that our features *do not already capture* — because if
any pocket descriptor predicted `b(R)`, that descriptor would already be in `S` and `b(R)` would shrink.

> **You cannot predict your own residual from features the model already used.** `b(R)` is orthogonal to
> our feature space *by definition*. Pocket descriptors aren't useless — their predictive part is already
> *inside* `S`. What's left in `b(R)` is precisely the physics our features miss.

That's why the metrics correlate so weakly with offset transfer:

| metric → predict `−|Δb|` | correlation |
|---|---|
| N-term sequence | +0.015 |
| pocket sequence | +0.035 |
| pocket composition | +0.008 |
| **pocket ProtDCal-3D (best)** | **+0.084** (<1% variance) |
| **directly LEARN `b(R)` (GBT/Ridge)** | **≈0** (worse than predict-the-mean) |

`b(R)` std = **2.14 kcal/mol**. Even the best metric leaves ≈2.13 — anchoring to a pocket-similar *other*
receptor injects its offset and shuffle ≥ similar (e273). **The 3-way pocket check finds similar shapes;
it cannot find similar *scorer-error*, which is what `b(R)` is.**

**Shortfall, precisely:** the idea assumes `b(R)` is a property of pocket *shape*. It is actually a
property of the *gap between true physics and our model* on that receptor — a different, orthogonal thing.
Two pockets can look identical and have opposite `b(R)` if our model mis-handles their specific chemistry
differently.

## 2. The MD reframe ("relax the peptide, don't model anything") — honest verdict

Your reframing is legitimate and I'll address it on its own terms: MD as **pose relaxation**, not free
energy. But it targets the **wrong error term**:

- `S = G + b(R) + c(P) + η`. Pose noise lives in **η** (the random, per-pose part). Relaxation reduces η.
- `b(R)` is the **systematic** part — and we measure it from **crystal/experimental poses** (SKEMPI e254,
  PPIKB structures). The offset is present *at the optimal pose*. Relaxing toward the optimal pose moves
  you toward where `b(R)` already sits — it cannot remove it.
- Direct evidence pose isn't the bottleneck: `corr(pose-RMSD, |error|) ≈ 0.01` (pose-quality audit).
  Within-receptor anchoring already hits the r≈0.75 ceiling *on crystal poses* — pose is not the limiter.

**So 0.1 ns relaxation helps η (a little), not `b(R)` (the wall).** It's a real but small lever aimed at a
term that isn't the problem. Not wrong — just not the fix for cross-receptor.

## 3. Your double-difference math — TESTED, and you were RIGHT (with one constraint)

Your 4-corner idea: `ΔG(P,R) ≈ y(P,Rk) + y(Pk,R) − y(Pk,Rk)`. The algebra (with `G = f(P)+g(R)+coupling`):
the double-difference cancels **both** `g(R)=b(R)` **and** `f(P)=c(P)`, leaving only the non-additive
coupling. So your instinct — "the extra information is going to waste" in single-anchoring — is **correct**.

**Empirical test on 31 real 2×2 grids (PPIKB, all 4 corners measured):**

| quantity | value |
|---|---|
| coupling (non-additivity) | mean **0.85**, median 0.63, max 2.05 kcal/mol — **small; additivity holds** |
| double-diff predicts held-out corner | **r = 0.955, MAE = 0.85 kcal/mol** |
| single-anchor (no scorer) baseline | MAE 2.89 |

**This is the strongest cross-receptor result we have, and it's yours.** When additivity holds (it does,
coupling ≈0.85), the double-difference nails the 4th corner.

**The constraint (the honest catch):** it needs the corner `y(P, Rk)` — *the query peptide measured on a
different receptor*. For a **de-novo designed** peptide you don't have that. But it is exactly available
for:
- **Repurposing / cross-reactivity:** P is a known binder of target A; predict its Kd on target B.
- **Selectivity panels:** a peptide measured on a reference receptor, predicted onto a new one.

This is the rigorous form of the **bridge-peptide** idea (I7): one shared peptide ties two receptors and
the double-difference cancels everything but coupling. **Deployable today for selectivity/repurposing.**

## 4. Where we fail on NEUTRAL — and how to beat it (it's model class, not physics)

You asked. We **lose on neutral**, and the cause is concrete (e282, fresh-305 neutral n=145):

| model (pooled, same 37 features) | NEUTRAL r | CHARGED r |
|---|---|---|
| **GBT (ours)** | 0.131 | **0.342** |
| **SVR (PPI-clone)** | **0.261** | 0.300 |
| Ridge | −0.03 | — |

**GBT and SVR are complementary:** GBT wins charged, SVR wins neutral (2×). We fail neutral because our
tree model overfits the neutral regime where the SVR's smooth kernel + feature selection generalizes
better. **Fix = charge-route:** neutral → SVR (0.131→0.261), charged → GBT (keep 0.342). That beats either
single model on its own weak axis — a real, free win. (Plain 50/50 blend gives all=0.322 but doesn't beat
pure SVR on neutral; route, don't blend.)

## 5. Salt-bridge ML — the one orthogonal-physics idea still worth trying

Your `(charge → receptor-spot)` salt-bridge ML is the right *kind* of idea: it adds physics **orthogonal**
to our aggregate features (explicit per-pair salt-bridge geometry + energy), which is the only thing that
*can* move `b(R)` (§1). What we know: the net salt-bridge ΔG is a small difference of large terms (Coulomb
−177 vs desolvation +209), so a naive per-pair sum amplifies error (e252 pooled 0.62 → clustered 0.13).
**But we have not tried a *learned* salt-bridge-residual model** that predicts the charged residual from
explicit (charge-product, distance, burial, local dielectric proxy) per-bridge features. Odds are modest
(the cancellation is the FEP-bound part), but it's the remaining untested orthogonal-physics lever and
cheap to prototype from structures. **Worth one shot.** Needs per-atom coords (716 PPIKB have structures).

## 6. The non-FEP synthesis (what to actually do)

The reason it "keeps coming back to FEP" is **elimination, not dogma**: `b(R)` is orthogonal to our
features (§1), and every *cheap* orthogonal physics we tried (RISM water, GIST, PB electrostatics) is flat
on it — leaving only explicit-water free energy. But you don't have to *predict* `b(R)`. The winning moves
**measure or cancel** it cheaply:

| path | what it does | status |
|---|---|---|
| **Double-difference (yours)** | cancels b(R)+c(P), r=0.955 | ✅ validated; repurposing/selectivity |
| **Probe-peptide fingerprint (I2)** | *measures* b(R) with 2–3 probes → r 0.52 | ✅ validated, wired |
| **Bridge peptide (I7)** | *measures* b(R)−b(Rk) for selectivity | ✅ = double-diff special case |
| **Charge-routed model** | neutral→SVR, charged→GBT | ✅ free win (§4) |
| **Salt-bridge residual ML** | orthogonal explicit physics | ◐ untested, one shot (§5) |
| pocket-similarity transfer | predict b(R) from shape | ✗ dead (§1) |
| short-MD relax for b(R) | targets η not b(R) | ✗ wrong term (§2) |

**Bottom line:** don't predict `b(R)` (impossible from our features, by construction). **Cancel it** (your
double-difference — validated r=0.955) or **measure it** (probe peptides — validated). Both are cheap,
non-FEP, and deployable. Plus two free model wins: pooled training (+0.04 overall) and charge-routing
(neutral 0.13→0.26).
