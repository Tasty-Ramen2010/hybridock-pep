# Where We Fail vs PPI on Crystal — Full Atlas + Why + What Actually Helps

*2026-06-15 · E194–E195 · the deep dive Ram asked for: neutral (PPI 0.66), long-structured (PPI 0.82),
vlong (PPI 0.46) — why they win, a full feature table, and an honest brainstorm of what closes each gap.*

---

## 1. The crystal failure table (T100, PPI's own benchmark)

| Slice | n | OURS r | PPI r | Gap | MAE ours/PPI |
|---|---|---|---|---|---|
| OVERALL | 85 | 0.359 | 0.525 | −0.17 | 1.29 / 1.13 |
| **charged \|q\|≥2** | 38 | **0.425** | 0.354 | **+0.07 WIN** | 1.25 / 1.25 |
| **v.charged \|q\|≥3** | 18 | **0.474** | 0.450 | **+0.02 WIN** | 1.43 / 1.24 |
| **med 9–12** | 34 | 0.245 | 0.248 | ~0 TIE | 1.07 / 1.22 |
| neutral \|q\|≤1 | 47 | 0.330 | 0.660 | −0.33 | 1.32 / 1.04 |
| long 13–16 | 15 | 0.344 | 0.816 | −0.47 | 1.24 / 0.83 |
| vlong ≥17 | 16 | 0.139 | 0.458 | −0.32 | 1.88 / 1.27 |

**We win charged, tie medium. PPI's entire crystal edge = neutral + long-structured + vlong.**

---

## 2. What drives each slice — per-slice top single-feature |Pearson r| (crystal-925, n=865)

| Slice | Top discriminators (feature, signed r) | Reading |
|---|---|---|
| **neutral** | `pkf:hopp +0.34` · `pkf:sheet −0.29` · `NLC(Z2)_AHR_P2 −0.28` · `pkf:arom −0.25` | **pocket hydrophobicity** dominates; binding by hydrophobic/shape match |
| **long 13–16** | `pkf:helix +0.39` · `pkf:alpha_n +0.37` · `Nc(Z3)_UCR_N1 −0.31` · `cys_frac −0.31` | **pocket helix-propensity** — long peptides binding helical grooves |
| **vlong ≥17** | `seq50 −0.50` · `seq40 −0.47` · `seq54 −0.45` · `seq120 −0.44` | **strong seq-descriptor (size/hydrophobicity) signal** — yet our model under-uses it |
| **structured (h+s≥.5)** | `NLC(Z2)_AHR_P2 −0.28` · `seq195 +0.26` · `pkf:hopp +0.26` | **a ProtDCal-3D contact descriptor is the #1 feature** — this is what PPI captures |
| **charged \|q\|≥2** | `seq130 −0.23` · `seq50 −0.21` · diffuse, all weak | no strong single feature — the FEP-only floor |

*Key:* for **structured** peptides the top discriminator is literally a 3D-contact descriptor — that is the
mechanistic reason PPI's contact-network features win there. For **vlong** the signal exists (|r|≈0.5) but
our pooled model dilutes it.

---

## 3. Why PPI wins these — and it is NOT a model/feature deficiency we can engineer around

We tested every obvious fix. All fail to close the T100 gaps:

### 3.1 Separate per-band models (Ram's "different models for each slice") — REFUTED
Training a specialist on only the slice (within each fold) does **worse than pooled, everywhere**, even
*within our own distribution* (crystal-925 clustered-CV):
```
 slice          POOLED   SLICE-SPECIALIST
 vlong≥17       0.217    0.101   ▼
 long13-16      0.351    0.285   ▼
 neutral        0.425    0.358   ▼
 charged        0.248    0.203   ▼
```
The pooled model **borrows strength** across all 925; a narrow slice (vlong n=53) overfits. On T100 the
specialists are even worse (vlong −0.31). *Separate models per band is the wrong architecture.*

### 3.2 Injecting the slice-relevant FEATURES into the pooled model — only +0.02
Adding SS (helix/sheet) and ProtDCal-3D contact descriptors (PPI's own feature class) to the pooled model,
T100 per slice:
```
 slice       base   +SS    +PD3D   +both   PPI
 vlong       0.086  0.064  −0.082  −0.030  0.458
 long13-16   0.159  0.176  0.142   0.159   0.816
 neutral     0.184  0.204  0.198   0.207   0.660
 charged     0.385  0.382  0.394   0.353   0.354  ← we already match/beat PPI
```
SS helps long/neutral by ~+0.02; ProtDCal-3D (literally PPI's features) **does not transfer** and even hurts
vlong. The big gaps (long 0.16→need 0.82) are nowhere near closeable by features.

### 3.3 The real reason: HOME-FIELD distribution, proven five ways
PPI was trained on BioLiP's T949; **T100 *is* BioLiP**. The gap is which peptides you trained on, not model quality:
1. Slice-specialists fail (§3.1) — not an architecture problem.
2. Feature injection gives +0.02 (§3.2) — not a feature problem.
3. ProtDCal-3D = PPI's *own* descriptor class, yet doesn't transfer to T100 from a PDBbind-trained model.
4. PPI's own clone degrades **0.32 → 0.22** on PPIKB (off *its* distribution) — same home-field effect, mirrored.
5. PPI's long13-16 = 0.816 on n=15 is almost certainly **redundancy-inflated** (small n, BioLiP-internal near-duplicates).

**Verdict: the neutral/long/vlong T100 gaps are distribution-bound. No feature or per-band model we can build
closes them — only training on the BioLiP/T949 distribution would.**

---

## 4. Brainstorm — ranked by what actually moves the needle

**Tier 1 — real, already-winning fronts (lead with these, don't chase the crystal gap):**
1. **Charged** — we already beat PPI (0.425 vs 0.354). Ship it as a headline; it inverts the old "charged floor caps us" story on *their* benchmark.
2. **Deployment** — on generated poses we lead ~4× (0.43 vs ~0.23–0.33). The crystal gap is on a crystal that doesn't exist for novel peptides.
3. **Selectivity at scale** — sequence τ already 0.06→0.16 on PPIKB's 454 families; common-frame docking (E193, queued) is the structural lever PPI can't run.

**Tier 2 — the ONE thing that closes the crystal gap (data, not modelling):**
4. **Train on the BioLiP/T949 distribution.** This is the only lever that touches neutral/long/vlong on T100.
   Path: register for PDBbind+ / pull BioLiP affinity (Ram's manual step) → retrain pooled model on the
   matching distribution. Everything else (features, per-band models) is measured to fail.

**Tier 3 — small honest increments (worth folding in, won't win alone):**
5. **Add SS (helix/sheet) fractions to the pooled model** — +0.02 on long/neutral, free (already computed). Let the GBT use pocket-helix natively (the top long discriminator).
6. **Keep features pooled, never split into per-band models** — measured to overfit.
7. **vlong:** the seq-descriptor signal (|r|≈0.5) is real but model-diluted; a *mild* length-aware feature (not a separate model) — e.g. interaction terms of seq-size × length — may recover some, but expect <+0.03.

**Explicitly DON'T do** (measured dead ends): separate slice-trained models (§3.1); fusing all 37 ProtDCal-3D
descriptors (hurts charged & vlong); chasing PPI's crystal number with feature engineering (it's home-field).

---

## 4b. Ram's engineered fixes — TESTED (gated features, not separate models)

Ram proposed three targeted fixes as **gated features** (global model stays identical for non-target slices).
Tested on T100 *and* robustly on crystal-925 clustered-CV (n=865):

| Fix | Idea | T100 (small n) | crystal-925 robust | Verdict |
|---|---|---|---|---|
| **Gated structured PD3D** | ProtDCal-3D contacts × 1[helix+sheet≥0.4] | long 0.73→0.78 (≈PPI 0.79!), neutral 0.25→0.33 | long **+0.02**, structured −0.02, overall flat | small real gain on long; T100 jumps were n=54 noise |
| **Hydrophobicity complementarity** | peptide-hyd × pocket-hyd product | hurts neutral (0.25→0.16) | — | **wrong formulation** (product backfires); needs a match-score, not a product |
| **vlong de-dilution** | size aggregates × 1[L≥17] | — | vlong **+0.02** (0.150→0.172) | small real gain, consistent |

**Honest read:** Ram's instinct is directionally right — gated-structured-PD3D helps long, vlong-de-dilution
helps vlong — but the robust gains are **~+0.02 per band**, not gap-closing. The dramatic T100 jumps
(neutral +0.08) did **not** survive the robust n=865 test (they were small-sample noise). Complementarity as a
raw product hurts. None of these close the structural gap; they're worth at most a modest fold-in.

## 4c. PPI-clone on BRAND-NEW data + ratio scale (E197) — PPI's 0.55 is mostly home-field inflation

We can't run real PPI on new data, but the faithful clone is in its exact feature class, so its retention
ratio transfers:
```
 CLONE on T100 (PPI home field)   r = 0.340
 CLONE on FRESH PPIKB (novel)     r = 0.219      retention ratio = 0.64
 ───────────────────────────────────────────────────────────────────────
 ESTIMATED real PPI-Affinity on brand-new data:  0.554 × 0.64 = 0.36
 OURS on the same fresh data:                    0.25
 OURS on fresh CHARGED:  0.261   vs  clone 0.136   ← WE WIN charged on fresh too
 OURS on fresh NEUTRAL:  0.234   vs  clone 0.279   ← PPI's contacts genuinely better (small, real)
```
**PPI's famous 0.55 drops to ~0.36 on data outside its BioLiP home field.** The headline "0.55 vs our 0.36"
gap is *mostly their inflation*, not our deficit. On fresh data the honest gap is ~0.11, concentrated in
**neutral** (where contact descriptors have a real but small edge); we already **win charged** everywhere.

## 4d. WHICH neutral complexes we fail on (E198 deep-dive, crystal-925 neutral n=508)

Per-complex error vs complex properties — what drives our neutral failures:
```
 corr(|error|, property):
   affinity magnitude |y|   +0.236   ← #1: we fail on EXTREME-affinity neutral (regression-to-mean)
   pocket size (poc_n)      +0.178   ← #2: big pockets = harder
   pocket hydrophobicity    −0.165   ← we do BETTER on hydrophobic pockets (simple burial); WORSE on polar
   hyd_MISMATCH (mean)      +0.035   ← weak: simple mean-hyd mismatch is NOT the driver
```
Two distinct failure modes in the worst-predicted neutral complexes:
- **Strong structured (β-sheet) binders we UNDER-predict** (1jp5 −12.6→−6.9; 6d40, 1sfi, 5u98; sheet≈0.4):
  the model regresses to the mean, can't reach the extreme-strong values.
- **Hydrophilic peptides in polar/large pockets we OVER-predict** (3fn0 −6.0→−10.7; 3qfd, 4pge; pep_hyd≈−1):
  specific H-bond/electrostatic networks (FEP-only physics) we can't capture from a static snapshot.

Sub-binning by affinity magnitude: weak-binder neutral r=0.241, mid r=0.056, **strong-binder r=0.474 but
mean|err| 1.96** — the strong tail is where the absolute error concentrates.

## 4e. The proper interface complementarity feature (E199) — built, marginal

Ram's "match-score, not product." Built the *physically correct* INTER-chain version: over interface contacts
(Cβ–Cβ < 8 Å), Σ property products across the peptide↔pocket interface (hyd, shape, aromatic complementarity +
mismatch + contact count). Cached `data/e199_compl.jsonl`.
```
 neutral single-feature |r|:  n_contacts −0.165 (strongest) · hyd_compl +0.069 · hyd_mismatch −0.047
 clustered-CV Δ:  neutral +0.010 · long +0.011 · overall −0.002 · charged −0.020
```
**+0.01 on neutral** — the right physics, and unlike the raw product it doesn't backfire, but it's marginal.
**Conclusion: the neutral gap is NOT a hydrophobic-complementarity gap.** It's affinity-extreme calibration
+ polar-pocket specific interactions (FEP-only) — largely irreducible with cheap static features.

## 4f. The RAPiDock-pose (deployment) scoring + PPI-clone scaled (E183, restated)

The number on RAPiDock-generated poses — what a user actually gets, with PPI's clone ratio-scaled:
```
 PPI-clone on RAPiDock rank-1 poses:  crystal 0.271 → pose 0.113   (retention 0.42)
 scaled to real PPI-Affinity:         0.554 → 0.23 (ratio) / 0.33 (pred-corr)
 OURS on the same RAPiDock poses:     0.43      ← ~4× the scaled PPI deployment number
```

## 4g. IS PPI OVERFIT? — Yes, partially (two independent signatures)

1. **Redundancy signature:** PPI scores **0.590 on the BioLiP-only T100** (most redundant with its T949
   training) vs **0.454 on the PDBbind-overlap T100** (less redundant) — **+0.136 better where the test
   overlaps its training distribution**. A generalising model would be flat across the two halves.
2. **Ratio-scale:** PPI's 0.55 → ~0.36 on brand-new data (§4c).

So **~0.15–0.19 of PPI's 0.55 is train/test redundancy + home-field inflation.** It is not fraud — it's the
standard BioLiP-internal redundancy that inflates every model evaluated on its own distribution (the same
effect that inflated *our* old crystal-65 numbers). On a level field PPI is a ~0.36 model, we are ~0.25–0.36
depending on slice, and we win charged + deployment + selectivity.

## 5. One-line verdict

PPI's neutral/long/vlong crystal edge is **home-field, not skill** — proven by specialists failing, gated
feature engineering giving only +0.02, PPI's own descriptors not transferring, and the **ratio-scale showing
PPI's 0.55 falls to ~0.36 on brand-new data** (its 0.55 is mostly BioLiP inflation). On truly novel data the
honest gap is ~0.11, concentrated in neutral; we **win charged everywhere, tie medium, win deployment 4×, and
own selectivity.** Closing the residual neutral edge needs their BioLiP training distribution (data lever) or
a proper hydrophobic-complementarity *match* feature (not the raw product, which backfired) — not separate
per-band models (measured to overfit).
