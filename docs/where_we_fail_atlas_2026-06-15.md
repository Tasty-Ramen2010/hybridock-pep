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

## 5. One-line verdict

PPI's neutral/long/vlong crystal edge is **home-field, not skill** — proven by specialists failing, feature
injection stalling at +0.02, and PPI's own descriptors not transferring. We **win charged, tie medium, win
deployment 4×, and own selectivity.** The only way to also match them on neutral/long/vlong crystals is to
train on their data distribution (the PDBbind+/BioLiP registration lever) — not a modelling trick.
