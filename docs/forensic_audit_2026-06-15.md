# Forensic Audit of Our Own Model — Mistakes, Redundancy, and Why PPI Wins the Bands

*2026-06-15 · E201 · the deep self-audit Ram asked for after ~200 experiments: feature-class contribution,
redundancy, size confound, bugs, and what PPI's predictions actually track. Found one genuine modeling
mistake (a shippable fix) + three structural issues.*

---

## 🔴 Finding 1 (the mistake) — geometry features SABOTAGE vlong (self-inflicted)

Feature-class ablation, clustered-CV crystal-925, per band:
```
 band         n     SEQ(220)  POCKET(22)  GEO(16)   SEQ+POC   ALL-258
 ALL         925     0.346      0.201      0.223     0.342     0.367
 neutral     508     0.409      0.246      0.316     0.401     0.433   (geometry helps)
 charged     417     0.240      0.129      0.105     0.244     0.259   (geometry helps)
 long13-16   160     0.399      0.170      0.422     0.381     0.386   (geometry-ONLY best)
 vlong≥17     53     0.236      0.058     −0.054     0.266     0.030   ← geometry DESTROYS it
```
**vlong with all features = 0.030; with geometry dropped (SEQ+POC) = 0.266** — an 8× recovery (Δ+0.237,
robust over 6 seeds). The geometry features individually *do* correlate with vlong affinity
(`mean_burial −0.36`, `poc_n −0.32`, `bsa_hyd −0.24`) — but they are **size-confounded**, and because med/short
peptides dominate training, the model learns their med/short slope and applies the **wrong sign/magnitude** to
big peptides. Feeding them to vlong is net negative.

**This fully explains the vlong "failure"** (and retro-explains why the old narrow vlong-specialist worked: it
simply excluded the sabotaging geometry). It is NOT FEP-bound, NOT data-limited — it was self-inflicted by
feeding band-miscalibrated geometry. **Shippable fix: route vlong (L≥17) to a geometry-free (SEQ+POC) model;
all other bands keep the full model.** Global predictions for non-vlong unchanged.

## 🟠 Finding 2 — our "physics" (geometry) is ~redundant with sequence ON CRYSTAL

SEQ-alone 0.346 ≈ SEQ+POC+GEO 0.367 overall — geometry adds only **+0.02** on crystal. This is consistent and
correct: geometry is the **deployment** lever (it discriminates pose quality on generated poses, +0.16 there),
but on a fixed crystal it's largely collinear with what sequence composition already encodes. We are, on
crystal absolute-Kd, essentially a sequence/composition model competing with PPI's — which is selected +
BioLiP-trained.

## 🟠 Finding 3 — size confound: the model over-uses length

```
 corr(prediction, length)  = −0.207     corr(truth, length)  = −0.103   ← model 2× too size-driven
 corr(prediction, poc_n)   = −0.156     corr(truth, poc_n)   = −0.120
```
The model leans on size ~2× more than the truth warrants — which is exactly what corrupts within-band
prediction (where size is ~constant) and feeds the vlong sabotage. Worth de-confounding (length-residualise
size-like features, or down-weight).

## 🟠 Finding 4 — redundancy: 258 features → 64 effective dimensions

PCA: 64 PCs explain 95% of variance — **4× feature redundancy** for 925 samples. Geometry is internally
collinear: `sasa_hb~hb_count 0.85`, `poc_n~mj_contact 0.79`, `bsa_hyd~mj_contact 0.76`, `poc_n~sasa_hb 0.73`
— four "interface-size" features ≈ one axis. (Global SelectKBest-37/80 does NOT help — it hurts every band;
the right move is band-aware feature-class routing, not global selection.)

## 🟡 Finding 5 — 579 NaNs in the sequence descriptors (minor)

The autocorrelation descriptors (`ac1/ac2`) are NaN for short/uniform peptides; production guards them, but
the experiment path `nan_to_num`'s 579 values to 0 (0.28% of the seq-descriptor matrix). Cosmetic now, but a
real source of spurious zeros — should be cleaned (impute to feature mean, not 0).

## 🟢 Finding 6 — what PPI's predictions actually track (no magic)

On the bands PPI wins (T100), its predictions track **simple composition features we already have**:
```
 long13-16 (PPI 0.816):  truth~length −0.57, pep_vol −0.32;  PPI_pred ~ pep_vol −0.52
 vlong≥17  (PPI 0.458):  truth~pocket_hyd +0.42;             PPI_pred ~ pocket_hyd +0.26
 neutral   (PPI 0.660):  truth~pep_helix_prop −0.31;         PPI_pred ~ pep_helix_prop −0.42
```
PPI isn't using physics we lack — it's using peptide volume / pocket hydrophobicity / helix propensity, all of
which we have. Its edge is (a) **feature selection** (37 clean vs our 258 noisy), (b) **BioLiP training**
(home field), and (c) **not sabotaging itself with size-confounded geometry**. On long, truth~length is −0.57
on n=15 — small-n, and PPI's 0.816 is partly that redundancy.

---

## Actionable summary

| # | Issue | Fix | Expected |
|---|---|---|---|
| 1 | **vlong sabotaged by geometry** | route vlong→SEQ+POC (geometry-free); validate on deployment poses | vlong 0.03→0.27 crystal; ship after deploy check |
| 3 | size over-reliance | length-residualise size features | reduces within-band error |
| 4 | 4× feature redundancy | prune 4 collinear geometry → 1; band-aware routing not global selection | cleaner, less overfit |
| 5 | 579 NaN→0 | impute to mean | removes spurious zeros |

**The headline:** our vlong failure was a **self-inflicted modeling mistake** (size-confounded geometry), not
physics. The fix is a geometry-free vlong route. Everything else PPI does on the bands, we have the features
for — their edge is selection + home-field, which the ratio-scale already showed evaporates on fresh data.
