# Where We Still Fail — and What the New Data Can (and Can't) Fix

*2026-06-15 · Epoch 7 · honest failure map after decoding PPI-Affinity + ingesting PPIKB + PepBenchmark.*

This is the "stop and ask" / brainstorm document CLAUDE.md §9 asks for: every place we still lose, the
**measured** reason, and whether the brand-new data (PPIKB, PepBenchmark, the ProtDCal-3D engine) actually
moves it. Numbers are from committed scripts E177–E190.

---

## 1. The four places we still fail

| # | Failure | Measured number | Root cause (measured, not guessed) |
|---|---|---|---|
| F1 | **Absolute Kd on crystal vs PPI** | us 0.36–0.52 vs **PPI 0.55** (T100) | PPI's edge = STRONG + LONG structured peptides + home-field (trained on BioLiP, T100 *is* BioLiP) |
| F2 | **Charged complexes** | r 0.29 (charged) vs 0.50 (low-charge) | electrostatics/desolvation = single-snapshot-uncapturable → **FEP-only** (proven across 4 prior epochs) |
| F3 | **vlong (≥17 res)** | pooled −0.03 (now +0.39 via band-specialist) | was a model-mixing artifact; remaining ceiling = degenerate labels + FEP physics |
| F4 | **Deployment short / data-sparse bands** | short real-pose r 0.11–0.13 | only ~40–54 real-pose short examples; data volume, not features |

---

## 2. What the NEW data actually does — tested, mostly honest negatives

### 2.1 PPIKB (`docs/Affinity Dataset(branch).xlsx`) — 2229 clean entries, 1652 Kd, 810 new PDBs

**Training expansion to beat PPI on crystal (E189): NEGATIVE.**
```
 trained on 925 only           → T100 r = 0.385
 trained on 925 + PPIKB (all)  → T100 r = 0.319   (HURTS −0.07)
 trained on 925 + PPIKB Kd-only→ T100 r = 0.336   (still below baseline)
 PPI shipped target            → 0.549
```
*Why:* PPIKB crystals are **heterogeneous** (deposition quality, mixed assay even within "Kd") and
**off-distribution** from BioLiP-T100. Dumping more PDBs moves us *away* from the target distribution.
**Lever confirmed: the crystal gap needs DISTRIBUTION-MATCHED, CLEAN data (the BioLiP/T949 set itself,
or registered PDBbind+), not raw volume.**

**Structure-based selectivity on PPIKB families (E190): NEGATIVE.**
```
 within-family τ (leave-family-out):  sequence +0.011 · structure −0.108 · both −0.029
 (sequence baseline on all 80 families, E187, was +0.059)
```
*Why:* family peptides come from **different crystals** → ProtDCal-3D contact descriptors capture
crystal artifacts (resolution, conditions, binding-mode), not affinity differences. Cross-crystal is the
wrong frame for selectivity.

### 2.2 PepBenchmark (github ZGCI-AI4S-Pep) — NOT USABLE

35 datasets but all **peptide-bioactivity** (antimicrobial, anticancer, hemolytic, ACE-inhibition IC50) —
*intrinsic peptide properties, not protein-peptide binding affinity or selectivity*. And the repo has **no
license** (null) → fails our OSI/MIT requirement (CLAUDE.md §2.6). Verdict: irrelevant to our task.

---

## 3. The brainstorm — what WOULD work, ranked by leverage

1. **Selectivity in a CONSISTENT frame (the real win).** The selectivity signal that *does* transfer is
   measured on a common structural frame: SKEMPI/ATLAS ΔΔG (same complex, WT-vs-mutant, real mutation
   context) gave **structure +0.165 over sequence** (E165). PPIKB's 80 families can be rescued the same way:
   **dock every family peptide into ONE receptor structure with RAPiDock**, then score — this removes the
   cross-crystal artifact that killed E190. That is *exactly our deployment pipeline* and PPI cannot do it
   (no pose generator). **Next experiment:** GPU-dock the 30 charged + 50 neutral PPIKB families into their
   shared receptor, re-measure within-family τ on consistently-posed structures.

2. **Beat PPI on crystal via clean on-distribution data, not volume (F1).** E189 proves raw PPIKB hurts.
   The honest path is the **registered PDBbind+ / BioLiP-T949 distribution** — Ram's manual registration
   step. Filter PPIKB to **Kd-only + high-resolution + single assay source** before adding (partial: Kd-only
   already recovered +0.017). Pre-screen by assay consistency.

3. **The deployment story is already a win — lead with it (F1 reframed).** We do NOT need to beat PPI's
   crystal-oracle 0.55 to be "the best." On **generated** poses (the real task) PPI collapses 0.55 → ~0.23–0.33
   (E183 haircut) while we hold 0.43. The honest headline is *deployment*, where we are #1, not crystal-oracle.

4. **Charged floor (F2): accept it's FEP-only, ship selectivity ΔΔG instead.** Four epochs proved single-
   snapshot electrostatics can't crack it; even real GB fails charged (0.066→0.064). The charged WIN is
   relative ΔΔG (floor cancels) — and PPIKB gives **30 charged selectivity families** to validate it on.

5. **vlong/short (F3/F4): band-isolated specialists + more real poses.** vlong specialist already +0.39
   deployment (global untouched). short is pure data-volume — the e176 campaign (running) is the fix.

---

## 4. One-line verdict

The new data did **not** hand us a crystal-Kd win (PPIKB raw hurts; PepBenchmark is off-task/unlicensed).
What it gave us is **80 selectivity families** — and the measured lesson that selectivity must be scored in a
**consistent docked frame (our pipeline)**, not heterogeneous crystals. That is the experiment that turns our
unique capability (pose generation + structural ΔΔG) into the number nobody else can produce.
