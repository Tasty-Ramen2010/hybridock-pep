# AI peptide-docking: overnight research synthesis (2026-08-01)

*Written while the corrected-loss run trains. Consolidates our v1→v6c history, the
loss-normalization breakthrough, and the 2025-26 field landscape into a decision map.*

---

## 0. TL;DR

1. **We found a real, confirmed bug**: our fine-tune loss was **unnormalized plain MSE**
   on a score that scales ~1/σ, so it was dominated by low-σ samples and starved
   PLACEMENT. **DiffDock (RAPiDock's parent) explicitly variance-weights the loss** so the
   trivial-prediction loss ≈ 1 at every noise level — "crucial for atomic accuracy." Our
   loss dropped that. **Every retrain v3→v6c shared this loss** → explains why none beat
   pretrained. The fix (σ-normalized loss) is live and **mechanically verified** (train
   loss 1600 → 3.7, the correct O(1) scale).
2. **RAPiDock is genuinely SOTA** (93.7% success @ top-25, +13.4% over AF2-Multimer,
   0.35 s/complex). Improving it is backing a strong horse, not a weak one.
3. **The placement cliff is a UNIVERSAL open problem** — AF3, Protenix, Chai-1, Boltz-2 all
   "struggle with precise placement" of flexible/long peptides. We are not fixing a
   RAPiDock quirk; we're at a field frontier. Expect modest gains; a real win is publishable.
4. **The field's proven answer to placement = two-stage coarse-to-fine** (global
   binding-site search → local refinement: pepATTRACT 70% near-native, PIPER-FlexPepDock,
   CABS-dock). RAPiDock does joint one-shot diffusion (no explicit global-placement stage),
   which is *why* placement is the fragile part.

---

## 1. Our history (v1→v6c) reinterpreted

All RAPiDock retrains used `train_lastlayer.py::compute_loss` (the `--v3/v4/v5/v6-mode`
flags only change which params are frozen, not the loss). That loss was **unnormalized
from its first commit** (3bf45b8, v6 era; earlier uncommitted versions almost certainly
same — no evidence it was ever normalized).

| run | lever | result | reinterpretation |
|---|---|---|---|
| v3/v3b/v3c | LR schedules (cosine, exp, adaptive-spike), cross_convs/score-heads | oscillation "structural", ep16 best | oscillation was the wrong-loss fighting itself |
| v5c | score-heads only, ultra-low LR | "best" 2.49Å, −0.14Å, hit@5 91% | measured on **superposed fragments** (wrong metric); −0.14 ceiling = wrong loss can only nudge before degrading |
| v6 | BN-freeze, pretrained-reg | implemented | same loss bug |
| Jul30-31 full-param / adapter | anchor, low-LR, σ-cap | collapsed / flat | same loss bug; anchor was a crutch for the wrong objective |

**The −0.14Å ceiling every version hit is the fingerprint of a wrong training objective.**
The current run is the **first** with the correct (variance-weighted) loss, evaluated on
the **correct metric** (direct docking RMSD on real complexes).

## 2. Field landscape (2025-26)

- **RAPiDock** (Nat. Mach. Intell. 2025): SOTA blind-ish peptide docking, 93.7% @ top-25.
- **AF3 / Protenix / Chai-1 / Boltz-1** (co-folding): 70-90% on protein-peptide complexes
  (Protenix 80.8% strict, AF3 89.9% moderate). **Multi-method combination (AF3+Protenix)
  → 89% high-quality** — ensembling is a validated lever (= our Track 1B).
- **Boltz-2**: near-**FEP affinity** prediction, 1000× faster — but "struggles with precise
  placement" and "limited power for highly dynamic systems." → **Boltz-2 is a lever for our
  AFFINITY/selectivity track, not placement.**
- **DiffPepDock** (in our third_party): SE(3) diffusion, FrameDiff-based, **normalizes the
  score loss AND adds a direct x0 position loss at low t** — i.e. it already does what we're
  now doing. We benchmarked HDP 0.80Å vs DiffPepDock 3.54Å on 1YCR (our pipeline was better
  there, but DiffPepDock's *loss design* is the reference).
- **Universal failure modes**: flexible/long peptides, cyclic peptides, apo (unbound)
  receptors, novel binding sites. "Training bias and sequence alignments shape AF peptide
  docking" — distribution matters (consistent with our Propedia-vs-RecentSet noise finding,
  though pocket-convention was ruled out).

## 3. What the field does for the placement cliff (candidate next levers)

Ranked by fit to our situation:

1. **σ-normalized loss (DONE, under test)** — makes coarse placement (high-σ) actually get
   gradient. First correct objective. Watch ep28/ep45.
2. **DiffPepDock-style low-t x0 placement term** — add a direct native-COM supervision at
   low noise. Safe (published), directly rewards placement. The natural *next* step if the
   normalization alone is flat. NOT reward-RL (that mode-collapses — Ram's instinct was right).
3. **Two-stage coarse-to-fine** (field's proven answer) — a global placement pass (crank N /
   wider tr_sigma search) to localize the site, then local refine. Heavier lift; matches how
   RAPiDock is *sampled* (N=100) more than how it's trained. Could be a *sampling-time*
   change (bias early diffusion steps toward site search) rather than retraining.
4. **Ensemble / orthogonal generators (Track 1B)** — RAPiDock + Boltz-1/DiffPepDock, take
   best/consensus. Field-validated (+multi-method → 89%). Zero retraining risk. Strong hedge
   and arguably the highest-EV move for the *tool* (vs. beating RAPiDock's weights).
5. **Boltz-2 for the selectivity/ΔG side** — near-FEP affinity is exactly our charged-wall /
   selectivity problem. Orthogonal to placement; worth a separate evaluation.

## 4. Decision map for the corrected-loss run

- **If it AVOIDS the collapse at ~ep28 and improves long/vlong** → the loss was the root
  cause. Let it run; then consider adding the low-t x0 term (#2) for more. Rebuild the DGX
  adapter bundle with the fixed loss.
- **If it's FLAT (≈pretrained, no collapse)** → the loss fix removed the *harm* but the
  objective still doesn't reward placement enough. Add the **low-t x0 placement term** (#2).
- **If it still COLLAPSES** → the loss wasn't sufficient; the cliff is deeper (architectural:
  single COM vector). Pivot to **ensemble (#4)** for the tool and treat retraining as research.
- **Regardless**: keep PRETRAINED as shipped Stage-1 until a retrain beats it on a HIGH-N
  (≥24) docking meter for ≥3 consecutive reads. N=8 has ~6Å noise — never ship on that.

## 5. Honest framing for iGEM

Beating RAPiDock's *weights* on the length cliff is a **field-frontier** problem even SOTA
labs haven't solved. The higher-EV, lower-risk win for the *tool* is the **ensemble +
physics-rescoring + selectivity** pipeline (where we already lead, and where Boltz-2-class
affinity is a real lever). The loss fix is the honest, well-grounded attempt at the hard
part; the ensemble is the pragmatic path to a strong Best-Software-Tool submission.

**Sources**: DiffDock (variance-weighted loss, NeurIPS ML4PS 2022); RAPiDock (Nat. Mach.
Intell. 2025, s42256-025-01077-9); "Benchmarking AF3-like methods for protein-peptide"
(bioRxiv 2025.03.09.642277); Boltz-2 (bioRxiv 2025.06.14.659707); pepATTRACT (Structure
2015); DiffPepDock (third_party code). Full links in session log.
