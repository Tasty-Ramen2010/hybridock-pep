# Second-Generation Fine-Tuning: V3C, V4C, V5C — Design Stats & Rationale

**Date:** 2026-05-29  
**Context:** HybriDock-Pep / RAPiDock fine-tuning on PepPC training set (2000 complexes)

---

## Background: Why a second generation?

All first-generation experiments (v1–v3b, v4n, v5n) degraded benchmark performance
relative to the pretrained model on 7 of 8 test complexes. The root causes were:

| Finding | Evidence |
|---|---|
| Pretrained manifold is near-optimal | Pretrained outperforms ALL finetuned on RMSD / diversity |
| Diversity collapses with any finetuning | 0.71 → 0.17–0.35 (across v1–v3b) |
| Long peptides need broad exploration | Only 7GUQ (5-mer) improved; all 8–15-mers degraded |
| Cosine LR re-injects instability cyclically | v3b spikes repeat at predictable phase intervals |
| Useful adaptation: epochs 4–9 only | After ep9, val loss plateaus or regresses |
| Model recovers from spikes | → pretrained manifold intact; disturbances are temporary |
| Score manifold is what's being compressed | Not the geometry prior — just the score field |

**Conclusion:** The pretrained diffusion prior is the model's most valuable learned
capability. Finetuning must be treated as a near-zero energy perturbation that only
recalibrates score magnitudes without reshaping the geometry exploration manifold.

---

## Common infrastructure added (all three experiments)

### `WarmupThenExponential` scheduler
Strictly monotone LR: linear warmup → exponential decay.

```
lr(epoch) = seg_base × rate^(epoch - seg_start)
rate      = (min_lr / init_lr) ^ (1 / decay_total)
```

- **No cosine rebound** — LR can only decrease
- **`permanent_reduce(factor)`**: on spike, permanently lowers all param-group LRs
  by `factor`, resets segment base to new floor, restarts decay from there.
  LR never recovers under any condition.

### Spike detection (all three)
Fires when `val tr_pred max_norm > prev × 10.0 AND > 100.0`.

Action (v3c/v4c/v5c): **PERMANENT** 50% LR reduction + 1-epoch EMA skip.  
(v3b/v4n/v5n used a 2-epoch cooldown followed by LR recovery — the wrong choice.)

### ESM frozen
ESM projection layer (`lm_embedding_layer`) frozen in all phases of all three experiments.

### Pretrained-reg (L2 toward pretrained init)
Only tracks `requires_grad=True` params. Patterns must match unfrozen layers.

---

## V3C — Minimal low-energy recalibration

**Philosophy:** Calibrate score magnitude without touching any geometry.  
Monotone exponential LR. Score heads + output convs only. No cross_conv touch in P1.

### Unfreeze patterns

| Phase | Layers unfrozen | Params unfrozen | % of total |
|---|---|---|---|
| P1 | tr/rot/tor_bb/tor_sc final_layer + final_conv + tor_bb/sc_bond_conv | 1,987,418 | 26.31% |
| P2 | P1 + cross_convs.3 | ~3,042,022 | ~40.3% |

### Hyperparameters

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| **Epochs** | 14 | 16 |
| **Base LR** | 4e-6 | 1e-6 |
| **Min LR (floor)** | 8e-7 | 1e-7 |
| **LR range** | 4e-6 → 8e-7 | 1e-6 → 1e-7 |
| **Warmup epochs** | 10 | 8 |
| **Optimizer tiers** | 2: output_convs=0.5×, heads=1.0× | 3: cc.3=0.1×, output=0.5×, heads=1.0× |
| **EMA decay** | 0.99997 | 0.99997 |
| **Grad clip** | 0.25 | 0.25 |
| **Weight decay** | 1e-5 | 2e-5 |
| **Pretrained-reg λ** | 3e-4 | 5e-4 |
| **Reg patterns** | final_conv, tor_\*_bond_conv | + cross_convs.3 |
| **Grad accum** | 4 | 4 |
| **Save cadence** | every 2 ep after ep6 | every ep from ep1 |

### Optimizer structure
```
Phase 1: make_v3cp1_optimizer
  lr=4.00e-06 (1.00×) — 12,674 params   [score heads: tr/rot/tor final_layer]
  lr=2.00e-06 (0.50×) — 1,974,744 params [output convs: final_conv + tor_*_bond_conv]

Phase 2: make_v3cp2_optimizer
  lr=1.00e-06 (1.00×) — 12,674 params   [score heads]
  lr=5.00e-07 (0.50×) — 1,974,744 params [output convs]
  lr=1.00e-07 (0.10×) — ~1,054,604 params [cross_convs.3]
```

### Design rationale
- P1 has NO cross_conv update — pure score magnitude calibration
- P2 adds cc.3 at only 0.1× so geometry shift is near-negligible
- 0.5× output_conv in P1 prevents over-adaptation of geometry-adjacent layers
- Strong reg (3e-4/5e-4) keeps structural layers near pretrained init

### Status: RUNNING (PID 2012034), Phase 1 epoch 1/14 at launch

---

## V4C — Tiny cross_conv adaptation test

**Philosophy:** Test whether TINY cc.3 learning in P1 can improve receptor-specific
interactions without destabilising the pretrained exploration manifold.  
Null hypothesis: if V4C diversity ratio ≥ V3C, then cc.3 at 0.15× is safe.

### Unfreeze patterns

| Phase | Layers unfrozen | Params unfrozen | % of total |
|---|---|---|---|
| P1 | score heads + output convs + **cross_convs.3 at 0.15×** | 3,042,022 | 40.27% |
| P2 | P1 + **cross_convs.2 at 0.08×** | ~4,096,000 | ~54% |

### Hyperparameters

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| **Epochs** | 14 | 18 |
| **Base LR** | 5e-6 | 1.2e-6 |
| **Min LR (floor)** | 1e-6 | 1e-7 |
| **LR range** | 5e-6 → 1e-6 | 1.2e-6 → 1e-7 |
| **Warmup epochs** | 10 | 8 |
| **Optimizer tiers** | 2: **cc.3=0.15×**, heads+output=1.0× | 3: **cc.2=0.08×**, cc.3=0.15×, heads+output=1.0× |
| **EMA decay** | 0.99997 | 0.99997 |
| **Grad clip** | 0.3 | 0.3 |
| **Weight decay** | 1e-5 | 2e-5 |
| **Pretrained-reg λ** | 3e-4 | 5e-4 |
| **Reg patterns** | final_conv, tor_\*_bond_conv, **cross_convs.3** | + **cross_convs.2** |
| **Grad accum** | 4 | 4 |
| **Save cadence** | every 2 ep after ep6 | every ep from ep1 |

### Optimizer structure
```
Phase 1: make_v4cp1_optimizer
  lr=5.00e-06 (1.00×) — 1,987,418 params  [score heads + output convs]
  lr=7.50e-07 (0.15×) — 1,054,604 params  [cross_convs.3]

Phase 2: make_v4cp2_optimizer
  lr=1.20e-06 (1.00×) — 1,987,418 params  [score heads + output convs]
  lr=1.80e-07 (0.15×) — 1,054,604 params  [cross_convs.3]
  lr=9.60e-08 (0.08×) — ~1,054,000 params [cross_convs.2]
```

### Design rationale
- cc.3 is the final cross-attention layer (most binding-geometry-specific)
- 0.15× in P1 means effective LR = 7.5e-7 — near-negligible receptor shift
- cc.2 at 0.08× in P2 ≈ near-frozen in practice
- Higher base LR than v3c (5e-6 vs 4e-6) compensates for 0.15× cc.3 scaling
- Slightly looser grad_clip (0.3) to accommodate cc.3 in P1

### Benchmark prediction
- If V4C diversity ≈ V3C: cc.3 touch at 0.15× is neutral → safe pattern to extend
- If V4C diversity < V3C: even 0.15× cc.3 compresses manifold → don't touch cc layers

### Status: RUNNING (PID 2015797), Phase 1 epoch 1/14 at launch (ep1 val=176.22)

---

## V5C — Ultra-minimal diversity-preserving recalibration

**Philosophy:** PRIMARY goal is preserving pretrained exploration diversity.
Do almost nothing. Treat preservation of the diffusion manifold as more important
than lower validation loss or aggressive specialisation.

### Key differences from v3c/v4c

| | v3c | v4c | **v5c** |
|---|---|---|---|
| P1 unfrozen | heads + output convs | heads + output + cc.3 | **heads ONLY** |
| Cross_conv touch | P2 only (cc.3 0.1×) | P1+P2 (cc.3 0.15×, cc.2 0.08×) | **NEVER** |
| Base LR P1 | 4e-6 | 5e-6 | **2e-6** |
| Base LR P2 | 1e-6 | 1.2e-6 | **5e-7** |
| EMA decay | 0.99997 | 0.99997 | **0.99998** |
| Grad clip | 0.25 | 0.30 | **0.20** |
| Pretrained-reg λ P1 | 3e-4 | 3e-4 | **1e-4** |
| Pretrained-reg λ P2 | 5e-4 | 5e-4 | **2e-4** |
| Total optimisation energy | low | medium-low | **minimal** |

### Unfreeze patterns

| Phase | Layers unfrozen | Params unfrozen | % of total |
|---|---|---|---|
| P1 | **ONLY tr/rot/tor_bb/tor_sc final_layer** | ~12,674 | **0.17%** |
| P2 | P1 + final_conv + tor_bb/sc_bond_conv | ~1,987,418 | ~26.3% |

Note: P1 is the smallest possible unfrozen set in any RAPiDock experiment to date.

### Hyperparameters

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| **Epochs** | 10 | 12 |
| **Base LR** | 2e-6 | 5e-7 |
| **Min LR (floor)** | 2e-7 | 5e-8 |
| **LR range** | 2e-6 → 2e-7 | 5e-7 → 5e-8 |
| **Warmup epochs** | 12 | 10 |
| **Optimizer tiers** | 1: heads=1.0× (single tier) | 2: output_convs=0.5×, heads=1.0× |
| **EMA decay** | 0.99998 | 0.99998 |
| **Grad clip** | 0.2 | 0.2 |
| **Weight decay** | 2e-5 | 3e-5 |
| **Pretrained-reg λ** | 1e-4 (weak) | 2e-4 (weak-moderate) |
| **Reg patterns** | tr/rot/tor_bb/tor_sc final_layer | + final_conv, tor_\*_bond_conv |
| **Grad accum** | 4 | 4 |
| **Save cadence** | every ep after ep4 | every ep after ep4 |

Note: warmup=12 with only 10 epochs means the LR peak is never reached — the
training runs entirely in the rising warmup ramp for P1, keeping effective LR
far below 2e-6. This is intentional: maximal conservatism in the first phase.

### Optimizer structure
```
Phase 1: make_v5cp1_optimizer
  lr=2.00e-06 (1.00×) — ~12,674 params   [score heads ONLY]

Phase 2: make_v5cp2_optimizer
  lr=5.00e-07 (1.00×) — ~12,674 params   [score heads]
  lr=2.50e-07 (0.50×) — ~1,974,744 params [output convs: final_conv + tor_*_bond_conv]
```

### Design rationale
- P1 with warmup=12 > epochs=10 means LR never exceeds 2e-6 × (10/12) = 1.67e-6
- This is the intended behaviour: the warmup ramp IS the training trajectory in P1
- P2 at 5e-7 base is 8× lower than v3c's P1 base — near-negligible output_conv shift
- Reg on score heads in P1 (unusual — but these ARE the unfrozen layers)
- No cross_conv update whatsoever across both phases

### Launch condition
Launched by `launch_v5c_when_ready.sh` (PID 2020698) when EITHER:
1. v3c Phase 2 completes (final.pt exists)
2. v4c Phase 2 completes (final.pt exists)
3. ≥90 minutes elapsed AND VRAM_free ≥ 4500 MiB

---

## Comparison matrix

### Optimisation energy budget (qualitative)

| Experiment | P1 energy | P2 energy | Cross_conv touch | Total budget |
|---|---|---|---|---|
| pretrained | — | — | — | baseline |
| v3c | low (output convs 0.5×, heads 1.0×) | low (+ cc.3 0.1×) | P2 only (0.1×) | low |
| v4c | medium-low (output 1.0×, cc.3 0.15×) | medium-low (+cc.2 0.08×) | P1+P2 (0.15×/0.08×) | medium-low |
| **v5c** | **near-zero (heads only, 1.0×)** | **minimal (+ output 0.5×)** | **NEVER** | **minimal** |

### Expected diversity outcomes (hypothesis)

| Experiment | Expected diversity ratio | Reasoning |
|---|---|---|
| pretrained | ~0.71 (baseline) | observed |
| v3c | 0.50–0.65 | low-energy, cc.3 only in P2 |
| v4c | 0.40–0.60 | cc.3 in P1 may compress; 0.15× mitigates |
| **v5c** | **0.60–0.70** | zero cross_conv touch; score-only calibration |

If v5c achieves diversity_ratio ≥ 0.60 AND improves val loss ≥ 5% over pretrained:
→ pure score-head recalibration is the correct minimal adaptation.

If v5c achieves diversity_ratio < 0.55: even score-head update compresses manifold
→ the pretrained model should be used as-is.

---

## Monitoring checklist (all three experiments)

**PRIMARY metrics (ranked by importance):**
1. `diversity_ratio` — target ≥ 0.60; hard stop < 0.35
2. Long-peptide RMSD (8–15 mers) — should not regress vs pretrained
3. Hit@5Å rate — should not drop below pretrained
4. `tr_pred max_norm` on val — spike indicator (>100 × 10× jump)

**SECONDARY metrics:**
5. Trimmed val loss (outliers>1k excluded)
6. Score variance (std / mean) — should remain stable
7. Oscillation amplitude in train loss — increasing amplitude = instability signal

**Checkpoint strategy:**
- All three: save every 1 epoch in P2 for post-hoc best-epoch selection
- v5c: save every 1 epoch in P1 too (after ep4) — fewer epochs, all valuable
- Recommended: benchmark the 3 most stable P2 checkpoints and average weights

---

## Infrastructure changes to `train_lastlayer.py`

### New constants
```python
_UNFREEZE_PATTERNS_V5CP1 = ["tr_final_layer", "rot_final_layer",
                              "tor_bb_final_layer", "tor_sc_final_layer"]
_UNFREEZE_PATTERNS_V5CP2 = _UNFREEZE_PATTERNS_V5CP1 + [
    "final_conv", "tor_bb_bond_conv", "tor_sc_bond_conv"
]
```

### New optimizer functions
- `make_v5cp1_optimizer`: single-tier (all heads at 1.0×)
- `make_v5cp2_optimizer`: 2-tier (heads=1.0×, output_convs=0.5×)

### New flag: `--v5c-mode`
Forces `--lr-schedule exponential`. All mode-branching code updated.

### Spike detection fix (applies to v3c/v4c/v5c)
Previously only v3c triggered `permanent_reduce()`. Fixed so v4c and v5c also
use permanent reduction (they always should have — was a latent bug).

```python
_permanent_reduce_mode = (v3c_mode or v4c_mode or v5c_mode)
if _permanent_reduce_mode:
    scheduler.permanent_reduce(0.5)
    ema_skip_remaining = 1
else:
    # v3b/v4n/v5n: 2-epoch cooldown
    spike_cooldown_remaining = 2
```

### `_ema_skip_modes` updated
```python
_ema_skip_modes = (v5n_mode or v3c_mode or v4c_mode or v5c_mode)
```

---

*Written: 2026-05-29 | All three experiments active or pending launch*

---

## Benchmark30 — SS-Stratified Evaluation Set

**Created: 2026-05-29**

### Motivation
The original 8-complex `inference_benchmark_set.csv` was too small and biased toward loop/coil peptides. The validation set (200 entries from PepPC-F) is overwhelmingly helical (115 HELIX, 69 SHEET, 11 PPII per Biopython phi/psi analysis). A proper evaluation benchmark must test all three structural modes because RAPiDock's failure modes differ by secondary structure:
- Helix: tests whether score heads correctly favor compacted α-conformations
- Sheet: tests whether generative model samples extended β-geometries (harder; diffusion tends to prefer helical due to training distribution)
- Unusual: tests diversity preservation for PPII, poly-Pro, long flexible, and disordered-to-ordered binders

### Composition

| Class    | Count | Source datasets                        | Notes |
|----------|-------|----------------------------------------|-------|
| HELIX    | 10    | PepPC-F val set                        | 100% Biopython-confirmed; 2 short, 5 medium, 3 long |
| SHEET    | 10    | PepPC-F val set                        | 50–100% β-strand; 8 short, 1 medium-9mer, 1 medium-11mer |
| UNUSUAL  | 10    | PepPC-F val + pepset + RefPepDB-Recent | 4 PPII, 2 poly-Pro/collagen, 1 poly-His, 1 Pro-Arg long, 1 V3-loop disordered, 1 disulfide-long |

**Total: 30 complexes. Benchmark CSV: `data/benchmark30.csv`**

### Secondary structure assignment method
Biopython `vectors.calc_dihedral` applied to Cα trace (φ/ψ angles):
- HELIX: ≥35% residues with φ∈[-100,-30]°, ψ∈[-70,+10]°
- SHEET: ≥30% residues with φ∈[-180,-60]° and ψ∈[60,180]° (extended strand)
- PPII: ≥30% residues with φ∈[-90,-50]°, ψ∈[110,170]°
- LOOP: everything else

UNUSUAL class deliberately mixes PPII-dominated (3-4 entries), poly-Pro (2 collagen-type), poly-His (charge anomaly), long partially-disordered (3 entries with ≥13 residues).

### Analysis pipeline

The benchmark runs 4 analysis tiers via `benchmark_inference_multi.py` (score-env):

**Analysis 1 — Overall performance**
Metrics: mean/median best RMSD, hit@2Å, hit@5Å, diversity ratio. All models compared.

**Analysis 2 — Per-class (highest priority)**
Same metrics split by HELIX / SHEET / UNUSUAL. Determines whether fine-tuning generalises across structural classes or only captures what's in the training distribution (mostly helix).

**Analysis 3 — Length stratification**
Bins: short (≤8), medium (9-11), long (12-15), very_long (16+). Tests if fine-tuning regresses on long-peptide difficulty.

**Analysis 4 — Diversity preservation**
Mean/median/min/max diversity ratio per model. Core metric for v5c hypothesis.

### Benchmark execution

Sequential chain (`sequential_chain_and_analysis.sh`):
1. v4c P2 completes → Analysis 1 (v4c vs pretrained, benchmark30, 5 samples)
2. v3c P2 completes → Analysis 2 (v3c vs pretrained, same benchmark)
3. v5c P2 completes → Analysis 3 (v5c vs pretrained, same benchmark)
4. Final comparison: all three finetuned vs pretrained (Analysis 4 outputs included in each)

5 samples per complex, seed=42. Running all 4 analyses produces 30 × 4 models × 5 poses = 600 total inference calls.

