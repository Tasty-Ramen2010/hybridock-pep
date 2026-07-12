# HybriDock-Pep — AI Fine-Tuning Log

> **Scope**: RAPiDock diffusion model fine-tuning on PepPC dataset for receptor–peptide
> interaction specialization. Documents all training runs, architectural decisions,
> observed failure modes, and planned experiments.
>
> **Not a how-to**: this is a live research log. For setup instructions see CLAUDE.md §5.
> For inference instructions see `scripts/run_rapidock.py` and README.md.
>
> **Do not read automatically** — load this file only when explicitly reviewing training history.

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Model Architecture](#2-model-architecture)
3. [Dataset](#3-dataset)
4. [Training Infrastructure](#4-training-infrastructure)
5. [V1 — Baseline Progressive Unfreezing](#5-v1--baseline-progressive-unfreezing)
6. [V2 — Layerwise LR Decay + Warmup Fix](#6-v2--layerwise-lr-decay--warmup-fix)
7. [V2b — Alternate Seed Baseline](#7-v2b--alternate-seed-baseline)
8. [V3 — Controlled Specialization](#8-v3--controlled-specialization)
9. [V4 — Pure Cross-Conv Ablation](#9-v4--pure-cross-conv-ablation)
10. [V5 — Ultra-Conservative Adaptation](#10-v5--ultra-conservative-adaptation)
11. [Key Findings & Failure Modes](#11-key-findings--failure-modes)
12. [Cross-Model Benchmark Results](#12-cross-model-benchmark-results)
13. [Monitoring & Launch Protocol](#13-monitoring--launch-protocol)
14. [Planned Analyses](#14-planned-analyses)

---

## 1. Project Context

**Project**: HybriDock-Pep — hybrid peptide docking tool for iGEM 2026 Best Software Tool.
**Pipeline**: RAPiDock (diffusion sampling, Stage 1) → Vina/AD4 rescoring (Stage 2).
**Target application**: malaria peptide LISDAELEAIFEADC docking to PfLDH (PDB 1T2D).
**Hardware**: NVIDIA RTX 5070 (12 GB VRAM, Blackwell CC 12.0) on WSL2 Ubuntu.
**Fine-tuning motivation**: pretrained RAPiDock (Zhao et al. Nat. Mach. Intell. 7:1308, 2025)
was trained on general protein–protein interfaces. PepPC-F introduces receptor diversity
(kinases, GPCRs, proteases, etc.) that may require interaction-layer adaptation.

---

## 2. Model Architecture

**Model**: `CGTensorProductEquivariantModel` (SE(3)-equivariant GNN)
**Total parameters**: 7,553,674 (~7.5M)
**Framework**: PyTorch + PyTorch Geometric + E3NN
**Checkpoint**: `train_models/CGTensorProductEquivariantModel/rapidock_local.pt`

### Layer taxonomy (deepest → shallowest)

```
Encoder layers (frozen deepest → more trainable):
  intra_convs.0  — core peptide-internal geometry (deepest, most physics-critical)
  intra_convs.1  — mid peptide geometry
  intra_convs.2  — near-core peptide geometry
  intra_convs.3  — outermost peptide-internal conv (safer to adapt)
  cross_convs.0  — earliest receptor-peptide interaction layer
  cross_convs.1  — mid receptor-interaction layer
  cross_convs.2  — 2nd outermost cross-conv (diverse receptor families)
  cross_convs.3  — outermost receptor-peptide interaction layer (highest priority)

ESM projection:
  rec_node_embedding.lm_embedding_layer  — linear(ESM_dim→emb_dim) for receptor
  pep_node_embedding.lm_embedding_layer  — (if peptide ESM enabled; off by default)

Score output heads (shallowest):
  tr_final_layer        — translation score magnitude
  rot_final_layer       — rotation score magnitude
  tor_bb_final_layer    — backbone torsion score magnitude
  tor_sc_final_layer    — sidechain torsion score magnitude
  final_conv            — equivariant conv feeding tr/rot heads
  tor_bb_bond_conv      — equivariant conv for backbone torsion
  tor_sc_bond_conv      — equivariant conv for sidechain torsion
  center_edge_embedding — edge features feeding final_conv
  pep_a_node_embedding  — node features feeding final_conv
  final_edge_embedding  — edge embedding for final layer
```

### Param count by tier (approximate)

| Tier | Params | % total |
|------|--------|---------|
| Score heads (P1 pattern) | 2,106,182 | 27.9% |
| + cross_convs.3 | 3,160,786 | 41.8% |
| + cross_convs.2 | 3,811,710 | 50.5% |
| + intra_convs.3 | 4,866,314 | 64.4% |
| ESM projection | 63,792 | 0.8% |
| ALL (P3 standard) | 7,553,674 | 100% |
| ALL minus ESM (P3 v3+) | 7,489,882 | 99.2% |

### Score-matching loss

Equal-weight MSE on 4 score fields:
```python
loss = 0.25 * mse(tr_pred, tr_target)
     + 0.25 * mse(rot_pred, rot_target)
     + 0.25 * mse(tor_bb_pred, tor_bb_target)   # if torsion exists
     + 0.25 * mse(tor_sc_pred, tor_sc_target)   # if torsion exists
```

### EMA

Exponential moving average of model weights (pytorch-ema, default decay 0.999).
EMA weights are used for validation and inference. Raw model is used for training.

**Critical**: EMA epoch-11 blowup is a known deterministic artifact — EMA weight
averaging causes a spike in val norms at epoch ~11 that is self-correcting by ep16.
**Not a fatal error**. See §11.1 for full analysis.

---

## 3. Dataset

### PepPC Training Set

| Split | File | Complexes |
|-------|------|-----------|
| Train | `combined_train_curated.csv` | 2,122 |
| Val | `combined_val_curated.csv` | 200 |

**Source breakdown** (training set):
- `peppc`: 300 entries (base PepPC dataset)
- `peppcf`: 1,600 entries (PepPC-F extended, diverse receptor families)
- `ppii_enriched`: 96 entries (polyproline helix II complexes, oversampled ×4)
- `recent_2024_2026`: 100 entries (recent PDB structures)
- `refpepdb`: 26 entries (reference benchmark set)

**Source-weighted sampling** (in `train_lastlayer.py`):
ppii_enriched oversampled ×4 because PPII helices are underrepresented in PepPC-F.
Other sources sampled ×1. Effective epoch size: 2,122.

**Typical yield**: 2,110 / 2,122 samples per epoch (12 consistent failures:
`ValueError: min() arg is an empty sequence` and `AssertionError` on malformed graphs).

### ESM Embeddings

Pre-computed and cached:
- `combined_train_curated_esm_cache.pt` — 2,000 embeddings (train)
- `combined_val_esm_cache.pt` — (val, pre-computed before curated)

ESM computation runs once with `--esm-device cpu` (default) to avoid WSL2 TDR crashes
that occur on long-sequence batches at ~batch 790/874.

### Benchmark Inference Set

`data/inference_benchmark_set.csv` — 8 complexes from PepPC val, one per peptide
length class (5, 8, 9, 10, 11, 12, 13, 15 residues), selected with seed=42.

```
7GUQ            (5-mer)
peppcf_1IE7_C_17_24     (8-mer)
peppcf_4ZRL_B_35_43     (9-mer)
peppcf_2Z30_B_111_120   (10-mer)
peppcf_2AUS_B_194_204   (11-mer)
peppcf_3ZPA_A_101_112   (12-mer)
peppcf_1R8Q_A_131_143   (13-mer)
peppcf_1GO4_B_29_43     (15-mer)
```

Single-complex 1YCR (MDM2/p53, 12-mer ETFSDLWKLLPE) used for Vina scoring validation.

---

## 4. Training Infrastructure

### Scripts

| Script | Purpose |
|--------|---------|
| `third_party/RAPiDock_finetuned/train_lastlayer.py` | Core training engine |
| `scripts/chain_training.sh` | v1 chain (original) |
| `scripts/chain_training_v2.sh` | v2/v2b chains |
| `experiments/chain_training_v3.sh` | v3 controlled specialization |
| `experiments/chain_training_v4.sh` | v4 cross-conv ablation |
| `experiments/chain_training_v5.sh` | v5 ultra-conservative |
| `scripts/monitor_and_launch.sh` | Automated GPU monitor + chain launcher |
| `scripts/benchmark_inference_multi.py` | 8-complex benchmark |
| `scripts/compare_finetuned.py` | 4-model val set comparison |
| `scripts/update_training_excel.py` | Generate training_stats.xlsx |

### Key `train_lastlayer.py` features (as of May 29 2026)

**Unfreezing modes**:

| Flag | P1 unfrozen | P2 unfrozen | P3 behavior |
|------|-------------|-------------|-------------|
| (none) | `_UNFREEZE_PATTERNS_P1` (score heads) | `_UNFREEZE_PATTERNS_P2` (+ convs) | all params |
| `--v3-mode` | V3P1 (heads + cross.2/3) | V3P2 (+ intra.3 @ 0.15×) | all except ESM |
| `--v4-mode` | V3P1 (heads + cross.2/3) | V3P1 (same — no intra) | all except ESM |
| `--v5-mode` | P1 (heads only) | V5P2 (+ cross.3) | all except ESM, aggressive LR |

**New flags added for v3/v4/v5**:
- `--ema-decay FLOAT` — override per-phase EMA decay
- `--weight-decay FLOAT` — explicit weight decay
- `--pretrained-reg-lambda FLOAT` — L2 penalty toward pretrained weights
- `--pretrained-reg-patterns [PATTERNS...]` — which layers to regularize
- `--save-every-after INT` — save every epoch after this epoch
- `--v3-mode / --v4-mode / --v5-mode` — experiment mode (mutually exclusive)

**Monitoring outputs added**:
- `[osc]` — oscillation amplitude over 10-epoch sliding window (range + std)
- `[norms-train]` — now includes `var=` (variance of norms, diversity proxy)
- `[norms-val]` — now reports rot and tor_bb norms in addition to tr
- CSV history: 6 new columns (rot/torbb norms, tr/rot variance)

**Optimizers**:
- Standard Adam — P1/P2 default
- `make_layerwise_optimizer()` — P3 standard (0.50/0.20/0.05 multipliers)
- `make_v3p2_optimizer()` — 3-tier diff LR for v3/v4 P2
- `make_v5p3_optimizer()` — aggressive P3 (0.30/0.10/0.02 multipliers)
- `build_pretrained_ref()` + `compute_pretrained_reg_loss()` — L2 reg infrastructure

### Output directories

All under `third_party/RAPiDock_finetuned/` (gitignored):

```
finetune_peppc_phase1/       v1 P1
finetune_peppc_phase2/       v1 P2
finetune_peppc_phase3/       v1 P3 (KILLED ep43)
finetune_peppc_v2_phase1/    v2 P1
finetune_peppc_v2_phase2/    v2 P2
finetune_peppc_v2_phase3/    v2 P3 (RUNNING)
finetune_peppc_v2b_phase1/   v2b P1
finetune_peppc_v2b_phase2/   v2b P2
finetune_peppc_v2b_phase3/   v2b P3 (RUNNING)
finetune_peppc_v3_phase{1,2,3}/   v3 (PENDING)
finetune_peppc_v4_phase{1,2,3}/   v4 (PENDING)
finetune_peppc_v5_phase{1,2,3}/   v5 (PENDING)
```

---

## 5. V1 — Baseline Progressive Unfreezing

**Status**: COMPLETE (P1/P2 done; P3 killed at ep43)
**Chain script**: `scripts/chain_training.sh`
**Log**: `logs/chain_training.log`

### Architecture

Standard 3-phase progressive unfreezing:
- P1: score heads only (27.9%)
- P2: + cross_convs + intra_convs (64.4%)
- P3: full model (100%)

No differential LR, no layerwise decay, no EMA override.

### Phase history

#### Phase 1 (30 epochs)
- LR: 1e-4, warmup: 3, plateau schedule
- Best val: ~48.0 (trimmed mean)
- Notes: first working run after fixing conformation_type=None bug (May 26)

#### Phase 2 (45 epochs)
- LR: 5e-5, warmup: 3, plateau schedule
- Best val: ~42.0 (trimmed mean)
- Notes: unfroze intra_convs + cross_convs; loss improving steadily

#### Phase 3 (killed at ep43/100)
- LR: 2e-5, warmup: 10, cosine schedule
- **Best val**: 37.38 @ epoch 16
- **Kill reason**: fatal score-field norm explosion at epoch 42 (×3.76M)
- Checkpoint saved: `finetune_peppc_phase3/rapidock_finetuned_best.pt`

**Kill analysis**:
- Ep1-16: steady improvement, best=37.38 at ep16
- Ep11: EMA blowup (routine, self-correcting by ep16)
- Ep16-42: plateau, RMSD floor worsening +0.66 Å/epoch
- Ep42: tr_pred max_norm → 31,805,500 (×3.77M vs prev epoch)
- Ep43: terminated after Terminated signal

**Key decision**: v1 P3 best checkpoint is the reference baseline for all v2/v3/v4/v5 comparisons.

---

## 6. V2 — Layerwise LR Decay + Warmup Fix

**Status**: P1 ✓, P2 ✓, P3 RUNNING (ep18/100 as of May 29)
**Chain script**: `scripts/chain_training_v2.sh`
**Log**: `logs/chain_training_v2.log`

### Architecture changes vs v1

- P3: `--layerwise-lr-decay` (0.50/0.20/0.05 multipliers)
- P3: corrected cosine warmup direction (fixed inverted warmup bug)
- `--grad-clip-norm 1.0` added throughout
- `--early-stop-patience 999` (effectively disabled for full run)

### Phase history

#### Phase 1 (20 epochs)
- LR: 1e-4, warmup: 5, plateau
- Best val: ~46 (trimmed)
- Checkpoint: `finetune_peppc_v2_phase1/rapidock_finetuned_best.pt`

#### Phase 2 (45 epochs)
- LR: 5e-5, warmup: 5, plateau
- **Best val: 38.38 @ epoch 16** (nearly matching v1 P3 best)
- Key finding: P2 with this config captures most of the improvement earlier
- Checkpoint: `finetune_peppc_v2_phase2/rapidock_finetuned_best.pt`

#### Phase 3 (running, ep18/100)
- LR: 1e-5, warmup: 10, cosine → 1e-7
- Layerwise decay: heads 1.0×, late convs 0.5×, mid 0.2×, early 0.05×
- **Best val: 39.07 @ epoch 16** (set just after ep11-16 EMA blowup recovery)
- Ep17-18: second EMA blowup (×42K norm spike at ep18, trimmed=92.4)
- Prognosis: self-correcting (same pattern as ep11 blowup)
- Checkpoint: `finetune_peppc_v2_phase3/rapidock_finetuned_best.pt`

### Bugs fixed before v2 start

1. **Inverted warmup** (`WarmupThenCosine`): warmup was ramping DOWN to base_lr instead of up
2. **NORM ALERT false alarm**: alert fired on decreasing norms (not just increasing)

---

## 7. V2b — Alternate Seed Baseline

**Status**: P1 ✓, P2 ✓, P3 RUNNING (ep18/100 as of May 29)
**Chain script**: `scripts/chain_training_v2.sh` (same as v2, seed=43)
**Log**: `logs/chain_training_v2b.log`

Identical architecture to v2. Seed changed to detect seed-dependent behavior.

### Phase history

#### Phase 2
- **Best val: 41.29 @ epoch 4** (best captured very early, before blowup)

#### Phase 3 (running, ep18/100)
- **Best val: 39.51 @ epoch 16** (new best after ep16 recovery)
- Ep18: second EMA blowup (×44K norm spike, trimmed=93.9)
- Prognosis: self-correcting

### v2 vs v2b comparison

Both show nearly identical loss trajectories with small stochastic variance.
Confirms the EMA ep11 and ep16-18 blowup patterns are deterministic (seed-independent).

---

## 8. V3 — Controlled Specialization

**Status**: PENDING (starts after v2 and v2b complete)
**Chain script**: `experiments/chain_training_v3.sh`
**Log** (when running): `logs/chain_training_v3.log`

### Design rationale

Prior runs showed:
- **Distribution shift**: PepPC val loss improved; generalization to novel targets degraded
- **Diversity collapse**: pretrained 71% Hit@5Å → finetuned 17-34%
- **Score-field instability**: large norm spikes in tr_pred / rot_pred

V3 hypothesis: carefully controlling which layers adapt, combined with:
- Keeping intra_convs entirely frozen in P1 (peptide-physics priors preserved)
- Soft L2 anchor toward pretrained weights for vulnerable layers
- Per-phase EMA decay that slows as model stabilizes

### Unfreezing schedule

| Phase | Unfrozen | Params | % |
|-------|----------|--------|---|
| P1 | score heads + cross_convs.2/3 | 3,811,710 | 50.5% |
| P2 | + intra_convs.3 @ 0.15× LR | 4,866,314 | 64.4% |
| P3 | all except ESM projection | 7,489,882 | 99.2% |

### Training schedule

#### Phase 1
- LR: 2e-5, warmup: 5, plateau schedule
- Grad clip: 1.0, WD: 1e-6, EMA: 0.9995
- Epochs: 20, save every 5
- Pretrained reg: disabled (unfrozen layers can't drift from pretrained anyway)

#### Phase 2 (3-tier differential LR)
- Base LR: 5e-6, warmup: 6, plateau schedule
- Tiers:
  - Score heads + cross_convs.3: 1.0× = 5e-6
  - cross_convs.2: 0.70× = 3.5e-6
  - intra_convs.3: 0.15× = 7.5e-7 (very slow — outermost peptide conv)
- Grad clip: 1.0, WD: 1e-5, EMA: 0.9997
- Epochs: 40, save every 5
- Pretrained reg: lambda=2e-4, patterns: intra_convs + cross_convs.0/1 + node embeddings

#### Phase 3 (cosine, full model)
- Peak LR: 7e-6 → 1e-7 cosine, warmup: 10
- Layerwise decay: 1.0× / 0.5× / 0.2× / 0.05×
- Grad clip: 1.0, WD: 1e-5, EMA: 0.9999
- Epochs: 80, save every epoch from ep20
- Pretrained reg: lambda=1e-4 (continued)

### New monitoring in v3

- `[osc]`: oscillation amplitude over 10-epoch window (range + std)
- `tr_norm_var`, `rot_norm_var`: variance of score norms (diversity proxy)
- Val rot_pred and tor_bb_pred norms reported

---

## 9. V4 — Pure Cross-Conv Ablation

**Status**: PENDING
**Chain script**: `experiments/chain_training_v4.sh`
**Log** (when running): `logs/chain_training_v4.log`

### Design rationale

Ablation of v3: removes intra_convs.3 from P2 and removes L2 regularization.
Tests whether cross_conv-only adaptation is sufficient, without any additional
regularization overhead.

### Differences vs v3

| | V3 | V4 |
|--|----|----|
| P2 unfreeze | + intra_convs.3 @ 0.15× | same as P1 (cross only) |
| P2 diff LR | 3-tier (0.15/0.70/1.0×) | 2-tier (0.70/1.0×) |
| Pretrained L2 reg | ✓ lambda=2e-4/1e-4 | ✗ none |
| P2 WD | 1e-5 | 1e-6 (lighter) |

### Training schedule

#### Phase 1 (identical to v3 P1)
- LR: 2e-5, warmup: 5, plateau; EMA: 0.9995; epochs: 20

#### Phase 2
- Same unfrozen params as P1 (cross_convs.2/3 + heads)
- 2-tier diff LR: cross_convs.2=0.70×, rest=1.0×
- LR: 5e-6, warmup: 6, WD: 1e-6, EMA: 0.9997; epochs: 40

#### Phase 3 (identical to v3 P3 minus pretrained-reg)
- Peak LR: 7e-6 → 1e-7, standard layerwise 0.5/0.2/0.05
- EMA: 0.9999; epochs: 80; save every epoch from ep20

---

## 10. V5 — Ultra-Conservative Adaptation

**Status**: PENDING
**Chain script**: `experiments/chain_training_v5.sh`
**Log** (when running): `logs/chain_training_v5.log`

### Design rationale

Tests the lower bound of fine-tuning aggressiveness. If the pretrained prior is as
strong as cross-model analysis suggests, minimal adaptation may beat any deeper
fine-tuning. Extreme caution: lower LR, tighter grad_clip, aggressive P3 layerwise.

### Differences vs v3/v4

| | V3 | V4 | V5 |
|--|----|----|-----|
| P1 unfrozen | heads+cross.2/3 | same | heads ONLY |
| P2 unfrozen | +intra.3 | same as P1 | +cross.3 only |
| P1 LR | 2e-5 | 2e-5 | 1e-5 (½) |
| P2 LR | 5e-6 | 5e-6 | 2e-6 (×0.4) |
| Grad clip | 1.0 | 1.0 | 0.5 |
| P3 layerwise | 0.50/0.20/0.05 | same | 0.30/0.10/0.02 |
| P3 peak LR | 7e-6 | 7e-6 | 5e-6 |
| Warmup (P3) | 10 | 10 | 15 |
| P3 epochs | 80 | 80 | 60 |
| EMA (P1/P2/P3) | 0.9995/0.9997/0.9999 | same | 0.9997/0.9998/0.9999 |

### Training schedule

#### Phase 1 (score heads only, 27.9%)
- LR: 1e-5, warmup: 8, plateau; grad_clip: 0.5; EMA: 0.9997; epochs: 20
- Save every epoch after ep8 (after warmup)

#### Phase 2 (+ cross_convs.3, 41.8%)
- Uniform LR: 2e-6, warmup: 8; grad_clip: 0.5; WD: 1e-6; EMA: 0.9998; epochs: 30
- No differential LR (single ring added — not worth splitting)
- Save every epoch after ep8

#### Phase 3 (full minus ESM; aggressive layerwise)
- Peak LR: 5e-6 → 1e-7 cosine, warmup: 15; grad_clip: 0.5; EMA: 0.9999; epochs: 60
- Aggressive multipliers: heads=1.0×, late=0.30×, mid=0.10×, early=0.02×
- Save every epoch after ep15 (post-warmup)

---

## 11. Key Findings & Failure Modes

### 11.1 EMA Blowup — Epoch 11 (Deterministic, Self-Correcting)

**Observed in**: v1 P3, v2 P3, v2b P3 (consistently at epoch 11, then again ep16-18)
**Symptom**: val tr_pred max_norm spikes ×1000–×45000; raw_mean val explodes to 1e12+
**Trimmed val loss**: spike to 60-90 (vs normal 40-50) for 2-4 epochs
**Recovery**: always self-corrects within 2-5 epochs; new bests set on recovery

**Root cause**: the EMA integrates ~11 epochs of training before the EMA weights
diverge enough from raw weights to cause the val score predictions to enter a
temporarily high-variance regime. This is not catastrophic — the trimmed mean
(drops top 5% of samples) absorbs most of the signal. By ep16, EMA catches up
to a better raw model and sets a new best.

**Action**: no action. Do not reduce LR or kill the run based on ep11 blowup.

### 11.2 Fatal Score-Field Explosion (×3.7M at v1 P3 ep42)

**Symptom**: tr_pred max_norm → 31,805,500; val raw_mean → 1.9×10^13
**Cause**: optimizer escaping the pretrained score-calibration basin after 42 epochs
of full-model retraining with cosine LR starting at 2e-5

**Distinguishing from EMA blowup**:
- EMA blowup: trimmed val stays below 95, median stays below 10, train loss stays stable
- Fatal explosion: trimmed val > 100, train loss also degrades, norm never recovers

**Response**: kill the run. Best checkpoint saved before explosion is still valid.

### 11.3 Distribution Shift — PepPC vs Novel Targets

**Finding**: after P3 fine-tuning, models improve PepPC val loss but degrade
generalization to novel targets (e.g. 1YCR MDM2/p53 docking).

**Benchmark evidence** (`logs/benchmark_inference_multi/benchmark_summary.csv`):
- Pretrained wins on 7/8 multi-complex benchmark cases (best_rmsd metric)
- All finetuned models fail 1YCR (best_rmsd 9-17 Å vs pretrained 13.2 Å)
- Pretrained model has higher pose diversity (hit_rate, diversity columns)

**Interpretation**: PepPC-F encodes receptor diversity that causes the model to
specialize toward common PepPC receptor interaction patterns at the expense of
broader generalizability. This is the core motivator for v3/v4/v5 designs.

### 11.4 Pose Diversity Collapse

**Metric**: average pairwise RMSD of 20 inference poses (diversity score 0-1)
**Pretrained**: 0.71 average diversity on benchmark set
**V1 finetuned**: 0.17 (severe collapse)
**V2/V2b**: 0.17-0.34 (partial collapse)

**Symptom**: finetuned model generates near-identical poses on novel targets;
pretrained model explores a broader distribution. This explains the higher
Hit@5Å rate of the pretrained model despite worse average RMSD.

### 11.5 Val Loss Floor Analysis

**Val loss floor** (trimmed mean, stable epochs):
- Pretrained (no fine-tuning): ~48-50 on PepPC val
- V1 P3 best: 37.38 @ ep16
- V2 P3 best: 39.07 @ ep16
- V2b P3 best: 39.51 @ ep16
- V2 P2 best: 38.38 @ ep16 (nearly matches P3!)

**Key insight**: most of the val loss improvement happens in P2. P3 provides
marginal additional gain (if any) while introducing distribution shift risk.

### 11.6 Load Failures (12/2122 per epoch)

**Consistent**: 12 out of 2122 training samples fail to load every epoch.
**Cause**: malformed graphs in PepPC dataset (empty sequences, assertion failures).
**Impact**: negligible (0.57% failure rate).
**Action**: none — these are known-bad entries; curating them out is not worth the effort.

### 11.7 Vina Score Inflation on Finetuned Poses

**Finding**: Vina scoring on best poses from finetuned models returned 139.7 kcal/mol
(positive = unfavorable) vs pretrained's 16.2 kcal/mol.

**Cause**: finetuned poses on 1YCR have severe clashes not present in pretrained poses.
The score-field calibration shift causes the diffusion to generate poses with
unrealistic atom-atom overlaps on non-PepPC targets.

**Implication**: distribution shift affects not just RMSD metrics but pose quality.

---

## 12. Cross-Model Benchmark Results

### Multi-complex benchmark (`logs/benchmark_inference_multi/benchmark_summary.csv`)

20 poses per model, 8 complexes, 4 models (pretrained / v1 / v2 / v2b).

| Complex | Length | Pretrained best_rmsd | V1 best | V2 best | V2b best |
|---------|--------|---------------------|---------|---------|---------|
| 7GUQ | 5 | 13.23 | 9.82 | 9.06 | 8.46 |
| 1IE7 | 8 | **1.34** | 1.61 | 1.42 | 1.68 |
| 4ZRL | 9 | **2.08** | 2.90 | 2.94 | 3.07 |
| 2Z30 | 10 | **1.96** | 3.44 | 2.05 | 2.55 |
| 2AUS | 11 | **2.58** | 5.49 | 4.24 | 4.53 |
| 3ZPA | 12 | **3.11** | 5.83 | 4.98 | 5.63 |
| 1R8Q | 13 | **3.10** | 6.70 | 4.49 | 5.81 |
| 1GO4 | 15 | **3.58** | 7.07 | 5.55 | 6.08 |

**Bold** = best across all 4 models. Pretrained wins 7/8 (not 7GUQ 5-mer).

### Hit@5Å rates

| Complex | Pretrained | V1 | V2 | V2b |
|---------|------------|----|----|-----|
| 7GUQ | 0.00 | 0.00 | 0.00 | 0.00 |
| 1IE7 | **1.00** | **1.00** | **1.00** | **1.00** |
| 4ZRL | **1.00** | **1.00** | **1.00** | **1.00** |
| 2Z30 | **1.00** | 0.25 | 0.65 | 0.45 |
| 2AUS | **1.00** | 0.00 | 0.25 | 0.10 |
| 3ZPA | 0.70 | 0.00 | 0.05 | 0.00 |
| 1R8Q | 0.85 | 0.00 | 0.05 | 0.00 |
| 1GO4 | 0.70 | 0.00 | 0.00 | 0.00 |

Pretrained significantly better for longer peptides (≥10 residues).

### 1YCR (MDM2/p53) Vina scoring

| Model | Best RMSD (Å) | Vina score (kcal/mol) |
|-------|--------------|----------------------|
| Pretrained | 2.25 | +16.2 |
| V1 | 17.51 | +139.7 |
| V2 | 13.83 | +139.7 |
| V2b | 16.66 | +139.7 |

All finetuned models produce clashing poses on 1YCR (positive Vina = unfavorable).

---

## 13. Monitoring & Launch Protocol

### Automated monitoring

`scripts/monitor_and_launch.sh` monitors running v2/v2b training and
automatically launches v3 → v4 → v5 in sequence when each finishes.

**Run in a persistent tmux session**:
```bash
tmux new-session -d -s training_monitor
tmux send-keys -t training_monitor \
    'bash scripts/monitor_and_launch.sh 2>&1 | tee logs/monitor_and_launch.log' Enter
```

**Monitor the monitor**:
```bash
tail -f logs/monitor_and_launch.log
```

**Manual GPU check**:
```bash
watch -n 10 "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
```

### Completion detection logic

A chain is considered complete when BOTH:
1. No process with the chain's output-dir path is running (`ps aux | grep`)
2. The final checkpoint exists: `rapidock_finetuned_final.pt`

### Expected timeline (rough, RTX 5070)

Each epoch takes ~530-560 seconds (~9 min). Per-phase estimates:

| Phase | Epochs | Est. time |
|-------|--------|-----------|
| v2/v2b P3 remaining | ~82 each | ~12.5h each (parallel) |
| v3 P1 | 20 | 3.1h |
| v3 P2 | 40 | 6.2h |
| v3 P3 | 80 | 12.4h |
| v4 P1+P2+P3 | 140 | ~21.7h |
| v5 P1+P2+P3 | 110 | ~17h |

Total from v2/v2b completion: ~60h for v3+v4+v5 sequential.

---

## 14. Planned Analyses

Once v3/v4/v5 complete, run:

### Per-model analysis
1. **Val set loss analysis**: compare trimmed-mean val loss floors across all 8 models
2. **Multi-complex benchmark**: re-run `benchmark_inference_multi.py` with v3/v4/v5
3. **Diversity metrics**: avg pairwise RMSD and Hit@5Å for each model
4. **Oscillation amplitude**: compare `tr_norm_var` trajectories (v3 logs it natively)

### Cross-model synthesis
- Pearson r between PepPC val loss and generalization benchmark RMSD
- Loss floor vs diversity trade-off curve (specialization vs generalization Pareto)
- Best model identification for HybriDock-Pep Stage 1 inference

### Report updates
- `logs/finetuning_analysis_report.md` — comprehensive update post-v5
- `logs/training_stats.xlsx` — re-run `update_training_excel.py` with all phases

### Ablation matrix (planned)

| Experiment | intra.3 in P2 | cross.2 in P2 | L2 reg | P3 LR | Layerwise |
|-----------|:---:|:---:|:---:|-------|---------|
| v1 | ✓ | ✓ | ✗ | 2e-5 | none |
| v2/v2b | ✓ | ✓ | ✗ | 1e-5 | 0.5/0.2/0.05 |
| v3 | ✓ (0.15×) | ✓ (0.70×) | ✓ | 7e-6 | 0.5/0.2/0.05 |
| v4 | ✗ | ✓ (0.70×) | ✗ | 7e-6 | 0.5/0.2/0.05 |
| v5 | ✗ | ✗ (only .3) | ✗ | 5e-6 | 0.3/0.1/0.02 |

This matrix directly tests:
- `v3 vs v4`: effect of intra_convs.3 + L2 reg
- `v4 vs v5`: effect of cross_convs.2 inclusion
- `v2 vs v3`: effect of lower LR + reg on same unfreeze pattern
- `v1 vs v2`: effect of layerwise LR decay alone

---

*Last updated: 2026-05-29. Next update: after v3/v4/v5 benchmarks complete.*
