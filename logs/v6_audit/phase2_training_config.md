# PHASE 2 — V6 Training Configuration
*Generated 2026-05-30 pre-v6 launch*

---

## 1. Starting State

| Item                        | Value                                             |
|-----------------------------|---------------------------------------------------|
| Starting checkpoint         | `third_party/RAPiDock/pretrained/rapidock_global.pt` |
| Validated against           | `freeze_audit §6` — pretrained BN stats baseline |
| Drift tolerance at start    | ≤ 1.0% per frozen module vs rapidock_global.pt   |
| BatchNorm fix               | `freeze_frozen_bn_stats()` called after load + at start of each epoch |
| BN drift monitoring         | `check_frozen_bn_drift()` called after each `train_epoch()` call |

---

## 2. Three-Phase Training Schedule

### Phase 1 (Epochs 1–8) — Torsion + Score Heads Warm-Up

| Parameter              | Value                          |
|------------------------|-------------------------------|
| Epochs                 | 1–8                           |
| Trainable modules      | `tor_bb_bond_conv`, `tr_final_layer`, `rot_final_layer`, `tor_bb_final_layer`, `tor_sc_final_layer` |
| Trainable params       | 980,258 (12.98%)              |
| Frozen modules         | all `intra_convs`, `cross_convs`, `tor_sc_bond_conv`, all embeddings |
| Learning rate          | 3e-6 (uniform across groups)  |
| LR schedule            | WarmupThenCosine (warmup 2 epochs), cosine to epoch 8 |
| Grad clip              | 0.5 (tight — score heads are sensitive)           |
| Oversampling           | None — uniform 1× all tiers  |
| Effective epoch size   | 1,200 (1000 gap-fill + 200 replay)                |
| L2 reg (cross_convs)   | Not applied (cross_convs frozen in P1)            |

**Purpose:** Adapt score heads and torsion backbone conv to the gap-fill distribution without
touching the cross-conv geometry. The cross_convs drive spatial reasoning; touching them
before the score heads have adapted causes instability (observed in v2/v3 runs).

---

### Phase 2 (Epochs 9–35) — Cross-Conv Specialization

| Parameter              | Value                          |
|------------------------|-------------------------------|
| Epochs                 | 9–35                          |
| Trainable modules      | All Phase 1 + `cross_convs.0–3` |
| Trainable params       | 3,648,228 (48.30%)            |
| Frozen modules         | `intra_convs`, `tor_sc_bond_conv`, all embeddings |
| Learning rate schedule | 5e-6 → 5e-7 cosine (peak at ep9, decay to ep35)  |
| Grad clip              | 1.0                           |
| Oversampling           | Tier-based (see §3)           |
| Effective epoch size   | 2,242 samples                 |
| L2 reg λ               | 3e-4 on `cross_convs.0–3` (pull toward pretrained weights) |

**Per-group LR multipliers:**

| Param group            | Multiplier | Effective LR at peak |
|------------------------|------------|---------------------|
| `cross_convs.0`        | 0.4×       | 2.00e-6             |
| `cross_convs.1`        | 0.4×       | 2.00e-6             |
| `cross_convs.2`        | 0.7×       | 3.50e-6             |
| `cross_convs.3`        | 0.7×       | 3.50e-6             |
| `tor_bb_bond_conv`     | 0.5×       | 2.50e-6             |
| Score heads (4)        | 1.0×       | 5.00e-6             |

**Rationale for graduated LRs:** Early cross_conv layers learn more general geometry;
higher LR on later layers (cc.2/3) allows faster specialization on long/very_long geometry.

---

### Phase 3 (Epochs 36–45) — Fine-Polish + Stability

| Parameter              | Value                          |
|------------------------|-------------------------------|
| Epochs                 | 36–45                         |
| Trainable modules      | Same as Phase 2               |
| Learning rate          | 5e-7 (flat — stability run)   |
| LR schedule            | Flat (cosine completed)       |
| Grad clip              | 1.0                           |
| Oversampling           | Back to uniform 1× (replay retained) |
| Effective epoch size   | 1,200 samples                 |
| L2 reg λ               | 3e-4 (retained)               |

**Purpose:** Prevent Phase 2's oversampling from baking in bias. Low uniform LR
allows the model to consolidate learned geometry without overfitting any specific tier.

---

## 3. Tier-Based Oversampling (Phase 2 Only)

| Tier                   | SS    | Length    | Count | Mult | Eff. Count |
|-----------------------|-------|-----------|------:|-----:|-----------:|
| T1_sheet_very_long     | SHEET | very_long |   106 |  3×  |        318 |
| T2_sheet_long          | SHEET | long      |   200 |  2×  |        400 |
| T3_helix_very_long     | HELIX | very_long |   150 |  3×  |        450 |
| T4_unusual_very_long   | UNUSUAL| very_long|    90 |  3×  |        270 |
| T5_helix_long          | HELIX | long      |   150 |  2×  |        300 |
| T6_sheet_medium        | SHEET | medium    |   100 |  1×  |        100 |
| T7_sheet_short         | SHEET | short     |   100 |  1×  |        100 |
| T8_sheet_medium_topoff | SHEET | medium    |   104 |  1×  |        104 |
| replay                 | mixed | mixed     |   200 |  1×  |        200 |
| **Total**              |       |           |**1200**|    |   **2,242**|

Oversampling is implemented via `WeightedRandomSampler` using per-sample weights
derived from the `tier` column of the training CSV. The 200 replay entries are tagged
with tier `"replay"` (weight=1).

---

## 4. Regularization

### L2 Pretrained-Weight Regularization (cross_convs)

Applied in Phase 2 and Phase 3 only (cross_convs unfrozen).

```
L_total = L_score_matching + λ × Σ_i ||θ_i − θ_i^pretrained||²
```

- **λ = 3e-4** (auto-set by `--v6-mode` when `--reg-lambda` is not specified)
- Applied to: `cross_convs.0`, `cross_convs.1`, `cross_convs.2`, `cross_convs.3`
- Pretrained weights loaded from `rapidock_global.pt` at training start

**Purpose:** Prevents catastrophic forgetting of general cross-complex interaction
geometry while still allowing specialization on long/very_long structures.

---

## 5. Validation Strategy

### Dataset
- **File:** `data/v6_val_200.csv`
- **Size:** 200 complexes (50 per length bucket)
- **Buckets:** short (pep_len=8), medium (9-12), long (13-19), very_long (≥20)
- **Source:** PepPC dirs not in bench300, gap-fill, curated_train, or replay set
- **Zero overlap** confirmed with bench300 (239 complexes) and gap-fill (1,000 complexes)

### Per-Epoch Val Metrics

Every epoch:
```
val_loss_overall   — trimmed-mean over full 200-complex val set
v6_val_short       — trimmed-mean on 50 short complexes (pep_len=8)
v6_val_medium      — trimmed-mean on 50 medium complexes (pep_len=9-12)
v6_val_long        — trimmed-mean on 50 long complexes (pep_len=13-19)
v6_val_very_long   — trimmed-mean on 50 very_long complexes (pep_len≥20)
```

EMA weights are used for all val evaluations (raw model weights for training).

### Guard Rails (Catastrophic Forgetting Detection)

Monitors short and medium bucket val_loss. Fires if:
- Bucket val_loss rises > 30% above its minimum seen for 3 consecutive post-warmup epochs
- **Action:** Print `[V6 GUARD RAIL ⚠]` alert with rollback guidance. No auto-stop.

### RMSD-Based Validation (Manual, Every 5 Epochs)

Not run per-epoch. Manual inference runs recommended at epochs 10, 15, 20, 25, 30, 35, 45
using `bench_very_long.csv` complexes (60 complexes, N=5 poses each).

---

## 6. Checkpointing

| Checkpoint file                        | Trigger                                              |
|---------------------------------------|------------------------------------------------------|
| `rapidock_finetuned_best.pt`           | Lowest overall val_loss (trimmed-mean, full 200)     |
| `rapidock_finetuned_best_long.pt`      | Lowest `v6_val_long` bucket loss                     |
| `rapidock_finetuned_best_very_long.pt` | Lowest `v6_val_very_long` bucket loss                |
| `rapidock_finetuned_best_combined.pt`  | Lowest `v6_val_long + v6_val_very_long` sum          |
| `rapidock_finetuned_epoch{N:03d}.pt`  | Every epoch ≥ 15; every `--save-every` epochs before |
| `rapidock_finetuned_final.pt`          | Always (end of training)                             |

**Primary inference model:** `best_very_long.pt` or `best_combined.pt`  
**Fallback (short/medium regression test):** `best.pt` overall

---

## 7. Training History Columns

Written to `<out_dir>/training_history.csv` at the end of training:

| Column              | Description                                              |
|--------------------|----------------------------------------------------------|
| `epoch`             | Epoch number                                             |
| `train_loss`        | Mean training loss (from train_epoch)                    |
| `val_loss`          | Overall trimmed-mean val loss (all 200 complexes)        |
| `val_raw_mean`      | Raw mean val loss (includes outliers)                    |
| `val_median`        | Median val loss                                          |
| `val_loss_std`      | Std of per-sample val losses                             |
| `val_max`           | Max single-sample val loss                               |
| `val_outliers`      | Count of samples with loss > 1000                        |
| `val_tr_norm_max`   | Max tr_pred norm in val (instability signal)             |
| `val_tr_norm_var`   | Variance of val tr_pred norms                            |
| `val_rot_norm_max`  | Max rot_pred norm in val                                 |
| `val_rot_norm_var`  | Variance of val rot_pred norms                           |
| `val_torbb_norm_max`| Max tor_bb norm in val                                   |
| `val_torsc_norm_mean`| Mean tor_sc norm in val                                 |
| `tr_norm_train`     | Mean tr_pred norm in training                            |
| `tr_norm_var`       | Variance of tr_pred norms (training diversity proxy)     |
| `rot_norm_train`    | Mean rot_pred norm in training                           |
| `rot_norm_var`      | Variance of rot_pred norms                               |
| `tor_bb_norm_train` | Mean tor_bb norm in training                             |
| `tor_bb_norm_max`   | Max tor_bb norm in training                              |
| `n_ok`              | Training samples processed without error                 |
| `n_total`           | Total training samples attempted                         |
| `lr`                | Current learning rate                                    |
| `v6_val_short`      | Val loss on short bucket (50 complexes, pep_len=8)       |
| `v6_val_medium`     | Val loss on medium bucket (50 complexes, pep_len=9-12)   |
| `v6_val_long`       | Val loss on long bucket (50 complexes, pep_len=13-19)    |
| `v6_val_very_long`  | Val loss on very_long bucket (50 complexes, pep_len≥20)  |

---

## 8. Effective Training Volume

| Phase   | Epochs | Epoch Size | Total Samples |
|---------|--------|----------:|-------------:|
| Phase 1 | 1–8    |     1,200 |        9,600 |
| Phase 2 | 9–35   |     2,242 |       60,534 |
| Phase 3 | 36–45  |     1,200 |       12,000 |
| **Total** |      |           |   **82,134** |

---

## 9. Loss Function

Score-matching MSE, 4-component weighted sum with equal component weights:

```
L = 0.25 × L_tr + 0.25 × L_rot + 0.25 × L_tor_bb + 0.25 × L_tor_sc
```

Where each component is the MSE between the model's predicted score vector
and the analytical score from the diffusion forward process.

With L2 reg (Phase 2+):
```
L_total = L + λ × Σ ||θ_cross_conv_i − θ_pretrained_i||²
```

---

## 10. ESM Embedding

ESM2 (facebook/esm2_t33_650M_UR50D) is fully frozen throughout all phases.
`--esm-device cpu` is the default (avoids VRAM contention with the GNN during training).
ESM embeddings are cached per-complex on first access.

---

## 11. Random Seeds

No explicit seed set for V6 (stochastic training is intentional for gradient diversity).
For reproducible val set construction: Python `random.seed(2026)`, `numpy.random.seed(2026)`.
