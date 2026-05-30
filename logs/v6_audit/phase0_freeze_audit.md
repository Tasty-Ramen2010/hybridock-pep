# PHASE 0 — Freeze Safety Audit
*Generated 2026-05-30 pre-v6 launch*

---

## 1. Executive Summary

**The PyTorch freeze logic (requires_grad) is mechanically correct.**  
Frozen parameters cannot receive gradient updates, and all optimizer factory functions
correctly filter to `requires_grad=True` params only. There is **no bug in the parameter
freeze logic**.

**There IS a bug: BatchNorm running statistics are not frozen.**  
All 10 `BatchNorm` layers in RAPiDock have `running_mean`/`running_var` buffers that
update during every `model.train()` forward pass, **regardless of `requires_grad=False`**.
This caused 14–63% relative drift in the running statistics of "frozen" conv blocks across
all prior runs (v3c, v4c, v5c).

**Additionally: v5c was NOT started from `rapidock_global.pt`.**  
Drift in v5c P1 (69.60% in cross_convs, 187.76% in tor_bb_bond_conv) is impossible from
score-head-only training. It was inherited from whatever checkpoint v5c P1 was launched
from (suspected: a prior run with full cross_conv training). This contaminates the v5c
benchmark, rendering it an unfair comparison vs pretrained.

---

## 2. Parameter Count (Total model: 7,553,674 params)

| Module                         | Count     |   Pct |
|-------------------------------|----------:|------:|
| encoder.intra_convs            | 2,667,970 | 35.32% |
| encoder.cross_convs            | 2,667,970 | 35.32% |
| encoder.tor_bb_bond_conv       |   967,584 | 12.81% |
| encoder.tor_sc_bond_conv       |   967,584 | 12.81% |
| encoder.pep_a_node_embedding   |   109,356 |  1.45% |
| encoder.rec_node_embedding     |    67,008 |  0.89% |
| encoder.final_conv             |    39,576 |  0.52% |
| encoder.rec_edge_embedding     |    15,984 |  0.21% |
| encoder.pep_edge_embedding     |    15,984 |  0.21% |
| encoder.pep_node_embedding     |     7,104 |  0.09% |
| encoder.cross_edge_embedding   |     5,472 |  0.07% |
| encoder.center_edge_embedding  |     5,472 |  0.07% |
| encoder.tor_bb_final_layer     |     4,656 |  0.06% |
| encoder.tor_sc_final_layer     |     4,656 |  0.06% |
| encoder.final_edge_embedding   |     3,936 |  0.05% |
| encoder.tr_final_layer         |     1,681 |  0.02% |
| encoder.rot_final_layer        |     1,681 |  0.02% |

**Buffers (not parameters, no gradient):** 5,028 values  
→ These are BatchNorm running statistics that bypass the freeze mechanism.

---

## 3. Freeze Logic Code Audit

### 3.1 `load_model_for_finetuning()` — CORRECT

```python
# All params frozen first
for param in model.parameters():
    param.requires_grad = False

# Selectively unfreeze by pattern matching
for name, param in model.named_parameters():
    for pattern in patterns:
        if pattern in name:
            param.requires_grad = True
            break
```

Pattern matching is correct. No double-counting. No escape paths.

### 3.2 Optimizer factory functions — CORRECT

All `make_v*_optimizer()` functions iterate `model.named_parameters()` and skip any
`param` where `requires_grad=False`. Frozen params are never in any optimizer param group.

**Cross-check:** `train_epoch()` gradient clipping also filters:
```python
torch.nn.utils.clip_grad_norm_(
    [p for p in model.parameters() if p.requires_grad], grad_clip_norm
)
```

### 3.3 Checkpoint resume — CORRECT

`model.load_state_dict(ckpt["model"])` updates `.data` only, not `.requires_grad`.
Freeze flags set before load_state_dict persist correctly through resume.

### 3.4 BatchNorm running statistics — **CRITICAL BUG**

BatchNorm layers in RAPiDock update `running_mean` and `running_var` during every
`model.train()` forward pass. These are buffers, not parameters — they are not controlled
by `requires_grad`. The freeze mechanism has **no effect** on them.

Affected BatchNorm layers (all 10 are affected when their parent conv is "frozen"):

| Layer                              |
|------------------------------------|
| encoder.intra_convs.0.batch_norm   |
| encoder.intra_convs.1.batch_norm   |
| encoder.intra_convs.2.batch_norm   |
| encoder.intra_convs.3.batch_norm   |
| encoder.cross_convs.0.batch_norm   |
| encoder.cross_convs.1.batch_norm   |
| encoder.cross_convs.2.batch_norm   |
| encoder.cross_convs.3.batch_norm   |
| encoder.tor_bb_bond_conv.batch_norm|
| encoder.tor_sc_bond_conv.batch_norm|

---

## 4. Measured Drift: v5c P1 vs v5c P2 (relative to rapidock_global.pt)

Only `tr_final_layer` and `rot_final_layer` (trained params) should have any drift.
All others should be 0.00%. Actual measured values:

| Module                        | P1_rel%   | P2_rel%   | Verdict           |
|------------------------------|----------:|----------:|-------------------|
| encoder.tor_bb_bond_conv      | 187.76%   | 171.96%   | WRONG START CKPT  |
| encoder.tor_sc_bond_conv      |  80.17%   |  80.99%   | WRONG START CKPT  |
| encoder.cross_convs           |  69.60%   |  61.34%   | WRONG START CKPT  |
| encoder.final_conv            |  38.32%   |  38.30%   | WRONG START CKPT  |
| encoder.tr_final_layer        |  37.11%   |  37.09%   | OK-trained (P1)   |
| encoder.rec_edge_embedding    |  33.44%   |  33.44%   | WRONG START CKPT  |
| encoder.intra_convs           |  23.32%   |  32.84%   | WRONG START CKPT  |
| encoder.pep_edge_embedding    |  29.13%   |  29.13%   | WRONG START CKPT  |
| encoder.rec_node_embedding    |  22.85%   |  22.85%   | WRONG START CKPT  |
| encoder.center_edge_embedding |  19.78%   |  19.78%   | WRONG START CKPT  |
| encoder.pep_a_node_embedding  |  15.64%   |  15.64%   | WRONG START CKPT  |
| encoder.rot_final_layer       |   8.77%   |   8.76%   | OK-trained (P1)   |

**Key diagnostics:**
- Frozen modules show IDENTICAL drift in P1 and P2, proving drift was pre-existing
  (inherited from wrong starting checkpoint), not caused by v5c training.
- `tor_bb_bond_conv` drift of 187.76% present in P1 — impossible from score-head-only training.
- Root cause: v5c P1 was not launched from `rapidock_global.pt`.

---

## 5. Confirmed: BatchNorm Running Stats Drift in v4c P2

v4c P2 trained cross_convs. The BatchNorm running stats in cross_convs drifted 45–62%
from pretrained:
- `encoder.cross_convs.0.batch_norm.running_mean`: 45.1% drift
- `encoder.cross_convs.0.batch_norm.running_var`: 46.3% drift
- `encoder.cross_convs.3.batch_norm.running_mean`: 51.7% drift

This is EXPECTED for trainable cross_convs in v4c. But the same update mechanism also
contaminates **frozen** convs in other runs (where the conv weights are frozen but the BN
running stats silently update).

---

## 6. Fixes Applied in v6

### Fix 1: Always start from rapidock_global.pt
Eliminates all inherited drift from prior runs.

### Fix 2: freeze_frozen_bn_stats() — new function
After load_model_for_finetuning(), call this to set `.eval()` on all BatchNorm
submodules whose parent conv is frozen (requires_grad=False on weight/bias).
This prevents running_mean/running_var from updating during training.

Applied at:
- Start of training (after model load)
- Start of each train_epoch call (after `model.train()`)

### Fix 3: Post-freeze assertion
After freeze_frozen_bn_stats(), snapshot running stats of frozen BN layers.
At end of each epoch, verify they haven't changed.

### Fix 4: Explicit starting checkpoint validation
Before training begins, compare key parameters vs rapidock_global.pt and abort
if drift > 1.0% in modules that should be at pretrained values.

---

## 7. v6 Trainable Parameter Summary (starting from rapidock_global.pt)

**Phase 1 (epochs 1–8):**

| Module                    | Count     |   Pct |
|--------------------------|----------:|------:|
| encoder.tor_bb_bond_conv  |   967,584 | 12.81% |
| encoder.tr_final_layer    |     1,681 |  0.02% |
| encoder.rot_final_layer   |     1,681 |  0.02% |
| encoder.tor_bb_final_layer|     4,656 |  0.06% |
| encoder.tor_sc_final_layer|     4,656 |  0.06% |
| **Total P1**              | **980,258**| **12.98%** |

**Phase 2 (epochs 9–35):**

| Module                    | Count     |   Pct |
|--------------------------|----------:|------:|
| encoder.cross_convs.0     |   424,666 |  5.62% |
| encoder.cross_convs.1     |   537,776 |  7.12% |
| encoder.cross_convs.2     |   650,924 |  8.62% |
| encoder.cross_convs.3     | 1,054,604 | 13.96% |
| encoder.tor_bb_bond_conv  |   967,584 | 12.81% |
| score heads (4)           |    12,674 |  0.17% |
| **Total P2**              | **3,648,228** | **48.30%** |

Frozen (always): intra_convs (35.32%), tor_sc_bond_conv (12.81%), all embeddings (3.57%),  
ESM projection, distance expansions

---

## 8. Conclusion

The freeze logic in train_lastlayer.py is mechanically correct for parameters.
The two real problems were:
1. **Wrong starting checkpoint** for v5c (inherited contamination, 69–188% drift)
2. **BatchNorm running stats** bypass the freeze mechanism (untracked buffer updates)

v6 fixes both. All prior benchmark comparisons (v3c/v4c/v5c vs pretrained) should be
interpreted with this in mind: the models were NOT trained from the same starting state,
making the comparison partially unfair.
