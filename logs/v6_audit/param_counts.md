# V6 Parameter Count Summary
*Generated 2026-05-30 from rapidock_global.pt*

---

## Model: RAPiDock-Reloaded (CGTensorProductEquivariantModel)

**Checkpoint:** `third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_global.pt`

### Total: 7,553,674 learnable parameters + 5,028 buffer values

Buffer values = BatchNorm running_mean / running_var for 10 BN layers.
Buffers are NOT parameters — they are not controlled by `requires_grad`.
The freeze fix (`freeze_frozen_bn_stats()`) targets these buffers explicitly.

---

## Full Module Table

| Module (encoder.*)                    | Count     |   Pct   |
|---------------------------------------|----------:|--------:|
| intra_convs.3                         | 1,054,996 | 13.96%  |
| cross_convs.3                         | 1,054,996 | 13.96%  |
| tor_bb_bond_conv.fc                   |   967,440 | 12.80%  |
| tor_sc_bond_conv.fc                   |   967,440 | 12.80%  |
| intra_convs.2                         |   651,316 |  8.62%  |
| cross_convs.2                         |   651,316 |  8.62%  |
| intra_convs.1                         |   538,072 |  7.12%  |
| cross_convs.1                         |   538,072 |  7.12%  |
| intra_convs.0                         |   424,850 |  5.62%  |
| cross_convs.0                         |   424,850 |  5.62%  |
| rec_node_embedding.lm_embedding_layer |    63,792 |  0.84%  |
| pep_a_node_embedding.atom_embedding_list|  55,380 |  0.73%  |
| pep_a_node_embedding.final_layer      |    48,828 |  0.65%  |
| final_conv.fc                         |    39,576 |  0.52%  |
| rec_edge_embedding.edge_embedding     |     9,312 |  0.12%  |
| pep_edge_embedding.edge_embedding     |     9,312 |  0.12%  |
| pep_a_node_embedding.linear           |     5,148 |  0.07%  |
| pep_node_embedding.amino_ebd          |     5,040 |  0.07%  |
| rec_edge_embedding.feature_ebd        |     4,992 |  0.07%  |
| pep_edge_embedding.feature_ebd        |     4,992 |  0.07%  |
| tor_bb_final_layer.0                  |     4,608 |  0.06%  |
| tor_sc_final_layer.0                  |     4,608 |  0.06%  |
| cross_edge_embedding.0                |     3,120 |  0.04%  |
| center_edge_embedding.0               |     3,120 |  0.04%  |
| cross_edge_embedding.3                |     2,352 |  0.03%  |
| center_edge_embedding.3               |     2,352 |  0.03%  |
| final_edge_embedding.3                |     2,352 |  0.03%  |
| tr_final_layer.0                      |     1,632 |  0.02%  |
| rot_final_layer.0                     |     1,632 |  0.02%  |
| *(sigma / distance embeddings)*       |   ~5,800  |  0.08%  |
| *(tp weight/mask tensors)*            |     1,840 |  0.02%  |
| **TOTAL**                             | **7,553,674** | **100%** |

---

## V6 Trainable by Phase

### Phase 1 (Epochs 1–8): 980,498 params (12.97%)

| Module                         | Count   |   Pct  |
|-------------------------------|--------:|-------:|
| encoder.tor_bb_bond_conv       | 967,728 | 12.80% |
| encoder.tr_final_layer         |   1,681 |  0.02% |
| encoder.rot_final_layer        |   1,681 |  0.02% |
| encoder.tor_bb_final_layer     |   4,656 |  0.06% |
| encoder.tor_sc_final_layer     |   4,656 |  0.06% |
| **P1 Total**                   |**980,402**|**12.97%**|

*Note: `tor_bb_bond_conv` includes fc (967,440) + batch_norm params (288) = 967,728.*

### Phase 2 (Epochs 9–35): 3,649,634 params (48.29%)

Adds cross_convs to Phase 1 trainable set:

| Module                    | Count     |   Pct  |
|--------------------------|----------:|-------:|
| encoder.cross_convs.0     |   424,850 |  5.62% |
| encoder.cross_convs.1     |   538,072 |  7.12% |
| encoder.cross_convs.2     |   651,316 |  8.62% |
| encoder.cross_convs.3     | 1,054,996 | 13.96% |
| Phase 1 params            |   980,402 | 12.97% |
| **P2 Total**              |**3,649,634**|**48.30%**|

### Phase 3 (Epochs 36–45)

Same trainable set as Phase 2. No new modules unfrozen.

---

## Always Frozen: 3,904,068 params (51.70%)

| Module                         | Count     |   Pct  |
|-------------------------------|----------:|-------:|
| encoder.intra_convs (all 4)    | 2,669,234 | 35.32% |
| encoder.tor_sc_bond_conv       |   967,728 | 12.81% |
| encoder.pep_a_node_embedding   |   109,356 |  1.45% |
| encoder.rec_node_embedding     |    67,008 |  0.89% |
| encoder.final_conv             |    39,576 |  0.52% |
| encoder.rec_edge_embedding     |    15,984 |  0.21% |
| encoder.pep_edge_embedding     |    15,984 |  0.21% |
| encoder.pep_node_embedding     |     7,104 |  0.09% |
| encoder.cross_edge_embedding   |     5,472 |  0.07% |
| encoder.center_edge_embedding  |     5,472 |  0.07% |
| encoder.final_edge_embedding   |     3,936 |  0.05% |
| *(other embeddings)*           |   ~14,220 |  0.19% |
| **Frozen Total**               |**~3,904,068**|**51.70%**|

ESM2 (facebook/esm2_t33_650M_UR50D): ~650M params, fully frozen and on CPU.
Not counted in the model parameter total above (separate encoder module).

---

## BatchNorm Layers (Buffer Values, NOT Parameters)

These 10 BN layers exist in the model but their `running_mean` / `running_var`
are registered as *buffers* — not parameters. They update during `model.train()`
forward passes regardless of `requires_grad`.

**V6 Fix:** `freeze_frozen_bn_stats()` calls `.eval()` on any BN layer whose parent
conv module has `requires_grad=False` on weight/bias. This is re-applied at the start
of every training epoch (since `model.train()` resets all modes).

| BN Layer                              | Affected in P1 | Affected in P2 |
|--------------------------------------|:--------------:|:--------------:|
| encoder.intra_convs.0.batch_norm      | frozen (eval)  | frozen (eval)  |
| encoder.intra_convs.1.batch_norm      | frozen (eval)  | frozen (eval)  |
| encoder.intra_convs.2.batch_norm      | frozen (eval)  | frozen (eval)  |
| encoder.intra_convs.3.batch_norm      | frozen (eval)  | frozen (eval)  |
| encoder.cross_convs.0.batch_norm      | frozen (eval)  | **train mode** |
| encoder.cross_convs.1.batch_norm      | frozen (eval)  | **train mode** |
| encoder.cross_convs.2.batch_norm      | frozen (eval)  | **train mode** |
| encoder.cross_convs.3.batch_norm      | frozen (eval)  | **train mode** |
| encoder.tor_bb_bond_conv.batch_norm   | **train mode** | **train mode** |
| encoder.tor_sc_bond_conv.batch_norm   | frozen (eval)  | frozen (eval)  |

*"train mode"* = BN running stats update normally (expected behavior for trainable conv).  
*"frozen (eval)"* = BN set to `.eval()` mode; running stats are locked.
