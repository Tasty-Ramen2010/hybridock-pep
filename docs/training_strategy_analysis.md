# RAPiDock Fine-tuning Strategy Analysis
**Date**: 2026-05-27  
**Context**: Fine-tuning pre-trained RAPiDock on PepPC (3.8K natural complexes) + PepPC-F (14.9K protein fragments) + RefPepDB/recent (0.8K) = 17.5K training complexes  
**Applies to**: `train_lastlayer.py`, `chain_training_phases.sh`, commits `57e0303` and `069e8bb`

---

## 1. What Each Model Component Encodes

Understanding gradient sensitivity requires knowing what information lives at each depth.

### Score Heads (`tr_final_layer`, `rot_final_layer`, `tor_bb_final_layer`, `tor_sc_final_layer`)
Linear projections from geometric representation space → diffusion score (gradient of log probability). These are **calibrated to the training data distribution**. When the data distribution changes (from original RAPiDock training data to PepPC-F fragments), the score heads are the first component to become wrong. They encode the *posterior* over good binding modes, not physics.

### Output Convolutions (`final_conv`, `tor_bb_bond_conv`, `tor_sc_bond_conv`)
Final equivariant aggregation step before the linear heads. More task-specific than the conv backbone but less so than the heads. Safe to adapt early.

### Final Embeddings (`center_edge_embedding`, `pep_a_node_embedding`, `final_edge_embedding`)
Output-side node/edge embeddings — map learned features into the representation space consumed by score heads. These sit at the boundary between geometric representation and score space. They need to adapt when input types change (e.g., protein fragments vs. natural peptides have different backbone geometry distributions).

### `cross_convs.2/3` — Receptor-Peptide Interaction Encoding
These layers encode **how the peptide sees the receptor**. The cross-conv aggregates peptide node features weighted by receptor neighbourhood features. This is where receptor family specificity is encoded.

**Why these need to adapt for PepPC/PepPC-F**: PepPC-F has fragments binding to many diverse receptor families (kinases, PDZ domains, WD40 repeats, leucine-rich repeats...). The original RAPiDock model's cross-conv layers are calibrated to whatever receptor families were in the original training set. Adapting `cross_convs.2/3` teaches the model "what does a productive receptor-peptide interaction look like in these new receptor families."

### `intra_convs.2/3` — Peptide-Internal Geometry Encoding
These layers encode the peptide's **own geometric structure** — bond angles, torsion constraints, chirality, local chemical environment. The physics of a peptide bond angle is identical in a kinase binding pocket and a PDZ domain binding groove. These layers encode **dataset-invariant physics**, not receptor-specific information.

**Key insight**: Applying gradients to `intra_convs.2` based on receptor-diverse data is largely noise injection — the peptide geometry physics it encodes doesn't change across receptor families. The gradients will be inconsistent across examples (different receptors don't agree on what "correct" peptide-internal geometry looks like) and will gradually erode the physically correct representations.

### `cross_convs.0/1`, `intra_convs.0/1` — Foundational Geometric Features
Encode atomic-level geometry: atom types, bond types, distances, angles. These represent invariances that exist in physics itself. They should never be touched in fine-tuning unless you have millions of training examples and are deliberately re-learning the physics from scratch.

### ESM-650M (Receptor Language Model)
Trained on 250M protein sequences from UniRef50. Contains the best available per-residue protein representation. With 17.5K training complexes vs. 250M training sequences, the ratio is ~14,000:1 in favour of the pre-trained ESM representations. **Keep frozen always.**

---

## 2. RAPiDock vs. DiffPepDock: Why the Training Paradigm Difference Matters

RAPiDock and DiffPepDock differ in:
- Diffusion noise schedule (different temperature parameterisation)
- Loss function weighting across timesteps
- Data preprocessing (how poses are extracted and noise is applied)

This creates a crucial distinction for fine-tuning:

**Scenario A: Same loss, new data** (what we're doing)  
→ Shifting the posterior (what binding modes are real) while keeping the prior (what poses are physically possible)  
→ Late layers + heads need adaptation; inner layers need much less  
→ Conservative unfreezing is correct

**Scenario B: Different loss, same or new data** (e.g., replacing RAPiDock's score matching with DiffPepDock's formulation)  
→ The entire gradient landscape changes; all layer calibrations become wrong  
→ Would need aggressive unfreezing close to full retraining from scratch

Since we're using RAPiDock's loss function on new data (Scenario A), the inner equivariant layers' calibration is approximately still correct — they were calibrated to produce features useful for RAPiDock's score matching, and that loss hasn't changed. Only the *distribution of good binding modes* has changed, which is captured by the late layers and score heads.

---

## 3. Why `cross_convs` Should Be Prioritised Over `intra_convs`

For a dataset of 17.5K complexes spanning diverse receptor families:

**Cross-conv gradient signal**: Each training example says "for this specific receptor family and binding site geometry, this peptide pose has score X." The cross-conv layers see different gradient signals for different receptor families → they can learn receptor-specific interaction patterns → clear learning signal.

**Intra-conv gradient signal**: Each training example says "for this peptide conformation, internal geometry score Y." The peptide's internal geometry constraints are the same regardless of receptor → gradient signals are consistent across examples → but they're also already learned correctly from the original training → low information gain, non-zero update cost.

The net effect of applying equal LR to both: the intra-conv layers receive a significant gradient signal that doesn't add information but does perturb physically correct representations. The cross-conv layers receive genuinely informative gradient signal about new receptor families.

**Resolution**: `cross_convs.3 > cross_convs.2 > intra_convs.3 > intra_convs.2` in terms of update priority. In Phase 2, we unfreeze in this priority order.

---

## 4. `tor_bb_final_layer` Is the Highest-Value Score Head

The four score heads are NOT equally important for this dataset:

| Head | Encodes | Dataset Sensitivity |
|---|---|---|
| `tr_final_layer` | Where in space the peptide goes (translation) | Medium — new receptor pockets have different centres |
| `rot_final_layer` | How the peptide is oriented (rotation) | Medium — new receptor geometries change optimal orientations |
| `tor_bb_final_layer` | Backbone torsion preference (phi/psi) | **HIGH** — PepPC-F fragments have structured backbone conformations (helices mid-fold, β-strands at interfaces) vs. natural peptide bias toward extended/helical |
| `tor_sc_final_layer` | Sidechain rotamer preference | Low-medium — rotamer preferences follow from backbone; physically valid rotamers are universal |

The `tor_bb_final_layer` needs to unlearn the natural-peptide phi/psi distribution bias and learn that fragments can adopt any structured backbone conformation. This is the most important single parameter block for structural diversity.

---

## 5. Phase 3 Full-Retrain Risk Analysis

Original Phase 3: 200 epochs, LR=1e-4, all 7.5M parameters.

**Why 1e-4 is too high for full retrain after partial fine-tuning**:

After Phase 1 (30 epochs, 2.1M params adapted) and Phase 2 (50 epochs, 5.5M params adapted), the model is already near its optimum for the unfrozen parameters. The inner layers (`intra_convs.0/1`, `cross_convs.0/1`) are still at their pre-trained values.

When Phase 3 unfreezes the inner layers and applies LR=1e-4:
- Inner layer gradients are computed from the new data distribution
- At LR=1e-4, these gradients accumulate as 100K+ parameter updates over 200 epochs
- Each update moves the inner layers away from their physically calibrated representations
- The data doesn't have enough signal to re-learn what the inner layers already know correctly
- Result: gradual erosion of physics-grounded representations → worse generalisation at inference

**Why cosine schedule is better than ReduceLROnPlateau for Phase 3**:

- Plateau scheduler: LR stays high until loss stops improving, then drops suddenly. In Phase 3, loss may plateau early (the model was already partially adapted), causing extended high-LR periods that erode inner layers.
- Cosine decay: Deterministic monotone decrease from `base_lr` to `min_lr`. The inner layers see a smoothly decreasing gradient signal that approaches zero, giving them minimum perturbation while still allowing global fine-tuning.

**Chosen parameters**: LR=2e-5 (5× lower than original), 100 epochs (not 200), cosine 2e-5→1e-7, warmup=10.

---

## 6. Dataset Scale → Unfreezing Depth Relationship

For reference when designing future training runs:

| Training examples | Safe max unfreeze depth | Notes |
|---|---|---|
| < 1,000 | Score heads only | Classic few-shot fine-tuning |
| 1,000–5,000 | Score heads + output convs + embeddings (Phase 1) | Original RAPiDock training scale |
| 5,000–20,000 | + last 2 cross-conv layers, last 1 intra-conv layer (Phase 2) | PepPC + PepPC-F scale |
| 20,000–100,000 | Full model fine-tune with low LR and cosine schedule | Large-scale dataset |
| > 100,000 | Full retrain (treat pre-trained as weight initialisation only) | DiffPepDock/RoseTTAFold scale |

With 17.5K examples, we're at the transition between "Phase 2 safe" and "Phase 3 risky" — which is exactly why Phase 3 needs LR=2e-5 (not 1e-4) and a bounded epoch count (not 200).

---

## 7. Implemented Phase Parameters (as of commit `069e8bb`)

```
Phase 1: unfreeze_phase=1  lr=1e-4  epochs=30  warmup=0  schedule=plateau
         Params: score heads, output convs, final embeddings (27.9% = 2.1M)

Phase 2: unfreeze_phase=2  lr=2e-5  epochs=50  warmup=5  schedule=plateau
         Params: P1 + cross_convs.3, cross_convs.2, intra_convs.3
         (intra_convs.2 deliberately frozen — held for Phase 3 full-model context)
         ~70% = 5.3M params

Phase 3: unfreeze_phase=3  lr=2e-5  epochs=100  warmup=10  schedule=cosine→1e-7
         Params: all 7,553,674 parameters
```

---

## 8. What to Monitor

**Phase 1 health check** (epoch 1):
- `n_ok/n_total` > 50% (expect 70–90%)
- `train_loss` between 0.1 and 5.0 (score matching loss)
- `val_loss` tracking `train_loss` (not zero, not 100× larger)

**Phase 2 health check** (epochs 1–5):
- Loss should NOT spike more than 2× above Phase 1 final loss in epoch 1
- If it does, LR 2e-5 may still be too high → consider 1e-5
- `val_loss` should eventually (by epoch 15–20) fall below Phase 1 best

**Phase 3 health check**:
- Loss at epoch 1 should be approximately equal to Phase 2 final loss (smooth handoff)
- Cosine LR at epoch 50: should be ~1e-5 (halfway between 2e-5 and 1e-7)
- If `val_loss` starts increasing after epoch 30–40, inner layers may be over-adapting

**Failure modes to watch**:
- Phase 3 `val_loss` < Phase 2 `val_loss` by > 20%: likely catastrophic forgetting of inner layers
- Cluster < 10 of 100 poses on held-out test: score function collapsed
- α recalibrates to > 1.2 or < 0.2 kcal/mol/residue after training: scoring pipeline broken

---

*Analysis conducted 2026-05-27. Read this file when: (a) re-designing training phases, (b) debugging Phase 2/3 loss spikes, (c) scaling to larger datasets.*
