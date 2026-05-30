"""Last-layer fine-tuning for RAPiDock — Phase 1 / 2 / 3.

Freezes all backbone layers and fine-tunes only the score heads (Phase 1),
or additionally unfreezes the last equivariant conv block (Phase 2),
or unfreezes ALL layers (Phase 3 = full retraining from pre-trained init).

Uses score-matching loss (MSE between predicted and analytical score) with the
same four-way weighted sum used during original RAPiDock training.

Runs in rapidock-env (Python 3.10). Do NOT use Python 3.11+ syntax.

Bug fixes applied (May 27 2026):
  - compute_loss uses preds["key"] dict access (not tuple unpacking → silent fail)
  - build_dataset uses conformation_type='E' (not None → KeyError)
  - val_epoch exceptions are caught and counted
  - sched_metric defined before use at end of training loop

Usage (from repo root, run in rapidock env):
    conda run --no-capture-output -n rapidock \\
        python third_party/RAPiDock_finetuned/train_lastlayer.py \\
            --train-csv  datasets/training_formatted_peppc/combined_train.csv \\
            --val-csv    datasets/training_formatted_peppc/combined_val.csv \\
            --checkpoint third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt \\
            --output-dir third_party/RAPiDock_finetuned/finetune_peppc_phase1/ \\
            --unfreeze-phase 1 --n-epochs 30 --lr 1e-4

    # Phase 2 (deep backbone unfreezing, start from Phase 1 best):
    conda run ... train_lastlayer.py \\
            --checkpoint .../finetune_peppc_phase1/rapidock_finetuned_best.pt \\
            --output-dir .../finetune_peppc_phase2/ \\
            --unfreeze-phase 2 --n-epochs 50 --lr 5e-5 --warmup-epochs 3

    # Dry-run (processes 10 samples, reports n_ok, then exits):
    ... --dry-run
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback as _traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import pandas as pd

# Add this directory to sys.path so RAPiDock modules resolve
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from argparse import Namespace
from utils.utils import get_model, ExponentialMovingAverage
from utils.transform import NoiseTransform
from utils.inference_utils import InferenceDataset


# ---------------------------------------------------------------------------
# Unfreeze pattern sets
# ---------------------------------------------------------------------------

# Phase 1 — score output heads + their direct feeding layers (~3–5% of params)
_UNFREEZE_PATTERNS_P1 = [
    # Scalar magnitude scalers
    "tr_final_layer",
    "rot_final_layer",
    "tor_bb_final_layer",
    "tor_sc_final_layer",
    # Equivariant convs that set prediction direction
    "final_conv",
    "tor_bb_bond_conv",
    "tor_sc_bond_conv",
    # Edge/node embeddings feeding those convolutions
    "center_edge_embedding",
    "pep_a_node_embedding",
    "final_edge_embedding",
]

# Phase 2 — additionally unfreeze the outermost equivariant conv layers.
# Verified against get_model() output (May 27 2026):
#   encoder.intra_convs.{0,1,2,3}.fc.{0,3}.{weight,bias}  + batch_norm.{weight,bias}
#   encoder.cross_convs.{0,1,2,3}.fc.{0,3}.{weight,bias}  + batch_norm.{weight,bias}
# "iegnn_interaction", "pep_a_node_norm", "rec_node_norm" do NOT exist in this model.
# "fc" is intentionally excluded: it matches ALL conv blocks (too broad).
#
# Cross-conv priority over intra-conv (analysis May 27 2026):
#   cross_convs encode receptor-peptide interactions → must adapt for diverse receptor families
#   intra_convs encode peptide-internal geometry → encodes universal physics, less data-dependent
#   cross_convs.2/3 adapt to new receptor families in PepPC/PepPC-F
#   intra_convs.3 (outermost only) is safe to update; encodes near-output peptide context
#   intra_convs.2 intentionally EXCLUDED — close enough to core physics; let Phase 3 handle it
#   with full-model context and lower LR rather than forcing it here
_UNFREEZE_PATTERNS_P2 = _UNFREEZE_PATTERNS_P1 + [
    "cross_convs.3",   # outermost receptor-peptide interaction layer — highest priority
    "cross_convs.2",   # 2nd outermost cross-conv — diverse receptor adaptation
    "intra_convs.3",   # outermost peptide-internal conv — safe near-output update
    # intra_convs.2 deliberately frozen here: encodes peptide geometry physics
    # that is dataset-invariant; adapts in Phase 3 under full-model gradient context
]

# Phase 3 — unfreeze everything (full retraining from pre-trained init)
# (handled by setting requires_grad=True for ALL params)

# ---------------------------------------------------------------------------
# V3 mode: controlled specialization — receptor adaptation + diversity preservation
# ---------------------------------------------------------------------------
# Design rationale (May 2026):
#   Prior runs (v1/v2/v2b) showed: specialization improved PepPC val loss but
#   degraded generalization (pretrained beats finetuned 7/8 benchmark complexes),
#   collapsed pose diversity (pretrained 71% hit-rate → finetuned 17–34%),
#   and caused score-field instability spikes. V3 addresses this with:
#     1. Tighter unfreezing: cross_convs only in P1 (intra_convs.0/1/2 frozen always)
#     2. intra_convs.3 allowed in P2 but at 0.15× LR — minimal drift
#     3. ESM projection layer always frozen — LM representations transfer well
#     4. Weak L2 regularization toward pretrained weights for frozen-pattern layers
#     5. Separate EMA decay schedules per phase (0.9995 → 0.9997 → 0.9999)
#        to slow EMA integration as the model stabilizes
#     6. Diversity monitoring: score variance + oscillation amplitude per epoch

# V3 Phase 1: score heads + cross_convs.2/3 ONLY (intra_convs entirely frozen)
#   Goal: adapt receptor-recognition cross-attention without touching peptide geometry
_UNFREEZE_PATTERNS_V3P1 = [
    "tr_final_layer",
    "rot_final_layer",
    "tor_bb_final_layer",
    "tor_sc_final_layer",
    "final_conv",
    "tor_bb_bond_conv",
    "tor_sc_bond_conv",
    "center_edge_embedding",
    "pep_a_node_embedding",
    "final_edge_embedding",
    "cross_convs.3",   # outermost receptor-peptide interaction layer
    "cross_convs.2",   # 2nd outermost cross-conv
    # intra_convs.*: all frozen — peptide-physics priors preserved entirely in P1
]

# V3 Phase 2: same as P1 + intra_convs.3 at very low LR (0.15×)
#   Allows outermost peptide-context layer to adapt to new receptor environments
#   without destabilizing deeper geometry priors in intra_convs.0/1/2
_UNFREEZE_PATTERNS_V3P2 = _UNFREEZE_PATTERNS_V3P1 + [
    "intra_convs.3",   # outermost peptide conv — minimal LR (0.15× via optimizer)
    # intra_convs.0/1/2: still frozen — core geometry priors untouched
]

# ---------------------------------------------------------------------------
# V4 mode: pure receptor-interaction adaptation — cross_convs ONLY, no intra_convs
# ---------------------------------------------------------------------------
# P1 = V3P1 (score heads + cross_convs.2/3)
# P2 = same as P1 — intra_convs.3 stays frozen throughout; 2-tier diff LR
#      cross_convs.3=1.0×, cross_convs.2=0.7× (make_v3p2_optimizer auto-degrades
#      to 2-tier when intra_convs.3 is frozen)
# P3 = full minus ESM (same as v3)
# No pretrained-weight regularization — cleaner ablation vs v3

# V4 reuses _UNFREEZE_PATTERNS_V3P1 for both P1 AND P2.

# ---------------------------------------------------------------------------
# V5 mode: ultra-conservative adaptation — minimal specialization
# ---------------------------------------------------------------------------
# P1 = score heads ONLY (= standard _UNFREEZE_PATTERNS_P1)
#   Goal: calibrate score outputs to PepPC scale without touching conv geometry
# P2 = score heads + cross_convs.3 ONLY
#   Adds one cross-conv ring at single uniform LR (no differential)
# P3 = full minus ESM; AGGRESSIVE layerwise decay (0.3/0.1/0.02 vs 0.5/0.2/0.05)
#   Hypothesis: pretrained prior is so strong that even mild late-epoch drift hurts
# Lower LR + higher grad clip conservatism throughout
_UNFREEZE_PATTERNS_V5P2 = _UNFREEZE_PATTERNS_P1 + [
    "cross_convs.3",   # single outermost cross-conv — minimal receptor adaptation
    # cross_convs.0/1/2: still frozen — preserve stable intermediate representations
    # intra_convs.*: entirely frozen throughout P1/P2
]

# ---------------------------------------------------------------------------
# V3B mode: stable controlled specialization — cosine in ALL phases, adaptive
#           spike LR, conservative progressive unfreezing
# ---------------------------------------------------------------------------
# Design rationale (May 2026):
#   v3 showed score-field instability from plateau+fixed-LR undamped oscillations.
#   v3b fixes this with cosine in ALL phases (deterministic LR decay guarantees
#   damping), tighter grad clip (0.5), and adaptive spike LR reduction.
#
# P1: score heads + output layers + cross_convs.3 ONLY
#   Same as V5-P2: just the outermost cross-conv ring; all intra_convs frozen
# P2: adds cross_convs.2 at 0.6× LR (gentle progressive unfreezing)
# P3: full except ESM; standard layerwise LR decay (0.5/0.2/0.05)
#
# Critical stability features:
#   - Cosine LR schedule in ALL phases (no plateau scheduler)
#   - Adaptive spike LR: on val tr_pred norm spike >10×, reduce LR 0.5× for 2 epochs
#   - grad_clip=0.5 throughout (tighter than v3's 1.0)
#   - EMA decay 0.9999 from P1 (slower EMA integration)

# V3B Phase 1: score heads + output layers + cross_convs.3 ONLY
# Same unfreeze set as V5P2 — identical patterns, different LR/schedule philosophy
_UNFREEZE_PATTERNS_V3BP1 = list(_UNFREEZE_PATTERNS_V5P2)

# V3B Phase 2: adds cross_convs.2 at 0.6× LR
# 2-tier differential LR via make_v3bp2_optimizer
_UNFREEZE_PATTERNS_V3BP2 = _UNFREEZE_PATTERNS_V3BP1 + [
    "cross_convs.2",   # 2nd outermost receptor ring — progressive specialisation
    # cross_convs.0/1: still frozen — deep receptor representation priors preserved
    # intra_convs.*: entirely frozen throughout P1/P2
]

# ---------------------------------------------------------------------------
# V5N mode: ultra-conservative manifold preservation — minimal adaptation
# ---------------------------------------------------------------------------
# Core philosophy: the pretrained RAPiDock prior is already highly optimized.
# Previous runs may have reduced exploration diversity by over-energizing the
# score field. V5N tests whether extremely gentle adaptation preserves the
# pretrained diffusion manifold while applying only minimal controlled biasing.
#
# P1: score heads + output convolution layers ONLY
#   Even more conservative than V5-P1 (which uses full _UNFREEZE_PATTERNS_P1
#   including embedding layers). V5N-P1 excludes all embeddings — no drift in
#   center_edge_embedding or pep_a_node_embedding.
#   Goal: calibrate score output magnitudes without touching conv geometry at all.
#
# P2: adds cross_convs.3 ONLY — one outermost receptor ring at uniform LR
#   Same unfreeze as V5-P2 but at lower LR (1e-6 vs V5's higher LR).
#   Goal: minimal receptor recognition adaptation, nothing else.
#
# P3: full except ESM; ultra-conservative layerwise 1.0/0.25/0.08/0.02
#   Heads stay at full speed. Late convs at 0.25×. Middle at 0.08×.
#   Early equivariant layers at 0.02× — essentially frozen in place.
#   Goal: let the score field adapt slightly to new PepPC distribution while
#   keeping equivariant geometry layers near the pretrained prior.
#
# Shared stability stack (same as v3b/v4n):
#   - Cosine LR schedule ALL phases (no plateau)
#   - grad_clip=0.5
#   - EMA decay=0.9999 from P1
#   - Adaptive spike LR: halve for 2 epochs on val tr_norm spike >10×
#   - EMA skip: pause EMA updates for 2 epochs after a spike (v5n only)
#     Rationale: spike epoch weights are unstable — don't average them into EMA

# V5N Phase 1: score heads + output convolution layers ONLY
# Excludes center_edge_embedding, pep_a_node_embedding, final_edge_embedding
# (these encode receptor/peptide context — freeze for maximal manifold preservation)
_UNFREEZE_PATTERNS_V5NP1 = [
    # Score prediction heads — scalar magnitude scalers
    "tr_final_layer",
    "rot_final_layer",
    "tor_bb_final_layer",
    "tor_sc_final_layer",
    # Output equivariant convolutions feeding the heads
    "final_conv",
    "tor_bb_bond_conv",
    "tor_sc_bond_conv",
    # Deliberately EXCLUDED vs _UNFREEZE_PATTERNS_P1:
    #   center_edge_embedding, pep_a_node_embedding, final_edge_embedding
    #   — keep ALL embeddings frozen; score-head calibration only
]

# V5N Phase 2: adds cross_convs.3 at uniform LR
# Same conv unfreeze as V5-P2 / V3B-P1, but P1 here is even smaller
_UNFREEZE_PATTERNS_V5NP2 = _UNFREEZE_PATTERNS_V5NP1 + [
    "cross_convs.3",   # outermost receptor-peptide interaction layer
    # cross_convs.0/1/2: frozen throughout P1/P2 — all intermediate representations preserved
    # intra_convs.*: entirely frozen in P1/P2 — peptide geometry priors untouched
]

# ---------------------------------------------------------------------------
# V3C mode: minimal low-energy recalibration — MONOTONIC decay, no rebounds
# ---------------------------------------------------------------------------
# Core philosophy (May 2026 analysis):
#   All previous runs confirmed the pretrained RAPiDock manifold is near-optimal
#   for exploration diversity.  Finetuning consistently collapses diversity (0.71→0.17–0.35)
#   while barely improving RMSD.  v3b/cosine showed that LR rebounds re-inject instability
#   energy cyclically — the stable operating window is ~3e-6 to 5e-6 and useful
#   adaptation ends very early (ep4–9).
#
#   V3C hypothesis: the ONLY safe adaptation is one that:
#     1. Never increases LR after warmup (monotone exponential decay)
#     2. Permanently reduces LR on any norm spike (no cooldown-then-resume)
#     3. Touches only score heads + output convs in P1 (calibrate output scale)
#     4. Adds cross_convs.3 at 0.1× in P2 (minimal receptor recognition shift)
#     5. No phase 3 — preserves 100% of equivariant geometry priors
#
# P1: score heads + output convolution layers ONLY  (identical to V5N-P1)
#   - No embeddings, no cross_convs, no intra_convs
#   - 2-tier diff LR: output_convs=0.5×, heads=1.0×
#   - Monotone exponential 4e-6 → 8e-7, warmup=10, epochs=14
# P2: + cross_convs.3 at 0.1× LR  (single outermost receptor ring, very gentle)
#   - 3-tier diff LR: cross_convs.3=0.1×, output_convs=0.5×, heads=1.0×
#   - Monotone exponential 1e-6 → 1e-7, warmup=8, epochs=16
#
# Spike handling (v3c only):
#   - On val tr_norm spike >10×: PERMANENT 50% LR cut via scheduler.permanent_reduce()
#   - EMA skip for 1 epoch (shorter than v5n's 2 — less aggressive adaptation = faster recovery)
#   - No cooldown countdown — the exponential simply continues from the new lower base
#   - LR can NEVER increase after warmup ends

# V3C Phase 1: score heads + output convs ONLY (identical to V5N-P1)
_UNFREEZE_PATTERNS_V3CP1 = list(_UNFREEZE_PATTERNS_V5NP1)

# V3C Phase 2: + cross_convs.3 at aggressive 0.1× LR (via make_v3cp2_optimizer)
_UNFREEZE_PATTERNS_V3CP2 = _UNFREEZE_PATTERNS_V3CP1 + [
    "cross_convs.3",   # outermost receptor ring — minimal adaptation at 0.1× LR
    # cross_convs.0/1/2: entirely frozen — deep receptor representations preserved
    # intra_convs.*: entirely frozen — ALL peptide geometry priors preserved
]

# ---------------------------------------------------------------------------
# V4C mode: tiny cross_conv probe — monotone decay, cross_convs.3 in P1 at 0.15×
# ---------------------------------------------------------------------------
# Core philosophy (May 2026):
#   V3C tests score-head calibration with NO cross_conv touch in P1.
#   V4C tests whether opening cross_convs.3 in P1 at very low LR (0.15× of 5e-6
#   = 7.5e-7 effective) provides additional receptor-specific benefit over V3C.
#   Hypothesis: cross_conv adaptation at near-noise LR biases receptor recognition
#   without destabilising the pretrained diffusion manifold.
#
# P1: score heads + output convs + cross_convs.3 at 0.15×
#   - 2-tier diff LR: heads+output=1.0×, cross_convs.3=0.15×
#   - Monotone exponential 5e-6 → 1e-6, warmup=10, epochs=14
# P2: + cross_convs.2 at 0.08×  (one ring deeper, even more conservative)
#   - 3-tier diff LR: heads+output=1.0×, cross_convs.3=0.15×, cross_convs.2=0.08×
#   - Monotone exponential 1.2e-6 → 1e-7, warmup=8, epochs=18
#
# Spike handling: identical to v3c — permanent_reduce(0.5) + EMA skip 1 epoch
# Shared with v3c: WarmupThenExponential, no phase 3, ESM frozen

# V4C Phase 1: score heads + output convs + cross_convs.3
# (identical param set to V5N-P2 / V3B-P1, but at lower LR and with diff optimizer)
_UNFREEZE_PATTERNS_V4CP1 = _UNFREEZE_PATTERNS_V3CP1 + [
    "cross_convs.3",   # outermost receptor ring — 0.15× effective LR in P1
]

# V4C Phase 2: + cross_convs.2 at 0.08× LR
_UNFREEZE_PATTERNS_V4CP2 = _UNFREEZE_PATTERNS_V4CP1 + [
    "cross_convs.2",   # 2nd outermost receptor ring — 0.08× effective LR in P2
    # cross_convs.0/1: entirely frozen — deep receptor representations preserved
    # intra_convs.*: entirely frozen — ALL peptide geometry priors preserved
]

# V5C mode: ultra-minimal diversity-preserving recalibration
#
# PRIMARY goal: preservation of pretrained exploration diversity and score-field geometry.
# Evidence from v1–v4c: pretrained manifold is extraordinarily well-calibrated and easy
# to damage. Long peptides depend on broad exploration; even tiny cross_conv updates
# (v4c 0.08–0.15×) may compress the manifold. Useful adaptation: first few epochs only.
#
# V5C hypothesis: only the final score-projection layers (tr/rot/tor heads) need
# recalibration; ALL geometry-forming layers (cross_convs, intra_convs, embeddings,
# equivariant) must remain frozen. Phase 2 adds output convolutions only.
#
# Key differences vs v3c/v4c:
#   - Phase 1: ONLY score heads (no output convs at all — v3c has both in P1)
#   - Phase 2: + output convs (no cross_conv touch whatsoever)
#   - Much lower LR: 2e-6→2e-7 (P1), 5e-7→5e-8 (P2)
#   - Very slow EMA: 0.99998 (vs 0.99997 in v3c/v4c)
#   - Tighter grad_clip: 0.2 (vs 0.25/0.3)
#   - Very weak pretrained-reg: λ=1e-4 (P1), λ=2e-4 (P2)
#   - Spike handling: permanent_reduce(0.5) + 1-epoch EMA skip (same as v3c)
#   - NO cross_conv adaptation in ANY phase
#   - NO phase 3
#   - Save every epoch after ep4 (preserve all stable checkpoints)
#
# V5C Phase 1: ONLY final score heads (NO output convs, NO cross_convs, NO intra_convs)
_UNFREEZE_PATTERNS_V5CP1 = [
    "tr_final_layer",         # translation score head
    "rot_final_layer",        # rotation score head
    "tor_bb_final_layer",     # backbone torsion score head
    "tor_sc_final_layer",     # side-chain torsion score head
    # ALL output convs FROZEN: final_conv, tor_*_bond_conv
    # ALL cross/intra convs FROZEN — zero geometry shift in P1
    # ALL embeddings FROZEN — representation priors preserved
]

# V5C Phase 2: + output convolutions at 0.5× (same 2-tier as v3c P1)
_UNFREEZE_PATTERNS_V5CP2 = _UNFREEZE_PATTERNS_V5CP1 + [
    "final_conv",           # main output conv — careful 0.5× LR in P2
    "tor_bb_bond_conv",     # backbone bond conv
    "tor_sc_bond_conv",     # side-chain bond conv
    # cross_convs.*: ALL frozen throughout — no receptor geometry updates
    # intra_convs.*: ALL frozen throughout — no peptide geometry updates
]

# ---------------------------------------------------------------------------
# V6 mode: targeted long-peptide adaptation — cross_convs + tor_bb_bond_conv
# ---------------------------------------------------------------------------
# Design rationale (May 2026):
#   Bench300 + N=20 analysis identified three independent failure modes:
#     1. RANKING (3.05Å gap at N=20) — dominant, can't fix with training alone
#     2. MODEL QUALITY (oracle 4.08Å at N=20) — very_long geometry not learned
#     3. SAMPLING (0.78Å gain from N=5→N=20) — secondary, diminishing returns
#
#   Root cause of model quality failure: ZERO very_long complexes in 2K training slice;
#   fine-tuning (v3c/v4c/v5c) only touched score heads and/or outermost receptor ring
#   at ultra-low LR — insufficient to learn very_long peptide backbone geometry.
#
#   V6 hypothesis: very_long peptide geometry is encoded in BOTH tor_bb_bond_conv
#   (backbone torsion prediction) AND cross_convs (receptor-peptide interaction). Both
#   need simultaneous adaptation with genuine signal. Previous ultra-conservative approach
#   was principled but not sufficient.
#
#   V6 design:
#     Phase 1 (ep 1–8):   score heads + tor_bb_bond_conv (12.98% params) — calibrate
#                          backbone torsion geometry before introducing cross-attention
#     Phase 2 (ep 9–35):  + all cross_convs (48.30% total) — full receptor adaptation
#                          4-tier differential LR (cc.0/1=0.4×, cc.2/3=0.7×,
#                          tor_bb=0.5×, heads=1.0×)
#     Phase 3 (ep 36–45): uniform sampling at 5e-7 LR — consolidation
#
#   Key differences vs all prior modes:
#     - Starts from rapidock_global.pt (NOT any previously fine-tuned checkpoint)
#     - Tier-based oversampling (3× for very_long tiers, 2× for long tiers)
#     - 20% replay from original training data (prevents short/medium forgetting)
#     - BatchNorm running stats frozen via freeze_frozen_bn_stats() [BUG FIX]
#     - L2 regularization on cross_convs (λ=3e-4) to prevent catastrophic forgetting
#     - Multi-best checkpointing: best_long, best_very_long, best_combined
#
#   CRITICAL BUG FIX (BatchNorm running stats):
#     All prior runs silently updated running_mean/running_var in "frozen" conv blocks
#     because model.train() propagates to all submodules regardless of requires_grad.
#     Fix: after load_model_for_finetuning(), call freeze_frozen_bn_stats() to set
#     .eval() on all BatchNorm submodules whose parameters are frozen.

# V6 Phase 1: score heads + tor_bb_bond_conv (backbone torsion geometry calibration)
_UNFREEZE_PATTERNS_V6P1 = [
    "tr_final_layer",         # translation score head
    "rot_final_layer",        # rotation score head
    "tor_bb_final_layer",     # backbone torsion score head
    "tor_sc_final_layer",     # side-chain torsion score head
    "tor_bb_bond_conv",       # backbone bond conv — learns very_long backbone geometry
    # ALL cross/intra_convs FROZEN: no receptor adaptation in P1
    # ALL embeddings FROZEN: representation priors preserved
    # tor_sc_bond_conv FROZEN: side-chain geometry is secondary concern
]

# V6 Phase 2: + all cross_convs (full receptor-peptide interaction adaptation)
_UNFREEZE_PATTERNS_V6P2 = _UNFREEZE_PATTERNS_V6P1 + [
    "cross_convs",   # all 4 rings — full receptor adaptation
    # cross_convs.0/1 at 0.4× LR (deeper, more fundamental representations)
    # cross_convs.2/3 at 0.7× LR (outermost, more data-dependent)
    # intra_convs.*: ALL frozen — peptide-internal geometry priors preserved
    # tor_sc_bond_conv: still frozen — side-chain geometry is secondary concern
]

# V6 Phase 3: same as P2 but with reduced LR (uniform sampling consolidation)
_UNFREEZE_PATTERNS_V6P3 = _UNFREEZE_PATTERNS_V6P2  # same params, lower LR

# V6 default L2 regularization patterns (applied to cross_convs to prevent forgetting)
_PRETRAINED_REG_PATTERNS_V6 = [
    "cross_convs.0",
    "cross_convs.1",
    "cross_convs.2",
    "cross_convs.3",
]

# V6 tier oversampling weights (keyed by tier column value in training CSV)
_V6_TIER_WEIGHTS = {
    "T1_sheet_very_long":    3,    # highest priority: SHEET very_long (hardest, rarest)
    "T2_sheet_long":         2,    # high: SHEET long (common failure mode)
    "T3_helix_very_long":    3,    # high: HELIX very_long
    "T4_unusual_very_long":  3,    # high: UNUSUAL very_long
    "T5_helix_long":         2,    # moderate: HELIX long
    "T6_sheet_medium":       1,    # baseline
    "T7_sheet_short":        1,    # baseline
    "T8_sheet_medium_topoff": 1,   # baseline
    "replay":                1,    # replay data: no oversampling (diversity preservation)
}

# Pattern to identify ESM projection layers (frozen in ALL v3/v4/v5/v3b/v5n/v3c/v4c/v5c/v6 phases)
_ESM_FREEZE_PATTERN = "lm_embedding_layer"

# Default patterns for L2 pretrained-weight regularization
# Applied to layers that are unfrozen but should not drift far from pretrained init
_PRETRAINED_REG_PATTERNS_DEFAULT = [
    "intra_convs",       # all intra_convs unfrozen in P2/P3 but should stay near prior
    "cross_convs.0",     # early cross_convs (only unfrozen in P3)
    "cross_convs.1",     # early cross_convs
    "rec_node_embedding",  # node embeddings — transfer well
    "pep_node_embedding",
]

# Score-matching loss weights (same as original RAPiDock training)
_TR_W = 0.25
_ROT_W = 0.25
_TOR_BB_W = 0.25
_TOR_SC_W = 0.25


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_finetuning(model_args, ckpt_path, device, unfreeze_phase,
                              v3_mode=False, v4_mode=False, v5_mode=False,
                              v3b_mode=False, v5n_mode=False, v3c_mode=False,
                              v4c_mode=False, v5c_mode=False):
    # type: (Namespace, str, torch.device, int, bool, bool, bool, bool, bool, bool, bool, bool) -> torch.nn.Module
    """Load pretrained weights and selectively freeze layers by phase.

    Exactly one of v3_mode/v4_mode/v5_mode/v3b_mode/v5n_mode/v3c_mode/v4c_mode/v5c_mode should be True (or all False).

    Args:
        model_args: Namespace with model hyperparameters.
        ckpt_path: Path to the .pt checkpoint file.
        device: torch.device to move the model to.
        unfreeze_phase: 1, 2, or 3.
        v3_mode: score heads + cross_convs.2/3 in P1; + intra_convs.3 in P2; ESM frozen in P3.
        v4_mode: score heads + cross_convs.2/3 in BOTH P1 AND P2; ESM frozen in P3.
        v5_mode: score heads only in P1; + cross_convs.3 in P2; aggressive P3; ESM frozen in P3.
        v3b_mode: stable controlled spec — cross_convs.3 in P1; + cross_convs.2 in P2;
                  cosine ALL phases; adaptive spike LR; ESM frozen in P3.
        v5n_mode: ultra-conservative manifold preservation — score heads+output only in P1;
                  + cross_convs.3 in P2; ultra-conservative layerwise P3; EMA skip on spike.
        v3c_mode: minimal low-energy recalibration — monotonic exponential decay, no rebounds;
                  score heads+output only in P1; + cross_convs.3 at 0.1× in P2; NO phase 3.
    """
    model = get_model(model_args, no_parallel=True)
    raw_ckpt = torch.load(ckpt_path, map_location="cpu")

    # Support both raw state dict and {"model": ...} wrapper
    if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        model.load_state_dict(raw_ckpt["model"], strict=True)
        if "ema_weights" in raw_ckpt:
            try:
                ema = ExponentialMovingAverage(model.parameters(),
                                               decay=model_args.ema_rate)
                ema.load_state_dict(raw_ckpt["ema_weights"], device=device)
                ema.copy_to(model.parameters())
                print("[INFO] Loaded EMA weights from checkpoint")
            except Exception as exc:
                print(f"[WARN] Could not apply EMA weights: {exc} — using raw weights")
    else:
        model.load_state_dict(raw_ckpt, strict=True)

    # Phase 3 = full retraining (all layers unfrozen)
    if unfreeze_phase == 3:
        for param in model.parameters():
            param.requires_grad = True
        any_esm_mode = v3_mode or v4_mode or v5_mode or v3b_mode or v5n_mode or v3c_mode or v4c_mode or v5c_mode
        if any_esm_mode:
            # All v3/v4/v5/v3b/v5n/v3c/v4c/v5c keep ESM projection layer frozen in P3
            esm_frozen = 0
            for name, param in model.named_parameters():
                if _ESM_FREEZE_PATTERN in name:
                    param.requires_grad = False
                    esm_frozen += param.numel()
            total_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mode_tag = ("V3" if v3_mode else "V4" if v4_mode
                        else "V5" if v5_mode else "V3B" if v3b_mode
                        else "V5N" if v5n_mode else "V3C" if v3c_mode
                        else "V4C" if v4c_mode else "V5C")
            print(f"[{mode_tag} Phase 3] {trainable:,} / {total_params:,} parameters unfrozen "
                  f"(ESM projection frozen: {esm_frozen:,} params)")
        else:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"[Phase 3] ALL {total_params:,} parameters unfrozen (full retraining)")
        return model.to(device)

    # Phase 1 or 2: freeze everything first, then selectively unfreeze
    for param in model.parameters():
        param.requires_grad = False

    if v3_mode:
        if unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V3P1
        else:
            patterns = _UNFREEZE_PATTERNS_V3P2  # P2: also includes intra_convs.3
        phase_label = f"V3-Phase {unfreeze_phase}"
    elif v4_mode:
        # V4: same cross_convs.2/3 + heads pattern for BOTH P1 and P2
        patterns = _UNFREEZE_PATTERNS_V3P1
        phase_label = f"V4-Phase {unfreeze_phase}"
    elif v5_mode:
        if unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_P1      # heads only
        else:
            patterns = _UNFREEZE_PATTERNS_V5P2    # heads + cross_convs.3
        phase_label = f"V5-Phase {unfreeze_phase}"
    elif v3b_mode:
        if unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V3BP1   # heads + cross_convs.3
        else:
            patterns = _UNFREEZE_PATTERNS_V3BP2   # heads + cross_convs.3 + cross_convs.2
        phase_label = f"V3B-Phase {unfreeze_phase}"
    elif v5n_mode:
        if unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V5NP1   # score heads + output convs ONLY
        else:
            patterns = _UNFREEZE_PATTERNS_V5NP2   # + cross_convs.3
        phase_label = f"V5N-Phase {unfreeze_phase}"
    elif v3c_mode:
        if unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V3CP1   # score heads + output convs ONLY (= V5N-P1)
        else:
            patterns = _UNFREEZE_PATTERNS_V3CP2   # + cross_convs.3 (at 0.1× via optimizer)
        phase_label = f"V3C-Phase {unfreeze_phase}"
    elif v4c_mode:
        if unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V4CP1   # score heads + output convs + cross_convs.3 at 0.15×
        else:
            patterns = _UNFREEZE_PATTERNS_V4CP2   # + cross_convs.2 at 0.08×
        phase_label = f"V4C-Phase {unfreeze_phase}"
    elif v5c_mode:
        if unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V5CP1   # score heads ONLY — zero geometry touch
        else:
            patterns = _UNFREEZE_PATTERNS_V5CP2   # + output convs at 0.5× (no cross_conv ever)
        phase_label = f"V5C-Phase {unfreeze_phase}"
    else:
        patterns = _UNFREEZE_PATTERNS_P1 if unfreeze_phase == 1 else _UNFREEZE_PATTERNS_P2
        phase_label = f"Phase {unfreeze_phase}"

    unfrozen_params = 0
    pattern_hits: dict = {p: [] for p in patterns}   # pattern → matching param names
    for name, param in model.named_parameters():
        for pattern in patterns:
            if pattern in name:
                param.requires_grad = True
                unfrozen_params += param.numel()
                pattern_hits[pattern].append(name)
                break  # don't double-count

    total_params = sum(p.numel() for p in model.parameters())
    pct = 100.0 * unfrozen_params / total_params
    print(f"\n[{phase_label}] Unfrozen: {unfrozen_params:,} / {total_params:,} ({pct:.2f}%)")
    print(f"Pattern → matched layers ({phase_label}):")
    for pattern, names in pattern_hits.items():
        if names:
            print(f"  ✓ {pattern!r:35s} → {len(names)} tensors ({names[0]}, ...)")
        else:
            print(f"  ✗ {pattern!r:35s} → NO MATCH ← WARN: pattern unused")

    n_unmatched = sum(1 for names in pattern_hits.values() if not names)
    if n_unmatched > 0:
        print(f"[WARN] {n_unmatched} patterns matched nothing — "
              f"check model architecture vs pattern names")
    if unfrozen_params == 0:
        raise RuntimeError("No parameters were unfrozen! Check patterns match model names.")

    return model.to(device)


# ---------------------------------------------------------------------------
# V6 load helper (adds v6_mode support to load_model_for_finetuning)
# ---------------------------------------------------------------------------

def load_model_for_finetuning_v6(model_args, ckpt_path, device, unfreeze_phase):
    # type: (Namespace, str, torch.device, int) -> torch.nn.Module
    """V6-specific model loading: score heads + tor_bb_bond_conv (P1) or + cross_convs (P2/P3).

    Starts ONLY from rapidock_global.pt — never from a previously fine-tuned checkpoint.

    Phase 1 (ep 1-8):  heads + tor_bb_bond_conv   (12.98% of params)
    Phase 2 (ep 9-35): + all cross_convs           (48.30% of params)
    Phase 3 (ep 36-45): same as P2 but different LR (consolidation)
    """
    model = get_model(model_args, no_parallel=True)
    raw_ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        state = raw_ckpt["model"]
    else:
        state = raw_ckpt

    model.load_state_dict(state, strict=True)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Select patterns based on phase
    if unfreeze_phase == 1:
        patterns = _UNFREEZE_PATTERNS_V6P1
        phase_label = "V6-Phase 1"
    elif unfreeze_phase == 2:
        patterns = _UNFREEZE_PATTERNS_V6P2
        phase_label = "V6-Phase 2"
    else:  # phase 3
        patterns = _UNFREEZE_PATTERNS_V6P3
        phase_label = "V6-Phase 3"

    unfrozen_params = 0
    pattern_hits = {p: [] for p in patterns}
    for name, param in model.named_parameters():
        for pattern in patterns:
            if pattern in name:
                param.requires_grad = True
                unfrozen_params += param.numel()
                pattern_hits[pattern].append(name)
                break

    total_params = sum(p.numel() for p in model.parameters())
    pct = 100.0 * unfrozen_params / total_params
    print(f"\n[{phase_label}] Unfrozen: {unfrozen_params:,} / {total_params:,} ({pct:.2f}%)")
    for pattern, names in pattern_hits.items():
        if names:
            print(f"  ✓ {pattern!r:35s} → {len(names)} tensors")
        else:
            print(f"  ✗ {pattern!r:35s} → NO MATCH ← WARN")

    if unfrozen_params == 0:
        raise RuntimeError("[V6] No parameters were unfrozen!")

    return model.to(device)


# ---------------------------------------------------------------------------
# BatchNorm running-stats freeze (BUG FIX for all modes)
# ---------------------------------------------------------------------------

def freeze_frozen_bn_stats(model):
    # type: (torch.nn.Module) -> int
    """Set .eval() on BatchNorm submodules whose weight/bias are frozen.

    CRITICAL BUG FIX: BatchNorm running_mean/running_var are buffers — they update
    during model.train() forward passes regardless of requires_grad=False on weight/bias.
    This function identifies frozen BatchNorm layers and calls .eval() on them, which
    stops running stat updates while keeping the model in train() mode for gradient flow.

    Must be called:
    1. After load_model_for_finetuning() before training starts.
    2. At the START of each train_epoch() call (after model.train()) to re-apply,
       since model.train() resets all submodule modes.

    Returns:
        Number of BatchNorm submodules frozen (set to eval mode).
    """
    n_frozen = 0
    for name, module in model.named_modules():
        # Check if this module has a weight parameter that is frozen
        weight = getattr(module, 'weight', None)
        if weight is None:
            continue
        module_type = type(module).__name__
        # Identify BatchNorm-style modules by type name
        if 'BatchNorm' not in module_type and 'batch_norm' not in name.lower():
            continue
        # If weight is not a parameter (e.g., buffer), skip
        if not isinstance(weight, torch.nn.Parameter):
            continue
        # Check if this BatchNorm's learnable params are frozen
        all_frozen = not weight.requires_grad
        bias = getattr(module, 'bias', None)
        if bias is not None and isinstance(bias, torch.nn.Parameter):
            all_frozen = all_frozen and not bias.requires_grad
        if all_frozen:
            module.eval()
            n_frozen += 1
    return n_frozen


def snapshot_frozen_bn_stats(model):
    # type: (torch.nn.Module) -> dict
    """Snapshot running_mean/running_var of all frozen BatchNorm layers.

    Used to verify frozen BN stats don't drift across epochs.
    Returns dict of {module_name: (running_mean_clone, running_var_clone)}.
    """
    snap = {}
    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if weight is None or not isinstance(weight, torch.nn.Parameter):
            continue
        if 'BatchNorm' not in type(module).__name__:
            continue
        if not weight.requires_grad:
            rm = getattr(module, 'running_mean', None)
            rv = getattr(module, 'running_var', None)
            if rm is not None and rv is not None:
                snap[name] = (rm.clone().detach(), rv.clone().detach())
    return snap


def check_frozen_bn_drift(model, snapshot, epoch):
    # type: (torch.nn.Module, dict, int) -> bool
    """Compare current frozen BN stats against snapshot.  Returns True if any drift detected."""
    drifted = False
    for name, module in model.named_modules():
        if name not in snapshot:
            continue
        rm_snap, rv_snap = snapshot[name]
        rm_cur = getattr(module, 'running_mean', None)
        rv_cur = getattr(module, 'running_var', None)
        if rm_cur is None or rv_cur is None:
            continue
        rm_diff = (rm_cur.float() - rm_snap.float()).abs().max().item()
        rv_diff = (rv_cur.float() - rv_snap.float()).abs().max().item()
        if rm_diff > 1e-6 or rv_diff > 1e-6:
            print(f"  [BN-DRIFT ALERT ep{epoch}] {name}: "
                  f"running_mean max_diff={rm_diff:.6f}  running_var max_diff={rv_diff:.6f}")
            drifted = True
    return drifted


# ---------------------------------------------------------------------------
# Batch attribute injection (single-sample fix)
# ---------------------------------------------------------------------------

def _inject_batch_for_single_sample(data, device):
    # type: (object, torch.device) -> None
    """Set PyG batch/ptr/num_graphs attributes on each node store for single-sample get().

    PyG's DataLoader sets .batch (node→graph-index tensor), .ptr (cumulative
    node-count tensor), and data.num_graphs when collating a batch.
    dataset.get(idx) called directly returns raw HeteroData WITHOUT these —
    the model's scatter and radius ops will crash:

    Missing batch/ptr → AttributeError: 'NodeStorage' has no attribute 'batch'
        at diffusion.py lines 459, 484, 517, 553, 603, 709-712, 725, 746
        and get_updated_peptide_feature lines 643-688

    Missing num_graphs → AttributeError: 'HeteroData' has no attribute 'num_graphs'
        at diffusion.py lines 466 (out_nodes=data.num_graphs) and 710
        (torch.zeros((data.num_graphs, 3)))

    For a single sample (batch_size=1):
        batch     = zeros(n_nodes, dtype=long)   ← every node belongs to graph 0
        ptr       = tensor([0, n_nodes])          ← cumulative size tensor
        num_graphs = 1                            ← one graph in this "batch"

    Attribute priority order for inferring n_nodes:
        pos → x → atom2res_index  (first non-None tensor found)

    Note on data.name: inference_utils sets complex_graph['name'] = name_string.
    diffusion.py line 484 does `for i in range(len(data.name))` expecting a list of
    length 1. With a string, len() = string length, but only i=0 yields non-empty
    tensors → correct result with wasted iterations. We leave it as-is (minor perf
    cost, not a correctness issue).
    """
    for node_type in data.node_types:
        store = data[node_type]
        n_nodes = None
        for attr_name in ("pos", "x", "atom2res_index"):
            val = getattr(store, attr_name, None)
            if val is not None and isinstance(val, torch.Tensor):
                n_nodes = val.size(0)
                break
        if not n_nodes:
            continue
        if not hasattr(store, "batch") or store.batch is None:
            store.batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
        if not hasattr(store, "ptr") or store.ptr is None:
            store.ptr = torch.tensor([0, n_nodes], dtype=torch.long, device=device)
    # Graph-level attribute: num_graphs = 1 for a single sample.
    # Used by diffusion.py line 466 (out_nodes=data.num_graphs) and line 710.
    if not hasattr(data, "num_graphs") or data.num_graphs is None:
        data.num_graphs = 1


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(model, data, transform, device, _norm_out=None):
    # type: (torch.nn.Module, object, NoiseTransform, torch.device, Optional[dict]) -> torch.Tensor
    """Apply noise to a crystal complex, run forward, return score-matching loss.

    ScoreModel.forward() returns a dict {"tr_pred": ..., "rot_pred": ...,
    "tor_pred_backbone": ..., "tor_pred_sidechain": ...}.

    CRITICAL: Access values by key — NOT by tuple unpacking. Tuple-unpacking
    a dict yields the string keys (not tensors) and produces a silent TypeError
    every time (n_ok=0 for entire training run).

    CRITICAL: _inject_batch_for_single_sample must be called after .to(device)
    so the batch/ptr tensors are on the same device as pos/x tensors.
    Without this, diffusion.py scatter ops crash with:
        AttributeError: 'NodeStorage' object has no attribute 'batch'

    DTYPE FIX (May 28 2026): atom2resid_index and atom2atomid_index are stored as
    int64 in the graph cache (torch.tensor(list_of_ints) defaults to int64).
    PyTorch's scatter_mean / F.mse_loss raise "Got: Int" on CUDA when integer tensors
    reach those ops.  The fix is explicit .float() on every score target and every
    prediction before the MSE computation.  .float() is a no-op for float32 tensors
    (no gradient-graph break) and safely casts any int/int64 to float32.
    Applying to predictions too is defensive — neural-net outputs should always be
    float32, but .float() costs nothing if they already are.

    Args:
        _norm_out: optional dict with keys "tr", "rot", "tor_bb", "tor_sc" — each
            a list that will receive the per-sample L2 norm of the corresponding
            score prediction.  Used to detect score-field instability before val
            explodes to 1e16.  Pass None (default) for no tracking overhead.
    """
    noisy = copy.deepcopy(data)
    noisy = transform(noisy)
    noisy = noisy.to(device)
    # Must come AFTER .to(device) — batch/ptr tensors must match device of pos/x
    _inject_batch_for_single_sample(noisy, device)

    preds = model(noisy)
    # Explicitly cast predictions to float32 — defensive against any int edge case.
    tr_pred     = preds["tr_pred"].float()
    rot_pred    = preds["rot_pred"].float()
    tor_bb_pred = preds["tor_pred_backbone"].float()
    tor_sc_pred = preds["tor_pred_sidechain"].float()

    # Score-norm tracking — catch instability BEFORE val explodes to 1e16.
    # We use torch.no_grad() to avoid polluting the autograd graph.
    if _norm_out is not None:
        with torch.no_grad():
            _norm_out["tr"].append(tr_pred.norm().item())
            _norm_out["rot"].append(rot_pred.norm().item())
            if tor_bb_pred.numel() > 0:
                _norm_out["tor_bb"].append(tor_bb_pred.norm().item())
            if tor_sc_pred.numel() > 0:
                _norm_out["tor_sc"].append(tor_sc_pred.norm().item())

    # Explicitly cast targets to float32.  Scores are computed from torch.normal()
    # (float32) so these should already be float, but int64 index tensors in the
    # graph can propagate through scatter ops and corrupt the dtype on CUDA.
    tr_target = noisy.tr_score.to(device).float()
    loss = _TR_W * F.mse_loss(tr_pred, tr_target)

    rot_target = noisy.rot_score.to(device).float()
    loss = loss + _ROT_W * F.mse_loss(rot_pred, rot_target)

    if tor_bb_pred.numel() > 0 and noisy.tor_backbone_score.numel() > 0:
        tor_bb_target = noisy.tor_backbone_score.to(device).float()
        if tor_bb_pred.shape == tor_bb_target.shape:
            loss = loss + _TOR_BB_W * F.mse_loss(tor_bb_pred, tor_bb_target)

    if tor_sc_pred.numel() > 0 and noisy.tor_sidechain_score.numel() > 0:
        tor_sc_target = noisy.tor_sidechain_score.to(device).float()
        if tor_sc_pred.shape == tor_sc_target.shape:
            loss = loss + _TOR_SC_W * F.mse_loss(tor_sc_pred, tor_sc_target)

    return loss


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _esm_cache_path(csv_path: str) -> str:
    """Return path for a .pt file that caches ESM embeddings for this CSV split.

    The cache is placed next to the CSV file so it is shared across all phases.
    Naming convention: <csv_stem>_esm_cache.pt  (e.g. combined_train_esm_cache.pt)
    """
    p = Path(csv_path)
    return str(p.parent / (p.stem + "_esm_cache.pt"))


def build_dataset(csv_path, model_args, output_dir, esm_device="cpu"):
    # type: (str, Namespace, str, str) -> InferenceDataset
    """Build InferenceDataset from a CSV (crystal poses as peptide_description).

    ESM embedding caching:
        After the first phase, embeddings are written to <csv_stem>_esm_cache.pt
        next to the CSV.  Subsequent phases (same CSV) load from cache, avoiding
        the ~3-4 h CPU re-computation per phase (saves ~7 h total for P2+P3).

    Args:
        esm_device: Passed to InferenceDataset.  Use 'cpu' (default) to avoid
            WSL2 TDR crashes during the one-time ESM embedding pre-computation.
    """
    df = pd.read_csv(csv_path)
    complex_names = df["complex_name"].tolist()
    protein_desc  = df["protein_description"].tolist()
    peptide_desc  = df["peptide_description"].tolist()

    os.makedirs(output_dir, exist_ok=True)
    for name in complex_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    # ── ESM embedding cache ─────────────────────────────────────────────────
    want_lm = model_args.esm_embeddings_path_train is not None
    precomputed = None
    cache_file  = _esm_cache_path(csv_path)

    if want_lm and os.path.exists(cache_file):
        print(f"[ESM cache] Loading from {cache_file}")
        try:
            cache_dict = torch.load(cache_file, map_location="cpu")
            # cache_dict maps complex_name → list[tensor per chain]
            # Reconstruct in CSV row order
            precomputed = [cache_dict.get(n) for n in complex_names]
            # Sanity: every entry must be a non-empty list (not None)
            missing = sum(1 for e in precomputed if e is None)
            if missing > 0:
                print(f"[ESM cache] WARNING: {missing} entries missing from cache; "
                      f"falling back to full ESM computation.")
                precomputed = None
            else:
                print(f"[ESM cache] Loaded {len(precomputed)} embeddings — "
                      f"skipping ESM re-computation.")
        except Exception as exc:
            print(f"[ESM cache] Failed to load cache ({exc}); recomputing.")
            precomputed = None
    # ───────────────────────────────────────────────────────────────────────

    # Pass cache_file to InferenceDataset so it saves INSIDE __init__ right
    # after compute_ESM_embeddings returns — bulletproof vs post-init failures.
    # On cache HIT (precomputed is not None), we don't re-save, so pass None.
    esm_save_path = cache_file if (want_lm and precomputed is None) else None

    dataset = InferenceDataset(
        output_dir=output_dir,
        complex_name_list=complex_names,
        protein_description_list=protein_desc,
        peptide_description_list=peptide_desc,
        lm_embeddings=want_lm,
        lm_embeddings_pep=(model_args.esm_embeddings_peptide_train is not None),
        precomputed_lm_embeddings=precomputed,
        # CRITICAL: was None → KeyError in get() → model never trained (fixed May 27)
        conformation_type='E',
        # Default CPU to avoid WSL2 TDR CUDA crash at batch ~790/874 on long sequences
        esm_device=esm_device,
        # Cache path: dataset saves raw ESM dict immediately after computation
        esm_cache_path=esm_save_path,
    )

    # ── On cache HIT: reconstruct list-of-lists from raw dict format ─────────
    # The raw cache stores {complex_name_chain_j: tensor}.
    # InferenceDataset stores self.lm_embeddings as list[list[tensor]].
    # When loading from cache, precomputed is already list[list[tensor]] so
    # no reconstruction needed — InferenceDataset uses it directly.
    # On cache MISS: InferenceDataset saves the raw dict and builds the list.
    # ─────────────────────────────────────────────────────────────────────────

    return dataset


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, dataset, indices, transform, optimizer, device,
                grad_accum, ema=None, grad_clip_norm=1.0,
                pretrained_ref=None, reg_lambda=0.0):
    # type: (...) -> tuple
    """Run one training epoch.

    Returns:
        (avg_loss, n_ok, n_total, norm_summary) where norm_summary is a dict
        with keys tr_mean, tr_max, rot_mean, rot_max, tor_bb_mean, tor_sc_mean,
        tr_var (variance of tr_pred norms — diversity proxy).
        These are the mean/max L2 norms of the score predictions across all
        successful samples — used to detect score-field instability.

    Args:
        pretrained_ref: Output of build_pretrained_ref(), or None.  When set,
            a weak L2 penalty toward the pretrained weights is added to each
            sample's loss before backprop.
        reg_lambda:     Regularization strength for pretrained-weight L2 penalty.
    """
    model.train()
    total_loss = 0.0
    n_ok = 0
    n_load_fail = 0
    n_loss_fail = 0
    _first_load_exc = None   # type: Optional[Exception]
    _first_loss_exc = None   # type: Optional[Exception]
    optimizer.zero_grad()

    # Score-norm accumulator — populated by compute_loss via _norm_out
    _norms = {"tr": [], "rot": [], "tor_bb": [], "tor_sc": []}

    for step, idx in enumerate(indices):
        try:
            data = dataset.get(idx)
        except Exception as exc:
            n_load_fail += 1
            if _first_load_exc is None:
                _first_load_exc = exc
            continue

        try:
            loss = compute_loss(model, data, transform, device, _norm_out=_norms)
            # Add weak pretrained-weight L2 regularization if configured
            if pretrained_ref and reg_lambda > 0.0:
                reg = compute_pretrained_reg_loss(model, pretrained_ref, reg_lambda)
                loss = loss + reg
            scaled = loss / grad_accum
            scaled.backward()
            total_loss += loss.item()
            n_ok += 1
        except Exception as exc:
            n_loss_fail += 1
            if _first_loss_exc is None:
                _first_loss_exc = exc
            optimizer.zero_grad()
            continue

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip_norm
            )
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model.parameters())

    # Flush remaining gradients
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], grad_clip_norm
    )
    optimizer.step()
    optimizer.zero_grad()
    if ema is not None:
        ema.update(model.parameters())

    # Always report diagnostics — failures must be visible
    n_total = len(indices)
    if n_load_fail > 0:
        print(f"  [train] load failures: {n_load_fail}/{n_total} "
              f"(first: {type(_first_load_exc).__name__}: {_first_load_exc})")
    if n_loss_fail > 0:
        print(f"  [train] loss failures: {n_loss_fail}/{n_total} "
              f"(first: {type(_first_loss_exc).__name__}: {_first_loss_exc})")
    if n_ok == 0:
        print(f"  [train] *** CRITICAL: 0/{n_total} samples produced a gradient — "
              f"check load/loss failures above ***")
    elif n_ok < n_total * 0.5:
        print(f"  [train] WARNING: low yield {n_ok}/{n_total} ({100*n_ok/n_total:.0f}%)")

    # Aggregate score norms
    def _smean(lst):  return float(np.mean(lst))  if lst else 0.0
    def _smax(lst):   return float(max(lst))       if lst else 0.0
    def _svar(lst):   return float(np.var(lst))    if len(lst) > 1 else 0.0
    norm_summary = {
        "tr_mean":      _smean(_norms["tr"]),
        "tr_max":       _smax(_norms["tr"]),
        "tr_var":       _svar(_norms["tr"]),    # diversity proxy: variance of tr norms
        "rot_mean":     _smean(_norms["rot"]),
        "rot_max":      _smax(_norms["rot"]),
        "rot_var":      _svar(_norms["rot"]),
        "tor_bb_mean":  _smean(_norms["tor_bb"]),
        "tor_bb_max":   _smax(_norms["tor_bb"]),
        "tor_sc_mean":  _smean(_norms["tor_sc"]),
    }
    print(f"  [norms-train] "
          f"tr={norm_summary['tr_mean']:.3f}(max {norm_summary['tr_max']:.1f}"
          f" var={norm_summary['tr_var']:.3f})  "
          f"rot={norm_summary['rot_mean']:.3f}(var={norm_summary['rot_var']:.3f})  "
          f"tor_bb={norm_summary['tor_bb_mean']:.3f}(max {norm_summary['tor_bb_max']:.1f})  "
          f"tor_sc={norm_summary['tor_sc_mean']:.3f}")

    return total_loss / max(n_ok, 1), n_ok, n_total, norm_summary


def val_epoch(model, dataset, indices, transform, device, _stats_out=None,
              prev_tr_norm_max=None):
    # type: (...) -> float
    """Run one validation pass; return robust trimmed-mean loss.

    Uses a trimmed mean (drops top 5% of per-sample losses) so that a single
    outlier sample with astronomical MSE (e.g. score prediction blowup on a
    rare geometry) does not dominate the epoch metric and cause false plateau
    signals or incorrect checkpoint selection.

    Also logs: raw mean, median, and number of per-sample outliers (>1000)
    so pathological samples are still visible in the log.

    Args:
        _stats_out: optional dict that will be populated with diagnostic stats:
            raw_mean, median, max_loss, n_outlier, tr_norm_mean, tr_norm_max.
            Allows the training loop to log and track val instability separately
            from the trimmed_mean metric used for checkpoint selection.
    """
    model.eval()
    per_sample = []          # individual loss values for robust aggregation
    n_fail = 0
    _first_exc = None  # type: Optional[Exception]
    _val_norms = {"tr": [], "rot": [], "tor_bb": [], "tor_sc": []}

    with torch.no_grad():
        for idx in indices:
            try:
                data = dataset.get(idx)
                loss = compute_loss(model, data, transform, device,
                                    _norm_out=_val_norms)
                per_sample.append(loss.item())
            except Exception as exc:
                n_fail += 1
                if _first_exc is None:
                    _first_exc = exc
                continue
    if n_fail > 0:
        print(f"  [val] {n_fail}/{len(indices)} samples failed "
              f"(first: {type(_first_exc).__name__}: {_first_exc})")
    if not per_sample:
        if _stats_out is not None:
            _stats_out.update({"raw_mean": float("nan"), "median": float("nan"),
                               "max_loss": float("nan"), "n_outlier": 0,
                               "tr_norm_mean": 0.0, "tr_norm_max": 0.0})
        return float("nan")

    # Robust trimmed-mean: drop top 5% (at least 1 sample)
    sorted_losses = sorted(per_sample)
    n_keep = max(1, int(len(sorted_losses) * 0.95))
    trimmed = sorted_losses[:n_keep]
    trimmed_mean = float(np.mean(trimmed))

    raw_mean   = float(np.mean(per_sample))
    median_val = float(np.median(per_sample))
    max_loss   = float(max(per_sample))
    n_outlier  = sum(1 for v in per_sample if v > 1000.0)

    # Val score norms (on EMA model) — largest early-warning signal for instability
    tr_nm     = float(np.mean(_val_norms["tr"]))   if _val_norms["tr"]    else 0.0
    tr_nmx    = float(max(_val_norms["tr"]))        if _val_norms["tr"]    else 0.0
    tr_var    = float(np.var(_val_norms["tr"]))     if len(_val_norms["tr"]) > 1 else 0.0
    rot_nm    = float(np.mean(_val_norms["rot"]))   if _val_norms["rot"]   else 0.0
    rot_nmx   = float(max(_val_norms["rot"]))       if _val_norms["rot"]   else 0.0
    rot_var   = float(np.var(_val_norms["rot"]))    if len(_val_norms["rot"]) > 1 else 0.0
    torbb_nm  = float(np.mean(_val_norms["tor_bb"])) if _val_norms["tor_bb"] else 0.0
    torbb_nmx = float(max(_val_norms["tor_bb"]))    if _val_norms["tor_bb"] else 0.0
    torsc_nm  = float(np.mean(_val_norms["tor_sc"])) if _val_norms["tor_sc"] else 0.0

    # Val loss distribution stats — diversity/stability proxies
    val_loss_std  = float(np.std(per_sample))  if len(per_sample) > 1 else 0.0

    # Populate stats dict for caller
    if _stats_out is not None:
        _stats_out["raw_mean"]       = raw_mean
        _stats_out["median"]         = median_val
        _stats_out["max_loss"]       = max_loss
        _stats_out["n_outlier"]      = n_outlier
        _stats_out["val_loss_std"]   = val_loss_std
        _stats_out["tr_norm_mean"]   = tr_nm
        _stats_out["tr_norm_max"]    = tr_nmx
        _stats_out["tr_norm_var"]    = tr_var
        _stats_out["rot_norm_mean"]  = rot_nm
        _stats_out["rot_norm_max"]   = rot_nmx
        _stats_out["rot_norm_var"]   = rot_var
        _stats_out["torbb_norm_max"] = torbb_nmx
        _stats_out["torsc_norm_mean"] = torsc_nm

    # Norm summary — printed every epoch (cheap, very useful for instability detection)
    print(f"  [norms-val]  "
          f"tr={tr_nm:.3f}(max {tr_nmx:.1f} var={tr_var:.3f})  "
          f"rot={rot_nm:.3f}(max {rot_nmx:.1f})  "
          f"tor_bb={torbb_nm:.3f}(max {torbb_nmx:.1f})  tor_sc={torsc_nm:.3f}")

    # Verbose stats when outliers detected
    if n_outlier > 0 or max_loss > trimmed_mean * 10:
        print(f"  [val] trimmed={trimmed_mean:.4f}  raw_mean={raw_mean:.1f}  "
              f"median={median_val:.4f}  max={max_loss:.1f}  "
              f"outliers(>1k)={n_outlier}/{len(per_sample)}")

        # Score-norm alert: alert only on actual destabilisation signals.
        #   • Significant increase vs previous epoch (>2×, >100): flag — optimizer
        #     escaping pretrained score-calibration basin.
        #   • Catastrophic absolute value (>5000) AND we have a prior baseline:
        #     flag — sustained blowup, not just pretrained model pathology.
        #   • Epoch 1 (prev_tr_norm_max is None): EMA ≈ 100% pretrained; high
        #     norms reflect the pretrained model itself, not fine-tuning damage.
        #     Never alert at epoch 1 — wait for epoch 2 to establish a baseline.
        #   • Decreasing norms are a GOOD sign — do NOT alarm.
        if tr_nmx > 50.0 and prev_tr_norm_max is not None:
            is_catastrophic = tr_nmx > 5000.0
            is_increasing   = (tr_nmx > prev_tr_norm_max * 2.0 and tr_nmx > 100.0)
            if is_catastrophic or is_increasing:
                ratio = tr_nmx / max(prev_tr_norm_max, 1e-9)
                if is_increasing:
                    print(f"  [NORM ALERT ⚠] val tr_pred max_norm INCREASING: "
                          f"{prev_tr_norm_max:.1f} → {tr_nmx:.1f} (×{ratio:.1f}) — "
                          f"score field destabilising. Consider reducing LR.")
                else:
                    print(f"  [NORM ALERT ⚠] val tr_pred max_norm SUSTAINED HIGH: "
                          f"{prev_tr_norm_max:.1f} → {tr_nmx:.1f} (×{ratio:.1f}) — "
                          f"not recovering after fine-tuning. Check grad-clip and LR.")

    return trimmed_mean


# ---------------------------------------------------------------------------
# LR warmup scheduler
# ---------------------------------------------------------------------------

class WarmupThenPlateau:
    """Linear LR warmup for `warmup_epochs`, then ReduceLROnPlateau.

    Epoch 1 trains at  base_lr / warmup_epochs  (lowest warmup LR).
    Each subsequent warmup epoch ramps linearly toward base_lr.
    After epoch `warmup_epochs`, ReduceLROnPlateau takes over.

    Fix history: original code initialised the optimiser at base_lr and
    applied the warmup factor AFTER epoch 1, causing epoch 1 to run at
    full LR and epochs 2–warmup to ramp back up ("inverted warmup").
    The __init__ now pre-sets the optimiser LR to the correct epoch-1
    value so the ramp is monotone from the very first epoch.
    """
    def __init__(self, optimizer, base_lr, warmup_epochs, plateau_kwargs):
        # type: (torch.optim.Optimizer, float, int, dict) -> None
        self._opt = optimizer
        self._base_lr = base_lr
        self._warmup = warmup_epochs
        self._plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **plateau_kwargs
        )
        self._epoch = 0
        # Pre-set optimiser to epoch-1 warmup LR so the first epoch trains
        # at the correct (low) value rather than the full base_lr.
        if warmup_epochs > 0:
            init_lr = base_lr / max(warmup_epochs, 1)
            for pg in self._opt.param_groups:
                pg["lr"] = init_lr

    def step(self, metric=None):
        self._epoch += 1
        if self._epoch <= self._warmup:
            frac = self._epoch / max(self._warmup, 1)
            lr = self._base_lr * frac
            for pg in self._opt.param_groups:
                pg["lr"] = lr
        else:
            if metric is not None:
                self._plateau.step(metric)

    @property
    def current_lr(self):
        return self._opt.param_groups[0]["lr"]


class WarmupThenCosine:
    """Linear LR warmup for `warmup_epochs`, then cosine decay to `min_lr`.

    Preferred over WarmupThenPlateau for Phase 3 full-retrain:
    - Deterministic schedule (no patience-dependent stalling)
    - Smooth monotone decay prevents oscillation around pre-trained representations
    - Reaches min_lr exactly at the final epoch (no wasted compute)
    - Layer-wise LR decay aware: if optimizer has multiple param groups with
      different initial LRs (set by make_layerwise_optimizer), this scheduler
      preserves the LR ratios throughout training.  Each group's LR scales
      proportionally relative to its initial value — so a group at 0.05× base_lr
      stays at 0.05× throughout warmup and cosine decay.
    """

    def __init__(self, optimizer, base_lr, warmup_epochs, n_epochs, min_lr=1e-7):
        # type: (torch.optim.Optimizer, float, int, int, float) -> None
        import math as _math
        self._math = _math
        self._opt = optimizer
        self._base_lr = base_lr
        self._warmup = warmup_epochs
        self._n_epochs = n_epochs
        self._min_lr = min_lr
        self._epoch = 0
        # Snapshot the initial LR of each param group.  This is set by
        # make_layerwise_optimizer() and encodes the layer-wise multipliers.
        # All schedule operations below apply the cosine/warmup factor
        # relative to these initial values, preserving the ratios.
        # IMPORTANT: snapshot BEFORE applying the epoch-1 warmup reduction.
        self._init_lrs = [pg["lr"] for pg in optimizer.param_groups]
        # Pre-set optimiser to epoch-1 warmup LR so the first epoch trains
        # at the correct (low) value rather than the full base_lr.
        # Ratios between param groups are preserved by scaling each group's
        # LR by the same factor.
        if warmup_epochs > 0:
            frac1 = 1.0 / max(warmup_epochs, 1)
            for pg, init_lr in zip(optimizer.param_groups, self._init_lrs):
                pg["lr"] = init_lr * frac1

    def step(self, metric=None):  # metric ignored; schedule is deterministic
        self._epoch += 1
        if self._epoch <= self._warmup:
            frac = self._epoch / max(self._warmup, 1)
            for pg, init_lr in zip(self._opt.param_groups, self._init_lrs):
                pg["lr"] = init_lr * frac
        else:
            cos_epoch = self._epoch - self._warmup
            cos_total = max(self._n_epochs - self._warmup, 1)
            cos_frac  = cos_epoch / cos_total
            for pg, init_lr in zip(self._opt.param_groups, self._init_lrs):
                # Scale min_lr proportionally to the group's initial LR,
                # so that the floor LR preserves the layer-wise ratio.
                mult = init_lr / max(self._base_lr, 1e-12)
                min_lr_grp = self._min_lr * mult
                pg["lr"] = (min_lr_grp
                            + 0.5 * (init_lr - min_lr_grp)
                            * (1 + self._math.cos(self._math.pi * cos_frac)))

    @property
    def current_lr(self):
        # Report the "head" group LR (first group = highest LR = most informative)
        return self._opt.param_groups[0]["lr"]


class WarmupThenExponential:
    """Linear warmup then MONOTONE exponential decay to min_lr.  LR never increases.

    Design rationale (v3c, May 2026):
      Cosine schedule re-injects instability energy each time it rebounds upward.
      This scheduler guarantees strict monotonicity: after warmup the LR decreases
      every epoch using a fixed per-epoch exponential rate, and spike events
      permanently lower the base via permanent_reduce() — the decay continues
      from the new (lower) floor with no recovery.

    LR trajectory (post-warmup for a single param group):
      decay_rate = (min_lr / init_lr)^(1 / decay_epochs)
      lr(t) = segment_base * decay_rate^t
      where t counts epochs since the last segment_base update.

    On spike: call permanent_reduce(factor=0.5).
      - Immediately lowers all group LRs by factor.
      - Updates each segment_base to the new (reduced) LR.
      - The exponential continues from here — same per-epoch rate, lower base.
      - LR never recovers: this is intentional and permanent.

    Layer-wise ratio preservation: if optimizer has multiple param groups with
    different initial LRs (e.g. from make_v3cp2_optimizer), the per-epoch
    decay_rate is computed from each group's own init_lr, so ratios are preserved.
    """

    def __init__(self, optimizer, base_lr, warmup_epochs, n_epochs, min_lr=1e-7):
        # type: (torch.optim.Optimizer, float, int, int, float) -> None
        import math as _math
        self._math = _math
        self._opt = optimizer
        self._base_lr = base_lr
        self._warmup = warmup_epochs
        self._n_epochs = n_epochs
        self._min_lr = min_lr
        self._epoch = 0
        # Snapshot initial per-group LRs (encode differential LR multipliers)
        self._init_lrs = [pg["lr"] for pg in optimizer.param_groups]
        # segment_bases: the "current top" of the exponential for each group.
        # Starts equal to init_lrs; decremented by permanent_reduce() on spikes.
        self._segment_bases = list(self._init_lrs)
        # segment_starts: epoch number when the current segment began.
        # Restarted by permanent_reduce() so decay counts from the new base.
        self._segment_starts = [warmup_epochs] * len(self._init_lrs)
        # Pre-set epoch-1 warmup LR (proportionally for all groups)
        if warmup_epochs > 0:
            frac1 = 1.0 / max(warmup_epochs, 1)
            for pg, init_lr in zip(optimizer.param_groups, self._init_lrs):
                pg["lr"] = init_lr * frac1

    def step(self, metric=None):  # metric ignored — schedule is deterministic
        self._epoch += 1
        decay_total = max(self._n_epochs - self._warmup, 1)
        if self._epoch <= self._warmup:
            frac = self._epoch / max(self._warmup, 1)
            for pg, init_lr in zip(self._opt.param_groups, self._init_lrs):
                pg["lr"] = init_lr * frac
            return
        # Post-warmup: exponential decay per segment
        for pg, init_lr, seg_base, seg_start in zip(
                self._opt.param_groups, self._init_lrs,
                self._segment_bases, self._segment_starts):
            mult = init_lr / max(self._base_lr, 1e-12)
            min_lr_grp = max(self._min_lr * mult, 1e-12)
            t = self._epoch - seg_start  # epochs into current segment
            # Per-epoch decay rate anchored to init_lr (consistent across segments)
            if init_lr > min_lr_grp:
                rate = (min_lr_grp / init_lr) ** (1.0 / decay_total)
                lr = seg_base * (rate ** t)
            else:
                lr = seg_base
            pg["lr"] = max(lr, min_lr_grp)

    def permanent_reduce(self, factor=0.5):
        """Permanently reduce all group LRs by factor.  LR can never recover.

        Updates both the actual optimizer LRs, the segment bases, AND the warmup
        init_lrs so the remaining warmup ramp continues from the reduced floor.

        BUG FIX (2026-05-29): previously only updated segment_bases/pg["lr"] but NOT
        _init_lrs.  During warmup, step() uses `init_lr × epoch/warmup` which overrides
        the reduction on the very next step — destroying the "permanent" guarantee during
        warmup.  Fix: scale down _init_lrs[i] so that the warmup ramp at the current
        epoch gives exactly new_lr, and future warmup epochs scale proportionally.

        Args:
            factor: multiplicative reduction (default 0.5 = halve the LR).
        """
        for i, pg in enumerate(self._opt.param_groups):
            new_lr = pg["lr"] * factor
            # Clamp to min_lr floor (per-group, proportional)
            mult = self._init_lrs[i] / max(self._base_lr, 1e-12)
            min_lr_grp = max(self._min_lr * mult, 1e-12)
            new_lr = max(new_lr, min_lr_grp)
            pg["lr"] = new_lr
            self._segment_bases[i] = new_lr
            self._segment_starts[i] = self._epoch   # decay restarts from here

            # FIX: if still in warmup, reduce _init_lrs so the warmup ramp
            # uses the new floor going forward instead of the original peak.
            if self._epoch < self._warmup:
                frac = self._epoch / max(self._warmup, 1)
                if frac > 0:
                    # new_init = new_lr / frac  →  warmup at current epoch gives new_lr
                    new_init_lr = new_lr / frac
                    # Only ever reduce — never let permanent_reduce RAISE the init_lr
                    self._init_lrs[i] = min(self._init_lrs[i], new_init_lr)

    @property
    def current_lr(self):
        return self._opt.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Layer-wise LR decay optimizer (Phase 3)
# ---------------------------------------------------------------------------

def make_layerwise_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with layer-wise LR decay for Phase 3 full-retrain.

    Why: the pretrained equivariant layers encode physically correct geometric
    representations of peptide-receptor interactions.  Applying a uniform lr to
    all 7.5M params in Phase 3 risks overwriting these representations (the
    'catastrophic forgetting' we observed as val-loss blowup in Phase 1/2).
    Layer-wise LR decay lets outer heads adapt quickly while the inner geometry
    layers barely move.

    LR multipliers (relative to base_lr):
        score heads + embeddings:     1.00× — high-level signal, can adapt fast
        late cross/intra_convs:       0.50× — receptor-interaction adaptation
        middle conv layers:           0.20× — geometry, move carefully
        early equivariant layers:     0.05× — core physics priors, minimal drift
        anything unmatched:           0.20× — conservative default

    Returns:
        torch.optim.Adam with per-group lr baked in.
        WarmupThenCosine automatically preserves these ratios throughout training.
    """
    # (patterns, multiplier) — first match wins; use trailing dots to avoid
    # partial matches (e.g. "cross_convs.3." does not match "cross_convs.30.")
    LAYERWISE = [
        (["cross_convs.0.", "intra_convs.0."],                        0.05),
        (["cross_convs.1.", "intra_convs.1.", "intra_convs.2."],      0.20),
        (["cross_convs.2.", "cross_convs.3.", "intra_convs.3."],      0.50),
        # Score output heads and their direct feed embeddings
        (["tr_final_layer", "rot_final_layer",
          "tor_bb_final_layer", "tor_sc_final_layer",
          "final_conv", "tor_bb_bond_conv", "tor_sc_bond_conv",
          "center_edge_embedding", "pep_a_node_embedding",
          "final_edge_embedding"],                                     1.00),
    ]
    DEFAULT_MULT = 0.20

    groups = {}   # multiplier (float) → list of params
    param_to_mult = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT
        for patterns, m in LAYERWISE:
            if any(pat in name for pat in patterns):
                mult = m
                break
        groups.setdefault(mult, []).append(param)
        param_to_mult[name] = mult

    # Build param group list, sorted descending by LR (heads first for logging)
    param_groups = [
        {"params": params,
         "lr": base_lr * mult,
         "weight_decay": weight_decay,
         "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[Phase 3] Layer-wise LR decay:")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v3p2_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 3-tier differential LR for v3 Phase 2.

    Tier assignment (first match wins, trailing dot prevents partial matches):
        intra_convs.3.*  →  0.15× base_lr  (outermost peptide conv: very slow drift)
        cross_convs.2.*  →  0.70× base_lr  (2nd receptor ring: conservative)
        everything else  →  1.00× base_lr  (score heads + cross_convs.3: full speed)

    Only unfrozen params (requires_grad=True) are included.
    """
    TIERS = [
        ("intra_convs.3.",  0.15),
        ("cross_convs.2.",  0.70),
    ]
    DEFAULT_MULT = 1.0

    groups: dict = {}  # mult → list of params
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT
        for pat, m in TIERS:
            if pat in name:
                mult = m
                break
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {
            "params": params,
            "lr": base_lr * mult,
            "weight_decay": weight_decay,
            "lr_mult": mult,
        }
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V3 Phase 2] Differential LR optimizer (3 tiers):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v5p3_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with AGGRESSIVE layer-wise LR decay for v5 Phase 3.

    Ultra-conservative multipliers preserve the strong pretrained geometric prior
    even during full-model retraining.  Contrast with standard Phase 3 (0.5/0.2/0.05).

    LR multipliers (relative to base_lr):
        score heads + embeddings:     1.00× — high-level signal, can adapt
        late cross/intra_convs:       0.30× — more conservative than v3 (was 0.50×)
        middle conv layers:           0.10× — strong prior preservation (was 0.20×)
        early equivariant layers:     0.02× — near-zero drift on core physics (was 0.05×)
        anything unmatched:           0.10× — conservative default
    """
    LAYERWISE_V5 = [
        (["cross_convs.0.", "intra_convs.0."],                        0.02),
        (["cross_convs.1.", "intra_convs.1.", "intra_convs.2."],      0.10),
        (["cross_convs.2.", "cross_convs.3.", "intra_convs.3."],      0.30),
        (["tr_final_layer", "rot_final_layer",
          "tor_bb_final_layer", "tor_sc_final_layer",
          "final_conv", "tor_bb_bond_conv", "tor_sc_bond_conv",
          "center_edge_embedding", "pep_a_node_embedding",
          "final_edge_embedding"],                                     1.00),
    ]
    DEFAULT_MULT_V5 = 0.10  # more conservative than standard 0.20

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT_V5
        for patterns, m in LAYERWISE_V5:
            if any(pat in name for pat in patterns):
                mult = m
                break
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult,
         "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V5 Phase 3] Aggressive layer-wise LR decay:")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v3bp2_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 2-tier differential LR for v3b Phase 2.

    Tier assignment (first match wins, trailing dot prevents partial matches):
        cross_convs.2.*  →  0.60× base_lr  (newly unfrozen ring: conservative)
        everything else  →  1.00× base_lr  (score heads + cross_convs.3: full speed)

    Rationale: cross_convs.2 is deeper than cross_convs.3 and encodes more
    fundamental receptor geometry — give it 60% of the heads LR to avoid
    overwriting pretrained structure while still adapting.

    Only unfrozen params (requires_grad=True) are included.
    """
    TIERS = [
        ("cross_convs.2.",  0.60),
    ]
    DEFAULT_MULT = 1.0

    groups: dict = {}  # mult → list of params
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT
        for pat, m in TIERS:
            if pat in name:
                mult = m
                break
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {
            "params": params,
            "lr": base_lr * mult,
            "weight_decay": weight_decay,
            "lr_mult": mult,
        }
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V3B Phase 2] Differential LR optimizer (2 tiers):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v4np2_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 2-tier differential LR for v4n Phase 2.

    More conservative than v3b: cross_convs.2 at 0.5× (v3b uses 0.6×).
    Rationale: v4n is a careful mechanistic probe — err on the side of less
    adaptation to preserve pretrained receptor geometry.

    Tier assignment:
        cross_convs.2.*  →  0.50× base_lr  (newly unfrozen ring: very conservative)
        everything else  →  1.00× base_lr  (score heads + cross_convs.3: full speed)
    """
    TIERS = [
        ("cross_convs.2.",  0.50),
    ]
    DEFAULT_MULT = 1.0

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT
        for pat, m in TIERS:
            if pat in name:
                mult = m
                break
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V4N Phase 2] Differential LR optimizer (2 tiers, conservative):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v3cp1_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 2-tier differential LR for v3c Phase 1.

    v3c Phase 1 unfreezes score heads + output convolutions only.
    The score magnitude scalers (tr/rot/tor_*_final_layer) get full LR;
    the output equivariant convolutions (final_conv, tor_*_bond_conv) get 0.5×
    to avoid destabilising the geometry-encoding part of the output stream.

    Tier assignment (first match wins):
        final_conv.*  /  tor_bb_bond_conv.*  /  tor_sc_bond_conv.*  →  0.50× base_lr
        tr_final_layer.*  /  rot_final_layer.*  /  tor_*_final_layer.*  →  1.00× base_lr

    Rationale: output convs sit closer to the equivariant geometry pipeline than
    the scalar heads; 0.5× preserves the geometry-encoding prior while still
    allowing score calibration.
    """
    OUTPUT_CONV_PATTERNS = ["final_conv.", "tor_bb_bond_conv.", "tor_sc_bond_conv."]
    OUTPUT_CONV_MULT = 0.50
    DEFAULT_MULT = 1.0

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = OUTPUT_CONV_MULT if any(p in name for p in OUTPUT_CONV_PATTERNS) else DEFAULT_MULT
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V3C Phase 1] Differential LR optimizer (2 tiers: heads=1.0×, output_convs=0.5×):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v3cp2_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 3-tier differential LR for v3c Phase 2.

    v3c Phase 2 adds cross_convs.3 at extremely conservative 0.1× LR.
    This is the most restricted cross-conv adaptation in the entire v3/v4/v5/v3c
    family — the goal is to absorb receptor distribution shift while keeping
    the manifold almost entirely intact.

    Tier assignment (first match wins):
        cross_convs.3.*                                             →  0.10× base_lr
        final_conv.*  /  tor_bb_bond_conv.*  /  tor_sc_bond_conv.* →  0.50× base_lr
        everything else (score heads)                               →  1.00× base_lr

    Rationale: cross_convs.3 at 0.1× gives ~10 epochs of meaningful adaptation
    before it effectively freezes at the exponential floor.  This is intentional.
    """
    TIERS = [
        ("cross_convs.3.",   0.10),   # one outermost receptor ring — ultra-conservative
    ]
    OUTPUT_CONV_PATTERNS = ["final_conv.", "tor_bb_bond_conv.", "tor_sc_bond_conv."]
    OUTPUT_CONV_MULT = 0.50
    DEFAULT_MULT = 1.0

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT
        for pat, m in TIERS:
            if pat in name:
                mult = m
                break
        else:
            # Not matched by TIERS — check output conv patterns
            if any(p in name for p in OUTPUT_CONV_PATTERNS):
                mult = OUTPUT_CONV_MULT
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V3C Phase 2] Differential LR optimizer (3 tiers: heads=1.0×, output_convs=0.5×, cross_convs.3=0.1×):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v4cp1_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 2-tier differential LR for v4c Phase 1.

    v4c Phase 1 unfreezes score heads + output convs + cross_convs.3.
    The cross_conv ring gets a very conservative 0.15× LR to probe receptor
    interaction without destabilising the pretrained manifold.

    Tier assignment (first match wins):
        cross_convs.3.*  →  0.15× base_lr  (outermost receptor ring — ultra-conservative)
        everything else  →  1.00× base_lr  (score heads + output convs)
    """
    TIERS = [
        ("cross_convs.3.",  0.15),
    ]
    DEFAULT_MULT = 1.0

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT
        for pat, m in TIERS:
            if pat in name:
                mult = m
                break
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V4C Phase 1] Differential LR optimizer (2 tiers: heads+output=1.0×, cross_convs.3=0.15×):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v4cp2_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 3-tier differential LR for v4c Phase 2.

    Adds cross_convs.2 at 0.08× — even more conservative than cross_convs.3 (0.15×).
    The logic: cc.2 encodes deeper receptor geometry than cc.3; it gets a lower LR
    to test whether any useful signal leaks in at near-zero energy.

    Tier assignment (first match wins):
        cross_convs.2.*  →  0.08× base_lr  (2nd outermost ring — near-frozen)
        cross_convs.3.*  →  0.15× base_lr  (outermost ring — continued from P1)
        everything else  →  1.00× base_lr  (score heads + output convs)
    """
    TIERS = [
        ("cross_convs.2.",  0.08),
        ("cross_convs.3.",  0.15),
    ]
    DEFAULT_MULT = 1.0

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT
        for pat, m in TIERS:
            if pat in name:
                mult = m
                break
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V4C Phase 2] Differential LR optimizer (3 tiers: heads+output=1.0×, cross_convs.3=0.15×, cross_convs.2=0.08×):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v5cp1_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with flat 1.0× LR for v5c Phase 1.

    v5c Phase 1 unfreezes ONLY the four score heads (tr/rot/tor_bb/tor_sc final layers).
    No output convolutions, no cross_convs, no intra_convs — zero geometry influence.
    Single-tier optimizer (all heads equally important for score calibration).

    LR multipliers:
        score heads (tr/rot/tor_bb/tor_sc _final_layer): 1.0× — full base LR
    """
    HEAD_PATTERNS = [
        "tr_final_layer", "rot_final_layer", "tor_bb_final_layer", "tor_sc_final_layer",
    ]
    heads_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        for p in HEAD_PATTERNS:
            if p in name:
                heads_params.append(param)
                break

    param_groups = [
        {"params": heads_params, "lr": base_lr, "lr_mult": 1.0},
    ]

    print("[V5C Phase 1] Single-tier optimizer (score heads only, 1.0×):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v5cp2_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 2-tier differential LR for v5c Phase 2.

    v5c Phase 2 adds output convolutions alongside score heads.
    Output convs get 0.5× (same conservative ratio as v3c P1).
    No cross_conv is ever touched in v5c.

    LR multipliers:
        score heads:          1.0× — full LR (small params, high-level signal)
        output convolutions:  0.5× — conservative (geometry-adjacent)
    """
    OUTPUT_CONV_PATTERNS = ["final_conv.", "tor_bb_bond_conv.", "tor_sc_bond_conv."]
    OUTPUT_CONV_MULT = 0.50
    HEAD_PATTERNS = [
        "tr_final_layer", "rot_final_layer", "tor_bb_final_layer", "tor_sc_final_layer",
    ]

    output_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_output = any(p in name for p in OUTPUT_CONV_PATTERNS)
        is_head = any(p in name for p in HEAD_PATTERNS)
        if is_output:
            output_params.append(param)
        elif is_head:
            head_params.append(param)
        # else: unfrozen but unmatched — add to heads group as default
        else:
            head_params.append(param)

    param_groups = [
        {"params": head_params,   "lr": base_lr * 1.0,             "lr_mult": 1.0},
        {"params": output_params, "lr": base_lr * OUTPUT_CONV_MULT, "lr_mult": OUTPUT_CONV_MULT},
    ]
    # Remove empty groups
    param_groups = [pg for pg in param_groups if pg["params"]]

    print("[V5C Phase 2] Differential LR optimizer (2 tiers: heads=1.0×, output_convs=0.5×):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v4np3_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with careful layer-wise LR decay for v4n Phase 3.

    Between standard (0.5/0.2/0.05) and v5-aggressive (0.3/0.1/0.02).
    Designed to probe adaptation limits while protecting early equivariant layers.

    LR multipliers:
        score heads + embeddings:     1.00× — high-level signal, adapt freely
        late cross/intra_convs:       0.40× — modest adaptation (vs 0.50× standard)
        middle conv layers:           0.15× — careful (vs 0.20× standard)
        early equivariant layers:     0.03× — near-frozen (vs 0.05× standard)
        anything unmatched:           0.15× — conservative default
    """
    LAYERWISE_V4N = [
        (["cross_convs.0.", "intra_convs.0."],                        0.03),
        (["cross_convs.1.", "intra_convs.1.", "intra_convs.2."],      0.15),
        (["cross_convs.2.", "cross_convs.3.", "intra_convs.3."],      0.40),
        (["tr_final_layer", "rot_final_layer",
          "tor_bb_final_layer", "tor_sc_final_layer",
          "final_conv", "tor_bb_bond_conv", "tor_sc_bond_conv",
          "center_edge_embedding", "pep_a_node_embedding",
          "final_edge_embedding"],                                     1.00),
    ]
    DEFAULT_MULT_V4N = 0.15

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT_V4N
        for patterns, m in LAYERWISE_V4N:
            if any(pat in name for pat in patterns):
                mult = m
                break
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult,
         "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V4N Phase 3] Careful layer-wise LR decay (0.40/0.15/0.03):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v5np3_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with ultra-conservative layer-wise LR decay for v5n Phase 3.

    Designed to preserve the pretrained diffusion manifold while allowing
    only the outermost score-producing layers to meaningfully adapt.

    LR multipliers (most conservative of all fine-tuning experiments):
        score heads + output convs:   1.00× — calibration target, adapt freely
        late cross/intra_convs:       0.25× — minimal receptor adaptation
        middle conv layers:           0.08× — near-frozen geometry
        early equivariant layers:     0.02× — essentially frozen in place
        anything unmatched:           0.08× — conservative default

    Rationale: v5n hypothesis is that the pretrained prior is near-optimal.
    Deep layers should barely move. Any meaningful signal should concentrate
    in the score heads.
    """
    LAYERWISE_V5N = [
        (["cross_convs.0.", "intra_convs.0.",
          "rec_node_embedding", "pep_node_embedding"],          0.02),
        (["cross_convs.1.", "intra_convs.1.", "intra_convs.2."], 0.08),
        (["cross_convs.2.", "cross_convs.3.", "intra_convs.3."], 0.25),
        (["tr_final_layer", "rot_final_layer",
          "tor_bb_final_layer", "tor_sc_final_layer",
          "final_conv", "tor_bb_bond_conv", "tor_sc_bond_conv",
          "center_edge_embedding", "pep_a_node_embedding",
          "final_edge_embedding"],                               1.00),
    ]
    DEFAULT_MULT_V5N = 0.08   # conservative fallback for unmatched layers

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT_V5N
        for patterns, m in LAYERWISE_V5N:
            if any(pat in name for pat in patterns):
                mult = m
                break
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult,
         "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V5N Phase 3] Ultra-conservative layer-wise LR decay (1.0/0.25/0.08/0.02):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v6p1_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 2-tier differential LR for V6 Phase 1.

    V6 Phase 1 unfreezes score heads + tor_bb_bond_conv.
    tor_bb_bond_conv is the backbone torsion conv — it encodes very_long peptide
    backbone geometry. Giving it 0.5× LR prevents destabilisation while allowing
    adaptation.

    Tier assignment:
        tor_bb_bond_conv.*  →  0.50× base_lr  (backbone geometry: careful adaptation)
        score heads         →  1.00× base_lr  (scalar projections: full speed)

    Rationale: tor_bb_bond_conv has 967,584 params and encodes complex equivariant
    geometry — a 0.5× LR gives it time to adapt steadily. The score heads are only
    ~12,674 params and can absorb full LR without instability risk.
    """
    TOR_BB_BOND_PATTERN = "tor_bb_bond_conv."
    TOR_BB_BOND_MULT = 0.50
    DEFAULT_MULT = 1.0

    groups = {}  # type: dict
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = TOR_BB_BOND_MULT if TOR_BB_BOND_PATTERN in name else DEFAULT_MULT
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V6 Phase 1] 2-tier optimizer (heads=1.0×, tor_bb_bond_conv=0.5×):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v6p2_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam with 4-tier differential LR for V6 Phase 2.

    V6 Phase 2 adds all cross_convs. Deep cross_convs (0/1) encode fundamental
    receptor geometry — they get 0.4× to preserve pretrained structure. Outer
    cross_convs (2/3) are more data-dependent — 0.7×.

    Tier assignment (first match wins, trailing dot prevents partial matches):
        cross_convs.0.*    →  0.40× base_lr  (deepest, most fundamental receptor repr)
        cross_convs.1.*    →  0.40× base_lr  (deep receptor geometry)
        cross_convs.2.*    →  0.70× base_lr  (outer receptor ring: conservative)
        cross_convs.3.*    →  0.70× base_lr  (outermost receptor ring)
        tor_bb_bond_conv.* →  0.50× base_lr  (backbone torsion: continued from P1)
        everything else    →  1.00× base_lr  (score heads)

    L2 regularization (applied separately via pretrained_ref):
        cross_convs.0/1/2/3: λ=3e-4 anchors to pretrained values
    """
    TIERS = [
        ("cross_convs.0.", 0.40),
        ("cross_convs.1.", 0.40),
        ("cross_convs.2.", 0.70),
        ("cross_convs.3.", 0.70),
        ("tor_bb_bond_conv.", 0.50),
    ]
    DEFAULT_MULT = 1.0

    groups = {}  # type: dict
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = DEFAULT_MULT
        for pat, m in TIERS:
            if pat in name:
                mult = m
                break
        groups.setdefault(mult, []).append(param)

    param_groups = [
        {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay, "lr_mult": mult}
        for mult, params in sorted(groups.items(), key=lambda x: -x[0])
        if params
    ]

    print("[V6 Phase 2] 4-tier optimizer (cc.2/3=0.7×, tor_bb=0.5×, cc.0/1=0.4×, heads=1.0×):")
    for pg in param_groups:
        n_p = sum(p.numel() for p in pg["params"])
        print(f"  lr={pg['lr']:.2e} ({pg['lr_mult']:.2f}×)  —  {n_p:,} params")

    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def make_v6p3_optimizer(model, base_lr, weight_decay=0.0):
    # type: (torch.nn.Module, float, float) -> torch.optim.Optimizer
    """Build Adam for V6 Phase 3 (consolidation) — same tiers as P2, lower LR.

    Phase 3 uses uniform sampling and a low flat LR (5e-7) to consolidate gains.
    Same 4-tier structure as P2 to preserve the relative LR ratios during consolidation.
    """
    return make_v6p2_optimizer(model, base_lr, weight_decay)


# ---------------------------------------------------------------------------
# Pretrained-weight L2 regularization
# ---------------------------------------------------------------------------

def build_pretrained_ref(model, ckpt_path, reg_patterns, device):
    # type: (torch.nn.Module, str, list, torch.device) -> dict
    """Extract frozen reference weights from the pretrained checkpoint.

    Returns a dict {param_name: frozen_tensor} for all params whose name
    contains at least one pattern in reg_patterns AND that param has
    requires_grad=True in the current model (i.e. will actually be trained).
    Only trainable params that need regularization are stored — ignores
    permanently frozen params since those can't drift anyway.

    The returned tensors are detached, on `device`, and NOT tracked by autograd.

    Args:
        model:        Current (already-loaded) model.
        ckpt_path:    Path to the pretrained .pt checkpoint.
        reg_patterns: List of substrings; a param is regularized if any match.
        device:       Target device for reference tensors.

    Returns:
        {name: tensor} for params matching any reg_pattern AND requires_grad.
    """
    raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        state = raw_ckpt["model"]
    else:
        state = raw_ckpt  # bare state dict

    ref = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen layer — can't drift, no need to regularize
        if not any(pat in name for pat in reg_patterns):
            continue  # not a regularized layer
        if name in state:
            ref[name] = state[name].to(device).detach()
        else:
            print(f"[pretrained-reg] WARN: {name!r} not in checkpoint — skipping")

    n_params = sum(t.numel() for t in ref.values())
    print(f"[pretrained-reg] Tracking {len(ref)} tensors ({n_params:,} params) "
          f"for L2 proximity to pretrained init")
    return ref


def compute_pretrained_reg_loss(model, pretrained_ref, reg_lambda):
    # type: (torch.nn.Module, dict, float) -> torch.Tensor
    """Compute weak L2 penalty: reg_lambda * sum_i ||theta_i - theta0_i||^2 / n_i.

    Normalized per-tensor by numel so lambda is scale-invariant across layer sizes.
    Returns a scalar tensor on the same device as the model.

    Args:
        model:          Current (training) model.
        pretrained_ref: Output of build_pretrained_ref().
        reg_lambda:     Regularization strength (e.g. 1e-4 to 1e-2 range).
    """
    if not pretrained_ref or reg_lambda <= 0.0:
        return torch.tensor(0.0)

    reg = None
    current = dict(model.named_parameters())
    for name, ref_tensor in pretrained_ref.items():
        if name not in current:
            continue
        param = current[name]
        if not param.requires_grad:
            continue
        diff = (param - ref_tensor).pow(2).mean()  # normalized by numel
        reg = diff if reg is None else reg + diff

    if reg is None:
        return torch.tensor(0.0)
    return reg_lambda * reg


# ---------------------------------------------------------------------------
# Checkpoint save / resume helpers
# ---------------------------------------------------------------------------

def find_resume_checkpoint(output_dir):
    # type: (Path) -> tuple
    """Scan output_dir for the latest epoch checkpoint.

    Returns:
        (path_str, epoch_int) for the highest-numbered epoch checkpoint found,
        or (None, 0) if none exist.
    """
    import glob as _glob
    import re as _re
    candidates = _glob.glob(str(output_dir / "rapidock_finetuned_epoch*.pt"))
    best_epoch = 0
    best_path = None
    for p in candidates:
        m = _re.search(r"epoch(\d+)\.pt$", p)
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_path = p
    return best_path, best_epoch


def save_checkpoint(model, optimizer, epoch, loss, path, ema=None, patterns=None):
    # type: (...) -> None
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": loss,
    }
    if patterns:
        payload["finetuned_heads"] = {
            k: v for k, v in model.state_dict().items()
            if any(p in k for p in patterns)
        }
    if ema is not None:
        payload["ema_weights"] = ema.state_dict()
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # type: () -> None
    parser = argparse.ArgumentParser(description="Phase-aware fine-tuning for RAPiDock")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--checkpoint", required=True,
                        help="Pre-trained .pt checkpoint to start from")
    parser.add_argument(
        "--model-params",
        default=str(_HERE / "train_models" / "CGTensorProductEquivariantModel" / "model_parameters.yml"),
    )
    parser.add_argument("--output-dir", default="finetune_out")
    parser.add_argument("--unfreeze-phase", type=int, choices=[1, 2, 3], default=1,
                        help="1=score heads only (~5%%), 2=+last equivariant block (~20%%), "
                             "3=all layers (full retrain)")
    parser.add_argument(
        "--v3-mode", action="store_true",
        help="Enable v3 controlled-specialization strategy. "
             "P1: unfreeze score heads + cross_convs.2/3 (intra_convs entirely frozen). "
             "P2: + intra_convs.3 at 0.15× LR; differential LR for all 3 tiers. "
             "P3: all EXCEPT ESM projection; weak L2 pretrained-weight regularization. "
             "Adds oscillation amplitude monitoring and score-norm variance tracking."
    )
    parser.add_argument(
        "--v4-mode", action="store_true",
        help="V4: pure cross_conv receptor-interaction adaptation. "
             "P1/P2: score heads + cross_convs.2/3 ONLY (intra_convs frozen throughout). "
             "P2: 2-tier differential LR (cross_convs.2=0.7×, rest=1.0×). "
             "P3: all EXCEPT ESM. No pretrained-weight regularization. "
             "Ablation vs v3 — tests cross_conv-only adaptation without L2 reg."
    )
    parser.add_argument(
        "--v5-mode", action="store_true",
        help="V5: ultra-conservative adaptation — maximal preservation of pretrained prior. "
             "P1: score heads ONLY (no conv adaptation at all). "
             "P2: score heads + cross_convs.3 ONLY (single cross-conv ring). "
             "P3: full EXCEPT ESM; aggressive layerwise decay (0.30/0.10/0.02). "
             "Lower LR + tighter grad_clip throughout. "
             "Hypothesis: pretrained prior already very strong — gentle touch wins."
    )
    parser.add_argument(
        "--v3b-mode", action="store_true",
        help="V3B: stable controlled specialization — cosine LR in ALL phases, adaptive "
             "spike LR reduction. "
             "P1: score heads + cross_convs.3 ONLY (intra_convs entirely frozen). "
             "P2: + cross_convs.2 at 0.6× differential LR. "
             "P3: full EXCEPT ESM; standard layerwise decay (0.50/0.20/0.05). "
             "Key fixes vs v3: cosine (no plateau), grad_clip=0.5, EMA=0.9999 from P1, "
             "adaptive spike LR (auto-halves LR for 2 epochs on norm spike detection)."
    )
    parser.add_argument(
        "--v4n-mode", action="store_true",
        help="V4N: careful mechanistic probe — cosine ALL phases, adaptive spike LR, "
             "more conservative than v3b. "
             "P1: score heads + cross_convs.3 ONLY (identical unfreeze to v3b P1). "
             "P2: + cross_convs.2 at 0.5× differential LR (vs v3b 0.6×). "
             "P3: full EXCEPT ESM; careful layerwise decay (0.40/0.15/0.03 vs 0.50/0.20/0.05). "
             "Lower P1/P2 LR budget and tighter layerwise P3 decay probe whether "
             "mild cross_conv adaptation improves interaction diversity without instability."
    )
    parser.add_argument(
        "--v5n-mode", action="store_true",
        help="V5N: ultra-conservative manifold preservation — minimal adaptation. "
             "P1: score heads + output convs ONLY (NO embeddings, NO cross_convs). "
             "P2: + cross_convs.3 ONLY at uniform LR. "
             "P3: full EXCEPT ESM; ultra-conservative layerwise (1.0/0.25/0.08/0.02). "
             "Adaptive spike LR + EMA skip on spike (pause EMA updates for 2 epochs "
             "after norm spike to prevent unstable weights leaking into EMA). "
             "Hypothesis: pretrained prior is near-optimal; only minimal biasing needed "
             "to improve PepPC performance without narrowing exploration diversity."
    )
    parser.add_argument(
        "--v4c-mode", action="store_true",
        help="V4C: tiny cross_conv probe — monotone exponential decay, cross_convs.3 in P1 at 0.15×. "
             "P1: score heads + output convs + cross_convs.3 at 0.15× LR. "
             "P2: + cross_convs.2 at 0.08× LR. NO phase 3. "
             "Spike handling identical to v3c: permanent_reduce(0.5) + 1-epoch EMA skip. "
             "Tests whether ultra-low cross_conv adaptation in P1 provides receptor benefit "
             "without collapsing exploration diversity."
    )
    parser.add_argument(
        "--v3c-mode", action="store_true",
        help="V3C: minimal low-energy recalibration — MONOTONIC exponential decay, no rebounds. "
             "P1: score heads + output convs ONLY; 2-tier diff LR (heads=1.0×, output_convs=0.5×). "
             "P2: + cross_convs.3 at 0.1× LR (3-tier diff LR). NO phase 3. "
             "Key design: LR only decreases (WarmupThenExponential); spike events permanently "
             "reduce LR by 50% and pause EMA for 1 epoch — LR never recovers. "
             "Strong pretrained-reg on equivariant layers + intra_convs + cross_convs.0/1/2. "
             "Use --lr-schedule exponential --cosine-min-lr <floor> for the exponential floor. "
             "Hypothesis: cosine rebounds re-inject instability; monotone decay eliminates "
             "resonant spike cycles while preserving the pretrained diffusion manifold."
    )
    parser.add_argument(
        "--v5c-mode", action="store_true",
        help="V5C: ultra-minimal diversity-preserving recalibration — PRIMARY goal is preserving "
             "pretrained exploration diversity and score-field geometry. "
             "P1: ONLY final score heads (tr/rot/tor_bb/tor_sc final layers); NO output convs, "
             "NO cross_convs, NO intra_convs, NO embeddings — zero geometry influence. "
             "P2: + output convolutions at 0.5× diff LR; STILL zero cross_conv touch. "
             "Key design: far lower LR than v3c/v4c (2e-6→2e-7 / 5e-7→5e-8); slower EMA "
             "(0.99998); tighter grad_clip (0.2); weak pretrained-reg (λ=1e-4/2e-4). "
             "Spike handling: permanent_reduce(0.5) + 1-epoch EMA skip (same as v3c). "
             "NO cross_conv adaptation in any phase. NO phase 3. Save every epoch after ep4. "
             "Hypothesis: score-head recalibration alone is sufficient; any cross_conv update "
             "risks compressing the diffusion exploration manifold."
    )
    parser.add_argument(
        "--v6-mode", action="store_true",
        help="V6: targeted long/very_long peptide adaptation — cross_convs + tor_bb_bond_conv. "
             "MUST start from rapidock_global.pt (not any fine-tuned checkpoint). "
             "P1: score heads + tor_bb_bond_conv at 0.5× LR (12.98%% of params). "
             "P2: + all cross_convs, 4-tier diff LR (cc.0/1=0.4×, cc.2/3=0.7×, tor_bb=0.5×). "
             "P3: same as P2 but with lower LR (consolidation, uniform sampling). "
             "Key fixes: (1) freeze_frozen_bn_stats() prevents BatchNorm running-stat drift; "
             "(2) tier-based oversampling (3× very_long, 2× long, 1× medium/short); "
             "(3) L2 regularization on cross_convs (default λ=3e-4); "
             "(4) multi-best checkpointing: best_long, best_very_long, best_combined; "
             "(5) stratified val loss tracking by length bucket."
    )
    parser.add_argument(
        "--v6-val-csv", default=None,
        help="V6: stratified validation CSV with 'length_bucket' column. "
             "Used to track val loss per bucket (short/medium/long/very_long) every epoch. "
             "If not provided, falls back to single-bucket val_loss."
    )
    parser.add_argument(
        "--v6-guard-patience", type=int, default=3,
        help="V6: number of consecutive epochs of short/medium val loss increase "
             "before flagging a guard-rail warning. Default: 3."
    )
    parser.add_argument(
        "--v6-guard-threshold", type=float, default=0.3,
        help="V6: fractional val loss increase threshold for guard rail trigger. "
             "If short or medium bucket val_loss increases by more than this fraction "
             "relative to the baseline (min seen so far) for --v6-guard-patience consecutive "
             "epochs, a CHECKPOINT UNSAFE warning is emitted. Default: 0.3 (30%% increase)."
    )
    parser.add_argument(
        "--ema-decay", type=float, default=None,
        help="Override EMA decay (default: from model checkpoint / model_args.ema_rate). "
             "Recommended per-phase: P1=0.9995, P2=0.9997, P3=0.9999."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="Explicit weight decay for the optimizer. Overrides the phase-default "
             "(0.0 for P1, 1e-5 for P2+). Recommended: P1=1e-6, P2=1e-5."
    )
    parser.add_argument(
        "--pretrained-reg-lambda", type=float, default=0.0,
        help="Strength of L2 regularization toward pretrained-checkpoint weights "
             "(added per-sample to training loss). 0 = disabled (default). "
             "Recommended for v3: 1e-4 (weak) to 5e-4 (moderate). "
             "Applied only to layers matching --pretrained-reg-patterns."
    )
    parser.add_argument(
        "--pretrained-reg-patterns", nargs="*",
        default=None,
        help="Layer name substrings to regularize toward pretrained weights. "
             "Default: intra_convs, cross_convs.0, cross_convs.1, "
             "rec_node_embedding, pep_node_embedding."
    )
    parser.add_argument(
        "--save-every-after", type=int, default=None,
        help="Save a checkpoint every epoch after this epoch number "
             "(overrides --save-every for late-stage stability averaging). "
             "E.g. --save-every-after 20 to save every epoch from ep20 onward."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest epoch checkpoint in --output-dir. "
             "Scans for rapidock_finetuned_epoch*.pt, loads model+optimizer+EMA state, "
             "and continues from the next epoch. Requires epoch checkpoints to exist "
             "(--save-every or --save-every-after must have been set in the prior run). "
             "If no checkpoint is found, starts from scratch (safe to always pass --resume)."
    )
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup epochs (recommended: 3 for Phase 2)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = grad_accum)")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--ppii-weight", type=int, default=4,
                        help="Oversample factor for ppii_enriched entries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model + 10 samples, check n_ok > 0, then exit")
    parser.add_argument("--bail-on-zero", action="store_true", default=True,
                        help="Abort training if n_ok=0 in first epoch (default: True)")
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="Stop early if val loss (trimmed-mean) does not improve for this many "
             "consecutive epochs after warmup. 0 = disabled (default). "
             "Recommended: 20 for Phase 3 cosine schedule."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=1.0,
        help="Max gradient L2 norm for clipping (default 1.0). "
             "Use 0.5 for Phase 1 to prevent score-head destabilisation at high LR."
    )
    parser.add_argument(
        "--layerwise-lr-decay", action="store_true",
        help="Phase 3: use layer-wise LR decay "
             "(heads 1.0×, late convs 0.5×, middle 0.2×, early equivariant 0.05×). "
             "Requires --lr-schedule cosine. Preserves physics priors in early layers."
    )
    parser.add_argument(
        "--lr-schedule", default="plateau", choices=["plateau", "cosine", "exponential"],
        help="LR schedule after warmup: "
             "'plateau' = ReduceLROnPlateau (default, good for P1/P2); "
             "'cosine' = deterministic cosine decay to --cosine-min-lr (recommended for P3 full retrain); "
             "'exponential' = MONOTONE exponential decay to --cosine-min-lr (v3c only — LR never increases)."
    )
    parser.add_argument(
        "--cosine-min-lr", type=float, default=1e-7,
        help="Floor LR for cosine or exponential schedule (ignored if --lr-schedule plateau). "
             "For exponential (v3c): set to the desired LR floor (e.g. 8e-7 for P1, 1e-7 for P2)."
    )
    parser.add_argument(
        "--esm-device", default="cpu", choices=["cpu", "cuda"],
        help="Device for ESM embedding pre-computation (default: cpu).  "
             "Use 'cpu' on WSL2 to avoid TDR crashes on long-sequence batches. "
             "ESM runs once before training; the ~40-min CPU cost beats a crash at batch 790."
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model hyperparameters
    with open(args.model_params) as fh:
        params = yaml.safe_load(fh)
    model_args = Namespace(**params)
    model_args.esm_embeddings_path_train   = True
    model_args.esm_embeddings_peptide_train = None

    # Validate: at most one experiment mode flag
    active_modes = [m for m in [args.v3_mode, args.v4_mode, args.v5_mode,
                                 args.v3b_mode, args.v4n_mode, args.v5n_mode,
                                 args.v3c_mode, args.v4c_mode, args.v5c_mode,
                                 args.v6_mode] if m]
    if len(active_modes) > 1:
        parser.error("Only one of --v3-mode, --v4-mode, --v5-mode, --v3b-mode, "
                     "--v4n-mode, --v5n-mode, --v3c-mode, --v4c-mode, --v5c-mode, "
                     "--v6-mode may be set at a time.")

    # v6 forces cosine schedule — warn if user passed something else
    if args.v6_mode and args.lr_schedule not in ("cosine",):
        print(f"[V6] Overriding --lr-schedule {args.lr_schedule!r} → 'cosine' "
              f"(V6 requires cosine decay for stable cross_conv adaptation).")
        args.lr_schedule = "cosine"

    # v3c/v4c/v5c force exponential schedule — warn if user passed something else
    if (args.v3c_mode or args.v4c_mode or args.v5c_mode) and args.lr_schedule != "exponential":
        _mode_name = "V3C" if args.v3c_mode else ("V4C" if args.v4c_mode else "V5C")
        print(f"[{_mode_name}] Overriding --lr-schedule {args.lr_schedule!r} → 'exponential' "
              f"({_mode_name} requires monotone schedule).")
        args.lr_schedule = "exponential"

    mode_tag = ("V3" if args.v3_mode else "V4" if args.v4_mode
                else "V5" if args.v5_mode else "V3B" if args.v3b_mode
                else "V4N" if args.v4n_mode else "V5N" if args.v5n_mode
                else "V3C" if args.v3c_mode else "V4C" if args.v4c_mode
                else "V5C" if args.v5c_mode else "V6" if args.v6_mode else "standard")
    print(f"\nLoading checkpoint: {args.checkpoint}")
    print(f"Experiment mode: {mode_tag}")

    # V6 uses its own dedicated loader (validates pretrained-only start)
    if args.v6_mode:
        model = load_model_for_finetuning_v6(model_args, args.checkpoint, device,
                                              unfreeze_phase=args.unfreeze_phase)
        # BUG FIX: freeze BatchNorm running stats for frozen conv blocks
        n_bn_frozen = freeze_frozen_bn_stats(model)
        print(f"[V6] BatchNorm freeze: {n_bn_frozen} frozen BN layers set to eval() "
              f"(prevents running_mean/var drift in frozen conv blocks)")
        # Snapshot frozen BN stats for drift detection
        _frozen_bn_snapshot = snapshot_frozen_bn_stats(model)
        print(f"[V6] Frozen BN snapshot: {len(_frozen_bn_snapshot)} modules tracked")
    else:
        _frozen_bn_snapshot = {}
        # v4n uses the same unfreeze patterns as v3b — pass v3b_mode=True for model loading
        model = load_model_for_finetuning(model_args, args.checkpoint, device,
                                          unfreeze_phase=args.unfreeze_phase,
                                          v3_mode=args.v3_mode,
                                          v4_mode=args.v4_mode,
                                          v5_mode=args.v5_mode,
                                          v3b_mode=args.v3b_mode or args.v4n_mode,
                                          v5n_mode=args.v5n_mode,
                                          v3c_mode=args.v3c_mode,
                                          v4c_mode=args.v4c_mode,
                                          v5c_mode=args.v5c_mode)

    if args.v3_mode:
        if args.unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V3P1
        elif args.unfreeze_phase == 2:
            patterns = _UNFREEZE_PATTERNS_V3P2
        else:
            patterns = None
    elif args.v4_mode:
        patterns = _UNFREEZE_PATTERNS_V3P1 if args.unfreeze_phase < 3 else None
    elif args.v5_mode:
        if args.unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_P1
        elif args.unfreeze_phase == 2:
            patterns = _UNFREEZE_PATTERNS_V5P2
        else:
            patterns = None
    elif args.v3b_mode or args.v4n_mode:
        # v4n uses identical unfreeze patterns to v3b
        if args.unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V3BP1
        elif args.unfreeze_phase == 2:
            patterns = _UNFREEZE_PATTERNS_V3BP2
        else:
            patterns = None
    elif args.v5n_mode:
        if args.unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V5NP1
        elif args.unfreeze_phase == 2:
            patterns = _UNFREEZE_PATTERNS_V5NP2
        else:
            patterns = None
    elif args.v3c_mode:
        if args.unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V3CP1
        elif args.unfreeze_phase == 2:
            patterns = _UNFREEZE_PATTERNS_V3CP2
        else:
            patterns = None  # v3c has no phase 3 by design
    elif args.v4c_mode:
        if args.unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V4CP1
        elif args.unfreeze_phase == 2:
            patterns = _UNFREEZE_PATTERNS_V4CP2
        else:
            patterns = None  # v4c has no phase 3 by design
    elif args.v5c_mode:
        if args.unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V5CP1
        elif args.unfreeze_phase == 2:
            patterns = _UNFREEZE_PATTERNS_V5CP2
        else:
            patterns = None  # v5c has no phase 3 by design
    elif args.v6_mode:
        if args.unfreeze_phase == 1:
            patterns = _UNFREEZE_PATTERNS_V6P1
        elif args.unfreeze_phase == 2:
            patterns = _UNFREEZE_PATTERNS_V6P2
        else:
            patterns = _UNFREEZE_PATTERNS_V6P3  # phase 3 = same as P2
    else:
        patterns = (_UNFREEZE_PATTERNS_P1 if args.unfreeze_phase == 1
                    else _UNFREEZE_PATTERNS_P2 if args.unfreeze_phase == 2
                    else None)

    # EMA decay: use --ema-decay override if given, else from model_args
    ema_decay = args.ema_decay if args.ema_decay is not None else getattr(model_args, "ema_rate", 0.999)
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
    print(f"EMA decay: {ema_decay}")

    # Weight decay: use --weight-decay if given, else phase default
    if args.weight_decay is not None:
        wd = args.weight_decay
    else:
        wd = 1e-5 if args.unfreeze_phase >= 2 else 0.0
    print(f"Weight decay: {wd}")

    if args.v5n_mode and args.unfreeze_phase == 3:
        # V5N Phase 3: ultra-conservative layerwise decay (1.0/0.25/0.08/0.02)
        optimizer = make_v5np3_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v5_mode and args.unfreeze_phase == 3:
        # V5 Phase 3: aggressive layerwise decay (0.30/0.10/0.02 instead of 0.50/0.20/0.05)
        optimizer = make_v5p3_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.layerwise_lr_decay and args.unfreeze_phase == 3:
        optimizer = make_layerwise_optimizer(model, base_lr=args.lr,
                                             weight_decay=wd)
    elif (args.v3_mode or args.v4_mode) and args.unfreeze_phase == 2:
        # V3 Phase 2: 3-tier differential LR (intra_convs.3=0.15×, cross_convs.2=0.70×, rest=1.0×)
        # V4 Phase 2: auto 2-tier (intra_convs.3 frozen → that tier is empty)
        optimizer = make_v3p2_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v3b_mode and args.unfreeze_phase == 2:
        # V3B Phase 2: 2-tier differential LR (cross_convs.2=0.60×, rest=1.0×)
        optimizer = make_v3bp2_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v4n_mode and args.unfreeze_phase == 2:
        # V4N Phase 2: 2-tier differential LR (cross_convs.2=0.50×, rest=1.0×)
        optimizer = make_v4np2_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v4n_mode and args.unfreeze_phase == 3 and args.layerwise_lr_decay:
        # V4N Phase 3: careful layerwise decay (0.40/0.15/0.03 vs standard 0.50/0.20/0.05)
        optimizer = make_v4np3_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v3c_mode and args.unfreeze_phase == 1:
        # V3C Phase 1: 2-tier diff LR (output_convs=0.5×, heads=1.0×)
        optimizer = make_v3cp1_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v3c_mode and args.unfreeze_phase == 2:
        # V3C Phase 2: 3-tier diff LR (cross_convs.3=0.1×, output_convs=0.5×, heads=1.0×)
        optimizer = make_v3cp2_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v4c_mode and args.unfreeze_phase == 1:
        # V4C Phase 1: 2-tier diff LR (cross_convs.3=0.15×, heads+output=1.0×)
        optimizer = make_v4cp1_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v4c_mode and args.unfreeze_phase == 2:
        # V4C Phase 2: 3-tier diff LR (cross_convs.2=0.08×, cross_convs.3=0.15×, heads+output=1.0×)
        optimizer = make_v4cp2_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v5c_mode and args.unfreeze_phase == 1:
        # V5C Phase 1: single-tier (score heads only at 1.0×)
        optimizer = make_v5cp1_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v5c_mode and args.unfreeze_phase == 2:
        # V5C Phase 2: 2-tier diff LR (heads=1.0×, output_convs=0.5×)
        optimizer = make_v5cp2_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v6_mode and args.unfreeze_phase == 1:
        # V6 Phase 1: 2-tier diff LR (heads=1.0×, tor_bb_bond_conv=0.5×)
        optimizer = make_v6p1_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v6_mode and args.unfreeze_phase == 2:
        # V6 Phase 2: 4-tier diff LR (cc.0/1=0.4×, cc.2/3=0.7×, tor_bb=0.5×, heads=1.0×)
        optimizer = make_v6p2_optimizer(model, base_lr=args.lr, weight_decay=wd)
    elif args.v6_mode and args.unfreeze_phase == 3:
        # V6 Phase 3: consolidation — same tiers as P2 but lower LR (passed via --lr)
        optimizer = make_v6p3_optimizer(model, base_lr=args.lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=wd,
        )
        if args.layerwise_lr_decay and args.unfreeze_phase != 3:
            print("[WARN] --layerwise-lr-decay is only applied for Phase 3; "
                  "ignoring for Phase %d." % args.unfreeze_phase)

    # Pretrained-weight regularization: build reference weight snapshot
    # V6 default: apply to all cross_convs (λ=3e-4)
    reg_lambda = args.pretrained_reg_lambda
    if args.v6_mode and reg_lambda == 0.0:
        reg_lambda = 3e-4
        print(f"[V6] Auto-enabling pretrained-reg λ=3e-4 on cross_convs (default for v6).")
    pretrained_ref = {}
    if reg_lambda > 0.0:
        if args.v6_mode:
            reg_patterns = _PRETRAINED_REG_PATTERNS_V6
        elif args.pretrained_reg_patterns:
            reg_patterns = args.pretrained_reg_patterns
        else:
            reg_patterns = _PRETRAINED_REG_PATTERNS_DEFAULT
        print(f"[pretrained-reg] lambda={reg_lambda:.2e}  "
              f"patterns={reg_patterns}")
        pretrained_ref = build_pretrained_ref(model, args.checkpoint,
                                              reg_patterns, device)
    else:
        print("[pretrained-reg] disabled (--pretrained-reg-lambda 0)")
    if args.lr_schedule == "cosine":
        scheduler = WarmupThenCosine(
            optimizer,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            n_epochs=args.n_epochs,
            min_lr=args.cosine_min_lr,
        )
        print(f"LR schedule: cosine  base={args.lr}  min={args.cosine_min_lr}  "
              f"warmup={args.warmup_epochs}")
    elif args.lr_schedule == "exponential":
        scheduler = WarmupThenExponential(
            optimizer,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            n_epochs=args.n_epochs,
            min_lr=args.cosine_min_lr,
        )
        print(f"LR schedule: exponential (MONOTONE)  base={args.lr}  "
              f"min={args.cosine_min_lr}  warmup={args.warmup_epochs}  "
              f"decay_epochs={args.n_epochs - args.warmup_epochs}")
    else:
        scheduler = WarmupThenPlateau(
            optimizer,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            plateau_kwargs={"mode": "min", "patience": 8, "factor": 0.5, "min_lr": 1e-6},
        )
        print(f"LR schedule: plateau  base={args.lr}  warmup={args.warmup_epochs}")
    transform = NoiseTransform(model_args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = str(out_dir / "processed_train")

    print(f"\nBuilding training dataset from {args.train_csv}")
    print(f"ESM device: {args.esm_device}  "
          f"({'CPU avoids WSL2 TDR; ~40 min one-time cost' if args.esm_device == 'cpu' else 'GPU fast but may crash on WSL2'})")
    train_ds = build_dataset(args.train_csv, model_args, processed_dir,
                             esm_device=args.esm_device)

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_train = len(train_ds)
    print(f"Training complexes loaded: {n_train}")

    # ── Source / Tier-weighted sampling ────────────────────────────────────
    _SOURCE_WEIGHTS = {
        "ppii_enriched":    args.ppii_weight,
        "recent_2024_2026": 2,
        "peppc":            1,
        "peppcf":           1,
        "refpepdb":         1,
    }
    df_train = pd.read_csv(args.train_csv)

    # V6 mode: tier-based oversampling supersedes source-based weighting
    if args.v6_mode and "tier" in df_train.columns:
        tiers = df_train["tier"].tolist()
        weighted_train_indices = []  # type: List[int]
        tier_counts = {}  # type: dict
        for i, tier in enumerate(tiers):
            tier_str = str(tier).strip() if str(tier) != "nan" else "replay"
            w = _V6_TIER_WEIGHTS.get(tier_str, 1)
            weighted_train_indices.extend([i] * w)
            tier_counts[tier_str] = tier_counts.get(tier_str, 0) + w
        print(f"[V6] Tier-weighted epoch size: {len(weighted_train_indices)}")
        print(f"[V6] Tier breakdown (effective counts): {tier_counts}")
    else:
        # Standard source-based weighting (all non-v6 modes)
        sources = (df_train["source"].tolist()
                   if "source" in df_train.columns
                   else ["refpepdb"] * n_train)
        weighted_train_indices = []
        for i, src in enumerate(sources):
            w = _SOURCE_WEIGHTS.get(str(src).strip(), 1)
            weighted_train_indices.extend([i] * w)
        src_counts = {}  # type: dict
        for src in sources:
            src_counts[src] = src_counts.get(src, 0) + _SOURCE_WEIGHTS.get(str(src).strip(), 1)
        print(f"Weighted epoch size: {len(weighted_train_indices)}  breakdown: {src_counts}")

    # ── Validation dataset ──────────────────────────────────────────────────
    val_ds = None
    # V6 stratified val: prefer --v6-val-csv if in v6 mode; fall back to --val-csv
    _val_csv_path = None
    if args.v6_mode and getattr(args, 'v6_val_csv', None) and Path(args.v6_val_csv).exists():
        _val_csv_path = args.v6_val_csv
        print(f"[V6] Using stratified validation CSV: {_val_csv_path}")
    elif args.val_csv and Path(args.val_csv).exists():
        _val_csv_path = args.val_csv

    # Load val bucket map for V6 stratified tracking
    _val_bucket_indices = {}   # type: dict  # bucket_name → list of val dataset indices
    if args.v6_mode and _val_csv_path:
        _df_val_check = pd.read_csv(_val_csv_path)
        if "length_bucket" in _df_val_check.columns:
            for bucket in ["short", "medium", "long", "very_long"]:
                _val_bucket_indices[bucket] = [
                    i for i, b in enumerate(_df_val_check["length_bucket"].tolist())
                    if str(b).strip() == bucket
                ]
            print(f"[V6] Val bucket sizes: " +
                  "  ".join(f"{k}={len(v)}" for k, v in _val_bucket_indices.items()))

    if _val_csv_path:
        print(f"Building validation dataset from {_val_csv_path}")
        val_ds = build_dataset(_val_csv_path, model_args, str(out_dir / "processed_val"),
                               esm_device=args.esm_device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Validation complexes: {len(val_ds)}")
    elif args.val_csv:
        pass  # val_csv was set but doesn't exist — fall through, val_ds = None

    # ── Dry-run check ───────────────────────────────────────────────────────
    if args.dry_run:
        print("\n── DRY RUN: testing 10 training samples ─────────────────────────")
        test_idx = list(range(min(10, n_train)))
        _, n_ok_test, n_total_test, _ = train_epoch(
            model, train_ds, test_idx, transform, optimizer, device,
            args.grad_accum, ema, grad_clip_norm=args.grad_clip_norm
        )
        if n_ok_test == 0:
            print("DRY RUN FAIL: 0/10 samples produced a gradient. "
                  "Check load/loss failures above.")
            sys.exit(1)
        print(f"DRY RUN PASS: {n_ok_test}/{n_total_test} samples OK. "
              f"Ready for full training.")
        sys.exit(0)

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    no_improve_count = 0      # epochs since last val improvement (post-warmup)
    history = []
    sched_metric = float("nan")
    norm_history = []         # per-epoch norm_summary dicts — used for jump detection
    prev_val_tr_norm_max = None   # previous epoch's val tr_pred max norm (for NORM ALERT)

    # V6 multi-best tracking: separate best metrics per length bucket
    # best_long uses val_loss on long bucket; best_very_long on very_long;
    # best_combined uses sum(long + very_long) val loss.
    _v6_best_long = float("inf")
    _v6_best_very_long = float("inf")
    _v6_best_combined = float("inf")

    # V6 guard rail: detect short/medium regression
    # Track min val_loss per bucket (baseline = min seen so far)
    _v6_bucket_min = {"short": float("inf"), "medium": float("inf")}
    _v6_bucket_exceed_count = {"short": 0, "medium": 0}  # consecutive epochs above threshold
    _v6_bucket_losses: dict = {}   # per-bucket trimmed-mean val loss; back-filled into history[-1]
    start_epoch = 1           # overridden by --resume if a checkpoint is found

    # Oscillation monitoring: track recent trimmed val losses
    from collections import deque
    _osc_window = 10          # epochs in sliding window for oscillation amplitude
    _recent_val_losses = deque(maxlen=_osc_window)

    # Adaptive spike LR reduction (v3b/v4n/v5n mode):
    # When val tr_pred norm spikes >10× previous epoch AND >100, halve the LR
    # for the next 2 epochs by applying a 0.5× factor after scheduler.step().
    spike_cooldown_remaining = 0   # countdown: epochs left with 0.5× LR factor applied

    # EMA skip on spike (v5n only):
    # After a spike, pause EMA updates for 2 epochs so that the unstable
    # spike-epoch weights don't leak into the EMA checkpoint.
    # Implemented by passing ema=None to train_epoch during cooldown.
    ema_skip_remaining = 0   # countdown: epochs left where EMA updates are paused

    # ── Resume from checkpoint ──────────────────────────────────────────────
    if args.resume:
        _resume_path, _resume_epoch = find_resume_checkpoint(out_dir)
        if _resume_path is None:
            print(f"[resume] No epoch checkpoints found in {out_dir} — starting from scratch.")
        elif _resume_epoch >= args.n_epochs:
            print(f"[resume] Latest checkpoint epoch {_resume_epoch} >= n_epochs {args.n_epochs} — "
                  f"nothing to resume.")
            sys.exit(0)
        else:
            print(f"[resume] Loading checkpoint: {_resume_path}  (epoch {_resume_epoch})")
            _ckpt = torch.load(_resume_path, map_location=device)
            model.load_state_dict(_ckpt["model"])
            optimizer.load_state_dict(_ckpt["optimizer"])
            if "ema_weights" in _ckpt and ema is not None:
                ema.load_state_dict(_ckpt["ema_weights"], device=device)
            # Advance scheduler to the correct position WITHOUT re-applying warmup:
            #   - WarmupThenCosine: deterministic, fast-forward by ticking _epoch
            #   - WarmupThenPlateau: optimizer LR is already restored from state_dict;
            #     just advance the epoch counter so step() stays past warmup.
            scheduler._epoch = _resume_epoch
            # Restore loop state
            best_val_loss = _ckpt.get("val_loss", float("inf"))
            best_epoch    = _resume_epoch
            start_epoch   = _resume_epoch + 1
            # Load existing training history so the final CSV covers the full run
            _hist_path = out_dir / "training_history.csv"
            if _hist_path.exists():
                import csv as _csv_mod
                with open(_hist_path, newline="") as _fh:
                    _reader = _csv_mod.DictReader(_fh)
                    for _row in _reader:
                        # Cast numerics back from str
                        _typed = {}
                        for _k, _v in _row.items():
                            try:
                                _typed[_k] = int(_v) if "." not in _v else float(_v)
                            except (ValueError, TypeError):
                                _typed[_k] = _v
                        history.append(_typed)
            print(f"[resume] Resuming from epoch {start_epoch}  "
                  f"best_val_loss={best_val_loss:.4f}  "
                  f"(loaded {len(history)} prior history rows)")

    if args.v3_mode:
        phase_label = f"V3-Phase {args.unfreeze_phase}"
    elif args.v4_mode:
        phase_label = f"V4-Phase {args.unfreeze_phase}"
    elif args.v5_mode:
        phase_label = f"V5-Phase {args.unfreeze_phase}"
    elif args.v3b_mode:
        phase_label = f"V3B-Phase {args.unfreeze_phase}"
    elif args.v4n_mode:
        phase_label = f"V4N-Phase {args.unfreeze_phase}"
    elif args.v5n_mode:
        phase_label = f"V5N-Phase {args.unfreeze_phase}"
    elif args.v3c_mode:
        phase_label = f"V3C-Phase {args.unfreeze_phase}"
    elif args.v4c_mode:
        phase_label = f"V4C-Phase {args.unfreeze_phase}"
    elif args.v5c_mode:
        phase_label = f"V5C-Phase {args.unfreeze_phase}"
    elif args.v6_mode:
        phase_label = f"V6-Phase {args.unfreeze_phase}"
    else:
        phase_label = f"Phase {args.unfreeze_phase}"
    print(f"\nStarting {phase_label} fine-tuning: "
          f"{args.n_epochs} epochs, lr={args.lr}, grad_accum={args.grad_accum}, "
          f"warmup={args.warmup_epochs}, grad_clip={args.grad_clip_norm}, "
          f"layerwise={args.layerwise_lr_decay}, mode={mode_tag}")
    if start_epoch > 1:
        print(f"  [resume] Starting from epoch {start_epoch} (epochs 1–{start_epoch - 1} loaded from history)")
    print("=" * 70)

    for epoch in range(start_epoch, args.n_epochs + 1):
        t0 = time.time()

        train_indices = list(np.random.permutation(weighted_train_indices))
        # v5n + v3c: pause EMA updates during spike recovery so unstable weights
        # don't leak into the EMA checkpoint. Other modes update EMA normally.
        # IMPORTANT: decrement happens HERE (start of epoch) so that "remaining=N"
        # set at spike detection gives exactly N skipped epochs.
        # v5n: ema_skip_remaining=2 → skips E+1 and E+2
        # v3c/v4c/v5c: ema_skip_remaining=1 → skips E+1 only (less aggressive adaptation)
        _ema_skip_modes = (getattr(args, 'v5n_mode', False)
                           or getattr(args, 'v3c_mode', False)
                           or getattr(args, 'v4c_mode', False)
                           or getattr(args, 'v5c_mode', False))
        _ema_skip_this_epoch = _ema_skip_modes and ema_skip_remaining > 0
        if _ema_skip_this_epoch:
            ema_skip_remaining -= 1
            _remaining_msg = (f"{ema_skip_remaining} epoch(s) remaining after this"
                              if ema_skip_remaining > 0 else "final skip epoch")
            print(f"  [EMA SKIP ⏸] EMA updates paused this epoch ({_remaining_msg}).")
            if ema_skip_remaining == 0:
                print(f"  [EMA SKIP] EMA resumes next epoch.")
        _train_ema = None if _ema_skip_this_epoch else ema

        # V6 BUG FIX: re-apply frozen BN eval() after train_epoch will call model.train()
        # (model.train() resets ALL submodule modes; we must re-freeze each epoch)
        # This is done BEFORE train_epoch to pre-configure the model state; train_epoch
        # calls model.train() internally which we then counter at the start of the epoch.
        # Note: freeze_frozen_bn_stats() is a no-op for non-v6 modes (returns 0).
        if args.v6_mode:
            freeze_frozen_bn_stats(model)

        train_loss, n_ok, n_total, epoch_norms = train_epoch(
            model, train_ds, train_indices, transform, optimizer, device,
            args.grad_accum, _train_ema, grad_clip_norm=args.grad_clip_norm,
            pretrained_ref=pretrained_ref, reg_lambda=reg_lambda,
        )

        # V6: check frozen BN stats haven't drifted (runs after train_epoch)
        if args.v6_mode and _frozen_bn_snapshot:
            check_frozen_bn_drift(model, _frozen_bn_snapshot, epoch)

        # ── Score-norm early warning (train model) ──────────────────────────
        # A 3× jump in tr_pred norms is a strong signal that the optimizer is
        # escaping the score-calibration basin — validate this NOW rather than
        # waiting for val to hit 1e16 next epoch.
        if norm_history:
            prev_tr = norm_history[-1]["tr_mean"]
            cur_tr  = epoch_norms["tr_mean"]
            if prev_tr > 0 and cur_tr > prev_tr * 3.0:
                ratio = cur_tr / prev_tr
                print(f"  [NORM ALERT ⚠] tr_pred norm: "
                      f"{prev_tr:.3f} → {cur_tr:.3f} (×{ratio:.1f}). "
                      f"Score field escaping pretrained basin. "
                      f"Consider stopping/reducing LR.")
            if cur_tr > 200.0:
                print(f"  [NORM ALERT ⚠] tr_pred mean_norm={cur_tr:.1f} "
                      f"is critically large — val blowup is imminent.")
        norm_history.append(epoch_norms)

        # ── BAIL if first epoch is completely broken ────────────────────────
        if epoch == 1 and n_ok == 0 and args.bail_on_zero:
            print(
                "\n*** ABORTING: n_ok=0 in epoch 1. ***\n"
                "This means every sample either failed to load or failed the forward pass.\n"
                "Check load/loss failures printed above before re-running.\n"
                "Pass --no-bail-on-zero to override (not recommended)."
            )
            sys.exit(2)

        # ── Validation (EMA model) ──────────────────────────────────────────
        val_loss   = float("nan")
        val_stats  = {}   # type: dict
        if val_ds is not None:
            val_indices = list(range(len(val_ds)))
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            val_loss = val_epoch(model, val_ds, val_indices, transform, device,
                                 _stats_out=val_stats,
                                 prev_tr_norm_max=prev_val_tr_norm_max)
            ema.restore(model.parameters())
            _cur_val_tr_norm_max = val_stats.get("tr_norm_max", None)

            # ── Adaptive spike LR detection (v3b/v4n/v5n/v3c mode) ──────────
            # Fires when val tr_pred max_norm jumps >10× previous AND >100.
            # Uses prev_val_tr_norm_max (BEFORE this epoch's update) as baseline.
            #
            # v3b/v4n/v5n: 2-epoch cooldown — LR halved for 2 epochs then resumes
            # v3c/v4c/v5c: PERMANENT reduction via scheduler.permanent_reduce(0.5)
            #              LR never increases after a spike; EMA skipped for 1 epoch
            _spike_mode = (getattr(args, 'v3b_mode', False)
                           or getattr(args, 'v4n_mode', False)
                           or getattr(args, 'v5n_mode', False)
                           or getattr(args, 'v3c_mode', False)
                           or getattr(args, 'v4c_mode', False)
                           or getattr(args, 'v5c_mode', False))
            if (_spike_mode
                    and _cur_val_tr_norm_max is not None
                    and prev_val_tr_norm_max is not None
                    and _cur_val_tr_norm_max > prev_val_tr_norm_max * 10.0
                    and _cur_val_tr_norm_max > 100.0
                    and spike_cooldown_remaining == 0):
                _spike_ratio = _cur_val_tr_norm_max / max(prev_val_tr_norm_max, 1e-9)
                _permanent_reduce_mode = (getattr(args, 'v3c_mode', False)
                                          or getattr(args, 'v4c_mode', False)
                                          or getattr(args, 'v5c_mode', False))
                if _permanent_reduce_mode:
                    # V3C / V4C / V5C: PERMANENT LR reduction — scheduler absorbs it, no cooldown
                    _mode_spike_name = ("V3C" if args.v3c_mode else ("V4C" if args.v4c_mode else "V5C"))
                    _old_lr = scheduler.current_lr
                    scheduler.permanent_reduce(0.5)
                    _new_lr = scheduler.current_lr
                    print(f"  [SPIKE LR ⚡] tr_norm spike ×{_spike_ratio:.1f}  "
                          f"({prev_val_tr_norm_max:.1f} → {_cur_val_tr_norm_max:.1f}). "
                          f"PERMANENT LR reduction: {_old_lr:.2e} → {_new_lr:.2e}. "
                          f"LR will never increase again (monotone schedule).")
                    ema_skip_remaining = 1
                    print(f"  [EMA SKIP ⏸] EMA updates paused for 1 epoch ({_mode_spike_name} spike recovery).")
                else:
                    # v3b/v4n/v5n: 2-epoch temporary cooldown
                    spike_cooldown_remaining = 2
                    print(f"  [SPIKE LR ⚡] tr_norm spike ×{_spike_ratio:.1f}  "
                          f"({prev_val_tr_norm_max:.1f} → {_cur_val_tr_norm_max:.1f}). "
                          f"LR factor 0.5× will be applied for next "
                          f"{spike_cooldown_remaining} epochs.")
                    # v5n: additionally pause EMA updates during spike recovery
                    if getattr(args, 'v5n_mode', False):
                        ema_skip_remaining = 2
                        print(f"  [EMA SKIP ⏸] EMA updates will be paused for "
                              f"{ema_skip_remaining} epochs (spike recovery).")

            prev_val_tr_norm_max = _cur_val_tr_norm_max

        elapsed = time.time() - t0
        lr_now = scheduler.current_lr
        _v_med = val_stats.get("median", float("nan"))
        _v_std = val_stats.get("val_loss_std", float("nan"))
        print(
            f"Epoch {epoch:3d}/{args.n_epochs}  "
            f"train={train_loss:.4f} ({n_ok}/{n_total})  "
            f"val={val_loss:.4f}(med={_v_med:.4f} std={_v_std:.3f})  "
            f"lr={lr_now:.2e}  t={elapsed:.0f}s"
        )

        history.append({
            "epoch":             epoch,
            "train_loss":        train_loss,
            "val_loss":          val_loss,
            "val_raw_mean":      val_stats.get("raw_mean",        float("nan")),
            "val_median":        val_stats.get("median",          float("nan")),
            "val_loss_std":      val_stats.get("val_loss_std",    float("nan")),
            "val_max":           val_stats.get("max_loss",        float("nan")),
            "val_outliers":      val_stats.get("n_outlier",       0),
            "val_tr_norm_max":   val_stats.get("tr_norm_max",     0.0),
            "val_tr_norm_var":   val_stats.get("tr_norm_var",     0.0),
            "val_rot_norm_max":  val_stats.get("rot_norm_max",    0.0),
            "val_rot_norm_var":  val_stats.get("rot_norm_var",    0.0),
            "val_torbb_norm_max": val_stats.get("torbb_norm_max", 0.0),
            "val_torsc_norm_mean": val_stats.get("torsc_norm_mean", 0.0),
            "tr_norm_train":     epoch_norms["tr_mean"],
            "tr_norm_var":       epoch_norms.get("tr_var", 0.0),
            "rot_norm_train":    epoch_norms["rot_mean"],
            "rot_norm_var":      epoch_norms.get("rot_var", 0.0),
            "tor_bb_norm_train": epoch_norms.get("tor_bb_mean", 0.0),
            "tor_bb_norm_max":   epoch_norms.get("tor_bb_max", 0.0),
            "n_ok":              n_ok,
            "n_total":           n_total,
            "lr":                lr_now,
            # V6 per-bucket val losses — populated below by history[-1].update()
            "v6_val_short":      float("nan"),
            "v6_val_medium":     float("nan"),
            "v6_val_long":       float("nan"),
            "v6_val_very_long":  float("nan"),
        })

        sched_metric = val_loss if not np.isnan(val_loss) else train_loss
        scheduler.step(sched_metric)

        # ── Adaptive spike LR application (v3b/v4n/v5n mode only) ─────────────
        # AFTER scheduler.step() so the cosine trajectory is not disrupted.
        # The factor is re-applied each affected epoch (scheduler overwrites LRs).
        # NOTE: v3c/v4c/v5c are EXCLUDED here — their spike handling is permanent_reduce()
        # called at spike detection time; the exponential schedule owns the LR.
        _spike_active = (getattr(args, 'v3b_mode', False)
                         or getattr(args, 'v4n_mode', False)
                         or getattr(args, 'v5n_mode', False))
        if _spike_active and spike_cooldown_remaining > 0:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.5
            spike_cooldown_remaining -= 1
            if spike_cooldown_remaining == 0:
                print(f"  [SPIKE LR] Cooldown expired — normal LR resumes next epoch.")
            else:
                print(f"  [SPIKE LR] Cooldown: {spike_cooldown_remaining} epoch(s) remaining.")

        # ── EMA skip countdown (v5n + v3c) ────────────────────────────────────
        # Decrement is handled at the START of the epoch (above), not here.
        # This block is intentionally empty — kept as a marker to show where the
        # countdown used to live, so reviewers know the logic moved upward.

        # ── Oscillation amplitude monitoring ────────────────────────────────
        if not np.isnan(val_loss):
            _recent_val_losses.append(val_loss)
        if len(_recent_val_losses) >= 4:
            win = list(_recent_val_losses)
            osc_amp = max(win) - min(win)
            osc_std = float(np.std(win))
            if osc_amp > 5.0 or osc_std > 2.0:
                print(f"  [osc] val loss oscillation over last {len(win)} epochs: "
                      f"range={osc_amp:.2f}  std={osc_std:.2f}  "
                      f"(>5.0 range or >2.0 std may indicate LR too high)")

        # ── V6 stratified validation + guard rail ────────────────────────────
        _v6_bucket_losses = {}   # type: dict  # bucket → trimmed-mean loss
        if args.v6_mode and val_ds is not None and _val_bucket_indices:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            for bucket, bidxs in _val_bucket_indices.items():
                if bidxs:
                    bl = val_epoch(model, val_ds, bidxs, transform, device)
                    _v6_bucket_losses[bucket] = bl
            ema.restore(model.parameters())
            _bstr = "  ".join("%s=%.3f" % (b, _v6_bucket_losses.get(b, float("nan")))
                              for b in ["short", "medium", "long", "very_long"])
            print(f"  [V6 val-buckets] {_bstr}")

            # Guard rail: track short/medium val loss degradation
            _guard_thresh = getattr(args, 'v6_guard_threshold', 0.3)
            _guard_pat    = getattr(args, 'v6_guard_patience', 3)
            for bucket in ["short", "medium"]:
                bl = _v6_bucket_losses.get(bucket, float("nan"))
                if bl != bl:  # nan
                    continue
                if bl < _v6_bucket_min[bucket]:
                    _v6_bucket_min[bucket] = bl
                    _v6_bucket_exceed_count[bucket] = 0
                elif _v6_bucket_min[bucket] < float("inf"):
                    frac_increase = (bl - _v6_bucket_min[bucket]) / max(_v6_bucket_min[bucket], 1e-9)
                    if frac_increase > _guard_thresh:
                        _v6_bucket_exceed_count[bucket] += 1
                        if _v6_bucket_exceed_count[bucket] >= _guard_pat:
                            print(f"  [V6 GUARD RAIL ⚠] {bucket} val_loss={bl:.3f} "
                                  f"exceeds min={_v6_bucket_min[bucket]:.3f} by "
                                  f"{frac_increase*100:.0f}% for "
                                  f"{_v6_bucket_exceed_count[bucket]} consecutive epochs "
                                  f"(threshold {_guard_thresh*100:.0f}% × {_guard_pat} epochs). "
                                  f"CHECKPOINT UNSAFE — {bucket}-peptide geometry may be degrading. "
                                  f"Consider rolling back to epoch {best_epoch}.")
                    else:
                        _v6_bucket_exceed_count[bucket] = 0

            # Back-fill bucket losses into the history entry appended earlier this epoch
            history[-1].update({
                "v6_val_short":     _v6_bucket_losses.get("short",     float("nan")),
                "v6_val_medium":    _v6_bucket_losses.get("medium",    float("nan")),
                "v6_val_long":      _v6_bucket_losses.get("long",      float("nan")),
                "v6_val_very_long": _v6_bucket_losses.get("very_long", float("nan")),
            })

        # ── Save best + early stopping tracking ──────────────────────────────
        best_metric = val_loss if not np.isnan(val_loss) else train_loss
        post_warmup = (epoch > args.warmup_epochs)
        if best_metric < best_val_loss:
            best_val_loss = best_metric
            best_epoch = epoch
            no_improve_count = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                            str(out_dir / "rapidock_finetuned_best.pt"), ema, patterns)
            print(f"  ✓ New best (loss={best_val_loss:.4f})")
        elif post_warmup:
            no_improve_count += 1
            if no_improve_count > 0 and no_improve_count % 5 == 0:
                print(f"  [plateau] No improvement for {no_improve_count} epochs "
                      f"(best={best_val_loss:.4f} at epoch {best_epoch}, "
                      f"lr={lr_now:.2e})")

        # V6 multi-best checkpointing: best_long, best_very_long, best_combined
        if args.v6_mode and _v6_bucket_losses:
            _bl_long = _v6_bucket_losses.get("long", float("nan"))
            _bl_vl   = _v6_bucket_losses.get("very_long", float("nan"))
            if _bl_long == _bl_long and _bl_long < _v6_best_long:
                _v6_best_long = _bl_long
                save_checkpoint(model, optimizer, epoch, _bl_long,
                                str(out_dir / "rapidock_finetuned_best_long.pt"), ema, patterns)
                print(f"  ✓ [V6] New best_long (loss={_bl_long:.4f})")
            if _bl_vl == _bl_vl and _bl_vl < _v6_best_very_long:
                _v6_best_very_long = _bl_vl
                save_checkpoint(model, optimizer, epoch, _bl_vl,
                                str(out_dir / "rapidock_finetuned_best_very_long.pt"), ema, patterns)
                print(f"  ✓ [V6] New best_very_long (loss={_bl_vl:.4f})")
            if _bl_long == _bl_long and _bl_vl == _bl_vl:
                _combined = _bl_long + _bl_vl
                if _combined < _v6_best_combined:
                    _v6_best_combined = _combined
                    save_checkpoint(model, optimizer, epoch, _combined,
                                    str(out_dir / "rapidock_finetuned_best_combined.pt"),
                                    ema, patterns)
                    print(f"  ✓ [V6] New best_combined (long+vl={_combined:.4f})")

        # Early stopping — only after warmup, only if enabled
        if (args.early_stop_patience > 0 and post_warmup
                and no_improve_count >= args.early_stop_patience):
            print(f"\n*** EARLY STOP: no val improvement for {no_improve_count} epochs "
                  f"(patience={args.early_stop_patience}).  "
                  f"Best={best_val_loss:.4f} at epoch {best_epoch}. ***")
            break

        # ── Checkpoint saving ─────────────────────────────────────────────────
        # V6: save every epoch after epoch 15
        save_every_after = args.save_every_after if args.save_every_after is not None else 999999
        if args.v6_mode and save_every_after == 999999:
            save_every_after = 15  # V6 default: save every epoch from ep15 onward
        if epoch >= save_every_after:
            save_checkpoint(model, optimizer, epoch, sched_metric,
                            str(out_dir / f"rapidock_finetuned_epoch{epoch:03d}.pt"),
                            ema, patterns)
        elif epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, sched_metric,
                            str(out_dir / f"rapidock_finetuned_epoch{epoch:03d}.pt"),
                            ema, patterns)

    # ── Save final ──────────────────────────────────────────────────────────
    # Use last sched_metric; if never assigned (no epochs ran), use best
    final_loss = sched_metric if not np.isnan(sched_metric) else best_val_loss
    save_checkpoint(model, optimizer, args.n_epochs, final_loss,
                    str(out_dir / "rapidock_finetuned_final.pt"), ema, patterns)

    # ── Write training history ──────────────────────────────────────────────
    import csv as csv_mod
    hist_path = out_dir / "training_history.csv"
    _HIST_FIELDS = [
        "epoch", "train_loss",
        "val_loss",            # trimmed-mean (checkpoint metric)
        "val_raw_mean",        # raw mean (for diagnosing outlier inflation)
        "val_median",          # median (robust central tendency)
        "val_loss_std",        # std of per-sample losses (diversity/spread proxy)
        "val_max",             # worst single-sample loss (flags outlier presence)
        "val_outliers",        # count of samples with loss > 1000
        "val_tr_norm_max",     # max tr_pred norm in val (early instability signal)
        "val_tr_norm_var",     # variance of val tr_pred norms (score diversity)
        "val_rot_norm_max",    # max rot_pred norm in val
        "val_rot_norm_var",    # variance of val rot_pred norms
        "val_torbb_norm_max",  # max tor_bb norm in val
        "val_torsc_norm_mean", # mean tor_sc norm in val
        "tr_norm_train",       # mean tr_pred norm in training
        "tr_norm_var",         # variance of tr_pred norms (diversity proxy)
        "rot_norm_train",      # mean rot_pred norm in training
        "rot_norm_var",        # variance of rot_pred norms
        "tor_bb_norm_train",   # mean tor_bb norm in training
        "tor_bb_norm_max",     # max tor_bb norm in training
        "n_ok", "n_total", "lr",
        # V6 per-bucket val losses (NaN when not in v6 mode)
        "v6_val_short", "v6_val_medium", "v6_val_long", "v6_val_very_long",
    ]
    with open(hist_path, "w", newline="") as fh:
        w = csv_mod.DictWriter(fh, fieldnames=_HIST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(history)

    final_phase_label = phase_label  # reuse from training loop setup
    print(f"\n{final_phase_label} training complete.")
    print(f"Best checkpoint → {out_dir / 'rapidock_finetuned_best.pt'}  "
          f"(loss={best_val_loss:.4f})")
    print(f"History → {hist_path}")


if __name__ == "__main__":
    main()
