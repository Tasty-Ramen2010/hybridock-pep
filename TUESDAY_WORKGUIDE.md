# HybriDock-Pep — Work Guide

**Last updated:** 2026-05-28 (auto-updated by Claude each session)  
**Project:** HybriDock-Pep · iGEM 2026 Best Software Tool  
**Maintainer:** Ram, Denmark HS iGEM Dry Lab

---

## 🟢 CURRENT STATUS — 2026-05-28 13:00 EDT

### Active jobs
| Job | Phase | Status | PID |
|-----|-------|--------|-----|
| `chain_training_phases.sh` | Phase 2/3 | **RUNNING** | 1597156 |
| `train_lastlayer.py` (P2) | epoch ~10/50 | Running, lr=2e-5 | 1597187 |
| `score_calibration_set.py` | Calibration (stale) | Orphaned from May 26 | 398431 |

### Training snapshot (Phase 2, as of epoch 10)
```
Phase 1  COMPLETE  30 epochs · best val=74.4 @ epoch 4  · best ckpt: finetune_peppc_phase1/rapidock_finetuned_best.pt
Phase 2  RUNNING   epoch ~10/50 · best val=75.5 @ epoch 4  · best ckpt: finetune_peppc_phase2/rapidock_finetuned_best.pt
Phase 3  PENDING   100 epochs · cosine LR · early-stop-patience=20
```

### Quick check commands
```bash
# Latest epoch
tail -5 logs/chain_training.log

# Full Phase 1 history
cat third_party/RAPiDock_finetuned/finetune_peppc_phase1/training_history.csv

# GPU utilisation
nvidia-smi

# Is training still alive?
ps aux | grep train_lastlayer | grep -v grep
```

---

## 🐛 BUGS FIXED (chronological)

### May 26 — Training no-op bug (CRITICAL)
**Symptom:** 50 epochs, all losses 0.0000, weights never changed, 0% GPU.  
**Root cause:** `build_dataset()` created `InferenceDataset` without `conformation_type`.
Defaults to `None`, causing `{'H':...,'E':...,'P':...}[None]` → KeyError on every sample.
`train_epoch()` catches silently → n_ok=0 every epoch.  
**Fix:** Added `conformation_type='H'` in build_dataset() call.  
**Files:** `train_lastlayer.py`

### May 26 — val_epoch silently returning 0.0
**Symptom:** val_loss=0.0000 in every checkpoint.  
**Root cause:** `compute_loss()` raised exceptions in eval mode, all caught, n_ok=0, 0/0=0.  
**Fix:** Added exception logging in val_epoch; loss now reports properly.  
**Files:** `train_lastlayer.py`

### May 26 — DataLoader dict-unpacking bug
**Symptom:** n_ok=0 in every epoch after the conformation fix.  
**Root cause:** `compute_loss()` was unpacking DataBatch as a dict (`data['pep']`) but
PyG returns attribute access (`data.pep`). Raised AttributeError every sample.  
**Fix:** Rewrote `compute_loss()` to use correct PyG attribute access.  
**Files:** `train_lastlayer.py`

### May 26 — ESM TDR crash on WSL2 (cuda, batch 790/874)
**Symptom:** CUDA kernel timeout (TDR) at ESM batch 790, killing the Python process.  
**Root cause:** Long receptor sequences (> 1022 AA) produce attention kernels > 2s, above WSL2's TDR limit.  
**Fix v1:** Changed `--esm-device cpu` (safe, slow ~40 min).  
**Fix v2:** Added sequence clipping at 1022 AA before ESM batching; restored `--esm-device cuda`
(pocket PDBs already trimmed < 300 AA; ~15 min GPU vs 31 h CPU).  
**Files:** `train_lastlayer.py`, `chain_training_phases.sh`

### May 26 — VRAM fragmentation (stall after epoch 1)
**Symptom:** After ~800 ESM batches, VRAM fills to 99.99%, training stalls.  
**Root cause:** PyTorch default CUDA allocator uses fixed-size blocks that fragment.  
**Fix:** `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in chain script.  
**Files:** `scripts/chain_training_phases.sh`

### May 27 — RAPiDock-Reloaded valence crash
**Symptom:** All peptides crash at Stage 1 with "Valence error: 5 bonds on N".  
**Root cause:** RAPiDock-Reloaded changed atom-type mapping; nitrogen with 5 connections
in some PDB files (e.g. nitro groups) broke RDKit valence check.  
**Fix:** Added name-based atom remapping + PeptideBuilder fallback.  
**Files:** `third_party/RAPiDock_finetuned/dataset/peptide_feature.py`

### May 27 — Unfrozen parameter count wrong (12k instead of 2.1M)
**Symptom:** Phase 1 unfroze only 12k/7.5M params (0.17%), almost nothing trained.  
**Root cause:** Pattern matching code skipped the embedding layers that should be unfrozen.  
**Fix:** Rewrote unfreeze logic; Phase 1 now unfreezes 2,106,182 params (27.88%).  
**Files:** `train_lastlayer.py`

### May 28 — e3nn FullyConnectedTensorProduct int32 bug (CRITICAL)
**Symptom:** 656/2122 loss failures per epoch (31%), `RuntimeError: mean() Got: Int`.  
**Root cause:** `e3nn.FullyConnectedTensorProduct` returns `torch.int32` (not float32) when
given empty input tensors (0 edges). This happens when diffusion noise moves the peptide
far enough from the receptor that zero cross-contact edges exist within the cutoff.
`torch_scatter.scatter` propagates the int32 dtype → e3nn BatchNorm `field.mean()` fails.  
**Fix:** Added `tp = tp.float()` immediately after `self.tensor_prod(...)` in CGTPEL.forward().
`.float()` is a no-op for float32 and costs nothing in the normal case.  
**Files:** `third_party/RAPiDock_finetuned/models/diffusion.py` (line 190)

### May 28 — Phase 1 val loss explosions (1.23 × 10¹⁶)
**Symptom:** Val loss periodic spikes to 10¹⁶ while train loss stays ~48-56.
Best checkpoint was epoch 4; remaining 26 Phase 1 epochs were wasted.  
**Root cause:** 1-2 outlier val samples produce astronomical MSE (score blowup on rare
geometries). Mean of 200 samples is dominated by these outliers.  
**Fix:** Changed `val_epoch()` to use a **trimmed mean** (drop top 5% of per-sample losses).
Added diagnostic logging of raw_mean / median / outlier count when spikes detected.
Added `--early-stop-patience N` arg (Phase 3 uses 20) to stop when no improvement.  
**Files:** `train_lastlayer.py`

### May 28 — 12 load failures per epoch (peptide_feature.py min() crash)
**Symptom:** 12/2122 complexes fail to load every epoch with `ValueError: min() arg is an empty sequence`.  
**Root cause:** `peptide_feature.py` line 570 calls `min([i for i in trans.keys() if isinstance(i,int)])`.
Some peptide PDBs have all-insertion-code residue numbering (e.g. "1A", "1B"…), so
the int-key list is empty.  
**Fix:** Added guard: extract `_int_keys` first; if empty, use `sorted(trans.keys())` directly.  
**Files:** `third_party/RAPiDock_finetuned/dataset/peptide_feature.py` (line 570)  
**Status:** Applied May 28, takes effect from Phase 3 onwards (Phase 2 already running).

---

## 📋 OPEN TO-DO LIST

### 🔴 High priority
- [ ] **Confirm Phase 2 best val improves past 75.5** — currently oscillating 75-160;
      if still at 75.5 by epoch 25, the cross_convs.2/3 changes may not be helping.
      Consider: reduce Phase 2 LR to 5e-6 in a re-run if Phase 3 results disappoint.
- [ ] **Verify Phase 3 starts cleanly** after Phase 2 completes (check for checkpoint assertion).
- [ ] **Post-training benchmark**: run `hybridock-pep benchmark` comparing finetuned vs.
      pretrained RAPiDock on 10 test complexes. Target: ≥0.1 improvement in Pearson r.
- [ ] **Investigate 12 AssertionError load failures** — some of the 12 failing complexes
      hit `assert len(embedding_idx) == len(lm_embedding_chain)`. The min() fix may
      resolve some but not all. May need to add these to a skip list.

### 🟡 Medium priority
- [ ] **Phase 2 restart option**: if Phase 3 val loss > Phase 1 best after 20 epochs,
      restart with a fresh Phase 2 at lower LR (5e-6) and skip Phase 2 cross_convs changes.
- [ ] **Add gradient norm logging** to train_epoch for debugging exploding gradients.
- [ ] **Calibration update**: once Phase 3 checkpoint is available, re-run calibration
      with the finetuned model to check if α changes.
- [ ] **PepSet-6 re-benchmark**: run the 6 gold-standard complexes (r=0.860) with the
      finetuned checkpoint to confirm no regression.
- [ ] **Fix stale `score_calibration_set.py` process** (PIDs 398431-398434, running since May 26).
      Kill with `kill 398431` if not needed.

### 🟢 Low priority
- [ ] **pytest after training changes**: run `pytest tests/` to ensure training script
      changes haven't broken the unit tests (val_epoch signature change, early stop args).
- [ ] **torch.cross deprecation warning**: line 668 in peptide_feature.py uses
      `torch.cross` without `dim` arg. Replace with `torch.linalg.cross(..., dim=-1)`.
- [ ] **Update docs/ai_training_guide_peppc.md** with final training results.
- [ ] **iGEM wiki page draft** — due before November Jamboree.

---

## 📊 TRAINING ARCHITECTURE REFERENCE

### 3-Phase schedule
| Phase | Unfrozen params | Epochs | LR | Schedule | Warmup |
|-------|-----------------|--------|----|----------|--------|
| 1 | 2.1M / 7.5M (27.9%) | 30 | 1e-4 | plateau (p=8) | 0 |
| 2 | +3.0M cross/intra convs | 50 | 2e-5 | plateau (p=8) | 5 |
| 3 | all 7.5M | 100 | 2e-5→1e-7 | cosine | 10 |

### What's unfrozen in each phase
```
Phase 1:  tr_final_layer, rot_final_layer, tor_bb_final_layer, tor_sc_final_layer,
          final_conv, tor_bb_bond_conv, tor_sc_bond_conv, center_edge_embedding,
          pep_a_node_embedding (25 tensors), final_edge_embedding

Phase 2:  Phase 1 + cross_convs.3, cross_convs.2, intra_convs.3
          (NOT intra_convs.2 — held for Phase 3 full-model context)

Phase 3:  All layers including intra_convs.0/1, cross_convs.0/1
```

### Checkpoints
```
finetune_peppc_phase1/rapidock_finetuned_best.pt    ← best P1 (val=74.4 epoch4)
finetune_peppc_phase2/rapidock_finetuned_best.pt    ← best P2 (val=75.5 epoch4, updating)
finetune_peppc_phase3/rapidock_finetuned_best.pt    ← best P3 (pending)
finetune_peppc_phase3/rapidock_finetuned_final.pt   ← P3 final epoch (pending)
```

### Key hyperparameter decisions (reasoning in training_strategy_analysis.md)
- `grad_accum=4` — effective batch=4; single-sample DataLoader required by PyG heterogeneous
- `weight_decay=1e-5` for Phase 2/3, 0 for Phase 1 (output heads only, no regularisation needed)
- `ema_decay=0.999` — slow EMA prevents val oscillation from raw weights
- `--early-stop-patience 20` for Phase 3 — cosine has no internal decay stop, needs external guard
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — prevents VRAM fragmentation on RTX 5070

---

## 🔧 QUICK REFERENCE COMMANDS

### Monitor training
```bash
# Live log tail
tail -f logs/chain_training.log

# Current epoch progress (last 5 epoch lines)
grep "^Epoch" logs/chain_training.log | tail -5

# GPU state
nvidia-smi

# Training process alive?
ps -o pid,etimes,pcpu,pmem -p $(pgrep -f train_lastlayer) 2>/dev/null
```

### After training completes
```bash
# Check all phase checkpoints exist
ls -lh third_party/RAPiDock_finetuned/finetune_peppc_phase{1,2,3}/*.pt 2>/dev/null

# Run benchmark with finetuned model
conda run -n score-env hybridock-pep benchmark \
    --test-csv data/test_complexes.csv \
    --baselines vina,adcp,rapidock \
    --report docs/benchmarks/finetune_benchmark_$(date +%Y%m%d).md

# Quick scoring comparison: finetune vs pretrained on PepSet-6
conda run -n score-env python scripts/calibrate_alpha.py \
    --training-csv data/training_complexes.csv \
    --model third_party/RAPiDock_finetuned/finetune_peppc_phase3/rapidock_finetuned_best.pt \
    --output data/calibration_finetuned.json
```

### Kill stale jobs
```bash
# Kill orphaned calibration workers from May 26
kill 398431 398433 398434 2>/dev/null; echo "done"

# Kill training (use only in emergency — prefer letting chain complete)
kill 1597156  # chain bash wrapper; kills Python child too
```

### Emergency restart (if chain died)
```bash
cd /home/igem/unknown_software
conda run -n rapidock bash scripts/chain_training_phases.sh \
    2>&1 | tee -a logs/chain_training.log &
echo "Restarted, PID $!"
```

---

## 📁 KEY FILE LOCATIONS

| File | Purpose |
|------|---------|
| `scripts/chain_training_phases.sh` | Master 3-phase training launcher |
| `third_party/RAPiDock_finetuned/train_lastlayer.py` | Training script (phases 1-3) |
| `third_party/RAPiDock_finetuned/models/diffusion.py` | CGTPEL + CGTensorProduct model |
| `third_party/RAPiDock_finetuned/dataset/peptide_feature.py` | Data loading (fixed May 28) |
| `logs/chain_training.log` | Live training log |
| `data/calibration.json` | Production calibration (PepSet-6, r=0.860) |
| `datasets/training_formatted_peppc/combined_train_curated.csv` | 2000-complex training set |
| `datasets/training_formatted_peppc/combined_val_curated.csv` | 200-complex val set |
| `docs/training_strategy_analysis.md` | Phase design rationale |
| `docs/benchmarks/` | Comparison reports vs DiffPepDock |
| `CLAUDE.md` | Project instructions for Claude Code |
| `TUESDAY_WORKGUIDE.md` | This file |

---

*Auto-updated by Claude Code · HybriDock-Pep v0.1 · iGEM 2026*
