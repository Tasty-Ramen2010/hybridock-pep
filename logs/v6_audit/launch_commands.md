# V6 Training Launch Commands
*Generated 2026-05-30*

---

## Prerequisites

```bash
# Verify all data files exist
ls -lh data/v6_train_combined.csv   # 1,200 rows (1000 gap-fill + 200 replay)
ls -lh data/v6_val_200.csv          # 200 rows (50 per length bucket)
ls -lh data/v6_replay_200.csv       # 200 rows (reference, already merged into combined)

# Verify starting checkpoint
ls -lh third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_global.pt

# Check GPU availability
nvidia-smi

# Activate the rapidock environment
conda activate rapidock
```

---

## Working Directory

All commands run from the **repo root**:
```bash
cd /home/igem/unknown_software
```

---

## Phase 1: Torsion + Score Heads (Epochs 1–8)

**Trainable:** `tor_bb_bond_conv`, `tr/rot/tor_bb/tor_sc_final_layer` (980,498 params, 12.97%)  
**Frozen:** everything else including `cross_convs`  
**Duration:** ~2–3 hours on RTX 5070

```bash
conda run -n rapidock python third_party/RAPiDock_finetuned/train_lastlayer.py \
  --v6-mode \
  --train-csv data/v6_train_combined.csv \
  --v6-val-csv data/v6_val_200.csv \
  --checkpoint third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_global.pt \
  --output-dir logs/v6_run/phase1 \
  --unfreeze-phase 1 \
  --n-epochs 8 \
  --lr 3e-6 \
  --lr-schedule cosine \
  --cosine-min-lr 5e-7 \
  --warmup-epochs 2 \
  --grad-clip-norm 0.5 \
  --grad-accum 4 \
  --save-every 2 \
  --save-every-after 5 \
  --esm-device cpu \
  --v6-guard-patience 3 \
  --v6-guard-threshold 0.3 \
  2>&1 | tee logs/v6_run/phase1.log
```

**Notes:**
- `--lr-schedule cosine` is **required** for V6; plateau mode is rejected.
- No `--pretrained-reg-lambda` in Phase 1 — cross_convs are frozen, reg is irrelevant.
- V6 auto-sets λ=3e-4 only when cross_convs are unfrozen (Phase 2+).
- `--save-every-after 5` checkpoints every epoch from ep5 onward.
- ESM cache is built on first run (~40 min CPU). Subsequent runs use the cache.

**Expected output:**
```
[V6] Loading from pretrained checkpoint: ...rapidock_global.pt
[V6] Val bucket sizes: short=50  medium=50  long=50  very_long=50
Epoch   1/8  train=X.XXXX  val=X.XXXX  lr=3.00e-06  t=XXXs
  [V6 val-buckets] short=X.XXX  medium=X.XXX  long=X.XXX  very_long=X.XXX
```

**After Phase 1:** Record the `val_loss` for each bucket at epoch 8.
Use `rapidock_finetuned_final.pt` as the Phase 2 starting checkpoint.

---

## Phase 2: Cross-Conv Specialization (Epochs 9–35)

**Trainable:** P1 set + `cross_convs.0–3` (3,649,634 params, 48.29%)  
**Duration:** ~15–20 hours on RTX 5070 (27 epochs × 2,242 effective samples)

```bash
conda run -n rapidock python third_party/RAPiDock_finetuned/train_lastlayer.py \
  --v6-mode \
  --train-csv data/v6_train_combined.csv \
  --v6-val-csv data/v6_val_200.csv \
  --checkpoint logs/v6_run/phase1/rapidock_finetuned_final.pt \
  --output-dir logs/v6_run/phase2 \
  --unfreeze-phase 2 \
  --n-epochs 27 \
  --lr 5e-6 \
  --lr-schedule cosine \
  --cosine-min-lr 5e-7 \
  --warmup-epochs 3 \
  --grad-clip-norm 1.0 \
  --grad-accum 4 \
  --save-every 5 \
  --save-every-after 15 \
  --pretrained-reg-lambda 3e-4 \
  --esm-device cpu \
  --v6-guard-patience 3 \
  --v6-guard-threshold 0.3 \
  2>&1 | tee logs/v6_run/phase2.log
```

**Notes:**
- `--checkpoint` points to Phase 1's `final.pt` (NOT `rapidock_global.pt`).
- `--n-epochs 27` runs epochs 1–27 internally (equivalent to global epochs 9–35).
  The history CSV will show epoch=1–27; subtract to get global epoch numbers.
- `--pretrained-reg-lambda 3e-4` applies L2 reg toward `rapidock_global.pt` weights
  for `cross_convs.0–3`. V6 auto-loads the pretrained ref from the initial checkpoint.
- Tier oversampling is active because `v6_train_combined.csv` has a `tier` column.
- Expected effective epoch size log: `[V6] Tier-weighted epoch size: 2242`

**After Phase 2:** Check `rapidock_finetuned_best_very_long.pt` and `best_combined.pt`.
Run bench inference on bench_very_long.csv to assess improvement before launching Phase 3.

---

## Phase 3: Fine-Polish + Stability (Epochs 36–45)

**Trainable:** Same as Phase 2 (48.29%)  
**Duration:** ~3–4 hours (10 epochs × 1,200 uniform samples)

```bash
conda run -n rapidock python third_party/RAPiDock_finetuned/train_lastlayer.py \
  --v6-mode \
  --train-csv data/v6_train_combined.csv \
  --v6-val-csv data/v6_val_200.csv \
  --checkpoint logs/v6_run/phase2/rapidock_finetuned_best_combined.pt \
  --output-dir logs/v6_run/phase3 \
  --unfreeze-phase 3 \
  --n-epochs 10 \
  --lr 5e-7 \
  --lr-schedule cosine \
  --cosine-min-lr 1e-7 \
  --warmup-epochs 0 \
  --grad-clip-norm 1.0 \
  --grad-accum 4 \
  --save-every 2 \
  --save-every-after 1 \
  --pretrained-reg-lambda 3e-4 \
  --esm-device cpu \
  --v6-guard-patience 3 \
  --v6-guard-threshold 0.3 \
  2>&1 | tee logs/v6_run/phase3.log
```

**Notes:**
- `--checkpoint` uses `best_combined.pt` from Phase 2 (lowest long+very_long sum loss).
  Alternative: use `best.pt` if you want the globally best overall checkpoint.
- `--n-epochs 10` runs epochs 36–45 (internal epoch numbering 1–10).
- `--save-every-after 1` checkpoints every epoch (only 10 epochs total).
- No oversampling in Phase 3 — uniform 1× (effective epoch = 1,200).
- After Phase 3, the final model is `rapidock_finetuned_final.pt`.

---

## Dry-Run Test (Before Full Launch)

Verify that the training CSV loads and processes correctly:

```bash
conda run -n rapidock python third_party/RAPiDock_finetuned/train_lastlayer.py \
  --v6-mode \
  --train-csv data/v6_train_combined.csv \
  --v6-val-csv data/v6_val_200.csv \
  --checkpoint third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_global.pt \
  --output-dir logs/v6_run/dryrun \
  --unfreeze-phase 1 \
  --n-epochs 1 \
  --lr 3e-6 \
  --lr-schedule cosine \
  --cosine-min-lr 5e-7 \
  --warmup-epochs 0 \
  --dry-run \
  --esm-device cpu \
  2>&1 | tee logs/v6_run/dryrun.log
```

**What to check in dry-run output:**
1. `[V6] Loading from pretrained checkpoint:` — confirms correct checkpoint
2. `[V6] Val bucket sizes: short=50  medium=50  long=50  very_long=50` — confirms val CSV
3. `[V6] Tier-weighted epoch size: 1200` — confirms tier column found
4. `n_ok=10 / n_total=10` — confirms forward pass works
5. No CUDA OOM or NaN warnings

---

## Output Artifacts

After full training (all 3 phases), the key files are:

| File | Description |
|------|-------------|
| `logs/v6_run/phase3/rapidock_finetuned_final.pt` | Final model (last epoch) |
| `logs/v6_run/phase2/rapidock_finetuned_best_very_long.pt` | Best checkpoint for very_long bucket |
| `logs/v6_run/phase2/rapidock_finetuned_best_combined.pt` | Best checkpoint for long + very_long |
| `logs/v6_run/phase2/rapidock_finetuned_best.pt` | Best overall checkpoint |
| `logs/v6_run/phase1/training_history.csv` | Per-epoch history (P1) |
| `logs/v6_run/phase2/training_history.csv` | Per-epoch history (P2), includes v6_val_* columns |
| `logs/v6_run/phase3/training_history.csv` | Per-epoch history (P3) |

---

## tmux Session

For long-running training (Phase 2 is ~18 hours), use tmux:

```bash
tmux new-session -s v6_training
# Inside tmux:
conda activate rapidock
cd /home/igem/unknown_software
# Paste Phase 1 command above
# After Phase 1 completes, paste Phase 2 command
# Press Ctrl+B, D to detach
```

To monitor in a separate pane:
```bash
tmux attach -t v6_training
# Ctrl+B, % to split vertically, then:
tail -f logs/v6_run/phase2.log | grep -E "Epoch|V6 val|GUARD|best"
```

---

## Benchmark Inference Commands (Run After Each Phase)

Run every 5 epochs to track RMSD improvement:

```bash
# Substitute CHECKPOINT_PATH with the checkpoint to evaluate
python scripts/run_bench.py \
  --model-checkpoint CHECKPOINT_PATH \
  --bench-csv data/bench_very_long.csv \
  --n-poses 5 \
  --output-dir logs/v6_bench_ep{N}
```

*(Assumes `scripts/run_bench.py` exists — use the equivalent inference script.)*
