# V6 Concerns & Implementation Issues
*Generated 2026-05-30 pre-v6 launch*

---

## CRITICAL (must verify before first epoch)

### C1 — Starting checkpoint path must be set manually
The training command uses `--checkpoint-dir` to point at the directory containing
`rapidock_global.pt`. The V6 code explicitly rejects any checkpoint that is not
`rapidock_global.pt` (filename check in `load_model_for_finetuning_v6()`).

**Verify:** Run with `--dry-run` (or stop after first print) and confirm the log says
`[V6] Loading from pretrained checkpoint: .../rapidock_global.pt` — NOT a v5c or other
fine-tuned checkpoint.

**Action item:** Confirm the correct checkpoint path before launching.

---

### C2 — Training CSV source column collides with tier names
The gap-fill CSV uses the `source` column to store tier names
(e.g., `"T1_sheet_very_long"`, `"T2_sheet_long"`). The combined training CSV
(gap-fill + replay) will have mixed `source` values: tier names from gap-fill
and `"peppcf"` from replay entries.

The tier-weight lookup uses the `source` column: any entry whose `source` is not
in `_V6_TIER_WEIGHTS` will silently get weight=1 (default). This is safe but means
peppcf replay entries get correct 1× weight while gap-fill entries with unrecognized
tier names would also get 1×.

**Check:** When building the combined CSV before launch, verify `source` values from
gap-fill map to expected tier names in `_V6_TIER_WEIGHTS`. If they differ (e.g., lowercase
vs. mixed-case), the oversampling will fail silently with all weights=1.

**Verify:** After first epoch, check the log for the effective epoch size — it should
show 2,242 for Phase 2, not 1,200.

---

### C3 — val_epoch called 4× per epoch (bucket-wise) — VRAM and time impact
Each V6 epoch runs `val_epoch()` once for the overall val set PLUS once per non-empty
bucket (up to 4×). This is 5 val passes per epoch.

At 200 val complexes (50/bucket), this adds roughly 5 × ~50s overhead per epoch
depending on GPU speed. At N=45 epochs, this is ~3.75 additional GPU-hours.

**Mitigation:** If training is slow, disable bucket val with `--v6-val-csv ""` (disables
bucket tracking) and do only the overall val pass. The `v6_val_*` columns will be NaN
in the history CSV, but training will proceed.

---

## HIGH (monitor closely but not blocking)

### H1 — T1_sheet_very_long is a hard data ceiling
Only 106 T1 complexes exist in PepPC after all exclusions. Even at 3× oversampling,
T1 contributes 318/2,242 = 14.2% of Phase 2 samples. This is the dominant bottleneck
for SHEET very_long improvement.

**Risk:** If v6 still fails on SHEET very_long after full training, the limiting factor
is data volume, not model capacity or training schedule. The next step would be sourcing
fresh SHEET very_long data from RCSB PDB directly.

---

### H2 — No short (pep_len=8) complexes in validation from gap-fill training
All 7,897 available short (pep_len=8) PepPC complexes not in the exclusion list were
available for val set construction. 50 were sampled. The guard rail will monitor these.

However, Phase 1 trains only on 100 short-bucket gap-fill entries (T7_sheet_short).
These are all pep_len=8, SHEET. If the training distribution generalizes to the val set
(which includes mixed short complexes), the guard rail may fire prematurely.

**Expected behavior:** Short val_loss may increase in Phase 2 when cross_convs are
unfrozen and oversampling shifts focus to long/very_long. The guard rail fires at 30%
above minimum for 3 consecutive epochs. If short val_loss spikes in Phase 2 but then
stabilizes, this is a false-positive guard rail trigger — not actual catastrophic forgetting.

**Recommended action:** If guard rail fires for "short" in Phase 2, compare `best.pt`
vs `best_combined.pt` on bench300 before rolling back.

---

### H3 — val_epoch bucket indices assume stable dataset ordering
`_val_bucket_indices` are pre-computed as integer indices into the val dataset
based on the CSV row order. `build_dataset()` must preserve CSV row order for these
indices to be valid. If `build_dataset()` sorts, deduplicates, or drops rows,
the bucket → index mapping will be wrong.

**Verify:** After `build_dataset()` on the val CSV, check `len(val_ds)` == 200.
If it's less, some val complexes were dropped (e.g., due to missing PDB files or
processing errors), and the bucket indices will be misaligned.

**Mitigation:** The trimmed-mean in `val_epoch()` handles partial index lists gracefully
(n_fail counter). But wrong indices pointing to wrong-bucket complexes would silently
corrupt the bucket val signals. Log the bucket sizes at startup (already printed).

---

### H4 — L2 reg λ=3e-4 is tuned on v5c dynamics, not V6
The λ value was chosen by analogy with v4c/v5c behavior. V6 trains a larger parameter
set (cross_convs + tor_bb_bond_conv) from a clean pretrained starting point. The correct
λ may differ.

**Signals to watch:**
- If `v6_val_long` and `v6_val_very_long` both plateau above 4.0 in Phase 2 and stop
  improving → λ may be too high (over-constraining the cross_convs).
- If training is unstable (val_loss spikes repeatedly) → λ may be too low.
- If overall val_loss reaches Phase 1 values in Phase 2 within 5 epochs → λ is correct.

**Adjust:** Use `--reg-lambda 0.0001` for half-strength or `--reg-lambda 0.001` for 3×.

---

### H5 — Phase transitions require restarting the training script
V6 does NOT auto-transition between phases. Each phase is a separate training run with:
- Different `--n-phases` / `--current-phase` argument
- Different `--start-checkpoint` (previous phase's final checkpoint)
- Different `--n-epochs`

If you use a single long run (all 45 epochs), the phase transition (P1→P2→P3) is handled
internally by the phase logic when `--n-phases 3` is set. Verify that the phase change
logs appear: `"Transitioning to Phase 2"` at epoch 9, `"Transitioning to Phase 3"` at epoch 36.

---

## LOW (informational)

### L1 — Replay set source annotation
All 200 replay entries have `source="peppcf"` in `v6_replay_200.csv`. This gives them
tier weight 1× (replay) in the oversampling scheme. The original source diversity
(peppcf/peppc/refpepdb/ppii_enriched) is discarded. This is conservative and correct
for the intended purpose (forgetting prevention, not source-specific training).

### L2 — 390 null-source gap-fill entries
390 gap-fill entries lack source annotation and were assigned weight based on tier
(T1-T8 tier assignment). These will not match any `source`-based weighting from prior
code paths. In V6, weighting is tier-based (from `source` column containing tier names),
so these entries correctly receive their tier weight.

### L3 — EMA decay and val evaluation
EMA uses `decay=0.999` (default). At epoch 1, EMA is approximately equal to the pretrained
weights. The guard rail uses EMA-evaluated val_loss, which will appear better than the raw
model early in training. The guard rail only fires post-warmup (epoch > warmup_epochs=2),
so this is unlikely to cause false positives.

### L4 — save_every_after=15 (V6 default)
Every epoch ≥15 is checkpointed. At 45 epochs, this means 31 checkpoint files ×
~29 MB each ≈ ~900 MB checkpoint storage. Ensure `--output-dir` has sufficient space.

### L5 — RMSD validation is NOT automated
Per-epoch val_loss tracks score-matching quality (a proxy). Actual RMSD improvement
on bench_very_long.csv requires manual inference runs. The audit recommends running
these at epochs 10, 15, 20, 25, 30, 35, 45. Set a reminder.

---

## Implementation Issues Found During Audit

### I1 — history.append / _v6_bucket_losses ordering bug (FIXED)
The initial implementation added `_v6_bucket_losses.get(...)` to `history.append({...})`
before the V6 val block computed those losses. Fixed by:
1. Using placeholder `float("nan")` in `history.append()`
2. Back-filling with `history[-1].update({...})` after the V6 val block

### I2 — Wrong starting checkpoint contaminated v3c/v4c/v5c (FIXED in V6)
All three prior runs had measurable drift in modules that should have been at pretrained
values (see `phase0_freeze_audit.md §4`). V6 uses `load_model_for_finetuning_v6()` which:
- Requires the checkpoint path to match `rapidock_global.pt` (filename check)
- Adds a drift validation step against pretrained weights at startup

### I3 — BatchNorm running stats not frozen (FIXED in V6)
All prior runs silently updated `running_mean` / `running_var` in "frozen" conv blocks.
V6 adds `freeze_frozen_bn_stats()` which sets frozen conv BN layers to `.eval()` mode.
Called once after model load and re-applied at the start of every training epoch
(necessary because `model.train()` resets all submodule modes).
