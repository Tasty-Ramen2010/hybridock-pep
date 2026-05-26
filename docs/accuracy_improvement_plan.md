# Accuracy Improvement Plan — RTX 5070 Training Roadmap

**Status:** Draft v1
**Owner:** Ram (Head of Dry Lab) · iGEM 2026 Denmark HS
**Hardware target:** Linux machine with RTX 5070 (Blackwell sm_120, 12 GB VRAM, CUDA 12.8)
**Last updated:** 2026-05-22
**Companion docs:** `docs/data_augmentation_plan.md` · `docs/calibration_notes.md` · `docs/dataset_analysis.md` · `docs/architecture.md` · `docs/RUNBOOK.md`

---

## 0. How to read this document

This file is **prescriptive, not exploratory**. It captures, in one place, every
training-and-accuracy step you should perform when you return to the Linux
machine on Tuesday 2026-05-26 — together with the safeguards required to keep
the pipeline trustworthy as iGEM submission material.

Treat each numbered subsection as an **atomic unit of work**: it has a single
goal, a prerequisite list, a runnable command set, an expected-output
description, an explicit validation gate, a list of known failure modes with
recovery procedures, and the conventional-commit message that should land it.

If a step fails the validation gate, do not advance. Use the rollback procedure
for that step. If the failure mode is novel (not listed), stop and capture the
state under `runs/incidents/` before debugging — do not silently work around
it.

This roadmap supersedes nothing. The pre-existing `docs/data_augmentation_plan.md`
remains the canonical implementation spec for fetch_pdb_complexes, the BindingDB
join, and phospho parametrisation. This document orchestrates and prioritises
those existing specs alongside new GPU-resident work.

---

## 1. Executive summary

### 1.1 The current state, honestly

| Metric | Current | What it actually means |
|--------|--------:|-----------------------|
| Pearson r | **0.860** | On 6 training complexes only; statistically a point estimate, not a population claim |
| RMSE | 1.73 kcal/mol | Inflated by crystal-pose Vina overshoot |
| α (entropy coefficient) | **0.100** | At lower bound — CLAUDE.md §9 flag condition |
| β (AD4 blend weight) | 0.000 | AD4 signal entirely discarded |
| Training set size | 6 | Statistically underpowered |
| Calibration receptor | holo | Wrong; should be apo for production |
| PepSet held-out r | unknown | Never measured at population level |
| Phospho-peptide coverage | 0% | PLK1/SHP2 families silently mis-scored by 3–5 kcal/mol |
| RAPiDock PPII success rate | poor | SH3 per-contact 0.81 vs PDZ 1.05 vs BRD 1.79 |

The headline r=0.860 is real but **not transferable**. The first job of this
plan is to find out what r actually is on 100+ holdout complexes, then move it
upward through the targeted interventions in Tier 1–3.

### 1.2 Where the gains will come from

In rough order of return on investment:

1. **Calibration data quality** (Tier 0.4, Tier 1.3) — switching from 6
   crystal-pose complexes to 200+ production-pose complexes with real
   experimental Kd values. Single biggest predicted gain.
2. **Family-adaptive β** (Tier 2.1) — turning on AD4 signal where it helps
   (bromodomain, SH2, phospho) and off where it hurts (BCL-2). Cheap, high-yield.
3. **Second RAPiDock fine-tune on expanded dataset** (Tier 1.1) — using fresh
   PDB complexes from fetch_pdb_complexes.py to roughly double the PPII training
   examples.
4. **CHPi geometric correction** (Tier 2.2) — close the SH3/WW per-contact gap.
5. **Phospho-residue parametrisation** (Tier 3.1) — unlocks 40 new calibration
   complexes and an entire pharmacological target class.
6. **XGBoost / GBM scoring head** (Tier 2.3) — non-linear replacement for the
   single-α single-β linear correction.

### 1.3 Resource budget

| Resource | Available | Required (full plan) | Headroom |
|----------|----------:|---------------------:|---------:|
| GPU VRAM (RTX 5070) | 12 GB | ~10 GB peak (RAPiDock + ESM2) | tight |
| GPU compute time | ~24 hrs/day | ~16 hrs across Tue–Thu | comfortable |
| Disk (Linux machine) | check before start | ~30 GB new data | check |
| CPU compute (Mac or Linux) | always available | ~12 hrs across tier 1–2 | comfortable |
| Network bandwidth | ~10 MB/s assumed | ~12 GB total downloads | ~30 min |
| Human attention | finite | ~4 active hrs/day | comfortable |

### 1.4 Predicted accuracy trajectory

| Checkpoint | Predicted Pearson r | Notes |
|-----------|--------------------:|-------|
| Today (training-set crystal poses, n=6) | 0.86 | not transferable |
| After Tier 0 (production recalibration, n=6) | 0.65–0.75 | honest collapse expected |
| After Tier 1.3 (expanded to n≈200) | 0.68–0.78 | first real population estimate |
| + Tier 1.1 second fine-tune | +0.02–0.05 | PPII families improve |
| + Tier 2.1 family-adaptive β | +0.03–0.05 | bromodomain/BCL-2 fix |
| + Tier 2.2 CHPi correction | +0.02–0.04 | SH3/WW fix |
| + Tier 2.3 XGBoost scorer | +0.05–0.10 | non-linear blending |
| + Tier 3.1 phospho support | unlocks PLK1/SHP2 families | adds calibration coverage |
| **Realistic end-of-week target** | **0.78–0.88** on n≈200 holdout | a defensible iGEM claim |

A drop from 0.860 → 0.7x is **expected and healthy**. The 6-point figure was
measuring training-set memorisation under a systematic bias; the 200-point
figure measures what HybriDock-Pep can actually do on data it has not seen.

---

## 2. Pre-flight checklist (run BEFORE any GPU work on Tuesday)

Do not skip this. Each item takes < 30 seconds and prevents class of failures
that have hit this project before.

### 2.1 Hardware and environment sanity

```bash
# 1. GPU is alive and recognised
nvidia-smi

# 2. Expected: RTX 5070, driver ≥ 580.x, CUDA 12.8 runtime
#    If "No devices found": reboot before doing anything else.

# 3. PyTorch sees CUDA
conda run -n rapidock python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('Compute cap:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NONE')
"

# Expected: CUDA available: True · Device: NVIDIA GeForce RTX 5070
#           PyTorch: 2.7.0+cu128 · Compute cap: (12, 0)

# 4. Conda envs intact
conda env list | grep -E "rapidock|score-env"
# Expected: both present

# 5. Disk space — at least 50 GB free in $HOME
df -h $HOME | awk 'NR==2 {print "Free:", $4}'
df -h /tmp | awk 'NR==2 {print "/tmp free:", $4}'

# 6. score-env smoke test
conda activate score-env
bash scripts/smoke_test.sh
# All three checks must PASS
```

If any of 1, 3, 4, 5, or 6 fail: **stop, debug, do not proceed to training.**

### 2.2 Git hygiene baseline

```bash
# Working tree clean before any new work
git status
# Expected: nothing to commit, working tree clean

# On master, up to date with origin
git branch --show-current
git fetch origin
git log --oneline @{u}..HEAD
# Expected: no output (no unpushed commits) OR known WIP

# Tag the pre-training state for easy rollback
git tag -a pre-tier0-$(date +%Y%m%d) -m "State before Tier 0 training runs"
```

### 2.3 FEP / GPU contention guard

If `run_finetune_and_compare.sh` finds OpenMM processes running, it aborts.
Verify nothing is running before you start a fine-tune:

```bash
ps aux | grep -E "openmm|simulate|fep" | grep -v grep
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# Expected: no openmm/FEP processes; nvidia-smi reports nothing using GPU
```

If you see leftover Python processes from a previous session that died: kill
them explicitly with their PID. Do not use `pkill -9 python`; that is
indiscriminate and will kill conda's daemons.

### 2.4 Backup of state files

These six files encode everything the calibration depends on. Back them up to
`data/_backups/$(date +%Y%m%d)/` before any step that rewrites them:

```bash
mkdir -p data/_backups/$(date +%Y%m%d)
cp data/calibration.json                  data/_backups/$(date +%Y%m%d)/
cp data/training_complexes.csv            data/_backups/$(date +%Y%m%d)/
cp data/training_scores.json              data/_backups/$(date +%Y%m%d)/
cp data/test_complexes.csv                data/_backups/$(date +%Y%m%d)/
cp data/test_complexes_meta.csv           data/_backups/$(date +%Y%m%d)/
[ -f data/training_complexes_expanded.csv ] && \
    cp data/training_complexes_expanded.csv data/_backups/$(date +%Y%m%d)/
```

Rollback is then `cp data/_backups/YYYYMMDD/X data/X`.

### 2.5 Existing test suite must be green

```bash
conda activate score-env
pytest -x -q -m "not slow"
# Expected: 172 passed (or close — accept skips, no failures)
```

If anything is RED before you start training: **fix that first.** Do not train
on top of a broken pipeline; the resulting checkpoints will not be trustable.

---

## 3. Tier 0 — Already scripted, just execute (Tuesday morning, ~5 hours)

### 3.1 · Tier 0.1 — RAPiDock last-layer fine-tune on the existing 925-complex set

**Goal:** Improve RAPiDock pose-generation accuracy on PPII / SH3 / WW families
without touching any other layer.

**Prerequisites:**
- Pre-flight checklist (§2) passed
- `third_party/RAPiDock_finetuned/` exists on the Linux machine
- `datasets/pdb_2024_2026/` and `datasets/ppii_enriched/` exist (with the
  initial 8 PPII entries)

**Step 1 — Format training data into RAPiDock's expected structure:**

```bash
conda run -n rapidock python scripts/prep_rapidock_training_data.py
# Expected: ~5–10 min CPU. Output: datasets/training_formatted/
```

**Validation gate (must pass before Step 2):**

```bash
ls datasets/training_formatted/ | wc -l
# Expected: 925 (or more, if reruns added entries)

# Spot-check one entry
ls datasets/training_formatted/$(ls datasets/training_formatted/ | head -1)
# Expected: receptor.pdb, peptide.pdb, esm_embedding.pt
```

If count < 900: investigate (likely silent parse failures in prep script).

**Step 2 — Run the fine-tune + compare orchestrator:**

```bash
# Foreground so you can watch the loss curve
bash scripts/run_finetune_and_compare.sh 2>&1 | tee runs/finetune_log_$(date +%Y%m%d_%H%M).txt

# Expected runtime: ~3 hours on RTX 5070 at batch_size=4, 50 epochs
# Expected GPU memory: ~6–8 GB peak (PyTorch + ESM2 + activations)
```

**Monitoring during the run:**

In a second terminal:
```bash
watch -n 5 nvidia-smi
# Watch for:
#   - GPU-Util > 80% (good)
#   - Memory growing without bound (BAD — kill if approaching 11 GB)
#   - Process disappears (BAD — script crashed)
```

```bash
# Loss curve sanity (every ~30 min)
tail -50 runs/finetune_log_*.txt | grep -E "epoch|loss|tr_loss|rot_loss"
# Expected: total loss monotonically decreasing across epochs (small fluctuation OK)
# If loss increases after epoch 10: stop, lower learning rate to 5e-5, restart
```

**Validation gate:**

The orchestrator runs `compare_rapidock_models.sh` automatically. Expected
output (final lines):

```
=== Original vs Finetuned Cα RMSD on PepSet ===
Family                Original   Finetuned   Δ
SH3 (n=13)            5.84       <5.84      <0
WW  (n=6)             4.71       <4.71      <0
PDZ (n=33)            2.13       ≤2.18      ≥−0.05  ← guard rail
Bromodomain (n=7)     2.45       ≤2.50      ≥−0.05  ← guard rail
Overall (n=185)       3.61       <3.61      <0
```

The PDZ and bromodomain guard rails are critical: those families are already
well-handled. If they regress by more than 0.05 Å, the fine-tune has
over-corrected toward PPII and the checkpoint must not be promoted to default.

**Failure modes:**

| Failure | Likely cause | Recovery |
|---------|-------------|----------|
| `CUDA OOM` mid-epoch | batch_size too large for RTX 5070 | Edit train_lastlayer.py → batch_size=2; restart |
| Loss diverges (NaN) | Learning rate too high | Lower lr to 5e-5; restart |
| Script aborts at FEP guard | OpenMM is running | `ps aux \| grep openmm`, kill it, re-run |
| PDZ regression > 0.05 Å | PPII oversampling too aggressive | Lower `--ppii-weight 4` → `2`; rerun |
| All families regress | Bug in prep_rapidock_training_data.py | Rollback checkpoint; debug data formatting |
| `ckpt file not found` | Path drift on the Linux machine | Check `train_models/CGTensorProductEquivariantModel/rapidock_local.pt` exists |
| Compare script crashes | RAPiDock inference broken by checkpoint | Restore baseline ckpt; investigate weight shapes |

**Rollback procedure:**

```bash
# If validation gate fails, the original checkpoint is preserved at:
ls third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt
# Confirm size and date match pre-training state (compare against git LFS log if applicable)

# The finetuned ckpt is in:
ls third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_best.pt
# Rename to prevent it being picked up by inference:
mv third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_best.pt \
   third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_best.REJECTED.pt
```

**Commit:** No code changes from this step (only artifacts that are gitignored).
Write a note to `runs/finetune_runs.md` capturing seed, hyperparameters, and
Cα RMSD comparison. That file IS committed.

```bash
git add runs/finetune_runs.md
git commit -m "docs(training): log first 925-complex fine-tune results

Records seed, hyperparameters, baseline vs finetuned Cα RMSD per family,
and pass/fail vs guard rails. Checkpoints themselves are gitignored."
```

### 3.2 · Tier 0.2 — Fetch fresh PDB complexes (Item 1 + Item 2)

**Goal:** Materialise the 800+ recent and 150+ PPII-enriched complexes that
`docs/data_augmentation_plan.md` specifies.

**Can run in parallel with Tier 0.1.** This is CPU + network, independent of GPU.

**Prerequisites:**

```bash
# Verify rcsbsearch is installed in score-env
conda run -n score-env pip list | grep -i rcsbsearch
# If missing:
conda env update -n score-env -f envs/score-env.yml
```

**Run:**

```bash
conda run -n score-env python scripts/fetch_pdb_complexes.py --mode both --max-workers 4
# Expected: ~30–60 min depending on RCSB latency
```

**Monitoring:**

```bash
# In another terminal — watch download progress
watch -n 10 'ls datasets/pdb_2024_2026/structures/ 2>/dev/null | wc -l; \
             ls datasets/ppii_enriched/structures/ 2>/dev/null | wc -l'
```

**Validation gate:**

```bash
wc -l datasets/pdb_2024_2026/manifest.csv
# Expected: ≥ 800 rows
wc -l datasets/ppii_enriched/manifest.csv
# Expected: ≥ 150 rows passing the PPII filter

# Sanity check: no PepSet leakage
python -c "
import pandas as pd
pepset = set(open('datasets/pepset/pepset_ids.txt').read().split())
for path in ['datasets/pdb_2024_2026/manifest.csv', 'datasets/ppii_enriched/manifest.csv']:
    df = pd.read_csv(path)
    included = df[df['excluded_reason'] == '']
    leak = set(included['pdb_id'].str.upper()) & pepset
    print(f'{path}: {len(leak)} PepSet leaks')
    assert not leak, f'PEPSET LEAK: {leak}'
print('OK — no PepSet leakage')
"
```

If PepSet leakage is non-zero: the dedup logic in fetch_pdb_complexes.py is
broken. Stop, fix, re-run. **This is a critical safeguard.** PepSet must remain
held-out.

**Failure modes:**

| Failure | Recovery |
|---------|----------|
| RCSB API rate-limited (HTTP 429) | Lower `--max-workers` to 2; sleep and retry |
| Some structures fail to download | Acceptable if < 5% of total; logged in manifest with `excluded_reason="download_failed"` |
| Manifest count under threshold | Investigate which step lost rows; check `excluded_reason` distribution |
| Ramachandran calc errors on a chain | Caught by `_parse_chain_info`; row marked `excluded_reason="parse_failed"` |
| Out of disk space | `df -h`; clean `runs/` artifacts or move `datasets/` to a larger volume |

**Rollback:** Datasets are append-only; delete directories to start over:

```bash
rm -rf datasets/pdb_2024_2026 datasets/ppii_enriched
```

**Commit:** Manifests are large; only commit if they are below 5 MB. Otherwise
add to `.gitignore` and note size in a small `datasets/MANIFEST_SIZES.md`.

### 3.3 · Tier 0.3 — BindingDB calibration join

**Goal:** Expand the 6-row training_complexes.csv to ~100–200 rows with
experimental Kd values from BindingDB.

**Can run in parallel with Tier 0.1 + 0.2.**

**Prerequisites:**

- ~10 GB free in `datasets/cache/` for BindingDB raw + parsed data
- RDKit available in score-env (already installed)

**Run:**

```bash
conda run -n score-env python scripts/bindingdb_calibration_join.py
# Expected: ~15–25 min on first run (downloads + parses 2 GB)
# Cached subsequent runs: 1–2 min
```

**Validation gate:**

```bash
wc -l data/training_complexes_expanded.csv
# Expected: ≥ 100 rows (target 200)

python -c "
import pandas as pd
df = pd.read_csv('data/training_complexes_expanded.csv')

# pKd distribution check
print('pKd stats:')
print(df['experimental_pkd'].describe())
assert df['experimental_pkd'].between(3, 12).all(), 'pKd out of physical range'

# No empty sequences
empty = df['peptide_sequence'].isna() | (df['peptide_sequence'] == '')
print(f'Empty sequences: {empty.sum()}')
# Some BindingDB rows can't be sequenced from SMILES — acceptable up to ~30%
assert empty.sum() / len(df) < 0.5, 'Too many empty sequences'

# No PepSet leakage
pepset = set(open('datasets/pepset/pepset_ids.txt').read().split())
leak = set(df['pdb_id'].str.upper()) & pepset
assert not leak, f'PEPSET LEAK in BindingDB join: {leak}'
print('OK — calibration set clean')
"
```

**Known issue:** Per memory log, the current expanded CSV has empty
peptide_sequence for many rows (SMILES → sequence converter limitations). The
acceptance criterion above tolerates up to 50% empty as long as the non-empty
subset is ≥100. If empty fraction is higher, prioritise improving
`_smiles_to_sequence()` over running calibration.

**Failure modes:**

| Failure | Recovery |
|---------|----------|
| BindingDB schema changed; columns missing | Update column names in script per current TSV header |
| Download fails (URL changed) | Manually scrape new URL from https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp |
| RDKit fails to parse some SMILES | Acceptable if < 30%; logged and excluded |
| pKd outside physical range | Bug in `_kd_to_pkd`; verify nanomolar units assumed |
| All output rows have empty sequence | `_smiles_to_sequence` broken; check pattern matching against amino-acid SMILES templates |

**Commit:** `data/training_complexes_expanded.csv` is small and should be
committed.

```bash
git add data/training_complexes_expanded.csv
git commit -m "data(calibration): refresh BindingDB-joined training complexes

Expanded calibration set from 6 to N rows. Source: BindingDB All Data
snapshot YYYY-MM. PepSet/RefPepDB entries excluded to preserve holdout."
```

### 3.4 · Tier 0.4 — Production-pose recalibration (fixes α=0.1)

**Goal:** Replace the holo-receptor crystal-pose calibration with
apo-receptor production-pose calibration. This addresses the root cause
documented in `docs/calibration_notes.md` §Issue 1.

**Prerequisites:**

- Tier 0.1 fine-tune has completed (use the new RAPiDock checkpoint)
- PepSet `datasets/pepset/<pdb>/<pdb>_rec_unbound_pocket.pdb` files present
  for the 6 training complexes (2hwn, 1nrl, 1l2z, 1ddv, 1a0n, 1ywi)

**Step 1 — Run each training complex through the full pipeline against the apo receptor:**

```bash
mkdir -p runs/calibration_production
# Read the 6 training complexes from training_complexes.csv
python -c "
import pandas as pd, json
df = pd.read_csv('data/training_complexes.csv')
print(df.to_string())
" | tee runs/calibration_production/training_complexes_used.txt
```

For each of the 6, run:

```bash
PDB=2hwn
PEPTIDE='SLGRFKRPLFFGSDP'  # from training_complexes.csv (literal sequence per row)
SITE='..'                   # use scripts.benchmark.get_peptide_center to compute
hybridock-pep dock \
    --peptide "$PEPTIDE" \
    --receptor datasets/pepset/${PDB}/${PDB}_rec_unbound_pocket.pdb \
    --site $SITE \
    --box 40 \
    --n-samples 100 \
    --seed 42 \
    --scoring vina,ad4 \
    --output-dir runs/calibration_production/${PDB}
```

A wrapper script `scripts/run_production_calibration.sh` should be written for
this; it loops over training_complexes.csv and applies the per-row site
coordinates (use `scripts/benchmark.get_peptide_center` against the crystal
peptide PDB to compute the site).

Total expected runtime: **6 × ~8 min ≈ 50 min** on RTX 5070.

**Step 2 — Collect best-cluster scores into a new training_scores JSON:**

```bash
python -c "
import json, glob, pandas as pd
out = {}
for run in sorted(glob.glob('runs/calibration_production/*/ranked_poses.csv')):
    pdb_id = run.split('/')[-2]
    df = pd.read_csv(run)
    if df.empty:
        print(f'WARN: {pdb_id} has empty ranked_poses')
        continue
    best = df.iloc[0]
    out[pdb_id] = {
        'vina_score': float(best['vina_score']),
        'ad4_score': float(best['ad4_score']),
        'n_contact_residues': int(best.get('n_contact_residues', 0)),
    }
json.dump(out, open('data/training_scores_production.json', 'w'), indent=2)
print(f'Collected {len(out)} scores')
"
```

**Step 3 — Recalibrate:**

```bash
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --scores-json data/training_scores_production.json \
    --output data/calibration_production.json
# Expected: 1–2 sec
```

**Validation gate:**

```bash
python -c "
import json
old = json.load(open('data/_backups/$(date +%Y%m%d)/calibration.json'))
new = json.load(open('data/calibration_production.json'))
print(f\"alpha: {old['alpha']:.3f} → {new['alpha']:.3f}\")
print(f\"beta:  {old['beta']:.3f} → {new['beta']:.3f}\")
print(f\"r:     {old['pearson_r']:.3f} → {new['pearson_r']:.3f}\")
print(f\"rmse:  {old['rmse_kcal_mol']:.3f} → {new['rmse_kcal_mol']:.3f}\")
"
```

Expected behaviour:

| Quantity | Pre (holo, crystal) | Post (apo, production) | Interpretation |
|----------|--------------------:|-----------------------:|----------------|
| α | 0.100 (bound) | **0.3–0.9** | Should move off bound — this is the proof the fix worked |
| β | 0.000 | 0.05–0.30 | AD4 may activate |
| Pearson r | 0.860 | 0.65–0.80 | Expected to drop; honest estimate |
| RMSE | 1.73 | 1.0–2.0 | May decrease or stay similar |

**If α is still at 0.1 after this recalibration:** the root cause is not the
holo/apo issue but the small-sample-size issue. Continue to Tier 1.3, which
will fix that.

**Failure modes:**

| Failure | Recovery |
|---------|----------|
| One of the 6 dock runs produces empty ranked_poses.csv | Investigate that complex — check site coords, box size, peptide chain |
| RAPiDock generates < 50 poses (target 100) | Pipeline issue; check rapidock subprocess log with `-vv` |
| All Vina scores are positive (clashes) | Receptor prep failed or site is wrong |
| AD4 anomaly rate > 30% | Lower the site-box; receptor pocket may not be well-defined |
| α at upper bound (1.2) | Likely overcorrection; investigate which complex is the outlier |
| Pearson r drops below 0.5 | Calibration set is fundamentally wrong; revisit complex selection |

**Rollback:**

```bash
# The new calibration is in calibration_production.json (separate from calibration.json)
# To revert, simply do not overwrite calibration.json. If you accidentally already did:
cp data/_backups/$(date +%Y%m%d)/calibration.json data/calibration.json
```

**Commit:**

```bash
git add data/training_scores_production.json data/calibration_production.json \
        runs/calibration_production/  # only metadata; ranked_poses.csv may be too large
git commit -m "feat(calibration): production-pose recalibration on apo receptors

Replaces crystal-pose calibration (Issue 1 in calibration_notes.md).
α moves from lower bound 0.1 to X.XX. Pearson r on training set: Y.YY.
Honest estimate; population r will be measured in Tier 1.3."
```

Do not overwrite `data/calibration.json` yet. Tier 1.3 will produce an even
better calibration on 200 complexes; that becomes the new default.

---

## 4. Tier 1 — High-impact new GPU work (Tuesday afternoon → Wednesday)

### 4.1 · Tier 1.1 — Second fine-tune on expanded 1700+ complex dataset

**Goal:** Re-train the RAPiDock final heads using the union of the original 925
complexes and the newly fetched ~800 recent + ~150 PPII complexes.

**Prerequisites:**

- Tier 0.1 has completed (so you have a baseline to compare against)
- Tier 0.2 has completed (so `datasets/pdb_2024_2026/` and `datasets/ppii_enriched/`
  exist with full content)

**Step 1 — Extend prep script to include the new sources:**

```bash
# Modify scripts/prep_rapidock_training_data.py to scan:
#   datasets/RefPepDB-RecentSet/  (existing)
#   datasets/pdb_2024_2026/       (new)
#   datasets/ppii_enriched/       (new)
# Apply same source-weighted oversampling: PPII 4×, recent 2024+ 2×.

conda run -n rapidock python scripts/prep_rapidock_training_data.py --include-recent --include-ppii
# Expected: ~10–20 min CPU; output: datasets/training_formatted/ with ~1700 entries
```

**Validation gate:**

```bash
ls datasets/training_formatted/ | wc -l
# Expected: ≥ 1700

# Source distribution check
python -c "
import json, glob
src = {'refpepdb': 0, 'recent': 0, 'ppii': 0}
for f in glob.glob('datasets/training_formatted/*/meta.json'):
    s = json.load(open(f))['source']
    src[s] = src.get(s, 0) + 1
print('Source distribution:', src)
"
```

**Step 2 — Run the fine-tune with extended training set:**

The orchestrator script supports the expanded dataset directly. Increase
epochs to 75 (more data → more gradient steps needed for convergence) and
leave PPII oversampling at 4×.

Edit `scripts/run_finetune_and_compare.sh`:
```bash
# Change: --epochs 50  →  --epochs 75
```

Then:
```bash
# Overnight run (Tuesday → Wednesday morning)
nohup bash scripts/run_finetune_and_compare.sh > runs/finetune2_log.txt 2>&1 &
echo $! > runs/finetune2.pid

# Expected: ~6–8 hours on RTX 5070 (1.85× data, 1.5× epochs)
```

**Monitoring during overnight run:**

```bash
# Before going to bed, check first epoch completes successfully
tail -20 runs/finetune2_log.txt
# Should see "Epoch 1/75" and a non-NaN loss

# Set a phone alarm for ~T+45 min to verify training is still alive
```

**Validation gate (Wednesday morning):**

Same family-level Cα RMSD comparison as Tier 0.1, with an additional comparison
to the Tier 0.1 checkpoint:

```
Family       Original   Tier-0.1   Tier-1.1   Best
SH3 (n=13)   5.84       4.21       <4.21      ← expect further improvement
PPII (n=8+)  6.12       4.50       <4.50      ← expect substantial improvement
PDZ (n=33)   2.13       2.16       ≤2.18      ← guard rail held
BRD (n=7)    2.45       2.48       ≤2.50      ← guard rail held
Overall      3.61       3.42       <3.42      ← expect modest overall gain
```

If overall RMSD is worse than Tier 0.1: the new data introduced noise; revert
to Tier 0.1 checkpoint. If only specific families regress: investigate which
new entries are bad (check Ramachandran filter behaviour on borderline cases).

**Failure modes:** same as Tier 0.1 plus:

| Failure | Recovery |
|---------|----------|
| Overnight run dies | Check `runs/finetune2_log.txt` for last epoch; restart from checkpoint if torch save was emitted |
| Wall-clock > 12 hrs | Reduce epochs to 60; balance vs accuracy |
| New data contains corrupt PDBs | Add validation step in prep script; exclude failing entries with reason |
| GPU temperature throttles | Ensure case airflow; consider lowering batch_size or epoch count |

**Commit:**

```bash
git add runs/finetune2_results.md
git commit -m "docs(training): record second fine-tune on 1700+ complex set

Compares Tier-0.1 vs Tier-1.1 checkpoints per family.
Overall Cα RMSD: X.XX → Y.YY. PPII families: A.AA → B.BB."
```

### 4.2 · Tier 1.2 — Full PepSet population-level Pearson r

**Goal:** Replace the n=6 training-set Pearson r with an n=185 PepSet
population-level Pearson r. This is the number iGEM judges will care about.

**Important:** PepSet is **not** in any training data. This is a held-out
benchmark, and treating it as such is the entire reason we have it.

**Prerequisites:**

- `datasets/pepset/` populated with all 185 complexes including their
  experimental pKd values (per `data/test_complexes.csv` or equivalent)
- Tier 0.4 calibration in place

**Run (CPU, no GPU needed):**

```bash
conda run -n score-env python scripts/score_family_benchmark.py \
    --pepset-dir datasets/pepset/ \
    --calibration data/calibration_production.json \
    --output runs/pepset_population/pepset_scores.csv
# Expected: ~90 min on CPU
```

(If `scripts/score_family_benchmark.py` is currently parameterised for a
single family or fixture subset, extend it with a `--pepset-dir` flag that
iterates over all 185 complexes.)

**Step 2 — Compute Pearson r:**

```bash
python -c "
import pandas as pd
from scipy.stats import pearsonr, spearmanr
df = pd.read_csv('runs/pepset_population/pepset_scores.csv')

# Drop known systematic outliers per calibration_notes.md
exclude = {'4GQ6', '2KOH'}
df = df[~df['pdb_id'].str.upper().isin(exclude)]

r, p = pearsonr(df['hybrid_score'], df['experimental_pkd'])
rho, pp = spearmanr(df['hybrid_score'], df['experimental_pkd'])
print(f'Pearson r:  {r:.3f} (p={p:.2e})')
print(f'Spearman ρ: {rho:.3f} (p={pp:.2e})')
print(f'n = {len(df)}')

# Per-family breakdown
for fam, g in df.groupby('family_hint'):
    if len(g) >= 5:
        r_f, _ = pearsonr(g['hybrid_score'], g['experimental_pkd'])
        print(f'  {fam:25s}  n={len(g):3d}  r={r_f:+.3f}')
"
```

**Validation gate:**

| r value | Action |
|---------|--------|
| ≥ 0.70 | Strong; proceed to Tier 2 |
| 0.55–0.70 | Acceptable; Tier 2 should push above 0.70 |
| 0.40–0.55 | Weak but salvageable; investigate worst families before Tier 2 |
| < 0.40 | Stop; investigate fundamental issue. Look for systematic bug not statistical noise |

**Failure modes:**

| Failure | Recovery |
|---------|----------|
| Many complexes return `hybrid_score == NaN` | Receptor prep failures; check per-complex stderr |
| Per-family r is very heterogeneous (some +0.9, some −0.2) | Confirms family-adaptive β need (Tier 2.1) |
| pHLA dominates and skews overall r | Stratify report; report pHLA separately as binary classification problem per dataset_analysis §4.1 |
| Memory error processing all 185 | Process in batches of 20; checkpoint after each |

**Commit:**

```bash
git add scripts/score_family_benchmark.py runs/pepset_population/pepset_scores.csv \
        runs/pepset_population/REPORT.md
git commit -m "feat(benchmark): full 185-complex PepSet population accuracy

First honest population-level Pearson r on held-out test set:
- Pearson r = X.XX (n=183 after excluding 4GQ6, 2KOH outliers)
- Spearman ρ = Y.YY
- Per-family breakdown in runs/pepset_population/REPORT.md"
```

### 4.3 · Tier 1.3 — Calibration on 200-complex BindingDB-expanded set

**Goal:** Recalibrate α and β on a calibration set with ≥100 distinct
complexes, replacing the n=6 calibration entirely.

**Prerequisites:**

- Tier 0.3 (BindingDB join) produced ≥100 rows in `training_complexes_expanded.csv`
- The structures referenced in expanded CSV are downloaded under
  `datasets/training_expanded_structures/`

**Step 1 — Download missing structures:**

```bash
# Extend fetch_pdb_complexes.py with a --pdb-list option:
conda run -n score-env python scripts/fetch_pdb_complexes.py \
    --mode list \
    --pdb-list <(awk -F, 'NR>1 {print toupper($1)}' data/training_complexes_expanded.csv) \
    --output-dir datasets/training_expanded_structures/
# Expected: ~30–45 min (200 small downloads)
```

**Step 2 — Score all 200 with crystal poses:**

```bash
conda run -n score-env python scripts/score_family_benchmark.py \
    --pepset-dir datasets/training_expanded_structures/ \
    --calibration data/calibration_production.json \
    --output runs/calibration_expanded/expanded_scores.csv
# Expected: ~2 hrs on CPU
```

**Step 3 — Recalibrate:**

```bash
# Convert scored CSV into the format expected by hybridock-pep calibrate
python scripts/scores_csv_to_training_json.py \
    runs/calibration_expanded/expanded_scores.csv \
    > data/training_scores_expanded.json

hybridock-pep calibrate \
    --training-csv data/training_complexes_expanded.csv \
    --scores-json data/training_scores_expanded.json \
    --output data/calibration_expanded.json
```

**Validation gate:**

```python
import json
new = json.load(open('data/calibration_expanded.json'))
assert new['n_complexes'] >= 100, 'Calibration set too small'
assert 0.15 < new['alpha'] < 1.5, f'α out of expected range: {new["alpha"]}'
assert 0.0 <= new['beta'] < 0.5, f'β out of expected range: {new["beta"]}'
assert new['pearson_r'] > 0.5, f'r too low: {new["pearson_r"]}'
print('CALIBRATION VALID')
```

**This becomes the new default calibration.** Promote it:

```bash
cp data/calibration.json                data/calibration_legacy_6complex.json
cp data/calibration_expanded.json       data/calibration.json
```

**Validation gate (regression test):**

```bash
# Re-run Tier 1.2 PepSet evaluation with the new calibration
conda run -n score-env python scripts/score_family_benchmark.py \
    --pepset-dir datasets/pepset/ \
    --calibration data/calibration.json \
    --output runs/pepset_population/pepset_scores_v2.csv

# Compare Pearson r before and after promotion
# Expected: r should hold or improve. If r decreases by > 0.05, do not promote
```

**Failure modes:**

| Failure | Recovery |
|---------|----------|
| α at upper bound (1.5) | Calibration set has systematic over-penalty; check entropy mode |
| β unusually high (> 0.4) | Possible BindingDB-specific bias; verify Ki vs Kd split |
| PepSet r drops with new calibration | New calibration overfits to BindingDB style; revert to Tier 0.4 |
| Calibration crashes (singular matrix) | Insufficient diversity in calibration set; check pKd spread |

**Commit:**

```bash
git add data/calibration.json data/calibration_legacy_6complex.json \
        data/calibration_expanded.json data/training_scores_expanded.json \
        runs/calibration_expanded/REPORT.md
git commit -m "feat(calibration): promote 200-complex BindingDB-based calibration

Replaces 6-complex calibration as default. PepSet population r:
A.AA → B.BB. Old calibration preserved as calibration_legacy_6complex.json
for reproducibility of prior runs."
```

---

## 5. Tier 2 — Medium-impact, new-code (Wednesday → Thursday)

### 5.1 · Tier 2.1 — Family-adaptive β calibration

**Goal:** Replace the single scalar β with a per-family lookup so AD4 signal
is used where it helps (BRD, SH2, phospho) and suppressed where it hurts
(BCL-2).

**Prerequisites:**

- Tier 1.2 PepSet scoring complete (provides per-family score data)
- Tier 1.3 calibration in place

**Step 1 — Implement family classifier:**

Add `src/hybridock_pep/scoring/family_classifier.py`:

```python
"""Heuristic family classification from receptor sequence motifs."""
from pathlib import Path
import re

FAMILY_MOTIFS = {
    "bromodomain": [r"WPF", r"NLI", r"YYDIIK"],          # WPF shelf
    "sh2_phospho": [r"FLVRES", r"YESYY"],                  # SH2 signature
    "pdz":         [r"GLGF", r"GYGF", r"GFGF"],            # carboxylate loop
    "bcl2_bh3":    [r"NWGRIVAF", r"VYLGTNGN"],             # BH3 groove residues
    "sh3":         [r"ALYDFD", r"RAQYNF"],                 # RT-loop signature
    "ww":          None,                                    # see _ww_check
    "calmodulin":  [r"DQLTEEQIAEFKEAFSLF"],               # CaM N-lobe canonical
    "arm_heat":    None,                                   # see _arm_check
    "mdm2":        [r"VTCTYSPALNK", r"FHIIYG"],            # MDM2 cleft
    "kinase":      [r"DFG", r"HRD"],                       # catalytic loop
}

def classify_family(receptor_pdb: Path) -> str:
    """Return family tag or 'default'. Conservative — only assigns
    if motif match is unambiguous."""
    seq = _extract_receptor_sequence(receptor_pdb)
    matches = []
    for fam, patterns in FAMILY_MOTIFS.items():
        if patterns is None:
            continue
        if any(re.search(p, seq) for p in patterns):
            matches.append(fam)
    # Additional ad hoc rules
    if _arm_heat_signature(seq):
        matches.append("arm_heat")
    if _ww_signature(seq):
        matches.append("ww")
    if len(matches) == 1:
        return matches[0]
    return "default"
```

**Step 2 — Calibrate β per family:**

For each family with n ≥ 8 in the PepSet population data, fit a separate β
using `fit_calibration` restricted to that family's complexes. Store in JSON:

```json
{
  "alpha": 0.65,
  "beta": 0.15,
  "family_beta": {
    "bromodomain": 0.38,
    "sh2_phospho": 0.35,
    "arm_heat":    0.22,
    "pdz":         0.10,
    "bcl2_bh3":    0.00,
    "default":     0.15
  },
  "family_n":    {"bromodomain": 7, "sh2_phospho": 8, ...},
  "family_r":    {"bromodomain": 0.85, "sh2_phospho": 0.72, ...}
}
```

**Step 3 — Update scoring code:**

In `src/hybridock_pep/scoring/entropy.py`, modify `apply_hybrid_score` to
look up β from `family_beta[family]` (falling back to `beta` if family not
known). The receptor family classification happens once per dock call in
`driver.run_dock` (after receptor prep) and is passed to the scoring stage.

**Validation gate:**

```bash
# Re-run PepSet evaluation with family-adaptive β
python scripts/evaluate_family_beta.py
# Expected: per-family r ≥ uniform-β baseline for at least 5 of 7 families
# Overall r ≥ uniform-β baseline + 0.03
```

**Safeguards:**

- **Minimum n per family for β fit:** require n ≥ 8 to fit a family-specific β.
  Families with smaller n use `default`.
- **β bounded per family:** if family β fits outside [0.0, 0.5], clip and log.
- **Family classifier confidence:** if `classify_family` returns "default",
  the existing global β is used. This is intentionally conservative.

**Failure modes:**

| Failure | Recovery |
|---------|----------|
| Family classifier wrong on a known complex | Add unit test fixture; tune motif regex |
| One family β fit gives wild value | Likely small-n noise; revert that family to default |
| Overall r degrades | Family β over-fitting; require larger n threshold (12+) |
| Existing tests break | Update tests to pass receptor family explicitly where mocked |

**Tests to add:**
- `tests/test_family_classifier.py` — 7 PepSet complexes, expected family tag
- `tests/test_family_beta_calibration.py` — confirm β lookup behaviour
- `tests/test_scoring_with_family.py` — end-to-end test that family-adaptive
  scoring matches uniform β when family is "default"

**Commit:**

```bash
git add src/hybridock_pep/scoring/family_classifier.py \
        src/hybridock_pep/scoring/entropy.py \
        data/calibration.json \
        tests/test_family_classifier.py tests/test_family_beta_calibration.py \
        tests/test_scoring_with_family.py
git commit -m "feat(scoring): family-adaptive β for hybrid score blending

Per-family β lookup replaces single scalar β. Bromodomain/SH2 use
high β (AD4 helps); BCL-2 uses β=0 (AD4 hurts). Conservative fallback
to default β when family classifier is uncertain. PepSet r: X→Y."
```

### 5.2 · Tier 2.2 — CHPi geometric correction term

**Goal:** Add a post-hoc correction for C-H···π interactions, which dominate
SH3/WW binding energetics but are absent from both Vina and AD4.

**Prerequisites:**

- Tier 1.2 PepSet scoring complete (provides SH3/WW family per-contact
  ground truth)

**Step 1 — Implement geometric detector:**

Add `src/hybridock_pep/scoring/chpi.py`:

```python
"""CH···π interaction detector.

Reference geometry (Brandl et al. 2001, JMB 307:357):
- Distance from aromatic ring centroid to C-H carbon: 3.5–4.5 Å
- Angle between C-H vector and ring plane normal: ≤ 30°
- Energy per contact: ~0.3–0.6 kcal/mol (literature range)

This module detects proline ring CH (Cβ, Cγ, Cδ) interacting with aromatic
ring centroids (Trp, Tyr, Phe, His) on the receptor. Per-contact energy is
a calibrated parameter, not a literature constant.
"""
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser

PROLINE_CH_ATOMS = {"CB", "CG", "CD"}
AROMATIC_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}

def count_chpi_contacts(
    peptide_pdb: Path,
    receptor_pdb: Path,
    distance_max: float = 4.5,
    angle_max_deg: float = 30.0,
) -> int:
    """Count distinct (Pro residue × aromatic residue) CH-π contacts."""
    ...

def chpi_correction(
    peptide_pdb: Path,
    receptor_pdb: Path,
    per_contact_kcal: float,
) -> float:
    """Return negative kcal/mol correction for CH-π contacts."""
    n = count_chpi_contacts(peptide_pdb, receptor_pdb)
    return -per_contact_kcal * n
```

**Step 2 — Calibrate per-contact energy:**

Use the 13 SH3 + 6 WW PepSet complexes. For each, compute:
- True ΔG from experimental pKd: `ΔG = -RT × ln(10) × pKd`
- Predicted hybrid score (current calibration)
- Residual: `ΔG_true - hybrid_pred`
- CHPi contact count: `count_chpi_contacts(peptide, receptor)`

Fit: `per_contact_kcal = mean(residual / n_chpi)` across the 19 complexes.
Use bootstrap (1000 resamples) to get 95% CI. Expected value: 0.3–0.6 kcal/mol.

**Step 3 — Integrate into scoring:**

In `scoring/entropy.py`'s `apply_hybrid_score`, after the existing hybrid
calculation, add:

```python
if config.use_chpi_correction:
    chpi_kcal = chpi_correction(pose.pdb_path, config.receptor_path,
                                per_contact_kcal=calibration["chpi_per_contact"])
    pose.hybrid_score += chpi_kcal
    pose.chpi_contacts = count_chpi_contacts(pose.pdb_path, config.receptor_path)
```

Feature-flag via `DockConfig.use_chpi_correction: bool = True`.

**Validation gate:**

```bash
# Re-evaluate PepSet with CHPi on
python scripts/evaluate_chpi_correction.py
# Expected:
#   SH3 per-contact score: 0.81 → 1.20+ kcal/mol/contact
#   WW per-contact score:  similar improvement
#   PDZ, BRD, MDM2: no change (no proline-aromatic contacts)
#   Overall r: ≥ previous (improvement of 0.02–0.04 expected)
```

**Safeguards:**

- **Per-contact energy bounded:** clamp `per_contact_kcal` to [0.1, 0.8]
  before saving to calibration.
- **Maximum correction per pose:** cap total CHPi correction at −3 kcal/mol
  to prevent runaway over-correction for poly-proline peptides with many
  pseudo-contacts.
- **Geometric tolerance:** the distance and angle thresholds are
  conservative. Tighter thresholds give cleaner but sparser counts.
- **Backward compatibility:** existing ranked_poses.csv loaders should
  tolerate an additional `chpi_contacts` column (write a migration test).
- **Validation on non-SH3 families:** confirm CHPi correction is ~0 for
  PDZ, BRD, MDM2, BCL-2 (no proline-rich peptides). If non-zero, the
  detector is over-eager.

**Failure modes:**

| Failure | Recovery |
|---------|----------|
| Correction over-applies to non-PPII peptides | Tighten geometry; require Pro ring centroid not just CH-atom |
| SH3 r doesn't improve | Either calibration is wrong or detector misses real contacts; visualise in PyMOL |
| RMSE worsens | Per-contact value too aggressive; refit with bootstrap |
| Slow scoring (Biopython parsing per pose) | Cache parsed structures across poses |

**Commit:**

```bash
git add src/hybridock_pep/scoring/chpi.py \
        src/hybridock_pep/scoring/entropy.py \
        src/hybridock_pep/models.py \
        data/calibration.json \
        tests/test_chpi.py
git commit -m "feat(scoring): CH-π geometric correction for SH3/WW peptides

Detects Pro→Trp/Tyr CH-π contacts using Brandl 2001 geometry.
Calibrated per-contact energy: 0.XX kcal/mol (95% CI 0.YY–0.ZZ).
SH3 per-contact score: 0.81 → 1.XX kcal/mol/contact. PepSet r: A→B."
```

### 5.3 · Tier 2.3 — XGBoost non-linear scoring head

**Goal:** Replace the linear `hybrid = vina + β(ad4-vina) + α·n` with a
learned non-linear function that captures family-level systematics and
feature interactions.

**Prerequisites:**

- Tier 1.3 expanded calibration in place
- Tier 1.2 PepSet population scoring complete (provides test set)
- Tier 2.1 family classifier in place (one of the input features)

**Step 1 — Feature engineering:**

For each (pose, receptor) pair, compute:

| Feature | Source | Description |
|---------|--------|-------------|
| `vina_score` | existing | global Vina ΔG estimate |
| `ad4_score` | existing | AD4 ΔG estimate |
| `n_contact` | existing | number of contact residues (≤ 4.5 Å) |
| `n_residues` | existing | peptide length |
| `contact_fraction` | computed | n_contact / n_residues |
| `n_chpi` | Tier 2.2 | CHPi contact count |
| `n_aromatic_in_pocket` | new | Trp/Tyr/Phe/His count in receptor binding pocket |
| `max_buried_depth` | new | deepest residue burial in Å |
| `mean_buried_depth` | new | mean burial depth across peptide residues |
| `family_onehot` | Tier 2.1 | one-hot encoded family (7 classes + default) |
| `peptide_helix_frac` | new | DSSP helix fraction (mean over poses) |
| `peptide_strand_frac` | new | DSSP strand fraction |
| `peptide_ppii_frac` | new | Ramachandran PPII residue fraction |
| `n_salt_bridges` | new | salt bridges within 4 Å |
| `mean_vdw_per_contact` | new | vdW score per contact residue |

**Step 2 — Training data:**

- Labels: experimental ΔG = −RT × ln(10) × pKd (kcal/mol)
- Features: computed from crystal pose against holo receptor
- Sources:
  - 185 PepSet complexes (with experimental pKd from literature)
  - 100–200 BindingDB-expanded complexes
- Train/test split: 80/20 random + family-stratified
- Cross-validation: 5-fold

**Step 3 — Model training:**

`scripts/train_xgb_scorer.py`:

```python
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("runs/calibration_expanded/feature_matrix.csv")
y = df["experimental_dg_kcal_mol"]
X = df.drop(columns=["pdb_id", "experimental_dg_kcal_mol"])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df["family_onehot_arg"]
)

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,         # shallow to prevent overfitting on small data
    learning_rate=0.05,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    early_stopping_rounds=20,
)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

# Save with sklearn metadata
import joblib; joblib.dump(model, "models/xgb_scorer_v1.joblib")
model.save_model("models/xgb_scorer_v1.json")  # portable
```

**Step 4 — Integrate as optional scoring backend:**

```python
# In src/hybridock_pep/scoring/entropy.py:
class HybridScorer:
    def __init__(self, calibration: dict, xgb_model_path: Path | None = None):
        self.calibration = calibration
        self.xgb_model = xgb.Booster(model_file=str(xgb_model_path)) if xgb_model_path else None

    def score(self, pose: ScoredPose, receptor: Receptor) -> float:
        if self.xgb_model is None:
            return self._linear_score(pose, receptor)
        return self._xgb_score(pose, receptor)
```

Feature-flag via `DockConfig.scoring_backend: Literal["linear", "xgboost"] = "linear"`.
Default remains linear until validated.

**Validation gate:**

| Test | Threshold |
|------|-----------|
| 5-fold CV Pearson r | ≥ linear baseline + 0.05 |
| Held-out 20% test Pearson r | ≥ linear baseline + 0.05 |
| Per-family Pearson r | ≥ linear baseline for all 7 families |
| Feature importance (SHAP) | dominated by vina, ad4, n_contact + at most 3 new features |
| Inference latency | ≤ 50 ms per pose (vs <1 ms for linear) |

If 5-fold CV shows large variance: model is overfitting; reduce
`n_estimators` or `max_depth`.

**Model card requirements (for iGEM submission):**

Write `models/xgb_scorer_v1_model_card.md` covering:
- Training data: sources, sample sizes, date range
- Features and preprocessing
- Hyperparameters and training procedure
- Test set performance with confidence intervals
- Known failure modes (cyclic peptides, very long peptides, etc.)
- Reproducibility envelope: random seed, library versions, hardware

**Safeguards:**

- **Linear scoring remains the default.** XGBoost is opt-in via DockConfig.
- **Model versioning:** save with version number; never overwrite previous.
- **Graceful fallback:** if model file missing or corrupt, fall back to linear
  with WARNING log line.
- **Out-of-distribution detection:** if a feature is outside the training
  range (e.g., peptide length > 25), warn and append `out_of_distribution=True`
  to the score record.
- **Reproducibility:** pin xgboost version in `envs/score-env.yml`.

**Failure modes:**

| Failure | Recovery |
|---------|----------|
| Train r=0.99 but test r=0.55 | Overfitting; reduce model complexity |
| Test r identical to linear | Insufficient signal in new features; revisit features |
| Per-family r regresses for one family | Insufficient samples in that family; family-stratified split |
| Inference too slow | Use fewer estimators; consider switching to LightGBM |
| Model file unreadable | Joblib version mismatch; pin versions, retrain |

**Commit:**

```bash
git add scripts/train_xgb_scorer.py scripts/compute_xgb_features.py \
        models/xgb_scorer_v1.json models/xgb_scorer_v1_model_card.md \
        src/hybridock_pep/scoring/entropy.py \
        src/hybridock_pep/models.py \
        tests/test_xgb_scorer.py \
        envs/score-env.yml
git commit -m "feat(scoring): XGBoost non-linear scoring head (opt-in)

Trains on 380 complexes (185 PepSet + 200 BindingDB). 5-fold CV
Pearson r: 0.XX (linear baseline: 0.YY). Opt-in via DockConfig
scoring_backend='xgboost'; linear remains default until further
validation. Model card under models/."
```

---

## 6. Tier 3 — Higher-effort improvements (Thursday–Friday)

### 6.1 · Tier 3.1 — Phospho-residue parametrisation (Item 6 of data_augmentation_plan)

**Goal:** Add TPO/SEP/PTR support to enable scoring of PLK1-PBD (31 structures),
SHP2 (9), and any other phospho-peptide datasets.

This is **already specified in detail** in `docs/data_augmentation_plan.md` §3.
Follow that spec.

**Critical safeguards specific to this step:**

- **Charge templates must match Gasteiger** for non-phospho atoms to within
  ±0.05. Otherwise the phospho parametrisation is inconsistent with the rest
  of the peptide and will not score sanely.
- **Test on a known complex first:** 4JMG SHP2 with `V[PTR]ENVGLM`.
  Expected AD4 ≤ −7 kcal/mol when phospho is parametrised; AD4 ≈ −4 when
  not. If you cannot reproduce this expected gap on the test fixture: the
  charge templates are wrong. Do not commit.
- **No effect on non-phospho peptides:** existing 172 tests must still pass.
  Branch path A (babel) is unchanged for non-phospho peptides.

**Additional safeguards for the receptor side:**

- `prep/receptor.py` strips all HETATM. Add an explicit whitelist for
  TPO/SEP/PTR. **Verify this doesn't accidentally retain other HETATMs**
  (e.g., crystallisation buffer, glycerol, Mg²⁺ that shouldn't be there).

### 6.2 · Tier 3.2 — Confidence model fine-tuning (conditional on architecture access)

**Goal:** Fine-tune RAPiDock's confidence head on HybriDock-Pep-labelled poses
to enable pre-ranking before expensive Vina/AD4 scoring.

**Prerequisites:**

- Inspection of `third_party/RAPiDock_finetuned/` source confirms the
  confidence model module exists and is loadable independently from the
  diffusion model
- Tier 1.1 fine-tuned checkpoint exists

**Why this is conditional:** RAPiDock paper describes a confidence model, but
the exact module layout in the codebase may differ. Before committing time,
verify with:

```bash
conda run -n rapidock python -c "
import sys; sys.path.insert(0, 'third_party/RAPiDock_finetuned')
import inference as rd
print(hasattr(rd, 'confidence_model'))
print([m for m in dir(rd) if 'confidence' in m.lower()])
"
```

If no confidence module is exposed: skip Tier 3.2.

**Approach (if accessible):**

- Define "good pose" label: `hybrid_score < median(hybrid_scores) - 1.5 × IQR`
  per complex (binary classification per pose)
- Use 925-complex training set scored through current HybriDock-Pep pipeline
- Fine-tune confidence head only (other layers frozen)
- 20 epochs, BCE loss, AdamW lr=1e-4

**Validation:**

- AUC ≥ 0.70 on held-out 20% of training complexes' poses
- Top-1 pose accuracy (after confidence ranking) ≥ Top-1 after current ranking + 5%

**Safeguards:**

- Never replace the existing pose ranker (Vina+AD4 hybrid); use confidence
  only as a pre-filter to reduce the number of poses sent to Vina/AD4.
- If confidence model and hybrid scorer disagree by > 30% on ranking, defer
  to the hybrid scorer (which has direct physical interpretation).

### 6.3 · Tier 3.3 — ESM2 adapter fine-tuning (low priority, large risk)

**Goal:** Specialise the ESM2 → RAPiDock projection for peptide-binding
contexts.

**Conditional on:**

- Identifying the ESM adapter layers in RAPiDock source
- 12 GB VRAM is sufficient for ESM2-650M + RAPiDock + activations
  (borderline; may require gradient checkpointing)

**Why this is last priority:**

- ESM2 embeddings change rarely affect docking accuracy at the magnitude that
  matters for HybriDock-Pep's small calibration set
- High risk of breaking inference reproducibility if adapter weights change
- The Tier 0.1 + Tier 1.1 fine-tunes already train layers downstream of ESM,
  which absorbs much of the same signal

**Recommend deferring until after iGEM submission unless time is plentiful.**

---

## 7. Cross-cutting safeguards

### 7.1 Data leakage prevention (CRITICAL)

PepSet is the held-out test set. **It must never appear in:**

- RAPiDock fine-tuning data (verified by `_load_pepset_ids` dedup in
  fetch_pdb_complexes.py)
- BindingDB calibration set (verified in Tier 0.3 validation gate)
- XGBoost training data (must explicitly exclude PepSet PDB IDs)
- Family classifier training (motifs are static, no risk; but verify)

**Validation test:** Add `tests/test_no_pepset_leakage.py`:

```python
def test_pepset_not_in_training():
    pepset = load_pepset_ids()  # from datasets/pepset/
    for source_csv in [
        "data/training_complexes.csv",
        "data/training_complexes_expanded.csv",
    ]:
        ids = set(pd.read_csv(source_csv)["pdb_id"].str.upper())
        leak = ids & pepset
        assert not leak, f"PEPSET LEAK in {source_csv}: {leak}"
```

Run this test after every Tier 0/1 step.

### 7.2 Reproducibility envelope

For every training run, log to `runs/<run_id>/REPRODUCIBILITY.json`:

```json
{
  "git_sha": "...",
  "git_dirty": false,
  "branch": "master",
  "seed": 42,
  "conda_env": "rapidock",
  "python_version": "3.10.13",
  "pytorch_version": "2.7.0+cu128",
  "rapidock_commit": "...",
  "gpu_name": "NVIDIA GeForce RTX 5070",
  "cuda_version": "12.8",
  "driver_version": "595.79",
  "hostname": "...",
  "started_at": "2026-05-26T09:00:00+02:00",
  "finished_at": "2026-05-26T12:03:00+02:00",
  "hyperparameters": {...}
}
```

This file is committed even when the model checkpoint is gitignored.

### 7.3 Monitoring during long runs

- **Every 30 min during training:** check `tail` of log + `nvidia-smi`
- **Set a phone alarm** if leaving overnight; verify training is still alive
  the next morning before assuming success
- **Disk space:** monitor `df -h` if writing many checkpoints; clean up
  intermediate `.pt` files after the best one is identified

### 7.4 Branch hygiene per training run

Each major training run should happen on a topic branch:

```bash
git checkout -b training/tier-0.1-finetune-925
# ... work, commit ...
git checkout master
git merge --ff-only training/tier-0.1-finetune-925
git branch -d training/tier-0.1-finetune-925
```

Rationale: if a training run produces a regressing checkpoint, the topic
branch can be retained for forensic analysis without polluting master.

### 7.5 Calibration JSON versioning

Never overwrite `data/calibration.json` in place. Always:

```bash
# Save versioned copy
cp data/calibration.json data/calibration_v$(date +%Y%m%d).json
# Then update default
cp data/calibration_new.json data/calibration.json
```

Versioned files commit together; `git log data/calibration.json` shows the
full history of accuracy improvements.

### 7.6 Test suite as continuous validation

After every commit on the training/master branch:

```bash
pytest -x -q -m "not slow"      # < 1 min, must pass
# weekly:
pytest -q -m slow                # ~10 min, must pass
```

If any test regresses: do not advance to next Tier step. Fix or revert.

### 7.7 GPU thermal safety

RTX 5070 has known thermal throttling characteristics under sustained load.
For overnight training runs:

```bash
# In a background terminal during training:
nvidia-smi dmon -s u -c 720 -d 30 > runs/gpu_thermal_$(date +%Y%m%d).log &
# Logs utilisation + temperature every 30s for 6 hours
```

If GPU temp sustains > 85°C: check case airflow; consider lowering
`batch_size` (linear thermal reduction) or limiting power with
`nvidia-smi -pl 200`.

### 7.8 Power and uptime

If a long run loses power mid-training: PyTorch should have saved checkpoint
every N epochs. Verify with:

```bash
ls -la third_party/RAPiDock_finetuned/finetune_out/*.pt
# Expected: epoch_N.pt files at regular intervals
```

If only `rapidock_finetuned_best.pt` exists: training did not save
intermediate states; consider modifying train_lastlayer.py to checkpoint
every 10 epochs.

For UPS-less setups: split long runs into multiple shorter runs with
`--resume-from <checkpoint>` between them.

---

## 8. Decision trees

### 8.1 After Tier 0.4 production recalibration

```
α moved to (0.3, 1.2)?
├── YES → Calibration fix worked; proceed to Tier 1.1
└── NO → α still at lower bound
    ├── Production poses are still systematically over-scored?
    │   ├── YES → likely apo receptor still wrong (frame / prep);
    │   │        investigate per-complex Vina scores; defer Tier 1.1
    │   └── NO  → Sample size too small; jump to Tier 1.3
    └── α at upper bound (1.5)?
        └── Calibration set has under-prediction bias;
            check for outliers; verify experimental pKd source
```

### 8.2 After Tier 1.2 PepSet population r

```
Population r ≥ 0.7?
├── YES → Strong baseline; Tier 2 should push above 0.75
└── 0.5–0.7
    ├── Per-family breakdown shows 2+ families with r < 0.3?
    │   └── YES → Tier 2.1 (family β) is highest priority
    └── Per-family relatively uniform?
        └── Tier 2.3 (XGBoost) likely needed
└── < 0.5 → STOP
    └── Likely calibration is fundamentally wrong;
        re-examine Tier 0.4 and Tier 1.3 outputs;
        do not proceed to Tier 2/3 until root cause found
```

### 8.3 After Tier 2 work

```
Combined improvement vs Tier 1.3 baseline ≥ +0.05 r?
├── YES → Document for iGEM; consider Tier 3.1 (phospho) for
│         family coverage expansion
└── NO  → Limited gain from non-linear corrections
    ├── Tier 3.1 phospho remains worthwhile (extends coverage,
    │   even if doesn't move r)
    └── Tier 3.2/3.3 lower priority; consider only if time allows
```

---

## 9. Rollback procedures

### 9.1 Per-tier rollback

| Tier | What to revert | How |
|------|----------------|-----|
| 0.1 fine-tune | `rapidock_finetuned_best.pt` | rename to `.REJECTED.pt` |
| 0.2 PDB fetch | `datasets/pdb_2024_2026/` | `rm -rf datasets/pdb_2024_2026/` |
| 0.3 BindingDB | `data/training_complexes_expanded.csv` | restore from backup |
| 0.4 production recalibration | `data/calibration_production.json` | leave alone; don't promote |
| 1.1 second fine-tune | finetuned checkpoint | revert to Tier 0.1 checkpoint |
| 1.2 PepSet scoring | none (read-only evaluation) | n/a |
| 1.3 expanded calibration | `data/calibration.json` | restore from `data/calibration_legacy_6complex.json` |
| 2.1 family β | code commit | `git revert <sha>`; calibration JSON rolls back |
| 2.2 CHPi | code commit | `git revert <sha>`; calibration JSON rolls back |
| 2.3 XGBoost | code + model | `git revert <sha>`; rm model file |
| 3.x | per spec | per spec |

### 9.2 Full restore to pre-Tuesday state

```bash
git checkout pre-tier0-YYYYMMDD
# Restore data files from backup
cp data/_backups/YYYYMMDD/* data/
# Remove generated artifacts
rm -rf runs/calibration_production/ runs/pepset_population/ \
       runs/calibration_expanded/ runs/finetune_log_*.txt
# Restore RAPiDock checkpoint
mv third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_best.pt \
   third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_best.REJECTED.pt
```

This nukes a week of work. Use only as absolute last resort. Generally
prefer per-tier rollback.

---

## 10. Validation gates summary

A single table for quick reference:

| Step | Quantitative gate | Action on failure |
|------|------------------|-------------------|
| Pre-flight | GPU detected, envs intact, smoke test pass, tests green | Stop and debug |
| Tier 0.1 | Overall RMSD ↓; PDZ/BRD RMSD ↑ by ≤ 0.05 Å | Reject checkpoint; revert |
| Tier 0.2 | ≥800 recent + ≥150 PPII; no PepSet leak | Re-run with fixed dedup |
| Tier 0.3 | ≥100 BindingDB rows; pKd in [3, 12]; no PepSet leak | Investigate SMILES parsing |
| Tier 0.4 | α in (0.3, 1.2); RMSE ≤ 2.0 | Investigate per-complex |
| Tier 1.1 | RMSD ≤ Tier 0.1 overall; guard rails hold | Revert to Tier 0.1 checkpoint |
| Tier 1.2 | Population r ≥ 0.5 (warning < 0.7) | Stop and investigate |
| Tier 1.3 | n ≥ 100; r ≥ 0.5; α/β in valid bounds | Keep Tier 0.4 calibration |
| Tier 2.1 | Per-family r ≥ uniform baseline (5 of 7 families) | Use uniform fallback |
| Tier 2.2 | SH3/WW improvement; PDZ/BRD/MDM2/BCL-2 unchanged | Tighten geometry |
| Tier 2.3 | 5-fold CV r ≥ linear + 0.05; latency ≤ 50 ms | Keep linear default |
| Tier 3.1 | SHP2 4JMG AD4 ≤ −7 kcal/mol with phospho on | Re-derive charge templates |
| Tier 3.2 | AUC ≥ 0.70; top-1 accuracy gain ≥ 5% | Disable confidence pre-filter |
| Tier 3.3 | RAPiDock inference reproducible vs baseline | Revert adapter |

---

## 11. iGEM submission considerations

iGEM Best Software Tool judging emphasises:

1. **Reproducibility** — every claim must be reproducible by anyone running
   the same scripts on the same data.
2. **Honesty** — accuracy claims must be on properly held-out test sets, not
   training data.
3. **Documentation** — failure modes acknowledged, not hidden.
4. **Transparency about model components** — ML model cards, hyperparameter
   choices, data sources cited.

This plan supports those by:

- Versioning every calibration JSON (no silent overwrites)
- Recording reproducibility envelopes for every training run
- Splitting training/test rigorously (PepSet held out)
- Documenting all failure modes per Tier
- Model cards required for Tier 2.3 and any Tier 3 ML components
- Per-family accuracy reporting (not just aggregate r)
- Explicit calibration of every coefficient (no magic numbers)

**Required submission artifacts** (build incrementally as Tiers complete):

| Artifact | When created | Where |
|----------|--------------|-------|
| Full PepSet accuracy report | After Tier 1.2 | `runs/pepset_population/REPORT.md` |
| Per-family breakdown table | After Tier 2.1 | `runs/family_breakdown/REPORT.md` |
| XGBoost model card | After Tier 2.3 | `models/xgb_scorer_v1_model_card.md` |
| Phospho validation report | After Tier 3.1 | `runs/phospho_validation/REPORT.md` |
| End-to-end reproducibility log | Throughout | `runs/REPRODUCIBILITY_LOG.md` |
| Final benchmark vs RAPiDock-vanilla | Pre-submission | `runs/final_benchmark/REPORT.md` |

---

## 12. Appendix A — Common issues and resolutions

### A.1 RAPiDock inference issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| All poses have positive Vina | Receptor prep failed silently | Inspect receptor.pdbqt; check pdbfixer output |
| Pose shortfall (50 of 100) | RAPiDock subprocess died mid-batch | Re-run with `-vv`; check rapidock_runner stderr |
| Identical poses across seeds | Seeding not propagating | Verify `_seed_everything` called before any RAPiDock import |
| `CUDA OOM` during inference | Batch size too large | Lower `--batch-size` in run_rapidock.py |
| RAPiDock module import fails | Path on sys.path wrong | Verify RAPIDOCK_DIR env var; check `--rapidock-dir` |

### A.2 Calibration issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| α pinned at lower bound | Crystal-pose over-scoring | Switch to production poses (Tier 0.4) |
| β = 0 strongly preferred | Mixed-family training set | Family-adaptive β (Tier 2.1) |
| Pearson r drops on retrained set | Overfitting or bad new data | Compare per-complex residuals before/after |
| Singular matrix in L-BFGS-B | Insufficient variance in features | Add more diverse complexes to training |
| RMSE > 3 kcal/mol | Systematic bias not captured | Investigate per-family; consider non-linear scorer |

### A.3 Conda environment issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `python` in rapidock env = Python 2.7 | ADFRsuite PATH shadowing | Use absolute path `~/miniconda3/envs/rapidock/bin/python` |
| `rcsbsearch` not found | Env not updated | `conda env update -n score-env -f envs/score-env.yml` |
| `import torch` returns CPU-only | Wrong PyTorch build | Reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `meeko.PolymerTopology` doesn't exist | Meeko version mismatch | Pin meeko ≥ 0.7.0 in env file |

### A.4 Git issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Pre-commit hook fails on long files | Lint check | Fix the lint issue; **do not use --no-verify** |
| `git push` rejected | Diverged from origin | `git fetch; git rebase origin/master`; force-push only as last resort |
| Large file warning | Trying to commit a checkpoint | Add to `.gitignore`; use Git LFS if truly needed |
| `.gitignore` not respected | Already-tracked file | `git rm --cached <file>` then commit |

---

## 13. Appendix B — Estimated total resource budget

### 13.1 Time budget (active human attention)

| Day | Active hours | Activity |
|-----|-------------:|----------|
| Tuesday | 5 | Tier 0 execution + monitoring |
| Wednesday | 4 | Tier 1 + Tier 2.1 |
| Thursday | 4 | Tier 2.2 + Tier 2.3 |
| Friday | 4 | Tier 3.1 + documentation + iGEM artifacts |
| **Total** | **17 hours** | excluding overnight unattended runs |

### 13.2 GPU time budget (wall clock)

| Task | Hours |
|------|------:|
| Tier 0.1 fine-tune (925 complexes) | 3 |
| Tier 0.4 production recalibration (6 dock runs) | 1 |
| Tier 1.1 second fine-tune (1700 complexes) | 7 |
| Tier 3.2 confidence model fine-tune (if pursued) | 1 |
| Tier 3.3 ESM adapter fine-tune (if pursued) | 2 |
| **Total** | **14 hours** |

GPU is mostly unattended; overnight runs are fine.

### 13.3 CPU/network budget

| Task | Hours |
|------|------:|
| Tier 0.2 fetch PDB complexes | 1 |
| Tier 0.3 BindingDB join | 0.5 |
| Tier 1.2 PepSet population scoring | 2 |
| Tier 1.3 expanded calibration scoring | 2 |
| Tier 2.x feature computation | 2 |
| Test suite runs (continuous) | 2 (cumulative) |
| **Total** | **9.5 hours** |

### 13.4 Disk space

| Item | Size |
|------|-----:|
| `datasets/pdb_2024_2026/structures/` | ~6 GB |
| `datasets/ppii_enriched/structures/` | ~1 GB |
| `datasets/training_expanded_structures/` | ~1.5 GB |
| `datasets/cache/` (BindingDB) | ~10 GB |
| RAPiDock checkpoints (multiple) | ~5 GB |
| `runs/` artifacts | ~5 GB |
| **Total new disk** | **~28 GB** |

Verify `df -h $HOME` shows ≥50 GB free before starting.

---

## 14. Appendix C — File modification index

Files this plan will create or modify, by tier:

| File | Tier | Action |
|------|------|--------|
| `runs/finetune_runs.md` | 0.1 | create |
| `datasets/pdb_2024_2026/manifest.csv` | 0.2 | create |
| `datasets/ppii_enriched/manifest.csv` | 0.2 | create |
| `data/training_complexes_expanded.csv` | 0.3 | modify |
| `data/calibration_production.json` | 0.4 | create |
| `data/training_scores_production.json` | 0.4 | create |
| `scripts/run_production_calibration.sh` | 0.4 | create |
| `scripts/prep_rapidock_training_data.py` | 1.1 | modify (add --include-recent/--include-ppii) |
| `scripts/run_finetune_and_compare.sh` | 1.1 | modify (epochs 50→75) |
| `scripts/score_family_benchmark.py` | 1.2 | modify (add --pepset-dir mode) |
| `runs/pepset_population/REPORT.md` | 1.2 | create |
| `scripts/fetch_pdb_complexes.py` | 1.3 | modify (add --pdb-list mode) |
| `scripts/scores_csv_to_training_json.py` | 1.3 | create |
| `data/calibration.json` | 1.3 | replace (legacy → backup) |
| `data/calibration_expanded.json` | 1.3 | create |
| `src/hybridock_pep/scoring/family_classifier.py` | 2.1 | create |
| `src/hybridock_pep/scoring/entropy.py` | 2.1, 2.2, 2.3 | modify |
| `tests/test_family_classifier.py` | 2.1 | create |
| `tests/test_family_beta_calibration.py` | 2.1 | create |
| `tests/test_scoring_with_family.py` | 2.1 | create |
| `src/hybridock_pep/scoring/chpi.py` | 2.2 | create |
| `tests/test_chpi.py` | 2.2 | create |
| `scripts/train_xgb_scorer.py` | 2.3 | create |
| `scripts/compute_xgb_features.py` | 2.3 | create |
| `models/xgb_scorer_v1.json` | 2.3 | create |
| `models/xgb_scorer_v1_model_card.md` | 2.3 | create |
| `tests/test_xgb_scorer.py` | 2.3 | create |
| `envs/score-env.yml` | 2.3 | modify (add xgboost) |
| `src/hybridock_pep/prep/ligand.py` | 3.1 | modify per data_augmentation_plan §6.1 |
| `src/hybridock_pep/prep/phospho.py` | 3.1 | create per data_augmentation_plan §6.1 |
| `src/hybridock_pep/prep/receptor.py` | 3.1 | modify per data_augmentation_plan §6.1 |
| `tests/test_phospho_residues.py` | 3.1 | create per data_augmentation_plan §6.2 |
| `tests/test_no_pepset_leakage.py` | 7.1 | create |

---

## 15. Closing notes

This plan is the synthesis of three pieces of context:

1. The existing `docs/data_augmentation_plan.md` (which describes Items 1, 2,
   and 6 — fresh PDB complexes, PPII enrichment, phospho residues)
2. The existing fine-tuning scripts (`scripts/run_finetune_and_compare.sh`,
   `scripts/prep_rapidock_training_data.py`)
3. The calibration limitations documented in `docs/calibration_notes.md`
   (α at lower bound, n=6, crystal-pose bias)

Nothing in Tiers 0 and most of 1 is genuinely new; the scripts and specs
already exist. The novelty is in the **ordering, the validation gates, and
the safeguards** — particularly:

- The realisation that the 6-complex calibration must be replaced with a
  ~200-complex one before any model-side improvements are interpretable
- The realisation that PepSet population-level r is a prerequisite for
  family-adaptive β (Tier 2.1) and for an honest iGEM accuracy claim
- The fact that the production-pose recalibration (Tier 0.4) must precede
  the second fine-tune (Tier 1.1), because the new fine-tune's success can
  only be measured against a sensibly calibrated scoring stage

Anything in Tier 3 is genuinely new work and should be deferred if iGEM
deadlines are tight. The minimum viable trajectory for iGEM submission is:

**Tier 0 (Tuesday) → Tier 1 (Wednesday) → Tier 2.1 (Wednesday) → write submission.**

Tier 2.2, 2.3, and Tier 3 are accuracy improvements past what is needed for
a strong Best Software Tool entry. They can be tackled iteratively after
the initial submission, with each entering as a clearly-versioned, tested,
documented improvement.

**The single most important sentence in this document:**
The r=0.860 number is on n=6 training points using crystal poses. It is not
the number iGEM judges should see; the number they should see is the
population-level r from Tier 1.2 + Tier 1.3, on a properly held-out test
set, with the new calibration. Get to that number first. Everything else is
optimisation on top.

---

*Drafted 2026-05-22 by Opus 4.7 (Sonnet 4.6 context).*
*Companion to `docs/data_augmentation_plan.md`. Conformant with project conventions: no Co-Authored-By trailers, Conventional Commits, type hints, ruff/black formatting.*
