# Tuesday RTX Work Guide
## HybriDock-Pep — Manual Working Guide for RTX 5070 Session

**Written:** 2026-05-23 (Claude session, pre-work done)  
**For:** Ram, working on Linux RTX machine Tue 2026-05-26  
**When to use:** When you're out of Claude tokens and need to work independently

---

## What Was Done in This Mac Session (2026-05-23)

Before you read anything else, here's exactly what I did so you know what's ready:

| Task | Status | Notes |
|------|--------|-------|
| BindingDB All Data download | ✅ Complete | `datasets/cache/bindingdb_all.zip` (581 MB, 8.81 GB uncompressed) |
| CIF retry for 269 failed PDBs | ✅ Complete | 252/269 recovered → 646 included (was 394) |
| PPII filter relaxed | ✅ Complete | 29→74 included structures (frac≥0.20, consec≥1) |
| PepSet IDs file | ✅ Created (CORRECTED) | `data/pepset_ids.txt` — **10 true held-out test IDs only** |
| BindingDB URL bug fixed | ✅ Fixed | `scripts/bindingdb_calibration_join.py` fallback URL corrected |
| Dataset validation report | ✅ Written | `datasets/VALIDATION_REPORT.md` |
| Historical PDB downloads (2010–2023) | ✅ Complete | 4,163 structures downloaded |
| Pre-2010 PDB downloads | ✅ Complete | 1,413 structures downloaded |
| Family-targeted downloads (SH3/WW/PDZ/BCL2/MDM2) | ✅ Complete | 1,428 structures |
| PPII extended | ✅ Complete | 27 structures |
| **RCSB bulk affinity query** | ✅ Complete | 2,689 records for 294 PDB IDs → calibration **284 entries** |
| Calibration set built | ✅ Complete | `data/training_complexes_full.csv` (284 rows, pKd 3.2–10.3) |
| All scripts committed | ✅ Complete | See commit `2f82877` |

**Total structures on disk: 8,732 files (1.5 GB)**

**Critical finding (fixed):** `data/pepset_ids.txt` initially had 21 IDs including training
complexes 1A0N and 1YWI. **Fixed** to 10 true held-out test IDs only:
`1EJ4, 1G73, 1PRM, 2FLU, 2VWF, 3DAB, 3EG6, 3EQS, 3EQY, 3TWR`

**BindingDB finding:** The BindingDB join script captured small-molecule inhibitor PDB entries,
not peptide-protein complexes. The 284-entry calibration set was built instead from RCSB bulk
affinity data (GraphQL query across all 6,982 structure IDs in manifests). See §4 for context.

---

## 1. First Things: Transfer Data from Mac to Linux

Before starting GPU work Tuesday, copy the downloaded data to the Linux machine.

```bash
# On the Linux machine, from the iGEMDryLab directory:
# Option A: rsync from Mac (if on same network)
rsync -avz --progress ram@<mac-ip>:~/Work/iGEMDryLab/hybridock-pep/datasets/cache/ \
    ~/Work/iGEMDryLab/hybridock-pep/datasets/cache/

# Option B: copy the zip manually (USB or cloud)
# The file is: hybridock-pep/datasets/cache/bindingdb_all.zip (581 MB)

# Option C: Re-download on Linux (fast link)
mkdir -p datasets/cache
curl -L "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202605_tsv.zip" \
    -o datasets/cache/bindingdb_all.zip
```

---

## 2. Pre-Flight Checklist (Do This First, No Exceptions)

From the accuracy_improvement_plan.md §2. Takes 5 minutes. Saves hours of debugging.

```bash
# 1. GPU alive?
nvidia-smi
# Expected: RTX 5070, driver ≥ 580.x, CUDA 12.8

# 2. PyTorch sees CUDA?
conda run -n rapidock python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('PyTorch:', torch.__version__)
print('Compute cap:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NONE')
"
# Expected: CUDA: True, Device: NVIDIA GeForce RTX 5070, compute (12, 0)

# 3. Conda envs intact?
conda env list | grep -E "rapidock|score-env"
# Expected: both present

# 4. Disk space (need ≥50 GB free)
df -h $HOME | awk 'NR==2 {print "Free:", $4}'
# Expected: ≥50 GB

# 5. Smoke test
conda activate score-env
bash scripts/smoke_test.sh
# Expected: all PASS

# 6. Tests green?
conda run -n score-env pytest -x -q -m "not slow"
# Expected: all pass (172+)

# 7. Git clean?
git status    # Expected: clean
git fetch origin
```

**If anything fails: STOP. Fix it before touching training.**

```bash
# Tag the pre-training state
git tag -a pre-tier0-$(date +%Y%m%d) -m "Pre-training state, Tuesday"
```

---

## 3. Backup State Files (Do Before Any Training)

```bash
mkdir -p data/_backups/$(date +%Y%m%d)
cp data/calibration.json         data/_backups/$(date +%Y%m%d)/
cp data/training_complexes.csv   data/_backups/$(date +%Y%m%d)/
cp data/training_scores.json     data/_backups/$(date +%Y%m%d)/
cp data/test_complexes.csv       data/_backups/$(date +%Y%m%d)/
cp data/test_complexes_meta.csv  data/_backups/$(date +%Y%m%d)/
[ -f data/training_complexes_expanded.csv ] && \
    cp data/training_complexes_expanded.csv data/_backups/$(date +%Y%m%d)/
ls -la data/_backups/$(date +%Y%m%d)/
# Expected: 6 files
```

---

## 4. Fix the BindingDB Join Script (CRITICAL — Do Before Tier 0.3)

**The Problem:**
`training_complexes_expanded.csv` currently has 39/42 empty peptide sequences.
These are small-molecule inhibitor PDB entries (HIV protease, etc.), not peptides.
The BindingDB scan found only ~84 genuine peptide-protein entries with PDB+affinity.

**Check current state:**
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/training_complexes_expanded.csv')
empty = df['peptide_sequence'].isna() | (df['peptide_sequence'] == '')
print(f'Rows: {len(df)}')
print(f'With sequence: {(~empty).sum()}')
print(f'Empty: {empty.sum()}')
print('PDB IDs with sequences:')
print(df[~empty][['pdb_id', 'peptide_sequence', 'experimental_pkd']].to_string())
"
```

**The Fix — run BindingDB join with fixed filtering:**

```bash
# First, check if rdkit is installed in score-env
conda run -n score-env python -c "from rdkit import Chem; print('rdkit ok')"

# If missing, install it:
conda install -n score-env -c conda-forge rdkit -y
# OR: conda run -n score-env pip install rdkit

# Run the fixed join script
conda run --no-capture-output -n score-env \
    python scripts/bindingdb_calibration_join.py \
    --use-ki   # include Ki in addition to Kd for more data
# Expected output: data/training_complexes_expanded.csv with ≥50 rows
# Expected runtime: ~15-25 min (reads 8.81 GB file)
```

**After running, validate:**
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/training_complexes_expanded.csv')
empty = df['peptide_sequence'].isna() | (df['peptide_sequence'] == '')
print(f'Total rows: {len(df)}')
print(f'With sequence: {(~empty).sum()}')
print(f'pKd range: {df[\"experimental_pkd\"].min():.1f} - {df[\"experimental_pkd\"].max():.1f}')
print('Source breakdown:')
print(df['source'].value_counts())
"
```

**Expected**: ≥50 rows, most with sequences, pKd in 3–12 range.

**If still mostly empty after the fix:**  
The BindingDB database genuinely has very few peptide-protein pairs with PDB structures (~84).
In that case, supplement manually:
```bash
# Add known peptide-protein pairs with experimental Kd
# Edit data/training_complexes_expanded.csv directly and add rows with source="manual"
# Good targets with known affinity:
# - MDM2/p53 peptides (1YCR: pKd=6.52 already there)
# - BCL-2/BH3 peptides (e.g., 2YJ1: NAVGIDLB, pKd=8.3)
# - SH2/pTyr peptides (1JYP, etc.)
```

---

## 5. Tier 0.2 — PDB Fetch (Mac Already Did Most of This)

**Current state:** 646 included structures (was 394). 74 PPII structures (was 29).

```bash
# Check what we have
wc -l datasets/pdb_2024_2026/manifest.csv  # should be 1027
python3 -c "
import pandas as pd
df = pd.read_csv('datasets/pdb_2024_2026/manifest.csv')
included = df[df['excluded_reason'].isna() | (df['excluded_reason'] == '')]
print('Included:', len(included))
print('On disk:', len(list(__import__('pathlib').Path('datasets/pdb_2024_2026/structures').glob('*.pdb.gz'))))
"
```

**Run an update fetch if you want more (optional):**
```bash
# Only 17 download failures remain (out of 1026). Retry if you want.
conda run --no-capture-output -n score-env \
    python scripts/fetch_pdb_complexes.py --mode both --max-workers 4
# Idempotent — skips already-downloaded structures
# Expected: ~30-60 min, picks up the 17 remaining failures
```

**Validation gate:**
```bash
# Check no PepSet leakage
conda run -n score-env python -c "
import pandas as pd
pepset = set(open('datasets/pepset/pepset_ids.txt').read().split())
for path in ['datasets/pdb_2024_2026/manifest.csv', 'datasets/ppii_enriched/manifest.csv']:
    df = pd.read_csv(path)
    included = df[df['excluded_reason'].fillna('') == '']
    leak = set(included['pdb_id'].str.upper()) & pepset
    print(f'{path}: {len(leak)} leaks')
    assert not leak, f'PEPSET LEAK: {leak}'
print('OK — no leakage')
"
```

---

## 6. Tier 0.1 — RAPiDock Fine-Tune (Tuesday Morning, GPU)

**Before running, prepare training data:**

```bash
# Step 1: Format for RAPiDock
conda run -n rapidock python scripts/prep_rapidock_training_data.py
# Expected: 5-10 min, creates datasets/training_formatted/
ls datasets/training_formatted/ | wc -l   # Expected: ~925

# Spot-check one entry
ls datasets/training_formatted/$(ls datasets/training_formatted/ | head -1)
# Expected: receptor.pdb, peptide.pdb, esm_embedding.pt
```

**Run fine-tune (foreground so you can watch):**

```bash
# Make sure nothing else is using GPU
ps aux | grep -E "openmm|python" | grep -v grep
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Start the fine-tune
bash scripts/run_finetune_and_compare.sh 2>&1 | tee runs/finetune_log_$(date +%Y%m%d_%H%M).txt
```

**What to watch (in a second terminal):**
```bash
# GPU utilization — should be >80%
watch -n 5 nvidia-smi

# Loss should decrease over epochs
tail -30 runs/finetune_log_*.txt | grep -E "epoch|loss|tr_loss"

# Memory check — if approaching 11 GB, lower batch_size
nvidia-smi | grep -E "MiB|%"
```

**If CUDA OOM:**
```bash
# Edit train_lastlayer.py: batch_size=2 (from 4)
grep -n "batch_size" third_party/RAPiDock_finetuned/train_lastlayer.py
# Then restart
```

**Expected runtime:** ~3 hours on RTX 5070.

**Validation gate (runs automatically at end):**
```bash
tail -20 runs/finetune_log_*.txt | grep -E "SH3|WW|PDZ|BRD|Overall"
# PDZ and BRD must NOT regress by > 0.05 Å
# If they do: mv third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_best.pt \
#                third_party/RAPiDock_finetuned/finetune_out/rapidock_finetuned_best.REJECTED.pt
```

**Log the results:**
```bash
# Write to finetune_runs.md
cat >> runs/finetune_runs.md << 'EOF'
## Run 1 — 2026-05-26

- Dataset: 925 complexes (RefPepDB-RecentSet)
- Epochs: 50
- Batch size: 4
- PPII oversampling: 4×
- Seed: 42
- Hardware: RTX 5070, CUDA 12.8
- Results: [fill from tail of log]
EOF

git add runs/finetune_runs.md
git commit -m "docs(training): log first 925-complex fine-tune results"
```

---

## 7. Tier 0.4 — Production-Pose Recalibration

**Goal:** Replace crystal-pose calibration (α=0.10 at lower bound) with apo-receptor production-pose calibration.

**Check what receptors you need:**
```bash
cat data/training_complexes.csv
# Needs: 2hwn, 1nrl, 1l2z, 1ddv, 1a0n, 1ywi (all in datasets/raw_pdbs/)
ls datasets/raw_pdbs/2HWN.pdb datasets/raw_pdbs/1NRL.pdb datasets/raw_pdbs/1L2Z.pdb \
   datasets/raw_pdbs/1DDV.pdb datasets/raw_pdbs/1A0N.pdb datasets/raw_pdbs/1YWI.pdb
```

**Step 1: Get peptide binding site centers:**
```bash
conda run -n score-env python -c "
from scripts.benchmark import get_peptide_center
from pathlib import Path

complexes = [
    ('2hwn', 'B'),  # peptide_chain from test_complexes_meta.csv
    ('1nrl', 'B'),
    ('1l2z', 'B'),
    ('1ddv', 'B'),
    ('1a0n', 'B'),
    ('1ywi', 'B'),
]
for pdb_id, pep_chain in complexes:
    path = Path(f'datasets/raw_pdbs/{pdb_id.upper()}.pdb')
    if not path.exists():
        path = Path(f'data/pdbs/{pdb_id.upper()}.pdb')
    center = get_peptide_center(path, pep_chain)
    print(f'{pdb_id}: {center}')
"
```

**Step 2: Run production docking for each:**
```bash
mkdir -p runs/calibration_production
# For each of the 6 complexes:
PDB=2hwn
PEPTIDE=EELAWKIAKMIVSDVMQQC    # from training_complexes.csv
# Get site coords from Step 1 above
SITE="X Y Z"   # fill from Step 1 output
RECEPTOR=datasets/raw_pdbs/2HWN.pdb

hybridock-pep dock \
    --peptide "$PEPTIDE" \
    --receptor "$RECEPTOR" \
    --site $SITE \
    --box 40 \
    --n-samples 100 \
    --seed 42 \
    --scoring vina,ad4 \
    --output-dir runs/calibration_production/${PDB}

# Repeat for 1nrl, 1l2z, 1ddv, 1a0n, 1ywi
# Expected: ~8 min per complex on RTX 5070 → ~50 min total
```

**Step 3: Collect scores and recalibrate:**
```bash
python3 -c "
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
print(json.dumps(out, indent=2))
"

# Recalibrate
hybridock-pep calibrate \
    --training-csv data/training_complexes.csv \
    --scores-json data/training_scores_production.json \
    --output data/calibration_production.json

# Check the result
python3 -c "
import json
old = json.load(open('data/calibration.json'))
new = json.load(open('data/calibration_production.json'))
print(f'BEFORE: alpha={old[\"alpha\"]:.3f}, beta={old[\"beta\"]:.3f}, r={old[\"pearson_r\"]:.3f}')
print(f'AFTER:  alpha={new[\"alpha\"]:.3f}, beta={new[\"beta\"]:.3f}, r={new[\"pearson_r\"]:.3f}')
"
```

**Validation gate:**
- α should move from 0.10 to **0.3–0.9** (proof the fix worked)
- If α is still ≤ 0.10 after this: small-sample-size problem, wait for Tier 1.3
- If α is at upper bound (≥ 1.5): overcorrection, check per-complex scores

**DO NOT** overwrite `data/calibration.json` yet. Keep production calibration in its own file
until Tier 1.3 produces the 200-complex version.

---

## 8. Tier 1.3 — Score 284-Entry Calibration Set (Tuesday, CPU, ~2 hrs)

**This replaces Tier 0.3 (BindingDB join was superseded by RCSB bulk affinity).**
`data/training_complexes_full.csv` already has 284 entries ready. You just need to
score them on the Linux machine (where Vina+AD4 are installed) and recalibrate.

**⚠️ Critical prerequisite: transfer the dataset directories from Mac to Linux first (§1)**

```bash
conda activate score-env

# Score all 284 calibration entries with Vina + AD4
# --workers 8 runs 8 complexes in parallel (all CPU, safe alongside GPU fine-tune)
conda run --no-capture-output -n score-env python scripts/score_calibration_set.py \
    --training-csv data/training_complexes_full.csv \
    --output-csv runs/calibration_full/scores.csv \
    --output-json data/training_scores_full.json \
    --workers 8 \
    --verbose
# Expected: ~1-2 hrs at 8 workers
# Checkpoint-safe: if interrupted, re-run the same command to resume
```

**Monitor progress:**
```bash
# Watch the checkpoint CSV grow
watch -n 30 "wc -l runs/calibration_full/scores.csv"
# Should grow from 1 (header only) to 285 (284 entries + header)
```

**After scoring completes, recalibrate:**
```bash
conda run --no-capture-output -n score-env python scripts/calibrate_alpha.py \
    --training-csv data/training_complexes_full.csv \
    --scores-json data/training_scores_full.json \
    --output data/calibration_full.json \
    --verbose

# Check result
python3 -c "
import json
c = json.load(open('data/calibration_full.json'))
print('α =', c['alpha'], '(should be 0.12-1.0)')
print('β =', c.get('beta', 0.0))
print('r =', c.get('pearson_r', 'N/A'), '(target ≥ 0.5)')
print('n =', c.get('n_complexes', '?'))
assert 0.10 < c['alpha'] < 1.5, f'α out of range: {c[\"alpha\"]}'
assert c.get('pearson_r', 0) > 0.4, f'r too low: {c.get(\"pearson_r\", 0)}'
print('CALIBRATION VALID')
"
```

**If you want Kd/Ki-only calibration (higher quality, fewer entries):**
```bash
conda run --no-capture-output -n score-env python scripts/score_calibration_set.py \
    --training-csv data/training_complexes_full.csv \
    --output-csv runs/calibration_kdki/scores.csv \
    --output-json data/training_scores_kdki.json \
    --affinity-types Kd Ki \
    --workers 8
# Expected: ~122 entries (67 Kd + 55 Ki)
```

**Promote new calibration:**
```bash
cp data/calibration.json data/calibration_legacy_6complex.json
cp data/calibration_full.json data/calibration.json
# Or for Kd/Ki only: cp data/calibration_kdki.json data/calibration.json
```

**Validation gate (no regression):**
```bash
# Verify no PepSet leak
python3 -c "
import pandas as pd
df = pd.read_csv('data/training_complexes_full.csv')
pepset = set(line.strip().upper() for line in open('data/pepset_ids.txt') if line.strip())
leak = set(df['pdb_id'].str.upper()) & pepset
assert not leak, f'PEPSET LEAK: {leak}'
print('PepSet check: OK (no leakage)')
print(f'Calibration entries: {len(df)}, pKd: {df[\"experimental_pkd\"].min():.1f}–{df[\"experimental_pkd\"].max():.1f}')
"
```

---

## 9. Tier 1.1 — Second Fine-Tune on Expanded Dataset (Tuesday Night, Overnight)

After Tier 0.1 completes and Tier 0.2 data is ready:

```bash
# Edit epochs to 75 (more data needs more epochs)
sed -i 's/--epochs 50/--epochs 75/' scripts/run_finetune_and_compare.sh
# Or manually edit the file

# Run overnight (nohup so it survives terminal close)
nohup bash scripts/run_finetune_and_compare.sh > runs/finetune2_log.txt 2>&1 &
echo $! > runs/finetune2.pid

# Check first epoch completes before going to bed
sleep 300 && tail -20 runs/finetune2_log.txt
# Should see "Epoch 1/75" with non-NaN loss

# GPU thermal monitor (let it run overnight)
nvidia-smi dmon -s u -c 720 -d 30 > runs/gpu_thermal_$(date +%Y%m%d).log &
```

**Expected runtime:** ~6-8 hours for the expanded dataset.

**Wednesday morning check:**
```bash
# Still running?
ps aux | grep finetune | grep -v grep
# Check loss decreased
grep -E "epoch|loss" runs/finetune2_log.txt | tail -30
# Check Cα RMSD comparison (end of log)
tail -30 runs/finetune2_log.txt
```

---

## 10. Tier 1.2 — Full Test Set Pearson r (Wednesday, CPU)

This is the number that matters for iGEM. Uses the 10 held-out test complexes.

```bash
# Run the full benchmark on test complexes
conda run --no-capture-output -n score-env python scripts/benchmark.py \
    --test-csv data/test_complexes.csv \
    --output-dir runs/pepset_population/ \
    --seed 42

# Check results
cat runs/pepset_population/benchmark_report.md
```

**What to look for:**
```bash
# Pearson r from benchmark output
# Target: ≥ 0.55 (already likely better with new calibration)
# On n=10 complexes, the error bars are ±0.15, so don't panic about
# small drops vs the training-set r=0.86
```

**Compute confidence intervals:**
```bash
conda run -n score-env python -c "
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import t
import numpy as np

df = pd.read_csv('runs/pepset_population/benchmark_results.csv')
r, p = pearsonr(df['hybrid_score'], df['experimental_pkd'])
n = len(df)
# 95% CI via Fisher z-transform
z = np.arctanh(r)
se = 1/np.sqrt(n-3)
lo, hi = np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)
print(f'Pearson r = {r:.3f} (95% CI: {lo:.3f}-{hi:.3f}), n={n}, p={p:.3f}')
"
```

---

## 11. Quick Data Commands (Reference Sheet)

### Check what's downloaded
```bash
# pdb_2024_2026 coverage
python3 -c "
import pandas as pd
df = pd.read_csv('datasets/pdb_2024_2026/manifest.csv')
inc = df[df['excluded_reason'].fillna('') == '']
print(f'pdb_2024_2026: {len(inc)} included of {len(df)} total')
print('Files on disk:', len(list(__import__('pathlib').Path('datasets/pdb_2024_2026/structures').glob('*.pdb.gz'))))
"

# ppii_enriched coverage
python3 -c "
import pandas as pd
df = pd.read_csv('datasets/ppii_enriched/manifest.csv')
inc = df[df['excluded_reason'].fillna('') == '']
print(f'ppii_enriched: {len(inc)} included of {len(df)} total')
print('mean PPII frac:', inc['ppii_fraction'].mean() if 'ppii_fraction' in inc.columns else 'N/A')
"
```

### Inspect a structure file
```bash
# Look at a downloaded pdb.gz file
PDB=7GUS  # any ID
python3 -c "
import gzip
from pathlib import Path
f = next(Path('datasets/pdb_2024_2026/structures').glob(f'${PDB}*'), None)
if f:
    print('File:', f, f.stat().st_size/1024, 'KB')
    with gzip.open(f) as g:
        text = g.read().decode('latin-1')
    print('ATOM lines:', text.count('ATOM'))
    print('HETATM lines:', text.count('HETATM'))
    print('Chains:', set(l[21] for l in text.split('\n') if l.startswith('ATOM')))
else:
    print('Not found')
"
```

### Spot-check a PDB in PyMOL or Chimera
```bash
# Extract a .pdb.gz for visual inspection
gzip -dc datasets/pdb_2024_2026/structures/7GUS.pdb.gz > /tmp/7GUS.pdb
open /tmp/7GUS.pdb  # Mac: opens in default viewer
# On Linux: pymol /tmp/7GUS.pdb
```

### Check calibration state
```bash
python3 -c "
import json
c = json.load(open('data/calibration.json'))
print('Current calibration:')
for k, v in c.items():
    if not isinstance(v, list):
        print(f'  {k}: {v}')
if c.get('alpha', 1) <= 0.11:
    print('⚠️ ALERT: alpha at lower bound — calibration is invalid')
"
```

### Check training complexes
```bash
cat data/training_complexes.csv
# Expected: 6 rows (2hwn, 1nrl, 1l2z, 1ddv, 1a0n, 1ywi)
```

### Check test complexes
```bash
cat data/test_complexes.csv
# Expected: 10 rows — the held-out benchmark set
# These PDB files are in datasets/raw_pdbs/
```

### PPII filter summary
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('datasets/ppii_enriched/manifest.csv')
inc = df[df['excluded_reason'].fillna('') == '']
print('Included PPII structures:', len(inc))
print()
print('Families:')
print(inc['family_hint'].value_counts().head(10))
print()
print('PPII fraction range:', inc['ppii_fraction'].min(), '-', inc['ppii_fraction'].max())
print()
print('Sample sequences:')
for _, r in inc.head(5).iterrows():
    print(f\"  {r['pdb_id']}: {r['peptide_seq']} (frac={r['ppii_fraction']:.2f})\")
"
```

---

## 12. Failure Mode Reference

### "CUDA OOM" during fine-tune
```bash
# Lower batch_size in the training script
grep -n "batch_size" third_party/RAPiDock_finetuned/train_lastlayer.py
# Change batch_size=4 → batch_size=2
# Restart training from scratch (or from last checkpoint if one was saved)
```

### Loss is NaN after epoch 1
```bash
# Lower learning rate
grep -n "learning_rate\|lr=" third_party/RAPiDock_finetuned/train_lastlayer.py
# Change lr=1e-4 → lr=5e-5
# Restart training
```

### Fine-tune checkpoint not improving PDZ/BRD
```bash
# The PPII oversampling (4×) is too aggressive
# Edit run_finetune_and_compare.sh: --ppii-weight 4 → --ppii-weight 2
# Retrain
```

### α still at 0.10 after production recalibration
```bash
# This means even apo-receptor production poses are still being over-scored
# Check individual dock runs:
for pdb in runs/calibration_production/*/; do
    echo "=== $pdb ==="
    head -3 ${pdb}/ranked_poses.csv 2>/dev/null
done
# If all Vina scores are strongly negative (< -12): might be using holo receptor
# Check the receptor path used in each dock run
```

### Git push rejected
```bash
git fetch && git rebase origin/master
# DO NOT force push unless absolutely certain
```

### RAPiDock checkpoint not found
```bash
ls -la third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/rapidock_local.pt
# If missing: the checkpoint path has drifted
find third_party/ -name "*.pt" | head -10
```

### score-env broken
```bash
# Rebuild from yaml
conda env remove -n score-env
conda env create -f envs/score-env.yml
conda activate score-env
bash scripts/smoke_test.sh
```

---

## 13. Dataset State Summary (Mac-side, 2026-05-23) — FINAL

| Dataset | On Disk | In Manifest | Notes |
|---------|---------|------------|-------|
| `pdb_2024_2026/structures` | 1,009 files | 1,026 | 2 still-failing (very recent) |
| `ppii_enriched/structures` | 324 files | 337 | Filter relaxed 29→74 included |
| `ppii_extended/structures` | 27 files | 27 | PP-motif all-time, new stream |
| `pdb_2019_2023/structures` | 1,717 files | 1,769 | 52 failed download |
| `pdb_2010_2018/structures` | **2,746 files** | 2,748 | 2 failed |
| `pdb_pre2010/structures` | **1,413 files** | 1,413 | 100% success |
| `family_targeted/structures` | 1,428 files | 1,444 | SH3/WW/PDZ/BCL2/MDM2 motifs |
| `raw_pdbs/` | 30 files | - | 6 train + 10 test + others |
| `cache/bindingdb_all.zip` | 581 MB | - | 8.81 GB uncompressed |
| **Total** | **8,732 files** | **8,764** | **1.5 GB** |

**Calibration set:**
- `data/training_complexes_full.csv`: **284 rows**, pKd 3.2–10.3
- Sources: rcsb_bulk(262), rcsb(13), manual(8), bindingdb_kd(1)
- Affinity types: IC50(127), Kd(67), Ki(55), EC50(35)
- All verified: peptide chain extracted from PDB file, no PepSet leakage

**Net improvements this session:**
- Structure files on disk: 30 → 8,732 (+8,702 across all datasets)
- Calibration entries: 6 → 284 (+278 from RCSB bulk affinity query)
- Calibration set now EXCEEDS the 200-complex plan target

---

## 14. What the Plan's Numbers Actually Mean Now

The `docs/accuracy_improvement_plan.md` was written expecting:
- ≥800 recent PDB complexes → **we have 646** (close enough, continue)
- ≥150 PPII complexes → **we have 74** (better than 29, still below target)
- 200 calibration rows → **we have 284** ✅ TARGET EXCEEDED
- 185 PepSet complexes → **we have 10** (the full RefPepDB set is not yet built)

**Revised expectations for the plan's accuracy trajectory:**

| Checkpoint | Expected Pearson r | Realistic r (n=10) | Notes |
|------------|-------------------|-------------------|-------|
| Current (n=6, crystal) | 0.860 | 0.860 | Not transferable |
| After Tier 0.4 (apo, n=6) | 0.65–0.75 | same | Honest collapse |
| After Tier 1.3 (n=284 calibration) | 0.72–0.82 | 0.60–0.78 | **284 entries now available** |
| + Tier 2.1 (family β) | +0.03–0.05 | same | Still useful |
| Population r, n=10 test set | 0.60–0.80 | 0.60–0.80 | Large CI, ±0.12 |

**Note on calibration quality:** 262/284 entries use IC50 or EC50 (less reliable than Kd/Ki).
The 122 Kd+Ki entries are the gold standard. Production calibration should weight Kd/Ki
entries 2× relative to IC50/EC50.

**Bottom line:** The key iGEM deliverable is an honest Pearson r on 10 held-out test complexes.
Even with n=10, r ≥ 0.60 with honest methodology is defensible for iGEM.
The "Tier 1.2 population r" from the plan = our benchmark.py output.

---

## 15. Commit Guide

After each major step, commit with Conventional Commits format:

```bash
# After Tier 0.1 (fine-tune)
git add runs/finetune_runs.md
git commit -m "docs(training): log first 925-complex fine-tune results

Records seed, hyperparameters, Cα RMSD per family vs baseline.
SH3: X.XX→Y.YY, WW: X.XX→Y.YY, PDZ guard rail: held."

# After Tier 0.4 (production calibration)
git add data/training_scores_production.json data/calibration_production.json
git commit -m "feat(calibration): production-pose recalibration on apo receptors

α: 0.100 → X.XXX. Pearson r on training set: Y.YYY.
Replaces crystal-pose calibration (Issue 1 from calibration_notes.md)."

# After BindingDB join
git add data/training_complexes_expanded.csv
git commit -m "data(calibration): BindingDB-expanded training complexes

N rows from BindingDB All Data 202605. pKd range: X.X–Y.Y.
PepSet excluded. Source: bindingdb_kd N rows, bindingdb_ki N rows."
```

---

## 16. Files Created/Modified in This Mac Session

| File | What changed |
|------|-------------|
| `scripts/retry_failed_as_cif.py` | New — downloads failed PDBs as CIF, converts to PDB.gz |
| `scripts/relax_ppii_filter.py` | New — relaxes PPII filter threshold in manifest |
| `scripts/fetch_expanded_sequences.py` | New — fetches peptide sequences from RCSB |
| `scripts/validate_all_datasets.py` | New — comprehensive validation script |
| `scripts/fetch_all_extended.py` | New — 4-stream fetcher: historical periods + family motifs + affinity + PPII |
| `scripts/fetch_affinity_supplement.py` | New — PDBe + ChEMBL + REMARK affinity fetcher |
| `scripts/fetch_rcsb_affinity_bulk.py` | New — RCSB GraphQL bulk affinity for all 6982 structure IDs |
| `scripts/build_calibration_from_affinity.py` | New — builds calibration CSV from affinity + structure files |
| `datasets/ppii_enriched/manifest.csv` | Modified — 29→74 included (relaxed filter) |
| `datasets/pdb_2024_2026/manifest.csv` | Modified — 394→646 included (CIF retry) |
| `datasets/pdb_2019_2023/` | New — 1717 structures downloaded |
| `datasets/pdb_2010_2018/` | New — 2746 structures downloaded |
| `datasets/pdb_pre2010/` | New — 1413 structures downloaded |
| `datasets/family_targeted/` | New — 1428 structures (SH3/WW/PDZ/BCL2/MDM2 motifs) |
| `datasets/ppii_extended/` | New — 27 structures (PP-motif all-time, deduped vs ppii_enriched) |
| `datasets/cache/bindingdb_all.zip` | New — 581 MB BindingDB All Data 202605 |
| `data/pepset_ids.txt` | Fixed — 10 true held-out test IDs only (was 21 including training data) |
| `data/rcsb_binding_affinity.csv` | New — 36 affinity records from RCSB |
| `data/rcsb_binding_affinity_bulk.csv` | New — 2689 affinity records for 294 PDB IDs |
| `data/training_complexes_full.csv` | New — **284 calibration entries** with sequences and pKd |
| `datasets/training_expanded_structures/` | New dir — ~40 structure files from BindingDB entries |
| `datasets/VALIDATION_REPORT.md` | New — comprehensive validation report |
| `scripts/bindingdb_calibration_join.py` | Fixed FALLBACK_URL (was /bind/, now /rwd/bind/) |
| `TUESDAY_WORKGUIDE.md` | This file |

**Key commits this session:**
- `d7ab323` feat(data): pre-training data acquisition and validation scripts
- `2f82877` feat(calibration): bulk affinity data acquisition and calibration set builder
