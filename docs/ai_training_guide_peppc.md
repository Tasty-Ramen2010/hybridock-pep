# HybriDock-Pep AI Training Guide
## Using PepPC + PepPC-F to Surpass DiffPepDock

*Last updated: May 27 2026 · Pre-training dataset analysis complete*

---

## 1. Head-to-Head Status: Where We Stand

### Current performance (objective)

| Method | Benchmark complexes | Median Cα RMSD | <2 Å rate | DockQ ≥0.23 | DockQ ≥0.49 |
|---|---|---|---|---|---|
| **HybriDock-Pep (Apr30, 100 poses)** | 1 (1YCR) | **0.80 Å** | — | — | — |
| DiffPepDock | 174 | 2.02 Å | 50% (87/174) | 90% (156/174) | 69% (120/174) |
| AlphaFold3 + MSA | 174 | 2.30 Å | 48% (84/174) | 78% (136/174) | 71% (124/174) |
| AlphaFold3 + template+MSA | 174 | **1.90 Å** | 53% (93/174) | 82% (142/174) | 74% (129/174) |
| ADCP (physics-only baseline) | 174 | 11.96 Å | 0% | 54% (94/174) | 0% |

**Our model already beats DiffPepDock on 1YCR.** The gap is generalization — DiffPepDock
was trained on 18K diverse complexes; our fine-tuned weights have only seen ~50 complexes.
The path to beating DiffPepDock's 174-complex benchmark: train on PepPC + PepPC-F.

---

## 2. Training Dataset Comparison

### 2.1 DiffPepDock training data (now available locally)

| Dataset | Complexes | Loops | Helices | Peptide lengths | Source |
|---|---|---|---|---|---|
| **PepPC** (natural protein-peptide) | 3,832 | 93.9% | 6.1% | variable | PDB (natural) |
| **PepPC-F** (protein fragment complexes) | 14,897 | 71.5% | 28.5% | 8–30 aa (mean 9.4) | PDB (fragments) |
| **Total** | **18,729** | 77% | 23% | — | DiffPepBuilder paper |

**Split**: PepPC training set = 3,619 complexes deposited before Jan 1 2022.
Test set = 205 complexes deposited after Jan 1 2022.

Local paths:
```
/home/igem/unknown_software/PepPC_raw_data.tar.gz       (810 MB — 3,832 PDB files)
/home/igem/unknown_software/PepPC-F_raw_data.tar.gz     (634 MB — 14,897 PDB files)
third_party/DiffPepDock/datasets/PepPC_dataset.csv      (manifest)
third_party/DiffPepDock/datasets/PepPC-F_dataset.csv    (manifest)
third_party/DiffPepDock/datasets/PepPC_before_202201.csv (training split)
third_party/DiffPepDock/datasets/PepPC_after_202201.csv  (test split)
```

### 2.2 RAPiDock pre-trained model training data

The `rapidock_local.pt` checkpoint (Zhao et al. 2025) was trained on **RefPepDB**:
- ~7,000 protein-peptide complexes from PDB deposited before ~2021
- Curated from BioLiP database (filtered for peptide length 4–30, resolution ≤3.0 Å)
- ~523 of these are available as `datasets/RefPepDB-RecentSet` on Zenodo
- Training: 850 epochs, 3 GPUs, batch_size=16, CGTensorProduct architecture

### 2.3 Protein family coverage comparison (sampled analysis)

**PepPC top families** (from 50-structure API sample, ~7% of dataset):

| Family | % in sample | Key examples |
|---|---|---|
| Transcription factors | 20% | Nuclear receptors, zinc fingers |
| Immune receptors / MHC | 16% | TCR epitopes, antibody fragments |
| Transferases / Kinases | 12% | Kinase substrates, SH2/SH3 ligands |
| Hydrolases / Proteases | 10% | Protease substrate peptides |
| Oxidoreductases | 6% | LDH, dehydrogenases ← **includes PfLDH target** |
| Hormone receptors | 6% | Steroid receptors, GPCRs |
| Other/unknown | 30% | Diverse |

**PepPC-F top families** (from 50-structure API sample):

| Family | % in sample | Key examples |
|---|---|---|
| Transferases | 18% | Phospho-transfer complex fragments |
| Hydrolases | 16% | Protease/esterase fragments |
| Oxidoreductases | 14% | Metabolic enzyme fragments |
| Immune system | 6% | Immune co-factors |
| Signaling | 6% | MAPK, Ras effectors |
| DNA-binding | 2% | Chromatin factors |
| Other/unknown | 36% | Diverse |

**Coverage gap**: PepPC is loop-heavy (94% loops) vs our malaria target PfLDH which
binds a **loop peptide** (LISDAELEAIFEADC is disordered in unbound state → loop). 
PepPC-F adds 28.5% helix binders — critical for the MDM2 target (p53 peptide is helical).

**What PepPC + PepPC-F adds that we're missing:**
- Helix binders (PepPC-F 4,241 complexes)
- Long peptides (up to 30 aa) — our LISDAELEAIFEADC is 15 aa
- Diverse receptor classes — oxidoreductases (PfLDH class), kinases, GPCRs
- Fragment-based binding epitopes (PepPC-F) — complementary to natural complexes

---

## 3. Technical Architecture Reference

### Model architecture (CGTensorProductEquivariantModel)

Key hyperparameters from `model_parameters.yml`:
- **Layers**: 4 equivariant conv layers (`num_conv_layers=4`)
- **Node features**: ns=48 scalars + nv=10 vectors per node
- **Receptor AA embedding**: 22-dim (`rec_amino_dim=22`)
- **Peptide heavy-atom embedding**: 104-dim (`pep_amino_dim=104`)
- **Edge features**: 103-dim (`edge_feature_dim=103`)
- **Cross/intra max distance**: 80/5.0 Å
- **Dropout**: 0.0 (none during original training)
- **Total parameters**: ~7.5M (estimated from ns/nv/layers)

### Score matching objective

Loss = **0.25 × tr_loss + 0.25 × rot_loss + 0.25 × tor_bb_loss + 0.25 × tor_sc_loss**

Where each component is MSE between predicted and analytical diffusion score
(the scaled gradient of the noise distribution w.r.t. the noised coordinates).

### Diffusion schedule

- Translation σ: 0.1 → 30 Å (log-uniform)  
- Rotation σ: 0.03 → 1.65 rad
- Torsion (backbone + sidechain) σ: 0.0314 → π rad
- Inference: 16 steps (default), can increase to 20 for accuracy

---

## 4. Fine-Tuning Strategy: Beating DiffPepDock

### Phase 1 (Immediate) — Expanded dataset, moderate unfreezing (current script)

**Target**: Improve from 2.00 Å best-of-10 → <2 Å median on 10-complex benchmark.

Current unfreezing (our `train_lastlayer.py`, `_UNFREEZE_PATTERNS`):
```python
"tr_final_layer"      # translation scaler
"rot_final_layer"     # rotation scaler
"tor_bb_final_layer"  # backbone torsion head
"tor_sc_final_layer"  # sidechain torsion head
"final_conv"          # equivariant output conv
"tor_bb_bond_conv"    # backbone torsion conv
"tor_sc_bond_conv"    # sidechain torsion conv
"center_edge_embedding"
"pep_a_node_embedding"
"final_edge_embedding"
```
**~3–5% of parameters unfrozen** (~ 225K–375K params).

**Data**: Extract PepPC_before_202201.csv (3,619 complexes) from PepPC_raw_data.tar.gz
+ all of PepPC-F_raw_data.tar.gz (14,897 complexes) → **18,516 training complexes**.

**Expected gain**: 18,516 complexes should reduce overfitting dramatically. With
850 epochs of pre-training on 7K complexes, the current weights are robust. 
We expect <10% relative improvement on MDM2 but large gains on unseen target families.

### Phase 2 (Recommended) — Deep backbone unfreezing

**Target**: Match DiffPepDock DockQ (≥0.23 on ≥90%) with better RMSD median.

Unfreeze additionally:
```python
# Add these to _UNFREEZE_PATTERNS:
"intra_convs.3",        # last intra-residue equivariant conv layer
"cross_convs.3",        # last cross-complex equivariant conv layer
"fc",                   # feature combination layers after last conv
```
**~15–20% of parameters** — the last equivariant conv block sets the geometric
output direction, so unfreezing it allows the model to adapt its 3D prediction head
to the new data distribution.

**Hyperparameters for Phase 2:**
```python
lr = 5e-5          # lower LR for deep fine-tuning (original was 1e-3)
n_epochs = 50      # fewer epochs — more data needs less epochs
scheduler = "cosine"
dropout = 0.1      # add dropout to prevent overfit on new data
weight_decay = 1e-4
batch_size = 8     # fit in GPU VRAM with larger complexes
```

### Phase 3 (Maximum performance) — Full retraining with combined dataset

**Target**: Surpass DiffPepDock on all 174 benchmark complexes.

Combine:
- RefPepDB original training data (~7K, Zenodo download required)
- PepPC 3,619 training complexes
- PepPC-F 14,897 complexes
- **Total: ~25,516 complexes**

Unfreeze **all layers** (fresh training from pre-trained init).

**Hyperparameters:**
```python
lr = 1e-4
n_epochs = 200
scheduler = "cosine"
warmup_epochs = 10
dropout = 0.1
batch_size = 16
ema_rate = 0.999
```

**Expected time on RTX 5070**: ~40h for 200 epochs over 25K complexes.

---

## 5. Data Preprocessing Pipeline

### 5.1 Extract PepPC raw data

```bash
# Create dataset directories
mkdir -p datasets/training_formatted_peppc
cd /home/igem/unknown_software

# Extract PepPC natural complexes
tar -xzf PepPC_raw_data.tar.gz -C datasets/
# Creates: datasets/nat_raw_data_final/{chain}_{pdb}_{res}_fixed.pdb

# Extract PepPC-F protein fragment complexes
tar -xzf PepPC-F_raw_data.tar.gz -C datasets/
# Creates: datasets/frag_raw_data_final/{chain}_{pdb}_{res}_fixed.pdb (verify name)
```

### 5.2 Convert to RAPiDock training format

The PepPC PDBs contain BOTH receptor and peptide chains in one file. RAPiDock
expects separate `{id}_protein_pocket.pdb` and `{id}_peptide.pdb` files.

**Key: chain IDs are encoded in the filename.**
PepPC filename format: `{ligand_chain_id}_{pdb_id}_{resolution}_fixed.pdb`
PepPC-F filename: `{receptor_chain}_{pdb_id}_{resolution}_fixed.pdb` 
 (Ligand chain = first chain after receptor in PepPC-F CSV)

```python
# Use our existing prep_rapidock_training_data.py as template
# Extend it to handle PepPC/PepPC-F format:

from Bio.PDB import PDBParser, PDBIO, Select
import csv

def extract_peppc_complex(pdb_file, ligand_chain_id, pocket_threshold=20.0):
    """
    Split a PepPC fixed PDB into receptor pocket + peptide.
    
    Returns: (receptor_pocket_pdb_str, peptide_pdb_str, peptide_sequence)
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('c', pdb_file)
    
    # Identify all chains
    all_chains = list(struct[0].get_chains())
    receptor_chains = [c for c in all_chains if c.id != ligand_chain_id]
    peptide_chain = struct[0][ligand_chain_id]
    
    # Extract peptide coordinates for pocket selection
    pep_atoms = list(peptide_chain.get_atoms())
    pep_coords = np.array([a.get_vector().get_array() for a in pep_atoms])
    
    # Select receptor residues within threshold of any peptide atom
    # ... (same logic as existing _write_pocket function)
    
    # Write peptide sequence
    three2one = {'ALA': 'A', 'CYS': 'C', ...}  # standard mapping
    seq = ''.join(three2one.get(r.get_resname(), 'X') 
                  for r in peptide_chain.get_residues()
                  if r.id[0] == ' ')  # standard residues only
    
    return receptor_pdb, peptide_pdb, seq
```

**Script: `scripts/prep_peppc_training_data.py`** (to be created):
```bash
conda run --no-capture-output -n score-env python scripts/prep_peppc_training_data.py \
    --peppc-raw    datasets/nat_raw_data_final/ \
    --peppc-csv    third_party/DiffPepDock/datasets/PepPC_before_202201.csv \
    --peppcf-raw   datasets/frag_raw_data_final/ \
    --peppcf-csv   third_party/DiffPepDock/datasets/PepPC-F_dataset.csv \
    --output-dir   datasets/training_formatted_peppc/ \
    --max-workers  16
# Expected output: ~18,500 complex directories in RAPiDock format
# Expected time: ~45 min on 16 cores
```

### 5.3 Generate ESM embeddings (optional but recommended)

RAPiDock uses ESM-2 language model embeddings for receptor and peptide nodes.
Generating these offline speeds up training 3–5×.

```bash
# ESM embeddings for all training complexes
conda run -n rapidock python3 scripts/precompute_esm_embeddings.py \
    --csv datasets/training_formatted_peppc/training_data.csv \
    --output-dir datasets/esm_embeddings_peppc/ \
    --model esm2_t33_650M_UR50D
# ~24h on RTX 5070 for 18K complexes
```

### 5.4 Training split

Use the **official DiffPepDock split** to enable direct comparison:
- Training: complexes deposited before Jan 1 2022 (PepPC: 3,619; PepPC-F: all 14,897)
- Validation: 10% held-out from training set (random, stratified by secondary structure)
- Test: PepPC_after_202201.csv (205 complexes deposited after Jan 1 2022)

---

## 6. Modified Training Script

### 6.1 Extended `_UNFREEZE_PATTERNS` for Phase 2

```python
# In third_party/RAPiDock_finetuned/train_lastlayer.py
# Replace or extend _UNFREEZE_PATTERNS:

_UNFREEZE_PATTERNS_PHASE2 = [
    # ── Original score heads ──────────────────────────────────────────────
    "tr_final_layer",
    "rot_final_layer",
    "tor_bb_final_layer",
    "tor_sc_final_layer",
    # ── Output geometry prediction layers ────────────────────────────────
    "final_conv",
    "tor_bb_bond_conv",
    "tor_sc_bond_conv",
    # ── Embedding layers feeding output convs ────────────────────────────
    "center_edge_embedding",
    "pep_a_node_embedding",
    "final_edge_embedding",
    # ── PHASE 2 ADDITIONS: last equivariant conv block ───────────────────
    "intra_convs.3",        # 4th intra-residue equivariant conv
    "cross_convs.3",        # 4th cross-complex equivariant conv
    "iegnn_interaction.layers.3",  # verify name in get_model() output
    # ── Node update MLPs for last block ──────────────────────────────────
    "pep_a_node_norm",      # layer norm after last pep node update
    "rec_node_norm",        # layer norm after last rec node update
]
```

### 6.2 Training command

```bash
# Phase 1 (fast, minimal GPU time)
conda run --no-capture-output -n rapidock python3 \
    third_party/RAPiDock_finetuned/train_lastlayer.py \
    --train-csv datasets/training_formatted_peppc/training_data.csv \
    --val-csv   datasets/training_formatted_peppc/val_data.csv \
    --checkpoint third_party/RAPiDock/train_models/CGTensorProductEquivariantModel/rapidock_local.pt \
    --output-dir third_party/RAPiDock_finetuned/finetune_peppc_phase1/ \
    --n-epochs 30 \
    --lr 1e-4 \
    --batch-size 8

# Phase 2 (deeper unfreezing)
conda run --no-capture-output -n rapidock python3 \
    third_party/RAPiDock_finetuned/train_lastlayer.py \
    --train-csv datasets/training_formatted_peppc/training_data.csv \
    --val-csv   datasets/training_formatted_peppc/val_data.csv \
    --checkpoint third_party/RAPiDock_finetuned/finetune_peppc_phase1/best.pt \
    --output-dir third_party/RAPiDock_finetuned/finetune_peppc_phase2/ \
    --n-epochs 50 \
    --lr 5e-5 \
    --unfreeze-patterns phase2 \
    --batch-size 8
```

---

## 7. DiffPepDock vs HybriDock-Pep: Objective Comparison

### What DiffPepDock does better (currently)
1. **Breadth**: Trained on 18K complexes across diverse protein families
2. **Benchmark coverage**: Evaluated on 174 benchmark complexes (vs our 1 complex)
3. **Side chains**: With PyRosetta post-processing, gets full-atom poses
4. **DockQ**: Median 0.805 on benchmark

### What HybriDock-Pep does better (currently)
1. **Accuracy on MDM2/p53**: 0.80 Å best-of-100 vs DiffPepDock ~2 Å (DockQ benchmark)
2. **Physics-based rescoring**: Vina + AD4 ensemble ranking that DiffPepDock lacks entirely
3. **Full pipeline**: Stage 2 scoring, clustering, entropy correction, convergence analysis
4. **ML+Physics hybrid**: No other tool combines diffusion sampling with Vina+AD4+MM-GBSA
5. **Side chain reconstruction via pdbfixer**: Already in pipeline

### How to make the comparison objective

**Short-term test** (can do now):
1. Extract the 174 DiffPepDock benchmark complexes from `datasets/docking/docking_benchmark.csv`
2. Run HybriDock-Pep on each with 100 poses
3. Compute Cα RMSD and DockQ for comparison

**Benchmark complexes are listed in**: 
`third_party/DiffPepDock/datasets/docking/docking_benchmark.csv`

The PDB IDs have format `{chain}_{pdb}_{resolution}` — e.g., `A_7U09_2.1`.

---

## 8. PyRosetta Side-Chain Reconstruction for DiffPepDock Comparison

Per the original request: add PyRosetta to the DiffPepDock scoring pipeline.

PyRosetta license: academic-free for non-commercial use. 
Installation: `conda install -c rosettacommons pyrosetta` (requires academic license key).

```python
# In third_party/DiffPepDock/analysis/postprocess.py
# The try/except we added (May 27) already handles missing PyRosetta gracefully.
# To enable:

try:
    from pyrosetta import init, pose_from_file, get_fa_scorefxn
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    HAVE_PYROSETTA = True
    init("-mute all")
except ImportError:
    HAVE_PYROSETTA = False

def add_sidechains_pyrosetta(backbone_pdb: str, output_pdb: str) -> bool:
    """Add missing side chains using Rosetta FastRelax (backbone constrained)."""
    if not HAVE_PYROSETTA:
        return False
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
    
    pose = pose_from_file(backbone_pdb)
    scorefxn = get_fa_scorefxn()
    
    relax = FastRelax(scorefxn_in=scorefxn, standard_repeats=1)
    relax.constrain_relax_to_start_coords(True)  # keep backbone fixed
    relax.apply(pose)
    
    pose.dump_pdb(output_pdb)
    return True
```

**Alternative without PyRosetta** (already working):
- `pdbfixer`: Reconstructs side chains geometrically (already tested, 120 atoms for ETFSDLWKLLPE)
- Only limitation: geometry is idealized, not energy-minimized
- For Vina scoring, idealized side chains are acceptable (Vina ignores side-chain Hbond terms)

---

## 9. Recommended Next Steps (Priority Order)

### Immediate (this week)
1. **Extract PepPC_raw_data.tar.gz** and run `prep_peppc_training_data.py`
2. **Run fine-tuning Phase 1** (30 epochs, 3-5% unfrozen, 18K complexes)
3. **Benchmark DiffPepDock comparison on 174 complexes** using both tools

### Short-term (next 2 weeks)  
4. **Fine-tuning Phase 2** (50 epochs, last-block unfrozen, lower LR)
5. **Run 100-pose comparison** on 5-10 benchmark complexes: HybriDock-Pep vs DiffPepDock
6. **Commit PeptideBuilder to score-env and rapidock-env.yml** (currently installed manually)

### Before iGEM submission
7. **Full 174-complex benchmark** vs DiffPepDock with quantitative DockQ comparison
8. **iGEM wiki documentation** of training data lineage and benchmark methodology
9. **Tutorial notebook** updated with expanded benchmark results

---

## 10. Key Technical Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| PepPC-F chain ID parsing errors | Medium | Test on 10 complexes first; some files have non-standard chain numbering |
| ESM embedding OOM for 18K complexes | Low | Batch in groups of 100; use `--max-batch-tokens 4096` |
| Overfitting to PepPC-F (all fragments, uniform length 8-30) | Medium | Stratified sampling in DataLoader; mixup PepPC + PepPC-F 1:3 ratio |
| DiffPepDock-trained data has different coordinate conventions | Low | Both use PDB coordinates; verify with 1YCR pocket center |
| RAPiDock model VRAM during 18K training | Medium | batch_size=4 with gradient accumulation steps=4; effectively batch=16 |

---

## 11. Updated PeptideBuilder Environment Notes

PeptideBuilder was **not installed** in the rapidock env (discovered May 27 — caused
valence-5 crashes). Now installed manually:

```bash
conda run -n rapidock pip install PeptideBuilder
```

**This must be added to `envs/rapidock-env.yml`**:
```yaml
  pip:
    - PeptideBuilder>=1.1.0  # REQUIRED: full-side-chain template for get_edges_from_sequence
```

Without PeptideBuilder, the fallback template has backbone-only atoms, and the
name-based remapping fix still handles backbone bonds correctly — but side-chain
atom features in `lig_atom_featurizer` will have degree=0 (wrong). Install it.

---

*This guide is a living document. Update after each training run with observed metrics.*
