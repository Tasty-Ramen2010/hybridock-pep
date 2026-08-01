# Arm B Handoff — RAPiDock fine-tune collapse: root cause, fix, and feasibility

**Audience:** the Claude Code instance running on the DGX Spark, working from
`rapidock_adapter_full_bundle.zip`.
**Author:** Claude Code (HybriDock-Pep repo), 2026-08-01.
**Status:** collapse root cause **CONFIRMED via checkpoint forensics**; primary fix
**implemented + unit-tested**; a complementary lever implemented; Arm B is **feasible**.

> Read this top-to-bottom before touching training code. §2 is the autopsy, §3 is the
> architecture you must respect, §4 is the two fixes (already in `train_lastlayer.py`), §5
> is the online/paper grounding + feasibility verdict, §6 is the exact run plan for you.

---

## 0. One-paragraph summary

Every RAPiDock fine-tune we have run degraded or collapsed on docking RMSD *while the training
loss fell*. We chased the loss (found and fixed a real σ-normalization bug) but the collapse
persisted. Forensics on the collapsed checkpoints now show the true driver: **BatchNorm
running-statistic drift.** ~99% of the model's change during fine-tuning was in BN *running
buffers*, not weights; the deepest cross-attention (placement) conv's `running_var` exploded
**30×** exactly when docking RMSD collapsed. The fix is standard transfer-learning practice —
**freeze all BN running statistics during fine-tuning** — and is now implemented behind
`--freeze-bn-stats`. A second, complementary lever (a direct x0 COM-placement term,
`--x0-lambda`) attacks the separate "loss ≠ placement" problem. Arm B (the zero-init translation
adapter) is *independently* safe (frozen base) and is the lowest-risk vehicle to test these.

---

## 1. What "Arm B" is (recap)

- **Arm A** = full-parameter / last-layer fine-tune of RAPiDock. This is what **collapsed**.
- **Arm B** = a **zero-init translation adapter** (ControlNet-style zero-conv) bolted onto a
  **fully frozen** pretrained RAPiDock. At init it is a bit-for-bit no-op (adds exactly 0 to the
  translation score), so it *structurally cannot regress* the base model. It can only *learn an
  additive placement correction*. This is why Arm B is the safe vehicle: even if a fix is wrong,
  the frozen base is untouched.

Arm B code already exists in the bundle: `models/diffusion.py` `use_tr_adapter` path
(lines ~401–549) + `train_lastlayer.py --adapter-only`. The open question this doc answers:
**given that Arm A collapsed, is Arm B worth running, and what must change first?** Answer: yes,
and the BN fix + x0 term are the changes.

---

## 2. The autopsy — why it collapsed (CONFIRMED)

The collapsed run was `finetune_normloss` (σ-normalized loss, `--v4-mode --unfreeze-phase 3`,
lr 1e-6, 400 epochs planned, killed at ep29). We kept checkpoints ep1 / ep19 / ep22 / ep28 and
the pretrained base. Here is what the weights actually show.

**See `collapse_forensics.png` (shipped with this doc) for the four-panel figure.**

### 2.1 The change was 99% BatchNorm buffers, not weights

Relative drift `‖W_ep28 − W_base‖ / ‖W_base‖`, aggregated:

| parameter class | relative drift (ep28 vs base) |
|---|---|
| all learnable weights + biases | **0.023** |
| BN running buffers (`running_mean`/`running_var`) | **2.157** |

The learnable weights barely moved (2%). The BN running statistics moved by **216%** — ~94×
more. Whatever broke the model lives in the BN buffers.

### 2.2 The cross-attention (placement) BN `running_var` exploded 5–30×

`‖running_var‖` per cross-attention conv (receptor↔peptide interaction layers):

| layer | base | ep22 | ep28 | ep28/base |
|---|---|---|---|---|
| `cross_convs.0.batch_norm` | 812,822 | 486,942 | 383,443 | 0.47× |
| `cross_convs.1.batch_norm` | 121,699 | 67,127 | 648,115 | **5.3×** |
| `cross_convs.2.batch_norm` | 181,942 | 129,835 | 1,412,316 | **7.8×** |
| `cross_convs.3.batch_norm` | 99,119 | 53,014 | **3,015,490** | **30.4×** |

The deep cross-attention layers were *stable through ep22*, then their variance blew up 5–30×
between ep22 and ep28.

### 2.3 The timing matches the docking collapse exactly

Best-of-8 direct RMSD (Å), `short5` class: ep16 = 3.85 → ep22 = 5.11 → **ep26 = 18.52** →
ep28 = 7.62. *Every* class collapses over ep22→28 — the same window the cross-attention BN
variance explodes. Correlation is not subtle; it is lockstep.

### 2.4 Why the loss kept falling while docking died — the mechanism

BatchNorm behaves differently in the two modes:
- **Training** (`model.train()`): BN normalizes each batch with the **batch's own** mean/var, and
  *updates* the running buffers by momentum. So the training forward pass is always well-scaled →
  **training loss falls happily**, blind to buffer corruption.
- **Inference / docking** (`model.eval()`): BN normalizes with the **running buffers**. With
  `running_var` 30× too large, activations are divided by ~√30 ≈ 5.5× too much → the
  receptor–peptide interaction signal is crushed → the translation score field is wrong → the
  reverse-diffusion places the peptide in the wrong spot. **Docking collapses.**

The fine-tune data (`adapter_train.csv`, ~3.4k Propedia long/vlong/β-sheet-heavy samples) has a
different activation-scale distribution than RAPiDock's original short-dominated pretraining set.
Over epochs, the BN momentum drags the running stats toward the fine-tune distribution and,
because the deep cross layers see large activations on long peptides, the variance runs away.

### 2.5 Secondary effect (not the cause): the translation magnitude head shrank

`tr_final_layer` (the learned score-magnitude MLP) had the largest *weight* drift of any head
(rel-drift 0.073) and its output weight norm shrank −46% (0.164 → 0.088). This is the head trying
to compensate for a moving BN-normalized input — a symptom, not the driver. (Panel D of the
figure.)

### 2.6 This unifies every past failure (v1→v6c, Arm A, adapter)

- **Why loss always dropped while docking degraded** — train-mode batch stats vs eval-mode drifted
  buffers. Every run had this.
- **Why placement specifically failed** while conformation survived — the *cross*-attention
  (placement) BN drifted hardest; the torsion/conformation convs' BN was comparatively stable.
- **Why σ-normalization "delayed but didn't prevent" it** — the loss fix changed activation scales
  slightly (shifting onset ~ep18→ep22) but BN drift is largely independent of loss weighting.
- **The −0.14 Å ceiling every version hit** — the best any run could do before BN drift overtook
  the small genuine gains.

**We even built the guard and then bypassed it.** `train_lastlayer.py` already has
`freeze_frozen_bn_stats()` and a whole `--v6-mode` for exactly this. But that guard only freezes BN
layers whose **learnable params are frozen** — under `--v4-mode --unfreeze-phase 3` (everything
unfrozen) it protects nothing, and the collapsed run didn't use `--v6-mode` anyway. The fix in §4.1
closes that gap.

---

## 3. Architecture you must respect — RAPiDock ≠ DiffPepDock

This matters for the x0 term (§4.2) and for anything you change. **Do not port DiffPepDock loss
terms verbatim.**

| | RAPiDock (this repo) | DiffPepDock (`third_party/DiffPepDock`) |
|---|---|---|
| paradigm | **DiffDock-style SCORE model** (predicts ∇log p) | **FrameDiff** (predicts x0 / denoised frames) |
| peptide graph | **bi-conformational all-atom** (peptide built in Helical/Extended/Polyproline init; two conformational contributions) | single-frame per-residue rigids |
| placement factorization | **rigid COM (`tr`) + orientation (`rot`) + backbone/sidechain torsions (`tor_bb`,`tor_sc`)** | per-residue translation frames |
| translation output | **composite**: `tr_pred = global_pred[:,:3] + global_pred[:,6:9]` (two conformational arms summed) — `diffusion.py:512` | per-residue `rigids[...,4:]` |
| `trans_x0` loss | must be reconstructed from the score (see §4.2) | trivial per-residue position MSE |

**Consequence.** RAPiDock has **no per-residue translation frame** to supervise. Its "placement"
is a *single composite COM vector* (`tr`) plus a separate orientation (`rot`). So the correct x0
adaptation supervises the **isolated COM channel**, taken *downstream* of the bi-conformational sum,
leaving `rot`/`tor` to their own score losses. DiffPepDock's per-residue `trans_x0_loss` would be
meaningless here. The bi-conformational representation is *why* RAPiDock handles peptide flexibility
well and is SOTA — don't fight it; add supervision on the channel it already exposes.

An optional **all-atom** x0 variant (folds `rot` in by applying the reverse rigid transform to
peptide heavy atoms) is stronger and more faithful to the all-atom graph; its derivation is in
§4.3. Start with the COM version.

---

## 4. The fixes (already implemented in `train_lastlayer.py`)

### 4.1 PRIMARY: freeze all BatchNorm running statistics — `--freeze-bn-stats`

New function `freeze_all_bn_stats(model)` sets **every** BN submodule to `eval()` during training,
so `running_mean`/`running_var` stay pinned at the pretrained values. The affine params (γ/β) still
receive gradients in eval mode, so you can still *learn the affine transform* while the buffers are
frozen. It is re-applied after every `model.train()` (which resets submodule modes) inside
`train_epoch`.

```python
def freeze_all_bn_stats(model):
    n = 0
    for _name, module in model.named_modules():
        if 'BatchNorm' in type(module).__name__:
            module.eval()      # stops running-buffer updates; γ/β still get grads
            n += 1
    return n
```

Wired: `--freeze-bn-stats` → `model._freeze_all_bn` → applied at setup and each epoch.
**Unit-tested:** with the flag on, `running_var` drift after 20 forward passes of wildly-scaled
input is **exactly 0** (was 5–30× over the run), and γ/β remain trainable.

This is the single most important change. It directly removes the confirmed collapse driver and
is standard practice (§5).

### 4.2 COMPLEMENTARY: direct x0 COM-placement term — `--x0-lambda`

Addresses the *separate* "loss ≠ placement" (M1) problem: score-matching is a proxy that never
directly rewards the peptide COM landing on native. This adds a physical-Ångström COM error term.

**Derivation (score-model → x0).** From `utils/transform.py`:
```
tr_update ~ N(0, tr_sigma)                 # applied COM displacement
tr_score  = -tr_update / tr_sigma**2       # training target; tr_pred is trained to match it
```
So the true reverse-move to native COM is `tr_sigma² · tr_score = −tr_update` (Å), and the model's
predicted reverse-move is `tr_sigma² · tr_pred` (Å). The COM placement error in Å² is therefore
```
x0_err² = ‖ tr_sigma² · (tr_pred − tr_score) ‖²  =  tr_sigma⁴ · ‖tr_pred − tr_score‖²
```
This is what score-MSE ignores. Implementation (in `compute_loss`, gated + clamped):
```python
x0_lambda = float(getattr(model, "_x0_lambda", 0.0))
if x0_lambda > 0.0 and x0_lo <= tr_sig <= x0_hi:      # placement/refine sigma band
    move_err = tr_sig**2 * (tr_pred - tr_target)      # Å vector
    x0_sq    = torch.clamp((move_err**2).sum(), max=x0_cap)   # Å², one far draw can't dominate
    loss     = loss + x0_lambda * x0_sq
```
**Unit-tested:** a known 3 Å COM error reads as exactly 9.0 Å²; a perfect prediction reads 0.0.
Flags: `--x0-lambda` (default 0/off, try 0.05–0.2), `--x0-sigma-hi` (default 12; `tr_sigma_max`=30),
`--x0-sigma-lo` (default 0), `--x0-cap` (default 100 = (10 Å)²).

**Why COM and not per-residue:** see §3 — RAPiDock has no per-residue translation frame; `tr_pred`
is the composite bi-conformational COM. This term sits downstream of that composite, so it rewards
native COM placement without disturbing how the two conformational arms combine.

### 4.3 OPTIONAL upgrade: all-atom x0 (folds in orientation) — *not yet implemented*

The COM term supervises where the peptide *center* goes, not its orientation. A stronger,
architecture-faithful term supervises predicted peptide heavy-atom positions after the reverse
rigid transform:
```
x_atom_pred = R(rot_pred) · (x_atom_noised − COM_noised) + (COM_noised + tr_sigma²·tr_pred)
x0_allatom  = mean_atoms ‖ x_atom_pred − x_atom_native ‖²    # Å², applied at low tr_sigma only
```
This mirrors DiffPepDock's `bb_atom_loss` but over RAPiDock's all-atom peptide and its `tr`+`rot`
factorization. It needs the peptide atom coords + native pose inside `compute_loss` and a
differentiable axis-angle→matrix (`utils/geometry`/`so3`). **Try this only if the COM term helps
but plateaus** — it is heavier and higher-risk. Keep it low-`tr_sigma`-gated for stability.

---

## 5. Online / paper grounding + feasibility verdict

### 5.1 BN-stat freezing is textbook (high confidence)

The fix is not exotic — it is the standard remedy for the exact symptom:
- *TensorFlow "Transfer learning & fine-tuning"*: keep BN layers in inference mode when
  fine-tuning so they "do not update their mean & variance statistics"; updating them on a small
  set "destroys what the model has learned."
- *"Efficient ConvBN Blocks for Transfer Learning and Beyond"* (arXiv 2305.11624): fine-tuning
  stability hinges on BN running-stat handling; Eval/Frozen modes stabilize training.
- *"The BatchNormalization layer of Keras is broken"* (Datumbox): the canonical writeup of
  running-stat/behavior mismatch between train and inference during fine-tuning.
- Community consensus: on small / distribution-shifted fine-tuning sets, **freeze BN entirely
  (`bn.eval()` during training) or swap to LayerNorm/GroupNorm.** Our forensics are a clean
  instance of this documented failure.

### 5.2 Score-matching loss design (context for §4.2)

- *DiffDock* (Corso et al., 2022): sums translation/rotation/torsion score-matching sublosses;
  **normalizes the translational/rotational forces and applies learned scaling via two MLPs
  conditioned on force magnitude and timestep t** — this is precisely RAPiDock's
  `tr_final_layer`/`rot_final_layer`. Confirms both our σ-normalization fix and that the magnitude
  head is a known, load-bearing component (so its −46% shrinkage in §2.5 is a real symptom).
- *Fine-Tuning DiffDock-L for Allosteric Kinase Docking* (JCIM 2025): direct precedent that
  fine-tuning DiffDock-family score models is viable — with care.
- *DiffPepDock*: the x0-at-low-t idea (§4.2/§4.3) comes from its `trans_x0`/`bb_atom` losses.

### 5.3 Feasibility verdict for Arm B

**Feasible, and now well-motivated.** Before the forensics, retraining looked like flailing (every
run collapsed for unknown reasons). Now we have (a) a *confirmed, single-line-explainable* root
cause, (b) a *standard, unit-tested* fix, and (c) a *structurally safe* vehicle (Arm B's frozen
base). The realistic outcomes, in order of probability:

1. **BN-freeze alone stops the collapse** and the fine-tune holds ≈ pretrained or slightly better.
   High probability — it removes the confirmed driver. Even "holds without collapse" is a big win:
   it means fine-tuning is finally *stable* and every future lever is testable.
2. **BN-freeze + x0 term yields a real long/vlong placement gain.** Plausible; this is the first
   time we'd be optimizing a placement-aware objective on a non-drifting model.
3. **Still flat after both.** Then placement is genuinely architecture-limited (single composite COM
   vector) and the right iGEM move is the ensemble/Track-1B path, not more retraining. This is an
   *informative* negative, not another mystery.

Caveats to keep honest: the fine-tune set is small and long/β-sheet-skewed; N=8 docking reads have
~6 Å run-to-run noise (use **N ≥ 24**); and pretrained is a strong baseline
(short 1.33 Å at N=48) that any retrain must beat on N-matched reads to ship.

---

## 6. Exact run plan for the DGX Claude Code

### 6.1 Environment
The bundle is self-contained aarch64. Build the `rapidock` env (Python 3.10, PyTorch 2.7 + PyG).
`torch_scatter`/`torch_cluster`/`torch_sparse` may need an **aarch64 source build** (no prebuilt
wheels) — allow time. Verify: `python -c "import torch, torch_scatter, torch_cluster; print(torch.cuda.is_available())"`.
Copy `train_models/CGTensorProductEquivariantModel/model_parameters.yml` into any new output dir
before evaluating (a known silent 0-pose failure if missing).

### 6.2 The experiment matrix — isolate one variable at a time
Run in this order; **do not** change multiple knobs at once (that mistake cost us a clean read
before):

| run | command additions | tests |
|---|---|---|
| **B0** control | `--adapter-only` (no fixes) | reproduces the frozen-base no-op baseline |
| **B1** BN fix only | `--adapter-only --freeze-bn-stats` | does BN-freeze alone stop the collapse? |
| **B2** BN + x0 | `--adapter-only --freeze-bn-stats --x0-lambda 0.1` | does placement supervision add gain? |

If you instead test on **full-param** (Arm A style), the same matrix applies with
`--v4-mode --unfreeze-phase 3` replacing `--adapter-only` — but note full-param has more ways to go
wrong; the adapter is the recommended first vehicle precisely because its base is frozen.

Base command (adapter, matches our config so only the fix differs):
```bash
python -u train_lastlayer.py \
  --train-csv data/adapter_train.csv --val-csv data/adapter_val.csv \
  --checkpoint train_models/CGTensorProductEquivariantModel/rapidock_local.pt \
  --output-dir runs/armB_bnfix --adapter-only \
  --freeze-bn-stats \
  --lr 1e-4 --lr-schedule cosine --warmup-epochs 5 --grad-accum 4 \
  --n-epochs 200 --seed 42 --save-every-after 1
```
(Adapter LR can be higher than full-param — the zero-init adapter needs signal; but if the
correction overshoots, `tr_adapter_cap` bounds it. For full-param, keep lr 1e-6.)

### 6.3 What to watch (the instruments)
1. **BN running-var monitor (the decisive one).** Every few epochs, print
   `‖cross_convs.3.batch_norm.running_var‖`. With `--freeze-bn-stats` it must stay **flat at the
   base value (~99k)**. If it moves, the flag isn't taking — stop and fix before wasting a run.
2. **Docking meter at N ≥ 24** (never trust N=8; ~6 Å noise). Watch the long/vlong/β-sheet classes
   vs the pretrained N-matched baseline.
3. **Regression guard:** short/med must NOT regress. If short degrades > ~0.6 Å, the run is going
   the wrong way.

### 6.4 Success / kill criteria
- **Success:** no collapse through ≥ 40 epochs AND target classes (long/vlong/β-sheet) improve vs
  pretrained on **≥ 3 consecutive N ≥ 24 reads**, with short/med not regressing.
- **Kill:** any class jumps > ~3 Å over baseline for 2 reads, OR the BN running-var monitor moves
  with the flag on (means a wiring bug), OR short collapses (the §2.3 fingerprint).
- **Keep pretrained as the shipped Stage-1** until a retrain clears the success bar. Pretrained at
  N=48: short 1.33 / med 2.82 / long 4.30 / vlong 8.38 / β-sheet(12mer) 5.52.

---

## 7. Files in the bundle relevant to this work

- `third_party/RAPiDock_finetuned/train_lastlayer.py` — **has the fixes**: `freeze_all_bn_stats()`,
  `--freeze-bn-stats`, the x0 term + `--x0-lambda/--x0-sigma-hi/--x0-sigma-lo/--x0-cap`,
  σ-normalized loss (`compute_loss`).
- `third_party/RAPiDock_finetuned/utils/transform.py` — `get_score` (stores `tr_sigma_val` etc. for
  the loss; training-only).
- `third_party/RAPiDock_finetuned/models/diffusion.py` — the model; `use_tr_adapter` (Arm B) path;
  `tr_pred` composite at line ~512 (§3).
- `data/adapter_train.csv`, `data/adapter_val.csv` — the fine-tune split.
- `collapse_forensics.png` — the four-panel autopsy figure (ships with this doc).
- `scripts/plot_collapse_forensics.py` — regenerates the figure (hard-coded numbers; no ckpts
  needed).

---

## 8. If you (DGX Claude Code) confirm or refute this

Please write back (via Ram) one line per run: config, whether BN running-var stayed flat, and the
N ≥ 24 target-class deltas vs pretrained. If **B1** holds without collapse, that alone confirms the
root cause and unblocks every future retrain. If **B2** beats pretrained on long/vlong for 3
consecutive N ≥ 24 reads, we rebuild the shipped Stage-1 model. If both are flat, we pivot the tool
to the ensemble path and treat retraining as closed. Any of the three is a *clean, informative*
outcome — which, after this autopsy, is the point.
