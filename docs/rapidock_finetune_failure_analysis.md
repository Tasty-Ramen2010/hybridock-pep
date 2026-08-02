# Why RAPiDock fine-tuning degrades placement — an adversarial forensic analysis

*2026-08-01. Investigation demanded after I too-quickly concluded "placement is
architecture-limited, give up." That conclusion is **refuted below.** Method: state each
claim, attack it with evidence from the actual weights, logs, code, held-out data, the
literature, and our own past notes. Training kept running throughout.*

---

## 0. The phenomenon to explain

Two independent fine-tune runs from pretrained RAPiDock, both with the BatchNorm fix:
- **B1** (plain BN-freeze): docking RMSD degrades monotonically on ALL length classes,
  N=48: vlong 7.4→8.4→**12.5** (ep8/15/25), long 4.3→4.4→6.5, even short 1.3→2.2.
- **B2** (BN-freeze + x0 placement term): same pattern, vlong 9.0→9.9→**10.3** (ep7/14/19).
- In BOTH, **validation score-matching loss DROPS** (B1 val 1.24→0.77) **while docking RMSD
  RISES.** The training objective improves as the actual goal worsens.

The question is not "does it fail" (it does) but **why**, and **whether it is fixable.**

---

## 1. Hypotheses raised and KILLED with evidence

Before the real diagnosis, four seductive explanations were ruled out by direct measurement.

### ✗ H-BN: "BatchNorm running stats are still drifting."
**Killed.** Measured `running_var` drift, x0-ep19 vs base = **0.000000**. The freeze holds
perfectly (it was the root cause of the *catastrophic collapse*, but that collapse is gone).

### ✗ H-EMA: "We evaluate raw fine-tuned weights but pretrained ships EMA-smoothed weights —
an apples-to-oranges confound."
**Killed by code.** `inference.py:65` — `ema_weights.copy_to(model.parameters())` — the
docking probe loads `"model"` then **overwrites with `"ema_weights"`**, for BOTH pretrained
and fine-tuned checkpoints (our `save_checkpoint` saves both keys). EMA IS applied on eval.
No confound. (Aside: ema_rate=0.999 ⇒ ~1-epoch smoothing window, so EMA can't rescue a
multi-epoch drift anyway.)

### ✗ H-Mag: "The translation head's output magnitude is collapsing (like it did in the
BN-collapse), under-shooting placement."
**Weakened to near-dead.** Rebuilt the `tr_final_layer` transfer function from weights and
swept it: base −0.240 → x0-ep19 −0.229 at ‖tr‖=1 — a **~5% magnitude change over 19 epochs.**
Far too small to explain long 4.3→6.5 (+50%). The [norms-train] tr trajectory (9.6→4.0) that
first looked alarming is dominated by per-batch σ-sampling, not a real head collapse.

### ✗ H-Leak: "The probe complexes leak into training, so the eval is meaningless."
**Killed.** All six probes (`7qqn 7tuc propedia_4ggn_D propedia_3dxc_B 7arx 7kei`) appear
**0 times** in `adapter_train.csv` and `adapter_val.csv`. The degradation is genuine loss of
generalization on **held-out** complexes.

### ✗ H-Local: "A specific layer/block breaks."
**Killed.** Per-block relative weight drift (x0-ep19 vs base) is **diffuse**: tr_head 4.8%,
rot_head 2.6%, cross_convs 1.9%, final_conv 1.9%, intra_convs 1.9%, tor_heads 1.6%,
embeddings 1.4%. No smoking-gun layer — the **whole converged model is gently pulled**.

---

## 2. THE CLAIM (what actually happens)

> **The failure is CATASTROPHIC FORGETTING from near-full-parameter, zero-dropout
> fine-tuning of a converged SOTA model on a small (3,432), narrow dataset, while optimizing
> a proxy objective (score-matching) that is decoupled from the deployed metric (docking
> RMSD). It is a METHOD failure, not a RAPiDock ARCHITECTURE limit.**

### Evidence FOR
1. **All classes degrade, including short.** Short (5-mer) is easy, robustly solved by
   pretrained (1.33 Å), and *not* a fine-tune target. It still degrades to ~2.0 Å. A model
   "failing to learn hard placement" would leave short alone; a model *forgetting* erodes
   even its strong skills. (Weak-corr r=0.25 between baseline difficulty and degradation ⇒
   damage is **broad/diffuse**, not concentrated on hard classes.)
2. **Proxy–metric decoupling.** Val score-loss ↓ monotonically while RMSD ↑. Documented:
   "lower loss ≠ better downstream" on small data, and behavioral-cloning work showing small
   datasets reach lower loss *via faster overfitting* (ICML 2023 "Same Pre-training Loss,
   Better Downstream"; behavioral-cloning small-data studies).
3. **We used the maximum-forgetting configuration.** Measured: **99.2% of params unfrozen**
   (7,490,074 / 7,553,866) and **dropout = 0.0**. Both are the *worst* choices per the
   literature.
4. **The exact architecture family shows the exact result.** *Fine-Tuning DiffDock-L for
   Allosteric Kinase Docking* (JCIM 2025; DiffDock = RAPiDock's parent) reports **"all
   fine-tuned models lead to a decrease in performance on [non-target] Type-I ligands"** —
   our short/general-class degradation, in the same model family. Their mitigations:
   **dropout=0.5** (lowest degradation), **freezing torsion / tuning translation+rotation
   only**, and **treating dataset size as a forgetting hyperparameter.**
5. **Parameter-efficient FT is the documented antidote.** LoRA/frozen-base "significantly
   reduces catastrophic forgetting by keeping pretrained weights frozen"; DiffFit reaches
   best FID tuning **0.12%** of params. Full FT is the forgetting-maximal end of the spectrum.
6. **Our own past notes agree.** [[project_tr_adapter_jul30]]: the zero-init frozen-base
   adapter proved **zero-regression** *and* **capacity to learn placement** — i.e. the base
   is improvable without forgetting when you don't touch it.

### Self-REFUTATION (where this claim is vulnerable)
- **R1.** *If it were classic forgetting, the class FARTHEST from the fine-tune data (short,
  since data is long/β-sheet-heavy) should degrade MOST. But long/vlong degrade more (~2 Å)
  than short (~0.7 Å).* This is the opposite of textbook forgetting-of-the-underrepresented.
- **R2.** *The x0 term directly supervised placement and still failed — maybe placement
  really is unreachable by this model.*
- **R3.** *Maybe the Propedia fine-tune poses are simply mislabeled/low-quality, so the model
  correctly learns a wrong target — a data problem, not forgetting.*

### Counter-arguments (why the claim survives)
- **vs R1.** Two forces superimpose: (a) uniform forgetting from full-FT, plus (b) damage is
  most *visible* where the skill is marginal. Short is robustly solved (deep basin) so equal
  weight-perturbation barely moves it; long/vlong sit on the length-cliff (shallow basin) so
  the same perturbation moves them more. The *diffuse* drift (§1 H-Local) + *weak* fragility
  correlation (r=0.25) fit "uniform forgetting × task fragility," not "targeted failure to
  learn." Short degrading AT ALL is the tell — a targeted-capability story predicts short
  untouched.
- **vs R2.** x0 changes the *objective* but not the *method*: it still full-tunes 99.2% of
  params at dropout 0. Forgetting operates on the whole weight vector regardless of the added
  loss term; a placement reward can't offset erosion it isn't structurally protected from.
  The JCIM result stands: DiffDock placement *is* improvable — with parameter-efficient
  method, not full FT. So R2's "unreachable" is unproven; "unreachable *by full FT*" is what
  we actually showed.
- **vs R3.** Partly conceded and testable — but it can't be the *whole* story: bad labels
  would mainly hurt fine-tune-similar classes (long/β-sheet), yet **short degrades too**, and
  short poses aren't even the bulk of the data. Data quality may *add* to the damage; it is
  not *necessary* to explain it. (Cheap future test: fine-tune on a tiny high-fidelity subset
  and see if short is spared.)

---

## 3. Refuting my OWN prior conclusion ("architecture-limited, give up")

I earlier wrote: *placement is architecture-limited (single composite COM vector); pivot to
ensemble.* That was premature. Refutation:
1. **The COM representation is demonstrably capable** — pretrained RAPiDock places short
   peptides at **1.33 Å** with the very same single-COM head. The head is not the ceiling.
2. **The length cliff is a training-DATA artifact, not an architecture wall** — RAPiDock's
   pretraining is **77% short/med, ~1% vlong** ([[reference_rapidock_methodology_jul29]]).
   Long/vlong are under-trained, not unrepresentable.
3. **The same architecture family was successfully fine-tuned for a placement task** (JCIM
   DiffDock-L allosteric) — once catastrophic forgetting was managed. Method, not ceiling.
4. Our degradation is fully explained by a *method* mechanism (§2) with no appeal to
   architecture. Invoking an architectural limit is unsupported and, given (1)–(3), wrong.

**Verdict flip:** the correct conclusion is **"full fine-tuning was the wrong method,"** not
"the model can't do it." The ensemble remains a strong *product* hedge, but the retrain path
is **not** closed on the evidence — we simply used the forgetting-maximal recipe.

---

## 4. What the evidence says to do next (ranked, all method-level)

1. **Frozen-base parameter-efficient fine-tune (our zero-init translation adapter, Arm B) at
   LOW, bounded LR + BN-freeze.** Structurally cannot forget (base weights untouched;
   zero-init ⇒ starts == pretrained ⇒ worst case is a no-op == pretrained, never worse). Past
   notes proved zero-regression + placement capacity; the only failure was lr 1e-4 overshoot
   ([[project_tr_adapter_jul30]]). This is the single highest-EV, theory-and-literature-and-
   our-own-data-supported lever, and we have NOT run it with the BN fix.
2. **If full-param at all: dropout 0.3–0.5 + freeze torsion, tune translation/rotation only
   + far fewer epochs.** Exactly the JCIM DiffDock-L anti-forgetting recipe. We ran the
   opposite (dropout 0, 99.2% unfrozen, 400-epoch schedule).
3. **Select checkpoints on a docking-RMSD validation set, never val loss** (proxy is
   decoupled). Even so, best epoch only *ties* pretrained under full FT — so this alone is
   not a win; it matters for the adapter runs.
4. **Data fidelity probe** (tests R3): fine-tune on a small curated high-quality subset;
   check whether short is spared. Cheap, and either sharpens or removes the data-quality term.
5. **Ensemble (Track-1B)** stays the product-level hedge regardless — highest EV for the
   *tool*, orthogonal to whether the retrain eventually wins.

---

## 5. One-paragraph honest summary

The fine-tune doesn't fail because RAPiDock *can't* place long peptides — pretrained places
short peptides at 1.3 Å with the same head, and the same architecture family was fine-tuned
successfully elsewhere. It fails because we fine-tuned **99.2% of a converged SOTA model's
parameters at zero dropout on 3,432 narrow examples**, which causes **catastrophic
forgetting** (documented for DiffDock-L), visible as broad, diffuse weight drift that lowers
the score-matching *proxy* loss while *raising* docking RMSD across every length class,
including ones we weren't even targeting. The fix the evidence points to is **not** abandoning
the model but **not full-tuning it** — a frozen-base, bounded, low-LR adapter (which cannot
forget by construction), optionally with high dropout and torsion-freezing if we full-tune at
all. My earlier "architecture-limited, give up" was wrong and is retracted.

**Sources:** JCIM 2025 *Fine-Tuning DiffDock-L for Allosteric Kinase Docking*
(pubs.acs.org/doi/10.1021/acs.jcim.5c02846; chemrxiv-2025-b8k76); DiffFit (arXiv 2304.06648);
ICML 2023 *Same Pre-training Loss, Better Downstream* (proceedings.mlr.press/v202/liu23ao);
LoRA-forgetting analyses (arXiv 2402.15415, 2503.16843). Internal: weight/BN forensics on
base·B1-ep25·x0-ep7/14/19 checkpoints; logs/bnfix_x0.log norm traces; inference.py EMA path;
[[project_tr_adapter_jul30]], [[reference_rapidock_methodology_jul29]],
[[project_bn_rootcause_aug1]], [[project_lossnorm_rootcause_jul31]].
