# Scoring Function Findings — Handoff for a New Scorer

**Purpose:** everything learned building/testing peptide scoring for HybriDock-Pep
(June 2026 session). Read this before building a new scoring function — it will
save you from re-running every dead end and, more importantly, from shipping an
inflated number. Every claim here is backed by a committed experiment.

**TL;DR for the impatient:**
1. There are **two different jobs** — *pose ranking* (which pose is native-like)
   and *affinity / Kd* (how tight). Don't conflate them; a feature can be great
   at one and useless at the other.
2. **Absolute peptide Kd is at a hard ceiling (~r 0.3, RMSE ~2 kcal/mol ≈ guessing
   the mean).** This is the enthalpy–entropy cancellation wall — fundamental, not
   a tooling gap. Nobody (us, CrankPep, FEP-free methods, AF3) breaks it cheaply.
3. **Vina and MM-GBSA are *backwards* for absolute affinity** (anti-correlated).
4. **Buried Surface Area (BSA) is the one correctly-signed cheap affinity signal.**
5. **For pose ranking, ref2015 ≈ BSA+clash ≈ 0.18 τ ceiling.** Tighter valid fit
   *is* the native pose (thermodynamically). BSA+clash is ~1000× cheaper.
6. **The biggest danger is the SIZE CONFOUND and several LEAKAGE traps** that
   inflate r/τ. Section 5 lists all of them. Validate length-controlled + held-out.

---

## 1. The two jobs (never blur them)

| | Pose RANKING | Affinity / Kd |
|---|---|---|
| Question | which of N poses is the real binding mode? | how tightly does it bind? |
| Ground truth | RMSD to native crystal (per pose) | experimental Kd/ΔG (per complex) |
| Metric | Kendall τ(score, RMSD); top-1 selected RMSD; hit@k | Pearson r(score, ΔG); RMSE kcal/mol |
| Incumbent | ref2015 | (was Vina/MM-GBSA — both backwards) |
| Honest ceiling | τ ≈ 0.18; top-1 ~4.4 Å; CAPRI ~55% | r ≈ 0.3 (size-controlled); RMSE ~2 |

**Key insight:** "tightest fit" and "native" are the SAME thing thermodynamically
— the crystal IS the lowest-free-energy pose, so RMSD-to-crystal is the ground-truth
test of whether your tightness-finder works. Confirmed: corr(BSA, RMSD) = −0.10
(tighter → lower RMSD → more native). You are not choosing between "tight" and
"native"; finding one finds the other.

---

## 2. The affinity ceiling (why absolute Kd is walled)

ΔG_bind = ΔH − TΔS, where ΔH ≈ −80 and TΔS ≈ +70 kcal/mol, so the observed
ΔG ≈ −10 is a small residual of two huge opposing terms (each scaling with
interface size). To predict ΔG to ±2 you need *both* big terms to ~2% — count-
based or empirical models have ~10–20% error each → ±10–20 kcal/mol on ΔG. The
error bars dwarf the signal. This is **enthalpy–entropy compensation** and it is
why peptide absolute affinity is unsolved by any cheap method.

Peptides are *worse* than small-molecule ligands here because (a) huge variable
conformational entropy, (b) huge variable buried interface (the size confound),
(c) shallow solvent-exposed grooves, (d) scarce clean training data. Ligands win
only because they're small/rigid/pocket-bound/data-rich — and even they only do
*relative* affinity well (congeneric series), not absolute.

---

## 3. All methods vs measured Kd (65 crystal complexes, the leaderboard)

`data/benchmark_crystal.json` (65 crystal complexes, measured Kd/Ki) + the
IC50/EC50 independent set (76). Size-controlled = partial-out peptide length.

| method | raw r (biased) | **size-controlled r** | sign |
|---|---|---|---|
| Vina | −0.56 | −0.39 | ✗ BACKWARDS |
| MM-GBSA (single-traj) | −0.43 | −0.19 | ✗ BACKWARDS |
| v1.2 calibration (was shipped) | −0.42 | — | ✗ BACKWARDS |
| AD4 | −0.32 | −0.11 | ✗ |
| **total BSA** | +0.45 | **+0.28** (held on new set: +0.35) | ✓ BEST |
| interface H-bonds | +0.50 | +0.29 | ✓ |
| desolvation | +0.48 | +0.26 | ✓ |
| hydrophobic packing | +0.45 | +0.26 | ✓ |
| peptide length (the confound) | +0.43 | — | — |
| BSA + interface combined | +0.42 | +0.16 | ✓ (collinear; combining HURTS) |

**Honest read:** absolute affinity tops out at r≈0.28–0.35 size-controlled,
RMSE ~1.95 kcal/mol vs ~2.1 for guessing the mean. It RANKS weakly-but-correctly;
it does NOT predict kcal/mol precisely. **BSA alone is the best single feature.
Combining interface terms HURTS (collinear, overfits on n=65).** The current
scorer (Vina/v1.2) is *backwards* — switching to BSA flips −0.42 → +0.28..0.45.

---

## 4. Pose ranking results (ref2015 vs BSA+clash vs everything)

### 4a. The ceiling decomposition (where the missing accuracy lives)
On bench300 (LOO-CV, 112 complexes, 5 poses):
```
              in-sample   held-out(LOO)
ref2015        0.180        0.175    ← no overfitting gap (fixed function)
encoder        0.270        0.123    ← HUGE gap = memorized its training targets
both           0.289        0.168
ORACLE (best feature per complex)  =  0.97
```
The oracle ≈ 0.97 means the right answer IS in the features — but *which* feature
matters is **target-specific**. A single global function averages over targets →
caps at ~0.18. To reach high τ you need target-specific knowledge: (a) per-target
training (legit only if your target is fixed, e.g. PfLDH), (b) evolutionary/MSA
priors (AF-Multimer — different model class), or (c) per-pose FEP (hours/pose).
**Global RL / a better universal physics function CANNOT exceed ~0.18** — RL is
an optimizer, not an information source.

### 4b. 3-way pose selection at N=100 (n=55, top-1 selected RMSD)
```
ranker            mean RMSD  ±sem   <4Å   <2Å   best-of-top5
OG RAPiDock        4.57 Å   0.32   53%   5%      3.53 Å
ref2015 rerank     4.62 Å   0.36   58%   9%      3.37 Å
BSA+clash          4.36 Å   0.29   53%   7%      3.37 Å
oracle             2.50 Å
```
**BSA+clash ≈ ref2015 (statistically equivalent; head-to-head 27/55 coin flip).**
BSA+clash: best mean + lowest variance. ref2015: best success rate. Both beat raw
diffusion order on best-of-top5. **BSA+clash ships as the ranker** because it's
equivalent accuracy at ~1000× lower cost (Biopython Shrake-Rupley SASA, no
PyRosetta) and doubles as the affinity signal. Score = −z(BSA) + z(n_clash),
z-normalised within the pose set. Code: `src/hybridock_pep/scoring/bsa_fit.py`.

### 4c. Things tried for ranking that FAILED (don't repeat)
- Hand-built FoldX geometric ranker (LJ/clash/H-bond/saltbridge): τ≈0.10 alone.
- Same + better physics (continuous fa_rep, LK desolvation, buried-unsat-polar):
  strong solo signals but all collinear → CV ridge 0.059; greedy-augment ref2015
  gave +0.021 in-sample → **+0.007 under nested CV with unstable picks = noise.**
- Knowledge-based MJ-style contact potential: r=−0.015 on affinity (size-controlled).
- Consensus / mode-population from N=100 diffusion poses: τ≈0.12 (weak).
- Interaction Entropy (IE) method: impractical (didn't finish 1 complex in 6 min)
  + scales with interface size = same dead axis.

---

## 5. THE TRAPS — how every "good" number turned out fake

Read this section twice. Every inflated result this session came from one of these.

1. **SIZE CONFOUND (the big one).** On peptides, interface size correlates with
   ΔG *in a given sample* by accident. Vina ≈ −(size) (corr −0.88 with n_contact),
   MM-GBSA ≈ size, BSA ≈ size. Any model hitting r≈0.5 does it by using a size
   proxy — and the sign flips on a different sample. **Always report
   length-controlled r** (partial out peptide length) AND within a fixed-length
   stratum. If the signal vanishes when length is held constant, it was size.
2. **Backwards-Vina ridge.** A ridge given Vina freely will assign it a NEGATIVE
   weight to exploit the size confound → looks like r=0.52, but it's using Vina
   *upside down* (good Vina score = predicted weaker binding). Physically nonsense,
   won't generalize. Constrain Vina's sign or drop it.
3. **LOO near-duplicate leakage.** The benchmark has co-crystals of the same
   protein (e.g. family 28 = 5× same protein, all ΔG −13.1). LOO leaves one out
   but keeps its twins → a per-family/per-cluster model memorizes the label via
   the family intercept. Per-family looked like r=0.65 LOO → **0.54 on true
   held-out (worse than single ridge).** Use GROUP-held-out splits (whole protein
   family out), not plain LOO.
4. **Encoder training overlap.** The diffusion encoder's 96-dim features inflate
   ranking on bench300 because **100% of bench300 is in the encoder's PepPC
   training set** (encoder in-sample τ=0.27 → held-out 0.12). Any learned feature
   must be tested on complexes absent from its training. Fixed physics (ref2015,
   BSA) can't leak this way.
5. **In-sample feature selection.** Greedy-picking the best features on the same
   data you evaluate on inflates τ (+0.021 → +0.007 under nested CV). Select on
   train folds only.
6. **Narrow dynamic range fools RMSE.** Peptide ΔG spans only ~2 kcal/mol, so
   guessing the mean gives RMSE ~2.1. A model at RMSE 1.95 is barely better than
   guessing. **RMSE alone is not evidence of skill — report correlation.**
7. **Reported 0.242 was optimistic.** The shipped RankerV2 figure used a trained
   NN head + tuned blend weight; clean LOO reproduction is ~0.21. Honest ranker
   number is ref2015 0.175 (leakage-free).

---

## 6. Datasets + how to validate honestly

- `data/benchmark_crystal.json` — **65 crystal complexes, measured Kd(34)/Ki(31)**,
  pKd 4.25–10.30, peptide len 9–26. THE affinity benchmark. Built by intersecting
  clean Kd+Ki affinities with PepPC crystals on disk. Use for affinity r.
- `data/eval_kd_ki_clean.json` — 101 Kd+Ki docked-pose feature rows.
- `logs/analysis_bench300/` — 240 complexes × 5 poses, RMSD labels, ref2015 phys
  (`logs/diagnosis/feats_bench300_physics.pkl`), encoder (`feats_bench300.pkl`),
  receptors (`*/scoring/receptor_cropped.pdb`). Use for pose ranking τ.
- `logs/gen_n100/` — 237 complexes × **100 poses**, RMSD labels, ref2015 phys
  (`feats_gen_n100_physics.pkl`). Receptors via `datasets/training_formatted_peppc/
  <ID>/*_protein_pocket.pdb`. Use for N=100 pose selection.
- `datasets/training_formatted_peppc/` — **9,121 crystal complexes** (pocket +
  peptide PDBs, sequence-named). No affinity labels (docking-training set).
- **REJECTED data:** BindingDB (its PDB matches are co-crystallized SMALL
  MOLECULES, not the peptide — wrong-ligand affinities; deleted 9.4 GB). PEPBI
  (Rosetta-modeled poses, ~10 kcal/mol bias, not crystals).

**Validation protocol (mandatory):** (a) length-controlled r AND within-length-
stratum r for affinity; (b) GROUP held-out (whole protein family out) not plain
LOO; (c) any learned feature evaluated only on complexes absent from its training;
(d) report correlation, not just RMSE; (e) compare against length-alone and
mean-predictor baselines.

---

## 7. Recommendations for the NEW scoring function

**Do:**
- For **pose ranking**: BSA + clash is the cheap, validated baseline (τ≈0.18,
  ties ref2015). To genuinely beat ref2015 you need a LEARNED feature on
  held-out targets (the encoder direction) — physics is saturated at ~0.18.
- For **affinity**: BSA is the best correctly-signed cheap feature. Report it as
  a *relative ranker / enrichment filter*, NOT a kcal/mol calculator.
- For **selectivity ΔΔG** (same peptide, two targets): the size confound CANCELS
  in the difference — this is the regime where physics genuinely works. Best
  honest use of the tool. MM-GBSA ΔΔG is valid here even though its absolute ΔG
  isn't.
- For a **fixed target (e.g. PfLDH)**: per-target calibration / pharmacophore from
  known binders is LEGIT (not leakage — the target never changes) and can reach
  high accuracy. This is the only honest path to high τ/r.

**Don't:**
- Don't expect to predict absolute Kd well — it's physically walled (~r 0.3).
- Don't use Vina or MM-GBSA for absolute affinity — they're backwards.
- Don't combine collinear interface features hoping for additivity — they overfit.
- Don't trust a number without length-control + group-held-out + training-overlap
  checks. Every fake win this session passed a naive check and failed an honest one.
- Don't recompile/fork Vina (spec §5.6, and it's a data-ceiling not a code problem).

**Compute-cost reality (the tool's real edge):**
- BSA / interface: milliseconds/pose. ref2015: seconds/pose (PyRosetta).
  MM-GBSA: minutes/pose. FEP: hours–days/pair.
- A full 100-pose run is ~5 min on one consumer GPU (RTX 5070). FEP costs
  10,000–100,000× more for a number only useful relatively. The honest pitch is
  **fast cheap screening + ranking + selectivity on consumer hardware**, not
  beating FEP on absolute Kd.

---

## 8. Reproduce / key scripts

- `scripts/benchmark_scoring.py` — CV affinity eval on the clean Kd+Ki set.
- `scripts/build_crystal_benchmark.py` — builds `data/benchmark_crystal.json`.
- `scripts/score_crystal_benchmark.py` — MM-GBSA (+IE/3traj/εin) on crystal poses.
- `scripts/analyze_crystal_benchmark.py` — size-confound controls.
- `scripts/knowledge_potential.py` — MJ contact potential test.
- `scripts/foldx_ranker_v2.py` — geometric ranker + ref2015-combine + nested CV.
- `scripts/three_way_rerank.py` — OG vs ref2015 vs BSA+clash at N=100.
- `scripts/pipeline_selection_test.py` — selection-quality test (bench300).
- `src/hybridock_pep/scoring/bsa_fit.py` — the shipped BSA+clash ranker.

Companion docs: `docs/scoring_overhaul_verdict.md`, `docs/ranker_validation.md`,
`docs/scoring_accuracy_analysis.md`, `docs/scoring_overhaul_plan.md`.

*Compiled June 2026 from the scoring-overhaul session
(branch phase-scoring/selectivity-and-entropy-fixes).*
