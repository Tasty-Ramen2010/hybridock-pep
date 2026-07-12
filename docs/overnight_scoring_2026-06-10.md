# Overnight Scoring Research — 2026-06-10 (morning report)

**Mandate (from Ram, before sleep):** research breakthroughs in peptide scoring,
test the enthalpy/entropy-balance thesis, find a cheaper-but-better FEP, try
secondary-structure conditioning and per-residue entropy/enthalpy, *don't stop*.
Autonomous run; decisions made solo; rigor non-negotiable (the **one-per-family /
family-mean cross-family test** is the only honest metric — everything that ever
looked like 0.55 was leakage of size / backwards-Vina / intra-family variation).

**Bottom line:** I tested your two best new ideas hard. **Physical per-residue
entropy is null cross-family** (clean negative — good to know definitively).
**NIS shows a real-looking cross-family signal on the one curated set (r≈−0.5,
p≈0.06) but I could not validate it on fresh families because off-the-shelf
extraction is too crude** — which pins the bottleneck exactly: *curated
independent-family Kd data*, not features or method. Nothing shipped to
production tonight; the within-target NIS module (committed earlier) is unchanged
and still correct.

---

## 1. Literature scan — what actually moves peptide kcal/mol

| Method | Best reported | Mechanism | Verdict for us |
|---|---|---|---|
| **LIE** (linear interaction energy) | R=0.79 on Aβ peptides — *beat* FEP (0.72), cheaper | α·Δ⟨V_vdw⟩ + β·Δ⟨V_elec⟩ + γ from 2 MD end-states | Promising but needs **MD** (bound+free); impractical in our WSL2/OpenMM-CPU env (same blocker that killed IE). Static single-point variant = interface sum = size-confounded. **Future work, needs real GPU MD.** |
| **Boltz-2** (2025 SOTA ML) | rp 0.66; >0.55 on only 3/8 assays | learned + MSA/co-evolution | cross-family signal lives in *evolution*, not cheap features |
| **PRODIGY** | 0.73 (protein-protein) | contact-type counts **+ %NIS** | %NIS is the one transferable idea — we already have it |
| **MM/GBSA part 9** | rp 0.75 | per-size-class ε tuning | requires size-class stratification |
| Per-residue conf. entropy (Creamer/D'Aquino/Baxa) | ~0.7 kcal/mol/res, SS-dependent | published per-AA scales | tested directly — see §2 |

Sources in `docs/kcalmol_research_synthesis.md`.

## 2. Experiments run tonight (all committed as `experiments/e3*.py`)

### E3 — physical per-residue entropy (your core thesis): **NULL cross-family**
Built sequence-based conformational entropy from published per-AA side-chain +
backbone scales (Abagyan–Totrov / D'Aquino) and a rotatable-bond scale for
robustness. Tested with the gold-standard **family-mean, length-residualized**
cross-family correlation:

- Raw entropy *sums* (ent_sc, ent_tot, ent_chi) hit one-per-family r≈0.4 — **but
  that is length re-discovered** (the sums scale with chain length; partial|L
  drops them to ~0.0–0.16; length itself correlates +0.43 with ΔG in this set).
- The length-free **composition** forms (per-residue, frac_flexible) are **null**
  cross-family (family-mean r ≈ −0.00, 0.03; CIs span zero).

**Conclusion:** the enthalpy/entropy thesis is physically right, but a *fixed
physical entropy composition* does not separate binders across families on this
data. Two peptides of equal length but different composition do **not** show the
predicted ΔG separation once length and family are controlled. Clean negative.

### E3c–E3d — NIS cross-family: **real on curated data, but n-limited (p≈0.06)**
This *corrects* a prior over-hasty conclusion (memory had "NIS one-per-family
0.065", from a single noisy n=20 predictive draw — the wrong test). The proper
**family-mean** cross-family correlation (one independent point per family):

| set | independent families | nis_p_frac r (len-resid) | jackknife CI | permutation p |
|---|---|---|---|---|
| ALL (Kd+Ki) | 20 | **−0.43** | [−0.74, −0.12] | 0.068 |
| Kd only | 14 | **−0.54** | [−0.83, −0.24] | 0.059 |

Robustness (kill-tests, `e3d`): **stable across all clustering thresholds**
(−0.41→−0.45), corr(nis_p, length) only −0.17, **survives nonparametric rank
residualization** (−0.37/−0.38). Correctly signed and physically sensible: high
non-interface polar fraction = peptide buries hydrophobics in the interface,
leaves polars solvent-exposed ("hydrophobic targeting") — a composition
determinant orthogonal to size. **No 2-feature combo helps** (adding any
partner barely moves r and worsens p by burning a DOF on 14 families).

So on the curated crystal-65 the signal looks real; it just sits at p≈0.06
because there are only ~14–20 independent families.

### E3f–E3g — fresh-family replication: **inconclusive (extraction too crude)**
Tried to push past p<0.05 by adding ~25 new Kd families from the bulk affinity
pool (structures already on disk). Result split:
- "Kd expanded" first pass looked significant (p=0.029) — **merge artifact.**
- Clean disentangle: **new-Kd alone r=+0.095** (null), combined r=−0.15.
- **But 88% of the new-Kd extractions are degenerate** (nis_p ≈ 0 or 1; median
  0.00) — my 1am shortest-chain/whole-receptor heuristic mis-identifies the
  peptide/pocket. So this is an **invalid test, not a negative** — the
  replication question is *unresolved*, blocked by extraction quality.

### E4 — secondary-structure conditioning: **untestable here (data-blocked)**
Your SS idea is sound, but the crystal Kd benchmark is **91% helix** (59/65; only
6 sheet, matching the known training-data SS bias — zero sheet peptides). There
is no loop/sheet diversity to learn an SS-conditioned model. `helix_frac`
correlates −0.39 with ΔG but lacks the SS spread to exploit. Needs sheet/loop Kd
data that doesn't exist off-the-shelf.

### E5 — per-residue interface propensity: subsumed by E3/NIS
The composition-feature family (per-AA interface propensities) is the same class
as entropy/NIS; all such features except NIS were null cross-family. Not pursued
separately — would re-hit the same wall.

### E6 — cheaper-than-FEP (LIE): **promising, but infeasible in this env**
LIE is the genuine "cheaper-but-competitive" method (beat FEP on Aβ peptides).
It needs two MD end-states. Our WSL2/OpenMM falls back to CPU for the real GBn2
protein system (the IE post-mortem already proved a single complex won't finish a
trajectory in 6 min). The static single-point elec/vdw decomposition is feasible
but is an interface sum → size-confounded like Vina/AD4. **Recommend: revisit LIE
only on a real CUDA box with proper MD; not viable here.**

---

## 3. The real conclusion (sharper than before)

The bottleneck is **curated independent-family Kd data with reliable
peptide/pocket extraction** — definitively, not features and not method:
- Physical entropy composition: tested, null cross-family.
- NIS: the one feature that looks real cross-family on curated data (r≈−0.5),
  but proving it needs *more curated families*, which a heuristic can't fake
  (88% degenerate extraction tonight).
- Only ~14–20 curated independent Kd families exist; ~30 more Kd PDBs are on
  disk but need **proper curation** (correct peptide-chain ID, pocket crop,
  validated binding mode) to be usable.

**Highest-value next step (a real project, not a feature tweak):** build a
clean extraction + curation pipeline for the ~30 on-disk new Kd PDBs (and mine
PepBDB/Propedia for more), then re-run the family-mean NIS test. If NIS holds
at p<0.05 across 30+ curated independent families, it becomes a defensible
cross-family *relative* affinity feature. If it doesn't, the wall is final and
proven at scale. Either outcome is publishable.

**Unchanged:** within-target NIS (variant ranking vs one receptor, r≈0.4) ships
as already committed — none of tonight's cross-family uncertainty affects it.

## 3b. PEPBI dataset — independent replication (added after Ram pointed at the Dryad zip)

The zip is the **PEPBI database** (Dryad doi:10.5061/dryad.wstqjq2wk): 329 protein–
peptide complexes with **ITC thermodynamics (ΔG, KD, and ΔH + TΔS separately)**,
clean predicted structures (A=protein/B=peptide), and Rosetta Interface Analyzer
features. ~31 binding groups = independent families. This is the clean,
independent dataset last night's degenerate extraction couldn't provide.

### Ram's enthalpy/entropy thesis — tested against MEASURED ΔH and TΔS
With experimental decomposition in hand: structural features predict **ΔG
(−0.41) better than they predict ΔH (−0.18) or TΔS (−0.08) separately**. This is
enthalpy–entropy **compensation** — the raw components are individually noisier
than their (canceling) sum, so decomposing *hurts*. The thesis is physically
reasonable but the data says predict the net, not the parts. Clean negative.

### The decisive cross-DATASET replication test — nothing survives
Two features looked significant on PEPBI's 31 groups (family-mean, length-resid,
permutation-tested): **hbonds_int r=−0.41 p=0.026**, per_residue_energy_int
−0.41 p=0.047, dSASA_polar −0.36 p=0.048. So I asked the real question: do they
replicate on the independent crystal-65 set?

| feature | crystal-65 | PEPBI | verdict |
|---|---|---|---|
| NIS (nis_p_frac) | −0.54 (p≈0.06, 14 fam) | **−0.21 (p=0.33, 27 fam)** | does NOT replicate |
| interface H-bond count | **+0.47 (p=0.03)** | **−0.41 (p=0.026)** | **SIGN FLIPS** |

`hbonds_int` is a count (0–54) on both, so the flip is real: more H-bonds →
*weaker* binding on crystal-65 (it's the backwards size proxy — bigger peptides
bind weaker in that sample), but → *stronger* on PEPBI's mutational variants.
Same name, opposite confound.

### The sharpened, stronger conclusion
**Single-dataset significance is illusory.** Every feature reaching p<0.05 on one
dataset either fails to replicate (NIS) or flips sign (H-bonds) on the other. The
real bar is **cross-dataset, sign-stable replication**, and across two independent
datasets (crystal-65 + PEPBI, ~50 independent families total) **nothing clears
it.** This is a stronger, more defensible negative result than last night's — and
it is genuinely publishable: it shows *why* cheap peptide-ΔG papers that report a
single-set 0.5 don't generalize. The within-target (variant-ranking) regime
remains the only place these features carry usable signal.

## 4. Files
- `experiments/e3_physical_entropy.py` — entropy features + one-per-family
- `experiments/e3b_length_resid_cross_family.py` — length-resid cross-family
- `experiments/e3c_family_mean.py` — gold-standard family-mean correlation
- `experiments/e3d_nis_killtests.py` — permutation / threshold / length-leak
- `experiments/e3f_expand_families.py` — family expansion (extraction caveat noted)
- `experiments/e7_pepbi_replication.py` — NIS on PEPBI clean structures (does not replicate)
- `experiments/e8_hbond_crossdataset.py` — H-bond count crystal-65 vs PEPBI (sign flips)
- (E4 SS, E3g disentangle, PEPBI ΔH/TΔS decomposition + RIA-feature significance
  were inline probes; findings captured in §3b)
- PEPBI source: `doi_10_5061_dryad_wstqjq2wk__v20250617.zip` (329 ITC complexes)
