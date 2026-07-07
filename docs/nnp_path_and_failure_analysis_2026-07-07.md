# The charged-FEP failure analysis + the NNP (option 4) path — honest assessment (2026-07-07)

Ram: option 4 (NNP) is appealing — research it, draft what truly went wrong, and how NNP proceeds + integrates.

## Part 1 — What went wrong, consolidated (the whole E329→E337 arc)

| stage | what | result | lesson |
|---|---|---|---|
| E329 | absolute annihilate | −12.4 ± **39.2** | subtracting two ~+330 kcal legs → noise |
| E332 | decouple (keep intramolecular) | +6.26 ± 0.73 | fixed the *cancellation* → precision 54× better |
| E333 | relative charge-morph (∫ diff-of-derivs) | +7.12 ± 1.50 | independent route agrees 0.86 kcal → **precise + self-consistent** |
| E334 | SKEMPI D75N (short) | +1.07 vs exp +5.90 | **accuracy fails** |
| E335 | D75N (NPT-equil + 11 win + 10× sampling) | +1.49 ± 0.25 | more sampling ⇒ 0.4 kcal → **not a sampling problem** |
| diag | Asp75–Lys101 distance over 200 ps | 2.6–3.0 Å, never breaks | **not a broken-bridge artifact** |
| E336 | scorer_neutral + FEP on 2jqk | −14.65 vs −4.63 | unvalidated FEP term ⇒ decomposition worse than raw |
| **E337** | 6-case accuracy map | see below | **MM FEP is not systematically off — it's qualitatively unreliable for charged interface mutations** |

**E337 map (calc vs exp, isosteric charge mutations) — FINAL:**
```
 1K8R D38N   calc +0.64  exp +1.97   under 1.3
 1E96 D38N   calc +0.22  exp +2.16   under 1.9
 1IAR E9Q    calc −6.12  exp +3.11   WRONG SIGN, off 9.2
 2O3B E24Q   calc +3.68  exp +5.40   under 1.7
 2O3B D75N   calc +3.13  exp +5.90   under 2.8
 ───────────────────────────────────────────────
 n=5  Pearson(calc,exp)=+0.54   mean signed err −3.40 (systematic UNDER)   MAE 3.40
```
Two failure modes, both disqualifying: (1) **1IAR wrong sign** (calc says the charge HELPS binding by 6, exp says
it HURTS by 3) → **kills the "clean offset → empirical correction" idea** (option 1); you cannot patch a sign flip
with a burial term. (2) **Not even reproducible:** D75N gave +1.07 (E334), +1.49 (E335), **+3.13 (E337)** for the
*identical* mutation — a ~2 kcal run-to-run spread that the ±0.5 per-run error bars completely hide. So the
celebrated "precision" (±0.73 on 2jqk) is itself optimistic; real reproducibility is ~±2 kcal. The MM Hamiltonian
is not merely *scaled* wrong for charged interfaces — it is *unreliable and irreproducible* there.

**Root cause (three-way confirmed):** fixed-charge amber14 lacks **electronic polarization**, which is essential
for buried/interfacial ion pairs (JACS 2022: a buried Glu–Lys pair was ">40 kcal/mol" too unstable without it).
LIE/TI/FEP are all estimators of the *same* wrong energy surface, so none of them fix it. **The lever is the
Hamiltonian.** That is precisely what option 4 (NNP) changes.

## Part 2 — Option 4 (NNP): genuinely the right direction, with one hard caveat for OUR problem

**The field is here and the infrastructure exists** (we have OpenMM 8.5 + torch + e3nn in `rapidock`):
- **Chodera 2020 "hybrid ML/MM"** — the foundational method: run cheap MM alchemical FEP, then **correct the
  endpoints to ML/MM accuracy by nonequilibrium reweighting** (post-processing, no re-run of the whole path).
  Cut kinase-inhibitor FEP error 0.97→0.47 kcal. **BUT it corrects the ligand INTRAMOLECULAR energetics**
  (mechanical embedding: subtract MM-ligand-in-vacuum, add ANI-ligand-in-vacuum).
- **OpenMM-ML + NNPOps** — makes a pretrained NNP a drop-in force (ANI/MACE), CUDA-accelerated. The practical
  integration layer.
- **AIMNet2** — the right NNP for us: built for **charged** species with an **explicit Coulomb term** to fix
  MACE-OFF's long-range weakness. MACE-OFF is accurate but local/expensive.
- **QuantumBind-RBFE (2025)** — accurate relative binding FE with NNPs; the modern proof it works.

**The caveat that matters for us — do not skip this.** Our failure is an **interfacial salt bridge**
(Asp75⁻···Lys101⁺ *across* the A/B interface) — a **long-range, intermolecular, polarization** effect. The
textbook ML/MM correction (Chodera 2020) fixes **intramolecular** ligand strain, which would **not** touch our
interface interaction. And the literature is explicit: *"most MLIPs rely on a locality assumption that
fundamentally limits their ability to capture long-range electrostatic interactions essential at charged
interfaces"*, and *"accurately handling salt bridges and charged residues at protein interfaces remains a
challenge."* So the NNP that fixes us must additionally:
1. put **both salt-bridge partners + local environment** inside the ML region (interface-spanning, not
   ligand-only);
2. use a **charge-aware** NNP (AIMNet2, explicit Coulomb) — not a purely-local one;
3. use **electrostatic embedding** (MM charges polarize the ML region and back) — harder than mechanical
   embedding, and where the long-range NNP limitation bites.

**Honest verdict on #4:** right direction, real infrastructure, *not a plug-in* for our case. The easy version
(mechanical-embedding ligand correction) doesn't address interface salt bridges; the version that does
(charge-aware NNP + electrostatic embedding + interface ML region) is at the research frontier. Effort: weeks,
new deps (openmm-ml, nnpops, aimnet2 weights), GPU-heavy, and success on interfacial polarization is **not
guaranteed** by current NNPs (long-range is their known weak spot).

## Part 3 — How #4 integrates with everything (the tiered architecture)

NNP is a **correction LAYER on the top-K charged candidates**, not a replacement — exactly the `--ultra` ladder:
```
 fast scorer (shape ΔG, accurate)                          ← shipped, ms
   │
 N5 charged-confidence flag (which complexes need more)     ← shipped
   │  (only "low"/charged escalate)
 MM charge-morph FEP (cheap charged term)                   ← built E332-337; precise, but polarization-blind
   │  (endpoints only)
 NNP/ML-MM correction (AIMNet2, electrostatic embedding,     ← option 4: the accuracy layer
   interface ML region; nonequilibrium endpoint reweighting)
   │
 final ΔG = scorer(shape) + [MM-FEP charged term + NNP correction]
```
Key efficiencies that make it tractable: (a) the NNP runs **only at the two physical endpoints** (reweighting),
not the whole λ-path; (b) **only** on N5-flagged charged top-K, never a screen; (c) NNPOps GPU kernels. This is
literally the milestone's **T2 tier**, and it slots under `--ultra` as its heaviest rung.

## Part 4 — Concrete first steps if we commit to #4 (de-risking order)
1. **Install + smoke-test** openmm-ml + nnpops + AIMNet2 in a new env; run a pretrained NNP as an OpenMM force on
   a tiny charged system (validate it loads + runs on Blackwell).
2. **Reproduce a published number** — a small-molecule hydration or a Chodera-style intramolecular correction —
   to prove the reweighting machinery (our G1-partial equivalent for NNP).
3. **The real test:** an interface-spanning ML region on 2O3B D75N with electrostatic embedding — does the NNP
   correction move +1.49 toward +5.90? *This is the make-or-break; if current NNPs can't do interfacial
   long-range polarization, #4 stalls here and we need a polarizable FF (AMOEBA/Drude) instead.*
4. Only then wire as `--ultra` Tier-4.

## Bottom line
- The MM-FEP charged tier is **precise but unreliable** for charged interfaces (E337) — root cause is missing
  polarization, a Hamiltonian problem no estimator (LIE/TI) fixes.
- **Option 4 is the principled path and the infrastructure exists**, but our specific failure (interfacial
  salt-bridge polarization) is the *hardest* case for NNPs (long-range/locality). The plug-in version won't fix
  it; the frontier version might, at weeks of effort and no guarantee.
- Cleanest integration: an endpoint NNP-correction layer on N5-flagged charged top-K — the `--ultra` T2/T4 rung.

Sources: Chodera ML/MM 2020 (biorxiv 2020.07.29.227959); OpenMM 8 ML (10.1021/acs.jpcb.3c06662); NNPOps
(PMC10577237); AIMNet2 (chemrxiv-2023-296ch); QuantumBind-RBFE (arxiv 2501.01811); JACS 2022 buried ion pairs
(10.1021/jacs.2c00312); MLIP long-range limitation (arxiv 2411.19728).
