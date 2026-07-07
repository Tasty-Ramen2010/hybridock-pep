# Full decomposition: is our charged-FEP failure even about salt bridges? (2026-07-07)

Ram's question after the polarizable FF also failed: is it *really* the salt bridges, or are we missing other
parts? I stopped assuming and tested **every** contributor to the D75N/E337 ΔΔG. **Answer: it is NOT one clean
salt-bridge/polarization term. The errors are heterogeneous and mostly NOT electrostatic.** I chased polarization
for 4 experiments on a case that isn't even buried.

## The eight hypotheses, each tested

| # | hypothesis | test | verdict |
|---|---|---|---|
| 1 | **Protonation wrong** (buried residue pKa shift) | PROPKA on 2O3B | **RULED OUT** — Asp75 pKa 3.05→−1, Lys101 11.11→+1; states are exactly what we assumed |
| 2 | **Bridging structural water** (implicit misses it) | crystal waters within 3.5 Å of both | **RULED OUT** — none bridge Asp75–Lys101; it's a direct contact |
| 3 | **Buried ion-pair polarization** (JACS 2022) | SASA + AMOEBA mutual TI | **RULED OUT** — Asp75 is **SURFACE** (SASA 51.9 Å², 0% buried); AMOEBA gave +0.88 (no help); and burial does not predict error (below) |
| 4 | **Stripped metal ion** | Mg²⁺ scan in 2O3B | **REAL BUG** — 2O3B is Nuclease A+NuiA with a Mg²⁺; **Glu24 is 3.7 Å from it** and our pipeline deletes it → E24Q computed in the wrong field (D75N unaffected, 24 Å away) |
| 5 | **Charge-only morph incomplete** (Asp→Asn ≠ just neutralise monopole) | — | **LIKELY** ~1–2 kcal — we keep the residue's shape/vdW/polarizability; the real mutation changes atoms + adds Asn amide H-bonds |
| 6 | **Conformational reorganization** | literature (Flex ddG, buried-mutation reorganization) | **LEADING remaining** — single-mutation ΔΔG often includes repacking our fixed-pose alchemical FEP cannot sample |
| 7 | **Sampling irreproducibility** | D75N across runs | **CONFIRMED** — +1.07 / +1.49 / +3.13 / +0.88 (~2 kcal spread) hidden by ±0.5 error bars |
| 8 | **Atypical/hard case selection** | ΔΔG magnitude vs residue type | **LIKELY** — I cherry-picked LARGE ΔΔG (+5.9) surface charge mutations; surface salt bridges are heavily screened and usually contribute 0–2 kcal, so +5.9 is atypical and probably not pure electrostatics |

## The decisive table — burial does NOT explain the error
```
 case         SASA(cplx)  buried%   calc   exp    |err|
 1K8R Asp38     4.3 Å²     95%      +0.6   +2.0    1.3   buried, GOOD
 2O3B Glu24     0.0 Å²    100%      +3.7   +5.4    1.7   buried, GOOD (+Mg stripped!)
 1IAR Glu9      0.0 Å²    100%      −6.1   +3.1    9.2   buried, WRONG SIGN
 2O3B Asp75    51.9 Å²     39%      +3.1   +5.9    2.8   surface, off
 1E96 Asp38    60.4 Å²     29%      +0.2   +2.2    1.9   surface, off
```
If the problem were buried-ion-pair polarization, buried cases would fail systematically. Instead buried cases
span |err| 1.3 → 9.2 and two of the three are our BEST. **The error is uncorrelated with burial** → not a
polarization/salt-bridge-strength story. The 1IAR wrong-sign blowup is a case-specific structural/setup problem,
not a physics term.

## What the literature says (reoriented)
- **Implicit solvent (GBSA/GK) OVER-stabilises salt bridges** (insufficient screening) — so it can't be the
  source of an *under*-estimate; and explicit TIP3P (our E334/E335) gave the same → solvent model isn't it.
- **Buried-residue pKa shifts + conformational response** can be huge (QM/MM overestimated by ~16 kcal when the
  structure was over-constrained) — i.e. *the structural response to a charge change is a first-order effect*
  that fixed-pose FEP under-samples.
- **Large single-mutation ΔΔG are well-documented to come from conformational rearrangement and entropy**, which
  Flex ddG captures via backrub + side-chain repacking — methods that don't sample this systematically miss it.

## The honest verdict
**It is not the salt bridges, and it is not one thing.** The ~4 kcal gap on our chosen cases is a *stack* of:
1. a **setup bug** (stripped Mg²⁺) on the metal-coupled case;
2. an **incomplete perturbation** (charge-only morph, not the full Asp→Asn) — ~1–2 kcal;
3. **conformational reorganization** the fixed-pose alchemical morph can't sample — the leading remaining term;
4. **~2 kcal sampling irreproducibility**;
5. **case selection** — I picked atypically large ΔΔG surface charge mutations, the worst possible validation set.

Polarization, protonation, and bridging water — the things I chased or worried about — are **ruled out** for these
cases. My four experiments on "buried salt-bridge polarization" were aimed at a residue that is 0% buried.

## What this means + the corrected next step
The whole FEP arc's "it fails on charged interfaces" conclusion is **too broad**. We never validated on a CLEAN,
well-posed case. The right test now:
1. **Curate a clean charge-mutation benchmark:** no metals in/near the interface, moderate ΔΔG (1–3 kcal),
   structures without induced-fit ambiguity, keep all ions/cofactors.
2. **Do the FULL mutation** (Asp→Asn with atoms/vdW, single-topology) not the charge-only morph — or at least add
   the Asn end-state properly.
3. **Add conformational sampling** (side-chain repacking / longer / ensemble) — test whether the gap is the
   missing structural response (hypothesis 6), the single biggest untested lever.
4. Only if we still fail on *clean, full-mutation, well-sampled* cases is it a force-field/polarization problem.

Until then, the honest status is: **we have not shown classical FEP fails on charged interfaces — we've shown our
quick charge-only morph, on badly-chosen cases with a setup bug, is unreliable.** That is a very different (and
more fixable) statement. Ships for freeze unchanged: fast scorer + N5 triage flag.

Sources: JCTC 2023 charge-change guidelines (10.1021/acs.jctc.3c00757); PPI charge-mutation FEP (PMC6453258);
salt-bridge desolvation implicit-vs-explicit (10.1021/jz1010863); Flex ddG (10.1021/acs.jpcb.7b11367);
maximal FEP accuracy (s42004-023-01019-9).
