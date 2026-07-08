# Why the charged tier fails, and the plan for a peptide-specific corrector ("HybriCharge")

**Date:** 2026-07-07 · Author: Dry Lab · Status: design draft (E345 campaign in flight, n=5/22 at writing)

The fast scorer is charge-blind. We built four physics engines to fill that gap (explicit FEP, ECC-scaled FEP,
GB-implicit, GFN2-xTB QM). Each fails differently, and they **collectively** fail on one class. This document
diagnoses each failure against the literature, reads it against our own feature↔error correlations, and drafts
a corrector that uses the best of each and knows when to abstain.

---

## 1. Two failure axes (from the feature↔signed-error correlations, E347)

Signed error = calc − exp; **+ = over-estimate**, **− = under-estimate / wrong sign**. Even at n=5 the structure
is clean and matches the physics:

| feature | expl | ecc | gb | qm | reading |
|---|---|---|---|---|---|
| **n_contacts** | +0.85 | +0.85 | +0.61 | +0.88 | more interface contacts → **over**-estimate |
| **cation_dist** | −0.88 | −0.82 | −0.77 | −0.65 | no salt-bridge partner (dist→∞) → **under**-estimate / wrong sign |
| has_saltbridge | +0.47 | +0.34 | **+0.75** | +0.05 | GB specifically over-stabilises salt bridges |
| complex_atoms | +0.47 | +0.53 | +0.09 | +0.75 | big complexes inflate explicit & QM |
| n_polar_neutral | +0.50 | +0.58 | +0.01 | +0.74 | QM over-binds polar-neutral contacts |

Everything collapses onto **two orthogonal axes**:

- **Axis A — over-stabilisation** (driven by contact count / salt-bridge presence): a **Hamiltonian** error.
  Fixed-charge FFs and GB over-count charge–charge attraction. Correctable.
- **Axis B — buried-charge wrong-sign** (driven by cation_dist, i.e. *absence* of a salt-bridge partner): a
  **dynamics/sampling** error. The hard one.

---

## 2. Why each engine fails (mechanism + literature)

### Explicit fixed-charge FEP — over-stabilises everything on Axis A
No electronic polarisation → fixed-charge FFs **over-stabilise salt bridges / buried ion pairs**; the ΔΔG is also
a small residual of two ~90-kcal desolvation legs (catastrophic-cancellation-adjacent) and carries a net-charge
finite-size artefact. Net: systematic over-estimate that scales with contact count (+0.85). *(JCTC 2023 charge-
change guidelines 10.1021/acs.jctc.3c00757.)*

### ECC (charges ×0.75) — right idea, uniform knob
Electronic-continuum correction (q/√ε_el = 1/√1.78 ≈ 0.75) restores the missing electronic screening and **halves
the over-stabilisation** (MAE 6.37→4.24). But it's a *uniform* scale — it can't know a given contact is a buried
ion pair vs a surface one, and it does nothing for Axis B. *(J. Chem. Phys. 153:050901; PMC12302216 insulin
salt-bridge overbinding fixed by scaling.)*

### GB-implicit — best consistency, two structural biases
Continuum solvent computes the desolvation reaction field **analytically** (no 90-kcal water legs), which is why
it's our most consistent engine so far. But the literature pins two biases we see directly:
1. **Salt bridges too stable by 3–4 kcal** in GB (our has_saltbridge corr = **+0.75**). *(JCTC 10.1021/ct050183l.)*
2. **Under-desolvates buried groups** because the dielectric-boundary model neglects solvent-excluded volume
   (10.1021/jz1010863) — feeds Axis B.

### GFN2-xTB QM — catches Hamiltonian H-bonds, but over-binds and is high-variance
Semi-empirical tight-binding **systematically over-binds charged/ionic clusters** (DFTB3 ~+10 kcal vs DFT on ionic
clusters), needs implicit solvent or it destabilises, and our single-point cluster is **high-variance** in which
residues get captured (great on 1AO7 +3.8, catastrophic on 1CHO −11, 1E96 −9.8). Its one unique win: it captures
**directional H-bond + polarisation** the FFs can't — which is why it *alone* got the 1IAR aromatic-pocket sign
right. *(GFN2-xTB JCTC 10.1021/acs.jctc.8b01176; DFTB ionic-cluster overbinding PMC4196743.)*

---

## 3. Why they COLLECTIVELY fail on buried-no-salt-bridge (1BRS, 1E96, 1IAR) — the unifying cause

This is the deepest hole and it has **one root cause**. Buried ionizable groups experience **apparent dielectric
constants of 10–20, not the 2–4** of a static hydrophobic interior — because the protein *dynamically responds*:
**water penetrates** and the **backbone/side chains reorganise** around the buried charge, screening and
stabilising it. The canonical finding: *"FDPB/electrostatics calculations with a **static structure** cannot
predict the pKa of internal ionizable groups."* *(PMC6413497 hydronium/water penetration; PNAS 10.1073/
pnas.1010750108 buried-Lys pKa shifts; PMC1861777 reorganization-coupled internal Asp.)*

**Every one of our engines uses a static or minimally-sampled structure.** So for a buried carboxylate with no
salt-bridge partner they all compute the *maximal* desolvation penalty (low apparent dielectric) → they predict
neutralising the charge **helps** binding (ΔΔG < 0) → **wrong sign**, because experiment says the charge helps
(+1.5). Our cation_dist correlation (−0.85) is exactly this: no partner ⇒ buried-desolvation trap ⇒ wrong sign.

The decisive consequence: **Axis B is not a Hamiltonian problem.** Better Hamiltonians (polarizable AMOEBA, QM)
*cannot* fix it — we proved AMOEBA didn't, and QM fails 1BRS/1E96 too. It needs the **dynamic reorganisation +
water-penetration** term. 1IAR is the lone exception QM rescues **only** because there the stabilisation is a
*Hamiltonian* H-bond effect (Glu–Tyr-OH), not pure reorganisation.

---

## 4. The plan — "HybriCharge": a peptide-specific, physics-informed charged corrector

Not a from-scratch force field (that needs 10⁴–10⁵ labelled points we do not have). Instead, the honest,
data-appropriate build: **a routed, bias-corrected, confidence-gated ensemble of the physics we already have,
plus the one missing physical term (reorganisation), with a Δ-learning layer once data allows.** This is a
"peptide charged force-field correction" in spirit — it *is* an energy model, just physics-informed and calibrated
rather than black-box.

### Layer 1 — per-route linear calibration (fixes Axis A, available now, n=22-supportable)
Each engine has a *systematic, correctable* bias (GB +3–4 on salt bridges, QM cluster over-bind, explicit contact-
scaling). Fit a **per-route linear map** (like LIE's fitted β; scale-invariant to Pearson r) with leave-one-out.
Interpretable, few parameters — fittable on 22 cases without overfitting.

### Layer 2 — environment router (uses the correlations as the decision surface)
Route each charged residue by interface geometry:
- **salt-bridge** (cation_dist < 4 Å) → calibrated **GB or ECC** (Axis A, well-corrected).
- **aromatic / polar-neutral H-bond pocket** (n_aromatic ≥ 1, no cation) → calibrated **QM** (the 1IAR win).
- **buried, no partner, no aromatic** (the 1BRS trap) → Layer 3, else **abstain** (low-confidence flag).

### Layer 3 — the reorganisation/water-penetration term (fixes Axis B, the real research)
The missing physics. Cheapest honest options, in order of build cost:
1. **Explicit-shell micro-sampling**: add a thin explicit-water shell around the buried residue + short (100–500 ps)
   restrained MD so water can penetrate and the local structure relaxes — recover the 10–20 apparent dielectric
   dynamically. This directly attacks the static-structure trap.
2. **Burial-scaled dielectric correction**: an empirical apparent-ε(burial, cavity, packing) that lifts the local
   dielectric for buried groups — cheap, calibratable, less rigorous.
3. Flag-and-abstain if neither is warranted — never emit a wrong-sign number silently.

### Layer 4 — confidence gate (knows when to act)
From the correlations: **high burial + high cation_dist + low aromatic = low confidence** unless Layer 3 ran. The
tier reports ΔΔG **with** a confidence label; `--ultra` trusts high-confidence corrections and flags the rest —
the honest UX we already ship for the N5 triage.

### Layer 5 — Δ-learning, once data allows (the scale-up)
With the full **SKEMPI charged-mutation set (hundreds of cases)** + our physics features (3 engine outputs +
burial, cation_dist, contacts, aromatic, cavity, sequence context), train a small gradient-boosted **Δ-model**
that predicts the residual to experiment — learning *which engine to trust where* and the reorganisation
correction from data. This is the modern paradigm (3D-ΔΔG, SAMPDI-3D, Δ-ML force fields, 2024–25). n=22 is the
proof-of-concept + bias characterisation; the model needs the data expansion first.

### Peptide-specific angle (why this is *ours*, for the iGEM tool)
Peptides bind in **extended, solvent-exposed, flexible** conformations — the reorganisation/water-penetration
physics (Layer 3) and the abstain-when-buried logic matter *more* for peptides than for rigid protein cores. The
corrector is tuned and validated on the charged-residue environments peptides actually present, and it plugs into
`--ultra` as the charged tier via the LRA decomposition ΔG = ΔG_neutral(scorer) + Σ ΔG_charged(HybriCharge).

---

## 5. Immediate next steps
1. Finish E345 (22 cases) → full correlation matrix → confirm the two axes at n=22.
2. Fit Layer-1 per-route calibration + Layer-2 router; report leave-one-out r/MAE/RMSE vs each raw engine.
3. Prototype Layer-3 explicit-shell micro-sampling on the 1BRS/1E96 buried cases — does dynamic water penetration
   flip the sign? This is the decisive experiment for Axis B.
4. Lock the `--ultra-charged` formula = calibrated router + confidence gate; wire the numeric leg into
   `scoring/charged_fep.py` (CALIBRATION constants).
5. Scope the SKEMPI charged-set expansion for the Layer-5 Δ-model.

**Sources:** JCTC 3c00757 (charge-change FEP); JCP 153:050901 & PMC12302216 (ECC); ct050183l & jz1010863 (GB salt-
bridge/buried desolvation); JCTC 8b01176 & PMC4196743 (GFN2-xTB / DFTB ionic overbinding); PMC6413497, PNAS
1010750108, PMC1861777 (buried-ionizable apparent dielectric / water penetration / reorganisation); 3D-ΔΔG
(PubMed 40375059), SAMPDI-3D (PMC11764785), Δ-ML force fields (PMC11500277).
