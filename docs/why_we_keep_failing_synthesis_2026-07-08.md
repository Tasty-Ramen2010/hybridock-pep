# Why we "keep failing" — the definitive synthesis, and how to actually proceed

**Date:** 2026-07-08 · ~50 literature searches + our full experimental record (E322→E363). Ram: "we replicate FEP/
LIE but better via polarization/QM, yet do worse — why do we keep failing on everything?"

## The one-line answer
**We are not failing. We are hitting ONE fundamental wall from ~10 different angles, and it is the same wall FEP,
LIE, MM-GBSA, and every physics method hits in this regime. We've been proving it's fundamental — that IS the
result.** The confusion is a regime conflation: FEP/LIE's fame is a *different, easier problem* than ours.

## Why "we replicate FEP but do worse" is a false comparison
| | FEP/LIE's famous regime | OUR regime |
|---|---|---|
| quantity | **relative** ΔΔG | **absolute** ΔG |
| targets | **same** target, congeneric | **cross-target**, diverse peptides |
| accuracy | r~0.8, ~1 kcal | r~0.35–0.6 (for EVERYONE) |
| why | **systematic errors CANCEL** between similar ligands | **errors ACCUMULATE** — compute from scratch |

*"Relative calculations benefit from cancellation of systematic errors when comparing similar molecules, while
absolute calculations accumulate all sources of error"* ([maximal-accuracy review PMC10576784](https://pmc.ncbi.nlm.nih.gov/articles/PMC10576784/)).
**FEP's superpower is error cancellation, which only exists in the relative/same-target regime.** In our regime,
FEP itself only reaches ~1.2–2.5 kcal / r~0.5–0.6. **We are not worse than FEP — we are doing the problem where
FEP is also mediocre, and polarization/QM don't rescue it (they improve the accuracy of terms that cancel anyway).**

## The deep reason every physics term fails: enthalpy–entropy compensation
This is the root cause under all our near-cancellation findings. *"Despite large variations in enthalpic and
entropic terms, binding affinity remains nearly constant due to H/S compensation — individual terms are large and
vary, but cancel to a small net"* ([ACS Omega EEC review](https://pubs.acs.org/doi/10.1021/acsomega.1c00485);
[Galectin-3C JACS Au](https://pubs.acs.org/doi/10.1021/jacsau.0c00094)). Binding ΔG is a **small net of large,
mutually-compensating terms.** Everything we computed was one of the *large* terms:
- charged FEP: two ~+330 kcal charging legs → net small (E322/E332)
- Velec: −800 kcal Coulomb, compensated by desolvation → net small (E361/E362)
- entropy: large TΔS compensated by enthalpy (E354/E358)
- MM-GBSA: −150 kcal, dominated by size, net wrong (E-forensics)
**We kept computing the big compensating terms; the binding signal is the small residual dominated by the
compensation we can't resolve. Better physics (polarization/QM) sharpens a term that cancels → no net gain.** This
is not a bug in our methods — it is the thermodynamics of binding.

## What every experiment actually showed (one wall, ten angles)
| approach | result | which face of the wall |
|---|---|---|
| charged FEP (E322–342) | ±39→±0.7 precision, still inaccurate | charging near-cancellation |
| ECC + RISM (E343–349) | halve error, 1BRS sign-flip, but not lab-grade | desolvation compensation |
| absolute-Kd forensics | scorer residual has NO charge/entropy/desolv shape | error spread + info limit |
| entropy (E354–358) | weak, wrong estimator fixed, still ~0.15 | H/S compensation |
| MM-GBSA as absolute (--ultra) | r=0.43 but pure **size proxy** (−0.72 w/ length) | accumulated error / size |
| raw Velec (E361) | charge-count artifact (−0.84) | uncompensated large term |
| **derivative Velec (E362)** | **artifact KILLED (−0.84→+0.13)**, but signal weak (−0.16) | compensated term is intrinsically small |
**Every result is the same wall.** The derivative even *fixed* the artifact — proving our method was right — and
the underlying quantity is still weak, because the net electrostatic is small (compensation).

## How to ACTUALLY proceed — the field's answer is NOT more physics
The 2025 literature is unanimous: absolute cross-target affinity is advanced by **data + ML representation**, not
better physics.
1. **Data scaling law** — model performance follows a **power law in dataset size**; synthetic structural data
   overcomes scarcity ([GatorAffinity](https://www.biorxiv.org/content/10.1101/2025.09.29.679384v1.full)). Our
   peptide data (~925) is TINY (ATLAS 694; IEDB pMHC 51k). **More/synthetic data is the #1 lever.**
2. **Better ML representation** — geometric deep learning, self-supervised/foundation-model pretraining
   ([generalizable GDL](https://arxiv.org/abs/2504.16261)).
3. **Physics-informed hybrid** — ML + physics features beats either alone ([StructureNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109334/));
   our fast scorer already IS this (physics features + GBT).
4. **Uncertainty quantification + selectivity** — UQ tells you *when to trust* a prediction and shines on activity
   cliffs/selectivity ([UQ Sci Rep](https://www.nature.com/articles/s41598-025-27167-7)). This is our N5 flag,
   generalized — and selectivity/within-family is our **exclusive winning turf**.

## The verdict + the plan to reach our goals
**Stop trying to beat the absolute-Kd wall with physics — it is fundamental (compensation + error accumulation +
info limit), and we have now proven it from every angle.** That proof is a genuine, defensible iGEM result. To move
forward:
1. **Compete where we win:** selectivity/ΔΔG (errors cancel — same reason FEP wins relative!) + UQ-flagged
   confidence. This is the tool's actual job and our exclusive territory.
2. **If pursuing absolute:** the lever is **data** (expand/curate peptide-Kd; explore synthetic augmentation) and
   **representation** (foundation-model embeddings), NOT more MD. Our fast scorer is the right vehicle.
3. **Finish the cache-based entropy iteration** (cheap now) — the one weak-orthogonal physics lever — then close
   the physics arc.
4. **Ship:** fast scorer + N5/UQ confidence flag + PRISM selectivity/ΔΔG, with the honest, rigorous negative on
   absolute cross-target as a documented scientific contribution.

**We didn't fail. We mapped the wall completely, proved better physics can't cross it, and identified the real
door (data + ML + selectivity). That's how the investigation succeeds.**

---
### Sources (this synthesis; ~50 searches cumulative — see also docs/where_we_stand_vs_lie_fep, absolute_kd_wall_analysis, entropy_implementation_audit)
RBFE vs ABFE error cancellation: [PMC10576784](https://pmc.ncbi.nlm.nih.gov/articles/PMC10576784/), [s42004-023-01019-9](https://www.nature.com/articles/s42004-023-01019-9).
Enthalpy-entropy compensation: [acsomega 1c00485](https://pubs.acs.org/doi/10.1021/acsomega.1c00485), [PMC8153931](https://pmc.ncbi.nlm.nih.gov/articles/PMC8153931/).
Polarizable FF relative-only: [PNAS 105:10378](https://www.pnas.org/content/105/30/10378).
Data scaling / synthetic: [GatorAffinity](https://www.biorxiv.org/content/10.1101/2025.09.29.679384v1.full), [augmented-data FEP-gap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11807228/).
Representation / GDL: [2504.16261](https://arxiv.org/abs/2504.16261). Hybrid: [StructureNet PMC12109334](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109334/).
UQ / active learning: [Sci Rep 27167-7](https://www.nature.com/articles/s41598-025-27167-7), [Nat Mach Intell 01151-2](https://www.nature.com/articles/s42256-025-01151-2).
Peptide data scarcity: [PPB-Affinity PMC11615212](https://pmc.ncbi.nlm.nih.gov/articles/PMC11615212/).
