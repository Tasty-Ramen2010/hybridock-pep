# Cross-domain ideas for the charged/absolute-ΔG wall (2026-07-06)

Ram's framing: our problem — *estimate a quantity that is a difference of two large, nearly-cancelling terms
(Coulomb vs desolvation), or that sits below the scorer's resolution* — is ancient and universal. It shows up
in metrology, numerical computing, quantum mechanics, surveying, finance, cooking. Each field evolved a trick.
This is a map of those tricks, each ADAPTED to us, with an honest verdict. Researched online + tested where
feasible. **The unifying result: every trick that helps us is either (a) measure RELATIVE to a reference so the
common error cancels, or (b) reduce VARIANCE by averaging. None CREATE the missing charged signal — that is a
BIAS (missing physics), and only a real per-atom energy (perturbation theory / FEP / a neural-net potential)
supplies it.**

## The universal problem in one sentence
You cannot measure the absolute; you *can* measure a difference against a reference, a derivative along a path,
or a variance-reduced average.

## The map — trick → adaptation → verdict

| # | Domain | Trick | Adaptation to HybriDock-Pep | Verdict |
|---|---|---|---|---|
| 1 | **Metrology** | Wheatstone bridge / interferometry (LIGO): measure the *difference* vs a balanced reference; common-mode cancels | ΔΔG **selectivity** (peptide on target vs off-target) — shared error cancels | ✅ **shipped** (selectivity primitive) |
| 2 | **Statistics** | **Control variates** `U−λ(V−μ)`; variance × (1−ρ²) with a correlated known-value reference | **Anchoring**: correct the query with measured reference peptides on the target | ✅ **validated** (r 0.25→0.71, E312) — this is the rigorous math behind our best same-receptor win |
| 3 | **ML robustness** | **Randomized smoothing / TTA**: perturb input, average predictions → lower variance | **`--ultra` mode**: perturb/fold peptide variants, score, collapse back | ◐ **tested** (E314): +2 pts ranking (variance↓), does NOT move absolute charged ceiling (bias) |
| 4 | **Econometrics** | **Difference-in-differences**: effect = Δtreated − Δcontrol; confounders cancel | double-difference thermodynamic cycle | ✗ **debunked** as a prediction (E312: additivity artifact, beaten by nearest-measured) |
| 5 | **Chemistry** | **Hess's law**: unmeasurable ΔH from a cycle of measurable ones (state function) | cycle-closure to constrain a ΔΔG network | ◐ could tighten shape-driven networks; charged ΔΔG is noise so closure can't help it |
| 6 | **Surveying** | Differential leveling + **loop closure**: chain relative heights, close the loop, distribute misclosure | predict a network of ΔΔGs, enforce closure (like FEP+ cycle-closure correction) | ◐ same as #5 — variance tool, needs signal to tighten |
| 7 | **Signal proc.** | **Lock-in amplifier**: modulate the signal at a known frequency, reject the huge background | modulate a charge parameter (scan charge/neutralize), detect the score's response | ✗ **blocked**: single-mutation/charge response is 51% coin-flip (E313) — below our resolution |
| 8 | **Quantum** | **Perturbation theory**: E = E₀ + ⟨ψ\|V\|ψ⟩ + …; corrections relative to a solvable H₀ | score = base(reference) + first-order correction | = thermodynamic perturbation / FEP; needs a real energy, not tree features |
| 9 | **Numerics** | **Catastrophic-cancellation reformulation** `x²−y²=(x−y)(x+y)`: rewrite so large terms never subtract | reformulate ΔG to avoid absolute contact−desolvation subtraction | partial: composition-IFP (ratios not counts) is a mild version; no clean separable terms in a black-box tree |
| 10 | **Monte Carlo** | **Common random numbers**: use the *same* randomness for two configs so their difference has low variance | score two peptides on the *same* receptor conformer so pose-noise cancels in ΔΔG | ◐ **new, untested** — cheap way to tighten ΔΔG; worth a spike |
| 11 | **Cooking** | **Baker's percentages / season-to-taste**: everything relative to a datum; adjust incrementally vs a reference | relative-to-reference scoring + iterative refinement | = anchoring (#2) |

## What this tells us (the honest synthesis)
- **Variance vs bias is the whole story.** Randomized smoothing (`--ultra`, #3), cycle/loop closure (#5,#6),
  and common-random-numbers (#10) all reduce *variance* → they tighten RANKING on targets where signal already
  exists (shape-driven), and do nothing for the charged *bias*.
- **Reference-subtraction is our real lever.** Bridge/control-variate/anchoring/selectivity (#1,#2,#11) all
  cancel the large common term — this is why anchoring and selectivity work while blind absolute does not. It
  is the same mathematical move FEP makes (relative, not absolute).
- **Signal creation needs physics.** Lock-in (#7) and perturbation theory (#8) *could* extract the charged
  contribution, but only with a well-conditioned per-atom energy/derivative. Our shape-dominated static scorer's
  single-residue derivative is noise (E313), so these are blocked until we have a differentiable per-atom energy
  (an ML force field / NN potential = NNP-FEP). That is a new milestone, not a tweak.

## Concretely worth building / spiking next
1. **`--ultra` ranking-refinement mode** (#3): honestly a variance-reduction pass — tightens shape-driven
   panels ~2 pts; cheap feature-TTA already captures most of it, so RAPiDock mutant-folding is likely not worth
   the cost. Ship it as a ranking mode with the explicit caveat that it does not improve charged absolute ΔG.
2. **Common-random-numbers ΔΔG** (#10): score candidate pairs on a *shared* receptor conformer to cancel
   pose-noise — the one untested variance trick that is cheap and could tighten within-target ΔΔG.
3. **NNP force-field track** (#7/#8): the only path that *creates* charged signal — a real project (MACE/ANI-
   style differentiable energy → NNP-FEP), scoped as its own milestone, not for the iGEM freeze.

## Addendum — desolvation-specific ideas (the charged wall's root cause), researched + tested

The charged failure's mechanism is desolvation: burying a charge costs the solvation energy it had in water, and
that penalty nearly cancels the favorable Coulomb of the salt bridge, leaving a net below static-scoring
resolution. Online desolvation methods and our tests:

| method (online) | what it is | our test | verdict |
|---|---|---|---|
| **3D-RISM-AI** (PMC9421647) | ML on 3D-RISM hydration descriptors (exchem, water sites); solvation <1 kcal | our `data/e230_rism`,`e240_ppikb_rism` | ◐ pocket hydration r=−0.35/−0.41 on e230 (n=49) — a real seq-orthogonal **receptor-offset** lever — but **≈0 on PPIKB** (n=90, doesn't generalize) and offset≠charged bottleneck (E311) |
| **GIST / WaterMap** | score high-energy waters displaced on binding | prior GIST partial-FEP | ✗ DEAD (non-reproducible, flat/wrong-sign) |
| **GB/PB single-point** (APBS local) | Born/Poisson-Boltzmann desolvation on one pose | E312 | ✗ NET (Coulomb−desolv) = noise; single-point error > the net |
| **Uncompensated-charge penalty** (EP2695097A1) | penalize formal charge NOT in a salt bridge <3.5 Å AND buried | E311 (unsatisfied buried charge) | ✗ flat (0.400→0.400), collinear with IFP |
| **Implicit-solvent ML potential** (ReSolv, arXiv:2406.00183) | ML-parametrised differentiable implicit solvent | not built | = the NNP track — the only one that could work, heavy |

**Why desolvation resists every cheap term (the unifying reason):** the decisive quantity is the *difference*
between the desolvation penalty and the compensating Coulomb — a small number hidden inside two large ones.
3D-RISM/GIST get solvation to <1 kcal only by *sampling/integrating over solvent* (RISM's integral equation,
GIST's grid MD); a single static descriptor of it inherits an error larger than the net it is trying to
resolve. Our RISM lever proves the ceiling: it captures the receptor's *average* hydration (an offset), which
is real but is not the within-target charged signal (E311). **Desolvation is FEP/RISM-integral-bound; no
static desolvation term recovers the charged net.** The one live option is a *differentiable* implicit-solvent
term (ReSolv-style) inside an NNP energy — i.e. the same NNP milestone, not a bolt-on. Reproduce the RISM
check: the e230/e240 correlations above.
