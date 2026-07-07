# Refining Ram's mutation/neutralization idea → a real, literature-grounded protocol (2026-07-07)

Ram's proposal (verbatim intent): for `--ultra`, a NEUTRAL peptide makes many copies + slightly refines (current
behaviour); a CHARGED peptide instead makes **sequence mutations** — neutralised variants and "more-charged"
variants — **scores them all, and works back the original** from the differences; for the best mutants, **run
MD with explicit water**, watch **desolvation**, **monitor the derivative on the charges**, and the **grand
gathering of all values** gives the answer.

## The honest headline: you re-derived the state of the art

Each piece of your intuition maps onto a *named, published, validated* method. This is not a coincidence — it
means the instinct is sound. But two pieces need correcting, and the corrections are what separate a number that
works from one that's systematically wrong.

| your intuition | the real method it is | key reference |
|---|---|---|
| "monitor the derivative on the charges" | **Thermodynamic Integration**: ΔG = ∫₀¹ ⟨∂U/∂λ⟩_λ dλ, λ scales charge | Kirkwood 1935; TI is the derivative you named |
| "run MD with clear water, watch desolvation" | **explicit-solvent alchemical charging leg** (the reorganisation E327's Born proxy couldn't see) | pmx / GROMACS, TIP3P |
| "mutate the sequence, score, work back from differences" | **alchemical mutation free energy** (single-topology morph), automated for peptides | pmx — Gapsys & de Groot 2015 |
| "make a bunch of mutants and gather all values at once" | **Multisite λ-dynamics (MSλD)** — many substituents/sites sampled *simultaneously* in one simulation | Knight & Brooks 2011; T4-lysozyme r=0.914 |
| "grand gathering of all values" | **MBAR / TI estimator** over all sampled states | Shirts & Chodera 2008 (MBAR) |
| "½·interaction" (the LIE ½ we kept hitting) | **LIE** electrostatic term β·⟨V_elec⟩, β=0.5 | Åqvist 1994 |

**MSλD is the important one for your specific framing.** It is *exactly* "make a bunch of mutations and gather
all the values in one go": it samples multiple mutations at multiple sites concurrently and is **20–50× faster**
than running each FEP separately, with **Pearson 0.914 / MUE 1.19 kcal/mol** on 32 protein mutations and scaling
to hundreds of sequences. Your "gather all values" is not hand-waving — it is the defining advantage of MSλD over
plain sequential FEP.

## The two corrections (this is where naive versions go wrong)

### 1. The differences must come from the MD, NOT from our scorer
"Score the mutants and work back the original" only works if the score *sees charge*. Our fast scorer is
charge-blind (poly-ALA moved ΔG 0.07 kcal, E308; single-mutation scoring was a 51% coin-flip, E313; the cheap
neutralisation double-difference was r≈0, E327). So the "difference between mutants" cannot be our model's ΔG —
it has to be the **alchemical free energy from the explicit-water MD** (the ∂U/∂λ you correctly want to monitor).
The scorer supplies the *shape* ΔG (its accurate regime); the MD supplies *only* the charged Δ. That division of
labour is the T1-charged design, and your idea is the same architecture from the mutation side.

### 2. Charged→neutral CHANGES NET CHARGE → mandatory finite-size correction
This is the piece missing from the raw idea, and it is not optional. When a mutation changes the solute's net
charge, explicit-solvent PBC simulations carry a **large artifact** from the periodic box self-interaction:
Rocklin et al. measured up to **17 kJ/mol (~4 kcal/mol)** of spurious signal on a +1 ligand, *box-size and
protein-charge dependent* — i.e. bigger than the effect we're trying to measure. Fixes (must apply one):
- **Rocklin/Hünenberger correction scheme** (periodicity-induced net-charge + undersolvation + residual
  integrated-potential + discrete-solvent terms), or
- **co-alchemical charge balancing**: simultaneously mutate a distant solvent-exposed residue the opposite way
  (or transmute a counter-ion) so net charge stays constant — but note JCTC 2025 ("Probing Limitations of
  Co-Alchemical Charge Changes") shows this itself has limits and must be validated.

Without this, the "grand gathering" gathers a box-size artifact. This is *the* reason charged FEP is hard and
why our cheap surrogates were doomed — even the real MD needs a correction the intuition doesn't include yet.

## The refined protocol (T1-charged, upgraded with your multi-mutant idea)

```
 fast pipeline → top-K charged-peptide poses (shape ΔG from our scorer, its accurate regime)
      │
      ▼  --charged-refine K   (NOT --ultra; see below)
 1. build single-topology alchemical map: charged side chains ⇄ neutral isosteres (K/R→Q, D/E→N/Q)
 2. solvate in EXPLICIT TIP3P + neutralising ions; equilibrate  (your "clear water")
 3. MSλD: couple ALL the charged-residue λ's in ONE simulation → the whole panel's charged Δ at once
    (your "bunch of mutants, gather all values" — 20–50× cheaper than sequential FEP)
 4. run BOTH legs (bound complex AND free peptide — the thermodynamic cycle) and monitor ⟨∂U/∂λ⟩
    (your "monitor the derivative"); TI or fast-switching work (pmx-style nonequilibrium)
 5. apply the Rocklin/Hünenberger net-charge finite-size correction  (the mandatory fix)
 6. MBAR the lot → ΔG_charged; final ΔG = scorer(shape) + ΔG_charged
```

### One correction to the `--ultra` framing
`--ultra` is *cheap variance reduction* — randomised smoothing that averages jittered copies to sharpen
**ranking**; it adds **no physical signal** and costs milliseconds. The charged path above is **GPU-hours of
explicit-water MD** — a fundamentally different computation. Conceptually they rhyme ("make copies") but they are
not the same operation, and folding MD into `--ultra` would mislead users about cost. So it belongs behind a
separate, clearly-expensive flag (`--charged-refine` / the milestone's `--fep-refine`), reserved for the final
2–3 candidates, not the whole panel. Your instinct to *branch behaviour on charge* is right; it's just two
different tools on the two branches.

## Why this is the ONLY thing left that can work
Every cheap route died for one reason (E305–E327): the charged term is the **reorganisation of water and the
pocket around the (dis)appearing charge** — a *fluctuation*, not a static energy. Static Coulomb/Born, ML ΔΔG,
3D-RISM, ensemble ⟨V_elec⟩, and the cheap neutralisation double-difference all miss it because none of them
sample that reorganisation. Your protocol is the first that does — because it runs the explicit-water MD and
watches the derivative. That is exactly why it can work where the others couldn't, and exactly why it costs what
it costs.

**Status:** this is the T1-charged milestone, now sharpened with (a) MSλD for the multi-mutant efficiency you
described and (b) the net-charge finite-size correction it was missing. Build gate remains G1-full (a converged
peptide ΔΔG, GPU-hours).

### Mechanism demo — the derivative IS finite and integrable in explicit water (E328)
`scripts/e328_explicit_water_ti.py` builds a 2269-atom explicit-TIP3P system (PME) and monitors ⟨∂U/∂λ⟩ — the
exact quantity you wanted to watch — as the solute's charges are scaled:

```
 λ_elec   ⟨dU/dλ⟩ (kcal/mol)   [sampled in real water]
  1.00      +10.3 ± 0.7
  0.75      +17.0 ± 0.7
  0.50      +28.9 ± 0.6
  0.25      +33.2 ± 0.6
  0.00      +37.5 ± 0.5
 ∫⟨dU/dλ⟩dλ (charging 0→1) ≈ +26 kcal/mol   (short/unconverged demo — magnitude is NOT production)
```

The curve is **finite, smooth, monotonic, and integrable** in explicit water — precisely the reorganisation
signal that every static term (Coulomb, Born, ⟨V_elec⟩, the E327 double-difference) failed to reproduce because
they never sampled the water. This is the mechanical proof that your "MD with clear water + monitor the
derivative" is the right and only lever. It is a *demonstration of the mechanism*, not a converged ΔG — that
needs the two legs + the Rocklin correction above.

### Sources
- Åqvist LIE — https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2020.00114/full
- MSλD (Knight & Brooks 2011) — https://pmc.ncbi.nlm.nih.gov/articles/PMC3223982/ ; T4 lysozyme scalable design — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6225981/ ; MMλD (2025) — https://pubs.acs.org/doi/10.1021/acs.jpclett.5c00467
- pmx (Gapsys & de Groot 2015) — https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.23804 ; docs — https://degrootlab.github.io/pmx/
- Charged-species finite-size correction (Rocklin/Hünenberger 2013) — https://pubmed.ncbi.nlm.nih.gov/24320250/ ; co-alchemical limits (2025) — https://pubs.acs.org/doi/10.1021/acs.jctc.5c00192
