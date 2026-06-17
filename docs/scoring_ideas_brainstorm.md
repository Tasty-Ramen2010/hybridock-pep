# Scoring ideas — brainstorm with rebuttals, counterarguments, literature, and combos

Ground truth (this session + prior): absolute-Kd clustered ceiling ≈0.35 (everyone, incl. PPI); charged
floor is FEP-bound (cancellation of large terms); `b(R)` unlearnable from static structure (e271/e276);
same-receptor anchoring works (r 0.26→0.63, e274); MISATO MD *dynamics* add +0.066 r orthogonally (e277,
but water-stripped so NO charge/desolvation); probe-fingerprint reaches r≈0.52 with 2–3 probes (e278, I2).

Each idea below: **claim → literature → rebuttal → counterargument → verdict**. Combos at the end.

---

## A. Analytical flexibility/entropy term (deliver MISATO's +0.066 with NO MD)

**Claim:** MISATO proved configurational dynamics carry orthogonal signal (+0.066 r). Capture it
analytically at deploy time via (i) sidechain **rotamer-entropy** loss on binding (rotamer-library
counting) and (ii) normalized crystallographic **B-factors** of binding-site residues.
**Literature:** rotamer entropy — Doig & Sternberg, *Protein Sci.* 1995; B-factor as entropy proxy —
Yuan et al. 2003; configurational entropy in FlexPepDock.
**Rebuttal:** rotamer counting and B-factors are crude, resolution-dependent surrogates; real MD only gave
+0.066, an analytical proxy captures less.
**Counterargument:** it's **free and deployable everywhere** (every crystal/pose), needs no 124 GB MD set,
and even half of +0.066 is worth more than the failed charge levers. Unlike MISATO it covers all targets.
**Verdict: TEST.** Cheap, directly cashes in the one positive MISATO finding without the MD dependency.

## B. MISATO-frame Boltzmann ensemble scoring

**Claim:** score all 100 MD frames per complex and Boltzmann-average — a *true* thermodynamic ensemble.
**Literature:** MM/PBSA ensemble averaging — Genheden & Ryde, *Expert Opin. Drug Discov.* 2015; LIE (Åqvist).
**Rebuttal:** we already found RAPiDock pose ensembles are a dead end (docking samples ≠ Boltzmann cloud,
6 Å spread; memory: pose-ensemble Jun11).
**Counterargument:** MISATO frames are *real MD samples* (a genuine Boltzmann ensemble), categorically
unlike docking decoys — the prior failure doesn't apply. This is the correct way to get the entropy term.
**Verdict: TEST on the 758** (cheap — frames already on disk). Caveat: only 758 complexes have MD.

## C. Siamese ΔΔG network + anchoring

**Claim:** train a twin network to predict the within-receptor **difference** in Kd between two peptides
(the learnable signal), then deploy by anchoring to a measured reference (predict ΔΔG to the probe).
**Literature:** Siamese/relative affinity nets — e.g. Jiménez-Luna relative scoring; pairwise learning-to-rank;
DiffNet MLE (Xu, *JCTC* 2019) for reconciling relative edges.
**Rebuttal:** within-receptor relative is what cheap anchoring already does; the net may just relearn it.
**Counterargument:** a net trained on **all** within-receptor pairs jointly learns transferable relative
physics better than per-receptor subtraction, and *natively* fits the I2 probe-fingerprint deployment.
**Verdict: STRONG — serves the validated lane.** Combine with I2 (below).

## D. Per-receptor random-effect GNN (mixed-effects baked into the architecture)

**Claim:** a GNN on the interaction graph with an explicit learned per-receptor intercept = `b(R)`;
fixed effects (transferable physics) and random intercept trained jointly.
**Literature:** PIGNet, GraphBAR, SchNet affinity GNNs; mixed-effects deep models.
**Rebuttal:** GNNs on PDBbind plateau at the same clustered ceiling (~0.5) and our 17 features already hit
~0.4; the independent benchmark showed all models ~0.2 cross-distribution.
**Counterargument:** the explicit intercept lets the GNN *stop* trying to predict `b(R)` (which e276 proved
impossible) and focus capacity on the transferable part — a cleaner objective than end-to-end absolute.
**Verdict: MEDIUM** — principled but ceiling-bound; only worth it if A–C plateau.

## E. Water term for the RIGHT regime (hydrophobic enclosed pockets only)

**Claim:** add a water-displacement term, but *only* for hydrophobic enclosed pockets (route by pocket type).
**Literature:** WaterMap — Abel et al., *JACS* 2008; GIST — Nguyen et al. 2012; 3D-RISM.
**Rebuttal:** we tested RISM + GIST → flat on the charged floor (memory: NIS/electrostatics; GIST writeup).
**Counterargument:** the GIST writeup explicitly says the failure was *wrong regime* — water-displacement
wins on rigid hydrophobic pockets (kinase ATP sites), untested on the *right* subset. Route by pocket type.
**Verdict: NARROW** — only the hydrophobic-enclosed subset; expensive; defer.

## F. Orthogonal ESM-2 head, blended (not replacing physics)

**Claim:** add an ESM-2 sequence head as a second opinion, blend with the physics model.
**Literature:** ESM-2 — Lin et al., *Science* 2023; ProtDCal (PPI-Affinity).
**Rebuttal:** ESM per-contact didn't help before (memory); sequence-only hits the same ceiling.
**Counterargument:** as a *blend* (like MISATO's +0.066), an orthogonal learned representation may add a
few points even if it can't stand alone.
**Verdict: CHEAP TEST** — low effort, low odds, but blends have repeatedly added small gains.

---

## Combos (the high-value syntheses)

**C1 — "Analytical dynamics scorer" = A (rotamer entropy + B-factor).** Delivers the MISATO-validated
dynamics signal (+0.066) at deploy time with no MD, on every target. The cheapest real upgrade.

**C2 — "Learned-relative anchoring" = C (Siamese ΔΔG) × I2 (probe fingerprint).** The Siamese net supplies
a *learned* within-receptor relative term; I2 supplies the measured probe anchors. Deployment: measure 2–3
probes (I2 says that reaches r≈0.52), let the Siamese net rank variants against them. This is the
strongest path for the iGEM mode-(b) hero result.

**C3 — "Ensemble-anchored absolute" = B (MISATO Boltzmann frames) × same-receptor anchor.** Ensemble-score
the relative term over real MD frames, then anchor to a same-receptor probe. Caps at the 758 with MD, but
is the most physically complete same-receptor estimator short of FEP.

**C4 — "Routed water" = E gated by pocket hydrophobicity, fused with C1.** Only spend the water term where
it's validated (hydrophobic enclosed), use the analytical dynamics term everywhere else.

---

## Recommended next (cheap → expensive)
1. **C1 (analytical dynamics term)** — cash in MISATO's +0.066 without the MD dependency. *Next experiment.*
2. **I2 wiring** — it works (r 0.165→0.52 with 2–3 probes); make it the deployment protocol.
3. **C2 (Siamese ΔΔG × I2)** — the learned-relative hero path.
4. **B/C3 (MISATO Boltzmann frames)** on the 758 — test the real-ensemble entropy term.
5. Defer D/E/F unless 1–4 plateau.
