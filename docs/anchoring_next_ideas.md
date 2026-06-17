# Cracking the receptor offset `b(R)` — idea backlog (brainstorm)

**Context:** the receptor offset `b(R)` (std 2.14 kcal/mol, e274) is the wall. It does not transfer across
receptors by *any* static similarity (sequence, pocket-seq, pocket-comp, pocket-3D — best corr +0.084,
e271; averaging can't remove it because it's bias not noise, e273), and short MD can't compute it (GIST
0.6 ns < null; equilibrium MD can't give a cross-protein ΔΔG). **Same-receptor anchoring works** (r
0.26→0.63, e274). These ideas are ranked by promise × cost, each tagged with the evidence it must beat.

## Tier 1 — lean into what works (high value, low risk)

**I1. Expand the same-receptor anchor library.** Mine BindingDB + IEDB MHC-allele panels (hundreds of
peptides per identical allele) → more receptors become anchorable (the only validated win). Pure data
work, no new physics. *Beats nothing — it just widens coverage of the r=0.63 lane.* **DO THIS.**

**I2. Receptor "fingerprint" via probe peptides (LIE-style per-system calibration).** To deploy on a new
target, dock/measure a fixed panel of K probe peptides; the vector `(y − S)` over the panel *is* a direct
measurement of `b(R)` (and a per-receptor calibration curve). Then every new peptide on that receptor uses
the fitted calibration. This is same-receptor anchoring reframed as "fingerprint the receptor with K
probes." Principled (LIE, Åqvist), and it's exactly the iGEM mode-(b) plan: measure 2–3 references on
PfLDH/hLDH. **DO THIS — it's the deployment protocol.**

**I3. Selectivity-only output (offset cancels).** For ranking peptides against ONE receptor, `b(R)` is a
shared constant → cancels → no anchor needed (already shipped). Lean the product on within-receptor
ranking + selectivity, quote absolute Kd only when anchored. *Honest scope.* **SHIPPED.**

## Tier 2 — principled, cheap to prototype (test next)

**I4. Directly LEARN `b(R)` (not pairwise similarity). — TESTED, DEAD (e276).** Leave-receptor-out GBT and
Ridge on pocket-3D ProtDCal + composition (181 receptors, OOF `b(R)`): GBT r=−0.131/MAE1.73, Ridge
r=+0.025/MAE1.58 — **neither beats predict-the-mean (MAE 1.54).** `b(R)` is not just non-transferable by
similarity (e271) but genuinely *unlearnable* from static pocket structure (confirms e255). Final nail on
the static route. → moves to Tier 4.

**I5. ESM-2 receptor embedding → `b(R)`. — DEMOTED.** A learned protein-LM representation is the one richer
static encoding untested, but with hand-crafted pocket-3D ProtDCal already at r≈0 (I4), the odds ESM
cracks a quantity that's small-difference-of-large-cancelling-terms physics are low. Low priority; only if
someone wants to fully close the "did we try every representation" question.

**I6. Mixed-effects / hierarchical-Bayes joint model.** Train on all data with a per-receptor random
intercept = `b(R)`; fixed effects transfer, intercept is fit from ≥1 anchor (sharp) or prior mean (0
anchors). Squeezes more from few anchors than the current 1-ref subtraction, and gives calibrated
uncertainty. The principled generalization of anchoring (doc §3 names DiffNet as the network form).
*Cheap to prototype; must beat plain 1-ref anchoring on the few-anchor regime.*

**I7. Bridge-peptide offset calibration (for selectivity).** If a single peptide is measured on BOTH
receptors A and B, then `b(A)−b(B) = [y−S]_A,bridge − [y−S]_B,bridge` — the offset *difference* is directly
measured from one shared compound. Turns case-B (same peptide, different receptors) into a selectivity
tool: one bridge peptide calibrates a receptor pair. *Common in selectivity panels.* **Test on case-B
data (peptides on ≥2 receptors give ready bridges).**

## Tier 3 — real physics, expensive (only for final candidates)

**I8. RBFE / FEP for the top-K only (hybrid pipeline).** ML+anchoring ranks/filters; run real
same-receptor relative FEP (mutate peptide, receptor fixed — the regime FEP *can* do) on the top 3–5 for
absolute numbers. The accuracy-ceiling lever. *Expensive but the only route to sub-kcal absolute.*

**I9. GIST done right (Track B, within-receptor entropy).** Properly converged (2–5 ns, holo-restrained,
hydrophobic pockets — the regime GIST wins) for the entropy term `−TΔS` of the *same-receptor* score, not
cross-receptor transfer. Narrow, expensive; only if I1–I7 plateau and we want the within-receptor ceiling.
*Per docs/water_why_failed_and_plan: untested in the right regime.*

## Tier 4 — long shots / likely-dead (documented so we don't re-run)

**I10. Contact-fingerprint similarity.** Richer than pocket-seq (actual residue–residue contacts + types).
But `b(R)` is receptor-intrinsic and e271 already showed pocket representations barely move; low odds.

**I11. Cross-receptor anchoring by any similarity / averaging / combo.** **CLOSED** (e266–e273). Do not
re-run: random ≥ similar (zero transfer signal); bias not noise (averaging can't help).

**I12. Short MD (≤0.6 ns) to compute the offset.** **CLOSED** (GIST < null; thermodynamic impossibility).

---

### Recommended order
1. **I1 + I2 + I3** now — the deployment value (same-receptor lane), no new research risk.
2. **I4** (running) → if any life, **I5**; then **I6** and **I7** (cheap, principled, could add real value).
3. **I8** as the top-K accuracy lever for the iGEM hero result.
4. Leave I9 unless I4–I7 plateau; never re-run I10–I12.
