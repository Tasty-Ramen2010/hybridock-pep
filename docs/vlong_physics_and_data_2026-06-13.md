# Does vlong "fail physics"? No — it fails the LABELS. Plus: the data door (BioLiP)

**Date:** 2026-06-13 · **Script:** `e101_vlong_physics.py` · **Set:** combined crystal (cr65+the98, n=156)

## Ram's principle holds: physics didn't lie — we graded it against a flat ruler

### The cr65 vlong "−0.5 failure" is a degenerate-label artifact
cr65 vlong = 15 complexes but only **8 unique Kd values**: five at −7.9, three at −8.64, two at −10.41
(10/15 are duplicate affinities — crystal structures of the same peptide families). **std = 0.94 kcal,
range 3.6 kcal.** There is essentially no affinity variance to correlate against. A correlation computed
on flat, duplicated labels is pure noise — it can land anywhere, including −0.5. **Physics never had a
signal to capture here.** This is not physics failing; it's a rigged test.

| set | band | n | y-range | std |
|---|---|---|---|---|
| cr65 | vlong ≥17 | 15 | 3.6 kcal | **0.94** ← degenerate |
| the98 | vlong ≥17 | 15 | 6.4 kcal | 1.66 |
| ALL | vlong ≥17 | 30 | 6.4 kcal | 1.51 |

### On the proper combined set, vlong physics WORKS
- **Full model, vlong band (combined): r = +0.43** — comparable to long (0.37) and pooled (0.544).
- **Enthalpy-only LOO on ≥13 (contacts/burial/H-bond/salt only): r = +0.27, POSITIVE.** The enthalpic
  physics genuinely tracks long-peptide affinity. Strongest signals: `org_density` (−0.41), `rg_per_L`
  (+0.30) — compactness/organization (intensive, transferable). `bsa_hyd` is −0.23 (extensive, over-counts
  for long peptides — Simpson), but the intensive features carry it.

### There is NO missing length-scaled entropy term (I was wrong about that too)
If a conformational-entropy penalty (∝ length × disorder) were missing, the residual would correlate with
length/flexibility. It does **not**:
- corr(residual, length) = **+0.009**, corr(residual, org_density) = −0.001, corr(residual, rg_per_L) = +0.001
- within ≥13: all ≈ 0; mean residual +0.06 kcal, 52% over-predicted (balanced, no bias)
- Adding an explicit `len×(1−org)` entropy term: pooled 0.544 → 0.541 (no help); raw length: 0.535 (hurts)

→ My earlier "s_free entropy is the vlong fix" is **unsupported on this ranking task.** (s_free remains a
general +0.08 pooled lever, but it is not specifically a length/vlong correction.)

## Verdict
vlong never failed physics. The cr65 −0.5 was flat/duplicated labels (8 unique Kd in 15). Given real
affinity range, the enthalpic features predict long-peptide ΔG fine (combined vlong 0.43, enthalpy-only
0.27). **The bottleneck is DATA RANGE/VOLUME, not a missing physical term.**

## The data door — BioLiP (this is how we beat PPI-Affinity 0.554)
PPI-Affinity's peptide set comes from **BioLiP** (Yang/Zhang lab, freely downloadable, HTTP 200 verified),
not a private source. Their recipe (from the paper, verbatim from Ram):
- Download nonredundant BioLiP (105,152 entries; incorporates PDBbind protein-ligand).
- Keep protein–peptide complexes with <90% identity between binding-site residues and full receptor seq.
- Single-chain receptor; peptide of standard residues, length ≥3; drop PTMs and fusion constructs.
- Keep only Kd or Ki; drop ambiguous (ranges); keep ΔG ∈ [−14.4, −3.6] kcal/mol.
- **Result: 1149 protein–peptide complexes**, peptides 3–29 aa, receptors 31–957 aa.

**Implications:**
1. **Reproducible by us** — BioLiP is public; the filters are explicit. We can rebuild the same ~1149 set.
2. **~7× our current 156** — gives every length band real affinity range (fixes the degenerate-vlong
   problem at the source) and the volume to fit/validate honestly.
3. **Reopens the M2 "data is sparse (~15–25)" verdict** — that was too pessimistic; BioLiP yields ~1149
   with Kd/Ki (caveat: includes Ki, noisier; families remain after the 90% dedup — true unique count TBD).
4. **Fair-comparison logic** — to beat 0.554 we should train/test on the *same data class* PPI-Affinity
   used. Our physics already hits 0.544 on 156; on matched BioLiP volume it should be competitive.

**Next:** pull BioLiP, apply the filters, dedup hard (sequence + binding-site), score our 16 features +
ML-best-5 poses, and grade pooled LOO vs PPI-Affinity's reported numbers on the same split.
