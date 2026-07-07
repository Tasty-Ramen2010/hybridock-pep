# External validation — scoring & ranking fresh literature peptides (2026-07-06)

Blind `crystal-score` on peptide–protein complexes pulled straight from the PDB/literature, **none in any
training split**. Two questions: (1) does the absolute ΔG land near the truth, and (2) can it *rank* a panel
of candidate peptides against one target (the "which do I test?" use case)? Every ΔG_exp is converted from
the literature K_d via ΔG = RT·ln(K_d) at 298 K. Honest results — including where it fails.

## Series 1 — MDM2 inhibitor panel (one target, ~160× affinity range): RANKING WORKS

| peptide | PDB | K_d | ΔG_exp | pred ΔG |
|---|---|---|---|---|
| p53 wt (ETFSDLWKLLPE) | 1YCR | 160 nM | −9.28 | −9.28 |
| p53 wt (2nd crystal) | 4HFZ | 160 nM | −9.28 | −9.48 |
| PMI-N8A (TSFAEYWALLS) | 3LNZ | ~50 nM | −9.97 | −9.98 |
| PMI (TSFAEYWNLLS) | 3EQS | 3.3 nM | −11.58 | −9.67 |
| pDIQ (ETFEHWWSQLLS) | 3JZS | 1.0 nM | −12.29 | −9.59 |

**Within-target Spearman ρ = +0.56.** It correctly separates the weak wild-type p53 peptides from the
optimised inhibitors — useful for "do my candidates beat wild-type?" — but **saturates among the sub-10 nM
binders** (PMI/pDIQ/PMI-N8A all score ~−9.6 to −10.0), so it cannot fine-rank the tight tail.

## Series 2 — Bcl-xL / BH3 panel (one target): RANKING FAILS (backwards)

| peptide | PDB | K_d | ΔG_exp | pred ΔG |
|---|---|---|---|---|
| Bak BH3 | 1BXL | 340 nM | −8.83 | −8.73 |
| Bim BH3 | 3FDL | 10 nM | −10.92 | −8.57 |
| Bad BH3 | 2BZW | 0.6 nM | −12.59 | −8.58 |
| Bad BH3 (2nd crystal) | 1G5J | 0.6 nM | −12.59 | −7.16 |

**Within-target Spearman ρ = −0.63 (anti-correlated).** It scores the *weakest* binder (Bak) as the
strongest and the tightest (Bad) as weakest, and the *same* Bad peptide scores −8.58 vs −7.16 across two
crystals. BH3 peptides are long amphipathic helices that all bury similarly in the groove; their affinity
differences come from sequence-specific electrostatics the geometry/IFP features do not capture.

## Verdict — honest and consistent

Within-target ranking is **target-dependent**: strong on aromatic/hydrophobic pockets (MDM2, ρ = +0.56),
unreliable-to-backwards on electrostatically-modulated helical grooves (BH3, ρ = −0.63). This is the two
tails of the 865-complex distribution (median ρ = 0.50, but ~20–30 % of targets rank in the wrong
direction — E306), and it matches the long-standing finding that the tool works where binding is
H-bond/aromatic-driven and we cannot always predict a priori which target that is. Absolute ΔG stays
compressed near −9 for both series regardless of the true −7 to −12.6 range — the blind-absolute ceiling.

**Practical guidance for other projects:** use the ranking to *prioritise* a panel and expect it to nail the
weak-vs-optimised split, but confirm the tight tail in the wet lab, and add 2–3 measured references on your
own target (reference-anchoring, r 0.25→0.61) when you need calibrated numbers. Do not trust a single
absolute ΔG.

## Live `dock` confirmation of the `rank_score` column (E309, wired)

Ran the MDM2 series through the **full `dock` pipeline** (`--input-poses` = each crystal pose) and read the
emitted `rank_score` column vs the default `delta_g`:

| peptide | K_d | `delta_g` | `rank_score` |
|---|---|---|---|
| p53 wt (1YCR) | 160 nM | −9.82 | −9.24 |
| p53 wt (4HFZ) | 160 nM | −9.23 | −9.53 |
| PMI-N8A (3LNZ) | 50 nM | −8.52 | −9.99 |
| PMI (3EQS) | 3.3 nM | −8.38 | −9.76 |
| pDIQ (3JZS) | 1 nM | −9.10 | −9.90 |

**Spearman vs experimental ΔG (n=5): `rank_score` +0.667 (correct) vs `delta_g` −0.667 (backwards).** In the
live pipeline the default `delta_g` (AI-pose affinity model) ranks this crystal-pose panel backwards; the
`rank_score` column rescues it and correctly places both wild-type p53 peptides below the optimised
inhibitors. n=5 and the sub-10 nM tail is still unranked (saturation), but the wired column reproduces the
E309 ranking signal end-to-end. Screening path validated; RAPiDock pose generation not exercised here.

## Independent panels — SH3 and PDZ (leave-receptor-out `rank_score`)

Two more real multi-peptide panels from the PDBbind crystal set, scored with the composition-IFP
`rank_score` **leave-receptor-out** (the model never trains on that target — a true-novel simulation):

| target | family | n peptides | Spearman | pairwise | verdict |
|---|---|---|---|---|---|
| Sem-5 / Grb2 SH3 (1prm/1prl/1qwe…) | proline-rich PxxP | 6 | **+0.91** | **92%** | strong |
| PDZ domain (4e34/4joj/4k6y…) | C-terminal motif | 8 | +0.26 | 62% | modest |

**SH3 is the strongest single-panel result yet** (+0.91) — proline-rich peptides bind by hydrophobic PxxP
packing, the shape-driven regime `rank_score` reads well (like MDM2). **PDZ is modest** (+0.26): its peptides
differ by single residues (SR**W**/**F**/**V**/**A**QTSII) and PDZ affinity is set by that side-chain readout,
exactly what the shape-dominated scorer under-reads (E308) — the tightest binder (ANSRWPTSII, −8.20) lands
mid-pack.

## Consolidated picture — `rank_score` is target-dependent, predictably

| panel | Spearman | binding driver |
|---|---|---|
| SH3 (Sem-5) | +0.91 | hydrophobic / proline packing ✅ |
| MDM2 inhibitors | +0.67 | hydrophobic Phe-Trp-Leu pocket ✅ |
| PDZ | +0.26 | single-residue C-terminal side-chain ◐ |
| Bcl-xL / BH3 | −0.63 | amphipathic-helix electrostatics ✗ |

The verdict tracks the binding mechanism, not chance: `rank_score` is reliable on shape/hydrophobic-driven
grooves (SH3, MDM2) and weak-to-backwards where affinity comes from side-chain chemistry the scorer under-reads
(PDZ, BH3). Use it to prioritise panels on hydrophobic/aromatic targets; treat charged/electrostatic helical
targets with caution and confirm in the wet lab.
