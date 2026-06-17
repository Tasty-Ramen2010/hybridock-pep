# HybriDock-Pep scoring scorecard — where we stand (consolidated)

**Date:** 2026-06-17. Consolidates e260–e280. Read this first; it is the single source of truth for what
works, what's dead, and on which dataset. Three axes — they are **different problems**, do not conflate.

---

## Axis 1 — Absolute Kd (the hard problem). Honest CV matters more than the headline number.

Random K-fold leaks receptors (same receptor in train+test) → optimistic. Leave-receptor-out is honest.

| dataset | optimistic r (leaky) | **honest r** (leave-receptor-out) | honest MAE |
|---|---|---|---|
| PPIKB (n=1274) | 0.608 | **0.259** | 2.03 |
| PDBbind peptides (n=925) | 0.413 | **0.398** | 1.35 |
| crystal-65 (prior) | 0.576 (LOO) | ~0.35 (clustered) | ~1.8 |
| PPIKB fresh-429 (leave-receptor) | — | 0.346 | 1.99 |

- **Honest absolute ceiling ≈ 0.26–0.40 for everyone, PPI-Affinity included** (its 0.554/0.63 are on
  homology-redundant splits; clustered ≈0.35). The big PPIKB "0.608" is a redundancy mirage (e280).
- **Head-to-head vs PPI-clone v2** (fresh PPIKB n=429, honest): OURS r=0.346/MAE1.99 vs PPI-clone
  r=0.309/MAE1.94 — **we lead r, tie MAE.** Competitive at the shared ceiling (e272).
- **Charged subset:** FEP-bound (cancellation of large terms). No static/ML/short-MD method cracks it.

**Verdict:** absolute Kd is a shared, FEP-bound ceiling. We are at/above parity with the best available
tool, but nobody beats ~0.35 honestly. Absolute Kd is **not** where we win.

## Axis 2 — Same-receptor anchoring / few-shot (THE WIN). This is our exclusive lane.

When ≥1 known-Kd peptide exists on the query receptor, the FEP-bound offset `b(R)` cancels.

| method | r | MAE | note |
|---|---|---|---|
| within-receptor COLD (absolute) | 0.250 | — | no anchor |
| **same-receptor anchored (simple)** | **0.614** | 1.67 | e280; the lever |
| I2 probe-fingerprint, K=2 probes | 0.513 | 1.94 | e278 |
| I2 probe-fingerprint, K=3 probes | 0.522 | 1.88 | measure 2–3 → deploy |
| I2 probe-fingerprint, K=5 probes | 0.548 | 1.95 | |
| pairwise/Siamese ΔΔG (C2) | 0.618 | 1.62 | ties simple — no gain |
| + MISATO MD dynamics (758 only) | +0.066 over static | | needs real MD |

**Verdict:** anchoring **doubles** within-receptor r (0.25→0.61) and is the iGEM mode-(b) deployment
protocol: measure 2–3 references on your actual target (PfLDH/hLDH) → r≈0.52. **Wired**
(`scoring/anchoring.py`, 6 tests). Simple subtraction is already optimal (Siamese/shrinkage don't beat it).

## Axis 3 — Cross-receptor transfer (CLOSED — do not re-open). The receptor offset `b(R)`.

`b(R)` std = 2.14 kcal/mol, and it does **not** transfer or learn by anything we tried:

| attempt | result | evidence |
|---|---|---|
| sequence-homolog anchoring | fails (n=14 group-B r=0.05) | e268 |
| peptide-similarity transfer | r 0.238 < absolute 0.280 | e269 |
| pocket-pkf similarity anchoring | no gain | e270 |
| offset-transfer corr (best metric, pocket-3D) | +0.084 (<1% variance) | e271 |
| full bake-off (M1–M4 ± fallback) | none beats ML on MAE | e272 |
| clean no-cheat + 5-ref combo | **SHUFFLE ≥ similar** = zero signal | e273 |
| same-peptide-diff-protein (pure wall) | \|Δy\| 2.01 mean / 8.53 max kcal/mol | e274 |
| directly LEARN b(R) (GBT/Ridge) | r≈0, worse than predict-mean | e276 |
| short MD (0.1–0.6 ns) for the offset | GIST < null; thermodynamically can't | GIST/e275 |

**Verdict:** the offset is FEP-bound and idiosyncratic. Cross-receptor transfer is impossible by any
similarity, learning, averaging, or short MD. **Closed from ~10 angles.** Not even FEP does it cheaply.

## Axis 4 — Within-target ranking / selectivity (SHIPPED). Offset cancels (shared constant).

- Within-target pose/peptide **ranking** needs no anchor (offset is a shared constant → cancels):
  pose-ranker τ≈0.41 (LOCO real poses), charge-complementarity for salt-bridge selectivity. **Shipped.**
- Cross-target selectivity `ΔΔG = [S(P,A)−b̂(A)]−[S(P,B)−b̂(B)]` needs anchors on **both** targets;
  propagated RMSE ≈ 1.9 kcal/mol (≤2.0 target). Math ready; needs the wet-lab references.

---

## Idea ledger (this session) — every idea, status, number

| idea | axis | status | number |
|---|---|---|---|
| Same-receptor anchoring | 2 | ✅ **WIN** | r 0.25→0.61 |
| I2 probe-peptide fingerprint | 2 | ✅ **WIN** | 2–3 probes → r 0.52 |
| MISATO MD dynamics | 1 | ✅ win (MD-only, 758) | +0.066 r |
| I6 mixed-effects shrinkage | 2 | ◐ marginal | +MAE, −r (1-anchor) |
| C2 Siamese ΔΔG | 2 | ◐ tie | 0.618 vs 0.624 |
| C1 analytical dynamics (rotamer) | 1 | ✗ dead | Δr −0.003 |
| MISATO coarse-charge | 1(charged) | ✗ dead | water stripped |
| Cross-receptor transfer (all metrics) | 3 | ✗ dead | shuffle ≥ similar |
| Learn b(R) | 3 | ✗ dead | r≈0 < mean |
| Short MD for offset | 3 | ✗ dead | GIST < null |

**Live next (docs/scoring_ideas_brainstorm.md):** C3 MISATO-frame Boltzmann ensemble (758), C2×I2 for the
mode-(b) hero result, routed water for hydrophobic-enclosed pockets only. Defer GNN/ESM.

## Bottom line for iGEM
- **Don't sell absolute Kd** (shared ~0.35 ceiling; we're at parity with PPI, nobody wins).
- **Sell same-receptor anchoring + selectivity** (Axis 2/4): it's our exclusive, validated lane —
  doubles within-receptor r, runs where PPI can't (no pose engine), and matches the deployment plan
  (measure 2–3 references on the target). Cross-receptor transfer is honestly, provably closed.
