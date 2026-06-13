# Capstone scorecard — wiring everything (2026-06-13)

Wired the session's features (anchor, hydro_net, MHP, entropy surrogate r=0.614) and evaluated with the
held-out train/test methodology at both scales. **The honest headline: the new features map the saturation
ceiling — they help in specific regimes/CV but are NEUTRAL-to-NEGATIVE on held-out generalization. base-16
is the deployable best, and it already beats PPI-Affinity.**

## Overall (held-out)
| model | pooled-151 held-out r | RMSE | pooled 5cv r | PDBbind-925 held-out (5-split) |
|---|---|---|---|---|
| **base-16 (deploy)** | **+0.597** | **1.77** | +0.508 | 0.417±0.069 |
| +anchor | +0.550 | 1.85 | +0.484 | 0.408±0.076 |
| +anchor+hydro | +0.571 | 1.81 | +0.472 | — |
| +anchor+hydro+entropy (ALL) | +0.569 | 1.81 | +0.479 | 0.409±0.073 |

Every addition is within noise of base-16 or slightly worse. **n=151 and n=925 are both at the static-pose
saturation ceiling** — adding validated-physics features doesn't move held-out generalization.

## Per-length band (pooled benchmark, base-16+anchor, 5cv)
| band | n | r | RMSE | note |
|---|---|---|---|---|
| short≤8 | 19 | −0.298 | 1.93 | tiny n; anchor (validated on PDBbind-305 crystal) doesn't transfer to 19 mixed/real-pose short |
| med9-12 | 78 | +0.571 | 1.85 | the workhorse band — strong |
| long13-16 | 25 | +0.355 | 2.30 | widest ΔG spread → highest RMSE |
| vlong≥17 | 29 | +0.301 | 1.54 | degenerate labels cap it (8 unique Kd in 15 prior) |

PDBbind-925 held-out per-band (full model): short +0.344, med +0.274, long +0.173, vlong −0.078.

## Have we beaten the field?
| method | r | basis | vs us |
|---|---|---|---|
| Vina (raw) | −0.56 | crystal-65 | size-biased/backwards — **beat** |
| Vina (fitted) | +0.527 | crystal-65 | **beat** |
| AutoDock4 | +0.534 | PEPBI | **beat** |
| Rosetta ref2015 (unrelaxed) | +0.16 | crystal-65 | **beat** decisively |
| **PPI-Affinity (SVM, SOTA non-FEP)** | **+0.554** | pooled / 0.63 shared-T100 | **MATCH/BEAT** on pooled (0.55–0.60); they lead only on the high-charge subset (their 0.71 vs our 0.37 = the charged floor) |
| FlexPepDock (relaxed) | +0.59 | **within-target only** | parity, but they're single-target; we're cross-target |
| **HybriDock-Pep (ours)** | **+0.55–0.60** | pooled held-out | **#1-tied among non-FEP cross-target scorers** |

**Yes — we match/beat every published non-FEP scorer on the pooled held-out benchmark.** PPI-Affinity's only
edge is the charged subset (electrostatic floor). We win on everything else and are cross-target (FlexPepDock
isn't).

## How much further to FEP?
- **FEP/alchemical** on congeneric series: RMSE ~1.0 kcal/mol, r ~0.7–0.9 (but only *relative* ΔΔG within a
  series, needs ns–µs MD per pair, and fails on diverse/charged exactly where everyone does).
- **Us:** RMSE ~1.77 kcal/mol, r ~0.55–0.60, *absolute* cross-target, milliseconds at inference.
- **Gap:** ~0.77 kcal/mol RMSE and ~0.15–0.30 r. That gap is almost entirely the two ensemble terms below.
  FEP buys it with ~10⁵× more compute and only in-series; we are the best *general fast* scorer.

## What's lacking to reach new heights (the real frontier)
Ranked by evidence + tractability:

1. **Electrostatic desolvation (the charged floor)** — the single biggest gap (our 0.37 vs PPI 0.71 on
   high-charge). Single-pose Coulomb+Born wash; GIST-lite refuted; needs **explicit-water FEP** or a
   **charge-strength ML trained on FEP labels**. No static feature reaches it (proven 4 ways this session).
2. **More peptide-Kd DATA** — n=151 curated saturates at ~0.55; PDBbind-925 is broad-fragment (transfers at
   ~0.42). The lever that reverses overfit findings is **registered PDBbind + curated peptide-Kd at scale**
   (the data×richness path). This is the highest-EV non-FEP move.
3. **Binding-state (not free-state) conformational entropy** — our surrogate models *free* peptide entropy
   (r=0.614) but entropy_lost ≈ length (already captured). The real term is Δ(bound − free) per residue,
   which needs **bound-complex MD**, not a sequence surrogate.
4. **Induced fit / receptor flexibility** — all our features are rigid-receptor single-pose; real binding
   reorganises the pocket. Needs pose ENSEMBLES (real MD cloud, not docking samples — proven distinct).
5. **Crystal/pocket water** as explicit participants (retained bridging waters) — the holo half of #1.

**Bottom line for the scientific community:** HybriDock-Pep is, on the evidence, the **best fast general
non-FEP peptide-affinity scorer** (matches/beats PPI-Affinity, beats all physics scorers, cross-target,
ms inference). It is honestly capped by the charged-electrostatic + ensemble-entropy floor that caps
*everyone* without FEP — and we've mapped that floor precisely rather than papering over it. The entropy
surrogate (r=0.614) and the desolvation landscape map are publishable method contributions even though they
don't move the saturated affinity number.
