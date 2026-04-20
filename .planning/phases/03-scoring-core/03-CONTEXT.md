# Phase 3: Scoring Core - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement per-pose scoring for Vina and AD4, apply backbone entropy correction, and produce a calibrated hybrid score. Delivers `scoring/vina.py`, `scoring/ad4.py`, `scoring/entropy.py`, and `scripts/calibrate_alpha.py`. Phase ends when every pose can be independently scored and a calibrated `calibration.json` is written. Sampling (RAPiDock), clustering, and CLI orchestration are NOT in scope.

</domain>

<decisions>
## Implementation Decisions

### Hybrid score formula
- **D-01:** Final hybrid score formula: `hybrid = vina + β×(ad4 − vina) + α×n_residues`
  - Vina is the primary scorer. The `β×(ad4−vina)` term captures the electrostatics delta without letting AD4 dominate. `α×n_residues` is the backbone entropy penalty.
  - When β=0: hybrid = vina (pure Vina). When β=1: hybrid = ad4 (pure AD4). β is calibrated to find the right balance.
- **D-02:** `n_residues = len(peptide_sequence)` — full peptide length, not contact-zone residues. Simpler, no contact detection needed in scoring phase.
- **D-03:** Sign convention: α is positive. Longer peptides pay a larger entropy penalty, making the hybrid score *less* negative. Physically correct: binding a longer, more flexible peptide costs more conformational entropy.
- **D-04:** α validated in `[0.2, 1.2]` kcal/mol/residue (per spec §8). If calibrated α falls outside this range, abort with diagnostic message. This is SCORE-03.
- **D-05:** β validated in `[0.0, 0.5]`. β > 0.5 means AD4 dominates over Vina, which contradicts the Vina-primary design. Abort with diagnostic if outside range.
- **D-06:** AD4 score > 0 → set `ScoredPose.is_ad4_anomaly = True` and flag in output (per SCORE-02). The pose is still scored and ranked — anomaly flag is informational, not a filter.
- **D-07:** Scoring failures (exception from Vina or AD4 API) → `PoseFailure(pose_idx, stage="scoring", error_msg=str(e))`. Batch continues. Scorer returns `(list[ScoredPose], list[PoseFailure])`. Consistent with the prep/ligand.py pattern.

### α and β calibration
- **D-08:** `training_complexes.csv` columns: `pdb_id`, `peptide_sequence`, `experimental_pkd`. The pKd column uses −log₁₀(Kd/mol).
- **D-09:** pKd → ΔG conversion: `ΔG = −RT × ln(10^−pKd)` = `−0.592 × pKd` kcal/mol at 298 K (R = 1.987 cal/mol/K). Conversion applied inside `calibrate_alpha.py` — training CSV stores raw pKd.
- **D-10:** Fit α and β jointly via `scipy.optimize.minimize` (method='L-BFGS-B'), minimizing sum of squared residuals `Σ(hybrid_i − ΔG_i)²` over the training set. Bounds: α ∈ [0.2, 1.2], β ∈ [0.0, 0.5] enforced by optimizer. Core fitting function lives in `scoring/entropy.py`; `scripts/calibrate_alpha.py` is a thin wrapper.
- **D-11:** `calibration.json` schema:
  ```json
  {
    "alpha": 0.65,
    "beta": 0.22,
    "n_complexes": 10,
    "pearson_r": 0.71,
    "rmse_kcal_mol": 1.2,
    "calibrated_at": "2026-04-20T...",
    "training_csv": "data/training_complexes.csv"
  }
  ```
- **D-12:** `scripts/calibrate_alpha.py` is a standalone script (direct `python scripts/calibrate_alpha.py`) AND wired as the `hybridock-pep calibrate` subcommand. The core fitting function lives in `scoring/entropy.py` so both entry points call the same code.
- **D-13:** Ship `data/calibration.json` in the repo with α≈0.65, β≈0.22 as a literature-reasonable default. `dock` reads this file at startup and works out of the box. Users can override by running `hybridock-pep calibrate`. This is required for the iGEM tutorial notebook to run top-to-bottom without a separate calibration step.

### Claude's Discretion
- Per-pose Vina+AD4 parallelism strategy (ThreadPoolExecutor vs sequential within the score batch) — use whatever achieves the 5-min wall-clock target on RTX 5070 + modern CPU.
- Exact Vina Python API call pattern (vina.Vina() instance lifecycle, receptor/ligand loading order).
- Whether to use `concurrent.futures` or `asyncio` for the Vina+AD4 per-pose dual-scoring.
- Grid boundary check implementation details (SCORE-01): validate atom x/y/z against DockConfig site_coords ± box_size/2; set `is_clipped=True` on ScoredPose and log to run_metadata.json.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Technical specification
- `docs/HybriDock-Pep_Technical_Specification.pdf` §4, §5 — Vina charge handling, AD4 parallel scoring, why Coulomb term is rejected
- `docs/HybriDock-Pep_Technical_Specification.pdf` §8 — Backbone entropy correction formula and α range [0.2, 1.2]
- `docs/HybriDock-Pep_Technical_Specification.pdf` §11, §12 — Scoring pipeline architecture

### Requirements
- `.planning/REQUIREMENTS.md` SCORE-01 — Vina Python API, grid boundary validation, clipped pose logging
- `.planning/REQUIREMENTS.md` SCORE-02 — AD4 parallel scoring, positive AD4 score anomaly flag
- `.planning/REQUIREMENTS.md` SCORE-03 — Entropy correction, α range validation [0.2, 1.2], abort if outside

### Project constraints
- `CLAUDE.md` §2.1 — Vina does NOT use partial charges (q column ignored). AD4 does use Gasteiger charges (`vina --scoring ad4`). This is why AD4 runs in parallel — it's not redundant.
- `CLAUDE.md` §2.4 — All scoring code runs in score-env (Python 3.11). No Python 3.10+ syntax restrictions here (only rapidock_runner.py is 3.9-restricted).

### Prior phase context
- `.planning/phases/02-preparation/02-CONTEXT.md` — prep outputs: `output_dir/receptor.pdbqt`, `output_dir/poses/*.pdbqt`, `output_dir/maps/` (AD4 affinity maps)
- `src/hybridock_pep/models.py` — `ScoredPose` fields: `vina_score`, `ad4_score`, `entropy_correction`, `hybrid_score`, `is_ad4_anomaly`, `is_clipped`, `pdbqt_path`; `DockConfig` fields: `site_coords`, `box_size`, `scoring`, `output_dir`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/hybridock_pep/models.py` `ScoredPose` — all score fields pre-defined: `vina_score`, `ad4_score`, `entropy_correction`, `hybrid_score`, `is_ad4_anomaly`, `is_clipped`. Scoring modules fill these fields; don't add new ones without checking models.py first.
- `src/hybridock_pep/models.py` `PoseFailure(stage="scoring")` — use for per-pose scoring failures; same as prep pattern.
- `src/hybridock_pep/prep/ligand.py` `prepare_ligand_batch()` — reference implementation for collect-all-failures batch pattern with ProcessPoolExecutor. Scoring batch should follow the same (scored, failures) return convention.
- `src/hybridock_pep/scoring/__init__.py` — stub exists, empty.

### Established Patterns
- `from __future__ import annotations` first line of every module.
- `logger = logging.getLogger(__name__)` + `logger.info("Running: %s", cmd)` before every external call.
- No bare `except:` — catch specific exceptions, reraise with context.
- Google-style docstrings with Args, Returns, Raises.
- Type hints everywhere; mypy strict mode.

### Integration Points
- `scoring/vina.py` → consumes `output_dir/receptor.pdbqt` + `output_dir/poses/{pose_idx}.pdbqt`, uses `DockConfig.site_coords` and `box_size` for grid boundary check.
- `scoring/ad4.py` → consumes `output_dir/maps/` (written by `prep/grids.py`), reads `.map` files via Vina API `--scoring ad4` mode.
- `scoring/entropy.py` → reads `calibration.json` (α, β), takes `ScoredPose` with vina+ad4 filled, computes and sets `entropy_correction` and `hybrid_score`.
- `scripts/calibrate_alpha.py` → reads `data/training_complexes.csv`, runs full scoring pipeline on each complex, fits α and β, writes `calibration.json`.
- `driver.py` Stage 2 → calls the three scoring modules in sequence per pose; aggregates (ScoredPose list, PoseFailure list).

</code_context>

<specifics>
## Specific Ideas

- Hybrid formula verbatim from discussion: `hybrid = vina + β×(ad4−vina) + α×n_residues`. The `β×(ad4−vina)` term is described by Ram as "the electrostatics delta" — it captures the charge penalty without letting AD4 dominate.
- Default calibration.json values: α≈0.65, β≈0.22. These are literature-reasonable starting points, not validated — users should run calibrate on their training set for production use.
- pKd→ΔG conversion at T=298K: `ΔG = −0.592 × pKd` kcal/mol. Hard-code T=298K for now (not a CLI flag).
- Ship `data/calibration.json` in the repo — required for tutorial notebook to run without a pre-calibration step.

</specifics>

<deferred>
## Deferred Ideas

- MM-GBSA OpenMM minimization before Vina scoring (CLAUDE.md §2.5 workaround for ref2015 cysteine issue) — deferred to Phase 7/OPT-01 unless it proves necessary for SCORE-01 to pass.
- Per-pose Vina+AD4 parallelism strategy — left to Claude's discretion; optimize for the 5-min wall-clock target.
- Temperature as a CLI flag for pKd→ΔG conversion — deferred to v2; T=298K hardcoded for now.

</deferred>

---

*Phase: 03-scoring-core*
*Context gathered: 2026-04-20*
