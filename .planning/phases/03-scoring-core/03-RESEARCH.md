# Phase 3: Scoring Core - Research

**Researched:** 2026-04-20
**Domain:** Molecular scoring via Vina Python API (Vina + AD4 modes) + scipy calibration
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Hybrid formula: `hybrid = vina + β×(ad4 − vina) + α×n_residues`
- **D-02:** `n_residues = len(peptide_sequence)` — full peptide length
- **D-03:** α is positive; longer peptides pay a larger entropy penalty
- **D-04:** α validated in [0.2, 1.2] kcal/mol/residue; abort if outside
- **D-05:** β validated in [0.0, 0.5]; abort if outside
- **D-06:** AD4 score > 0 → `ScoredPose.is_ad4_anomaly = True` (informational, not a filter)
- **D-07:** Scoring failures → `PoseFailure(pose_idx, stage="scoring", error_msg=str(e))`. Scorer returns `(list[ScoredPose], list[PoseFailure])`
- **D-08:** Training CSV columns: `pdb_id`, `peptide_sequence`, `experimental_pkd`
- **D-09:** pKd → ΔG: `ΔG = −0.592 × pKd` kcal/mol at T=298K (hardcoded)
- **D-10:** Joint α+β fit via `scipy.optimize.minimize` (method='L-BFGS-B'); minimize `Σ(hybrid_i − ΔG_i)²`; bounds α ∈ [0.2, 1.2], β ∈ [0.0, 0.5]
- **D-11:** `calibration.json` schema: `{alpha, beta, n_complexes, pearson_r, rmse_kcal_mol, calibrated_at, training_csv}`
- **D-12:** Core fitting function in `scoring/entropy.py`; `scripts/calibrate_alpha.py` is a thin CLI wrapper; `hybridock-pep calibrate` calls the same function
- **D-13:** Ship `data/calibration.json` with α≈0.65, β≈0.22 as literature-reasonable defaults

### Claude's Discretion
- Per-pose Vina+AD4 parallelism strategy (ThreadPoolExecutor vs sequential within score batch)
- Exact Vina Python API call pattern (instance lifecycle, receptor/ligand loading order)
- Whether to use `concurrent.futures` or `asyncio` for Vina+AD4 dual-scoring per pose
- Grid boundary check implementation details: validate atom x/y/z from PDBQT against DockConfig site_coords ± box_size/2; set `is_clipped=True` on ScoredPose

### Deferred Ideas (OUT OF SCOPE)
- MM-GBSA OpenMM minimization before Vina scoring (Phase 7/OPT-01)
- Temperature as CLI flag for pKd→ΔG (v2)
- Per-pose Vina+AD4 parallelism optimization details (Claude's discretion)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SCORE-01 | Score each pose with `vina --score_only` via Vina Python API; validate all atoms against grid bounds before scoring; log clipped poses to run_metadata.json (never silently drop) | Vina Python API: `Vina(sf_name='vina')`, `set_receptor()`, `load_maps()` or `compute_vina_maps()`, `set_ligand_from_file()`, `score()` — see Pattern 1 |
| SCORE-02 | Score each pose with `vina --scoring ad4` in parallel with Vina; flag positive AD4 scores as anomalies | AD4 Python API: `Vina(sf_name='ad4')`, `load_maps(map_prefix)` WITHOUT `set_receptor()`, then `set_ligand_from_file()`, `score()` — see Critical Pitfall below |
| SCORE-03 | Apply backbone entropy correction using calibrated α from `calibration.json`; validate α ∈ [0.2, 1.2]; abort if outside | `scipy.optimize.minimize` L-BFGS-B with bounds; entropy formula in `scoring/entropy.py` |
</phase_requirements>

---

## Summary

Phase 3 implements the per-pose scoring pipeline: two independent Vina calls (one with the standard `vina` scoring function and one with the `ad4` scoring function), followed by a backbone entropy correction that applies a linear correction term calibrated against experimental binding affinities.

The Vina Python API (package `vina`, latest stable 1.2.7) provides a `Vina` class that wraps the C++ backend via SWIG bindings. The API surface is straightforward: constructor with `sf_name`, receptor/ligand loading methods, and `score()` which returns a numpy array. The key architectural distinction is that **Vina mode and AD4 mode require different initialization sequences** — AD4 mode does not call `set_receptor()` at all; it calls `load_maps()` with the autogrid4 map prefix. Conflating these two patterns is the dominant pitfall in this phase.

The calibration workflow uses `scipy.optimize.minimize` with `method='L-BFGS-B'` and explicit bounds to jointly fit α and β against training complexes. The `result.x` array contains `[α, β]` at convergence. The shipped `data/calibration.json` provides defaults so the tutorial notebook runs without a prior calibration step.

**Primary recommendation:** One `Vina` instance per scoring session (Vina mode), reused across all poses by calling `set_ligand_from_file()` each time. One separate `Vina(sf_name='ad4')` instance with maps pre-loaded, also reused. Both run sequentially per pose (not threaded) to avoid C++ state contention — the Vina SWIG bindings have no documented thread safety and share internal state.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Vina score_only per pose | API / Backend (`score-env`) | — | Pure CPU computation; reads PDBQT files from disk |
| AD4 score per pose | API / Backend (`score-env`) | — | Reads pre-computed autogrid4 maps from Phase 2 |
| Grid boundary validation | API / Backend (`scoring/vina.py`) | — | Must happen before Vina call; uses DockConfig.site_coords + box_size |
| Entropy correction + hybrid score | API / Backend (`scoring/entropy.py`) | — | Pure arithmetic on ScoredPose fields; no I/O |
| α+β calibration | API / Backend (`scoring/entropy.py` + `scripts/`) | — | scipy optimization; reads CSV + runs full scoring pipeline |
| Batch orchestration | API / Backend (`driver.py` Stage 2) | — | Calls three scoring modules in sequence; aggregates results |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| vina | 1.2.7 [VERIFIED: PyPI 2026-04-20] | Vina Python API for `score()` calls | Official binding; avoids 200 fork+exec cycles per run |
| scipy | ≥1.11 (score-env) [ASSUMED] | L-BFGS-B calibration optimizer | Standard scientific Python; provides bounded minimize() |
| numpy | ≥1.24 (score-env) [ASSUMED] | Array ops, `score()` return type | Already in score-env; Vina API returns np.ndarray |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | — | Read/write calibration.json | JSON I/O without dependency |
| concurrent.futures | stdlib | Optional ThreadPoolExecutor for Vina+AD4 dual-scoring | Only if sequential proves too slow for 5-min target |
| pearsonr (scipy.stats) | — | Compute Pearson r in calibration output | Reports diagnostic metric in calibration.json |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Vina Python API | subprocess vina --score_only | API avoids 200 fork+exec per run; 2× faster startup |
| scipy L-BFGS-B | scipy Nelder-Mead | L-BFGS-B enforces bounds natively; Nelder-Mead needs penalty trick |
| ThreadPoolExecutor | ProcessPoolExecutor | Vina C++ bindings may not release GIL; process isolation is safer but has pickling overhead |

**Installation:**
```bash
# Already in score-env.yml — verify:
conda run -n score-env pip show vina
```

**Version verification:**
```bash
pip index versions vina  # Latest: 1.2.7 (2025-02-26)
```

---

## Architecture Patterns

### System Architecture Diagram

```
pose_idx  ──▶  [grid boundary check]  ──▶  is_clipped flag
                       │
                       ▼
          pdbqt_path (output_dir/poses/{i}.pdbqt)
                /              \
               ▼                ▼
  Vina(sf_name='vina')    Vina(sf_name='ad4')
  set_receptor(...)       [NO set_receptor call]
  compute_vina_maps(...)  load_maps(maps_dir/'receptor')
  set_ligand_from_file()  set_ligand_from_file()
  score()[0]              score()[0]
      │                       │
      ▼                       ▼
  vina_score            ad4_score
      │                       │
      └───────────┬───────────┘
                  ▼
         [entropy.py: apply_hybrid_score()]
         hybrid = vina + β×(ad4−vina) + α×n_residues
                  │
                  ▼
           ScoredPose (all fields filled)
```

### Recommended Project Structure
```
src/hybridock_pep/
├── scoring/
│   ├── __init__.py          # exports score_pose_batch()
│   ├── vina.py              # Vina mode scorer (SCORE-01)
│   ├── ad4.py               # AD4 mode scorer (SCORE-02)
│   └── entropy.py           # entropy correction + fit_calibration() (SCORE-03)
scripts/
└── calibrate_alpha.py       # thin CLI wrapper calling entropy.fit_calibration()
data/
└── calibration.json         # shipped defaults α≈0.65, β≈0.22
```

### Pattern 1: Vina score_only per pose
**What:** Initialize one `Vina(sf_name='vina')` instance, load receptor once, reuse across all poses by calling `set_ligand_from_file()` per pose.
**When to use:** All poses in the scoring batch for Vina mode (SCORE-01).
**Example:**
```python
# Source: https://autodock-vina.readthedocs.io/en/latest/docking_python.html
from vina import Vina
from pathlib import Path

def score_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    site_coords: tuple[float, float, float],
    box_size: float,
) -> float:
    """Return Vina score_only for one pose. Re-use instance externally for batch."""
    v = Vina(sf_name='vina', verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_file(str(ligand_pdbqt))
    v.compute_vina_maps(
        center=list(site_coords),
        box_size=[box_size, box_size, box_size],
    )
    energies = v.score()  # np.ndarray; index 0 = total kcal/mol
    return float(energies[0])
```

**Note:** `compute_vina_maps()` behavior differs when called before vs. after `set_ligand_from_file()`. Calling it AFTER ligand loading computes maps only for the atom types present in the ligand (more efficient for per-pose scoring). Call order above is correct for score_only. [CITED: autodock-vina.readthedocs.io/en/latest/docking_python.html]

### Pattern 2: AD4 score per pose (CRITICAL — different from Vina mode)
**What:** For AD4 mode, `set_receptor()` must NOT be called. Call `load_maps()` with the autogrid4 map prefix instead.
**When to use:** All poses in AD4 scoring batch (SCORE-02).
**Example:**
```python
# Source: Derived from https://autodock-vina.readthedocs.io/en/latest/vina.html
# and confirmed via GitHub issue investigation (issues #69, #254)
from vina import Vina
from pathlib import Path

def score_ad4(
    maps_dir: Path,
    ligand_pdbqt: Path,
) -> float:
    """Return AD4 score for one pose. maps_dir is output_dir/maps/ from Phase 2."""
    v = Vina(sf_name='ad4', verbosity=0)
    # DO NOT call v.set_receptor() — AD4 mode uses maps, not a receptor PDBQT.
    # Calling set_receptor() with AD4 raises:
    #   "Only flexible residues allowed with the AD4 scoring function. No (rigid) receptor."
    map_prefix = str(maps_dir / "receptor")  # prefix for receptor.C.map, receptor.HD.map, etc.
    v.load_maps(map_prefix)
    v.set_ligand_from_file(str(ligand_pdbqt))
    energies = v.score()  # np.ndarray; index 0 = total kcal/mol
    return float(energies[0])
```

**score() return array for AD4:**
`[total, lig_inter, flex_inter, other_inter, flex_intra, lig_intra, torsions, -lig_intra]`
Index 0 is the total binding energy in kcal/mol. [CITED: github.com/ccsb-scripps/AutoDock-Vina/blob/develop/build/python/vina/vina.py]

**score() return array for Vina:**
`[total, lig_inter, flex_inter, other_inter, flex_intra, lig_intra, torsions, lig_intra_best_pose]`
Same indexing convention. [CITED: github.com/ccsb-scripps/AutoDock-Vina]

### Pattern 3: Grid boundary check for SCORE-01
**What:** Parse PDBQT atom coordinates and check against `site_coords ± box_size/2` before scoring.
**When to use:** Every pose before Vina scoring call.
**Example:**
```python
# PDBQT format: columns 30-37 (x), 38-45 (y), 46-53 (z) — same as PDB format
# Source: PDBQT is a PDB superset; coordinate columns are PDB-standard
def check_grid_boundary(
    pdbqt_path: Path,
    site_coords: tuple[float, float, float],
    box_size: float,
) -> bool:
    """Return True if any atom coordinate falls outside grid bounds."""
    cx, cy, cz = site_coords
    half = box_size / 2.0
    xlo, xhi = cx - half, cx + half
    ylo, yhi = cy - half, cy + half
    zlo, zhi = cz - half, cz + half

    for line in pdbqt_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue  # malformed coordinate line — skip
        if not (xlo <= x <= xhi and ylo <= y <= yhi and zlo <= z <= zhi):
            return True  # at least one atom outside bounds
    return False
```

### Pattern 4: L-BFGS-B joint calibration of α and β
**What:** `scipy.optimize.minimize` with method='L-BFGS-B' and bounds to fit two parameters against training data.
**When to use:** `entropy.fit_calibration()` called by `calibrate_alpha.py` and `hybridock-pep calibrate`.
**Example:**
```python
# Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

def fit_calibration(
    vina_scores: list[float],
    ad4_scores: list[float],
    n_residues_list: list[int],
    experimental_pkd: list[float],
    RT: float = 0.592,  # kcal/mol at 298K
) -> dict[str, float]:
    """Fit α and β jointly via L-BFGS-B minimizing sum-of-squared residuals."""
    delta_g = [-RT * pkd for pkd in experimental_pkd]  # D-09: ΔG = -0.592 × pKd

    def objective(params: np.ndarray) -> float:
        alpha, beta = params
        residuals = []
        for vina, ad4, n_res, dg in zip(vina_scores, ad4_scores, n_residues_list, delta_g):
            hybrid = vina + beta * (ad4 - vina) + alpha * n_res
            residuals.append(hybrid - dg)
        return sum(r**2 for r in residuals)

    x0 = np.array([0.65, 0.22])  # start from shipped defaults
    bounds = [(0.2, 1.2), (0.0, 0.5)]  # D-04, D-05
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    alpha, beta = result.x

    # Compute diagnostics
    hybrids = [v + beta*(a-v) + alpha*n for v, a, n in zip(vina_scores, ad4_scores, n_residues_list)]
    r, _ = pearsonr(hybrids, delta_g)
    rmse = float(np.sqrt(np.mean([(h-d)**2 for h, d in zip(hybrids, delta_g)])))

    return {"alpha": float(alpha), "beta": float(beta), "pearson_r": float(r), "rmse_kcal_mol": rmse}
```

### Anti-Patterns to Avoid
- **Calling `set_receptor()` with `sf_name='ad4'`:** Raises "Only flexible residues allowed with the AD4 scoring function." AD4 mode must use `load_maps()` instead.
- **Creating a new Vina instance per pose:** Vina instance initialization loads the receptor PDBQT (for Vina mode) or maps (for AD4 mode) each time. Create one instance per scoring session, reuse across poses via `set_ligand_from_file()`.
- **Using `compute_vina_maps()` in AD4 mode:** AD4 requires pre-computed autogrid4 maps. `compute_vina_maps()` computes Vina-style maps, not AD4 affinity maps — calling it in AD4 mode will produce incorrect scores.
- **Threading Vina instances:** The Vina SWIG bindings have no documented thread safety and no visible GIL-release macros. Multiple Vina instances in threads may cause undefined C++ state. Use sequential scoring per pose or ProcessPoolExecutor with one instance per process.
- **Extracting per-atom charge contributions from Vina scores:** Vina ignores the q column in PDBQT. The `score()` array does not break out electrostatic contributions for Vina mode. This is why AD4 runs in parallel (CLAUDE.md §2.1).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Molecular docking scoring | Custom scoring function | `vina` Python API | Implements force field correctly; handles all atom types |
| Bounded parameter optimization | Custom gradient descent | `scipy.optimize.minimize` L-BFGS-B | Handles bounds, convergence, gradient estimation |
| Pearson correlation | Manual implementation | `scipy.stats.pearsonr` | Edge cases with constant arrays |
| PDBQT coordinate parsing | Full PDBQT parser | Inline fixed-column parsing (cols 30-54) | PDBQT is PDB-format; coordinates are always fixed-width |

**Key insight:** Calibration is numerically sensitive. The bounds on α and β are load-bearing constraints from the spec (§8) — they are not just documentation. L-BFGS-B natively respects them without penalty tricks.

---

## Common Pitfalls

### Pitfall 1: Calling set_receptor() in AD4 mode
**What goes wrong:** `Vina(sf_name='ad4')` then `v.set_receptor(...)` raises "Only flexible residues allowed with the AD4 scoring function. No (rigid) receptor."
**Why it happens:** AD4 scoring in Vina is designed around pre-computed autogrid4 affinity maps, not on-the-fly rigid receptor computation. The Vina C++ backend enforces this.
**How to avoid:** For `sf_name='ad4'`, call `load_maps(map_prefix)` instead of `set_receptor()`. The map prefix should be `str(maps_dir / "receptor")` — this resolves to `maps_dir/receptor.C.map`, `maps_dir/receptor.HD.map`, etc.
**Warning signs:** `AssertionError` or `RuntimeError` mentioning "flexible residues" during AD4 scoring.

### Pitfall 2: Incorrect map prefix path for load_maps()
**What goes wrong:** `load_maps()` silently fails to find maps when given a directory path instead of the prefix string, or when given an absolute path that doesn't match where autogrid4 wrote files.
**Why it happens:** autogrid4 writes `receptor.C.map`, `receptor.HD.map` etc. into `maps_dir/`. The prefix must be `str(maps_dir / "receptor")`, not `str(maps_dir)` or `str(maps_dir) + "/"`.
**How to avoid:** Derive prefix as `str(Path(output_dir) / "maps" / "receptor")`. Verify `receptor.HD.map` exists before calling `load_maps()` (Phase 2 guard already ensures this, but add a defensive check).
**Warning signs:** `IOError` or `FileNotFoundError` mentioning `.map` files at a wrong path.

### Pitfall 3: Instance reuse vs. new instance per pose (performance)
**What goes wrong:** Creating `Vina()` inside a per-pose loop causes 200× receptor loading overhead. On a 100-pose batch, this adds seconds of wall time.
**Why it happens:** `set_receptor()` parses the PDBQT file into the C++ backend every call.
**How to avoid:** Create one Vina instance for Vina mode and one for AD4 mode before the pose loop. Only call `set_ligand_from_file()` inside the loop.
**Warning signs:** Scoring batch takes > 60s for 100 poses when Vina scoring alone should be < 5s.

### Pitfall 4: Thread safety with Vina C++ backend
**What goes wrong:** Running Vina scoring in threads (e.g., ThreadPoolExecutor) causes sporadic segfaults or incorrect scores due to shared C++ state.
**Why it happens:** The Vina SWIG bindings have no visible GIL release macros and no documented thread safety. The C++ object maintains internal mutable state.
**How to avoid:** Use sequential scoring within a single process. If parallelism is needed, use ProcessPoolExecutor with one Vina instance per subprocess — process isolation guarantees no shared state.
**Warning signs:** Non-deterministic scores on identical poses; segmentation faults under concurrent load.

### Pitfall 5: α or β validation failure with unhelpful error
**What goes wrong:** Calibrated α or β falls outside validated range; abort message is cryptic.
**Why it happens:** Training data too small, noisy, or systematically biased can cause the optimizer to converge at a boundary or extrapolate.
**How to avoid:** The abort message must quote the calibrated value AND the allowed range (e.g., "Calibrated α=1.45 is outside valid range [0.2, 1.2] kcal/mol/residue — check training data coverage."). This is a SCORE-03 requirement.
**Warning signs:** See CLAUDE.md §9 — α > 1.2 or < 0.2 is a signal that the pipeline is broken; do not patch around it.

### Pitfall 6: Vina score() return value indexing
**What goes wrong:** Using `score()` return value without indexing (it's a numpy array, not a float).
**Why it happens:** Documentation examples show `energy[0]`, but code written hastily might use `energy` directly.
**How to avoid:** Always `float(v.score()[0])` — index 0 is total energy; other indices are energy components.
**Warning signs:** TypeError when comparing score to 0 in AD4 anomaly check.

---

## Code Examples

### Complete per-pose scoring flow (both modes)
```python
# Source: Vina API from autodock-vina.readthedocs.io + AD4 pattern from GitHub issue #69/#254 investigation
from __future__ import annotations

import logging
from pathlib import Path

from vina import Vina

from hybridock_pep.models import DockConfig, PoseFailure, ScoredPose

logger = logging.getLogger(__name__)


def score_pose_batch(
    poses: list[ScoredPose],
    config: DockConfig,
    receptor_pdbqt: Path,
    maps_dir: Path,
) -> tuple[list[ScoredPose], list[PoseFailure]]:
    """Score all poses with Vina and AD4 in sequence.

    Creates one Vina instance per mode (receptor/maps loaded once).
    Reuses instances across poses via set_ligand_from_file().

    Args:
        poses: ScoredPose records with pdbqt_path populated.
        config: DockConfig for site_coords, box_size, peptide_sequence.
        receptor_pdbqt: Path to output_dir/receptor.pdbqt.
        maps_dir: Path to output_dir/maps/ (contains receptor.*.map files).

    Returns:
        Tuple of (scored_poses, failures).
    """
    # --- One Vina instance per mode ---
    v_vina = Vina(sf_name='vina', verbosity=0)
    v_vina.set_receptor(str(receptor_pdbqt))

    v_ad4 = Vina(sf_name='ad4', verbosity=0)
    # AD4: load_maps, NOT set_receptor (Pitfall 1)
    v_ad4.load_maps(str(maps_dir / "receptor"))

    n_residues = len(config.peptide_sequence)
    scored: list[ScoredPose] = []
    failures: list[PoseFailure] = []

    for pose in poses:
        try:
            # --- Grid boundary check (SCORE-01) ---
            pose.is_clipped = _check_grid_boundary(
                pose.pdbqt_path, config.site_coords, config.box_size
            )
            if pose.is_clipped:
                logger.warning("Pose %d: atoms outside grid bounds", pose.pose_idx)

            # --- Vina score ---
            v_vina.set_ligand_from_file(str(pose.pdbqt_path))
            v_vina.compute_vina_maps(
                center=list(config.site_coords),
                box_size=[config.box_size] * 3,
            )
            pose.vina_score = float(v_vina.score()[0])

            # --- AD4 score ---
            v_ad4.set_ligand_from_file(str(pose.pdbqt_path))
            pose.ad4_score = float(v_ad4.score()[0])
            pose.is_ad4_anomaly = pose.ad4_score > 0  # D-06

            scored.append(pose)

        except Exception as e:  # noqa: BLE001
            failures.append(PoseFailure(
                pose_idx=pose.pose_idx,
                stage="scoring",
                error_msg=f"{type(e).__name__}: {e}",
            ))
            logger.warning("Pose %d scoring failed: %s", pose.pose_idx, e)

    return scored, failures
```

### calibration.json read/write pattern
```python
# Source: [ASSUMED] — standard Python json module patterns
import json
from datetime import datetime, timezone
from pathlib import Path

CALIBRATION_SCHEMA = {
    "alpha": float,
    "beta": float,
    "n_complexes": int,
    "pearson_r": float,
    "rmse_kcal_mol": float,
    "calibrated_at": str,
    "training_csv": str,
}

def load_calibration(path: Path) -> dict[str, float | int | str]:
    """Load and validate calibration.json.

    Raises:
        ValueError: If alpha or beta are outside valid ranges (SCORE-03).
        FileNotFoundError: If calibration.json does not exist.
    """
    with path.open() as fh:
        cal = json.load(fh)
    alpha = cal["alpha"]
    beta = cal["beta"]
    if not (0.2 <= alpha <= 1.2):
        raise ValueError(
            f"Calibrated α={alpha:.3f} is outside valid range [0.2, 1.2] kcal/mol/residue "
            "— check training data coverage. SCORE-03 abort."
        )
    if not (0.0 <= beta <= 0.5):
        raise ValueError(
            f"Calibrated β={beta:.3f} is outside valid range [0.0, 0.5] "
            "— check training data or use default calibration.json."
        )
    return cal

def write_calibration(path: Path, alpha: float, beta: float, **kwargs: float | int | str) -> None:
    """Write calibration.json with current timestamp."""
    payload = {
        "alpha": alpha,
        "beta": beta,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| vina subprocess fork per pose | Vina Python API `score()` | Vina 1.2.0 (2021) | No fork overhead; receptor loaded once |
| Separate AD4 binary | `vina --scoring ad4` / Vina Python API `sf_name='ad4'` | Vina 1.2.0 (2021) | Single tool for both scoring functions |
| Manual charge handling for AD4 | Meeko Gasteiger charges + AD4 maps | Current standard | Meeko + autogrid4 handles charge assignment |

**Deprecated/outdated:**
- Using `vina --score_only` CLI flag from subprocess for per-pose scoring: Works but creates 100–200 subprocess processes per run. Python API is the current approach.
- AutoDockTools Python 2.x scripts (ADT): Meeko is the modern replacement for ligand preparation.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | scipy ≥1.11 is installed in score-env | Standard Stack | Low — scipy is in score-env.yml; version constraint is conservative |
| A2 | Vina C++ SWIG bindings do not release the GIL during `score()` | Pitfall 4 / Common Pitfalls | Medium — if GIL IS released, ThreadPoolExecutor parallelism becomes safe; sequential is still correct |
| A3 | `load_maps()` glob pattern resolves correctly given an absolute path prefix | Pattern 2 | Medium — glob behavior with absolute paths should work, but test in score-env with actual map files |
| A4 | Reusing a Vina instance across poses via `set_ligand_from_file()` produces identical scores to creating a fresh instance | Pattern 1 / Pitfall 3 | Low — confirmed by Vina API design (set_ligand replaces internal state) |
| A5 | `compute_vina_maps()` call after `set_ligand_from_file()` correctly maps only atom types present in the ligand (not all 22) | Pattern 1 | Low — documented in official Vina docs |

---

## Open Questions

1. **Does Vina instance reuse across poses require re-calling `compute_vina_maps()` each time?**
   - What we know: After `set_ligand_from_file()`, the ligand's atom type set may change. `compute_vina_maps()` behavior differs if called before vs. after ligand loading.
   - What's unclear: For `score_only` (no docking search), does changing the ligand require recomputing maps, or can the maps from the previous ligand be reused?
   - Recommendation: Call `compute_vina_maps()` once before the pose loop (before any `set_ligand_from_file()`), using the default 22-atom-type set. This avoids per-pose map recomputation and is the pattern documented for batch screening.

2. **ThreadPoolExecutor safety for dual Vina+AD4 scoring**
   - What we know: Vina has no documented thread safety; no GIL-release macros found in SWIG wrapper.
   - What's unclear: Whether separate Vina instances in separate threads share any global C++ state.
   - Recommendation: Default to sequential within a single process. If the 5-min wall-clock target is at risk, profile first before introducing threading. ProcessPoolExecutor with one Vina instance per subprocess is the safe parallel option.

3. **Does `score()` in Vina mode produce `--score_only` equivalent output?**
   - What we know: `v.score()` scores the current pose without minimization. This is the Python equivalent of `vina --score_only`.
   - What's unclear: Whether internal preprocessing (unbound energy correction) differs between Python API and CLI.
   - Recommendation: Trust the official API documentation; `score()` is described as evaluating the current pose.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| vina Python package | SCORE-01, SCORE-02 | Unverified (score-env only) | 1.2.7 latest [VERIFIED: PyPI] | — (no fallback; blocking) |
| scipy | SCORE-03 calibration | Unverified (score-env only) | ≥1.11 [ASSUMED] | — |
| autogrid4 maps from Phase 2 | SCORE-02 AD4 scoring | Phase 2 complete [VERIFIED: STATE.md] | — | — |
| receptor.pdbqt from Phase 2 | SCORE-01 | Phase 2 complete [VERIFIED: STATE.md] | — | — |
| poses/*.pdbqt from Phase 2 | SCORE-01, SCORE-02 | Phase 2 complete [VERIFIED: STATE.md] | — | — |

**Missing dependencies with no fallback:**
- None in Phase 2 outputs (all complete). The `vina` Python package and `scipy` must be confirmed installed in `score-env` before running tests.

**Missing dependencies with fallback:**
- None.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (current in project) |
| Config file | `pyproject.toml` [ASSUMED — standard for this project] |
| Quick run command | `python -m pytest tests/test_scoring.py -x -q` |
| Full suite command | `python -m pytest -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SCORE-01 | Grid boundary check sets `is_clipped=True` when atom outside box | unit | `python -m pytest tests/test_scoring.py::TestGridBoundaryCheck -x` | ❌ Wave 0 |
| SCORE-01 | Vina `score()` returns a float for valid PDBQT | unit (Vina mocked) | `python -m pytest tests/test_scoring.py::TestVinaScorerImports -x` | ❌ Wave 0 |
| SCORE-01 | Scoring failure returns `PoseFailure(stage='scoring')` | unit | `python -m pytest tests/test_scoring.py::TestScoringFailures -x` | ❌ Wave 0 |
| SCORE-02 | AD4 scorer uses `load_maps()` not `set_receptor()` | unit (source inspection) | `python -m pytest tests/test_scoring.py::TestAD4ScorerImports -x` | ❌ Wave 0 |
| SCORE-02 | AD4 score > 0 sets `is_ad4_anomaly=True` | unit | `python -m pytest tests/test_scoring.py::TestAD4AnomalyFlag -x` | ❌ Wave 0 |
| SCORE-03 | `apply_hybrid_score()` computes correct formula | unit | `python -m pytest tests/test_scoring.py::TestHybridFormula -x` | ❌ Wave 0 |
| SCORE-03 | `load_calibration()` raises ValueError when α outside [0.2, 1.2] | unit | `python -m pytest tests/test_scoring.py::TestCalibrationValidation -x` | ❌ Wave 0 |
| SCORE-03 | `fit_calibration()` returns α and β within bounds | unit | `python -m pytest tests/test_scoring.py::TestFitCalibration -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_scoring.py -x -q`
- **Per wave merge:** `python -m pytest -x -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_scoring.py` — covers all SCORE-01, SCORE-02, SCORE-03 requirements
- [ ] `data/calibration.json` — shipped defaults file (D-13)
- [ ] `src/hybridock_pep/scoring/__init__.py` — currently empty stub; needs exports

*(No new fixtures needed — can mock Vina API and use existing `tests/fixtures/pose_tiny.pdb`)*

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Validated via DockConfig (Pydantic) in Phase 1; PDBQT path existence check before scoring |
| V6 Cryptography | no | — |

### Known Threat Patterns for {stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal in PDBQT paths | Tampering | Paths derived from `DockConfig.output_dir` (validated); no user-supplied file names at scoring time |
| Malformed PDBQT causing parse error | Denial of Service | catch specific exceptions per pose; collect as PoseFailure (D-07) |
| calibration.json injection (tampered α/β) | Tampering | `load_calibration()` validates α ∈ [0.2, 1.2] and β ∈ [0.0, 0.5] on every read |

---

## Project Constraints (from CLAUDE.md)

These CLAUDE.md directives directly constrain implementation choices in this phase:

| Directive | Section | Impact on Phase 3 |
|-----------|---------|-------------------|
| Vina does NOT use partial charges | §2.1 | Do not extract per-atom charge from Vina score(); AD4 provides charge signal |
| All scoring code runs in score-env (Python 3.11) | §2.4 | No Python 3.10+ syntax restrictions for scoring modules (only rapidock_runner.py is 3.9) |
| No bare `except:` | §4 | Use `except Exception as e:` with reraise or PoseFailure collection |
| `from __future__ import annotations` first line | §4 | All scoring modules |
| Type hints everywhere; mypy strict | §4 | All public functions fully typed |
| Google-style docstrings with Args, Returns, Raises | §4 | Every module in `scoring/` |
| Every subprocess call logs full command at INFO | §4 | N/A — no subprocess calls in scoring; Vina Python API used |
| ADFRsuite binaries not redistributable | §2.6 | Not affected — scoring uses Vina Python package, not ADFRsuite |
| α calibrates to > 1.2 or < 0.2 → stop and ask | §9 | Enforced by `load_calibration()` ValueError; do not silently clamp |
| Vina and AD4 scores disagree in sign on > 20% of poses → flag and stop | §9 | Handled by `is_ad4_anomaly` flag; driver.py should surface anomaly count |

---

## Sources

### Primary (HIGH confidence)
- [autodock-vina.readthedocs.io — Vina class API reference](https://autodock-vina.readthedocs.io/en/latest/vina.html) — `__init__`, `set_receptor`, `set_ligand_from_file`, `load_maps`, `compute_vina_maps`, `score` signatures
- [autodock-vina.readthedocs.io — Python scripting tutorial](https://autodock-vina.readthedocs.io/en/latest/docking_python.html) — Vina mode score_only example; ligand loading order
- [github.com/ccsb-scripps/AutoDock-Vina — vina.py source](https://github.com/ccsb-scripps/AutoDock-Vina/blob/develop/build/python/vina/vina.py) — `score()` return array docstring; AD4 constraint; instance reuse
- [docs.scipy.org — scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) — L-BFGS-B bounds format; result.x

### Secondary (MEDIUM confidence)
- [github.com/ccsb-scripps/AutoDock-Vina/issues/254](https://github.com/ccsb-scripps/AutoDock-Vina/issues/254) — "Only flexible residues" AD4 constraint; `load_maps` without `set_receptor` for AD4
- [github.com/ccsb-scripps/AutoDock-Vina/issues/69](https://github.com/ccsb-scripps/AutoDock-Vina/issues/69) — AD4 rigid receptor failure; confirmed load_maps pattern
- [PyPI vina 1.2.7](https://pypi.org/project/vina/) — latest version, release date 2025-02-26

### Tertiary (LOW confidence)
- WebSearch results on thread safety — no authoritative source found; marked as ASSUMED in pitfall

---

## Metadata

**Confidence breakdown:**
- Standard Stack: HIGH — Vina API verified from official docs and source; scipy is standard
- Architecture: HIGH — Vina + AD4 instance lifecycle confirmed from source and GitHub issues
- AD4 API constraint: HIGH — confirmed from two GitHub issues and source code inspection
- Pitfalls: HIGH for Pitfall 1–3; MEDIUM for Pitfall 4 (thread safety unverified)
- Calibration pattern: HIGH — scipy.optimize.minimize L-BFGS-B well-documented

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (Vina API is stable; scipy patterns are stable)
