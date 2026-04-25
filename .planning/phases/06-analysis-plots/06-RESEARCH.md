# Phase 6: Analysis & Plots - Research

**Researched:** 2026-04-24
**Domain:** sklearn clustering, scipy stats, matplotlib Agg, Biopython Cα extraction, numpy RMSD
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Contact-zone residues = any peptide residue whose Cα is within 6 Å of any receptor Cα. Receptor Cα parsed from `config.receptor_path` (raw PDB, not PDBQT) using Biopython at clustering time.
- **D-02:** Fallback to full-peptide Cα RMSD when a pose has fewer than 3 contact-zone residues. Pose stays in clustering — not discarded.
- **D-03:** Convergence plot uses score-sorted order (ascending `hybrid_score`). Running mean ± σ tests ranking stability, NOT sampling arrival-order convergence.
- **D-04:** X-axis = N (1..len(scored_poses)). Y-axis = running mean ± σ of `hybrid_score` over top-N poses.
- **D-05:** k selection = `argmax(silhouette_scores)` for k = 2..k_max.
- **D-06:** `k_max = min(15, n_poses // 5)`. If k_max < 2, fall back to k=2 without silhouette search.
- **D-07:** `silhouette_plot.png` shows full silhouette score curve across the k range.
- **D-08:** `cluster_poses(scored_poses, config) -> ClusterResult` mutates `ScoredPose.cluster_id` in-place (mirrors `apply_hybrid_score()` pattern).
- **D-09:** `ClusterResult` is a `@dataclass` defined in `analysis/clustering.py`. Fields: `k_optimal: int`, `silhouette_score: float`, `per_cluster_stats: list[dict]`.
- **D-10:** `cluster_poses()` writes all three output files: `cluster_summary.csv`, `convergence_plot.png`, `silhouette_plot.png`.
- **D-11:** Replace driver.py Stage 3 stub (lines 147-148) with `cluster_result = cluster_poses(scored_poses, config)`.

### Claude's Discretion

- Matplotlib backend: `Agg` — set via `matplotlib.use("Agg")` at module import in plotting.py.
- Figure size: 8×5 inches at 150 DPI.
- Clustering linkage: `average`.
- RMSD matrix: `sklearn.metrics.pairwise_distances` with callable RMSD metric.
- 95% CI: `scipy.stats.t.interval` (t-distribution, loc=mean, scale=SEM, df=n-1).
- `cluster_summary.csv` column order: `cluster_id, n_poses, mean_hybrid_score, std_hybrid_score, ci95_lower, ci95_upper, best_pose_idx`.

### Deferred Ideas (OUT OF SCOPE)

- VIZ-01: Cluster dendrogram plot — deferred to v2.
- Arrival-order convergence (sampling convergence) — deferred to v2.
- `--n-clusters` CLI flag — not added in v1.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ANAL-01 | Pipeline clusters poses by pairwise Cα RMSD over contact-zone residues only, using agglomerative clustering with average linkage and precomputed metric; reports silhouette score per run | sklearn AgglomerativeClustering(metric='precomputed', linkage='average'), pairwise_distances callable, silhouette_score(metric='precomputed') |
| ANAL-02 | Pipeline computes per-cluster ensemble statistics (mean, std, 95% CI of hybrid score) and writes `cluster_summary.csv` | scipy.stats.t.interval, numpy per-group stats, csv.DictWriter |
| ANAL-03 | Pipeline generates `convergence_plot.png` showing running mean ± σ of hybrid score vs. number of poses N | matplotlib Agg backend, np.cumsum running stats, ax.fill_between for σ band |
| OUT-04 | Pipeline generates `convergence_plot.png` (running mean ± σ vs N) confirming ensemble convergence | Same as ANAL-03 |
| OUT-05 | Pipeline generates `silhouette_plot.png` showing cluster quality validation scores across cluster counts | matplotlib bar/line plot of silhouette scores for k=2..k_max, argmax annotation |
</phase_requirements>

---

## Summary

Phase 6 transforms a `list[ScoredPose]` (with populated `hybrid_score` and `ca_coords`) into clustered binding modes with ensemble statistics and two diagnostic plots. The work divides cleanly into three modules: `analysis/clustering.py` (RMSD matrix, silhouette search, AgglomerativeClustering, ClusterResult dataclass), `analysis/statistics.py` (per-cluster mean/std/95% CI, CSV writer), and `analysis/plotting.py` (convergence and silhouette plots). The three output files (`cluster_summary.csv`, `convergence_plot.png`, `silhouette_plot.png`) are all owned and written by `cluster_poses()`.

The most technically load-bearing piece is the contact-zone RMSD sub-selection: for each pose pair, the RMSD is computed only over peptide residue indices that are within 6 Å of any receptor Cα. This index set must be computed once per pose, not once per pair — per-pose contact indices are determined by comparing that pose's Cα coordinates against receptor Cα. The fallback to full-peptide RMSD when fewer than 3 contact residues are found (D-02) must be applied per pose, not globally; a mixed matrix (some rows use contact-zone, some use full-peptide RMSD) is acceptable and tested.

All three sklearn/scipy/matplotlib calls have well-verified exact APIs available. The key API correctness risks are: (1) `AgglomerativeClustering` uses `metric='precomputed'` in sklearn >= 1.4 (not the old `affinity='precomputed'` which was deprecated in 1.2, renamed in 1.4), (2) `silhouette_score` requires `2 <= n_labels <= n_samples - 1` and raises `ValueError` for k=1 or k=n_samples, (3) `scipy.stats.t.interval` takes `df=n-1` not `df=n`.

**Primary recommendation:** Build `clustering.py` as a single `cluster_poses()` function that owns the full pipeline (RMSD matrix → silhouette loop → fit → in-place mutation → delegate to `statistics.py` and `plotting.py` for CSV and plots). Keep plotting headless-safe by setting `matplotlib.use("Agg")` at the top of `plotting.py` before any `import matplotlib.pyplot`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Contact-zone residue index selection | analysis/clustering.py | — | Per-pose receptor-distance check; uses Biopython for receptor Cα, numpy for distance filter |
| Pairwise RMSD matrix computation | analysis/clustering.py | — | Input to AgglomerativeClustering; must be precomputed as n×n float64 array |
| Silhouette k-search loop | analysis/clustering.py | — | Owns k_max derivation and argmax selection; k=2 fallback lives here |
| AgglomerativeClustering fit + cluster_id mutation | analysis/clustering.py | — | In-place mutation of ScoredPose.cluster_id is primary side effect |
| ClusterResult construction | analysis/clustering.py | — | Dataclass defined in this module; not in models.py (analysis output, not pipeline contract) |
| Per-cluster stats + 95% CI | analysis/statistics.py | analysis/clustering.py | statistics.py computes; clustering.py calls it and passes results into ClusterResult |
| cluster_summary.csv write | analysis/statistics.py | — | CSV write delegated to statistics module; cluster_poses() calls it |
| Convergence plot | analysis/plotting.py | — | Agg backend, score-sorted order, fill_between for σ band |
| Silhouette plot | analysis/plotting.py | — | Bar/line across k range, argmax annotation |
| Driver Stage 3 wiring | driver.py | — | One-line replacement of lines 147-148 stub |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | >=1.4 (score-env.yml) | AgglomerativeClustering, silhouette_score, pairwise_distances | The only correct choice for precomputed-metric hierarchical clustering in Python; no alternatives warranted |
| scipy | >=1.13 (score-env.yml) | scipy.stats.t.interval for 95% CI | Already a dependency; t-distribution CI is appropriate for small cluster sizes (n < 30) |
| matplotlib | >=3.8 (score-env.yml) | convergence_plot.png, silhouette_plot.png | Already a dependency; Agg backend is headless-safe and required |
| numpy | >=1.26 (score-env.yml) | RMSD computation, running mean/std cumsum | Already a dependency; vectorised Cα distance computation |
| biopython | >=1.83 (score-env.yml) | Receptor Cα extraction from PDB | Already used in pose_io.py; no new dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| csv (stdlib) | Python 3.11 | cluster_summary.csv write | DictWriter for ordered column output |
| dataclasses (stdlib) | Python 3.11 | ClusterResult @dataclass | Already used pattern in models.py |

**No new dependencies are introduced by Phase 6.** All libraries are already in score-env.yml.

**Version verification:**
```bash
# score-env.yml pins these minimums:
# scikit-learn>=1.4, scipy>=1.13, matplotlib>=3.8, numpy>=1.26, biopython>=1.83
# All confirmed via envs/score-env.yml [VERIFIED: file read]
```

---

## Architecture Patterns

### System Architecture Diagram

```
cluster_poses(scored_poses: list[ScoredPose], config: DockConfig)
         │
         ├─► _load_receptor_ca_coords(config.receptor_path)
         │          └── Biopython PDBParser → receptor_ca: np.ndarray [n_rec, 3]
         │
         ├─► _build_rmsd_matrix(scored_poses, receptor_ca)
         │          ├── for each pose: _contact_zone_indices(pose.ca_coords, receptor_ca, cutoff=6.0)
         │          │       └── if len(indices) < 3: use full-peptide Cα (D-02 fallback)
         │          └── pairwise_distances(pose_ca_arrays, metric=_rmsd_fn)
         │                   → dist_matrix: np.ndarray [n, n]  (precomputed)
         │
         ├─► _select_k(dist_matrix, labels=None, k_max=min(15, n//5))
         │          ├── for k in range(2, k_max+1):
         │          │       agg = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
         │          │       labels = agg.fit_predict(dist_matrix)
         │          │       sil_scores[k] = silhouette_score(dist_matrix, labels, metric='precomputed')
         │          └── k_optimal = argmax(sil_scores)
         │
         ├─► AgglomerativeClustering(n_clusters=k_optimal, ...).fit_predict(dist_matrix)
         │          └── mutate pose.cluster_id = label  (in-place, D-08)
         │
         ├─► compute_cluster_stats(scored_poses)   [statistics.py]
         │          └── per cluster: mean, std, 95% CI via t.interval, best_pose_idx
         │
         ├─► write_cluster_summary_csv(stats, output_path)   [statistics.py]
         │
         ├─► plot_convergence(scored_poses, output_path)   [plotting.py]
         │          └── sort by hybrid_score ascending → running mean ± σ → savefig
         │
         └─► plot_silhouette(sil_scores, k_optimal, output_path)   [plotting.py]
                    └── bar/line across k range, vertical line at k_optimal → savefig

Returns: ClusterResult(k_optimal, silhouette_score, per_cluster_stats)
Side effect: all ScoredPose.cluster_id mutated, 3 output files written
```

### Recommended Project Structure

```
src/hybridock_pep/analysis/
├── __init__.py          # export cluster_poses
├── clustering.py        # cluster_poses(), ClusterResult, RMSD matrix, silhouette loop
├── statistics.py        # compute_cluster_stats(), write_cluster_summary_csv()
└── plotting.py          # plot_convergence(), plot_silhouette(); matplotlib.use("Agg") at top

tests/
└── test_clustering.py   # New file; covers RMSD, contact-zone, clustering, silhouette
```

### Pattern 1: AgglomerativeClustering with Precomputed Distance Matrix

**What:** Pass a precomputed n×n distance matrix to AgglomerativeClustering using `metric='precomputed'`.
**When to use:** Any time the distance function is not a standard Euclidean metric (e.g., Cα RMSD over a subset of residues).

```python
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# [VERIFIED: WebFetch official docs]
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# dist_matrix: np.ndarray shape (n_poses, n_poses), dtype float64
# Must be square, symmetric, zero on diagonal
clustering = AgglomerativeClustering(
    n_clusters=k,
    metric="precomputed",   # NOT 'affinity' — that was renamed in sklearn 1.4
    linkage="average",
)
labels = clustering.fit_predict(dist_matrix)  # shape (n_poses,), dtype int
```

**Critical:** `metric='precomputed'` requires passing the distance matrix (not feature vectors) to `fit_predict`. The matrix must be square and symmetric. `linkage='ward'` is INCOMPATIBLE with `metric='precomputed'`; `'average'` is correct and compatible.

### Pattern 2: silhouette_score with Precomputed Distances

**What:** Compute silhouette score from a precomputed distance matrix (not feature vectors).
**When to use:** Inside the k-search loop, after each `fit_predict`.

```python
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
# [VERIFIED: WebFetch official docs]
from sklearn.metrics import silhouette_score

# Requires: 2 <= n_unique_labels <= n_samples - 1
# Raises ValueError if k=1 or k=n_samples
score = silhouette_score(
    dist_matrix,      # precomputed n×n distance matrix
    labels,           # cluster labels from fit_predict
    metric="precomputed",
)
```

**Critical pitfall:** `silhouette_score` raises `ValueError: Number of labels is 1` when all samples are assigned to one cluster (can happen if k_max=1 due to very small n). Guard with `if len(set(labels)) < 2: continue`. Also raises if `n_unique_labels == n_samples` (every pose its own cluster). Both are edge cases that must be try/except guarded in the k-search loop.

### Pattern 3: Pairwise RMSD Matrix via pairwise_distances Callable

**What:** Build the n×n RMSD matrix by passing a callable metric to `pairwise_distances`. The callable receives two 1D arrays (flattened Cα coords for two poses) and returns a float.
**When to use:** When RMSD is computed over a variable-length contact-zone subset.

```python
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
# [VERIFIED: WebFetch official docs]
from sklearn.metrics import pairwise_distances
import numpy as np

# contact_indices[i]: np.ndarray of residue indices for pose i
# ca_arrays[i]: np.ndarray shape (n_residues, 3) — full Cα coords for pose i

def _build_rmsd_matrix(
    ca_arrays: list[np.ndarray],
    contact_indices: list[np.ndarray],
) -> np.ndarray:
    n = len(ca_arrays)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            # Use intersection of contact zones; fallback handled in contact_indices
            idx_i = contact_indices[i]
            idx_j = contact_indices[j]
            # Both poses must agree on which residues to compare
            # Use the intersection if they differ (defensive; should be same peptide)
            common = np.intersect1d(idx_i, idx_j)
            if len(common) == 0:
                common = np.arange(len(ca_arrays[i]))  # full-peptide fallback
            coords_i = ca_arrays[i][common]
            coords_j = ca_arrays[j][common]
            rmsd = np.sqrt(np.mean(np.sum((coords_i - coords_j) ** 2, axis=1)))
            dist[i, j] = dist[j, i] = rmsd
    return dist
```

**Note:** The above explicit loop is clearer and safer than using `pairwise_distances` with a callable when the metric depends on per-pose state (contact indices). `pairwise_distances` with a callable receives 1D vectors; reshaping and index-subsetting inside the callable would require closure over `contact_indices`, which is fragile. The explicit loop is preferred here.

### Pattern 4: scipy.stats.t.interval for 95% CI

**What:** Compute 95% CI using Student's t-distribution, appropriate for small per-cluster sample sizes.
**When to use:** Per-cluster stats in statistics.py.

```python
# Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
# [VERIFIED: WebFetch official docs]
from scipy.stats import t as t_dist
import numpy as np

def cluster_ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        mean = float(values[0]) if values else float("nan")
        return mean, mean  # CI undefined for n=1; return point estimate
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n))   # standard error of mean
    lo, hi = t_dist.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return float(lo), float(hi)
```

**Critical:** `df=n-1` (degrees of freedom), NOT `df=n`. `scale=sem` (standard error of mean = std/sqrt(n)), NOT `scale=std`. Getting `df` or `scale` wrong silently produces incorrect CI bounds with no error.

### Pattern 5: Matplotlib Agg Backend (Headless)

**What:** Set the Agg non-interactive backend before importing pyplot to prevent display errors in headless environments.
**When to use:** Top of `analysis/plotting.py`.

```python
# Source: https://matplotlib.org/stable/users/explain/figure/backends.html
# [VERIFIED: Context7 /matplotlib/matplotlib]
import matplotlib
matplotlib.use("Agg")  # MUST be before any import of matplotlib.pyplot
import matplotlib.pyplot as plt

def plot_convergence(scored_poses, output_path, figsize=(8, 5), dpi=150):
    scores = sorted([p.hybrid_score for p in scored_poses])  # ascending
    n = len(scores)
    ns = np.arange(1, n + 1)
    running_mean = np.array([np.mean(scores[:i]) for i in ns])
    running_std  = np.array([np.std(scores[:i], ddof=0) for i in ns])
    # ddof=0 for population std (consistent with "σ" notation)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ns, running_mean, color="steelblue", label="Running mean")
    ax.fill_between(
        ns,
        running_mean - running_std,
        running_mean + running_std,
        alpha=0.3,
        color="steelblue",
        label="±σ",
    )
    ax.set_xlabel("Top-N poses (score-sorted)")
    ax.set_ylabel("Hybrid score (kcal/mol)")
    ax.set_title("Score-sorted convergence: ranking stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)  # CRITICAL: release memory; avoids ResourceWarning in tests
```

**Critical:** `plt.close(fig)` after each `savefig` is mandatory in headless batch usage. Not closing figures causes a `RuntimeWarning: More than 20 figures have been opened` in test suites and leaks memory in long runs.

### Pattern 6: Receptor Cα Extraction with Biopython

**What:** Parse a receptor PDB with Biopython and collect all Cα coordinates for contact-zone distance filtering. Same library already used in `pose_io.py`.
**When to use:** Once per `cluster_poses()` call, before building the RMSD matrix.

```python
# Source: pose_io.py — established pattern [VERIFIED: file read]
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import numpy as np

def _load_receptor_ca_coords(receptor_path) -> np.ndarray:
    """Return shape [n_rec_residues, 3] float64 array of receptor Cα coords."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(receptor_path))
    model = next(iter(structure))
    coords = []
    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            if "CA" not in residue:
                continue
            coords.append(list(residue["CA"].get_vector().get_array()))
    if not coords:
        raise ValueError(f"No standard AA Cα atoms found in receptor {receptor_path}")
    return np.array(coords, dtype=np.float64)
```

### Pattern 7: Contact-Zone Index Selection

**What:** For a single pose, determine which residue indices are "in contact" with any receptor Cα (within 6 Å).
**When to use:** Called per pose before building the RMSD matrix.

```python
# [ASSUMED] — numpy broadcasting approach; standard pattern
def _contact_zone_indices(
    pose_ca: np.ndarray,      # shape [n_pep, 3]
    receptor_ca: np.ndarray,  # shape [n_rec, 3]
    cutoff: float = 6.0,
) -> np.ndarray:
    """Return indices of peptide residues within cutoff Å of any receptor Cα."""
    # Broadcasting: [n_pep, 1, 3] - [1, n_rec, 3] → [n_pep, n_rec, 3]
    diff = pose_ca[:, np.newaxis, :] - receptor_ca[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # [n_pep, n_rec]
    in_contact = np.any(dists < cutoff, axis=1)  # [n_pep] bool
    indices = np.where(in_contact)[0]
    return indices  # dtype int64
```

### Anti-Patterns to Avoid

- **Using `affinity='precomputed'`:** Deprecated since sklearn 1.2, renamed to `metric` in 1.4. score-env pins sklearn>=1.4, so `affinity` raises a deprecation warning or error. Always use `metric='precomputed'`.
- **Using `linkage='ward'` with `metric='precomputed'`:** Ward linkage requires Euclidean metric. It will raise `ValueError`. Use `linkage='average'` (locked by D-02).
- **Calling `silhouette_score` with k=1 labels:** Raises `ValueError`. Always check `len(set(labels)) >= 2` before calling.
- **Setting `matplotlib.use("Agg")` after `import matplotlib.pyplot`:** Has no effect; the backend must be set before pyplot is imported. If pyplot is imported elsewhere first (e.g., in test harness), the `use()` call is a no-op. Use `matplotlib.use("Agg", force=True)` in tests if needed.
- **Not calling `plt.close(fig)`:** Leaks figure handles in test runs (pytest runs all tests in one process). Always `plt.close(fig)` after `savefig`.
- **Computing contact zone once globally:** The contact zone depends on EACH pose's Cα coordinates (each pose is a different conformation). Computing it once from the first pose and applying to all is incorrect. Per-pose contact indices must be computed per pose.
- **Using `ddof=1` for the σ band in the convergence plot:** The plot shows population σ (±σ), not sample std. Use `ddof=0` for consistency with the plot label. The CSV column `std_hybrid_score` uses `ddof=1` (sample std for statistical reporting).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hierarchical clustering with precomputed distances | Custom dendrogram merge loop | `sklearn.cluster.AgglomerativeClustering(metric='precomputed', linkage='average')` | Handles all linkage math, cluster assignment, edge cases |
| Silhouette coefficient | Custom (a,b) computation per sample | `sklearn.metrics.silhouette_score(dist_matrix, labels, metric='precomputed')` | Correct handling of border cases, vectorised |
| Confidence interval | `mean ± z*std/sqrt(n)` (z-distribution) | `scipy.stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)` | t-distribution is correct for n<30; z-approx understates CI width |
| Pairwise distance matrix | Double-loop with explicit RMSD calls | Numpy broadcasting `diff = ca[i][:,np.newaxis] - ca[j][np.newaxis,:]` | Vectorised, avoids Python loop overhead for n_residues dimension |

**Key insight:** The only genuinely custom code in Phase 6 is the contact-zone index selection and the per-pair RMSD computation (which varies by pose because contact indices differ). Everything else delegates to sklearn/scipy.

---

## Common Pitfalls

### Pitfall 1: `affinity` vs `metric` in AgglomerativeClustering
**What goes wrong:** Code uses `AgglomerativeClustering(affinity='precomputed', ...)`. In sklearn >= 1.4 this raises a deprecation warning; in future versions it will error.
**Why it happens:** Old tutorials and StackOverflow answers use `affinity`.
**How to avoid:** Always use `metric='precomputed'` when score-env pins sklearn>=1.4.
**Warning signs:** `FutureWarning: The `affinity` parameter is deprecated` in test output.

### Pitfall 2: silhouette_score ValueError for Edge Cases
**What goes wrong:** `silhouette_score` raises `ValueError: Number of labels is 1` (k=1 result) or `ValueError: Number of labels is n_samples` (every pose in its own cluster).
**Why it happens:** For very small n_poses (e.g., 5 poses, k_max=1 after D-06), or degenerate clustering where all pairwise distances are equal.
**How to avoid:** Wrap the silhouette call in `try/except ValueError`: skip that k value and continue the loop. Also enforce k_max >= 2 per D-06 before entering the loop.
**Warning signs:** Test suite fails with `ValueError` on fixture with 3-5 poses.

### Pitfall 3: plt.close() Memory Leak in Tests
**What goes wrong:** pytest runs 10+ test functions that each call a plot function. Without `plt.close(fig)`, matplotlib opens 20+ figures and emits `RuntimeWarning: More than 20 figures have been opened`.
**Why it happens:** Matplotlib tracks all open figures globally within a process; pytest does not close them between tests.
**How to avoid:** Every plotting function ends with `plt.close(fig)`. Test fixtures that call plotting functions also call `plt.close("all")` in teardown.
**Warning signs:** `RuntimeWarning` in test output; increasing memory usage during `pytest`.

### Pitfall 4: Contact Zone Computed Once Instead of Per-Pose
**What goes wrong:** `_contact_zone_indices` is called once on the first pose and the indices applied to all poses. Poses with different backbone conformations have different contact zones.
**Why it happens:** Performance optimisation assumption (receptor is fixed, so contact zone is fixed). But peptide Cα positions vary per pose, so the set of "close" residues varies.
**How to avoid:** Call `_contact_zone_indices(pose.ca_coords, receptor_ca)` per pose inside the loop.
**Warning signs:** All poses use same contact zone even when conformations differ visibly in PDB viewer.

### Pitfall 5: D-02 Fallback Applied Globally Instead of Per-Pose
**What goes wrong:** If ANY pose has <3 contact residues, fall back to full-peptide RMSD for ALL poses. This changes the distance definition for well-contacted poses.
**Why it happens:** Simpler implementation: check minimum contact count globally, then decide.
**How to avoid:** Apply D-02 fallback per-pose: each pose independently decides whether to use contact-zone or full-peptide Cα. The distance matrix then has pairs that may mix contact-zone and full-peptide distances — this is acceptable and consistent with the spec.
**Warning signs:** Test asserts contact-zone RMSD for well-docked poses but full-peptide RMSD is used instead.

### Pitfall 6: t.interval scale = std instead of SEM
**What goes wrong:** `t.interval(0.95, df=n-1, loc=mean, scale=std)` produces CI far too wide.
**Why it happens:** Confusing population std with standard error of mean.
**How to avoid:** `scale = std / sqrt(n)` (SEM). The CI shrinks as n grows, as expected.
**Warning signs:** 95% CI for a cluster of 50 poses spans ±3 kcal/mol — unrealistically wide.

### Pitfall 7: Matplotlib Agg Set After pyplot Import
**What goes wrong:** `matplotlib.use("Agg")` has no effect if `matplotlib.pyplot` was already imported by another module in the same process.
**Why it happens:** Test files or conftest.py that import plotting utilities before setting the backend.
**How to avoid:** Set `matplotlib.use("Agg")` at the very top of `plotting.py`, before all other matplotlib imports. In test `conftest.py`, set it before importing the module under test.
**Warning signs:** `UserWarning: Matplotlib is currently using <TkAgg>` or display errors on headless CI.

---

## Code Examples

Verified patterns from official sources and codebase inspection:

### Contact-Zone Index Selection + RMSD Matrix Build

```python
# [VERIFIED: numpy broadcasting — standard pattern; Biopython pattern verified against pose_io.py]
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

def _load_receptor_ca_coords(receptor_path) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(receptor_path))
    model = next(iter(structure))
    coords = [
        list(res["CA"].get_vector().get_array())
        for chain in model
        for res in chain
        if is_aa(res, standard=True) and "CA" in res
    ]
    return np.array(coords, dtype=np.float64)

def _contact_zone_indices(pose_ca, receptor_ca, cutoff=6.0):
    # [n_pep, n_rec, 3] pairwise diff
    diff = pose_ca[:, np.newaxis, :] - receptor_ca[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # [n_pep, n_rec]
    return np.where(np.any(dists < cutoff, axis=1))[0]

def _pose_pair_rmsd(ca_i, ca_j, idx_i, idx_j):
    common = np.intersect1d(idx_i, idx_j)
    if len(common) < 3:
        common = np.arange(min(len(ca_i), len(ca_j)))  # D-02 full-peptide fallback
    coords_i = ca_i[common]
    coords_j = ca_j[common]
    return float(np.sqrt(np.mean(np.sum((coords_i - coords_j) ** 2, axis=1))))
```

### Silhouette k-Search Loop

```python
# Source: sklearn.metrics.silhouette_score docs [VERIFIED: WebFetch]
# Source: sklearn.cluster.AgglomerativeClustering docs [VERIFIED: WebFetch]
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

def _select_k_silhouette(dist_matrix: np.ndarray) -> tuple[int, dict[int, float]]:
    n = dist_matrix.shape[0]
    k_max = min(15, n // 5)
    if k_max < 2:
        return 2, {}  # D-06 fallback

    sil_scores: dict[int, float] = {}
    for k in range(2, k_max + 1):
        agg = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        )
        labels = agg.fit_predict(dist_matrix)
        try:
            sil_scores[k] = silhouette_score(
                dist_matrix, labels, metric="precomputed"
            )
        except ValueError:
            # k produced degenerate clustering (all one cluster or each its own)
            sil_scores[k] = float("-inf")

    k_optimal = max(sil_scores, key=sil_scores.__getitem__)
    return k_optimal, sil_scores
```

### 95% CI with scipy.stats.t.interval

```python
# Source: scipy.stats.t docs [VERIFIED: WebFetch]
from scipy.stats import t as t_dist
import numpy as np

def _ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        v = float(values[0]) if values else float("nan")
        return v, v
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    lo, hi = t_dist.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return float(lo), float(hi)
```

### ClusterResult Dataclass

```python
# Pattern: mirrors models.py @dataclass pattern [VERIFIED: models.py read]
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class ClusterResult:
    """Result of cluster_poses() analysis.

    Args:
        k_optimal: Number of clusters selected by silhouette argmax.
        silhouette_score: Silhouette score at k_optimal.
        per_cluster_stats: List of per-cluster stat dicts, each with keys:
            cluster_id, n_poses, mean_hybrid_score, std_hybrid_score,
            ci95_lower, ci95_upper, best_pose_idx.
    """
    k_optimal: int
    silhouette_score: float
    per_cluster_stats: list[dict] = field(default_factory=list)
```

### Driver Integration (lines 147-148 replacement)

```python
# Before (lines 147-148 in driver.py):
#   # Stage 3 stub: Clustering and output writing are Phase 6/7 scope
#   logger.info("Clustering and output: Phase 6/7 not yet implemented")
#
# After (Phase 6 replacement):
from hybridock_pep.analysis import cluster_poses

    # Stage 3: Cluster poses and write analysis outputs
    cluster_result = cluster_poses(scored_poses, config)
    logger.info(
        "Stage 3 complete: k=%d clusters, silhouette=%.3f",
        cluster_result.k_optimal,
        cluster_result.silhouette_score,
    )
```

---

## ScoredPose Interface (from Phase 4)

**Confirmed by file read of `src/hybridock_pep/models.py`** [VERIFIED]:

```python
@dataclass
class ScoredPose(PoseRecord):
    vina_score: float | None = None
    ad4_score: float | None = None
    entropy_correction: float | None = None
    hybrid_score: float | None = None
    cluster_id: int | None = None       # Phase 6 writes this
    pdbqt_path: Path | None = None
    is_ad4_anomaly: bool = False
    is_clipped: bool = False

# PoseRecord base:
#   pose_idx: int
#   pdb_path: Path
#   sequence: str
#   ca_coords: np.ndarray   # shape [n_residues, 3], float64
```

`ca_coords` is populated at parse time in `pose_io.py` and stored in-memory — no re-read from disk at clustering time. Shape is `[n_residues, 3]`, dtype `float64`. This is confirmed by `pose_io.py` line 148: `ca_coords = np.array(ca_coords_list, dtype=np.float64)`.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `AgglomerativeClustering(affinity='precomputed')` | `AgglomerativeClustering(metric='precomputed')` | sklearn 1.2 (deprecated), 1.4 (renamed) | `affinity` param raises FutureWarning in 1.4+; use `metric` |
| `matplotlib.use("Agg")` optional | `matplotlib.use("Agg")` required for headless | Always been best practice | CI / headless servers have no display; without Agg, pyplot import fails |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | numpy broadcasting `pose_ca[:, np.newaxis, :] - receptor_ca[np.newaxis, :, :]` produces correct [n_pep, n_rec, 3] diff for distance computation | Pattern 7, Code Examples | If axes are wrong, distances are computed per-chain not per-residue; wrong contact zone |
| A2 | `plt.close(fig)` is sufficient to release figure memory; no additional matplotlib cleanup needed | Pattern 5, Pitfall 3 | Minor: if wrong, tests may emit ResourceWarning but won't fail |
| A3 | `per_cluster_stats: list[dict]` is sufficient for Phase 7 to read `cluster_summary.csv` without needing a richer type | D-09, ClusterResult | Phase 7 reads from CSV, not from ClusterResult in memory — low risk |

**Table is short:** Most claims are verified via WebFetch of official docs or direct file read of the codebase.

---

## Open Questions

1. **Contact-zone RMSD when peptide sequences differ across poses**
   - What we know: All poses should be the same peptide sequence (same RAPiDock run). `pose_io.py` parses sequence per pose.
   - What's unclear: If parsing produces slightly different residue counts across poses (e.g., one pose missing terminal residue), `np.intersect1d(idx_i, idx_j)` falls back correctly — but the RMSD is computed over fewer residues than expected.
   - Recommendation: Add a `logger.debug` when intersection is shorter than contact zone, but do not fail. The D-02 fallback handles the extreme case (< 3 common residues).

2. **Silhouette score store for ClusterResult.silhouette_score when k_max < 2**
   - What we know: D-06 specifies k=2 fallback when k_max < 2. No silhouette search is run.
   - What's unclear: What value to store in `ClusterResult.silhouette_score` in this case.
   - Recommendation: Store `float("nan")` and document it. Phase 7 reads `cluster_summary.csv`, not `ClusterResult.silhouette_score`, so `nan` here is harmless.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scikit-learn | AgglomerativeClustering, silhouette_score | score-env.yml pin | >=1.4 | None — required |
| scipy | t.interval for 95% CI | score-env.yml pin | >=1.13 | None — required |
| matplotlib | Plot generation | score-env.yml pin | >=3.8 | None — required |
| numpy | RMSD computation | score-env.yml pin | >=1.26 | None — required |
| biopython | Receptor Cα extraction | score-env.yml pin | >=1.83 | None — required |

All dependencies confirmed present in `envs/score-env.yml`. No new packages are needed for Phase 6. [VERIFIED: envs/score-env.yml file read]

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (project-wide, per CLAUDE.md) |
| Config file | pytest.ini or pyproject.toml `[tool.pytest]` — check Wave 0 |
| Quick run command | `pytest tests/test_clustering.py -x` |
| Full suite command | `pytest --cov=hybridock_pep` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ANAL-01 | Contact-zone index selection returns correct residue indices for known distances | unit | `pytest tests/test_clustering.py::test_contact_zone_indices -x` | ❌ Wave 0 |
| ANAL-01 | Full-peptide fallback when <3 contact residues | unit | `pytest tests/test_clustering.py::test_contact_zone_fallback -x` | ❌ Wave 0 |
| ANAL-01 | Pairwise RMSD matrix is square, symmetric, zero diagonal | unit | `pytest tests/test_clustering.py::test_rmsd_matrix_symmetry -x` | ❌ Wave 0 |
| ANAL-01 | AgglomerativeClustering assigns cluster_id to each ScoredPose | unit | `pytest tests/test_clustering.py::test_cluster_poses_assigns_ids -x` | ❌ Wave 0 |
| ANAL-01 | Silhouette k-search returns k_optimal in [2, k_max] | unit | `pytest tests/test_clustering.py::test_silhouette_k_selection -x` | ❌ Wave 0 |
| ANAL-02 | cluster_summary.csv is written with correct columns and values | unit | `pytest tests/test_clustering.py::test_cluster_summary_csv -x` | ❌ Wave 0 |
| ANAL-02 | 95% CI computed correctly via t.interval (compare to known value) | unit | `pytest tests/test_clustering.py::test_ci95 -x` | ❌ Wave 0 |
| ANAL-03 | convergence_plot.png is written as valid PNG | unit | `pytest tests/test_clustering.py::test_convergence_plot_written -x` | ❌ Wave 0 |
| OUT-04 | convergence_plot.png exists at expected output path after cluster_poses() | unit | Same as ANAL-03 | ❌ Wave 0 |
| OUT-05 | silhouette_plot.png is written as valid PNG | unit | `pytest tests/test_clustering.py::test_silhouette_plot_written -x` | ❌ Wave 0 |

### Key Test Design Notes

**Deterministic fixture:** Use a fixed `dist_matrix` of known shape (e.g., 10×10 with two clear clusters of distance 0.1 within-cluster and 5.0 between-cluster). This gives deterministic k=2 silhouette optimum and makes cluster assignment assertions reliable.

**Fake ScoredPose fixtures:** Construct `ScoredPose` objects directly with `ca_coords=np.zeros((5,3))` and `hybrid_score=-7.0`. No PDB file required for unit tests of statistics.py and plotting.py — only clustering.py's `_load_receptor_ca_coords` needs a PDB file (use `tests/fixtures/receptor_tiny.pdb`).

**PNG validation:** Assert file exists and `os.path.getsize(path) > 0`. Optionally use `PIL.Image.open(path)` to verify it's a valid image, but PIL is not in score-env; size > 0 check is sufficient.

**matplotlib isolation:** Test module should call `matplotlib.use("Agg", force=True)` or import plotting.py before pytest collects (since plotting.py sets it at module level, importing the module handles it).

### Sampling Rate

- **Per task commit:** `pytest tests/test_clustering.py -x`
- **Per wave merge:** `pytest --cov=hybridock_pep`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_clustering.py` — covers ANAL-01, ANAL-02, ANAL-03, OUT-04, OUT-05
- [ ] Fixture: a minimal `np.ndarray` RMSD matrix (10×10) with known cluster structure
- [ ] Fixture: list of 10 `ScoredPose` objects with `ca_coords` and `hybrid_score` set

---

## Security Domain

**Not applicable.** Phase 6 performs in-memory numerical analysis and writes PNG/CSV files to the run output directory. No network I/O, authentication, user input parsing, or external service calls. ASVS categories V5 (Input Validation) applies minimally: `cluster_poses()` should validate `len(scored_poses) > 0` and that `hybrid_score` is not None on all poses before proceeding.

---

## Project Constraints (from CLAUDE.md)

| Directive | Implication for Phase 6 |
|-----------|------------------------|
| Python 3.11 for all in-repo code (score-env) | clustering.py, statistics.py, plotting.py use Python 3.11 features; match/case and X\|Y unions are fine |
| `from __future__ import annotations` at top of every module | Required in all 3 new modules |
| Type hints everywhere; mypy strict on CI | All functions must be fully typed; `np.ndarray` not `Any` |
| Ruff + black; line length 100 | Code formatted to 100-char lines |
| Docstrings in Google style with Args/Returns/Raises | All public functions need full docstrings |
| No bare `except:` | Silhouette loop uses `except ValueError:`, not bare `except:` |
| `logging` not `print` | All logging via `logging.getLogger(__name__)` |
| `pytest --cov` target ≥ 70% before merging to main | test_clustering.py must cover RMSD, contact-zone, silhouette, and CSV paths |
| OSI-licensed dependencies only; no copyleft | sklearn (BSD), scipy (BSD), matplotlib (BSD), biopython (BSD-like) — all compliant |
| No bare `except:` — catch specific exceptions | Confirmed above |

---

## Sources

### Primary (HIGH confidence)
- `src/hybridock_pep/models.py` — ScoredPose, PoseRecord, DockConfig dataclasses; ca_coords shape confirmed
- `src/hybridock_pep/driver.py` — Stage 3 stub lines 147-148 confirmed
- `src/hybridock_pep/sampling/pose_io.py` — Biopython PDBParser pattern for Cα extraction
- `src/hybridock_pep/scoring/entropy.py` — apply_hybrid_score() in-place mutation pattern
- `envs/score-env.yml` — library version minimums (sklearn>=1.4, scipy>=1.13, etc.)
- `.planning/phases/06-analysis-plots/06-CONTEXT.md` — all locked decisions D-01..D-11
- `.planning/REQUIREMENTS.md` — ANAL-01, ANAL-02, ANAL-03, OUT-04, OUT-05 definitions
- [WebFetch: scikit-learn.org/stable — AgglomerativeClustering] — metric='precomputed', linkage compatibility, fit_predict API
- [WebFetch: scikit-learn.org/stable — silhouette_score] — metric='precomputed', n_labels constraint
- [WebFetch: scikit-learn.org/stable — pairwise_distances] — callable metric support
- [WebFetch: scikit-learn.org/stable/whats_new/v1.2] — affinity→metric deprecation timeline
- [WebFetch: docs.scipy.org — scipy.stats.t] — interval(confidence, df, loc, scale) signature
- [Context7: /matplotlib/matplotlib — Agg backend, savefig] — matplotlib.use("Agg") placement

### Secondary (MEDIUM confidence)
- [WebFetch: scikit-learn.org/stable/whats_new/v1.4] — metric=None deprecation in AgglomerativeClustering confirmed

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all confirmed from score-env.yml and official docs
- Architecture: HIGH — all locked decisions from CONTEXT.md; API signatures verified
- Pitfalls: HIGH — affinity/metric rename verified from changelog; others are direct API constraints verified from docs
- Test strategy: HIGH — test design follows established Phase 5 test patterns in codebase

**Research date:** 2026-04-24
**Valid until:** 2026-05-24 (sklearn, scipy, matplotlib are stable; API unlikely to change)
