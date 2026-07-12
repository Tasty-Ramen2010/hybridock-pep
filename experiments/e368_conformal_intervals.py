"""E368 — honest calibrated uncertainty for HybriDock-Pep ΔG via split conformal prediction.

Length/charge do NOT predict our error (E-checks: corr≈0, R²=0.003), so a per-peptide error bar keyed to
those would be miscalibrated. Instead we calibrate a **conformal** interval on the leave-cluster-out residuals:
a validated "pred ± q kcal/mol at (1−α) coverage" with a finite-sample marginal-coverage guarantee under
exchangeability. We test coverage honestly by splitting CLUSTERS (not rows) into calibrate/test, repeated.

We also test whether an ADAPTIVE (normalized) width — scaling q by a per-peptide difficulty σ(x) — beats the
global constant width, and whether the global interval is FAIR across length/charge subgroups.

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python experiments/e368_conformal_intervals.py
"""
from __future__ import annotations
import csv, json, math, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import cluster_by_identity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KB, KD = set("KR"), set("DE")
LEVELS = [0.50, 0.80, 0.90]
RNG = np.random.default_rng(0)


def q_conformal(cal_res, cov):
    """Finite-sample split-conformal quantile of |residual| for target coverage `cov`."""
    n = len(cal_res)
    k = math.ceil((n + 1) * cov)
    if k > n:
        return float(np.max(cal_res))  # coverage not attainable at this n → widest
    return float(np.sort(cal_res)[k - 1])


def main():
    rows = list(csv.DictReader(open(ROOT / "data/hybridock_blind_925.csv")))
    y = np.array([float(r["exp_dG_kcal_mol"]) for r in rows])
    p = np.array([float(r["pred_dG_kcal_mol"]) for r in rows])
    res = np.abs(y - p)
    seqs = [r["peptide"] for r in rows]
    L = np.array([len(s) for s in seqs])
    Q = np.array([abs(sum(c in KB for c in s) - sum(c in KD for c in s)) for s in seqs])
    clusters = cluster_by_identity(seqs, 0.60)
    uclu = np.array(sorted(set(clusters.tolist())))
    print(f"n={len(rows)}  MAE={res.mean():.2f}  RMSE={np.sqrt((res**2).mean()):.2f}  "
          f"{len(uclu)} identity clusters\n")

    # ---- cluster-aware split conformal, repeated ----
    REP = 300
    cov = {a: [] for a in LEVELS}
    wid = {a: [] for a in LEVELS}
    for _ in range(REP):
        RNG.shuffle(uclu)
        cal_cl = set(uclu[: len(uclu) // 2].tolist())
        cal = np.array([c in cal_cl for c in clusters])
        for a in LEVELS:
            qa = q_conformal(res[cal], a)
            cov[a].append(float((res[~cal] <= qa).mean()))
            wid[a].append(2 * qa)
    print("=== split conformal, cluster-split calibrate/test (300 repeats) ===")
    print(f"  {'target':>7} {'coverage (mean±sd)':>22} {'interval width':>16}")
    for a in LEVELS:
        print(f"  {int(a*100):>6}% {np.mean(cov[a])*100:>10.1f}% ± {np.std(cov[a])*100:>4.1f}"
              f"      ±{np.mean(wid[a])/2:>4.2f} kcal (full {np.mean(wid[a]):.2f})")

    # ---- deployable calibration on ALL residuals (what would ship) ----
    print("\n=== shipped calibration (q on all 925 leave-cluster-out residuals) ===")
    for a in LEVELS:
        print(f"  {int(a*100)}% interval: pred ± {q_conformal(res, a):.2f} kcal/mol")

    # ---- is the global interval FAIR across subgroups? (conditional coverage at 80%) ----
    q80 = q_conformal(res, 0.80)
    print(f"\n=== conditional coverage of the global 80% band (±{q80:.2f}) across subgroups ===")
    def cond(mask, name):
        if mask.sum() >= 20:
            print(f"  {name:16s} n={mask.sum():4d}  coverage={ (res[mask]<=q80).mean()*100:5.1f}%  "
                  f"mean|err|={res[mask].mean():.2f}")
    cond(L <= 8, "short ≤8"); cond((L > 8) & (L <= 15), "mid 9-15"); cond(L > 15, "long 16+")
    cond(Q == 0, "neutral"); cond((Q >= 1) & (Q <= 2), "charge 1-2"); cond(Q >= 3, "charge 3+")

    # ---- does ADAPTIVE (normalized) width beat global? test σ = prediction extremity ----
    print("\n=== adaptive (normalized) conformal vs global — does per-peptide width help? ===")
    sigma = np.abs(p - np.median(p)) + 0.5  # difficulty proxy: extreme predictions (+floor)
    for label, s in [("GLOBAL (σ=const)", np.ones_like(res)), ("ADAPTIVE (σ=|pred−median|)", sigma)]:
        cvs, wsd = [], []
        for _ in range(200):
            RNG.shuffle(uclu)
            cal_cl = set(uclu[: len(uclu) // 2].tolist())
            cal = np.array([c in cal_cl for c in clusters])
            score = res[cal] / s[cal]
            qn = q_conformal(score, 0.80)
            w = qn * s[~cal]                 # per-point half width
            cvs.append(float((res[~cal] <= w).mean()))
            wsd.append(float(np.std(2 * w)))
        print(f"  {label:28s} 80%-coverage={np.mean(cvs)*100:5.1f}%   width-spread(sd)={np.mean(wsd):.2f} kcal")
    print("  (adaptive is only worth it if coverage holds AND width-spread > 0 meaningfully — i.e. it"
          " genuinely tightens easy cases and widens hard ones.)")


if __name__ == "__main__":
    main()
