"""E47 — partial ensemble from the N=100 RAPiDock poses we already have (the cheapest rung up
the FEP/LIE ladder, CPU-only). The diffusion model already sampled the bound ensemble; we've been
collapsing it to pose_0. Build the discrete partition function over the poses and test whether the
ensemble free energy + a BOUND configurational-entropy term beat the single rank-1 pose.

Per complex, for the N poses, with a cheap per-pose energy E_i (geometry linear ΔG prediction from
the production calibration's geometry block — self-consistent, no Vina needed):
  G_ens   = -kT ln Σ e^{-E_i/kT}        discrete partition-function free energy (≈ -kT ln Z)
  w_i     = e^{-E_i/kT} / Σ             Boltzmann weights
  N_eff   = exp(-Σ w_i ln w_i)          effective # populated basins = BOUND conf. entropy proxy
  E_min   = min_i E_i ; E_mean10 = mean of 10 lowest ; rmsd_spread = Cα-RMSD spread of low-E poses
Tests vs experimental ΔG: does any ensemble observable beat the rank-1 (pose_0) baseline, and does
N_eff (bound entropy) add to it? kT = 0.593 kcal/mol (310 K).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.ensemble import EnsembleCalibration  # noqa: E402
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

KT = 0.593  # kcal/mol at ~310 K
CAL = EnsembleCalibration.load(ROOT / "data/ensemble_calibration.json")
FEATS = CAL.feature_names
GM, GS = np.array(CAL.geo_mean), np.array(CAL.geo_std)


def geo_energy(feat: dict) -> float:
    z = (np.array([feat.get(f, 0.0) for f in FEATS]) - GM) / (GS + 1e-9)
    return float(CAL.geo_intercept + np.dot(CAL.geo_weights, z))


def ca_coords(pdb: Path):
    xs = []
    for ln in pdb.read_text().splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            xs.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return np.array(xs)


def main():
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    cache = Path("/tmp/e47_ens.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for pdb, meta in bench.items():
        if pdb in out:
            continue
        pdir = ROOT / f"logs/crystal65_n100/cr_{pdb}/poses"
        rec = ROOT / meta["pocket_pdb"]
        poses = sorted(pdir.glob("pose_*.pdb"), key=lambda p: int(p.stem.split("_")[1]))
        if len(poses) < 20 or not rec.exists():
            continue
        E, cas = [], []
        for p in poses:
            f = compute_geometry_features(p, rec)
            if f:
                E.append(geo_energy(f)); cas.append(p)
        if len(E) < 20:
            continue
        E = np.array(E)
        w = np.exp(-(E - E.min()) / KT); w /= w.sum()
        n_eff = float(np.exp(-(w * np.log(w + 1e-12)).sum()))
        g_ens = float(-KT * (np.log(np.exp(-(E - E.min()) / KT).sum())) + E.min())
        order = np.argsort(E)
        lowc = [ca_coords(cas[i]) for i in order[:10]]
        m = min(len(c) for c in lowc)
        spread = float(np.mean([np.sqrt(((lowc[a][:m] - lowc[b][:m]) ** 2).sum(1).mean())
                                for a in range(len(lowc)) for b in range(a + 1, len(lowc))])) if m else 0.0
        out[pdb] = dict(y=meta["dg_exp"], e_rank1=float(E[0]), e_min=float(E.min()),
                        e_mean10=float(np.sort(E)[:10].mean()), g_ens=g_ens,
                        n_eff=n_eff, rmsd_spread=spread, n=len(E))
        cache.write_text(json.dumps(out))
        if len(out) % 10 == 0:
            print(f"  {len(out)} complexes done", flush=True)

    ks = list(out)
    y = np.array([out[k]["y"] for k in ks])
    print(f"\n=== partial ensemble over N≈100 RAPiDock poses (n={len(ks)} complexes) ===")
    print(f"  {'observable':<14}{'r vs ΔG':>10}   meaning")
    desc = {"e_rank1": "pose_0 only (BASELINE)", "e_min": "best single pose",
            "e_mean10": "mean of 10 lowest", "g_ens": "−kT ln Σe^−E/kT (partition fn)",
            "n_eff": "bound conf. entropy (N_eff)", "rmsd_spread": "low-E pose spread"}
    for f in ["e_rank1", "e_min", "e_mean10", "g_ens", "n_eff", "rmsd_spread"]:
        v = np.array([out[k][f] for k in ks])
        r = pearsonr(v, y).statistic if v.std() > 0 else 0.0
        print(f"  {f:<14}{r:>+10.3f}   {desc[f]}")

    def loo(feats):
        X = np.array([[out[k][f] for f in feats] for k in ks]); p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
            w2, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            p[i] = np.r_[1, (X[i] - mu) / sd] @ w2
        return pearsonr(p, y).statistic
    print("\n=== does the ensemble (+ bound entropy) beat rank-1? (LOO) ===")
    print(f"  rank-1 alone          r={loo(['e_rank1']):+.3f}")
    print(f"  g_ens alone           r={loo(['g_ens']):+.3f}")
    print(f"  g_ens + n_eff         r={loo(['g_ens', 'n_eff']):+.3f}")
    print(f"  g_ens + n_eff + spread r={loo(['g_ens', 'n_eff', 'rmsd_spread']):+.3f}")
    print("  >> if g_ens/n_eff beat e_rank1, the discarded 99 poses carry partition-function signal")


if __name__ == "__main__":
    main()
