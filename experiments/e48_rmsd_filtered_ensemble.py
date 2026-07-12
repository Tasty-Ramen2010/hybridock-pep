"""E48 — Ram's fix: RMSD-to-best filter -> same-mode ensemble (the CORRECT partial ensemble).

E47 built the partition function over all 100 RAPiDock poses, but those are spread 6 Å RMSD across
the site (docking uncertainty, many misdocks) — averaging in garbage. The cheap fix (pipeline
already has Kabsch/RMSD): pick the best pose, RMSD every pose against it IN THE RECEPTOR FRAME (no
re-superposition — flipped registers stay 'far', which is correct), keep only poses within tau Å =
the genuine thermal neighbourhood of one bound mode. Build g_ens / N_eff (bound entropy) over THAT
subset; the dropped fraction is a docking-confidence signal, not entropy.

Caches per-pose energy + rmsd-to-best so tau can be swept without recompute. Compares rank-1 vs
naive(all 100) vs RMSD-filtered ensembles at tau in {2,3,4} Å.
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

KT = 0.593
CAL = EnsembleCalibration.load(ROOT / "data/ensemble_calibration.json")
FEATS = CAL.feature_names
GM, GS = np.array(CAL.geo_mean), np.array(CAL.geo_std)


def geo_energy(feat):
    z = (np.array([feat.get(f, 0.0) for f in FEATS]) - GM) / (GS + 1e-9)
    return float(CAL.geo_intercept + np.dot(CAL.geo_weights, z))


def ca(p):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in Path(p).read_text().splitlines()
                     if l.startswith("ATOM") and l[12:16].strip() == "CA"])


def build():
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    cache = Path("/tmp/e48_perpose.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for pdb, meta in bench.items():
        if pdb in out:
            continue
        pdir = ROOT / f"logs/crystal65_n100/cr_{pdb}/poses"
        rec = ROOT / meta["pocket_pdb"]
        poses = sorted(pdir.glob("pose_*.pdb"), key=lambda p: int(p.stem.split("_")[1]))
        if len(poses) < 20 or not rec.exists():
            continue
        E, C = [], []
        for p in poses:
            f = compute_geometry_features(p, rec)
            if f:
                E.append(geo_energy(f)); C.append(ca(p))
        if len(E) < 20:
            continue
        E = np.array(E)
        # Reference = pose_0 (diffusion rank-1), the reliable "best pose" — NOT argmin(E):
        # e47 showed our geometry energy is a poor re-ranker (e_min r=0.33 << e_rank1 0.72),
        # so its argmin picks a decoy. Filter the same-mode neighbourhood around pose_0.
        cb = C[0]
        rmsd = []
        for c in C:
            m = min(len(c), len(cb))
            rmsd.append(float(np.sqrt(((c[:m] - cb[:m]) ** 2).sum(1).mean())) if m else 99.0)
        out[pdb] = dict(y=meta["dg_exp"], E=[float(x) for x in E], rmsd=rmsd)
        cache.write_text(json.dumps(out))
        if len(out) % 10 == 0:
            print(f"  {len(out)} done", flush=True)
    return out


def ens(E, keep):
    """partition-function free energy + bound-entropy N_eff over the kept (same-mode) poses."""
    e = E[keep]
    w = np.exp(-(e - e.min()) / KT); w /= w.sum()
    g = float(-KT * np.log(np.exp(-(e - e.min()) / KT).sum()) + e.min())
    n_eff = float(np.exp(-(w * np.log(w + 1e-12)).sum()))
    return g, n_eff, len(e)


def main():
    out = build()
    ks = list(out)
    y = np.array([out[k]["y"] for k in ks])

    def loo(vecs):
        X = np.column_stack(vecs); p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
            w, *_ = np.linalg.lstsq(A, y[tr], rcond=None); p[i] = np.r_[1, (X[i] - mu) / sd] @ w
        return pearsonr(p, y).statistic

    rank1 = np.array([out[k]["E"][0] for k in ks])
    allg = np.array([ens(np.array(out[k]["E"]), np.arange(len(out[k]["E"])))[0] for k in ks])
    r_r1 = pearsonr(rank1, y).statistic
    print(f"\n=== RMSD-to-pose_0 filter: does same-mode ensemble ADD to rank-1? (n={len(ks)}) ===")
    print(f"  rank-1 (pose_0)  r={r_r1:+.3f}   [strong baseline]")
    print(f"  naive g_ens(100) r={pearsonr(allg,y).statistic:+.3f}   [averages in 6Å misdocks]")
    print(f"  {'tau(Å)':>6}{'g_ens':>9}{'meanE':>9}{'r1+Neff':>10}{'r1+g+Ne':>10}{'kept':>7}{'N_eff':>7}")
    for tau in (1.5, 2.0, 3.0):
        g, me, ne, nk = [], [], [], []
        for k in ks:
            E = np.array(out[k]["E"]); rm = np.array(out[k]["rmsd"])
            keep = np.where(rm <= tau)[0]
            if len(keep) < 2:
                keep = np.array([0])  # fall back to pose_0
            gg, nn, kk = ens(E, keep)
            g.append(gg); me.append(float(E[keep].mean())); ne.append(nn); nk.append(kk)
        g, me, ne = np.array(g), np.array(me), np.array(ne)
        print(f"  {tau:>6.1f}{pearsonr(g,y).statistic:>+9.3f}{pearsonr(me,y).statistic:>+9.3f}"
              f"{loo([rank1,ne]):>+10.3f}{loo([rank1,g,ne]):>+10.3f}{np.mean(nk):>7.1f}{np.mean(ne):>7.1f}")
    print("  >> r1+Neff > rank-1 ? then BOUND configurational entropy (same-mode pose spread) is")
    print("     real new signal on top of the diffusion model's best pose — the cheap ensemble win.")


if __name__ == "__main__":
    main()
