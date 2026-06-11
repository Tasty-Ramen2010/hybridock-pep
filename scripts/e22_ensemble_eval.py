"""E22 eval — geometry+Vina ensemble on REAL poses + "good Vina parts" test.

Combines:
  geometry features on real top-5 poses (recomputed here, same as e19_realpose top5_mean)
  vina on real poses (rigid score_only, /tmp/e22_vina_real.json) -> "good part" (no torsion penalty)
  vina flexible benchmark (data/benchmark_crystal.json vina_docked) -> size-biased "all of vina"

Tests, all LOO on crystal-65:
  1. geometry alone (real poses)
  2. + flexible benchmark Vina (size-biased)  ensemble
  3. + rigid real-pose Vina ("good part")     ensemble    <- the hypothesis
  4. size-bias check: corr(each vina, length)
Fits + saves the production EnsembleCalibration for the winning variant.
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
sys.path.insert(0, str(ROOT / "scripts"))
from Bio.PDB import PDBParser  # noqa: E402
from e19_decompose_recover import interface_features, pocket_descriptors  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

P = PDBParser(QUIET=True)
GEN = ROOT / "logs/crystal65_n100"
POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]
GEO = POCK + IFACE


def merge(pep_pose, poc_pdb, out):
    L = [l[:21] + "P" + l[22:] for l in Path(pep_pose).read_text().splitlines()
         if l.startswith(("ATOM", "HETATM")) and l[17:20] != "HOH"]
    L += [l[:21] + "R" + l[22:] for l in Path(poc_pdb).read_text().splitlines()
          if l.startswith(("ATOM", "HETATM")) and l[17:20] != "HOH"]
    Path(out).write_text("\n".join(L) + "\nEND\n")


def geo_top5(pdb, meta):
    posedir = GEN / f"cr_{pdb}" / "poses"
    poc = str((ROOT / meta["pocket_pdb"]).resolve())
    pep_free = str((ROOT / meta["peptide_pdb"]).resolve())
    seq = meta["peptide_seq"]
    feats = []
    for i in range(5):
        pose = posedir / f"pose_{i}.pdb"
        if not pose.exists():
            continue
        tmp = f"/tmp/e22geo/{pdb}_{i}.pdb"
        Path(tmp).parent.mkdir(exist_ok=True)
        merge(str(pose), poc, tmp)
        fi = interface_features(pep_free, tmp, "P", len(seq))
        pk = pocket_descriptors(P.get_structure("m", tmp)[0], "P")
        if fi and pk:
            feats.append({**fi, **pk})
    if not feats:
        return None
    return {k: float(np.mean([f[k] for f in feats])) for k in feats[0]}


def loo(X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return p


def rr(p, y):
    return pearsonr(p, y).statistic, float(np.sqrt(((p - y) ** 2).mean()))


def main():
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    vr = json.loads(Path("/tmp/e22_vina_real.json").read_text())
    rows = []
    for pdb, meta in bench.items():
        if pdb not in vr:
            continue
        g = geo_top5(pdb, meta)
        if not g:
            continue
        g = dict(g); g["L"] = float(meta["peptide_len"])
        rows.append(dict(pdb=pdb, y=meta["dg_exp"],
                         vina_rigid=vr[pdb]["vina_total"], vina_flex=meta["vina_docked"], **g))
    y = np.array([r["y"] for r in rows])
    L = np.array([r["L"] for r in rows], float)
    Xg = np.array([[r.get(f, 0.0) for f in GEO] for r in rows])
    v_rigid = np.array([r["vina_rigid"] for r in rows])
    v_flex = np.array([r["vina_flex"] for r in rows])
    print(f"n={len(rows)} complexes with real-pose geometry + Vina\n")

    print("=== size-bias check: corr(vina, peptide length) ===")
    print(f"  flexible benchmark Vina : {pearsonr(v_flex, L).statistic:+.3f}  (more negative = more size-biased)")
    print(f"  rigid real-pose Vina    : {pearsonr(v_rigid, L).statistic:+.3f}  <- the 'good part'?")

    pg = loo(Xg, y)
    pvf = loo(v_flex.reshape(-1, 1), y)
    pvr = loo(v_rigid.reshape(-1, 1), y)
    print("\n=== individual (real-pose LOO) ===")
    print(f"  geometry top5         r={rr(pg,y)[0]:+.3f}  RMSE {rr(pg,y)[1]:.2f}")
    print(f"  vina flexible (bench) r={rr(pvf,y)[0]:+.3f}  RMSE {rr(pvf,y)[1]:.2f}")
    print(f"  vina rigid (good part)r={rr(pvr,y)[0]:+.3f}  RMSE {rr(pvr,y)[1]:.2f}")

    def z(p):
        return (p - p.mean()) / p.std()
    print("\n=== ENSEMBLE geometry + vina (z-blend) ===")
    best = None
    for label, pv in [("flex(all-vina)", pvf), ("rigid(good-part)", pvr)]:
        for w in (0.4, 0.5, 0.6):
            blend = w * pg + (1 - w) * pv
            r, rmse = rr(blend, y)
            tag = ""
            if best is None or r > best[0]:
                best = (r, rmse, label, w)
            print(f"  {w:.1f}*geo + {1-w:.1f}*{label:<16} r={r:+.3f}  RMSE {rmse:.2f}")
    print(f"\n  >> BEST: {best[2]} blend={best[3]} -> r={best[0]:+.3f} RMSE={best[1]:.2f}")
    print(f"  guess-mean RMSE={y.std():.2f} | oracle ensemble was 0.62 | real-pose geom alone ~0.47")

    # fit + save production calibration for the winning variant (use total vina from real poses)
    from hybridock_pep.scoring.ensemble import fit_ensemble_calibration
    recs = [dict(r, vina=r["vina_rigid"] if "rigid" in best[2] else r["vina_flex"]) for r in rows]
    cal = fit_ensemble_calibration(recs, blend=best[3], vina_mode="total")
    out = ROOT / "data/ensemble_calibration.json"
    cal.save(out)
    print(f"\n  saved production calibration -> {out} (blend={best[3]}, vina={best[2]})")


if __name__ == "__main__":
    main()
