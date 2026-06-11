"""E19 final — recompute pocket+interface features on REAL RAPiDock poses and re-eval.

The deployment-faithful test. For each of the 65 Kd complexes we now have N=100 RAPiDock
diffusion poses. Score several pose choices and re-run crystal-65 LOO:
  - crystal (oracle)         : upper bound (peptide_pdb)
  - rank1                    : pose_0 = default user pose (true deployed, single pose)
  - top5/25/100_mean         : ENSEMBLE-average features over top-K poses (denoise pose noise)
  - bestrmsd                 : closest-to-crystal of the 100 (sampler ceiling)

Reports LOO r (pocket / interface / combined) AND combined RMSE in kcal/mol vs the
guess-the-mean baseline. Tests Ram's hypothesis: averaging features over many poses
rescues the pose-fragile interface term.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from Bio.PDB import PDBParser  # noqa: E402
from e19_decompose_recover import interface_features, pocket_descriptors  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

P = PDBParser(QUIET=True)
GEN = ROOT / "logs/crystal65_n100"
POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]


def merge(pep_pose, poc_pdb, out):
    lines = []
    for l in Path(pep_pose).read_text().splitlines():
        if l.startswith(("ATOM", "HETATM")) and l[17:20] != "HOH":
            lines.append(l[:21] + "P" + l[22:])
    for l in Path(poc_pdb).read_text().splitlines():
        if l.startswith(("ATOM", "HETATM")) and l[17:20] != "HOH":
            lines.append(l[:21] + "R" + l[22:])
    Path(out).write_text("\n".join(lines) + "\nEND\n")


def feats_for_pose(pep_pose, pep_free_pdb, poc_pdb, seq):
    tmp = Path(f"/tmp/e19real/{Path(pep_pose).parent.parent.name}_{Path(pep_pose).stem}.pdb")
    tmp.parent.mkdir(exist_ok=True)
    merge(pep_pose, poc_pdb, str(tmp))
    fi = interface_features(pep_free_pdb, str(tmp), "P", len(seq))
    pk = pocket_descriptors(P.get_structure("m", str(tmp))[0], "P")
    if not fi or not pk:
        return None
    return {**fi, **pk}


def ensemble_mean_feats(posedir, pep_free, poc, seq, k):
    """Average pocket+interface features over the top-k RAPiDock poses (denoise pose noise)."""
    feats = []
    for i in range(k):
        p = posedir / f"pose_{i}.pdb"
        if not p.exists():
            continue
        f = feats_for_pose(str(p), pep_free, poc, seq)
        if f:
            feats.append(f)
    if not feats:
        return None
    return {kk: float(np.mean([f[kk] for f in feats])) for kk in feats[0].keys()}


def _loo_pred(rows, feats, y):
    X = np.array([[r.get(f, 0.0) for f in feats] for r in rows], float)
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pred


def loo(rows, feats, y):
    return pearsonr(_loo_pred(rows, feats, y), y).statistic


def main():
    b = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    res_path = GEN / "benchmark_results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    variants = ["crystal", "rank1", "top5_mean", "top25_mean", "top100_mean", "bestrmsd"]
    rows = {v: [] for v in variants}
    ys = {v: [] for v in variants}
    done = 0
    for pdb, meta in b.items():
        cname = f"cr_{pdb}"
        posedir = GEN / cname / "poses"
        if not posedir.exists():
            continue
        seq = meta["peptide_seq"]
        poc = str((ROOT / meta["pocket_pdb"]).resolve())
        pep_free = str((ROOT / meta["peptide_pdb"]).resolve())
        y = meta["dg_exp"]

        def add(key, f):
            if f:
                rows[key].append(f); ys[key].append(y)

        add("crystal", feats_for_pose(pep_free, pep_free, poc, seq))
        p0 = posedir / "pose_0.pdb"
        add("rank1", feats_for_pose(str(p0), pep_free, poc, seq) if p0.exists() else None)
        add("top5_mean", ensemble_mean_feats(posedir, pep_free, poc, seq, 5))
        add("top25_mean", ensemble_mean_feats(posedir, pep_free, poc, seq, 25))
        add("top100_mean", ensemble_mean_feats(posedir, pep_free, poc, seq, 100))
        rm = (res.get(cname, {}) or {}).get("ref_rmsds") or []
        if rm:
            bi = int(np.argmin(rm))
            pb = posedir / f"pose_{bi}.pdb"
            add("bestrmsd", feats_for_pose(str(pb), pep_free, poc, seq) if pb.exists() else None)
        done += 1
        if done % 10 == 0:
            print(f"  {done} complexes processed", flush=True)

    print("\nDEPLOYMENT-FAITHFUL EVAL - real RAPiDock poses (LOO r):")
    print(f"{'pose choice':<14}{'n':>4}{'pocket':>9}{'interface':>11}{'pock+iface':>12}{'RMSE_kcal':>11}")
    for k in variants:
        if len(rows[k]) < 5:
            print(f"{k:<14}{len(rows[k]):>4}  (too few)"); continue
        y = np.array(ys[k])
        rb = loo(rows[k], POCK + IFACE, y)
        pred = _loo_pred(rows[k], POCK + IFACE, y)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        print(f"{k:<14}{len(rows[k]):>4}{loo(rows[k],POCK,y):>9.3f}"
              f"{loo(rows[k],IFACE,y):>11.3f}{rb:>12.3f}{rmse:>11.2f}")
    yall = np.array([m["dg_exp"] for m in b.values()])
    print(f"\nguess-the-mean baseline RMSE = {yall.std():.2f} kcal/mol  (beat this)")
    print("crystal=oracle | rank1=default single pose | topK_mean=ENSEMBLE avg over K | bestrmsd=ceiling")
    allrm = [min(v["ref_rmsds"]) for v in res.values() if v.get("ref_rmsds")]
    if allrm:
        print(f"median best-of-100 RMSD = {np.median(allrm):.2f} A  (n={len(allrm)})")


if __name__ == "__main__":
    main()
