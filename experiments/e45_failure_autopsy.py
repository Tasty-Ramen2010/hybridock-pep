"""E45 — deep failure autopsy: which peptides do WE and FLEXPEPDOCK both fail on, and why.

Signed residuals (over/undershoot), shared failures, and the physics drivers: charge, proline/
glycine (rigidity/flex), aromatic, terminal tails, secondary structure, and STRUCTURE QUALITY
(Rosetta fa_rep = clashes / 'unreal' structure). Pooled crystal-65 + the-98.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import PDBParser  # noqa: E402
from hybridock_pep.scoring.geometry_features import (GEOMETRY_FEATURE_KEYS,  # noqa: E402
                                                     compute_geometry_features)
from scipy.stats import pearsonr  # noqa: E402

P = PDBParser(QUIET=True)
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
ROS = ["fa_atr", "fa_rep", "fa_sol", "fa_elec", "hbond_lr_bb", "hbond_sc", "hbond_bb_sc", "lk_ball_wtd"]


def seq_of(pdb):
    return "".join(A3.get(r.resname.upper(), "X") for r in P.get_structure("p", str(pdb))[0].get_residues()
                   if r.id[0] == " ")


def seqfeat(seq):
    L = max(1, len(seq))
    return dict(
        charged_frac=sum(c in "DEKR" for c in seq) / L,
        net_charge=(seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")),
        pro_frac=seq.count("P") / L, gly_frac=seq.count("G") / L,
        arom_frac=sum(c in "FWYH" for c in seq) / L,
        bulky_frac=sum(c in "FWYLIM" for c in seq) / L,
        # terminal tail flexibility: G/P/S/charged at the ends
        term_flex=sum(c in "GSDEKR" for c in (seq[:2] + seq[-2:])) / 4.0,
        L=len(seq),
    )


def main():
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    ros_cr = json.loads(Path("/tmp/e43_cr.json").read_text())
    ros_b = json.loads(Path("/tmp/e43_b98.json").read_text())
    work = Path("/tmp/ppep_work")
    rows = []
    for k, rt in ros_cr.items():
        if k not in bench:
            continue
        g = compute_geometry_features(Path(bench[k]["peptide_pdb"]), Path(bench[k]["pocket_pdb"]))
        if not g:
            continue
        sq = bench[k]["peptide_seq"]
        rows.append(dict(g, pdb=k, set="cr", y=rt["y"], seq=sq, ss="?",
                         **{f"ros_{t}": rt[t] for t in ROS}, **seqfeat(sq)))
    for k, v in e28.items():
        if k not in ros_b:
            continue
        g = compute_geometry_features(work / f"{k}_pep.pdb", work / f"{k}_rec.pdb")
        if not g:
            continue
        sq = seq_of(work / f"{k}_pep.pdb")
        rows.append(dict(g, pdb=k, set="b98", y=v["y"], seq=sq, ss=v.get("ss", "?"),
                         **{f"ros_{t}": ros_b[k][t] for t in ROS}, **seqfeat(sq)))
    y = np.array([r["y"] for r in rows])
    G = GEOMETRY_FEATURE_KEYS

    def loo(feats):
        X = np.array([[r.get(f, 0.0) for f in feats] for r in rows]); p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd]); w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            p[i] = np.r_[1, (X[i] - mu) / sd] @ w
        return p
    pred_us = loo(G)
    pred_ros = loo([f"ros_{t}" for t in ROS])
    res_us = pred_us - y          # >0 = predict too WEAK (undershoot strength)
    res_ros = pred_ros - y
    print(f"n={len(rows)} | ours r={pearsonr(pred_us,y).statistic:+.3f} | rosetta r={pearsonr(pred_ros,y).statistic:+.3f}")
    print(f"corr(our residual, rosetta residual) = {pearsonr(res_us,res_ros).statistic:+.3f} "
          "(high = we fail on the SAME peptides)\n")

    # SHARED failures: both models far off
    au, ar = np.abs(res_us), np.abs(res_ros)
    shared = (au > np.percentile(au, 70)) & (ar > np.percentile(ar, 70))
    print(f"=== peptides BOTH models fail on (n={shared.sum()}): what do they share? ===")
    print(f"  {'property':<14}{'FAIL mean':>11}{'rest mean':>11}{'enriched?':>11}")
    props = ["charged_frac", "pro_frac", "gly_frac", "arom_frac", "term_flex", "L",
             "ros_fa_rep", "ros_fa_elec", "frac_pol_satisfied"]
    for pname in props:
        v = np.array([r.get(pname, 0.0) for r in rows])
        fm, rm = v[shared].mean(), v[~shared].mean()
        flag = "YES" if abs(fm - rm) > 0.3 * (v.std() + 1e-9) else ""
        print(f"  {pname:<14}{fm:>11.2f}{rm:>11.2f}{flag:>11}")

    # UNDERSHOOT vs OVERSHOOT: what drives signed error?
    print("\n=== UNDER(+)/OVER(-)shoot drivers (corr of SIGNED residual with property) ===")
    print("  (+ = property makes us predict too WEAK; − = too STRONG)")
    for pname in ["charged_frac", "net_charge", "pro_frac", "gly_frac", "arom_frac", "bulky_frac",
                  "term_flex", "L", "ros_fa_rep", "y"]:
        v = np.array([r.get(pname, 0.0) for r in rows])
        if v.std() > 0:
            print(f"  corr(signed_resid, {pname:<14}) = {pearsonr(res_us, v).statistic:+.3f}")

    # worst 12 with characterization
    order = np.argsort(-au)
    print("\n=== 12 WORST for us (signed err; +under/−over) ===")
    print(f"  {'pdb':<7}{'set':<5}{'exp':>6}{'pred':>6}{'err':>6}{'chg':>5}{'pro':>5}{'fa_rep':>8}  seq")
    for i in order[:12]:
        r = rows[i]
        print(f"  {r['pdb']:<7}{r['set']:<5}{y[i]:>6.1f}{pred_us[i]:>6.1f}{res_us[i]:>+6.1f}"
              f"{r['charged_frac']:>5.2f}{r['pro_frac']:>5.2f}{r['ros_fa_rep']:>8.1f}  {r['seq'][:20]}")


if __name__ == "__main__":
    main()
