"""E150 — ProtDCal-scale descriptors to close the charged gap (PPI-Affinity 0.71 on SAME data).

Verified: the98/T100 (where PPI got 0.71 high-charge) overlaps PDBbind heavily — same complexes, same Kd.
So the gap is FEATURES, not data: PPI computes ProtDCal's 23040 descriptors → selects 37; we had 29 hand-made.
This builds a ProtDCal-style pool — ~22 amino-acid property scales × {mean,std,max,min,sum,range,Nterm,
Cterm,autocorr-lag1/2/3} aggregations + structure-based charge descriptors (3D dipole, spatial charge
clustering, buried/interface charge) — then GBT (internal feature selection) with grouped CV on the charged
subset. Target: lift charged 0.42 → toward PPI's 0.71 on the same complexes.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
POS, NEG = set("KR"), set("DE")
PLROOT = ROOT / "data/drive_pull/pl/P-L"
THREE1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
          "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
          "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

# 22 amino-acid property scales (AAindex-style) — hydrophobicity variants, charge, size, electronic, SS.
SCALES = {
    "kd": {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2},
    "eisen": {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.9, "C": 0.29, "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.4, "I": 1.38, "L": 1.06, "K": -1.5, "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08},
    "hopp": {"A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0, "G": 0.0, "H": -0.5, "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5, "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5},
    "charge": {a: (1.0 if a in "KR" else -1.0 if a in "DE" else 0.5 if a == "H" else 0.0) for a in "ACDEFGHIKLMNPQRSTVWY"},
    "vol": {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0},
    "polar": {"A": 0.046, "R": 0.291, "N": 0.134, "D": 0.105, "C": 0.128, "Q": 0.18, "E": 0.151, "G": 0.0, "H": 0.23, "I": 0.186, "L": 0.186, "K": 0.219, "M": 0.221, "F": 0.29, "P": 0.131, "S": 0.062, "T": 0.108, "W": 0.409, "Y": 0.298, "V": 0.14},
    "pol_grantham": {"A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9},
    "mw": {"A": 89, "R": 174, "N": 132, "D": 133, "C": 121, "Q": 146, "E": 147, "G": 75, "H": 155, "I": 131, "L": 131, "K": 146, "M": 149, "F": 165, "P": 115, "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117},
    "bulk": {"A": 11.5, "R": 14.28, "N": 12.28, "D": 11.68, "C": 13.46, "Q": 14.45, "E": 13.57, "G": 3.4, "H": 13.69, "I": 21.4, "L": 21.4, "K": 15.71, "M": 16.25, "F": 19.8, "P": 17.43, "S": 9.47, "T": 15.77, "W": 21.67, "Y": 18.03, "V": 21.57},
    "flex": {"A": 0.36, "R": 0.53, "N": 0.46, "D": 0.51, "C": 0.35, "Q": 0.49, "E": 0.5, "G": 0.54, "H": 0.32, "I": 0.46, "L": 0.37, "K": 0.47, "M": 0.3, "F": 0.31, "P": 0.51, "S": 0.51, "T": 0.44, "W": 0.31, "Y": 0.42, "V": 0.39},
    "helix": {"A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.7, "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.0, "I": 1.08, "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57, "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06},
    "sheet": {"A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19, "Q": 1.1, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.6, "L": 1.3, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55, "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.7},
    "asa": {"A": 115, "R": 225, "N": 160, "D": 150, "C": 135, "Q": 180, "E": 190, "G": 75, "H": 195, "I": 175, "L": 170, "K": 200, "M": 185, "F": 210, "P": 145, "S": 115, "T": 140, "W": 255, "Y": 230, "V": 155},
    "refract": {"A": 4.34, "R": 26.66, "N": 13.28, "D": 12.0, "C": 35.77, "Q": 17.56, "E": 17.26, "G": 0.0, "H": 21.81, "I": 19.06, "L": 18.78, "K": 21.29, "M": 21.64, "F": 29.4, "P": 10.93, "S": 6.35, "T": 11.01, "W": 42.53, "Y": 31.53, "V": 13.92},
    "pI": {"A": 6.0, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07, "Q": 5.65, "E": 3.22, "G": 5.97, "H": 7.59, "I": 6.02, "L": 5.98, "K": 9.74, "M": 5.74, "F": 5.48, "P": 6.3, "S": 5.68, "T": 5.6, "W": 5.89, "Y": 5.66, "V": 5.96},
    "transfer": {"A": 0.5, "R": -11.2, "N": -0.2, "D": -7.4, "C": -2.8, "Q": -9.38, "E": -9.9, "G": 0.0, "H": -0.5, "I": 2.5, "L": 1.8, "K": -4.2, "M": 1.3, "F": 2.5, "P": -3.3, "S": -0.3, "T": -0.4, "W": 3.4, "Y": 2.3, "V": 1.5},
    "isa": {"A": 0.31, "R": -1.01, "N": -0.6, "D": -0.77, "C": 1.54, "Q": -0.22, "E": -0.64, "G": 0.0, "H": 0.13, "I": 1.8, "L": 1.7, "K": -0.99, "M": 1.23, "F": 1.79, "P": 0.72, "S": -0.04, "T": 0.26, "W": 2.25, "Y": 0.96, "V": 1.22},
    "nci": {"A": 0.007, "R": 0.043, "N": -0.014, "D": -0.024, "C": 0.038, "Q": -0.011, "E": -0.012, "G": 0.018, "H": -0.04, "I": 0.022, "L": 0.052, "K": 0.018, "M": 0.003, "F": 0.038, "P": 0.24, "S": -0.005, "T": 0.003, "W": 0.05, "Y": 0.023, "V": 0.057},
    "alpha_n": {"A": 0.42, "R": 0.36, "N": 0.21, "D": 0.25, "C": 0.17, "Q": 0.36, "E": 0.42, "G": 0.13, "H": 0.27, "I": 0.3, "L": 0.39, "K": 0.32, "M": 0.38, "F": 0.3, "P": 0.13, "S": 0.2, "T": 0.21, "W": 0.32, "Y": 0.25, "V": 0.27},
    "hbond": {a: (1.0 if a in "STNQYHKRWDE" else 0.0) for a in "ACDEFGHIKLMNPQRSTVWY"},
    "arom": {a: (1.0 if a in "FWY" else 0.0) for a in "ACDEFGHIKLMNPQRSTVWY"},
    "sidechain_vol": {"A": 27, "R": 105, "N": 58, "D": 52, "C": 44, "Q": 80, "E": 73, "G": 0, "H": 79, "I": 93, "L": 93, "K": 100, "M": 94, "F": 115, "P": 41, "S": 29, "T": 51, "W": 145, "Y": 117, "V": 67},
}
AGG = ["mean", "std", "max", "min", "sum", "range", "nterm", "cterm", "ac1", "ac2"]


def seq_descriptors(seq):
    L = max(1, len(seq))
    out = []
    for sc in SCALES.values():
        v = np.array([sc.get(c, 0.0) for c in seq], float)
        ac1 = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if L > 2 and np.std(v) > 0 else 0.0
        ac2 = float(np.corrcoef(v[:-2], v[2:])[0, 1]) if L > 3 and np.std(v) > 0 else 0.0
        out += [v.mean(), v.std(), v.max(), v.min(), v.sum(), v.max() - v.min(),
                v[0], v[-1], ac1, ac2]
    return out


def struct_charge_descriptors(pep_res):
    """3D charge descriptors from the bound peptide pose: dipole, spatial charge clustering."""
    chg, xyz = [], []
    for r in pep_res:
        aa = THREE1.get(r["rn"], "X")
        q = 1.0 if aa in POS else -1.0 if aa in NEG else (0.5 if aa == "H" else 0.0)
        c = np.mean(r["xyz"], 0)
        chg.append(q); xyz.append(c)
    chg = np.array(chg); xyz = np.array(xyz)
    if len(chg) == 0:
        return [0.0] * 5
    centroid = xyz.mean(0)
    dipole = float(np.linalg.norm((chg[:, None] * (xyz - centroid)).sum(0)))   # charge-weighted dipole
    chg_idx = np.where(np.abs(chg) > 0.9)[0]
    if len(chg_idx) > 1:
        cd = xyz[chg_idx]
        pair = np.linalg.norm(cd[:, None, :] - cd[None, :, :], axis=2)
        clust = float(pair[np.triu_indices(len(cd), 1)].mean())
        rg_chg = float(np.sqrt(((cd - cd.mean(0)) ** 2).sum(1).mean()))
    else:
        clust = rg_chg = 0.0
    net = float(chg.sum())
    return [dipole, clust, rg_chg, net, float(len(chg_idx))]


def parse_pep_mol2(mol2):
    lines = mol2.read_text().splitlines()
    if "@<TRIPOS>ATOM" not in lines:
        return None
    a = lines.index("@<TRIPOS>ATOM")
    res, cur = [], None
    for ln in lines[a + 1:]:
        if ln.startswith("@"):
            break
        f = ln.split()
        if len(f) < 9 or f[1][0] == "H":
            continue
        try:
            xyz = [float(f[2]), float(f[3]), float(f[4])]
        except ValueError:
            continue
        nm, rn = f[1], "".join(c for c in f[7] if c.isalpha()).upper()[:3]
        if nm == "N" or cur is None:
            cur = {"rn": rn, "xyz": []}; res.append(cur)
        cur["xyz"].append(xyz)
    return res


def cv(rows, mode, k=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(rows))
    y = np.array([r["y"] for r in rows])
    X = []
    for r in rows:
        row = list(r["feat"])
        if mode in ("protdcal", "all"):
            row += r["pdesc"]
        if mode in ("struct", "all"):
            row += r["sdesc"]
        X.append(row)
    X = np.array(X, float)
    pred = np.full(len(rows), np.nan)
    for f in range(k):
        tr = fold != f
        m = HistGradientBoostingRegressor(max_iter=600, max_depth=4, learning_rate=0.03,
                                          l2_regularization=4.0, min_samples_leaf=15, random_state=0).fit(X[tr], y[tr])
        pred[fold == f] = m.predict(X[fold == f])
    return pred, y


def metr(p, y):
    return pearsonr(p, y)[0], float(np.mean(np.abs(p - y))), float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    pdbb = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    rows = []
    for r in pdbb:
        q = abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))
        d = next((Path(p).parent for p in glob.glob(str(PLROOT / f"*/{r['pdb']}/{r['pdb']}_ligand.mol2"))), None)
        sdesc = [0.0] * 5
        if d is not None:
            pep = parse_pep_mol2(d / f"{r['pdb']}_ligand.mol2")
            if pep:
                sdesc = struct_charge_descriptors(pep)
        rows.append({"y": r["y"], "absq": q, "length": r["length"], "feat": [r[c] for c in PROD],
                     "pdesc": seq_descriptors(r["seq"]), "sdesc": sdesc})
    print(f"=== E150 ProtDCal-scale descriptors ({len(SCALES)} scales×{len(AGG)} aggs = {len(SCALES)*len(AGG)} + 5 struct) ===")
    absq = np.array([r["absq"] for r in rows])
    for name, sub in [("ALL (n=%d)" % len(rows), rows),
                      ("charged |q|≥2", [r for r in rows if r["absq"] >= 2]),
                      ("high |q|≥3", [r for r in rows if r["absq"] >= 3])]:
        if len(sub) < 25:
            continue
        print(f"\n--- {name} ---  r / MAE / RMSE")
        for lbl, mode in [("base-16", "base"), ("+protdcal", "protdcal"), ("+struct", "struct"), ("+ALL", "all")]:
            r, mae, rmse = metr(*cv(sub, mode))
            print(f"    {lbl:<14}{r:>+8.3f}{mae:>7.2f}{rmse:>7.2f}")
    print("\n  target charged → PPI 0.71. If +protdcal/+ALL lifts charged toward 0.6+, the gap WAS features.")


if __name__ == "__main__":
    main()
