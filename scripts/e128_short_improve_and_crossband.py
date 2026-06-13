"""E128 — anchor features across ALL bands + deeper short features to push past 0.54.

(1) Test the e127 anchor features on short/med/long/vlong — confirm short-specific (Ram expects yes).
(2) The short residual: we still miss deep-hydrophobic (under-pred) and polyproline (over-pred). Add
    chemistry-aware anchor features and find what pushes short higher:
      hyd_anchor_depth   = max_i (burial_i · max(0,KD_i))        deep HYDROPHOBIC plug (AVGIGAV/FLSYK)
      charged_anchor     = max_i (burial_i · is_saltbridged_i)   buried salt-bridged charge
      arom_anchor_depth  = max_i (burial_i · is_aromatic_i)      aromatic stacking anchor
      hbond_per_contact  = hb_count / n_contact_res              specific favorable density (polyPro low)
      pro_run            = longest proline run / L                polyproline penalty (PPPLPP/RPPG)
      buried_inert       = max_burial · (1 - hbond_per_contact)  buried-but-inert (over-pred signature)
Final: best short feature set + cross-band Δ.
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
PLROOT = ROOT / "data/drive_pull/pl/P-L"
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
      "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
THREE1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
          "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
          "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def cc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    return pearsonr(x[m], y[m])[0] if m.sum() > 4 and np.std(x[m]) > 0 else np.nan


def parse_pep(mol2):
    lines = mol2.read_text().splitlines()
    if "@<TRIPOS>ATOM" not in lines:
        return None
    a = lines.index("@<TRIPOS>ATOM")
    atoms = []
    for ln in lines[a + 1:]:
        if ln.startswith("@"):
            break
        f = ln.split()
        if len(f) < 9 or f[1][0] == "H":
            continue
        try:
            atoms.append((f[1], "".join(c for c in f[7] if c.isalpha()).upper()[:3],
                          np.array([float(f[2]), float(f[3]), float(f[4])])))
        except ValueError:
            continue
    res, cur = [], None
    for nm, rn, xyz in atoms:
        if nm == "N":
            cur = {"rn": rn, "xyz": [], "at": {}}
            res.append(cur)
        if cur is None:
            cur = {"rn": rn, "xyz": [], "at": {}}
            res.append(cur)
        cur["xyz"].append(xyz)
        cur["at"][nm] = xyz
    return res


def rec_data(pdb):
    heavy, rows = [], {}
    for ln in pdb.read_text().splitlines():
        if not ln.startswith("ATOM") or ln[12:16].strip()[:1] == "H":
            continue
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        heavy.append(xyz)
        rn = ln[17:20].strip()
        if rn in ("LYS", "ARG", "ASP", "GLU"):
            rows.setdefault((ln[21], ln[22:27], rn), {})[ln[12:16].strip()] = xyz
    charged = []
    for (c, n, rn), at in rows.items():
        if rn in ("LYS", "ARG"):
            p = at.get("NZ", at.get("CZ"))
            if p is not None:
                charged.append((+1, p))
        else:
            o = [at[k] for k in ("OD1", "OD2", "OE1", "OE2", "CG", "CD") if k in at]
            if o:
                charged.append((-1, np.mean(o, 0)))
    return (np.array(heavy) if heavy else np.zeros((1, 3))), charged


def features(pid, hb_count):
    d = next((Path(p).parent for p in glob.glob(str(PLROOT / f"*/{pid}/{pid}_ligand.mol2"))), None)
    if d is None:
        return None
    res = parse_pep(d / f"{pid}_ligand.mol2")
    if not res:
        return None
    rec, rcharge = rec_data(d / f"{pid}_protein.pdb")
    burial, hyd_depth, arom_depth, chg_depth, contacts_tot = [], 0.0, 0.0, 0.0, 0
    salt = 0
    pro_run = cur_run = 0
    for r in res:
        aa = THREE1.get(r["rn"], "X")
        rx = np.array(r["xyz"])
        nb = int((np.linalg.norm(rec - rx.mean(0), axis=1) < 8.0).sum()) if rx.size else 0
        nc = int((np.linalg.norm(rec[:, None, :] - rx[None, :, :], axis=2).min(0) < 4.5).sum()) if rx.size else 0
        burial.append(nb)
        contacts_tot += (nc > 0)
        hyd_depth = max(hyd_depth, nb * max(0.0, KD.get(aa, 0)))
        if aa in "FWY":
            arom_depth = max(arom_depth, nb)
        # salt bridge for this residue
        cc_ = None
        if aa == "K" and "NZ" in r["at"]:
            cc_ = (+1, r["at"]["NZ"])
        elif aa == "R" and "CZ" in r["at"]:
            cc_ = (+1, r["at"]["CZ"])
        elif aa == "D":
            o = [r["at"][k] for k in ("OD1", "OD2", "CG") if k in r["at"]]
            cc_ = (-1, np.mean(o, 0)) if o else None
        elif aa == "E":
            o = [r["at"][k] for k in ("OE1", "OE2", "CD") if k in r["at"]]
            cc_ = (-1, np.mean(o, 0)) if o else None
        if cc_ and rcharge and any(cc_[0] * sr < 0 and np.linalg.norm(cc_[1] - xr) < 4.5 for sr, xr in rcharge):
            salt += 1
            chg_depth = max(chg_depth, float(nb))
        cur_run = cur_run + 1 if aa == "P" else 0
        pro_run = max(pro_run, cur_run)
    burial = np.array(burial)
    n_contact = max(1, contacts_tot)
    hpc = hb_count / n_contact
    return {"max_burial": float(burial.max()), "burial_concentration": float(burial.max() / (burial.sum() + 1e-9)),
            "best_salt_bridge": float(salt), "hyd_anchor_depth": hyd_depth, "arom_anchor_depth": arom_depth,
            "charged_anchor": chg_depth, "hbond_per_contact": hpc, "pro_run": pro_run / max(1, len(res)),
            "buried_inert": float(burial.max()) * (1.0 / (1.0 + hpc))}


ANCHOR_BASE = ["max_burial", "burial_concentration", "best_salt_bridge"]
ANCHOR_FULL = ANCHOR_BASE + ["hyd_anchor_depth", "arom_anchor_depth", "charged_anchor", "hbond_per_contact",
                             "pro_run", "buried_inert"]


def cvr(rows, cols, extra):
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(rows))
    y = np.array([r["y"] for r in rows])
    X = np.array([[r["feat"][c] for c in cols] + [r["a"][e] for e in extra] for r in rows], float)
    pred = np.full(len(rows), np.nan)
    for f in range(5):
        tr = fold != f
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=2.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
        pred[fold == f] = m.predict(X[fold == f])
    ok = ~np.isnan(pred)
    return pearsonr(pred[ok], y[ok])[0], float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2)))


def main():
    pdbb = [json.loads(ln) for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    rows = []
    for r in pdbb:
        a = features(r["pdb"], r["hb_count"] if "hb_count" in r else r["hb_count"])
        if a is None:
            continue
        rows.append({"y": r["y"], "length": r["length"], "feat": {c: r[c] for c in PROD}, "a": a})
    print(f"=== E128 anchor cross-band + deeper short (n={len(rows)}) ===\n")
    L = np.array([r["length"] for r in rows])

    print("PART 1 — base anchor features (max_burial, burial_conc, best_salt_bridge) by band:")
    print(f"{'band':<12}{'n':>5}{'16 r':>9}{'+anchor r':>11}{'Δ':>8}")
    for b in ["short≤8", "med9-12", "long13-16", "vlong≥17"]:
        sub = [r for r in rows if band(r["length"]) == b]
        if len(sub) < 20:
            continue
        r0 = cvr(sub, PROD, [])[0]
        r1 = cvr(sub, PROD, ANCHOR_BASE)[0]
        print(f"{b:<12}{len(sub):>5}{r0:>+9.3f}{r1:>+11.3f}{r1-r0:>+8.3f}")

    print("\nPART 2 — deeper short features, corr with ΔG (short only):")
    short = [r for r in rows if r["length"] <= 8]
    ys = np.array([r["y"] for r in short])
    for e in ANCHOR_FULL:
        print(f"     {e:<22} {cc([r['a'][e] for r in short], ys):+.3f}")

    print("\nPART 3 — short model, incremental feature sets (5-fold):")
    for nm, extra in [("16 base", []), ("+anchor base(3)", ANCHOR_BASE),
                      ("+hyd_anchor_depth", ANCHOR_BASE + ["hyd_anchor_depth"]),
                      ("+chem anchors(6)", ANCHOR_BASE + ["hyd_anchor_depth", "arom_anchor_depth", "charged_anchor"]),
                      ("+inert/proline", ANCHOR_BASE + ["hyd_anchor_depth", "hbond_per_contact", "pro_run", "buried_inert"]),
                      ("ALL anchor(9)", ANCHOR_FULL)]:
        r, rmse = cvr(short, PROD, extra)
        print(f"     {nm:<22} r={r:+.3f}  RMSE={rmse:.2f}")
    print("\n  reading: best short set = the feature combo that maximizes short r. Wire that; anchor base")
    print("  should help short most and med/long/vlong little (size features already work there).")


if __name__ == "__main__":
    main()
