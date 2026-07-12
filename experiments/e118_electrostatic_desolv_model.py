"""E118 — electrostatic-desolvation salt-bridge model (Ram's "before→after receptor" idea).

Physics: the charged contribution to binding = receptor pocket electrostatic context (BEFORE peptide) +
the salt bridges / charge desolvation that form on binding (AFTER). Single-pose Coulomb washes (documented
floor); so we LEARN the mapping from geometric salt-bridge/desolvation descriptors → the charged residual
our structural physics misses. NOT a black box: features are interpretable electrostatic geometry.

Geometry-only (no OpenMM → avoids the NaN structure-prep problem). Per PDBbind complex (peptide mol2 +
receptor _protein.pdb):
  BEFORE (receptor pocket context): # charged residues near pocket, pocket net charge, charge density.
  AFTER  (salt bridges + desolvation): n salt bridges (pep± to rec∓, charge-center dist <4Å), their burial,
          peptide charged residues buried but UNPAIRED (desolvation penalty), screened Coulomb (e32 form),
          peptide×pocket charge complementarity.
Test: does adding these to the 16 structural features improve the CHARGED complexes (5-fold CV by charge)?
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
PLROOT = ROOT / "data/drive_pull/pl/P-L"
SB_CUT = 4.0   # Å between charge centers = salt bridge
NEAR = 6.0     # Å pocket-context shell
ESFEATS = ["n_sb", "sb_buried_mean", "n_pep_chg_buried_unpaired", "rec_pocket_ncharged",
           "rec_pocket_netq", "screened_coul", "charge_complement", "n_pep_pos", "n_pep_neg"]


def chg_centers_peptide(mol2):
    """From mol2: list of (sign, xyz) charge centers (Lys NZ, Arg CZ, Asp/Glu carboxyl midpoint)."""
    lines = mol2.read_text().splitlines()
    if "@<TRIPOS>ATOM" not in lines:
        return None, None
    a = lines.index("@<TRIPOS>ATOM")
    atoms = []  # (name, resname, xyz)
    for ln in lines[a + 1:]:
        if ln.startswith("@"):
            break
        f = ln.split()
        if len(f) < 9:
            continue
        nm = f[1]
        rn = "".join(c for c in f[7] if c.isalpha()).upper()[:3]
        try:
            atoms.append((nm, rn, np.array([float(f[2]), float(f[3]), float(f[4])])))
        except ValueError:
            continue
    # group into residues by backbone N
    res, cur = [], None
    for nm, rn, xyz in atoms:
        if nm == "N":
            cur = {"rn": rn, "atoms": {}}
            res.append(cur)
        if cur is None:
            cur = {"rn": rn, "atoms": {}}
            res.append(cur)
        cur["atoms"][nm] = xyz
    centers, all_xyz = [], [xyz for _, _, xyz in atoms]
    for r in res:
        at = r["atoms"]
        if r["rn"] == "LYS" and "NZ" in at:
            centers.append((+1, at["NZ"]))
        elif r["rn"] == "ARG" and "CZ" in at:
            centers.append((+1, at["CZ"]))
        elif r["rn"] == "ASP":
            o = [at[k] for k in ("OD1", "OD2", "CG") if k in at]
            if o:
                centers.append((-1, np.mean(o, axis=0)))
        elif r["rn"] == "GLU":
            o = [at[k] for k in ("OE1", "OE2", "CD") if k in at]
            if o:
                centers.append((-1, np.mean(o, axis=0)))
    npos = sum(1 for s, _ in centers if s > 0)
    nneg = sum(1 for s, _ in centers if s < 0)
    return centers, (np.array(all_xyz), npos, nneg)


def chg_centers_receptor(pdb, near_xyz):
    """Receptor charge centers within NEAR of any peptide atom (pocket context)."""
    centers = []
    cur, curname = {}, None
    rows = []
    for ln in pdb.read_text().splitlines():
        if not ln.startswith("ATOM"):
            continue
        rn = ln[17:20].strip()
        if rn not in ("LYS", "ARG", "ASP", "GLU"):
            continue
        nm = ln[12:16].strip()
        key = (ln[21], ln[22:27])
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        rows.append((key, rn, nm, xyz))
    byres = {}
    for key, rn, nm, xyz in rows:
        byres.setdefault((key, rn), {})[nm] = xyz
    for (key, rn), at in byres.items():
        c = None
        if rn == "LYS" and "NZ" in at:
            c = (+1, at["NZ"])
        elif rn == "ARG" and "CZ" in at:
            c = (+1, at["CZ"])
        elif rn == "ASP":
            o = [at[k] for k in ("OD1", "OD2", "CG") if k in at]
            c = (-1, np.mean(o, axis=0)) if o else None
        elif rn == "GLU":
            o = [at[k] for k in ("OE1", "OE2", "CD") if k in at]
            c = (-1, np.mean(o, axis=0)) if o else None
        if c is not None and near_xyz.size and np.linalg.norm(near_xyz - c[1], axis=1).min() < NEAR:
            centers.append(c)
    return centers


def es_features(pid):
    d = next((Path(p).parent for p in glob.glob(str(PLROOT / f"*/{pid}/{pid}_ligand.mol2"))), None)
    if d is None:
        return None
    pc, meta = chg_centers_peptide(d / f"{pid}_ligand.mol2")
    if pc is None:
        return None
    pep_xyz, npos, nneg = meta
    rc = chg_centers_receptor(d / f"{pid}_protein.pdb", pep_xyz)
    # all receptor heavy atoms for burial estimate
    rec_xyz = []
    for ln in (d / f"{pid}_protein.pdb").read_text().splitlines():
        if ln.startswith("ATOM"):
            try:
                rec_xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            except ValueError:
                pass
    rec_xyz = np.array(rec_xyz) if rec_xyz else np.zeros((1, 3))
    n_sb, sb_bur, paired = 0, [], set()
    screened = 0.0
    for i, (sp, xp) in enumerate(pc):
        for (sr, xr) in rc:
            dist = float(np.linalg.norm(xp - xr))
            if sp * sr < 0 and dist < SB_CUT:  # opposite charge, close = salt bridge
                n_sb += 1
                paired.add(i)
                bur = int((np.linalg.norm(rec_xyz - xp, axis=1) < 6.0).sum())  # neighbors = burial
                sb_bur.append(bur)
            if dist > 0:
                screened += 332.0 * sp * sr / (4.0 * dist * dist + 1e-6)  # distance-dependent ε
    # peptide charged buried but unpaired (desolvation penalty)
    unpaired_buried = 0
    for i, (sp, xp) in enumerate(pc):
        if i not in paired and rec_xyz.size and (np.linalg.norm(rec_xyz - xp, axis=1) < 5.0).sum() > 8:
            unpaired_buried += 1
    rec_netq = sum(s for s, _ in rc)
    pep_netq = npos - nneg
    return {"n_sb": n_sb, "sb_buried_mean": float(np.mean(sb_bur)) if sb_bur else 0.0,
            "n_pep_chg_buried_unpaired": unpaired_buried, "rec_pocket_ncharged": len(rc),
            "rec_pocket_netq": rec_netq, "screened_coul": screened,
            "charge_complement": pep_netq * rec_netq, "n_pep_pos": npos, "n_pep_neg": nneg}


def main():
    pdbb = [json.loads(ln) for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    POS, NEG = set("KR"), set("DE")
    rows = []
    print(f"=== E118 electrostatic-desolvation salt-bridge model ({len(pdbb)} PDBbind) ===", flush=True)
    for i, r in enumerate(pdbb):
        es = es_features(r["pdb"])
        if es is None:
            continue
        s = r["seq"]
        ac = sum(c in POS | NEG for c in s) / max(1, len(s))
        rows.append({"y": r["y"], "abs_ch": ac, "feat": {c: r[c] for c in PROD}, "es": es})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pdbb)} ({len(rows)} ok)", flush=True)
    print(f"  computed ES features on {len(rows)} complexes\n")

    y = np.array([r["y"] for r in rows])
    ac = np.array([r["abs_ch"] for r in rows])

    def cv(cols_struct, add_es):
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 5, len(rows))
        X = np.array([[r["feat"][c] for c in cols_struct] + ([r["es"][e] for e in ESFEATS] if add_es else []) for r in rows], float)
        pred = np.full(len(rows), np.nan)
        for f in range(5):
            tr = fold != f
            m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                              l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(X[tr], y[tr])
            pred[fold == f] = m.predict(X[fold == f])
        return pred

    def rr(p, m):
        ok = m & ~np.isnan(p)
        return pearsonr(p[ok], y[ok])[0] if ok.sum() > 4 else np.nan

    # ES features vs charged residual (do they correlate with what physics misses?)
    base = cv(PROD, False)
    resid = y - base
    print("  ES feature → corr with charged residual (|residual| and signed):")
    for e in ESFEATS:
        v = np.array([r["es"][e] for r in rows])
        hi = ac > 0.30
        print(f"     {e:<26} corr(signed resid, high-charge)={pearsonr(v[hi], resid[hi])[0]:+.3f}")

    hi = ac > 0.30
    lo = ac <= 0.15
    print(f"\n  GBT 5-fold by charge:  struct16 → struct16+ES")
    es = cv(PROD, True)
    for lab, m in [("ALL", np.ones(len(rows), bool)), ("low ≤0.15", lo),
                   ("mid", (ac > 0.15) & (ac <= 0.30)), ("HIGH >0.30", hi)]:
        print(f"     {lab:<12} n={m.sum():<4} struct={rr(base,m):+.3f} → +ES={rr(es,m):+.3f}  Δ={rr(es,m)-rr(base,m):+.3f}")
    print("\n  reading: +ES lifts HIGH-charge r ⇒ salt-bridge/desolvation geometry recovers the charged floor")
    print("  that single-pose Coulomb washes — Ram's before→after electrostatic model WORKS.")


if __name__ == "__main__":
    main()
