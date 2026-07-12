"""E124 — burial-weighted BORN desolvation (the pocket-dielectric physics e118's counts lacked).

e118 used salt-bridge COUNTS/geometry → didn't crack the charged floor (+0.017). The missing ingredient:
the desolvation ENERGY under a burial-dependent dielectric. A charge buried in the low-ε pocket pays a
large Born desolvation penalty unless a salt bridge repays it (favorable Coulomb at low ε); an exposed
charge (ε~80) is screened/weak. This is the pocket-dielectric effect uniform-ε GB washes out.

Per charged group g (peptide Lys/Arg+, Asp/Glu−):
  burial_g   = min(1, n_receptor_heavy_within_8Å / NORM)             (0 exposed → 1 buried)
  ε_eff_g    = ε_w − (ε_w − ε_pocket)·burial_g                       (80 exposed → ~10 buried)
  desolv_g   = +166·q²/r · (1/ε_eff_g − 1/ε_w)                        (Born penalty, ≥0 unfavorable)
  coulomb_g  = Σ_recpartner 332·q_g·q_r / (ε_eff_g · d)              (favorable if salt-bridged at low ε)
Net charged term = Σ_g (coulomb_g + desolv_g).  Test vs charged residual + does GBT high-charge improve?
If it gives a computable signal → distill a surrogate (like entropy). If flat → desolvation is FEP-only.
"""
from __future__ import annotations

import glob
import json
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
EPS_W, EPS_POCK = 80.0, 10.0
BURNORM = 40.0  # receptor heavy atoms within 8Å for full burial
R_BORN = 3.0    # effective Born radius of a charged group (Å)


def peptide_charges(mol2):
    lines = mol2.read_text().splitlines()
    if "@<TRIPOS>ATOM" not in lines:
        return None
    a = lines.index("@<TRIPOS>ATOM")
    atoms = []
    for ln in lines[a + 1:]:
        if ln.startswith("@"):
            break
        f = ln.split()
        if len(f) < 9:
            continue
        try:
            atoms.append((f[1], "".join(c for c in f[7] if c.isalpha()).upper()[:3],
                          np.array([float(f[2]), float(f[3]), float(f[4])])))
        except ValueError:
            continue
    res, cur = [], None
    for nm, rn, xyz in atoms:
        if nm == "N":
            cur = {"rn": rn, "at": {}}
            res.append(cur)
        if cur is None:
            cur = {"rn": rn, "at": {}}
            res.append(cur)
        cur["at"][nm] = xyz
    ch = []
    for r in res:
        at = r["at"]
        if r["rn"] == "LYS" and "NZ" in at:
            ch.append((+1, at["NZ"]))
        elif r["rn"] == "ARG" and "CZ" in at:
            ch.append((+1, at["CZ"]))
        elif r["rn"] == "ASP":
            o = [at[k] for k in ("OD1", "OD2", "CG") if k in at]
            if o:
                ch.append((-1, np.mean(o, 0)))
        elif r["rn"] == "GLU":
            o = [at[k] for k in ("OE1", "OE2", "CD") if k in at]
            if o:
                ch.append((-1, np.mean(o, 0)))
    return ch


def receptor_charges_and_xyz(pdb):
    rows, heavy = {}, []
    for ln in pdb.read_text().splitlines():
        if not ln.startswith("ATOM"):
            continue
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        heavy.append(xyz)
        rn = ln[17:20].strip()
        if rn in ("LYS", "ARG", "ASP", "GLU"):
            rows.setdefault((ln[21], ln[22:27], rn), {})[ln[12:16].strip()] = xyz
    ch = []
    for (c, n, rn), at in rows.items():
        if rn == "LYS" and "NZ" in at:
            ch.append((+1, at["NZ"]))
        elif rn == "ARG" and "CZ" in at:
            ch.append((+1, at["CZ"]))
        elif rn == "ASP":
            o = [at[k] for k in ("OD1", "OD2", "CG") if k in at]
            if o:
                ch.append((-1, np.mean(o, 0)))
        elif rn == "GLU":
            o = [at[k] for k in ("OE1", "OE2", "CD") if k in at]
            if o:
                ch.append((-1, np.mean(o, 0)))
    return ch, (np.array(heavy) if heavy else np.zeros((1, 3)))


def born_features(pid):
    d = next((Path(p).parent for p in glob.glob(str(PLROOT / f"*/{pid}/{pid}_ligand.mol2"))), None)
    if d is None:
        return None
    pc = peptide_charges(d / f"{pid}_ligand.mol2")
    if pc is None:
        return None
    rc, rec_xyz = receptor_charges_and_xyz(d / f"{pid}_protein.pdb")
    desolv_tot = coul_tot = 0.0
    n_buried_unpaired = 0
    sb_low_eps = 0  # salt bridges in low-dielectric (buried, strong)
    for sp, xp in pc:
        nb = (np.linalg.norm(rec_xyz - xp, axis=1) < 8.0).sum()
        burial = min(1.0, nb / BURNORM)
        eps_eff = EPS_W - (EPS_W - EPS_POCK) * burial
        desolv = 166.0 * 1.0 / R_BORN * (1.0 / eps_eff - 1.0 / EPS_W)  # ≥0 penalty
        desolv_tot += desolv * burial  # only buried charges pay (exposed ~ fully solvated)
        paired = False
        for sr, xr in rc:
            dd = float(np.linalg.norm(xp - xr))
            if dd < 6.0 and sp * sr < 0:
                coul_tot += 332.0 * sp * sr / (eps_eff * dd + 1e-6)  # favorable, low-ε if buried
                paired = True
                if burial > 0.5:
                    sb_low_eps += 1
        if not paired and burial > 0.5:
            n_buried_unpaired += 1
    return {"born_desolv": desolv_tot, "born_coul": coul_tot, "born_net": coul_tot + desolv_tot,
            "sb_low_eps": sb_low_eps, "buried_unpaired_chg": n_buried_unpaired}


BORNF = ["born_desolv", "born_coul", "born_net", "sb_low_eps", "buried_unpaired_chg"]


def main():
    pdbb = [json.loads(ln) for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    POS, NEG = set("KR"), set("DE")
    rows = []
    print(f"=== E124 burial-weighted Born desolvation ({len(pdbb)} PDBbind) ===", flush=True)
    for i, r in enumerate(pdbb):
        b = born_features(r["pdb"])
        if b is None:
            continue
        s = r["seq"]
        ac = sum(c in POS | NEG for c in s) / max(1, len(s))
        rows.append({"y": r["y"], "abs_ch": ac, "feat": {c: r[c] for c in PROD}, "born": b})
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(pdbb)}", flush=True)
    y = np.array([r["y"] for r in rows])
    ac = np.array([r["abs_ch"] for r in rows])
    hi = ac > 0.30
    print(f"  computed on {len(rows)} complexes; high-charge n={hi.sum()}\n")

    def cv(add):
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 5, len(rows))
        X = np.array([[r["feat"][c] for c in PROD] + ([r["born"][b] for b in BORNF] if add else []) for r in rows], float)
        pred = np.full(len(rows), np.nan)
        for f in range(5):
            tr = fold != f
            m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                              l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(X[tr], y[tr])
            pred[fold == f] = m.predict(X[fold == f])
        return pred

    base = cv(False)
    resid = y - base
    print("  Born feature → corr with charged residual (high-charge subset):")
    for b in BORNF:
        v = np.array([r["born"][b] for r in rows])
        print(f"     {b:<20} corr(resid,high)={pearsonr(v[hi], resid[hi])[0]:+.3f}  corr(feat,y all)={pearsonr(v, y)[0]:+.3f}")

    born = cv(True)

    def rr(p, m):
        ok = m & ~np.isnan(p)
        return pearsonr(p[ok], y[ok])[0] if ok.sum() > 4 else np.nan
    print("\n  GBT 5-fold by charge:  struct16 → struct16+Born")
    for lab, m in [("ALL", np.ones(len(rows), bool)), ("low ≤0.15", ac <= 0.15),
                   ("mid", (ac > 0.15) & (ac <= 0.30)), ("HIGH >0.30", hi)]:
        print(f"     {lab:<12} n={m.sum():<4} struct={rr(base,m):+.3f} → +Born={rr(born,m):+.3f}  Δ={rr(born,m)-rr(base,m):+.3f}")
    print("\n  reading: +Born lifts HIGH-charge ⇒ pocket-dielectric desolvation IS a computable signal →")
    print("  worth distilling a surrogate (Ram's ML desolvation model). If flat ⇒ desolvation is FEP-only,")
    print("  unlike entropy (MD-computable). That asymmetry is the honest answer on the two missing terms.")


if __name__ == "__main__":
    main()
