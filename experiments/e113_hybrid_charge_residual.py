"""E113 — physics base + ML charge-residual hybrid (Ram's design: ML models charges where physics fails).

Architecture (physics-first, NOT a ProtDCal black-box clone):
    final ΔG = physics_ridge(16 structural features)        [interpretable base]
             + charge_residual_GBT(charge-specific features) [ML only for the floor]
The residual model sees ONLY charge features (net/abs charge, K/R/D/E counts, charge/len, pocket charge,
salt-bridge SASA, peptide×pocket charge complementarity) — so it corrects the charged regime without
turning the whole model into a black box. Residuals trained OUT-OF-FOLD (no leakage into the base fit).

Evaluated on the held-out test (e112 split) and the PPI-Affinity shared subset, SPLIT BY CHARGE:
does the hybrid beat physics-alone on high-charge, and approach/beat PPI-Affinity (0.55/0.63)?
Reference: a full-feature GBT (the "mere copy") — to show the hybrid competes without being a black box.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "data" / "biolip" / "ppiaffinity_si" / "SI"
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
POS, NEG = set("KR"), set("DE")


def charge_feats(r):
    s = r["seq"]
    L = max(1, len(s))
    nk, nr, nd, ne = s.count("K"), s.count("R"), s.count("D"), s.count("E")
    net = (nk + nr) - (nd + ne)
    absf = (nk + nr + nd + ne) / L
    pocn = r["feat"]["poc_net"]
    return [net, net / L, absf, nk / L, nr / L, nd / L, ne / L, pocn, r["feat"]["poc_eis"],
            r["feat"]["sasa_sb"], net * pocn, abs(net)]


def load():
    rows = {}
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            rows[r["pdb"]] = {"id": r["pdb"], "pdb4": r["pdb"].lower()[:4], "seq": r.get("seq", ""),
                              "y": float(r["y"]), "length": int(float(r["length"])),
                              "feat": {c: float(r[c]) for c in PROD}}
    oseq = {r["seq"] for r in rows.values() if r["seq"]}
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["seq"] in oseq or r["pdb"] in rows:
            continue
        oseq.add(r["seq"])
        rows[r["pdb"]] = {"id": r["pdb"], "pdb4": r["pdb"].lower()[:4], "seq": r["seq"], "y": r["y"],
                          "length": r["length"], "feat": {c: r[c] for c in PROD}}
    return rows


def ridge_fit(rows, cols, lam=1.0):
    X = np.array([[r["feat"][c] for c in cols] for r in rows], float); y = np.array([r["y"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]) * lam; R[0, 0] = 0
    return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ y)


def ridge_pred(r, cols, p):
    mu, sd, w = p
    return float(np.r_[1.0, (np.array([r["feat"][c] for c in cols]) - mu) / sd] @ w)


def Rc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return pearsonr(a[m], b[m])[0] if m.sum() > 4 else np.nan


def main():
    allrows = load()
    split = json.load(open(ROOT / "data/pep_split.json"))
    train = [allrows[i] for i in split["train"] if i in allrows]
    test = [allrows[i] for i in split["test"] if i in allrows]
    sh = {}
    for r in csv.DictReader(open(SI / "SI-File-6-protein-peptide-test-set-1.csv")):
        m = re.match(r"([0-9a-zA-Z]{4})", r["PDB_NAME"])
        if m:
            sh[m.group(1).lower()] = float(r["PPI-Affinity"])
    yte = np.array([r["y"] for r in test])
    print(f"=== E113 physics+charge-residual hybrid (train {len(train)}, test {len(test)}) ===\n")

    # base physics, fit on train
    base = ridge_fit(train, PROD)
    phys_test = np.array([ridge_pred(r, PROD, base) for r in test])

    # out-of-fold physics residuals on train → train charge-residual GBT
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(train))
    resid = np.zeros(len(train))
    for k in range(5):
        tr = [train[j] for j in range(len(train)) if fold[j] != k]
        bk = ridge_fit(tr, PROD)
        for j in range(len(train)):
            if fold[j] == k:
                resid[j] = train[j]["y"] - ridge_pred(train[j], PROD, bk)
    Xc_tr = np.array([charge_feats(r) for r in train])
    cg = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                       l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(Xc_tr, resid)
    charge_corr = cg.predict(np.array([charge_feats(r) for r in test]))
    hybrid = phys_test + charge_corr

    # reference: full-feature GBT (struct+charge) = the "mere copy"
    Xf_tr = np.array([[r["feat"][c] for c in PROD] + charge_feats(r) for r in train])
    full = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                         l2_regularization=2.0, min_samples_leaf=25, random_state=0).fit(Xf_tr, [r["y"] for r in train])
    full_test = full.predict(np.array([[r["feat"][c] for c in PROD] + charge_feats(r) for r in test]))

    ac = np.array([(r["seq"].count("K") + r["seq"].count("R") + r["seq"].count("D") + r["seq"].count("E")) / max(1, len(r["seq"])) for r in test])
    in91 = np.array([r["pdb4"] in sh for r in test])
    ppi = np.array([sh.get(r["pdb4"], np.nan) for r in test])

    def line(nm, pred):
        hi = ac > 0.30
        print(f"   {nm:<26} all={Rc(pred,yte):+.3f}  low-ch={Rc(pred[~hi],yte[~hi]):+.3f}  "
              f"high-ch={Rc(pred[hi],yte[hi]):+.3f}  shared={Rc(pred[in91],yte[in91]):+.3f}")

    print(f"   {'model':<26} {'all':>9} {'low-ch':>9} {'high-ch':>9} {'shared91':>9}")
    line("physics base (ridge16)", phys_test)
    line("HYBRID phys+charge-resid", hybrid)
    line("full GBT (struct+charge)", full_test)
    print(f"   {'PPI-Affinity (shared)':<26} {'—':>9} {'—':>9} {'—':>9} {Rc(ppi[in91],yte[in91]):>+9.3f}")
    # high-charge on shared
    hi_sh = in91 & (ac > 0.30)
    print(f"\n   high-charge ON shared-91 (n={hi_sh.sum()}): physics={Rc(phys_test[hi_sh],yte[hi_sh]):+.3f}  "
          f"hybrid={Rc(hybrid[hi_sh],yte[hi_sh]):+.3f}  PPI={Rc(ppi[hi_sh],yte[hi_sh]):+.3f}")
    print(f"   RMSE test: physics={np.sqrt(np.mean((phys_test-yte)**2)):.2f}  hybrid={np.sqrt(np.mean((hybrid-yte)**2)):.2f}  "
          f"full={np.sqrt(np.mean((full_test-yte)**2)):.2f}")
    print("\n  reading: hybrid > physics on high-charge ⇒ Ram's 'ML models charges where we fail' WORKS.")
    print("  hybrid ≈ full GBT ⇒ we get the accuracy WITHOUT a black box (physics stays the interpretable base).")


if __name__ == "__main__":
    main()
