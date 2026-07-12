"""E112 — even train/test split + feature sign-stability + WHERE physics fails (charge error map).

Ram's plan (physics-first, ML-for-the-floor — NOT a ProtDCal clone):
  1. Build a stratified train/test split of the pooled PDBs (ours 156 + PDBbind 925), balanced on
     charge × length × source × ΔG. Force the 91 PPI-Affinity shared complexes into TEST (head-to-head).
  2. Validate our 16 physics features still correlate on BOTH splits (sign-stability gate — the core
     discipline; a feature that flips train→test is unreliable at scale).
  3. Error map: fit physics on train, predict test, categorize WHERE we fail (charge / length / source).
     Confirms the residual to hand to ML is the CHARGED floor (so the hybrid targets the right thing).

Persists the split to data/pep_split.json for the hybrid model (e113).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "data" / "biolip" / "ppiaffinity_si" / "SI"
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
POS, NEG = set("KR"), set("DE")


def abscharge(s):
    return sum(c in POS | NEG for c in s) / max(1, len(s))


def netcharge(s):
    return sum(c in POS for c in s) - sum(c in NEG for c in s)


def load():
    rows = []
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            rows.append({"id": r["pdb"], "pdb4": r["pdb"].lower()[:4], "seq": r.get("seq", ""),
                         "y": float(r["y"]), "length": int(float(r["length"])), "src": "ours",
                         "feat": {c: float(r[c]) for c in PROD}})
    oseq = {r["seq"] for r in rows if r["seq"]}
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["seq"] in oseq:
            continue
        oseq.add(r["seq"])
        rows.append({"id": r["pdb"], "pdb4": r["pdb"].lower()[:4], "seq": r["seq"], "y": r["y"],
                     "length": r["length"], "src": "pdbbind", "feat": {c: r[c] for c in PROD}})
    return rows


def shared91():
    sh = {}
    for r in csv.DictReader(open(SI / "SI-File-6-protein-peptide-test-set-1.csv")):
        import re
        m = re.match(r"([0-9a-zA-Z]{4})", r["PDB_NAME"])
        if m:
            sh[m.group(1).lower()] = float(r["PPI-Affinity"])
    return sh


def stratum(r, qy):
    L = r["length"]
    lb = 0 if L <= 8 else 1 if L <= 12 else 2 if L <= 16 else 3
    cb = 0 if abscharge(r["seq"]) <= 0.15 else 1 if absharge_ok(r) else 2
    yb = int(np.digitize(r["y"], qy))
    return (r["src"], lb, cb, yb)


def absharge_ok(r):
    return abscharge(r["seq"]) <= 0.30


def main():
    rows = load()
    sh = shared91()
    y = np.array([r["y"] for r in rows])
    qy = np.quantile(y, [1 / 3, 2 / 3])
    print(f"=== E112 split + error map ({len(rows)} pooled: "
          f"{sum(r['src']=='ours' for r in rows)} ours + {sum(r['src']=='pdbbind' for r in rows)} PDBbind) ===\n")

    # ---- 1. stratified split, force shared-91 into TEST ----
    rng = np.random.default_rng(7)
    for r in rows:
        r["abs_ch"] = abscharge(r["seq"])
        r["net_ch"] = netcharge(r["seq"])
        cb = 0 if r["abs_ch"] <= 0.15 else (1 if r["abs_ch"] <= 0.30 else 2)
        yb = int(np.digitize(r["y"], qy))
        r["stratum"] = (r["src"], min(r["length"] // 5, 4), cb, yb)
        r["in91"] = r["pdb4"] in sh
    test, train = [], []
    by = {}
    for r in rows:
        by.setdefault(r["stratum"], []).append(r)
    for st, grp in by.items():
        rng.shuffle(grp)
        for r in grp:
            if r["in91"]:
                test.append(r)  # force all shared-91 into test
        rest = [r for r in grp if not r["in91"]]
        ntest = max(0, round(0.25 * len(grp)) - sum(1 for r in grp if r["in91"]))
        test += rest[:ntest]
        train += rest[ntest:]
    json.dump({"train": [r["id"] for r in train], "test": [r["id"] for r in test]},
              open(ROOT / "data/pep_split.json", "w"))
    print(f"1. SPLIT: train={len(train)} test={len(test)} (shared-91 in test={sum(r['in91'] for r in test)})")
    for lab, S in [("train", train), ("test", test)]:
        a = np.array([r["abs_ch"] for r in S])
        print(f"   {lab}: ΔG {np.mean([r['y'] for r in S]):.2f}±{np.std([r['y'] for r in S]):.2f}  "
              f"abs-charge {a.mean():.2f}  len {np.mean([r['length'] for r in S]):.1f}  "
              f"pdbbind-frac {np.mean([r['src']=='pdbbind' for r in S]):.2f}")

    # ---- 2. feature sign-stability on BOTH splits ----
    print("\n2. FEATURE SIGN-STABILITY  corr(feature, ΔG): train | test | STABLE?")
    yt = np.array([r["y"] for r in train]); yte = np.array([r["y"] for r in test])
    nstable = 0
    for c in PROD:
        rt = pearsonr([r["feat"][c] for r in train], yt)[0]
        rte = pearsonr([r["feat"][c] for r in test], yte)[0]
        stable = (np.sign(rt) == np.sign(rte)) and abs(rt) > 0.05 and abs(rte) > 0.05
        nstable += stable
        print(f"   {c:<14} {rt:+.3f} | {rte:+.3f}   {'OK' if stable else 'flip/weak' if np.sign(rt)!=np.sign(rte) else 'weak'}")
    print(f"   → {nstable}/16 features sign-stable & non-trivial across the split")

    # ---- 3. error map: physics fit on train, predict test, fail-by-category ----
    def ridge(rows_, cols, lam=1.0):
        X = np.array([[r["feat"][c] for c in cols] for r in rows_], float); yy = np.array([r["y"] for r in rows_])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]) * lam; R[0, 0] = 0
        return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ yy)
    p = ridge(train, PROD)

    def pred(r):
        mu, sd, w = p
        return float(np.r_[1.0, (np.array([r["feat"][c] for c in PROD]) - mu) / sd] @ w)
    pr = np.array([pred(r) for r in test])
    err = yte - pr
    print(f"\n3. PHYSICS error map (train→test, r={pearsonr(pr,yte)[0]:+.3f} RMSE={np.sqrt(np.mean(err**2)):.2f}):")
    ac = np.array([r["abs_ch"] for r in test])
    for lab, m in [("low-charge ≤0.15", ac <= 0.15), ("mid 0.15–0.30", (ac > 0.15) & (ac <= 0.30)),
                   ("high-charge >0.30", ac > 0.30)]:
        if m.sum() >= 5:
            print(f"   {lab:<18} n={m.sum():<4} r={pearsonr(pr[m],yte[m])[0]:+.3f}  RMSE={np.sqrt(np.mean(err[m]**2)):.2f}  "
                  f"mean|err|={np.mean(np.abs(err[m])):.2f}")
    print(f"\n   corr(|err|, abs_charge)={pearsonr(np.abs(err), ac)[0]:+.3f}  "
          f"corr(|err|, |net_charge|)={pearsonr(np.abs(err), [abs(r['net_ch']) for r in test])[0]:+.3f}")
    print("   → if |err| grows with charge, the residual to hand ML is the CHARGED FLOOR (build e113 hybrid).")
    # persist test residuals for the hybrid
    json.dump([{"id": r["id"], "resid": float(yte[i] - pr[i]), "abs_ch": r["abs_ch"]}
               for i, r in enumerate(test)], open(ROOT / "data/pep_test_residuals.json", "w"))


if __name__ == "__main__":
    main()
