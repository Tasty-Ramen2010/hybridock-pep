"""E105 — is our physics COMPLEMENTARY to PPI-Affinity? (orthogonal signal → ensemble beats the leader)

We can't out-predict PPI-Affinity on 156 (data-scale limited). But if our STRUCTURAL physics is orthogonal
to their SEQUENCE-descriptor SVM, an equal-weight ensemble can beat either alone — a genuine, honest,
deployable contribution ("use both; HybriDock adds structural signal the sequence model misses").

On the 91 shared crystal complexes:
  1. corr(our_pred, ppi_pred) — orthogonality (low = complementary)
  2. equal-weight z-ensemble z(ours)+z(ppi)  — NO fitted weights (honest, can't overfit)
  3. add each weak method to the ensemble; does ours add the most?
  4. LOCO-fitted optimal 2-way blend (realistic upper bound, weight learned out-of-sample)
Bar: PPI-Affinity alone r=0.629.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "data" / "biolip" / "ppiaffinity_si" / "SI"
STRUCT = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
          "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density",
          "cys_frac", "net_dewet", "polar_desolv"]
SHORT = ["bsa_hyd", "mj_contact", "strength_bur"]


def load_ours():
    rows = {}
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            rows[r["pdb"].lower()[:4]] = {"pdb4": r["pdb"].lower()[:4], "y": float(r["y"]),
                                          "length": int(float(r["length"])),
                                          "feat": {c: float(r[c]) for c in STRUCT}}
    return rows


def load_t100():
    f = SI / "SI-File-6-protein-peptide-test-set-1.csv"
    out = []
    for r in csv.DictReader(open(f)):
        m = re.match(r"([0-9a-zA-Z]{4})", r["PDB_NAME"])
        if not m:
            continue
        rec = {k.strip(): r[k] for k in r}
        rec["pdb4"] = m.group(1).lower()
        rec["y"] = float(r["Binding_affinity"])
        out.append(rec)
    return out


def ridge(rows, cols, lam=1.0):
    X = np.array([[r["feat"][c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    R = np.eye(A.shape[1]) * lam
    R[0, 0] = 0
    return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ y)


def rpred(feat, cols, p):
    mu, sd, w = p
    return float(np.r_[1.0, (np.array([feat[c] for c in cols]) - mu) / sd] @ w)


def our_loco(allrows, targets):
    out = {}
    for p4 in targets:
        tgt = allrows[p4]
        tr = [r for r in allrows.values() if r["pdb4"] != p4]
        if tgt["length"] <= 8 and sum(r["length"] <= 8 for r in tr) >= 6:
            cols, base = SHORT, [r for r in tr if r["length"] <= 8]
        else:
            cols, base = STRUCT, tr
        out[p4] = rpred(tgt["feat"], cols, ridge(base, cols))
    return out


def z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-9)


def R(a, y):
    return pearsonr(np.asarray(a, float), np.asarray(y, float))[0]


def main():
    ours = load_ours()
    t100 = load_t100()
    shared = [t for t in t100 if t["pdb4"] in ours]
    y = np.array([t["y"] for t in shared])
    op = np.array([v for v in (our_loco(ours, [t["pdb4"] for t in shared])).values()])
    op = np.array([our_loco(ours, [t["pdb4"] for t in shared])[t["pdb4"]] for t in shared])

    def colvec(name):
        key = next((k for k in shared[0] if k.replace(" ", "") == name.replace(" ", "")), None)
        try:
            return np.array([float(t[key]) for t in shared])
        except (ValueError, KeyError, TypeError):
            return None

    ppi = colvec("PPI-Affinity")
    print(f"=== E105 complementarity on {len(shared)} shared crystal complexes ===\n")
    # sign-align everything to truth (more-negative ΔG = stronger; align each predictor's sign)
    def align(v):
        return v if R(v, y) >= 0 else -v
    opa, ppia = align(op), align(ppi)

    print(f"1. ORTHOGONALITY:  r(ours, PPI-Affinity) = {pearsonr(opa, ppia)[0]:+.3f}  "
          f"(low ⇒ complementary signal)")
    print(f"   ours r={R(opa, y):+.3f} | PPI r={R(ppia, y):+.3f}\n")

    print("2. EQUAL-WEIGHT z-ENSEMBLE (no fitted weights — honest):")
    ens = z(opa) + z(ppia)
    print(f"   z(ours)+z(PPI)  r={R(ens, y):+.3f}   vs PPI alone {R(ppia, y):+.3f}   "
          f"{'<== ENSEMBLE WINS' if R(ens, y) > R(ppia, y) else '(no gain)'}")

    print("\n3. DOES OURS ADD THE MOST? ensemble PPI+each-method (equal-weight z):")
    for nm in ["HybriDock(ours)", "Kdeep", "DFIRE", "CP_PIE", "RF-Score", "PRODIGY"]:
        v = opa if nm.startswith("HybriDock") else colvec(nm)
        if v is None:
            continue
        va = align(v)
        e = z(ppia) + z(va)
        print(f"   PPI + {nm:<16} r={R(e, y):+.3f}  (Δ over PPI {R(e, y)-R(ppia, y):+.3f}; "
              f"orthog r={pearsonr(va, ppia)[0]:+.2f})")

    print("\n4. LOCO-FITTED 2-way blend ours+PPI (weight learned out-of-sample, realistic upper bound):")
    n = len(shared)
    blend = np.full(n, np.nan)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        A = np.column_stack([z(opa)[tr], z(ppia)[tr]])
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), A]), y[tr], rcond=None)
        blend[i] = np.r_[1.0, z(opa)[i], z(ppia)[i]] @ w
    print(f"   LOCO blend r={R(blend, y):+.3f}  vs PPI {R(ppia, y):+.3f}  "
          f"{'<== WINS' if R(blend, y) > R(ppia, y) else ''}")
    print("\n  reading: if ensemble > PPI alone, HybriDock contributes orthogonal STRUCTURAL signal the")
    print("  sequence-descriptor model misses → honest 'use both' value, even though we don't out-predict solo.")


if __name__ == "__main__":
    main()
