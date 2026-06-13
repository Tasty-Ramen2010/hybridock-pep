"""E107 — WHERE does PPI-Affinity's edge over us come from? (mechanistic; tells us what data would buy)

On the 91 shared crystal complexes PPI-Affinity leads 0.629 vs our 0.449. Is that edge uniform, or
concentrated in a regime our physics is KNOWN to miss (the charged/polar floor — single-pose electrostatics
wash, documented)? If PPI's advantage lives where we have the floor, then PDBbind-scale data specifically
buys us THAT regime (the model would learn the floor statistically, as PPI did).

Per shared complex: |err_ours| vs |err_ppi| (both sign-aligned, z-scored to truth). Then partition the
ADVANTAGE (|err_ours|−|err_ppi|, >0 = PPI better here) by charge, length, hydrophobicity.
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
POS, NEG = set("KR"), set("DE")


def load():
    rows = {}
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            seq = r.get("seq", "")
            rows[r["pdb"].lower()[:4]] = {"pdb4": r["pdb"].lower()[:4], "y": float(r["y"]),
                                          "length": int(float(r["length"])), "seq": seq,
                                          "net_charge": float(r["net_charge"]),
                                          "abs_charge": (sum(c in POS | NEG for c in seq) / max(1, len(seq))),
                                          "feat": {c: float(r[c]) for c in STRUCT}}
    return rows


def ridge(rows, cols, lam=1.0):
    X = np.array([[r["feat"][c] for c in cols] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]) * lam; R[0, 0] = 0
    return mu, sd, np.linalg.solve(A.T @ A + R, A.T @ y)


def rp(feat, cols, p):
    mu, sd, w = p
    return float(np.r_[1.0, (np.array([feat[c] for c in cols]) - mu) / sd] @ w)


def main():
    ours = load()
    sh = []
    for r in csv.DictReader(open(SI / "SI-File-6-protein-peptide-test-set-1.csv")):
        m = re.match(r"([0-9a-zA-Z]{4})", r["PDB_NAME"])
        if m and m.group(1).lower() in ours:
            sh.append((m.group(1).lower(), float(r["Binding_affinity"]), float(r["PPI-Affinity"])))
    y = np.array([s[1] for s in sh])
    ppi = np.array([s[2] for s in sh])
    op = []
    for p4, _, _ in sh:
        tr = [r for r in ours.values() if r["pdb4"] != p4]
        t = ours[p4]
        cols, base = (SHORT, [r for r in tr if r["length"] <= 8]) if (t["length"] <= 8 and sum(r["length"] <= 8 for r in tr) >= 6) else (STRUCT, tr)
        op.append(rp(t["feat"], cols, ridge(base, cols)))
    op = np.array(op)
    op = op if pearsonr(op, y)[0] >= 0 else -op
    ppi = ppi if pearsonr(ppi, y)[0] >= 0 else -ppi

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)
    # put both predictors and truth on z-scale so |err| is comparable
    zy = z(y)
    e_ours = np.abs(z(op) - zy)
    e_ppi = np.abs(z(ppi) - zy)
    adv = e_ours - e_ppi  # >0 => PPI better on this complex

    print(f"=== E107 where PPI-Affinity beats us ({len(sh)} shared) ===")
    print(f"  overall: ours r={pearsonr(op,y)[0]:+.3f}  PPI r={pearsonr(ppi,y)[0]:+.3f}  "
          f"mean adv(PPI−ours)={adv.mean():+.3f}  PPI-better on {(adv>0).mean():.0%} of complexes\n")

    feats = {
        "abs_charge_frac": np.array([ours[s[0]]["abs_charge"] for s in sh]),
        "|net_charge|": np.array([abs(ours[s[0]]["net_charge"]) for s in sh]),
        "length": np.array([ours[s[0]]["length"] for s in sh]),
        "frac_hyd(poc)": np.array([ours[s[0]]["feat"]["poc_f_hyd"] for s in sh]),
        "polar_desolv": np.array([ours[s[0]]["feat"]["polar_desolv"] for s in sh]),
        "bsa_hyd": np.array([ours[s[0]]["feat"]["bsa_hyd"] for s in sh]),
    }
    print("  corr(feature, PPI-advantage)  — POSITIVE = PPI's edge grows with this feature:")
    for nm, v in sorted(feats.items(), key=lambda kv: -abs(pearsonr(kv[1], adv)[0])):
        print(f"     {nm:<18} r={pearsonr(v, adv)[0]:+.3f}")

    print("\n  PARTITION by charge (median split on abs_charge_frac):")
    ac = feats["abs_charge_frac"]
    med = np.median(ac)
    for lab, m in [("low-charge", ac <= med), ("high-charge", ac > med)]:
        if m.sum() >= 5:
            ro = pearsonr(op[m], y[m])[0]
            rp_ = pearsonr(ppi[m], y[m])[0]
            print(f"     {lab:<12} n={m.sum():<3} ours r={ro:+.3f}  PPI r={rp_:+.3f}  gap={rp_-ro:+.3f}")
    print("\n  reading: if PPI's edge concentrates in high-charge (our documented floor), PDBbind-scale data")
    print("  buys us exactly that regime — the model learns the floor statistically, as PPI did. That is the")
    print("  mechanistic case for the data lever (not a cleverer single-pose physics term, which washes).")


if __name__ == "__main__":
    main()
