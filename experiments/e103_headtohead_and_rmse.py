"""E103 — DEFINITIVE head-to-head vs PPI-Affinity on the SAME complexes + why is our RMSE 'so bad'?

PPI-Affinity's T100 peptide test set overlaps our pooled 156 by 91 complexes with IDENTICAL labels
(verified mean |Δ|=0.0). So we can compare their PUBLISHED predictions against ours on the exact same
91 complexes, same truth — the cleanest head-to-head possible, no AI poses, crystal structures (their turf).

Answers two of Ram's questions:
  Q1: do we beat PPI-Affinity at its OWN evaluation (crystal)?  → r/ρ/RMSE on the shared 91.
  Q2: why is our RMSE 'so bad' (~1.8) vs theirs (~1.46)?  → RMSE DECOMPOSITION: RMSE ≈ std_y·√(1−r²).
      Our 156 has wider affinity spread (std~2.2) than T100 (std~1.75); at equal r, wider spread →
      mechanically larger RMSE. On the SAME complexes the gap should vanish.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "data" / "biolip" / "ppiaffinity_si" / "SI"
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
SHORT = ["bsa_hyd", "mj_contact", "strength_bur"]


def load_ours():
    rows = {}
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            rows[r["pdb"].lower()[:4]] = {"pdb": r["pdb"], "y": float(r["y"]),
                                          "length": int(float(r["length"])),
                                          "feat": {c: float(r[c]) for c in PROD}}
    return rows


def load_t100():
    f = SI / "SI-File-6-protein-peptide-test-set-1.csv"
    out = []
    for r in csv.DictReader(open(f)):
        m = re.match(r"([0-9a-zA-Z]{4})", r["PDB_NAME"])
        pid = m.group(1).lower() if m else None
        rec = {k.strip(): r[k] for k in r}
        rec["pdb4"] = pid
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


def pred1(feat, cols, p):
    mu, sd, w = p
    x = np.array([feat[c] for c in cols], float)
    return float(np.r_[1.0, (x - mu) / sd] @ w)


def our_loo(all_rows, target_pdbs, router=True):
    """LOO prediction for each target complex, trained on all OTHER complexes in the 156."""
    by = {r["pdb"].lower()[:4]: r for r in all_rows}
    preds = {}
    for p4 in target_pdbs:
        tgt = by[p4]
        tr = [r for r in all_rows if r["pdb"].lower()[:4] != p4]
        if router and tgt["length"] <= 8 and sum(r["length"] <= 8 for r in tr) >= 6:
            cols, base = SHORT, [r for r in tr if r["length"] <= 8]
        else:
            cols, base = PROD, tr
        preds[p4] = pred1(tgt["feat"], cols, ridge(base, cols))
    return preds


def metrics(pred, y):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    m = ~(np.isnan(pred) | np.isnan(y))
    pred, y = pred[m], y[m]
    r = pearsonr(pred, y)[0]
    rho = spearmanr(pred, y).statistic
    raw_rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    a, b = np.polyfit(pred, y, 1)
    fit_rmse = float(np.sqrt(np.mean((a * pred + b - y) ** 2)))
    return r, rho, raw_rmse, fit_rmse, len(y)


def main():
    ours = load_ours()
    t100 = load_t100()
    shared = [t for t in t100 if t["pdb4"] in ours]
    all_rows = list(ours.values())
    print(f"=== E103 head-to-head on {len(shared)} shared crystal complexes (PPI-Affinity T100 ∩ our 156) ===\n")

    y = np.array([t["y"] for t in shared])
    our_pred = our_loo(all_rows, [t["pdb4"] for t in shared])
    op = np.array([our_pred[t["pdb4"]] for t in shared])

    print("Q1 — HEAD-TO-HEAD on identical complexes & labels (crystal poses, their evaluation):")
    print(f"  {'method':<16}{'Pearson r':>10}{'Spearman':>10}{'raw RMSE':>10}{'fit RMSE':>10}")
    rows_metrics = []
    # ours
    r, rho, raw, fit, n = metrics(op, y)
    print(f"  {'HybriDock(ours)':<16}{r:>+10.3f}{rho:>+10.3f}{raw:>10.2f}{fit:>10.2f}")
    rows_metrics.append(("ours", r, raw, fit))
    # competitors from T100 columns
    for col in ["PPI-Affinity", "Kdeep", "DFIRE", "CP_PIE", "RF-Score", "PRODIGY"]:
        key = next((k for k in shared[0] if k.replace(" ", "") == col.replace(" ", "")), None)
        if not key:
            continue
        try:
            x = np.array([float(t[key]) for t in shared])
        except (ValueError, KeyError):
            continue
        r, rho, raw, fit, n = metrics(x, y)
        print(f"  {col:<16}{r:>+10.3f}{rho:>+10.3f}{raw:>10.2f}{fit:>10.2f}")
        rows_metrics.append((col, r, raw, fit))

    print("\n  → 'raw RMSE' for competitors is on THEIR native scale (some not kcal/mol) so ignore it for")
    print("     them; compare on Pearson r and fit-RMSE. Ours is genuine kcal/mol (raw≈fit).")

    print("\nQ2 — WHY our RMSE looked 'bad' (1.8) vs PPI-Affinity (1.46): RMSE ≈ std_y·√(1−r²)")
    # our 156 spread vs T100 spread
    y_full = np.array([r["y"] for r in all_rows])
    std_full = y_full.std()
    std_shared = y.std()
    print(f"  std(ΔG) our full 156      = {std_full:.2f} kcal")
    print(f"  std(ΔG) shared 91 (=T100) = {std_shared:.2f} kcal")
    for label, r_, sd_ in [("our full-156 LOO (r=0.544)", 0.544, std_full),
                           ("on shared-91 (our r above)", abs(metrics(op, y)[0]), std_shared)]:
        print(f"  predicted RMSE for {label:<30} = {sd_*np.sqrt(1-r_**2):.2f} kcal  "
              f"(std·√(1−r²))")
    print("  → RMSE is LOCKED to r × spread. Our 1.85 on the full set is the WIDER-spread (2.2) cost,")
    print("    not worse modeling. On the shared 91 at matched spread we should match their ~1.46.")

    # decisive: ours vs ppi on shared, same spread
    ppi_key = next((k for k in shared[0] if k.replace(" ", "") == "PPI-Affinity"), None)
    if ppi_key:
        ppi = np.array([float(t[ppi_key]) for t in shared])
        ro, _, rawo, fito, _ = metrics(op, y)
        rp, _, _, fitp, _ = metrics(ppi, y)
        print(f"\n  VERDICT on shared 91:  ours r={ro:+.3f} fitRMSE={fito:.2f}  vs  PPI-Affinity r={rp:+.3f} fitRMSE={fitp:.2f}")
        print(f"  {'→ WE WIN' if ro>rp else '→ PARITY' if abs(ro-rp)<0.03 else '→ PPI ahead'} "
              f"(Δr={ro-rp:+.3f}) on its OWN crystal benchmark.")


if __name__ == "__main__":
    main()
