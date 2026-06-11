"""E21 — WHY are we at 0.47? Per-complex failure autopsy + missing-term search.

1. LOO residuals for OUR model vs Vina-fit. Which complexes do we miss? Same as Vina or different?
2. Correlate |residual| with complex properties (length, charge, aromatic, hydrophobicity, SS).
3. Test terms Vina has that we lack: rotatable-bond/torsion entropy (Nrot~length), explicit
   vdW-like contact count, electrostatic complementarity.
4. us + Vina ENSEMBLE: if we fail on different complexes, blending beats either alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]
GEO = POCK + IFACE


def loo_pred(X, y):
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pred


def main():
    cr = json.loads(Path("/tmp/e19_cr.json").read_text())
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    # attach vina + length + extra raw terms
    rows = []
    for r in cr:
        b = bench.get(r["pdb"].upper())
        if not b:
            continue
        r = dict(r)
        r["vina"] = b.get("vina_docked")
        r["L"] = b.get("peptide_len", len(r["seq"]))
        rows.append(r)
    y = np.array([r["y"] for r in rows])
    Xg = np.array([[r.get(f, 0.0) for f in GEO] for r in rows], float)
    vina = np.array([r["vina"] for r in rows], float)
    L = np.array([r["L"] for r in rows], float)

    pred_us = loo_pred(Xg, y)
    pred_vina = loo_pred(vina.reshape(-1, 1), y)
    res_us = pred_us - y
    res_vina = pred_vina - y
    r_us = pearsonr(pred_us, y).statistic
    r_vina = pearsonr(pred_vina, y).statistic
    print(f"OUR geometry LOO r={r_us:+.3f} RMSE={np.sqrt((res_us**2).mean()):.2f} | "
          f"Vina-fit r={r_vina:+.3f} RMSE={np.sqrt((res_vina**2).mean()):.2f}")
    print(f"corr(our residual, Vina residual) = {pearsonr(res_us,res_vina).statistic:+.3f}  "
          f"(low => we fail on DIFFERENT complexes => ensemble helps)\n")

    # worst complexes for us
    order = np.argsort(-np.abs(res_us))
    print("=== 10 WORST complexes for OUR model (|residual| kcal/mol) ===")
    print(f"{'pdb':<6}{'L':>3}{'exp':>7}{'pred':>7}{'|err|':>7}{'vina_err':>9}  seq")
    for i in order[:10]:
        r = rows[i]
        print(f"{r['pdb']:<6}{int(L[i]):>3}{y[i]:>7.1f}{pred_us[i]:>7.1f}"
              f"{abs(res_us[i]):>7.1f}{abs(res_vina[i]):>9.1f}  {r['seq'][:24]}")

    # what predicts our error?
    print("\n=== what correlates with OUR |residual|? (property -> error driver) ===")
    props = {"length": L, "poc_eis": np.array([r.get("poc_eis",0) for r in rows]),
             "poc_net": np.array([r.get("poc_net",0) for r in rows]),
             "arom_cc": np.array([r.get("arom_cc",0) for r in rows]),
             "hb_count": np.array([r.get("hb_count",0) for r in rows]),
             "bsa_hyd": np.array([r.get("bsa_hyd",0) for r in rows]),
             "sasa_sb": np.array([r.get("sasa_sb",0) for r in rows]),
             "exp_dG": y}
    ae = np.abs(res_us)
    for nm, v in sorted(props.items(), key=lambda kv: -abs(pearsonr(kv[1], np.abs(res_us)).statistic) if kv[1].std()>0 else 0):
        if v.std() > 0:
            print(f"  corr(|err|, {nm:<10}) = {pearsonr(ae, v).statistic:+.3f}")

    # missing-term tests: add Vina-style terms
    print("\n=== add terms Vina has that we lack (LOO r) ===")
    Nrot = L  # rotatable bonds ~ length (Vina's torsional penalty)
    tests = {
        "ours (geometry)": Xg,
        "+ length/Nrot (torsion)": np.column_stack([Xg, L]),
        "+ vina raw": np.column_stack([Xg, vina]),
        "+ length + vina": np.column_stack([Xg, L, vina]),
    }
    for nm, X in tests.items():
        p = loo_pred(X, y)
        print(f"  {nm:<26}{pearsonr(p,y).statistic:+.3f}  RMSE {np.sqrt(((p-y)**2).mean()):.2f}")

    # ensemble us + vina (z-average of two LOO preds)
    zu = (pred_us - pred_us.mean()) / pred_us.std()
    zv = (pred_vina - pred_vina.mean()) / pred_vina.std()
    for w in (0.3, 0.5, 0.7):
        ze = w * zu + (1 - w) * zv
        print(f"  ENSEMBLE {w:.1f}*us+{1-w:.1f}*vina   r={pearsonr(ze,y).statistic:+.3f}")
    print(f"\n  guess-mean RMSE={y.std():.2f}")


if __name__ == "__main__":
    main()
