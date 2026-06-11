"""E40 — REAL free-state conformational entropy from MD (GPU), not a sequence proxy.

Runs 60ps free-peptide MD, measures actual dihedral-histogram entropy S_free + RMSF. The
peptide LOSES this on binding → +TΔS penalty. Tests whether the ACTUAL free entropy (vs the
weak sequence proxy in e39) bridges cross-dataset generalization. Feature S_free × buried_frac
= the entropy that actually freezes. ~8s/peptide.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e18v2_md import run_free_dynamics  # noqa: E402


def compute(which):
    out_path = Path(f"/tmp/e40_{which}.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    if which == "cr":
        e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
        geo = json.loads(Path("/tmp/e19_cr.json").read_text())
        items = [(g["pdb"].upper(), e0[g["pdb"].upper()].get("pep_pdb"))
                 for g in geo if g["pdb"].upper() in e0 and e0[g["pdb"].upper()].get("pep_pdb")]
    else:
        e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
        work = Path("/tmp/ppep_work")
        items = [(k, str(work / f"{k}_pep.pdb")) for k in e28 if (work / f"{k}_pep.pdb").exists()]
    for key, pep in items:
        if key in out or not pep:
            continue
        try:
            rmsf, sdih = run_free_dynamics(pep, 60)
            out[key] = dict(s_free=float(np.nanmean(sdih)), s_free_total=float(np.nansum(sdih)),
                            rmsf=float(np.nanmean(rmsf)))
            out_path.write_text(json.dumps(out))
            if len(out) % 10 == 0:
                print(f"  {which} {len(out)} done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {type(e).__name__}", flush=True)
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("cr", "both"):
        print("=== crystal-65 free MD entropy ===", flush=True); compute("cr")
    if which in ("b98", "both"):
        print("=== the-98 free MD entropy ===", flush=True); compute("b98")
    # eval
    cp, bp = Path("/tmp/e40_cr.json"), Path("/tmp/e40_b98.json")
    if not (cp.exists() and bp.exists()):
        return
    from scipy.stats import pearsonr
    inten = json.loads(Path("/tmp/e31_intensive.json").read_text())
    geo = json.loads(Path("/tmp/e19_cr.json").read_text())
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    sc = json.loads(cp.read_text()); sb = json.loads(bp.read_text())
    cr = []
    for r, it in zip(geo, inten["cr"]):
        k = r["pdb"].upper()
        if k in sc:
            cr.append(dict(it, s_free=sc[k]["s_free"], s_free_total=sc[k]["s_free_total"],
                           rmsf=sc[k]["rmsf"], s_free_bur=sc[k]["s_free"] * min(1.0, it.get("f_hyd_iface", 0.5))))
    b98 = []
    for (k, v), it in zip(e28.items(), inten["b98"]):
        if k in sb:
            b98.append(dict(it, s_free=sb[k]["s_free"], s_free_total=sb[k]["s_free_total"],
                            rmsf=sb[k]["rmsf"], s_free_bur=sb[k]["s_free"] * min(1.0, it.get("f_hyd_iface", 0.5))))
    ycr = np.array([r["y"] for r in cr]); y98 = np.array([r["y"] for r in b98])
    print(f"\n=== REAL MD free-state entropy: sign-consistency (cr n={len(cr)}, 98 n={len(b98)}) ===")
    for f in ["s_free", "s_free_total", "rmsf", "s_free_bur"]:
        rc = pearsonr([r[f] for r in cr], ycr).statistic
        r9 = pearsonr([r[f] for r in b98], y98).statistic
        print(f"  {f:<14}{rc:+.3f} / {r9:+.3f}  {'UNIVERSAL' if rc*r9>0 and min(abs(rc),abs(r9))>0.1 else 'flip/weak'}")
    UNI = ["bsa_hyd", "mj_per_contact", "f_hyd_iface", "frac_pol_satisfied"]

    def mat(rows, feats):
        return np.array([[r.get(f, 0.0) for f in feats] for r in rows])

    def transfer(tr, te, feats):
        Xtr, yt = mat(tr, feats), np.array([r["y"] for r in tr]); Xte, ye = mat(te, feats), np.array([r["y"] for r in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd]); w, *_ = np.linalg.lstsq(A, yt, rcond=None)
        return pearsonr(np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w, ye).statistic

    def loo(rows, feats):
        y = np.array([r["y"] for r in rows]); X = mat(rows, feats); p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd]); w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            p[i] = np.r_[1, (X[i] - mu) / sd] @ w
        return pearsonr(p, y).statistic, np.sqrt(((p - y) ** 2).mean())
    pool = cr + b98
    print("\n=== does REAL MD free entropy bridge generalization? ===")
    print(f"  {'feature set':<34}{'cr->98':>9}{'98->cr':>9}{'pool LOO':>10}")
    for nm, fs in [("universal (4) [baseline]", UNI), ("+ s_free", UNI + ["s_free"]),
                   ("+ s_free_bur", UNI + ["s_free_bur"]), ("+ rmsf", UNI + ["rmsf"]),
                   ("+ s_free + rmsf", UNI + ["s_free", "rmsf"])]:
        print(f"  {nm:<34}{transfer(cr,b98,fs):>+9.3f}{transfer(b98,cr,fs):>+9.3f}{loo(pool,fs)[0]:>+10.3f}")
    print("  baseline +0.24/+0.37 pool 0.421 >> does ACTUAL entropy beat the sequence proxy (e39: no)?")


if __name__ == "__main__":
    main()
