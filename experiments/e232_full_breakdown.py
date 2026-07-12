"""E232 — full ours-vs-PPI breakdown by LENGTH x CHARGE on T100 (PPI home) and fresh PPIKB (de-redundant),
plus: does the RISM receptor-baseline signal CONCENTRATE on charged receptors (i.e. can it address the floor
where PPI beats us on T100)?  Uses e210's trained models.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import e210_curated_fresh_headtohead as e210  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e179_protdcal_3d as e179  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_regression  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

PPI_T100 = 0.424   # PPI-Affinity real fair r on T100 (non-vlong)


def qnet(seq):
    return sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16"


def rmae(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok])))


def breakdown(tag, seqs, yv, qv, Lv, pred_ours, pred_clone, clone_home_r):
    print(f"\n=== {tag} (n={len(yv)}) ===")
    ro, mo = rmae(pred_ours, yv); rc, mc = rmae(pred_clone, yv)
    ppi = rc / clone_home_r * PPI_T100 if clone_home_r > 0 else float("nan")
    print(f"  OVERALL   ours r={ro:+.3f} MAE={mo:.2f}   clone r={rc:+.3f}   est.PPI r={ppi:+.3f}")
    yv = np.asarray(yv); qv = np.asarray(qv); Lv = np.asarray(Lv)
    print("  by LENGTH:")
    for b in ("short≤8", "med9-12", "long13-16"):
        m = np.array([band(int(L)) == b for L in Lv])
        if m.sum() >= 6:
            ro2, _ = rmae(pred_ours[m], yv[m]); rc2, _ = rmae(pred_clone[m], yv[m])
            print(f"    {b:<10} n={int(m.sum()):<4} ours={ro2:+.3f}  est.PPI={rc2/clone_home_r*PPI_T100:+.3f}")
    print("  by CHARGE:")
    for nm, m in [("charged|q|≥2", np.abs(qv) >= 2), ("neutral|q|≤1", np.abs(qv) <= 1)]:
        if m.sum() >= 6:
            ro2, mo2 = rmae(pred_ours[m], yv[m]); rc2, _ = rmae(pred_clone[m], yv[m])
            print(f"    {nm:<12} n={int(m.sum()):<4} ours={ro2:+.3f} MAE={mo2:.2f}  est.PPI={rc2/clone_home_r*PPI_T100:+.3f}")


def main():
    # --- train both models on 925 (reuse e210) ---
    base = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
    Xo, Xc, ytr = [], [], []
    for b in base:
        ps = e158.pocket_seq(b["pdb"])
        if ps is None:
            continue
        pn = (sum(c in "KR" for c in ps) - sum(c in "DE" for c in ps)) / max(len(ps), 1)
        Xo.append(e210.our_feat(b["seq"], pn, e210.pocket_pkf_from_seq(ps)))
        Xc.append(b["desc"]); ytr.append(b["y"])
    Xo = np.nan_to_num(Xo); Xc = np.nan_to_num(Xc); ytr = np.array(ytr)
    ours = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
            l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(Xo, ytr)
    clone = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                      ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xc, ytr)

    # --- T100 (PPI home) ---
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    t = {"seq": [], "y": [], "q": [], "L": [], "xo": [], "xc": []}
    for l in open(ROOT / "data/t100_extra_features.jsonl"):
        d = json.loads(l); pid = d["pdb"].lower()
        if pid not in man or len(d["seq"]) >= 17:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        if res is None:
            continue
        ps = e158.pocket_seq(pid) or ""
        pn = (sum(c in "KR" for c in ps) - sum(c in "DE" for c in ps)) / max(len(ps), 1)
        t["xo"].append(e210.our_feat(d["seq"], pn, e210.pocket_pkf_from_seq(ps)))
        t["xc"].append(e179.descriptors(res, 6.0, 3))
        t["seq"].append(d["seq"]); t["y"].append(float(man[pid]["dg_exp"]))
        t["q"].append(qnet(d["seq"])); t["L"].append(len(d["seq"]))
    po_t = ours.predict(np.nan_to_num(t["xo"])); pc_t = clone.predict(np.nan_to_num(t["xc"]))
    clone_home_r, _ = rmae(pc_t, t["y"])
    breakdown("T100 (PPI HOME — redundant w/ PPI training)", t["seq"], t["y"], t["q"], t["L"], po_t, pc_t, clone_home_r)

    # --- fresh curated PPIKB (de-redundant) ---
    cur = e210.curate()
    Xo_f = np.nan_to_num([e210.our_feat(r["seq"],
            (sum(c in "KR" for c in (e158.pocket_seq(r["pdb"]) or "")) -
             sum(c in "DE" for c in (e158.pocket_seq(r["pdb"]) or ""))) / max(r.get("npocket", 1), 1),
            r["pocket_pkf"]) for r in cur])
    Xc_f = np.nan_to_num([r["desc3d"] for r in cur])
    po_f = ours.predict(Xo_f); pc_f = clone.predict(Xc_f)
    yv = [r["y"] for r in cur]; qv = [r["net_charge"] for r in cur]; Lv = [r["length"] for r in cur]
    breakdown("FRESH PPIKB (de-redundant, honest)", [r["seq"] for r in cur], yv, qv, Lv, po_f, pc_f, clone_home_r)

    # --- does RISM baseline signal concentrate on CHARGED receptors? ---
    print("\n=== RISM receptor-baseline: charged vs neutral receptors (the floor where PPI wins T100) ===")
    rism = [json.loads(l) for l in open(ROOT / "data/e230_rism.jsonl")]
    manf = {r["peptides"][0]["pdb"]: r for r in json.load(open(ROOT / "data/e228_pilot_manifest.json"))["receptors"]}
    y = np.array([r["y_mean"] for r in rism])
    # receptor net charge (per residue) from rec_seq
    rq = np.array([qnet(manf[r["rep_pdb"]]["rec_seq"]) / max(len(manf[r["rep_pdb"]]["rec_seq"]), 1) for r in rism])
    nsites = np.array([r["n_sites"] for r in rism]); meang = np.array([r["mean_g"] for r in rism])
    size = np.array([manf[r["rep_pdb"]]["receptor_len"] for r in rism], float)
    def pcorr(x, t, z):
        rx = x - np.polyval(np.polyfit(z, x, 1), z); rt = t - np.polyval(np.polyfit(z, t, 1), z)
        return float(np.corrcoef(rx, rt)[0, 1])
    hi = rq >= np.median(rq)
    for nm, m in [("more-charged receptors", hi), ("less-charged receptors", ~hi)]:
        if m.sum() >= 5:
            print(f"  {nm:<24} n={int(m.sum())}  n_sites→base r={np.corrcoef(nsites[m], y[m])[0,1]:+.3f}  "
                  f"mean_g→base r={np.corrcoef(meang[m], y[m])[0,1]:+.3f}")
    print(f"  ALL (size-controlled): n_sites→base r={pcorr(nsites, y, size):+.3f}  mean_g→base r={pcorr(meang, y, size):+.3f}")


if __name__ == "__main__":
    main()
