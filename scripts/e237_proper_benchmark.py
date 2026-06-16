"""E237 — THE proper, single benchmark Ram asked for.
  (i)   T100: OURS (with length router) vs PPI-Affinity's REAL predictions (SI-File-6, full-100 = 0.554).
  (iii) PPIKB curated Kd set: same breakdown.
  (iv)  ablation: does adding (contacts/L, buried-area/L) fix the long-band over-crediting?
  (v)   calibration: shrinkage slope before/after linear calibration (note: linear cal fixes MAE/slope, not r).
All models trained on PDBbind-925 MINUS the test set (no leakage). Breakdown by LENGTH x CHARGE.
Our model defined ONCE: structural-16 + ProtDCal(seq) + charge-compl + length [+ per-length norms],
with the SHORT(<=8) length router -> lean ridge on (bsa_hyd, mj_contact, strength_bur).
"""
from __future__ import annotations

import csv
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
from hybridock_pep.scoring.affinity_model import _protdcal_descriptors, _charge_complementarity  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

STRUCT = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
          "arom_cc", "hb_count", "mj_contact", "strength_bur", "rg_per_L", "org_density", "cys_frac", "mean_burial"]
SHORT_FEATS = ["bsa_hyd", "mj_contact", "strength_bur"]
PPI_FULL_T100 = 0.554


def g(r, k):
    return float(r.get(k, 0) or 0)


def perlen(r):
    """(contacts or buried-area) / length — removes the long-peptide over-crediting bias."""
    L = max(len(r["seq"]), 1)
    return [g(r, "mj_contact") / L, g(r, "bsa_hyd") / L, g(r, "hb_count") / L, g(r, "poc_n") / L]


def feat(r, perL):
    base = [g(r, k) for k in STRUCT] + _protdcal_descriptors(r["seq"]) + \
           _charge_complementarity(r["seq"], g(r, "poc_net")) + [float(len(r["seq"]))]
    return base + (perlen(r) if perL else [])


def rmae(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), float("nan"), int(ok.sum())
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))), int(ok.sum())


def _hgb():
    return HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
            l2_regularization=3.0, min_samples_leaf=12, random_state=0)


class RoutedModel:
    """short<=8 -> lean ridge(SHORT_FEATS); else -> HGB(full feat). Length router, no leakage."""
    def __init__(self, perL):
        self.perL = perL

    def fit(self, rows, y):
        y = np.asarray(y)
        sh = np.array([len(r["seq"]) <= 8 for r in rows])
        self.full = _hgb().fit(np.nan_to_num([feat(r, self.perL) for r in rows]), y)
        if sh.sum() >= 12:
            Xs = np.nan_to_num([[g(r, k) for k in SHORT_FEATS] for r in rows if len(r["seq"]) <= 8])
            self.sc = StandardScaler().fit(Xs)
            self.short = Ridge(alpha=2.0).fit(self.sc.transform(Xs), y[sh])
        else:
            self.short = None
        return self

    def predict(self, rows):
        out = self.full.predict(np.nan_to_num([feat(r, self.perL) for r in rows]))
        if self.short is not None:
            for i, r in enumerate(rows):
                if len(r["seq"]) <= 8:
                    xs = self.sc.transform([[g(r, k) for k in SHORT_FEATS]])
                    out[i] = self.short.predict(xs)[0]
        return out


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def breakdown(tag, rows, y, pred, ppi_pred=None, ppi_full=None):
    y = np.asarray(y); pred = np.asarray(pred); L = np.array([len(r["seq"]) for r in rows])
    q = np.array([sum(c in "KR" for c in r["seq"]) - sum(c in "DE" for c in r["seq"]) for r in rows])
    r, m, n = rmae(pred, y)
    slope = np.polyfit(pred[~np.isnan(pred)], y[~np.isnan(pred)], 1)[0]
    print(f"\n=== {tag} (n={n}) ===")
    print(f"  OURS overall   r={r:+.3f}  MAE={m:.2f}  shrink-slope={slope:.2f}")
    if ppi_pred is not None:
        rp, mp, npp = rmae(ppi_pred, y)
        print(f"  PPI  overall   r={rp:+.3f}  MAE={mp:.2f}  (n={npp})" + (f"   [PPI full-100={ppi_full}]" if ppi_full else ""))
    for b in ("short≤8", "med9-12", "long13-16", "vlong≥17"):
        mb = np.array([band(int(x)) == b for x in L])
        if mb.sum() >= 5:
            r2, m2, _ = rmae(pred[mb], y[mb])
            extra = ""
            if ppi_pred is not None:
                rp2, _, _ = rmae(np.asarray(ppi_pred)[mb], y[mb]); extra = f"   PPI r={rp2:+.3f}"
            print(f"    {b:<10} n={int(mb.sum()):<3} ours r={r2:+.3f} MAE={m2:.2f}{extra}")
    for nm, mb in [("charged|q|≥2", np.abs(q) >= 2), ("neutral|q|≤1", np.abs(q) <= 1)]:
        if mb.sum() >= 5:
            r2, m2, _ = rmae(pred[mb], y[mb]); extra = ""
            if ppi_pred is not None:
                rp2, _, _ = rmae(np.asarray(ppi_pred)[mb], y[mb]); extra = f"   PPI r={rp2:+.3f}"
            print(f"    {nm:<12} n={int(mb.sum()):<3} ours r={r2:+.3f} MAE={m2:.2f}{extra}")


def main():
    # ---------- T100 ----------
    si = {row["PDB_NAME"].strip()[:4].lower(): row for row in
          csv.DictReader(open(ROOT / "data/biolip/ppiaffinity_si/SI/SI-File-6-protein-peptide-test-set-1.csv"))}
    ppi_all = np.array([float(v["PPI-Affinity"]) for v in si.values()])
    y_all = np.array([float(v["Binding_affinity"]) for v in si.values()])
    rp_full = float(np.corrcoef(ppi_all, y_all)[0, 1])
    print(f"=== PPI-Affinity REAL on full T100 (SI-File-6): r={rp_full:.3f}  (published 0.554) ===")

    t100 = [json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")]
    tids = {r["pdb"].lower() for r in t100}
    tr = [r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")) if r["pdb"].lower() not in tids]
    ytr = [r["y"] for r in tr]
    t_ppi = np.array([float(si[r["pdb"].lower()]["PPI-Affinity"]) if r["pdb"].lower() in si else np.nan for r in t100])

    for perL in (False, True):
        mdl = RoutedModel(perL).fit(tr, ytr)
        breakdown(f"T100  [ours router{'+perlen' if perL else ''}]  vs PPI(on same subset)",
                  t100, [float(r["y"]) for r in t100], mdl.predict(t100), t_ppi, f"{rp_full:.3f}")

    # ---------- PPIKB curated Kd ----------
    ppikb = []
    seen = set()
    for r in (json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl")):
        if r["pdb"].lower() in tids or r.get("aff_type") not in ("Kd", "KD", "pKd"):
            continue
        if not (2 <= r["length"] <= 25) or not (-18 < r["y"] < -2):
            continue
        if r["pdb"].lower() in {x["pdb"].lower() for x in tr}:   # fresh vs training
            continue
        if r["seq"] in seen:
            continue
        seen.add(r["seq"])
        # ppikb_features uses different keys? map structural ones if present
        ppikb.append(r)
    print(f"\n  (PPIKB curated Kd fresh set: n={len(ppikb)} — only seq+desc3d available, so TRANSFERABLE model)")
    # PPIKB has no structural geometry -> transferable model (ProtDCal-seq + charge + len) vs PPI-CLONE (desc3d)
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.svm import SVR

    def tfeat(r):
        return _protdcal_descriptors(r["seq"]) + _charge_complementarity(r["seq"], 0.0) + [float(len(r["seq"]))]
    Xt = np.nan_to_num([tfeat(r) for r in tr])
    ours_t = _hgb().fit(Xt, np.array(ytr))
    base925 = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl")
               if json.loads(l).get("desc") and json.loads(l)["pdb"].lower() not in tids]
    clone = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                      ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(
                          np.nan_to_num([b["desc"] for b in base925]), np.array([b["y"] for b in base925]))
    dlen = len(base925[0]["desc"])
    ppikb = [r for r in ppikb if isinstance(r.get("desc3d"), list) and len(r["desc3d"]) == dlen]
    print(f"  (with valid desc3d: n={len(ppikb)})")
    yk = np.array([float(r["y"]) for r in ppikb])
    po = ours_t.predict(np.nan_to_num([tfeat(r) for r in ppikb]))
    pc = clone.predict(np.nan_to_num([r["desc3d"] for r in ppikb]))
    breakdown(f"PPIKB-{len(ppikb)}  [ours TRANSFERABLE vs PPI-clone]", ppikb, yk, po, pc)


if __name__ == "__main__":
    main()
