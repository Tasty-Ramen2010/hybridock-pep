"""E210 — CURATED fresh benchmark from PPIKB + head-to-head: our crystal model vs PPI-clone, ratio-scaled to
estimate real PPI on fresh data (Ram's ask). Both trained on 925, predicted on a clean PPIKB subset that is:
  fresh (not in our 925), Kd-only, unique sequence, non-vlong, valid ΔG (−18<y<−2),
  structure-clean (|npep−len|<=2, npocket>=10).
OUR model = transferable part (seq-ProtDCal + pocket-ProtDCal + charge-compl + length) — no interface geometry
(not available for PPIKB crystals, and it's the non-transferable deployment-only part anyway).
PPI-clone = ProtDCal-3D intra-peptide contact descriptors (their feature class).
Ratio-scale: estimated PPI_fresh = PPI_T100_fair × (clone_fresh / clone_T100).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_regression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import _protdcal_descriptors, _charge_complementarity, _SCALES  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e179_protdcal_3d as e179  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
SN = list(_SCALES.keys())
PPI_T100_FAIR = 0.424  # PPI's real T100 r excluding vlong (E208)
PPI_T100_FULL = 0.525


def rmae(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = ~(np.isnan(p) | np.isnan(y))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def pocket_pkf_from_seq(ps):
    return [float(np.mean([_SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else [0.0] * len(SN)


def our_feat(seq, poc_net, pocket_pkf):
    return (_protdcal_descriptors(seq) + list(pocket_pkf)
            + _charge_complementarity(seq, poc_net) + [float(len(seq))])


def curate():
    ours = {json.loads(l)["pdb"].lower() for l in open(ROOT / "data/pdbbind_peptides.jsonl")}
    rows = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl") if json.loads(l).get("desc3d")]
    seen = set(); out = []
    for r in sorted(rows, key=lambda x: x["pdb"]):
        if r["pdb"].lower() in ours:                       # fresh for us
            continue
        if r["aff_type"] not in ("Kd", "KD", "pKd"):       # clean Kd
            continue
        if not (2 <= r["length"] <= 16):                   # non-vlong, sane
            continue
        if not (-18.0 < r["y"] < -2.0):                    # valid binding ΔG
            continue
        if abs(r.get("npep", r["length"]) - r["length"]) > 2 or r.get("npocket", 0) < 10:
            continue
        if r["seq"] in seen:                               # unique sequence
            continue
        seen.add(r["seq"]); out.append(r)
    return out


def main():
    cur = curate()
    print(f"=== CURATED fresh PPIKB benchmark: n={len(cur)} (Kd, unique-seq, non-vlong, structure-clean, fresh) ===")
    yc = np.array([r["y"] for r in cur]); qc = np.array([abs(r["net_charge"]) for r in cur])
    print(f"  y: [{yc.min():.1f},{yc.max():.1f}] std={yc.std():.2f}  charged|q|>=2: {(qc>=2).sum()}")

    # ---- training data: 925 (build both feature sets) ----
    base = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
    tr_ours, tr_clone, ytr = [], [], []
    for b in base:
        ps = e158.pocket_seq(b["pdb"])
        if ps is None:
            continue
        r925 = None  # poc_net not in e180; approximate from pocket seq
        pn = (sum(c in "KR" for c in ps) - sum(c in "DE" for c in ps)) / max(len(ps), 1)
        tr_ours.append(our_feat(b["seq"], pn, pocket_pkf_from_seq(ps)))
        tr_clone.append(b["desc"]); ytr.append(b["y"])
    Xo = np.nan_to_num(tr_ours); Xc = np.nan_to_num(tr_clone); ytr = np.array(ytr)

    # ---- T100 (clone home reference) ----
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    t100 = []
    for pid, d in seqc.items():
        if pid not in man:
            continue
        if len(d["seq"]) >= 17:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        if res is None:
            continue
        t100.append({"d3": e179.descriptors(res, 6.0, 3), "y": float(man[pid]["dg_exp"])})

    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

    def hgb():
        return HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                             l2_regularization=3.0, min_samples_leaf=12, random_state=0)
    ours_model = hgb().fit(Xo, ytr)
    clone = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                      ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xc, ytr)

    # predict fresh
    Xo_f = np.nan_to_num([our_feat(r["seq"], (sum(c in "KR" for c in (e158.pocket_seq(r["pdb"]) or "")) -
                          sum(c in "DE" for c in (e158.pocket_seq(r["pdb"]) or ""))) / max(r.get("npocket", 1), 1),
                          r["pocket_pkf"]) for r in cur])
    Xc_f = np.nan_to_num([r["desc3d"] for r in cur])
    ro_f, mo_f = rmae(ours_model.predict(Xo_f), yc)
    rc_f, mc_f = rmae(clone.predict(Xc_f), yc)
    # clone home (T100)
    rc_h, _ = rmae(clone.predict(np.nan_to_num([t["d3"] for t in t100])), np.array([t["y"] for t in t100]))
    ratio = rc_f / rc_h if rc_h > 0 else float("nan")

    print(f"\n  OURS (seq+pocket, trained 925) on fresh:   r={ro_f:+.3f}  MAE={mo_f:.2f}")
    print(f"  PPI-CLONE (ProtDCal-3D, trained 925) on fresh: r={rc_f:+.3f}  MAE={mc_f:.2f}")
    print(f"  PPI-clone home (T100 non-vlong): r={rc_h:+.3f}  → retention ratio fresh/home = {ratio:.2f}")
    print(f"\n  EXTRAPOLATED real PPI-Affinity on this fresh set:")
    print(f"     via fair T100 ({PPI_T100_FAIR}):  {PPI_T100_FAIR*ratio:+.3f}")
    print(f"     via full T100 ({PPI_T100_FULL}):  {PPI_T100_FULL*ratio:+.3f}")
    print(f"  OURS on fresh:                       {ro_f:+.3f}")
    win = "WE WIN" if ro_f > PPI_T100_FAIR * ratio else "PPI"
    print(f"  → on fresh curated data: {win}  (ours {ro_f:.3f} vs est. PPI {PPI_T100_FAIR*ratio:.3f})")
    # charged subslice
    for nm, mk in [("charged|q|>=2", qc >= 2), ("neutral|q|<=1", qc <= 1)]:
        if mk.sum() >= 5:
            roc, _ = rmae(ours_model.predict(Xo_f)[mk], yc[mk]); rcc, _ = rmae(clone.predict(Xc_f)[mk], yc[mk])
            print(f"    {nm:<14} n={int(mk.sum()):<4} ours={roc:+.3f}  clone={rcc:+.3f}  est.PPI={rcc/rc_h*PPI_T100_FAIR if rc_h>0 else float('nan'):+.3f}")


if __name__ == "__main__":
    main()
