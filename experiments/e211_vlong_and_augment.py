"""E211 — Ram's two asks:
  A) curated fresh head-to-head WITH vlong included (vs E210 which excluded it).
  B) AUGMENT training with PPIKB long+vlong ("everything we have"): does adding the brand-new dataset's
     long/vlong complexes to the seq+pocket model lift T100 long/vlong (the home-field bands we lose)?
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
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
SN = list(_SCALES.keys())
PPI_FAIR, PPI_FULL = 0.424, 0.525


def rmae(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = ~(np.isnan(p) | np.isnan(y))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def pkf(ps):
    return [float(np.mean([_SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else [0.0] * len(SN)


def ofeat(seq, pn, pocket_pkf):
    return _protdcal_descriptors(seq) + list(pocket_pkf) + _charge_complementarity(seq, pn) + [float(len(seq))]


def hgb():
    return HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                         l2_regularization=3.0, min_samples_leaf=12, random_state=0)


def main():
    ours = {json.loads(l)["pdb"].lower() for l in open(ROOT / "data/pdbbind_peptides.jsonl")}
    ppikb = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl") if json.loads(l).get("desc3d")]

    def pn_of(pid, npocket):
        ps = e158.pocket_seq(pid) or ""
        return (sum(c in "KR" for c in ps) - sum(c in "DE" for c in ps)) / max(npocket, 1)

    # ---- A. curated fresh WITH vlong ----
    seen = set(); cur = []
    for r in sorted(ppikb, key=lambda x: x["pdb"]):
        if r["pdb"].lower() in ours or r["aff_type"] not in ("Kd", "KD", "pKd"):
            continue
        if not (2 <= r["length"] <= 50) or not (-18 < r["y"] < -2):
            continue
        if abs(r.get("npep", r["length"]) - r["length"]) > 2 or r.get("npocket", 0) < 10 or r["seq"] in seen:
            continue
        seen.add(r["seq"]); cur.append(r)
    # train on 925
    base = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
    Xo, Xc, ytr = [], [], []
    for b in base:
        ps = e158.pocket_seq(b["pdb"])
        if ps is None:
            continue
        Xo.append(ofeat(b["seq"], pn_of(b["pdb"], len(ps)), pkf(ps))); Xc.append(b["desc"]); ytr.append(b["y"])
    Xo = np.nan_to_num(Xo); Xc = np.nan_to_num(Xc); ytr = np.array(ytr)
    om = hgb().fit(Xo, ytr)
    cm = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                   ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xc, ytr)
    yc = np.array([r["y"] for r in cur]); Lc = np.array([r["length"] for r in cur])
    Xo_f = np.nan_to_num([ofeat(r["seq"], pn_of(r["pdb"], r.get("npocket", 1)), r["pocket_pkf"]) for r in cur])
    Xc_f = np.nan_to_num([r["desc3d"] for r in cur])
    # clone home (T100 WITH vlong this time, full)
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    t100d3, t100y = [], []
    for pid, d in seqc.items():
        if pid not in man:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        if res:
            t100d3.append(e179.descriptors(res, 6.0, 3)); t100y.append(float(man[pid]["dg_exp"]))
    rc_h, _ = rmae(cm.predict(np.nan_to_num(t100d3)), np.array(t100y))
    ro_f, _ = rmae(om.predict(Xo_f), yc); rc_f, _ = rmae(cm.predict(Xc_f), yc)
    ratio = rc_f / rc_h if rc_h > 0 else float("nan")
    print(f"=== A. CURATED FRESH WITH VLONG (n={len(cur)}, vlong={int((Lc>=17).sum())}) ===")
    print(f"  OURS r={ro_f:+.3f}  PPI-clone r={rc_f:+.3f}  clone-home(T100full)={rc_h:+.3f}  ratio={ratio:.2f}")
    print(f"  extrapolated PPI fresh: {PPI_FULL*ratio:+.3f}  → {'WE WIN' if ro_f>PPI_FULL*ratio else 'PPI'}")
    for nm, mk in [("vlong>=17", Lc >= 17), ("long13-16", (Lc >= 13) & (Lc <= 16)), ("<=12", Lc <= 12)]:
        if mk.sum() >= 4:
            r1, _ = rmae(om.predict(Xo_f)[mk], yc[mk]); r2, _ = rmae(cm.predict(Xc_f)[mk], yc[mk])
            print(f"    {nm:<10} n={int(mk.sum()):<4} ours={r1:+.3f} clone={r2:+.3f}")

    # ---- B. AUGMENT training with PPIKB long+vlong, test T100 long/vlong ----
    print(f"\n=== B. AUGMENT 925 with PPIKB long+vlong → T100 long/vlong (seq+pocket model) ===")
    # STRICT no-leak: exclude 925, T100 pdbs, AND T100 sequences (E211 found 6 pdb + 1 seq leaks)
    t100_seqs = {seqc[p]["seq"] for p in man if p in seqc}
    aug = []
    for r in ppikb:
        if r["pdb"].lower() in ours or r["pdb"].lower() in man or r["seq"] in t100_seqs:
            continue
        if r["aff_type"] not in ("Kd", "KD", "pKd"):
            continue
        if r["length"] < 13 or not (-18 < r["y"] < -2):
            continue
        if abs(r.get("npep", r["length"]) - r["length"]) > 2 or r.get("npocket", 0) < 8:
            continue
        aug.append(r)
    print(f"  PPIKB long+vlong augmentation pool (T100 pdb+seq excluded): {len(aug)}")
    Xa = np.nan_to_num([ofeat(r["seq"], pn_of(r["pdb"], r.get("npocket", 1)), r["pocket_pkf"]) for r in aug])
    ya = np.array([r["y"] for r in aug])
    # T100 features (seq+pocket) held out
    tt = []
    for pid, d in seqc.items():
        if pid not in man or pid in ours:
            pass
        m = man.get(pid)
        if m is None:
            continue
        ps = e158.pocket_seq(pid) or ""
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        tt.append({"x": ofeat(d["seq"], (sum(c in "KR" for c in ps) - sum(c in "DE" for c in ps)) / max(len(ps), 1), pkf(ps)),
                   "y": float(m["dg_exp"]), "ship": ship, "L": len(d["seq"])})
    Xtt = np.nan_to_num([t["x"] for t in tt]); ytt = np.array([t["y"] for t in tt]); Ltt = np.array([t["L"] for t in tt])
    ship = np.array([t["ship"] for t in tt])
    for label, Xadd, yadd in [("925 only", None, None), ("925 + PPIKB long/vlong", Xa, ya)]:
        if Xadd is None:
            m = hgb().fit(Xo, ytr)
        else:
            m = hgb().fit(np.vstack([Xo, Xadd]), np.concatenate([ytr, yadd]))
        p = m.predict(Xtt)
        rl, _ = rmae(p[(Ltt >= 13) & (Ltt <= 16)], ytt[(Ltt >= 13) & (Ltt <= 16)])
        rv, _ = rmae(p[Ltt >= 17], ytt[Ltt >= 17])
        ro, _ = rmae(p, ytt)
        print(f"  {label:<24}: T100 overall={ro:+.3f}  long={rl:+.3f}  vlong={rv:+.3f}")
    rlp, _ = rmae(ship[(Ltt >= 13) & (Ltt <= 16)], ytt[(Ltt >= 13) & (Ltt <= 16)])
    rvp, _ = rmae(ship[Ltt >= 17], ytt[Ltt >= 17])
    print(f"  {'PPI shipped':<24}: T100 overall={rmae(ship,ytt)[0]:+.3f}  long={rlp:+.3f}  vlong={rvp:+.3f}")


if __name__ == "__main__":
    main()
