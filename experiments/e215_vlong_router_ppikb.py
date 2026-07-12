"""E215 — Ram's two asks:
  PART 1: VLONG-ONLY ROUTER on T100 — production pocket model for L<17, augmented (925+PPIKB-vlong) model for
          L>=17. Non-vlong predictions BYTE-IDENTICAL to production; only vlong changes. Validate.
  PART 2: UNBIASED PPIKB test WITH vlong — curated fresh PPIKB (Kd, unique-seq, structure-clean, not in 925,
          not in T100), INCLUDING vlong. OUR routed model vs PPI-clone, per band + ratio-scale to real PPI.
          Strict no-leak: fresh test pdbs/seqs excluded from any PPIKB augmentation.
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
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import (build_feature_vector, GEOMETRY_KEYS, SIZE_IDX,  # noqa: E402
                                                  _protdcal_descriptors, _SCALES)
import e158_overfit_failure_analysis as e158  # noqa: E402
import e179_protdcal_3d as e179  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
SN = list(_SCALES.keys())


def rmae(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return (float("nan"), float("nan"))
    return (float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok]))))


def pkf(ps):
    return [float(np.mean([_SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else [0.0] * len(SN)


def vec262(g0, seq, pocket_seq):
    g = {k: float(g0.get(k, 0.0)) for k in GEOMETRY_KEYS}
    g["pocket_seq"] = pocket_seq or ""
    x = build_feature_vector(g, seq)
    return x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))


def size_regs(X, L):
    return {j: LinearRegression().fit(L.reshape(-1, 1), X[:, j]) for j in SIZE_IDX}


def apply_regs(X, L, regs):
    X = X.copy()
    for j, lr in regs.items():
        X[:, j] = X[:, j] - lr.predict(L.reshape(-1, 1))
    return X


def part1_t100_router():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    # 925 train (262-feat)
    tr = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        if r["pdb"].lower() in man:
            continue
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        tr.append((vec262(r, r["seq"], ps), float(r["y"]), r["length"]))
    # PPIKB vlong augmentation (geom-complete, e212), no T100 leak
    aug = []
    for ln in (ROOT / "data/e212_ppikb_geom.jsonl").read_text().splitlines():
        e = json.loads(ln)
        if not e.get("geom") or e["length"] < 17:
            continue
        g = {k: float(e["geom"].get(k, 0.0)) for k in GEOMETRY_KEYS}
        x = build_feature_vector(g, e["seq"])
        x = (x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))).copy()
        if e.get("pocket_pkf") and len(e["pocket_pkf"]) == 22:
            x[240:262] = np.array(e["pocket_pkf"], float)
        aug.append((np.nan_to_num(x), float(e["y"]), e["length"]))
    # test
    test = []
    for pid, m in man.items():
        d = have.get(pid) or cache.get(pid)
        if d is None:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        ps = e158.pocket_seq(pid) or ""
        test.append((vec262(d, d["seq"], ps), float(m["dg_exp"]), len(d["seq"]), ship,
                     abs(sum(c in "KR" for c in d["seq"]) - sum(c in "DE" for c in d["seq"]))))
    Xte = np.nan_to_num([t[0] for t in test]); y = np.array([t[1] for t in test]); L = np.array([t[2] for t in test])
    ship = np.array([t[3] for t in test]); q = np.array([t[4] for t in test])

    X9 = np.nan_to_num([r[0] for r in tr]); y9 = np.array([r[1] for r in tr]); L9 = np.array([r[2] for r in tr])
    rg9 = size_regs(X9, L9); prod = e202._hgb().fit(apply_regs(X9, L9, rg9), y9)
    Xa = np.vstack([X9] + ([np.array([a[0] for a in aug])] if aug else []))
    ya = np.concatenate([y9] + ([np.array([a[1] for a in aug])] if aug else []))
    La = np.concatenate([L9] + ([np.array([a[2] for a in aug])] if aug else []))
    rga = size_regs(Xa, La); augm = e202._hgb().fit(apply_regs(Xa, La, rga), ya)

    p_prod = prod.predict(apply_regs(Xte, L, rg9))
    p_routed = p_prod.copy()
    vl = L >= 17
    p_routed[vl] = augm.predict(apply_regs(Xte[vl], L[vl], rga))
    print("=== PART 1: VLONG-ONLY ROUTER on T100 ===")
    print(f"  non-vlong byte-identical: max|Δ|={np.max(np.abs(p_prod[~vl]-p_routed[~vl])):.6f} (must be 0)")
    for nm, mk in [("OVERALL", np.ones(len(y), bool)), ("non-vlong", ~vl), ("vlong>=17", vl), ("charged|q|>=2", q >= 2)]:
        rp, _ = rmae(p_prod, y, mk); rr, _ = rmae(p_routed, y, mk); rpp, _ = rmae(ship, y, mk)
        print(f"  {nm:<14} n={int(mk.sum()):<4} prod={rp:+.3f}  routed={rr:+.3f}  PPI={rpp:+.3f}")


def part2_ppikb_unbiased():
    print("\n=== PART 2: UNBIASED PPIKB fresh WITH vlong — ours vs PPI-clone ===")
    man = {m["pdb"].lower() for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    ours = {json.loads(l)["pdb"].lower() for l in open(ROOT / "data/pdbbind_peptides.jsonl")}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    t100_seqs = {seqc[p]["seq"] for p in man if p in seqc}
    ppikb = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl") if json.loads(l).get("desc3d")]
    seen = set(); fresh = []
    for r in sorted(ppikb, key=lambda x: x["pdb"]):
        if r["pdb"].lower() in ours or r["aff_type"] not in ("Kd", "KD", "pKd"):
            continue
        if not (2 <= r["length"] <= 50) or not (-18 < r["y"] < -2):
            continue
        if abs(r.get("npep", r["length"]) - r["length"]) > 2 or r.get("npocket", 0) < 10 or r["seq"] in seen:
            continue
        seen.add(r["seq"]); fresh.append(r)
    fresh_seqs = {r["seq"] for r in fresh}; fresh_pdbs = {r["pdb"].lower() for r in fresh}
    # train on 925 only (clean — fresh held out)
    base = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
    Xo, Xc, ytr = [], [], []
    for b in base:
        ps = e158.pocket_seq(b["pdb"])
        if ps is None:
            continue
        pn = (sum(c in "KR" for c in ps) - sum(c in "DE" for c in ps)) / max(len(ps), 1)
        Xo.append(_protdcal_descriptors(b["seq"]) + pkf(ps) + [pn * 0, 0, 0, float(len(b["seq"]))])
        Xc.append(b["desc"]); ytr.append(b["y"])
    Xo = np.nan_to_num(Xo); Xc = np.nan_to_num(Xc); ytr = np.array(ytr)
    om = e202._hgb().fit(Xo, ytr)
    cm = Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=37)),
                   ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))]).fit(Xc, ytr)
    yc = np.array([r["y"] for r in fresh]); Lc = np.array([r["length"] for r in fresh]); qc = np.array([abs(r["net_charge"]) for r in fresh])
    Xo_f = np.nan_to_num([_protdcal_descriptors(r["seq"]) + list(r["pocket_pkf"]) + [0, 0, 0, float(r["length"])] for r in fresh])
    Xc_f = np.nan_to_num([r["desc3d"] for r in fresh])
    po = om.predict(Xo_f); pc = cm.predict(Xc_f)
    # clone home (T100 full) for ratio-scale
    t100d3, t100y = [], []
    for pid in man:
        d = seqc.get(pid)
        if d is None:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        if res:
            t100d3.append(e179.descriptors(res, 6.0, 3))
            import json as _j
            man2 = {m["pdb"].lower(): m for m in _j.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
            t100y.append(float(man2[pid]["dg_exp"]))
    rc_h, _ = rmae(cm.predict(np.nan_to_num(t100d3)), np.array(t100y))
    print(f"  curated fresh n={len(fresh)} (vlong={int((Lc>=17).sum())}), charged={int((qc>=2).sum())}")
    print(f"  {'band':<14}{'n':>4}{'OURS':>8}{'CLONE':>8}{'est.PPI':>9}")
    for nm, mk in [("OVERALL", np.ones(len(yc), bool)), ("<=12", Lc <= 12), ("long13-16", (Lc >= 13) & (Lc <= 16)),
                   ("vlong>=17", Lc >= 17), ("charged|q|>=2", qc >= 2), ("neutral|q|<=1", qc <= 1)]:
        if mk.sum() < 4:
            continue
        ro, _ = rmae(po, yc, mk); rcl, _ = rmae(pc, yc, mk)
        est_ppi = (rcl / rc_h * 0.525) if rc_h > 0 else float("nan")
        win = "← WE WIN" if ro > est_ppi else ""
        print(f"  {nm:<14}{int(mk.sum()):>4}{ro:>+8.3f}{rcl:>+8.3f}{est_ppi:>+9.3f}  {win}")
    print(f"  (est.PPI = clone_band/clone_T100home × PPI_0.525; clone home r={rc_h:+.3f})")


def main():
    part1_t100_router()
    part2_ppikb_unbiased()


if __name__ == "__main__":
    main()
