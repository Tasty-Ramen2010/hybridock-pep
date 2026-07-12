"""E238 — validate a LONG (13-16) specialist the same way E215 validated vlong: production model for L<=12,
925+PPIKB-long-augmented model for 13<=L<=16, 925+PPIKB-vlong for L>=17. Non-long/non-vlong predictions must
stay byte-identical to production. Report T100 per band (prod vs routed vs PPI). Ship only if long improves
and the untouched bands are byte-identical.
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
import e215_vlong_router_ppikb as e215  # noqa: E402  (reuse vec262, size_regs, apply_regs, rmae)
from hybridock_pep.scoring.affinity_model import GEOMETRY_KEYS, build_feature_vector  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402


def ppikb_aug(lo, hi):
    aug = []
    for ln in (ROOT / "data/e212_ppikb_geom.jsonl").read_text().splitlines():
        e = json.loads(ln)
        if not e.get("geom") or not (lo <= e["length"] <= hi):
            continue
        g = {k: float(e["geom"].get(k, 0.0)) for k in GEOMETRY_KEYS}
        x = build_feature_vector(g, e["seq"])
        x = (x[:262] if x.shape[0] >= 262 else np.pad(x, (0, 262 - x.shape[0]))).copy()
        if e.get("pocket_pkf") and len(e["pocket_pkf"]) == 22:
            x[240:262] = np.array(e["pocket_pkf"], float)
        aug.append((np.nan_to_num(x), float(e["y"]), e["length"]))
    return aug


def fit_aug(X9, y9, L9, aug):
    Xa = np.vstack([X9] + ([np.array([a[0] for a in aug])] if aug else []))
    ya = np.concatenate([y9] + ([np.array([a[1] for a in aug])] if aug else []))
    La = np.concatenate([L9] + ([np.array([a[2] for a in aug])] if aug else []))
    rg = e215.size_regs(Xa, La)
    return e202._hgb().fit(e215.apply_regs(Xa, La, rg), ya), rg


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    have = {r["pdb"].lower(): r for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl"))}
    cache = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}

    tr = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        if r["pdb"].lower() in man:
            continue
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        tr.append((e215.vec262(r, r["seq"], ps), float(r["y"]), r["length"]))
    X9 = np.nan_to_num([r[0] for r in tr]); y9 = np.array([r[1] for r in tr]); L9 = np.array([r[2] for r in tr])
    rg9 = e215.size_regs(X9, L9)
    prod = e202._hgb().fit(e215.apply_regs(X9, L9, rg9), y9)
    long_m, rgl = fit_aug(X9, y9, L9, ppikb_aug(13, 16))
    vlong_m, rgv = fit_aug(X9, y9, L9, ppikb_aug(17, 999))
    print(f"  aug counts: long13-16={len(ppikb_aug(13,16))}  vlong>=17={len(ppikb_aug(17,999))}")

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
        test.append((e215.vec262(d, d["seq"], ps), float(m["dg_exp"]), len(d["seq"]), ship,
                     abs(sum(c in "KR" for c in d["seq"]) - sum(c in "DE" for c in d["seq"]))))
    Xte = np.nan_to_num([t[0] for t in test]); y = np.array([t[1] for t in test]); L = np.array([t[2] for t in test])
    ship = np.array([t[3] for t in test]); q = np.array([t[4] for t in test])

    p_prod = prod.predict(e215.apply_regs(Xte, L, rg9))
    p_routed = p_prod.copy()
    lo = (L >= 13) & (L <= 16); vl = L >= 17
    if lo.any():
        p_routed[lo] = long_m.predict(e215.apply_regs(Xte[lo], L[lo], rgl))
    if vl.any():
        p_routed[vl] = vlong_m.predict(e215.apply_regs(Xte[vl], L[vl], rgv))
    untouched = ~(lo | vl)
    print("=== LONG+VLONG ROUTER on T100 (E238) ===")
    print(f"  untouched (L<=12) byte-identical: max|Δ|={np.max(np.abs(p_prod[untouched]-p_routed[untouched])):.6f}")
    for nm, mk in [("OVERALL", np.ones(len(y), bool)), ("<=12", untouched), ("long13-16", lo),
                   ("vlong>=17", vl), ("charged|q|>=2", q >= 2)]:
        if mk.sum() < 4:
            continue
        rp, mp = e215.rmae(p_prod, y, mk); rr, mr = e215.rmae(p_routed, y, mk); rpp, _ = e215.rmae(ship, y, mk)
        print(f"  {nm:<14} n={int(mk.sum()):<4} prod={rp:+.3f}  routed={rr:+.3f} (MAE {mp:.2f}->{mr:.2f})  PPI={rpp:+.3f}")


if __name__ == "__main__":
    main()
