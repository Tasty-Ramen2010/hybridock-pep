"""FoldX/Rosetta-inspired pose ranker v2 — genuine attempt to beat ref2015 (0.176).

Fixes the over-insertion confound that capped v1 at τ=0.10, by adding the terms
ref2015 uses to defeat it:
  fa_rep       continuous steep repulsion (magnitude, not a binary clash count)
  desolv_polar Lazaridis-Karplus-style desolvation: burying peptide N/O costs energy
  buried_unsat buried polar atoms with NO H-bond partner (specificity, size-free)
  shape_var    interface fit quality (low nearest-distance variance = conformal)
  hyd_pack     hydrophobic C-C packing in the ideal band
  n_hb, n_sb   satisfied H-bonds, salt bridges

Parses every bench300 pose ONCE, caches features to JSON, then runs CV ranking
experiments cheaply — including COMBINING with ref2015's 16 physics terms
(the "can we add value to ref2015" test). Metric: mean per-complex Kendall τ
vs interface-RMSD.

Usage:
  python scripts/foldx_ranker_v2.py --build      # parse + cache features
  python scripts/foldx_ranker_v2.py              # run experiments on cache
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import scipy.stats as ss

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "logs" / "analysis_bench300"
LABELS = BENCH / "interface_rmsd_labels.json"
PHYS = REPO / "logs" / "diagnosis" / "feats_bench300_physics.pkl"
CACHE = REPO / "logs" / "foldx_v2_features.json"
MODEL = "pretrained"

_POS = {"ARG", "LYS", "HIS"}
_NEG = {"ASP", "GLU"}
FEAT_KEYS = ("fa_rep", "n_clash", "hyd_pack", "desolv_polar", "n_hb",
             "buried_unsat", "n_sb", "shape_var", "band_frac", "n_contact")


def parse(pdb: Path):
    out = []
    for ln in pdb.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        try:
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        out.append((ln[22:27].strip(), ln[17:20].strip(), an, an[0], np.array(xyz)))
    return out


def featurize(pa, ra) -> dict | None:
    if not pa or not ra:
        return None
    P = np.array([a[4] for a in pa]); R = np.array([a[4] for a in ra])
    D = np.sqrt(np.maximum(((P[:, None] - R[None]) ** 2).sum(-1), 1e-6))
    nmin = D.min(1)  # nearest receptor atom per peptide atom

    # continuous steep repulsion: pairs inside 3.6 Å, quadratic ramp
    close = D[(D > 1.5) & (D < 3.6)]
    fa_rep = float(np.sum((3.6 - close) ** 2)) if close.size else 0.0
    n_clash = float((D < 2.9).sum())

    # hydrophobic C-C packing in ideal band
    pe = np.array([a[3] for a in pa]); re_ = np.array([a[3] for a in ra])
    pC = P[pe == "C"]; rC = R[re_ == "C"]
    hyd_pack = 0.0
    if len(pC) and len(rC):
        dc = np.sqrt(((pC[:, None] - rC[None]) ** 2).sum(-1))
        hyd_pack = float(((dc >= 3.4) & (dc < 4.8)).sum())

    # peptide polar atoms (N,O) and their burial / satisfaction
    pol_mask = (pe == "N") | (pe == "O")
    pPol = P[pol_mask]
    rN = R[re_ == "N"]; rO = R[re_ == "O"]
    rNO = np.vstack([rN, rO]) if len(rN) and len(rO) else (rN if len(rN) else rO)
    desolv_polar = 0.0; buried_unsat = 0.0
    if len(pPol):
        dpol = np.sqrt(((pPol[:, None] - R[None]) ** 2).sum(-1))
        burial = (dpol < 4.0).sum(1)  # receptor heavy atoms near each polar
        desolv_polar = float(burial.sum())
        if len(rNO):
            dhb = np.sqrt(((pPol[:, None] - rNO[None]) ** 2).sum(-1))
            satisfied = ((dhb > 2.5) & (dhb < 3.4)).any(1)
        else:
            satisfied = np.zeros(len(pPol), bool)
        buried_unsat = float(((burial >= 4) & (~satisfied)).sum())

    # satisfied interface H-bonds (peptide N↔rec O, peptide O↔rec N)
    pN = P[pe == "N"]; pO = P[pe == "O"]
    def hb(A, B):
        if not len(A) or not len(B):
            return 0
        d = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
        return int((((d > 2.5) & (d < 3.4)).sum()))
    n_hb = float(hb(pN, rO) + hb(pO, rN))

    # salt bridges
    def charged(atoms, names):
        pts = {}
        for rs, rn, an, el, xyz in atoms:
            if rn in names and an not in ("N", "CA", "C", "O"):
                pts.setdefault(rs, []).append(xyz)
        return [np.mean(v, 0) for v in pts.values()]
    n_sb = 0
    for A, B in ((charged(pa, _POS), charged(ra, _NEG)), (charged(pa, _NEG), charged(ra, _POS))):
        for a in A:
            for b in B:
                if np.linalg.norm(a - b) < 5.0:
                    n_sb += 1

    # shape: variance of nearest-receptor distance over buried peptide atoms
    buried_atoms = nmin[nmin < 6.0]
    shape_var = float(np.var(buried_atoms)) if buried_atoms.size > 2 else 0.0
    band_frac = float(((nmin >= 3.3) & (nmin < 4.6)).sum()) / max(1, (nmin < 6.0).sum())
    n_contact = float((D < 4.5).sum())

    return {"fa_rep": fa_rep, "n_clash": n_clash, "hyd_pack": hyd_pack,
            "desolv_polar": desolv_polar, "n_hb": n_hb, "buried_unsat": buried_unsat,
            "n_sb": float(n_sb), "shape_var": shape_var, "band_frac": band_frac,
            "n_contact": n_contact}


def build_cache():
    labels = json.loads(LABELS.read_text())
    out = {}
    cns = sorted(d.name for d in BENCH.glob("peppc_*") if d.is_dir())
    for j, cn in enumerate(cns):
        rec = BENCH / cn / "scoring" / "receptor_cropped.pdb"
        posedir = BENCH / cn / MODEL / "poses"
        lab = labels.get(cn, {}).get(MODEL, {})
        rm = lab.get("interface_rmsds")
        if isinstance(rm, str):
            rm = json.loads(rm)
        if not rec.exists() or not posedir.exists() or not rm:
            continue
        ra = parse(rec)
        rows, rms = [], []
        for i, r in enumerate(rm):
            if r is None:
                continue
            pp = posedir / f"pose_{i}.pdb"
            if not pp.exists():
                continue
            f = featurize(parse(pp), ra)
            if f:
                f["_pose_idx"] = i
                rows.append(f); rms.append(float(r))
        if len(rows) >= 3:
            out[cn] = {"feats": rows, "rmsd": rms}
        if (j + 1) % 30 == 0:
            print(f"  parsed {j+1}/{len(cns)}", flush=True)
    CACHE.write_text(json.dumps(out))
    print(f"Cached {len(out)} complexes → {CACHE.relative_to(REPO)}")


# ── experiments ───────────────────────────────────────────────────────────────

def _solo_taus(pool):
    print("  solo τ (raw feature vs rmsd; sign-agnostic |τ| shows signal):")
    for k in FEAT_KEYS:
        tt = []
        for cn, d in pool.items():
            v = np.array([f[k] for f in d["feats"]]); r = np.array(d["rmsd"])
            t, _ = ss.kendalltau(v, r)
            if not np.isnan(t):
                tt.append(t)
        print(f"     {k:13s} τ={np.mean(tt):+.3f}")


def _cv_ridge_tau(pool, keys, extra=None):
    """LOO-complex ridge predicting rmsd from `keys` (+ optional extra arrays). τ vs rmsd."""
    cns = list(pool)
    X = {cn: np.array([[f[k] for k in keys] for f in pool[cn]["feats"]]) for cn in cns}
    if extra is not None:
        X = {cn: np.hstack([X[cn], extra[cn]]) for cn in cns}
    allX = np.vstack(list(X.values()))
    mu, sd = allX.mean(0), allX.std(0) + 1e-9
    taus = []
    for held in cns:
        tr = [c for c in cns if c != held]
        Xtr = np.vstack([(X[c] - mu) / sd for c in tr])
        ytr = np.concatenate([np.array(pool[c]["rmsd"]) for c in tr])
        A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        reg = 1.0 * np.eye(A.shape[1]); reg[-1, -1] = 0
        w = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
        pred = (X[held] - mu) / sd @ w[:-1] + w[-1]
        t, _ = ss.kendalltau(pred, np.array(pool[held]["rmsd"]))
        if not np.isnan(t):
            taus.append(t)
    return float(np.mean(taus))


def _load_phys(pool):
    """ref2015 16-dim physics per pose, aligned to cached poses via _pose_idx."""
    phys = pickle.load(open(PHYS, "rb"))
    out = {}
    for cn, d in pool.items():
        arrs = []
        ok = True
        for f in d["feats"]:
            key = (cn, MODEL, f["_pose_idx"])
            if key not in phys:
                ok = False; break
            arrs.append(np.asarray(phys[key], float))
        if ok:
            out[cn] = np.array(arrs)
    return out


def experiments():
    pool = json.loads(CACHE.read_text())
    print(f"Loaded {len(pool)} complexes (ref2015 baseline τ=0.176)\n")
    _solo_taus(pool)

    geo = _cv_ridge_tau(pool, FEAT_KEYS)
    print(f"\n  [A] FoldX-v2 geometric (CV ridge, {len(FEAT_KEYS)} terms): τ = {geo:+.3f}")

    phys = _load_phys(pool)
    common = {cn: pool[cn] for cn in pool if cn in phys}
    print(f"  (ref2015 physics aligned for {len(common)}/{len(pool)} complexes)")
    if len(common) >= 20:
        ref = _cv_ridge_tau(common, [], extra=phys)
        comb = _cv_ridge_tau(common, FEAT_KEYS, extra=phys)
        geo_c = _cv_ridge_tau(common, FEAT_KEYS)
        print(f"  [B] ref2015 physics-16 alone (CV):     τ = {ref:+.3f}")
        print(f"  [C] FoldX-v2 geometric alone (CV):     τ = {geo_c:+.3f}")
        print(f"  [D] COMBINED ref2015 + FoldX-v2 (CV):  τ = {comb:+.3f}")
        print(f"\n  → FoldX-v2 adds {comb-ref:+.3f} over ref2015 physics alone "
              f"({'GENUINE GAIN' if comb > ref + 0.005 else 'no real gain'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.build or not CACHE.exists():
        build_cache()
    if not args.build:
        experiments()


if __name__ == "__main__":
    main()
