#!/usr/bin/env python3
"""
ranker_failure_modes.py — Where do ref2015 and BSA+clash each fail, and can we route?

Both rankers sit at τ≈0.10-0.14 globally. This asks a DIFFERENT question:
do they fail on different STRUCTURAL classes (deep grooves, long tails, flat
interfaces...)? If errors are complementary, a per-complex router beats either.

Steps:
  1. Cache per-pose BSA + clash + peptide free-SASA (Shrake-Rupley, once).
  2. Per complex: τ for ref2015 and BSA+clash vs RMSD.
  3. Structural descriptors per complex:
       burial_frac   BSA(best pose)/SASA_pep_free   (deep groove = high)
       enclosure     receptor heavy atoms within 6Å of best pose / pep atoms
       pep_len       peptide length
       oracle_rmsd   best ref_rmsd (are good poses even present?)
       pose_spread   mean pairwise Cα RMSD (pose diversity)
  4. Spearman(τ_ranker, descriptor) → where each ranker fails.
  5. Router: oracle upper bound max(τ_ref,τ_bsa); descriptor-threshold router.

Run (rapidock/score-env): python3 scripts/ranker_failure_modes.py [--rebuild]
"""
from __future__ import annotations

import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy import stats as sp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.bsa_tail_test import read_heavy, per_atom_sasa, CLASH_DIST, CROP

BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")
GEN_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
ENC_PKL  = REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"
PHYS_PKL = REPO / "logs" / "diagnosis" / "feats_gen_n100_physics.pkl"
BSA_CACHE = REPO / "logs" / "diagnosis" / "feats_gen_n100_bsa.pkl"
OUT = REPO / "logs" / "training_campaign" / "ranker_failure_modes.json"


def read_ca(pdb: str) -> np.ndarray:
    xyz = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
           for l in open(pdb) if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return np.array(xyz, np.float32) if xyz else np.empty((0, 3), np.float32)


def build_bsa_cache() -> dict:
    """Per-pose (bsa, n_clash, sasa_pep_free, n_rec_within6). Cache to pkl."""
    bjson = json.load(open(GEN_JSON))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))
    cache: dict = {}
    t0 = time.time()
    for ci, cn in enumerate(cxs):
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        pdir = Path(entry.get("poses_dir", ""))
        rec_pdb = BASE / cn / f"{cn}_protein_pocket.pdb"
        if not rec_pdb.exists() or len(rr) < 10:
            continue
        rec_lines, rec_xyz, _ = read_heavy(str(rec_pdb))
        for pi in range(len(rr)):
            pp = pdir / f"pose_{pi}.pdb"
            if not pp.exists():
                continue
            pep_lines, pep_xyz, _ = read_heavy(str(pp))
            if len(pep_xyz) < 4:
                cache[(cn, pi)] = None; continue
            d2 = ((rec_xyz[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1)
            near = d2.min(1) <= CROP ** 2
            crop = [rec_lines[i] for i in np.where(near)[0]]
            s_free = per_atom_sasa(pep_lines)
            s_cx = per_atom_sasa(pep_lines + crop)[:len(pep_lines)]
            if len(s_free) != len(pep_lines) or len(s_cx) != len(pep_lines):
                cache[(cn, pi)] = None; continue
            bsa = float(np.maximum(s_free - s_cx, 0.0).sum())
            pd2 = ((pep_xyz[:, None, :] - rec_xyz[None, :, :]) ** 2).sum(-1)
            n_clash = float((pd2.min(1) < CLASH_DIST ** 2).sum())
            n_rec6 = float((pd2.min(0) < 6.0 ** 2).sum())  # rec atoms near peptide
            cache[(cn, pi)] = (bsa, n_clash, float(s_free.sum()), n_rec6)
        if (ci + 1) % 10 == 0:
            print(f"  cached {ci+1} cx  {time.time()-t0:.0f}s", flush=True)
    pickle.dump(cache, open(BSA_CACHE, "wb"), protocol=4)
    print(f"  Saved BSA cache → {BSA_CACHE}  ({time.time()-t0:.0f}s)")
    return cache


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def spearman(a, b):
    r, _ = sp.spearmanr(a, b)
    return float(r) if not np.isnan(r) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    if a.rebuild or not BSA_CACHE.exists():
        print("Building BSA cache...", flush=True)
        cache = build_bsa_cache()
    else:
        print(f"Loading BSA cache {BSA_CACHE}", flush=True)
        cache = pickle.load(open(BSA_CACHE, "rb"))

    bjson = json.load(open(GEN_JSON))
    phys = pickle.load(open(PHYS_PKL, "rb"))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))

    rows = []  # one per complex
    for cn in cxs:
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        pdir = Path(entry.get("poses_dir", ""))
        if len(rr) < 10:
            continue
        bsas, clashes, frees, rec6, refs, rmsds, cas, pis = [], [], [], [], [], [], [], []
        for pi in range(len(rr)):
            c = cache.get((cn, pi)); pv = phys.get((cn, "pretrained", pi))
            if c is None or pv is None:
                continue
            bsa, ncl, sfree, n6 = c
            bsas.append(bsa); clashes.append(ncl); frees.append(sfree); rec6.append(n6)
            refs.append(float(pv[13])); rmsds.append(rr[pi]); pis.append(pi)
            cas.append(read_ca(str(pdir / f"pose_{pi}.pdb")))
        if len(rmsds) < 10:
            continue
        bsas = np.array(bsas); clashes = np.array(clashes); frees = np.array(frees)
        rec6 = np.array(rec6); refs = np.array(refs); rmsds = np.array(rmsds)

        bsa_score = -_z(bsas) + _z(clashes)            # lower = better
        tau_ref, _ = sp.kendalltau(-refs, -rmsds)
        tau_bsa, _ = sp.kendalltau(-bsa_score, -rmsds)

        best = int(np.argmin(rmsds))
        burial_frac = bsas[best] / (frees[best] + 1e-9)
        L = min(len(c) for c in cas) if cas else 0
        if L >= 2:
            A = np.stack([c[:L] for c in cas])
            sp_spread = float(np.mean([np.sqrt(((A - A[i])**2).sum(-1).mean(-1)).mean()
                                       for i in range(0, len(A), 5)]))
        else:
            sp_spread = 0.0
        rows.append({
            "cn": cn,
            "tau_ref": float(tau_ref) if not np.isnan(tau_ref) else 0.0,
            "tau_bsa": float(tau_bsa) if not np.isnan(tau_bsa) else 0.0,
            "burial_frac": float(burial_frac),
            "enclosure": float(rec6[best] / max(L, 1)),
            "pep_len": int(L),
            "oracle_rmsd": float(rmsds.min()),
            "pose_spread": sp_spread,
        })

    n = len(rows)
    tr = np.array([r["tau_ref"] for r in rows])
    tb = np.array([r["tau_bsa"] for r in rows])
    descs = ["burial_frac", "enclosure", "pep_len", "oracle_rmsd", "pose_spread"]

    print(f"\n{'='*68}")
    print(f"RANKER FAILURE MODES  ({n} complexes)")
    print(f"{'='*68}")
    print(f"\n  ref2015 τ = {tr.mean():+.4f}    BSA+clash τ = {tb.mean():+.4f}")
    print(f"  corr(τ_ref, τ_bsa) = {np.corrcoef(tr, tb)[0,1]:+.3f}  "
          f"(high → same failures → routing won't help)")

    print(f"\n  Where each ranker FAILS — Spearman(τ, descriptor):")
    print(f"  (negative = ranker does WORSE as descriptor rises)")
    print(f"  {'descriptor':<14} {'vs τ_ref':>10} {'vs τ_bsa':>10} {'vs (ref-bsa)':>13}")
    print(f"  {'-'*50}")
    diff = tr - tb
    for d in descs:
        dv = np.array([r[d] for r in rows], float)
        print(f"  {d:<14} {spearman(tr,dv):>+10.3f} {spearman(tb,dv):>+10.3f} "
              f"{spearman(diff,dv):>+13.3f}")

    # routing
    oracle_router = np.maximum(tr, tb).mean()
    print(f"\n  ROUTING:")
    print(f"  ref2015 alone           τ = {tr.mean():+.4f}")
    print(f"  BSA+clash alone         τ = {tb.mean():+.4f}")
    print(f"  oracle router (best/cx) τ = {oracle_router:+.4f}  "
          f"(+{oracle_router-max(tr.mean(),tb.mean()):.4f} headroom)")
    # best single-descriptor threshold router
    best_router = (max(tr.mean(), tb.mean()), "none", 0.0)
    for d in descs:
        dv = np.array([r[d] for r in rows], float)
        for thr in np.percentile(dv, [25, 50, 75]):
            use_ref = dv >= thr
            routed = np.where(use_ref, tr, tb).mean()
            if routed > best_router[0]:
                best_router = (routed, f"{d}>= {thr:.2f}→ref", routed)
    print(f"  best descriptor router  τ = {best_router[0]:+.4f}  ({best_router[1]})")

    OUT.write_text(json.dumps({
        "n": n, "tau_ref": float(tr.mean()), "tau_bsa": float(tb.mean()),
        "corr_ref_bsa": float(np.corrcoef(tr, tb)[0,1]),
        "oracle_router": float(oracle_router),
        "best_descriptor_router": best_router[0], "router_rule": best_router[1],
        "rows": rows,
    }, indent=2))
    print(f"\n  Saved → {OUT}")


if __name__ == "__main__":
    main()
