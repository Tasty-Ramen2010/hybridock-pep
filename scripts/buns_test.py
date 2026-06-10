#!/usr/bin/env python3
"""
buns_test.py — Buried Unsatisfied Polar atoms as a per-pose ranking feature.

REF2015 blind spot #3: it under-penalizes buried polar atoms with no H-bond
partner. This signal VARIES per pose and is ORTHOGONAL to BSA (BSA = how much
area is buried; BUNS = whether the buried polar atoms are satisfied). Two poses
can bury equal area; the one leaving a dangling buried Asp is worse — BSA can't
see that, BUNS can.

Per pose: count peptide polar atoms (N/O) that are BURIED (per-atom SASA in
complex < BURY_SASA) AND have NO polar heavy-atom partner within H-bond range
(1.8–3.5 Å, excluding same residue). That count = BUNS.

Tests:
  τ(−BUNS, −RMSD) per complex                 (does fewer unsats → lower RMSD?)
  BSA+clash         vs  BSA+clash + w·BUNS     (does BUNS add orthogonal signal?)
  BUNS-only

Reuses BSA cache (feats_gen_n100_bsa.pkl) for the BSA/clash terms; caches BUNS.
Run (rapidock env, has Biopython): python3 scripts/buns_test.py
"""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy import stats as sp
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.bsa_tail_test import per_atom_sasa, CROP

BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")
GEN_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
ENC_PKL  = REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"
BSA_CACHE = REPO / "logs" / "diagnosis" / "feats_gen_n100_bsa.pkl"
BUNS_CACHE = REPO / "logs" / "diagnosis" / "feats_gen_n100_buns.pkl"
OUT = REPO / "logs" / "training_campaign" / "buns_test.json"

BURY_SASA = 5.0      # Å² — peptide polar atom buried if complex SASA below this
HB_MIN, HB_MAX = 1.8, 3.5   # Å heavy-atom polar-polar H-bond distance window


def read_atoms(pdb: str):
    """Return (lines, xyz[N,3], elem[N], resid[N]) for heavy atoms."""
    lines, xyz, elem, rid = [], [], [], []
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        # element: prefer cols 76-78, else first alpha of atom name
        e = ln[76:78].strip() or "".join(c for c in an if c.isalpha())[:1]
        e = e.upper()
        try:
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except ValueError:
            continue
        lines.append(ln); elem.append(e); rid.append((ln[21], ln[22:27]))
    return lines, (np.array(xyz, np.float32) if xyz else np.empty((0, 3), np.float32)), elem, rid


def compute_buns(pep_pdb: str, rec_lines, rec_xyz, rec_elem):
    pep_lines, pep_xyz, pep_elem, pep_rid = read_atoms(pep_pdb)
    if len(pep_xyz) < 4 or len(rec_xyz) < 4:
        return None
    # crop receptor near peptide
    d2 = ((rec_xyz[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1)
    near = d2.min(1) <= CROP ** 2
    crop_lines = [rec_lines[i] for i in np.where(near)[0]]
    crop_xyz = rec_xyz[near]; crop_elem = [rec_elem[i] for i in np.where(near)[0]]

    # per-atom SASA of peptide IN COMPLEX (peptide portion)
    s_cx = per_atom_sasa(pep_lines + crop_lines)[:len(pep_lines)]
    if len(s_cx) != len(pep_lines):
        return None

    pep_polar = np.array([e in ("N", "O") for e in pep_elem])
    rec_polar_xyz = crop_xyz[np.array([e in ("N", "O") for e in crop_elem])] \
        if len(crop_xyz) else np.empty((0, 3))

    buns = 0
    for i in range(len(pep_xyz)):
        if not pep_polar[i] or s_cx[i] >= BURY_SASA:
            continue  # not a buried polar atom
        # partners: any polar heavy atom (peptide other-residue OR receptor) in window
        satisfied = False
        # receptor partners
        if len(rec_polar_xyz):
            dr = np.sqrt(((rec_polar_xyz - pep_xyz[i]) ** 2).sum(1))
            if np.any((dr > HB_MIN) & (dr <= HB_MAX)):
                satisfied = True
        # peptide partners (different residue)
        if not satisfied:
            for j in range(len(pep_xyz)):
                if j == i or not pep_polar[j] or pep_rid[j] == pep_rid[i]:
                    continue
                dd = np.linalg.norm(pep_xyz[j] - pep_xyz[i])
                if HB_MIN < dd <= HB_MAX:
                    satisfied = True; break
        if not satisfied:
            buns += 1
    return float(buns)


def build_buns_cache():
    bjson = json.load(open(GEN_JSON))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))
    cache = pickle.load(open(BUNS_CACHE, "rb")) if BUNS_CACHE.exists() else {}
    t0 = time.time(); n_new = 0
    for ci, cn in enumerate(cxs):
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        pdir = Path(entry.get("poses_dir", ""))
        rec_pdb = BASE / cn / f"{cn}_protein_pocket.pdb"
        if not rec_pdb.exists() or len(rr) < 10:
            continue
        rec_lines, rec_xyz, rec_elem, _ = read_atoms(str(rec_pdb))
        for pi in range(len(rr)):
            if (cn, pi) in cache:
                continue
            pp = pdir / f"pose_{pi}.pdb"
            cache[(cn, pi)] = compute_buns(str(pp), rec_lines, rec_xyz, rec_elem) \
                if pp.exists() else None
            n_new += 1
        if (ci + 1) % 10 == 0:
            pickle.dump(cache, open(BUNS_CACHE, "wb"), protocol=4)
            print(f"  BUNS cached {ci+1} cx  {time.time()-t0:.0f}s", flush=True)
    pickle.dump(cache, open(BUNS_CACHE, "wb"), protocol=4)
    print(f"  BUNS cache done ({n_new} new, {time.time()-t0:.0f}s)", flush=True)
    return cache


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def main():
    print("Building/loading BUNS cache...", flush=True)
    buns = build_buns_cache()
    bjson = json.load(open(GEN_JSON))
    bsa_cache = pickle.load(open(BSA_CACHE, "rb"))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))

    tau = {"buns": [], "bsa": [], "bsa_buns": []}
    hit1 = {"bsa": [], "bsa_buns": []}
    hit5 = {"bsa": [], "bsa_buns": []}
    mean_buns = []
    for cn in cxs:
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        if len(rr) < 10:
            continue
        bs, cl, bn, rm = [], [], [], []
        for pi in range(len(rr)):
            c = bsa_cache.get((cn, pi)); b = buns.get((cn, pi))
            if c is None or b is None:
                continue
            bs.append(c[0]); cl.append(c[1]); bn.append(b); rm.append(rr[pi])
        if len(rm) < 10:
            continue
        bs = np.array(bs); cl = np.array(cl); bn = np.array(bn); rm = np.array(rm)
        mean_buns.append(bn.mean())

        bsa_score = -_z(bs) + _z(cl)                    # lower = better
        bsa_buns  = -_z(bs) + _z(cl) + 0.5 * _z(bn)     # + BUNS penalty
        # τ: higher "goodness" = lower RMSD → use -score
        for name, sc in [("buns", _z(bn)), ("bsa", bsa_score), ("bsa_buns", bsa_buns)]:
            t, _ = sp.kendalltau(-sc, -rm)
            if not np.isnan(t):
                tau[name].append(t)
        for name, sc in [("bsa", bsa_score), ("bsa_buns", bsa_buns)]:
            o = np.argsort(sc)
            hit1[name].append(float(rm[o[0]] <= 2.0))
            hit5[name].append(float(rm[o[:5]].min() <= 2.0))

    n = len(hit1["bsa"])
    print(f"\n{'='*62}")
    print(f"BURIED UNSATISFIED POLAR (BUNS) TEST  ({n} complexes)")
    print(f"{'='*62}")
    print(f"  mean BUNS per pose = {np.mean(mean_buns):.1f}\n")
    print(f"  Kendall τ (vs RMSD):")
    print(f"    BUNS only       τ = {np.mean(tau['buns']):+.4f}")
    print(f"    BSA+clash       τ = {np.mean(tau['bsa']):+.4f}")
    print(f"    BSA+clash+BUNS  τ = {np.mean(tau['bsa_buns']):+.4f}  "
          f"(Δ={np.mean(tau['bsa_buns'])-np.mean(tau['bsa']):+.4f})")
    print(f"\n  Top-1 / Top-5 Hit@2Å:")
    print(f"    BSA+clash       {100*np.mean(hit1['bsa']):>5.1f}% / {100*np.mean(hit5['bsa']):>5.1f}%")
    print(f"    BSA+clash+BUNS  {100*np.mean(hit1['bsa_buns']):>5.1f}% / {100*np.mean(hit5['bsa_buns']):>5.1f}%")
    dtau = np.mean(tau['bsa_buns']) - np.mean(tau['bsa'])
    print(f"\n  → {'BUNS ADDS orthogonal signal' if dtau > 0.02 else 'BUNS does NOT help over BSA+clash'} "
          f"(Δτ={dtau:+.4f})")

    OUT.write_text(json.dumps({
        "n": n, "mean_buns": float(np.mean(mean_buns)),
        "tau": {k: float(np.mean(v)) for k, v in tau.items()},
        "hit1": {k: float(np.mean(v)) for k, v in hit1.items()},
        "hit5": {k: float(np.mean(v)) for k, v in hit5.items()},
    }, indent=2))
    print(f"\n  Saved → {OUT}")


if __name__ == "__main__":
    main()
