#!/usr/bin/env python3
"""
bsa_nis_combo_test.py — Test BSA+clash+NIS combinations as pose rankers.

NIS (non-interface surface composition) has within-target variant signal (~r=0.4)
but hasn't been tested as a per-pose ranker. This script tests whether blending
NIS into the BSA+clash score lifts τ or Hit@2Å.

Combinations tested (all z-normalized within complex, ascending = best first):
  bsa_only        -z(BSA)
  clash_only      +z(n_clash)
  nis_only        +z(nis_score)        [nis_score = charged_frac - polar_frac; lower=better]
  bsa_clash       -z(BSA) + z(n_clash)        [current production]
  bsa_nis         -z(BSA) + z(nis_score)
  bsa_clash_nis   -z(BSA) + z(n_clash) + z(nis_score)
  bsa_clash_nis05 -z(BSA) + z(n_clash) + 0.5*z(nis_score)
  bsa_clash_nis2  -z(BSA) + z(n_clash) + 2.0*z(nis_score)

Baseline: ref2015 physics (feats_gen_n100_physics.pkl col 0 = total_score, negated for ranking).

Metrics: mean per-complex Kendall τ (vs RMSD), Hit@1Å, Hit@2Å (top-1 and top-5).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BSA_PKL  = REPO / "logs/diagnosis/feats_gen_n100_bsa.pkl"
RMSD_PKL = REPO / "logs/diagnosis/gen_n100_rmsd_recomputed.pkl"
PHYS_PKL = REPO / "logs/diagnosis/feats_gen_n100_physics.pkl"
BM_JSON  = REPO / "logs/gen_n100/benchmark_results.json"
PEPPC    = REPO / "datasets/training_formatted_peppc"

CONTACT_CUT = 5.5   # Å heavy-atom cutoff for NIS contact definition
_CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
_POLAR   = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}


# ---------------------------------------------------------------------------
# PDB helpers
# ---------------------------------------------------------------------------

def _read_heavy(pdb: Path) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Return ([(resname, chain_resid), ...], xyz) for heavy atoms."""
    meta, xyz = [], []
    for ln in pdb.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        try:
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except ValueError:
            continue
        resname = ln[17:20].strip()
        chain_resid = ln[21] + ln[22:27].strip()
        meta.append((resname, chain_resid))
    return meta, (np.array(xyz, dtype=np.float32) if xyz else np.empty((0, 3), np.float32))


def compute_nis(pose_pdb: Path, receptor_pdb: Path) -> float | None:
    """Return nis_score = charged_frac - polar_frac of non-contacting residues."""
    pep_meta, pep_xyz = _read_heavy(pose_pdb)
    rec_meta, rec_xyz = _read_heavy(receptor_pdb)
    if len(pep_xyz) == 0 or len(rec_xyz) == 0:
        return None

    # group peptide atoms by residue
    res_map: dict[str, list[int]] = {}
    for i, (_, rid) in enumerate(pep_meta):
        res_map.setdefault(rid, []).append(i)

    res_ids  = list(res_map.keys())
    contacting = set()
    for rid, idx in res_map.items():
        pep_block = pep_xyz[idx]
        d = cdist(pep_block, rec_xyz).min()
        if d < CONTACT_CUT:
            contacting.add(rid)

    non_int = [rid for rid in res_ids if rid not in contacting]
    if not non_int:
        # all residues contact — NIS undefined
        return None

    # resname for first atom of each residue
    def resname_of(rid: str) -> str:
        idx0 = res_map[rid][0]
        return pep_meta[idx0][0].upper()

    n_charged = sum(1 for r in non_int if resname_of(r) in _CHARGED)
    n_polar   = sum(1 for r in non_int if resname_of(r) in _POLAR)
    n = len(non_int)
    return (n_charged - n_polar) / n


# ---------------------------------------------------------------------------
# Load caches
# ---------------------------------------------------------------------------

print("Loading caches...")
bsa_raw  = pickle.load(open(BSA_PKL, "rb"))
rmsd_raw = pickle.load(open(RMSD_PKL, "rb"))
phys_raw = pickle.load(open(PHYS_PKL, "rb"))
bm       = json.load(open(BM_JSON))

# remap BSA keys: (cx, pose) -> (cx, 'pretrained', pose)
bsa_map = {(k[0], "pretrained", k[1]): v for k, v in bsa_raw.items()}

# build per-complex pose lists
complexes: dict[str, list[dict]] = {}
for key, rmsd in rmsd_raw.items():
    cx, model, pose_idx = key
    if model != "pretrained":
        continue
    if key not in bsa_map:
        continue
    bsa_tuple = bsa_map[key]
    bsa_val   = float(bsa_tuple[0])
    clash_val = float(bsa_tuple[1])

    phys_vec = phys_raw.get(key)
    # col 13 = ref2015 interface score; already negative for good poses → ascending sort = best first
    ref2015  = float(phys_vec[13]) if phys_vec is not None else None

    complexes.setdefault(cx, []).append({
        "pose_idx": pose_idx,
        "rmsd":     float(rmsd),
        "bsa":      bsa_val,
        "n_clash":  clash_val,
        "ref2015":  ref2015,
        "nis":      None,   # filled below
    })

print(f"Complexes loaded: {len(complexes)}, total poses: {sum(len(v) for v in complexes.values())}")

# ---------------------------------------------------------------------------
# Compute NIS for all poses
# ---------------------------------------------------------------------------

print("Computing NIS features...")
nis_ok = nis_fail = 0
for cx, poses in complexes.items():
    rec_pdb = PEPPC / cx / f"{cx}_protein_pocket.pdb"
    if not rec_pdb.exists():
        nis_fail += len(poses)
        continue
    poses_dir = REPO / "logs/gen_n100" / cx / "poses"
    for p in poses:
        pose_pdb = poses_dir / f"pose_{p['pose_idx']}.pdb"
        if not pose_pdb.exists():
            nis_fail += 1
            continue
        nis_val = compute_nis(pose_pdb, rec_pdb)
        if nis_val is not None:
            p["nis"] = nis_val
            nis_ok += 1
        else:
            nis_fail += 1

print(f"NIS computed: {nis_ok} ok, {nis_fail} failed/skipped")


# ---------------------------------------------------------------------------
# Ranking evaluation
# ---------------------------------------------------------------------------

def z_norm(vals: list[float]) -> np.ndarray:
    a = np.array(vals, dtype=np.float64)
    sd = a.std()
    return (a - a.mean()) / (sd if sd > 1e-9 else 1.0)


COMBOS = {
    "bsa_only":        lambda b, c, n: -b,
    "clash_only":      lambda b, c, n: c,
    "nis_only":        lambda b, c, n: n,
    "bsa_clash":       lambda b, c, n: -b + c,
    "bsa_nis":         lambda b, c, n: -b + n,
    "bsa_clash_nis":   lambda b, c, n: -b + c + n,
    "bsa_clash_nis05": lambda b, c, n: -b + c + 0.5 * n,
    "bsa_clash_nis2":  lambda b, c, n: -b + c + 2.0 * n,
    "ref2015":         None,  # handled separately
}

results: dict[str, dict] = {name: {"taus": [], "hit1_t1": [], "hit2_t1": [], "hit2_t5": []} for name in COMBOS}

for cx, poses in complexes.items():
    # require at least 3 poses and NIS computed for all
    valid = [p for p in poses if p["nis"] is not None and p["ref2015"] is not None]
    if len(valid) < 3:
        continue

    rmsds = np.array([p["rmsd"] for p in valid])

    zb = z_norm([p["bsa"]     for p in valid])
    zc = z_norm([p["n_clash"] for p in valid])
    zn = z_norm([p["nis"]     for p in valid])
    r2 = np.array([p["ref2015"] for p in valid])

    for name, fn in COMBOS.items():
        if fn is not None:
            scores = fn(zb, zc, zn)
        else:
            scores = r2

        tau, _ = sp.kendalltau(scores, rmsds)
        if np.isnan(tau):
            continue

        order    = np.argsort(scores)
        top1_r   = rmsds[order[0]]
        top5_r   = rmsds[order[:5]].min() if len(order) >= 5 else top1_r

        results[name]["taus"].append(tau)
        results[name]["hit1_t1"].append(float(top1_r <= 1.0))
        results[name]["hit2_t1"].append(float(top1_r <= 2.0))
        results[name]["hit2_t5"].append(float(top5_r <= 2.0))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print(f"\n{'Combo':<20} {'τ mean':>8} {'τ med':>7} {'Hit@1Å t1':>10} {'Hit@2Å t1':>10} {'Hit@2Å t5':>10}  (n_cx)")
print("-" * 80)

# sort by mean τ descending
order = sorted(results.keys(), key=lambda k: -np.mean(results[k]["taus"]) if results[k]["taus"] else -99)
for name in order:
    r = results[name]
    if not r["taus"]:
        continue
    taus = np.array(r["taus"])
    n = len(taus)
    print(
        f"{name:<20} {taus.mean():>8.4f} {np.median(taus):>7.4f}"
        f" {np.mean(r['hit1_t1']):>10.1%}"
        f" {np.mean(r['hit2_t1']):>10.1%}"
        f" {np.mean(r['hit2_t5']):>10.1%}"
        f"  ({n})"
    )

print()
# Highlight best on each metric
best_tau  = max(results, key=lambda k: np.mean(results[k]["taus"]) if results[k]["taus"] else -99)
best_h2t1 = max(results, key=lambda k: np.mean(results[k]["hit2_t1"]) if results[k]["hit2_t1"] else -99)
best_h2t5 = max(results, key=lambda k: np.mean(results[k]["hit2_t5"]) if results[k]["hit2_t5"] else -99)
print(f"Best τ:        {best_tau}")
print(f"Best Hit@2Å t1: {best_h2t1}")
print(f"Best Hit@2Å t5: {best_h2t5}")
