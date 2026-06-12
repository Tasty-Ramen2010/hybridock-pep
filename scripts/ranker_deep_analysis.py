#!/usr/bin/env python3
"""
ranker_deep_analysis.py — Where is the missing τ signal?

Four parallel investigations:
  A. Interface-RMSD label: global Cα-RMSD includes floppy tails → noisy label.
     Recompute RMSD using only crystal-contacting residues. Does τ jump?
     This directly quantifies how much "missing 0.8" is label noise.

  B. Consensus stream: mean pairwise Cα-RMSD to rest of ensemble (free, no
     PyRosetta). Near-native poses cluster together → low consensus score.
     Orthogonal to burial (ρ=0.33 on bench300). Test as z-blend stream.

  C. Z-blend sweeps: ref2015 × bsa_clash_nis × consensus with weight grids.
     Find the combo that wins on BOTH τ AND Hit@2Å.

  D. Per-complex failure anatomy: for each complex, classify failure as
     (a) no near-native poses exist  → generator problem, not scorer
     (b) near-native exists but scores last → scorer problem
     (c) ranker is correct (τ > 0.2) → success cases
     Maps the 0.8 variance to its causes.
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

CONTACT_CUT  = 5.5   # Å cutoff for NIS / interface residue definition
_CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
_POLAR   = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}


# ─── PDB helpers ────────────────────────────────────────────────────────────

def read_heavy(pdb: Path) -> tuple[list[tuple[str, str, int]], np.ndarray]:
    """(meta[(resname, chain_resid, seq_pos)], xyz) for heavy atoms, chain order."""
    meta, xyz, res_order, res_idx = [], [], [], {}
    pos = 0
    for ln in pdb.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        try:
            c = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        rn   = ln[17:20].strip()
        rid  = ln[21] + ln[22:27].strip()
        if rid not in res_idx:
            res_idx[rid] = pos
            res_order.append(rid)
            pos += 1
        meta.append((rn, rid, res_idx[rid]))
        xyz.append(c)
    arr = np.array(xyz, dtype=np.float32) if xyz else np.empty((0, 3), np.float32)
    return meta, arr


def ca_coords_ordered(pdb: Path) -> np.ndarray | None:
    """Cα coordinates in chain residue order. None if < 3 residues."""
    ca: dict[str, tuple[int, list]] = {}
    order = []
    for ln in pdb.read_text().splitlines():
        if not ln.startswith("ATOM"):
            continue
        if ln[12:16].strip() != "CA":
            continue
        try:
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        rid = ln[21] + ln[22:27].strip()
        if rid not in ca:
            ca[rid] = (len(order), [])
            order.append(rid)
        ca[rid][1].append(xyz)
    if len(order) < 3:
        return None
    return np.array([ca[r][1][0] for r in order], dtype=np.float32)


def nis_score(pose_pdb: Path, rec_xyz: np.ndarray, pep_meta: list, pep_xyz: np.ndarray) -> float | None:
    """NIS score from pre-loaded arrays. nis_score = charged_frac - polar_frac."""
    if len(pep_xyz) == 0 or len(rec_xyz) == 0:
        return None
    res_map: dict[str, list[int]] = {}
    for i, (_, rid, _) in enumerate(pep_meta):
        res_map.setdefault(rid, []).append(i)
    res_ids = list(res_map.keys())
    contacting = set()
    for rid, idx in res_map.items():
        if cdist(pep_xyz[idx], rec_xyz).min() < CONTACT_CUT:
            contacting.add(rid)
    non_int = [r for r in res_ids if r not in contacting]
    if not non_int:
        return None
    def rname(rid):
        return pep_meta[res_map[rid][0]][0].upper()
    n = len(non_int)
    nc = sum(1 for r in non_int if rname(r) in _CHARGED)
    np_ = sum(1 for r in non_int if rname(r) in _POLAR)
    return (nc - np_) / n


def interface_rmsd(pose_ca: np.ndarray, crystal_ca: np.ndarray, iface_idx: list[int]) -> float | None:
    """RMSD over interface residues only (by sequential index)."""
    if not iface_idx:
        return None
    idx = [i for i in iface_idx if i < len(pose_ca) and i < len(crystal_ca)]
    if len(idx) < 2:
        return None
    d = pose_ca[idx] - crystal_ca[idx]
    return float(np.sqrt((d ** 2).sum(1).mean()))


def pairwise_ca_rmsd(ca_list: list[np.ndarray]) -> np.ndarray:
    """NxN pairwise Cα-RMSD matrix (common-length prefix only)."""
    n = len(ca_list)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            L = min(len(ca_list[i]), len(ca_list[j]))
            if L < 2:
                mat[i, j] = mat[j, i] = np.nan
                continue
            d = ca_list[i][:L] - ca_list[j][:L]
            r = float(np.sqrt((d ** 2).sum(1).mean()))
            mat[i, j] = mat[j, i] = r
    return mat


# ─── Load caches ─────────────────────────────────────────────────────────────

print("Loading caches …")
bsa_raw  = pickle.load(open(BSA_PKL, "rb"))
rmsd_raw = pickle.load(open(RMSD_PKL, "rb"))
phys_raw = pickle.load(open(PHYS_PKL, "rb"))
bm       = json.load(open(BM_JSON))

bsa_map = {(k[0], "pretrained", k[1]): v for k, v in bsa_raw.items()}

# ─── Per-complex data assembly ────────────────────────────────────────────────

print("Assembling per-complex pose data …")
complexes: dict[str, dict] = {}
for key, rmsd_val in rmsd_raw.items():
    cx, model, pose_idx = key
    if model != "pretrained" or key not in bsa_map:
        continue
    bt = bsa_map[key]
    pv = phys_raw.get(key)
    complexes.setdefault(cx, {"poses": []})["poses"].append({
        "idx":     pose_idx,
        "rmsd":    float(rmsd_val),
        "bsa":     float(bt[0]),
        "n_clash": float(bt[1]),
        "ref2015": float(pv[13]) if pv is not None else None,
        "nis":     None,
        "ca":      None,
        "irmsd":   None,
    })

print(f"  {len(complexes)} complexes, {sum(len(v['poses']) for v in complexes.values())} poses")

# ─── Feature computation ──────────────────────────────────────────────────────

print("Computing NIS, Cα coords, interface-RMSD …")
iface_ok = nis_ok = ca_ok = 0

for cx, data in complexes.items():
    rec_pdb     = PEPPC / cx / f"{cx}_protein_pocket.pdb"
    crystal_pdb = PEPPC / cx / f"{cx}_peptide.pdb"
    poses_dir   = REPO / "logs/gen_n100" / cx / "poses"

    if not rec_pdb.exists() or not crystal_pdb.exists():
        continue

    rec_meta, rec_xyz = read_heavy(rec_pdb)
    crystal_ca = ca_coords_ordered(crystal_pdb)
    if crystal_ca is None:
        continue

    # identify crystal interface residue indices (by seq position)
    cryst_meta, cryst_xyz = read_heavy(crystal_pdb)
    cryst_res: dict[str, list[int]] = {}
    for i, (_, rid, _) in enumerate(cryst_meta):
        cryst_res.setdefault(rid, []).append(i)
    cryst_res_order = list(dict.fromkeys(m[1] for m in cryst_meta))
    iface_idx = []
    for seq_i, rid in enumerate(cryst_res_order):
        atoms = cryst_xyz[cryst_res[rid]]
        if len(rec_xyz) > 0 and cdist(atoms, rec_xyz).min() < CONTACT_CUT:
            iface_idx.append(seq_i)
    data["iface_idx"] = iface_idx
    data["crystal_ca"] = crystal_ca
    iface_ok += 1

    for p in data["poses"]:
        pose_pdb = poses_dir / f"pose_{p['idx']}.pdb"
        if not pose_pdb.exists():
            continue
        pep_meta, pep_xyz = read_heavy(pose_pdb)
        pose_ca = ca_coords_ordered(pose_pdb)
        if pose_ca is not None:
            p["ca"] = pose_ca
            ca_ok += 1
        n = nis_score(pose_pdb, rec_xyz, pep_meta, pep_xyz)
        if n is not None:
            p["nis"] = n
            nis_ok += 1
        if pose_ca is not None and iface_idx:
            ir = interface_rmsd(pose_ca, crystal_ca, iface_idx)
            if ir is not None:
                p["irmsd"] = ir

print(f"  iface complexes: {iface_ok}, Cα: {ca_ok}, NIS: {nis_ok}")

# ─── Consensus scoring ────────────────────────────────────────────────────────

print("Computing pairwise consensus …")
consensus_ok = 0
for cx, data in complexes.items():
    valid = [p for p in data["poses"] if p["ca"] is not None]
    if len(valid) < 5:
        continue
    ca_list = [p["ca"] for p in valid]
    mat = pairwise_ca_rmsd(ca_list)
    means = np.nanmean(mat, axis=1)
    for p, m in zip(valid, means):
        p["consensus"] = float(m)
        consensus_ok += 1

print(f"  consensus computed for {consensus_ok} poses")

# ─── Ranking evaluation engine ────────────────────────────────────────────────

def z_norm(vals: list[float]) -> np.ndarray:
    a = np.array(vals, dtype=np.float64)
    sd = a.std()
    return (a - a.mean()) / (sd if sd > 1e-9 else 1.0)


def eval_complex(poses: list[dict], score_arr: np.ndarray,
                 rmsd_key: str = "rmsd") -> dict | None:
    rmsds = np.array([p[rmsd_key] for p in poses])
    if np.isnan(rmsds).any() or len(rmsds) < 3:
        return None
    tau, _ = sp.kendalltau(score_arr, rmsds)
    if np.isnan(tau):
        return None
    order = np.argsort(score_arr)
    t1r   = rmsds[order[0]]
    t5r   = rmsds[order[:5]].min() if len(order) >= 5 else t1r
    best  = rmsds.min()
    return {"tau": tau, "hit1_t1": t1r <= 1.0, "hit2_t1": t1r <= 2.0,
            "hit2_t5": t5r <= 2.0, "best_rmsd": best, "n": len(rmsds)}


def aggregate(rows: list[dict]) -> dict:
    taus = np.array([r["tau"] for r in rows])
    return {
        "n_cx":    len(rows),
        "tau_mean": taus.mean(),
        "tau_med":  float(np.median(taus)),
        "hit1_t1":  np.mean([r["hit1_t1"] for r in rows]),
        "hit2_t1":  np.mean([r["hit2_t1"] for r in rows]),
        "hit2_t5":  np.mean([r["hit2_t5"] for r in rows]),
    }


# ─── Part A: global-RMSD vs interface-RMSD label ─────────────────────────────

print("\n══════════════════════════════════════════════════════")
print("A. LABEL NOISE — global-RMSD vs interface-RMSD")
print("══════════════════════════════════════════════════════")

global_rows, iface_rows = [], []
for cx, data in complexes.items():
    poses = [p for p in data["poses"]
             if p["ref2015"] is not None and p["nis"] is not None
             and p.get("consensus") is not None]
    if len(poses) < 5:
        continue
    scores = np.array([p["ref2015"] for p in poses])

    r_global = eval_complex(poses, scores, "rmsd")
    if r_global:
        r_global["cx"] = cx
        global_rows.append(r_global)

    if all(p.get("irmsd") is not None for p in poses):
        r_iface = eval_complex(poses, scores, "irmsd")
        if r_iface:
            iface_rows.append(r_iface)

g = aggregate(global_rows)
i = aggregate(iface_rows) if iface_rows else None

print(f"\nref2015 vs global-RMSD  : τ={g['tau_mean']:.4f}  Hit@2Å-t5={g['hit2_t5']:.1%}  (n={g['n_cx']})")
if i:
    print(f"ref2015 vs iface-RMSD   : τ={i['tau_mean']:.4f}  Hit@2Å-t5={i['hit2_t5']:.1%}  (n={i['n_cx']})")
    delta = i["tau_mean"] - g["tau_mean"]
    print(f"  Δτ from label fix     : {delta:+.4f}  ← fraction of 'missing' τ from tail noise")
else:
    print("  interface-RMSD: insufficient coverage")

# near-native stats
all_rmsds = [p["rmsd"] for data in complexes.values() for p in data["poses"]]
nn_rate = np.mean(np.array(all_rmsds) <= 2.0)
print(f"\nNear-native rate (≤2Å)  : {nn_rate:.1%} of {len(all_rmsds)} poses")
print(f"Oracle Hit@2Å (any pose): {np.mean([any(p['rmsd']<=2.0 for p in data['poses']) for data in complexes.values()]):.1%} of complexes have ≥1 near-native pose")

# ─── Part B: Consensus signal ─────────────────────────────────────────────────

print("\n══════════════════════════════════════════════════════")
print("B. CONSENSUS STREAM — pairwise Cα-RMSD to ensemble")
print("══════════════════════════════════════════════════════")

COMBOS_BASE = {
    "ref2015":             lambda b, c, n, cs: b,
    "bsa_clash":           lambda b, c, n, cs: -z_norm(list(c)) if False else None,  # placeholder
    "consensus":           lambda b, c, n, cs: cs,
}

# manual z-blend evaluation (cleaner than lambda over pre-normed arrays)
combo_results: dict[str, list[dict]] = {}
COMBO_NAMES = [
    "ref2015", "bsa_clash", "nis_only", "consensus",
    "bsa_clash_nis", "bsa_clash_nis05",
    "ref2015+consensus_50", "ref2015+bsa_clash_nis_50",
    "ref2015+bsa_clash_nis_33+consensus_33",
    "bsa_clash_nis+consensus",
]
for n in COMBO_NAMES:
    combo_results[n] = []

for cx, data in complexes.items():
    poses = [p for p in data["poses"]
             if p["ref2015"] is not None
             and p["nis"] is not None
             and p.get("consensus") is not None]
    if len(poses) < 5:
        continue

    bsa   = z_norm([p["bsa"]       for p in poses])
    clash = z_norm([p["n_clash"]   for p in poses])
    nis   = z_norm([p["nis"]       for p in poses])
    ref   = np.array([p["ref2015"] for p in poses])
    zref  = z_norm(ref.tolist())
    cons  = z_norm([p["consensus"] for p in poses])

    scores_map = {
        "ref2015":                          ref,
        "bsa_clash":                        -bsa + clash,
        "nis_only":                         nis,
        "consensus":                        cons,
        "bsa_clash_nis":                    -bsa + clash + nis,
        "bsa_clash_nis05":                  -bsa + clash + 0.5 * nis,
        "ref2015+consensus_50":             zref + cons,
        "ref2015+bsa_clash_nis_50":         zref + (-bsa + clash + nis),
        "ref2015+bsa_clash_nis_33+consensus_33": zref + 0.33*(-bsa+clash+nis) + 0.33*cons,
        "bsa_clash_nis+consensus":          -bsa + clash + nis + cons,
    }

    for name, scores in scores_map.items():
        r = eval_complex(poses, scores, "rmsd")
        if r:
            combo_results[name].append(r)

print(f"\n{'Combo':<40} {'τ mean':>8} {'τ med':>7} {'Hit@1Å':>7} {'Hit@2Å t1':>10} {'Hit@2Å t5':>10}  n")
print("-" * 100)
ordered = sorted(combo_results.keys(),
                 key=lambda k: -np.mean([r["tau"] for r in combo_results[k]]) if combo_results[k] else 99)
for name in ordered:
    rows = combo_results[name]
    if not rows:
        continue
    taus = np.array([r["tau"] for r in rows])
    print(f"{name:<40} {taus.mean():>8.4f} {np.median(taus):>7.4f}"
          f" {np.mean([r['hit1_t1'] for r in rows]):>7.1%}"
          f" {np.mean([r['hit2_t1'] for r in rows]):>10.1%}"
          f" {np.mean([r['hit2_t5'] for r in rows]):>10.1%}"
          f"  {len(rows)}")

# ─── Part C: Weight sweep for best of both worlds ────────────────────────────

print("\n══════════════════════════════════════════════════════")
print("C. WEIGHT SWEEP — ref2015 × (bsa_clash_nis) × consensus")
print("   goal: max τ without losing Hit@2Å t5")
print("══════════════════════════════════════════════════════")

# baseline ref2015
ref_rows = combo_results["ref2015"]
base_tau  = np.mean([r["tau"] for r in ref_rows])
base_h2t5 = np.mean([r["hit2_t5"] for r in ref_rows])
print(f"\nBaseline (ref2015): τ={base_tau:.4f}  Hit@2Å-t5={base_h2t5:.1%}")

best_balanced = {"score": -99, "w_r": 0, "w_b": 0, "w_c": 0, "tau": 0, "h2t5": 0}
sweep_rows = []
for w_r in np.arange(0.0, 1.01, 0.2):
    for w_b in np.arange(0.0, 1.01 - w_r, 0.2):
        w_c = round(1.0 - w_r - w_b, 2)
        if w_c < -0.01:
            continue
        rows = []
        for cx, data in complexes.items():
            poses = [p for p in data["poses"]
                     if p["ref2015"] is not None
                     and p["nis"] is not None
                     and p.get("consensus") is not None]
            if len(poses) < 5:
                continue
            bsa   = z_norm([p["bsa"]       for p in poses])
            clash = z_norm([p["n_clash"]   for p in poses])
            nis   = z_norm([p["nis"]       for p in poses])
            zref  = z_norm([p["ref2015"]   for p in poses])
            cons  = z_norm([p["consensus"] for p in poses])
            bsa_nis_clash = -bsa + clash + nis
            scores = w_r * zref + w_b * bsa_nis_clash + w_c * cons
            r = eval_complex(poses, scores, "rmsd")
            if r:
                rows.append(r)
        if not rows:
            continue
        tau   = np.mean([r["tau"] for r in rows])
        h2t5  = np.mean([r["hit2_t5"] for r in rows])
        h2t1  = np.mean([r["hit2_t1"] for r in rows])
        # balanced score: mean of (τ rank + h2t5 rank) — we track best by both
        balanced = tau + h2t5  # simple sum; maximise both
        sweep_rows.append((w_r, w_b, w_c, tau, h2t5, h2t1, len(rows)))
        if balanced > best_balanced["score"]:
            best_balanced = {"score": balanced, "w_r": w_r, "w_b": w_b,
                             "w_c": w_c, "tau": tau, "h2t5": h2t5, "h2t1": h2t1}

# print top-10 by balanced score
sweep_rows.sort(key=lambda x: -(x[3] + x[4]))
print(f"\n{'w_ref':>6} {'w_bsanis':>9} {'w_cons':>7} {'τ':>8} {'H@2t1':>7} {'H@2t5':>7}  n")
for row in sweep_rows[:12]:
    w_r, w_b, w_c, tau, h2t5, h2t1, n = row
    print(f"{w_r:>6.1f} {w_b:>9.1f} {w_c:>7.1f} {tau:>8.4f} {h2t1:>7.1%} {h2t5:>7.1%}  {n}")

bb = best_balanced
print(f"\nBest balanced (max τ+Hit@2Å-t5): w_ref={bb['w_r']:.1f}  w_bsa_nis={bb['w_b']:.1f}  w_cons={bb['w_c']:.1f}")
print(f"  τ={bb['tau']:.4f}  Hit@2Å-t1={bb['h2t1']:.1%}  Hit@2Å-t5={bb['h2t5']:.1%}")

# ─── Part D: Per-complex failure anatomy ─────────────────────────────────────

print("\n══════════════════════════════════════════════════════")
print("D. WHERE IS THE 0.8 GOING — per-complex anatomy")
print("══════════════════════════════════════════════════════")

# Use ref2015 (best single feature) for this analysis
failures_no_nn  = []  # no pose ≤2Å — generator problem
failures_scorer = []  # has near-native but ranker τ < 0 — scorer problem
successes       = []  # τ > 0.2

ref_rows_cx = {r["cx"]: r for r in global_rows if "cx" in r}

for cx, data in complexes.items():
    poses = [p for p in data["poses"]
             if p["ref2015"] is not None and p["nis"] is not None
             and p.get("consensus") is not None]
    if len(poses) < 5:
        continue
    rmsds = np.array([p["rmsd"] for p in poses])
    has_nn = rmsds.min() <= 2.0
    nn_rate_cx = (rmsds <= 2.0).mean()

    scores = np.array([p["ref2015"] for p in poses])
    r = eval_complex(poses, scores, "rmsd")
    tau = r["tau"] if r else 0.0

    rec = {"cx": cx, "tau": tau, "has_nn": has_nn, "nn_rate": nn_rate_cx,
           "best_rmsd": rmsds.min(), "n_poses": len(poses),
           "n_nn": int((rmsds <= 2.0).sum())}

    if not has_nn:
        failures_no_nn.append(rec)
    elif tau < 0.0:
        failures_scorer.append(rec)
    elif tau > 0.2:
        successes.append(rec)

print(f"\nComplexes with NO near-native pose (≤2Å):   {len(failures_no_nn):3d}")
print(f"  → generator problem; scorer irrelevant")
print(f"  mean best_rmsd: {np.mean([r['best_rmsd'] for r in failures_no_nn]):.2f} Å")

print(f"\nComplexes with near-native BUT τ < 0:        {len(failures_scorer):3d}")
print(f"  → scorer ANTI-correlates; near-native scored last")
print(f"  mean nn_rate: {np.mean([r['nn_rate'] for r in failures_scorer]):.1%}")
print(f"  mean n_nn_poses: {np.mean([r['n_nn'] for r in failures_scorer]):.1f}")

print(f"\nSuccess (has near-native AND τ > 0.2):       {len(successes):3d}")
print(f"  mean τ: {np.mean([r['tau'] for r in successes]):.3f}")
print(f"  mean nn_rate: {np.mean([r['nn_rate'] for r in successes]):.1%}")

# middle ground
others = [r for cx, data in complexes.items()
          for r in [{"cx": cx,
                     "has_nn": any(p["rmsd"]<=2.0 for p in data["poses"]
                                   if p["ref2015"] is not None and p.get("nis") is not None
                                   and p.get("consensus") is not None)}]
          if r["cx"] not in {x["cx"] for x in failures_no_nn + failures_scorer + successes}]

print(f"\nMiddle (has near-native, 0 ≤ τ ≤ 0.2):      {len(others):3d}")

# Variance decomposition summary
total_cx = len(failures_no_nn) + len(failures_scorer) + len(successes) + len(others)
print(f"\n─── VARIANCE DECOMPOSITION (of the missing τ) ───")
print(f"Total evaluated complexes: {total_cx}")
print(f"  {len(failures_no_nn)/total_cx:.0%} → generator problem (no near-native to rank)")
print(f"  {len(failures_scorer)/total_cx:.0%} → scorer actively wrong (anti-correlated)")
print(f"  {len(successes)/total_cx:.0%} → τ > 0.2 (working well)")
print(f"  {len(others)/total_cx:.0%} → weak positive signal (τ 0-0.2)")

# Feature coverage for the scorer-failure group
if failures_scorer:
    print(f"\nScorer-failure cases (anti-corr complexes):")
    for r in sorted(failures_scorer, key=lambda x: x["tau"])[:8]:
        print(f"  {r['cx']:<35} τ={r['tau']:+.3f}  best={r['best_rmsd']:.2f}Å  n_nn={r['n_nn']}")

# ─── Summary ─────────────────────────────────────────────────────────────────

print("\n══════════════════════════════════════════════════════")
print("SUMMARY")
print("══════════════════════════════════════════════════════")

best_tau_combo  = max(combo_results, key=lambda k: np.mean([r["tau"] for r in combo_results[k]]) if combo_results[k] else -99)
best_h2t5_combo = max(combo_results, key=lambda k: np.mean([r["hit2_t5"] for r in combo_results[k]]) if combo_results[k] else -99)
best_h2t1_combo = max(combo_results, key=lambda k: np.mean([r["hit2_t1"] for r in combo_results[k]]) if combo_results[k] else -99)

print(f"Best τ:         {best_tau_combo}  τ={np.mean([r['tau'] for r in combo_results[best_tau_combo]]):.4f}")
print(f"Best Hit@2Å t5: {best_h2t5_combo}  {np.mean([r['hit2_t5'] for r in combo_results[best_h2t5_combo]]):.1%}")
print(f"Best Hit@2Å t1: {best_h2t1_combo}  {np.mean([r['hit2_t1'] for r in combo_results[best_h2t1_combo]]):.1%}")
