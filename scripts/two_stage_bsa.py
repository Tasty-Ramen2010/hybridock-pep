#!/usr/bin/env python3
"""
two_stage_bsa.py — Ram's two-stage idea: condition on the groove, then BSA the tail.

Reweighting BSA toward the tail across all 100 poses FAILED (glob_tail τ=0.089 ≈
global 0.100) because core-mode variance dominates. Fix: FIRST restrict to poses
that agree on the core (same groove), THEN BSA+clash varies (mostly) on the tail.

Two claims tested:
  (1) MECHANISM: is τ(BSA, RMSD) sharper WITHIN a core-consistent subset
      (top-10 by ref2015, or largest Cα-RMSD cluster) than across all 100?
  (2) END-TO-END: does two-stage (coarse core-finder → BSA tail-refine) beat
      single-stage top-1 Hit@2Å?

Poses are all in the receptor frame (docking output), so pose-pose Cα RMSD needs
NO superposition. ref2015 total_score from feats_gen_n100_physics.pkl[...,13].
BSA+clash via Shrake-Rupley (reused from bsa_tail_test).

Run (rapidock/score-env, needs Biopython + sklearn):
  python3 scripts/two_stage_bsa.py
"""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy import stats as sp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.bsa_tail_test import read_heavy, per_atom_sasa, CLASH_DIST, CROP  # reuse

BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")
GEN_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
ENC_PKL  = REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"
PHYS_PKL = REPO / "logs" / "diagnosis" / "feats_gen_n100_physics.pkl"
OUT = REPO / "logs" / "training_campaign" / "two_stage_bsa.json"

TOPK = 10
CLUSTER_THRESH = 4.0   # Å Cα-RMSD agglomerative distance threshold


def read_ca(pdb: str) -> np.ndarray:
    xyz = []
    for ln in open(pdb):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            try:
                xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            except ValueError:
                pass
    return np.array(xyz, np.float32) if xyz else np.empty((0, 3), np.float32)


def bsa_clash(pep_pdb: str, rec_lines, rec_xyz):
    pep_lines, pep_xyz, _ = read_heavy(pep_pdb)
    if len(pep_xyz) < 4 or len(rec_xyz) < 4:
        return None
    d2 = ((rec_xyz[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1)
    near = d2.min(1) <= CROP ** 2
    crop = [rec_lines[i] for i in np.where(near)[0]]
    s_free = per_atom_sasa(pep_lines)
    s_cx = per_atom_sasa(pep_lines + crop)[:len(pep_lines)]
    if len(s_free) != len(pep_lines) or len(s_cx) != len(pep_lines):
        return None
    bsa = float(np.maximum(s_free - s_cx, 0.0).sum())
    pd2 = ((pep_xyz[:, None, :] - rec_xyz[None, :, :]) ** 2).sum(-1)
    n_clash = float((pd2.min(1) < CLASH_DIST ** 2).sum())
    return bsa, n_clash


def pairwise_ca_rmsd(cas: list[np.ndarray]) -> np.ndarray:
    """No-superposition pose-pose Cα RMSD (same frame). [N,N]."""
    n = len(cas)
    L = min(len(c) for c in cas)
    A = np.stack([c[:L] for c in cas])      # [N, L, 3]
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        diff = A - A[i]                     # [N, L, 3]
        D[i] = np.sqrt((diff ** 2).sum(-1).mean(-1))
    return D


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def main():
    from sklearn.cluster import AgglomerativeClustering
    bjson = json.load(open(GEN_JSON))
    phys = pickle.load(open(PHYS_PKL, "rb"))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))

    # claim 1: conditional τ
    tau_all, tau_top10, tau_clust = [], [], []
    # claim 2: top-1 Hit@2Å for pipelines
    pipes = ["ref2015_only", "bsa_only",
             "ref10_then_bsa", "cluster_then_bsa", "ref10_then_glob"]
    hit1 = {p: [] for p in pipes}
    hit3 = {p: [] for p in pipes}
    oracle = []
    # how often does the stage-1 subset even contain a near-native?
    near_in_top10 = []
    t0 = time.time()
    n_done = 0

    for ci, cn in enumerate(cxs):
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        pdir = Path(entry.get("poses_dir", ""))
        rec_pdb = BASE / cn / f"{cn}_protein_pocket.pdb"
        if not rec_pdb.exists() or len(rr) < 10:
            continue
        rec_lines, rec_xyz, _ = read_heavy(str(rec_pdb))

        bsas, clashes, refs, cas, rmsds = [], [], [], [], []
        for pi in range(len(rr)):
            pp = pdir / f"pose_{pi}.pdb"
            pv = phys.get((cn, "pretrained", pi))
            if not pp.exists() or pv is None:
                continue
            bc = bsa_clash(str(pp), rec_lines, rec_xyz)
            ca = read_ca(str(pp))
            if bc is None or len(ca) < 2:
                continue
            bsas.append(bc[0]); clashes.append(bc[1])
            refs.append(float(pv[13])); cas.append(ca); rmsds.append(rr[pi])
        if len(rmsds) < 10:
            continue
        bsas = np.array(bsas); clashes = np.array(clashes)
        refs = np.array(refs); rmsds = np.array(rmsds)
        N = len(rmsds)

        # BSA+clash score (lower = better)
        bsa_score = -_z(bsas) + _z(clashes)
        ref_score = refs                       # ref2015 total, lower = better

        # ── claim 1: conditional discrimination of BSA ──────────────────
        t, _ = sp.kendalltau(-bsa_score, -rmsds)
        if not np.isnan(t): tau_all.append(t)

        top10 = np.argsort(ref_score)[:TOPK]
        if len(top10) >= 4:
            t, _ = sp.kendalltau(-bsa_score[top10], -rmsds[top10])
            if not np.isnan(t): tau_top10.append(t)
            near_in_top10.append(float(rmsds[top10].min() <= 2.0))

        # largest Cα-RMSD cluster (consensus groove)
        D = pairwise_ca_rmsd(cas)
        cl = AgglomerativeClustering(
            n_clusters=None, distance_threshold=CLUSTER_THRESH,
            metric="precomputed", linkage="average").fit_predict(D)
        labels, counts = np.unique(cl, return_counts=True)
        big = labels[counts.argmax()]
        cmask = np.where(cl == big)[0]
        if len(cmask) >= 4:
            t, _ = sp.kendalltau(-bsa_score[cmask], -rmsds[cmask])
            if not np.isnan(t): tau_clust.append(t)

        # ── claim 2: end-to-end top-1/top-3 ─────────────────────────────
        def h(idxs_sorted):
            return (float(rmsds[idxs_sorted[0]] <= 2.0),
                    float(rmsds[idxs_sorted[:3]].min() <= 2.0))

        h1, h3 = h(np.argsort(ref_score));  hit1["ref2015_only"].append(h1); hit3["ref2015_only"].append(h3)
        h1, h3 = h(np.argsort(bsa_score));   hit1["bsa_only"].append(h1);     hit3["bsa_only"].append(h3)
        # ref top-10 → re-rank by BSA
        sub = top10[np.argsort(bsa_score[top10])]
        h1, h3 = h(sub);                     hit1["ref10_then_bsa"].append(h1); hit3["ref10_then_bsa"].append(h3)
        # ref top-10 → keep ref order (control: does BSA re-rank add anything?)
        sub2 = top10[np.argsort(ref_score[top10])]
        h1, h3 = h(sub2);                    hit1["ref10_then_glob"].append(h1); hit3["ref10_then_glob"].append(h3)
        # cluster → re-rank by BSA
        subc = cmask[np.argsort(bsa_score[cmask])]
        h1, h3 = h(subc);                    hit1["cluster_then_bsa"].append(h1); hit3["cluster_then_bsa"].append(h3)

        oracle.append(float(rmsds.min() <= 2.0))
        n_done += 1
        if n_done % 5 == 0:
            print(f"  {n_done} cx  {time.time()-t0:.0f}s  "
                  f"τ: all={np.mean(tau_all):+.3f} top10={np.mean(tau_top10):+.3f} "
                  f"clust={np.mean(tau_clust):+.3f}", flush=True)

    print(f"\n{'='*66}")
    print(f"TWO-STAGE BSA  ({n_done} complexes, {time.time()-t0:.0f}s)")
    print(f"{'='*66}")
    print(f"\nCLAIM 1 — is BSA a sharper discriminator within a core-consistent subset?")
    print(f"  τ(BSA, RMSD) across all 100 :  {np.mean(tau_all):+.4f} ± {np.std(tau_all):.3f}")
    print(f"  τ(BSA, RMSD) within ref top-10: {np.mean(tau_top10):+.4f} ± {np.std(tau_top10):.3f}")
    print(f"  τ(BSA, RMSD) within Cα cluster: {np.mean(tau_clust):+.4f} ± {np.std(tau_clust):.3f}")
    d10 = np.mean(tau_top10) - np.mean(tau_all)
    dcl = np.mean(tau_clust) - np.mean(tau_all)
    print(f"  → top-10 Δ={d10:+.4f}   cluster Δ={dcl:+.4f}")
    print(f"  (stage-1 top-10 contains a near-native in {100*np.mean(near_in_top10):.0f}% of complexes)")

    print(f"\nCLAIM 2 — end-to-end top-1 / top-3 Hit@2Å (oracle = {100*np.mean(oracle):.1f}%):")
    print(f"  {'pipeline':<20} {'top-1':>8} {'top-3':>8}")
    print(f"  {'-'*38}")
    for p in pipes:
        print(f"  {p:<20} {100*np.mean(hit1[p]):>7.1f}% {100*np.mean(hit3[p]):>7.1f}%")

    OUT.write_text(json.dumps({
        "tau_all": float(np.mean(tau_all)),
        "tau_top10": float(np.mean(tau_top10)),
        "tau_cluster": float(np.mean(tau_clust)),
        "near_in_top10": float(np.mean(near_in_top10)),
        "hit1": {p: float(np.mean(hit1[p])) for p in pipes},
        "hit3": {p: float(np.mean(hit3[p])) for p in pipes},
        "oracle": float(np.mean(oracle)), "n": n_done,
    }, indent=2))
    print(f"\n  Saved → {OUT}")


if __name__ == "__main__":
    main()
