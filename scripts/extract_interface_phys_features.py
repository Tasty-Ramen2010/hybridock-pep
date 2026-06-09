#!/usr/bin/env python3
"""
extract_interface_phys_features.py — Two routes to interface-specific physics features.

Route A: per-residue energies summed over interface peptide residues only.
    Contact = any peptide heavy atom within CONTACT_A Å of any receptor heavy atom
    (in the POSED complex, not the crystal). Uses Rosetta per-residue energy decomp.

Route B: cross-chain pairwise interaction energies via Rosetta EnergyGraph.
    Sums only two-body interactions between peptide residues and receptor residues.
    This captures pure interface interaction energy, no peptide self-energy.

Output: logs/diagnosis/feats_bench300_interface_phys.pkl
    key: (complex_name, model_key, pose_idx)
    value: {"route_a": np.array(12,), "route_b": np.array(12,)} or None on failure

Same 12 energy terms as SCORE_TERMS (no interface_ddG/total_score/resp_* —
those are global by definition). Features are raw, unscaled.

Runtime: ~25 min on CPU (0.6s/pose × 2300 poses).

Usage:
    conda run -n score-env python3 scripts/extract_interface_phys_features.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BJSON = json.load(open(REPO / "logs" / "analysis_bench300" / "benchmark_results.json"))
PHYS_PKL = REPO / "logs" / "diagnosis" / "feats_bench300_physics.pkl"
OUT_PKL = REPO / "logs" / "diagnosis" / "feats_bench300_interface_phys.pkl"
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")

CONTACT_A = 4.5      # Å cutoff for Route A contact detection (slightly looser)
SCORE_TERMS = [
    "fa_atr", "fa_rep", "fa_sol", "fa_intra_rep", "fa_elec",
    "hbond_bb_sc", "hbond_sc", "hbond_lr_bb", "hbond_sr_bb",
    "rama_prepro", "fa_dun", "p_aa_pp",
]


def _count_ca(pdb_path: str) -> int:
    return sum(1 for l in open(pdb_path)
               if l.startswith("ATOM") and l[12:16].strip() == "CA")


def _receptor_heavy_coords(pose, n_rec: int) -> np.ndarray:
    """(M,3) array of non-H receptor heavy atom coords."""
    pts = []
    for ri in range(1, n_rec + 1):
        res = pose.residue(ri)
        for ai in range(1, res.natoms() + 1):
            if not res.atom_is_hydrogen(ai):
                xyz = res.xyz(ai)
                pts.append([xyz.x, xyz.y, xyz.z])
    return np.array(pts) if pts else np.zeros((0, 3))


def _peptide_heavy_by_residue(pose, n_rec: int, n_total: int) -> dict[int, np.ndarray]:
    """{pose_residue_idx: (k,3)} heavy atom coords per peptide residue."""
    out = {}
    for pi in range(n_rec + 1, n_total + 1):
        res = pose.residue(pi)
        pts = []
        for ai in range(1, res.natoms() + 1):
            if not res.atom_is_hydrogen(ai):
                xyz = res.xyz(ai)
                pts.append([xyz.x, xyz.y, xyz.z])
        if pts:
            out[pi] = np.array(pts)
    return out


def score_pose(pose_pdb: str, rec_pdb: str, pyrosetta, sfxn,
               ScoreType, rec_cache: dict) -> dict | None:
    """Score one pose and return Route A + B interface feature vectors."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
            tmp.write(open(rec_pdb).read().rstrip())
            tmp.write("\nTER\n")
            tmp.write(open(pose_pdb).read())
            tmp_path = tmp.name

        pose = pyrosetta.pose_from_pdb(tmp_path)
        os.unlink(tmp_path)
        sfxn(pose)

        # receptor residue count from cache or count CA atoms
        if rec_pdb not in rec_cache:
            rec_pose = pyrosetta.pose_from_pdb(rec_pdb)
            rec_cache[rec_pdb] = rec_pose.total_residue()
        n_rec = rec_cache[rec_pdb]
        n_total = pose.total_residue()

        if n_total <= n_rec:
            return None  # no peptide residues

        # ── Route A: per-residue energies filtered to interface residues ──────
        rec_heavy = _receptor_heavy_coords(pose, n_rec)
        pep_by_res = _peptide_heavy_by_residue(pose, n_rec, n_total)

        # identify contact residues in POSED structure
        contact_res = set()
        if rec_heavy.shape[0] > 0:
            for pi, heavy in pep_by_res.items():
                # min distance between this peptide residue and all receptor atoms
                diffs = rec_heavy[None, :, :] - heavy[:, None, :]  # (k,M,3)
                dists = np.sqrt((diffs ** 2).sum(-1))
                if dists.min() < CONTACT_A:
                    contact_res.add(pi)
        if len(contact_res) < 3:
            contact_res = set(range(n_rec + 1, n_total + 1))  # fallback: all

        route_a = np.zeros(len(SCORE_TERMS), dtype=np.float32)
        energies = pose.energies()
        for pi in contact_res:
            re = energies.residue_total_energies(pi)
            for ti, term in enumerate(SCORE_TERMS):
                try:
                    route_a[ti] += re[getattr(ScoreType, term)]
                except Exception:
                    pass

        # ── Route B: cross-chain pairwise interaction energies ────────────────
        route_b = np.zeros(len(SCORE_TERMS), dtype=np.float32)
        eg = energies.energy_graph()
        for pi in range(n_rec + 1, n_total + 1):
            for ri in range(1, n_rec + 1):
                edge = eg.find_energy_edge(pi, ri)
                if edge is not None:
                    em = edge.fill_energy_map()
                    for ti, term in enumerate(SCORE_TERMS):
                        try:
                            route_b[ti] += em[getattr(ScoreType, term)]
                        except Exception:
                            pass

        return {"route_a": route_a, "route_b": route_b,
                "n_contact": len(contact_res), "n_total_pep": n_total - n_rec}

    except Exception as exc:
        print(f"  FAIL {pose_pdb}: {exc}", flush=True)
        return None


def main():
    import pyrosetta
    pyrosetta.init("-mute all -ex1 -ex2aro", silent=True)
    from pyrosetta.rosetta.core.scoring import get_score_function, ScoreType
    sfxn = get_score_function(True)

    phys = pickle.load(open(PHYS_PKL, "rb"))
    keys = sorted(phys.keys())
    print(f"{len(keys)} poses to process", flush=True)

    rec_cache: dict[str, int] = {}
    results: dict = {}
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    for i, k in enumerate(keys):
        cn, mk, pi = k
        entry = BJSON.get(cn, {}).get(mk, {})
        poses_dir = entry.get("poses_dir", "")
        pose_pdb = str(Path(poses_dir) / f"pose_{pi}.pdb")
        rec_pdb = str(BASE / cn / f"{cn}_protein_pocket.pdb")

        if not Path(pose_pdb).exists() or not Path(rec_pdb).exists():
            n_fail += 1
            results[k] = None
            continue

        r = score_pose(pose_pdb, rec_pdb, pyrosetta, sfxn, ScoreType, rec_cache)
        results[k] = r
        if r is not None:
            n_ok += 1
        else:
            n_fail += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(keys) - i - 1)
            print(f"  {i+1}/{len(keys)}  ok={n_ok} fail={n_fail}  "
                  f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)

    print(f"\nDone. {n_ok} ok, {n_fail} failed. Total: {(time.time()-t0)/60:.1f} min")
    pickle.dump(results, open(OUT_PKL, "wb"), protocol=4)
    print(f"Saved → {OUT_PKL}")


if __name__ == "__main__":
    main()
