#!/usr/bin/env python3
"""
extract_interface_rmsds.py — Compute interface-RMSD labels for all bench300 poses.

For each complex, identifies which peptide residues contact the receptor
in the crystal structure (heavy-atom distance < CONTACT_CUTOFF Å), then
re-computes per-pose RMSD using only those Cα atoms.

Output: logs/analysis_bench300/interface_rmsd_labels.json
    {complex_name: {model_key: {
        "interface_residues": [res_num_str, ...],
        "n_interface": int,
        "n_total": int,
        "interface_rmsds": [float, ...]
    }}}

Usage:
    PYTHONPATH=$(pwd) ~/miniconda3/envs/rapidock/bin/python \\
        scripts/extract_interface_rmsds.py [--cutoff 4.0] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")
BJSON_PATH = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
OUT_PATH = REPO / "logs" / "analysis_bench300" / "interface_rmsd_labels.json"

CONTACT_CUTOFF = 4.0   # Å, heavy-atom distance


# ── Kabsch RMSD ──────────────────────────────────────────────────────────────

def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Minimum RMSD after Kabsch optimal rotation of Q onto P."""
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = Q.T @ P
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    diff = P - (Q @ R.T)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ── PDB parsing ──────────────────────────────────────────────────────────────

def _heavy_atoms_by_residue(pdb: Path) -> dict[str, list[np.ndarray]]:
    """Return {resnum_str: [xyz, ...]} of non-hydrogen heavy atoms."""
    res: dict[str, list[np.ndarray]] = {}
    for line in open(pdb):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        aname = line[12:16].strip()
        if aname.startswith("H"):
            continue
        elem = line[76:78].strip() if len(line) > 76 else ""
        if elem == "H":
            continue
        try:
            resnum = line[22:26].strip()
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            res.setdefault(resnum, []).append(xyz)
        except ValueError:
            pass
    return res


def _all_heavy_atoms(pdb: Path) -> np.ndarray:
    """Return (N, 3) array of all non-hydrogen heavy atom coords."""
    pts = []
    for line in open(pdb):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        aname = line[12:16].strip()
        if aname.startswith("H"):
            continue
        elem = line[76:78].strip() if len(line) > 76 else ""
        if elem == "H":
            continue
        try:
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            pass
    return np.array(pts) if pts else np.empty((0, 3))


def _ca_by_residue(pdb: Path) -> dict[str, np.ndarray]:
    """Return {resnum_str: xyz} of Cα atoms."""
    ca: dict[str, np.ndarray] = {}
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            try:
                resnum = line[22:26].strip()
                ca[resnum] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
    return ca


# ── Interface detection ───────────────────────────────────────────────────────

def interface_residues(
    receptor_pdb: Path, peptide_pdb: Path, cutoff: float = CONTACT_CUTOFF
) -> list[str]:
    """Return sorted list of peptide residue numbers (as str) within `cutoff` Å
    of any receptor heavy atom in the crystal structure."""
    rec_heavy = _all_heavy_atoms(receptor_pdb)
    pep_by_res = _heavy_atoms_by_residue(peptide_pdb)
    pep_ca = _ca_by_residue(peptide_pdb)

    if rec_heavy.shape[0] == 0:
        # fallback: all residues
        return sorted(pep_ca.keys(), key=lambda r: int(r))

    iface = []
    for resnum, atoms in pep_by_res.items():
        for xyz in atoms:
            dists = np.linalg.norm(rec_heavy - xyz, axis=1)
            if dists.min() < cutoff:
                iface.append(resnum)
                break

    # guarantee at least 3 residues so RMSD isn't pathological
    if len(iface) < 3:
        iface = sorted(pep_ca.keys(), key=lambda r: int(r))

    return sorted(iface, key=lambda r: int(r))


# ── Per-pose interface RMSD ───────────────────────────────────────────────────

def interface_rmsd(
    ref_ca: dict[str, np.ndarray],
    pose_ca: dict[str, np.ndarray],
    iface_res: list[str],
) -> float | None:
    """Kabsch RMSD over interface residues only (superpose on interface Cα)."""
    pairs = [(ref_ca[r], pose_ca[r]) for r in iface_res
             if r in ref_ca and r in pose_ca]
    if len(pairs) < 3:
        return None
    ref_pts = np.array([p[0] for p in pairs])
    pose_pts = np.array([p[1] for p in pairs])
    return _kabsch_rmsd(ref_pts, pose_pts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF,
                    help="contact distance cutoff in Å (default 4.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print stats for first 5 complexes and exit")
    a = ap.parse_args()

    bjson = json.load(open(BJSON_PATH))
    complexes = sorted(bjson.keys())

    results: dict = {}
    missing_rec = 0
    missing_pose = 0
    n_fallback = 0

    for i, cx in enumerate(complexes):
        if a.dry_run and i >= 5:
            break

        rec_pdb = BASE / cx / f"{cx}_protein_pocket.pdb"
        pep_pdb = BASE / cx / f"{cx}_peptide.pdb"

        if not rec_pdb.exists() or not pep_pdb.exists():
            missing_rec += 1
            continue

        iface = interface_residues(rec_pdb, pep_pdb, a.cutoff)
        ref_ca = _ca_by_residue(pep_pdb)
        n_total = len(ref_ca)
        n_iface = len(iface)

        # check if fallback was triggered (< 3 → all residues used)
        if n_iface == n_total:
            n_fallback += 1

        results[cx] = {}
        for mk, entry in bjson[cx].items():
            poses_dir = Path(entry["poses_dir"])
            n_poses = entry["n_poses"]
            i_rmsds = []
            for pi in range(n_poses):
                pose_pdb = poses_dir / f"pose_{pi}.pdb"
                if not pose_pdb.exists():
                    missing_pose += 1
                    i_rmsds.append(None)
                    continue
                pose_ca = _ca_by_residue(pose_pdb)
                r = interface_rmsd(ref_ca, pose_ca, iface)
                i_rmsds.append(r)

            results[cx][mk] = {
                "interface_residues": iface,
                "n_interface": n_iface,
                "n_total": n_total,
                "interface_rmsds": i_rmsds,
            }

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(complexes)}  {cx}  iface={n_iface}/{n_total}")

    print(f"\nDone. {len(results)} complexes | missing_rec={missing_rec} "
          f"| missing_pose={missing_pose} | fallback_all={n_fallback}")

    if not a.dry_run:
        json.dump(results, open(OUT_PATH, "w"), indent=2)
        print(f"Saved → {OUT_PATH}")
    else:
        # print summary table
        for cx, mk_data in list(results.items())[:5]:
            for mk, v in mk_data.items():
                orig = bjson[cx][mk]["ref_rmsds"]
                new = [x for x in v["interface_rmsds"] if x is not None]
                orig_f = [x for x in orig if x is not None]
                print(f"{cx} [{mk}]  iface={v['n_interface']}/{v['n_total']}  "
                      f"orig_rmsds={[round(x,2) for x in orig_f[:3]]}  "
                      f"iface_rmsds={[round(x,2) for x in new[:3]]}")


if __name__ == "__main__":
    main()
